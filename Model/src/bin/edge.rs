use tokio::prelude::*;
use actix::prelude::*;
use tokio::io::WriteHalf;
use tokio::net::TcpStream;
use tokio::codec::{FramedRead,FramedWrite};
use model::{EdgeResponse, DeviceRequest, TensorRepresentation, EdgeRequest, CloudResponse};
use std::collections::HashMap;
use std::net::SocketAddr;
use tch::{CModule, Tensor};
use std::path::{Path, PathBuf};
use tch::nn::Module;
use bytes::Bytes;
use tokio::sync::mpsc::Sender;
use flate2::write::{GzEncoder, GzDecoder};
use std::io::prelude::*;
use std::str::FromStr;
use std::fs::DirBuilder;
use std::process::ExitStatus;

pub fn entropy(t:&Tensor) -> f64 {
    let c = (t.size()[1] as f64).log(2.);
    let mut sum = 0.;
    for i in t.get(0).iter::<f64>().unwrap() {
        sum += (i*i.log(2.))/c
    }
    -sum
}

struct EdgeWorker {
    classifier:CModule,
    main_path:CModule,
    threshold:f64,
    trainset_buffer:Vec<Tensor>,
    trainlabel_buffer:Vec<i64>,
    edge_service:Option<Addr<EdgeService>>
}

impl EdgeWorker {
    pub fn new<T:AsRef<Path>>(classifier:T,main_path:T) -> EdgeWorker {
        EdgeWorker {
            classifier: tch::CModule::load(classifier).unwrap(),
            main_path: tch::CModule::load(main_path).unwrap(),
            threshold: 0.1,
            trainset_buffer: vec![],
            trainlabel_buffer: vec![],
            edge_service: None
        }
    }
}

impl Actor for EdgeWorker {
    type Context = Context<Self>;
}

impl Handler<Eval> for EdgeWorker {
    type Result = Result<EvalResult,()>;

    fn handle(&mut self, msg: Eval, ctx: &mut Self::Context) -> Self::Result {
        let extracted = self.main_path.forward(&msg.xs);
        let classifier_result = self.classifier.forward(&extracted).softmax(-1);
        if entropy(&classifier_result) < self.threshold {
            // save sample
            let pred = classifier_result.argmax(-1,false).int64_value(&[]);
            self.trainset_buffer.push(msg.xs);
            self.trainlabel_buffer.push(pred);
            if self.trainset_buffer.len() > 100 {
                msg.reply.try_send(
                    TrainDeviceClassifier {
                        trainset: self.trainset_buffer.drain(0..).collect(),
                        label: self.trainlabel_buffer.drain(0..).collect()
                    }
                );
            }
            // exit
            println!("Task {}",msg.task_id);
            classifier_result.argmax(-1,false).print();
            Ok(EvalResult::ToClient {
                client_address: msg.client_address,
                task_id: msg.task_id,
                result: classifier_result
            })
        }
        else {
            // To cloud
            Ok(EvalResult::ToCloud {
                client_address: msg.client_address,
                task_id: msg.task_id,
                tensor: extracted
            })
        }
    }
}

impl Handler<EdgeServiceStart> for EdgeWorker {
    type Result = ();

    fn handle(&mut self, msg: EdgeServiceStart, ctx: &mut Self::Context) -> Self::Result {
        self.edge_service.replace(msg.0);
    }
}

struct EdgeService {
    // clients
    clients: HashMap<SocketAddr,Addr<ClientAgent>>,
    worker: Addr<EdgeWorker>,
    cloud_writer:Sender<Bytes>,
    device_trainer:Addr<DeviceClassifierTraniner>
}

impl Actor for EdgeService {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.worker.send(EdgeServiceStart(ctx.address()));
        self.device_trainer.try_send(EdgeServiceStart(ctx.address()));
    }
}

impl StreamHandler<TcpStream,std::io::Error> for EdgeService {
    fn handle(&mut self, item: TcpStream, ctx: &mut Self::Context) {
        let client_address = item.peer_addr().unwrap();
        let my_addr = ctx.address();
        println!("device of {} connected",&client_address);
        let (r,w) = item.split();
        let w = FramedWrite::new(w,tokio::codec::LengthDelimitedCodec::new());
        let r = FramedRead::new(r,tokio::codec::LengthDelimitedCodec::new());
        let (sender,receiver) = tokio::sync::mpsc::channel(1024);
        tokio::spawn(receiver
            .map_err(|e|())
            .forward(w.sink_map_err(|x|()))
            .map(|_|())
        );
        let worker = self.worker.clone();
        let cloud_writer = self.cloud_writer.clone();
        let client_agent = ClientAgent::create(move |c|{
            c.add_stream(r);
            ClientAgent {
                remote_address: client_address,
                worker,
                writer: sender,
                edge: my_addr,
                recv_bytes: 0,
                cloud_writer
            }
        });
        self.clients.insert(client_address,client_agent);
    }

    fn error(&mut self, err: std::io::Error, ctx: &mut Self::Context) -> Running {
        dbg!(err);
        Running::Stop
    }
}

impl Handler<ClientAgentStopped> for EdgeService {
    type Result = ();

    fn handle(&mut self, msg: ClientAgentStopped, ctx: &mut Self::Context) -> Self::Result {
        println!("Client {} disconnected.", &msg.0);
        self.clients.remove(&msg.0);
    }
}

impl Handler<TrainDeviceClassifier> for EdgeService {
    type Result = ();

    fn handle(&mut self, msg: TrainDeviceClassifier, ctx: &mut Self::Context) -> Self::Result {
        self.device_trainer.try_send(msg).unwrap();
    }
}

// cloud traffic
impl StreamHandler<bytes::BytesMut, std::io::Error> for EdgeService {
    fn handle(&mut self, item: bytes::BytesMut, ctx: &mut Self::Context) {
        if let Ok(response) = bincode::deserialize::<CloudResponse>(item.as_ref()) {
            self.clients.get(&response.client_address).unwrap().try_send(EdgeResponse::EvalResult {
                id: response.id,
                prob: response.prob
            }).unwrap();
        }
        else {
            println!("bad cloud response");
        }
    }
}

struct ClientAgent {
    remote_address: SocketAddr,
    worker: Addr<EdgeWorker>,
    writer: tokio::sync::mpsc::Sender<bytes::Bytes>,
    edge: Addr<EdgeService>,
    recv_bytes: u64,
    cloud_writer: tokio::sync::mpsc::Sender<bytes::Bytes>
}

impl Actor for ClientAgent {
    type Context = Context<Self>;
}

impl StreamHandler<bytes::BytesMut,std::io::Error> for ClientAgent {
    fn handle(&mut self, item: bytes::BytesMut, ctx: &mut Self::Context) {
        self.recv_bytes += item.len() as u64;
//        let mut writer = Vec::new();
//        let mut decoder = GzDecoder::new(writer);
//        decoder.write_all(item.as_ref()).unwrap();
//        writer = decoder.finish().unwrap();
        if let Ok(request) = bincode::deserialize::<DeviceRequest>(item.as_ref()) {
            match request {
                DeviceRequest::Eval {
                    id,
                    tensor
                } => {
//                    println!("recv eval request: {} bytes from {}",item.len(),self.remote_address);
                    let my_addr = ctx.address();
                    let task = self.worker.send(
                        Eval {
                            reply: self.edge.clone(),
                            client_address: self.remote_address,
                            task_id: id,
                            xs: tensor.to_tensor()
                        }
                    ).and_then(move|x|{
                        my_addr.send(x.unwrap())
                    }).map_err(|e|{
                        println!("Mailbox error when sending Eval")
                    }).into_actor(self);
                    ctx.spawn(task);
                }
            }
        }
        else {
            println!("bad request from {}",self.remote_address);
        }
    }

    fn finished(&mut self, ctx: &mut Self::Context) {
        self.edge.do_send(ClientAgentStopped(self.remote_address));
        println!("Recv data == {} MB", self.recv_bytes as f64 / (1024.*1024.));
        ctx.stop()
    }
}

impl Handler<EvalResult> for ClientAgent {
    type Result = ();

    fn handle(&mut self, msg: EvalResult, ctx: &mut Self::Context) -> Self::Result {
        match msg {
            EvalResult::ToClient {
                client_address, task_id, result
            } => {
                let edge_result = bincode::serialize(&EdgeResponse::EvalResult {
                    id: task_id,
                    prob: TensorRepresentation::from_tensor(&result)
                }).unwrap();
                self.writer.try_send(Bytes::from(edge_result)).unwrap();
            }
            EvalResult::ToCloud {
                client_address, task_id, tensor
            } => {
                let request = bincode::serialize(&EdgeRequest::Eval {
                    client_address,
                    id: task_id,
                    tensor: TensorRepresentation::from_tensor(&tensor)
                }).unwrap();
                self.cloud_writer.try_send(Bytes::from(request)).unwrap();
            }
        }
    }
}

impl Handler<EdgeResponse> for ClientAgent {
    type Result = ();

    fn handle(&mut self, msg: EdgeResponse, ctx: &mut Self::Context) -> Self::Result {
        let edge_result = bincode::serialize(&msg).unwrap();
        self.writer.try_send(Bytes::from(edge_result)).unwrap();
    }
}

impl Handler<SendTrainedClassifier> for ClientAgent {
    type Result = ();

    fn handle(&mut self, msg: SendTrainedClassifier, ctx: &mut Self::Context) -> Self::Result {
        let edge_result = bincode::serialize(&EdgeResponse::ClassifierUpdate {
            module: msg.file
        }).unwrap();
        self.writer.try_send(Bytes::from(edge_result)).unwrap();
    }
}

struct DeviceClassifierTraniner {
    pub edge_service:Option<Addr<EdgeService>>,
    pub old_classifier_path:PathBuf,
    pub old_classifier:CModule,
    pub indexer:usize
}

impl DeviceClassifierTraniner {
    pub fn new(classifier:&str) -> DeviceClassifierTraniner {
        DeviceClassifierTraniner {
            edge_service: None,
            old_classifier_path: PathBuf::from_str(classifier).unwrap(),
            old_classifier: tch::CModule::load(classifier).unwrap(),
            indexer: 0
        }
    }
}

impl Actor for DeviceClassifierTraniner {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
//        std::fs::remove_dir("/tmp/sddnn").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/train_set/0").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/train_set/1").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/train_set/2").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/train_set/3").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/train_set/4").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/train_set/5").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/test_set/0").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/test_set/1").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/test_set/2").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/test_set/3").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/test_set/4").unwrap();
        DirBuilder::new().recursive(true).create("/tmp/sddnn/test_set/5").unwrap();
    }
}

impl Handler<EdgeServiceStart> for DeviceClassifierTraniner {
    type Result = ();

    fn handle(&mut self, msg: EdgeServiceStart, ctx: &mut Self::Context) -> Self::Result {
        self.edge_service = Some(msg.0);
    }
}

impl Handler<TrainDeviceClassifier> for DeviceClassifierTraniner {
    type Result = ();

    fn handle(&mut self, mut msg: TrainDeviceClassifier, ctx: &mut Self::Context) -> Self::Result {
        if msg.trainset.len()!=msg.label.len() {
            println!("[ERR] len(trainset)!=len(label)");
            return;
        }
        let len = msg.trainset.len()/2;
        let train_set: Vec<Tensor> = msg.trainset.drain(0..len).collect();
        let train_label:Vec<i64> = msg.label.drain(0..len).collect();
        let test_set = msg.trainset;
        let test_label = msg.label;
        for (tensor,target) in train_set.iter().zip(train_label.iter()) {
            let path = format!("/tmp/sddnn/train_set/{}/{}.npy",target,self.indexer);
            tensor.write_npy(&path).unwrap();
            self.indexer+=1;
        }
        for (tensor,target) in test_set.iter().zip(test_label.iter()) {
            let path = format!("/tmp/sddnn/test_set/{}/{}.npy",target,self.indexer);
            tensor.write_npy(&path).unwrap();
            self.indexer+=1;
        }
        let addr= self.edge_service.as_ref().unwrap().clone();
        actix::spawn(futures::future::lazy(move|| {
            let mut exit = std::process::Command::new("python3")
                .arg("/home/skye/SDDNN/train_device_classifier.py")
                .current_dir("/home/skye/SDDNN")
                .output()
                .unwrap();
            let mut outputs = exit
                .status;
            match outputs.code() {
                Some(0) => {
                    if let Ok(module) = std::fs::read("/tmp/sddnn/device_classifier.pt") {
                        addr.try_send(SendTrainedClassifier {
                            path: "/tmp/sddnn/device_classifier.pt",
                            file: module
                        });
                    }
                }
                _ => {
                    println!("not trained: {}", String::from_utf8_lossy(exit.stdout.as_ref()));
                }
            }
            Ok(())
        }));
    }
}

impl Handler<SendTrainedClassifier> for EdgeService {
    type Result = ();

    fn handle(&mut self, msg: SendTrainedClassifier, ctx: &mut Self::Context) -> Self::Result {
        for (_,client) in self.clients.iter() {
            client.try_send(msg.clone()).unwrap();
        }
    }
}

#[derive(Message)]
pub struct ClientAgentStopped(pub SocketAddr);

#[derive(Message,Clone)]
struct SendTrainedClassifier {
    path:&'static str,
    file:Vec<u8>
}

struct Eval {
    pub reply:Addr<EdgeService>,
    pub client_address:SocketAddr,
    pub task_id:usize,
    pub xs:Tensor
}

#[derive(Message)]
pub enum EvalResult {
    ToClient {
        client_address:SocketAddr,
        task_id:usize,
        result:Tensor
    },
    ToCloud {
        client_address:SocketAddr,
        task_id:usize,
        tensor:Tensor
    }
}

#[derive(Message)]
struct EdgeServiceStart(pub Addr<EdgeService>);

#[derive(Message)]
pub struct TrainDeviceClassifier{
    trainset:Vec<Tensor>,
    label:Vec<i64>
}

impl Message for Eval {
    type Result = Result<EvalResult,()>;
}

fn main() {
    System::run(||{
        let work_addr = EdgeWorker::create(|_| EdgeWorker::new("edge_classifier.pt","edge_main.pt"));

        let edge_address = "127.0.0.1:12345".parse().unwrap();
        let cloud_address = "127.0.0.1:12346".parse().unwrap();
        let tcp_listener = tokio::net::TcpListener::bind(&edge_address).unwrap();
        let tcp_stream = tokio::net::TcpStream::connect(&cloud_address).and_then(|x|{
            let (reader,writer) = x.split();
            let edge_server = EdgeService::create(move |x|{
                x.add_stream(tcp_listener.incoming());
                x.add_stream(FramedRead::new(reader, tokio::codec::LengthDelimitedCodec::new()));
                let (sender,receiver) = tokio::sync::mpsc::channel(1024);
                let tcp_w = FramedWrite::new(writer,tokio::codec::LengthDelimitedCodec::new());
                tokio::spawn(receiver
                    .map_err(|e|())
                    .forward(tcp_w.sink_map_err(|x|()))
                    .map(|_|())
                );
                let device_trainer = DeviceClassifierTraniner::create(|ctx|{
                    DeviceClassifierTraniner::new("device_classifier.pt")
                });
                EdgeService {
                    clients: Default::default(),
                    worker: work_addr,
                    cloud_writer: sender,
                    device_trainer
                }
            });
            futures::future::ok(())
        }).map_err(|e|{
            dbg!(e);
        });
        actix::spawn(tcp_stream);
    }).unwrap();
}