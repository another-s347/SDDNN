use tokio::prelude::*;
use actix::prelude::*;
use tokio::io::WriteHalf;
use tokio::net::TcpStream;
use tokio::codec::{FramedRead,FramedWrite};
use model::{EdgeResponse, DeviceRequest, TensorRepresentation, EdgeRequest, CloudResponse};
use std::collections::HashMap;
use std::net::SocketAddr;
use tch::{CModule, Tensor};
use std::path::Path;
use tch::nn::Module;
use bytes::Bytes;
use tokio::sync::mpsc::Sender;
use flate2::write::{GzEncoder, GzDecoder};
use std::io::prelude::*;

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
    threshold:f64
}

impl EdgeWorker {
    pub fn new<T:AsRef<Path>>(classifier:T,main_path:T) -> EdgeWorker {
        EdgeWorker {
            classifier: tch::CModule::load(classifier).unwrap(),
            main_path: tch::CModule::load(main_path).unwrap(),
            threshold: 0.1
        }
    }
}

impl Actor for EdgeWorker {
    type Context = SyncContext<Self>;
}

impl Handler<Eval> for EdgeWorker {
    type Result = Result<EvalResult,()>;

    fn handle(&mut self, msg: Eval, ctx: &mut Self::Context) -> Self::Result {
        let classifier_result = self.classifier.forward(&msg.xs).softmax(-1);
        if entropy(&classifier_result) < self.threshold {
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
            let extracted = self.main_path.forward(&msg.xs);
            Ok(EvalResult::ToCloud {
                client_address: msg.client_address,
                task_id: msg.task_id,
                tensor: extracted
            })
        }
    }
}

struct EdgeService {
    // clients
    clients: HashMap<SocketAddr,Addr<ClientAgent>>,
    worker: Addr<EdgeWorker>,
    cloud_writer:Sender<Bytes>
}

impl Actor for EdgeService {
    type Context = Context<Self>;
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

// cloud traffic
impl StreamHandler<bytes::BytesMut, std::io::Error> for EdgeService {
    fn handle(&mut self, item: bytes::BytesMut, ctx: &mut Self::Context) {
        if let Ok(response) = bincode::deserialize::<CloudResponse>(item.as_ref()) {
            self.clients.get(&response.client_address).unwrap().try_send(EdgeResponse {
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
                let edge_result = bincode::serialize(&EdgeResponse {
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

#[derive(Message)]
pub struct ClientAgentStopped(pub SocketAddr);

pub struct Eval {
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

impl Message for Eval {
    type Result = Result<EvalResult,()>;
}

fn main() {
    System::run(||{
        let work_addr = SyncArbiter::start(1, || EdgeWorker::new("edge_classifier.pt","edge_main.pt"));

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
                EdgeService {
                    clients: Default::default(),
                    worker: work_addr,
                    cloud_writer: sender
                }
            });
            futures::future::ok(())
        }).map_err(|e|{
            dbg!(e);
        });
        actix::spawn(tcp_stream);
    }).unwrap();
}