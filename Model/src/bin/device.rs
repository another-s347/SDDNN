#![feature(bind_by_move_pattern_guards)]

use tokio::prelude::*;
use actix::prelude::*;
use tokio::io::WriteHalf;
use tokio::net::{TcpStream, UdpSocket, UdpFramed};
use tokio::codec::{FramedRead,FramedWrite};
use model::{EdgeResponse, DeviceRequest, TensorRepresentation, read_npy, read_pred};
use tch::{CModule, IValue, Tensor, Kind};
use std::path::Path;
use tch::vision::dataset::Dataset;
use tch::data::Iter2;
use std::ops::Deref;
use bytes::{BytesMut, BufMut};
use tch::nn::{Module, ModuleT};
use std::time::{Instant, Duration};
use flate2::{GzBuilder, Compression};
use flate2::write::GzEncoder;
use serde::de::value::U32Deserializer;
use std::path::PathBuf;
use structopt::StructOpt;
use std::net::{SocketAddr, Ipv4Addr};
use serde::{Deserialize,Serialize};

static CLASSES:[&'static str;6] = ["buildings", "forest", "glacier", "mountain", "sea", "street"];

pub fn entropy(t:&Tensor) -> f64 {
    let c = (t.size()[1] as f64).log(2.);
    let mut sum = 0.;
    for i in t.get(0).iter::<f64>().unwrap() {
        sum += (i*i.log(2.))/c
    }
    -sum
}

pub struct DeviceWorker {
    main_path:CModule,
    classifier:CModule,
    extractor:CModule,
    threshold:f64,
    id:u32
}

impl DeviceWorker {
    pub fn new<T:AsRef<Path>>(id:u32,main_path:T,classifier:T,extractor:T) -> DeviceWorker {
        DeviceWorker {
            main_path: tch::CModule::load(main_path).unwrap(),
            classifier: tch::CModule::load(classifier).unwrap(),
            extractor: tch::CModule::load(extractor).unwrap(),
            threshold: 0.2,
            id
        }
    }
}

impl Actor for DeviceWorker {
    type Context = SyncContext<Self>;
}

impl Handler<DeviceEval> for DeviceWorker {
    type Result = Result<DeviceEvalResult,String>;

    fn handle(&mut self, msg: DeviceEval, ctx: &mut Self::Context) -> Self::Result {
        let xs = msg.images;
        let main_outputs = self.main_path.forward_t(&xs,false);
        let extracted = self.extractor.forward(&main_outputs);
        let classifier_result = self.classifier.forward_t(&extracted,false).softmax(-1);
        let ent = entropy(&classifier_result);
        if ent < self.threshold {
            // exit
//            println!("Early exit on task {}, entropy={}",msg.index,ent);
            Ok(DeviceEvalResult::Success {
                index: msg.index,
                prediciton: classifier_result,
                xs
            })
        }
        else {
//            println!("Pass to edge on task {}, entropy={}",msg.index,ent);
            // to edge
            Ok(DeviceEvalResult::ToEdge {
                index: msg.index,
                tensor: extracted
            })
        }
    }
}

impl Handler<UpdateClassifier> for DeviceWorker {
    type Result = ();

    fn handle(&mut self, mut msg: UpdateClassifier, ctx: &mut Self::Context) -> Self::Result {
        let path = format!("worker/{}.pth",self.id);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path).unwrap();
        file.write_all(msg.0.as_mut()).unwrap();
        if let Ok(module) = tch::CModule::load(&path) {
            self.classifier = module;
            println!("Classifier updated");
        }
        else {
            println!("Bad classifier module data");
        }
    }
}

pub struct EdgeConnector {
    worker: Addr<DeviceWorker>,
    tcp_writer: tokio::sync::mpsc::Sender<bytes::Bytes>,
    test_loader: Option<Addr<TestLoader>>
}

impl Actor for EdgeConnector {
    type Context = Context<Self>;
}

impl Handler<DeviceRequest> for EdgeConnector {
    type Result = ();

    fn handle(&mut self, msg: DeviceRequest, ctx: &mut Self::Context) -> Self::Result {
        let bytes = bincode::serialize(&msg).unwrap();
//        self.controller.try_send((Task {
//            instruction_number: bytes.len() as u32,
//            identification: self.id.clone(),
//            amount_data: bytes.len() as u32
//        }, SocketAddr::new(Ipv4Addr::BROADCAST.into(), 12345))).unwrap();
//        println!("send {} MB",bytes.len() as f64/(1024.*1024.));
//        let mut encoder = GzEncoder::new(Vec::new(),Compression::best());
//        encoder.write(bytes.as_ref());
        self.tcp_writer.try_send(bytes::Bytes::from(bytes)).unwrap();
    }
}

impl StreamHandler<bytes::BytesMut,std::io::Error> for EdgeConnector {
    fn handle(&mut self, item: bytes::BytesMut, ctx: &mut Self::Context) {
        if let Ok(response) = bincode::deserialize::<EdgeResponse>(item.as_ref()) {
            self.test_loader.as_ref().unwrap().try_send(response).unwrap();
        }
    }
}

impl Handler<SetTestloader> for EdgeConnector {
    type Result = ();

    fn handle(&mut self, msg: SetTestloader, ctx: &mut Self::Context) -> Self::Result {
        if self.test_loader.is_some() {
            panic!("Test loader already set");
        }
        self.test_loader = Some(msg.0);
    }
}

pub struct TestLoader {
    xss:Vec<Tensor>,
    yss:Vec<Tensor>,
    worker:Addr<DeviceWorker>,
    edge:Addr<EdgeConnector>,
    correct:i64,
    total:i64,
    preds:Vec<(usize,Tensor)>,
    current_index:usize,
    timestamp:Vec<Instant>,
    duras:f64,
    pred_set:bool,
    start:Instant
}

impl TestLoader {
    pub fn new(worker: Addr<DeviceWorker>, edge_connector:Addr<EdgeConnector>, pred_set:bool) -> Self {
        println!("Loading...");
        if pred_set {
            let preds = read_pred("/home/skye/SDDNN/unknown");
            let total = preds.len();
            println!("Pred set loaded");
            TestLoader {
                xss:vec![],
                yss:vec![],
                worker,
                edge: edge_connector,
                correct: 0,
                total: 0,
                preds,
                current_index: 0,
                timestamp: vec![Instant::now();total],
                duras: 0.0,
                pred_set,
                start: Instant::now()
            }
        }
        else {
            let mut xss:Vec<Tensor> = vec![];
            let mut yss:Vec<Tensor> = vec![];
            for xs in read_npy("/home/skye/SDDNN/buildings") {
                xss.push(xs);
                yss.push(Tensor::of_slice(&[0]).to_kind(Kind::Int64));
            }
            for xs in read_npy("/home/skye/SDDNN/forest") {
                xss.push(xs);
                yss.push(Tensor::of_slice(&[1]).to_kind(Kind::Int64));
            }
            for xs in read_npy("/home/skye/SDDNN/glacier") {
                xss.push(xs);
                yss.push(Tensor::of_slice(&[2]).to_kind(Kind::Int64));
            }
            for xs in read_npy("/home/skye/SDDNN/mountain") {
                xss.push(xs);
                yss.push(Tensor::of_slice(&[3]).to_kind(Kind::Int64));
            }
            for xs in read_npy("/home/skye/SDDNN/sea") {
                xss.push(xs);
                yss.push(Tensor::of_slice(&[4]).to_kind(Kind::Int64));
            }
            for xs in read_npy("/home/skye/SDDNN/street") {
                xss.push(xs);
                yss.push(Tensor::of_slice(&[5]).to_kind(Kind::Int64));
            }
            let total = xss.len();
            println!("Test set loaded");
            TestLoader {
                xss,
                yss,
                worker,
                edge: edge_connector,
                correct: 0,
                total: 0,
                preds: vec![],
                current_index: 0,
                timestamp: vec![Instant::now();total],
                duras: 0.0,
                pred_set,
                start: Instant::now()
            }
        }
    }
}

impl Actor for TestLoader {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.edge.try_send(SetTestloader(ctx.address())).unwrap();
        self.start = Instant::now();
        ctx.notify(NextTest)
    }
}

impl Handler<EdgeResponse> for TestLoader {
    type Result = ();

    fn handle(&mut self, msg: EdgeResponse, ctx: &mut Self::Context) -> Self::Result {
        match msg {
            EdgeResponse::EvalResult { id, prob } => {
                let prob = prob.to_tensor();
                if self.pred_set {
                    let class_index = prob.argmax(-1, false).f_int64_value(&[]).unwrap() as usize;
                    let prob = prob.max().f_double_value(&[]).unwrap();
                    println!("Id: {} is {} with {:.3}% | From edge.", id, CLASSES[class_index], 100.*prob);
                } else {
                    let ys = self.yss.get(id).unwrap();
                    let correct = prob.argmax(-1, false)
                        .eq1(&ys)
                        .sum().f_int64_value(&[]).unwrap();
                    self.total += ys.size()[0];
                    self.correct += correct;
                    let dur = Instant::now().duration_since(self.timestamp.get(id).unwrap().clone());
                    self.duras += dur.as_secs_f64();
                }
                ctx.notify(NextTest);
            }
            EdgeResponse::ClassifierUpdate { module } => {
                self.worker.try_send(UpdateClassifier(module)).unwrap();
            }
        }
    }
}

impl Handler<NextTest> for TestLoader {
    type Result = ();

    fn handle(&mut self, msg: NextTest, ctx: &mut Self::Context) -> Self::Result {
        let index = self.current_index;
        if !self.pred_set && (index % 10 == 0 || index+1==self.xss.len()) {
            println!("Acc {:?}% {}/{}, duration == {}",100.*(self.correct as f64 / self.total as f64),self.correct,self.total,self.duras/self.total as f64);
        }
        if self.current_index >= 800 {
            let now = Instant::now();
            let dura = now.duration_since(self.start);
            println!("Total time: {}",dura.as_secs_f64());
            actix::System::current().stop();
            return;
        }
        if self.current_index >= self.timestamp.len() {
            actix::System::current().stop();
            return;
        }
        self.current_index += 1;
//        println!("start test {}", index);
        if !self.pred_set {
            if let Some(images) = self.xss.get(index).map(|x|x.shallow_clone()) {
                let mut my_addr = ctx.address().clone();
                let task = self.worker.send(DeviceEval {
                    index,
                    images
                }).and_then(move |x|{
                    my_addr.send(x.unwrap())
                }).map_err(|e|{
                    println!("Mailbox error when sending DeviceEval")
                }).into_actor(self);
                *self.timestamp.get_mut(index).unwrap() = Instant::now();
                ctx.spawn(task);
            }
        }
        else {
            if let Some((id, images)) = self.preds.get(index).map(|(id,x)|(*id,x.shallow_clone())) {
                let mut my_addr = ctx.address().clone();
                let task = self.worker.send(DeviceEval {
                    index:id,
                    images
                }).and_then(move |x|{
                    my_addr.send(x.unwrap())
                }).map_err(|e|{
                    println!("Mailbox error when sending DeviceEval")
                }).into_actor(self);
                *self.timestamp.get_mut(index).unwrap() = Instant::now();
                ctx.spawn(task);
            }
        }
    }
}

impl Handler<DeviceEvalResult> for TestLoader {
    type Result = ();

    fn handle(&mut self, msg: DeviceEvalResult, ctx: &mut Self::Context) -> Self::Result {
        //
        match msg {
            DeviceEvalResult::Success {
                index, prediciton, xs
            } => {
                if self.pred_set {
                    let class_index = prediciton.argmax(-1, false).f_int64_value(&[]).unwrap() as usize;
                    let prob = prediciton.max().f_double_value(&[]).unwrap();
                    println!("Id: {} is {} with {:.3}%", index, CLASSES[class_index], 100.*prob);
                }
                else {
                    let ys = self.yss.get(index).unwrap();
                    self.correct += prediciton.argmax(-1, false)
                        .eq1(&ys)
                        .sum().f_int64_value(&[]).unwrap();
                    self.total += ys.size()[0];
                    let dur = Instant::now().duration_since(self.timestamp.get(index).unwrap().clone());
                    self.duras+=dur.as_secs_f64();
                }
                ctx.notify(NextTest);
            }
            DeviceEvalResult::ToEdge {
                index, tensor
            } => {
                self.edge.do_send(DeviceRequest::Eval {
                    id: index,
                    tensor: TensorRepresentation::from_tensor(&tensor)
                });
//                ctx.notify(NextTest);
            }
        }
    }
}

#[derive(Message)]
struct NextTest;

struct DeviceEval {
    pub index:usize,
    pub images:Tensor
}

#[derive(Message)]
struct UpdateClassifier(pub Vec<u8>);

impl Message for DeviceEval {
    type Result = Result<DeviceEvalResult,String>;
}

#[derive(Message)]
enum DeviceEvalResult {
    Success {
        index:usize,
        prediciton:Tensor,
        xs:Tensor
    },
    ToEdge {
        index:usize,
        tensor:Tensor
    }
}

#[derive(Message)]
struct SetTestloader(pub Addr<TestLoader>);

#[derive(StructOpt, Debug,Clone)]
#[structopt(name = "device")]
struct Opt {
    #[structopt(short, long,default_value="127.0.0.1:12345")]
    edge:String,
    #[structopt(short, long)]
    id:u32,
    #[structopt(short, long)]
    pred:bool
}

fn main() {
    let opt:Opt = Opt::from_args();

    System::run(move||{
        let id = opt.id;
        let work_addr = SyncArbiter::start(1, move || DeviceWorker::new(id,"device_main.pt","device_classifier.pt","device_extract.pt"));

        let edge_address = opt.edge.parse().unwrap();
        let tcp_stream = tokio::net::TcpStream::connect(&edge_address);
        let split_stream = tcp_stream.and_then(|x|{
            futures::future::ok(x.split())
        });
        actix::spawn(split_stream.and_then(move |(reader,writer)|{
            let worker = work_addr.clone();
            let edge_connector: Addr<EdgeConnector> = EdgeConnector::create(move |x|{
                x.add_stream(FramedRead::new(reader, tokio::codec::LengthDelimitedCodec::new()));
                let (sender,receiver) = tokio::sync::mpsc::channel(1024);
                let tcp_w = FramedWrite::new(writer,tokio::codec::LengthDelimitedCodec::new());
                tokio::spawn(receiver
                    .map_err(|e|())
                    .forward(tcp_w.sink_map_err(|x|()))
                    .map(|_|())
                );
                EdgeConnector {
                    worker: work_addr,
                    tcp_writer: sender,
                    test_loader: None,
                }
            });
            TestLoader::create(move|_|TestLoader::new(worker ,edge_connector,opt.pred));
            futures::future::ok(())
        }).map_err(|e|{
            dbg!(e);
        }));
    }).unwrap();
}