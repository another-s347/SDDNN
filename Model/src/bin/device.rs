#![feature(bind_by_move_pattern_guards)]

use tokio::prelude::*;
use actix::prelude::*;
use tokio::io::WriteHalf;
use tokio::net::TcpStream;
use tokio::codec::{FramedRead,FramedWrite};
use model::{EdgeResponse, DeviceRequest, TensorRepresentation, read_npy};
use tch::{CModule, IValue, Tensor, Kind};
use std::path::Path;
use tch::vision::dataset::Dataset;
use tch::data::Iter2;
use std::ops::Deref;
use bytes::BytesMut;
use tch::nn::{Module, ModuleT};
use chrono::{DateTime, Utc};
use std::time::{Instant, Duration};

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
    threshold:f64
}

impl DeviceWorker {
    pub fn new<T:AsRef<Path>>(main_path:T,classifier:T,extractor:T) -> DeviceWorker {
        DeviceWorker {
            main_path: tch::CModule::load(main_path).unwrap(),
            classifier: tch::CModule::load(classifier).unwrap(),
            extractor: tch::CModule::load(extractor).unwrap(),
            threshold: 0.0
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
        let classifier_result = self.classifier.forward_t(&main_outputs,false).softmax(-1);
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
            let extracted = self.extractor.forward(&main_outputs);
            Ok(DeviceEvalResult::ToEdge {
                index: msg.index,
                tensor: extracted
            })
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
//        println!("send {} MB",bytes.len() as f64/(1024.*1024.));
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
    current_index:usize,
    timestamp:Vec<Instant>,
    duras:f64
}

impl TestLoader {
    pub fn new(worker: Addr<DeviceWorker>, edge_connector:Addr<EdgeConnector>) -> Self {
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
//        println!("Total: {}",xss.len());
        TestLoader {
            xss,
            yss,
            worker,
            edge: edge_connector,
            correct: 0,
            total: 0,
            current_index: 0,
            timestamp: vec![Instant::now();3000],
            duras: 0.0
        }
    }
}

impl Actor for TestLoader {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.edge.try_send(SetTestloader(ctx.address())).unwrap();
        ctx.notify(NextTest)
    }
}

impl Handler<EdgeResponse> for TestLoader {
    type Result = ();

    fn handle(&mut self, msg: EdgeResponse, ctx: &mut Self::Context) -> Self::Result {
        let ys = self.yss.get(msg.id).unwrap();
        let correct = msg.prob.to_tensor().argmax(-1, false)
            .eq1(&ys)
            .sum().f_int64_value(&[]).unwrap();
        self.total += ys.size()[0];
        self.correct += correct;
        let dur = Instant::now().duration_since(self.timestamp.get(msg.id).unwrap().clone());
        self.duras += dur.as_secs_f64();
        ctx.notify(NextTest);
    }
}

impl Handler<NextTest> for TestLoader {
    type Result = ();

    fn handle(&mut self, msg: NextTest, ctx: &mut Self::Context) -> Self::Result {
        let index = self.current_index;
        if index % 10 == 0 || index+1==self.xss.len() {
            println!("Acc {:?}% {}/{}, duration == {}",100.*(self.correct as f64 / self.total as f64),self.correct,self.total,self.duras/self.total as f64);
        }
        if self.current_index >= self.xss.len() {
            return;
        }
        self.current_index += 1;
//        println!("start test {}", index);
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
}

impl Handler<DeviceEvalResult> for TestLoader {
    type Result = ();

    fn handle(&mut self, msg: DeviceEvalResult, ctx: &mut Self::Context) -> Self::Result {
        //
        match msg {
            DeviceEvalResult::Success {
                index, prediciton, xs
            } => {
                let ys = self.yss.get(index).unwrap();
                self.correct += prediciton.argmax(-1, false)
                    .eq1(&ys)
                    .sum().f_int64_value(&[]).unwrap();
                self.total += ys.size()[0];
                let dur = Instant::now().duration_since(self.timestamp.get(index).unwrap().clone());
                self.duras+=dur.as_secs_f64();
                ctx.notify(NextTest);
            }
            DeviceEvalResult::ToEdge {
                index, tensor
            } => {
                self.edge.do_send(DeviceRequest::Eval {
                    id: index,
                    tensor: TensorRepresentation::from_tensor(&tensor)
                })
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

fn main() {
    System::run(||{
        let work_addr = SyncArbiter::start(1, || DeviceWorker::new("device_main.pt","device_classifier.pt","device_extract.pt"));

        let edge_address = "127.0.0.1:12345".parse().unwrap();
        let tcp_stream = tokio::net::TcpStream::connect(&edge_address);
        let split_stream = tcp_stream.and_then(|x|{
            futures::future::ok(x.split())
        });
        actix::spawn(split_stream.and_then(move |(reader,writer)|{
            let worker = work_addr.clone();
            let edge_connector = EdgeConnector::create(move |x|{
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
                    test_loader: None
                }
            });
            TestLoader::create(|_|TestLoader::new(worker ,edge_connector));
            futures::future::ok(())
        }).map_err(|e|{
            dbg!(e);
        }));
    }).unwrap();
}