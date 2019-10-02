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
use structopt::StructOpt;

struct CloudWorker {
    model:CModule
}

impl CloudWorker {
    pub fn new<T:AsRef<Path>>(path:T) -> CloudWorker {
        CloudWorker {
            model: tch::CModule::load(path).unwrap()
        }
    }
}

impl Actor for CloudWorker {
    type Context = SyncContext<Self>;
}

impl Handler<Eval> for CloudWorker {
    type Result = Result<EvalResult,()>;

    fn handle(&mut self, msg: Eval, ctx: &mut Self::Context) -> Self::Result {
        let result = self.model.forward(&msg.xs).softmax(-1);
        println!("Task {}",msg.task_id);
        result.print();
        Ok(EvalResult {
            client_address: msg.client_address,
            edge_address: msg.edge_address,
            task_id: msg.task_id,
            result
        })
    }
}

struct CloudService {
    // clients
    edge: HashMap<SocketAddr,Addr<EdgeAgent>>,
    worker: Addr<CloudWorker>
}

impl Actor for CloudService {
    type Context = Context<Self>;
}

impl StreamHandler<TcpStream,std::io::Error> for CloudService {
    fn handle(&mut self, item: TcpStream, ctx: &mut Self::Context) {
        let edge_address = item.peer_addr().unwrap();
        let my_addr = ctx.address();
        println!("edge of {} connected",&edge_address);
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
        let edge_agent = EdgeAgent::create(move |c|{
            c.add_stream(r);
            EdgeAgent {
                remote_address: edge_address,
                worker,
                writer: sender,
                edge: my_addr,
                recv_bytes: 0
            }
        });
        self.edge.insert(edge_address, edge_agent);
    }

    fn error(&mut self, err: std::io::Error, ctx: &mut Self::Context) -> Running {
        dbg!(err);
        Running::Stop
    }
}

impl Handler<EdgeAgentStopped> for CloudService {
    type Result = ();

    fn handle(&mut self, msg: EdgeAgentStopped, ctx: &mut Self::Context) -> Self::Result {
        println!("Client {} disconnected.", &msg.0);
        self.edge.remove(&msg.0);
    }
}

struct EdgeAgent {
    remote_address: SocketAddr,
    worker: Addr<CloudWorker>,
    writer: tokio::sync::mpsc::Sender<bytes::Bytes>,
    edge: Addr<CloudService>,
    recv_bytes: u64
}

impl Actor for EdgeAgent {
    type Context = Context<Self>;
}

impl StreamHandler<bytes::BytesMut,std::io::Error> for EdgeAgent {
    fn handle(&mut self, item: bytes::BytesMut, ctx: &mut Self::Context) {
        self.recv_bytes += item.len() as u64;
        if let Ok(request) = bincode::deserialize::<EdgeRequest>(item.as_ref()) {
            match request {
                EdgeRequest::Eval {
                    client_address, id,
                    tensor
                } => {
//                    println!("recv eval request: {} bytes from {}",item.len(),self.remote_address);
                    let my_addr = ctx.address();
                    let task = self.worker.send(
                        Eval {
                            client_address,
                            edge_address: self.remote_address,
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
        self.edge.do_send(EdgeAgentStopped(self.remote_address));
        println!("Recv data == {} MB", self.recv_bytes as f64 / (1024.*1024.));
        ctx.stop()
    }
}

impl Handler<EvalResult> for EdgeAgent {
    type Result = ();

    fn handle(&mut self, msg: EvalResult, ctx: &mut Self::Context) -> Self::Result {
        let edge_result = bincode::serialize(&CloudResponse {
            client_address: msg.client_address,
            id: msg.task_id,
            prob: TensorRepresentation::from_tensor(&msg.result)
        }).unwrap();
        self.writer.try_send(Bytes::from(edge_result)).unwrap();
    }
}

#[derive(Message)]
pub struct EdgeAgentStopped(pub SocketAddr);

pub struct Eval {
    pub client_address:SocketAddr,
    pub edge_address:SocketAddr,
    pub task_id:usize,
    pub xs:Tensor
}

#[derive(Message)]
pub struct EvalResult {
    pub client_address:SocketAddr,
    pub edge_address:SocketAddr,
    pub task_id:usize,
    pub result:Tensor
}

impl Message for Eval {
    type Result = Result<EvalResult,()>;
}

#[derive(StructOpt, Debug,Clone)]
#[structopt(name = "cloud")]
struct Opt {
    #[structopt(short, long,default_value="127.0.0.1:12346")]
    listen:String
}

fn main() {
    let opt:Opt = Opt::from_args();

    System::run(move||{
        let work_addr = SyncArbiter::start(1, || CloudWorker::new("cloud.pt"));

        let cloud_address = opt.listen.parse().unwrap();
        let tcp_listener = tokio::net::TcpListener::bind(&cloud_address).unwrap();
        let cloud_server = CloudService::create(move |x|{
            x.add_stream(tcp_listener.incoming());
            CloudService {
                edge: Default::default(),
                worker: work_addr
            }
        });
    }).unwrap();
}