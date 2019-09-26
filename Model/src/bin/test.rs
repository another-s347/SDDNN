#![feature(bind_by_move_pattern_guards)]

use tch::{nn, Device, Tensor, CModule, IValue, Kind};
use tch::nn::{ModuleT, OptimizerConfig, Module};
use tch::data::Iter2;
use model::{read_npy, TensorRepresentation, read_pred};

pub fn entropy(t:&Tensor) -> f64 {
    let c = (t.size()[1] as f64).log(2.);
    let mut sum = 0.;
    for i in t.get(0).iter::<f64>().unwrap() {
        sum += (i*i.log(2.))/c
    }
    -sum
}

pub fn main() -> Result<(),Box<dyn std::error::Error>> {
    dbg!(std::process::Command::new("python3")
        .arg("/home/skye/SDDNN/train_device_classifier.py")
        .current_dir("/home/skye/SDDNN")
        .output());
//    let device_main = tch::CModule::load("device_main.pt")?;
//    let device_extract = tch::CModule::load("device_extract.pt")?;
//    let device_classifier = tch::CModule::load("device_classifier.pt")?;
//    let edge_main = tch::CModule::load("edge_main.pt")?;
//    let edge_classifier = tch::CModule::load("edge_classifier.pt")?;
//    let cloud = tch::CModule::load("cloud.pt")?;
//    let testloader = TestLoader::new();
//    let mut mid_exit = 0;
//    let mut early_exit = 0;
//    let mut correct = 0;
//    println!("all {}",testloader.xss.len());
//    for i in 0..testloader.xss.len() {
//        let xs = testloader.xss.get(i).unwrap();
//        let ys = testloader.yss.get(i).unwrap();
//        let device_main_outputs = device_main.forward(xs);
//        let device_outputs = device_classifier.forward(&device_main_outputs).softmax(-1);
//        let device_t = entropy(&device_outputs);
//        if device_t < 0.2 {
//            early_exit += 1;
//            mid_exit += 1;
//            correct += device_outputs.argmax(-1, false)
//                .eq1(&ys)
//                .sum().f_int64_value(&[]).unwrap();
//        }
//        else {
//            let mut device_to_edge = device_extract.forward(&device_main_outputs);
//            let edge_outputs = edge_classifier.forward(&device_to_edge).softmax(-1);
//            if entropy(&edge_outputs) < 0.1 {
//                correct += edge_outputs.argmax(-1, false)
//                    .eq1(&ys)
//                    .sum().f_int64_value(&[]).unwrap();
//                mid_exit += 1;
//            }
//            else {
//                let edge_to_cloud = edge_main.forward(&device_to_edge);
//                let result = cloud.forward(&edge_to_cloud);
//                correct += result.argmax(-1, false)
//                    .eq1(&ys)
//                    .sum().f_int64_value(&[]).unwrap();
//            }
//        }
//        if i % 100 == 0 {
//            println!("Early exit:{}, mid exit:{}, correct:{}, acc:{}",early_exit,mid_exit,correct,correct as f32 / i as f32);
//        }
//    }
    Ok(())
}

fn to_vec(a:&Tensor) -> Vec<f32> {
    let len = a.numel();
    let mut ret:Vec<u8> = vec![0;len*4];
    a.copy_data(&mut ret,len);
    let new = Tensor::of_data_size(ret.as_ref(),a.size().as_ref(),a.kind());
    a.eq1(&new).sum().print();
    Vec::new()
}

pub struct TestLoader {
    xss:Vec<Tensor>,
    yss:Vec<Tensor>,
}

impl TestLoader {
    pub fn new() -> Self {
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
        TestLoader {
            xss,
            yss,
        }
    }
}