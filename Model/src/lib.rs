#[macro_use]
extern crate serde;

use serde::{Deserialize,Serialize};
use tch::{Tensor, Kind};
use std::io::Bytes;
use actix::prelude::*;
use std::net::SocketAddr;
use std::path::Path;

#[derive(Serialize,Deserialize,Debug,Message)]
pub enum DeviceRequest {
    Eval {
        id: usize,
        tensor: TensorRepresentation
    }
}

#[derive(Serialize,Deserialize,Debug,Message)]
pub enum EdgeRequest {
    Eval {
        client_address: SocketAddr,
        id: usize,
        tensor: TensorRepresentation
    }
}

#[derive(Serialize,Deserialize,Debug,Message)]
pub struct EdgeResponse {
    pub id:usize,
    pub prob: TensorRepresentation
}

#[derive(Serialize,Deserialize,Debug,Message)]
pub struct CloudResponse {
    pub client_address:SocketAddr,
    pub id:usize,
    pub prob: TensorRepresentation
}

#[derive(Debug,Serialize,Deserialize,Clone)]
pub struct TensorRepresentation {
    pub size:Vec<i64>,
    pub data:Vec<u8>,
    pub kind:TensorKind
}


// TODO: zero copy
impl TensorRepresentation {
    pub fn from_tensor(tensor:&Tensor)-> Self {
        let kind = TensorKind::from_tch(tensor.kind());
        let size = tensor.size();
        let len = tensor.numel();
        let mut data:Vec<u8> = vec![0;len*4];
        tensor.copy_data(&mut data,len);
        Self {
            size,
            data,
            kind
        }
    }

    pub fn to_tensor(&self)-> Tensor {
        let kind = self.kind.into_tch();
        Tensor::of_data_size(&self.data,&self.size,kind)
    }
}

#[derive(Debug,Serialize,Deserialize,Copy,Clone)]
pub enum TensorKind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
}

impl TensorKind {
    pub fn from_tch(kind:tch::kind::Kind) -> Self {
        match kind {
            Kind::Uint8 => Self::Uint8,
            Kind::Int8 => Self::Int8,
            Kind::Int16 => unimplemented!(),
            Kind::Int => Self::Int,
            Kind::Int64 => Self::Int64,
            Kind::Half => unimplemented!(),
            Kind::Float => Self::Float,
            Kind::Double => Self::Double,
            Kind::ComplexHalf => unimplemented!(),
            Kind::ComplexFloat => unimplemented!(),
            Kind::ComplexDouble => unimplemented!(),
        }
    }

    pub fn into_tch(self) -> tch::Kind {
        match self {
            TensorKind::Uint8 => Kind::Uint8,
            TensorKind::Int8 => Kind::Int8,
            TensorKind::Int16 => Kind::Int16,
            TensorKind::Int => Kind::Int,
            TensorKind::Int64 => Kind::Int64,
            TensorKind::Half => Kind::Half,
            TensorKind::Float => Kind::Float,
            TensorKind::Double => Kind::Double,
            TensorKind::ComplexHalf => Kind::ComplexHalf,
            TensorKind::ComplexFloat => Kind::ComplexFloat,
            TensorKind::ComplexDouble => Kind::ComplexDouble,
        }
    }
}

pub fn read_npy<T:AsRef<Path>>(dir: T) -> Vec<Tensor> {
    let mut files = Vec::new();
    let mut tensors = Vec::new();
    visit_dirs(dir.as_ref(), &mut files).unwrap();
    for file in files {
        tensors.push(Tensor::read_npy(file.path()).unwrap());
    }
    tensors
}

fn visit_dirs(dir: &Path, files: &mut Vec<std::fs::DirEntry>) -> Result<(),std::io::Error> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else if entry
                .file_name()
                .to_str()
                .map_or(false, |s| s.ends_with(".npy"))
            {
                files.push(entry);
            }
        }
    }
    Ok(())
}

fn main() {
    let bytes:Vec<u8> = Tensor::new().into();
    println!("Hello, world!");
}
