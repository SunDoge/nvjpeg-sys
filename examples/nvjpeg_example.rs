#[macro_use]
extern crate structopt;

use std::ffi;
// use std::ptr;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct DecodeParams {
    #[structopt(short = "i")]
    input_dir: String,

    #[structopt(short = "b")]
    batch_size: usize,

    #[structopt(long = "dev")]
    dev: i32,
}

use nvjpeg_sys::*;

macro_rules! cuda_check_error {
    ($call:expr) => {
        unsafe {
            $call;
            let e = cudaGetLastError();
            if e != cudaError_cudaSuccess {
                panic!(
                    "Cuda failure: {}",
                    ffi::CStr::from_ptr(cudaGetErrorString(e)).to_str().unwrap()
                );
            }
        }
    };
}

macro_rules! nvjpeg_check_error {
    ($call:expr) => {
        let e = $call;
        if e != NVJPEG_STATUS_SUCCESS {
            panic!("nvjpeg failure: error #{}", e);
        }
    };
}

fn main() {
    let params = DecodeParams::from_args();
    println!("{:?}", params);

    let mut props = cudaDeviceProp::default() ;

    cuda_check_error!(cudaSetDevice(params.dev));
    cuda_check_error!(cudaGetDeviceProperties(&mut props, params.dev));
}
