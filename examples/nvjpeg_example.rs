#[macro_use]
extern crate structopt;

use nvjpeg_sys::*;
use std::ffi;
use std::ptr;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    #[structopt(short = "i")]
    input_dir: String,

    #[structopt(short = "b")]
    batch_size: i32,

    #[structopt(long = "dev")]
    dev: i32,

    #[structopt(short = "w")]
    warmup: i32,

    #[structopt(long = "fmt")]
    fmt: nvjpegOutputFormat_t,
}

struct DecodeParams {
    nvjpeg_state: nvjpegJpegState_t,
    nvjpeg_handle: nvjpegHandle_t,
    stream: cudaStream_t,
}

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
        let e = unsafe { $call };

        if e != nvjpegStatus_t_NVJPEG_STATUS_SUCCESS {
            panic!("nvjpeg failure: error #{}", e);
        }
    };
}

fn main() {
    let opt = Opt::from_args();
    let mut params = DecodeParams {
        nvjpeg_state: ptr::null_mut(),
        nvjpeg_handle: ptr::null_mut(),
        stream: ptr::null_mut(),
    };
    println!("{:?}", opt);

    let mut props = cudaDeviceProp::default();

    cuda_check_error!(cudaSetDevice(opt.dev));
    cuda_check_error!(cudaGetDeviceProperties(&mut props, opt.dev));
    println!(
        "Using GPU {} ({}, {} SMs, {} th/SM max, CC {}.{}, ECC {})",
        opt.dev,
        unsafe { ffi::CStr::from_ptr(props.name.as_ptr()) }
            .to_str()
            .unwrap(),
        props.multiProcessorCount,
        props.maxThreadsPerMultiProcessor,
        props.major,
        props.minor,
        if props.ECCEnabled != 0 { "on" } else { "off" }
    );

    cuda_check_error!(cudaFree(0 as *mut ffi::c_void));

    let mut dev_allocator = nvjpegDevAllocator_t {
        dev_malloc: Some(dev_malloc),
        dev_free: Some(dev_free),
    };

    nvjpeg_check_error!(nvjpegCreate(
        nvjpegBackend_t_NVJPEG_BACKEND_DEFAULT,
        &mut dev_allocator,
        &mut params.nvjpeg_handle
    ));
    nvjpeg_check_error!(nvjpegJpegStateCreate(params.nvjpeg_handle, &mut params.nvjpeg_state));
    nvjpeg_check_error!(nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state, opt.batch_size, 1, opt.fmt));
}

unsafe extern "C" fn dev_malloc(p: *mut *mut ffi::c_void, s: usize) -> i32 {
    cudaMalloc(p, s) as i32
}

unsafe extern "C" fn dev_free(p: *mut ffi::c_void) -> i32 {
    cudaFree(p) as i32
}

