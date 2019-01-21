#[macro_use]
extern crate structopt;

use nvjpeg_sys::*;
use std::ffi;
use std::ptr;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use walkdir::WalkDir;

type FileNames = Vec<String>;
type FileData = Vec<Vec<u8>>;

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

    #[structopt(short = "t")]
    total_images: i32,
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
    let mut opt = Opt::from_args();
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
    nvjpeg_check_error!(nvjpegJpegStateCreate(
        params.nvjpeg_handle,
        &mut params.nvjpeg_state
    ));
    nvjpeg_check_error!(nvjpegDecodeBatchedInitialize(
        params.nvjpeg_handle,
        params.nvjpeg_state,
        opt.batch_size,
        1,
        opt.fmt
    ));
    let image_names: Vec<String> = WalkDir::new(&opt.input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().into())
        .collect();

    if opt.total_images == -1 {
        opt.total_images = image_names.len() as i32;
    } else if opt.total_images % opt.batch_size != 0 {
        opt.total_images = (opt.total_images / opt.batch_size) * opt.batch_size;
        println!(
            "Changing total_images number to {} to be multiple of batch_size - {}",
            opt.total_images, opt.batch_size
        );
    }

    println!(
        "Decoding images in directory: {}, total {}, batchsize {}",
        &opt.input_dir, opt.total_images, opt.batch_size
    );

    let total = process_images(&image_names, &opt, &mut params).unwrap();

    println!("Total decoding time: {}", total);
    println!(
        "Avg decoding time per image: {}",
        total / opt.total_images as f64
    );
    println!("Avg images per sec: {}", opt.total_images as f64 / total);
    println!(
        "Avg decoding time per batch: {}",
        total / ((opt.total_images + opt.batch_size - 1) as f64 / opt.batch_size as f64)
    );

    nvjpeg_check_error!(nvjpegJpegStateDestroy(params.nvjpeg_state));
    nvjpeg_check_error!(nvjpegDestroy(params.nvjpeg_handle));

    cuda_check_error!(cudaDeviceReset());
}

fn process_images(
    image_names: &[String],
    opt: &Opt,
    params: &mut DecodeParams,
) -> Result<f64, String> {
    let mut file_data: FileData = Vec::with_capacity(opt.batch_size as usize);
    let mut file_len: Vec<usize> = Vec::with_capacity(opt.batch_size as usize);
    let mut current_names: FileNames = Vec::with_capacity(opt.batch_size as usize);
    let mut widths: Vec<i32> = Vec::with_capacity(opt.batch_size as usize);
    let mut heights: Vec<i32> = Vec::with_capacity(opt.batch_size as usize);

    cuda_check_error!(cudaStreamCreateWithFlags(
        &mut params.stream,
        cudaStreamNonBlocking
    ));

    let mut total_processed = 0;

    let mut iout: Vec<nvjpegImage_t> = Vec::with_capacity(opt.batch_size as usize);
    let mut isz: Vec<nvjpegImage_t> = Vec::with_capacity(opt.batch_size as usize);

    for i in 0..iout.len() {
        for c in 0..NVJPEG_MAX_COMPONENT as usize {
            iout[i].channel[c] = ptr::null_mut();
            iout[i].pitch[c] = 0;
            isz[i].pitch[c] = 0;
        }
    }

    Ok(0.0)
}

unsafe extern "C" fn dev_malloc(p: *mut *mut ffi::c_void, s: usize) -> i32 {
    cudaMalloc(p, s) as i32
}

unsafe extern "C" fn dev_free(p: *mut ffi::c_void) -> i32 {
    cudaFree(p) as i32
}
