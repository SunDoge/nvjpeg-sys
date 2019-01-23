#[macro_use]
extern crate structopt;

use nvjpeg_sys::*;
use std::ffi;
use std::fs::File;
use std::io::Read;
use std::ptr;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use walkdir::WalkDir;

type FileNames = Vec<String>;
type FileData = Vec<Vec<u8>>;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    #[structopt(name = "images_dir", short = "i")]
    /// Path to single image or directory of images
    input_dir: String,

    #[structopt(short = "b")]
    /// Decode images from input by batches of specified size
    batch_size: i32,

    #[structopt(name = "device_id", long = "dev")]
    /// Which device to use for decoding
    dev: i32,

    #[structopt(name = "warmup_iterations", short = "w")]
    /// Run this amount of batches first without measuring performance
    warmup: i32,

    #[structopt(name = "output_format", long = "fmt")]
    fmt: nvjpegOutputFormat_t,

    #[structopt(short = "t")]
    /// Decode this much images, if there are less images
    /// in the input than total images, decoder will loop over the input
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
    let mut image_names: Vec<String> = WalkDir::new(&opt.input_dir)
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

    let total = process_images(&mut image_names, &opt, &mut params).unwrap();

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
    image_names: &mut Vec<String>,
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

    let mut test_time = 0.0;
    let mut warmup = 0;

    while total_processed < opt.total_images {
        read_next_batch(
            image_names,
            opt.batch_size,
            &mut file_data,
            &file_len,
            &mut current_names,
        )
        .unwrap();
    }

    Ok(0.0)
}

fn read_next_batch(
    image_names: &mut FileNames,
    batch_size: i32,
    raw_data: &mut FileData,
    raw_len: &Vec<usize>,
    current_names: &FileNames,
) -> Option<f64> {

    image_names.retain(|image_name| File::open(image_name).is_ok());

    // while counter < batch_size as usize {
    //     if let Some(image_name) = cur_iter.next() {
    //         let input = File::open(image_name);
    //         if input.is_err() {
    //             eprintln!(
    //                 "Cannot open image: {}, removing it from image list",
    //                 image_name
    //             );
    //             continue;
    //         }
    //         let mut input = input.unwrap();
    //         let file_size = input.metadata().unwrap().len() as usize;

    //         if raw_data[counter].len() < file_size {
    //             raw_data[counter].resize(file_size, b'0');
    //         }

    //         match input.read_exact(&mut raw_data[counter]) {
    //             Err(_) => {
    //                 eprintln!(
    //                     "Cannot read from file: {}, removing it from image list",
    //                     image_name
    //                 );
    //                 image_names.remove_item(image_name);
    //             }
    //             Ok(_) => {}
    //         };
    //     } else {
    //         cur_iter = image_names.iter_mut();
    //     }
    // }

    None
}

unsafe extern "C" fn dev_malloc(p: *mut *mut ffi::c_void, s: usize) -> i32 {
    cudaMalloc(p, s) as i32
}

unsafe extern "C" fn dev_free(p: *mut ffi::c_void) -> i32 {
    cudaFree(p) as i32
}
