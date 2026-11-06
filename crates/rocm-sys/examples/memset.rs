use std::ptr;

use rocm_sys::hip::{
    hipError_t, hipFree, hipGetDevice, hipMalloc, hipMemsetAsync, hipStream_t, hipStreamCreate,
    hipStreamDestroy, hipStreamSynchronize,
};

fn main() {
    println!("trying to get device...");

    let mut device = -1;
    unsafe {
        check(hipGetDevice(&mut device), "hipDeviceGet");
    }
    if device == -1 {
        unreachable!("device unset with success");
    }

    println!("creating stream...");

    let mut stream: hipStream_t = ptr::null_mut();
    unsafe {
        check(hipStreamCreate(&mut stream), "hipStreamCreate");
    }

    println!("allocating buffer...");

    let mut ptr = ptr::null_mut();
    unsafe {
        check(hipMalloc(&mut ptr, 4096), "hipMalloc");
    }

    println!("zeroing buffer...");

    let err = unsafe { hipMemsetAsync(ptr, 0, 4096, stream) };

    println!("  hipMemsetAsync = {err:?}");

    check(err, "hipMemsetAsync");

    println!("synchronizing...");

    unsafe {
        check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    }

    println!("SUCCESS!");

    println!("cleaning up...");

    unsafe {
        check(hipFree(ptr), "hipFree");
        check(hipStreamDestroy(stream), "hipStreamDestroy");
    }
}

fn check(err: hipError_t, name: &str) {
    if err != hipError_t::hipSuccess {
        panic!("{name}: {err:?}");
    }
}
