use std::ptr;

use rocm_sys::hip::{
    hipError_t, hipFree, hipGetDevice, hipMalloc, hipMemsetAsync, hipSetDevice, hipStream_t,
    hipStreamCreate, hipStreamDestroy, hipStreamSynchronize,
};

fn main() {
    unsafe {
        // Current device.
        let mut device = 0;
        check(hipGetDevice(&mut device), "hipDeviceGet");
        println!("device = {device}");

        check(hipSetDevice(device), "hipSetDevice");

        // Raw stream.
        let mut stream: hipStream_t = ptr::null_mut();
        check(hipStreamCreate(&mut stream), "hipStreamCreate");
        println!("stream = {stream:p}");

        // Raw device allocation.
        let mut ptr = ptr::null_mut();
        check(hipMalloc(&mut ptr, 4096), "hipMalloc");
        println!("ptr = {ptr:p}");

        // The actual test.
        println!("hipMemsetAsync...");

        let err = hipMemsetAsync(ptr, 0, 4096, stream);

        println!("hipMemsetAsync = {err:?}");

        // Don't continue if enqueue itself failed.
        check(err, "hipMemsetAsync");

        println!("hipStreamSynchronize...");

        check(hipStreamSynchronize(stream), "hipStreamSynchronize");

        println!("success");

        check(hipFree(ptr), "hipFree");
        check(hipStreamDestroy(stream), "hipStreamDestroy");
    }
}

fn check(err: hipError_t, name: &str) {
    if err != hipError_t::hipSuccess {
        panic!("{name}: {err:?}");
    }
}
