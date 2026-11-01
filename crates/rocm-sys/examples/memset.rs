use std::ptr;

use rocm_sys::hip::{
    hipError_t, hipFree, hipGetDevice, hipMalloc, hipMemsetAsync, hipSetDevice, hipStream_t,
    hipStreamCreate, hipStreamDestroy, hipStreamSynchronize,
};

fn main() {
    unsafe {
        let mut device = -1;
        check(hipGetDevice(&mut device), "hipDeviceGet");
        if device == -1 {
            unreachable!("device unset with success");
        }

        check(hipSetDevice(device), "hipSetDevice");

        let mut stream: hipStream_t = ptr::null_mut();
        check(hipStreamCreate(&mut stream), "hipStreamCreate");

        let mut ptr = ptr::null_mut();
        check(hipMalloc(&mut ptr, 4096), "hipMalloc");

        let err = hipMemsetAsync(ptr, 0, 4096, stream);

        println!("hipMemsetAsync = {err:?}");

        check(err, "hipMemsetAsync");

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
