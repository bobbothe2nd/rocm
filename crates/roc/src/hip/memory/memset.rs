use rocm_sys::hip::{
    hipMemsetD8, hipMemsetD8Async, hipMemsetD16, hipMemsetD16Async, hipMemsetD32, hipMemsetD32Async,
};

use crate::hip::{HipError, stream::HipStream};

pub unsafe trait Memset {
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError>;
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError>;
}

unsafe impl Memset for u8 {
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemsetD8Async(dst.cast(), value, size, stream.raw),
                Ok(())
            )
        }
    }

    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { try_err!(hipMemsetD8(dst.cast(), value, count), Ok(())) }
    }
}

unsafe impl Memset for u16 {
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemsetD16Async(dst.cast(), value, size, stream.raw),
                Ok(())
            )
        }
    }

    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { try_err!(hipMemsetD16(dst.cast(), value, count), Ok(())) }
    }
}

unsafe impl Memset for i32 {
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemsetD32Async(dst.cast(), value, size, stream.raw),
                Ok(())
            )
        }
    }

    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { try_err!(hipMemsetD32(dst.cast(), value, count), Ok(())) }
    }
}
