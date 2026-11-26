use rocm_sys::hip::{
    hipMemsetD8, hipMemsetD8Async, hipMemsetD16, hipMemsetD16Async, hipMemsetD32, hipMemsetD32Async,
};

use crate::hip::{HipError, stream::HipStream};

pub unsafe trait Memset: Copy {
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError>;
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError>;
}

unsafe impl Memset for u8 {
    #[inline(always)]
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

    #[inline(always)]
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { try_err!(hipMemsetD8(dst.cast(), value, count), Ok(())) }
    }
}

unsafe impl Memset for i8 {
    #[inline(always)]
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe { u8::memset_async(dst.cast(), value.cast_unsigned(), size, stream) }
    }

    #[inline(always)]
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { u8::memset_sync(dst.cast(), value.cast_unsigned(), count) }
    }
}

unsafe impl Memset for u16 {
    #[inline(always)]
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

    #[inline(always)]
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { try_err!(hipMemsetD16(dst.cast(), value, count), Ok(())) }
    }
}

unsafe impl Memset for i16 {
    #[inline(always)]
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe { u16::memset_async(dst.cast(), value.cast_unsigned(), size, stream) }
    }

    #[inline(always)]
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { u16::memset_sync(dst.cast(), value.cast_unsigned(), count) }
    }
}

unsafe impl Memset for i32 {
    #[inline(always)]
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

    #[inline(always)]
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { try_err!(hipMemsetD32(dst.cast(), value, count), Ok(())) }
    }
}

unsafe impl Memset for u32 {
    #[inline(always)]
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe { i32::memset_async(dst.cast(), value.cast_signed(), size, stream) }
    }

    #[inline(always)]
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { i32::memset_sync(dst.cast(), value.cast_signed(), count) }
    }
}

unsafe impl Memset for f32 {
    #[inline(always)]
    unsafe fn memset_async(
        dst: *mut Self,
        value: Self,
        size: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe { u32::memset_async(dst.cast(), value.to_bits(), size, stream) }
    }

    #[inline(always)]
    unsafe fn memset_sync(dst: *mut Self, value: Self, count: usize) -> Result<(), HipError> {
        unsafe { u32::memset_sync(dst.cast(), value.to_bits(), count) }
    }
}
