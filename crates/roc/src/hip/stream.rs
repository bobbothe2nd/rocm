use alloc::sync::Arc;
use core::{
    ffi::{c_int, c_ulonglong},
    mem::transmute,
    ptr,
};
use rocm_sys::hip::{
    hipError_t, hipFreeAsync, hipMemcpyDtoDAsync, hipMemcpyDtoHAsync, hipMemcpyHtoDAsync,
    hipStream_t, hipStreamCreate, hipStreamDestroy, hipStreamGetDevice, hipStreamGetId,
    hipStreamGetPriority, hipStreamQuery, hipStreamSynchronize,
};

use crate::hip::{
    HipError,
    device::{Device, DeviceAttr},
    memory::{Buffer, HostSlice, HostSliceMut},
};

#[repr(C)]
#[derive(Debug)]
pub struct HipStream {
    pub(crate) raw: hipStream_t,
    pub(crate) has_async_alloc: bool,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn create() -> Result<Arc<Self>, HipError> {
        let raw: Result<hipStream_t, HipError> = unsafe {
            wrap_sys_res!(|stream: hipStream_t| hipStreamCreate(stream.as_mut_ptr().cast()))
        };
        unsafe { Self::from_raw(raw?) }
    }

    #[inline(always)]
    pub fn as_raw(&self) -> hipStream_t {
        self.raw
    }

    #[inline(always)]
    pub fn into_raw(self: Arc<Self>) -> hipStream_t {
        self.raw
    }

    #[inline(always)]
    pub unsafe fn from_raw(raw: hipStream_t) -> Result<Arc<Self>, HipError> {
        let dev_raw: Result<i32, HipError> =
            unsafe { wrap_sys_res!(|dev| hipStreamGetDevice(raw, dev.as_mut_ptr())) };
        let dev = unsafe { Device::from_raw(dev_raw?) };
        let has_async_alloc = dev
            .get_attr(DeviceAttr::MemoryPoolsSupported)
            .is_ok_and(|x| x > 0);
        Ok(Arc::new(Self {
            raw,
            has_async_alloc,
        }))
    }

    pub unsafe fn copy_htod(
        &self,
        src: &HostSlice<'_>,
        dst: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > src.size as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.ptr.len() as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_htod_unchecked(src, dst, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_htod_unchecked(
        &self,
        src: &HostSlice<'_>,
        dst: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoDAsync(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.add(src_off as usize).cast(),
                    len as usize,
                    self.raw,
                ),
                Ok(())
            )
        }
    }

    pub unsafe fn copy_dtoh(
        &self,
        src: &Buffer,
        dst: &HostSliceMut<'_>,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > src.ptr.len() as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.size as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_dtoh_unchecked(src, dst, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_dtoh_unchecked(
        &self,
        src: &Buffer,
        dst: &HostSliceMut<'_>,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoHAsync(
                    dst.ptr.add(dst_off as usize) as *mut _,
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                    self.raw,
                ),
                Ok(())
            )
        }
    }

    pub unsafe fn copy_dtod(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > src.ptr.len() as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.ptr.len() as u32)
            || ptr::eq(src.ptr, dst.ptr)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_dtod_unchecked(src, dst, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_dtod_unchecked(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoDAsync(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                    self.raw,
                ),
                Ok(())
            )
        }
    }

    /// Asynchronously free a GPU buffer
    ///
    /// # Safety
    ///
    /// The buffer must not be dropped or freed again
    #[inline(always)]
    pub unsafe fn free_unchecked(&self, buf: &Buffer) -> Result<(), HipError> {
        unsafe { try_err!(hipFreeAsync(buf.ptr.cast(), self.raw,), Ok(())) }
    }

    /// Returns `true` if the stream has finished all queued tasks
    pub fn is_ready(&self) -> Result<bool, HipError> {
        let res = unsafe { hipStreamQuery(self.raw) };
        match res {
            hipError_t::hipSuccess => Ok(true),
            hipError_t::hipErrorNotReady => Ok(false),
            e => Err(unsafe { transmute::<hipError_t, HipError>(e) }),
        }
    }

    /// Waits until the stream has finished all queued tasks
    #[inline(always)]
    pub fn sync(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipStreamSynchronize(self.raw), Ok(())) }
    }

    #[inline(always)]
    pub fn get_id(&self) -> Result<u64, HipError> {
        unsafe { wrap_sys_res!(|id| hipStreamGetId(self.raw, id.as_mut_ptr())) }
            .map(|id: c_ulonglong| id as u64)
    }

    #[inline(always)]
    pub fn get_priority(&self) -> Result<i32, HipError> {
        unsafe { wrap_sys_res!(|prio| hipStreamGetPriority(self.raw, prio.as_mut_ptr())) }
            .map(|prio: c_int| prio as i32)
    }

    #[inline(always)]
    pub fn get_dev(&self) -> Result<Device, HipError> {
        unsafe {
            wrap_sys_res!(|dev: Device| hipStreamGetDevice(self.raw, dev.as_mut_ptr().cast()))
        }
    }

    #[inline(always)]
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        self.raw == other.raw
    }

    #[inline(always)]
    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipStreamDestroy(self.raw), Ok(())) }
    }
}

impl Drop for HipStream {
    #[inline(always)]
    fn drop(&mut self) {
        log_if_err!("failed to destroy HipStream": unsafe { self.destroy_unchecked() });
    }
}
