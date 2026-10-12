use core::{
    ffi::{c_int, c_ulonglong},
    mem::transmute,
};
use alloc::rc::Rc;
use rocm_sys::hip::{
    hipError_t, hipFreeAsync, hipMemcpyDtoDAsync, hipMemcpyDtoHAsync, hipMemcpyHtoDAsync, hipStream_t, hipStreamCreate, hipStreamDestroy, hipStreamGetDevice, hipStreamGetId, hipStreamGetPriority, hipStreamQuery, hipStreamSynchronize,
};

use crate::hip::{
    HipError, device::Device, memory::{Buffer, RefMapped},
};

#[derive(Debug, Clone)]
pub struct Stream {
    pub(crate) raw: Rc<StreamInner>,
}

impl Stream {
    pub fn create() -> Result<Self, HipError> {
        unsafe { wrap_sys_res!(|stream| hipStreamCreate((&raw mut stream).cast())) }
    }

    #[inline(always)]
    pub fn as_raw(&self) -> hipStream_t {
        self.raw.raw
    }

    #[inline(always)]
    pub fn into_raw(self) -> hipStream_t {
        self.raw.raw
    }

    #[inline(always)]
    pub unsafe fn from_raw(raw: hipStream_t) -> Self {
        Self { raw: Rc::new(StreamInner { raw }) }
    }

    pub fn copy_htod(
        &self,
        src: &RefMapped<'_>,
        dst: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > src.size())
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.size)
            || dst.dev != self.get_dev()?
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_htod_unchecked(src, dst, src_off, dst_off, len) }
    }

    pub unsafe fn copy_htod_unchecked(
        &self,
        src: &RefMapped<'_>,
        dst: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoDAsync(
                    dst.ptr.add(dst_off as usize).cast(),
                    src.data.as_ptr().add(src_off as usize).cast(),
                    len as usize,
                    self.raw.raw,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_dtoh(
        &self,
        src: &Buffer,
        dst: &RefMapped<'_>,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > src.size)
            || dst_off.checked_add(len).is_none_or(|end| end > dst.size())
            || src.dev != self.get_dev()?
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_dtoh_unchecked(src, dst, src_off, dst_off, len) }
    }

    pub unsafe fn copy_dtoh_unchecked(
        &self,
        src: &Buffer,
        dst: &RefMapped<'_>,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoHAsync(
                    dst.data.as_ptr().add(dst_off as usize) as *mut _,
                    src.ptr.add(src_off as usize).cast(),
                    len as usize,
                    self.raw.raw,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_dtod(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > src.size)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.size)
            || src.ptr == dst.ptr
            || src.dev != dst.dev
            || src.dev != self.get_dev()?
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_dtod_unchecked(src, dst, src_off, dst_off, len) }
    }

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
                    dst.ptr.add(dst_off as usize).cast(),
                    src.ptr.add(src_off as usize).cast(),
                    len as usize,
                    self.raw.raw,
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
    pub unsafe fn free(&self, buf: &Buffer) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipFreeAsync(
                    buf.ptr.cast(),
                    self.raw.raw,
                ),
                Ok(())
            )
        }
    }

    /// Returns `true` if the stream has finished all queued tasks
    pub fn is_ready(&self) -> Result<bool, HipError> {
        let res = unsafe { hipStreamQuery(self.raw.raw) };
        match res {
            hipError_t::hipSuccess => Ok(true),
            hipError_t::hipErrorNotReady => Ok(false),
            e => Err(unsafe { transmute::<hipError_t, HipError>(e) }),
        }
    }

    /// Waits until the stream has finished all queued tasks
    pub fn sync(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipStreamSynchronize(self.raw.raw), Ok(())) }
    }

    pub fn get_id(&self) -> Result<u64, HipError> {
        unsafe { wrap_sys_res!(|id| hipStreamGetId(self.raw.raw, (&raw mut id).cast())) }
            .map(|id: c_ulonglong| id as u64)
    }

    pub fn get_priority(&self) -> Result<i32, HipError> {
        unsafe { wrap_sys_res!(|prio| hipStreamGetPriority(self.raw.raw, (&raw mut prio).cast())) }
            .map(|prio: c_int| prio as i32)
    }

    pub fn get_dev(&self) -> Result<Device, HipError> {
        unsafe { wrap_sys_res!(|dev| hipStreamGetDevice(self.raw.raw, (&raw mut dev).cast())) }
    }
}

#[derive(Debug)]
pub struct StreamInner {
    pub(crate) raw: hipStream_t,
}

impl StreamInner {
    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipStreamDestroy(self.raw), Ok(())) }
    }
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.destroy_unchecked();
        }
    }
}
