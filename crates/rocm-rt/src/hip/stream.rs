use core::{
    ffi::{c_int, c_ulonglong},
    mem::{ManuallyDrop, transmute},
};

use rocm_sys::hip::{
    hipError_t, hipMemcpyDtoDAsync, hipMemcpyDtoHAsync, hipMemcpyHtoDAsync, hipStream_t,
    hipStreamCreate, hipStreamDestroy, hipStreamGetDevice, hipStreamGetId, hipStreamGetPriority,
    hipStreamQuery, hipStreamSynchronize,
};

use crate::hip::{
    HipError,
    device::Device,
    memory::{Buffer, DevMapped},
};

#[derive(Debug)]
pub struct Stream {
    pub(crate) raw: hipStream_t,
}

impl Stream {
    pub fn create() -> Result<Self, HipError> {
        unsafe { wrap_sys_res!(|stream| hipStreamCreate((&raw mut stream).cast())) }
    }

    pub fn into_raw(self) -> hipStream_t {
        self.raw
    }

    pub unsafe fn from_raw(raw: hipStream_t) -> Self {
        Self { raw }
    }

    pub fn copy_htod(
        &self,
        src: &DevMapped,
        dst: &Buffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > src.size)
            || dst_off.checked_add(len).is_none_or(|end| end > dst.size as usize)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.copy_htod_unchecked(src, dst, src_off, dst_off, len)
        }
    }

    pub unsafe fn copy_htod_unchecked(
        &self,
        src: &DevMapped,
        dst: &Buffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoDAsync(
                    dst.ptr.add(dst_off).cast(),
                    src.ptr.add(src_off).cast(),
                    len,
                    self.raw,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_dtoh(
        &self,
        src: &Buffer,
        dst: &DevMapped,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > src.size as usize)
            || dst_off.checked_add(len).is_none_or(|end| end > dst.size)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.copy_dtoh_unchecked(src, dst, src_off, dst_off, len)
        }
    }

    pub unsafe fn copy_dtoh_unchecked(
        &self,
        src: &Buffer,
        dst: &DevMapped,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoHAsync(
                    dst.ptr.add(dst_off).cast(),
                    src.ptr.add(src_off).cast(),
                    len,
                    self.raw,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_dtod(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > src.size as usize)
            || dst_off.checked_add(len).is_none_or(|end| end > dst.size as usize)
            || src.ptr == dst.ptr
            || src.dev != dst.dev
        {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.copy_dtod_unchecked(src, dst, src_off, dst_off, len)
        }
    }

    pub unsafe fn copy_dtod_unchecked(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoDAsync(
                    dst.ptr.add(dst_off).cast(),
                    src.ptr.add(src_off).cast(),
                    len,
                    self.raw,
                ),
                Ok(())
            )
        }
    }

    /// Destroys this stream
    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe {
            handle.destroy_unchecked()?;
        }

        Ok(())
    }

    /// Destroys this stream
    ///
    /// # Safety
    ///
    /// Must not destroyed twice via dropping, calling this function twice, or calling [`Self::destroy`].
    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipStreamDestroy(self.raw), Ok(())) }
    }

    /// Returns `true` if the stream has finished all queued tasks
    pub fn query(&self) -> Result<bool, HipError> {
        let res = unsafe { hipStreamQuery(self.raw) };
        match res {
            hipError_t::hipSuccess => Ok(true),
            hipError_t::hipErrorNotReady => Ok(false),
            e => Err(unsafe { transmute::<hipError_t, HipError>(e) }),
        }
    }

    /// Waits until the stream has finished all queued tasks
    pub fn sync(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipStreamSynchronize(self.raw), Ok(())) }
    }

    pub fn get_id(&self) -> Result<u64, HipError> {
        unsafe { wrap_sys_res!(|id| hipStreamGetId(self.raw, (&raw mut id).cast())) }
            .map(|id: c_ulonglong| id as u64)
    }

    pub fn get_priority(&self) -> Result<i32, HipError> {
        unsafe { wrap_sys_res!(|prio| hipStreamGetPriority(self.raw, (&raw mut prio).cast())) }
            .map(|prio: c_int| prio as i32)
    }

    pub fn get_dev(&self) -> Result<Device, HipError> {
        unsafe { wrap_sys_res!(|dev| hipStreamGetDevice(self.raw, (&raw mut dev).cast())) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = self.destroy_unchecked();
        }
    }
}
