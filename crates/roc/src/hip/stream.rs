use alloc::sync::Arc;
use core::{
    ffi::{c_int, c_ulonglong},
    mem::transmute,
};
use rocm_sys::hip::{
    hipError_t, hipStream_t, hipStreamCreateWithFlags, hipStreamCreateWithPriority,
    hipStreamDestroy, hipStreamGetDevice, hipStreamGetId, hipStreamGetPriority, hipStreamQuery,
    hipStreamSynchronize,
};

use crate::{
    aux::RocHandle,
    hip::{HipError, device::Device},
};

#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct HipStream {
    pub(crate) raw: hipStream_t,
    pub(crate) dev: Device,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn create(flags: StreamKind) -> Result<Arc<Self>, HipError> {
        let raw: Result<hipStream_t, HipError> = unsafe {
            wrap_sys_res!(|stream: hipStream_t| hipStreamCreateWithFlags(
                stream.as_mut_ptr().cast(),
                flags as u32
            ))
        };
        unsafe { Self::from_raw(raw?) }
    }

    /// lower value = higher priority
    pub fn create_with_prio(flags: StreamKind, prio: i32) -> Result<Arc<Self>, HipError> {
        let raw: Result<hipStream_t, HipError> = unsafe {
            wrap_sys_res!(|stream: hipStream_t| hipStreamCreateWithPriority(
                stream.as_mut_ptr().cast(),
                flags as u32,
                prio
            ))
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
        let dev = unsafe { Device::from_raw(dev_raw?)? };
        Ok(Arc::new(Self { raw, dev }))
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
    pub fn get_dev(&self) -> Device {
        // unsafe { wrap_sys_res!(|dev: Device| hipStreamGetDevice(self.raw, dev.as_mut_ptr().cast())) }
        self.dev
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

impl RocHandle for HipStream {
    type Error = HipError;
}

impl Drop for HipStream {
    #[inline(always)]
    fn drop(&mut self) {
        log_if_err!("failed to destroy HipStream": unsafe { self.destroy_unchecked() });
    }
}

#[repr(u32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone)]
pub enum StreamKind {
    /// Default stream creation flags. These are used with `hipStreamCreate()`.
    Default = 0,
    /// Stream does not implicitly synchronize with null stream.
    NonBlocking = 1,
}
