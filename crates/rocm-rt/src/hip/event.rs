use core::mem::{ManuallyDrop, transmute};

use rocm_sys::hip::{
    hipError_t, hipEvent_t, hipEventCreate, hipEventDestroy, hipEventQuery, hipEventRecord,
    hipEventSynchronize,
};

use crate::hip::{HipError, stream::HipStream};

#[repr(transparent)]
#[derive(Debug)]
pub struct HipEvent {
    raw: hipEvent_t,
}

impl HipEvent {
    pub fn create() -> Result<Self, HipError> {
        unsafe { wrap_sys_res!(|event| hipEventCreate((&raw mut event).cast())) }
    }

    pub fn record(&self, stream: &HipStream) -> Result<(), HipError> {
        unsafe { try_err!(hipEventRecord(self.raw, stream.raw.raw), Ok(())) }
    }

    pub fn is_ready(&self) -> Result<bool, HipError> {
        let res = unsafe { hipEventQuery(self.raw) };
        match res {
            hipError_t::hipSuccess => Ok(true),
            hipError_t::hipErrorNotReady => Ok(false),
            e => Err(unsafe { transmute::<hipError_t, HipError>(e) }),
        }
    }

    pub fn sync(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipEventSynchronize(self.raw), Ok(())) }
    }

    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe {
            handle.destroy_unchecked()?;
        }

        Ok(())
    }

    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipEventDestroy(self.raw), Ok(())) }
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = self.destroy_unchecked();
        }
    }
}
