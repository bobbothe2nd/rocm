use core::mem::{ManuallyDrop, transmute};

use bitflags::bitflags;
use rocm_sys::hip::{
    hipError_t, hipEvent_t, hipEventCreate, hipEventCreateWithFlags, hipEventDestroy,
    hipEventQuery, hipEventRecord, hipEventRecordWithFlags, hipEventSynchronize,
};

use crate::hip::{HipError, stream::HipStream};

#[repr(transparent)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone)]
pub struct HipEvent {
    raw: hipEvent_t,
}

unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl HipEvent {
    pub fn create(flags: Option<EventFlags>) -> Result<Self, HipError> {
        if let Some(flags) = flags {
            unsafe {
                wrap_sys_res!(|event| hipEventCreateWithFlags(
                    (&raw mut event).cast(),
                    flags.bits()
                ))
            }
        } else {
            unsafe { wrap_sys_res!(|event| hipEventCreate((&raw mut event).cast())) }
        }
    }

    pub fn record(
        &self,
        stream: &HipStream,
        flags: Option<EventRecordKind>,
    ) -> Result<(), HipError> {
        if let Some(flags) = flags {
            unsafe {
                try_err!(
                    hipEventRecordWithFlags(self.raw, stream.raw, flags as u32),
                    Ok(())
                )
            }
        } else {
            unsafe { try_err!(hipEventRecord(self.raw, stream.raw), Ok(())) }
        }
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

bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
    #[derive(Debug, Clone)]
    pub struct EventFlags: u32 {
        /// Default flags.
        const DEFAULT = 0;
        /// Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.
        const BLOCKING_SYNC = 1;
        /// Disable event's capability to record timing information. May improve performance.
        const DISABLE_TIMING = 2;
        /// Event can support IPC. hipEventDisableTiming also must be set.
        const INTERPROCESS = 4;
    }
}

#[repr(u32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone)]
pub enum EventRecordKind {
    /// Default flags.
    Default = 0,
    /// Event is captured in the graph as an external event node when performing stream capture.
    External = 1,
}

#[repr(u32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone)]
pub enum EventWaitKind {
    /// Default flags.
    Default = 0,
    /// Wait is captured in the graph as an external event node when performing stream capture.
    External = 1,
}
