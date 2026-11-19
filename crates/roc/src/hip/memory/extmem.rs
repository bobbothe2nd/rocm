use core::mem::ManuallyDrop;
use std::fs::File;

use alloc::sync::Arc;
use rocm_sys::hip::{hipDestroyExternalMemory, hipExternalMemory_t};

use crate::hip::{HipError, stream::HipStream};

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct ExtBuffer {
    ptr: hipExternalMemory_t,
    size: usize,
    ctx: Arc<HipStream>,
    _file: ManuallyDrop<File>,
}

impl HipStream {
    pub fn import_external_memory(
        self: &Arc<Self>,
        _file: File,
        _size: usize,
    ) -> Result<ExtBuffer, HipError> {
        todo!()
    }
}

impl ExtBuffer {
    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe { handle.destroy_unchecked() }
    }

    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipDestroyExternalMemory(self.ptr), Ok(())) }
    }
}

impl Drop for ExtBuffer {
    fn drop(&mut self) {
        log_if_err!("failed to destroy ExtBuffer": unsafe { self.destroy_unchecked() });

        #[cfg(windows)]
        unsafe {
            ManuallyDrop::<File>::drop(&mut self._file);
        }
    }
}
