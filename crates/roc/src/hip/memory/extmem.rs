use core::mem::ManuallyDrop;
use std::fs::File;

use alloc::sync::Arc;
use rocm_sys::hip::hipExternalMemory_t;

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
