use rocm_sys::hip::{hipStream_t, hipStreamCreate};

use crate::hip::HipError;

pub struct Stream {
    stream: hipStream_t,
}

impl Stream {
    pub fn new() -> Result<Self, HipError> {
        unsafe { wrap_sys_res!(|stream| hipStreamCreate((&raw mut stream).cast())) }
    }

    pub fn into_raw(self) -> hipStream_t {
        self.stream
    }

    pub unsafe fn from_raw(stream: hipStream_t) -> Self {
        Self { stream }
    }
}
