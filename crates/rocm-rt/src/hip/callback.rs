use core::{ffi::c_void, mem::transmute};

use rocm_sys::hip::hipLaunchHostFunc;

use crate::hip::{HipError, stream::Stream};

impl Stream {
    /// Launches a host callback to run on the stream asynchronously
    ///
    /// # Safety
    ///
    /// This function is only as safe as `func`
    pub unsafe fn callback(&self, func: Callback, data: *mut u8) -> Result<(), HipError> {
        unsafe {
            try_err!(hipLaunchHostFunc(self.raw, Some(func.func), data.cast()), Ok(()))
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Callback {
    func: unsafe extern "C" fn(*mut c_void),
}

impl Callback {
    pub const fn new(func: unsafe extern "C" fn(*mut u8)) -> Self {
        let func = unsafe {
            transmute::<
                unsafe extern "C" fn(*mut u8),
                unsafe extern "C" fn(*mut c_void),
            >(func)
        };

        Self { func }
    }
}
