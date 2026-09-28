use core::{ffi::c_void, mem::transmute};

use rocm_sys::hip::hipLaunchHostFunc;

use crate::hip::{HipError, stream::Stream};

impl Stream {
    /// Launches a host callback to run on the stream asynchronously
    ///
    /// # Safety
    ///
    /// This function is only as safe as `func`
    ///
    /// Undefined behavior if capturing in a graph and replaying
    #[cfg(feature = "alloc")]
    pub unsafe fn callback<F>(&self, func: F, data: *mut u8) -> Result<(), HipError>
    where
        F: FnOnce(*mut u8) + Send + 'static,
    {
        use alloc::boxed::Box;

        unsafe extern "C" fn wrapper<F>(data: *mut c_void)
        where
            F: FnOnce(*mut u8),
        {
            let func = unsafe { Box::from_raw(data.cast::<(F, *mut u8)>()) };

            (func.0)(func.1);
        }

        let func = Box::new((func, data));
        let data = Box::into_raw(func);

        unsafe {
            try_err!(
                hipLaunchHostFunc(
                    self.raw,
                    Some(wrapper::<F>),
                    data.cast(),
                ),
                Ok(())
            )
        }
    }

    /// Launches a host callback to run on the stream asynchronously
    ///
    /// # Safety
    ///
    /// This function is only as safe as `func`
    pub unsafe fn launch_host(&self, func: Callback, data: *mut u8) -> Result<(), HipError> {
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
