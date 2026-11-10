use alloc::boxed::Box;
use core::{ffi::c_void, mem::transmute};
use rocm_sys::hip::hipLaunchHostFunc;

use crate::hip::{HipError, stream::HipStream};

impl HipStream {
    /// Launches a host callback to run on the stream asynchronously
    ///
    /// # Safety
    ///
    /// The function's lifetimes must last at least until this function is finished.
    ///
    /// Undefined behavior if capturing in a graph and replaying or calling HIP API functions inside the callback.
    pub unsafe fn callback<T, F>(&self, func: F, data: *mut T) -> Result<(), HipError>
    where
        F: FnOnce(*mut T) + Send + 'static,
    {
        unsafe extern "C" fn wrapper<T, F>(data: *mut (F, *mut T))
        where
            F: FnOnce(*mut T),
        {
            let func = unsafe { Box::from_raw(data) };
            (func.0)(func.1);
        }

        let func = Box::new((func, data));
        let data = Box::into_raw(func);

        unsafe { self.launch_host(wrapper::<T, F>, data) }
    }

    /// Launches a host callback to run on the stream asynchronously
    ///
    /// # Safety
    ///
    /// This function is only as safe as `func`.
    /// The function must be safe to send across threads and all lifetimes must last at least until this function is finished.
    ///
    /// Undefined behavior if capturing in a graph and replaying or calling HIP API functions inside the callback.
    #[inline(always)]
    pub unsafe fn launch_host<T>(
        &self,
        func: unsafe extern "C" fn(*mut T),
        data: *mut T,
    ) -> Result<(), HipError> {
        let func = unsafe {
            transmute::<unsafe extern "C" fn(*mut T), unsafe extern "C" fn(*mut c_void)>(func)
        };
        unsafe {
            try_err!(
                hipLaunchHostFunc(self.raw, Some(func), data.cast()),
                Ok(())
            )
        }
    }
}
