use core::{ffi::CStr, ptr::null_mut};

use alloc::{ffi::CString, sync::Arc};

use rocm_sys::hip::{
    hipFunction_t, hipModule_t, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData,
    hipModuleUnload,
};

use crate::{
    hip::{HipError, stream::Stream},
    hiprtc::program::Hsaco,
};

#[derive(Clone)]
#[repr(transparent)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

impl Module {
    #[cfg(feature = "alloc")]
    pub fn get_func(&self, name: &str) -> Result<Func, HipError> {
        let name = CString::new(name).map_err(|_| HipError::InvalidValue)?;
        self.get_func_c(&name)
    }

    pub fn get_func_c(&self, name: &CStr) -> Result<Func, HipError> {
        let func: Result<hipFunction_t, HipError> = unsafe {
            wrap_sys_res!(|func| hipModuleGetFunction(
                (&raw mut func).cast(),
                self.inner.raw,
                name.as_ptr()
            ))
        };

        Ok(Func {
            raw: func?,
            _module: self.clone(),
        })
    }
}

#[cfg(all(
    feature = "hiprtc",
    any(hiprtc, feature = "dynamic-loading")
))]
mod hiprtc {
    use super::*;

    impl Hsaco {
        pub fn load(&self) -> Result<Module, HipError> {
            Module::load(self)
        }
    }

    impl Module {
        pub fn load(code: &Hsaco) -> Result<Self, HipError> {
            let module: Result<hipModule_t, HipError> = unsafe {
                wrap_sys_res!(|module| hipModuleLoadData
                    (
                    (&raw mut module).cast(),
                    code.code.as_ptr().cast()
                ))
            };

            let inner = Arc::new(ModuleInner { raw: module? });

            Ok(Module { inner })
        }
    }
}

struct ModuleInner {
    raw: hipModule_t,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        unsafe {
            hipModuleUnload(self.raw);
        }
    }
}

#[derive(Clone)]
pub struct Func {
    pub(crate) raw: hipFunction_t,
    _module: Module,
}

impl Stream {
    /// Launches a kernel/function on this stream
    ///
    /// # Safety
    ///
    /// The caller must ensure that `func` refers to a valid kernel and that
    /// `args` contains correctly typed and correctly laid-out kernel arguments
    /// for that kernel. All referenced memory must remain valid for the duration
    /// required by the launched kernel.
    pub unsafe fn launch(
        &self,
        func: &Func,
        args: &mut [*mut u8],
        conf: LaunchConfig,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipModuleLaunchKernel(
                    func.raw,
                    conf.grid[0],
                    conf.grid[1],
                    conf.grid[2],
                    conf.block[0],
                    conf.block[1],
                    conf.block[2],
                    0,
                    self.raw,
                    args.as_mut_ptr().cast(),
                    null_mut(),
                ),
                Ok(())
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
}
