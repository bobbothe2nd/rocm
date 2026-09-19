use core::{ffi::CStr, marker::PhantomData, ptr::null_mut};

use alloc::{rc::Rc, ffi::CString};

use rocm_sys::hip::{
    hipFunction_t, hipModule_t, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData, hipModuleUnload,
};

use crate::{
    hip::{HipError, stream::Stream},
    hiprtc::program::Hsaco,
};

impl Hsaco {
    pub fn load<'a>(&'a self) -> Result<Module<'a>, HipError> {
        Module::load(self)
    }
}

impl<'a> Module<'a> {
    pub fn load(code: &'a Hsaco) -> Result<Self, HipError> {
        let module: Result<hipModule_t, HipError> = unsafe {
            wrap_sys_res!(|module| hipModuleLoadData(
                (&raw mut module).cast(),
                code.code.as_ptr().cast()
            ))
        };

        let inner = Rc::new(ModuleInner {
            raw: module?,
            _code: PhantomData,
        });

        Ok(Module {
            inner,
            _code: PhantomData,
        })
    }
}

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct Module<'a> {
    inner: Rc<ModuleInner<'a>>,
    _code: PhantomData<&'a Hsaco>,
}

impl<'a> Module<'a> {
    #[cfg(feature = "alloc")]
    pub fn get_func(&self, name: &str) -> Result<Func<'a>, HipError> {
        let name = CString::new(name).map_err(|_| HipError::InvalidValue)?;
        self.get_func_c(&name)
    }

    pub fn get_func_c(&self, name: &CStr) -> Result<Func<'a>, HipError> {
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

#[derive(Debug)]
#[repr(transparent)]
struct ModuleInner<'a> {
    raw: hipModule_t,
    _code: PhantomData<&'a Hsaco>,
}

impl Drop for ModuleInner<'_> {
    fn drop(&mut self) {
        unsafe {
            hipModuleUnload(self.raw);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Func<'a> {
    raw: hipFunction_t,
    _module: Module<'a>,
}

impl Stream {
    pub unsafe fn launch(
        &self,
        func: &Func<'_>,
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
