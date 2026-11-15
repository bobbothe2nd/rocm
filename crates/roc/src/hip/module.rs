use core::{
    ffi::CStr,
    mem::{offset_of, transmute},
    ptr::null_mut,
};

use alloc::{ffi::CString, sync::Arc};

use rocm_sys::hip::{
    hipFuncGetAttribute, hipFunction_attribute, hipFunction_t, hipModule_t, hipModuleGetFunction,
    hipModuleLaunchKernel, hipModuleOccupancyMaxPotentialBlockSize,
    hipModuleOccupancyMaxPotentialBlockSizeWithFlags, hipModuleUnload,
    hipOccupancyMaxActiveBlocksPerMultiprocessor,
    hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
};

use crate::hip::{HipError, stream::HipStream};

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Clone)]
#[repr(transparent)]
pub struct Module {
    inner: Arc<private::ModuleInner>,
}

impl Module {
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
            module: self.clone(),
        })
    }

    #[inline(always)]
    pub fn as_raw(&self) -> hipModule_t {
        self.inner.raw
    }

    #[inline(always)]
    pub fn into_raw(self) -> hipModule_t {
        self.inner.raw
    }
}

#[cfg(all(feature = "hiprtc", any(hiprtc, feature = "dynamic-loading")))]
mod hiprtc {
    use super::*;
    use crate::hiprtc::program::Hsaco;

    impl Hsaco {
        pub fn load(&self) -> Result<Module, HipError> {
            Module::load(self)
        }
    }

    impl Module {
        pub fn load(code: &Hsaco) -> Result<Self, HipError> {
            let module: Result<hipModule_t, HipError> = unsafe {
                wrap_sys_res!(|module| rocm_sys::hip::hipModuleLoadData(
                    (&raw mut module).cast(),
                    code.code.as_ptr().cast()
                ))
            };

            let inner = Arc::new(private::ModuleInner { raw: module? });

            Ok(Module { inner })
        }
    }
}

mod private {
    use super::*;

    #[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
    pub struct ModuleInner {
        pub(super) raw: hipModule_t,
    }

    unsafe impl Send for ModuleInner {}
    unsafe impl Sync for ModuleInner {}

    impl Drop for ModuleInner {
        fn drop(&mut self) {
            log_if_err!("failed to unload Module": HipError::wrap(unsafe { hipModuleUnload(self.raw) }));
        }
    }
}

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Clone)]
pub struct Func {
    pub(crate) raw: hipFunction_t,
    module: Module,
}

impl Func {
    #[inline(always)]
    pub fn module_handle(&self) -> Module {
        self.module.clone()
    }

    pub fn into_module_handle(self) -> Module {
        self.module
    }

    pub fn get_attribute(&self, attr: FuncAttr) -> Result<i32, HipError> {
        unsafe { wrap_sys_res!(|val| hipFuncGetAttribute(val.as_mut_ptr(), attr.into(), self.raw)) }
    }

    /// Maximum number of resident blocks per multiprocessor for this kernel
    /// at the given block size and dynamic shared memory.
    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: i32,
        dynamic_smem_bytes: usize,
        flags: Option<u32>,
    ) -> Result<i32, HipError> {
        match flags {
            None => unsafe {
                wrap_sys_res!(|blocks| hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    blocks.as_mut_ptr(),
                    self.raw as *const _,
                    block_size,
                    dynamic_smem_bytes,
                ))
            },
            Some(flags) => unsafe {
                wrap_sys_res!(
                    |blocks| hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                        blocks.as_mut_ptr(),
                        self.raw as *const _,
                        block_size,
                        dynamic_smem_bytes,
                        flags,
                    )
                )
            },
        }
    }

    /// Returns the smallest grid that saturates the device
    /// and the block size that achieves the saturation.
    pub fn occupancy_max_potential_block_size(
        &self,
        dynamic_smem_bytes: usize,
        block_size: i32,
        flags: Option<u32>,
    ) -> Result<LaunchShape, HipError> {
        match flags {
            None => unsafe {
                wrap_sys_res!(|conf: LaunchShape| hipModuleOccupancyMaxPotentialBlockSize(
                    conf.as_mut_ptr()
                        .byte_add(offset_of!(LaunchShape, grid))
                        .cast(),
                    conf.as_mut_ptr()
                        .byte_add(offset_of!(LaunchShape, block))
                        .cast(),
                    self.raw,
                    dynamic_smem_bytes,
                    block_size,
                ))
            },
            Some(flags) => unsafe {
                wrap_sys_res!(
                    |conf: LaunchShape| hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
                        conf.as_mut_ptr()
                            .byte_add(offset_of!(LaunchShape, grid))
                            .cast(),
                        conf.as_mut_ptr()
                            .byte_add(offset_of!(LaunchShape, block))
                            .cast(),
                        self.raw,
                        dynamic_smem_bytes,
                        block_size,
                        flags,
                    )
                )
            },
        }
    }
}

/// On AMD devices and some Nvidia devices, these hints and controls are ignored.
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum FuncAttr {
    /// The maximum number of bytes requested for dynamically allocated shared memory
    MaxDynamicSharedMemorySize = 8,
    /// Sets the percentage of total shared memory allocated as the shared memory carveout
    PreferredSharedMemoryCarveout = 9,
    /// The kernel must launch with a valid cluster size specified.
    ClusterDimMustBeSet = 10,
    /// The required cluster width in blocks
    RequiredClusterWidth = 11,
    /// The required cluster height in blocks
    RequiredClusterHeight = 12,
    /// The required cluster depth in blocks
    RequiredClusterDepth = 13,
    /// Is the function allowed to launch with non-portable cluster size.
    NonPortableClusterSizeAllowed = 14,
    /// The block scheduling policy of a function.
    ClusterSchedulingPolicyPreference = 15,
    Max = 16,
}

impl From<hipFunction_attribute> for FuncAttr {
    #[inline(always)]
    fn from(value: hipFunction_attribute) -> Self {
        unsafe { transmute::<hipFunction_attribute, Self>(value) }
    }
}

impl From<FuncAttr> for hipFunction_attribute {
    #[inline(always)]
    fn from(value: FuncAttr) -> Self {
        unsafe { transmute::<FuncAttr, Self>(value) }
    }
}

impl HipStream {
    /// Launches a kernel/function on this stream
    ///
    /// Each element of `args` points to the value of that argument
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
        conf: LaunchShape,
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

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug, Clone)]
pub struct LaunchShape {
    pub grid: [u32; 3],
    pub block: [u32; 3],
}
