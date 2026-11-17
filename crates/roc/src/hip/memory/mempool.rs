use core::{ffi::c_int, marker::PhantomData, mem::ManuallyDrop, ptr::null_mut};

use alloc::sync::Arc;
use rocm_sys::{
    hip::{hipMallocFromPoolAsync, hipMemPool_t, hipMemPoolCreate, hipMemPoolDestroy},
    platform::windows::LPSECURITYATTRIBUTES,
};

use crate::hip::{DevRepr, HipError, event::HipEvent, memory::HipBuffer, stream::HipStream};

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug, Clone)]
pub struct MemPool {
    raw: hipMemPool_t,
    stream: Arc<HipStream>,
}

impl HipStream {
    pub fn create_mem_pool(self: &Arc<Self>, prop: MemPoolProps) -> Result<MemPool, HipError> {
        if !self.dev.has_mem_pools {
            return Err(HipError::NotSupported);
        }

        let mem_pool: Result<hipMemPool_t, HipError> = unsafe {
            wrap_sys_res!(|mem_pool| hipMemPoolCreate(
                mem_pool.as_mut_ptr(),
                (&raw const prop).cast()
            ))
        };
        Ok(MemPool {
            raw: mem_pool?,
            stream: self.clone(),
        })
    }
}

impl MemPool {
    pub fn as_raw(&self) -> hipMemPool_t {
        self.raw
    }

    /// Allocates `len` elements
    pub fn alloc<T: DevRepr>(&self, len: usize) -> Result<HipBuffer<T>, HipError> {
        let event = HipEvent::create(None)?;

        let ptr: Result<*mut T, HipError> = unsafe {
            wrap_sys_res!(|ptr: *mut T| hipMallocFromPoolAsync((&raw mut ptr).cast(), len, self.raw, self.stream.raw))
        };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(HipBuffer {
            ptr,
            len,
            stream: self.stream.clone(),
            event,
            _marker: PhantomData,
        })
    }

    /// Allocates `len` zeroed elements
    pub fn alloc_zeroed<T: DevRepr>(&self, len: usize) -> Result<HipBuffer<T>, HipError> {
        self.alloc(len).and_then(|mut buf| {
            buf.memset(0u8)?;
            Ok(buf)
        })
    }

    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);
        unsafe { handle.destroy_unchecked() }
    }

    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipMemPoolDestroy(self.raw), Ok(())) }
    }
}

impl Drop for MemPool {
    fn drop(&mut self) {
        log_if_err!("failed to destroy MemPool": unsafe {
            self.destroy_unchecked()
        });
    }
}

/// Specifies the properties of allocations made from the pool.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct MemPoolProps {
    /// Allocation type. Currently must be specified as `hipMemAllocationTypePinned`
    alloc_type: HipMemAllocType,
    /// Handle types that will be supported by allocations from the pool
    handle_types: HipMemAllocHandleType,
    /// Location where allocations should reside
    location: HipMemLoc,
    /// Windows-specific `LPSECURITYATTRIBUTES` required when `hipMemHandleTypeWin32` is specified
    win32_security_attributes: LPSECURITYATTRIBUTES,
    /// Maximum pool size. When set to 0, defaults to a system dependent value
    max_size: usize,
    /// Reserved for future use, must be 0
    reserved: [::core::ffi::c_uchar; 56usize],
}

impl MemPoolProps {
    pub const fn new() -> Self {
        Self {
            alloc_type: HipMemAllocType::Invalid,
            handle_types: HipMemAllocHandleType::None,
            location: HipMemLoc {
                type_: HipMemLocType::Invalid,
                id: -1,
            },
            win32_security_attributes: null_mut(),
            max_size: 0,
            reserved: [0; 56],
        }
    }

    pub const fn alloc_type(mut self, alloc_type: HipMemAllocType) -> Self {
        self.alloc_type = alloc_type;
        self
    }

    pub const fn handle_types(mut self, handle_types: HipMemAllocHandleType) -> Self {
        self.handle_types = handle_types;
        self
    }

    pub const fn max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    #[cfg(windows)]
    pub const unsafe fn win32_security_attributes(
        mut self,
        win32_security_attributes: LPSECURITYATTRIBUTES,
    ) -> Self {
        self.win32_security_attributes = win32_security_attributes;
        self
    }
}

/// Defines the allocation types
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HipMemAllocType {
    Invalid = 0,
    /// This allocation type is 'pinned', i.e. cannot migrate from its current
    /// location while the application is actively using it
    Pinned = 1,
    Managed = 2,
    Uncached = 1073741824,
    Max = 2147483647,
}

/// Flags for specifying handle types for memory pool allocations
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HipMemAllocHandleType {
    /// Does not allow any export mechanism
    None = 0,
    PosixFileDescriptor = 1,
    /// Allows a Win32 NT handle for exporting. (`HANDLE`)
    Win32 = 2,
    /// Allows a Win32 KMT handle for exporting. (`D3DKMT_HANDLE`)
    Win32Kmt = 4,
    /// Allows a fabric handle to be used for exporting.
    Fabric = 8,
}

/// Specifies a memory location.
/// To specify a gpu, set type = `hipMemLocationTypeDevice` and set id = the gpu's device ID
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HipMemLoc {
    /// Specifies the location type, which describes the meaning of id
    pub type_: HipMemLocType,
    /// Identifier for the provided location type `HipMemLocType`
    pub id: c_int,
}

#[repr(i32)]
/// Specifies the type of location
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HipMemLocType {
    Invalid = 0,
    /// Device location, thus it's HIP device ID
    Device = 1,
    /// Host location, id is ignored
    Host = 2,
    /// Host NUMA node location, id is host NUMA node id
    HostNuma = 3,
    HostNumaCurrent = 4,
}
