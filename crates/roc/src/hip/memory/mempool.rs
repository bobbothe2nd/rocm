use core::{
    ffi::c_int,
    marker::PhantomData,
    mem::{ManuallyDrop, transmute},
    ptr::null_mut,
};

use alloc::sync::Arc;
use rocm_sys::{
    hip::{
        hipDeviceSetMemPool, hipMallocFromPoolAsync, hipMemAllocationHandleType, hipMemPool_t,
        hipMemPoolCreate, hipMemPoolDestroy, hipMemPoolExportPointer,
        hipMemPoolExportToShareableHandle, hipMemPoolImportFromShareableHandle,
        hipMemPoolPtrExportData, hipMemPoolTrimTo,
    },
    platform::windows::LPSECURITYATTRIBUTES,
};

use crate::hip::{
    DevRepr, HipError,
    device::Device,
    event::HipEvent,
    memory::{DevPtr, HipBuffer},
    stream::HipStream,
};

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipMemPool {
    inner: RefMemPool,
}

impl HipStream {
    pub fn create_mem_pool(self: &Arc<Self>, prop: MemPoolProps) -> Result<HipMemPool, HipError> {
        if !self.dev.has_mem_pools {
            return Err(HipError::NotSupported);
        }

        let mem_pool: Result<hipMemPool_t, HipError> = unsafe {
            wrap_sys_res!(|mem_pool| hipMemPoolCreate(
                mem_pool.as_mut_ptr(),
                (&raw const prop).cast()
            ))
        };
        Ok(HipMemPool {
            inner: RefMemPool {
                raw: mem_pool?,
                stream: Some(self.clone()),
            },
        })
    }
}

impl Device {
    pub fn set_mem_pool<T: AsRef<RefMemPool>>(self, mem_pool: &T) -> Result<(), HipError> {
        let mem_pool = mem_pool.as_ref();

        if mem_pool.stream.is_none() {
            return Err(HipError::NotSupported);
        }

        try_err!(
            unsafe { hipDeviceSetMemPool(self.dev, mem_pool.raw) },
            Ok(())
        )
    }
}

impl HipMemPool {
    #[inline(always)]
    pub fn as_raw(&self) -> hipMemPool_t {
        self.inner.raw
    }

    pub fn export(&self, handle_type: HipMemAllocHandleType) -> Result<HipMemPoolExport, HipError> {
        self.inner.export(handle_type)
    }

    /// you should probably use [`RefMemPool::import`] instead
    pub unsafe fn import(export_data: HipMemPoolExport) -> Result<Self, HipError> {
        Ok(Self {
            inner: unsafe { RefMemPool::import(export_data)? },
        })
    }

    /// Allocates `len` elements
    pub fn alloc<T: DevRepr>(&self, len: usize) -> Result<HipBuffer<T>, HipError> {
        self.inner.alloc(len)
    }

    /// Allocates `len` zeroed elements
    pub fn alloc_zeroed<T: DevRepr>(&self, len: usize) -> Result<HipBuffer<T>, HipError> {
        self.inner.alloc_zeroed(len)
    }

    #[inline(always)]
    pub fn trim_to(&self, min_size: usize) -> Result<(), HipError> {
        self.inner.trim_to(min_size)
    }

    #[inline(always)]
    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);
        unsafe { handle.destroy_unchecked() }
    }

    #[inline(always)]
    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { self.inner.destroy_unchecked() }
    }
}

impl Drop for HipMemPool {
    fn drop(&mut self) {
        log_if_err!("failed to destroy HipMemPool": unsafe {
            self.destroy_unchecked()
        });
    }
}

impl AsRef<RefMemPool> for HipMemPool {
    fn as_ref(&self) -> &RefMemPool {
        &self.inner
    }
}

impl AsMut<RefMemPool> for HipMemPool {
    fn as_mut(&mut self) -> &mut RefMemPool {
        &mut self.inner
    }
}

pub trait Export<T>: DevPtr<T> {
    fn export_ptr(&mut self) -> Result<HipPtrExport<'_, Self>, HipError>;
}

impl<P: DevPtr<T>, T> Export<T> for P {
    fn export_ptr(&mut self) -> Result<HipPtrExport<'_, Self>, HipError> {
        if !self.get_dev().has_mem_pools {
            return Err(HipError::NotSupported);
        }

        unsafe {
            wrap_sys_res!(
                |export_data: HipPtrExport<'_, Self>| hipMemPoolExportPointer(
                    export_data.as_mut_ptr().cast(),
                    self.as_ptr() as *mut _
                )
            )
        }
    }
}

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct RefMemPool {
    raw: hipMemPool_t,
    stream: Option<Arc<HipStream>>,
}

impl RefMemPool {
    #[inline(always)]
    pub fn as_raw(&self) -> hipMemPool_t {
        self.raw
    }

    pub fn export(&self, handle_type: HipMemAllocHandleType) -> Result<HipMemPoolExport, HipError> {
        if matches!(handle_type, HipMemAllocHandleType::Fabric) {
            return Err(HipError::NotSupported);
        }

        #[cfg(windows)]
        if matches!(handle_type, HipMemAllocHandleType::Win32Kmt) {
            return Err(HipError::NotSupported);
        }

        let export_data: Result<_, HipError> = unsafe {
            wrap_sys_res!(
                |data: __private_owned_shareable_handle::OwnedShareableHandle| {
                    hipMemPoolExportToShareableHandle(
                        data.as_mut_ptr().cast(),
                        self.raw,
                        transmute::<HipMemAllocHandleType, hipMemAllocationHandleType>(handle_type),
                        0,
                    )
                }
            )
        };
        Ok(HipMemPoolExport {
            export_data: export_data?,
            handle_type,
        })
    }

    pub unsafe fn import(export_data: HipMemPoolExport) -> Result<Self, HipError> {
        let mem_pool: Result<hipMemPool_t, HipError> = unsafe {
            wrap_sys_res!(|mem_pool| hipMemPoolImportFromShareableHandle(
                mem_pool.as_mut_ptr(),
                export_data
                    .export_data
                    .as_hip_handle(export_data.handle_type),
                transmute::<HipMemAllocHandleType, hipMemAllocationHandleType>(
                    export_data.handle_type
                ),
                0,
            ))
        };
        Ok(RefMemPool {
            raw: mem_pool?,
            stream: None,
        })
    }

    /// Allocates `len` elements
    pub fn alloc<T: DevRepr>(&self, len: usize) -> Result<HipBuffer<T>, HipError> {
        let Some(stream) = &self.stream else {
            return Err(HipError::NotSupported);
        };

        let event = HipEvent::create(None)?;

        let ptr: Result<*mut T, HipError> = unsafe {
            wrap_sys_res!(|ptr: *mut T| hipMallocFromPoolAsync(
                (&raw mut ptr).cast(),
                len * size_of::<T>(),
                self.raw,
                stream.raw
            ))
        };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(HipBuffer {
            ptr,
            len,
            stream: stream.clone(),
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

    #[inline(always)]
    pub fn trim_to(&self, min_size: usize) -> Result<(), HipError> {
        try_err!(unsafe { hipMemPoolTrimTo(self.raw, min_size) }, Ok(()))
    }

    #[inline(always)]
    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipMemPoolDestroy(self.raw), Ok(())) }
    }
}

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipMemPoolExport {
    export_data: __private_owned_shareable_handle::OwnedShareableHandle,
    handle_type: HipMemAllocHandleType,
}

impl Drop for HipMemPoolExport {
    fn drop(&mut self) {
        unsafe {
            self.export_data.release(self.handle_type);
        }
    }
}

mod __private_owned_shareable_handle {
    use core::{ffi::c_void, fmt::Debug};

    use crate::hip::memory::HipMemAllocHandleType;

    #[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
    pub union OwnedShareableHandle {
        #[cfg(unix)]
        fd: i32,
        #[cfg(windows)]
        win32: *mut c_void,
        // #[cfg(windows)] kmt: u32,
        // fabric: rocm_sys::hip::hipMemFabricHandle_t,
    }

    impl OwnedShareableHandle {
        pub(super) unsafe fn release(&mut self, handle_type: HipMemAllocHandleType) {
            #[cfg(unix)]
            {
                use core::ffi::c_int;

                if handle_type == HipMemAllocHandleType::PosixFileDescriptor {
                    unsafe {
                        libc::close(self.fd as c_int);
                    }
                }
            }
            #[cfg(windows)]
            {
                match handle_type {
                    HipMemAllocHandleType::Win32 => unsafe {
                        windows_sys::Win32::Foundation::CloseHandle(self.win32);
                    },
                    HipMemAllocHandleType::Win32Kmt => {}
                    _ => {}
                }
            }
        }

        pub(super) fn as_hip_handle(&self, handle_type: HipMemAllocHandleType) -> *mut c_void {
            match handle_type {
                #[cfg(unix)]
                HipMemAllocHandleType::PosixFileDescriptor => unsafe { self.fd as *mut c_void },

                #[cfg(windows)]
                HipMemAllocHandleType::Win32 => unsafe { self.win32 },

                _ => core::ptr::null_mut(),
            }
        }
    }

    impl Debug for OwnedShareableHandle {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("OwnedShareableHandle")
        }
    }
}

#[repr(transparent)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipPtrExport<'a, T: ?Sized> {
    export_data: hipMemPoolPtrExportData,
    _marker: PhantomData<&'a mut T>,
}

/// Specifies the properties of allocations made from the pool.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MemPoolProps {
    /// Allocation type. Currently must be specified as [`hipMemAllocationTypePinned`](MemAllocType)
    alloc_type: HipMemAllocType,
    /// Handle types that will be supported by allocations from the pool
    handle_types: HipMemAllocHandleTypes,
    /// Location where allocations should reside
    location: HipMemLoc,
    /// Windows-specific [`LPSECURITYATTRIBUTES`] required when `hipMemHandleTypeWin32` is specified
    win32_security_attributes: LPSECURITYATTRIBUTES,
    /// Maximum pool size. When set to 0, defaults to a system dependent value
    max_size: usize,
    /// Reserved for future use, must be 0
    reserved: [::core::ffi::c_uchar; 56usize],
}

impl Default for MemPoolProps {
    fn default() -> Self {
        Self::new()
    }
}

impl MemPoolProps {
    pub const fn new() -> Self {
        Self {
            alloc_type: HipMemAllocType::Invalid,
            handle_types: HipMemAllocHandleTypes::empty(),
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

    pub const fn handle_types(mut self, handle_types: HipMemAllocHandleTypes) -> Self {
        self.handle_types = handle_types;
        self
    }

    pub const fn max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    pub const fn location(mut self, location: HipMemLoc) -> Self {
        self.location = location;
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
    #[cfg(unix)]
    PosixFileDescriptor = 1,
    /// Allows a Win32 NT handle for exporting. ([`HANDLE`](rocm_sys::platform::windows::HANDLE))
    #[cfg(windows)]
    Win32 = 2,
    /// Allows a Win32 KMT handle for exporting. ([`D3DKMT_HANDLE`](rocm_sys::platform::windows::D3DKMT_HANDLE))
    #[cfg(windows)]
    Win32Kmt = 4,
    /// Allows a fabric handle to be used for exporting.
    Fabric = 8,
}

bitflags::bitflags! {
    /// Flags for specifying handle types for memory pool allocations
    #[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub struct HipMemAllocHandleTypes: i32 {
        #[cfg(unix)]
        const POSIX_FILE_DESCRIPTOR = 1;
        /// Allows a Win32 NT handle for exporting. ([`HANDLE`](rocm_sys::platform::windows::HANDLE))
        #[cfg(windows)]
        const WIN32 = 2;
        /// Allows a Win32 KMT handle for exporting. ([`D3DKMT_HANDLE`](rocm_sys::platform::windows::D3DKMT_HANDLE))
        #[cfg(windows)]
        const WIN32_KMT = 4;
        /// Allows a fabric handle to be used for exporting.
        const FABRIC = 8;
    }
}

/// Specifies a memory location.
/// To specify a gpu, set type = `hipMemLocationTypeDevice` and set id = the gpu's device ID
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
    /// Host location, `id` is ignored
    Host = 2,
    /// Host NUMA node location, `id` is host NUMA node id
    HostNuma = 3,
    HostNumaCurrent = 4,
}
