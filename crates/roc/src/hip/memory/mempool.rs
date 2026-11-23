use core::{marker::PhantomData, mem::ManuallyDrop, ops::{Deref, DerefMut}, ptr::null_mut};

use alloc::sync::Arc;
use rocm_sys::hip::{
    hipDeviceSetMemPool, hipMallocFromPoolAsync, hipMemAllocationHandleType, hipMemAllocationType,
    hipMemLocation, hipMemLocationType, hipMemPool_t, hipMemPoolCreate, hipMemPoolDestroy,
    hipMemPoolExportPointer, hipMemPoolExportToShareableHandle,
    hipMemPoolImportFromShareableHandle, hipMemPoolImportPointer, hipMemPoolProps,
    hipMemPoolPtrExportData, hipMemPoolTrimTo,
};

use crate::hip::{
    DevRepr, HipError,
    device::Device,
    event::HipEvent,
    memory::{DevPtr, HipBuffer, RawPtr},
    stream::HipStream,
};

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipMemPool {
    inner: RefMemPool,
}

const DEFAULT_ALLOCATION_TYPE: hipMemAllocationHandleType = {
    #[cfg(unix)]
    {
        hipMemAllocationHandleType::hipMemHandleTypePosixFileDescriptor
    }
    #[cfg(windows)]
    {
        hipMemAllocationHandleType::hipMemHandleTypeWin32
    }
};

impl HipStream {
    pub fn create_mem_pool(self: &Arc<Self>) -> Result<HipMemPool, HipError> {
        if !self.dev.has_mem_pools {
            return Err(HipError::NotSupported);
        }

        let prop = hipMemPoolProps {
            allocType: hipMemAllocationType::hipMemAllocationTypePinned,
            handleTypes: DEFAULT_ALLOCATION_TYPE,
            location: hipMemLocation {
                type_: hipMemLocationType::hipMemLocationTypeDevice,
                id: 0,
            },
            win32SecurityAttributes: null_mut(),
            maxSize: 0,
            reserved: [0; 56],
        };

        let mem_pool: Result<hipMemPool_t, HipError> = unsafe {
            wrap_sys_res!(|mem_pool| hipMemPoolCreate(mem_pool.as_mut_ptr(), &raw const prop))
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
            unsafe { hipDeviceSetMemPool(mem_pool.raw, self.dev) },
            Ok(())
        )
    }
}

impl HipMemPool {
    #[inline(always)]
    pub fn as_raw(&self) -> hipMemPool_t {
        self.inner.raw
    }

    pub fn export<'a>(&'a self) -> Result<HipMemPoolExport<'a>, HipError> {
        self.inner.export()
    }

    /// you should probably use [`RefMemPool::import`] instead
    ///
    /// # Safety
    ///
    /// Only one mem pool can be dropped
    pub unsafe fn import(export_data: HipMemPoolExport) -> Result<Self, HipError> {
        Ok(Self {
            inner: RefMemPool::import(export_data)?,
        })
    }

    #[inline(always)]
    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);
        unsafe { handle.destroy_unchecked() }
    }
}

impl Drop for HipMemPool {
    fn drop(&mut self) {
        log_if_err!("failed to destroy HipMemPool": unsafe {
            self.destroy_unchecked()
        });
    }
}

impl Deref for HipMemPool {
    type Target = RefMemPool;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for HipMemPool {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
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

impl<P: DevPtr<T>, T: DevRepr> Export<T> for P {
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

impl HipMemPool {
    pub fn import_ptr<'a, P: DevPtr<T>, T: DevRepr>(
        &self,
        ptr: &mut HipPtrExport<'a, P>,
    ) -> Result<RawPtr<'a, T>, HipError> {
        unsafe {
            wrap_sys_res!(|dev_ptr: RawPtr<T>| hipMemPoolImportPointer(
                dev_ptr.as_mut_ptr().cast(),
                self.inner.raw,
                &raw mut ptr.export_data
            ))
        }
    }
}

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct RefMemPool {
    pub(crate) raw: hipMemPool_t,
    pub(crate) stream: Option<Arc<HipStream>>,
}

impl RefMemPool {
    #[inline(always)]
    pub fn as_raw(&self) -> hipMemPool_t {
        self.raw
    }

    pub fn export<'a>(&'a self) -> Result<HipMemPoolExport<'a>, HipError> {
        unsafe {
            wrap_sys_res!(
                |data: HipMemPoolExport| {
                    hipMemPoolExportToShareableHandle(
                        data.as_mut_ptr().cast(),
                        self.raw,
                        DEFAULT_ALLOCATION_TYPE,
                        0,
                    )
                }
            )
        }
    }

    pub fn import(export_data: HipMemPoolExport) -> Result<Self, HipError> {
        let mem_pool: Result<hipMemPool_t, HipError> = unsafe {
            wrap_sys_res!(|mem_pool| hipMemPoolImportFromShareableHandle(
                mem_pool.as_mut_ptr(),
                export_data.export_data.as_hip_handle(),
                DEFAULT_ALLOCATION_TYPE,
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

    pub fn set_stream(&mut self, stream: &Arc<HipStream>) -> Option<Arc<HipStream>> {
        self.stream.replace(stream.clone())
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

impl AsRef<Self> for RefMemPool {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl AsMut<Self> for RefMemPool {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

#[repr(transparent)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipMemPoolExport<'a> {
    export_data: __private_owned_shareable_handle::OwnedShareableHandle,
    _marker: PhantomData<&'a HipMemPool>,
}

impl Drop for HipMemPoolExport<'_> {
    fn drop(&mut self) {
        unsafe {
            self.export_data.release();
        }
    }
}

mod __private_owned_shareable_handle {
    use core::{ffi::c_void, fmt::Debug};

    #[repr(transparent)]
    #[derive(Debug)]
    #[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
    pub struct OwnedShareableHandle {
        #[cfg(unix)]
        fd: i32,
        #[cfg(windows)]
        win32: *mut c_void,
    }

    impl OwnedShareableHandle {
        pub(super) unsafe fn release(&mut self) {
            #[cfg(unix)]
            {
                unsafe {
                    libc::close(self.fd as core::ffi::c_int);
                }
            }
            #[cfg(windows)]
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.win32);
            }
        }

        pub(super) fn as_hip_handle(&self) -> *mut c_void {
            #[cfg(unix)]
            {
                self.fd as *mut c_void
            }

            #[cfg(windows)]
            {
                self.win32
            }
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
