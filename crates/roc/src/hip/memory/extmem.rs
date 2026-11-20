use core::{marker::PhantomData, mem::{ManuallyDrop, MaybeUninit}, ops::RangeBounds};
use std::fs::File;

use alloc::sync::Arc;
use rocm_sys::hip::{
    hipDestroyExternalMemory, hipDeviceptr_t, hipExternalMemory_t, hipExternalMemoryBufferDesc_st, hipExternalMemoryGetMappedBuffer, hipExternalMemoryHandleDesc_st, hipExternalMemoryHandleDesc_st__bindgen_ty_1, hipExternalMemoryHandleType, hipFree, hipFreeAsync, hipImportExternalMemory,
};

use crate::{
    auxiliary::{ResolvedRange, resolve_range}, hip::{
        DevRepr, HipError, device::Device, event::HipEvent, memory::{DevPtr, EventSync, HipSlice, HipSliceMut, MutPtr, StreamBound}, stream::HipStream,
    },
};

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct ExtBuffer {
    ptr: hipExternalMemory_t,
    size: usize,
    stream: Arc<HipStream>,
    file: ManuallyDrop<File>,
    _marker: PhantomData<[u8]>,
}

impl StreamBound for ExtBuffer {
    fn stream(&self) -> &HipStream {
        &self.stream
    }
}

impl HipStream {
    pub fn import_external_memory(
        self: &Arc<Self>,
        file: File,
        size: usize,
    ) -> Result<ExtBuffer, HipError> {
        #[cfg(unix)]
        let extmem = {
            use std::os::fd::AsRawFd;
            let fd = file.as_raw_fd();

            let handle_desc = hipExternalMemoryHandleDesc_st {
                type_: hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueFd,
                handle: hipExternalMemoryHandleDesc_st__bindgen_ty_1 { fd },
                size: size as u64,
                flags: 0,
                reserved: [0; 16],
            };

            unsafe {
                wrap_sys_res!(|extmem| hipImportExternalMemory(extmem.as_mut_ptr(), &handle_desc))
            }
        };
        #[cfg(windows)]
        let extmem = {
            use std::os::windows::io::AsRawHandle;
            let handle = file.as_raw_handle();

            let handle_desc = hipExternalMemoryHandleDesc_st {
                type_: hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueWin32,
                handle: hipExternalMemoryHandleDesc_st__bindgen_ty_1 {
                    win32:
                        rocm_sys::hip::hipExternalMemoryHandleDesc_st__bindgen_ty_1__bindgen_ty_1 {
                            handle,
                            name: core::ptr::null(),
                        },
                },
                size: size as u64,
                flags: 0,
                reserved: [0; 16],
            };

            unsafe { wrap_sys_res!(|ptr| hipImportExternalMemory(ptr.as_mut_ptr(), &handle_desc)) }
        };

        Ok(ExtBuffer {
            ptr: extmem?,
            size,
            stream: self.clone(),
            file: ManuallyDrop::new(file),
            _marker: PhantomData,
        })
    }
}

impl ExtBuffer {
    pub const fn as_raw(&self) -> hipExternalMemory_t {
        self.ptr
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    /// Map a byte-range as a buffer.
    ///
    /// Only one mapped buffer is allowed at a time — this method
    /// consumes `self`, transferring ownership into the [`MappedBuffer`].
    /// More restrictive than HIP itself requires, but keeps safety
    /// mechanical.
    pub fn map(self, range: impl RangeBounds<usize>) -> Result<MappedBuffer, HipError> {
        let ResolvedRange { start, end: _, len } = resolve_range(range, self.size);

        let buffer_desc = hipExternalMemoryBufferDesc_st {
            offset: start as u64,
            size: len as u64,
            flags: 0,
            reserved: [0; 16],
        };
        let device_ptr = unsafe {
            wrap_sys_res!(|ptr| hipExternalMemoryGetMappedBuffer(
                ptr.as_mut_ptr(),
                self.ptr,
                &raw const buffer_desc
            ))
        };

        Ok(MappedBuffer {
            ptr: device_ptr?,
            len,
            event: HipEvent::create(None)?,
            extmem: self,
        })
    }

    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe { handle.destroy_unchecked() }
    }

    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipDestroyExternalMemory(self.ptr), Ok(())) }
    }
}

impl Drop for ExtBuffer {
    fn drop(&mut self) {
        log_if_err!("failed to destroy ExtBuffer": unsafe { self.destroy_unchecked() });

        #[cfg(windows)]
        unsafe {
            ManuallyDrop::<File>::drop(&mut self.file);
        }
    }
}

/// A device-accessible buffer mapped from an [`ExtBuffer`]. Frees the mapping on drop.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct MappedBuffer {
    ptr: hipDeviceptr_t,
    len: usize,
    event: HipEvent,
    extmem: ExtBuffer,
}

impl MappedBuffer {
    pub const fn as_raw(&self) -> hipDeviceptr_t {
        self.ptr
    }

    pub const fn as_extmem(&self) -> &ExtBuffer {
        &self.extmem
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Deallocates a buffer
    ///
    /// This allows you to safely raw errors which would normally panic when dropped.
    #[inline(always)]
    pub fn dealloc(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe { handle.dealloc_unchecked() }
    }

    /// Deallocates a buffer
    ///
    /// # Safety
    ///
    /// The buffer must never be deallocated again or dropped.
    #[inline(always)]
    pub unsafe fn dealloc_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipFree(self.ptr.cast()), Ok(())) }
    }

    /// Deallocates a buffer asynchronously
    #[inline(always)]
    pub fn dealloc_async(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe { handle.dealloc_async_unchecked() }
    }

    /// Deallocates a buffer asynchronously
    #[inline(always)]
    pub unsafe fn dealloc_async_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipFreeAsync(self.ptr.cast(), self.extmem.stream.raw,), Ok(())) }
    }

    /// # Safety
    ///
    /// Must be initialized
    pub const unsafe fn as_hip_slice<T: DevRepr>(&self) -> &HipSlice<'_, T> {
        unsafe { &*(self as *const Self).cast() }
    }

    /// # Safety
    ///
    /// Must be initialized
    pub const unsafe fn as_hip_slice_mut<T: DevRepr>(&mut self) -> &mut HipSliceMut<'_, T> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

unsafe impl DevPtr<MaybeUninit<u8>> for MappedBuffer {
    fn as_ptr(&self) -> *const MaybeUninit<u8> {
        self.ptr.cast()
    }

    fn get_dev(&self) -> Device {
        self.extmem.stream.dev
    }

    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl MutPtr for MappedBuffer {}

impl EventSync for MappedBuffer {
    fn latesync(&self) -> Result<(), HipError> {
        self.event.record(&self.extmem.stream, None)
    }

    fn sync(&self) -> Result<(), HipError> {
        self.event.sync()
    }
}

impl StreamBound for MappedBuffer {
    fn stream(&self) -> &HipStream {
        &self.extmem.stream
    }
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        if self.extmem.stream.dev.has_mem_pools {
            unsafe {
                log_if_err!("failed to unmap MappedBuffer": self.dealloc_async_unchecked());
            }
        } else {
            log_if_err!("failed to synchronize MappedBuffer": self.event.sync());
            unsafe {
                log_if_err!("failed to unmap MappedBuffer": self.dealloc_unchecked());
            }
        }
    }
}
