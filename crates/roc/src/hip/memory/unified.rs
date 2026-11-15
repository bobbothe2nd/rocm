use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, ops::RangeBounds};

use alloc::sync::Arc;
use rocm_sys::hip::{hipFree, hipFreeAsync};

use crate::{
    auxiliary::{ResolvedRange, resolve_range},
    hip::{HipError, event::HipEvent, stream::HipStream},
};

/// Attach mode for a [`UnifiedBuffer`]. Mirrors HIP's
/// `hipMemAttachGlobal` / `Host` / `Single` flag triple, which are
/// `#define`s in the C headers (bindgen drops them).
#[repr(u32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemAttachFlags {
    /// Any device or stream may access the memory.
    Global = 1,
    /// Only the host may access; device access requires
    /// `hipDeviceAttributeConcurrentManagedAccess`.
    Host = 2,
    /// Only the attached stream may access. Host access synchronizes
    /// that stream.
    Single = 4,
}

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct UnifiedSlice<'a> {
    pub(crate) ptr: *const u8,
    pub(crate) size: usize,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    pub(crate) attach_mode: MemAttachFlags,
    pub(crate) concurrent_managed_access: bool,
    pub(crate) _marker: PhantomData<&'a UnsafeCell<[u8]>>,
}

unsafe impl Send for UnifiedSlice<'_> {}
unsafe impl Sync for UnifiedSlice<'_> {}

impl UnifiedSlice<'_> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange {
            start,
            size,
            end: _,
        } = resolve_range(range, self.size);
        unsafe {
            Self::from_raw_parts(
                self.ptr.add(start),
                size,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.concurrent_managed_access,
            )
        }
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *const u8,
        size: usize,
        stream: &Arc<HipStream>,
        event: &HipEvent,
        attach_mode: MemAttachFlags,
        concurrent_managed_access: bool,
    ) -> Self {
        Self {
            ptr,
            size,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            attach_mode,
            concurrent_managed_access,
            _marker: PhantomData,
        }
    }

    /// Gets the pointer to GPU memory
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.size
    }
}

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct UnifiedSliceMut<'a> {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    pub(crate) attach_mode: MemAttachFlags,
    pub(crate) concurrent_managed_access: bool,
    pub(crate) _marker: PhantomData<&'a mut UnsafeCell<[u8]>>,
}

unsafe impl Send for UnifiedSliceMut<'_> {}
unsafe impl Sync for UnifiedSliceMut<'_> {}

impl UnifiedSliceMut<'_> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange {
            start,
            size,
            end: _,
        } = resolve_range(range, self.size);
        unsafe {
            Self::from_raw_parts(
                self.ptr.add(start),
                size,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.concurrent_managed_access,
            )
        }
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: usize,
        stream: &Arc<HipStream>,
        event: &HipEvent,
        attach_mode: MemAttachFlags,
        concurrent_managed_access: bool,
    ) -> Self {
        Self {
            ptr,
            size,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            attach_mode,
            concurrent_managed_access,
            _marker: PhantomData,
        }
    }

    /// Gets the pointer to GPU memory
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.size
    }
}

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct UnifiedBuffer {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    pub(crate) attach_mode: MemAttachFlags,
    pub(crate) concurrent_managed_access: bool,
    pub(crate) _marker: PhantomData<UnsafeCell<[u8]>>,
}

unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

impl UnifiedBuffer {
    /// Gets the pointer to GPU memory
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.size
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> UnifiedSlice<'_> {
        let ResolvedRange {
            start,
            size,
            end: _,
        } = resolve_range(range, self.size);
        unsafe {
            UnifiedSlice::from_raw_parts(
                self.ptr.add(start),
                size,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.concurrent_managed_access,
            )
        }
    }

    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> UnifiedSliceMut<'_> {
        let ResolvedRange {
            start,
            size,
            end: _,
        } = resolve_range(range, self.size);
        unsafe {
            UnifiedSliceMut::from_raw_parts(
                self.ptr.add(start),
                size,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.concurrent_managed_access,
            )
        }
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
        unsafe { try_err!(hipFreeAsync(self.ptr.cast(), self.stream.raw), Ok(())) }
    }
}

impl Drop for UnifiedBuffer {
    fn drop(&mut self) {
        if self.stream.dev.has_async_alloc {
            unsafe {
                log_if_err!("failed to deallocate HipBuffer": self.dealloc_async_unchecked());
            }
        } else {
            log_if_err!("failed to synchronize HipBuffer": self.event.sync());
            unsafe {
                log_if_err!("failed to deallocate HipBuffer": self.dealloc_unchecked());
            }
        }
    }
}

impl<'a> AsRef<UnifiedSlice<'a>> for UnifiedBuffer {
    #[inline(always)]
    fn as_ref(&self) -> &UnifiedSlice<'a> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a> AsRef<UnifiedSlice<'a>> for UnifiedSliceMut<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &UnifiedSlice<'a> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a> AsMut<UnifiedSliceMut<'a>> for UnifiedBuffer {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut UnifiedSliceMut<'a> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl AsRef<Self> for UnifiedSlice<'_> {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl AsMut<Self> for UnifiedSliceMut<'_> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Self {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}
