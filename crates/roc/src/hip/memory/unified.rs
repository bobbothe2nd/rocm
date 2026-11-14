use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop};

use alloc::sync::Arc;

use crate::hip::{event::HipEvent, stream::HipStream};

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
    pub(crate) marker: PhantomData<&'a UnsafeCell<[u8]>>,
}

unsafe impl Send for UnifiedSlice<'_> {}
unsafe impl Sync for UnifiedSlice<'_> {}

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
    pub(crate) marker: PhantomData<&'a mut UnsafeCell<[u8]>>,
}

unsafe impl Send for UnifiedSliceMut<'_> {}
unsafe impl Sync for UnifiedSliceMut<'_> {}

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
    pub(crate) marker: PhantomData<UnsafeCell<[u8]>>,
}

unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

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
