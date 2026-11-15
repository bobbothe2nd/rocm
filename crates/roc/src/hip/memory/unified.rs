use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, ops::{Deref, DerefMut, RangeBounds}, slice};
use std::ptr::write_bytes;

use alloc::sync::Arc;
use rocm_sys::hip::{hipFree, hipFreeAsync, hipMallocManaged};

use crate::{
    auxiliary::{ResolvedRange, resolve_range},
    hip::{
        HipError,
        event::{EventFlags, HipEvent},
        stream::HipStream,
    },
};

/// Attach mode for a [`UnifiedBuffer`]. Mirrors HIP's
/// `hipMemAttachGlobal` / `Host` / `Single` flag triple, which are
/// `#define`s in the C headers,
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
    pub(crate) is_init: bool,
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
                self.is_init,
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
        is_init: bool,
    ) -> Self {
        Self {
            ptr,
            size,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            attach_mode,
            is_init,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn expect_init(&self, msg: Option<&str>) {
        assert!(
            self.is_init,
            "{}",
            match msg {
                Some(msg) => msg,
                None => "expected init UnifiedSlice, encountered uninit",
            }
        );
    }

    #[inline(always)]
    pub const unsafe fn assume_init(&mut self) {
        self.is_init = true;
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

    pub fn as_slice(&self) -> &[u8] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedSlice": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }
}

impl Deref for UnifiedSlice<'_> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
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
    pub(crate) is_init: bool,
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
                self.is_init,
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
        is_init: bool,
    ) -> Self {
        Self {
            ptr,
            size,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            attach_mode,
            is_init,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn expect_init(&self, msg: Option<&str>) {
        assert!(
            self.is_init,
            "{}",
            match msg {
                Some(msg) => msg,
                None => "expected init UnifiedSliceMut, encountered uninit",
            }
        );
    }

    #[inline(always)]
    pub const unsafe fn assume_init(&mut self) {
        self.is_init = true;
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

    pub fn as_slice(&self) -> &[u8] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    pub fn memset(&self, byte: u8) {
        unsafe {
            write_bytes(self.ptr, byte, self.size);
        }
    }
}

impl Deref for UnifiedSliceMut<'_> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for UnifiedSliceMut<'_> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
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
    pub(crate) is_init: bool,
    pub(crate) _marker: PhantomData<UnsafeCell<[u8]>>,
}

unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

impl HipStream {
    /// Allocate unified (managed) memory.
    ///
    /// `attach_global = true` uses `hipMemAttachGlobal` (any device/stream
    /// can access); `false` uses `hipMemAttachHost`. To switch to
    /// `hipMemAttachSingle` after allocation use
    /// [`HipUnifiedSlice::attach`].
    ///
    /// Returns [`sys::hipError_t::hipErrorNotSupported`] if the device
    /// does not advertise `hipDeviceAttributeManagedMemory`.
    pub fn alloc_unified(
        self: &Arc<Self>,
        size: usize,
        attach_global: bool,
    ) -> Result<UnifiedBuffer, HipError> {
        if !self.dev.has_managed_memory {
            return Err(HipError::NotSupported);
        }

        let attach_mode = if attach_global {
            MemAttachFlags::Global
        } else {
            MemAttachFlags::Host
        };

        let raw = unsafe {
            wrap_sys_res!(|raw| hipMallocManaged(raw.as_mut_ptr(), size, attach_mode as u32))
        };

        let event = HipEvent::create(Some(EventFlags::BLOCKING_SYNC))?;

        Ok(UnifiedBuffer {
            ptr: raw?.cast(),
            size,
            stream: self.clone(),
            event,
            attach_mode,
            is_init: false,
            _marker: PhantomData,
        })
    }

    /// Allocate unified (managed) memory.
    pub fn alloc_unified_zeroed(
        self: &Arc<Self>,
        size: usize,
        attach_global: bool,
    ) -> Result<UnifiedBuffer, HipError> {
        self.alloc_unified(size, attach_global).map(|mut buf| {
            buf.memset(0);
            unsafe {
                buf.assume_init();
            }
            buf
        })
    }
}

impl UnifiedBuffer {
    #[inline(always)]
    pub const fn expect_init(&self, msg: Option<&str>) {
        assert!(
            self.is_init,
            "{}",
            match msg {
                Some(msg) => msg,
                None => "expected init UnifiedBuffer, encountered uninit",
            }
        );
    }

    #[inline(always)]
    pub const unsafe fn assume_init(&mut self) {
        self.is_init = true;
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
                self.is_init,
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
                self.is_init,
            )
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedBuffer": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedBuffer": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
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

    pub fn memset(&self, byte: u8) {
        unsafe {
            write_bytes(self.ptr, byte, self.size);
        }
    }
}

impl Deref for UnifiedBuffer {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for UnifiedBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for UnifiedBuffer {
    fn drop(&mut self) {
        if self.stream.dev.has_async_alloc {
            unsafe {
                log_if_err!("failed to deallocate UnifiedBuffer": self.dealloc_async_unchecked());
            }
        } else {
            log_if_err!("failed to synchronize UnifiedBuffer": self.event.sync());
            unsafe {
                log_if_err!("failed to deallocate UnifiedBuffer": self.dealloc_unchecked());
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
