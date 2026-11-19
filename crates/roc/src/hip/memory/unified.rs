use core::{
    cell::UnsafeCell,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, RangeBounds},
    slice,
};
use std::ptr::write_bytes;

use alloc::sync::Arc;
use rocm_sys::hip::{
    hipCpuDeviceId, hipFree, hipFreeAsync, hipMallocManaged, hipMemPrefetchAsync,
    hipStreamAttachMemAsync,
};

use crate::{
    auxiliary::{ResolvedRange, resolve_range}, hip::{
        DevRepr, HipError, event::{EventFlags, HipEvent}, memory::{HipSlice, HipSliceMut, HostSlice, HostSliceMut, Memset}, stream::HipStream,
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
pub struct UnifiedSlice<'a, T: DevRepr> {
    pub(crate) ptr: *const T,
    pub(crate) len: usize,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    pub(crate) true_pad: bool,
    pub(crate) is_init: bool,
    pub(crate) attach_mode: MemAttachFlags,
    pub(crate) _marker: PhantomData<&'a UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for UnifiedSlice<'_, T> {}
unsafe impl<T: DevRepr> Sync for UnifiedSlice<'_, T> {}

impl<T: DevRepr> UnifiedSlice<'_, T> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe {
            Self::from_raw_parts(
                self.ptr.add(start),
                len,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.is_init,
            )
        }
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *const T,
        len: usize,
        stream: &Arc<HipStream>,
        event: &HipEvent,
        attach_mode: MemAttachFlags,
        is_init: bool,
    ) -> Self {
        Self {
            ptr,
            len,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            attach_mode,
            is_init,
            true_pad: true,
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
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Gets the length of the allocation
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn as_slice(&self) -> &[T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedSlice": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.len as usize) }
    }

    /// Hint HIP to prefetch the buffer to the appropriate location based
    /// on the current [`attach_mode`](`Self::attach_mode`).
    ///
    /// - `Global` / `Single`: prefetch to the stream's device (requires
    ///   `hipDeviceAttributeConcurrentManagedAccess`).
    /// - `Host`: prefetch to host memory ([`hipCpuDeviceId`]).
    pub fn prefetch(&self) -> Result<(), HipError> {
        let device = match self.attach_mode {
            MemAttachFlags::Global | MemAttachFlags::Single => {
                if !self.stream.dev.concurrent_managed_access {
                    return Err(HipError::NotSupported);
                }
                self.stream.dev.as_raw()
            }
            MemAttachFlags::Host => hipCpuDeviceId,
        };
        unsafe {
            try_err!(
                hipMemPrefetchAsync(self.ptr as *const _, self.size(), device, self.stream.raw,),
                Ok(())
            )
        }
    }

    /// Synchronize the stream if needed to guarantee safe host access.
    pub fn sync_host(&self) -> Result<(), HipError> {
        if matches!(self.attach_mode, MemAttachFlags::Single) {
            self.event.sync()?;
        }
        Ok(())
    }

    /// Validate that the given stream may access this buffer. Returns
    /// `Err(hipErrorNotSupported)` when the attach mode forbids it.
    pub fn check_dev_access(&self) -> Result<(), HipError> {
        match self.attach_mode {
            MemAttachFlags::Global => {}
            MemAttachFlags::Host => {
                if !self.stream.dev.concurrent_managed_access {
                    return Err(HipError::NotSupported);
                }
            }
            MemAttachFlags::Single => return Err(HipError::NotSupported),
        }
        Ok(())
    }
}

impl<T: DevRepr> Deref for UnifiedSlice<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct UnifiedSliceMut<'a, T: DevRepr> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    pub(crate) true_pad: bool,
    pub(crate) is_init: bool,
    pub(crate) attach_mode: MemAttachFlags,
    pub(crate) _marker: PhantomData<&'a mut UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for UnifiedSliceMut<'_, T> {}
unsafe impl<T: DevRepr> Sync for UnifiedSliceMut<'_, T> {}

impl<T: DevRepr> UnifiedSliceMut<'_, T> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe {
            Self::from_raw_parts(
                self.ptr.add(start),
                len,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.is_init,
            )
        }
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut T,
        len: usize,
        stream: &Arc<HipStream>,
        event: &HipEvent,
        attach_mode: MemAttachFlags,
        is_init: bool,
    ) -> Self {
        Self {
            ptr,
            len,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            attach_mode,
            is_init,
            true_pad: true,
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
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Gets the length of the allocation
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn as_slice(&self) -> &[T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len as usize) }
    }

    /// Re-attach to a new stream and buffer
    pub fn attach(&mut self, flags: MemAttachFlags) -> Result<(), HipError> {
        self.event.sync()?;
        self.attach_mode = flags;
        unsafe {
            try_err!(
                hipStreamAttachMemAsync(
                    self.stream.raw,
                    self.ptr as *mut _,
                    self.size(),
                    self.attach_mode as i32,
                ),
                Ok(())
            )
        }
    }

    /// Hint HIP to prefetch the buffer to the appropriate location based
    /// on the current [`attach_mode`](`Self::attach_mode`).
    ///
    /// - `Global` / `Single`: prefetch to the stream's device (requires
    ///   `hipDeviceAttributeConcurrentManagedAccess`).
    /// - `Host`: prefetch to host memory ([`hipCpuDeviceId`]).
    pub fn prefetch(&self) -> Result<(), HipError> {
        let device = match self.attach_mode {
            MemAttachFlags::Global | MemAttachFlags::Single => {
                if !self.stream.dev.concurrent_managed_access {
                    return Err(HipError::NotSupported);
                }
                self.stream.dev.as_raw()
            }
            MemAttachFlags::Host => hipCpuDeviceId,
        };
        unsafe {
            try_err!(
                hipMemPrefetchAsync(self.ptr as *const _, self.size(), device, self.stream.raw,),
                Ok(())
            )
        }
    }

    pub fn memset(&mut self, byte: u8) {
        unsafe {
            write_bytes(self.ptr, byte, self.len);
        }
    }

    /// Synchronize the stream if needed to guarantee safe host access.
    pub fn sync_host(&self) -> Result<(), HipError> {
        if matches!(self.attach_mode, MemAttachFlags::Single) {
            self.event.sync()?;
        }
        Ok(())
    }

    /// Validate that the given stream may access this buffer. Returns
    /// `Err(hipErrorNotSupported)` when the attach mode forbids it.
    pub fn check_dev_access(&self) -> Result<(), HipError> {
        match self.attach_mode {
            MemAttachFlags::Global => {}
            MemAttachFlags::Host => {
                if !self.stream.dev.concurrent_managed_access {
                    return Err(HipError::NotSupported);
                }
            }
            MemAttachFlags::Single => return Err(HipError::NotSupported),
        }
        Ok(())
    }
}

impl<T: DevRepr> Deref for UnifiedSliceMut<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DevRepr> DerefMut for UnifiedSliceMut<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct UnifiedBuffer<T: DevRepr> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    pub(crate) true_pad: bool,
    pub(crate) is_init: bool,
    pub(crate) attach_mode: MemAttachFlags,
    pub(crate) _marker: PhantomData<UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for UnifiedBuffer<T> {}
unsafe impl<T: DevRepr> Sync for UnifiedBuffer<T> {}

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
    pub fn alloc_unified<T: DevRepr>(
        self: &Arc<Self>,
        len: usize,
        attach_global: bool,
    ) -> Result<UnifiedBuffer<T>, HipError> {
        if !self.dev.has_managed_memory {
            return Err(HipError::NotSupported);
        }

        let attach_mode = if attach_global {
            MemAttachFlags::Global
        } else {
            MemAttachFlags::Host
        };

        let raw = unsafe {
            wrap_sys_res!(|raw| hipMallocManaged(raw.as_mut_ptr(), len * size_of::<T>(), attach_mode as u32))
        };

        let event = HipEvent::create(Some(EventFlags::BLOCKING_SYNC))?;

        Ok(UnifiedBuffer {
            ptr: raw?.cast(),
            len,
            stream: self.clone(),
            event,
            attach_mode,
            is_init: false,
            true_pad: true,
            _marker: PhantomData,
        })
    }

    /// Allocate unified (managed) memory.
    pub fn alloc_unified_zeroed<T: DevRepr>(
        self: &Arc<Self>,
        len: usize,
        attach_global: bool,
    ) -> Result<UnifiedBuffer<T>, HipError> {
        self.alloc_unified(len, attach_global).map(|mut buf| {
            if attach_global {
                buf.memset_host(0);
            } else {
                buf.memset_async(0).expect("failed to zero UnifiedBuffer");
            };

            unsafe {
                buf.assume_init();
            }
            buf
        })
    }
}

impl<T: DevRepr> UnifiedBuffer<T> {
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

    #[inline(always)]
    pub const fn attach_mode(&self) -> MemAttachFlags {
        self.attach_mode
    }

    /// Gets the pointer to GPU memory
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Gets the length of the allocation
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> UnifiedSlice<'_, T> {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe {
            UnifiedSlice::from_raw_parts(
                self.ptr.add(start),
                len,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.is_init,
            )
        }
    }

    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> UnifiedSliceMut<'_, T> {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe {
            UnifiedSliceMut::from_raw_parts(
                self.ptr.add(start),
                len,
                &self.stream,
                &self.event,
                self.attach_mode,
                self.is_init,
            )
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedBuffer": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize UnifiedBuffer": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len as usize) }
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

    /// Re-attach to a new stream and buffer
    pub fn attach(&mut self, flags: MemAttachFlags) -> Result<(), HipError> {
        self.event.sync()?;
        self.attach_mode = flags;
        unsafe {
            try_err!(
                hipStreamAttachMemAsync(
                    self.stream.raw,
                    self.ptr as *mut _,
                    self.size(),
                    self.attach_mode as i32,
                ),
                Ok(())
            )
        }
    }

    /// Hint HIP to prefetch the buffer to the appropriate location based
    /// on the current [`attach_mode`](`Self::attach_mode`).
    ///
    /// - `Global` / `Single`: prefetch to the stream's device (requires
    ///   `hipDeviceAttributeConcurrentManagedAccess`).
    /// - `Host`: prefetch to host memory ([`hipCpuDeviceId`]).
    pub fn prefetch(&self) -> Result<(), HipError> {
        let device = match self.attach_mode {
            MemAttachFlags::Global | MemAttachFlags::Single => {
                if !self.stream.dev.concurrent_managed_access {
                    return Err(HipError::NotSupported);
                }
                self.stream.dev.as_raw()
            }
            MemAttachFlags::Host => hipCpuDeviceId,
        };
        unsafe {
            try_err!(
                hipMemPrefetchAsync(self.ptr as *const _, self.size(), device, self.stream.raw,),
                Ok(())
            )
        }
    }

    pub fn memset_host(&mut self, byte: u8) {
        unsafe {
            write_bytes(self.ptr, byte, self.len);
        }
    }

    pub fn memset_dev<D: Memset>(&mut self, val: D) -> Result<(), HipError> {
        let hip_slice: &HipSliceMut<'_, T> = self.as_mut();
        hip_slice.memset(val)
    }

    pub fn memset_async<D: Memset>(&mut self, val: D) -> Result<(), HipError> {
        let hip_slice: &HipSliceMut<'_, T> = self.as_mut();
        hip_slice.memset_async(val)
    }

    /// Synchronize the stream if needed to guarantee safe host access.
    pub fn sync_host(&self) -> Result<(), HipError> {
        if matches!(self.attach_mode, MemAttachFlags::Single) {
            self.event.sync()?;
        }
        Ok(())
    }

    /// Validate that the given stream may access this buffer. Returns
    /// `Err(hipErrorNotSupported)` when the attach mode forbids it.
    pub fn check_dev_access(&self) -> Result<(), HipError> {
        if !self.is_init {
            return Err(HipError::InvalidValue);
        }
        match self.attach_mode {
            MemAttachFlags::Global => {}
            MemAttachFlags::Host => {
                if !self.stream.dev.concurrent_managed_access {
                    return Err(HipError::NotSupported);
                }
            }
            MemAttachFlags::Single => return Err(HipError::NotSupported),
        }
        Ok(())
    }
}

impl<T: DevRepr> Deref for UnifiedBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DevRepr> DerefMut for UnifiedBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: DevRepr> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        if self.stream.dev.has_mem_pools {
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

impl<'a, T: DevRepr> AsRef<UnifiedSlice<'a, T>> for UnifiedBuffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &UnifiedSlice<'a, T> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<UnifiedSlice<'a, T>> for UnifiedSliceMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &UnifiedSlice<'a, T> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsMut<UnifiedSliceMut<'a, T>> for UnifiedBuffer<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut UnifiedSliceMut<'a, T> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<T: DevRepr> AsRef<Self> for UnifiedSlice<'_, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<T: DevRepr> AsMut<Self> for UnifiedSliceMut<'_, T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Self {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HostSlice<'a, T>> for UnifiedBuffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &HostSlice<'a, T> {
        log_if_err!("failed to synchronize UnifiedBuffer": self.sync_host());
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HostSlice<'a, T>> for UnifiedSlice<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &HostSlice<'a, T> {
        log_if_err!("failed to synchronize UnifiedSlice": self.sync_host());
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HostSlice<'a, T>> for UnifiedSliceMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &HostSlice<'a, T> {
        log_if_err!("failed to synchronize UnifiedSliceMut": self.sync_host());
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsMut<HostSliceMut<'a, T>> for UnifiedBuffer<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut HostSliceMut<'a, T> {
        log_if_err!("failed to synchronize UnifiedBuffer": self.sync_host());
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T: DevRepr> AsMut<HostSliceMut<'a, T>> for UnifiedSliceMut<'_, T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut HostSliceMut<'a, T> {
        log_if_err!("failed to synchronize UnifiedSliceMut": self.sync_host());
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HipSlice<'a, T>> for UnifiedBuffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &HipSlice<'a, T> {
        self.check_dev_access().expect("unsafe device access on UnifiedBuffer");
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HipSlice<'a, T>> for UnifiedSlice<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &HipSlice<'a, T> {
        self.check_dev_access().expect("unsafe device access on UnifiedSlice");
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HipSlice<'a, T>> for UnifiedSliceMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &HipSlice<'a, T> {
        self.expect_init(None);
        self.check_dev_access().expect("unsafe device access on UnifiedSliceMut");
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsMut<HipSliceMut<'a, T>> for UnifiedBuffer<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut HipSliceMut<'a, T> {
        self.check_dev_access().expect("unsafe device access on UnifiedBuffer");
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T: DevRepr> AsMut<HipSliceMut<'a, T>> for UnifiedSliceMut<'_, T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut HipSliceMut<'a, T> {
        self.check_dev_access().expect("unsafe device access on UnifiedSliceMut");
        unsafe { &mut *(self as *mut Self).cast() }
    }
}
