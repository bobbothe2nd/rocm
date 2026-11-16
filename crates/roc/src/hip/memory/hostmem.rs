use crate::{
    auxiliary::{ResolvedRange, resolve_range}, hip::{
        DevRepr, HipError, event::HipEvent, memory::{CopyTo, DevPtr, StreamBound}, stream::HipStream,
    },
};
use alloc::sync::Arc;
use core::{
    cell::UnsafeCell,
    hint::unreachable_unchecked,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, RangeBounds},
    ptr::write_bytes,
    slice,
};
use rocm_sys::hip::{hipError_t, hipHostFree, hipHostMalloc, hipHostRegister, hipHostUnregister};

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HostSlice<'a, T: DevRepr> {
    pub(crate) ptr: *const T,
    pub(crate) len: usize,
    pub(crate) premapped: bool,
    pub(crate) is_init: bool,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    _marker: PhantomData<&'a UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for HostSlice<'_, T> {}
unsafe impl<T: DevRepr> Sync for HostSlice<'_, T> {}

impl<'a, T: DevRepr> HostSlice<'a, T> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange {
            start,
            len,
            end: _,
        } = resolve_range(range, self.len);
        unsafe {
            Self::from_raw_parts(
                self.ptr.add(start),
                len,
                self.premapped,
                &self.stream,
                &self.event,
                self.is_init,
            )
        }
    }

    /// Maps the ptr and return a handle
    pub fn new(ptr: &'a [T], stream: &Arc<HipStream>) -> Result<Self, HipError> {
        let res = unsafe { hipHostRegister(ptr.as_ptr() as *mut _, ptr.len(), 0) };
        let premapped = match res {
            hipError_t::hipSuccess => true,
            hipError_t::hipErrorHostMemoryAlreadyRegistered => false,
            res => try_err!(res, unsafe { unreachable_unchecked() }),
        };

        Ok(Self {
            ptr: ptr.as_ptr(),
            len: ptr.len(),
            premapped,
            is_init: true,
            event: ManuallyDrop::new(HipEvent::create(None)?),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        })
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *const T,
        len: usize,
        premapped: bool,
        stream: &Arc<HipStream>,
        event: &HipEvent,
        is_init: bool,
    ) -> Self {
        Self {
            ptr,
            len,
            premapped,
            is_init,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
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
                None => "expected init HostSlice, encountered uninit",
            }
        );
    }

    #[inline(always)]
    pub const unsafe fn assume_init(&mut self) {
        self.is_init = true;
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_slice(&self) -> &[T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize HostSlice": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Maps the buffer such that is is accessible
    #[inline(always)]
    pub fn map(&self) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipHostRegister(self.ptr as *mut _, self.len, 0),
                Ok(())
            )
        }
    }

    /// Unmaps the buffer such that is is inaccessible
    ///
    /// # Safety
    ///
    /// The buffer must never be deallocated again or dropped.
    #[inline(always)]
    pub unsafe fn unmap(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipHostUnregister(self.ptr as *mut _), Ok(())) }
    }

    #[inline]
    pub fn copy_from_sync<Src>(
        &mut self,
        src: &Src,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Src: CopyTo<T, Self>,
    {
        super::copy(src, self, src_off, dst_off, len)
    }

    #[inline]
    pub fn copy_to_sync<Dst>(
        &self,
        dst: &mut Dst,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Self: CopyTo<T, Dst>,
        Dst: DevPtr<T>,
    {
        super::copy(self, dst, src_off, dst_off, len)
    }

    #[inline]
    pub fn copy_from_async<Src>(
        &mut self,
        src: &Src,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Src: CopyTo<T, Self> + StreamBound,
    {
        super::copy_async(src, self, src_off, dst_off, len)
    }

    #[inline]
    pub fn copy_to_async<Dst>(
        &self,
        dst: &mut Dst,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Self: CopyTo<T, Dst>,
        Dst: DevPtr<T> + StreamBound,
    {
        super::copy_async(self, dst, src_off, dst_off, len)
    }
}

impl<T: DevRepr> Drop for HostSlice<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        log_if_err!("failed to synchronize HostSlice": self.event.sync());
        if !self.premapped {
            log_if_err!("failed to deallocate HostSlice": unsafe { self.unmap() });
        }
    }
}

impl<T: DevRepr> Deref for HostSlice<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HostSliceMut<'a, T: DevRepr> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) premapped: bool,
    pub(crate) is_init: bool,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    _marker: PhantomData<&'a mut UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for HostSliceMut<'_, T> {}
unsafe impl<T: DevRepr> Sync for HostSliceMut<'_, T> {}

impl<'a, T: DevRepr> HostSliceMut<'a, T> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange {
            start,
            len,
            end: _,
        } = resolve_range(range, self.len);
        unsafe {
            Self::from_raw_parts(
                self.ptr.add(start),
                len,
                self.premapped,
                &self.stream,
                &self.event,
                self.is_init,
            )
        }
    }

    /// Maps the ptr and return a handle
    pub fn new(ptr: &'a mut [T], stream: &Arc<HipStream>) -> Result<Self, HipError> {
        let res = unsafe { hipHostRegister(ptr.as_mut_ptr().cast(), ptr.len(), 0) };
        let premapped = match res {
            hipError_t::hipSuccess => true,
            hipError_t::hipErrorHostMemoryAlreadyRegistered => false,
            res => try_err!(res, unsafe { unreachable_unchecked() }),
        };

        Ok(Self {
            ptr: ptr.as_mut_ptr(),
            len: ptr.len(),
            premapped,
            is_init: true,
            event: ManuallyDrop::new(HipEvent::create(None)?),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        })
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut T,
        len: usize,
        premapped: bool,
        stream: &Arc<HipStream>,
        event: &HipEvent,
        is_init: bool,
    ) -> Self {
        Self {
            ptr,
            len,
            premapped,
            is_init,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    #[inline(always)]
    pub const fn expect_init(&self, msg: Option<&str>) {
        assert!(
            self.is_init,
            "{}",
            match msg {
                Some(msg) => msg,
                None => "expected init HostSliceMut, encountered uninit",
            }
        );
    }

    #[inline(always)]
    pub const unsafe fn assume_init(&mut self) {
        self.is_init = true;
    }

    pub fn as_slice(&self) -> &[T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize HostSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize HostSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Maps the buffer such that is is accessible
    #[inline(always)]
    pub fn map(&self) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipHostRegister(self.ptr.cast(), self.len, 0),
                Ok(())
            )
        }
    }

    /// Unmaps the buffer such that is is inaccessible
    ///
    /// # Safety
    ///
    /// The buffer must never be deallocated again or dropped.
    #[inline(always)]
    pub unsafe fn unmap(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipHostUnregister(self.ptr.cast()), Ok(())) }
    }

    #[inline]
    pub fn copy_from_sync<Src>(
        &mut self,
        src: &Src,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Src: CopyTo<T, Self>,
    {
        super::copy(src, self, src_off, dst_off, len)
    }

    #[inline]
    pub fn copy_to_sync<Dst>(
        &self,
        dst: &mut Dst,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Self: CopyTo<T, Dst>,
        Dst: DevPtr<T>,
    {
        super::copy(self, dst, src_off, dst_off, len)
    }

    #[inline]
    pub fn copy_from_async<Src>(
        &mut self,
        src: &Src,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Src: CopyTo<T, Self> + StreamBound,
    {
        super::copy_async(src, self, src_off, dst_off, len)
    }

    #[inline]
    pub fn copy_to_async<Dst>(
        &self,
        dst: &mut Dst,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Self: CopyTo<T, Dst>,
        Dst: DevPtr<T> + StreamBound,
    {
        super::copy_async(self, dst, src_off, dst_off, len)
    }

    pub fn memset(&self, byte: u8) {
        unsafe {
            write_bytes(self.ptr, byte, self.len);
        }
    }
}

impl<T: DevRepr> Drop for HostSliceMut<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        log_if_err!("failed to synchronize HostSlice": self.event.sync());
        if !self.premapped {
            log_if_err!("failed to deallocate HostSlice": unsafe { self.unmap() });
        }
    }
}

impl<T: DevRepr> Deref for HostSliceMut<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DevRepr> DerefMut for HostSliceMut<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HostBuffer<T: DevRepr> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) is_mapped: bool,
    pub(crate) is_init: bool,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    _marker: PhantomData<UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for HostBuffer<T> {}
unsafe impl<T: DevRepr> Sync for HostBuffer<T> {}

impl HipStream {
    /// Allocates `len` elements and makes it accessible from all HIP devices
    pub fn alloc_host<T: DevRepr>(self: &Arc<Self>, len: usize) -> Result<HostBuffer<T>, HipError> {
        let ptr: Result<*mut T, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipHostMalloc((&raw mut ptr).cast(), len, 0)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(HostBuffer {
            ptr,
            len,
            is_mapped: true,
            is_init: false,
            stream: self.clone(),
            event: HipEvent::create(None)?,
            _marker: PhantomData,
        })
    }

    /// Allocates `len` elements and makes it accessible from all HIP devices
    pub fn alloc_host_zeroed<T: DevRepr>(self: &Arc<Self>, len: usize) -> Result<HostBuffer<T>, HipError> {
        self.alloc_host(len).map(|mut buf| {
            buf.memset(0);
            unsafe {
                buf.assume_init();
            }
            buf
        })
    }

    /// Allocates a slice of elements and makes it accessible from all HIP devices
    pub fn alloc_host_init<T: DevRepr>(self: &Arc<Self>, ptr: &[T]) -> Result<HostBuffer<T>, HipError> {
        let len = ptr.len();

        let mut buf = self.alloc_host::<T>(len)?;

        unsafe {
            buf.ptr.copy_from_nonoverlapping(ptr.as_ptr(), len);

            buf.assume_init();
        }

        Ok(buf)
    }

    /// Borrows elements and makes it accessible from all HIP devices
    pub fn map_slice<'a, T: DevRepr>(self: &Arc<Self>, ptr: &'a [T]) -> Result<HostSlice<'a, T>, HipError> {
        HostSlice::new(ptr, self)
    }

    /// Borrows elements and makes it accessible from all HIP devices
    pub fn map_slice_mut<'a, T: DevRepr>(
        self: &Arc<Self>,
        ptr: &'a mut [T],
    ) -> Result<HostSliceMut<'a, T>, HipError> {
        HostSliceMut::new(ptr, self)
    }
}

impl<T: DevRepr> HostBuffer<T> {
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut T,
        len: usize,
        stream: Arc<HipStream>,
        event: HipEvent,
    ) -> Self {
        Self {
            ptr,
            len,
            is_mapped: true,
            is_init: true,
            stream,
            event,
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
                None => "expected init HostBuffer, encountered uninit",
            }
        );
    }

    #[inline(always)]
    pub const unsafe fn assume_init(&mut self) {
        self.is_init = true;
    }

    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn as_slice(&self) -> &[T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize HostBuffer": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.expect_init(None);
        log_if_err!("failed to synchronize HostBuffer": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> HostSlice<'_, T> {
        let ResolvedRange {
            start,
            len,
            end: _,
        } = resolve_range(range, self.len);
        unsafe {
            HostSlice::from_raw_parts(
                self.ptr.add(start),
                len,
                self.is_mapped,
                &self.stream,
                &self.event,
                self.is_init,
            )
        }
    }

    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> HostSliceMut<'_, T> {
        let ResolvedRange {
            start,
            len,
            end: _,
        } = resolve_range(range, self.len);
        unsafe {
            HostSliceMut::from_raw_parts(
                self.ptr.add(start),
                len,
                self.is_mapped,
                &self.stream,
                &self.event,
                self.is_init,
            )
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Deallocates the GPU-accessible buffer and synchronizes
    ///
    /// This allows you to safely handle errors which would normally panic when dropped.
    pub fn dealloc(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe {
            handle.dealloc_unchecked()?;
        }

        Ok(())
    }

    /// Deallocates the GPU-accessible buffer and synchronizes
    ///
    /// # Safety
    ///
    /// The buffer must never be deallocated again or dropped and the buffer can't be used again.
    pub unsafe fn dealloc_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipHostFree(self.ptr.cast()), Ok(())) }
    }

    pub fn memset(&self, byte: u8) {
        unsafe {
            write_bytes(self.ptr, byte, self.len);
        }
    }
}

impl<T: DevRepr> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        log_if_err!("failed to synchronize HostBuffer": self.event.sync());
        log_if_err!("failed to deallocate HostBuffer": unsafe { self.dealloc_unchecked() });
    }
}

impl<T: DevRepr> Deref for HostBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DevRepr> DerefMut for HostBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, T: DevRepr> AsRef<HostSlice<'a, T>> for HostBuffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &HostSlice<'a, T> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsMut<HostSliceMut<'a, T>> for HostBuffer<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut HostSliceMut<'a, T> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HostSlice<'a, T>> for HostSliceMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &HostSlice<'a, T> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<T: DevRepr> AsRef<Self> for HostSlice<'_, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T: DevRepr> AsMut<Self> for HostSliceMut<'_, T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}
