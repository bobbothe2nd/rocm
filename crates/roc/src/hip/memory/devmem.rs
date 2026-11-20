use crate::{
    auxiliary::{ResolvedRange, resolve_range},
    hip::{
        DevRepr, HipError,
        event::HipEvent,
        memory::{CopyTo, DevPtr, EventSync, Memset, StreamBound},
        stream::HipStream,
    },
};
use alloc::sync::Arc;
use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, ops::RangeBounds};
use rocm_sys::hip::{hipFree, hipFreeAsync, hipMalloc, hipMallocAsync};

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipSlice<'a, T: DevRepr> {
    pub(crate) ptr: *const T,
    pub(crate) len: usize,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    _marker: PhantomData<&'a UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for HipSlice<'_, T> {}
unsafe impl<T: DevRepr> Sync for HipSlice<'_, T> {}

impl<T: DevRepr> HipSlice<'_, T> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe { Self::from_raw_parts(self.ptr.add(start), len, &self.stream, &self.event) }
    }

    /// Constructs a GPU buffer from a raw pointer, len, stream, and event
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *const T,
        len: usize,
        stream: &Arc<HipStream>,
        event: &HipEvent,
    ) -> Self {
        Self {
            ptr,
            len,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        }
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

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
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
    pub fn copy_to_async<Dst>(
        &self,
        dst: &mut Dst,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>
    where
        Self: CopyTo<T, Dst>,
        Dst: DevPtr<T> + StreamBound + EventSync,
    {
        super::copy_async(self, dst, src_off, dst_off, len)
    }
}

impl<'a, T: DevRepr> HipSlice<'a, T> {
    /// Reinterpret the buffer as `[U]`.
    ///
    /// Returns `None` if the byte length isn't divisible by `size_of::<U>()`.
    ///
    /// Consumes `self`; the returned slice continues to own the
    /// underlying allocation and any read/write events.
    pub fn transmute<U: DevRepr>(self) -> Option<HipSlice<'a, U>> {
        let bytes = self.size();
        let u_size = size_of::<U>();
        if u_size == 0 || !bytes.is_multiple_of(u_size) {
            return None;
        }
        let new_len = bytes / u_size;
        let handle = ManuallyDrop::new(self);
        Some(HipSlice {
            ptr: handle.ptr.cast(),
            len: new_len,
            event: unsafe { core::ptr::read(&handle.event) },
            stream: unsafe { core::ptr::read(&handle.stream) },
            _marker: PhantomData,
        })
    }
}

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipSliceMut<'a, T: DevRepr> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    _marker: PhantomData<&'a mut UnsafeCell<[T]>>,
}

unsafe impl<T: DevRepr> Send for HipSliceMut<'_, T> {}
unsafe impl<T: DevRepr> Sync for HipSliceMut<'_, T> {}

impl<T: DevRepr> HipSliceMut<'_, T> {
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe { Self::from_raw_parts(self.ptr.add(start), len, &self.stream, &self.event) }
    }

    /// Constructs a GPU buffer from a raw pointer, len, stream, and event
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut T,
        len: usize,
        stream: &Arc<HipStream>,
        event: &HipEvent,
    ) -> Self {
        Self {
            ptr,
            len,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        }
    }

    /// Gets the pointer to GPU memory
    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Gets the length of the allocation
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
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
        Dst: DevPtr<T> + StreamBound + EventSync,
    {
        super::copy_async(self, dst, src_off, dst_off, len)
    }

    pub fn memset<D: Memset>(&self, val: D) -> Result<(), HipError> {
        let bytes = self.len * size_of::<T>();

        if !bytes.is_multiple_of(size_of::<D>()) {
            return Err(HipError::InvalidValue);
        }

        self.sync()?;

        unsafe { D::memset_sync(self.ptr.cast(), val, bytes / size_of::<D>()) }
    }

    pub fn memset_async<D: Memset>(&self, val: D) -> Result<(), HipError> {
        let bytes = self.len * size_of::<T>();

        if !bytes.is_multiple_of(size_of::<D>()) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            D::memset_async(self.ptr.cast(), val, bytes / size_of::<D>(), &self.stream)?;
        }

        self.latesync()?;

        Ok(())
    }
}

impl<'a, T: DevRepr> HipSliceMut<'a, T> {
    /// Reinterpret the buffer as `[U]`.
    ///
    /// Returns `None` if the byte length isn't divisible by `size_of::<U>()`.
    ///
    /// Consumes `self`; the returned slice continues to own the
    /// underlying allocation and any read/write events.
    pub fn transmute<U: DevRepr>(self) -> Option<HipSliceMut<'a, U>> {
        let bytes = self.size();
        let u_size = size_of::<U>();
        if u_size == 0 || !bytes.is_multiple_of(u_size) {
            return None;
        }
        let new_len = bytes / u_size;
        let handle = ManuallyDrop::new(self);
        Some(HipSliceMut {
            ptr: handle.ptr.cast(),
            len: new_len,
            event: unsafe { core::ptr::read(&handle.event) },
            stream: unsafe { core::ptr::read(&handle.stream) },
            _marker: PhantomData,
        })
    }
}

/// Handle to an allocation on the GPU
#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug)]
pub struct HipBuffer<T: DevRepr> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    pub(super) _marker: PhantomData<UnsafeCell<[T]>>,
}

impl<T: DevRepr> Clone for HipBuffer<T> {
    fn clone(&self) -> Self {
        let mut dst = self
            .stream
            .alloc::<T>(self.len)
            .expect("HIP allocation failed");
        self.copy_to_async(&mut dst, 0, 0, self.len)
            .expect("HIP DtoD async copy failed");
        dst
    }
}

unsafe impl<T: DevRepr> Send for HipBuffer<T> {}
unsafe impl<T: DevRepr> Sync for HipBuffer<T> {}

impl HipStream {
    /// Allocates `len` elements
    pub fn alloc<T: DevRepr>(self: &Arc<Self>, len: usize) -> Result<HipBuffer<T>, HipError> {
        let event = HipEvent::create(None)?;

        let ptr: Result<*mut T, HipError> = if self.dev.has_mem_pools {
            unsafe {
                wrap_sys_res!(|ptr: *mut T| hipMallocAsync(
                    ptr.as_mut_ptr().cast(),
                    len * size_of::<T>(),
                    self.raw
                ))
            }
        } else {
            unsafe {
                wrap_sys_res!(|ptr: *mut T| hipMalloc(
                    ptr.as_mut_ptr().cast(),
                    len * size_of::<T>()
                ))
            }
        };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(HipBuffer {
            ptr,
            len,
            stream: self.clone(),
            event,
            _marker: PhantomData,
        })
    }

    /// Allocates `len` zeroed elements
    pub fn alloc_zeroed<T: DevRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<HipBuffer<T>, HipError> {
        self.alloc(len).and_then(|mut buf| {
            buf.memset_async(0)?;
            Ok(buf)
        })
    }
}

impl<T: DevRepr> HipBuffer<T> {
    /// Constructs a GPU buffer from a raw pointer, len, stream, and event
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
            stream,
            event,
            _marker: PhantomData,
        }
    }

    /// Reinterpret the buffer as `[U]`.
    ///
    /// Returns `None` if the byte length isn't divisible by `size_of::<U>()`.
    ///
    /// Consumes `self`; the returned slice continues to own the
    /// underlying allocation and any read/write events.
    pub fn transmute<U: DevRepr>(self) -> Option<HipBuffer<U>> {
        let bytes = self.size();
        let u_size = size_of::<U>();
        if u_size == 0 || !bytes.is_multiple_of(u_size) {
            return None;
        }
        let new_len = bytes / u_size;
        let handle = ManuallyDrop::new(self);
        Some(HipBuffer {
            ptr: handle.ptr.cast(),
            len: new_len,
            event: unsafe { core::ptr::read(&handle.event) },
            stream: unsafe { core::ptr::read(&handle.stream) },
            _marker: PhantomData,
        })
    }

    /// Gets the pointer to GPU memory
    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Gets the length of the allocation
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> HipSlice<'_, T> {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe { HipSlice::from_raw_parts(self.ptr.add(start), len, &self.stream, &self.event) }
    }

    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> HipSliceMut<'_, T> {
        let ResolvedRange { start, len, end: _ } = resolve_range(range, self.len);
        unsafe { HipSliceMut::from_raw_parts(self.ptr.add(start), len, &self.stream, &self.event) }
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
        unsafe { try_err!(hipFreeAsync(self.ptr.cast(), self.stream.raw,), Ok(())) }
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
        Dst: DevPtr<T> + StreamBound + EventSync,
    {
        super::copy_async(self, dst, src_off, dst_off, len)
    }

    #[inline(always)]
    pub fn memset<D: Memset>(&mut self, val: D) -> Result<(), HipError> {
        let bytes = self.len * size_of::<T>();

        if !bytes.is_multiple_of(size_of::<D>()) {
            return Err(HipError::InvalidValue);
        }

        unsafe { D::memset_sync(self.ptr.cast(), val, bytes / size_of::<D>()) }
    }

    pub fn memset_async<D: Memset>(&mut self, val: D) -> Result<(), HipError> {
        let bytes = self.len * size_of::<T>();

        if !bytes.is_multiple_of(size_of::<D>()) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            D::memset_async(self.ptr.cast(), val, bytes / size_of::<D>(), &self.stream)?;
        }

        self.latesync()?;

        Ok(())
    }
}

impl<T: DevRepr> Drop for HipBuffer<T> {
    fn drop(&mut self) {
        if self.stream.dev.has_mem_pools {
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

impl<'a, T: DevRepr> AsRef<HipSlice<'a, T>> for HipBuffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &HipSlice<'a, T> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsRef<HipSlice<'a, T>> for HipSliceMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &HipSlice<'a, T> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a, T: DevRepr> AsMut<HipSliceMut<'a, T>> for HipBuffer<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut HipSliceMut<'a, T> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<T: DevRepr> AsRef<Self> for HipSlice<'_, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<T: DevRepr> AsMut<Self> for HipSliceMut<'_, T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Self {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}
