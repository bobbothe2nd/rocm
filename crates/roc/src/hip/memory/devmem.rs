use crate::hip::{
    HipError,
    event::HipEvent,
    memory::{CopyTo, DevPtr, EventSync, Memset, StreamBound},
    stream::HipStream,
};
use alloc::sync::Arc;
use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop};
use rocm_sys::hip::{hipFree, hipMalloc, hipMemsetD8};

/// Handle to an allocation on the GPU
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct HipBuffer {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    _marker: PhantomData<UnsafeCell<()>>,
}

unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

impl HipStream {
    /// Allocates `size` bytes on the default device
    pub fn alloc(self: &Arc<Self>, size: usize) -> Result<HipBuffer, HipError> {
        let event = HipEvent::create(None)?;

        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipMalloc((&raw mut ptr).cast(), size)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(HipBuffer {
            ptr,
            size,
            stream: self.clone(),
            event,
            _marker: PhantomData,
        })
    }

    /// Allocates `size` zeroed bytes on the default device
    pub fn alloc_zeroed(self: &Arc<Self>, size: usize) -> Result<HipBuffer, HipError> {
        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipMalloc((&raw mut ptr).cast(), size)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            try_err!(hipMemsetD8(ptr.cast(), 0, size as usize));
        }

        Ok(HipBuffer {
            ptr,
            size,
            stream: self.clone(),
            event: HipEvent::create(None)?,
            _marker: PhantomData,
        })
    }
}

impl HipBuffer {
    /// Constructs a GPU buffer from a raw pointer, size, stream, and event
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: usize,
        stream: Arc<HipStream>,
        event: HipEvent,
    ) -> Self {
        Self {
            ptr,
            size,
            stream,
            event,
            _marker: PhantomData,
        }
    }

    /// Gets the pointer to GPU memory
    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr.cast()
    }

    /// Gets the size of the allocation
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Deallocates a buffer
    ///
    /// This allows you to safely raw errors which would normally panic when dropped.
    #[inline(always)]
    pub fn dealloc(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe {
            handle.dealloc_unchecked()?;
        }

        Ok(())
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

        unsafe { handle.stream.free_unchecked(&handle) }
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
        Src: CopyTo<Self>,
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
        Self: CopyTo<Dst>,
        Dst: DevPtr,
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
        Src: CopyTo<Self> + StreamBound,
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
        Self: CopyTo<Dst>,
        Dst: DevPtr + StreamBound,
    {
        super::copy_async(self, dst, src_off, dst_off, len)
    }

    pub fn memset<T: Memset>(&self, val: T) -> Result<(), HipError> {
        if !self.size.is_multiple_of(size_of::<T>()) {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.memset_unchecked(val) }
    }

    #[inline(always)]
    pub unsafe fn memset_unchecked<T: Memset>(&self, val: T) -> Result<(), HipError> {
        unsafe { T::memset_sync(self.ptr.cast(), val, self.size) }
    }

    pub fn memset_async<T: Memset>(&self, val: T) -> Result<(), HipError> {
        if !self.size.is_multiple_of(size_of::<T>()) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            T::memset_async(self.ptr.cast(), val, self.size, &self.stream)?;
        }

        self.latesync()?;

        Ok(())
    }
}

impl EventSync for HipBuffer {
    fn latesync(&self) -> Result<(), crate::hip::HipError> {
        self.event.record(&self.stream, None)
    }

    fn sync(&self) -> Result<(), crate::hip::HipError> {
        self.event.sync()
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if self.stream.has_async_alloc {
            unsafe {
                log_if_err!("failed to deallocate HipBuffer": self.stream.free_unchecked(self));
            }
        } else {
            log_if_err!("failed to deallocate HipBuffer": self.event.sync());
            unsafe {
                log_if_err!("failed to deallocate HipBuffer": self.dealloc_unchecked());
            }
        }
    }
}
