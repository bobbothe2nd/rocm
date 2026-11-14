use crate::hip::{
    HipError,
    event::HipEvent,
    memory::{CopyTo, DevPtr, StreamBound},
    stream::HipStream,
};
use alloc::sync::Arc;
use core::{
    cell::UnsafeCell,
    hint::unreachable_unchecked,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Bound, Deref, DerefMut, RangeBounds},
    slice,
};
use rocm_sys::hip::{hipError_t, hipHostFree, hipHostMalloc, hipHostRegister, hipHostUnregister};

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct HostSlice<'a> {
    pub(crate) ptr: *const u8,
    pub(crate) size: usize,
    pub(crate) premapped: bool,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    _marker: PhantomData<&'a UnsafeCell<[u8]>>,
}

unsafe impl Send for HostSlice<'_> {}
unsafe impl Sync for HostSlice<'_> {}

impl<'a> HostSlice<'a> {
    /// Maps the ptr and return a handle
    pub fn new(ptr: &'a [u8], stream: &Arc<HipStream>) -> Result<Self, HipError> {
        let res = unsafe { hipHostRegister(ptr.as_ptr() as *mut _, ptr.len(), 0) };
        let premapped = match res {
            hipError_t::hipSuccess => true,
            hipError_t::hipErrorHostMemoryAlreadyRegistered => false,
            res => try_err!(res, unsafe { unreachable_unchecked() }),
        };

        Ok(Self {
            ptr: ptr.as_ptr(),
            size: ptr.len() as usize,
            premapped,
            event: ManuallyDrop::new(HipEvent::create(None)?),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        })
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: usize,
        premapped: bool,
        stream: &Arc<HipStream>,
        event: &HipEvent,
    ) -> Self {
        Self {
            ptr,
            size,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            premapped,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        log_if_err!("failed to synchronize HostSlice": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Maps the buffer such that is is accessible
    #[inline(always)]
    pub fn map(&self) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipHostRegister(self.ptr as *mut _, self.size as usize, 0),
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
}

impl Drop for HostSlice<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        log_if_err!("failed to synchronize HostSlice": self.event.sync());
        if !self.premapped {
            log_if_err!("failed to deallocate HostSlice": unsafe { self.unmap() });
        }
    }
}

impl Deref for HostSlice<'_> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct HostSliceMut<'a> {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    pub(crate) premapped: bool,
    pub(crate) event: ManuallyDrop<HipEvent>,
    pub(crate) stream: ManuallyDrop<Arc<HipStream>>,
    _marker: PhantomData<&'a mut UnsafeCell<[u8]>>,
}

unsafe impl Send for HostSliceMut<'_> {}
unsafe impl Sync for HostSliceMut<'_> {}

impl<'a> HostSliceMut<'a> {
    /// Maps the ptr and return a handle
    pub fn new(ptr: &'a mut [u8], stream: &Arc<HipStream>) -> Result<Self, HipError> {
        let res = unsafe { hipHostRegister(ptr.as_mut_ptr().cast(), ptr.len(), 0) };
        let premapped = match res {
            hipError_t::hipSuccess => true,
            hipError_t::hipErrorHostMemoryAlreadyRegistered => false,
            res => try_err!(res, unsafe { unreachable_unchecked() }),
        };

        Ok(Self {
            ptr: ptr.as_mut_ptr(),
            size: ptr.len() as usize,
            premapped,
            event: ManuallyDrop::new(HipEvent::create(None)?),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        })
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: usize,
        premapped: bool,
        stream: &Arc<HipStream>,
        event: &HipEvent,
    ) -> Self {
        Self {
            ptr,
            size,
            premapped,
            event: ManuallyDrop::new(unsafe { core::ptr::read(event) }),
            stream: ManuallyDrop::new(unsafe { core::ptr::read(stream) }),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        log_if_err!("failed to synchronize HostSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        log_if_err!("failed to synchronize HostSliceMut": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Maps the buffer such that is is accessible
    #[inline(always)]
    pub fn map(&self) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipHostRegister(self.ptr.cast(), self.size as usize, 0),
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
}

impl Drop for HostSliceMut<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        log_if_err!("failed to synchronize HostSlice": self.event.sync());
        if !self.premapped {
            log_if_err!("failed to deallocate HostSlice": unsafe { self.unmap() });
        }
    }
}

impl Deref for HostSliceMut<'_> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for HostSliceMut<'_> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct HostBuffer {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    pub(crate) is_mapped: bool,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    _marker: PhantomData<UnsafeCell<[u8]>>,
}

unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HipStream {
    /// Allocates `size` bytes and makes it accessible from all HIP devices
    pub fn alloc_host(self: &Arc<Self>, size: usize) -> Result<HostBuffer, HipError> {
        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipHostMalloc((&raw mut ptr).cast(), size as usize, 0)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(HostBuffer {
            ptr,
            size,
            is_mapped: true,
            stream: self.clone(),
            event: HipEvent::create(None)?,
            _marker: PhantomData,
        })
    }

    /// Allocates a slice of bytes and makes it accessible from all HIP devices
    pub fn alloc_host_init(self: &Arc<Self>, ptr: &[u8]) -> Result<HostBuffer, HipError> {
        let size = ptr.len();

        let buf = self.alloc_host(size as usize)?;

        unsafe {
            buf.ptr.copy_from_nonoverlapping(ptr.as_ptr(), size);
        }

        Ok(buf)
    }

    /// Borrows bytes and makes it accessible from all HIP devices
    pub fn host_slice<'a>(self: &Arc<Self>, ptr: &'a [u8]) -> Result<HostSlice<'a>, HipError> {
        HostSlice::new(ptr, &self)
    }

    /// Borrows bytes and makes it accessible from all HIP devices
    pub fn host_slice_mut<'a>(
        self: &Arc<Self>,
        ptr: &'a mut [u8],
    ) -> Result<HostSliceMut<'a>, HipError> {
        HostSliceMut::new(ptr, &self)
    }
}

impl HostBuffer {
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
            is_mapped: false,
            stream,
            event,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        log_if_err!("failed to synchronize HostBuffer": self.event.sync());
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        log_if_err!("failed to synchronize HostBuffer": self.event.sync());
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> HostSlice<'_> {
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Excluded(start) => start + 1,
            Bound::Included(start) => *start,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => self.size,
            Bound::Excluded(end) => end - 1,
            Bound::Included(end) => *end,
        };
        let size = end - start;

        unsafe {
            HostSlice::from_raw_parts(
                self.ptr.add(start),
                size,
                self.is_mapped,
                &self.stream,
                &self.event,
            )
        }
    }

    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> HostSliceMut<'_> {
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Excluded(start) => start + 1,
            Bound::Included(start) => *start,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => self.size,
            Bound::Excluded(end) => end - 1,
            Bound::Included(end) => *end,
        };
        let size = end - start;

        unsafe {
            HostSliceMut::from_raw_parts(
                self.ptr.add(start),
                size,
                self.is_mapped,
                &self.stream,
                &self.event,
            )
        }
    }

    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.size
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
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        log_if_err!("failed to synchronize HostBuffer": self.event.sync());
        log_if_err!("failed to deallocate HostBuffer": unsafe { self.dealloc_unchecked() });
    }
}

impl Deref for HostBuffer {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for HostBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a> AsRef<HostSlice<'a>> for HostBuffer {
    #[inline(always)]
    fn as_ref(&self) -> &HostSlice<'a> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl<'a> AsMut<HostSliceMut<'a>> for HostBuffer {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut HostSliceMut<'a> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a> AsRef<HostSlice<'a>> for HostSliceMut<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &HostSlice<'a> {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl AsRef<Self> for HostSlice<'_> {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl AsMut<Self> for HostSliceMut<'_> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}
