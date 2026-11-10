use crate::hip::{HipError, event::HipEvent, memory::Buffer, stream::HipStream};
use alloc::sync::Arc;
use core::{
    cell::UnsafeCell,
    hint::unreachable_unchecked,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    slice,
};
use rocm_sys::hip::{
    hipError_t, hipHostFree, hipHostMalloc, hipHostRegister, hipHostUnregister, hipMemcpyDtoH,
    hipMemcpyHtoD,
};

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[derive(Debug)]
pub struct HostSlice<'a> {
    pub(crate) ptr: *const u8,
    pub(crate) size: u32,
    pub(crate) premapped: bool,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    _marker: PhantomData<&'a HostBuffer>,
}

impl<'a> HostSlice<'a> {
    /// Maps the ptr and return a handle
    pub fn new(ptr: &'a [u8], stream: Arc<HipStream>) -> Result<Self, HipError> {
        let res = unsafe { hipHostRegister(ptr.as_ptr() as *mut _, ptr.len(), 0) };
        let premapped = match res {
            hipError_t::hipSuccess => true,
            hipError_t::hipErrorHostMemoryAlreadyRegistered => false,
            res => try_err!(res, unsafe { unreachable_unchecked() }),
        };

        Ok(Self {
            ptr: ptr.as_ptr(),
            size: ptr.len() as u32,
            premapped,
            event: HipEvent::create()?,
            stream,
            _marker: PhantomData,
        })
    }

    pub fn latesync(&self) -> Result<(), HipError> {
        self.event.record(&self.stream)
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: u32,
        premapped: bool,
        stream: Arc<HipStream>,
        event: HipEvent,
    ) -> Self {
        Self {
            ptr,
            size,
            event,
            stream,
            premapped,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    #[inline(always)]
    pub const fn size(&self) -> u32 {
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

    pub fn copy_to_dev(
        &self,
        other: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > self.size)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > other.ptr.len() as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_to_dev_unchecked(other, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_to_dev_unchecked(
        &self,
        other: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    other.ptr.byte_add(dst_off as usize).cast(),
                    self.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }
}

impl Drop for HostSlice<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        let _ = self.event.sync();
        if !self.premapped {
            let _ = unsafe { self.unmap() };
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
#[derive(Debug)]
pub struct HostSliceMut<'a> {
    pub(crate) ptr: *mut u8,
    pub(crate) size: u32,
    pub(crate) premapped: bool,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    _marker: PhantomData<&'a HostBuffer>,
}

impl<'a> HostSliceMut<'a> {
    /// Maps the ptr and return a handle
    pub fn new(ptr: &'a mut [u8], stream: Arc<HipStream>) -> Result<Self, HipError> {
        let res = unsafe { hipHostRegister(ptr.as_mut_ptr().cast(), ptr.len(), 0) };
        let premapped = match res {
            hipError_t::hipSuccess => true,
            hipError_t::hipErrorHostMemoryAlreadyRegistered => false,
            res => try_err!(res, unsafe { unreachable_unchecked() }),
        };

        Ok(Self {
            ptr: ptr.as_mut_ptr(),
            size: ptr.len() as u32,
            premapped,
            event: HipEvent::create()?,
            stream,
            _marker: PhantomData,
        })
    }

    pub fn latesync(&self) -> Result<(), HipError> {
        self.event.record(&self.stream)
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: u32,
        premapped: bool,
        stream: Arc<HipStream>,
        event: HipEvent,
    ) -> Self {
        Self {
            ptr,
            size,
            premapped,
            event,
            stream,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    #[inline(always)]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    #[inline(always)]
    pub const fn size(&self) -> u32 {
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

    pub fn copy_to_dev(
        &self,
        other: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > self.size)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > other.ptr.len() as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_to_dev_unchecked(other, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_to_dev_unchecked(
        &self,
        other: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    other.ptr.byte_add(dst_off as usize).cast(),
                    self.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_from_dev(
        &self,
        other: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > self.size)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > other.ptr.len() as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_from_dev_unchecked(other, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_from_dev_unchecked(
        &self,
        other: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoH(
                    self.ptr.byte_add(src_off as usize) as *mut _,
                    other.ptr.byte_add(dst_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }
}

impl Drop for HostSliceMut<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        let _ = self.event.sync();
        if !self.premapped {
            let _ = unsafe { self.unmap() };
        }
    }
}

impl Deref for HostSliceMut<'_> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }
}

impl DerefMut for HostSliceMut<'_> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }
}

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[derive(Debug)]
pub struct HostBuffer {
    pub(crate) ptr: *mut u8,
    pub(crate) size: u32,
    pub(crate) is_mapped: bool,
    pub(crate) event: HipEvent,
    pub(crate) stream: Arc<HipStream>,
    _marker: PhantomData<UnsafeCell<()>>,
}

impl HipStream {
    /// Allocates `size` bytes and makes it accessible from all HIP devices
    pub fn alloc_host(self: &Arc<Self>, size: u32) -> Result<HostBuffer, HipError> {
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
            event: HipEvent::create()?,
            _marker: PhantomData,
        })
    }

    /// Allocates a slice of bytes and makes it accessible from all HIP devices
    pub fn alloc_host_init(self: &Arc<Self>, ptr: &[u8]) -> Result<HostBuffer, HipError> {
        let size = ptr.len();

        let buf = self.alloc_host(size as u32)?;

        unsafe {
            buf.ptr.copy_from_nonoverlapping(ptr.as_ptr(), size);
        }

        Ok(buf)
    }

    /// Borrows bytes and makes it accessible from all HIP devices
    pub fn host_slice<'a>(self: &Arc<Self>, ptr: &'a [u8]) -> Result<HostSlice<'a>, HipError> {
        HostSlice::new(ptr, self.clone())
    }

    /// Borrows bytes and makes it accessible from all HIP devices
    pub fn host_slice_mut<'a>(
        self: &Arc<Self>,
        ptr: &'a mut [u8],
    ) -> Result<HostSliceMut<'a>, HipError> {
        HostSliceMut::new(ptr, self.clone())
    }
}

impl HostBuffer {
    pub fn latesync(&self) -> Result<(), HipError> {
        self.event.record(&self.stream)
    }

    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: u32,
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

    #[inline(always)]
    pub const fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    #[inline(always)]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    #[inline(always)]
    pub const fn size(&self) -> u32 {
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
        let _ = self.event.sync();

        unsafe {
            let _ = self.dealloc_unchecked();
        }
    }
}

impl Deref for HostBuffer {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr, self.size as usize) }
    }
}

impl DerefMut for HostBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size as usize) }
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
