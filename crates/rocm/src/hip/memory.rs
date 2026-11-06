use crate::hip::{HipError, event::HipEvent, stream::HipStream};
use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, slice::from_raw_parts};
use rocm_sys::hip::{
    hipFree, hipHostFree, hipHostMalloc, hipHostRegister, hipHostUnregister, hipMalloc,
    hipMemPool_t, hipMemcpyDtoD, hipMemcpyDtoH, hipMemcpyHtoD, hipMemsetAsync, hipMemsetD8,
};

/// Handle to CPU memory, mapped to all GPUs on the device
#[derive(Debug)]
pub struct RefMapped<'a> {
    pub(crate) data: &'a [u8],
    _marker: PhantomData<UnsafeCell<()>>,
}

impl<'a> RefMapped<'a> {
    /// Maps the data and return a handle
    ///
    /// # Safety
    ///
    /// Any asynchronous operations on this handle must be completed before it returns
    pub unsafe fn new(data: &'a mut [u8]) -> Result<Self, HipError> {
        unsafe {
            try_err!(hipHostRegister(data.as_mut_ptr().cast(), data.len(), 0));
        }

        Ok(Self {
            data,
            _marker: PhantomData,
        })
    }

    pub const unsafe fn from_slice(data: &'a [u8]) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    pub const fn as_slice(&self) -> &[u8] {
        self.data
    }

    pub const fn size(&self) -> u32 {
        self.data.len() as u32
    }

    /// Maps the buffer such that is is
    pub fn map(&self) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipHostRegister(self.data.as_ptr() as *mut _, self.data.len(), 0),
                Ok(())
            )
        }
    }

    /// Unmaps the buffer such that is is inaccessible
    ///
    /// # Safety
    ///
    /// The buffer must never be deallocated again or dropped.
    pub unsafe fn unmap(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipHostUnregister(self.data.as_ptr() as *mut _), Ok(())) }
    }

    pub fn copy_to_dev(
        &self,
        other: &Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > self.size())
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > other.size as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_to_dev_unchecked(other, src_off, dst_off, len) }
    }

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
                    other.ptr.add(dst_off as usize).cast(),
                    self.data.as_ptr().add(src_off as usize).cast(),
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
        if src_off.checked_add(len).is_none_or(|end| end > self.size())
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > other.size as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_from_dev_unchecked(other, src_off, dst_off, len) }
    }

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
                    self.data.as_ptr().add(src_off as usize) as *mut _,
                    other.ptr.add(dst_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }
}

/// Handle to CPU memory, mapped to all GPUs on the device
#[repr(C)]
#[derive(Debug)]
pub struct DevMapped {
    pub(crate) ptr: *mut u8,
    pub(crate) size: u32,
    pub(crate) event: HipEvent,
    pub(crate) stream: HipStream,
    _marker: PhantomData<UnsafeCell<()>>,
}

impl HipStream {
    /// Allocates `size` bytes and makes it accessible from all HIP devices
    pub fn alloc_host(&self, size: u32) -> Result<DevMapped, HipError> {
        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipHostMalloc((&raw mut ptr).cast(), size as usize, 0)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(DevMapped {
            ptr,
            size,
            stream: self.clone(),
            event: HipEvent::create()?,
            _marker: PhantomData,
        })
    }

    /// Allocates `size` bytes and makes it accessible from all HIP devices
    pub fn alloc_host_init(&self, data: &[u8]) -> Result<DevMapped, HipError> {
        let size = data.len();

        let buf = self.alloc_host(size as u32)?;

        unsafe {
            buf.ptr.copy_from_nonoverlapping(data.as_ptr(), size);
        }

        Ok(buf)
    }
}

impl DevMapped {
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: u32,
        stream: HipStream,
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

    pub fn borrowed(&self) -> RefMapped<'_> {
        unsafe { RefMapped::from_slice(self.as_slice()) }
    }

    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub const fn as_slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.ptr, self.size as usize) }
    }

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

impl Drop for DevMapped {
    fn drop(&mut self) {
        let _ = self.event.sync();

        unsafe {
            let _ = self.dealloc_unchecked();
        }
    }
}

/// Handle to an allocation on the GPU
#[repr(C)]
#[derive(Debug)]
pub struct Buffer {
    pub(crate) ptr: *mut u8,
    pub(crate) size: u32,
    pub(crate) event: HipEvent,
    pub(crate) stream: HipStream,
    _marker: PhantomData<UnsafeCell<()>>,
}

impl HipStream {
    /// Allocates `size` bytes on the default device
    pub fn alloc(&self, size: u32) -> Result<Buffer, HipError> {
        let event = HipEvent::create()?;

        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipMalloc((&raw mut ptr).cast(), size as usize)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(Buffer {
            ptr,
            size,
            stream: self.clone(),
            event,
            _marker: PhantomData,
        })
    }

    /// Allocates `size` zeroed bytes on the default device
    pub fn alloc_zeroed(&self, size: u32) -> Result<Buffer, HipError> {
        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipMalloc((&raw mut ptr).cast(), size as usize)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            try_err!(hipMemsetD8(ptr.cast(), 0, size as usize));
        }

        Ok(Buffer {
            ptr,
            size,
            stream: self.clone(),
            event: HipEvent::create()?,
            _marker: PhantomData,
        })
    }
}

impl Buffer {
    /// Constructs a GPU buffer from a raw pointer, size, stream, and event
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: u32,
        stream: HipStream,
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
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Gets the size of the allocation
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// Deallocates a buffer
    ///
    /// This allows you to safely handle errors which would normally panic when dropped.
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
    pub unsafe fn dealloc_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipFree(self.ptr.cast()), Ok(())) }
    }

    pub fn copy_to_host(
        &self,
        dst: &RefMapped,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > self.size as u32)
            || dst_off.checked_add(len).is_none_or(|end| end > dst.size())
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_to_host_unchecked(dst, src_off, dst_off, len) }
    }

    pub unsafe fn copy_to_host_unchecked(
        &self,
        dst: &RefMapped,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoH(
                    dst.data.as_ptr().add(dst_off as usize) as *mut _,
                    self.ptr.add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_from_host(
        &self,
        src: &RefMapped,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > src.size())
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > self.size as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_from_host_unchecked(src, src_off, dst_off, len) }
    }

    pub unsafe fn copy_from_host_unchecked(
        &self,
        src: &RefMapped,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    self.ptr.add(dst_off as usize).cast(),
                    src.data.as_ptr().add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_from(
        &self,
        src: &Self,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > src.size as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > self.size as u32)
            || self.ptr == src.ptr
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_from_unchecked(src, src_off, dst_off, len) }
    }

    pub fn copy_to(
        &self,
        dst: &Self,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > self.size as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.size as u32)
            || self.ptr == dst.ptr
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { dst.copy_from_unchecked(self, src_off, dst_off, len) }
    }

    pub unsafe fn copy_from_unchecked(
        &self,
        src: &Self,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoD(
                    self.ptr.add(dst_off as usize).cast(),
                    src.ptr.add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_from_async(
        &self,
        src: &Self,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if !self.stream.same_as(&src.stream) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.stream.copy_dtod(src, self, src_off, dst_off, len)?;
        }

        self.event.record(&self.stream)?;
        src.event.record(&self.stream)?;

        Ok(())
    }

    pub fn copy_to_async(
        &self,
        dst: &Self,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if !self.stream.same_as(&dst.stream) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.stream.copy_dtod(self, dst, src_off, dst_off, len)?;
        }

        self.event.record(&self.stream)?;
        dst.event.record(&self.stream)?;

        Ok(())
    }

    pub fn copy_from_host_async(
        &self,
        src: &DevMapped,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if !self.stream.same_as(&src.stream) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.stream
                .copy_htod(&src.borrowed(), self, src_off, dst_off, len)?;
        }

        self.event.record(&self.stream)?;

        src.event.record(&self.stream)?;

        Ok(())
    }

    pub fn copy_to_host_async(
        &self,
        dst: &DevMapped,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError> {
        if !self.stream.same_as(&dst.stream) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.stream
                .copy_dtoh(self, &dst.borrowed(), src_off, dst_off, len)?;
        }

        self.event.record(&self.stream)?;
        dst.event.record(&self.stream)?;

        Ok(())
    }

    pub fn memset(&self, byte: u8) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemsetD8(self.ptr.cast(), byte, self.size as usize,),
                Ok(())
            )
        }
    }

    pub fn memset_async(&self, byte: u8) -> Result<(), HipError> {
        unsafe {
            try_err!(hipMemsetAsync(
                self.ptr.cast(),
                byte as i32,
                self.size as usize,
                self.stream.raw.raw,
            ),)
        }

        self.event.record(&self.stream)?;

        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self.event.sync();

        unsafe {
            let _ = self.dealloc_unchecked();
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemPool {
    mem_pool: hipMemPool_t,
}
