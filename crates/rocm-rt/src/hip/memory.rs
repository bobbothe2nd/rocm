use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, slice::from_raw_parts};

use rocm_sys::hip::{
    hipFree, hipHostFree, hipHostMalloc, hipMalloc, hipMemPool_t, hipMemcpyDtoD, hipMemcpyDtoH,
    hipMemcpyHtoD,
};

use crate::hip::HipError;

/// Handle to CPU memory, mapped to all GPUs on the device
#[derive(Debug)]
pub struct DevMapped {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    _marker: PhantomData<UnsafeCell<()>>,
}

impl DevMapped {
    pub fn new(data: &[u8]) -> Result<Self, HipError> {
        let size = data.len();

        let buf = Self::alloc(size)?;

        unsafe {
            buf.ptr.copy_from_nonoverlapping(data.as_ptr(), size);
        }

        Ok(buf)
    }

    /// Allocates `size` bytes and makes it accessible from all HIP device
    pub fn alloc(size: usize) -> Result<Self, HipError> {
        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipHostMalloc((&raw mut ptr).cast(), size, 0)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(Self {
            ptr,
            size,
            _marker: PhantomData,
        })
    }

    pub const unsafe fn from_raw_parts(ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            _marker: PhantomData,
        }
    }

    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub const fn as_slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.ptr, self.size) }
    }

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
    /// The buffer must never be deallocated again or dropped.
    pub unsafe fn dealloc_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipHostFree(self.ptr.cast()), Ok(())) }
    }

    pub fn copy_to_dev(
        &self,
        other: &Buffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > self.size)
            || dst_off.checked_add(len).is_none_or(|end| end > other.size)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.copy_to_dev_unchecked(other, src_off, dst_off, len)
        }
    }

    pub unsafe fn copy_to_dev_unchecked(
        &self,
        other: &Buffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    other.ptr.add(dst_off).cast(),
                    self.ptr.add(src_off).cast(),
                    len
                ),
                Ok(())
            )
        }
    }
}

impl Drop for DevMapped {
    fn drop(&mut self) {
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
    pub(crate) size: usize,
    _marker: PhantomData<UnsafeCell<()>>,
}

impl Buffer {
    /// Allocates `size` bytes on the default device
    pub fn new(size: usize) -> Result<Self, HipError> {
        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipMalloc((&raw mut ptr).cast(), size)) };
        let ptr = ptr?;

        if ptr.is_null() {
            return Err(HipError::InvalidValue);
        }

        Ok(Self {
            ptr,
            size,
            _marker: PhantomData,
        })
    }

    /// Constructs a GPU buffer from a raw pointer and size
    pub const unsafe fn from_raw_parts(ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            _marker: PhantomData,
        }
    }

    /// Gets the pointer to GPU memory
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Gets the size of the allocation
    pub const fn size(&self) -> usize {
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
        dst: &DevMapped,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > self.size)
            || dst_off.checked_add(len).is_none_or(|end| end > dst.size)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.copy_to_host_unchecked(dst, src_off, dst_off, len)
        }
    }
    
    pub unsafe fn copy_to_host_unchecked(
        &self,
        dst: &DevMapped,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoH(
                    dst.ptr.add(dst_off).cast(),
                    self.ptr.add(src_off).cast(),
                    len
                ),
                Ok(())
            )
        }
    }

    pub fn copy_from(
        &self,
        src: &Self,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        if src_off.checked_add(len).is_none_or(|end| end > self.size)
            || dst_off.checked_add(len).is_none_or(|end| end > src.size)
            || self.ptr == src.ptr
        {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.copy_from_unchecked(src, src_off, dst_off, len)
        }
    }

    pub unsafe fn copy_from_unchecked(
        &self,
        src: &Self,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoD(
                    self.ptr.add(dst_off).cast(),
                    src.ptr.add(src_off).cast(),
                    len
                ),
                Ok(())
            )
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
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
