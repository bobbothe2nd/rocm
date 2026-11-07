use crate::hip::{
    HipError, event::HipEvent, memory::{HostSlice, HostSliceMut, Memset}, stream::HipStream,
};
use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, ptr};
use rocm_sys::hip::{
    hipFree, hipMalloc, hipMemcpyDtoD, hipMemcpyDtoH, hipMemcpyHtoD, hipMemsetD8,
};

/// Handle to an allocation on the GPU
#[repr(C)]
#[derive(Debug)]
pub struct Buffer {
    pub(crate) ptr: *mut [u8],
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

        let slice = core::ptr::slice_from_raw_parts_mut(ptr, size as usize);
        Ok(Buffer {
            ptr: slice,
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

        let slice = core::ptr::slice_from_raw_parts_mut(ptr, size as usize);
        Ok(Buffer {
            ptr: slice,
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
        let slice = core::ptr::slice_from_raw_parts_mut(ptr, size as usize);
        Self {
            ptr: slice,
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
    pub const fn size(&self) -> u32 {
        self.ptr.len() as u32
    }

    /// Deallocates a buffer
    ///
    /// This allows you to safely handle errors which would normally panic when dropped.
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

    pub fn copy_to_host<'a, S>(
        &self,
        dst: &mut S,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    where
        S: AsMut<HostSliceMut<'a>>,
    {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > self.ptr.len() as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.as_mut().size())
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_to_host_unchecked(dst, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_to_host_unchecked<'a, S>(
        &self,
        dst: &mut S,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    where
        S: AsMut<HostSliceMut<'a>>,
    {
        unsafe {
            try_err!(
                hipMemcpyDtoH(
                    dst.as_mut().data.byte_add(dst_off as usize) as *mut _,
                    self.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    pub fn copy_from_host<'a, S>(
        &self,
        src: &S,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    where
        S: AsRef<HostSlice<'a>>,
    {
        if src_off
            .checked_add(len)
            .is_none_or(|end| end > src.as_ref().size())
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > self.ptr.len() as u32)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.copy_from_host_unchecked(src, src_off, dst_off, len) }
    }

    #[inline(always)]
    pub unsafe fn copy_from_host_unchecked<'a, S>(
        &self,
        src: &S,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    where
        S: AsRef<HostSlice<'a>>,
    {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    self.ptr.byte_add(dst_off as usize).cast(),
                    src.as_ref().data.byte_add(src_off as usize).cast(),
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
            .is_none_or(|end| end > src.ptr.len() as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > self.ptr.len() as u32)
            || ptr::eq(self.ptr, src.ptr)
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
            .is_none_or(|end| end > self.ptr.len() as u32)
            || dst_off
                .checked_add(len)
                .is_none_or(|end| end > dst.ptr.len() as u32)
            || ptr::eq(self.ptr, dst.ptr)
        {
            return Err(HipError::InvalidValue);
        }

        unsafe { dst.copy_from_unchecked(self, src_off, dst_off, len) }
    }

    #[inline(always)]
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
                    self.ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
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

    pub fn copy_from_host_async<'a, S>(
        &self,
        src: &S,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    where
        S: AsRef<HostSlice<'a>>,
    {
        let src = src.as_ref();

        if !self.stream.same_as(&src.stream) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.stream.copy_htod(&src, self, src_off, dst_off, len)?;
        }

        self.event.record(&self.stream)?;
        src.event.record(&self.stream)?;

        Ok(())
    }

    pub fn copy_to_host_async<'a, S>(
        &self,
        dst: &mut S,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    where
        S: AsMut<HostSliceMut<'a>>,
    {
        let dst = dst.as_mut();

        if !self.stream.same_as(&dst.stream) {
            return Err(HipError::InvalidValue);
        }

        unsafe {
            self.stream.copy_dtoh(self, dst, src_off, dst_off, len)?;
        }

        self.event.record(&self.stream)?;
        dst.event.record(&self.stream)?;

        Ok(())
    }

    pub fn memset<T: Memset>(&self, val: T) -> Result<(), HipError> {
        if !self.ptr.len().is_multiple_of(size_of::<T>()) {
            return Err(HipError::InvalidValue);
        }

        unsafe { self.memset_unchecked(val) }
    }

    #[inline(always)]
    pub unsafe fn memset_unchecked<T: Memset>(&self, val: T) -> Result<(), HipError> {
        unsafe { T::memset_sync(self.ptr.cast(), val, self.ptr.len()) }
    }

    pub fn memset_async<T: Memset>(&self, val: T) -> Result<(), HipError> {
        if !self.ptr.len().is_multiple_of(size_of::<T>()) {
            return Err(HipError::InvalidValue);
        }

        unsafe { T::memset_async(self.ptr.cast(), val, self.ptr.len(), &self.stream)?; }

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
