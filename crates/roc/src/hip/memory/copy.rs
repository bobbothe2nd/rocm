use rocm_sys::hip::{
    hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoD,
    hipMemcpyHtoDAsync,
};

use crate::hip::{
    HipError,
    device::Device,
    memory::{EventSync, HipBuffer, HostBuffer, HostSlice, HostSliceMut},
    stream::HipStream,
};

#[inline]
pub fn copy<Src, Dst>(
    src: &Src,
    dst: &mut Dst,
    src_off: usize,
    dst_off: usize,
    len: usize,
) -> Result<(), HipError>
where
    Src: CopyTo<Dst>,
    Dst: DevPtr,
{
    if src_off
        .checked_add(len)
        .is_none_or(|end| end > src.size() as usize)
        || dst_off
            .checked_add(len)
            .is_none_or(|end| end > dst.size() as usize)
        || core::ptr::eq(src.as_ptr(), dst.as_ptr())
    {
        return Err(HipError::InvalidValue);
    }

    unsafe { Src::copy(src, dst, src_off, dst_off, len) }
}

#[inline]
pub fn copy_async<Src, Dst>(
    src: &Src,
    dst: &mut Dst,
    src_off: usize,
    dst_off: usize,
    len: usize,
) -> Result<(), HipError>
where
    Src: CopyTo<Dst> + StreamBound,
    Dst: DevPtr + StreamBound,
{
    let stream = src.stream();

    if !stream.same_as(dst.stream())
        || src_off
            .checked_add(len)
            .is_none_or(|end| end > src.size() as usize)
        || dst_off
            .checked_add(len)
            .is_none_or(|end| end > dst.size() as usize)
        || core::ptr::eq(src.as_ptr(), dst.as_ptr())
    {
        return Err(HipError::InvalidValue);
    }

    unsafe {
        Src::copy_async(src, dst, src_off, dst_off, len, stream)?;
    }

    Ok(())
}

pub trait StreamBound {
    fn stream(&self) -> &HipStream;
}

impl StreamBound for HipBuffer {
    #[inline(always)]
    fn stream(&self) -> &HipStream {
        &self.stream
    }
}

impl StreamBound for HostBuffer {
    #[inline(always)]
    fn stream(&self) -> &HipStream {
        &self.stream
    }
}

impl StreamBound for HostSlice<'_> {
    #[inline(always)]
    fn stream(&self) -> &HipStream {
        &self.stream
    }
}

impl StreamBound for HostSliceMut<'_> {
    #[inline(always)]
    fn stream(&self) -> &HipStream {
        &self.stream
    }
}

pub trait DevPtr {
    fn as_ptr(&self) -> *const u8;
    fn size(&self) -> usize;
    fn dev(&self) -> Device;
}

impl DevPtr for HipBuffer {
    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self.ptr.cast()
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    fn dev(&self) -> Device {
        self.stream
            .get_dev()
            .expect("valid Stream should have device")
    }
}

impl DevPtr for HostBuffer {
    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn dev(&self) -> Device {
        self.stream
            .get_dev()
            .expect("valid Stream should have device")
    }
}

impl DevPtr for HostSlice<'_> {
    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn dev(&self) -> Device {
        self.stream
            .get_dev()
            .expect("valid Stream should have device")
    }
}

impl DevPtr for HostSliceMut<'_> {
    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn dev(&self) -> Device {
        self.stream
            .get_dev()
            .expect("valid Stream should have device")
    }
}

pub unsafe trait CopyTo<Dst: DevPtr>: DevPtr {
    unsafe fn copy(
        src: &Self,
        dst: &mut Dst,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError>;

    unsafe fn copy_async(
        src: &Self,
        dst: &mut Dst,
        src_off: usize,
        dst_off: usize,
        len: usize,
        stream: &HipStream,
    ) -> Result<(), HipError>
    where
        Self: StreamBound,
        Dst: StreamBound;
}

unsafe impl CopyTo<HipBuffer> for HostSlice<'_> {
    #[inline(always)]
    unsafe fn copy(
        src: &Self,
        dst: &mut HipBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn copy_async(
        src: &Self,
        dst: &mut HipBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(hipMemcpyHtoDAsync(
                dst.ptr.byte_add(dst_off as usize).cast(),
                src.ptr.byte_add(src_off as usize).cast(),
                len as usize,
                stream.raw,
            ),)
        }

        dst.latesync()?;
        src.latesync()?;

        Ok(())
    }
}

unsafe impl CopyTo<HipBuffer> for HostSliceMut<'_> {
    #[inline(always)]
    unsafe fn copy(
        src: &Self,
        dst: &mut HipBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.as_ref().ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn copy_async(
        src: &Self,
        dst: &mut HipBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(hipMemcpyHtoDAsync(
                dst.ptr.byte_add(dst_off as usize).cast(),
                src.as_ref().ptr.byte_add(src_off as usize).cast(),
                len as usize,
                stream.raw,
            ),)
        }

        dst.latesync()?;
        src.latesync()?;

        Ok(())
    }
}

unsafe impl CopyTo<HipBuffer> for HostBuffer {
    #[inline(always)]
    unsafe fn copy(
        src: &Self,
        dst: &mut HipBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyHtoD(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.as_ref().ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn copy_async(
        src: &Self,
        dst: &mut HipBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(hipMemcpyHtoDAsync(
                dst.ptr.byte_add(dst_off as usize).cast(),
                src.as_ref().ptr.byte_add(src_off as usize).cast(),
                len as usize,
                stream.raw,
            ),)
        }

        dst.latesync()?;
        src.latesync()?;

        Ok(())
    }
}

unsafe impl CopyTo<HostSliceMut<'_>> for HipBuffer {
    #[inline(always)]
    unsafe fn copy(
        src: &Self,
        dst: &mut HostSliceMut,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoH(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn copy_async(
        src: &Self,
        dst: &mut HostSliceMut,
        src_off: usize,
        dst_off: usize,
        len: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(hipMemcpyDtoHAsync(
                dst.ptr.byte_add(dst_off as usize).cast(),
                src.ptr.byte_add(src_off as usize).cast(),
                len as usize,
                stream.raw,
            ),)
        }

        dst.latesync()?;
        src.latesync()?;

        Ok(())
    }
}

unsafe impl CopyTo<HostBuffer> for HipBuffer {
    #[inline(always)]
    unsafe fn copy(
        src: &Self,
        dst: &mut HostBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoH(
                    dst.as_mut().ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn copy_async(
        src: &Self,
        dst: &mut HostBuffer,
        src_off: usize,
        dst_off: usize,
        len: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(hipMemcpyDtoHAsync(
                dst.as_mut().ptr.byte_add(dst_off as usize).cast(),
                src.ptr.byte_add(src_off as usize).cast(),
                len as usize,
                stream.raw,
            ),)
        }

        dst.latesync()?;
        src.latesync()?;

        Ok(())
    }
}

unsafe impl CopyTo<Self> for HipBuffer {
    #[inline(always)]
    unsafe fn copy(
        src: &Self,
        dst: &mut Self,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipMemcpyDtoD(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                ),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn copy_async(
        src: &Self,
        dst: &mut Self,
        src_off: usize,
        dst_off: usize,
        len: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        unsafe {
            try_err!(hipMemcpyDtoDAsync(
                dst.ptr.byte_add(dst_off as usize).cast(),
                src.ptr.byte_add(src_off as usize).cast(),
                len as usize,
                stream.raw,
            ),)
        }

        dst.latesync()?;
        src.latesync()?;

        Ok(())
    }
}
