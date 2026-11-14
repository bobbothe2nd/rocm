use crate::hip::{
    HipError, memory::{DevPtr, EventSync, HipBuffer, HipSlice, HipSliceMut, HostBuffer, HostSlice, HostSliceMut, StreamBound}, stream::HipStream,
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

macro_rules! impl_copy {
    ($($src:ty > ($sync:ident | $async:ident) > $dst:ty),*$(,)?) => {$(
        unsafe impl CopyTo<$dst> for $src {
            #[inline(always)]
            unsafe fn copy(
                src: &Self,
                dst: &mut $dst,
                src_off: usize,
                dst_off: usize,
                len: usize,
            ) -> Result<(), HipError> {
                unsafe {
                    try_err!(
                        rocm_sys::hip::$sync(
                            dst.ptr.byte_add(dst_off as usize) as *mut _,
                            src.ptr.byte_add(src_off as usize) as *mut _,
                            len as usize,
                        ),
                        Ok(())
                    )
                }
            }

            #[inline(always)]
            unsafe fn copy_async(
                src: &Self,
                dst: &mut $dst,
                src_off: usize,
                dst_off: usize,
                len: usize,
                stream: &HipStream,
            ) -> Result<(), HipError> {
                unsafe {
                    try_err!(
                        rocm_sys::hip::$async(
                            dst.ptr.byte_add(dst_off as usize) as *mut _,
                            src.ptr.byte_add(src_off as usize) as *mut _,
                            len as usize,
                            stream.raw,
                        )
                    );
                }
                src.latesync()?;
                dst.latesync()
            }
        }
    )*};
}

impl_copy!(
    HipBuffer > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > Self,
    HipBuffer > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipSliceMut<'_>,
    HipSlice<'_> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipBuffer,
    HipSlice<'_> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipSliceMut<'_>,
    HipSliceMut<'_> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipBuffer,
    HipSliceMut<'_> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > Self,

    HipBuffer > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostBuffer,
    HipBuffer > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostSliceMut<'_>,
    HipSlice<'_> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostBuffer,
    HipSlice<'_> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostSliceMut<'_>,
    HipSliceMut<'_> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostBuffer,
    HipSliceMut<'_> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostSliceMut<'_>,

    HostBuffer > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipBuffer,
    HostBuffer > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipSliceMut<'_>,
    HostSlice<'_> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipBuffer,
    HostSlice<'_> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipSliceMut<'_>,
    HostSliceMut<'_> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipBuffer,
    HostSliceMut<'_> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipSliceMut<'_>,
);
