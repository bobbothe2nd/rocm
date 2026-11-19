use crate::hip::{
    HipError,
    memory::{
        DevPtr, EventSync, HipBuffer, HipSlice, HipSliceMut, HostBuffer, HostSlice, HostSliceMut,
        StreamBound,
    },
    stream::HipStream,
};

#[inline]
pub fn copy<T, Src, Dst>(
    src: &Src,
    dst: &mut Dst,
    src_off: usize,
    dst_off: usize,
    len: usize,
) -> Result<(), HipError>
where
    Src: CopyTo<T, Dst>,
    Dst: DevPtr<T>,
{
    if src.get_dev() == dst.get_dev()
        || src_off
        .checked_add(len)
        .is_none_or(|end| end > src.len() as usize)
        || dst_off
            .checked_add(len)
            .is_none_or(|end| end > dst.len() as usize)
        || core::ptr::eq(src.as_ptr(), dst.as_ptr())
    {
        return Err(HipError::InvalidValue);
    }

    unsafe { Src::copy(src, dst, src_off, dst_off, len) }
}

#[inline]
pub fn copy_async<T, Src, Dst>(
    src: &Src,
    dst: &mut Dst,
    src_off: usize,
    dst_off: usize,
    len: usize,
) -> Result<(), HipError>
where
    Src: CopyTo<T, Dst> + StreamBound,
    Dst: DevPtr<T> + StreamBound + EventSync,
{
    if src.get_dev() == dst.get_dev()
        || src_off
            .checked_add(len)
            .is_none_or(|end| end > src.len() as usize)
        || dst_off
            .checked_add(len)
            .is_none_or(|end| end > dst.len() as usize)
        || core::ptr::eq(src.as_ptr(), dst.as_ptr())
    {
        return Err(HipError::InvalidValue);
    }

    let stream = src.stream();

    if !dst.stream().same_as(stream) {
        dst.sync()?;
    }

    unsafe {
        Src::copy_async(src, dst, src_off, dst_off, len, stream)?;
    }

    Ok(())
}

pub unsafe trait CopyTo<T, Dst: DevPtr<T>>: DevPtr<T> {
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
        unsafe impl<T: crate::hip::DevRepr> CopyTo<T, $dst> for $src {
            #[inline(always)]
            unsafe fn copy(
                src: &Self,
                dst: &mut $dst,
                src_off: usize,
                dst_off: usize,
                len: usize,
            ) -> Result<(), HipError> {
                src.sync()?;
                dst.sync()?;

                unsafe {
                    try_err!(
                        rocm_sys::hip::$sync(
                            dst.ptr.add(dst_off as usize) as *mut _,
                            src.ptr.add(src_off as usize) as *mut _,
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
                            dst.ptr.add(dst_off as usize) as *mut _,
                            src.ptr.add(src_off as usize) as *mut _,
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
    HipBuffer<T> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > Self,
    HipBuffer<T> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipSliceMut<'_, T>,
    HipSlice<'_, T> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipBuffer<T>,
    HipSlice<'_, T> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipSliceMut<'_, T>,
    HipSliceMut<'_, T> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > HipBuffer<T>,
    HipSliceMut<'_, T> > (hipMemcpyDtoD | hipMemcpyDtoDAsync) > Self,

    HipBuffer<T> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostBuffer<T>,
    HipBuffer<T> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostSliceMut<'_, T>,
    HipSlice<'_, T> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostBuffer<T>,
    HipSlice<'_, T> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostSliceMut<'_, T>,
    HipSliceMut<'_, T> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostBuffer<T>,
    HipSliceMut<'_, T> > (hipMemcpyDtoH | hipMemcpyDtoHAsync) > HostSliceMut<'_, T>,

    HostBuffer<T> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipBuffer<T>,
    HostBuffer<T> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipSliceMut<'_, T>,
    HostSlice<'_, T> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipBuffer<T>,
    HostSlice<'_, T> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipSliceMut<'_, T>,
    HostSliceMut<'_, T> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipBuffer<T>,
    HostSliceMut<'_, T> > (hipMemcpyHtoD | hipMemcpyHtoDAsync) > HipSliceMut<'_, T>,
);
