use rocm_sys::hip::{hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoD, hipMemcpyHtoDAsync};

use crate::hip::{HipError, memory::{Buffer, HostSlice, HostSliceMut}, stream::HipStream};

pub trait CopyTo<Dst> {
    unsafe fn copy(
        src: &Self,
        dst: &mut Dst,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>;

    unsafe fn copy_async(
        src: &Self,
        dst: &mut Dst,
        src_off: u32,
        dst_off: u32,
        len: u32,
        stream: HipStream,
    ) -> Result<(), HipError>;
}

impl<'a, T: AsRef<HostSlice<'a>>> CopyTo<Buffer> for T {
    unsafe fn copy(
        src: &Self,
        dst: &mut Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    {
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

    unsafe fn copy_async(
        src: &Self,
        dst: &mut Buffer,
        src_off: u32,
        dst_off: u32,
        len: u32,
        stream: HipStream,
    ) -> Result<(), HipError>
    {
        unsafe {
            try_err!(
                hipMemcpyHtoDAsync(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.as_ref().ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                    stream.raw,
                ),
                Ok(())
            )
        }
    }
}

impl<'a, T: AsMut<HostSliceMut<'a>>> CopyTo<T> for Buffer {
    unsafe fn copy(
        src: &Self,
        dst: &mut T,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    {
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

    unsafe fn copy_async(
        src: &Self,
        dst: &mut T,
        src_off: u32,
        dst_off: u32,
        len: u32,
        stream: HipStream,
    ) -> Result<(), HipError>
    {
        unsafe {
            try_err!(
                hipMemcpyDtoHAsync(
                    dst.as_mut().ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                    stream.raw,
                ),
                Ok(())
            )
        }
    }
}

impl CopyTo<Self> for Buffer {
    unsafe fn copy(
        src: &Self,
        dst: &mut Self,
        src_off: u32,
        dst_off: u32,
        len: u32,
    ) -> Result<(), HipError>
    {
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

    unsafe fn copy_async(
        src: &Self,
        dst: &mut Self,
        src_off: u32,
        dst_off: u32,
        len: u32,
        stream: HipStream,
    ) -> Result<(), HipError>
    {
        unsafe {
            try_err!(
                hipMemcpyDtoHAsync(
                    dst.ptr.byte_add(dst_off as usize).cast(),
                    src.ptr.byte_add(src_off as usize).cast(),
                    len as usize,
                    stream.raw,
                ),
                Ok(())
            )
        }
    }
}
