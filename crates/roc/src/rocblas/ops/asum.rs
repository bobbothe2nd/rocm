use rocm_sys::rocblas::{
    rocblas_dasum, rocblas_dasum_64, rocblas_int, rocblas_sasum, rocblas_sasum_64,
};

use crate::{
    blas::{Asum, Asum_64},
    rocblas::RocblasHandle,
};

impl Asum_64 for RocblasHandle {}

unsafe impl Asum<f32, f32> for RocblasHandle {
    #[inline(always)]
    unsafe fn blas_asum(
        &self,
        n: u32,
        x: *const f32,
        incx: u32,
        result: *mut f32,
    ) -> Result<(), Self::Error> {
        unsafe {
            try_err!(
                rocblas_sasum(self.raw, n as rocblas_int, x, incx as rocblas_int, result),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn blas_asum_64(
        &self,
        n: u64,
        x: *const f32,
        incx: u64,
        result: *mut f32,
    ) -> Result<(), Self::Error>
    where
        Self: Asum_64,
    {
        unsafe {
            try_err!(
                rocblas_sasum_64(self.raw, n as i64, x, incx as i64, result),
                Ok(())
            )
        }
    }
}

unsafe impl Asum<f64, f64> for RocblasHandle {
    #[inline(always)]
    unsafe fn blas_asum(
        &self,
        n: u32,
        x: *const f64,
        incx: u32,
        result: *mut f64,
    ) -> Result<(), Self::Error> {
        unsafe {
            try_err!(
                rocblas_dasum(self.raw, n as rocblas_int, x, incx as rocblas_int, result),
                Ok(())
            )
        }
    }

    #[inline(always)]
    unsafe fn blas_asum_64(
        &self,
        n: u64,
        x: *const f64,
        incx: u64,
        result: *mut f64,
    ) -> Result<(), Self::Error>
    where
        Self: Asum_64,
    {
        unsafe {
            try_err!(
                rocblas_dasum_64(self.raw, n as i64, x, incx as i64, result),
                Ok(())
            )
        }
    }
}
