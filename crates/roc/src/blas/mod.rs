#![allow(non_camel_case_types)]

use crate::RocHandle;

pub trait Axpy<Alpha, X, Y>: RocHandle {
    unsafe fn __blas_axpy(
        &self,
        n: u32,
        alpha: Alpha,
        x: *const X,
        incx: u32,
        y: *mut Y,
        incy: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn __blas_axpy_64(
        &self,
        n: u64,
        alpha: Alpha,
        x: *const X,
        incx: u64,
        y: *mut Y,
        incy: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Axpy_64;
}

pub trait Axpy_64 {}

pub trait Asum<X, Y>: RocHandle {
    unsafe fn __blas_asum(
        &self,
        n: u32,
        x: *const X,
        incx: u32,
        result: *mut Y,
    ) -> Result<(), Self::Error>;
    unsafe fn __blas_asum_64(
        &self,
        n: u64,
        x: *const X,
        incx: u64,
        result: *mut Y,
    ) -> Result<(), Self::Error>
    where
        Self: Asum_64;
}

pub trait Asum_64 {}

pub trait Dot<X, Y, R>: RocHandle {
    unsafe fn __blas_dot(
        &self,
        n: u32,
        x: *const X,
        incx: u32,
        y: *const Y,
        incy: u32,
        result: *mut R,
    ) -> Result<(), Self::Error>;
    unsafe fn __blas_dot_64(
        &self,
        n: u64,
        x: *const X,
        incx: u64,
        y: *const Y,
        incy: u64,
        result: *mut R,
    ) -> Result<(), Self::Error>
    where
        Self: Dot_64;
}

pub trait Dot_64 {}

pub trait Gemm<Alpha, A, B, Beta, C>: RocHandle {
    unsafe fn __blas_gemm(
        &self,
        trans_a: MatOp,
        trans_b: MatOp,
        m: u32,
        n: u32,
        k: u32,
        alpha: Alpha,
        a: *const A,
        lda: u32,
        b: *const B,
        ldb: u32,
        beta: Beta,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn __blas_gemm_64(
        &self,
        trans_a: MatOp,
        trans_b: MatOp,
        m: u64,
        n: u64,
        k: u64,
        alpha: Alpha,
        a: *const A,
        lda: u64,
        b: *const B,
        ldb: u64,
        beta: Beta,
        c: *mut C,
        ldc: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Gemm_64;
}

pub trait Gemm_64 {}

pub trait Gemv<Alpha, A, X, Beta, Y>: RocHandle {
    unsafe fn __blas_gemv(
        &self,
        trans: MatOp,
        m: u32,
        n: u32,
        alpha: Alpha,
        a: *const A,
        lda: u32,
        x: *const X,
        incx: u32,
        beta: Beta,
        y: *mut Y,
        incy: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn __blas_gemv_64(
        &self,
        trans: MatOp,
        m: u64,
        n: u64,
        alpha: Alpha,
        a: *const A,
        lda: u64,
        x: *const X,
        incx: u64,
        beta: Beta,
        y: *mut Y,
        incy: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Gemv_64;
}

pub trait Gemv_64 {}

pub trait Nrm2<X, R>: RocHandle {
    unsafe fn __blas_nrm2(
        &self,
        n: u32,
        x: *const X,
        incx: u32,
        result: *mut R,
    ) -> Result<(), Self::Error>;
    unsafe fn __blas_nrm2_64(
        &self,
        n: u64,
        x: *const X,
        incx: u64,
        result: *mut R,
    ) -> Result<(), Self::Error>
    where
        Self: Gemv_64;
}

pub trait Nrm2_64 {}

pub trait Scal<Alpha, X>: RocHandle {
    unsafe fn __blas_scal(
        &self,
        n: u32,
        alpha: Alpha,
        x: *mut X,
        incx: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn __blas_scal_64(
        &self,
        n: u64,
        alpha: Alpha,
        x: *mut X,
        incx: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Scal_64;
}

pub trait Scal_64 {}

pub enum MatOp {
    /// Operate with the matrix directly
    None = 111,

    /// Operate with the transpose of the matrix
    Transpose = 112,

    /// Operate with the transpose of the conjugate of the matrix (for complex)
    ConjugateTranspose = 113,
}
