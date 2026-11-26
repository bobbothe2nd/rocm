#![allow(non_camel_case_types)]

use crate::aux::RocHandle;

pub unsafe trait Axpy<Alpha, X, Y>: RocHandle {
    unsafe fn blas_axpy(
        &self,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        y: *mut Y,
        incy: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_axpy_64(
        &self,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        y: *mut Y,
        incy: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Axpy_64;
}

pub trait Axpy_64 {}

pub unsafe trait Asum<X, Y>: RocHandle {
    unsafe fn blas_asum(
        &self,
        n: u32,
        x: *const X,
        incx: u32,
        result: *mut Y,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_asum_64(
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

pub unsafe trait Dot<X, Y, R>: RocHandle {
    unsafe fn blas_dot(
        &self,
        n: u32,
        x: *const X,
        incx: u32,
        y: *const Y,
        incy: u32,
        result: *mut R,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_dot_64(
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

pub unsafe trait Gemm<Alpha, A, B, Beta, C>: RocHandle {
    unsafe fn blas_gemm(
        &self,
        trans_a: MatOp,
        trans_b: MatOp,
        m: u32,
        n: u32,
        k: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        b: *const B,
        ldb: u32,
        beta: Beta,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_gemm_64(
        &self,
        trans_a: MatOp,
        trans_b: MatOp,
        m: u64,
        n: u64,
        k: u64,
        alpha: *const Alpha,
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

pub unsafe trait Gemv<Alpha, A, X, Beta, Y>: RocHandle {
    unsafe fn blas_gemv(
        &self,
        trans: MatOp,
        m: u32,
        n: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        x: *const X,
        incx: u32,
        beta: Beta,
        y: *mut Y,
        incy: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_gemv_64(
        &self,
        trans: MatOp,
        m: u64,
        n: u64,
        alpha: *const Alpha,
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

pub unsafe trait Geam<Alpha, A, Beta, B, C>: RocHandle {
    unsafe fn blas_geam(
        &self,
        trans_a: MatOp,
        trans_b: MatOp,
        m: u32,
        n: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        beta: Beta,
        b: *const B,
        ldb: u32,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_geam_64(
        &self,
        trans_a: MatOp,
        trans_b: MatOp,
        m: u64,
        n: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        beta: Beta,
        b: *const B,
        ldb: u64,
        c: *mut C,
        ldc: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Gemv_64;
}

pub trait Geam_64 {}

pub unsafe trait Ger<Alpha, X, Y, A>: RocHandle {
    unsafe fn blas_ger(
        &self,
        m: u32,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        y: *const Y,
        incy: u32,
        a: *mut A,
        lda: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_ger_64(
        &self,
        m: u64,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        y: *const Y,
        incy: u64,
        a: *mut A,
        lda: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Ger_64;
}

pub trait Ger_64 {}

pub unsafe trait Gemmt<Alpha, A, B, C>: RocHandle {
    unsafe fn blas_gemmt(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        trans_b: MatOp,
        n: u32,
        k: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        b: *const B,
        ldb: u32,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_gemmt_64(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        trans_b: MatOp,
        n: u64,
        k: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        b: *const B,
        ldb: u64,
        c: *mut C,
        ldc: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Gemmt_64;
}

pub trait Gemmt_64 {}

pub unsafe trait Nrm2<X, R>: RocHandle {
    unsafe fn blas_nrm2(
        &self,
        n: u32,
        x: *const X,
        incx: u32,
        result: *mut R,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_nrm2_64(
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

pub unsafe trait Scal<Alpha, X>: RocHandle {
    unsafe fn blas_scal(
        &self,
        n: u32,
        alpha: *const Alpha,
        x: *mut X,
        incx: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_scal_64(
        &self,
        n: u64,
        alpha: *const Alpha,
        x: *mut X,
        incx: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Scal_64;
}

pub trait Scal_64 {}

pub unsafe trait Spr<Alpha, X, AP>: RocHandle {
    unsafe fn blas_spr(
        &self,
        uplo: MatFill,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        ap: *mut AP,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_spr_64(
        &self,
        uplo: MatFill,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        ap: *mut AP,
    ) -> Result<(), Self::Error>
    where
        Self: Spr_64;
}

pub trait Spr_64 {}

pub unsafe trait Spr2<Alpha, X, Y, AP>: RocHandle {
    unsafe fn blas_spr2(
        &self,
        uplo: MatFill,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        y: *const Y,
        incy: u32,
        ap: *mut AP,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_spr2_64(
        &self,
        uplo: MatFill,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        y: *const Y,
        incy: u64,
        ap: *mut AP,
    ) -> Result<(), Self::Error>
    where
        Self: Spr_64;
}

pub trait Spr2_64 {}

pub unsafe trait Syr<Alpha, X, A>: RocHandle {
    unsafe fn blas_syr(
        &self,
        uplo: MatFill,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        a: *mut A,
        lda: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_syr_64(
        &self,
        uplo: MatFill,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        a: *mut A,
        lda: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Syr_64;
}

pub trait Syr_64 {}

pub unsafe trait Syr2<Alpha, X, Y, A>: RocHandle {
    unsafe fn blas_syr2(
        &self,
        uplo: MatFill,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        y: *const Y,
        incy: u32,
        a: *mut A,
        lda: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_syr2_64(
        &self,
        uplo: MatFill,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        y: *const Y,
        incy: u64,
        a: *mut A,
        lda: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Syr2_64;
}

pub trait Syr2_64 {}

pub unsafe trait Syrk<Alpha, A, Beta, C>: RocHandle {
    unsafe fn blas_syrk(
        &self,
        uplo: MatFill,
        trans: MatOp,
        n: u32,
        k: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        beta: *const Beta,
        c: *mut C,
        ldx: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_syrk_64(
        &self,
        uplo: MatFill,
        trans: MatOp,
        n: u64,
        k: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        c: *mut C,
        ldx: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Syrk_64;
}

pub trait Syrk_64 {}

pub unsafe trait Syrkx<Alpha, A, B, Beta, C>: RocHandle {
    unsafe fn blas_syrkx(
        &self,
        uplo: MatFill,
        trans: MatOp,
        n: u32,
        k: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        b: *const B,
        ldb: u32,
        beta: *const Beta,
        c: *mut C,
        ldx: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_syrkx_64(
        &self,
        uplo: MatFill,
        trans: MatOp,
        n: u64,
        k: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        b: *const B,
        ldb: u64,
        beta: *const Beta,
        c: *mut C,
        ldx: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Syrkx_64;
}

pub trait Syrkx_64 {}

pub unsafe trait Dgmm<A, X, C>: RocHandle {
    unsafe fn blas_dgmm(
        &self,
        side: MatSide,
        m: u32,
        n: u32,
        a: *const A,
        lda: u32,
        x: *const X,
        incx: u32,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_dgmm_64(
        &self,
        side: MatSide,
        m: u64,
        n: u64,
        a: *const A,
        lda: u64,
        x: *const X,
        incx: u64,
        c: *mut C,
        ldc: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Dgmm_64;
}

pub trait Dgmm_64 {}

pub unsafe trait Trmv<A, X>: RocHandle {
    unsafe fn blas_trmv(
        &self,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        n: u32,
        a: *const A,
        lda: u32,
        x: *mut X,
        incx: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_trmv_64(
        &self,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        n: u64,
        a: *const A,
        lda: u64,
        x: *mut X,
        incx: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Trmv_64;
}

pub trait Trmv_64 {}

pub unsafe trait Trsv<A, X>: RocHandle {
    unsafe fn blas_trsv(
        &self,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        n: u32,
        a: *const A,
        lda: u32,
        x: *mut X,
        incx: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_trsv_64(
        &self,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        n: u64,
        a: *const A,
        lda: u64,
        x: *mut X,
        incx: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Trsv_64;
}

pub trait Trsv_64 {}

pub unsafe trait Trmm<Alpha, A, B, C>: RocHandle {
    unsafe fn blas_trmm(
        &self,
        side: MatSide,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        m: u32,
        n: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        b: *const B,
        ldb: u32,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_trmm_64(
        &self,
        side: MatSide,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        m: u64,
        n: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        b: *const B,
        ldb: u64,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Trmm_64;
}

pub trait Trmm_64 {}

pub unsafe trait Trsm<Alpha, A, B>: RocHandle {
    unsafe fn blas_trsm(
        &self,
        side: MatSide,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        m: u32,
        n: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        b: *mut B,
        ldb: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_trsm_64(
        &self,
        side: MatSide,
        uplo: MatFill,
        trans: MatOp,
        diag: MatDiag,
        m: u64,
        n: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        b: *mut B,
        ldb: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Trsm_64;
}

pub trait Trsm_64 {}

pub unsafe trait Gbmv<Alpha, A, B, Beta, Y>: RocHandle {
    unsafe fn blas_gbmv(
        &self,
        trans: MatOp,
        m: u32,
        n: u32,
        kl: u32,
        ku: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        x: *const B,
        incx: u32,
        beta: *const Beta,
        y: *mut Y,
        incy: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_gbmv_64(
        &self,
        trans: MatOp,
        m: u64,
        n: u64,
        kl: u64,
        ku: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        x: *const B,
        incx: u64,
        beta: *const Beta,
        y: *mut Y,
        incy: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Gbmv_64;
}

pub trait Gbmv_64 {}

pub unsafe trait Rot<X, Y, C, S>: RocHandle {
    unsafe fn blas_rot(
        &self,
        n: u32,
        x: *mut X,
        incx: u32,
        y: *mut Y,
        incy: u32,
        c: *const C,
        s: *const S,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_rot_64(
        &self,
        n: u64,
        x: *mut X,
        incx: u64,
        y: *mut Y,
        incy: u64,
        c: *const C,
        s: *const S,
    ) -> Result<(), Self::Error>
    where
        Self: Rot_64;
}

pub trait Rot_64 {}

pub unsafe trait Rotg<A, B, C, S>: RocHandle {
    unsafe fn blas_rotg(
        &self,
        a: *mut A,
        b: *mut B,
        c: *mut C,
        s: *mut S,
    ) -> Result<(), Self::Error>;
}

pub unsafe trait Rotm<X, Y, Param>: RocHandle {
    unsafe fn blas_rotm(
        &self,
        n: u32,
        x: *mut X,
        incx: u32,
        y: *mut Y,
        incy: u32,
        param: *const Param,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_rotm_64(
        &self,
        n: u32,
        x: *mut X,
        incx: u32,
        y: *mut Y,
        incy: u32,
        param: *const Param,
    ) -> Result<(), Self::Error>
    where
        Self: Rotm_64;
}

pub trait Rotm_64 {}

pub unsafe trait Rotmg<D1, D2, X1, Y1, Param>: RocHandle {
    unsafe fn blas_rotmg(
        &self,
        d1: *mut D1,
        d2: *mut D2,
        x1: *const X1,
        y1: *const Y1,
        param: *mut Param,
    ) -> Result<(), Self::Error>;
}

pub unsafe trait Symm<Alpha, A, B, C>: RocHandle {
    unsafe fn blas_symm(
        &self,
        side: MatSide,
        uplo: MatFill,
        m: u32,
        n: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        b: *const B,
        ldb: u32,
        c: *mut C,
        ldx: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_symm_64(
        &self,
        side: MatSide,
        uplo: MatFill,
        m: u64,
        n: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        b: *const B,
        ldb: u64,
        c: *mut C,
        ldx: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Symm_64;
}

pub trait Symm_64 {}

pub unsafe trait Symv<Alpha, A, X, Beta, Y>: RocHandle {
    unsafe fn blas_symv(
        &self,
        uplo: MatFill,
        n: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        x: *const X,
        incx: u32,
        beta: *const Beta,
        y: *mut Y,
        incy: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_symv_64(
        &self,
        uplo: MatFill,
        n: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        x: *const X,
        incx: u64,
        beta: *const Beta,
        y: *mut Y,
        incy: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Symv_64;
}

pub trait Symv_64 {}

pub unsafe trait Her<Alpha, X, A>: RocHandle {
    unsafe fn blas_her(
        &self,
        uplo: MatFill,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        a: *mut A,
        lda: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_her_64(
        &self,
        uplo: MatFill,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        a: *mut A,
        lda: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Her_64;
}

pub trait Her_64 {}

pub unsafe trait Her2<Alpha, X, Y, A>: RocHandle {
    unsafe fn blas_her2(
        &self,
        uplo: MatFill,
        n: u32,
        alpha: *const Alpha,
        x: *const X,
        incx: u32,
        y: *const Y,
        incy: u32,
        a: *mut A,
        lda: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_her2_64(
        &self,
        uplo: MatFill,
        n: u64,
        alpha: *const Alpha,
        x: *const X,
        incx: u64,
        y: *const Y,
        incy: u64,
        a: *mut A,
        lda: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Her2_64;
}

pub trait Her2_64 {}

pub unsafe trait Herk<Alpha, A, Beta, C>: RocHandle {
    unsafe fn blas_herk(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        n: u32,
        k: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        beta: *const Beta,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_herk_64(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        n: u64,
        k: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        beta: *const Beta,
        c: *mut C,
        ldc: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Herk_64;
}

pub trait Herk_64 {}

pub unsafe trait Her2k<Alpha, A, B, Beta, C>: RocHandle {
    unsafe fn blas_her2k(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        n: u32,
        k: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        b: *const B,
        ldb: u32,
        beta: *const Beta,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_her2k_64(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        n: u64,
        k: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        b: *const B,
        ldb: u64,
        beta: *const Beta,
        c: *mut C,
        ldc: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Her2k_64;
}

pub trait Her2k_64 {}

pub unsafe trait Herkx<Alpha, A, Beta, C>: RocHandle {
    unsafe fn blas_herkx(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        n: u32,
        k: u32,
        alpha: *const Alpha,
        a: *const A,
        lda: u32,
        beta: *const Beta,
        c: *mut C,
        ldc: u32,
    ) -> Result<(), Self::Error>;
    unsafe fn blas_herkx_64(
        &self,
        uplo: MatFill,
        trans_a: MatOp,
        n: u64,
        k: u64,
        alpha: *const Alpha,
        a: *const A,
        lda: u64,
        beta: *const Beta,
        c: *mut C,
        ldc: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Herkx_64;
}

pub trait Herkx_64 {}

#[repr(i32)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MatOp {
    /// Operate with the matrix directly
    None = 111,

    /// Operate with the transpose of the matrix
    Transpose = 112,

    /// Operate with the transpose of the conjugate of the matrix (for complex)
    ConjugateTranspose = 113,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MatFill {
    /// Upper triangle.
    Upper = 121,

    /// Lower triangle.
    Lower = 122,

    Full = 123,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MatSide {
    /// Multiply general matrix by symmetric,
    /// Hermitian, or triangular matrix on the left.
    Left = 141,
    /// Multiply general matrix by symmetric,
    /// Hermitian, or triangular matrix on the right.
    Right = 142,
    Both = 143,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MatDiag {
    /// Non-unit triangular.
    NonUnit,
    /// Unit triangular.
    Unit,
}
