impl hipblasStatus_t {
    /// Function succeeds
    pub const HIPBLAS_STATUS_SUCCESS: hipblasStatus_t = hipblasStatus_t(0);
    /// HIPBLAS library not initialized
    pub const HIPBLAS_STATUS_NOT_INITIALIZED: hipblasStatus_t = hipblasStatus_t(1);
    /// resource allocation failed
    pub const HIPBLAS_STATUS_ALLOC_FAILED: hipblasStatus_t = hipblasStatus_t(2);
    /// unsupported numerical value was passed to function
    pub const HIPBLAS_STATUS_INVALID_VALUE: hipblasStatus_t = hipblasStatus_t(3);
    /// access to GPU memory space failed
    pub const HIPBLAS_STATUS_MAPPING_ERROR: hipblasStatus_t = hipblasStatus_t(4);
    /// GPU program failed to execute
    pub const HIPBLAS_STATUS_EXECUTION_FAILED: hipblasStatus_t = hipblasStatus_t(5);
    /// an internal HIPBLAS operation failed
    pub const HIPBLAS_STATUS_INTERNAL_ERROR: hipblasStatus_t = hipblasStatus_t(6);
    /// function not implemented
    pub const HIPBLAS_STATUS_NOT_SUPPORTED: hipblasStatus_t = hipblasStatus_t(7);
    /// architecture mismatch
    pub const HIPBLAS_STATUS_ARCH_MISMATCH: hipblasStatus_t = hipblasStatus_t(8);
    /// hipBLAS handle is null pointer
    pub const HIPBLAS_STATUS_HANDLE_IS_NULLPTR: hipblasStatus_t = hipblasStatus_t(9);
    ///  unsupported enum value was passed to function
    pub const HIPBLAS_STATUS_INVALID_ENUM: hipblasStatus_t = hipblasStatus_t(10);
    ///  back-end returned an unsupported status code
    pub const HIPBLAS_STATUS_UNKNOWN: hipblasStatus_t = hipblasStatus_t(11);
}
#[repr(transparent)]
/// \\brief hipblas status codes definition
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasStatus_t(::core::ffi::c_int);
impl hipblasOperation_t {
    ///  Operate with the matrix.
    pub const HIPBLAS_OP_N: hipblasOperation_t = hipblasOperation_t(111);
    ///  Operate with the transpose of the matrix.
    pub const HIPBLAS_OP_T: hipblasOperation_t = hipblasOperation_t(112);
    /// Operate with the conjugate transpose of the matrix.
    pub const HIPBLAS_OP_C: hipblasOperation_t = hipblasOperation_t(113);
}
#[repr(transparent)]
/// \\brief Used to specify whether the matrix is to be transposed or not.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasOperation_t(pub ::core::ffi::c_int);
impl hipblasComputeType_t {
    /// compute will be at least 16-bit precision
    pub const HIPBLAS_COMPUTE_16F: hipblasComputeType_t = hipblasComputeType_t(0);
    /// compute will be exactly 16-bit precision
    pub const HIPBLAS_COMPUTE_16F_PEDANTIC: hipblasComputeType_t = hipblasComputeType_t(1);
    /// compute will be at least 32-bit precision
    pub const HIPBLAS_COMPUTE_32F: hipblasComputeType_t = hipblasComputeType_t(2);
    /// compute will be exactly 32-bit precision
    pub const HIPBLAS_COMPUTE_32F_PEDANTIC: hipblasComputeType_t = hipblasComputeType_t(3);
    /// 32-bit input can use 16-bit compute
    pub const HIPBLAS_COMPUTE_32F_FAST_16F: hipblasComputeType_t = hipblasComputeType_t(4);
    /// 32-bit input can is bf16 compute
    pub const HIPBLAS_COMPUTE_32F_FAST_16BF: hipblasComputeType_t = hipblasComputeType_t(5);
    pub const HIPBLAS_COMPUTE_32F_FAST_TF32: hipblasComputeType_t = hipblasComputeType_t(6);
    /// compute will be at least 64-bit precision
    pub const HIPBLAS_COMPUTE_64F: hipblasComputeType_t = hipblasComputeType_t(7);
    /// compute will be exactly 64-bit precision
    pub const HIPBLAS_COMPUTE_64F_PEDANTIC: hipblasComputeType_t = hipblasComputeType_t(8);
    /// compute will be at least 32-bit integer precision
    pub const HIPBLAS_COMPUTE_32I: hipblasComputeType_t = hipblasComputeType_t(9);
    /// compute will be exactly 32-bit integer precision
    pub const HIPBLAS_COMPUTE_32I_PEDANTIC: hipblasComputeType_t = hipblasComputeType_t(10);
    /// 32-bit compute using fp8 mfma instruction
    pub const HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ: hipblasComputeType_t = hipblasComputeType_t(100);
    /// 32-bit compute using bf8 mfma instruction
    pub const HIPBLAS_COMPUTE_32F_FAST_8BF_FNUZ: hipblasComputeType_t = hipblasComputeType_t(101);
    /// 32-bit compute using f8bf8 mfma instruction
    pub const HIPBLAS_COMPUTE_32F_FAST_8F8BF_FNUZ: hipblasComputeType_t = hipblasComputeType_t(102);
    /// 32-bit compute using bf8f8 mfma instruction
    pub const HIPBLAS_COMPUTE_32F_FAST_8BF8F_FNUZ: hipblasComputeType_t = hipblasComputeType_t(103);
}
#[repr(transparent)]
/// \\brief The compute type to be used. Currently only used with GemmEx with the HIPBLAS_V2 interface.\n         Note that support for compute types is largely dependent on backend.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasComputeType_t(pub ::core::ffi::c_int);
/// \\brief `hipblasHandle_t` is a void pointer that stores the library context (either rocBLAS or cuBLAS).
pub type hipblasHandle_t = *mut ::core::ffi::c_void;
pub type hipblasHalf = u16;
/// \\brief  To specify the datatype as a signed char
pub type hipblasInt8 = i8;
/// \\brief  Stride between matrices or vectors in strided_batched functions.
pub type hipblasStride = i64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct hipblasBfloat16 {
    pub data: u16,
}
impl hipblasPointerMode_t {
    /// Scalar values affected by this variable will be located on the host.
    pub const HIPBLAS_POINTER_MODE_HOST: hipblasPointerMode_t = hipblasPointerMode_t(0);
    ///  Scalar values affected by this variable will be located on the device.
    pub const HIPBLAS_POINTER_MODE_DEVICE: hipblasPointerMode_t = hipblasPointerMode_t(1);
}
#[repr(transparent)]
/// \\brief Indicates whether scalar pointers are on the host or device. This is used for scalars alpha and beta and for scalar function return values.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasPointerMode_t(pub ::core::ffi::c_int);
impl hipblasFillMode_t {
    ///  Upper triangle.
    pub const HIPBLAS_FILL_MODE_UPPER: hipblasFillMode_t = hipblasFillMode_t(121);
    ///  Lower triangle.
    pub const HIPBLAS_FILL_MODE_LOWER: hipblasFillMode_t = hipblasFillMode_t(122);
    pub const HIPBLAS_FILL_MODE_FULL: hipblasFillMode_t = hipblasFillMode_t(123);
}
#[repr(transparent)]
/// \\brief Used by the Hermitian, symmetric, and triangular matrix routines to specify whether the upper or lower triangle is being referenced.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasFillMode_t(pub ::core::ffi::c_int);
impl hipblasDiagType_t {
    ///  Non-unit triangular.
    pub const HIPBLAS_DIAG_NON_UNIT: hipblasDiagType_t = hipblasDiagType_t(131);
    ///  Unit triangular.
    pub const HIPBLAS_DIAG_UNIT: hipblasDiagType_t = hipblasDiagType_t(132);
}
#[repr(transparent)]
/// \\brief Used by the triangular matrix routines to specify whether the matrix is unit triangular.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasDiagType_t(pub ::core::ffi::c_int);
impl hipblasSideMode_t {
    pub const HIPBLAS_SIDE_LEFT: hipblasSideMode_t = hipblasSideMode_t(141);
    pub const HIPBLAS_SIDE_RIGHT: hipblasSideMode_t = hipblasSideMode_t(142);
    pub const HIPBLAS_SIDE_BOTH: hipblasSideMode_t = hipblasSideMode_t(143);
}
#[repr(transparent)]
/// \\brief Indicates the side matrix A is located on, relative to matrix B, during multiplication.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasSideMode_t(pub ::core::ffi::c_int);
impl hipblasMath_t {
    pub const HIPBLAS_DEFAULT_MATH: hipblasMath_t = hipblasMath_t(0);
    pub const HIPBLAS_XF32_XDL_MATH: hipblasMath_t = hipblasMath_t(1);
    pub const HIPBLAS_PEDANTIC_MATH: hipblasMath_t = hipblasMath_t(2);
    pub const HIPBLAS_TF32_TENSOR_OP_MATH: hipblasMath_t = hipblasMath_t(3);
    pub const HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: hipblasMath_t = hipblasMath_t(4);
    pub const HIPBLAS_TENSOR_OP_MATH: hipblasMath_t = hipblasMath_t(5);
}
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasMath_t(pub ::core::ffi::c_int);
impl hipblasGemmAlgo_t {
    ///  enumerator `rocblas_gemm_algo_standard`
    pub const HIPBLAS_GEMM_DEFAULT: hipblasGemmAlgo_t = hipblasGemmAlgo_t(160);
}
#[repr(transparent)]
/// \\brief Indicates whether the layer is active through the use of a bitmask.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasGemmAlgo_t(pub ::core::ffi::c_int);
impl hipblasAtomicsMode_t {
    ///  Algorithms will refrain from atomics where applicable.
    pub const HIPBLAS_ATOMICS_NOT_ALLOWED: hipblasAtomicsMode_t = hipblasAtomicsMode_t(0);
    /// Algorithms will take advantage of atomics where applicable.
    pub const HIPBLAS_ATOMICS_ALLOWED: hipblasAtomicsMode_t = hipblasAtomicsMode_t(1);
}
#[repr(transparent)]
/// \\brief Indicates whether atomics operations are allowed. Not allowing atomic operations can generally improve determinism and repeatability of results at a cost of performance.\n         By default, the rocBLAS backend allows atomic operations, while the cuBLAS backend disallows atomic operations. See the backend documentation\n         for more details.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasAtomicsMode_t(pub ::core::ffi::c_int);
impl hipblasGemmFlags_t {
    /// Default empty flags
    pub const HIPBLAS_GEMM_FLAGS_NONE: hipblasGemmFlags_t = hipblasGemmFlags_t(0);
    pub const HIPBLAS_GEMM_FLAGS_USE_CU_EFFICIENCY: hipblasGemmFlags_t = hipblasGemmFlags_t(2);
    /// enumerator rocblas_gemm_flags_fp16_alt_impl
    pub const HIPBLAS_GEMM_FLAGS_FP16_ALT_IMPL: hipblasGemmFlags_t = hipblasGemmFlags_t(4);
    pub const HIPBLAS_GEMM_FLAGS_CHECK_SOLUTION_INDEX: hipblasGemmFlags_t = hipblasGemmFlags_t(8);
    pub const HIPBLAS_GEMM_FLAGS_FP16_ALT_IMPL_RNZ: hipblasGemmFlags_t = hipblasGemmFlags_t(16);
}
#[repr(transparent)]
/// \\brief Control flags passed into gemm ex with flags algorithms. Only relevant with rocBLAS backend. See rocBLAS documentation\n         for more information.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasGemmFlags_t(pub ::core::ffi::c_int);
impl hipblasLibraryProperty_t {
    /// Major version number
    pub const HIPBLAS_MAJOR_VERSION: hipblasLibraryProperty_t = hipblasLibraryProperty_t(0);
    /// Minor version number
    pub const HIPBLAS_MINOR_VERSION: hipblasLibraryProperty_t = hipblasLibraryProperty_t(1);
    /// Patch level
    pub const HIPBLAS_PATCH_LEVEL: hipblasLibraryProperty_t = hipblasLibraryProperty_t(2);
}
#[repr(transparent)]
/// \\brief Library property types for querying version components.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasLibraryProperty_t(pub ::core::ffi::c_int);
