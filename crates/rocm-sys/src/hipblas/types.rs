#[repr(i32)]
/// hipblas status codes definition
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasStatus_t {
    /// Function succeeds
    HIPBLAS_STATUS_SUCCESS = 0,
    /// HIPBLAS library not initialized
    HIPBLAS_STATUS_NOT_INITIALIZED = 1,
    /// resource allocation failed
    HIPBLAS_STATUS_ALLOC_FAILED = 2,
    /// unsupported numerical value was passed to function
    HIPBLAS_STATUS_INVALID_VALUE = 3,
    /// access to GPU memory space failed
    HIPBLAS_STATUS_MAPPING_ERROR = 4,
    /// GPU program failed to execute
    HIPBLAS_STATUS_EXECUTION_FAILED = 5,
    /// an internal HIPBLAS operation failed
    HIPBLAS_STATUS_INTERNAL_ERROR = 6,
    /// function not implemented
    HIPBLAS_STATUS_NOT_SUPPORTED = 7,
    /// architecture mismatch
    HIPBLAS_STATUS_ARCH_MISMATCH = 8,
    /// hipBLAS handle is null pointer
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9,
    ///  unsupported enum value was passed to function
    HIPBLAS_STATUS_INVALID_ENUM = 10,
    ///  back-end returned an unsupported status code
    HIPBLAS_STATUS_UNKNOWN = 11,
}
#[repr(i32)]
/// Used to specify whether the matrix is to be transposed or not.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasOperation_t {
    ///  Operate with the matrix.
    HIPBLAS_OP_N = 111,
    ///  Operate with the transpose of the matrix.
    HIPBLAS_OP_T = 112,
    /// Operate with the conjugate transpose of the matrix.
    HIPBLAS_OP_C = 113,
}
#[repr(i32)]
/// The compute type to be used. Currently only used with GemmEx with the HIPBLAS_V2 interface.\n         Note that support for compute types is largely dependent on backend.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasComputeType_t {
    /// compute will be at least 16-bit precision
    HIPBLAS_COMPUTE_16F = 0,
    /// compute will be exactly 16-bit precision
    HIPBLAS_COMPUTE_16F_PEDANTIC = 1,
    /// compute will be at least 32-bit precision
    HIPBLAS_COMPUTE_32F = 2,
    /// compute will be exactly 32-bit precision
    HIPBLAS_COMPUTE_32F_PEDANTIC = 3,
    /// 32-bit input can use 16-bit compute
    HIPBLAS_COMPUTE_32F_FAST_16F = 4,
    /// 32-bit input can is bf16 compute
    HIPBLAS_COMPUTE_32F_FAST_16BF = 5,
    HIPBLAS_COMPUTE_32F_FAST_TF32 = 6,
    /// compute will be at least 64-bit precision
    HIPBLAS_COMPUTE_64F = 7,
    /// compute will be exactly 64-bit precision
    HIPBLAS_COMPUTE_64F_PEDANTIC = 8,
    /// compute will be at least 32-bit integer precision
    HIPBLAS_COMPUTE_32I = 9,
    /// compute will be exactly 32-bit integer precision
    HIPBLAS_COMPUTE_32I_PEDANTIC = 10,
    /// 32-bit compute using fp8 mfma instruction
    HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ = 100,
    /// 32-bit compute using bf8 mfma instruction
    HIPBLAS_COMPUTE_32F_FAST_8BF_FNUZ = 101,
    /// 32-bit compute using f8bf8 mfma instruction
    HIPBLAS_COMPUTE_32F_FAST_8F8BF_FNUZ = 102,
    /// 32-bit compute using bf8f8 mfma instruction
    HIPBLAS_COMPUTE_32F_FAST_8BF8F_FNUZ = 103,
}
/// ``hipblasHandle_t`` is a void pointer that stores the library context (either rocBLAS or cuBLAS).
pub type hipblasHandle_t = *mut ::core::ffi::c_void;
pub type hipblasHalf = u16;
///  To specify the datatype as a signed char
pub type hipblasInt8 = i8;
///  Stride between matrices or vectors in strided_batched functions.
pub type hipblasStride = i64;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipblasBfloat16 {
    pub data: u16,
}
#[repr(i32)]
/// Indicates whether scalar pointers are on the host or device. This is used for scalars alpha and beta and for scalar function return values.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasPointerMode_t {
    /// Scalar values affected by this variable will be located on the host.
    HIPBLAS_POINTER_MODE_HOST = 0,
    ///  Scalar values affected by this variable will be located on the device.
    HIPBLAS_POINTER_MODE_DEVICE = 1,
}
#[repr(i32)]
/// Used by the Hermitian, symmetric, and triangular matrix routines to specify whether the upper or lower triangle is being referenced.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasFillMode_t {
    ///  Upper triangle.
    HIPBLAS_FILL_MODE_UPPER = 121,
    ///  Lower triangle.
    HIPBLAS_FILL_MODE_LOWER = 122,
    HIPBLAS_FILL_MODE_FULL = 123,
}
#[repr(i32)]
/// Used by the triangular matrix routines to specify whether the matrix is unit triangular.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasDiagType_t {
    ///  Non-unit triangular.
    HIPBLAS_DIAG_NON_UNIT = 131,
    ///  Unit triangular.
    HIPBLAS_DIAG_UNIT = 132,
}
#[repr(i32)]
/// Indicates the side matrix A is located on, relative to matrix B, during multiplication.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasSideMode_t {
    HIPBLAS_SIDE_LEFT = 141,
    HIPBLAS_SIDE_RIGHT = 142,
    HIPBLAS_SIDE_BOTH = 143,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasMath_t {
    HIPBLAS_DEFAULT_MATH = 0,
    HIPBLAS_XF32_XDL_MATH = 1,
    HIPBLAS_PEDANTIC_MATH = 2,
    HIPBLAS_TF32_TENSOR_OP_MATH = 3,
    HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 4,
    HIPBLAS_TENSOR_OP_MATH = 5,
}
#[repr(i32)]
/// Indicates whether the layer is active through the use of a bitmask.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasGemmAlgo_t {
    ///  enumerator ``rocblas_gemm_algo_standard``
    HIPBLAS_GEMM_DEFAULT = 160,
}
#[repr(i32)]
/// Indicates whether atomics operations are allowed. Not allowing atomic operations can generally improve determinism and repeatability of results at a cost of performance.\n         By default, the rocBLAS backend allows atomic operations, while the cuBLAS backend disallows atomic operations. See the backend documentation\n         for more details.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasAtomicsMode_t {
    ///  Algorithms will refrain from atomics where applicable.
    HIPBLAS_ATOMICS_NOT_ALLOWED = 0,
    /// Algorithms will take advantage of atomics where applicable.
    HIPBLAS_ATOMICS_ALLOWED = 1,
}
#[repr(i32)]
/// Control flags passed into gemm ex with flags algorithms. Only relevant with rocBLAS backend. See rocBLAS documentation\n         for more information.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasGemmFlags_t {
    /// Default empty flags
    HIPBLAS_GEMM_FLAGS_NONE = 0,
    HIPBLAS_GEMM_FLAGS_USE_CU_EFFICIENCY = 2,
    /// enumerator rocblas_gemm_flags_fp16_alt_impl
    HIPBLAS_GEMM_FLAGS_FP16_ALT_IMPL = 4,
    HIPBLAS_GEMM_FLAGS_CHECK_SOLUTION_INDEX = 8,
    HIPBLAS_GEMM_FLAGS_FP16_ALT_IMPL_RNZ = 16,
}
#[repr(i32)]
/// Library property types for querying version components.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLibraryProperty_t {
    /// Major version number
    HIPBLAS_MAJOR_VERSION = 0,
    /// Minor version number
    HIPBLAS_MINOR_VERSION = 1,
    /// Patch level
    HIPBLAS_PATCH_LEVEL = 2,
}
