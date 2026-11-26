/// Struct to represent a 16-bit Brain floating-point number.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_bfloat16 {
    pub data: u16,
}
#[repr(C)]
#[derive(Debug)]
pub struct _rocblas_handle {
    _unused: [u8; 0],
}
/// rocblas_handle is a structure holding the rocBLAS library context.
///  It must be initialized using rocblas_create_handle(),
///  and the returned handle must be passed
///  to all subsequent library function calls.
///  It should be destroyed at the end using rocblas_destroy_handle().
pub type rocblas_handle = *mut _rocblas_handle;
/// Opaque base class for device memory allocation
#[repr(C)]
#[derive(Debug)]
pub struct rocblas_device_malloc_base {
    _unused: [u8; 0],
}
pub type rocblas_int = i32;
/// Stride between matrices or vectors in strided_batched functions
pub type rocblas_stride = i64;
/// Single precision floating point type
pub type rocblas_float = f32;
/// Double precision floating point type
pub type rocblas_double = f64;
/// Structure to represent an unsigned 16-bit integer type.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_half {
    pub data: u16,
}
/// Struct to represent a complex number with single-precision real and imaginary parts.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_float_complex {
    pub x: f32,
    pub y: f32,
}
/// Struct to represent a complex number with double-precision real and imaginary parts.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_double_complex {
    pub x: f64,
    pub y: f64,
}
impl rocblas_operation_ {
    /// Operate with the matrix.
    pub const rocblas_operation_none: rocblas_operation_ = rocblas_operation_(111);
    /// Operate with the transpose of the matrix.
    pub const rocblas_operation_transpose: rocblas_operation_ = rocblas_operation_(112);
    pub const rocblas_operation_conjugate_transpose: rocblas_operation_ = rocblas_operation_(113);
}
#[repr(transparent)]
/// Used to specify whether the matrix is to be transposed or not.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_operation_(pub ::core::ffi::c_int);
/// Used to specify whether the matrix is to be transposed or not.
pub use self::rocblas_operation_ as rocblas_operation;
impl rocblas_fill_ {
    /// Upper triangle.
    pub const rocblas_fill_upper: rocblas_fill_ = rocblas_fill_(121);
    /// Lower triangle.
    pub const rocblas_fill_lower: rocblas_fill_ = rocblas_fill_(122);
    pub const rocblas_fill_full: rocblas_fill_ = rocblas_fill_(123);
}
#[repr(transparent)]
/// Used by the Hermitian, symmetric, and triangular matrix
///  routines to specify whether the upper or lower triangle is being referenced.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_fill_(pub ::core::ffi::c_int);
/// Used by the Hermitian, symmetric, and triangular matrix
///  routines to specify whether the upper or lower triangle is being referenced.
pub use self::rocblas_fill_ as rocblas_fill;
impl rocblas_diagonal_ {
    /// Non-unit triangular.
    pub const rocblas_diagonal_non_unit: rocblas_diagonal_ = rocblas_diagonal_(131);
    /// Unit triangular.
    pub const rocblas_diagonal_unit: rocblas_diagonal_ = rocblas_diagonal_(132);
}
#[repr(transparent)]
/// Used by the triangular matrix routines to specify whether the
///  matrix is unit triangular.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_diagonal_(pub ::core::ffi::c_int);
/// Used by the triangular matrix routines to specify whether the
///  matrix is unit triangular.
pub use self::rocblas_diagonal_ as rocblas_diagonal;
impl rocblas_side_ {
    /// Multiply general matrix by symmetric,
    /// Hermitian, or triangular matrix on the left.
    pub const rocblas_side_left: rocblas_side_ = rocblas_side_(141);
    /// Multiply general matrix by symmetric,
    /// Hermitian, or triangular matrix on the right.
    pub const rocblas_side_right: rocblas_side_ = rocblas_side_(142);
    pub const rocblas_side_both: rocblas_side_ = rocblas_side_(143);
}
#[repr(transparent)]
/// Indicates the side matrix A is located on, relative to matrix B, during multiplication.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_side_(pub ::core::ffi::c_int);
/// Indicates the side matrix A is located on, relative to matrix B, during multiplication.
pub use self::rocblas_side_ as rocblas_side;
impl rocblas_datatype_ {
    /// 16-bit floating point, real
    pub const rocblas_datatype_f16_r: rocblas_datatype_ = rocblas_datatype_(150);
    /// 32-bit floating point, real
    pub const rocblas_datatype_f32_r: rocblas_datatype_ = rocblas_datatype_(151);
    /// 64-bit floating point, real
    pub const rocblas_datatype_f64_r: rocblas_datatype_ = rocblas_datatype_(152);
    /// 16-bit floating point, complex
    pub const rocblas_datatype_f16_c: rocblas_datatype_ = rocblas_datatype_(153);
    /// 32-bit floating point, complex
    pub const rocblas_datatype_f32_c: rocblas_datatype_ = rocblas_datatype_(154);
    /// 64-bit floating point, complex
    pub const rocblas_datatype_f64_c: rocblas_datatype_ = rocblas_datatype_(155);
    ///  8-bit signed integer, real
    pub const rocblas_datatype_i8_r: rocblas_datatype_ = rocblas_datatype_(160);
    ///  8-bit unsigned integer, real
    pub const rocblas_datatype_u8_r: rocblas_datatype_ = rocblas_datatype_(161);
    /// 32-bit signed integer, real
    pub const rocblas_datatype_i32_r: rocblas_datatype_ = rocblas_datatype_(162);
    /// 32-bit unsigned integer, real
    pub const rocblas_datatype_u32_r: rocblas_datatype_ = rocblas_datatype_(163);
    ///  8-bit signed integer, complex
    pub const rocblas_datatype_i8_c: rocblas_datatype_ = rocblas_datatype_(164);
    ///  8-bit unsigned integer, complex
    pub const rocblas_datatype_u8_c: rocblas_datatype_ = rocblas_datatype_(165);
    /// 32-bit signed integer, complex
    pub const rocblas_datatype_i32_c: rocblas_datatype_ = rocblas_datatype_(166);
    /// 32-bit unsigned integer, complex
    pub const rocblas_datatype_u32_c: rocblas_datatype_ = rocblas_datatype_(167);
    /// 16-bit bfloat, real
    pub const rocblas_datatype_bf16_r: rocblas_datatype_ = rocblas_datatype_(168);
    /// 16-bit bfloat, complex
    pub const rocblas_datatype_bf16_c: rocblas_datatype_ = rocblas_datatype_(169);
    /// Invalid datatype value. Do not use.
    pub const rocblas_datatype_invalid: rocblas_datatype_ = rocblas_datatype_(255);
}
#[repr(transparent)]
/// Indicates the precision width of the data stored in a BLAS type.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_datatype_(pub ::core::ffi::c_int);
/// Indicates the precision width of the data stored in a BLAS type.
pub use self::rocblas_datatype_ as rocblas_datatype;
impl rocblas_status_ {
    /// Success.
    pub const rocblas_status_success: rocblas_status_ = rocblas_status_(0);
    /// Handle is uninitialized, invalid, or null.
    pub const rocblas_status_invalid_handle: rocblas_status_ = rocblas_status_(1);
    /// Function is not implemented.
    pub const rocblas_status_not_implemented: rocblas_status_ = rocblas_status_(2);
    /// Invalid pointer argument.
    pub const rocblas_status_invalid_pointer: rocblas_status_ = rocblas_status_(3);
    /// Invalid size argument.
    pub const rocblas_status_invalid_size: rocblas_status_ = rocblas_status_(4);
    /// Failed internal memory allocation, copy, or dealloc.
    pub const rocblas_status_memory_error: rocblas_status_ = rocblas_status_(5);
    /// Other internal library failure.
    pub const rocblas_status_internal_error: rocblas_status_ = rocblas_status_(6);
    /// Performance degraded due to low device memory.
    pub const rocblas_status_perf_degraded: rocblas_status_ = rocblas_status_(7);
    /// Unmatched start/stop size query.
    pub const rocblas_status_size_query_mismatch: rocblas_status_ = rocblas_status_(8);
    /// Queried device memory size increased.
    pub const rocblas_status_size_increased: rocblas_status_ = rocblas_status_(9);
    /// Queried device memory size unchanged.
    pub const rocblas_status_size_unchanged: rocblas_status_ = rocblas_status_(10);
    /// Passed argument not valid.
    pub const rocblas_status_invalid_value: rocblas_status_ = rocblas_status_(11);
    /// Nothing preventing function from proceeding.
    pub const rocblas_status_continue: rocblas_status_ = rocblas_status_(12);
    pub const rocblas_status_check_numerics_fail: rocblas_status_ = rocblas_status_(13);
    pub const rocblas_status_excluded_from_build: rocblas_status_ = rocblas_status_(14);
    pub const rocblas_status_arch_mismatch: rocblas_status_ = rocblas_status_(15);
}
#[repr(transparent)]
///   rocBLAS status codes definition.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_status_(::core::ffi::c_int);
///   rocBLAS status codes definition.
pub use self::rocblas_status_ as rocblas_status;
impl rocblas_pointer_mode_ {
    /// Scalar values affected by this variable are located on the host.
    pub const rocblas_pointer_mode_host: rocblas_pointer_mode_ = rocblas_pointer_mode_(0);
    /// Scalar values affected by this variable are located on the device.
    pub const rocblas_pointer_mode_device: rocblas_pointer_mode_ = rocblas_pointer_mode_(1);
}
#[repr(transparent)]
/// Indicates whether scalar pointers are on the host or device. This is used for
/// scalars alpha and beta and for scalar function return values.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_pointer_mode_(pub ::core::ffi::c_int);
/// Indicates whether scalar pointers are on the host or device. This is used for
/// scalars alpha and beta and for scalar function return values.
pub use self::rocblas_pointer_mode_ as rocblas_pointer_mode;
impl rocblas_atomics_mode_ {
    /// Algorithms refrain from atomics where applicable
    pub const rocblas_atomics_not_allowed: rocblas_atomics_mode_ = rocblas_atomics_mode_(0);
    /// Algorithms take advantage of atomics where applicable
    pub const rocblas_atomics_allowed: rocblas_atomics_mode_ = rocblas_atomics_mode_(1);
}
#[repr(transparent)]
/// Indicates whether atomics operations are allowed. Not allowing atomic operations
/// can generally improve determinism and repeatability of results at a cost to performance.
/// Defaults to `rocblas_atomics_allowed`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_atomics_mode_(pub ::core::ffi::c_int);
/// Indicates whether atomics operations are allowed. Not allowing atomic operations
/// can generally improve determinism and repeatability of results at a cost to performance.
/// Defaults to `rocblas_atomics_allowed`.
pub use self::rocblas_atomics_mode_ as rocblas_atomics_mode;
impl rocblas_performance_metric_ {
    /// Use Tensile's default performance metric for solution selection
    pub const rocblas_default_performance_metric: rocblas_performance_metric_ =
        rocblas_performance_metric_(0);
    /// Select the solution with the highest GFlops across all compute units
    pub const rocblas_device_efficiency_performance_metric: rocblas_performance_metric_ =
        rocblas_performance_metric_(1);
    /// Select the solution with the highest GFlops per compute unit it uses. This
    ///  may be useful when running multiple small gemm problems simultaneously
    pub const rocblas_cu_efficiency_performance_metric: rocblas_performance_metric_ =
        rocblas_performance_metric_(2);
}
#[repr(transparent)]
/// Indicates which performance metric Tensile uses when selecting the optimal
/// solution for gemm problems.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_performance_metric_(pub ::core::ffi::c_int);
/// Indicates which performance metric Tensile uses when selecting the optimal
/// solution for gemm problems.
pub use self::rocblas_performance_metric_ as rocblas_performance_metric;
impl rocblas_layer_mode_ {
    /// No logging will take place.
    pub const rocblas_layer_mode_none: rocblas_layer_mode_ = rocblas_layer_mode_(0);
    /// A line containing the function name and value of arguments passed is printed with each rocBLAS function call.
    pub const rocblas_layer_mode_log_trace: rocblas_layer_mode_ = rocblas_layer_mode_(1);
    /// Outputs a line each time a rocBLAS function is called. This line can be used with rocblas-bench to make the same call again.
    pub const rocblas_layer_mode_log_bench: rocblas_layer_mode_ = rocblas_layer_mode_(2);
    /// Outputs a YAML description of each rocBLAS function called, along with its arguments and number of times it was called.
    pub const rocblas_layer_mode_log_profile: rocblas_layer_mode_ = rocblas_layer_mode_(4);
    /// Outputs to the same stream as trace logging with limited internal API details, such as the GEMM backend that was used.
    pub const rocblas_layer_mode_log_internal: rocblas_layer_mode_ = rocblas_layer_mode_(8);
}
#[repr(transparent)]
/// Indicates whether the layer is active by means of a bitmask.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_layer_mode_(pub ::core::ffi::c_int);
/// Indicates whether the layer is active by means of a bitmask.
pub use self::rocblas_layer_mode_ as rocblas_layer_mode;
impl rocblas_gemm_algo_ {
    pub const rocblas_gemm_algo_standard: rocblas_gemm_algo_ = rocblas_gemm_algo_(0);
    pub const rocblas_gemm_algo_solution_index: rocblas_gemm_algo_ = rocblas_gemm_algo_(1);
}
#[repr(transparent)]
/// Indicates whether the layer is active by means of a bitmask.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_gemm_algo_(pub ::core::ffi::c_int);
/// Indicates whether the layer is active by means of a bitmask.
pub use self::rocblas_gemm_algo_ as rocblas_gemm_algo;
impl rocblas_geam_ex_operation_ {
    pub const rocblas_geam_ex_operation_min_plus: rocblas_geam_ex_operation_ =
        rocblas_geam_ex_operation_(0);
    pub const rocblas_geam_ex_operation_plus_min: rocblas_geam_ex_operation_ =
        rocblas_geam_ex_operation_(1);
}
#[repr(transparent)]
/// Which mathematical geam-like operation to perform for geam_ex
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_geam_ex_operation_(pub ::core::ffi::c_int);
/// Which mathematical geam-like operation to perform for geam_ex
pub use self::rocblas_geam_ex_operation_ as rocblas_geam_ex_operation;
impl rocblas_gemm_flags_ {
    /// Default empty flags
    pub const rocblas_gemm_flags_none: rocblas_gemm_flags_ = rocblas_gemm_flags_(0);
    /// Before ROCm 6.0, `rocblas_gemm_flags_pack_int8x4 = 0x1` was in use. This has now been removed and is available for future use. */
    /// /*! Select the GEMM problem with the highest efficiency per compute unit used, which is useful for running multiple smaller problems
    ///  simultaneously. This takes precedence over the performance metric set in `rocblas_handle` and currently only works for
    ///  gemm_*_ex problems.
    pub const rocblas_gemm_flags_use_cu_efficiency: rocblas_gemm_flags_ = rocblas_gemm_flags_(2);
    /// Select an alternate implementation for the Instinct MI200 `FP16` HPA
    ///  (High Precision Accumulate) GEMM kernel. This implementation uses the `BF16` matrix
    ///  instructions with reduced accuracy in cases where computation cannot
    ///  tolerate the `FP16` matrix instructions, flushing subnormal `FP16`
    ///  input/output data to zero. See the \"MI200 (gfx90a) Considerations\"
    ///  section of the rocBLAS design notes documentation for more details.
    pub const rocblas_gemm_flags_fp16_alt_impl: rocblas_gemm_flags_ = rocblas_gemm_flags_(4);
    /// Select an alternate implementation for the Instinct MI200 `FP16` HPA
    ///  (High Precision Accumulate) GEMM kernel. This implementation uses the `BF16` matrix
    ///  instructions with reduced accuracy in cases where computation cannot
    ///  tolerate the `FP16` matrix instructions, flushing subnormal `FP16`
    ///  input/output data to zero. See the \"MI200 (gfx90a) Considerations\"
    ///  section of the rocBLAS design notes documentation for more details.
    pub const rocblas_gemm_flags_check_solution_index: rocblas_gemm_flags_ = rocblas_gemm_flags_(8);
    /// Select an alternate implementation for the Instinct MI200 `FP16` HPA
    ///  (High Precision Accumulate) GEMM kernel. This implementation uses the `BF16` matrix
    ///  instructions with reduced accuracy in cases where computation cannot
    ///  tolerate the `FP16` matrix instructions, flushing subnormal `FP16`
    ///  input/output data to zero. See the \"MI200 (gfx90a) Considerations\"
    ///  section of the rocBLAS design notes documentation for more details.
    pub const rocblas_gemm_flags_fp16_alt_impl_rnz: rocblas_gemm_flags_ = rocblas_gemm_flags_(16);
    /// Select an alternate implementation for the Instinct MI200 `FP16` HPA
    ///  (High Precision Accumulate) GEMM kernel. This implementation uses the `BF16` matrix
    ///  instructions with reduced accuracy in cases where computation cannot
    ///  tolerate the `FP16` matrix instructions, flushing subnormal `FP16`
    ///  input/output data to zero. See the \"MI200 (gfx90a) Considerations\"
    ///  section of the rocBLAS design notes documentation for more details.
    pub const rocblas_gemm_flags_stochastic_rounding: rocblas_gemm_flags_ = rocblas_gemm_flags_(32);
}
#[repr(transparent)]
/// Controls the flags passed into GEMM algorithms invoked by the Tensile host.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_gemm_flags_(pub ::core::ffi::c_int);
/// Controls the flags passed into GEMM algorithms invoked by the Tensile host.
pub use self::rocblas_gemm_flags_ as rocblas_gemm_flags;
/// Union for representing scalar values
#[repr(C)]
#[derive(Copy, Clone)]
pub union rocblas_union_u {
    pub h: rocblas_half,
    pub s: f32,
    pub d: f64,
    pub i: i32,
    pub c: rocblas_float_complex,
    pub z: rocblas_double_complex,
}
/// Union for representing scalar values
pub type rocblas_union_t = rocblas_union_u;
impl rocblas_check_numerics_mode_ {
    pub const rocblas_check_numerics_mode_no_check: rocblas_check_numerics_mode_ =
        rocblas_check_numerics_mode_(0);
    pub const rocblas_check_numerics_mode_info: rocblas_check_numerics_mode_ =
        rocblas_check_numerics_mode_(1);
    pub const rocblas_check_numerics_mode_warn: rocblas_check_numerics_mode_ =
        rocblas_check_numerics_mode_(2);
    pub const rocblas_check_numerics_mode_fail: rocblas_check_numerics_mode_ =
        rocblas_check_numerics_mode_(4);
    pub const rocblas_check_numerics_mode_only_nan_inf: rocblas_check_numerics_mode_ =
        rocblas_check_numerics_mode_(8);
}
#[repr(transparent)]
/// Numerical checking for verifying the Input and Output vector/matrix of the rocBLAS functions for a NaN, zero, infinity and denormal value
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_check_numerics_mode_(pub ::core::ffi::c_int);
/// Numerical checking for verifying the Input and Output vector/matrix of the rocBLAS functions for a NaN, zero, infinity and denormal value
pub use self::rocblas_check_numerics_mode_ as rocblas_check_numerics_mode;
impl rocblas_math_mode_ {
    pub const rocblas_default_math: rocblas_math_mode_ = rocblas_math_mode_(0);
    pub const rocblas_xf32_xdl_math_op: rocblas_math_mode_ = rocblas_math_mode_(1);
}
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocblas_math_mode_(pub ::core::ffi::c_int);
pub use self::rocblas_math_mode_ as rocblas_math_mode;
