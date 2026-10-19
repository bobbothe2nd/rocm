pub type __u_char = ::core::ffi::c_uchar;
pub type __u_short = ::core::ffi::c_ushort;
pub type __u_int = ::core::ffi::c_uint;
pub type __u_long = ::core::ffi::c_ulong;
pub type __int8_t = ::core::ffi::c_schar;
pub type __uint8_t = ::core::ffi::c_uchar;
pub type __int16_t = ::core::ffi::c_short;
pub type __uint16_t = ::core::ffi::c_ushort;
pub type __int32_t = ::core::ffi::c_int;
pub type __uint32_t = ::core::ffi::c_uint;
pub type __int64_t = ::core::ffi::c_long;
pub type __uint64_t = ::core::ffi::c_ulong;
pub type __int_least8_t = __int8_t;
pub type __uint_least8_t = __uint8_t;
pub type __int_least16_t = __int16_t;
pub type __uint_least16_t = __uint16_t;
pub type __int_least32_t = __int32_t;
pub type __uint_least32_t = __uint32_t;
pub type __int_least64_t = __int64_t;
pub type __uint_least64_t = __uint64_t;
pub type __quad_t = ::core::ffi::c_long;
pub type __u_quad_t = ::core::ffi::c_ulong;
pub type __intmax_t = ::core::ffi::c_long;
pub type __uintmax_t = ::core::ffi::c_ulong;
pub type __dev_t = ::core::ffi::c_ulong;
pub type __uid_t = ::core::ffi::c_uint;
pub type __gid_t = ::core::ffi::c_uint;
pub type __ino_t = ::core::ffi::c_ulong;
pub type __ino64_t = ::core::ffi::c_ulong;
pub type __mode_t = ::core::ffi::c_uint;
pub type __nlink_t = ::core::ffi::c_ulong;
pub type __off_t = ::core::ffi::c_long;
pub type __off64_t = ::core::ffi::c_long;
pub type __pid_t = ::core::ffi::c_int;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __fsid_t {
    pub __val: [::core::ffi::c_int; 2usize],
}
pub type __clock_t = ::core::ffi::c_long;
pub type __rlim_t = ::core::ffi::c_ulong;
pub type __rlim64_t = ::core::ffi::c_ulong;
pub type __id_t = ::core::ffi::c_uint;
pub type __time_t = ::core::ffi::c_long;
pub type __useconds_t = ::core::ffi::c_uint;
pub type __suseconds_t = ::core::ffi::c_long;
pub type __suseconds64_t = ::core::ffi::c_long;
pub type __daddr_t = ::core::ffi::c_int;
pub type __key_t = ::core::ffi::c_int;
pub type __clockid_t = ::core::ffi::c_int;
pub type __timer_t = *mut ::core::ffi::c_void;
pub type __blksize_t = ::core::ffi::c_long;
pub type __blkcnt_t = ::core::ffi::c_long;
pub type __blkcnt64_t = ::core::ffi::c_long;
pub type __fsblkcnt_t = ::core::ffi::c_ulong;
pub type __fsblkcnt64_t = ::core::ffi::c_ulong;
pub type __fsfilcnt_t = ::core::ffi::c_ulong;
pub type __fsfilcnt64_t = ::core::ffi::c_ulong;
pub type __fsword_t = ::core::ffi::c_long;
pub type __ssize_t = ::core::ffi::c_long;
pub type __syscall_slong_t = ::core::ffi::c_long;
pub type __syscall_ulong_t = ::core::ffi::c_ulong;
pub type __loff_t = __off64_t;
pub type __caddr_t = *mut ::core::ffi::c_char;
pub type __intptr_t = ::core::ffi::c_long;
pub type __socklen_t = ::core::ffi::c_uint;
pub type __sig_atomic_t = ::core::ffi::c_int;
pub type int_least8_t = __int_least8_t;
pub type int_least16_t = __int_least16_t;
pub type int_least32_t = __int_least32_t;
pub type int_least64_t = __int_least64_t;
pub type uint_least8_t = __uint_least8_t;
pub type uint_least16_t = __uint_least16_t;
pub type uint_least32_t = __uint_least32_t;
pub type uint_least64_t = __uint_least64_t;
pub type int_fast8_t = ::core::ffi::c_schar;
pub type int_fast16_t = ::core::ffi::c_long;
pub type int_fast32_t = ::core::ffi::c_long;
pub type int_fast64_t = ::core::ffi::c_long;
pub type uint_fast8_t = ::core::ffi::c_uchar;
pub type uint_fast16_t = ::core::ffi::c_ulong;
pub type uint_fast32_t = ::core::ffi::c_ulong;
pub type uint_fast64_t = ::core::ffi::c_ulong;
pub type intmax_t = __intmax_t;
pub type uintmax_t = __uintmax_t;
#[doc = " \\brief Struct to represent a 16-bit Brain floating-point number."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_bfloat16 {
    pub data: u16,
}
pub type _Float32 = f32;
pub type _Float64 = f64;
pub type _Float32x = f64;
pub type _Float64x = u128;
pub type float_t = f32;
pub type double_t = f64;
pub const FP_NAN: _bindgen_ty_1 = _bindgen_ty_1::FP_NAN;
pub const FP_INFINITE: _bindgen_ty_1 = _bindgen_ty_1::FP_INFINITE;
pub const FP_ZERO: _bindgen_ty_1 = _bindgen_ty_1::FP_ZERO;
pub const FP_SUBNORMAL: _bindgen_ty_1 = _bindgen_ty_1::FP_SUBNORMAL;
pub const FP_NORMAL: _bindgen_ty_1 = _bindgen_ty_1::FP_NORMAL;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum _bindgen_ty_1 {
    FP_NAN = 0,
    FP_INFINITE = 1,
    FP_ZERO = 2,
    FP_SUBNORMAL = 3,
    FP_NORMAL = 4,
}
pub type wchar_t = ::core::ffi::c_int;
#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Copy, Clone)]
pub struct max_align_t {
    pub __clang_max_align_nonce1: ::core::ffi::c_longlong,
    pub __clang_max_align_nonce2: u128,
}
#[repr(C)]
#[derive(Debug)]
pub struct _rocblas_handle {
    _unused: [u8; 0],
}
#[doc = " \\brief rocblas_handle is a structure holding the rocBLAS library context.\n It must be initialized using rocblas_create_handle(),\n and the returned handle must be passed\n to all subsequent library function calls.\n It should be destroyed at the end using rocblas_destroy_handle()."]
pub type rocblas_handle = *mut _rocblas_handle;
#[doc = " \\brief Opaque base class for device memory allocation"]
#[repr(C)]
#[derive(Debug)]
pub struct rocblas_device_malloc_base {
    _unused: [u8; 0],
}
pub type rocblas_int = i32;
#[doc = " \\brief Stride between matrices or vectors in strided_batched functions"]
pub type rocblas_stride = i64;
#[doc = " \\brief Single precision floating point type"]
pub type rocblas_float = f32;
#[doc = " \\brief Double precision floating point type"]
pub type rocblas_double = f64;
#[doc = " \\brief Structure to represent an unsigned 16-bit integer type."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_half {
    pub data: u16,
}
#[doc = " \\brief Struct to represent a complex number with single-precision real and imaginary parts."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_float_complex {
    pub x: f32,
    pub y: f32,
}
#[doc = " \\brief Struct to represent a complex number with double-precision real and imaginary parts."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocblas_double_complex {
    pub x: f64,
    pub y: f64,
}
#[repr(u32)]
#[doc = " \\brief Used to specify whether the matrix is to be transposed or not."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_operation_ {
    #[doc = "< Operate with the matrix."]
    rocblas_operation_none = 111,
    #[doc = "< Operate with the transpose of the matrix."]
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
}
#[doc = " \\brief Used to specify whether the matrix is to be transposed or not."]
pub use self::rocblas_operation_ as rocblas_operation;
#[repr(u32)]
#[doc = " \\brief Used by the Hermitian, symmetric, and triangular matrix\n routines to specify whether the upper or lower triangle is being referenced."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_fill_ {
    #[doc = "< Upper triangle."]
    rocblas_fill_upper = 121,
    #[doc = "< Lower triangle."]
    rocblas_fill_lower = 122,
    rocblas_fill_full = 123,
}
#[doc = " \\brief Used by the Hermitian, symmetric, and triangular matrix\n routines to specify whether the upper or lower triangle is being referenced."]
pub use self::rocblas_fill_ as rocblas_fill;
#[repr(u32)]
#[doc = " \\brief Used by the triangular matrix routines to specify whether the\n matrix is unit triangular."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_diagonal_ {
    #[doc = "< Non-unit triangular."]
    rocblas_diagonal_non_unit = 131,
    #[doc = "< Unit triangular."]
    rocblas_diagonal_unit = 132,
}
#[doc = " \\brief Used by the triangular matrix routines to specify whether the\n matrix is unit triangular."]
pub use self::rocblas_diagonal_ as rocblas_diagonal;
#[repr(u32)]
#[doc = " \\brief Indicates the side matrix A is located on, relative to matrix B, during multiplication."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_side_ {
    #[doc = "< Multiply general matrix by symmetric,\nHermitian, or triangular matrix on the left."]
    rocblas_side_left = 141,
    #[doc = "< Multiply general matrix by symmetric,\nHermitian, or triangular matrix on the right."]
    rocblas_side_right = 142,
    rocblas_side_both = 143,
}
#[doc = " \\brief Indicates the side matrix A is located on, relative to matrix B, during multiplication."]
pub use self::rocblas_side_ as rocblas_side;
#[repr(u32)]
#[doc = " \\brief Indicates the precision width of the data stored in a BLAS type."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_datatype_ {
    #[doc = "< 16-bit floating point, real"]
    rocblas_datatype_f16_r = 150,
    #[doc = "< 32-bit floating point, real"]
    rocblas_datatype_f32_r = 151,
    #[doc = "< 64-bit floating point, real"]
    rocblas_datatype_f64_r = 152,
    #[doc = "< 16-bit floating point, complex"]
    rocblas_datatype_f16_c = 153,
    #[doc = "< 32-bit floating point, complex"]
    rocblas_datatype_f32_c = 154,
    #[doc = "< 64-bit floating point, complex"]
    rocblas_datatype_f64_c = 155,
    #[doc = "<  8-bit signed integer, real"]
    rocblas_datatype_i8_r = 160,
    #[doc = "<  8-bit unsigned integer, real"]
    rocblas_datatype_u8_r = 161,
    #[doc = "< 32-bit signed integer, real"]
    rocblas_datatype_i32_r = 162,
    #[doc = "< 32-bit unsigned integer, real"]
    rocblas_datatype_u32_r = 163,
    #[doc = "<  8-bit signed integer, complex"]
    rocblas_datatype_i8_c = 164,
    #[doc = "<  8-bit unsigned integer, complex"]
    rocblas_datatype_u8_c = 165,
    #[doc = "< 32-bit signed integer, complex"]
    rocblas_datatype_i32_c = 166,
    #[doc = "< 32-bit unsigned integer, complex"]
    rocblas_datatype_u32_c = 167,
    #[doc = "< 16-bit bfloat, real"]
    rocblas_datatype_bf16_r = 168,
    #[doc = "< 16-bit bfloat, complex"]
    rocblas_datatype_bf16_c = 169,
    #[doc = "< Invalid datatype value. Do not use."]
    rocblas_datatype_invalid = 255,
}
#[doc = " \\brief Indicates the precision width of the data stored in a BLAS type."]
pub use self::rocblas_datatype_ as rocblas_datatype;
#[repr(u32)]
#[doc = "   @brief rocBLAS status codes definition."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_status_ {
    #[doc = "< Success."]
    rocblas_status_success = 0,
    #[doc = "< Handle is uninitialized, invalid, or null."]
    rocblas_status_invalid_handle = 1,
    #[doc = "< Function is not implemented."]
    rocblas_status_not_implemented = 2,
    #[doc = "< Invalid pointer argument."]
    rocblas_status_invalid_pointer = 3,
    #[doc = "< Invalid size argument."]
    rocblas_status_invalid_size = 4,
    #[doc = "< Failed internal memory allocation, copy, or dealloc."]
    rocblas_status_memory_error = 5,
    #[doc = "< Other internal library failure."]
    rocblas_status_internal_error = 6,
    #[doc = "< Performance degraded due to low device memory."]
    rocblas_status_perf_degraded = 7,
    #[doc = "< Unmatched start/stop size query."]
    rocblas_status_size_query_mismatch = 8,
    #[doc = "< Queried device memory size increased."]
    rocblas_status_size_increased = 9,
    #[doc = "< Queried device memory size unchanged."]
    rocblas_status_size_unchanged = 10,
    #[doc = "< Passed argument not valid."]
    rocblas_status_invalid_value = 11,
    #[doc = "< Nothing preventing function from proceeding."]
    rocblas_status_continue = 12,
    rocblas_status_check_numerics_fail = 13,
    rocblas_status_excluded_from_build = 14,
    rocblas_status_arch_mismatch = 15,
}
#[doc = "   @brief rocBLAS status codes definition."]
pub use self::rocblas_status_ as rocblas_status;
#[repr(u32)]
#[doc = " \\brief Indicates whether scalar pointers are on the host or device. This is used for\n    scalars alpha and beta and for scalar function return values."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_pointer_mode_ {
    #[doc = " \\brief Scalar values affected by this variable are located on the host."]
    rocblas_pointer_mode_host = 0,
    #[doc = " \\brief Scalar values affected by this variable are located on the device."]
    rocblas_pointer_mode_device = 1,
}
#[doc = " \\brief Indicates whether scalar pointers are on the host or device. This is used for\n    scalars alpha and beta and for scalar function return values."]
pub use self::rocblas_pointer_mode_ as rocblas_pointer_mode;
#[repr(u32)]
#[doc = " \\brief Indicates whether atomics operations are allowed. Not allowing atomic operations\n    can generally improve determinism and repeatability of results at a cost to performance.\n    Defaults to ``rocblas_atomics_allowed``."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_atomics_mode_ {
    #[doc = " \\brief Algorithms refrain from atomics where applicable"]
    rocblas_atomics_not_allowed = 0,
    #[doc = " \\brief Algorithms take advantage of atomics where applicable"]
    rocblas_atomics_allowed = 1,
}
#[doc = " \\brief Indicates whether atomics operations are allowed. Not allowing atomic operations\n    can generally improve determinism and repeatability of results at a cost to performance.\n    Defaults to ``rocblas_atomics_allowed``."]
pub use self::rocblas_atomics_mode_ as rocblas_atomics_mode;
#[repr(u32)]
#[doc = " \\brief Indicates which performance metric Tensile uses when selecting the optimal\n    solution for gemm problems."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_performance_metric_ {
    #[doc = " \\brief Use Tensile's default performance metric for solution selection"]
    rocblas_default_performance_metric = 0,
    #[doc = " \\brief Select the solution with the highest GFlops across all compute units"]
    rocblas_device_efficiency_performance_metric = 1,
    #[doc = " \\brief Select the solution with the highest GFlops per compute unit it uses. This\n may be useful when running multiple small gemm problems simultaneously"]
    rocblas_cu_efficiency_performance_metric = 2,
}
#[doc = " \\brief Indicates which performance metric Tensile uses when selecting the optimal\n    solution for gemm problems."]
pub use self::rocblas_performance_metric_ as rocblas_performance_metric;
#[repr(u32)]
#[doc = " \\brief Indicates whether the layer is active by means of a bitmask."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_layer_mode_ {
    #[doc = " \\brief No logging will take place."]
    rocblas_layer_mode_none = 0,
    #[doc = " \\brief A line containing the function name and value of arguments passed is printed with each rocBLAS function call."]
    rocblas_layer_mode_log_trace = 1,
    #[doc = " \\brief Outputs a line each time a rocBLAS function is called. This line can be used with rocblas-bench to make the same call again."]
    rocblas_layer_mode_log_bench = 2,
    #[doc = " \\brief Outputs a YAML description of each rocBLAS function called, along with its arguments and number of times it was called."]
    rocblas_layer_mode_log_profile = 4,
    #[doc = " \\brief Outputs to the same stream as trace logging with limited internal API details, such as the GEMM backend that was used."]
    rocblas_layer_mode_log_internal = 8,
}
#[doc = " \\brief Indicates whether the layer is active by means of a bitmask."]
pub use self::rocblas_layer_mode_ as rocblas_layer_mode;
#[repr(u32)]
#[doc = " \\brief Indicates whether the layer is active by means of a bitmask."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_gemm_algo_ {
    rocblas_gemm_algo_standard = 0,
    rocblas_gemm_algo_solution_index = 1,
}
#[doc = " \\brief Indicates whether the layer is active by means of a bitmask."]
pub use self::rocblas_gemm_algo_ as rocblas_gemm_algo;
#[repr(u32)]
#[doc = " \\brief Which mathematical geam-like operation to perform for geam_ex"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_geam_ex_operation_ {
    rocblas_geam_ex_operation_min_plus = 0,
    rocblas_geam_ex_operation_plus_min = 1,
}
#[doc = " \\brief Which mathematical geam-like operation to perform for geam_ex"]
pub use self::rocblas_geam_ex_operation_ as rocblas_geam_ex_operation;
#[repr(u32)]
#[doc = " \\brief Controls the flags passed into GEMM algorithms invoked by the Tensile host."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_gemm_flags_ {
    #[doc = " \\brief Default empty flags"]
    rocblas_gemm_flags_none = 0,
    #[doc = " \\brief Before ROCm 6.0, ``rocblas_gemm_flags_pack_int8x4 = 0x1`` was in use. This has now been removed and is available for future use. */\n/*! \\brief Select the GEMM problem with the highest efficiency per compute unit used, which is useful for running multiple smaller problems\n simultaneously. This takes precedence over the performance metric set in ``rocblas_handle`` and currently only works for\n gemm_*_ex problems."]
    rocblas_gemm_flags_use_cu_efficiency = 2,
    #[doc = " \\brief Select an alternate implementation for the Instinct MI200 ``FP16`` HPA\n (High Precision Accumulate) GEMM kernel. This implementation uses the ``BF16`` matrix\n instructions with reduced accuracy in cases where computation cannot\n tolerate the ``FP16`` matrix instructions, flushing subnormal ``FP16``\n input/output data to zero. See the \"MI200 (gfx90a) Considerations\"\n section of the rocBLAS design notes documentation for more details."]
    rocblas_gemm_flags_fp16_alt_impl = 4,
    #[doc = " \\brief Select an alternate implementation for the Instinct MI200 ``FP16`` HPA\n (High Precision Accumulate) GEMM kernel. This implementation uses the ``BF16`` matrix\n instructions with reduced accuracy in cases where computation cannot\n tolerate the ``FP16`` matrix instructions, flushing subnormal ``FP16``\n input/output data to zero. See the \"MI200 (gfx90a) Considerations\"\n section of the rocBLAS design notes documentation for more details."]
    rocblas_gemm_flags_check_solution_index = 8,
    #[doc = " \\brief Select an alternate implementation for the Instinct MI200 ``FP16`` HPA\n (High Precision Accumulate) GEMM kernel. This implementation uses the ``BF16`` matrix\n instructions with reduced accuracy in cases where computation cannot\n tolerate the ``FP16`` matrix instructions, flushing subnormal ``FP16``\n input/output data to zero. See the \"MI200 (gfx90a) Considerations\"\n section of the rocBLAS design notes documentation for more details."]
    rocblas_gemm_flags_fp16_alt_impl_rnz = 16,
    #[doc = " \\brief Select an alternate implementation for the Instinct MI200 ``FP16`` HPA\n (High Precision Accumulate) GEMM kernel. This implementation uses the ``BF16`` matrix\n instructions with reduced accuracy in cases where computation cannot\n tolerate the ``FP16`` matrix instructions, flushing subnormal ``FP16``\n input/output data to zero. See the \"MI200 (gfx90a) Considerations\"\n section of the rocBLAS design notes documentation for more details."]
    rocblas_gemm_flags_stochastic_rounding = 32,
}
#[doc = " \\brief Controls the flags passed into GEMM algorithms invoked by the Tensile host."]
pub use self::rocblas_gemm_flags_ as rocblas_gemm_flags;
#[doc = " \\brief Union for representing scalar values"]
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
#[doc = " \\brief Union for representing scalar values"]
pub type rocblas_union_t = rocblas_union_u;
#[repr(u32)]
#[doc = " \\brief Numerical checking for verifying the Input and Output vector/matrix of the rocBLAS functions for a NaN, zero, infinity and denormal value"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_check_numerics_mode_ {
    rocblas_check_numerics_mode_no_check = 0,
    rocblas_check_numerics_mode_info = 1,
    rocblas_check_numerics_mode_warn = 2,
    rocblas_check_numerics_mode_fail = 4,
    rocblas_check_numerics_mode_only_nan_inf = 8,
}
#[doc = " \\brief Numerical checking for verifying the Input and Output vector/matrix of the rocBLAS functions for a NaN, zero, infinity and denormal value"]
pub use self::rocblas_check_numerics_mode_ as rocblas_check_numerics_mode;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocblas_math_mode_ {
    rocblas_default_math = 0,
    rocblas_xf32_xdl_math_op = 1,
}
pub use self::rocblas_math_mode_ as rocblas_math_mode;
