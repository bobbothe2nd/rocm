#[repr(u32)]
#[doc = " hipJitOption"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitOption {
    #[doc = "< CUDA Only Maximum registers may be used in a thread,\n< passed to compiler"]
    hipJitOptionMaxRegisters = 0,
    #[doc = "< CUDA Only Number of thread per block"]
    hipJitOptionThreadsPerBlock = 1,
    #[doc = "< CUDA Only Value for total wall clock time"]
    hipJitOptionWallTime = 2,
    #[doc = "< CUDA Only Pointer to the buffer with logged information"]
    hipJitOptionInfoLogBuffer = 3,
    #[doc = "< CUDA Only Size of the buffer in bytes for logged info"]
    hipJitOptionInfoLogBufferSizeBytes = 4,
    #[doc = "< CUDA Only Pointer to the buffer with logged error(s)"]
    hipJitOptionErrorLogBuffer = 5,
    #[doc = "< CUDA Only Size of the buffer in bytes for logged\n< error(s)"]
    hipJitOptionErrorLogBufferSizeBytes = 6,
    #[doc = "< Value of optimization level for generated codes, acceptable\n< options -O0, -O1, -O2, -O3"]
    hipJitOptionOptimizationLevel = 7,
    #[doc = "< CUDA Only The target context, which is the default"]
    hipJitOptionTargetFromContext = 8,
    #[doc = "< CUDA Only JIT target"]
    hipJitOptionTarget = 9,
    #[doc = "< CUDA Only Fallback strategy"]
    hipJitOptionFallbackStrategy = 10,
    #[doc = "< CUDA Only Generate debug information"]
    hipJitOptionGenerateDebugInfo = 11,
    #[doc = "< CUDA Only Generate log verbose"]
    hipJitOptionLogVerbose = 12,
    #[doc = "< CUDA Only Generate line number information"]
    hipJitOptionGenerateLineInfo = 13,
    #[doc = "< CUDA Only Set cache mode"]
    hipJitOptionCacheMode = 14,
    #[doc = "< @deprecated CUDA Only New SM3X option."]
    hipJitOptionSm3xOpt = 15,
    #[doc = "< CUDA Only Set fast compile"]
    hipJitOptionFastCompile = 16,
    #[doc = "< CUDA Only Array of device symbol names to be relocated to the\n< host"]
    hipJitOptionGlobalSymbolNames = 17,
    #[doc = "< CUDA Only Array of host addresses to be relocated to the\n< device"]
    hipJitOptionGlobalSymbolAddresses = 18,
    #[doc = "< CUDA Only Number of symbol count."]
    hipJitOptionGlobalSymbolCount = 19,
    #[doc = "< @deprecated CUDA Only Enable link-time optimization for device code"]
    hipJitOptionLto = 20,
    #[doc = "< @deprecated CUDA Only Set single-precision denormals."]
    hipJitOptionFtz = 21,
    #[doc = "< @deprecated CUDA Only Set single-precision floating-point division\n< and reciprocals"]
    hipJitOptionPrecDiv = 22,
    #[doc = "< @deprecated CUDA Only Set single-precision floating-point square root"]
    hipJitOptionPrecSqrt = 23,
    #[doc = "< @deprecated CUDA Only Enable floating-point multiplies and\n< adds/subtracts operations"]
    hipJitOptionFma = 24,
    #[doc = "< CUDA Only Generates Position Independent code"]
    hipJitOptionPositionIndependentCode = 25,
    #[doc = "< CUDA Only Hints to JIT compiler the minimum number of CTAs frin\n< kernel's grid to be mapped to SM"]
    hipJitOptionMinCTAPerSM = 26,
    #[doc = "< CUDA only Maximum number of threads in a thread block"]
    hipJitOptionMaxThreadsPerBlock = 27,
    #[doc = "< Cuda only Override Directive values"]
    hipJitOptionOverrideDirectiveValues = 28,
    #[doc = "< Number of options"]
    hipJitOptionNumOptions = 29,
    #[doc = "< Hip Only Linker options to be passed on to compiler"]
    hipJitOptionIRtoISAOptExt = 10000,
    #[doc = "< Hip Only Count of linker options to be passed on to compiler"]
    hipJitOptionIRtoISAOptCountExt = 10001,
}
#[repr(u32)]
#[doc = " hipJitInputType"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitInputType {
    #[doc = "< Cuda only Input cubin"]
    hipJitInputCubin = 0,
    #[doc = "< Cuda only Input PTX"]
    hipJitInputPtx = 1,
    #[doc = "< Cuda Only Input FAT Binary"]
    hipJitInputFatBinary = 2,
    #[doc = "< Cuda Only Host Object with embedded device code"]
    hipJitInputObject = 3,
    #[doc = "< Cuda Only Archive of Host Objects with embedded\n< device code"]
    hipJitInputLibrary = 4,
    #[doc = "< @deprecated Cuda only High Level intermediate\n< code for LTO"]
    hipJitInputNvvm = 5,
    #[doc = "< Count of Legacy Input Types"]
    hipJitNumLegacyInputTypes = 6,
    #[doc = "< HIP Only LLVM Bitcode or IR assembly"]
    hipJitInputLLVMBitcode = 100,
    #[doc = "< HIP Only LLVM Clang Bundled Code"]
    hipJitInputLLVMBundledBitcode = 101,
    #[doc = "< HIP Only LLVM Archive of Bundled Bitcode"]
    hipJitInputLLVMArchivesOfBundledBitcode = 102,
    #[doc = "< HIP Only SPIRV Code Object"]
    hipJitInputSpirv = 103,
    #[doc = "< Count of Input Types"]
    hipJitNumInputTypes = 10,
}
#[repr(u32)]
#[doc = " hipJitCacheMode"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitCacheMode {
    hipJitCacheOptionNone = 0,
    hipJitCacheOptionCG = 1,
    hipJitCacheOptionCA = 2,
}
#[repr(u32)]
#[doc = " hipJitFallback"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitFallback {
    hipJitPreferPTX = 0,
    hipJitPreferBinary = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipLibraryOption_e {
    hipLibraryHostUniversalFunctionAndDataTable = 0,
    hipLibraryBinaryIsPreserved = 1,
}
pub use self::hipLibraryOption_e as hipLibraryOption;
pub type wchar_t = ::core::ffi::c_int;
pub type _Float32 = f32;
pub type _Float64 = f64;
pub type _Float32x = f64;
pub type _Float64x = u128;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct div_t {
    pub quot: ::core::ffi::c_int,
    pub rem: ::core::ffi::c_int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ldiv_t {
    pub quot: ::core::ffi::c_long,
    pub rem: ::core::ffi::c_long,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct lldiv_t {
    pub quot: ::core::ffi::c_longlong,
    pub rem: ::core::ffi::c_longlong,
}
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
pub type u_char = __u_char;
pub type u_short = __u_short;
pub type u_int = __u_int;
pub type u_long = __u_long;
pub type quad_t = __quad_t;
pub type u_quad_t = __u_quad_t;
pub type fsid_t = __fsid_t;
pub type loff_t = __loff_t;
pub type ino_t = __ino_t;
pub type dev_t = __dev_t;
pub type gid_t = __gid_t;
pub type mode_t = __mode_t;
pub type nlink_t = __nlink_t;
pub type uid_t = __uid_t;
pub type off_t = __off_t;
pub type pid_t = __pid_t;
pub type id_t = __id_t;
pub type daddr_t = __daddr_t;
pub type caddr_t = __caddr_t;
pub type key_t = __key_t;
pub type clock_t = __clock_t;
pub type clockid_t = __clockid_t;
pub type time_t = __time_t;
pub type timer_t = __timer_t;
pub type ulong = ::core::ffi::c_ulong;
pub type ushort = ::core::ffi::c_ushort;
pub type uint = ::core::ffi::c_uint;
pub type u_int8_t = __uint8_t;
pub type u_int16_t = __uint16_t;
pub type u_int32_t = __uint32_t;
pub type u_int64_t = __uint64_t;
pub type register_t = ::core::ffi::c_long;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __sigset_t {
    pub __val: [::core::ffi::c_ulong; 16usize],
}
pub type sigset_t = __sigset_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct timeval {
    pub tv_sec: __time_t,
    pub tv_usec: __suseconds_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct timespec {
    pub tv_sec: __time_t,
    pub tv_nsec: __syscall_slong_t,
}
pub type suseconds_t = __suseconds_t;
pub type __fd_mask = ::core::ffi::c_long;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct fd_set {
    pub __fds_bits: [__fd_mask; 16usize],
}
pub type fd_mask = __fd_mask;
pub type blksize_t = __blksize_t;
pub type blkcnt_t = __blkcnt_t;
pub type fsblkcnt_t = __fsblkcnt_t;
pub type fsfilcnt_t = __fsfilcnt_t;
#[repr(C)]
#[derive(Copy, Clone)]
pub union __atomic_wide_counter {
    pub __value64: ::core::ffi::c_ulonglong,
    pub __value32: __atomic_wide_counter__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __atomic_wide_counter__bindgen_ty_1 {
    pub __low: ::core::ffi::c_uint,
    pub __high: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __pthread_internal_list {
    pub __prev: *mut __pthread_internal_list,
    pub __next: *mut __pthread_internal_list,
}
pub type __pthread_list_t = __pthread_internal_list;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __pthread_internal_slist {
    pub __next: *mut __pthread_internal_slist,
}
pub type __pthread_slist_t = __pthread_internal_slist;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __pthread_mutex_s {
    pub __lock: ::core::ffi::c_int,
    pub __count: ::core::ffi::c_uint,
    pub __owner: ::core::ffi::c_int,
    pub __nusers: ::core::ffi::c_uint,
    pub __kind: ::core::ffi::c_int,
    pub __spins: ::core::ffi::c_short,
    pub __elision: ::core::ffi::c_short,
    pub __list: __pthread_list_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __pthread_rwlock_arch_t {
    pub __readers: ::core::ffi::c_uint,
    pub __writers: ::core::ffi::c_uint,
    pub __wrphase_futex: ::core::ffi::c_uint,
    pub __writers_futex: ::core::ffi::c_uint,
    pub __pad3: ::core::ffi::c_uint,
    pub __pad4: ::core::ffi::c_uint,
    pub __cur_writer: ::core::ffi::c_int,
    pub __shared: ::core::ffi::c_int,
    pub __rwelision: ::core::ffi::c_schar,
    pub __pad1: [::core::ffi::c_uchar; 7usize],
    pub __pad2: ::core::ffi::c_ulong,
    pub __flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct __pthread_cond_s {
    pub __wseq: __atomic_wide_counter,
    pub __g1_start: __atomic_wide_counter,
    pub __g_refs: [::core::ffi::c_uint; 2usize],
    pub __g_size: [::core::ffi::c_uint; 2usize],
    pub __g1_orig_size: ::core::ffi::c_uint,
    pub __wrefs: ::core::ffi::c_uint,
    pub __g_signals: [::core::ffi::c_uint; 2usize],
}
pub type __tss_t = ::core::ffi::c_uint;
pub type __thrd_t = ::core::ffi::c_ulong;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __once_flag {
    pub __data: ::core::ffi::c_int,
}
pub type pthread_t = ::core::ffi::c_ulong;
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_mutexattr_t {
    pub __size: [::core::ffi::c_char; 4usize],
    pub __align: ::core::ffi::c_int,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_condattr_t {
    pub __size: [::core::ffi::c_char; 4usize],
    pub __align: ::core::ffi::c_int,
}
pub type pthread_key_t = ::core::ffi::c_uint;
pub type pthread_once_t = ::core::ffi::c_int;
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_attr_t {
    pub __size: [::core::ffi::c_char; 56usize],
    pub __align: ::core::ffi::c_long,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_mutex_t {
    pub __data: __pthread_mutex_s,
    pub __size: [::core::ffi::c_char; 40usize],
    pub __align: ::core::ffi::c_long,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_cond_t {
    pub __data: __pthread_cond_s,
    pub __size: [::core::ffi::c_char; 48usize],
    pub __align: ::core::ffi::c_longlong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_rwlock_t {
    pub __data: __pthread_rwlock_arch_t,
    pub __size: [::core::ffi::c_char; 56usize],
    pub __align: ::core::ffi::c_long,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_rwlockattr_t {
    pub __size: [::core::ffi::c_char; 8usize],
    pub __align: ::core::ffi::c_long,
}
pub type pthread_spinlock_t = ::core::ffi::c_int;
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_barrier_t {
    pub __size: [::core::ffi::c_char; 32usize],
    pub __align: ::core::ffi::c_long,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union pthread_barrierattr_t {
    pub __size: [::core::ffi::c_char; 4usize],
    pub __align: ::core::ffi::c_int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct random_data {
    pub fptr: *mut i32,
    pub rptr: *mut i32,
    pub state: *mut i32,
    pub rand_type: ::core::ffi::c_int,
    pub rand_deg: ::core::ffi::c_int,
    pub rand_sep: ::core::ffi::c_int,
    pub end_ptr: *mut i32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct drand48_data {
    pub __x: [::core::ffi::c_ushort; 3usize],
    pub __old_x: [::core::ffi::c_ushort; 3usize],
    pub __c: ::core::ffi::c_ushort,
    pub __init: ::core::ffi::c_ushort,
    pub __a: ::core::ffi::c_ulonglong,
}
pub type __compar_fn_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: *const ::core::ffi::c_void,
        arg2: *const ::core::ffi::c_void,
    ) -> ::core::ffi::c_int,
>;
#[repr(u32)]
#[doc = " @addtogroup GlobalDefs\n @{\n\n/\n/**\n hiprtc error code"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hiprtcResult {
    #[doc = "< Success"]
    HIPRTC_SUCCESS = 0,
    #[doc = "< Out of memory"]
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,
    #[doc = "< Failed to create program"]
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    #[doc = "< Invalid input"]
    HIPRTC_ERROR_INVALID_INPUT = 3,
    #[doc = "< Invalid program"]
    HIPRTC_ERROR_INVALID_PROGRAM = 4,
    #[doc = "< Invalid option"]
    HIPRTC_ERROR_INVALID_OPTION = 5,
    #[doc = "< Compilation error"]
    HIPRTC_ERROR_COMPILATION = 6,
    #[doc = "< Failed in builtin operation"]
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    #[doc = "< No name expression after compilation"]
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    #[doc = "< No lowered names before compilation"]
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    #[doc = "< Invalid name expression"]
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    #[doc = "< Internal error"]
    HIPRTC_ERROR_INTERNAL_ERROR = 11,
    #[doc = "< Error in linking"]
    HIPRTC_ERROR_LINKING = 100,
}
#[repr(C)]
#[derive(Debug)]
pub struct ihiprtcLinkState {
    _unused: [u8; 0],
}
#[doc = "  hiprtc link state\n"]
pub type hiprtcLinkState = *mut ihiprtcLinkState;
#[repr(C)]
#[derive(Debug)]
pub struct _hiprtcProgram {
    _unused: [u8; 0],
}
#[doc = "  hiprtc program\n"]
pub type hiprtcProgram = *mut _hiprtcProgram;
