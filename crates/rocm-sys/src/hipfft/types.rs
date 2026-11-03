#[repr(i32)]
#[doc = " @brief Result/status/error codes"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipfftResult_t {
    #[doc = " hipFFT operation was successful"]
    HIPFFT_SUCCESS = 0,
    #[doc = " hipFFT was passed an invalid plan handle"]
    HIPFFT_INVALID_PLAN = 1,
    #[doc = " hipFFT failed to allocate GPU or CPU memory"]
    HIPFFT_ALLOC_FAILED = 2,
    #[doc = " No longer used"]
    HIPFFT_INVALID_TYPE = 3,
    #[doc = " User specified an invalid pointer or parameter"]
    HIPFFT_INVALID_VALUE = 4,
    #[doc = " Driver or internal hipFFT library error"]
    HIPFFT_INTERNAL_ERROR = 5,
    #[doc = " Failed to execute an FFT on the GPU"]
    HIPFFT_EXEC_FAILED = 6,
    #[doc = " hipFFT failed to initialize"]
    HIPFFT_SETUP_FAILED = 7,
    #[doc = " User specified an invalid transform size"]
    HIPFFT_INVALID_SIZE = 8,
    #[doc = " No longer used"]
    HIPFFT_UNALIGNED_DATA = 9,
    #[doc = " Missing parameters in call"]
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10,
    #[doc = " Execution of a plan was on different GPU than plan creation"]
    HIPFFT_INVALID_DEVICE = 11,
    #[doc = " Internal plan database error"]
    HIPFFT_PARSE_ERROR = 12,
    #[doc = " No workspace has been provided prior to plan execution"]
    HIPFFT_NO_WORKSPACE = 13,
    #[doc = " Function does not implement functionality for parameters given."]
    HIPFFT_NOT_IMPLEMENTED = 14,
    #[doc = " Operation is not supported for parameters given."]
    HIPFFT_NOT_SUPPORTED = 16,
}
#[doc = " @brief Result/status/error codes"]
pub use self::hipfftResult_t as hipfftResult;
#[repr(i32)]
#[doc = " @brief Transform type\n  @details This type is used to declare the Fourier transform type that will be executed."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipfftType_t {
    #[doc = " Real to complex (interleaved)"]
    HIPFFT_R2C = 42,
    #[doc = " Complex (interleaved) to real"]
    HIPFFT_C2R = 44,
    #[doc = " Complex to complex (interleaved)"]
    HIPFFT_C2C = 41,
    #[doc = " Double to double-complex (interleaved)"]
    HIPFFT_D2Z = 106,
    #[doc = " Double-complex (interleaved) to double"]
    HIPFFT_Z2D = 108,
    #[doc = " Double-complex to double-complex (interleaved)"]
    HIPFFT_Z2Z = 105,
}
#[doc = " @brief Transform type\n  @details This type is used to declare the Fourier transform type that will be executed."]
pub use self::hipfftType_t as hipfftType;
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipfftLibraryPropertyType_t {
    HIPFFT_MAJOR_VERSION = 0,
    HIPFFT_MINOR_VERSION = 1,
    HIPFFT_PATCH_LEVEL = 2,
}
pub use self::hipfftLibraryPropertyType_t as hipfftLibraryPropertyType;
#[repr(C)]
#[derive(Debug)]
pub struct hipfftHandle_t {
    _unused: [u8; 0],
}
pub type hipfftHandle = *mut hipfftHandle_t;
#[doc = " @brief Single-precision floating point complex type"]
pub type hipfftComplex = crate::hip::hipComplex;
#[doc = " @brief Double-precision floating point complex type"]
pub type hipfftDoubleComplex = crate::hip::hipDoubleComplex;
#[doc = " @brief Single-precision floating point type"]
pub type hipfftReal = f32;
#[doc = " @brief Double-precision floating point type"]
pub type hipfftDoubleReal = f64;
