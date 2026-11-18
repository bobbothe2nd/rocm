impl hipfftResult_t {
    /// hipFFT operation was successful
    pub const HIPFFT_SUCCESS: hipfftResult_t = hipfftResult_t(0);
    /// hipFFT was passed an invalid plan handle
    pub const HIPFFT_INVALID_PLAN: hipfftResult_t = hipfftResult_t(1);
    /// hipFFT failed to allocate GPU or CPU memory
    pub const HIPFFT_ALLOC_FAILED: hipfftResult_t = hipfftResult_t(2);
    /// No longer used
    pub const HIPFFT_INVALID_TYPE: hipfftResult_t = hipfftResult_t(3);
    /// User specified an invalid pointer or parameter
    pub const HIPFFT_INVALID_VALUE: hipfftResult_t = hipfftResult_t(4);
    /// Driver or internal hipFFT library error
    pub const HIPFFT_INTERNAL_ERROR: hipfftResult_t = hipfftResult_t(5);
    /// Failed to execute an FFT on the GPU
    pub const HIPFFT_EXEC_FAILED: hipfftResult_t = hipfftResult_t(6);
    /// hipFFT failed to initialize
    pub const HIPFFT_SETUP_FAILED: hipfftResult_t = hipfftResult_t(7);
    /// User specified an invalid transform size
    pub const HIPFFT_INVALID_SIZE: hipfftResult_t = hipfftResult_t(8);
    /// No longer used
    pub const HIPFFT_UNALIGNED_DATA: hipfftResult_t = hipfftResult_t(9);
    /// Missing parameters in call
    pub const HIPFFT_INCOMPLETE_PARAMETER_LIST: hipfftResult_t = hipfftResult_t(10);
    /// Execution of a plan was on different GPU than plan creation
    pub const HIPFFT_INVALID_DEVICE: hipfftResult_t = hipfftResult_t(11);
    /// Internal plan database error
    pub const HIPFFT_PARSE_ERROR: hipfftResult_t = hipfftResult_t(12);
    /// No workspace has been provided prior to plan execution
    pub const HIPFFT_NO_WORKSPACE: hipfftResult_t = hipfftResult_t(13);
    /// Function does not implement functionality for parameters given.
    pub const HIPFFT_NOT_IMPLEMENTED: hipfftResult_t = hipfftResult_t(14);
    /// Operation is not supported for parameters given.
    pub const HIPFFT_NOT_SUPPORTED: hipfftResult_t = hipfftResult_t(16);
}
#[repr(transparent)]
/// Result/status/error codes
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipfftResult_t(pub ::core::ffi::c_int);
/// Result/status/error codes
pub use self::hipfftResult_t as hipfftResult;
impl hipfftType_t {
    /// Real to complex (interleaved)
    pub const HIPFFT_R2C: hipfftType_t = hipfftType_t(42);
    /// Complex (interleaved) to real
    pub const HIPFFT_C2R: hipfftType_t = hipfftType_t(44);
    /// Complex to complex (interleaved)
    pub const HIPFFT_C2C: hipfftType_t = hipfftType_t(41);
    /// Double to double-complex (interleaved)
    pub const HIPFFT_D2Z: hipfftType_t = hipfftType_t(106);
    /// Double-complex (interleaved) to double
    pub const HIPFFT_Z2D: hipfftType_t = hipfftType_t(108);
    /// Double-complex to double-complex (interleaved)
    pub const HIPFFT_Z2Z: hipfftType_t = hipfftType_t(105);
}
#[repr(transparent)]
/// Transform type
///
/// This type is used to declare the Fourier transform type that will be executed.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipfftType_t(pub ::core::ffi::c_int);
/// Transform type
///
/// This type is used to declare the Fourier transform type that will be executed.
pub use self::hipfftType_t as hipfftType;
impl hipfftLibraryPropertyType_t {
    pub const HIPFFT_MAJOR_VERSION: hipfftLibraryPropertyType_t = hipfftLibraryPropertyType_t(0);
    pub const HIPFFT_MINOR_VERSION: hipfftLibraryPropertyType_t = hipfftLibraryPropertyType_t(1);
    pub const HIPFFT_PATCH_LEVEL: hipfftLibraryPropertyType_t = hipfftLibraryPropertyType_t(2);
}
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipfftLibraryPropertyType_t(pub ::core::ffi::c_int);
pub use self::hipfftLibraryPropertyType_t as hipfftLibraryPropertyType;
#[repr(C)]
#[derive(Debug)]
pub struct hipfftHandle_t {
    _unused: [u8; 0],
}
pub type hipfftHandle = *mut hipfftHandle_t;
/// Single-precision floating point complex type
pub type hipfftComplex = crate::hip::hipComplex;
/// Double-precision floating point complex type
pub type hipfftDoubleComplex = crate::hip::hipDoubleComplex;
/// Single-precision floating point type
pub type hipfftReal = f32;
/// Double-precision floating point type
pub type hipfftDoubleReal = f64;
