#[deprecated(since = "0.6.1", note = "use rocm_sys::hip::hipJitOption instead")]
pub type hipJitOption = crate::hip::hipJitOption;
#[deprecated(since = "0.6.1", note = "use rocm_sys::hip::hipJitInputType instead")]
pub type hipJitInputType = crate::hip::hipJitInputType;
#[deprecated(since = "0.6.1", note = "use rocm_sys::hip::hipJitCacheMode instead")]
pub type hipJitCacheMode = crate::hip::hipJitCacheMode;
#[deprecated(since = "0.6.1", note = "use rocm_sys::hip::hipJitFallback instead")]
pub type hipJitFallback = crate::hip::hipJitFallback;
#[deprecated(since = "0.6.1", note = "use rocm_sys::hip::hipLibraryOption instead")]
pub type hipLibraryOption = crate::hip::hipLibraryOption;
#[deprecated(
    since = "0.6.1",
    note = "use rocm_sys::hip::hipLibraryOption_e instead"
)]
pub type hipLibraryOption_e = crate::hip::hipLibraryOption_e;
#[repr(u32)]
/// hiprtc error code
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hiprtcResult {
    /// Success
    HIPRTC_SUCCESS = 0,
    /// Out of memory
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,
    /// Failed to create program
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    /// Invalid input
    HIPRTC_ERROR_INVALID_INPUT = 3,
    /// Invalid program
    HIPRTC_ERROR_INVALID_PROGRAM = 4,
    /// Invalid option
    HIPRTC_ERROR_INVALID_OPTION = 5,
    /// Compilation error
    HIPRTC_ERROR_COMPILATION = 6,
    /// Failed in builtin operation
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    /// No name expression after compilation
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    /// No lowered names before compilation
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    /// Invalid name expression
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    /// Internal error
    HIPRTC_ERROR_INTERNAL_ERROR = 11,
    /// Error in linking
    HIPRTC_ERROR_LINKING = 100,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihiprtcLinkState {
    _unused: [u8; 0],
}
/// hiprtc link state
///
///
pub type hiprtcLinkState = *mut ihiprtcLinkState;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct _hiprtcProgram {
    _unused: [u8; 0],
}
/// hiprtc program
///
///
pub type hiprtcProgram = *mut _hiprtcProgram;
