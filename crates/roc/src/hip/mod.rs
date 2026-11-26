macro_rules! try_err {
    ($func:expr$(, $sucess:expr)?$(,)?) => {
        match $func {
            ::rocm_sys::hip::hipError_t::hipSuccess => { $($sucess)? }
            err => return Err(<$crate::hip::HipError as ::core::convert::From::<::rocm_sys::hip::hipError_t>>::from(err)),
        }
    };
}

mod callback;
pub mod device;
pub mod event;
pub mod gfx;
pub mod graph;
pub mod memory;
pub mod module;
pub mod sord;
pub mod stream;
pub mod vector;

use rocm_sys::hip::hipError_t;

#[repr(u32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HipError {
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    InvalidConfiguration = 9,
    InvalidPitchValue = 12,
    InvalidSymbol = 13,
    InvalidDevicePointer = 17,
    InvalidMemcpyDirection = 21,
    InsufficientDriver = 35,
    MissingConfiguration = 52,
    PriorLaunchFailure = 53,
    InvalidDeviceFunction = 98,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    ECCNotCorrectable = 214,
    UnsupportedLimit = 215,
    ContextAlreadyInUse = 216,
    PeerAccessUnsupported = 217,
    InvalidKernelFile = 218,
    InvalidGraphicsContext = 219,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystem = 304,
    InvalidHandle = 400,
    IllegalState = 401,
    NotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeOut = 702,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    SetOnActiveProcess = 708,
    ContextIsDestroyed = 709,
    Assert = 710,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    LaunchFailure = 719,
    CooperativeLaunchTooLarge = 720,
    NotSupported = 801,
    StreamCaptureUnsupported = 900,
    StreamCaptureInvalidated = 901,
    StreamCaptureMerge = 902,
    StreamCaptureUnmatched = 903,
    StreamCaptureUnjoined = 904,
    StreamCaptureIsolation = 905,
    StreamCaptureImplicit = 906,
    CapturedEvent = 907,
    StreamCaptureWrongThread = 908,
    GraphExecUpdateFailure = 910,
    InvalidChannelDescriptor = 911,
    InvalidTexture = 912,
    Unknown = 999,
    RuntimeMemory = 1052,
    RuntimeOther = 1053,
    Tbd = 1054,
}

impl HipError {
    #[inline(always)]
    pub fn get_last() -> Self {
        unsafe { rocm_sys::hip::hipGetLastError().into() }
    }
}

err_ty!(type HipError = hipError_t is !hipSuccess from rocm_sys::hip::hipGetErrorString;);

/// # Safety
///
/// Must be safe to be written to from a GPU:
/// - no `impl Drop`
/// - memcpy-safe
pub unsafe trait DevRepr: Copy + Send + Sync {}

unsafe impl<T: DevWrRepr> DevRepr for T {}

unsafe impl DevRepr for bool {}
unsafe impl DevRepr for char {}

#[cfg(feature = "lowp")]
unsafe impl DevRepr for lowp::f4 {}
#[cfg(feature = "lowp")]
unsafe impl DevRepr for lowp::bf6 {}
#[cfg(feature = "lowp")]
unsafe impl DevRepr for lowp::hf6 {}

pub unsafe trait DevWrRepr: DevRepr {}

unsafe impl DevWrRepr for f32 {}
unsafe impl DevWrRepr for f64 {}

unsafe impl DevWrRepr for i8 {}
unsafe impl DevWrRepr for u8 {}

unsafe impl DevWrRepr for i16 {}
unsafe impl DevWrRepr for u16 {}

unsafe impl DevWrRepr for i32 {}
unsafe impl DevWrRepr for u32 {}

unsafe impl DevWrRepr for i64 {}
unsafe impl DevWrRepr for u64 {}

unsafe impl DevWrRepr for i128 {}
unsafe impl DevWrRepr for u128 {}

#[cfg(feature = "half")]
unsafe impl DevWrRepr for half::bf16 {}
#[cfg(feature = "half")]
unsafe impl DevWrRepr for half::f16 {}

#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::bf6x4 {}
#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::hf6x4 {}
#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::f4x2 {}
#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::bf8 {}
#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::bf8uz {}
#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::hf8 {}
#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::hf8fn {}
#[cfg(feature = "lowp")]
unsafe impl DevWrRepr for lowp::hf8uz {}

use crate::hip::vector::{
    f32x1, f32x2, f32x3, f32x4, f64x1, f64x2, f64x3, f64x4, i8x1, i8x2, i8x3, i8x4, i16x1, i16x2,
    i16x3, i16x4, i32x1, i32x2, i32x3, i32x4, i64x1, i64x2, i64x3, i64x4, u8x1, u8x2, u8x3, u8x4,
    u16x1, u16x2, u16x3, u16x4, u32x1, u32x2, u32x3, u32x4, u64x1, u64x2, u64x3, u64x4,
};

unsafe impl DevWrRepr for f32x1 {}
unsafe impl DevWrRepr for f32x2 {}
unsafe impl DevWrRepr for f32x3 {}
unsafe impl DevWrRepr for f32x4 {}

unsafe impl DevWrRepr for f64x1 {}
unsafe impl DevWrRepr for f64x2 {}
unsafe impl DevWrRepr for f64x3 {}
unsafe impl DevWrRepr for f64x4 {}

unsafe impl DevWrRepr for i8x1 {}
unsafe impl DevWrRepr for i8x2 {}
unsafe impl DevWrRepr for i8x3 {}
unsafe impl DevWrRepr for i8x4 {}

unsafe impl DevWrRepr for u8x1 {}
unsafe impl DevWrRepr for u8x2 {}
unsafe impl DevWrRepr for u8x3 {}
unsafe impl DevWrRepr for u8x4 {}

unsafe impl DevWrRepr for i16x1 {}
unsafe impl DevWrRepr for i16x2 {}
unsafe impl DevWrRepr for i16x3 {}
unsafe impl DevWrRepr for i16x4 {}

unsafe impl DevWrRepr for u16x1 {}
unsafe impl DevWrRepr for u16x2 {}
unsafe impl DevWrRepr for u16x3 {}
unsafe impl DevWrRepr for u16x4 {}

unsafe impl DevWrRepr for i32x1 {}
unsafe impl DevWrRepr for i32x2 {}
unsafe impl DevWrRepr for i32x3 {}
unsafe impl DevWrRepr for i32x4 {}

unsafe impl DevWrRepr for u32x1 {}
unsafe impl DevWrRepr for u32x2 {}
unsafe impl DevWrRepr for u32x3 {}
unsafe impl DevWrRepr for u32x4 {}

unsafe impl DevWrRepr for i64x1 {}
unsafe impl DevWrRepr for i64x2 {}
unsafe impl DevWrRepr for i64x3 {}
unsafe impl DevWrRepr for i64x4 {}

unsafe impl DevWrRepr for u64x1 {}
unsafe impl DevWrRepr for u64x2 {}
unsafe impl DevWrRepr for u64x3 {}
unsafe impl DevWrRepr for u64x4 {}
