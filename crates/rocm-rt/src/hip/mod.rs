macro_rules! try_err {
    ($func:expr$(, $sucess:expr)?$(,)?) => {
        match $func {
            ::rocm_sys::hip::hipError_t::hipSuccess => { $($sucess)? }
            err => return Err(<$crate::hip::HipError as ::core::convert::From::<::rocm_sys::hip::hipError_t>>::from(err)),
        }
    };
}

pub mod device;
pub mod memory;
pub mod stream;
pub mod graph;

#[cfg(all(
    feature = "alloc",
    feature = "hiprtc",
    any(feature = "dynamic-loading", hiprtc)
))]
pub mod module;

use core::{
    error::Error,
    fmt::{self, Display, Formatter, Result},
};

use rocm_sys::hip::hipError_t;

#[repr(u32)]
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

impl Display for HipError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let cstr_ptr = unsafe {
            rocm_sys::hip::hipGetErrorString(core::mem::transmute_copy::<Self, hipError_t>(self))
        };
        let cstr = unsafe { core::ffi::CStr::from_ptr(cstr_ptr) };
        let formatted = cstr.to_str().map_err(|_| fmt::Error)?;
        f.write_str(formatted)
    }
}

impl Error for HipError {}

impl From<hipError_t> for HipError {
    fn from(value: hipError_t) -> Self {
        match value {
            hipError_t::hipSuccess => panic!("converting hipSuccess to HipError"),
            err => unsafe { core::mem::transmute::<hipError_t, Self>(err) },
        }
    }
}

impl From<HipError> for hipError_t {
    fn from(value: HipError) -> Self {
        unsafe { core::mem::transmute::<HipError, Self>(value) }
    }
}
