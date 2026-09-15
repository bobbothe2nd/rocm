macro_rules! try_err {
    ($func:expr$(, $sucess:expr)?$(,)?) => {
        match $func {
            ::rocm_sys::hip::hipError_t::hipSuccess => { $($sucess)? }
            err => return Err(<$crate::hip::HipError as ::core::convert::From::<::rocm_sys::hip::hipError_t>>::from(err)),
        }
    };
}

pub mod device;

pub mod stream;

use core::{error::Error, fmt::{self, Display, Formatter, Result}};

use rocm_sys::hip::hipError_t;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HipError {
    HipErrorInvalidValue = 1,
    HipErrorOutOfMemory = 2,
    HipErrorNotInitialized = 3,
    HipErrorDeinitialized = 4,
    HipErrorProfilerDisabled = 5,
    HipErrorProfilerNotInitialized = 6,
    HipErrorProfilerAlreadyStarted = 7,
    HipErrorProfilerAlreadyStopped = 8,
    HipErrorInvalidConfiguration = 9,
    HipErrorInvalidPitchValue = 12,
    HipErrorInvalidSymbol = 13,
    HipErrorInvalidDevicePointer = 17,
    HipErrorInvalidMemcpyDirection = 21,
    HipErrorInsufficientDriver = 35,
    HipErrorMissingConfiguration = 52,
    HipErrorPriorLaunchFailure = 53,
    HipErrorInvalidDeviceFunction = 98,
    HipErrorNoDevice = 100,
    HipErrorInvalidDevice = 101,
    HipErrorInvalidImage = 200,
    HipErrorInvalidContext = 201,
    HipErrorContextAlreadyCurrent = 202,
    HipErrorMapFailed = 205,
    HipErrorUnmapFailed = 206,
    HipErrorArrayIsMapped = 207,
    HipErrorAlreadyMapped = 208,
    HipErrorNoBinaryForGpu = 209,
    HipErrorAlreadyAcquired = 210,
    HipErrorNotMapped = 211,
    HipErrorNotMappedAsArray = 212,
    HipErrorNotMappedAsPointer = 213,
    HipErrorECCNotCorrectable = 214,
    HipErrorUnsupportedLimit = 215,
    HipErrorContextAlreadyInUse = 216,
    HipErrorPeerAccessUnsupported = 217,
    HipErrorInvalidKernelFile = 218,
    HipErrorInvalidGraphicsContext = 219,
    HipErrorInvalidSource = 300,
    HipErrorFileNotFound = 301,
    HipErrorSharedObjectSymbolNotFound = 302,
    HipErrorSharedObjectInitFailed = 303,
    HipErrorOperatingSystem = 304,
    HipErrorInvalidHandle = 400,
    HipErrorIllegalState = 401,
    HipErrorNotFound = 500,
    HipErrorNotReady = 600,
    HipErrorIllegalAddress = 700,
    HipErrorLaunchOutOfResources = 701,
    HipErrorLaunchTimeOut = 702,
    HipErrorPeerAccessAlreadyEnabled = 704,
    HipErrorPeerAccessNotEnabled = 705,
    HipErrorSetOnActiveProcess = 708,
    HipErrorContextIsDestroyed = 709,
    HipErrorAssert = 710,
    HipErrorHostMemoryAlreadyRegistered = 712,
    HipErrorHostMemoryNotRegistered = 713,
    HipErrorLaunchFailure = 719,
    HipErrorCooperativeLaunchTooLarge = 720,
    HipErrorNotSupported = 801,
    HipErrorStreamCaptureUnsupported = 900,
    HipErrorStreamCaptureInvalidated = 901,
    HipErrorStreamCaptureMerge = 902,
    HipErrorStreamCaptureUnmatched = 903,
    HipErrorStreamCaptureUnjoined = 904,
    HipErrorStreamCaptureIsolation = 905,
    HipErrorStreamCaptureImplicit = 906,
    HipErrorCapturedEvent = 907,
    HipErrorStreamCaptureWrongThread = 908,
    HipErrorGraphExecUpdateFailure = 910,
    HipErrorInvalidChannelDescriptor = 911,
    HipErrorInvalidTexture = 912,
    HipErrorUnknown = 999,
    HipErrorRuntimeMemory = 1052,
    HipErrorRuntimeOther = 1053,
    HipErrorTbd = 1054,
}

impl Display for HipError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let cstr_ptr = unsafe {
            rocm_sys::hip::hipGetErrorString(core::mem::transmute_copy::<Self, hipError_t>(self))
        };
        let cstr = unsafe { core::ffi::CStr::from_ptr(cstr_ptr) };
        let formatted = cstr.to_str().map_err(|_| fmt::Error)?;

        write!(f, "{}", formatted)
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
