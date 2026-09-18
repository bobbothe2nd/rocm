macro_rules! try_err {
    ($func:expr$(, $sucess:expr)?$(,)?) => {
        match $func {
            ::rocm_sys::hiprtc::hiprtcResult::HIPRTC_SUCCESS => { $($sucess)? }
            err => return Err(<$crate::hiprtc::HiprtcError as ::core::convert::From::<::rocm_sys::hiprtc::hiprtcResult>>::from(err)),
        }
    };
}

#[cfg(feature = "alloc")]
pub mod program;

use core::{
    error::Error,
    fmt::{self, Display, Formatter, Result},
};

use rocm_sys::hiprtc::hiprtcResult;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HiprtcError {
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    InvalidOption = 5,
    Compilation = 6,
    BuiltinOperationFailure = 7,
    NoNameExpressionsAfterCompilation = 8,
    NoLoweredNamesBeforeCompilation = 9,
    NameExpressionNotValid = 10,
    InternalError = 11,
    Linking = 100,
}

impl Display for HiprtcError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let cstr_ptr = unsafe {
            rocm_sys::hiprtc::hiprtcGetErrorString(core::mem::transmute_copy::<Self, hiprtcResult>(
                self,
            ))
        };
        let cstr = unsafe { core::ffi::CStr::from_ptr(cstr_ptr) };
        let formatted = cstr.to_str().map_err(|_| fmt::Error)?;
        f.write_str(formatted)
    }
}

impl Error for HiprtcError {}

impl From<hiprtcResult> for HiprtcError {
    fn from(value: hiprtcResult) -> Self {
        match value {
            hiprtcResult::HIPRTC_SUCCESS => panic!("converting HIPRTC_SUCCESS to HiprtcError"),
            err => unsafe { core::mem::transmute::<hiprtcResult, Self>(err) },
        }
    }
}

impl From<HiprtcError> for hiprtcResult {
    fn from(value: HiprtcError) -> Self {
        unsafe { core::mem::transmute::<HiprtcError, Self>(value) }
    }
}
