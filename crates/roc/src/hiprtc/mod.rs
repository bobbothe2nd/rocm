#[allow(unused_macros)]
macro_rules! try_err {
    ($func:expr$(, $sucess:expr)?$(,)?) => {
        match $func {
            ::rocm_sys::hiprtc::hiprtcResult::HIPRTC_SUCCESS => { $($sucess)? }
            err => return Err(<$crate::hiprtc::HiprtcError as ::core::convert::From::<::rocm_sys::hiprtc::hiprtcResult>>::from(err)),
        }
    };
}

pub mod program;

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

err_ty!(type HiprtcError = hiprtcResult is !HIPRTC_SUCCESS from rocm_sys::hiprtc::hiprtcGetErrorString;);
