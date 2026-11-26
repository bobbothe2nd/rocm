#![allow(clippy::unnecessary_cast, clippy::missing_safety_doc)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[allow(unused_macros)]
macro_rules! wrap_sys_res {
    (|$val:ident$(:$ty:ty)?| $expr:expr) => {{
        let mut $val$(: ::core::mem::MaybeUninit<$ty>)? = ::core::mem::MaybeUninit::uninit();
        let res = $crate::aux::MapToErr::map_to_err($expr);
        res.map(|_| $val.assume_init())
    }};
}

#[allow(unused_macros)]
macro_rules! log_if_err {
    ($msg:literal: $expr:expr) => {{
        if let Err(err) = $crate::aux::MapToErr::map_to_err($expr) {
            err.log($msg);
        }
    }};
}

#[allow(unused_macros)]
macro_rules! log_lock_if_err {
    ($msg:literal: $expr:expr) => {{
        if let Err(err) = $expr {
            err.log($msg);
            loop {}
        }
    }};
}

#[allow(unused_macros)]
macro_rules! sync_ret_if_err {
    ($stream:expr, $expr:expr) => {{
        if let Err(err) = $expr {
            log_lock_if_err!("failed to synchronize stream": ($stream).sync());
            return Err(err.into());
        }
    }};
}

#[allow(unused_macros)]
macro_rules! err_ty {
    (type $err:ident = $sys_err:ident is !$success:ident from $err_cvt:path;) => {
        impl ::core::fmt::Display for $err {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let cstr_ptr = unsafe { $err_cvt((*self).into()) };
                let cstr = unsafe { ::core::ffi::CStr::from_ptr(cstr_ptr) };
                let formatted = cstr.to_str().map_err(|_| ::core::fmt::Error)?;
                f.write_str(formatted)
            }
        }

        impl ::core::error::Error for $err {}

        impl ::core::convert::From<$sys_err> for $err {
            fn from(value: $sys_err) -> Self {
                match value {
                    $sys_err::$success => panic!("converting $success to $err"),
                    err => unsafe { ::core::mem::transmute::<$sys_err, Self>(err) },
                }
            }
        }

        impl ::core::convert::From<$err> for $sys_err {
            fn from(value: $err) -> Self {
                unsafe { ::core::mem::transmute::<$err, Self>(value) }
            }
        }

        impl $crate::aux::MapToErr<$err> for $sys_err {
            fn map_to_err(self) -> Result<(), $err> {
                match self {
                    Self::$success => Ok(()),
                    err => Err(err.into()),
                }
            }
        }

        impl $err {
            pub const fn wrap(err: $sys_err) -> Result<(), Self> {
                match err {
                    $sys_err::$success => Ok(()),
                    err => Err(unsafe { ::core::mem::transmute::<$sys_err, Self>(err) }),
                }
            }

            pub fn log(self, msg: &::core::primitive::str) {
                let _ = msg;
                #[cfg(feature = "log")]
                {
                    ::log::error!("{msg}: {self}");
                }
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! cvt_hip_err {
    (
        $err:ident {
            $(
                $self_err:ident => [$($hip_err:ident),+ $(,)?]
            ),+$(,)?
        }
        reverse {
            $(
                $reverse_err:ident => $reverse_hip_err:ident
            ),+$(,)?
        }
        fallback $fallback:ident
    ) => {
        impl From<$crate::hip::HipError> for $err {
            fn from(err: $crate::hip::HipError) -> Self {
                match err {
                    $($($crate::hip::HipError::$hip_err)|+ => Self::$self_err,)+
                    _ => Self::$fallback,
                }
            }
        }
        impl From<$err> for $crate::hip::HipError {
            fn from(err: $err) -> Self {
                match err {
                    $($err::$reverse_err => $crate::hip::HipError::$reverse_hip_err,)+
                }
            }
        }
    };
}

mod auxiliary;
pub use auxiliary::public as aux;

#[cfg(feature = "blas")]
pub mod blas;

#[cfg(feature = "fft")]
pub mod fft;

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub mod hiprtc;

#[cfg(all(feature = "hipblas", any(feature = "dynamic-loading", hipblas)))]
pub mod hipblas;

#[cfg(all(feature = "hipblaslt", any(feature = "dynamic-loading", hipblaslt)))]
pub use hipblaslt;

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub mod rocblas;

#[cfg(all(feature = "rocfft", any(feature = "dynamic-loading", rocfft)))]
pub use rocfft;

#[cfg(all(feature = "hipfft", any(feature = "dynamic-loading", hipfft)))]
pub use hipfft;

#[cfg(all(feature = "miopen", any(feature = "dynamic-loading", miopen)))]
pub use miopen;

#[cfg(any(feature = "dynamic-loading", all(hip, hiprtc)))]
rocm_include::doctest!(["hip", "hiprtc"] readme: "../README.md");

#[cfg(not(any(windows, unix)))]
compile_error!("unsupported target, `roc` requires either Windows or Unix");
