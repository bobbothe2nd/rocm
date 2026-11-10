#![allow(clippy::unnecessary_cast, clippy::missing_safety_doc)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[allow(unused_macros)]
macro_rules! wrap_sys_res {
    (|$val:ident$(:$ty:ty)?| $expr:expr) => {{
        let mut $val$(: ::core::mem::MaybeUninit<$ty>)? = ::core::mem::MaybeUninit::uninit();
        Ok(try_err!($expr, $val.assume_init()))
    }};
}

#[allow(unused_macros)]
macro_rules! log_if_err {
    ($msg:literal: $expr:expr) => {{
        if let Err(err) = $expr {
            err.log($msg);
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

        impl $err {
            pub const fn wrap(err: $sys_err) -> Result<(), Self> {
                match err {
                    $sys_err::$success => Ok(()),
                    err => Err(unsafe { ::core::mem::transmute::<$sys_err, Self>(err) }),
                }
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

        impl $err {
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

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub use rocm_sys::is_amdhip64_present;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub mod hiprtc;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub use rocm_sys::is_hiprtc_present;

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub mod rocblas;

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub use rocm_sys::is_rocblas_present;

#[cfg(all(feature = "hipblas", any(feature = "dynamic-loading", hipblas)))]
pub use hipblas;

#[cfg(all(feature = "hipblas", any(feature = "dynamic-loading", hipblas)))]
pub use rocm_sys::is_hipblas_present;

#[cfg(all(feature = "hipblaslt", any(feature = "dynamic-loading", hipblaslt)))]
pub use hipblaslt;

#[cfg(all(feature = "hipblaslt", any(feature = "dynamic-loading", hipblaslt)))]
pub use rocm_sys::is_hipblaslt_present;

#[cfg(all(feature = "rocfft", any(feature = "dynamic-loading", rocfft)))]
pub use rocfft;

#[cfg(all(feature = "rocfft", any(feature = "dynamic-loading", rocfft)))]
pub use rocm_sys::is_rocfft_present;

#[cfg(all(feature = "hipfft", any(feature = "dynamic-loading", hipfft)))]
pub use hipfft;

#[cfg(all(feature = "hipfft", any(feature = "dynamic-loading", hipfft)))]
pub use rocm_sys::is_hipfft_present;

#[cfg(all(feature = "miopen", any(feature = "dynamic-loading", miopen)))]
pub use miopen;

#[cfg(all(feature = "miopen", any(feature = "dynamic-loading", miopen)))]
pub use rocm_sys::is_miopen_present;

#[cfg(any(feature = "dynamic-loading", all(hip, hiprtc)))]
rocm_include::doctest!(["hip", "hiprtc"] readme: "../README.md");
