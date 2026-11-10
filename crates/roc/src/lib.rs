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
