#![allow(
    non_upper_case_globals,
    non_snake_case,
    non_camel_case_types,
    unused_comparisons,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    clippy::ptr_offset_with_cast,
    clippy::manual_div_ceil,
    clippy::useless_transmute
)]
#![cfg_attr(not(feature = "dynamic-loading"), no_std)]

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;
#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub mod hiprtc;

#[cfg(all(feature = "hipblas", any(feature = "dynamic-loading", hipblas)))]
pub mod hipblas;
#[cfg(all(feature = "hipblaslt", any(feature = "dynamic-loading", hipblaslt)))]
pub mod hipblaslt;
#[cfg(all(feature = "hipfft", any(feature = "dynamic-loading", hipfft)))]
pub mod hipfft;

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub mod rocblas;
#[cfg(all(feature = "rocfft", any(feature = "dynamic-loading", rocfft)))]
pub mod rocfft;

#[cfg(all(feature = "miopen", any(feature = "dynamic-loading", miopen)))]
pub mod miopen;

#[cfg(all(feature = "hip", feature = "dynamic-loading"))]
pub fn is_amdhip64_present() -> bool {
    hip::amdhip64::is_rocm_lib_present()
}

#[cfg(all(feature = "hiprtc", feature = "dynamic-loading"))]
pub fn is_hiprtc_present() -> bool {
    hiprtc::hiprtc::is_rocm_lib_present()
}

#[cfg(all(feature = "rocblas", feature = "dynamic-loading"))]
pub fn is_rocblas_present() -> bool {
    rocblas::rocblas::is_rocm_lib_present()
}

#[cfg(all(feature = "hipblas", feature = "dynamic-loading"))]
pub fn is_hipblas_present() -> bool {
    hipblas::hipblas::is_rocm_lib_present()
}

#[cfg(all(feature = "hipblaslt", feature = "dynamic-loading"))]
pub fn is_hipblaslt_present() -> bool {
    hipblaslt::hipblaslt::is_rocm_lib_present()
}

#[cfg(all(feature = "hipfft", feature = "dynamic-loading"))]
pub fn is_hipfft_present() -> bool {
    hipfft::hipfft::is_rocm_lib_present()
}

#[cfg(all(feature = "rocfft", feature = "dynamic-loading"))]
pub fn is_rocfft_present() -> bool {
    rocfft::rocfft::is_rocm_lib_present()
}

#[cfg(all(feature = "rocfft", feature = "dynamic-loading"))]
pub fn is_miopen_present() -> bool {
    miopen::MIOpen::is_rocm_lib_present()
}

#[cfg(any(feature = "dynamic-loading", hip))]
rocm_include::doctest!(["hip"] readme: "../README.md");
