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

#[allow(unused_macros)]
macro_rules! link {
    (
        $lib:ident : [$($version:literal),*$(,)?] : $cfg_ver:ident;

        $(
            $(
                #[internal($cfg_internal:ident)]
            )?
            $(
                #[doc = $docs:literal]
            )*
            $(#[deprecated = $note:literal$(, $cfg_strong:ident)?])?
            $(#[since = $major_version:literal.$minor_version:literal,$version_string:literal])?
            $(*$ret_mut:ident)*$ret:ident $name:ident($($(*$mut:ident)*$arg_ty:ident $arg:ident),*$(,)?);
        )*
    ) => {
        pub(crate) mod $lib {
            #[cfg(feature = "dynamic-loading")]
            pub(crate) fn is_rocm_lib_present() -> bool {
                for path in $crate::get_lib_name_candidates(stringify!($lib)) {
                    if matches!(::std::fs::exists(path), Ok(true)) {
                        return true;
                    }
                }

                false
            }
        }

        #[cfg(feature = "dynamic-loading")]
        static $lib: ::std::sync::LazyLock<libloading::Library> = {
            ::std::sync::LazyLock::new(|| {
                let paths = $crate::get_lib_name_candidates(stringify!($lib));

                for path in &paths {
                    if let Ok(lib) = unsafe {
                        libloading::Library::new(path)
                    } {
                        return lib;
                    }
                }

                panic!(concat!("Unable to dynamically load the \"", stringify!($lib), "\" shared library - searched for library names: {:?}. \
If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue."), paths);
            })
        };

        $(
            $(
                #[$cfg_internal(feature = "internal")]
            )?
            $(
                #[doc = $docs]
            )*
            $(
                $(#[$cfg_strong(feature = "deprecated")])?
                #[deprecated = $note]
            )?
            #[cfg(feature = "dynamic-loading")]
            pub unsafe extern "C" fn $name(
                $($arg: $(*$mut)* $arg_ty),*
            ) -> $(*$ret_mut)* $ret {
                type F = unsafe extern "C" fn(
                    $($(*$mut)* $arg_ty),*
                ) -> $(*$ret_mut)* $ret;

                static SYMBOL: ::std::sync::OnceLock<F> = ::std::sync::OnceLock::new();

                let f = SYMBOL.get_or_init(|| unsafe {
                    $(
                        let major = *MAJOR;
                        let minor = *MINOR;

                        assert!(major >= $major_version && minor >= $minor_version, concat!(stringify!($name), " requires ", stringify!($library), " version ^{}.{}"), major, minor);
                    )?

                    *$lib.get::<F>(concat!(stringify!($name), "\0")).unwrap()
                });

                unsafe {
                    f($($arg),*)
                }
            }

            unsafe extern "C" {
                $(
                    #[$cfg_internal(feature = "internal")]
                )?
                $(
                    #[doc = $docs]
                )*
                $(
                    $(#[$cfg_strong(feature = "deprecated")])?
                    #[deprecated = $note]
                )?
                #[inline]
                #[cfg(all(
                    not(feature = "dynamic-loading"),
                    $(
                        $cfg_ver = $version_string
                    )?
                ))]
                unsafe fn $name(
                    $($arg: $(*$mut)* $arg_ty),*
                ) -> $(*$ret_mut)* $ret;
            }
        )*
    };
}

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

pub mod version;

pub mod internal;

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub fn is_amdhip64_present() -> bool {
    hip::amdhip64::is_rocm_lib_present()
}

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub fn is_hiprtc_present() -> bool {
    hiprtc::hiprtc::is_rocm_lib_present()
}

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub fn is_rocblas_present() -> bool {
    rocblas::rocblas::is_rocm_lib_present()
}

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub fn is_hipblas_present() -> bool {
    hipblas::hipblas::is_rocm_lib_present()
}

#[cfg(feature = "dynamic-loading")]
fn get_lib_name_candidates(lib_name: &str) -> Vec<String> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
    use crate::version::{ROCM_MAJOR, ROCM_MINOR};

    let rocm_path = std::env::var("ROCM_PATH").ok();

    let major = *ROCM_MAJOR;
    let minor = *ROCM_MINOR;

    let stems = [
        format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.{major}"),
        format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.{major}.{minor}"),
    ];

    let mut out = Vec::with_capacity(stems.len() * (2 + rocm_path.is_some() as usize));
    for stem in &stems {
        out.push(stem.clone());
        out.push(format!("/opt/rocm/lib/{stem}"));
        if let Some(rocm_path) = &rocm_path {
            out.push(format!("{rocm_path}/lib/{stem}"));
        }
    }
    out
}
