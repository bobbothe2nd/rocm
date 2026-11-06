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
        $lib:ident : $cfg_ver:ident;

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

                static SYMBOL: ::std::sync::LazyLock<F> = ::std::sync::LazyLock::new(|| unsafe {
                    $(
                        let ver = $crate::HIP_VER.expect("failed to get HIP version");
                        let min_ver = ($major_version, $minor_version);
                        assert!(ver >= min_ver, concat!(stringify!($name), " requires ", stringify!($lib), " version ^", stringify!($major_version), ".", stringify!($minor_version), ", found {}.{}"), ver.0, ver.1);
                    )?

                    *$lib.get::<F>(stringify!($name)).unwrap()
                });

                unsafe {
                    (SYMBOL)($($arg),*)
                }
            }

            $(
                #[$cfg_internal(feature = "internal")]
            )?
            #[cfg(all(
                not(feature = "dynamic-loading"),
                $(
                    $cfg_ver = $version_string
                )?
            ))]
            $($(#[$cfg_strong(feature = "deprecated")])?)?
            unsafe extern "C" {
                $(
                    #[doc = $docs]
                )*
                $(
                    #[deprecated = $note]
                )?
                #[inline]
                unsafe fn $name(
                    $($arg: $(*$mut)* $arg_ty),*
                ) -> $(*$ret_mut)* $ret;
            }

            $(
                #[$cfg_internal(feature = "internal")]
            )?
            #[cfg(any(
                feature = "dynamic-loading",
                all(
                    not(feature = "dynamic-loading"),
                    $(
                        $cfg_ver = $version_string
                    )?
                ),
            ))]
            $($(#[$cfg_strong(feature = "deprecated")])?)?
            #[cfg(test)]
            pastey::paste! {
                #[test]
                fn [<test_$name>]() {
                    #[cfg(feature = "dynamic-loading")]
                    {
                        type F = unsafe extern "C" fn(
                            $($(*$mut)* $arg_ty),*
                        ) -> $(*$ret_mut)* $ret;

                        let f: F = unsafe {
                            $(
                                let Some((major, minor)) = *$crate::HIP_VER else { return; };
                                if !(major >= $major_version && minor >= $minor_version) { return; }
                            )?

                            *$lib.get::<F>(stringify!($name)).unwrap()
                        };
                        let _ = ::core::hint::black_box(f);
                    }
                    #[cfg(not(feature = "dynamic-loading"))]
                    {
                        let _ = ::core::hint::black_box($name);
                    }
                }
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

pub mod internal;

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

#[cfg(feature = "dynamic-loading")]
fn get_lib_name_candidates(lib_name: &str) -> Vec<String> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let rocm_path = std::env::var("ROCM_PATH").ok();

    let stem = format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}");

    let mut out = Vec::with_capacity(2 + rocm_path.is_some() as usize);
    out.push(format!("{stem}"));
    out.push(format!("/opt/rocm/lib/{stem}"));
    if let Some(rocm_path) = &rocm_path {
        out.push(format!("{rocm_path}/lib/{stem}"));
    }
    out
}

#[cfg(feature = "dynamic-loading")]
static HIP_VER: std::sync::LazyLock<Option<(u32, u32)>> = std::sync::LazyLock::new(hip_runtime_version);

#[cfg(feature = "dynamic-loading")]
fn hip_runtime_version() -> Option<(u32, u32)> {
    fn parse_define(contents: &str, name: &str) -> Option<u32> {
        contents
            .lines()
            .find_map(|line| {
                let mut parts = line.split_whitespace();

                if parts.next()? == "#define" && parts.next()? == name {
                    parts.next()?.parse().ok()
                } else {
                    None
                }
            })
    }

    let rocm_path = std::env::var("ROCM_PATH").ok();

    let mut headers = Vec::with_capacity(1 + rocm_path.is_some() as usize);
    headers.push(format!("/opt/rocm/include/hip/hip_version.h"));
    if let Some(rocm_path) = &rocm_path {
        headers.push(format!("{rocm_path}/include/hip/hip_version.h"));
    }

    for header in headers {
        let Ok(contents) = std::fs::read_to_string(&header) else {
            continue;
        };

        let major = parse_define(&contents, "HIP_VERSION_MAJOR")?;
        let minor = parse_define(&contents, "HIP_VERSION_MINOR")?;

        return Some((major, minor));
    }

    None
}
