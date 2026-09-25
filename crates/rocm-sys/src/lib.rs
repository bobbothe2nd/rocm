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
            $(#[since = $major_version:literal.$minor_version:literal,
                $version_string:literal])?
            $vis:vis fn $name:ident($($arg:ident: $arg_ty:ty),*$(,)?) $(-> $ret:ty)?;
        )*
    ) => {
        pub(crate) mod $lib {
            pub(crate) const LIBRARIES: &[&str] = {
                #[cfg(unix)]
                {
                    &[
                        concat!("lib", stringify!($lib), ".so"),
                        $(
                            concat!("lib", stringify!($lib), ".so.", stringify!($version)),
                        )*
                    ]
                }

                #[cfg(windows)]
                {
                    &[
                        concat!(stringify!($lib), ".dll"),
                        $(
                            concat!(stringify!($lib), "-", stringify!($version), ".dll"),
                        )*
                    ]
                }

                #[cfg(not(any(windows, unix)))]
                {
                    &[]
                }
            };

            pub(crate) fn is_rocm_lib_present() -> bool {
                let roots = [
                    std::env::var_os("ROCM_PATH"),
                    Some("/opt/rocm".into()),
                    Some("/usr/local/rocm".into()),
                ];

                for root in roots.into_iter().flatten() {
                    let lib_dir = std::path::Path::new(&root).join("lib");

                    for choice in LIBRARIES {
                        let path = lib_dir.join(choice);

                        if path.exists() {
                            return true;
                        }
                    }
                }

                false
            }
        }

        #[cfg(feature = "dynamic-loading")]
        static $lib: ::std::sync::LazyLock<libloading::Library> = {
            ::std::sync::LazyLock::new(|| {
                let roots = [
                    std::env::var_os("ROCM_PATH"),
                    Some("/opt/rocm".into()),
                    Some("/usr/local/rocm".into()),
                ];

                for root in roots.into_iter().flatten() {
                    let lib_dir = std::path::Path::new(&root).join("lib");

                    for choice in $lib::LIBRARIES {
                        let path = lib_dir.join(choice);

                        if let Ok(lib) = unsafe {
                            libloading::Library::new(&path)
                        } {
                            return lib;
                        }
                    }
                }

                panic!("Unable to dynamically load the {:?} shared library - searched for library names: {:?}. \
If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.", stringify!($lib), $lib::LIBRARIES);
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
            $vis unsafe fn $name(
                $($arg: $arg_ty),*
            ) $(-> $ret)? {
                type F = unsafe extern "C" fn(
                    $($arg_ty),*
                ) $(-> $ret)?;

                static SYMBOL: ::std::sync::OnceLock<F> = ::std::sync::OnceLock::new();

                let f = SYMBOL.get_or_init(|| unsafe {
                    $(
                        let major = *MAJOR;
                        let minor = *MINOR;

                        assert!(major >= $major_version && minor >= $minor_version, "{} requires ROCm version ^{major}.{minor}", stringify!($name));
                    )?

                    *$lib.get::<F>(concat!(stringify!($name), "\0")).unwrap()
                });

                unsafe {
                    f($($arg),*)
                }
            }

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
            $vis unsafe fn $name(
                $($arg: $arg_ty),*
            ) $(-> $ret)? {
                unsafe extern "C" {
                    #[link_name = stringify!($name)]
                    unsafe fn inner(
                        $($arg: $arg_ty),*
                    ) $(-> $ret)?;
                }

                unsafe {
                    inner($($arg),*)
                }
            }
        )*
    };
}

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;

#[cfg(all(feature = "hipblas", any(feature = "dynamic-loading", hipblas)))]
pub mod hipblas;

#[cfg(all(feature = "hipblaslt", any(feature = "dynamic-loading", hipblaslt)))]
pub mod hipblaslt;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub mod hiprtc;

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub mod rocblas;

#[cfg(all(feature = "rocfft", any(feature = "dynamic-loading", rocfft)))]
pub mod rocfft;

mod version;

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub fn is_amdhip64_present() -> bool {
    hip::amdhip64::is_rocm_lib_present()
}

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub fn is_hiprtc_present() -> bool {
    hiprtc::hiprtc::is_rocm_lib_present()
}
