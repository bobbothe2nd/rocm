#![allow(non_upper_case_globals, non_snake_case, non_camel_case_types, unused_comparisons)]
#![cfg_attr(not(feature = "dynamic-loading"), no_std)]

#[allow(unused_macros)]
macro_rules! link {
    (
        $lib:ident: [$($version:literal),*$(,)?];

        $(
            $(
                #[doc = $docs:literal]
            )*
            $(#[since = $major_version:literal.$minor_version:literal])?
            $vis:vis fn $name:ident($($arg:ident: $arg_ty:ty),*$(,)?) $(-> $ret:ty)?;
        )*
    ) => {
        #[cfg(feature = "dynamic-loading")]
        static $lib: ::std::sync::LazyLock<libloading::Library> = {
            const LIBRARIES: &[&str] = {
                #[cfg(target_os = "linux")]
                {
                    &[
                        concat!("lib", stringify!($lib), ".so"),
                        $(
                            concat!("lib", stringify!($lib), ".so.", stringify!($version)),
                        )*
                    ]
                }

                #[cfg(target_os = "windows")]
                {
                    &[
                        concat!(stringify!($lib), ".dll"),
                        $(
                            concat!(stringify!($lib), "-", stringify!($version), ".dll"),
                        )*
                    ]
                }

                #[cfg(not(any(target_os = "windows", target_os = "linux")))]
                {
                    &[]
                }
            };

            ::std::sync::LazyLock::new(|| {
                let roots = [
                    std::env::var_os("ROCM_PATH"),
                    Some("/opt/rocm".into()),
                    Some("/usr/local/rocm".into()),
                ];

                for root in roots.into_iter().flatten() {
                    let lib_dir = std::path::Path::new(&root).join("lib");

                    for choice in LIBRARIES {
                        let path = lib_dir.join(choice);

                        if let Ok(lib) = unsafe {
                            libloading::Library::new(&path)
                        } {
                            return lib;
                        }
                    }
                }

                panic!("Unable to dynamically load the {:?} shared library - searched for library names: {:?}. \
If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.", stringify!($lib), LIBRARIES);
            })
        };

        $(
            $(
                #[doc = $docs]
            )*
            #[inline]
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

            #[inline]
            #[cfg(not(feature = "dynamic-loading"))]
            $vis unsafe fn $name(
                $($arg: $arg_ty),*
            ) $(-> $ret)? {
                extern "C" {
                    $(
                        #[doc = $docs]
                    )*
                    #[link_name = stringify!($name)]
                    unsafe fn inner(
                        $($arg: $arg_ty),*
                    ) $(-> $ret)?;
                }

                $(
                    const {
                        assert!(MAJOR >= $major_version && MINOR >= $minor_version, concat!(
                            stringify!($name),
                            " requires ROCm version ^",
                            stringify!(MAJOR),
                            ".",
                            stringify!(MINOR)
                        ));
                    }
                )?

                inner($($arg),*)
            }
        )*
    };
}

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub mod hiprtc;

#[cfg(all(feature = "rocblas", any(feature = "dynamic-loading", rocblas)))]
pub mod rocblas;

mod version;
