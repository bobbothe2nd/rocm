/// Links functions at a given directory using dynamic loading when `std` is available
#[macro_export]
#[cfg(feature = "std")]
macro_rules! link {
    (
        $lib:ident : $cfg_ver:ident;

        $(
            $(#[internal($cfg_internal:ident)])?
            $(#[doc = $docs:literal])*
            $(#[deprecated = $note:literal$(, $cfg_strong:ident)?])?
            $(#[since = $major_version:literal.$minor_version:literal,$version_string:literal])?
            $(*$ret_mut:ident)*$ret:ident $name:ident($($(*$mut:ident)*$arg_ty:ident $arg:ident),*$(,)?);
        )*
    ) => {
        pub(crate) mod $lib {
            pub(crate) fn is_rocm_lib_present() -> bool {
                for path in $crate::link::get_lib_name_candidates(::core::stringify!($lib)) {
                    if matches!(::std::fs::exists(path), Ok(true)) {
                        return true;
                    }
                }
                false
            }

            #[cfg(test)]
            #[allow(dead_code)]
            pub(super) static HAS_ROCMLIB: ::std::sync::LazyLock<bool> = ::std::sync::LazyLock::new(is_rocm_lib_present);
        }

        static $lib: ::std::sync::LazyLock<::libloading::Library> = {
            ::std::sync::LazyLock::new(|| {
                let paths = $crate::link::get_lib_name_candidates(::core::stringify!($lib));
                for path in &paths {
                    if let Ok(lib) = unsafe {
                        ::libloading::Library::new(path)
                    } {
                        return lib;
                    }
                }
                panic!(::core::concat!("Unable to dynamically load the \"", ::core::stringify!($lib), "\" shared library - searched for library names: {:?}. \
If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue."), paths);
            })
        };

        $(
            $(#[$cfg_internal(feature = "internal")])?
            $(#[doc = $docs])*
            $(
                $(#[$cfg_strong(feature = "deprecated")])?
                #[deprecated = $note]
            )?
            pub unsafe extern "C" fn $name(
                $($arg: $(*$mut)* $arg_ty),*
            ) -> $(*$ret_mut)* $ret {
                type F = unsafe extern "C" fn(
                    $($(*$mut)* $arg_ty),*
                ) -> $(*$ret_mut)* $ret;
                static SYMBOL: ::std::sync::LazyLock<F> = ::std::sync::LazyLock::new(|| unsafe {
                    $(
                        let ver = ::rocm_include::ver::ROCM_VERSION.expect("failed to get HIP version");
                        let min_ver = ::rocm_include::ver::Version { major: $major_version, minor: $minor_version };
                        assert!(ver >= min_ver, ::core::concat!(::core::stringify!($name), " requires ", ::core::stringify!($lib), " version ^", ::core::stringify!($major_version), ".", ::core::stringify!($minor_version), ", found {}"), ver);
                    )?
                    *$lib.get::<F>(::core::stringify!($name)).unwrap()
                });
                unsafe {
                    (SYMBOL)($($arg),*)
                }
            }

            $(#[$cfg_internal(feature = "internal")])?
            $($(#[$cfg_strong(feature = "deprecated")])?)?
            #[cfg(test)]
            pastey::paste! {
                #[test]
                fn [<test_$name>]() {
                    type F = unsafe extern "C" fn(
                        $($(*$mut)* $arg_ty),*
                    ) -> $(*$ret_mut)* $ret;
                    if !*$lib::HAS_ROCMLIB { return; }
                    let f: F = unsafe {
                        $(
                            let Some(ver) = *::rocm_include::ver::ROCM_VERSION else { return; };
                            let min_ver = ::rocm_include::ver::Version { major: $major_version, minor: $minor_version };
                            if ver < min_ver { return; }
                        )?
                        *$lib.get::<F>(::core::stringify!($name)).unwrap()
                    };
                    let _ = ::core::hint::black_box(f);
                }
            }
        )*
    };
}

#[macro_export]
#[cfg(not(feature = "std"))]
macro_rules! link {
    (
        $lib:ident : $cfg_ver:ident;

        $(
            $(#[internal($cfg_internal:ident)])?
            $(#[doc = $docs:literal])*
            $(#[deprecated = $note:literal$(, $cfg_strong:ident)?])?
            $(#[since = $major_version:literal.$minor_version:literal,$version_string:literal])?
            $(*$ret_mut:ident)*$ret:ident $name:ident($($(*$mut:ident)*$arg_ty:ident $arg:ident),*$(,)?);
        )*
    ) => {
        $(
            $(#[$cfg_internal(feature = "internal")])?
            $(#[cfg($cfg_ver = $version_string)])?
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

            $(#[$cfg_internal(feature = "internal")])?
            $(#[cfg($cfg_ver = $version_string)])?
            $($(#[$cfg_strong(feature = "deprecated")])?)?
            #[cfg(test)]
            pastey::paste! {
                #[test]
                fn [<test_$name>]() {
                    let _ = ::core::hint::black_box($name);
                }
            }
        )*
    };
}

#[cfg(feature = "std")]
pub fn get_lib_name_candidates(lib_name: &str) -> Vec<String> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let rocm_path = std::env::var("ROCM_PATH").ok();

    let stem = format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}");

    let mut out = Vec::with_capacity(2 + rocm_path.is_some() as usize);
    out.push(stem.clone());
    out.push(format!("/opt/rocm/lib/{stem}"));
    if let Some(rocm_path) = &rocm_path {
        out.push(format!("{rocm_path}/lib/{stem}"));
    }
    out
}
