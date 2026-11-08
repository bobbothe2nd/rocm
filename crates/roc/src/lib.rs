#![allow(clippy::unnecessary_cast, clippy::missing_safety_doc)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[allow(unused_macros)]
macro_rules! wrap_sys_res {
    (|$val:ident| $expr:expr) => {{
        let mut $val = ::core::mem::MaybeUninit::uninit();

        Ok(try_err!($expr, $val.assume_init()))
    }};
}

#[allow(unused_macros)]
macro_rules! raii {
    ($name:ident($inner:ident)) => {
        #[::core::derive(::core::fmt::Debug)]
        pub struct $name {
            pub(crate) inner: ::alloc::sync::Arc<$inner>,
        }

        impl ::core::clone::Clone for $name {
            fn clone(&self) -> Self {
                Self {
                    inner: ::alloc::sync::Arc::clone(&self.inner),
                }
            }
        }

        impl ::core::ops::Deref for $name {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl $inner {
            pub fn managed(self) -> $name {
                $name {
                    inner: ::alloc::sync::Arc::new(self),
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
pub use rocm_sys::is_rocblas_present;

#[cfg(all(feature = "hipblas", any(feature = "dynamic-loading", hipblas)))]
pub use rocm_sys::is_hipblas_present;

#[cfg(all(feature = "hipblaslt", any(feature = "dynamic-loading", hipblaslt)))]
pub use rocm_sys::is_hipblaslt_present;

#[cfg(all(feature = "rocfft", any(feature = "dynamic-loading", rocfft)))]
pub use rocm_sys::is_rocfft_present;

#[cfg(all(feature = "hipfft", any(feature = "dynamic-loading", hipfft)))]
pub use rocm_sys::is_hipfft_present;

#[cfg(any(feature = "dynamic-loading", all(hip, hiprtc)))]
rocm_include::doctest!(["hip", "hiprtc"] readme: "../README.md");
