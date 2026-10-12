#![allow(clippy::unnecessary_cast)]
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

pub mod shared;

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub use rocm_sys::is_amdhip64_present;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub mod hiprtc;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub use rocm_sys::is_hiprtc_present;
