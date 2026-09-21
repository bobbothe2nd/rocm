#![allow(clippy::unnecessary_cast)]
#![no_std]

#[cfg(feature = "alloc")]
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

pub mod shared;

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;

#[cfg(all(feature = "hiprtc", any(feature = "dynamic-loading", hiprtc)))]
pub mod hiprtc;
