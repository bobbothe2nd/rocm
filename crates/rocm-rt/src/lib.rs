#![no_std]

macro_rules! wrap_sys_res {
    (|$val:ident| $expr:expr) => {{
        let mut $val = ::core::mem::MaybeUninit::uninit();

        Ok(try_err!($expr, $val.assume_init()))
    }};
}

#[cfg(all(feature = "hip", any(feature = "dynamic-loading", hip)))]
pub mod hip;
