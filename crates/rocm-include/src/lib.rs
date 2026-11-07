#![cfg_attr(not(feature = "std"), no_std)]

pub mod link;

pub mod cparse;

pub mod ctype;

pub mod ver;

#[cfg(feature = "std")]
mod dynamic;

#[deprecated(since = "0.1.1", note = "moved to rocm_include::ver")]
pub mod consts;
