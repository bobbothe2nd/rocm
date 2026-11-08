#![cfg_attr(not(feature = "std"), no_std)]

pub mod link;

pub mod cparse;

pub mod ctype;

pub mod test;

pub mod ver;

#[deprecated(since = "0.1.1", note = "moved to rocm_include::ver")]
pub mod consts;

#[cfg(feature = "std")]
mod dynamic;

mod strum;

doctest!(readme: "../README.md");
