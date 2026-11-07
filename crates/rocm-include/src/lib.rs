#![cfg_attr(not(feature = "std"), no_std)]

pub mod consts;

pub mod cparse;

pub mod ctype;

pub mod ver;

#[cfg(feature = "std")]
mod dynamic;
