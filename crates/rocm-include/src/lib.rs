#![cfg_attr(not(feature = "std"), no_std)]

pub mod link;
pub mod cparse;
pub mod ctype;
pub mod test;
pub mod ver;

#[cfg(feature = "std")]
mod dynamic;

mod strum;

doctest!(readme: "../README.md");

macro_rules! doc_api {
    ($release:ident) => {
        #[doc = include_str!(concat!("../new_api/", stringify!($release), ".md"))]
        pub const $release: () = {};
    };
}

pub mod updates;
