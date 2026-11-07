#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
}

#[cfg(feature = "std")]
pub use super::dynamic::{ROCM_VERSION, find_rocm_version};
