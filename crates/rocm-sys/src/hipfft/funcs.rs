use super::types::*;
use crate::{internal::*, hip::*};

// #[cfg(feature = "dynamic-loading")]
// use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

link! {
    hipfft : [7, 8, 9, 10] : rocm_ver;
}
