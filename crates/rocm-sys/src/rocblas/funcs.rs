use super::types::*;

use core::ffi::*;

#[cfg(feature = "dynamic-loading")]
use crate::version::{ROCBLAS_MAJOR as MAJOR, ROCBLAS_MINOR as MINOR};

link! {
    rocblas : [7, 8, 9, 10] : rocblas_ver;

    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;

    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;

//

    pub fn rocblas_get_version_string_size(len: *mut usize) -> rocblas_status;

    pub fn rocblas_get_version_string(buf: *mut c_char, len: usize) -> rocblas_status;
}
