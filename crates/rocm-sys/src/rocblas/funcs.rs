use super::types::*;
use crate::hip::*;

use core::ffi::*;

// #[cfg(feature = "dynamic-loading")]
// use crate::version::{ROCBLAS_MAJOR as MAJOR, ROCBLAS_MINOR as MINOR};

link! {
    rocblas : [7, 8, 9, 10] : rocblas_ver;

    rocblas_status rocblas_create_handle(*mut rocblas_handle handle);

    rocblas_status rocblas_destroy_handle(rocblas_handle handle);

    rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream);

    rocblas_status rocblas_get_stream(rocblas_handle handle, *mut hipStream_t stream);

    rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode pointer_mode);

    rocblas_status rocblas_get_pointer_mode(rocblas_handle handle, *mut rocblas_pointer_mode pointer_mode);

    rocblas_status rocblas_set_atomics_mode(rocblas_handle handle, rocblas_atomics_mode atomics_mode);

    //

    rocblas_status rocblas_get_version_string_size(*mut usize len);

    rocblas_status rocblas_get_version_string(*mut c_char buf, usize len);
}
