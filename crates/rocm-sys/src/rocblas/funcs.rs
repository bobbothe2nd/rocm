use super::types::*;
use crate::hip::*;

use core::ffi::*;

type Unit = ();

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

    rocblas_status rocblas_get_atomics_mode(rocblas_handle handle, *mut rocblas_atomics_mode atomics_mode);

    rocblas_status rocblas_set_batch_alpha_stride(rocblas_handle handle, rocblas_stride alpha_stride);

    rocblas_status rocblas_get_batch_alpha_stride(rocblas_handle handle, *mut rocblas_stride alpha_stride);

    rocblas_status rocblas_set_batch_beta_stride(rocblas_handle handle, rocblas_stride beta_stride);

    rocblas_status rocblas_get_batch_beta_stride(rocblas_handle handle, *mut rocblas_stride beta_stride);

    rocblas_pointer_mode rocblas_pointer_to_mode(*mut c_void ptr);

    Unit rocblas_initialize();

    *const char rocblas_status_to_string(rocblas_status status);

    rocblas_status rocblas_set_vector(rocblas_int n, rocblas_int elem_size, *const c_void x, rocblas_int incx, *mut c_void y, rocblas_int incy);

    rocblas_status rocblas_get_vector(rocblas_int n, rocblas_int elem_size, *const c_void x, rocblas_int incx, *mut c_void y, rocblas_int incy);

    rocblas_status rocblas_set_vector_async(rocblas_int n, rocblas_int elem_size, *const c_void x, rocblas_int incx, *mut c_void y, rocblas_int incy, hipStream_t stream);

    rocblas_status rocblas_get_vector_async(rocblas_int n, rocblas_int elem_size, *const c_void x, rocblas_int incx, *mut c_void y, rocblas_int incy, hipStream_t stream);

    rocblas_status rocblas_set_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const c_void a, rocblas_int lda, *mut c_void b, rocblas_int ldb);

    rocblas_status rocblas_get_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const c_void a, rocblas_int lda, *mut c_void b, rocblas_int ldb);

    rocblas_status rocblas_set_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const c_void a, rocblas_int lda, *mut c_void b, rocblas_int ldb, hipStream_t stream);

    rocblas_status rocblas_get_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const c_void a, rocblas_int lda, *mut c_void b, rocblas_int ldb, hipStream_t stream);

    rocblas_status rocblas_start_device_memory_size_query(rocblas_handle handle);

    rocblas_status rocblas_stop_device_memory_size_query(rocblas_handle handle, *mut usize size);

    rocblas_status rocblas_get_device_memory_size(rocblas_handle handle, *mut usize size);

    rocblas_status rocblas_set_workspace(rocblas_handle handle, *mut c_void addr, usize size);

    bool rocblas_is_managing_device_memory(rocblas_handle handle);

    rocblas_status rocblas_get_version_string_size(*mut usize len);

    rocblas_status rocblas_get_version_string(*mut c_char buf, usize len);

    // rocBLAS Level-1 functions

    rocblas_status rocblas_isamax(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_idamax(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_icamax(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_izamax(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_isamax_batched(rocblas_handle handle, rocblas_int n, *const *const c_float x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamax_batched(rocblas_handle handle, rocblas_int n, *const *const c_double x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamax_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamax_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_isamax_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamax_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamax_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamax_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_isamin(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_idamin(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_icamin(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_izamin(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_isamin_batched(rocblas_handle handle, rocblas_int n, *const *const c_float x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamin_batched(rocblas_handle handle, rocblas_int n, *const *const c_double x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamin_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamin_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_isamin_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamin_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamin_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamin_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_sasum(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, *mut c_float result);

    rocblas_status rocblas_dasum(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, *mut c_double result);

    rocblas_status rocblas_scasum(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut c_float result);

    rocblas_status rocblas_dzasum(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut c_double result);

    rocblas_status rocblas_sasum_batched(rocblas_handle handle, rocblas_int n, *const *const c_float x, rocblas_int incx, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dasum_batched(rocblas_handle handle, rocblas_int n, *const *const c_double x, rocblas_int incx, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_scasum_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dzasum_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_sasum_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dasum_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_scasum_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dzasum_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_saxpy(rocblas_handle handle, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_daxpy(rocblas_handle handle, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, *mut c_double y, rocblas_int incy);

    rocblas_status rocblas_haxpy(rocblas_handle handle, rocblas_int n, *const rocblas_half alpha, *const rocblas_half x, rocblas_int incx, *mut rocblas_half y, rocblas_int incy);

    rocblas_status rocblas_caxpy(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zaxpy(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_saxpy_batched(rocblas_handle handle, rocblas_int n, *const c_float alpha, *const *const c_float x, rocblas_int incx, *const c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_daxpy_batched(rocblas_handle handle, rocblas_int n, *const c_double alpha, *const *const c_double x, rocblas_int incx, *const c_double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_haxpy_batched(rocblas_handle handle, rocblas_int n, *const rocblas_half alpha, *const *const rocblas_half x, rocblas_int incx, *const rocblas_half y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_caxpy_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zaxpy_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_saxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, rocblas_stride stridex, *mut c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_daxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, rocblas_stride stridex, *mut c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_haxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_half alpha, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_half y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_caxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zaxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_scopy(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_dcopy(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, *mut c_double y, rocblas_int incy);

    rocblas_status rocblas_ccopy(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zcopy(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_scopy_batched(rocblas_handle handle, rocblas_int n, *const *const c_float x, rocblas_int incx, *const c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dcopy_batched(rocblas_handle handle, rocblas_int n, *const *const c_double x, rocblas_int incx, *const c_double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_ccopy_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zcopy_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_scopy_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, rocblas_stride stridex, *mut c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dcopy_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, rocblas_stride stridex, *mut c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_ccopy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zcopy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sdot(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, *const c_float y, rocblas_int incy, *mut c_float result);

    rocblas_status rocblas_ddot(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, *const c_double y, rocblas_int incy, *mut c_double result);

    rocblas_status rocblas_hdot(rocblas_handle handle, rocblas_int n, *const rocblas_half x, rocblas_int incx, *const rocblas_half y, rocblas_int incy, *mut rocblas_half result);

    rocblas_status rocblas_bfdot(rocblas_handle handle, rocblas_int n, *const rocblas_bfloat16 x, rocblas_int incx, *const rocblas_bfloat16 y, rocblas_int incy, *mut rocblas_bfloat16 result);

    rocblas_status rocblas_cdotu(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex result);

    rocblas_status rocblas_cdotc(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex result);

    rocblas_status rocblas_zdotu(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex result);

    rocblas_status rocblas_zdotc(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_double_complex result);

    rocblas_status rocblas_sdot_batched(rocblas_handle handle, rocblas_int n, *const *const c_float x, rocblas_int incx, *const *const c_float y, rocblas_int incy, rocblas_int batch_count, *mut c_float result);

    rocblas_status rocblas_ddot_batched(rocblas_handle handle, rocblas_int n, *const *const c_double x, rocblas_int incx, *const *const c_double y, rocblas_int incy, rocblas_int batch_count, *mut c_double result);

    rocblas_status rocblas_hdot_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_half x, rocblas_int incx, *const *const rocblas_half y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_half result);

    rocblas_status rocblas_bfdot_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_bfloat16 x, rocblas_int incx, *const *const rocblas_bfloat16 y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_bfloat16 result);

    rocblas_status rocblas_cdotu_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_cdotc_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_zdotu_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_zdotc_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_sdot_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, rocblas_stride stridex, *const c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut c_float result);

    rocblas_status rocblas_ddot_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, rocblas_stride stridex, *const c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut c_double result);

    rocblas_status rocblas_hdot_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *const rocblas_half y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_half result);

    rocblas_status rocblas_bfdot_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_bfloat16 x, rocblas_int incx, rocblas_stride stridex, *const rocblas_bfloat16 y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_bfloat16 result);

    rocblas_status rocblas_cdotu_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_cdotc_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_zdotu_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_zdotc_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_snrm2(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, *mut c_float result);

    rocblas_status rocblas_dnrm2(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, *mut c_double result);

    rocblas_status rocblas_scnrm2(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut c_float result);

    rocblas_status rocblas_dznrm2(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut c_double result);

    rocblas_status rocblas_snrm2_batched(rocblas_handle handle, rocblas_int n, *const *const c_float x, rocblas_int incx, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dnrm2_batched(rocblas_handle handle, rocblas_int n, *const *const c_double x, rocblas_int incx, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_scnrm2_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dznrm2_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_snrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dnrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_scnrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_float results);

    rocblas_status rocblas_dznrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut c_double results);

    rocblas_status rocblas_srot(rocblas_handle handle, rocblas_int n, *mut c_float x, rocblas_int incx, *mut c_float y, rocblas_int incy, *const c_float c, *const c_float s);

    rocblas_status rocblas_drot(rocblas_handle handle, rocblas_int n, *mut c_double x, rocblas_int incx, *mut c_double y, rocblas_int incy, *const c_double c, *const c_double s);

    rocblas_status rocblas_crot(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy, *const c_float c, *const rocblas_float_complex s);

    rocblas_status rocblas_csrot(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy, *const c_float c, *const c_float s);

    rocblas_status rocblas_zrot(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy, *const c_double c, *const rocblas_double_complex s);

    rocblas_status rocblas_zdrot(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy, *const c_double c, *const c_double s);

    rocblas_status rocblas_srot_batched(rocblas_handle handle, rocblas_int n, *const *mut c_float x, rocblas_int incx, *const *mut c_float y, rocblas_int incy, *const c_float c, *const c_float s, rocblas_int batch_count);

    rocblas_status rocblas_drot_batched(rocblas_handle handle, rocblas_int n, *const *mut c_double x, rocblas_int incx, *const *mut c_double y, rocblas_int incy, *const c_double c, *const c_double s, rocblas_int batch_count);

    rocblas_status rocblas_crot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex y, rocblas_int incy, *const c_float c, *const rocblas_float_complex s, rocblas_int batch_count);

    rocblas_status rocblas_csrot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex y, rocblas_int incy, *const c_float c, *const c_float s, rocblas_int batch_count);

    rocblas_status rocblas_zrot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex y, rocblas_int incy, *const c_double c, *const rocblas_double_complex s, rocblas_int batch_count);

    rocblas_status rocblas_zdrot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex y, rocblas_int incy, *const c_double c, *const c_double s, rocblas_int batch_count);

    rocblas_status rocblas_srot_strided_batched(rocblas_handle handle, rocblas_int n, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, *mut c_float y, rocblas_int incy, rocblas_stride stride_y, *const c_float c, *const c_float s, rocblas_int batch_count);

    rocblas_status rocblas_drot_strided_batched(rocblas_handle handle, rocblas_int n, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, *mut c_double y, rocblas_int incy, rocblas_stride stride_y, *const c_double c, *const c_double s, rocblas_int batch_count);

    rocblas_status rocblas_crot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *const c_float c, *const rocblas_float_complex s, rocblas_int batch_count);

    rocblas_status rocblas_csrot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *const c_float c, *const c_float s, rocblas_int batch_count);

    rocblas_status rocblas_zrot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *const c_double c, *const rocblas_double_complex s, rocblas_int batch_count);

    rocblas_status rocblas_zdrot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *const c_double c, *const c_double s, rocblas_int batch_count);

    rocblas_status rocblas_srotg(rocblas_handle handle, *mut c_float a, *mut c_float b, *mut c_float c, *mut c_float s);

    rocblas_status rocblas_drotg(rocblas_handle handle, *mut c_double a, *mut c_double b, *mut c_double c, *mut c_double s);

    rocblas_status rocblas_crotg(rocblas_handle handle, *mut rocblas_float_complex a, *mut rocblas_float_complex b, *mut c_float c, *mut rocblas_float_complex s);

    rocblas_status rocblas_zrotg(rocblas_handle handle, *mut rocblas_double_complex a, *mut rocblas_double_complex b, *mut c_double c, *mut rocblas_double_complex s);

    rocblas_status rocblas_srotg_batched(rocblas_handle handle, *const *mut c_float a, *const *mut c_float b, *const *mut c_float c, *const *mut c_float s, rocblas_int batch_count);

    rocblas_status rocblas_drotg_batched(rocblas_handle handle, *const *mut c_double a, *const *mut c_double b, *const *mut c_double c, *const *mut c_double s, rocblas_int batch_count);

    rocblas_status rocblas_crotg_batched(rocblas_handle handle, *const rocblas_float_complex a, *const rocblas_float_complex b, *const c_float c, *const rocblas_float_complex s, rocblas_int batch_count);

    rocblas_status rocblas_zrotg_batched(rocblas_handle handle, *const *mut rocblas_double_complex a, *const *mut rocblas_double_complex b, *const *mut c_double c, *const rocblas_double_complex s, rocblas_int batch_count);

    rocblas_status rocblas_srotg_strided_batched(rocblas_handle handle, *mut c_float a, rocblas_stride stride_a, *mut c_float b, rocblas_stride stride_b, *mut c_float c, rocblas_stride stride_c, *mut c_float s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_drotg_strided_batched(rocblas_handle handle, *mut c_double a, rocblas_stride stride_a, *mut c_double b, rocblas_stride stride_b, *mut c_double c, rocblas_stride stride_c, *mut c_double s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_crotg_strided_batched(rocblas_handle handle, *mut rocblas_float_complex a, rocblas_stride stride_a, *mut rocblas_float_complex b, rocblas_stride stride_b, *mut c_float c, rocblas_stride stride_c, *mut rocblas_float_complex s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_zrotg_strided_batched(rocblas_handle handle, *mut rocblas_double_complex a, rocblas_stride stride_a, *mut rocblas_double_complex b, rocblas_stride stride_b, *mut c_double c, rocblas_stride stride_c, *mut rocblas_double_complex s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_srotm(rocblas_handle handle, rocblas_int n, *mut c_float x, rocblas_int incx, *mut c_float y, rocblas_int incy, *const c_float param);

    rocblas_status rocblas_drotm(rocblas_handle handle, rocblas_int n, *mut c_double x, rocblas_int incx, *mut c_double y, rocblas_int incy, *const c_double param);

    rocblas_status rocblas_srotm_batched(rocblas_handle handle, rocblas_int n, *const *mut c_float x, rocblas_int incx, *const *mut c_float y, rocblas_int incy, *const *const c_float param, rocblas_int batch_count);

    rocblas_status rocblas_drotm_batched(rocblas_handle handle, rocblas_int n, *const *mut c_double x, rocblas_int incx, *const *mut c_double y, rocblas_int incy, *const *const c_double param, rocblas_int batch_count);

    rocblas_status rocblas_srotm_strided_batched(rocblas_handle handle, rocblas_int n, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, *mut c_float y, rocblas_int incy, rocblas_stride stride_y, *const c_float param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_drotm_strided_batched(rocblas_handle handle, rocblas_int n, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, *mut c_double y, rocblas_int incy, rocblas_stride stride_y, *const c_double param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_srotmg(rocblas_handle handle, *mut c_float d1, *mut c_float d2, *mut c_float x1, *const c_float y1, *mut c_float param);

    rocblas_status rocblas_drotmg(rocblas_handle handle, *mut c_double d1, *mut c_double d2, *mut c_double x1, *const c_double y1, *mut c_double param);

    rocblas_status rocblas_srotmg_batched(rocblas_handle handle, *const *mut c_float d1, *const *mut c_float d2, *const *mut c_float x1, *const *const c_float y1, *const *mut c_float param, rocblas_int batch_count);

    rocblas_status rocblas_drotmg_batched(rocblas_handle handle, *const *mut c_double d1, *const *mut c_double d2, *const *mut c_double x1, *const *const c_double y1, *const *mut c_double param, rocblas_int batch_count);

    rocblas_status rocblas_srotmg_strided_batched(rocblas_handle handle, *mut c_float d1, rocblas_stride stride_d1, *mut c_float d2, rocblas_stride stride_d2, *mut c_float x1, rocblas_stride stride_x1, *const c_float y1, rocblas_stride stride_y1, *mut c_float param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_drotmg_strided_batched(rocblas_handle handle, *mut c_double d1, rocblas_stride stride_d1, *mut c_double d2, rocblas_stride stride_d2, *mut c_double x1, rocblas_stride stride_x1, *const c_double y1, rocblas_stride stride_y1, *mut c_double param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_sscal(rocblas_handle handle, rocblas_int n, *const c_float alpha, *mut c_float x, rocblas_int incx);

    rocblas_status rocblas_dscal(rocblas_handle handle, rocblas_int n, *const c_double alpha, *mut c_double x, rocblas_int incx);

    rocblas_status rocblas_cscal(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_zscal(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_csscal(rocblas_handle handle, rocblas_int n, *const c_float alpha, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_zdscal(rocblas_handle handle, rocblas_int n, *const c_double alpha, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_sscal_batched(rocblas_handle handle, rocblas_int n, *const c_float alpha, *const *mut c_float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dscal_batched(rocblas_handle handle, rocblas_int n, *const c_double alpha, *const *mut c_double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_cscal_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_zscal_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_csscal_batched(rocblas_handle handle, rocblas_int n, *const c_float alpha, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_zdscal_batched(rocblas_handle handle, rocblas_int n, *const c_double alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_sscal_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float alpha, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dscal_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double alpha, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_cscal_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_zscal_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_csscal_strided_batched(rocblas_handle handle, rocblas_int n, *const c_float alpha, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_zdscal_strided_batched(rocblas_handle handle, rocblas_int n, *const c_double alpha, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_sswap(rocblas_handle handle, rocblas_int n, *mut c_float x, rocblas_int incx, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_dswap(rocblas_handle handle, rocblas_int n, *mut c_double x, rocblas_int incx, *mut c_double y, rocblas_int incy);

    rocblas_status rocblas_cswap(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zswap(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_sswap_batched(rocblas_handle handle, rocblas_int n, *const *mut c_float x, rocblas_int incx, *const *mut c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dswap_batched(rocblas_handle handle, rocblas_int n, *const *mut c_double x, rocblas_int incx, *const *mut c_double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_cswap_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zswap_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut c_float x, rocblas_int incx, rocblas_stride stridex, *mut c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut c_double x, rocblas_int incx, rocblas_stride stridex, *mut c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_cswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    // rocBLAS Level-2 functions
    rocblas_status rocblas_sgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const c_float alpha, *const c_float A, rocblas_int lda, *const c_float x, rocblas_int incx, *const c_float beta, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_dgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const c_double alpha, *const c_double A, rocblas_int lda, *const c_double x, rocblas_int incx, *const c_double beta, *mut c_double y, rocblas_int incy);

    rocblas_status rocblas_cgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_sgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const c_float alpha, *const *const c_float A, rocblas_int lda, *const *const c_float x, rocblas_int incx, *const *const c_float beta, c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const c_double alpha, *const *const c_double A, rocblas_int lda, *const *const c_double x, rocblas_int incx, *const c_double beta, *const c_double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_cgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const c_float alpha, *const c_float A, rocblas_int lda, rocblas_stride stride_A, *const c_float x, rocblas_int incx, rocblas_stride stride_x, *const c_float beta, *mut c_float y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_dgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const c_double alpha, *const c_double A, rocblas_int lda, rocblas_stride stride_A, *const c_double x, rocblas_int incx, rocblas_stride stride_x, *const c_double beta, *mut c_double y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_cgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_zgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_sgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_float alpha, *const c_float A, rocblas_int lda, *const c_float x, rocblas_int incx, *const c_float beta, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_dgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_double alpha, *const c_double A, rocblas_int lda, *const c_double x, rocblas_int incx, *const c_double beta, *mut c_double y, rocblas_int incy);

    rocblas_status rocblas_cgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_sgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_float alpha, *const *const c_float A, rocblas_int lda, *const *const c_float x, rocblas_int incx, *const c_float beta, *const *mut c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_double alpha, *const *const c_double A, rocblas_int lda, *const *const c_double x, rocblas_int incx, *const c_double beta, *const c_double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_cgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_hshgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_float alpha, *const *const rocblas_half A, rocblas_int lda, *const *const rocblas_half x, rocblas_int incx, *const c_float beta, *const *mut rocblas_half y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_hssgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_float alpha, *const *const rocblas_half A, rocblas_int lda, *const *const rocblas_half x, rocblas_int incx, *const c_float beta, *const *mut c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_tstgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_float alpha, *const *const rocblas_bfloat16 A, rocblas_int lda, *const *const rocblas_bfloat16 x, rocblas_int incx, *const c_float beta, *const *mut rocblas_bfloat16 y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_tssgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const c_float alpha, *const *const rocblas_bfloat16 A, rocblas_int lda, *const *const rocblas_bfloat16 x, rocblas_int incx, *const c_float beta, *const *mut c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const c_float alpha, *const c_float A, rocblas_int lda, rocblas_stride strideA, *const c_float x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, *mut c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const c_double alpha, *const c_double A, rocblas_int lda, rocblas_stride strideA, *const c_double x, rocblas_int incx, rocblas_stride stridex, *const c_double beta, *mut c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_cgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_hshgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const c_float alpha, *const rocblas_half A, rocblas_int lda, rocblas_stride strideA, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, *mut rocblas_half y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_hssgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const c_float alpha, *const rocblas_half A, rocblas_int lda, rocblas_stride strideA, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_tstgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const c_float alpha, *const rocblas_bfloat16 A, rocblas_int lda, rocblas_stride strideA, *const rocblas_bfloat16 x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, *mut rocblas_bfloat16 y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_tssgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const c_float alpha, *const rocblas_bfloat16 A, rocblas_int lda, rocblas_stride strideA, *const rocblas_bfloat16 x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, *mut c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sger(rocblas_handle handle, rocblas_int m, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, *const c_float y, rocblas_int incy, *mut c_float A, rocblas_int lda);

    rocblas_status rocblas_dger(rocblas_handle handle, rocblas_int m, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, *const c_double y, rocblas_int incy, *mut c_double A, rocblas_int lda);

    rocblas_status rocblas_cgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_cgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_sger_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const c_float alpha, *const *const c_float x, rocblas_int incx, *const *const c_float y, rocblas_int incy, *const *mut c_float A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_dger_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const c_double alpha, *const *const c_double x, rocblas_int incx, *const *const c_double y, rocblas_int incy, *const *mut c_double A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cgeru_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zgeru_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cgerc_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zgerc_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_sger_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, rocblas_stride stridex, *const c_float y, rocblas_int incy, rocblas_stride stridey, *mut c_float A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_dger_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, rocblas_stride stridex, *const c_double y, rocblas_int incy, rocblas_stride stridey, *mut c_double A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_cgeru_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zgeru_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_cgerc_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zgerc_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_ssbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const c_float alpha, *const c_float A, rocblas_int lda, *const c_float x, rocblas_int incx, *const c_float beta, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_dsbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const c_double alpha, *const c_double A, rocblas_int lda, *const c_double x, rocblas_int incx, *const c_double beta, *mut c_double y, rocblas_int incy);

    rocblas_status rocblas_ssbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const c_float alpha, *const *const c_float A, rocblas_int lda, *const *const c_float x, rocblas_int incx, *const c_float beta, *const *mut c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_ssbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const c_float alpha, *const c_float A, rocblas_int lda, rocblas_stride strideA, *const c_float x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dsbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const c_double alpha, *const c_double A, rocblas_int lda, rocblas_stride strideA, *const c_double x, rocblas_int incx, rocblas_stride stridex, *const c_double beta, *mut c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float A, *const c_float x, rocblas_int incx, *const c_float beta, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_dspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double A, *const c_double x, rocblas_int incx, *const c_double beta, *mut c_double y, rocblas_int incy);

    rocblas_status rocblas_sspmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const c_float A, *const *const c_float x, rocblas_int incx, *const c_float beta, *const *mut c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dspmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const c_double A, *const *const c_double x, rocblas_int incx, *const c_double beta, *const *mut c_double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sspmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float A, rocblas_stride strideA, *const c_float x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, *mut c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dspmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double A, rocblas_stride strideA, *const c_double x, rocblas_int incx, rocblas_stride stridex, *const c_double beta, *mut c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, *mut c_float AP);

    rocblas_status rocblas_dspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, *mut c_double AP);

    rocblas_status rocblas_cspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex AP);

    rocblas_status rocblas_zspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex AP);

    rocblas_status rocblas_sspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const c_float x, rocblas_int incx, *const *mut c_float AP, rocblas_int batch_count);

    rocblas_status rocblas_dspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const c_double x, rocblas_int incx, *const *mut c_double AP, rocblas_int batch_count);

    rocblas_status rocblas_cspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_zspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_sspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, rocblas_stride stride_x, *mut c_float AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_dspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, rocblas_stride stride_x, *mut c_double AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_cspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_sspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, *const c_float y, rocblas_int incy, *mut c_float AP);

    rocblas_status rocblas_dspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, *const c_double y, rocblas_int incy, *mut c_double AP);

    rocblas_status rocblas_sspr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const c_float x, rocblas_int incx, *const *const c_float y, rocblas_int incy, *const *mut c_float AP, rocblas_int batch_count);

    rocblas_status rocblas_dspr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const c_double x, rocblas_int incx, *const *const c_double y, rocblas_int incy, *const *mut c_double AP, rocblas_int batch_count);

    rocblas_status rocblas_sspr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, rocblas_stride stride_x, *const c_float y, rocblas_int incy, rocblas_stride stride_y, *mut c_float AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_dspr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, rocblas_stride stride_x, *const c_double y, rocblas_int incy, rocblas_stride stride_y, *mut c_double AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_ssymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float A, rocblas_int lda, *const c_float x, rocblas_int incx, *const c_float beta, *mut c_float y, rocblas_int incy);

    rocblas_status rocblas_dsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double A, rocblas_int lda, *const c_double x, rocblas_int incx, *const c_double beta, c_double y, rocblas_int incy);

    rocblas_status rocblas_csymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_ssymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const c_float A, rocblas_int lda, *const *const c_float x, rocblas_int incx, *const c_float beta, *const *mut c_float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dsymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const c_double A, rocblas_int lda, *const *const c_double x, rocblas_int incx, *const c_double beta, *const c_double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_csymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zsymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_ssymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float A, rocblas_int lda, rocblas_stride strideA, *const c_float x, rocblas_int incx, rocblas_stride stridex, *const c_float beta, *mut c_float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dsymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double A, rocblas_int lda, rocblas_stride strideA, *const c_double x, rocblas_int incx, rocblas_stride stridex, *const c_double beta, *mut c_double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_csymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zsymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_ssyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, *mut c_float A, rocblas_int lda);

    rocblas_status rocblas_dsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, *mut c_double A, rocblas_int lda);

    rocblas_status rocblas_csyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_ssyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const c_float x, rocblas_int incx, *const *mut c_float A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_dsyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const c_double x, rocblas_int incx, *const *mut c_double A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_csyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zsyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_ssyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, rocblas_stride stridex, *mut c_float A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_dsyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, rocblas_stride stridex, *mut c_double A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_csyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zsyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_ssyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, *const c_float y, rocblas_int incy, *mut c_float A, rocblas_int lda);

    rocblas_status rocblas_dsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, *const c_double y, rocblas_int incy, *mut c_double A, rocblas_int lda);

    rocblas_status rocblas_csyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_ssyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const c_float x, rocblas_int incx, *const *const c_float y, rocblas_int incy, *const *mut c_float A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_dsyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const c_double x, rocblas_int incx, *const *const c_double y, rocblas_int incy, *const *mut c_double A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_csyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zsyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_ssyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const c_float x, rocblas_int incx, rocblas_stride stridex, *const c_float y, rocblas_int incy, rocblas_stride stridey, *mut c_float A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_dsyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const c_double x, rocblas_int incx, rocblas_stride stridex, *const c_double y, rocblas_int incy, rocblas_stride stridey, *mut c_double A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_csyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zsyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_stbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_float A, rocblas_int lda, *mut c_float x, rocblas_int incx);

    rocblas_status rocblas_dtbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_double A, rocblas_int lda, *mut c_double x, rocblas_int incx);

    rocblas_status rocblas_ctbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_stbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const c_float A, rocblas_int lda, *const *mut c_float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const c_double A, rocblas_int lda, *const *mut c_double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_float A, rocblas_int lda, rocblas_stride stride_A, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_double A, rocblas_int lda, rocblas_stride stride_A, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_stbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_float A, rocblas_int lda, *mut c_float x, rocblas_int incx);

    rocblas_status rocblas_dtbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_double A, rocblas_int lda, *mut c_double x, rocblas_int incx);

    rocblas_status rocblas_ctbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_stbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const c_float A, rocblas_int lda, *const *mut c_float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const c_double A, rocblas_int lda, *const *mut c_double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_float A, rocblas_int lda, rocblas_stride stride_A, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const c_double A, rocblas_int lda, rocblas_stride stride_A, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_stpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float A, *mut c_float x, rocblas_int incx);

    rocblas_status rocblas_dtpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double A, *mut c_double x, rocblas_int incx);

    rocblas_status rocblas_ctpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, *mut rocblas_double_complex x, rocblas_int incx);
    
    rocblas_status rocblas_stpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_float A, *const *mut c_float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_double A, *const *mut c_double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex A, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex A, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float A, rocblas_stride stride_A, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double A, rocblas_stride stride_A, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_stpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float AP, *mut c_float x, rocblas_int incx);

    rocblas_status rocblas_dtpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double AP, *mut c_double x, rocblas_int incx);

    rocblas_status rocblas_ctpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex AP, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex AP, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_stpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_float AP, *const *mut c_float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_double AP, *const *mut c_double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex AP, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex AP, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float AP, rocblas_stride stride_A, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double AP, rocblas_stride stride_A, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex AP, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex AP, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_strmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float A, rocblas_int lda, *mut c_float x, rocblas_int incx);

    rocblas_status rocblas_dtrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double A, rocblas_int lda, *mut c_double x, rocblas_int incx);

    rocblas_status rocblas_ctrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_strmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_float A, rocblas_int lda, *const *mut c_float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtrmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_double A, rocblas_int lda, *const *mut c_double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctrmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztrmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_strmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float A, rocblas_int lda, rocblas_stride stride_A, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtrmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double A, rocblas_int lda, rocblas_stride stride_A, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctrmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztrmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_strsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float A, rocblas_int lda, *mut c_float x, rocblas_int incx);

    rocblas_status rocblas_dtrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double A, rocblas_int lda, *mut c_double x, rocblas_int incx);

    rocblas_status rocblas_ctrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_strsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_float A, rocblas_int lda, *const *mut c_float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtrsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const c_double A, rocblas_int lda, *const *mut c_double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctrsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztrsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_strsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_float A, rocblas_int lda, rocblas_stride stride_A, *mut c_float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtrsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const c_double A, rocblas_int lda, rocblas_stride stride_A, *mut c_double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctrsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztrsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_chemv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zhemv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_chemv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zhemv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_chemv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_zhemv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_chbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zhbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_chbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zhbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_chbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_zhbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_chpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex AP, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zhpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex AP, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_chpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex AP, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zhpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex AP, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_chpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex AP, rocblas_stride stride_A, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_zhpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex AP, rocblas_stride stride_A, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_cher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_cher_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zher_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cher_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zher_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_cher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_cher2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zher2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cher2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zher2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_chpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex AP);

    rocblas_status rocblas_zhpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex AP);

    rocblas_status rocblas_chpr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_zhpr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_chpr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_float alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zhpr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const c_double alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_chpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex AP);

    rocblas_status rocblas_zhpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex AP);

    rocblas_status rocblas_chpr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_zhpr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_chpr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_float_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zhpr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_double_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    // rocBLAS Level-3 functions

    // rocBLAS extension

    // rocblas_axpy_ex + batched, strided_batched
    rocblas_status rocblas_axpy_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void alpha,
        rocblas_datatype alpha_type,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_axpy_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void alpha,
        rocblas_datatype alpha_type,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_axpy_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void alpha,
        rocblas_datatype alpha_type,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stridex,
        *mut c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stridey,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    // rocblas_dot_ex + batched, strided_batched
    rocblas_status rocblas_dot_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *const c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *mut c_void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_dot_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *const c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_int batch_count,
        *mut c_void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_dot_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        *const c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stride_y,
        rocblas_int batch_count,
        *mut c_void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    // rocblas_nrm2_ex + batched, strided_batched
    rocblas_status rocblas_nrm2_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut c_void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_nrm2_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_int batch_count,
        *mut c_void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_nrm2_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        rocblas_int batch_count,
        *mut c_void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    // rocblas_rot_ex + batched, strided_batched
    rocblas_status rocblas_rot_ex(
        rocblas_handle handle,
        rocblas_int n,
        *mut c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *const c_void c,
        *const c_void s,
        rocblas_datatype cs_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_rot_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *mut c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *const c_void c,
        *const c_void s,
        rocblas_datatype cs_type,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_rot_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *mut c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut c_void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stride_y,
        *const c_void c,
        *const c_void s,
        rocblas_datatype cs_type,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    // rocblas_scal_ex + batched, strided_batched
    rocblas_status rocblas_scal_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void alpha,
        rocblas_datatype alpha_type,
        *mut c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_scal_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void alpha,
        rocblas_datatype alpha_type,
        *mut c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_scal_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const c_void alpha,
        rocblas_datatype alpha_type,
        *mut c_void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stridex,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    // rocblas_gemm_ex + batched, strided_batched
    rocblas_status rocblas_gemm_ex(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const c_void alpha,
        *const c_void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const c_void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const c_void beta,
        *const c_void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut c_void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        i32 solution_index,
        u32 flags,
    );

    rocblas_status rocblas_gemm_batched_ex(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const c_void alpha,
        *const c_void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const c_void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const c_void beta,
        *const c_void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut c_void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_int batch_count,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        i32 solution_index,
        u32 flags,
    );

    rocblas_status rocblas_gemm_strided_batched_ex(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const c_void alpha,
        *const c_void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const c_void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const c_void beta,
        *const c_void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        rocblas_stride stride_c,
        *mut c_void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_stride stride_d,
        rocblas_int batch_count,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        i32 solution_index,
        u32 flags,
    );

    // rocblas_gemm_grouped_batched_ex
    rocblas_status rocblas_gemm_grouped_batched_ex(
        rocblas_handle handle,
        *const rocblas_operation transa_array,
        *const rocblas_operation transb_array,
        *const rocblas_int m_array,
        *const rocblas_int n_array,
        *const rocblas_int k_array,
        *const c_void alpha_array,
        *const *const c_void Aarray,
        rocblas_datatype a_type,
        *const rocblas_int lda_array,
        *const *const c_void Barray,
        rocblas_datatype b_type,
        *const rocblas_int ldb_array,
        *const c_void beta_array,
        *const *const c_void Carray,
        rocblas_datatype c_type,
        *const rocblas_int ldc_array,
        *const *mut c_void Darray,
        rocblas_datatype d_type,
        *const rocblas_int ldd_array,
        rocblas_int group_count,
        *const rocblas_int group_size,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        u32 flags,
    );

    // rocblas_syrk_ex
    rocblas_status rocblas_syrk_ex(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_int n,
        rocblas_int k,
        *const c_void alpha,
        *const c_void A,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const c_void beta,
        *mut c_void C,
        rocblas_datatype c_type,
        rocblas_int ldc,
        rocblas_datatype execution_type,
    );

    // rocblas_herk_ex
    rocblas_status rocblas_herk_ex(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_int n,
        rocblas_int k,
        *const c_void alpha,
        *const c_void A,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const c_void beta,
        *mut c_void C,
        rocblas_datatype c_type,
        rocblas_int ldc,
        rocblas_datatype execution_type,
    );

    // rocblas_trsm_ex + batched, strided_batched
    rocblas_status rocblas_trsm_ex(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_diagonal diag,
        rocblas_int m,
        rocblas_int n,
        *const c_void alpha,
        *const c_void A,
        rocblas_int lda,
        *mut c_void B,
        rocblas_int ldb,
        *const c_void invA,
        rocblas_int invA_size,
        rocblas_datatype compute_type,
    );

    rocblas_status rocblas_trsm_batched_ex(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_diagonal diag,
        rocblas_int m,
        rocblas_int n,
        *const c_void alpha,
        *const c_void A,
        rocblas_int lda,
        *mut c_void B,
        rocblas_int ldb,
        rocblas_int batch_count,
        *const c_void invA,
        rocblas_int invA_size,
        rocblas_datatype compute_type,
    );

    rocblas_status rocblas_trsm_strided_batched_ex(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_diagonal diag,
        rocblas_int m,
        rocblas_int n,
        *const c_void alpha,
        *const c_void A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *mut c_void B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        rocblas_int batch_count,
        *const c_void invA,
        rocblas_int invA_size,
        rocblas_stride stride_invA,
        rocblas_datatype compute_type,
    );

    // rocblas_Xgeam + batched, strided_batched
    rocblas_status rocblas_sgeam(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const c_float alpha,
        *const c_float A,
        rocblas_int lda,
        *const c_float beta,
        *const c_float B,
        rocblas_int ldb,
        c_float C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_dgeam(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const c_double alpha,
        *const c_double A,
        rocblas_int lda,
        *const c_double beta,
        *const c_double B,
        rocblas_int ldb,
        *mut c_double C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_cgeam(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_float_complex alpha,
        *const rocblas_float_complex A,
        rocblas_int lda,
        *const rocblas_float_complex beta,
        *const rocblas_float_complex B,
        rocblas_int ldb,
        *mut rocblas_float_complex C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_zgeam(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_double_complex alpha,
        *const rocblas_double_complex A,
        rocblas_int lda,
        *const rocblas_double_complex beta,
        *const rocblas_double_complex B,
        rocblas_int ldb,
        *mut rocblas_double_complex C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_sgeam_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const c_float alpha,
        *const *const c_float A,
        rocblas_int lda,
        *const c_float beta,
        *const *const c_float B,
        rocblas_int ldb,
        *const *mut c_float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_dgeam_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const c_double alpha,
        *const *const c_double A,
        rocblas_int lda,
        *const c_double beta,
        *const *const c_double B,
        rocblas_int ldb,
        *const *mut c_double C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_cgeam_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_float_complex alpha,
        *const *const rocblas_float_complex A,
        rocblas_int lda,
        *const rocblas_float_complex beta,
        *const *const rocblas_float_complex B,
        rocblas_int ldb,
        *const *mut rocblas_float_complex C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_zgeam_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_double_complex alpha,
        *const *const rocblas_double_complex A,
        rocblas_int lda,
        *const rocblas_double_complex beta,
        *const *const rocblas_double_complex B,
        rocblas_int ldb,
        *const *mut rocblas_double_complex C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_sgeam_strided_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const c_float alpha,
        *const c_float A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const c_float beta,
        *const c_float B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut c_float C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_dgeam_strided_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const c_double alpha,
        *const c_double A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const c_double beta,
        *const c_double B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut c_double C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_cgeam_strided_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_float_complex alpha,
        *const rocblas_float_complex A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const rocblas_float_complex beta,
        *const rocblas_float_complex B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut rocblas_float_complex C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_zgeam_strided_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_double_complex alpha,
        *const rocblas_double_complex A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const rocblas_double_complex beta,
        *const rocblas_double_complex B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut rocblas_double_complex C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    // rocblas_geam_ex
    rocblas_status rocblas_geam_ex(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const c_void alpha,
        *const c_void A,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const c_void B,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const c_void beta,
        *const c_void C,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut c_void D,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_datatype compute_type,
        rocblas_geam_ex_operation geam_ex_op,
    );

    rocblas_status rocblas_sdgmm(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const c_float A,
        rocblas_int lda,
        *const c_float x,
        rocblas_int incx,
        *mut c_float C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_ddgmm(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const c_double A,
        rocblas_int lda,
        *const c_double x,
        rocblas_int incx,
        c_double C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_cdgmm(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_float_complex A,
        rocblas_int lda,
        *const rocblas_float_complex x,
        rocblas_int incx,
        *mut rocblas_float_complex C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_zdgmm(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_double_complex A,
        rocblas_int lda,
        *const rocblas_double_complex x,
        rocblas_int incx,
        *mut rocblas_double_complex C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_sdgmm_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const *const c_float A,
        rocblas_int lda,
        *const *const c_float x,
        rocblas_int incx,
        *const c_float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_ddgmm_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const *const c_double A,
        rocblas_int lda,
        *const *const c_double x,
        rocblas_int incx,
        *const *mut c_double C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_cdgmm_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const *const rocblas_float_complex A,
        rocblas_int lda,
        *const *const rocblas_float_complex x,
        rocblas_int incx,
        *const *mut rocblas_float_complex C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_zdgmm_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const *const rocblas_double_complex A,
        rocblas_int lda,
        *const *const rocblas_double_complex x,
        rocblas_int incx,
        *const *mut rocblas_double_complex C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_sdgmm_strided_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const c_float A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const c_float x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut c_float C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_ddgmm_strided_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const c_double A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const c_double x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut c_double C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_cdgmm_strided_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_float_complex A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const rocblas_float_complex x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut rocblas_float_complex C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_zdgmm_strided_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const rocblas_double_complex A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const rocblas_double_complex x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut rocblas_double_complex C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    // rocblas_Xgemmt + batched, strided_batched
    rocblas_status rocblas_sgemmt(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const c_float alpha,
        *const c_float A,
        rocblas_int lda,
        *const c_float B,
        rocblas_int ldb,
        *const c_float beta,
        *mut c_float C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_dgemmt(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const c_double alpha,
        *const c_double A,
        rocblas_int lda,
        *const c_double B,
        rocblas_int ldb,
        *const c_double beta,
        *mut c_double C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_cgemmt(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const rocblas_float_complex alpha,
        *const rocblas_float_complex A,
        rocblas_int lda,
        *const rocblas_float_complex B,
        rocblas_int ldb,
        *const rocblas_float_complex beta,
        *mut rocblas_float_complex C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_zgemmt(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const rocblas_double_complex alpha,
        *const rocblas_double_complex A,
        rocblas_int lda,
        *const rocblas_double_complex B,
        rocblas_int ldb,
        *const rocblas_double_complex beta,
        *mut rocblas_double_complex C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_sgemmt_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const c_float alpha,
        *const *const c_float A,
        rocblas_int lda,
        *const *const c_float B,
        rocblas_int ldb,
        *const c_float beta,
        *const *mut c_float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_dgemmt_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const c_double alpha,
        *const *const c_double A,
        rocblas_int lda,
        *const *const c_double B,
        rocblas_int ldb,
        *const c_double beta,
        *const *mut c_double C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_cgemmt_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const rocblas_float_complex alpha,
        *const *const rocblas_float_complex A,
        rocblas_int lda,
        *const *const rocblas_float_complex B,
        rocblas_int ldb,
        *const rocblas_float_complex beta,
        *const *mut rocblas_float_complex C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_zgemmt_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const rocblas_double_complex alpha,
        *const *const rocblas_double_complex A,
        rocblas_int lda,
        *const *const rocblas_double_complex B,
        rocblas_int ldb,
        *const rocblas_double_complex beta,
        *const *mut rocblas_double_complex C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_sgemmt_strided_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const c_float alpha,
        *const c_float A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const c_float B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const c_float beta,
        *mut c_float C,
        rocblas_int ldc,
        rocblas_stride stride_c,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_dgemmt_strided_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const c_double alpha,
        *const c_double A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const c_double B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const c_double beta,
        *mut c_double C,
        rocblas_int ldc,
        rocblas_stride stride_c,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_cgemmt_strided_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const rocblas_float_complex alpha,
        *const rocblas_float_complex A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const rocblas_float_complex B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const rocblas_float_complex beta,
        *mut rocblas_float_complex C,
        rocblas_int ldc,
        rocblas_stride stride_c,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_zgemmt_strided_batched(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const rocblas_double_complex alpha,
        *const rocblas_double_complex A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const rocblas_double_complex B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const rocblas_double_complex beta,
        rocblas_double_complex C,
        rocblas_int ldc,
        rocblas_stride stride_c,
        rocblas_int batch_count,
    );
}
