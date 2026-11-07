use super::types::*;
use crate::hip::*;
use rocm_include::ctype::*;

#[cfg(feature = "dynamic-loading")]
use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

link! {
    rocblas : rocm_ver;

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

    rocblas_pointer_mode rocblas_pointer_to_mode(*mut void ptr);

    rvoid rocblas_initialize();

    *const char rocblas_status_to_string(rocblas_status status);

    rocblas_status rocblas_set_vector(rocblas_int n, rocblas_int elem_size, *const void x, rocblas_int incx, *mut void y, rocblas_int incy);

    rocblas_status rocblas_get_vector(rocblas_int n, rocblas_int elem_size, *const void x, rocblas_int incx, *mut void y, rocblas_int incy);

    rocblas_status rocblas_set_vector_async(rocblas_int n, rocblas_int elem_size, *const void x, rocblas_int incx, *mut void y, rocblas_int incy, hipStream_t stream);

    rocblas_status rocblas_get_vector_async(rocblas_int n, rocblas_int elem_size, *const void x, rocblas_int incx, *mut void y, rocblas_int incy, hipStream_t stream);

    rocblas_status rocblas_set_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const void a, rocblas_int lda, *mut void b, rocblas_int ldb);

    rocblas_status rocblas_get_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const void a, rocblas_int lda, *mut void b, rocblas_int ldb);

    rocblas_status rocblas_set_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const void a, rocblas_int lda, *mut void b, rocblas_int ldb, hipStream_t stream);

    rocblas_status rocblas_get_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, *const void a, rocblas_int lda, *mut void b, rocblas_int ldb, hipStream_t stream);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_set_vector_64(int64_t n, int64_t elem_size, *const void x, int64_t incx, *mut void y, int64_t incy);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_get_vector_64(int64_t n, int64_t elem_size, *const void x, int64_t incx, *mut void y, int64_t incy);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_set_vector_async_64(int64_t n, int64_t elem_size, *const void x, int64_t incx, *mut void y, int64_t incy, hipStream_t stream);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_get_vector_async_64(int64_t n, int64_t elem_size, *const void x, int64_t incx, *mut void y, int64_t incy, hipStream_t stream);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_set_matrix_64(int64_t rows, int64_t cols, int64_t elem_size, *const void a, int64_t lda, *mut void b, int64_t ldb);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_get_matrix_64(int64_t rows, int64_t cols, int64_t elem_size, *const void a, int64_t lda, *mut void b, int64_t ldb);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_set_matrix_async_64(int64_t rows, int64_t cols, int64_t elem_size, *const void a, int64_t lda, *mut void b, int64_t ldb, hipStream_t stream);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_get_matrix_async_64(int64_t rows, int64_t cols, int64_t elem_size, *const void a, int64_t lda, *mut void b, int64_t ldb, hipStream_t stream);

    rocblas_status rocblas_start_device_memory_size_query(rocblas_handle handle);

    rocblas_status rocblas_stop_device_memory_size_query(rocblas_handle handle, *mut size_t size);

    rocblas_status rocblas_get_device_memory_size(rocblas_handle handle, *mut size_t size);

    rocblas_status rocblas_set_workspace(rocblas_handle handle, *mut void addr, size_t size);

    bool rocblas_is_managing_device_memory(rocblas_handle handle);

    rocblas_status rocblas_get_version_string_size(*mut size_t len);

    rocblas_status rocblas_get_version_string(*mut char buf, size_t len);

    // rocBLAS Level-1 functions
    rocblas_status rocblas_isamax(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_idamax(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_icamax(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_izamax(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_isamax_batched(rocblas_handle handle, rocblas_int n, *const *const float x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamax_batched(rocblas_handle handle, rocblas_int n, *const *const double x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamax_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamax_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_isamax_strided_batched(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamax_strided_batched(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamax_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamax_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_isamin(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_idamin(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_icamin(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_izamin(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_int result);

    rocblas_status rocblas_isamin_batched(rocblas_handle handle, rocblas_int n, *const *const float x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamin_batched(rocblas_handle handle, rocblas_int n, *const *const double x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamin_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamin_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_isamin_strided_batched(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_idamin_strided_batched(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_icamin_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_izamin_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut rocblas_int result);

    rocblas_status rocblas_sasum(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, *mut float result);

    rocblas_status rocblas_dasum(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, *mut double result);

    rocblas_status rocblas_scasum(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut float result);

    rocblas_status rocblas_dzasum(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut double result);

    rocblas_status rocblas_sasum_batched(rocblas_handle handle, rocblas_int n, *const *const float x, rocblas_int incx, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dasum_batched(rocblas_handle handle, rocblas_int n, *const *const double x, rocblas_int incx, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_scasum_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dzasum_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_sasum_strided_batched(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dasum_strided_batched(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_scasum_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dzasum_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_saxpy(rocblas_handle handle, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, *mut float y, rocblas_int incy);

    rocblas_status rocblas_daxpy(rocblas_handle handle, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, *mut double y, rocblas_int incy);

    rocblas_status rocblas_haxpy(rocblas_handle handle, rocblas_int n, *const rocblas_half alpha, *const rocblas_half x, rocblas_int incx, *mut rocblas_half y, rocblas_int incy);

    rocblas_status rocblas_caxpy(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zaxpy(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_saxpy_batched(rocblas_handle handle, rocblas_int n, *const float alpha, *const *const float x, rocblas_int incx, *const float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_daxpy_batched(rocblas_handle handle, rocblas_int n, *const double alpha, *const *const double x, rocblas_int incx, *const double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_haxpy_batched(rocblas_handle handle, rocblas_int n, *const rocblas_half alpha, *const *const rocblas_half x, rocblas_int incx, *const rocblas_half y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_caxpy_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zaxpy_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_saxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, rocblas_stride stridex, *mut float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_daxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, rocblas_stride stridex, *mut double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_haxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_half alpha, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_half y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_caxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zaxpy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_scopy(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, *mut float y, rocblas_int incy);

    rocblas_status rocblas_dcopy(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, *mut double y, rocblas_int incy);

    rocblas_status rocblas_ccopy(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zcopy(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_scopy_batched(rocblas_handle handle, rocblas_int n, *const *const float x, rocblas_int incx, *const float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dcopy_batched(rocblas_handle handle, rocblas_int n, *const *const double x, rocblas_int incx, *const double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_ccopy_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zcopy_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_scopy_strided_batched(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, rocblas_stride stridex, *mut float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dcopy_strided_batched(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, rocblas_stride stridex, *mut double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_ccopy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zcopy_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sdot(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, *const float y, rocblas_int incy, *mut float result);

    rocblas_status rocblas_ddot(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, *const double y, rocblas_int incy, *mut double result);

    rocblas_status rocblas_hdot(rocblas_handle handle, rocblas_int n, *const rocblas_half x, rocblas_int incx, *const rocblas_half y, rocblas_int incy, *mut rocblas_half result);

    rocblas_status rocblas_bfdot(rocblas_handle handle, rocblas_int n, *const rocblas_bfloat16 x, rocblas_int incx, *const rocblas_bfloat16 y, rocblas_int incy, *mut rocblas_bfloat16 result);

    rocblas_status rocblas_cdotu(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex result);

    rocblas_status rocblas_cdotc(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex result);

    rocblas_status rocblas_zdotu(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex result);

    rocblas_status rocblas_zdotc(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_double_complex result);

    rocblas_status rocblas_sdot_batched(rocblas_handle handle, rocblas_int n, *const *const float x, rocblas_int incx, *const *const float y, rocblas_int incy, rocblas_int batch_count, *mut float result);

    rocblas_status rocblas_ddot_batched(rocblas_handle handle, rocblas_int n, *const *const double x, rocblas_int incx, *const *const double y, rocblas_int incy, rocblas_int batch_count, *mut double result);

    rocblas_status rocblas_hdot_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_half x, rocblas_int incx, *const *const rocblas_half y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_half result);

    rocblas_status rocblas_bfdot_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_bfloat16 x, rocblas_int incx, *const *const rocblas_bfloat16 y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_bfloat16 result);

    rocblas_status rocblas_cdotu_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_cdotc_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_zdotu_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_zdotc_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_sdot_strided_batched(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, rocblas_stride stridex, *const float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut float result);

    rocblas_status rocblas_ddot_strided_batched(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, rocblas_stride stridex, *const double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut double result);

    rocblas_status rocblas_hdot_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *const rocblas_half y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_half result);

    rocblas_status rocblas_bfdot_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_bfloat16 x, rocblas_int incx, rocblas_stride stridex, *const rocblas_bfloat16 y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_bfloat16 result);

    rocblas_status rocblas_cdotu_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_cdotc_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_float_complex result);

    rocblas_status rocblas_zdotu_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_zdotc_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count, *mut rocblas_double_complex result);

    rocblas_status rocblas_snrm2(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, *mut float result);

    rocblas_status rocblas_dnrm2(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, *mut double result);

    rocblas_status rocblas_scnrm2(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, *mut float result);

    rocblas_status rocblas_dznrm2(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *mut double result);

    rocblas_status rocblas_snrm2_batched(rocblas_handle handle, rocblas_int n, *const *const float x, rocblas_int incx, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dnrm2_batched(rocblas_handle handle, rocblas_int n, *const *const double x, rocblas_int incx, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_scnrm2_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dznrm2_batched(rocblas_handle handle, rocblas_int n, *const *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_snrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const float x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dnrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const double x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_scnrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut float results);

    rocblas_status rocblas_dznrm2_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, rocblas_int batch_count, *mut double results);

    rocblas_status rocblas_srot(rocblas_handle handle, rocblas_int n, *mut float x, rocblas_int incx, *mut float y, rocblas_int incy, *const float c, *const float s);

    rocblas_status rocblas_drot(rocblas_handle handle, rocblas_int n, *mut double x, rocblas_int incx, *mut double y, rocblas_int incy, *const double c, *const double s);

    rocblas_status rocblas_crot(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy, *const float c, *const rocblas_float_complex s);

    rocblas_status rocblas_csrot(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy, *const float c, *const float s);

    rocblas_status rocblas_zrot(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy, *const double c, *const rocblas_double_complex s);

    rocblas_status rocblas_zdrot(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy, *const double c, *const double s);

    rocblas_status rocblas_srot_batched(rocblas_handle handle, rocblas_int n, *const *mut float x, rocblas_int incx, *const *mut float y, rocblas_int incy, *const float c, *const float s, rocblas_int batch_count);

    rocblas_status rocblas_drot_batched(rocblas_handle handle, rocblas_int n, *const *mut double x, rocblas_int incx, *const *mut double y, rocblas_int incy, *const double c, *const double s, rocblas_int batch_count);

    rocblas_status rocblas_crot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex y, rocblas_int incy, *const float c, *const rocblas_float_complex s, rocblas_int batch_count);

    rocblas_status rocblas_csrot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex y, rocblas_int incy, *const float c, *const float s, rocblas_int batch_count);

    rocblas_status rocblas_zrot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex y, rocblas_int incy, *const double c, *const rocblas_double_complex s, rocblas_int batch_count);

    rocblas_status rocblas_zdrot_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex y, rocblas_int incy, *const double c, *const double s, rocblas_int batch_count);

    rocblas_status rocblas_srot_strided_batched(rocblas_handle handle, rocblas_int n, *mut float x, rocblas_int incx, rocblas_stride stride_x, *mut float y, rocblas_int incy, rocblas_stride stride_y, *const float c, *const float s, rocblas_int batch_count);

    rocblas_status rocblas_drot_strided_batched(rocblas_handle handle, rocblas_int n, *mut double x, rocblas_int incx, rocblas_stride stride_x, *mut double y, rocblas_int incy, rocblas_stride stride_y, *const double c, *const double s, rocblas_int batch_count);

    rocblas_status rocblas_crot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *const float c, *const rocblas_float_complex s, rocblas_int batch_count);

    rocblas_status rocblas_csrot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *const float c, *const float s, rocblas_int batch_count);

    rocblas_status rocblas_zrot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *const double c, *const rocblas_double_complex s, rocblas_int batch_count);

    rocblas_status rocblas_zdrot_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *const double c, *const double s, rocblas_int batch_count);

    rocblas_status rocblas_srotg(rocblas_handle handle, *mut float a, *mut float b, *mut float c, *mut float s);

    rocblas_status rocblas_drotg(rocblas_handle handle, *mut double a, *mut double b, *mut double c, *mut double s);

    rocblas_status rocblas_crotg(rocblas_handle handle, *mut rocblas_float_complex a, *mut rocblas_float_complex b, *mut float c, *mut rocblas_float_complex s);

    rocblas_status rocblas_zrotg(rocblas_handle handle, *mut rocblas_double_complex a, *mut rocblas_double_complex b, *mut double c, *mut rocblas_double_complex s);

    rocblas_status rocblas_srotg_batched(rocblas_handle handle, *const *mut float a, *const *mut float b, *const *mut float c, *const *mut float s, rocblas_int batch_count);

    rocblas_status rocblas_drotg_batched(rocblas_handle handle, *const *mut double a, *const *mut double b, *const *mut double c, *const *mut double s, rocblas_int batch_count);

    rocblas_status rocblas_crotg_batched(rocblas_handle handle, *const rocblas_float_complex a, *const rocblas_float_complex b, *const float c, *const rocblas_float_complex s, rocblas_int batch_count);

    rocblas_status rocblas_zrotg_batched(rocblas_handle handle, *const *mut rocblas_double_complex a, *const *mut rocblas_double_complex b, *const *mut double c, *const rocblas_double_complex s, rocblas_int batch_count);

    rocblas_status rocblas_srotg_strided_batched(rocblas_handle handle, *mut float a, rocblas_stride stride_a, *mut float b, rocblas_stride stride_b, *mut float c, rocblas_stride stride_c, *mut float s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_drotg_strided_batched(rocblas_handle handle, *mut double a, rocblas_stride stride_a, *mut double b, rocblas_stride stride_b, *mut double c, rocblas_stride stride_c, *mut double s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_crotg_strided_batched(rocblas_handle handle, *mut rocblas_float_complex a, rocblas_stride stride_a, *mut rocblas_float_complex b, rocblas_stride stride_b, *mut float c, rocblas_stride stride_c, *mut rocblas_float_complex s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_zrotg_strided_batched(rocblas_handle handle, *mut rocblas_double_complex a, rocblas_stride stride_a, *mut rocblas_double_complex b, rocblas_stride stride_b, *mut double c, rocblas_stride stride_c, *mut rocblas_double_complex s, rocblas_stride stride_s, rocblas_int batch_count);

    rocblas_status rocblas_srotm(rocblas_handle handle, rocblas_int n, *mut float x, rocblas_int incx, *mut float y, rocblas_int incy, *const float param);

    rocblas_status rocblas_drotm(rocblas_handle handle, rocblas_int n, *mut double x, rocblas_int incx, *mut double y, rocblas_int incy, *const double param);

    rocblas_status rocblas_srotm_batched(rocblas_handle handle, rocblas_int n, *const *mut float x, rocblas_int incx, *const *mut float y, rocblas_int incy, *const *const float param, rocblas_int batch_count);

    rocblas_status rocblas_drotm_batched(rocblas_handle handle, rocblas_int n, *const *mut double x, rocblas_int incx, *const *mut double y, rocblas_int incy, *const *const double param, rocblas_int batch_count);

    rocblas_status rocblas_srotm_strided_batched(rocblas_handle handle, rocblas_int n, *mut float x, rocblas_int incx, rocblas_stride stride_x, *mut float y, rocblas_int incy, rocblas_stride stride_y, *const float param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_drotm_strided_batched(rocblas_handle handle, rocblas_int n, *mut double x, rocblas_int incx, rocblas_stride stride_x, *mut double y, rocblas_int incy, rocblas_stride stride_y, *const double param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_srotmg(rocblas_handle handle, *mut float d1, *mut float d2, *mut float x1, *const float y1, *mut float param);

    rocblas_status rocblas_drotmg(rocblas_handle handle, *mut double d1, *mut double d2, *mut double x1, *const double y1, *mut double param);

    rocblas_status rocblas_srotmg_batched(rocblas_handle handle, *const *mut float d1, *const *mut float d2, *const *mut float x1, *const *const float y1, *const *mut float param, rocblas_int batch_count);

    rocblas_status rocblas_drotmg_batched(rocblas_handle handle, *const *mut double d1, *const *mut double d2, *const *mut double x1, *const *const double y1, *const *mut double param, rocblas_int batch_count);

    rocblas_status rocblas_srotmg_strided_batched(rocblas_handle handle, *mut float d1, rocblas_stride stride_d1, *mut float d2, rocblas_stride stride_d2, *mut float x1, rocblas_stride stride_x1, *const float y1, rocblas_stride stride_y1, *mut float param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_drotmg_strided_batched(rocblas_handle handle, *mut double d1, rocblas_stride stride_d1, *mut double d2, rocblas_stride stride_d2, *mut double x1, rocblas_stride stride_x1, *const double y1, rocblas_stride stride_y1, *mut double param, rocblas_stride stride_param, rocblas_int batch_count);

    rocblas_status rocblas_sscal(rocblas_handle handle, rocblas_int n, *const float alpha, *mut float x, rocblas_int incx);

    rocblas_status rocblas_dscal(rocblas_handle handle, rocblas_int n, *const double alpha, *mut double x, rocblas_int incx);

    rocblas_status rocblas_cscal(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_zscal(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_csscal(rocblas_handle handle, rocblas_int n, *const float alpha, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_zdscal(rocblas_handle handle, rocblas_int n, *const double alpha, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_sscal_batched(rocblas_handle handle, rocblas_int n, *const float alpha, *const *mut float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dscal_batched(rocblas_handle handle, rocblas_int n, *const double alpha, *const *mut double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_cscal_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_zscal_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_csscal_batched(rocblas_handle handle, rocblas_int n, *const float alpha, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_zdscal_batched(rocblas_handle handle, rocblas_int n, *const double alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_sscal_strided_batched(rocblas_handle handle, rocblas_int n, *const float alpha, *mut float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dscal_strided_batched(rocblas_handle handle, rocblas_int n, *const double alpha, *mut double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_cscal_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_float_complex alpha, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_zscal_strided_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex alpha, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_csscal_strided_batched(rocblas_handle handle, rocblas_int n, *const float alpha, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_zdscal_strided_batched(rocblas_handle handle, rocblas_int n, *const double alpha, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_sswap(rocblas_handle handle, rocblas_int n, *mut float x, rocblas_int incx, *mut float y, rocblas_int incy);

    rocblas_status rocblas_dswap(rocblas_handle handle, rocblas_int n, *mut double x, rocblas_int incx, *mut double y, rocblas_int incy);

    rocblas_status rocblas_cswap(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zswap(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_sswap_batched(rocblas_handle handle, rocblas_int n, *const *mut float x, rocblas_int incx, *const *mut float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dswap_batched(rocblas_handle handle, rocblas_int n, *const *mut double x, rocblas_int incx, *const *mut double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_cswap_batched(rocblas_handle handle, rocblas_int n, *const *mut rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zswap_batched(rocblas_handle handle, rocblas_int n, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut float x, rocblas_int incx, rocblas_stride stridex, *mut float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut double x, rocblas_int incx, rocblas_stride stridex, *mut double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_cswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zswap_strided_batched(rocblas_handle handle, rocblas_int n, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_isamax_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_idamax_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_icamax_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_izamax_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_isamax_batched_64(rocblas_handle handle, int64_t n, *const *const float x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_idamax_batched_64(rocblas_handle handle, int64_t n, *const *const double x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_icamax_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_float_complex x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_izamax_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_double_complex x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_isamax_strided_batched_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_idamax_strided_batched_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_icamax_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_izamax_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_isamin_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_idamin_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_icamin_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_izamin_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_isamin_batched_64(rocblas_handle handle, int64_t n, *const *const float x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_idamin_batched_64(rocblas_handle handle, int64_t n, *const *const double x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_icamin_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_float_complex x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_izamin_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_double_complex x, int64_t incx, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_isamin_strided_batched_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_idamin_strided_batched_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_icamin_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_izamin_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut int64_t result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sasum_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, *mut float result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dasum_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, *mut double result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scasum_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, *mut float result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dzasum_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *mut double result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sasum_batched_64(rocblas_handle handle, int64_t n, *const *const float x, int64_t incx, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dasum_batched_64(rocblas_handle handle, int64_t n, *const *const double x, int64_t incx, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scasum_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_float_complex x, int64_t incx, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dzasum_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_double_complex x, int64_t incx, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sasum_strided_batched_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dasum_strided_batched_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scasum_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dzasum_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_saxpy_64(rocblas_handle handle, int64_t n, *const float alpha, *const float x, int64_t incx, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_daxpy_64(rocblas_handle handle, int64_t n, *const double alpha, *const double x, int64_t incx, *mut double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_haxpy_64(rocblas_handle handle, int64_t n, *const rocblas_half alpha, *const rocblas_half x, int64_t incx, *mut rocblas_half y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_caxpy_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zaxpy_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_saxpy_batched_64(rocblas_handle handle, int64_t n, *const float alpha, *const *const float x, int64_t incx, *const float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_daxpy_batched_64(rocblas_handle handle, int64_t n, *const double alpha, *const *const double x, int64_t incx, *const double y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_haxpy_batched_64(rocblas_handle handle, int64_t n, *const rocblas_half alpha, *const *const rocblas_half x, int64_t incx, *const rocblas_half y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_caxpy_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zaxpy_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_saxpy_strided_batched_64(rocblas_handle handle, int64_t n, *const float alpha, *const float x, int64_t incx, rocblas_stride stridex, *mut float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_daxpy_strided_batched_64(rocblas_handle handle, int64_t n, *const double alpha, *const double x, int64_t incx, rocblas_stride stridex, *mut double y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_haxpy_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_half alpha, *const rocblas_half x, int64_t incx, rocblas_stride stridex, *mut rocblas_half y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_caxpy_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zaxpy_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scopy_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dcopy_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, *mut double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ccopy_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zcopy_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scopy_batched_64(rocblas_handle handle, int64_t n, *const *const float x, int64_t incx, *const float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dcopy_batched_64(rocblas_handle handle, int64_t n, *const *const double x, int64_t incx, *const double y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ccopy_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zcopy_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scopy_strided_batched_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, rocblas_stride stridex, *mut float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dcopy_strided_batched_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, rocblas_stride stridex, *mut double y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ccopy_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zcopy_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sdot_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, *const float y, int64_t incy, *mut float result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ddot_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, *const double y, int64_t incy, *mut double result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hdot_64(rocblas_handle handle, int64_t n, *const rocblas_half x, int64_t incx, *const rocblas_half y, int64_t incy, *mut rocblas_half result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_bfdot_64(rocblas_handle handle, int64_t n, *const rocblas_bfloat16 x, int64_t incx, *const rocblas_bfloat16 y, int64_t incy, *mut rocblas_bfloat16 result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cdotu_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, *mut rocblas_float_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cdotc_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, *mut rocblas_float_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdotu_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, *mut rocblas_double_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdotc_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, rocblas_double_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sdot_batched_64(rocblas_handle handle, int64_t n, *const *const float x, int64_t incx, *const *const float y, int64_t incy, int64_t batch_count, *mut float result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ddot_batched_64(rocblas_handle handle, int64_t n, *const *const double x, int64_t incx, *const *const double y, int64_t incy, int64_t batch_count, *mut double result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hdot_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_half x, int64_t incx, *const *const rocblas_half y, int64_t incy, int64_t batch_count, *mut rocblas_half result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_bfdot_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_bfloat16 x, int64_t incx, *const *const rocblas_bfloat16 y, int64_t incy, int64_t batch_count, *mut rocblas_bfloat16 result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cdotu_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_float_complex x, int64_t incx, *const *const rocblas_float_complex y, int64_t incy, int64_t batch_count, *mut rocblas_float_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cdotc_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_float_complex x, int64_t incx, *const *const rocblas_float_complex y, int64_t incy, int64_t batch_count, *mut rocblas_float_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdotu_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_double_complex x, int64_t incx, *const *const rocblas_double_complex y, int64_t incy, int64_t batch_count, *mut rocblas_double_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdotc_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_double_complex x, int64_t incx, *const *const rocblas_double_complex y, int64_t incy, int64_t batch_count, *mut rocblas_double_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sdot_strided_batched_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, rocblas_stride stridex, *const float y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut float result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ddot_strided_batched_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, rocblas_stride stridex, *const double y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut double result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hdot_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_half x, int64_t incx, rocblas_stride stridex, *const rocblas_half y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut rocblas_half result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_bfdot_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_bfloat16 x, int64_t incx, rocblas_stride stridex, *const rocblas_bfloat16 y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut rocblas_bfloat16 result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cdotu_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_float_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut rocblas_float_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cdotc_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_float_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut rocblas_float_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdotu_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_double_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut rocblas_double_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdotc_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_double_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count, *mut rocblas_double_complex result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_snrm2_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, *mut float result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dnrm2_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, *mut double result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scnrm2_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, *mut float result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dznrm2_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *mut double result);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_snrm2_batched_64(rocblas_handle handle, int64_t n, *const *const float x, int64_t incx, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dnrm2_batched_64(rocblas_handle handle, int64_t n, *const *const double x, int64_t incx, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scnrm2_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_float_complex x, int64_t incx, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dznrm2_batched_64(rocblas_handle handle, int64_t n, *const *const rocblas_double_complex x, int64_t incx, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_snrm2_strided_batched_64(rocblas_handle handle, int64_t n, *const float x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dnrm2_strided_batched_64(rocblas_handle handle, int64_t n, *const double x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_scnrm2_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut float results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dznrm2_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, int64_t batch_count, *mut double results);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srot_64(rocblas_handle handle, int64_t n, *mut float x, int64_t incx, *mut float y, int64_t incy, *const float c, *const float s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drot_64(rocblas_handle handle, int64_t n, *mut double x, int64_t incx, *mut double y, int64_t incy, *const double c, *const double s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_crot_64(rocblas_handle handle, int64_t n, *mut rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex y, int64_t incy, *const float c, *const rocblas_float_complex s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csrot_64(rocblas_handle handle, int64_t n, *mut rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex y, int64_t incy, *const float c, *const float s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zrot_64(rocblas_handle handle, int64_t n, *mut rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex y, int64_t incy, *const double c, *const rocblas_double_complex s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdrot_64(rocblas_handle handle, int64_t n, *mut rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex y, int64_t incy, *const double c, *const double s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srot_batched_64(rocblas_handle handle, int64_t n, *const *mut float x, int64_t incx, *const *mut float y, int64_t incy, *const float c, *const float s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drot_batched_64(rocblas_handle handle, int64_t n, *const *mut double x, int64_t incx, *const *mut double y, int64_t incy, *const double c, *const double s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_crot_batched_64(rocblas_handle handle, int64_t n, *const *mut rocblas_float_complex x, int64_t incx, *const *mut rocblas_float_complex y, int64_t incy, *const float c, *const rocblas_float_complex s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csrot_batched_64(rocblas_handle handle, int64_t n, *const *mut rocblas_float_complex x, int64_t incx, *const *mut rocblas_float_complex y, int64_t incy, *const float c, *const float s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zrot_batched_64(rocblas_handle handle, int64_t n, *const *mut rocblas_double_complex x, int64_t incx, *const *mut rocblas_double_complex y, int64_t incy, *const double c, *const rocblas_double_complex s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdrot_batched_64(rocblas_handle handle, int64_t n, *const *mut rocblas_double_complex x, int64_t incx, *const *mut rocblas_double_complex y, int64_t incy, *const double c, *const double s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srot_strided_batched_64(rocblas_handle handle, int64_t n, *mut float x, int64_t incx, rocblas_stride stride_x, *mut float y, int64_t incy, rocblas_stride stride_y, *const float c, *const float s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drot_strided_batched_64(rocblas_handle handle, int64_t n, *mut double x, int64_t incx, rocblas_stride stride_x, *mut double y, int64_t incy, rocblas_stride stride_y, *const double c, *const double s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_crot_strided_batched_64(rocblas_handle handle, int64_t n, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, *const float c, *const rocblas_float_complex s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csrot_strided_batched_64(rocblas_handle handle, int64_t n, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, *const float c, *const float s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zrot_strided_batched_64(rocblas_handle handle, int64_t n, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, *const double c, *const rocblas_double_complex s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdrot_strided_batched_64(rocblas_handle handle, int64_t n, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, *const double c, *const double s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotg_64(rocblas_handle handle, *mut float a, *mut float b, *mut float c, *mut float s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotg_64(rocblas_handle handle, *mut double a, *mut double b, *mut double c, *mut double s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_crotg_64(rocblas_handle handle, *mut rocblas_float_complex a, *mut rocblas_float_complex b, *mut float c, *mut rocblas_float_complex s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zrotg_64(rocblas_handle handle, *mut rocblas_double_complex a, *mut rocblas_double_complex b, *mut double c, *mut rocblas_double_complex s);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotg_batched_64(rocblas_handle handle, *const *mut float a, *const *mut float b, *const *mut float c, *const *mut float s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotg_batched_64(rocblas_handle handle, *const *mut double a, *const *mut double b, *const *mut double c, *const *mut double s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_crotg_batched_64(rocblas_handle handle, *const rocblas_float_complex a, *const rocblas_float_complex b, *const float c, *const rocblas_float_complex s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zrotg_batched_64(rocblas_handle handle, *const *mut rocblas_double_complex a, *const *mut rocblas_double_complex b, *const *mut double c, *const rocblas_double_complex s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotg_strided_batched_64(rocblas_handle handle, *mut float a, rocblas_stride stride_a, *mut float b, rocblas_stride stride_b, *mut float c, rocblas_stride stride_c, *mut float s, rocblas_stride stride_s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotg_strided_batched_64(rocblas_handle handle, *mut double a, rocblas_stride stride_a, *mut double b, rocblas_stride stride_b, *mut double c, rocblas_stride stride_c, *mut double s, rocblas_stride stride_s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_crotg_strided_batched_64(rocblas_handle handle, *mut rocblas_float_complex a, rocblas_stride stride_a, *mut rocblas_float_complex b, rocblas_stride stride_b, *mut float c, rocblas_stride stride_c, *mut rocblas_float_complex s, rocblas_stride stride_s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zrotg_strided_batched_64(rocblas_handle handle, *mut rocblas_double_complex a, rocblas_stride stride_a, *mut rocblas_double_complex b, rocblas_stride stride_b, *mut double c, rocblas_stride stride_c, *mut rocblas_double_complex s, rocblas_stride stride_s, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotm_64(rocblas_handle handle, int64_t n, *mut float x, int64_t incx, *mut float y, int64_t incy, *const float param);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotm_64(rocblas_handle handle, int64_t n, *mut double x, int64_t incx, *mut double y, int64_t incy, *const double param);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotm_batched_64(rocblas_handle handle, int64_t n, *const *mut float x, int64_t incx, *const *mut float y, int64_t incy, *const *const float param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotm_batched_64(rocblas_handle handle, int64_t n, *const *mut double x, int64_t incx, *const *mut double y, int64_t incy, *const *const double param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotm_strided_batched_64(rocblas_handle handle, int64_t n, *mut float x, int64_t incx, rocblas_stride stride_x, *mut float y, int64_t incy, rocblas_stride stride_y, *const float param, rocblas_stride stride_param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotm_strided_batched_64(rocblas_handle handle, int64_t n, *mut double x, int64_t incx, rocblas_stride stride_x, *mut double y, int64_t incy, rocblas_stride stride_y, *const double param, rocblas_stride stride_param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotmg_64(rocblas_handle handle, *mut float d1, *mut float d2, *mut float x1, *const float y1, *mut float param);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotmg_64(rocblas_handle handle, *mut double d1, *mut double d2, *mut double x1, *const double y1, *mut double param);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotmg_batched_64(rocblas_handle handle, *const *mut float d1, *const *mut float d2, *const *mut float x1, *const *const float y1, *const *mut float param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotmg_batched_64(rocblas_handle handle, *const *mut double d1, *const *mut double d2, *const *mut double x1, *const *const double y1, *const *mut double param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_srotmg_strided_batched_64(rocblas_handle handle, *mut float d1, rocblas_stride stride_d1, *mut float d2, rocblas_stride stride_d2, *mut float x1, rocblas_stride stride_x1, *const float y1, rocblas_stride stride_y1, *mut float param, rocblas_stride stride_param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_drotmg_strided_batched_64(rocblas_handle handle, *mut double d1, rocblas_stride stride_d1, *mut double d2, rocblas_stride stride_d2, *mut double x1, rocblas_stride stride_x1, *const double y1, rocblas_stride stride_y1, *mut double param, rocblas_stride stride_param, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sscal_64(rocblas_handle handle, int64_t n, *const float alpha, *mut float x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dscal_64(rocblas_handle handle, int64_t n, *const double alpha, *mut double x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cscal_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex alpha, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zscal_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex alpha, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csscal_64(rocblas_handle handle, int64_t n, *const float alpha, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdscal_64(rocblas_handle handle, int64_t n, *const double alpha, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sscal_batched_64(rocblas_handle handle, int64_t n, *const float alpha, *const *mut float x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dscal_batched_64(rocblas_handle handle, int64_t n, *const double alpha, *const *mut double x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cscal_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex alpha, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zscal_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex alpha, *const *mut rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csscal_batched_64(rocblas_handle handle, int64_t n, *const float alpha, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdscal_batched_64(rocblas_handle handle, int64_t n, *const double alpha, *const rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sscal_strided_batched_64(rocblas_handle handle, int64_t n, *const float alpha, *mut float x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dscal_strided_batched_64(rocblas_handle handle, int64_t n, *const double alpha, *mut double x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cscal_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_float_complex alpha, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zscal_strided_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex alpha, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csscal_strided_batched_64(rocblas_handle handle, int64_t n, *const float alpha, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zdscal_strided_batched_64(rocblas_handle handle, int64_t n, *const double alpha, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sswap_64(rocblas_handle handle, int64_t n, *mut float x, int64_t incx, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dswap_64(rocblas_handle handle, int64_t n, *mut double x, int64_t incx, *mut double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cswap_64(rocblas_handle handle, int64_t n, *mut rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zswap_64(rocblas_handle handle, int64_t n, *mut rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sswap_batched_64(rocblas_handle handle, int64_t n, *const *mut float x, int64_t incx, *const *mut float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dswap_batched_64(rocblas_handle handle, int64_t n, *const *mut double x, int64_t incx, *const *mut double y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cswap_batched_64(rocblas_handle handle, int64_t n, *const *mut rocblas_float_complex x, int64_t incx, *const *mut rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zswap_batched_64(rocblas_handle handle, int64_t n, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sswap_strided_batched_64(rocblas_handle handle, int64_t n, *mut float x, int64_t incx, rocblas_stride stridex, *mut float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dswap_strided_batched_64(rocblas_handle handle, int64_t n, *mut double x, int64_t incx, rocblas_stride stridex, *mut double y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cswap_strided_batched_64(rocblas_handle handle, int64_t n, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zswap_strided_batched_64(rocblas_handle handle, int64_t n, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    // rocBLAS Level-2 functions
    rocblas_status rocblas_sgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const float alpha, *const float A, rocblas_int lda, *const float x, rocblas_int incx, *const float beta, *mut float y, rocblas_int incy);

    rocblas_status rocblas_dgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const double alpha, *const double A, rocblas_int lda, *const double x, rocblas_int incx, *const double beta, *mut double y, rocblas_int incy);

    rocblas_status rocblas_cgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_sgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const float alpha, *const *const float A, rocblas_int lda, *const *const float x, rocblas_int incx, *const *const float beta, float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const double alpha, *const *const double A, rocblas_int lda, *const *const double x, rocblas_int incx, *const double beta, *const double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_cgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zgbmv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const float alpha, *const float A, rocblas_int lda, rocblas_stride stride_A, *const float x, rocblas_int incx, rocblas_stride stride_x, *const float beta, *mut float y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_dgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_A, *const double x, rocblas_int incx, rocblas_stride stride_x, *const double beta, *mut double y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_cgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_zgbmv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, rocblas_int batch_count);

    rocblas_status rocblas_sgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, *const float x, rocblas_int incx, *const float beta, *mut float y, rocblas_int incy);

    rocblas_status rocblas_dgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, *const double x, rocblas_int incx, *const double beta, *mut double y, rocblas_int incy);

    rocblas_status rocblas_cgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_sgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const float alpha, *const *const float A, rocblas_int lda, *const *const float x, rocblas_int incx, *const float beta, *const *mut float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const double alpha, *const *const double A, rocblas_int lda, *const *const double x, rocblas_int incx, *const double beta, *const double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_cgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_hshgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const float alpha, *const *const rocblas_half A, rocblas_int lda, *const *const rocblas_half x, rocblas_int incx, *const float beta, *const *mut rocblas_half y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_hssgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const float alpha, *const *const rocblas_half A, rocblas_int lda, *const *const rocblas_half x, rocblas_int incx, *const float beta, *const *mut float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_tstgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const float alpha, *const *const rocblas_bfloat16 A, rocblas_int lda, *const *const rocblas_bfloat16 x, rocblas_int incx, *const float beta, *const *mut rocblas_bfloat16 y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_tssgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, *const float alpha, *const *const rocblas_bfloat16 A, rocblas_int lda, *const *const rocblas_bfloat16 x, rocblas_int incx, *const float beta, *const *mut float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, rocblas_stride strideA, *const float x, rocblas_int incx, rocblas_stride stridex, *const float beta, *mut float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, rocblas_stride strideA, *const double x, rocblas_int incx, rocblas_stride stridex, *const double beta, *mut double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_cgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_hshgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const float alpha, *const rocblas_half A, rocblas_int lda, rocblas_stride strideA, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *const float beta, *mut rocblas_half y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_hssgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const float alpha, *const rocblas_half A, rocblas_int lda, rocblas_stride strideA, *const rocblas_half x, rocblas_int incx, rocblas_stride stridex, *const float beta, float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_tstgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const float alpha, *const rocblas_bfloat16 A, rocblas_int lda, rocblas_stride strideA, *const rocblas_bfloat16 x, rocblas_int incx, rocblas_stride stridex, *const float beta, *mut rocblas_bfloat16 y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_tssgemv_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n, *const float alpha, *const rocblas_bfloat16 A, rocblas_int lda, rocblas_stride strideA, *const rocblas_bfloat16 x, rocblas_int incx, rocblas_stride stridex, *const float beta, *mut float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sger(rocblas_handle handle, rocblas_int m, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, *const float y, rocblas_int incy, *mut float A, rocblas_int lda);

    rocblas_status rocblas_dger(rocblas_handle handle, rocblas_int m, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, *const double y, rocblas_int incy, *mut double A, rocblas_int lda);

    rocblas_status rocblas_cgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_cgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_sger_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const float alpha, *const *const float x, rocblas_int incx, *const *const float y, rocblas_int incy, *const *mut float A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_dger_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const double alpha, *const *const double x, rocblas_int incx, *const *const double y, rocblas_int incy, *const *mut double A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cgeru_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zgeru_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cgerc_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zgerc_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_sger_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, rocblas_stride stridex, *const float y, rocblas_int incy, rocblas_stride stridey, *mut float A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_dger_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, rocblas_stride stridex, *const double y, rocblas_int incy, rocblas_stride stridey, *mut double A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_cgeru_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zgeru_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_cgerc_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zgerc_strided_batched(rocblas_handle handle, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_ssbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, *const float x, rocblas_int incx, *const float beta, *mut float y, rocblas_int incy);

    rocblas_status rocblas_dsbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, *const double x, rocblas_int incx, *const double beta, *mut double y, rocblas_int incy);

    rocblas_status rocblas_ssbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const float alpha, *const *const float A, rocblas_int lda, *const *const float x, rocblas_int incx, *const float beta, *const *mut float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_ssbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, rocblas_stride strideA, *const float x, rocblas_int incx, rocblas_stride stridex, *const float beta, float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dsbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, rocblas_stride strideA, *const double x, rocblas_int incx, rocblas_stride stridex, *const double beta, *mut double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float A, *const float x, rocblas_int incx, *const float beta, *mut float y, rocblas_int incy);

    rocblas_status rocblas_dspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double A, *const double x, rocblas_int incx, *const double beta, *mut double y, rocblas_int incy);

    rocblas_status rocblas_sspmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const float A, *const *const float x, rocblas_int incx, *const float beta, *const *mut float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dspmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const double A, *const *const double x, rocblas_int incx, *const double beta, *const *mut double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_sspmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float A, rocblas_stride strideA, *const float x, rocblas_int incx, rocblas_stride stridex, *const float beta, *mut float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dspmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double A, rocblas_stride strideA, *const double x, rocblas_int incx, rocblas_stride stridex, *const double beta, *mut double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_sspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, *mut float AP);

    rocblas_status rocblas_dspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, *mut double AP);

    rocblas_status rocblas_cspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex AP);

    rocblas_status rocblas_zspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex AP);

    rocblas_status rocblas_sspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const float x, rocblas_int incx, *const *mut float AP, rocblas_int batch_count);

    rocblas_status rocblas_dspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const double x, rocblas_int incx, *const *mut double AP, rocblas_int batch_count);

    rocblas_status rocblas_cspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_zspr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_sspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, rocblas_stride stride_x, *mut float AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_dspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, rocblas_stride stride_x, *mut double AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_cspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zspr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_sspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, *const float y, rocblas_int incy, *mut float AP);

    rocblas_status rocblas_dspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, *const double y, rocblas_int incy, *mut double AP);

    rocblas_status rocblas_sspr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const float x, rocblas_int incx, *const *const float y, rocblas_int incy, *const *mut float AP, rocblas_int batch_count);

    rocblas_status rocblas_dspr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const double x, rocblas_int incx, *const *const double y, rocblas_int incy, *const *mut double AP, rocblas_int batch_count);

    rocblas_status rocblas_sspr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, rocblas_stride stride_x, *const float y, rocblas_int incy, rocblas_stride stride_y, *mut float AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_dspr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, rocblas_stride stride_x, *const double y, rocblas_int incy, rocblas_stride stride_y, *mut double AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_ssymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, *const float x, rocblas_int incx, *const float beta, *mut float y, rocblas_int incy);

    rocblas_status rocblas_dsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, *const double x, rocblas_int incx, *const double beta, double y, rocblas_int incy);

    rocblas_status rocblas_csymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy);

    rocblas_status rocblas_zsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy);

    rocblas_status rocblas_ssymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const float A, rocblas_int lda, *const *const float x, rocblas_int incx, *const float beta, *const *mut float y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_dsymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const double A, rocblas_int lda, *const *const double x, rocblas_int incx, *const double beta, *const double y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_csymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex beta, *const rocblas_float_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_zsymv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex beta, *const rocblas_double_complex y, rocblas_int incy, rocblas_int batch_count);

    rocblas_status rocblas_ssymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, rocblas_stride strideA, *const float x, rocblas_int incx, rocblas_stride stridex, *const float beta, *mut float y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_dsymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, rocblas_stride strideA, *const double x, rocblas_int incx, rocblas_stride stridex, *const double beta, *mut double y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_csymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex beta, *mut rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_zsymv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex beta, *mut rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, rocblas_int batch_count);

    rocblas_status rocblas_ssyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, *mut float A, rocblas_int lda);

    rocblas_status rocblas_dsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, *mut double A, rocblas_int lda);

    rocblas_status rocblas_csyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_ssyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const float x, rocblas_int incx, *const *mut float A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_dsyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const double x, rocblas_int incx, *const *mut double A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_csyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zsyr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_ssyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, rocblas_stride stridex, *mut float A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_dsyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, rocblas_stride stridex, *mut double A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_csyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zsyr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_ssyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, *const float y, rocblas_int incy, *mut float A, rocblas_int lda);

    rocblas_status rocblas_dsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, *const double y, rocblas_int incy, *mut double A, rocblas_int lda);

    rocblas_status rocblas_csyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_ssyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const float x, rocblas_int incx, *const *const float y, rocblas_int incy, *const *mut float A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_dsyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const double x, rocblas_int incx, *const *const double y, rocblas_int incy, *const *mut double A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_csyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zsyr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_ssyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const float x, rocblas_int incx, rocblas_stride stridex, *const float y, rocblas_int incy, rocblas_stride stridey, *mut float A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_dsyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const double x, rocblas_int incx, rocblas_stride stridex, *const double y, rocblas_int incy, rocblas_stride stridey, *mut double A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_csyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_zsyr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stridex, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stridey, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride strideA, rocblas_int batch_count);

    rocblas_status rocblas_stbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const float A, rocblas_int lda, *mut float x, rocblas_int incx);

    rocblas_status rocblas_dtbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const double A, rocblas_int lda, *mut double x, rocblas_int incx);

    rocblas_status rocblas_ctbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_stbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const float A, rocblas_int lda, *const *mut float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const double A, rocblas_int lda, *const *mut double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztbmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const float A, rocblas_int lda, rocblas_stride stride_A, *mut float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const double A, rocblas_int lda, rocblas_stride stride_A, *mut double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztbmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_stbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const float A, rocblas_int lda, *mut float x, rocblas_int incx);

    rocblas_status rocblas_dtbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const double A, rocblas_int lda, *mut double x, rocblas_int incx);

    rocblas_status rocblas_ctbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_stbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const float A, rocblas_int lda, *const *mut float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const double A, rocblas_int lda, *const *mut double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztbsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const float A, rocblas_int lda, rocblas_stride stride_A, *mut float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const double A, rocblas_int lda, rocblas_stride stride_A, *mut double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztbsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_stpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float A, *mut float x, rocblas_int incx);

    rocblas_status rocblas_dtpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double A, *mut double x, rocblas_int incx);

    rocblas_status rocblas_ctpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_stpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const float A, *const *mut float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const double A, *const *mut double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex A, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztpmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex A, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float A, rocblas_stride stride_A, *mut float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double A, rocblas_stride stride_A, *mut double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztpmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_stpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float AP, *mut float x, rocblas_int incx);

    rocblas_status rocblas_dtpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double AP, *mut double x, rocblas_int incx);

    rocblas_status rocblas_ctpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex AP, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex AP, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_stpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const float AP, *const *mut float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const double AP, *const *mut double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex AP, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztpsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex AP, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_stpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float AP, rocblas_stride stride_A, *mut float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double AP, rocblas_stride stride_A, *mut double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex AP, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztpsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex AP, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_strmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float A, rocblas_int lda, *mut float x, rocblas_int incx);

    rocblas_status rocblas_dtrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double A, rocblas_int lda, *mut double x, rocblas_int incx);

    rocblas_status rocblas_ctrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_strmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const float A, rocblas_int lda, *const *mut float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtrmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const double A, rocblas_int lda, *const *mut double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctrmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztrmv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_strmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float A, rocblas_int lda, rocblas_stride stride_A, *mut float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtrmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double A, rocblas_int lda, rocblas_stride stride_A, *mut double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ctrmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_ztrmv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *mut rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_strsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float A, rocblas_int lda, *mut float x, rocblas_int incx);

    rocblas_status rocblas_dtrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double A, rocblas_int lda, *mut double x, rocblas_int incx);

    rocblas_status rocblas_ctrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex x, rocblas_int incx);

    rocblas_status rocblas_ztrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex x, rocblas_int incx);

    rocblas_status rocblas_strsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const float A, rocblas_int lda, *const *mut float x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_dtrsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const double A, rocblas_int lda, *const *mut double x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ctrsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_ztrsv_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex x, rocblas_int incx, rocblas_int batch_count);

    rocblas_status rocblas_strsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const float A, rocblas_int lda, rocblas_stride stride_A, *mut float x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

    rocblas_status rocblas_dtrsv_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, *const double A, rocblas_int lda, rocblas_stride stride_A, *mut double x, rocblas_int incx, rocblas_stride stride_x, rocblas_int batch_count);

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

    rocblas_status rocblas_cher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_cher_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zher_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cher_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zher_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_cher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex A, rocblas_int lda);

    rocblas_status rocblas_zher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex A, rocblas_int lda);

    rocblas_status rocblas_cher2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_zher2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex A, rocblas_int lda, rocblas_int batch_count);

    rocblas_status rocblas_cher2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zher2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_chpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const rocblas_float_complex x, rocblas_int incx, *mut rocblas_float_complex AP);

    rocblas_status rocblas_zhpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const rocblas_double_complex x, rocblas_int incx, *mut rocblas_double_complex AP);

    rocblas_status rocblas_chpr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *mut rocblas_float_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_zhpr_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *mut rocblas_double_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_chpr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const float alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_float_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zhpr_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const double alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *mut rocblas_double_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_chpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, *const rocblas_float_complex y, rocblas_int incy, *mut rocblas_float_complex AP);

    rocblas_status rocblas_zhpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, *const rocblas_double_complex y, rocblas_int incy, *mut rocblas_double_complex AP);

    rocblas_status rocblas_chpr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, rocblas_int incx, *const *const rocblas_float_complex y, rocblas_int incy, *const *mut rocblas_float_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_zhpr2_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, rocblas_int incx, *const *const rocblas_double_complex y, rocblas_int incy, *const *mut rocblas_double_complex AP, rocblas_int batch_count);

    rocblas_status rocblas_chpr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_float_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_float_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    rocblas_status rocblas_zhpr2_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, rocblas_int incx, rocblas_stride stride_x, *const rocblas_double_complex y, rocblas_int incy, rocblas_stride stride_y, *mut rocblas_double_complex AP, rocblas_stride stride_A, rocblas_int batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgbmv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const float alpha, *const float A, int64_t lda, *const float x, int64_t incx, *const float beta, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgbmv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const double alpha, *const double A, int64_t lda, *const double x, int64_t incx, *const double beta, *mut double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgbmv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgbmv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgbmv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const float alpha, *const *const float A, int64_t lda, *const *const float x, int64_t incx, *const *const float beta, float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgbmv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const double alpha, *const *const double A, int64_t lda, *const *const double x, int64_t incx, *const double beta, *const double y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgbmv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgbmv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgbmv_strided_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const float alpha, *const float A, int64_t lda, rocblas_stride stride_A, *const float x, int64_t incx, rocblas_stride stride_x, *const float beta, *mut float y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgbmv_strided_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_A, *const double x, int64_t incx, rocblas_stride stride_x, *const double beta, *mut double y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgbmv_strided_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgbmv_strided_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, int64_t kl, int64_t ku, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgemv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, *const float x, int64_t incx, *const float beta, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgemv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, *const double x, int64_t incx, *const double beta, *mut double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgemv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgemv_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const float alpha, *const *const float A, int64_t lda, *const *const float x, int64_t incx, *const float beta, *const *mut float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const double alpha, *const *const double A, int64_t lda, *const *const double x, int64_t incx, *const double beta, *const double y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *const rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hshgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const float alpha, *const *const rocblas_half A, int64_t lda, *const *const rocblas_half x, int64_t incx, *const float beta, *const *mut rocblas_half y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hssgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const float alpha, *const *const rocblas_half A, int64_t lda, *const *const rocblas_half x, int64_t incx, *const float beta, *const *mut float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_tstgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const float alpha, *const *const rocblas_bfloat16 A, int64_t lda, *const *const rocblas_bfloat16 x, int64_t incx, *const float beta, *const *mut rocblas_bfloat16 y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_tssgemv_batched_64(rocblas_handle handle, rocblas_operation trans, int64_t m, int64_t n, *const float alpha, *const *const rocblas_bfloat16 A, int64_t lda, *const *const rocblas_bfloat16 x, int64_t incx, *const float beta, *const *mut float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, rocblas_stride strideA, *const float x, int64_t incx, rocblas_stride stridex, *const float beta, *mut float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, rocblas_stride strideA, *const double x, int64_t incx, rocblas_stride stridex, *const double beta, *mut double y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride strideA, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride strideA, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hshgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const float alpha, *const rocblas_half A, int64_t lda, rocblas_stride strideA, *const rocblas_half x, int64_t incx, rocblas_stride stridex, *const float beta, *mut rocblas_half y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hssgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const float alpha, *const rocblas_half A, int64_t lda, rocblas_stride strideA, *const rocblas_half x, int64_t incx, rocblas_stride stridex, *const float beta, float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_tstgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const float alpha, *const rocblas_bfloat16 A, int64_t lda, rocblas_stride strideA, *const rocblas_bfloat16 x, int64_t incx, rocblas_stride stridex, *const float beta, *mut rocblas_bfloat16 y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_tssgemv_strided_batched_64(rocblas_handle handle, rocblas_operation transA, int64_t m, int64_t n, *const float alpha, *const rocblas_bfloat16 A, int64_t lda, rocblas_stride strideA, *const rocblas_bfloat16 x, int64_t incx, rocblas_stride stridex, *const float beta, *mut float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sger_64(rocblas_handle handle, int64_t m, int64_t n, *const float alpha, *const float x, int64_t incx, *const float y, int64_t incy, *mut float A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dger_64(rocblas_handle handle, int64_t m, int64_t n, *const double alpha, *const double x, int64_t incx, *const double y, int64_t incy, *mut double A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgeru_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, *mut rocblas_float_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgeru_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, *mut rocblas_double_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgerc_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, *mut rocblas_float_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgerc_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, *mut rocblas_double_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sger_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const float alpha, *const *const float x, int64_t incx, *const *const float y, int64_t incy, *const *mut float A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dger_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const double alpha, *const *const double x, int64_t incx, *const *const double y, int64_t incy, *const *mut double A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgeru_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, int64_t incx, *const *const rocblas_float_complex y, int64_t incy, *const *mut rocblas_float_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgeru_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const *const rocblas_double_complex y, int64_t incy, *const *mut rocblas_double_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgerc_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const *const *const rocblas_float_complex x, int64_t incx, *const *const rocblas_float_complex y, int64_t incy, *const *mut rocblas_float_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgerc_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const *const rocblas_double_complex y, int64_t incy, *const *mut rocblas_double_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sger_strided_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const float alpha, *const float x, int64_t incx, rocblas_stride stridex, *const float y, int64_t incy, rocblas_stride stridey, *mut float A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dger_strided_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const double alpha, *const double x, int64_t incx, rocblas_stride stridex, *const double y, int64_t incy, rocblas_stride stridey, *mut double A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgeru_strided_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_float_complex y, int64_t incy, rocblas_stride stridey, *mut rocblas_float_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgeru_strided_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_double_complex y, int64_t incy, rocblas_stride stridey, *mut rocblas_double_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgerc_strided_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_float_complex y, int64_t incy, rocblas_stride stridey, *mut rocblas_float_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgerc_strided_batched_64(rocblas_handle handle, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_double_complex y, int64_t incy, rocblas_stride stridey, *mut rocblas_double_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssbmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, *const float x, int64_t incx, *const float beta, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsbmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, *const double x, int64_t incx, *const double beta, *mut double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssbmv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const float alpha, *const *const float A, int64_t lda, *const *const float x, int64_t incx, *const float beta, *const *mut float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, rocblas_stride strideA, *const float x, int64_t incx, rocblas_stride stridex, *const float beta, float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, rocblas_stride strideA, *const double x, int64_t incx, rocblas_stride stridex, *const double beta, *mut double y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float A, *const float x, int64_t incx, *const float beta, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double A, *const double x, int64_t incx, *const double beta, *mut double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspmv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const float A, *const *const float x, int64_t incx, *const float beta, *const *mut float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspmv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const double A, *const *const double x, int64_t incx, *const double beta, *const *mut double y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float A, rocblas_stride strideA, *const float x, int64_t incx, rocblas_stride stridex, *const float beta, *mut float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double A, rocblas_stride strideA, *const double x, int64_t incx, rocblas_stride stridex, *const double beta, *mut double y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, *mut float AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, *mut double AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cspr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zspr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const float x, int64_t incx, *const *mut float AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const double x, int64_t incx, *const *mut double AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cspr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, int64_t incx, *const *mut rocblas_float_complex AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zspr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const *mut rocblas_double_complex AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, rocblas_stride stride_x, *mut float AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, rocblas_stride stride_x, *mut double AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cspr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_float_complex AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zspr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_double_complex AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, *const float y, int64_t incy, *mut float AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, *const double y, int64_t incy, *mut double AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const float x, int64_t incx, *const *const float y, int64_t incy, *const *mut float AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const double x, int64_t incx, *const *const double y, int64_t incy, *const *mut double AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sspr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, rocblas_stride stride_x, *const float y, int64_t incy, rocblas_stride stride_y, *mut float AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dspr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, rocblas_stride stride_x, *const double y, int64_t incy, rocblas_stride stride_y, *mut double AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssymv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float A, int64_t lda, *const float x, int64_t incx, *const float beta, *mut float y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsymv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double A, int64_t lda, *const double x, int64_t incx, *const double beta, double y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csymv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsymv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssymv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const float A, int64_t lda, *const *const float x, int64_t incx, *const float beta, *const *mut float y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsymv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const double A, int64_t lda, *const *const double x, int64_t incx, *const double beta, *const double y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csymv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *const rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsymv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *const rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssymv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float A, int64_t lda, rocblas_stride strideA, *const float x, int64_t incx, rocblas_stride stridex, *const float beta, *mut float y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsymv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double A, int64_t lda, rocblas_stride strideA, *const double x, int64_t incx, rocblas_stride stridex, *const double beta, *mut double y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csymv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride strideA, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsymv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride strideA, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stridey, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, *mut float A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, *mut double A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const float x, int64_t incx, *const *mut float A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const double x, int64_t incx, *const *mut double A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, int64_t incx, *const *mut rocblas_float_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const *mut rocblas_double_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, rocblas_stride stridex, *mut float A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, rocblas_stride stridex, *mut double A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_float_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *mut rocblas_double_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, *const float y, int64_t incy, *mut float A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, *const double y, int64_t incy, *mut double A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, *mut rocblas_float_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, *mut rocblas_double_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const float x, int64_t incx, *const *const float y, int64_t incy, *const *mut float A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const double x, int64_t incx, *const *const double y, int64_t incy, *const *mut double A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, int64_t incx, *const *const rocblas_float_complex y, int64_t incy, *const *mut rocblas_float_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const *const rocblas_double_complex y, int64_t incy, *const *mut rocblas_double_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const float x, int64_t incx, rocblas_stride stridex, *const float y, int64_t incy, rocblas_stride stridey, *mut float A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const double x, int64_t incx, rocblas_stride stridex, *const double y, int64_t incy, rocblas_stride stridey, *mut double A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_float_complex y, int64_t incy, rocblas_stride stridey, *mut rocblas_float_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stridex, *const rocblas_double_complex y, int64_t incy, rocblas_stride stridey, *mut rocblas_double_complex A, int64_t lda, rocblas_stride strideA, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stbmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const float A, int64_t lda, *mut float x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtbmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const double A, int64_t lda, *mut double x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctbmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_float_complex A, int64_t lda, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztbmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_double_complex A, int64_t lda, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stbmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const *const float A, int64_t lda, *const *mut float x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtbmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const *const double A, int64_t lda, *const *mut double x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctbmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const *const rocblas_float_complex A, int64_t lda, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztbmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const *const rocblas_double_complex A, int64_t lda, *const *mut rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const float A, int64_t lda, rocblas_stride stride_A, *mut float x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const double A, int64_t lda, rocblas_stride stride_A, *mut double x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stbsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const float A, int64_t lda, *mut float x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtbsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const double A, int64_t lda, *mut double x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctbsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_float_complex A, int64_t lda, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztbsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_double_complex A, int64_t lda, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stbsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const *const float A, int64_t lda, *const *mut float x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtbsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const *const double A, int64_t lda, *const *mut double x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctbsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const *const rocblas_float_complex A, int64_t lda, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztbsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const *const rocblas_double_complex A, int64_t lda, *const *mut rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stbsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const float A, int64_t lda, rocblas_stride stride_A, *mut float x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtbsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const double A, int64_t lda, rocblas_stride stride_A, *mut double x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctbsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztbsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, int64_t k, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stpmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float A, *mut float x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtpmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double A, *mut double x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctpmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex A, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztpmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex A, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stpmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const float A, *const *mut float x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtpmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const double A, *const *mut double x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctpmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_float_complex A, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztpmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_double_complex A, *const *mut rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stpmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float A, rocblas_stride stride_A, *mut float x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtpmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double A, rocblas_stride stride_A, *mut double x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctpmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex A, rocblas_stride stride_A, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztpmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex A, rocblas_stride stride_A, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stpsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float AP, *mut float x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtpsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double AP, *mut double x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctpsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex AP, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztpsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex AP, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stpsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const float AP, *const *mut float x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtpsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const double AP, *const *mut double x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctpsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_float_complex AP, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztpsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_double_complex AP, *const *mut rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_stpsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float AP, rocblas_stride stride_A, *mut float x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtpsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double AP, rocblas_stride stride_A, *mut double x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctpsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex AP, rocblas_stride stride_A, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztpsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex AP, rocblas_stride stride_A, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float A, int64_t lda, *mut float x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double A, int64_t lda, *mut double x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex A, int64_t lda, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrmv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex A, int64_t lda, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const float A, int64_t lda, *const *mut float x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const double A, int64_t lda, *const *mut double x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_float_complex A, int64_t lda, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrmv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_double_complex A, int64_t lda, *const *mut rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float A, int64_t lda, rocblas_stride stride_A, *mut float x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double A, int64_t lda, rocblas_stride stride_A, *mut double x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float A, int64_t lda, *mut float x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double A, int64_t lda, *mut double x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex A, int64_t lda, *mut rocblas_float_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrsv_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex A, int64_t lda, *mut rocblas_double_complex x, int64_t incx);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const float A, int64_t lda, *const *mut float x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const double A, int64_t lda, *const *mut double x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_float_complex A, int64_t lda, *const *mut rocblas_float_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrsv_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const *const rocblas_double_complex A, int64_t lda, *const *mut rocblas_double_complex x, int64_t incx, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const float A, int64_t lda, rocblas_stride stride_A, *mut float x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const double A, int64_t lda, rocblas_stride stride_A, *mut double x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrsv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t n, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *mut rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chemv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhemv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chemv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhemv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chemv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhemv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chbmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhbmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chbmv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhbmv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *const *mut rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhbmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex AP, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpmv_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex AP, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpmv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex AP, *const *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex beta, *const *mut rocblas_float_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpmv_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex AP, *const *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex beta, *const rocblas_double_complex y, int64_t incy, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex AP, rocblas_stride stride_A, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_float_complex beta, *mut rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpmv_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex AP, rocblas_stride stride_A, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_double_complex beta, *mut rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const rocblas_float_complex x, int64_t incx, *const *mut rocblas_float_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const rocblas_double_complex x, int64_t incx, *const *mut rocblas_double_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, *mut rocblas_float_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, *mut rocblas_double_complex A, int64_t lda);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, int64_t incx, *const *const rocblas_float_complex y, int64_t incy, *const *mut rocblas_float_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const *const rocblas_double_complex y, int64_t incy, *const *mut rocblas_double_complex A, int64_t lda, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, *mut rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, *mut rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const rocblas_float_complex x, int64_t incx, *mut rocblas_float_complex AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpr_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const rocblas_double_complex x, int64_t incx, *mut rocblas_double_complex AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const *const rocblas_float_complex x, int64_t incx, *const *mut rocblas_float_complex AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpr_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const *const rocblas_double_complex x, int64_t incx, *const *mut rocblas_double_complex AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const float alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_float_complex AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpr_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const double alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *mut rocblas_double_complex AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, *const rocblas_float_complex y, int64_t incy, *mut rocblas_float_complex AP);

    #[since = 6 .2, "6.2"]
    rocblas_status rocblas_zhpr2_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, *const rocblas_double_complex y, int64_t incy, *mut rocblas_double_complex AP);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex x, int64_t incx, *const *const rocblas_float_complex y, int64_t incy, *const *mut rocblas_float_complex AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpr2_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex x, int64_t incx, *const *const rocblas_double_complex y, int64_t incy, *const *mut rocblas_double_complex AP, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chpr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_float_complex y, int64_t incy, rocblas_stride stride_y, *mut rocblas_float_complex AP, rocblas_stride stride_A, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhpr2_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex x, int64_t incx, rocblas_stride stride_x, *const rocblas_double_complex y, int64_t incy, rocblas_stride stride_y, *mut rocblas_double_complex AP, rocblas_stride stride_A, int64_t batch_count);

    // rocBLAS Level-3 functions
    rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, *const float B, rocblas_int ldb, *const float beta, *mut float C, rocblas_int ldc);

    rocblas_status rocblas_dgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, *const double B, rocblas_int ldb, *const double beta, *mut double C, rocblas_int ldc);

    rocblas_status rocblas_hgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_half alpha, *const rocblas_half A, rocblas_int lda, *const rocblas_half B, rocblas_int ldb, *const rocblas_half beta, *mut rocblas_half C, rocblas_int ldc);

    rocblas_status rocblas_cgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_sgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const float alpha, *const *const float A, rocblas_int lda, *const *const float B, rocblas_int ldb, *const float beta, *const *mut float C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_dgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const double alpha, *const *const double A, rocblas_int lda, *const *const double B, rocblas_int ldb, *const double beta, *const *mut double C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_hgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_half alpha, *const *const rocblas_half A, rocblas_int lda, *const *const rocblas_half B, rocblas_int ldb, *const rocblas_half beta, *const *mut rocblas_half C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_cgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, rocblas_stride stride_a, *const float B, rocblas_int ldb, rocblas_stride stride_b, *const float beta, *mut float C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);

    rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_a, *const double B, rocblas_int ldb, rocblas_stride stride_b, *const double beta, *mut double C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);

    rocblas_status rocblas_hgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_half alpha, *const rocblas_half A, rocblas_int lda, rocblas_stride stride_a, *const rocblas_half B, rocblas_int ldb, rocblas_stride stride_b, *const rocblas_half beta, *mut rocblas_half C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);

    rocblas_status rocblas_cgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_a, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_b, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);

    rocblas_status rocblas_zgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_a, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_b, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);

    // rocblas_status rocblas_sgemm_grouped_batched(rocblas_handle handle, *const rocblas_operation transa_array, *const rocblas_operation transb_array, *const rocblas_int m_array, *const rocblas_int n_array, *const rocblas_int k_array, *const float alpha_array, *const *const float Aarray, *const rocblas_int lda_array, *const *const float Barray, *const rocblas_int ldb_array, *const float beta_array, *const *mut float Carray, *const rocblas_int ldc_array, rocblas_int group_count, *const rocblas_int group_size);

    // rocblas_status rocblas_dgemm_grouped_batched(rocblas_handle handle, *const rocblas_operation transa_array, *const rocblas_operation transb_array, *const rocblas_int m_array, *const rocblas_int n_array, *const rocblas_int k_array, *const double alpha_array, *const *const double Aarray, *const rocblas_int lda_array, *const *const double Barray, *const rocblas_int ldb_array, *const double beta_array, *const *mut double Carray, *const rocblas_int ldc_array, rocblas_int group_count, *const rocblas_int group_size);

    rocblas_status rocblas_ssymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, *const float B, rocblas_int ldb, *const float beta, *mut float C, rocblas_int ldc);

    rocblas_status rocblas_dsymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, *const double B, rocblas_int ldb, *const double beta, *mut double C, rocblas_int ldc);

    rocblas_status rocblas_csymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zsymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_ssymm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const float alpha, *const *const float A, rocblas_int lda, *const *const float B, rocblas_int ldb, *const float beta, *const *mut float C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_dsymm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const double alpha, *const *const double A, rocblas_int lda, *const *const double B, rocblas_int ldb, *const double beta, *const *mut double C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_csymm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zsymm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_ssymm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, rocblas_stride stride_A, *const float B, rocblas_int ldb, rocblas_stride stride_B, *const float beta, *mut float C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_dsymm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_A, *const double B, rocblas_int ldb, rocblas_stride stride_B, *const double beta, *mut double C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_csymm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zsymm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_ssyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, *const float beta, *mut float C, rocblas_int ldc);

    rocblas_status rocblas_dsyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, *const double beta, *mut double C, rocblas_int ldc);

    rocblas_status rocblas_csyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zsyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_ssyrk_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const float alpha, *const *const float A, rocblas_int lda, *const float beta, *const *mut float C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_dsyrk_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const double alpha, *const *const double A, rocblas_int lda, *const double beta, *const *mut double C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_csyrk_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zsyrk_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_ssyrk_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, rocblas_stride stride_A, *const float beta, *mut float C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_dsyrk_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_A, *const double beta, *mut double C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_csyrk_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zsyrk_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_ssyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, *const float B, rocblas_int ldb, *const float beta, *mut float C, rocblas_int ldc);

    rocblas_status rocblas_dsyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, *const double B, rocblas_int ldb, *const double beta, *mut double C, rocblas_int ldc);

    rocblas_status rocblas_csyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zsyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_ssyr2k_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const float alpha, *const *const float A, rocblas_int lda, *const *const float B, rocblas_int ldb, *const float beta, *const *mut float C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_dsyr2k_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const double alpha, *const *const double A, rocblas_int lda, *const *const double B, rocblas_int ldb, *const double beta, *const *mut double C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_csyr2k_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zsyr2k_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_ssyr2k_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, rocblas_stride stride_A, *const float B, rocblas_int ldb, rocblas_stride stride_B, *const float beta, *mut float C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_dsyr2k_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_A, *const double B, rocblas_int ldb, rocblas_stride stride_B, *const double beta, *mut double C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_csyr2k_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zsyr2k_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_ssyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const float alpha, *const float A, rocblas_int lda, *const float B, rocblas_int ldb, *const float beta, *mut float C, rocblas_int ldc);

    rocblas_status rocblas_dsyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, *const double B, rocblas_int ldb, *const double beta, *mut double C, rocblas_int ldc);

    rocblas_status rocblas_csyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zsyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_ssyrkx_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const float alpha, *const *const float A, rocblas_int lda, *const *const float B, rocblas_int ldb, *const float beta, *const *mut float C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_dsyrkx_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const double alpha, *const *const double A, rocblas_int lda, *const *const double B, rocblas_int ldb, *const double beta, *const *mut double C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_csyrkx_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zsyrkx_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_ssyrkx_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const float alpha, *const *const float A, rocblas_int lda, rocblas_stride stride_A, *const float B, rocblas_int ldb, rocblas_stride stride_B, *const float beta, *mut float C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_dsyrkx_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_A, *const double B, rocblas_int ldb, rocblas_stride stride_B, *const double beta, *mut double C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_csyrkx_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zsyrkx_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_strmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, *const float B, rocblas_int ldb, *mut float C, rocblas_int ldc);

    rocblas_status rocblas_dtrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, *const double B, rocblas_int ldb, *mut double C, rocblas_int ldc);

    rocblas_status rocblas_ctrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_ztrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_strmm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const float alpha, *const *const float A, rocblas_int lda, *const *const float B, rocblas_int ldb, *const *mut float C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_dtrmm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const double alpha, *const *const double A, rocblas_int lda, *const *const double B, rocblas_int ldb, *const *mut double C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_ctrmm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_ztrmm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_strmm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, rocblas_stride stride_A, *const float B, rocblas_int ldb, rocblas_stride stride_B, *mut float C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_dtrmm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_A, *const double B, rocblas_int ldb, rocblas_stride stride_B, *mut double C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_ctrmm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_B, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_ztrmm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_B, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_strsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, *mut float B, rocblas_int ldb);

    rocblas_status rocblas_dtrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, *mut double B, rocblas_int ldb);

    rocblas_status rocblas_ctrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *mut rocblas_float_complex B, rocblas_int ldb);

    rocblas_status rocblas_ztrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *mut rocblas_double_complex B, rocblas_int ldb);

    rocblas_status rocblas_strsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const float alpha, *const *const float A, rocblas_int lda, *const *mut float B, rocblas_int ldb, rocblas_int batch_count);

    rocblas_status rocblas_dtrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const double alpha, *const *const double A, rocblas_int lda, *const *mut double B, rocblas_int ldb, rocblas_int batch_count);

    rocblas_status rocblas_ctrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *mut rocblas_float_complex B, rocblas_int ldb, rocblas_int batch_count);

    rocblas_status rocblas_ztrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *mut rocblas_double_complex B, rocblas_int ldb, rocblas_int batch_count);

    rocblas_status rocblas_strsm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const float alpha, *const float A, rocblas_int lda, rocblas_stride stride_a, *mut float B, rocblas_int ldb, rocblas_stride stride_b, rocblas_int batch_count);

    rocblas_status rocblas_dtrsm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const double alpha, *const double A, rocblas_int lda, rocblas_stride stride_a, *mut double B, rocblas_int ldb, rocblas_stride stride_b, rocblas_int batch_count);

    rocblas_status rocblas_ctrsm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_a, *mut rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_b, rocblas_int batch_count);

    rocblas_status rocblas_ztrsm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_a, *mut rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_b, rocblas_int batch_count);

    rocblas_status rocblas_chemm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zhemm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_chemm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zhemm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_chemm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zhemm_strided_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_cherk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const float alpha, *const rocblas_float_complex A, rocblas_int lda, *const float beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zherk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const double alpha, *const rocblas_double_complex A, rocblas_int lda, *const double beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_cherk_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const float alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const float beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zherk_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const double alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const double beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_cherk_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const float alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const float beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zherk_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, *const double alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const double beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_cher2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *const float beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zher2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *const double beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_cher2k_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const float beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zher2k_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const double beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_cher2k_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_B, *const float beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zher2k_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_B, *const double beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_cherkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, *const rocblas_float_complex B, rocblas_int ldb, *const float beta, *mut rocblas_float_complex C, rocblas_int ldc);

    rocblas_status rocblas_zherkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, *const rocblas_double_complex B, rocblas_int ldb, *const double beta, *mut rocblas_double_complex C, rocblas_int ldc);

    rocblas_status rocblas_cherkx_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, rocblas_int lda, *const *const rocblas_float_complex B, rocblas_int ldb, *const float beta, *const *mut rocblas_float_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_zherkx_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, rocblas_int lda, *const *const rocblas_double_complex B, rocblas_int ldb, *const double beta, *const *mut rocblas_double_complex C, rocblas_int ldc, rocblas_int batch_count);

    rocblas_status rocblas_cherkx_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_float_complex B, rocblas_int ldb, rocblas_stride stride_B, *const float beta, *mut rocblas_float_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_zherkx_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, rocblas_int lda, rocblas_stride stride_A, *const rocblas_double_complex B, rocblas_int ldb, rocblas_stride stride_B, *const double beta, *mut rocblas_double_complex C, rocblas_int ldc, rocblas_stride stride_C, rocblas_int batch_count);

    rocblas_status rocblas_strtri(rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag, rocblas_int n, *const float A, rocblas_int lda, *mut float invA, rocblas_int ldinvA);

    rocblas_status rocblas_dtrtri(rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag, rocblas_int n, *const double A, rocblas_int lda, *mut double invA, rocblas_int ldinvA);

    rocblas_status rocblas_strtri_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag, rocblas_int n, *const *const float A, rocblas_int lda, *const *mut float invA, rocblas_int ldinvA, rocblas_int batch_count);

    rocblas_status rocblas_dtrtri_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag, rocblas_int n, *const *const double A, rocblas_int lda, *const *mut double invA, rocblas_int ldinvA, rocblas_int batch_count);

    rocblas_status rocblas_strtri_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag, rocblas_int n, *const float A, rocblas_int lda, rocblas_stride stride_a, *mut float invA, rocblas_int ldinvA, rocblas_stride stride_invA, rocblas_int batch_count);

    rocblas_status rocblas_dtrtri_strided_batched(rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag, rocblas_int n, *const double A, rocblas_int lda, rocblas_stride stride_a, *mut double invA, rocblas_int ldinvA, rocblas_stride stride_invA, rocblas_int batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgemm_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, *const float B, int64_t ldb, *const float beta, *mut float C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgemm_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, *const double B, int64_t ldb, *const double beta, *mut double C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hgemm_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_half alpha, *const rocblas_half A, int64_t lda, *const rocblas_half B, int64_t ldb, *const rocblas_half beta, *mut rocblas_half C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgemm_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgemm_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgemm_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const float alpha, *const *const float A, int64_t lda, *const *const float B, int64_t ldb, *const float beta, *const *mut float C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgemm_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const double alpha, *const *const double A, int64_t lda, *const *const double B, int64_t ldb, *const double beta, *const *mut double C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hgemm_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_half alpha, *const *const rocblas_half A, int64_t lda, *const *const rocblas_half B, int64_t ldb, *const rocblas_half beta, *const *mut rocblas_half C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgemm_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgemm_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_sgemm_strided_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, rocblas_stride stride_a, *const float B, int64_t ldb, rocblas_stride stride_b, *const float beta, *mut float C, int64_t ldc, rocblas_stride stride_c, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dgemm_strided_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_a, *const double B, int64_t ldb, rocblas_stride stride_b, *const double beta, *mut double C, int64_t ldc, rocblas_stride stride_c, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_hgemm_strided_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_half alpha, *const rocblas_half A, int64_t lda, rocblas_stride stride_a, *const rocblas_half B, int64_t ldb, rocblas_stride stride_b, *const rocblas_half beta, *mut rocblas_half C, int64_t ldc, rocblas_stride stride_c, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cgemm_strided_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_a, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_b, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_c, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zgemm_strided_batched_64(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, int64_t m, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_a, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_b, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_c, int64_t batch_count);

    // #[since = 6 .1, "6.1"]
    // rocblas_status rocblas_sgemm_grouped_batched_64(rocblas_handle handle, *const rocblas_operation transa_array, *const rocblas_operation transb_array, *const int64_t m_array, *const int64_t n_array, *const int64_t k_array, *const float alpha_array, *const *const float Aarray, *const int64_t lda_array, *const *const float Barray, *const int64_t ldb_array, *const float beta_array, *const *mut float Carray, *const int64_t ldc_array, int64_t group_count, *const int64_t group_size);

    // #[since = 6 .1, "6.1"]
    // rocblas_status rocblas_dgemm_grouped_batched_64(rocblas_handle handle, *const rocblas_operation transa_array, *const rocblas_operation transb_array, *const int64_t m_array, *const int64_t n_array, *const int64_t k_array, *const double alpha_array, *const *const double Aarray, *const int64_t lda_array, *const *const double Barray, *const int64_t ldb_array, *const double beta_array, *const *mut double Carray, *const int64_t ldc_array, int64_t group_count, *const int64_t group_size);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssymm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, *const float B, int64_t ldb, *const float beta, *mut float C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsymm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, *const double B, int64_t ldb, *const double beta, *mut double C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csymm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsymm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssymm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const float alpha, *const *const float A, int64_t lda, *const *const float B, int64_t ldb, *const float beta, *const *mut float C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsymm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const double alpha, *const *const double A, int64_t lda, *const *const double B, int64_t ldb, *const double beta, *const *mut double C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csymm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsymm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssymm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, rocblas_stride stride_A, *const float B, int64_t ldb, rocblas_stride stride_B, *const float beta, *mut float C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsymm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_A, *const double B, int64_t ldb, rocblas_stride stride_B, *const double beta, *mut double C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csymm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsymm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyrk_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, *const float beta, *mut float C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyrk_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, *const double beta, *mut double C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyrk_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyrk_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyrk_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const float alpha, *const *const float A, int64_t lda, *const float beta, *const *mut float C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyrk_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const double alpha, *const *const double A, int64_t lda, *const double beta, *const *mut double C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyrk_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyrk_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyrk_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, rocblas_stride stride_A, *const float beta, *mut float C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyrk_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_A, *const double beta, *mut double C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyrk_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyrk_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr2k_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, *const float B, int64_t ldb, *const float beta, *mut float C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr2k_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, *const double B, int64_t ldb, *const double beta, *mut double C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr2k_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr2k_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr2k_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const float alpha, *const *const float A, int64_t lda, *const *const float B, int64_t ldb, *const float beta, *const *mut float C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr2k_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const double alpha, *const *const double A, int64_t lda, *const *const double B, int64_t ldb, *const double beta, *const *mut double C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr2k_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr2k_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyr2k_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, rocblas_stride stride_A, *const float B, int64_t ldb, rocblas_stride stride_B, *const float beta, *mut float C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyr2k_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_A, *const double B, int64_t ldb, rocblas_stride stride_B, *const double beta, *mut double C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyr2k_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyr2k_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyrkx_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const float alpha, *const float A, int64_t lda, *const float B, int64_t ldb, *const float beta, *mut float C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyrkx_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, *const double B, int64_t ldb, *const double beta, *mut double C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyrkx_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyrkx_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyrkx_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const float alpha, *const *const float A, int64_t lda, *const *const float B, int64_t ldb, *const float beta, *const *mut float C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyrkx_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const double alpha, *const *const double A, int64_t lda, *const *const double B, int64_t ldb, *const double beta, *const *mut double C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyrkx_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyrkx_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ssyrkx_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const float alpha, *const *const float A, int64_t lda, rocblas_stride stride_A, *const float B, int64_t ldb, rocblas_stride stride_B, *const float beta, *mut float C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dsyrkx_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_A, *const double B, int64_t ldb, rocblas_stride stride_B, *const double beta, *mut double C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_csyrkx_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zsyrkx_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strmm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, *const float B, int64_t ldb, *mut float C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrmm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, *const double B, int64_t ldb, *mut double C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrmm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrmm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strmm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const float alpha, *const *const float A, int64_t lda, *const *const float B, int64_t ldb, *const *mut float C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrmm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const double alpha, *const *const double A, int64_t lda, *const *const double B, int64_t ldb, *const *mut double C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrmm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrmm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strmm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, rocblas_stride stride_A, *const float B, int64_t ldb, rocblas_stride stride_B, *mut float C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrmm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_A, *const double B, int64_t ldb, rocblas_stride stride_B, *mut double C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrmm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_B, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrmm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_B, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strsm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, *mut float B, int64_t ldb);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrsm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, *mut double B, int64_t ldb);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrsm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *mut rocblas_float_complex B, int64_t ldb);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrsm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *mut rocblas_double_complex B, int64_t ldb);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strsm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const float alpha, *const *const float A, int64_t lda, *const *mut float B, int64_t ldb, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrsm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const double alpha, *const *const double A, int64_t lda, *const *mut double B, int64_t ldb, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrsm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *mut rocblas_float_complex B, int64_t ldb, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrsm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *mut rocblas_double_complex B, int64_t ldb, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_strsm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const float alpha, *const float A, int64_t lda, rocblas_stride stride_a, *mut float B, int64_t ldb, rocblas_stride stride_b, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_dtrsm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const double alpha, *const double A, int64_t lda, rocblas_stride stride_a, *mut double B, int64_t ldb, rocblas_stride stride_b, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ctrsm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_a, *mut rocblas_float_complex B, int64_t ldb, rocblas_stride stride_b, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_ztrsm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_a, *mut rocblas_double_complex B, int64_t ldb, rocblas_stride stride_b, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chemm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhemm_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chemm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const rocblas_float_complex beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhemm_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const rocblas_double_complex beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_chemm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_float_complex beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zhemm_strided_batched_64(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, int64_t m, int64_t n, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_B, *const rocblas_double_complex beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cherk_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const float alpha, *const rocblas_float_complex A, int64_t lda, *const float beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zherk_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const double alpha, *const rocblas_double_complex A, int64_t lda, *const double beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cherk_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const float alpha, *const *const rocblas_float_complex A, int64_t lda, *const float beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zherk_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const double alpha, *const *const rocblas_double_complex A, int64_t lda, *const double beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cherk_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const float alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const float beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zherk_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, int64_t n, int64_t k, *const double alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const double beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher2k_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *const float beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher2k_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *const double beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher2k_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const float beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher2k_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const double beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cher2k_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_B, *const float beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zher2k_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_B, *const double beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cherkx_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, *const rocblas_float_complex B, int64_t ldb, *const float beta, *mut rocblas_float_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zherkx_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, *const rocblas_double_complex B, int64_t ldb, *const double beta, *mut rocblas_double_complex C, int64_t ldc);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cherkx_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const *const rocblas_float_complex A, int64_t lda, *const *const rocblas_float_complex B, int64_t ldb, *const float beta, *const *mut rocblas_float_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zherkx_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const *const rocblas_double_complex A, int64_t lda, *const *const rocblas_double_complex B, int64_t ldb, *const double beta, *const *mut rocblas_double_complex C, int64_t ldc, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_cherkx_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_float_complex alpha, *const rocblas_float_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_float_complex B, int64_t ldb, rocblas_stride stride_B, *const float beta, *mut rocblas_float_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    #[since = 6 .1, "6.1"]
    rocblas_status rocblas_zherkx_strided_batched_64(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, int64_t n, int64_t k, *const rocblas_double_complex alpha, *const rocblas_double_complex A, int64_t lda, rocblas_stride stride_A, *const rocblas_double_complex B, int64_t ldb, rocblas_stride stride_B, *const double beta, *mut rocblas_double_complex C, int64_t ldc, rocblas_stride stride_C, int64_t batch_count);

    // rocBLAS extension

    // rocblas_axpy_ex + batched, strided_batched
    rocblas_status rocblas_axpy_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_axpy_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_axpy_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stridex,
        *mut void y,
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
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *const void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *mut void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_dot_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *const void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_int batch_count,
        *mut void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_dot_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        *const void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stride_y,
        rocblas_int batch_count,
        *mut void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    // rocblas_nrm2_ex + batched, strided_batched
    rocblas_status rocblas_nrm2_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_nrm2_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_int batch_count,
        *mut void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_nrm2_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        rocblas_int batch_count,
        *mut void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    // rocblas_rot_ex + batched, strided_batched
    rocblas_status rocblas_rot_ex(
        rocblas_handle handle,
        rocblas_int n,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *const void c,
        *const void s,
        rocblas_datatype cs_type,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_rot_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *const void c,
        *const void s,
        rocblas_datatype cs_type,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_rot_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stride_y,
        *const void c,
        *const void s,
        rocblas_datatype cs_type,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    // rocblas_scal_ex + batched, strided_batched
    rocblas_status rocblas_scal_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_scal_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    rocblas_status rocblas_scal_strided_batched_ex(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *mut void x,
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
        *const void alpha,
        *const void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const void beta,
        *const void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        int32_t solution_index,
        uint32_t flags,
    );

    rocblas_status rocblas_gemm_batched_ex(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const void alpha,
        *const void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const void beta,
        *const void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_int batch_count,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        int32_t solution_index,
        uint32_t flags,
    );

    rocblas_status rocblas_gemm_strided_batched_ex(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const void alpha,
        *const void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const void beta,
        *const void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        rocblas_stride stride_c,
        *mut void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_stride stride_d,
        rocblas_int batch_count,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        int32_t solution_index,
        uint32_t flags,
    );

    /*
    // rocblas_gemm_grouped_batched_ex
    rocblas_status rocblas_gemm_grouped_batched_ex(
        rocblas_handle handle,
        *const rocblas_operation transa_array,
        *const rocblas_operation transb_array,
        *const rocblas_int m_array,
        *const rocblas_int n_array,
        *const rocblas_int k_array,
        *const void alpha_array,
        *const *const void Aarray,
        rocblas_datatype a_type,
        *const rocblas_int lda_array,
        *const *const void Barray,
        rocblas_datatype b_type,
        *const rocblas_int ldb_array,
        *const void beta_array,
        *const *const void Carray,
        rocblas_datatype c_type,
        *const rocblas_int ldc_array,
        *const *mut void Darray,
        rocblas_datatype d_type,
        *const rocblas_int ldd_array,
        rocblas_int group_count,
        *const rocblas_int group_size,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        uint32_t flags,
    );
    */

    // rocblas_syrk_ex
    rocblas_status rocblas_syrk_ex(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_int n,
        rocblas_int k,
        *const void alpha,
        *const void A,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const void beta,
        *mut void C,
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
        *const void alpha,
        *const void A,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const void beta,
        *mut void C,
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
        *const void alpha,
        *const void A,
        rocblas_int lda,
        *mut void B,
        rocblas_int ldb,
        *const void invA,
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
        *const void alpha,
        *const void A,
        rocblas_int lda,
        *mut void B,
        rocblas_int ldb,
        rocblas_int batch_count,
        *const void invA,
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
        *const void alpha,
        *const void A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *mut void B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        rocblas_int batch_count,
        *const void invA,
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
        *const float alpha,
        *const float A,
        rocblas_int lda,
        *const float beta,
        *const float B,
        rocblas_int ldb,
        float C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_dgeam(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const double alpha,
        *const double A,
        rocblas_int lda,
        *const double beta,
        *const double B,
        rocblas_int ldb,
        *mut double C,
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
        *const float alpha,
        *const *const float A,
        rocblas_int lda,
        *const float beta,
        *const *const float B,
        rocblas_int ldb,
        *const *mut float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_dgeam_batched(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const double alpha,
        *const *const double A,
        rocblas_int lda,
        *const double beta,
        *const *const double B,
        rocblas_int ldb,
        *const *mut double C,
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
        *const float alpha,
        *const float A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const float beta,
        *const float B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut float C,
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
        *const double alpha,
        *const double A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const double beta,
        *const double B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut double C,
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
        *const void alpha,
        *const void A,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const void B,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const void beta,
        *const void C,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut void D,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_datatype compute_type,
        rocblas_geam_ex_operation geam_ex_op,
    );

    // rocblas_Xdgmm + batched, strided_batched
    rocblas_status rocblas_sdgmm(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const float A,
        rocblas_int lda,
        *const float x,
        rocblas_int incx,
        *mut float C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_ddgmm(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const double A,
        rocblas_int lda,
        *const double x,
        rocblas_int incx,
        double C,
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
        *const *const float A,
        rocblas_int lda,
        *const *const float x,
        rocblas_int incx,
        *const float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_ddgmm_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const *const double A,
        rocblas_int lda,
        *const *const double x,
        rocblas_int incx,
        *const *mut double C,
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
        *const float A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const float x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut float C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    rocblas_status rocblas_ddgmm_strided_batched(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const double A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const double x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut double C,
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
        *const float alpha,
        *const float A,
        rocblas_int lda,
        *const float B,
        rocblas_int ldb,
        *const float beta,
        *mut float C,
        rocblas_int ldc,
    );

    rocblas_status rocblas_dgemmt(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const double alpha,
        *const double A,
        rocblas_int lda,
        *const double B,
        rocblas_int ldb,
        *const double beta,
        *mut double C,
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
        *const float alpha,
        *const *const float A,
        rocblas_int lda,
        *const *const float B,
        rocblas_int ldb,
        *const float beta,
        *const *mut float C,
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
        *const double alpha,
        *const *const double A,
        rocblas_int lda,
        *const *const double B,
        rocblas_int ldb,
        *const double beta,
        *const *mut double C,
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
        *const float alpha,
        *const float A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const float B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const float beta,
        *mut float C,
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
        *const double alpha,
        *const double A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const double B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const double beta,
        *mut double C,
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

    // rocblas_axpy_ex + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_axpy_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_axpy_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_axpy_strided_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stridex,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stridey,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    // rocblas_dot_ex + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dot_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *const void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *mut void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dot_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *const void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_int batch_count,
        *mut void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dot_strided_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        *const void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stride_y,
        rocblas_int batch_count,
        *mut void result,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    // rocblas_nrm2_ex + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_nrm2_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_nrm2_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_int batch_count,
        *mut void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_nrm2_strided_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        rocblas_int batch_count,
        *mut void results,
        rocblas_datatype result_type,
        rocblas_datatype execution_type,
    );

    // rocblas_rot_ex + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_rot_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *const void c,
        *const void s,
        rocblas_datatype cs_type,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_rot_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        *const void c,
        *const void s,
        rocblas_datatype cs_type,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_rot_strided_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut void y,
        rocblas_datatype y_type,
        rocblas_int incy,
        rocblas_stride stride_y,
        *const void c,
        *const void s,
        rocblas_datatype cs_type,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    // rocblas_scal_ex + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_scal_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_scal_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_scal_strided_batched_ex_64(
        rocblas_handle handle,
        rocblas_int n,
        *const void alpha,
        rocblas_datatype alpha_type,
        *mut void x,
        rocblas_datatype x_type,
        rocblas_int incx,
        rocblas_stride stridex,
        rocblas_int batch_count,
        rocblas_datatype execution_type,
    );

    // rocblas_gemm_ex + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_gemm_ex_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const void alpha,
        *const void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const void beta,
        *const void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        int32_t solution_index,
        uint32_t flags,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_gemm_batched_ex_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const void alpha,
        *const void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        *const void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        *const void beta,
        *const void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        *mut void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_int batch_count,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        int32_t solution_index,
        uint32_t flags,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_gemm_strided_batched_ex_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        *const void alpha,
        *const void a,
        rocblas_datatype a_type,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const void b,
        rocblas_datatype b_type,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const void beta,
        *const void c,
        rocblas_datatype c_type,
        rocblas_int ldc,
        rocblas_stride stride_c,
        *mut void d,
        rocblas_datatype d_type,
        rocblas_int ldd,
        rocblas_stride stride_d,
        rocblas_int batch_count,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        int32_t solution_index,
        uint32_t flags,
    );

    /*
    // rocblas_gemm_grouped_batched_ex
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_gemm_grouped_batched_ex_64(
        rocblas_handle handle,
        *const rocblas_operation transa_array,
        *const rocblas_operation transb_array,
        *const rocblas_int m_array,
        *const rocblas_int n_array,
        *const rocblas_int k_array,
        *const void alpha_array,
        *const *const void Aarray,
        rocblas_datatype a_type,
        *const rocblas_int lda_array,
        *const *const void Barray,
        rocblas_datatype b_type,
        *const rocblas_int ldb_array,
        *const void beta_array,
        *const *const void Carray,
        rocblas_datatype c_type,
        *const rocblas_int ldc_array,
        *const *mut void Darray,
        rocblas_datatype d_type,
        *const rocblas_int ldd_array,
        rocblas_int group_count,
        *const rocblas_int group_size,
        rocblas_datatype compute_type,
        rocblas_gemm_algo algo,
        uint32_t flags,
    );
    */

    // rocblas_Xgeam + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sgeam_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const float alpha,
        *const float A,
        rocblas_int lda,
        *const float beta,
        *const float B,
        rocblas_int ldb,
        float C,
        rocblas_int ldc,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dgeam_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const double alpha,
        *const double A,
        rocblas_int lda,
        *const double beta,
        *const double B,
        rocblas_int ldb,
        *mut double C,
        rocblas_int ldc,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cgeam_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zgeam_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sgeam_batched_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const float alpha,
        *const *const float A,
        rocblas_int lda,
        *const float beta,
        *const *const float B,
        rocblas_int ldb,
        *const *mut float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dgeam_batched_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const double alpha,
        *const *const double A,
        rocblas_int lda,
        *const double beta,
        *const *const double B,
        rocblas_int ldb,
        *const *mut double C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cgeam_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zgeam_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sgeam_strided_batched_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const float alpha,
        *const float A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const float beta,
        *const float B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut float C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dgeam_strided_batched_64(
        rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        *const double alpha,
        *const double A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const double beta,
        *const double B,
        rocblas_int ldb,
        rocblas_stride stride_B,
        *mut double C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cgeam_strided_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zgeam_strided_batched_64(
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

    // rocblas_Xdgmm + batched, strided_batched
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sdgmm_64(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const float A,
        rocblas_int lda,
        *const float x,
        rocblas_int incx,
        *mut float C,
        rocblas_int ldc,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_ddgmm_64(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const double A,
        rocblas_int lda,
        *const double x,
        rocblas_int incx,
        double C,
        rocblas_int ldc,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cdgmm_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zdgmm_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sdgmm_batched_64(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const *const float A,
        rocblas_int lda,
        *const *const float x,
        rocblas_int incx,
        *const float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_ddgmm_batched_64(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const *const double A,
        rocblas_int lda,
        *const *const double x,
        rocblas_int incx,
        *const *mut double C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cdgmm_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zdgmm_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sdgmm_strided_batched_64(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const float A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const float x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut float C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_ddgmm_strided_batched_64(
        rocblas_handle handle,
        rocblas_side side,
        rocblas_int m,
        rocblas_int n,
        *const double A,
        rocblas_int lda,
        rocblas_stride stride_A,
        *const double x,
        rocblas_int incx,
        rocblas_stride stride_x,
        *mut double C,
        rocblas_int ldc,
        rocblas_stride stride_C,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cdgmm_strided_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zdgmm_strided_batched_64(
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
    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sgemmt_64(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const float alpha,
        *const float A,
        rocblas_int lda,
        *const float B,
        rocblas_int ldb,
        *const float beta,
        *mut float C,
        rocblas_int ldc,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dgemmt_64(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const double alpha,
        *const double A,
        rocblas_int lda,
        *const double B,
        rocblas_int ldb,
        *const double beta,
        *mut double C,
        rocblas_int ldc,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cgemmt_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zgemmt_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sgemmt_batched_64(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const float alpha,
        *const *const float A,
        rocblas_int lda,
        *const *const float B,
        rocblas_int ldb,
        *const float beta,
        *const *mut float C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dgemmt_batched_64(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const double alpha,
        *const *const double A,
        rocblas_int lda,
        *const *const double B,
        rocblas_int ldb,
        *const double beta,
        *const *mut double C,
        rocblas_int ldc,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cgemmt_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zgemmt_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_sgemmt_strided_batched_64(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const float alpha,
        *const float A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const float B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const float beta,
        *mut float C,
        rocblas_int ldc,
        rocblas_stride stride_c,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_dgemmt_strided_batched_64(
        rocblas_handle handle,
        rocblas_fill uplo,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int n,
        rocblas_int k,
        *const double alpha,
        *const double A,
        rocblas_int lda,
        rocblas_stride stride_a,
        *const double B,
        rocblas_int ldb,
        rocblas_stride stride_b,
        *const double beta,
        *mut double C,
        rocblas_int ldc,
        rocblas_stride stride_c,
        rocblas_int batch_count,
    );

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_cgemmt_strided_batched_64(
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

    #[since = 6 .3, "6.3"]
    rocblas_status rocblas_zgemmt_strided_batched_64(
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
