use super::types::*;
use crate::internal::*;

// #[cfg(feature = "dynamic-loading")]
// use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

link! {
    rocfft : rocm_ver;

    rocfft_status rocfft_setup();

    rocfft_status rocfft_cleanup();

    rocfft_status rocfft_plan_create(*mut rocfft_plan plan, rocfft_result_placement placement, rocfft_transform_type transform_type, rocfft_precision precision, size_t dimensions, *const size_t lengths, size_t number_of_transforms, rocfft_plan_description description);

    rocfft_status rocfft_execute(rocfft_plan plan, *mut void in_buffer, *mut *mut void out_buffer, rocfft_execution_info info);

    rocfft_status rocfft_plan_destroy(rocfft_plan plan);

    rocfft_status rocfft_plan_description_set_scale_factor(rocfft_plan_description description, double scale_factor);

    rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description, rocfft_array_type in_array_type, rocfft_array_type out_array_type, *const size_t in_offsets, *const size_t out_offsets, size_t in_strides_size, *const size_t in_strides, size_t in_distance, size_t out_strides_size, *const size_t out_strides, size_t out_distance);

    rocfft_status rocfft_field_create(*mut rocfft_field field);

    rocfft_status rocfft_field_destroy(rocfft_field field);

    rocfft_status rocfft_get_version_string(*mut char buf, size_t len);

    rocfft_status rocfft_plan_description_set_comm(rocfft_plan_description description, rocfft_comm_type comm_type, *mut void comm_handle);

    rocfft_status rocfft_plan_description_set_load_callback(rocfft_plan_description description, *const char symbol_name, *const void bitcode_data, size_t bitcode_len_bytes, size_t shared_mem_bytes);

    rocfft_status rocfft_plan_description_set_store_callback(rocfft_plan_description description, *const char symbol_name, *const void bitcode_data, size_t bitcode_len_bytes, size_t shared_mem_bytes);

    rocfft_status rocfft_brick_create(*mut rocfft_brick brick, *const size_t field_lower, *const size_t field_upper, *const size_t brick_stride, size_t dim_with_batch, int deviceID);

    rocfft_status rocfft_brick_destroy(rocfft_brick brick);

    rocfft_status rocfft_field_add_brick(rocfft_field field, rocfft_brick brick);

    rocfft_status rocfft_plan_description_add_infield(rocfft_plan_description description, rocfft_field field);

    rocfft_status rocfft_plan_description_add_outfield(rocfft_plan_description description, rocfft_field field);

    rocfft_status rocfft_plan_get_work_buffer_size(rocfft_plan plan, *mut size_t size_in_bytes);

    rocfft_status rocfft_plan_get_print(rocfft_plan plan);

    rocfft_status rocfft_plan_description_create(*mut rocfft_plan_description description);

    rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description);

    rocfft_status rocfft_execution_info_create(*mut rocfft_execution_info info);

    rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info);

    rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info info, *mut void work_buffer, size_t size_in_bytes);

    rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info, *mut void stream);

    rocfft_status rocfft_execution_info_set_load_callback(rocfft_execution_info info, *mut *mut void cb_functions, *mut *mut void cb_data, size_t shared_mem_bytes);

    rocfft_status rocfft_execution_info_set_load_callback_data(rocfft_execution_info info, *mut *mut void cb_data, size_t count);

    rocfft_status rocfft_execution_info_set_store_callback(rocfft_execution_info info, *mut *mut void cb_functions, *mut *mut void cb_data, size_t shared_mem_bytes);

    rocfft_status rocfft_execution_info_set_store_callback_data(rocfft_execution_info info, *mut *mut void cb_data, size_t count);

    rocfft_status rocfft_cache_serialize(*mut *mut void buffer, *mut size_t buffer_len_bytes);

    rocfft_status rocfft_cache_buffer_free(*mut void buffer);

    rocfft_status rocfft_cache_deserialize(*const void buffer, size_t buffer_len_bytes);
}
