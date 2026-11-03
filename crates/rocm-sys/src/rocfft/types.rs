#[repr(C)]
#[derive(Debug)]
pub struct rocfft_plan_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to plan structure\n  @details This type is used to declare a plan handle that can be initialized\n with ::rocfft_plan_create."]
pub type rocfft_plan = *mut rocfft_plan_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_plan_description_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to plan description structure\n  @details This type is used to declare a plan description handle that can be\n initialized with ::rocfft_plan_description_create."]
pub type rocfft_plan_description = *mut rocfft_plan_description_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_execution_info_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to execution info structure\n  @details This type is used to declare an execution info handle that can be\n initialized with ::rocfft_execution_info_create."]
pub type rocfft_execution_info = *mut rocfft_execution_info_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_field_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to a rocFFT field structure.\n\n  @details rocFFT fields are used to hold data decomposition information which is then passed to a\n  \\ref rocfft_plan via a \\ref rocfft_plan_description\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
pub type rocfft_field = *mut rocfft_field_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_brick_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to a rocFFT brick structure.\n\n  @details rocFFT bricks are used to describe the data decomposition of fields.\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
pub type rocfft_brick = *mut rocfft_brick_t;
#[repr(i32)]
#[doc = " @brief rocFFT status/error codes"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocfft_status_e {
    rocfft_status_success = 0,
    rocfft_status_failure = 1,
    rocfft_status_invalid_arg_value = 2,
    rocfft_status_invalid_dimensions = 3,
    rocfft_status_invalid_array_type = 4,
    rocfft_status_invalid_strides = 5,
    rocfft_status_invalid_distance = 6,
    rocfft_status_invalid_offset = 7,
    rocfft_status_invalid_work_buffer = 8,
}
#[doc = " @brief rocFFT status/error codes"]
pub use self::rocfft_status_e as rocfft_status;
#[repr(i32)]
#[doc = " @brief Type of transform"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocfft_transform_type_e {
    rocfft_transform_type_complex_forward = 0,
    rocfft_transform_type_complex_inverse = 1,
    rocfft_transform_type_real_forward = 2,
    rocfft_transform_type_real_inverse = 3,
}
#[doc = " @brief Type of transform"]
pub use self::rocfft_transform_type_e as rocfft_transform_type;
#[repr(i32)]
#[doc = " @brief Precision"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocfft_precision_e {
    rocfft_precision_single = 0,
    rocfft_precision_double = 1,
    rocfft_precision_half = 2,
}
#[doc = " @brief Precision"]
pub use self::rocfft_precision_e as rocfft_precision;
#[repr(i32)]
#[doc = " @brief Result placement\n  @details Declares where the output of the transform should be\n  placed.  Note that input buffers may still be overwritten\n  during execution of a transform, even if the transform is not\n  in-place."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocfft_result_placement_e {
    rocfft_placement_inplace = 0,
    rocfft_placement_notinplace = 1,
}
#[doc = " @brief Result placement\n  @details Declares where the output of the transform should be\n  placed.  Note that input buffers may still be overwritten\n  during execution of a transform, even if the transform is not\n  in-place."]
pub use self::rocfft_result_placement_e as rocfft_result_placement;
#[repr(i32)]
#[doc = " @brief Array type"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocfft_array_type_e {
    rocfft_array_type_complex_interleaved = 0,
    rocfft_array_type_complex_planar = 1,
    rocfft_array_type_real = 2,
    rocfft_array_type_hermitian_interleaved = 3,
    rocfft_array_type_hermitian_planar = 4,
    rocfft_array_type_unset = 5,
}
#[doc = " @brief Array type"]
pub use self::rocfft_array_type_e as rocfft_array_type;
#[repr(i32)]
#[doc = " @brief Communicator type for distributed transforms"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum rocfft_comm_type_e {
    rocfft_comm_none = 0,
    rocfft_comm_mpi = 1,
}
#[doc = " @brief Communicator type for distributed transforms"]
pub use self::rocfft_comm_type_e as rocfft_comm_type;
