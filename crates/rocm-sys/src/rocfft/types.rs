#[repr(C)]
#[derive(Debug)]
pub struct rocfft_plan_t {
    _unused: [u8; 0],
}
/// Pointer type to plan structure\n  This type is used to declare a plan handle that can be initialized\n with ::rocfft_plan_create.
pub type rocfft_plan = *mut rocfft_plan_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_plan_description_t {
    _unused: [u8; 0],
}
/// Pointer type to plan description structure\n  This type is used to declare a plan description handle that can be\n initialized with ::rocfft_plan_description_create.
pub type rocfft_plan_description = *mut rocfft_plan_description_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_execution_info_t {
    _unused: [u8; 0],
}
/// Pointer type to execution info structure\n  This type is used to declare an execution info handle that can be\n initialized with ::rocfft_execution_info_create.
pub type rocfft_execution_info = *mut rocfft_execution_info_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_field_t {
    _unused: [u8; 0],
}
/// Pointer type to a rocFFT field structure.\n\n  rocFFT fields are used to hold data decomposition information which is then passed to a\n  \\ref rocfft_plan via a \\ref rocfft_plan_description\n\n  @warning Experimental!  This feature is part of an experimental API preview.
pub type rocfft_field = *mut rocfft_field_t;
#[repr(C)]
#[derive(Debug)]
pub struct rocfft_brick_t {
    _unused: [u8; 0],
}
/// Pointer type to a rocFFT brick structure.\n\n  rocFFT bricks are used to describe the data decomposition of fields.\n\n  @warning Experimental!  This feature is part of an experimental API preview.
pub type rocfft_brick = *mut rocfft_brick_t;
impl rocfft_status_e {
    pub const rocfft_status_success: rocfft_status_e = rocfft_status_e(0);
    pub const rocfft_status_failure: rocfft_status_e = rocfft_status_e(1);
    pub const rocfft_status_invalid_arg_value: rocfft_status_e = rocfft_status_e(2);
    pub const rocfft_status_invalid_dimensions: rocfft_status_e = rocfft_status_e(3);
    pub const rocfft_status_invalid_array_type: rocfft_status_e = rocfft_status_e(4);
    pub const rocfft_status_invalid_strides: rocfft_status_e = rocfft_status_e(5);
    pub const rocfft_status_invalid_distance: rocfft_status_e = rocfft_status_e(6);
    pub const rocfft_status_invalid_offset: rocfft_status_e = rocfft_status_e(7);
    pub const rocfft_status_invalid_work_buffer: rocfft_status_e = rocfft_status_e(8);
}
#[repr(transparent)]
/// rocFFT status/error codes
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocfft_status_e(::core::ffi::c_int);
/// rocFFT status/error codes
pub use self::rocfft_status_e as rocfft_status;
impl rocfft_transform_type_e {
    pub const rocfft_transform_type_complex_forward: rocfft_transform_type_e =
        rocfft_transform_type_e(0);
    pub const rocfft_transform_type_complex_inverse: rocfft_transform_type_e =
        rocfft_transform_type_e(1);
    pub const rocfft_transform_type_real_forward: rocfft_transform_type_e =
        rocfft_transform_type_e(2);
    pub const rocfft_transform_type_real_inverse: rocfft_transform_type_e =
        rocfft_transform_type_e(3);
}
#[repr(transparent)]
/// Type of transform
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocfft_transform_type_e(pub ::core::ffi::c_int);
/// Type of transform
pub use self::rocfft_transform_type_e as rocfft_transform_type;
impl rocfft_precision_e {
    pub const rocfft_precision_single: rocfft_precision_e = rocfft_precision_e(0);
    pub const rocfft_precision_double: rocfft_precision_e = rocfft_precision_e(1);
    pub const rocfft_precision_half: rocfft_precision_e = rocfft_precision_e(2);
}
#[repr(transparent)]
/// Precision
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocfft_precision_e(pub ::core::ffi::c_int);
/// Precision
pub use self::rocfft_precision_e as rocfft_precision;
impl rocfft_result_placement_e {
    pub const rocfft_placement_inplace: rocfft_result_placement_e = rocfft_result_placement_e(0);
    pub const rocfft_placement_notinplace: rocfft_result_placement_e = rocfft_result_placement_e(1);
}
#[repr(transparent)]
/// Result placement\n  Declares where the output of the transform should be\n  placed.  Note that input buffers may still be overwritten\n  during execution of a transform, even if the transform is not\n  in-place.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocfft_result_placement_e(pub ::core::ffi::c_int);
/// Result placement\n  Declares where the output of the transform should be\n  placed.  Note that input buffers may still be overwritten\n  during execution of a transform, even if the transform is not\n  in-place.
pub use self::rocfft_result_placement_e as rocfft_result_placement;
impl rocfft_array_type_e {
    pub const rocfft_array_type_complex_interleaved: rocfft_array_type_e = rocfft_array_type_e(0);
    pub const rocfft_array_type_complex_planar: rocfft_array_type_e = rocfft_array_type_e(1);
    pub const rocfft_array_type_real: rocfft_array_type_e = rocfft_array_type_e(2);
    pub const rocfft_array_type_hermitian_interleaved: rocfft_array_type_e = rocfft_array_type_e(3);
    pub const rocfft_array_type_hermitian_planar: rocfft_array_type_e = rocfft_array_type_e(4);
    pub const rocfft_array_type_unset: rocfft_array_type_e = rocfft_array_type_e(5);
}
#[repr(transparent)]
/// Array type
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocfft_array_type_e(pub ::core::ffi::c_int);
/// Array type
pub use self::rocfft_array_type_e as rocfft_array_type;
impl rocfft_comm_type_e {
    pub const rocfft_comm_none: rocfft_comm_type_e = rocfft_comm_type_e(0);
    pub const rocfft_comm_mpi: rocfft_comm_type_e = rocfft_comm_type_e(1);
}
#[repr(transparent)]
/// Communicator type for distributed transforms
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct rocfft_comm_type_e(pub ::core::ffi::c_int);
/// Communicator type for distributed transforms
pub use self::rocfft_comm_type_e as rocfft_comm_type;
