use super::types::*;

use core::ffi::*;

link! {
    hiprtc : [7, 8, 9, 10] : rocm_ver;

    // In HIP, this function returns the name of the error, if the hiprtc result is defined, it will return “Invalid HIPRTC error code”
    *const c_char hiprtcGetErrorString(hiprtcResult result);

    hiprtcResult hiprtcVersion(
        *mut c_int major,
        *mut c_int minor,
    );

    hiprtcResult hiprtcAddNameExpression(
        hiprtcProgram prog,
        *const c_char name_expression,
    );

    hiprtcResult hiprtcCompileProgram(
        hiprtcProgram prog,
        c_int numOptions,
        *const c_char options,
    );

    hiprtcResult hiprtcCreateProgram(
        *mut hiprtcProgram prog,
        *const c_char src,
        *const c_char name,
        c_int numHeaders,
        *const *const c_char headers,
        *const *const c_char includeNames,
    );

    hiprtcResult hiprtcDestroyProgram(*mut hiprtcProgram prog);

    hiprtcResult hiprtcGetLoweredName(
        hiprtcProgram prog,
        *const c_char name_expression,
        *mut *const c_char lowered_name,
    );

    hiprtcResult hiprtcGetProgramLog(
        hiprtcProgram prog,
        *mut c_char log,
    );

    hiprtcResult hiprtcGetCode(
        hiprtcProgram prog,
        *mut c_char code,
    );

    hiprtcResult hiprtcGetCodeSize(
        hiprtcProgram prog,
        *mut usize codeSizeRet,
    );

    hiprtcResult hiprtcGetBitcode(
        hiprtcProgram prog,
        *mut c_char bitcode,
    );

    hiprtcResult hiprtcGetBitcodeSize(
        hiprtcProgram prog,
        *mut usize bitcode_size,
    );

    hiprtcResult hiprtcLinkCreate(
        c_uint num_options,
        *mut hipJitOption options_ptr,
        *mut *mut c_void option_vals_pptr,
        *mut hiprtcLinkState hip_link_state_ptr,
    );

    hiprtcResult hiprtcLinkAddFile(
        hiprtcLinkState hip_link_state,
        hipJitInputType input_type,
        *const c_char file_path,
        c_uint num_options,
        *mut hipJitOption options_ptr,
        *mut *mut c_void option_values,
    );

    hiprtcResult hiprtcLinkAddData(
        hiprtcLinkState hip_link_state,
        hipJitInputType input_type,
        *mut c_void image,
        usize image_size,
        *const c_char name,
        c_uint num_options,
        *mut hipJitOption options_ptr,
        *mut *mut c_void option_values,
    );

    hiprtcResult hiprtcLinkComplete(
        hiprtcLinkState hip_link_state,
        *mut *mut c_void bin_out,
        *mut usize size_out,
    );

    hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state);

    hiprtcResult hiprtcGetProgramLogSize(
        hiprtcProgram prog,
        *mut usize logSizeRet,
    );
}
