use super::types::*;
use rocm_include::ctype::*;

rocm_include::link! {
    hiprtc : rocm_ver;

    // In HIP, this function returns the name of the error, if the hiprtc result is defined, it will return “Invalid HIPRTC error code”
    *const char hiprtcGetErrorString(hiprtcResult result);

    hiprtcResult hiprtcVersion(
        *mut int major,
        *mut int minor,
    );

    hiprtcResult hiprtcAddNameExpression(
        hiprtcProgram prog,
        *const char name_expression,
    );

    hiprtcResult hiprtcCompileProgram(
        hiprtcProgram prog,
        int numOptions,
        *const char options,
    );

    hiprtcResult hiprtcCreateProgram(
        *mut hiprtcProgram prog,
        *const char src,
        *const char name,
        int numHeaders,
        *const *const char headers,
        *const *const char includeNames,
    );

    hiprtcResult hiprtcDestroyProgram(*mut hiprtcProgram prog);

    hiprtcResult hiprtcGetLoweredName(
        hiprtcProgram prog,
        *const char name_expression,
        *mut *const char lowered_name,
    );

    hiprtcResult hiprtcGetProgramLog(
        hiprtcProgram prog,
        *mut char log,
    );

    hiprtcResult hiprtcGetCode(
        hiprtcProgram prog,
        *mut char code,
    );

    hiprtcResult hiprtcGetCodeSize(
        hiprtcProgram prog,
        *mut size_t codeSizeRet,
    );

    hiprtcResult hiprtcGetBitcode(
        hiprtcProgram prog,
        *mut char bitcode,
    );

    hiprtcResult hiprtcGetBitcodeSize(
        hiprtcProgram prog,
        *mut size_t bitcode_size,
    );

    hiprtcResult hiprtcLinkCreate(
        uint num_options,
        *mut hipJitOption options_ptr,
        *mut *mut void option_vals_pptr,
        *mut hiprtcLinkState hip_link_state_ptr,
    );

    hiprtcResult hiprtcLinkAddFile(
        hiprtcLinkState hip_link_state,
        hipJitInputType input_type,
        *const char file_path,
        uint num_options,
        *mut hipJitOption options_ptr,
        *mut *mut void option_values,
    );

    hiprtcResult hiprtcLinkAddData(
        hiprtcLinkState hip_link_state,
        hipJitInputType input_type,
        *mut void image,
        size_t image_size,
        *const char name,
        uint num_options,
        *mut hipJitOption options_ptr,
        *mut *mut void option_values,
    );

    hiprtcResult hiprtcLinkComplete(
        hiprtcLinkState hip_link_state,
        *mut *mut void bin_out,
        *mut size_t size_out,
    );

    hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state);

    hiprtcResult hiprtcGetProgramLogSize(
        hiprtcProgram prog,
        *mut size_t logSizeRet,
    );
}
