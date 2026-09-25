use super::types::*;

use core::ffi::*;

link! {
    hiprtc : [7, 8, 9, 10] : rocm_ver;

    #[doc = "In HIP, this function returns the name of the error, if the hiprtc result is defined, it will return “Invalid HIPRTC error code”"]
    pub fn hiprtcGetErrorString(result: hiprtcResult) -> *const c_char;

    pub fn hiprtcVersion(
        major: *mut c_int,
        minor: *mut c_int,
    ) -> hiprtcResult;

    pub fn hiprtcAddNameExpression(
        prog: hiprtcProgram,
        name_expression: *const c_char,
    ) -> hiprtcResult;

    pub fn hiprtcCompileProgram(
        prog: hiprtcProgram,
        numOptions: c_int,
        options: *const c_char,
    ) -> hiprtcResult;

    pub fn hiprtcCreateProgram(
        prog: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        numHeaders: c_int,
        headers: *const *const c_char,
        includeNames: *const *const c_char,
    ) -> hiprtcResult;

    pub fn hiprtcDestroyProgram(prog: *mut hiprtcProgram) -> hiprtcResult;

    pub fn hiprtcGetLoweredName(
        prog: hiprtcProgram,
        name_expression: *const c_char,
        lowered_name: *mut *const c_char,
    ) -> hiprtcResult;

    pub fn hiprtcGetProgramLog(
        prog: hiprtcProgram,
        log: *mut c_char,
    ) -> hiprtcResult;

    pub fn hiprtcGetCode(
        prog: hiprtcProgram,
        code: *mut c_char,
    ) -> hiprtcResult;

    pub fn hiprtcGetCodeSize(
        prog: hiprtcProgram,
        codeSizeRet: *mut usize,
    ) -> hiprtcResult;

    pub fn hiprtcGetBitcode(
        prog: hiprtcProgram,
        bitcode: *mut c_char,
    ) -> hiprtcResult;

    pub fn hiprtcGetBitcodeSize(
        prog: hiprtcProgram,
        bitcode_size: *mut usize,
    ) -> hiprtcResult;

    pub fn hiprtcLinkCreate(
        num_options: c_uint,
        options_ptr: *mut hipJitOption,
        option_vals_pptr: *mut *mut c_void,
        hip_link_state_ptr: *mut hiprtcLinkState,
    ) -> hiprtcResult;

    pub fn hiprtcLinkAddFile(
        hip_link_state: hiprtcLinkState,
        input_type: hipJitInputType,
        file_path: *const c_char,
        num_options: c_uint,
        options_ptr: *mut hipJitOption,
        option_values: *mut *mut c_void,
    ) -> hiprtcResult;

    pub fn hiprtcLinkAddData(
        hip_link_state: hiprtcLinkState,
        input_type: hipJitInputType,
        image: *mut c_void,
        image_size: usize,
        name: *const c_char,
        num_options: c_uint,
        options_ptr: *mut hipJitOption,
        option_values: *mut *mut c_void,
    ) -> hiprtcResult;

    pub fn hiprtcLinkComplete(
        hip_link_state: hiprtcLinkState,
        bin_out: *mut *mut c_void,
        size_out: *mut usize,
    ) -> hiprtcResult;

    pub fn hiprtcLinkDestroy(hip_link_state: hiprtcLinkState) -> hiprtcResult;

    pub fn hiprtcGetProgramLogSize(
        prog: hiprtcProgram,
        logSizeRet: *mut usize,
    ) -> hiprtcResult;
}
