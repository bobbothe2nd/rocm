pub const MIOPEN_HIP_COMPILER: &[u8; 36] = b"/usr/lib64/rocm/llvm/bin/amdclang++\0";
pub const MIOPEN_CACHE_DIR: &[u8; 17] = b"~/.cache/miopen/\0";
pub type miopenAcceleratorQueue_t = crate::hip::hipStream_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenHandle {
    pub _address: u8,
}
pub type miopenHandle_t = *mut miopenHandle;
pub const miopenStatus_t_miopenStatusSuccess: miopenStatus_t = 0;
pub const miopenStatus_t_miopenStatusNotInitialized: miopenStatus_t = 1;
pub const miopenStatus_t_miopenStatusInvalidValue: miopenStatus_t = 2;
pub const miopenStatus_t_miopenStatusBadParm: miopenStatus_t = 3;
pub const miopenStatus_t_miopenStatusAllocFailed: miopenStatus_t = 4;
pub const miopenStatus_t_miopenStatusInternalError: miopenStatus_t = 5;
pub const miopenStatus_t_miopenStatusNotImplemented: miopenStatus_t = 6;
pub const miopenStatus_t_miopenStatusUnknownError: miopenStatus_t = 7;
pub const miopenStatus_t_miopenStatusUnsupportedOp: miopenStatus_t = 8;
pub const miopenStatus_t_miopenStatusGpuOperationsSkipped: miopenStatus_t = 9;
pub const miopenStatus_t_miopenStatusVersionMismatch: miopenStatus_t = 10;
pub type miopenStatus_t = ::std::os::raw::c_uint;
pub type miopenAllocatorFunction = ::std::option::Option<
    unsafe extern "C" fn(
        context: *mut ::std::os::raw::c_void,
        sizeBytes: usize,
    ) -> *mut ::std::os::raw::c_void,
>;
pub type miopenDeallocatorFunction = ::std::option::Option<
    unsafe extern "C" fn(context: *mut ::std::os::raw::c_void, memory: *mut ::std::os::raw::c_void),
>;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenFusionOpDescriptor {
    pub _address: u8,
}
pub type miopenFusionOpDescriptor_t = *mut miopenFusionOpDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenTensorDescriptor {
    pub _address: u8,
}
pub type miopenTensorDescriptor_t = *mut miopenTensorDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenSeqTensorDescriptor {
    pub _address: u8,
}
pub type miopenSeqTensorDescriptor_t = *mut miopenSeqTensorDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenConvolutionDescriptor {
    pub _address: u8,
}
pub type miopenConvolutionDescriptor_t = *mut miopenConvolutionDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenPoolingDescriptor {
    pub _address: u8,
}
pub type miopenPoolingDescriptor_t = *mut miopenPoolingDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenLRNDescriptor {
    pub _address: u8,
}
pub type miopenLRNDescriptor_t = *mut miopenLRNDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenActivationDescriptor {
    pub _address: u8,
}
pub type miopenActivationDescriptor_t = *mut miopenActivationDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenRNNDescriptor {
    pub _address: u8,
}
pub type miopenRNNDescriptor_t = *mut miopenRNNDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenCTCLossDescriptor {
    pub _address: u8,
}
pub type miopenCTCLossDescriptor_t = *mut miopenCTCLossDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenDropoutDescriptor {
    pub _address: u8,
}
pub type miopenDropoutDescriptor_t = *mut miopenDropoutDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenReduceTensorDescriptor {
    pub _address: u8,
}
pub type miopenReduceTensorDescriptor_t = *mut miopenReduceTensorDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenMhaDescriptor {
    pub _address: u8,
}
pub type miopenMhaDescriptor_t = *mut miopenMhaDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenSoftmaxDescriptor {
    pub _address: u8,
}
pub type miopenSoftmaxDescriptor_t = *mut miopenSoftmaxDescriptor;
pub const miopenDataType_t_miopenHalf: miopenDataType_t = 0;
pub const miopenDataType_t_miopenFloat: miopenDataType_t = 1;
pub const miopenDataType_t_miopenInt32: miopenDataType_t = 2;
pub const miopenDataType_t_miopenInt8: miopenDataType_t = 3;
pub const miopenDataType_t_miopenBFloat16: miopenDataType_t = 5;
pub const miopenDataType_t_miopenDouble: miopenDataType_t = 6;
pub const miopenDataType_t_miopenFloat8_fnuz: miopenDataType_t = 7;
pub const miopenDataType_t_miopenBFloat8_fnuz: miopenDataType_t = 8;
pub const miopenDataType_t_miopenInt64: miopenDataType_t = 9;
pub type miopenDataType_t = ::std::os::raw::c_uint;
pub const miopenTensorLayout_t_miopenTensorNCHW: miopenTensorLayout_t = 0;
pub const miopenTensorLayout_t_miopenTensorNHWC: miopenTensorLayout_t = 1;
pub const miopenTensorLayout_t_miopenTensorCHWN: miopenTensorLayout_t = 2;
pub const miopenTensorLayout_t_miopenTensorNCHWc4: miopenTensorLayout_t = 3;
pub const miopenTensorLayout_t_miopenTensorNCHWc8: miopenTensorLayout_t = 4;
pub const miopenTensorLayout_t_miopenTensorCHWNc4: miopenTensorLayout_t = 5;
pub const miopenTensorLayout_t_miopenTensorCHWNc8: miopenTensorLayout_t = 6;
pub const miopenTensorLayout_t_miopenTensorNCDHW: miopenTensorLayout_t = 7;
pub const miopenTensorLayout_t_miopenTensorNDHWC: miopenTensorLayout_t = 8;
pub type miopenTensorLayout_t = ::std::os::raw::c_uint;
pub const miopenIndexType_t_miopenIndexUint8: miopenIndexType_t = 0;
pub const miopenIndexType_t_miopenIndexUint16: miopenIndexType_t = 1;
pub const miopenIndexType_t_miopenIndexUint32: miopenIndexType_t = 2;
pub const miopenIndexType_t_miopenIndexUint64: miopenIndexType_t = 3;
pub type miopenIndexType_t = ::std::os::raw::c_uint;
pub const miopenTensorOp_t_miopenTensorOpAdd: miopenTensorOp_t = 0;
pub const miopenTensorOp_t_miopenTensorOpMul: miopenTensorOp_t = 1;
pub const miopenTensorOp_t_miopenTensorOpMin: miopenTensorOp_t = 2;
pub const miopenTensorOp_t_miopenTensorOpMax: miopenTensorOp_t = 3;
pub type miopenTensorOp_t = ::std::os::raw::c_uint;
pub const miopenConvolutionMode_t_miopenConvolution: miopenConvolutionMode_t = 0;
pub const miopenConvolutionMode_t_miopenTranspose: miopenConvolutionMode_t = 1;
pub const miopenConvolutionMode_t_miopenGroupConv: miopenConvolutionMode_t = 2;
pub const miopenConvolutionMode_t_miopenDepthwise: miopenConvolutionMode_t = 3;
pub type miopenConvolutionMode_t = ::std::os::raw::c_uint;
pub const miopenPaddingMode_t_miopenPaddingDefault: miopenPaddingMode_t = 0;
pub const miopenPaddingMode_t_miopenPaddingSame: miopenPaddingMode_t = 1;
pub const miopenPaddingMode_t_miopenPaddingValid: miopenPaddingMode_t = 2;
pub type miopenPaddingMode_t = ::std::os::raw::c_uint;
pub const miopenPoolingMode_t_miopenPoolingMax: miopenPoolingMode_t = 0;
pub const miopenPoolingMode_t_miopenPoolingAverage: miopenPoolingMode_t = 1;
pub const miopenPoolingMode_t_miopenPoolingAverageInclusive: miopenPoolingMode_t = 2;
pub type miopenPoolingMode_t = ::std::os::raw::c_uint;
pub const miopenPoolingWorkspaceIndexMode_t_miopenPoolingWorkspaceIndexMask:
    miopenPoolingWorkspaceIndexMode_t = 0;
pub const miopenPoolingWorkspaceIndexMode_t_miopenPoolingWorkspaceIndexImage:
    miopenPoolingWorkspaceIndexMode_t = 1;
pub type miopenPoolingWorkspaceIndexMode_t = ::std::os::raw::c_uint;
pub const miopenLRNMode_t_miopenLRNWithinChannel: miopenLRNMode_t = 0;
pub const miopenLRNMode_t_miopenLRNCrossChannel: miopenLRNMode_t = 1;
pub type miopenLRNMode_t = ::std::os::raw::c_uint;
pub const miopenBatchNormMode_t_miopenBNPerActivation: miopenBatchNormMode_t = 0;
pub const miopenBatchNormMode_t_miopenBNSpatial: miopenBatchNormMode_t = 1;
pub type miopenBatchNormMode_t = ::std::os::raw::c_uint;
pub const miopenActivationMode_t_miopenActivationPASTHRU: miopenActivationMode_t = 0;
pub const miopenActivationMode_t_miopenActivationLOGISTIC: miopenActivationMode_t = 1;
pub const miopenActivationMode_t_miopenActivationTANH: miopenActivationMode_t = 2;
pub const miopenActivationMode_t_miopenActivationRELU: miopenActivationMode_t = 3;
pub const miopenActivationMode_t_miopenActivationSOFTRELU: miopenActivationMode_t = 4;
pub const miopenActivationMode_t_miopenActivationABS: miopenActivationMode_t = 5;
pub const miopenActivationMode_t_miopenActivationPOWER: miopenActivationMode_t = 6;
pub const miopenActivationMode_t_miopenActivationCLIPPEDRELU: miopenActivationMode_t = 7;
pub const miopenActivationMode_t_miopenActivationLEAKYRELU: miopenActivationMode_t = 8;
pub const miopenActivationMode_t_miopenActivationELU: miopenActivationMode_t = 9;
pub const miopenActivationMode_t_miopenActivationCLAMP: miopenActivationMode_t = 10;
pub type miopenActivationMode_t = ::std::os::raw::c_uint;
pub const miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_FAST: miopenSoftmaxAlgorithm_t = 0;
pub const miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_ACCURATE: miopenSoftmaxAlgorithm_t = 1;
pub const miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_LOG: miopenSoftmaxAlgorithm_t = 2;
pub type miopenSoftmaxAlgorithm_t = ::std::os::raw::c_uint;
pub const miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_INSTANCE: miopenSoftmaxMode_t = 0;
pub const miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_CHANNEL: miopenSoftmaxMode_t = 1;
pub type miopenSoftmaxMode_t = ::std::os::raw::c_uint;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_ADD: miopenReduceTensorOp_t = 0;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MUL: miopenReduceTensorOp_t = 1;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MIN: miopenReduceTensorOp_t = 2;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MAX: miopenReduceTensorOp_t = 3;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_AMAX: miopenReduceTensorOp_t = 4;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_AVG: miopenReduceTensorOp_t = 5;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_NORM1: miopenReduceTensorOp_t = 6;
pub const miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_NORM2: miopenReduceTensorOp_t = 7;
pub type miopenReduceTensorOp_t = ::std::os::raw::c_uint;
pub const miopenNanPropagation_t_MIOPEN_NOT_PROPAGATE_NAN: miopenNanPropagation_t = 0;
pub const miopenNanPropagation_t_MIOPEN_PROPAGATE_NAN: miopenNanPropagation_t = 1;
pub type miopenNanPropagation_t = ::std::os::raw::c_uint;
pub const miopenReduceTensorIndices_t_MIOPEN_REDUCE_TENSOR_NO_INDICES: miopenReduceTensorIndices_t =
    0;
pub const miopenReduceTensorIndices_t_MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES:
    miopenReduceTensorIndices_t = 1;
pub type miopenReduceTensorIndices_t = ::std::os::raw::c_uint;
pub const miopenIndicesType_t_MIOPEN_32BIT_INDICES: miopenIndicesType_t = 0;
pub const miopenIndicesType_t_MIOPEN_64BIT_INDICES: miopenIndicesType_t = 1;
pub const miopenIndicesType_t_MIOPEN_16BIT_INDICES: miopenIndicesType_t = 2;
pub const miopenIndicesType_t_MIOPEN_8BIT_INDICES: miopenIndicesType_t = 3;
pub type miopenIndicesType_t = ::std::os::raw::c_uint;
pub const miopenConvolutionAttrib_t_MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL:
    miopenConvolutionAttrib_t = 0;
pub const miopenConvolutionAttrib_t_MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC:
    miopenConvolutionAttrib_t = 1;
pub type miopenConvolutionAttrib_t = ::std::os::raw::c_uint;
pub const miopenConvolutionFindMode_t_miopenConvolutionFindModeNormal: miopenConvolutionFindMode_t =
    1;
pub const miopenConvolutionFindMode_t_miopenConvolutionFindModeFast: miopenConvolutionFindMode_t =
    2;
pub const miopenConvolutionFindMode_t_miopenConvolutionFindModeHybrid: miopenConvolutionFindMode_t =
    3;
pub const miopenConvolutionFindMode_t_miopenConvolutionFindModeDynamicHybrid:
    miopenConvolutionFindMode_t = 5;
pub const miopenConvolutionFindMode_t_miopenConvolutionFindModeTrustVerify:
    miopenConvolutionFindMode_t = 6;
pub const miopenConvolutionFindMode_t_miopenConvolutionFindModeTrustVerifyFull:
    miopenConvolutionFindMode_t = 7;
pub const miopenConvolutionFindMode_t_miopenConvolutionFindModeDefault:
    miopenConvolutionFindMode_t = 5;
pub type miopenConvolutionFindMode_t = ::std::os::raw::c_uint;
pub const miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoGEMM: miopenConvFwdAlgorithm_t = 0;
pub const miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoDirect: miopenConvFwdAlgorithm_t = 1;
pub const miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoFFT: miopenConvFwdAlgorithm_t = 2;
pub const miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoWinograd: miopenConvFwdAlgorithm_t = 3;
pub const miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoImplicitGEMM: miopenConvFwdAlgorithm_t =
    5;
pub type miopenConvFwdAlgorithm_t = ::std::os::raw::c_uint;
pub const miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoGEMM:
    miopenConvBwdWeightsAlgorithm_t = 0;
pub const miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoDirect:
    miopenConvBwdWeightsAlgorithm_t = 1;
pub const miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoWinograd:
    miopenConvBwdWeightsAlgorithm_t = 3;
pub const miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoImplicitGEMM:
    miopenConvBwdWeightsAlgorithm_t = 5;
pub type miopenConvBwdWeightsAlgorithm_t = ::std::os::raw::c_uint;
pub const miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoGEMM:
    miopenConvBwdDataAlgorithm_t = 0;
pub const miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoDirect:
    miopenConvBwdDataAlgorithm_t = 1;
pub const miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoFFT:
    miopenConvBwdDataAlgorithm_t = 2;
pub const miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoWinograd:
    miopenConvBwdDataAlgorithm_t = 3;
pub const miopenConvBwdDataAlgorithm_t_miopenTransposeBwdDataAlgoGEMM:
    miopenConvBwdDataAlgorithm_t = 4;
pub const miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoImplicitGEMM:
    miopenConvBwdDataAlgorithm_t = 5;
pub type miopenConvBwdDataAlgorithm_t = ::std::os::raw::c_uint;
pub const miopenConvAlgorithm_t_miopenConvolutionAlgoGEMM: miopenConvAlgorithm_t = 0;
pub const miopenConvAlgorithm_t_miopenConvolutionAlgoDirect: miopenConvAlgorithm_t = 1;
pub const miopenConvAlgorithm_t_miopenConvolutionAlgoFFT: miopenConvAlgorithm_t = 2;
pub const miopenConvAlgorithm_t_miopenConvolutionAlgoWinograd: miopenConvAlgorithm_t = 3;
pub const miopenConvAlgorithm_t_miopenConvolutionAlgoImplicitGEMM: miopenConvAlgorithm_t = 5;
pub type miopenConvAlgorithm_t = ::std::os::raw::c_uint;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct miopenConvAlgoPerf_t {
    pub __bindgen_anon_1: miopenConvAlgoPerf_t__bindgen_ty_1,
    pub time: f32,
    pub memory: usize,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union miopenConvAlgoPerf_t__bindgen_ty_1 {
    pub fwd_algo: miopenConvFwdAlgorithm_t,
    pub bwd_weights_algo: miopenConvBwdWeightsAlgorithm_t,
    pub bwd_data_algo: miopenConvBwdDataAlgorithm_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenConvSolution_t {
    pub time: f32,
    pub workspace_size: usize,
    pub solution_id: u64,
    pub algorithm: miopenConvAlgorithm_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenFusionPlanDescriptor {
    pub _address: u8,
}
pub type miopenFusionPlanDescriptor_t = *mut miopenFusionPlanDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenOperatorDescriptor {
    pub _address: u8,
}
pub type miopenOperatorDescriptor_t = *mut miopenOperatorDescriptor;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenOperatorArgs {
    pub _address: u8,
}
pub type miopenOperatorArgs_t = *mut miopenOperatorArgs;
pub const miopenFusionDirection_t_miopenVerticalFusion: miopenFusionDirection_t = 0;
pub const miopenFusionDirection_t_miopenHorizontalFusion: miopenFusionDirection_t = 1;
pub type miopenFusionDirection_t = ::std::os::raw::c_uint;
pub const miopenRNNMode_t_miopenRNNRELU: miopenRNNMode_t = 0;
pub const miopenRNNMode_t_miopenRNNTANH: miopenRNNMode_t = 1;
pub const miopenRNNMode_t_miopenLSTM: miopenRNNMode_t = 2;
pub const miopenRNNMode_t_miopenGRU: miopenRNNMode_t = 3;
pub type miopenRNNMode_t = ::std::os::raw::c_uint;
pub const miopenRNNInputMode_t_miopenRNNlinear: miopenRNNInputMode_t = 0;
pub const miopenRNNInputMode_t_miopenRNNskip: miopenRNNInputMode_t = 1;
pub type miopenRNNInputMode_t = ::std::os::raw::c_uint;
pub const miopenRNNAlgo_t_miopenRNNdefault: miopenRNNAlgo_t = 0;
pub const miopenRNNAlgo_t_miopenRNNfundamental: miopenRNNAlgo_t = 1;
pub const miopenRNNAlgo_t_miopenRNNroundedDynamic: miopenRNNAlgo_t = 2;
pub type miopenRNNAlgo_t = ::std::os::raw::c_uint;
pub const miopenRNNDirectionMode_t_miopenRNNunidirection: miopenRNNDirectionMode_t = 0;
pub const miopenRNNDirectionMode_t_miopenRNNbidirection: miopenRNNDirectionMode_t = 1;
pub type miopenRNNDirectionMode_t = ::std::os::raw::c_uint;
pub const miopenRNNBiasMode_t_miopenRNNNoBias: miopenRNNBiasMode_t = 0;
pub const miopenRNNBiasMode_t_miopenRNNwithBias: miopenRNNBiasMode_t = 1;
pub type miopenRNNBiasMode_t = ::std::os::raw::c_uint;
pub const miopenRNNGEMMalgoMode_t_miopenRNNAlgoGEMM: miopenRNNGEMMalgoMode_t = 0;
pub type miopenRNNGEMMalgoMode_t = ::std::os::raw::c_uint;
pub const miopenRNNPaddingMode_t_miopenRNNIONotPadded: miopenRNNPaddingMode_t = 0;
pub const miopenRNNPaddingMode_t_miopenRNNIOWithPadding: miopenRNNPaddingMode_t = 1;
pub type miopenRNNPaddingMode_t = ::std::os::raw::c_uint;
pub const miopenRNNFWDMode_t_miopenRNNTraining: miopenRNNFWDMode_t = 0;
pub const miopenRNNFWDMode_t_miopenRNNInference: miopenRNNFWDMode_t = 1;
pub type miopenRNNFWDMode_t = ::std::os::raw::c_uint;
pub const miopenRNNBaseLayout_t_miopenRNNDataUnknownLayout: miopenRNNBaseLayout_t = 0;
pub const miopenRNNBaseLayout_t_miopenRNNDataSeqMajorNotPadded: miopenRNNBaseLayout_t = 1;
pub const miopenRNNBaseLayout_t_miopenRNNDataSeqMajorPadded: miopenRNNBaseLayout_t = 2;
pub const miopenRNNBaseLayout_t_miopenRNNDataBatchMajorPadded: miopenRNNBaseLayout_t = 3;
pub type miopenRNNBaseLayout_t = ::std::os::raw::c_uint;
pub const miopenCTCLossAlgo_t_MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC: miopenCTCLossAlgo_t = 0;
pub type miopenCTCLossAlgo_t = ::std::os::raw::c_uint;
pub const miopenRNGType_t_MIOPEN_RNG_PSEUDO_XORWOW: miopenRNGType_t = 0;
pub type miopenRNGType_t = ::std::os::raw::c_uint;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenProblem {
    pub _address: u8,
}
pub type miopenProblem_t = *mut miopenProblem;
pub const miopenProblemDirection_t_miopenProblemDirectionForward: miopenProblemDirection_t = 0;
pub const miopenProblemDirection_t_miopenProblemDirectionBackward: miopenProblemDirection_t = 1;
pub const miopenProblemDirection_t_miopenProblemDirectionBackwardWeights: miopenProblemDirection_t =
    2;
pub type miopenProblemDirection_t = ::std::os::raw::c_uint;
pub const miopenTensorArgumentId_t_miopenTensorArgumentIdInvalid: miopenTensorArgumentId_t = 0;
pub const miopenTensorArgumentId_t_miopenTensorConvolutionX: miopenTensorArgumentId_t = 1;
pub const miopenTensorArgumentId_t_miopenTensorConvolutionW: miopenTensorArgumentId_t = 2;
pub const miopenTensorArgumentId_t_miopenTensorConvolutionY: miopenTensorArgumentId_t = 3;
pub const miopenTensorArgumentId_t_miopenTensorMhaK: miopenTensorArgumentId_t = 4;
pub const miopenTensorArgumentId_t_miopenTensorMhaQ: miopenTensorArgumentId_t = 5;
pub const miopenTensorArgumentId_t_miopenTensorMhaV: miopenTensorArgumentId_t = 6;
pub const miopenTensorArgumentId_t_miopenTensorMhaDescaleK: miopenTensorArgumentId_t = 7;
pub const miopenTensorArgumentId_t_miopenTensorMhaDescaleQ: miopenTensorArgumentId_t = 8;
pub const miopenTensorArgumentId_t_miopenTensorMhaDescaleV: miopenTensorArgumentId_t = 9;
pub const miopenTensorArgumentId_t_miopenTensorMhaDescaleS: miopenTensorArgumentId_t = 10;
pub const miopenTensorArgumentId_t_miopenTensorMhaScaleS: miopenTensorArgumentId_t = 11;
pub const miopenTensorArgumentId_t_miopenTensorMhaScaleO: miopenTensorArgumentId_t = 12;
pub const miopenTensorArgumentId_t_miopenTensorMhaDropoutProbability: miopenTensorArgumentId_t = 13;
pub const miopenTensorArgumentId_t_miopenTensorMhaDropoutSeed: miopenTensorArgumentId_t = 14;
pub const miopenTensorArgumentId_t_miopenTensorMhaDropoutOffset: miopenTensorArgumentId_t = 15;
pub const miopenTensorArgumentId_t_miopenTensorMhaO: miopenTensorArgumentId_t = 16;
pub const miopenTensorArgumentId_t_miopenTensorMhaAmaxO: miopenTensorArgumentId_t = 17;
pub const miopenTensorArgumentId_t_miopenTensorMhaAmaxS: miopenTensorArgumentId_t = 18;
pub const miopenTensorArgumentId_t_miopenTensorMhaM: miopenTensorArgumentId_t = 19;
pub const miopenTensorArgumentId_t_miopenTensorMhaZInv: miopenTensorArgumentId_t = 20;
pub const miopenTensorArgumentId_t_miopenTensorMhaDO: miopenTensorArgumentId_t = 21;
pub const miopenTensorArgumentId_t_miopenTensorMhaDescaleO: miopenTensorArgumentId_t = 22;
pub const miopenTensorArgumentId_t_miopenTensorMhaDescaleDO: miopenTensorArgumentId_t = 23;
pub const miopenTensorArgumentId_t_miopenTensorMhaDescaleDS: miopenTensorArgumentId_t = 24;
pub const miopenTensorArgumentId_t_miopenTensorMhaScaleDS: miopenTensorArgumentId_t = 25;
pub const miopenTensorArgumentId_t_miopenTensorMhaScaleDQ: miopenTensorArgumentId_t = 26;
pub const miopenTensorArgumentId_t_miopenTensorMhaScaleDK: miopenTensorArgumentId_t = 27;
pub const miopenTensorArgumentId_t_miopenTensorMhaScaleDV: miopenTensorArgumentId_t = 28;
pub const miopenTensorArgumentId_t_miopenTensorMhaDQ: miopenTensorArgumentId_t = 29;
pub const miopenTensorArgumentId_t_miopenTensorMhaDK: miopenTensorArgumentId_t = 30;
pub const miopenTensorArgumentId_t_miopenTensorMhaDV: miopenTensorArgumentId_t = 31;
pub const miopenTensorArgumentId_t_miopenTensorMhaAmaxDQ: miopenTensorArgumentId_t = 32;
pub const miopenTensorArgumentId_t_miopenTensorMhaAmaxDK: miopenTensorArgumentId_t = 33;
pub const miopenTensorArgumentId_t_miopenTensorMhaAmaxDV: miopenTensorArgumentId_t = 34;
pub const miopenTensorArgumentId_t_miopenTensorMhaAmaxDS: miopenTensorArgumentId_t = 35;
pub const miopenTensorArgumentId_t_miopenTensorMhaBias: miopenTensorArgumentId_t = 36;
pub const miopenTensorArgumentId_t_miopenTensorArgumentIsScalar: miopenTensorArgumentId_t =
    2147483648;
pub const miopenTensorArgumentId_t_miopenTensorMhaMask: miopenTensorArgumentId_t = 2147483649;
pub type miopenTensorArgumentId_t = ::std::os::raw::c_uint;
pub const miopenFindResultsOrder_t_miopenFindResultsOrderByTime: miopenFindResultsOrder_t = 0;
pub const miopenFindResultsOrder_t_miopenFindResultsOrderByWorkspaceSize: miopenFindResultsOrder_t =
    1;
pub type miopenFindResultsOrder_t = ::std::os::raw::c_uint;
pub const miopenMhaMask_t_miopenMhaMaskNone: miopenMhaMask_t = 0;
pub const miopenMhaMask_t_miopenMhaMaskCausal: miopenMhaMask_t = 1;
pub type miopenMhaMask_t = ::std::os::raw::c_uint;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenFindOptions {
    pub _address: u8,
}
pub type miopenFindOptions_t = *mut miopenFindOptions;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenSolution {
    pub _address: u8,
}
pub type miopenSolution_t = *mut miopenSolution;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct miopenTensorArgument_t {
    pub id: miopenTensorArgumentId_t,
    pub descriptor: *mut miopenTensorDescriptor_t,
    pub buffer: *mut ::std::os::raw::c_void,
}
pub const miopenTuningPolicy_t_miopenTuningPolicyNone: miopenTuningPolicy_t = 1;
pub const miopenTuningPolicy_t_miopenTuningPolicyDbUpdate: miopenTuningPolicy_t = 2;
pub const miopenTuningPolicy_t_miopenTuningPolicySearch: miopenTuningPolicy_t = 3;
pub const miopenTuningPolicy_t_miopenTuningPolicySearchDbUpdate: miopenTuningPolicy_t = 4;
pub const miopenTuningPolicy_t_miopenTuningPolicyDbClean: miopenTuningPolicy_t = 5;
pub type miopenTuningPolicy_t = ::std::os::raw::c_uint;
