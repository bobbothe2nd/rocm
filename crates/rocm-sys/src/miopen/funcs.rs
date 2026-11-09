use super::types::*;
use rocm_include::ctype::*;

rocm_include::link! {
    MIOpen : rocm_ver;
    dev miopen_ _exists pastey::paste;

    *const void miopenGetErrorString(miopenStatus_t error);

    miopenStatus_t miopenGetVersion(
        *mut size_t major,
        *mut size_t minor,
        *mut size_t patch,
    );

    miopenStatus_t miopenCreate(*mut miopenHandle_t handle);

    miopenStatus_t miopenCreateWithStream(
        *mut miopenHandle_t handle,
        miopenAcceleratorQueue_t stream,
    );

    miopenStatus_t miopenDestroy(miopenHandle_t handle);

    miopenStatus_t miopenSetStream(
        miopenHandle_t handle,
        miopenAcceleratorQueue_t streamID,
    );

    miopenStatus_t miopenGetStream(
        miopenHandle_t handle,
        *mut miopenAcceleratorQueue_t streamID,
    );

    miopenStatus_t miopenSetAllocator(
        miopenHandle_t handle,
        miopenAllocatorFunction allocator,
        miopenDeallocatorFunction deallocator,
        *mut void allocatorContext,
    );

    miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, *mut float time);

    miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable);

    miopenStatus_t miopenCreateTensorDescriptor(
        *mut miopenTensorDescriptor_t tensorDesc,
    );

    miopenStatus_t miopenSet4dTensorDescriptor(
        miopenTensorDescriptor_t tensorDesc,
        miopenDataType_t dataType,
        int n,
        int c,
        int h,
        int w,
    );

    miopenStatus_t miopenSetNdTensorDescriptorWithLayout(
        miopenTensorDescriptor_t tensorDesc,
        miopenDataType_t dataType,
        miopenTensorLayout_t tensorLayout,
        *const int lens,
        int num_lens,
    );

    miopenStatus_t miopenSet4dTensorDescriptorEx(
        miopenTensorDescriptor_t tensorDesc,
        miopenDataType_t dataType,
        int n,
        int c,
        int h,
        int w,
        int nStride,
        int cStride,
        int hStride,
        int wStride,
    );
    
    miopenStatus_t miopenGet4dTensorDescriptor(
        miopenTensorDescriptor_t tensorDesc,
        *mut miopenDataType_t dataType,
        *mut int n,
        *mut int c,
        *mut int h,
        *mut int w,
        *mut int nStride,
        *mut int cStride,
        *mut int hStride,
        *mut int wStride,
    );
    
    miopenStatus_t miopenSetTensorDescriptor(
        miopenTensorDescriptor_t tensorDesc,
        miopenDataType_t dataType,
        int nbDims,
        *const int dimsA,
        *const int stridesA,
    );
    
    miopenStatus_t miopenGetTensorDescriptorSize(
        miopenTensorDescriptor_t tensorDesc,
        *mut int size,
    );
    
    miopenStatus_t miopenGetTensorDescriptor(
        miopenTensorDescriptor_t tensorDesc,
        *mut miopenDataType_t dataType,
        *mut int dimsA,
        *mut int stridesA,
    );
    
    miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc);
    
    miopenStatus_t miopenCreateSeqTensorDescriptor(
        *mut miopenSeqTensorDescriptor_t tensorDesc,
    );
    
    miopenStatus_t miopenDestroySeqTensorDescriptor(
        miopenSeqTensorDescriptor_t tensorDesc,
    );
    
    miopenStatus_t miopenOpTensor(
        miopenHandle_t handle,
        miopenTensorOp_t tensorOp,
        *const void alpha1,
        miopenTensorDescriptor_t aDesc,
        *const void A,
        *const void alpha2,
        miopenTensorDescriptor_t bDesc,
        *const void B,
        *const void beta,
        miopenTensorDescriptor_t cDesc,
        *mut void C,
    );
    
    miopenStatus_t miopenSetTensor(
        miopenHandle_t handle,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        *const void alpha,
    );
    
    miopenStatus_t miopenScaleTensor(
        miopenHandle_t handle,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        *const void alpha,
    );
    
    miopenStatus_t miopenGetTensorNumBytes(
        miopenTensorDescriptor_t tensorDesc,
        *mut size_t numBytes,
    );
    
    miopenStatus_t miopenTransformTensor(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
    );
    
    miopenStatus_t miopenCreateConvolutionDescriptor(
        *mut miopenConvolutionDescriptor_t convDesc,
    );
    
    miopenStatus_t miopenInitConvolutionDescriptor(
        miopenConvolutionDescriptor_t convDesc,
        miopenConvolutionMode_t c_mode,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
    );
    
    miopenStatus_t miopenInitConvolutionNdDescriptor(
        miopenConvolutionDescriptor_t convDesc,
        int spatialDim,
        *const int padA,
        *const int strideA,
        *const int dilationA,
        miopenConvolutionMode_t c_mode,
    );
    
    miopenStatus_t miopenGetConvolutionSpatialDim(
        miopenConvolutionDescriptor_t convDesc,
        *mut int spatialDim,
    );
    
    miopenStatus_t miopenGetConvolutionDescriptor(
        miopenConvolutionDescriptor_t convDesc,
        *mut miopenConvolutionMode_t c_mode,
        *mut int pad_h,
        *mut int pad_w,
        *mut int stride_h,
        *mut int stride_w,
        *mut int dilation_h,
        *mut int dilation_w,
    );
    
    miopenStatus_t miopenGetConvolutionNdDescriptor(
        miopenConvolutionDescriptor_t convDesc,
        int requestedSpatialDim,
        *mut int spatialDim,
        *mut int padA,
        *mut int strideA,
        *mut int dilationA,
        *mut miopenConvolutionMode_t c_mode,
    );
    
    miopenStatus_t miopenGetConvolutionGroupCount(
        miopenConvolutionDescriptor_t convDesc,
        *mut int groupCount,
    );
    
    miopenStatus_t miopenSetConvolutionGroupCount(
        miopenConvolutionDescriptor_t convDesc,
        int groupCount,
    );
    
    miopenStatus_t miopenSetTransposeConvOutputPadding(
        miopenConvolutionDescriptor_t convDesc,
        int adj_h,
        int adj_w,
    );
    
    miopenStatus_t miopenSetTransposeConvNdOutputPadding(
        miopenConvolutionDescriptor_t convDesc,
        int spatialDim,
        *const int adjA,
    );
    
    miopenStatus_t miopenGetConvolutionForwardOutputDim(
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t inputTensorDesc,
        miopenTensorDescriptor_t filterDesc,
        *mut int n,
        *mut int c,
        *mut int h,
        *mut int w,
    );

    miopenStatus_t miopenGetConvolutionNdForwardOutputDim(
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t inputTensorDesc,
        miopenTensorDescriptor_t filterDesc,
        *mut int nDim,
        *mut int outputTensorDimA,
    );

    miopenStatus_t miopenDestroyConvolutionDescriptor(
        miopenConvolutionDescriptor_t convDesc,
    );

    miopenStatus_t miopenSetConvolutionAttribute(
        miopenConvolutionDescriptor_t convDesc,
        miopenConvolutionAttrib_t attr,
        int value,
    );

    miopenStatus_t miopenGetConvolutionAttribute(
        miopenConvolutionDescriptor_t convDesc,
        miopenConvolutionAttrib_t attr,
        *mut int value,
    );

    miopenStatus_t miopenSetConvolutionFindMode(
        miopenConvolutionDescriptor_t convDesc,
        miopenConvolutionFindMode_t findMode,
    );

    miopenStatus_t miopenGetConvolutionFindMode(
        miopenConvolutionDescriptor_t convDesc,
        *mut miopenConvolutionFindMode_t findMode,
    );
    
    miopenStatus_t miopenConvolutionForwardGetSolutionCount(
        miopenHandle_t handle,
        miopenTensorDescriptor_t wDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t yDesc,
        *mut size_t solutionCount,
    );

    miopenStatus_t miopenConvolutionForwardGetSolution(
        miopenHandle_t handle,
        miopenTensorDescriptor_t wDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t yDesc,
        size_t maxSolutionCount,
        *mut size_t solutionCount,
        *mut miopenConvSolution_t solutions,
    );

    miopenStatus_t miopenConvolutionForwardGetSolutionWorkspaceSize(
        miopenHandle_t handle,
        miopenTensorDescriptor_t wDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t yDesc,
        uint64_t solution_id,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenConvolutionForwardCompileSolution(
        miopenHandle_t handle,
        miopenTensorDescriptor_t wDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t yDesc,
        uint64_t solution_id,
    );

    miopenStatus_t miopenConvolutionForwardImmediate(
        miopenHandle_t handle,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        *mut void workSpace,
        size_t workSpaceSize,
        uint64_t solution_id,
    );

    miopenStatus_t miopenConvolutionBackwardDataGetSolutionCount(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t wDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dxDesc,
        *mut size_t solutionCount,
    );

    miopenStatus_t miopenConvolutionBackwardDataGetSolution(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t wDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dxDesc,
        size_t maxSolutionCount,
        *mut size_t solutionCount,
        *mut miopenConvSolution_t solutions,
    );

    miopenStatus_t miopenConvolutionBackwardDataGetSolutionWorkspaceSize(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t wDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dxDesc,
        uint64_t solution_id,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenConvolutionBackwardDataCompileSolution(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t wDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dxDesc,
        uint64_t solution_id,
    );

    miopenStatus_t miopenConvolutionBackwardDataImmediate(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        *mut void workSpace,
        size_t workSpaceSize,
        uint64_t solution_id,
    );

    miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionCount(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dwDesc,
        *mut size_t solutionCount,
    );

    miopenStatus_t miopenConvolutionBackwardWeightsGetSolution(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dwDesc,
        size_t maxSolutionCount,
        *mut size_t solutionCount,
        *mut miopenConvSolution_t solutions,
    );

    miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dwDesc,
        uint64_t solution_id,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenConvolutionBackwardWeightsCompileSolution(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dwDesc,
        uint64_t solution_id,
    );

    miopenStatus_t miopenConvolutionBackwardWeightsImmediate(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dwDesc,
        *mut void dw,
        *mut void workSpace,
        size_t workSpaceSize,
        uint64_t solution_id,
    );

    miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
        miopenHandle_t handle,
        miopenTensorDescriptor_t wDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t yDesc,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenFindConvolutionForwardAlgorithm(
        miopenHandle_t handle,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        int requestAlgoCount,
        *mut int returnedAlgoCount,
        *mut miopenConvAlgoPerf_t perfResults,
        *mut void workSpace,
        size_t workSpaceSize,
        bool exhaustiveSearch,
    );

    miopenStatus_t miopenConvolutionForward(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenConvolutionDescriptor_t convDesc,
        miopenConvFwdAlgorithm_t algo,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        *mut void workSpace,
        size_t workSpaceSize,
    );

    miopenStatus_t miopenConvolutionForwardBias(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t bDesc,
        *const void b,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
    );

    miopenStatus_t miopenConvolutionBackwardDataGetWorkSpaceSize(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t wDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dxDesc,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenFindConvolutionBackwardDataAlgorithm(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        int requestAlgoCount,
        *mut int returnedAlgoCount,
        *mut miopenConvAlgoPerf_t perfResults,
        *mut void workSpace,
        size_t workSpaceSize,
        bool exhaustiveSearch,
    );

    miopenStatus_t miopenConvolutionBackwardData(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenConvolutionDescriptor_t convDesc,
        miopenConvBwdDataAlgorithm_t algo,
        *const void beta,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        *mut void workSpace,
        size_t workSpaceSize,
    );

    miopenStatus_t miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        miopenTensorDescriptor_t xDesc,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dwDesc,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenFindConvolutionBackwardWeightsAlgorithm(
        miopenHandle_t handle,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t dwDesc,
        *mut void dw,
        int requestAlgoCount,
        *mut int returnedAlgoCount,
        *mut miopenConvAlgoPerf_t perfResults,
        *mut void workSpace,
        size_t workSpaceSize,
        bool exhaustiveSearch,
    );

    miopenStatus_t miopenConvolutionBackwardWeights(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenConvolutionDescriptor_t convDesc,
        miopenConvBwdWeightsAlgorithm_t algo,
        *const void beta,
        miopenTensorDescriptor_t dwDesc,
        *mut void dw,
        *mut void workSpace,
        size_t workSpaceSize,
    );

    miopenStatus_t miopenConvolutionBackwardBias(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        *const void beta,
        miopenTensorDescriptor_t dbDesc,
        *mut void db,
    );

    miopenStatus_t miopenCreatePoolingDescriptor(
        *mut miopenPoolingDescriptor_t poolDesc,
    );

    miopenStatus_t miopenSetPoolingIndexType(
        miopenPoolingDescriptor_t poolDesc,
        miopenIndexType_t index_type,
    );

    miopenStatus_t miopenGetPoolingIndexType(
        miopenPoolingDescriptor_t poolDesc,
        *mut miopenIndexType_t index_type,
    );

    miopenStatus_t miopenSetPoolingWorkSpaceIndexMode(
        miopenPoolingDescriptor_t poolDesc,
        miopenPoolingWorkspaceIndexMode_t workspace_index,
    );

    miopenStatus_t miopenGetPoolingWorkSpaceIndexMode(
        miopenPoolingDescriptor_t poolDesc,
        *mut miopenPoolingWorkspaceIndexMode_t workspace_index,
    );

    miopenStatus_t miopenSet2dPoolingDescriptor(
        miopenPoolingDescriptor_t poolDesc,
        miopenPoolingMode_t mode,
        int windowHeight,
        int windowWidth,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
    );

    miopenStatus_t miopenGet2dPoolingDescriptor(
        miopenPoolingDescriptor_t poolDesc,
        *mut miopenPoolingMode_t mode,
        *mut int windowHeight,
        *mut int windowWidth,
        *mut int pad_h,
        *mut int pad_w,
        *mut int stride_h,
        *mut int stride_w,
    );

    miopenStatus_t miopenGetPoolingForwardOutputDim(
        miopenPoolingDescriptor_t poolDesc,
        miopenTensorDescriptor_t tensorDesc,
        *mut int n,
        *mut int c,
        *mut int h,
        *mut int w,
    );

    miopenStatus_t miopenSetNdPoolingDescriptor(
        miopenPoolingDescriptor_t poolDesc,
        miopenPoolingMode_t mode,
        int nbDims,
        *const int windowDimA,
        *const int padA,
        *const int stridesA,
    );

    miopenStatus_t miopenGetNdPoolingDescriptor(
        miopenPoolingDescriptor_t poolDesc,
        int nbDimsRequested,
        *mut miopenPoolingMode_t mode,
        *mut int nbDims,
        *mut int windowDimA,
        *mut int padA,
        *mut int stridesA,
    );

    miopenStatus_t miopenGetPoolingNdForwardOutputDim(
        miopenPoolingDescriptor_t poolDesc,
        miopenTensorDescriptor_t tensorDesc,
        int dims,
        *mut int tensorDimArr,
    );

    miopenStatus_t miopenPoolingGetWorkSpaceSize(
        miopenTensorDescriptor_t yDesc,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenPoolingGetWorkSpaceSizeV2(
        miopenPoolingDescriptor_t poolDesc,
        miopenTensorDescriptor_t yDesc,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenPoolingForward(
        miopenHandle_t handle,
        miopenPoolingDescriptor_t poolDesc,
        *const void alpha,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        bool do_backward,
        *mut void workSpace,
        size_t workSpaceSize,
    );

    miopenStatus_t miopenPoolingBackward(
        miopenHandle_t handle,
        miopenPoolingDescriptor_t poolDesc,
        *const void alpha,
        miopenTensorDescriptor_t yDesc,
        *const void y,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        *mut void workSpace,
    );

    miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc);

    miopenStatus_t miopenCreateLRNDescriptor(*mut miopenLRNDescriptor_t lrnDesc);

    miopenStatus_t miopenSetLRNDescriptor(
        miopenLRNDescriptor_t lrnDesc,
        miopenLRNMode_t mode,
        uint lrnN,
        double lrnAlpha,
        double lrnBeta,
        double lrnK,
    );

    miopenStatus_t miopenGetLRNDescriptor(
        miopenLRNDescriptor_t lrnDesc,
        *mut miopenLRNMode_t mode,
        *mut uint lrnN,
        *mut double lrnAlpha,
        *mut double lrnBeta,
        *mut double lrnK,
    );

    miopenStatus_t miopenLRNGetWorkSpaceSize(
        miopenTensorDescriptor_t yDesc,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenLRNForward(
        miopenHandle_t handle,
        miopenLRNDescriptor_t lrnDesc,
        *const void alpha,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        bool do_backward,
        *mut void workSpace,
    );

    miopenStatus_t miopenLRNBackward(
        miopenHandle_t handle,
        miopenLRNDescriptor_t lrnDesc,
        *const void alpha,
        miopenTensorDescriptor_t yDesc,
        *const void y,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        *const void workSpace,
    );

    miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc);

    miopenStatus_t miopenDeriveBNTensorDescriptor(
        miopenTensorDescriptor_t derivedBnDesc,
        miopenTensorDescriptor_t xDesc,
        miopenBatchNormMode_t bn_mode,
    );

    miopenStatus_t miopenBatchNormalizationForwardTraining(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *mut void alpha,
        *mut void beta,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
        *mut void bnScale,
        *mut void bnBias,
        double expAvgFactor,
        *mut void resultRunningMean,
        *mut void resultRunningVariance,
        double epsilon,
        *mut void resultSaveMean,
        *mut void resultSaveInvVariance,
    );

    miopenStatus_t miopenBatchNormalizationForwardTraining_V2(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *mut void alpha,
        *mut void beta,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t scaleDesc,
        miopenTensorDescriptor_t biasVarDesc,
        miopenTensorDescriptor_t savedMeanDesc,
        miopenTensorDescriptor_t savedVarDesc,
        *mut void bnScale,
        *mut void bnBias,
        double expAvgFactor,
        *mut void resultRunningMean,
        *mut void resultRunningVariance,
        double epsilon,
        *mut void resultSaveMean,
        *mut void resultSaveInvVariance,
    );

    miopenStatus_t miopenBatchNormForwardTrainingActivation(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *mut void alpha,
        *mut void beta,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t scaleDesc,
        miopenTensorDescriptor_t biasVarDesc,
        miopenTensorDescriptor_t savedMeanDesc,
        miopenTensorDescriptor_t savedVarDesc,
        *mut void bnScale,
        *mut void bnBias,
        double expAvgFactor,
        *mut void resultRunningMean,
        *mut void resultRunningVariance,
        double epsilon,
        *mut void resultSaveMean,
        *mut void resultSaveInvVariance,
        miopenActivationDescriptor_t activDesc,
    );

    miopenStatus_t miopenBatchNormalizationForwardInference(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *mut void alpha,
        *mut void beta,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
        *mut void bnScale,
        *mut void bnBias,
        *mut void estimatedMean,
        *mut void estimatedVariance,
        double epsilon,
    );

    miopenStatus_t miopenBatchNormalizationForwardInference_V2(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *mut void alpha,
        *mut void beta,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t scaleDesc,
        miopenTensorDescriptor_t biasDesc,
        miopenTensorDescriptor_t estMeanDesc,
        miopenTensorDescriptor_t estVarianceDesc,
        *mut void bnScale,
        *mut void bnBias,
        *mut void estimatedMean,
        *mut void estimatedVariance,
        double epsilon,
    );

    miopenStatus_t miopenBatchNormForwardInferenceActivation(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *mut void alpha,
        *mut void beta,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t scaleDesc,
        miopenTensorDescriptor_t biasDesc,
        miopenTensorDescriptor_t estMeanDesc,
        miopenTensorDescriptor_t estVarianceDesc,
        *mut void bnScale,
        *mut void bnBias,
        *mut void estimatedMean,
        *mut void estimatedVariance,
        double epsilon,
        miopenActivationDescriptor_t activDesc,
    );

    miopenStatus_t miopenBatchNormalizationBackward(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *const void alphaDataDiff,
        *const void betaDataDiff,
        *const void alphaParamDiff,
        *const void betaParamDiff,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        miopenTensorDescriptor_t bnScaleBiasDiffDesc,
        *const void bnScale,
        *mut void resultBnScaleDiff,
        *mut void resultBnBiasDiff,
        double epsilon,
        *const void savedMean,
        *const void savedInvVariance,
    );

    miopenStatus_t miopenBatchNormalizationBackward_V2(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *const void alphaDataDiff,
        *const void betaDataDiff,
        *const void alphaParamDiff,
        *const void betaParamDiff,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        miopenTensorDescriptor_t scaleDesc,
        miopenTensorDescriptor_t biasDesc,
        miopenTensorDescriptor_t savedMeanDesc,
        miopenTensorDescriptor_t savedVarDesc,
        *const void bnScale,
        *mut void resultBnScaleDiff,
        *mut void resultBnBiasDiff,
        double epsilon,
        *const void savedMean,
        *const void savedInvVariance,
    );

    miopenStatus_t miopenBatchNormBackwardActivation(
        miopenHandle_t handle,
        miopenBatchNormMode_t bn_mode,
        *const void alphaDataDiff,
        *const void betaDataDiff,
        *const void alphaParamDiff,
        *const void betaParamDiff,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        miopenTensorDescriptor_t scaleDesc,
        miopenTensorDescriptor_t biasDesc,
        miopenTensorDescriptor_t savedMeanDesc,
        miopenTensorDescriptor_t savedVarianceDesc,
        *const void bnScale,
        *const void bnBias,
        *mut void resultBnScaleDiff,
        *mut void resultBnBiasDiff,
        double epsilon,
        *const void savedMean,
        *const void savedInvVariance,
        miopenActivationDescriptor_t activDesc,
    );

    miopenStatus_t miopenCreateActivationDescriptor(
        *mut miopenActivationDescriptor_t activDesc,
    );

    miopenStatus_t miopenSetActivationDescriptor(
        miopenActivationDescriptor_t activDesc,
        miopenActivationMode_t mode,
        double activAlpha,
        double activBeta,
        double activGamma,
    );

    miopenStatus_t miopenGetActivationDescriptor(
        miopenActivationDescriptor_t activDesc,
        *mut miopenActivationMode_t mode,
        *mut double activAlpha,
        *mut double activBeta,
        *mut double activGamma,
    );

    miopenStatus_t miopenActivationForward(
        miopenHandle_t handle,
        miopenActivationDescriptor_t activDesc,
        *const void alpha,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
    );

    miopenStatus_t miopenActivationBackward(
        miopenHandle_t handle,
        miopenActivationDescriptor_t activDesc,
        *const void alpha,
        miopenTensorDescriptor_t yDesc,
        *const void y,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
    );

    miopenStatus_t miopenDestroyActivationDescriptor(
        miopenActivationDescriptor_t activDesc,
    );

    miopenStatus_t miopenSoftmaxForward(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
    );

    miopenStatus_t miopenSoftmaxBackward(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t yDesc,
        *const void y,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        *const void beta,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
    );

    miopenStatus_t miopenSoftmaxForward_V2(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        *const void beta,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenSoftmaxAlgorithm_t algorithm,
        miopenSoftmaxMode_t mode,
    );

    miopenStatus_t miopenSoftmaxBackward_V2(
        miopenHandle_t handle,
        *const void alpha,
        miopenTensorDescriptor_t yDesc,
        *const void y,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        *const void beta,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        miopenSoftmaxAlgorithm_t algorithm,
        miopenSoftmaxMode_t mode,
    );
    

    miopenStatus_t miopenCreateFusionPlan(
        *mut miopenFusionPlanDescriptor_t fusePlanDesc,
        miopenFusionDirection_t fuseDirection,
        miopenTensorDescriptor_t inputDesc,
    );

    miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc);

    miopenStatus_t miopenCompileFusionPlan(
        miopenHandle_t handle,
        miopenFusionPlanDescriptor_t fusePlanDesc,
    );

    miopenStatus_t miopenFusionPlanGetOp(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        int op_idx,
        *mut miopenFusionOpDescriptor_t op,
    );

    miopenStatus_t miopenFusionPlanGetWorkSpaceSize(
        miopenHandle_t handle,
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut size_t workSpaceSize,
        miopenConvFwdAlgorithm_t algo,
    );

    miopenStatus_t miopenFusionPlanConvolutionGetAlgo(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        int requestAlgoCount,
        *mut int returnedAlgoCount,
        *mut miopenConvFwdAlgorithm_t returnedAlgos,
    );

    miopenStatus_t miopenFusionPlanConvolutionSetAlgo(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        miopenConvFwdAlgorithm_t algo,
    );

    miopenStatus_t miopenCreateOpConvForward(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut miopenFusionOpDescriptor_t convOp,
        miopenConvolutionDescriptor_t convDesc,
        miopenTensorDescriptor_t wDesc,
    );

    miopenStatus_t miopenCreateOpActivationForward(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut miopenFusionOpDescriptor_t activFwdOp,
        miopenActivationMode_t mode,
    );

    miopenStatus_t miopenCreateOpActivationBackward(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut miopenFusionOpDescriptor_t activBwdOp,
        miopenActivationMode_t mode,
    );

    miopenStatus_t miopenCreateOpBiasForward(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut miopenFusionOpDescriptor_t biasOp,
        miopenTensorDescriptor_t bDesc,
    );

    miopenStatus_t miopenCreateOpBatchNormInference(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut miopenFusionOpDescriptor_t bnOp,
        miopenBatchNormMode_t bn_mode,
        miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
    );

    miopenStatus_t miopenCreateOpBatchNormForward(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut miopenFusionOpDescriptor_t bnFwdOp,
        miopenBatchNormMode_t bn_mode,
        bool runningMeanVariance,
    );

    miopenStatus_t miopenCreateOpBatchNormBackward(
        miopenFusionPlanDescriptor_t fusePlanDesc,
        *mut miopenFusionOpDescriptor_t bnBwdOp,
        miopenBatchNormMode_t bn_mode,
    );

    miopenStatus_t miopenCreateOperatorArgs(*mut miopenOperatorArgs_t args);

    miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args);

    miopenStatus_t miopenSetOpArgsConvForward(
        miopenOperatorArgs_t args,
        miopenFusionOpDescriptor_t convOp,
        *const void alpha,
        *const void beta,
        *const void w,
    );

    miopenStatus_t miopenSetOpArgsActivForward(
        miopenOperatorArgs_t args,
        miopenFusionOpDescriptor_t activFwdOp,
        *const void alpha,
        *const void beta,
        double activAlpha,
        double activBeta,
        double activGamma,
    );

    miopenStatus_t miopenSetOpArgsActivBackward(
        miopenOperatorArgs_t args,
        miopenFusionOpDescriptor_t activBwdOp,
        *const void alpha,
        *const void beta,
        *const void y,
        *const void reserved,
        double activAlpha,
        double activBeta,
        double activGamma,
    );

    miopenStatus_t miopenSetOpArgsBatchNormInference(
        miopenOperatorArgs_t args,
        miopenFusionOpDescriptor_t bnOp,
        *const void alpha,
        *const void beta,
        *const void bnScale,
        *const void bnBias,
        *const void estimatedMean,
        *const void estimatedVariance,
        double epsilon,
    );

    miopenStatus_t miopenSetOpArgsBatchNormForward(
        miopenOperatorArgs_t args,
        miopenFusionOpDescriptor_t bnOp,
        *const void alpha,
        *const void beta,
        *const void bnScale,
        *const void bnBias,
        *mut void savedMean,
        *mut void savedInvVariance,
        *mut void runningMean,
        *mut void runningVariance,
        double expAvgFactor,
        double epsilon,
    );

    miopenStatus_t miopenSetOpArgsBatchNormBackward(
        miopenOperatorArgs_t args,
        miopenFusionOpDescriptor_t bnOp,
        *const void alpha,
        *const void beta,
        *const void x,
        *const void bnScale,
        *const void bnBias,
        *mut void resultBnScaleDiff,
        *mut void resultBnBiasDiff,
        *const void savedMean,
        *const void savedInvVariance,
    );

    miopenStatus_t miopenSetOpArgsBiasForward(
        miopenOperatorArgs_t args,
        miopenFusionOpDescriptor_t biasOp,
        *const void alpha,
        *const void beta,
        *const void bias,
    );

    miopenStatus_t miopenExecuteFusionPlan(
        miopenHandle_t handle,
        miopenFusionPlanDescriptor_t fusePlanDesc,
        miopenTensorDescriptor_t inputDesc,
        *const void input,
        miopenTensorDescriptor_t outputDesc,
        *mut void output,
        miopenOperatorArgs_t args,
    );

    miopenStatus_t miopenExecuteFusionPlan_v2(
        miopenHandle_t handle,
        miopenFusionPlanDescriptor_t fusePlanDesc,
        miopenTensorDescriptor_t inputDesc,
        *const void input,
        miopenTensorDescriptor_t outputDesc,
        *mut void output,
        miopenOperatorArgs_t args,
        *mut void workspace,
        size_t workspaceSize,
    );

    miopenStatus_t miopenConvolutionBiasActivationForward(
        miopenHandle_t handle,
        *const void alpha1,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenConvolutionDescriptor_t convDesc,
        miopenConvFwdAlgorithm_t algo,
        *mut void workspace,
        size_t workspaceSizeInBytes,
        *const void alpha2,
        miopenTensorDescriptor_t zDesc,
        *const void z,
        miopenTensorDescriptor_t biasDesc,
        *const void bias,
        miopenActivationDescriptor_t activationDesc,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
    );

    miopenStatus_t miopenCreateRNNDescriptor(*mut miopenRNNDescriptor_t rnnDesc);

    miopenStatus_t miopenGetRNNDescriptor(
        miopenRNNDescriptor_t rnnDesc,
        *mut miopenRNNMode_t rnnMode,
        *mut miopenRNNAlgo_t algoMode,
        *mut miopenRNNInputMode_t inputMode,
        *mut miopenRNNDirectionMode_t dirMode,
        *mut miopenRNNBiasMode_t biasMode,
        *mut int hiddenSize,
        *mut int layer,
    );

    miopenStatus_t miopenGetRNNDescriptor_V2(
        miopenRNNDescriptor_t rnnDesc,
        *mut int hiddenSize,
        *mut int layer,
        *mut miopenDropoutDescriptor_t dropoutDesc,
        *mut miopenRNNInputMode_t inputMode,
        *mut miopenRNNDirectionMode_t dirMode,
        *mut miopenRNNMode_t rnnMode,
        *mut miopenRNNBiasMode_t biasMode,
        *mut miopenRNNAlgo_t algoMode,
        *mut miopenDataType_t dataType,
    );

    miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc);

    miopenStatus_t miopenSetRNNDescriptor(
        miopenRNNDescriptor_t rnnDesc,
        int hsize,
        int nlayers,
        miopenRNNInputMode_t inMode,
        miopenRNNDirectionMode_t direction,
        miopenRNNMode_t rnnMode,
        miopenRNNBiasMode_t biasMode,
        miopenRNNAlgo_t algo,
        miopenDataType_t dataType,
    );

    miopenStatus_t miopenSetRNNDescriptor_V2(
        miopenRNNDescriptor_t rnnDesc,
        int hsize,
        int nlayers,
        miopenDropoutDescriptor_t dropoutDesc,
        miopenRNNInputMode_t inMode,
        miopenRNNDirectionMode_t direction,
        miopenRNNMode_t rnnMode,
        miopenRNNBiasMode_t biasMode,
        miopenRNNAlgo_t algo,
        miopenDataType_t dataType,
    );

    miopenStatus_t miopenSetRNNDataSeqTensorDescriptor(
        miopenSeqTensorDescriptor_t seqTensorDesc,
        miopenDataType_t dataType,
        miopenRNNBaseLayout_t layout,
        int maxSequenceLen,
        int batchSize,
        int vectorSize,
        *const int sequenceLenArray,
        *mut void paddingMarker,
    );

    miopenStatus_t miopenGetRNNDataSeqTensorDescriptor(
        miopenSeqTensorDescriptor_t seqTensorDesc,
        *mut miopenDataType_t dataType,
        *mut miopenRNNBaseLayout_t layout,
        *mut int maxSequenceLen,
        *mut int batchSize,
        *mut int vectorSize,
        int sequenceLenArrayLimit,
        *mut int sequenceLenArray,
        *mut void paddingMarker,
    );

    miopenStatus_t miopenGetRNNWorkspaceSize(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int sequenceLen,
        *const miopenTensorDescriptor_t xDesc,
        *mut size_t numBytes,
    );

    miopenStatus_t miopenGetRNNTrainingReserveSize(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int sequenceLen,
        *const miopenTensorDescriptor_t xDesc,
        *mut size_t numBytes,
    );

    miopenStatus_t miopenGetRNNTempSpaceSizes(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        miopenSeqTensorDescriptor_t xDesc,
        miopenRNNFWDMode_t fwdMode,
        *mut size_t workSpaceSize,
        *mut size_t reserveSpaceSize,
    );

    miopenStatus_t miopenGetRNNParamsSize(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        miopenTensorDescriptor_t xDesc,
        *mut size_t numBytes,
        miopenDataType_t dtype,
    );

    miopenStatus_t miopenGetRNNParamsDescriptor(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        miopenTensorDescriptor_t xDesc,
        miopenTensorDescriptor_t wDesc,
        miopenDataType_t dtype,
    );

    miopenStatus_t miopenGetRNNInputTensorSize(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int seqLen,
        *mut miopenTensorDescriptor_t xDesc,
        *mut size_t numBytes,
    );

    miopenStatus_t miopenGetRNNHiddenTensorSize(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int seqLen,
        *mut miopenTensorDescriptor_t xDesc,
        *mut size_t numBytes,
    );

    miopenStatus_t miopenGetRNNLayerParamSize(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        miopenTensorDescriptor_t xDesc,
        int paramID,
        *mut size_t numBytes,
    );

    miopenStatus_t miopenGetRNNLayerBiasSize(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        int biasID,
        *mut size_t numBytes,
    );

    miopenStatus_t miopenGetRNNLayerParam(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        miopenTensorDescriptor_t xDesc,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        int paramID,
        miopenTensorDescriptor_t paramDesc,
        *mut void layerParam,
    );

    miopenStatus_t miopenGetRNNLayerBias(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        miopenTensorDescriptor_t xDesc,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        int biasID,
        miopenTensorDescriptor_t biasDesc,
        *mut void layerBias,
    );

    miopenStatus_t miopenGetRNNLayerParamOffset(
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        miopenTensorDescriptor_t xDesc,
        int paramID,
        miopenTensorDescriptor_t paramDesc,
        *mut size_t layerParamOffset,
    );

    miopenStatus_t miopenGetRNNLayerBiasOffset(
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        miopenTensorDescriptor_t xDesc,
        int biasID,
        miopenTensorDescriptor_t biasDesc,
        *mut size_t layerBiasOffset,
    );

    miopenStatus_t miopenSetRNNLayerParam(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        miopenTensorDescriptor_t xDesc,
        miopenTensorDescriptor_t wDesc,
        *mut void w,
        int paramID,
        miopenTensorDescriptor_t paramDesc,
        *const void layerParam,
    );

    miopenStatus_t miopenSetRNNLayerBias(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int layer,
        miopenTensorDescriptor_t xDesc,
        miopenTensorDescriptor_t wDesc,
        *mut void w,
        int biasID,
        miopenTensorDescriptor_t biasDesc,
        *const void layerBias,
    );

    miopenStatus_t miopenSetRNNPaddingMode(
        miopenRNNDescriptor_t rnnDesc,
        miopenRNNPaddingMode_t paddingMode,
    );

    miopenStatus_t miopenGetRNNPaddingMode(
        miopenRNNDescriptor_t rnnDesc,
        *mut miopenRNNPaddingMode_t paddingMode,
    );

    miopenStatus_t miopenRNNForward(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        miopenRNNFWDMode_t fwdMode,
        miopenSeqTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t hDesc,
        *const void hx,
        *mut void hy,
        miopenTensorDescriptor_t cDesc,
        *const void cx,
        *mut void cy,
        miopenSeqTensorDescriptor_t yDesc,
        *mut void y,
        *const void w,
        size_t weightSpaceSize,
        *mut void workSpace,
        size_t workSpaceNumBytes,
        *mut void reserveSpace,
        size_t reserveSpaceNumBytes,
    );

    miopenStatus_t miopenRNNBackwardSeqData(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        miopenSeqTensorDescriptor_t yDesc,
        *const void y,
        *const void dy,
        miopenTensorDescriptor_t hDesc,
        *const void hx,
        *const void dhy,
        *mut void dhx,
        miopenTensorDescriptor_t cDesc,
        *const void cx,
        *const void dcy,
        *mut void dcx,
        miopenSeqTensorDescriptor_t xDesc,
        *mut void dx,
        *const void w,
        size_t weightSpaceSize,
        *mut void workSpace,
        size_t workSpaceNumBytes,
        *mut void reserveSpace,
        size_t reserveSpaceNumBytes,
    );

    miopenStatus_t miopenRNNBackwardWeightsSeqTensor(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        miopenSeqTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t hDesc,
        *const void hx,
        miopenSeqTensorDescriptor_t yDesc,
        *const void y,
        *mut void dw,
        size_t weightSpaceSize,
        *mut void workSpace,
        size_t workSpaceNumBytes,
        *const void reserveSpace,
        size_t reserveSpaceNumBytes,
    );

    miopenStatus_t miopenRNNForwardTraining(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int sequenceLen,
        *const miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t hxDesc,
        *const void hx,
        miopenTensorDescriptor_t cxDesc,
        *const void cx,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        *const miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t hyDesc,
        *mut void hy,
        miopenTensorDescriptor_t cyDesc,
        *mut void cy,
        *mut void workSpace,
        size_t workSpaceNumBytes,
        *mut void reserveSpace,
        size_t reserveSpaceNumBytes,
    );

    miopenStatus_t miopenRNNBackwardData(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int sequenceLen,
        *const miopenTensorDescriptor_t yDesc,
        *const void y,
        *const miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t dhyDesc,
        *const void dhy,
        miopenTensorDescriptor_t dcyDesc,
        *const void dcy,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        miopenTensorDescriptor_t hxDesc,
        *const void hx,
        miopenTensorDescriptor_t cxDesc,
        *const void cx,
        *const miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        miopenTensorDescriptor_t dhxDesc,
        *mut void dhx,
        miopenTensorDescriptor_t dcxDesc,
        *mut void dcx,
        *mut void workSpace,
        size_t workSpaceNumBytes,
        *mut void reserveSpace,
        size_t reserveSpaceNumBytes,
    );

    miopenStatus_t miopenRNNBackwardWeights(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int sequenceLen,
        *const miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t hxDesc,
        *const void hx,
        *const miopenTensorDescriptor_t yDesc,
        *const void y,
        miopenTensorDescriptor_t dwDesc,
        *mut void dw,
        *mut void workSpace,
        size_t workSpaceNumBytes,
        *const void reserveSpace,
        size_t reserveSpaceNumBytes,
    );

    miopenStatus_t miopenRNNForwardInference(
        miopenHandle_t handle,
        miopenRNNDescriptor_t rnnDesc,
        int sequenceLen,
        *const miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t hxDesc,
        *const void hx,
        miopenTensorDescriptor_t cxDesc,
        *const void cx,
        miopenTensorDescriptor_t wDesc,
        *const void w,
        *const miopenTensorDescriptor_t yDesc,
        *mut void y,
        miopenTensorDescriptor_t hyDesc,
        *mut void hy,
        miopenTensorDescriptor_t cyDesc,
        *mut void cy,
        *mut void workSpace,
        size_t workSpaceNumBytes,
    );

    miopenStatus_t miopenCreateCTCLossDescriptor(
        *mut miopenCTCLossDescriptor_t ctcLossDesc,
    );

    miopenStatus_t miopenGetCTCLossDescriptor(
        miopenCTCLossDescriptor_t ctcLossDesc,
        *mut miopenDataType_t dataType,
        *mut int blank_label_id,
        *mut bool apply_softmax_layer,
    );

    miopenStatus_t miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc)
       ;

    miopenStatus_t miopenSetCTCLossDescriptor(
        miopenCTCLossDescriptor_t ctcLossDesc,
        miopenDataType_t dataType,
        int blank_label_id,
        bool apply_softmax_layer,
    );

    miopenStatus_t miopenGetCTCLossWorkspaceSize(
        miopenHandle_t handle,
        miopenTensorDescriptor_t probsDesc,
        miopenTensorDescriptor_t gradientsDesc,
        *const int labels,
        *const int labelLengths,
        *const int inputLengths,
        miopenCTCLossAlgo_t algo,
        miopenCTCLossDescriptor_t ctcLossDesc,
        *mut size_t workSpaceSize,
    );

    miopenStatus_t miopenCTCLoss(
        miopenHandle_t handle,
        miopenTensorDescriptor_t probsDesc,
        *const void probs,
        *const int labels,
        *const int labelLengths,
        *const int inputLengths,
        *mut void losses,
        miopenTensorDescriptor_t gradientsDesc,
        *mut void gradients,
        miopenCTCLossAlgo_t algo,
        miopenCTCLossDescriptor_t ctcLossDesc,
        *mut void workSpace,
        size_t workSpaceSize,
    );


    miopenStatus_t miopenCreateDropoutDescriptor(
        *mut miopenDropoutDescriptor_t dropoutDesc,
    );

    miopenStatus_t miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc)
       ;

    miopenStatus_t miopenDropoutGetReserveSpaceSize(
        miopenTensorDescriptor_t xDesc,
        *mut size_t reserveSpaceSizeInBytes,
    );

    miopenStatus_t miopenDropoutGetStatesSize(
        miopenHandle_t handle,
        *mut size_t stateSizeInBytes,
    );

    miopenStatus_t miopenGetDropoutDescriptor(
        miopenDropoutDescriptor_t dropoutDesc,
        miopenHandle_t handle,
        *mut float dropout,
        *mut *mut void states,
        *mut ulonglong seed,
        *mut bool use_mask,
        *mut bool state_evo,
        *mut miopenRNGType_t rng_mode,
    );

    miopenStatus_t miopenRestoreDropoutDescriptor(
        miopenDropoutDescriptor_t dropoutDesc,
        miopenHandle_t handle,
        float dropout,
        *mut void states,
        size_t stateSizeInBytes,
        ulonglong seed,
        bool use_mask,
        bool state_evo,
        miopenRNGType_t rng_mode,
    );

    miopenStatus_t miopenSetDropoutDescriptor(
        miopenDropoutDescriptor_t dropoutDesc,
        miopenHandle_t handle,
        float dropout,
        *mut void states,
        size_t stateSizeInBytes,
        ulonglong seed,
        bool use_mask,
        bool state_evo,
        miopenRNGType_t rng_mode,
    );

    miopenStatus_t miopenDropoutForward(
        miopenHandle_t handle,
        miopenDropoutDescriptor_t dropoutDesc,
        miopenTensorDescriptor_t noise_shape,
        miopenTensorDescriptor_t xDesc,
        *const void x,
        miopenTensorDescriptor_t yDesc,
        *mut void y,
        *mut void reserveSpace,
        size_t reserveSpaceSizeInBytes,
    );

    miopenStatus_t miopenDropoutBackward(
        miopenHandle_t handle,
        miopenDropoutDescriptor_t dropoutDesc,
        miopenTensorDescriptor_t noise_shape,
        miopenTensorDescriptor_t dyDesc,
        *const void dy,
        miopenTensorDescriptor_t dxDesc,
        *mut void dx,
        *mut void reserveSpace,
        size_t reserveSpaceSizeInBytes,
    );

    miopenStatus_t miopenCreateReduceTensorDescriptor(
        *mut miopenReduceTensorDescriptor_t reduceTensorDesc,
    );

    miopenStatus_t miopenDestroyReduceTensorDescriptor(
        miopenReduceTensorDescriptor_t reduceTensorDesc,
    );

    miopenStatus_t miopenSetReduceTensorDescriptor(
        miopenReduceTensorDescriptor_t reduceTensorDesc,
        miopenReduceTensorOp_t reduceTensorOp,
        miopenDataType_t reduceTensorCompType,
        miopenNanPropagation_t reduceTensorNanOpt,
        miopenReduceTensorIndices_t reduceTensorIndices,
        miopenIndicesType_t reduceTensorIndicesType,
    );

    miopenStatus_t miopenGetReduceTensorDescriptor(
        miopenReduceTensorDescriptor_t reduceTensorDesc,
        *mut miopenReduceTensorOp_t reduceTensorOp,
        *mut miopenDataType_t reduceTensorCompType,
        *mut miopenNanPropagation_t reduceTensorNanOpt,
        *mut miopenReduceTensorIndices_t reduceTensorIndices,
        *mut miopenIndicesType_t reduceTensorIndicesType,
    );

    miopenStatus_t miopenGetReductionIndicesSize(
        miopenHandle_t handle,
        miopenReduceTensorDescriptor_t reduceTensorDesc,
        miopenTensorDescriptor_t aDesc,
        miopenTensorDescriptor_t cDesc,
        *mut size_t sizeInBytes,
    );

    miopenStatus_t miopenGetReductionWorkspaceSize(
        miopenHandle_t handle,
        miopenReduceTensorDescriptor_t reduceTensorDesc,
        miopenTensorDescriptor_t aDesc,
        miopenTensorDescriptor_t cDesc,
        *mut size_t sizeInBytes,
    );

    miopenStatus_t miopenReduceTensor(
        miopenHandle_t handle,
        miopenReduceTensorDescriptor_t reduceTensorDesc,
        *mut void indices,
        size_t indicesSizeInBytes,
        *mut void workspace,
        size_t workspaceSizeInBytes,
        *const void alpha,
        miopenTensorDescriptor_t aDesc,
        *const void A,
        *const void beta,
        miopenTensorDescriptor_t cDesc,
        *mut void C,
    );

    miopenStatus_t miopenCreateConvProblem(
        *mut miopenProblem_t problem,
        miopenConvolutionDescriptor_t operatorDesc,
        miopenProblemDirection_t direction,
    );


    miopenStatus_t miopenCreateMhaProblem(
        *mut miopenProblem_t problem,
        miopenMhaDescriptor_t operatorDesc,
        miopenProblemDirection_t direction,
    );

    miopenStatus_t miopenCreateMhaDescriptor(*mut miopenMhaDescriptor_t mhaDesc);

    miopenStatus_t miopenSetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float scale);

    miopenStatus_t miopenGetMhaDescriptor(
        miopenMhaDescriptor_t mhaDesc,
        *mut float scale,
    );

    miopenStatus_t miopenCreateSoftmaxDescriptor(
        *mut miopenSoftmaxDescriptor_t softmaxDesc,
    );

    miopenStatus_t miopenSetSoftmaxDescriptor(
        miopenSoftmaxDescriptor_t softmaxDesc,
        float alpha,
        float beta,
        miopenSoftmaxAlgorithm_t algorithm,
        miopenSoftmaxMode_t mode,
    );

    miopenStatus_t miopenGetSoftmaxDescriptor(
        miopenSoftmaxDescriptor_t softmaxDesc,
        *mut float alpha,
        *mut float beta,
        *mut miopenSoftmaxAlgorithm_t algorithm,
        *mut miopenSoftmaxMode_t mode,
    );

    miopenStatus_t miopenDestroyProblem(miopenProblem_t problem);

    miopenStatus_t miopenSetProblemTensorDescriptor(
        miopenProblem_t problem,
        miopenTensorArgumentId_t id,
        miopenTensorDescriptor_t descriptor,
    );
    
    miopenStatus_t miopenCreateFindOptions(*mut miopenFindOptions_t options);

    miopenStatus_t miopenDestroyFindOptions(miopenFindOptions_t options);

    miopenStatus_t miopenSetFindOptionTuning(
        miopenFindOptions_t options,
        int value,
    );

    miopenStatus_t miopenSetFindOptionResultsOrder(
        miopenFindOptions_t options,
        miopenFindResultsOrder_t value,
    );

    miopenStatus_t miopenSetFindOptionWorkspaceLimit(
        miopenFindOptions_t options,
        size_t value,
    );

    miopenStatus_t miopenSetFindOptionPreallocatedWorkspace(
        miopenFindOptions_t options,
        *mut void buffer,
        size_t size,
    );

    miopenStatus_t miopenSetFindOptionPreallocatedTensor(
        miopenFindOptions_t options,
        miopenTensorArgumentId_t id,
        *mut void buffer,
    );

    miopenStatus_t miopenSetFindOptionAttachBinaries(
        miopenFindOptions_t options,
        uint attach,
    );

    miopenStatus_t miopenFindSolutions(
        miopenHandle_t handle,
        miopenProblem_t problem,
        miopenFindOptions_t options,
        *mut miopenSolution_t solutions,
        *mut size_t numSolutions,
        size_t maxSolutions,
    );

    miopenStatus_t miopenRunSolution(
        miopenHandle_t handle,
        miopenSolution_t solution,
        size_t nInputs,
        *const miopenTensorArgument_t tensors,
        *mut void workspace,
        size_t workspaceSize,
    );

    miopenStatus_t miopenDestroySolution(miopenSolution_t solution);

    miopenStatus_t miopenLoadSolution(
        *mut miopenSolution_t solution,
        *const char data,
        size_t size,
    );

    miopenStatus_t miopenSaveSolution(
        miopenSolution_t solution,
        *mut char data,
    );

    miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, *mut size_t size);

    miopenStatus_t miopenGetSolutionWorkspaceSize(
        miopenSolution_t solution,
        *mut size_t workspaceSize,
    );

    miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, *mut float time);

    miopenStatus_t miopenGetSolutionSolverId(
        miopenSolution_t solution,
        *mut uint64_t solverId,
    );

    miopenStatus_t miopenGetSolverIdConvAlgorithm(
        uint64_t solverId,
        *mut miopenConvAlgorithm_t result,
    );
}
