use super::types::*;
use crate::{hip::*, hipblas::*};
use rocm_include::ctype::*;

rocm_include::link! {
    hipblaslt : rocm_ver;
    dev exists_ pastey::paste;

    hipblasStatus_t hipblasLtCreate(*mut hipblasLtHandle_t handle);

    hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle);

    hipblasStatus_t hipblasLtMatrixLayoutCreate(*mut hipblasLtMatrixLayout_t matLayout, hipDataType type_, uint64_t rows, uint64_t cols, int64_t ld);

    hipblasStatus_t hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t matLayout);

    hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t matLayout, hipblasLtMatrixLayoutAttribute_t attr, *const void buf, size_t sizeInBytes);

    hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(hipblasLtMatrixLayout_t matLayout, hipblasLtMatrixLayoutAttribute_t attr, *mut void buf, size_t sizeInBytes, *mut size_t sizeWritten);

    hipblasStatus_t hipblasLtMatmulDescCreate(*mut hipblasLtMatmulDesc_t matmulDesc, hipblasComputeType_t computeType, hipDataType scaleType);

    hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t matmulDesc);

    hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatmulDescAttributes_t attr, *const void buf, size_t sizeInBytes);

    hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatmulDescAttributes_t attr, *mut void buf, size_t sizeInBytes, *mut size_t sizeWritten);

    hipblasStatus_t hipblasLtMatmulPreferenceCreate(*mut hipblasLtMatmulPreference_t pref);

    hipblasStatus_t hipblasLtMatmulPreferenceDestroy(hipblasLtMatmulPreference_t pref);

    hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, *const void buf, size_t sizeInBytes);

    hipblasStatus_t hipblasLtMatmulPreferenceGetAttribute(hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, *mut void buf, size_t sizeInBytes, *mut size_t sizeWritten);

    hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc, hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc, hipblasLtMatmulPreference_t pref, int requestedAlgoCount, *mut hipblasLtMatmulHeuristicResult_t heuristicResultsArray, *mut int returnAlgoCount);

    hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc, *const void alpha, *const void A, hipblasLtMatrixLayout_t Adesc, *const void B, hipblasLtMatrixLayout_t Bdesc, *const void beta, *const void C, hipblasLtMatrixLayout_t Cdesc, *mut void D, hipblasLtMatrixLayout_t Ddesc, *const hipblasLtMatmulAlgo_t algo, *mut void workspace, size_t workspaceSizeInBytes, hipStream_t stream);

    hipblasStatus_t hipblasLtMatrixTransformDescCreate(*mut hipblasLtMatrixTransformDesc_t transformDesc, hipDataType scaleType);

    hipblasStatus_t hipblasLtMatrixTransformDescDestroy(hipblasLtMatrixTransformDesc_t transformDesc);

    hipblasStatus_t hipblasLtMatrixTransformDescSetAttribute(hipblasLtMatrixTransformDesc_t transformDesc, hipblasLtMatrixTransformDescAttributes_t attr, *const void buf, size_t sizeInBytes);

    hipblasStatus_t hipblasLtMatrixTransformDescGetAttribute(hipblasLtMatrixTransformDesc_t transformDesc, hipblasLtMatrixTransformDescAttributes_t attr, *mut void buf, size_t sizeInBytes, *mut size_t sizeWritten);

    hipblasStatus_t hipblasLtMatrixTransform(hipblasLtHandle_t lightHandle, hipblasLtMatrixTransformDesc_t transformDesc, *const void alpha, *const void A, hipblasLtMatrixLayout_t Adesc, *const void beta, *const void B, hipblasLtMatrixLayout_t Bdesc, *mut void C, hipblasLtMatrixLayout_t Cdesc, hipStream_t stream);
}
