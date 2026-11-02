use super::types::*;
use crate::{hip::*, internal::*};

link! {
    hipblas : [7, 8, 9, 10] : rocm_ver;

    hipblasStatus_t hipblasCreate(*mut hipblasHandle_t handle);

    hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
    
    hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId);

    hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, *mut hipStream_t streamId);

    hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode);

    hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, *mut hipblasPointerMode_t mode);

    hipblasStatus_t hipblasSetVector(int n, int elemSize, *const void x, int incx, *mut void y, int incy);

    hipblasStatus_t hipblasGetVector(int n, int elemSize, *const void x, int incx, *mut void y, int incy);

    hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, *const void AP, int lda, *mut void BP, int ldb);

    hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, *const void AP, int lda, *mut void BP, int ldb);

    hipblasStatus_t hipblasSetVectorAsync(int n, int elemSize, *const void x, int incx, *mut void y, int incy, hipStream_t stream);

    hipblasStatus_t hipblasGetVectorAsync(int n, int elemSize, *const void x, int incx, *mut void y, int incy, hipStream_t stream);

    hipblasStatus_t hipblasSetMatrixAsync(int rows, int cols, int elemSize, *const void AP, int lda, *mut void BP, int ldb, hipStream_t stream);

    hipblasStatus_t hipblasGetMatrixAsync(int rows, int cols, int elemSize, *const void AP, int lda, *mut void BP, int ldb, hipStream_t stream);

    hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode);

    hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle, *mut hipblasAtomicsMode_t atomics_mode);

    hipblasStatus_t hipblasSetBatchAlphaStride(hipblasHandle_t handle, hipblasStride alpha_stride);

    hipblasStatus_t hipblasGetBatchAlphaStride(hipblasHandle_t handle, *mut hipblasStride alpha_stride);

    hipblasStatus_t hipblasSetBatchBetaStride(hipblasHandle_t handle, hipblasStride beta_stride);

    hipblasStatus_t hipblasGetBatchBetaStride(hipblasHandle_t handle, *mut hipblasStride beta_stride);

    *const char hipblasStatusToString(hipblasStatus_t status);

    hipblasStatus_t hipblasGetProperty(hipblasLibraryProperty_t type_, *mut int value);

    // Level 1 BLAS
    hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, *const float x, int incx, *mut int result);

    hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, *const double x, int incx, *mut int result);

    hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n, *const hipComplex x, int incx, *mut int result);

    hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, *mut int result);

    hipblasStatus_t hipblasIsamaxBatched(hipblasHandle_t handle, int n, *const *const float x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIdamaxBatched(hipblasHandle_t handle, int n, *const *const double x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIcamaxBatched(hipblasHandle_t handle, int n, *const *const hipComplex x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIzamaxBatched(hipblasHandle_t handle, int n, *const *const hipDoubleComplex x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIsamaxStridedBatched(hipblasHandle_t handle, int n, *const float x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle, int n, *const double x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t handle, int n, *const hipComplex x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, *const float x, int incx, *mut int result);

    hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, *const double x, int incx, *mut int result);

    hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n, *const hipComplex x, int incx, *mut int result);

    hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, *mut int result);

    hipblasStatus_t hipblasIsaminBatched(hipblasHandle_t handle, int n, *const *const float x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIdaminBatched(hipblasHandle_t handle, int n, *const *const double x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIcaminBatched(hipblasHandle_t handle, int n, *const *const hipComplex x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIzaminBatched(hipblasHandle_t handle, int n, *const *const hipDoubleComplex x, int incx, int batchCount, *mut int result);

    hipblasStatus_t hipblasIsaminStridedBatched(hipblasHandle_t handle, int n, *const float x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle, int n, *const double x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t handle, int n, *const hipComplex x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount, *mut int result);

    hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, *const float x, int incx, *mut float result);

    hipblasStatus_t hipblasDasum(hipblasHandle_t handle, int n, *const double x, int incx, *mut double result);

    hipblasStatus_t hipblasScasum(hipblasHandle_t handle, int n, *const hipComplex x, int incx, *mut float result);

    hipblasStatus_t hipblasDzasum(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, *mut double result);

    hipblasStatus_t hipblasSasumBatched(hipblasHandle_t handle, int n, *const *const float x, int incx, int batchCount, *mut float result);

    hipblasStatus_t hipblasDasumBatched(hipblasHandle_t handle, int n, *const *const double x, int incx, int batchCount, *mut double result);

    hipblasStatus_t hipblasScasumBatched(hipblasHandle_t handle, int n, *const *const hipComplex x, int incx, int batchCount, *mut float result);

    hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t handle, int n, *const *const hipDoubleComplex x, int incx, int batchCount, *mut double result);

    hipblasStatus_t hipblasSasumStridedBatched(hipblasHandle_t handle, int n, *const float x, int incx, hipblasStride stridex, int batchCount, *mut float result);

    hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle, int n, *const double x, int incx, hipblasStride stridex, int batchCount, *mut double result);

    hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t handle, int n, *const hipComplex x, int incx, hipblasStride stridex, int batchCount, *mut float result);
    
    hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount, *mut double result);

    hipblasStatus_t hipblasHaxpy(hipblasHandle_t handle, int n, *const hipblasHalf alpha, *const hipblasHalf x, int incx, *mut hipblasHalf y, int incy);

    hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, *const float alpha, *const float x, int incx, *mut float y, int incy);

    hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, *const double alpha, *const double x, int incx, *mut double y, int incy);
    
    hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle, int n, *const hipComplex alpha, *const hipComplex x, int incx, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasHaxpyBatched(hipblasHandle_t handle, int n, *const hipblasHalf alpha, *const *const hipblasHalf x, int incx, *const *mut hipblasHalf y, int incy, int batchCount);

    hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, *const float alpha, *const *const float x, int incx, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDaxpyBatched(hipblasHandle_t handle, int n, *const double alpha, *const *const double x, int incx, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasCaxpyBatched(hipblasHandle_t handle, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZaxpyBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasHaxpyStridedBatched(hipblasHandle_t handle, int n, *const hipblasHalf alpha, *const hipblasHalf x, int incx, hipblasStride stridex, *mut hipblasHalf y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasSaxpyStridedBatched(hipblasHandle_t handle, int n, *const float alpha, *const float x, int incx, hipblasStride stridex, *mut float y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasDaxpyStridedBatched(hipblasHandle_t handle, int n, *const double alpha, *const double x, int incx, hipblasStride stridex, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasCaxpyStridedBatched(hipblasHandle_t handle, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZaxpyStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, *const float x, int incx, *mut float y, int incy);

    hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, *const double x, int incx, *mut double y, int incy);

    hipblasStatus_t hipblasCcopy(hipblasHandle_t handle, int n, *const hipComplex x, int incx, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZcopy(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, *const *const float x, int incx, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, *const *const double x, int incx, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t handle, int n, *const *const hipComplex x, int incx, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t handle, int n, *const *const hipDoubleComplex x, int incx, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasScopyStridedBatched(hipblasHandle_t handle, int n, *const float x, int incx, hipblasStride stridex, *mut float y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle, int n, *const double x, int incx, hipblasStride stridex, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t handle, int n, *const hipComplex x, int incx, hipblasStride stridex, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasHdot(hipblasHandle_t handle, int n, *const hipblasHalf x, int incx, *const hipblasHalf y, int incy, *mut hipblasHalf result);

    hipblasStatus_t hipblasBfdot(hipblasHandle_t handle, int n, *const hipblasBfloat16 x, int incx, *const hipblasBfloat16 y, int incy, *mut hipblasBfloat16 result);

    hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, *const float x, int incx, *const float y, int incy, *mut float result);

    hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n, *const double x, int incx, *const double y, int incy, *mut double result);

    hipblasStatus_t hipblasCdotc(hipblasHandle_t handle, int n, *const hipComplex x, int incx, *const hipComplex y, int incy, *mut hipComplex result);

    hipblasStatus_t hipblasCdotu(hipblasHandle_t handle, int n, *const hipComplex x, int incx, *const hipComplex y, int incy, *mut hipComplex result);

    hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, *const hipDoubleComplex y, int incy, *mut hipDoubleComplex result);

    hipblasStatus_t hipblasZdotu(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, *const hipDoubleComplex y, int incy, *mut hipDoubleComplex result);

    hipblasStatus_t hipblasHdotBatched(hipblasHandle_t handle, int n, *const *const hipblasHalf x, int incx, *const *const hipblasHalf y, int incy, int batchCount, *mut hipblasHalf result);

    hipblasStatus_t hipblasBfdotBatched(hipblasHandle_t handle, int n, *const *const hipblasBfloat16 x, int incx, *const *const hipblasBfloat16 y, int incy, int batchCount, *mut hipblasBfloat16 result);

    hipblasStatus_t hipblasSdotBatched(hipblasHandle_t handle, int n, *const *const float x, int incx, *const *const float y, int incy, int batchCount, *mut float result);

    hipblasStatus_t hipblasDdotBatched(hipblasHandle_t handle, int n, *const *const double x, int incx, *const *const double y, int incy, int batchCount, *mut double result);

    hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t handle, int n, *const *const hipComplex x, int incx, *const *const hipComplex y, int incy, int batchCount, *mut hipComplex result);

    hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t handle, int n, *const *const hipComplex x, int incx, *const *const hipComplex y, int incy, int batchCount, *mut hipComplex result);

    hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t handle, int n, *const *const hipDoubleComplex x, int incx, *const *const hipDoubleComplex y, int incy, int batchCount, *mut hipDoubleComplex result);

    hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t handle, int n, *const *const hipDoubleComplex x, int incx, *const *const hipDoubleComplex y, int incy, int batchCount, *mut hipDoubleComplex result);

    hipblasStatus_t hipblasHdotStridedBatched(hipblasHandle_t handle, int n, *const hipblasHalf x, int incx, hipblasStride stridex, *const hipblasHalf y, int incy, hipblasStride stridey, int batchCount, *mut hipblasHalf result);

    hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t handle, int n, *const hipblasBfloat16 x, int incx, hipblasStride stridex, *const hipblasBfloat16 y, int incy, hipblasStride stridey, int batchCount, *mut hipblasBfloat16 result);

    hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle, int n, *const float x, int incx, hipblasStride stridex, *const float y, int incy, hipblasStride stridey, int batchCount, *mut float result);

    hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle, int n, *const double x, int incx, hipblasStride stridex, *const double y, int incy, hipblasStride stridey, int batchCount, *mut double result);

    hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t handle, int n, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex y, int incy, hipblasStride stridey, int batchCount, *mut hipComplex result);

    hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t handle, int n, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex y, int incy, hipblasStride stridey, int batchCount, *mut hipComplex result);

    hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount, *mut hipDoubleComplex result);

    hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount, *mut hipDoubleComplex result);

    hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, *const float x, int incx, *mut float result);

    hipblasStatus_t hipblasDnrm2(hipblasHandle_t handle, int n, *const double x, int incx, *mut double result);

    hipblasStatus_t hipblasScnrm2(hipblasHandle_t handle, int n, *const hipComplex x, int incx, *mut float result);

    hipblasStatus_t hipblasDznrm2(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, *mut double result);

    hipblasStatus_t hipblasSnrm2Batched(hipblasHandle_t handle, int n, *const *const float x, int incx, int batchCount, *mut float result);

    hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t handle, int n, *const *const double x, int incx, int batchCount, *mut double result);

    hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t handle, int n, *const *const hipComplex x, int incx, int batchCount, *mut float result);

    hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t handle, int n, *const *const hipDoubleComplex x, int incx, int batchCount, *mut double result);

    hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle, int n, *const float x, int incx, hipblasStride stridex, int batchCount, *mut float result);

    hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle, int n, *const double x, int incx, hipblasStride stridex, int batchCount, *mut double result);

    hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t handle, int n, *const hipComplex x, int incx, hipblasStride stridex, int batchCount, *mut float result);

    hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount, *mut double result);

    hipblasStatus_t hipblasSrot(hipblasHandle_t handle, int n, *mut float x, int incx, *mut float y, int incy, *const float c, *const float s);

    hipblasStatus_t hipblasDrot(hipblasHandle_t handle, int n, *mut double x, int incx, *mut double y, int incy, *const double c, *const double s);

    hipblasStatus_t hipblasCrot(hipblasHandle_t handle, int n, *mut hipComplex x, int incx, *mut hipComplex y, int incy, *const float c, *const hipComplex s);

    hipblasStatus_t hipblasCsrot(hipblasHandle_t handle, int n, *mut hipComplex x, int incx, *mut hipComplex y, int incy, *const float c, *const float s);

    hipblasStatus_t hipblasZrot(hipblasHandle_t handle, int n, *mut hipDoubleComplex x, int incx, *mut hipDoubleComplex y, int incy, *const double c, *const hipDoubleComplex s);

    hipblasStatus_t hipblasZdrot(hipblasHandle_t handle, int n, *mut hipDoubleComplex x, int incx, *mut hipDoubleComplex y, int incy, *const double c, *const double s);

    hipblasStatus_t hipblasSrotBatched(hipblasHandle_t handle, int n, *const *mut float x, int incx, *const *mut float y, int incy, *const float c, *const float s, int batchCount);

    hipblasStatus_t hipblasDrotBatched(hipblasHandle_t handle, int n, *const *mut double x, int incx, *const *mut double y, int incy, *const double c, *const double s, int batchCount);

    hipblasStatus_t hipblasCrotBatched(hipblasHandle_t handle, int n, *const *mut hipComplex x, int incx, *const *mut hipComplex y, int incy, *const float c, *const hipComplex s, int batchCount);

    hipblasStatus_t hipblasCsrotBatched(hipblasHandle_t handle, int n, *const *mut hipComplex x, int incx, *const *mut hipComplex y, int incy, *const float c, *const float s, int batchCount);

    hipblasStatus_t hipblasZrotBatched(hipblasHandle_t handle, int n, *const *mut hipDoubleComplex x, int incx, *const *mut hipDoubleComplex y, int incy, *const double c, *const hipDoubleComplex s, int batchCount);

    hipblasStatus_t hipblasZdrotBatched(hipblasHandle_t handle, int n, *const *mut hipDoubleComplex x, int incx, *const *mut hipDoubleComplex y, int incy, *const double c, *const double s, int batchCount);

    hipblasStatus_t hipblasSrotStridedBatched(hipblasHandle_t handle, int n, *mut float x, int incx, hipblasStride stridex, *mut float y, int incy, hipblasStride stridey, *const float c, *const float s, int batchCount);

    hipblasStatus_t hipblasDrotStridedBatched(hipblasHandle_t handle, int n, *mut double x, int incx, hipblasStride stridex, *mut double y, int incy, hipblasStride stridey, *const double c, *const double s, int batchCount);

    hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t handle, int n, *mut hipComplex x, int incx, hipblasStride stridex, *mut hipComplex y, int incy, hipblasStride stridey, *const float c, *const hipComplex s, int batchCount);

    hipblasStatus_t hipblasCsrotStridedBatched(hipblasHandle_t handle, int n, *mut hipComplex x, int incx, hipblasStride stridex, *mut hipComplex y, int incy, hipblasStride stridey, *const float c, *const float s, int batchCount);

    hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t handle, int n, *mut hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex y, int incy, hipblasStride stridey, *const double c, *const hipDoubleComplex s, int batchCount);

    hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t handle, int n, *mut hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex y, int incy, hipblasStride stridey, *const double c, *const double s, int batchCount);

    hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, *mut float a, *mut float b, *mut float c, *mut float s);

    hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, *mut double a, *mut double b, *mut double c, *mut double s);

    hipblasStatus_t hipblasCrotg(hipblasHandle_t handle, *mut hipComplex a, *mut hipComplex b, *mut float c, *mut hipComplex s);

    hipblasStatus_t hipblasZrotg(hipblasHandle_t handle, *mut hipDoubleComplex a, *mut hipDoubleComplex b, *mut double c, *mut hipDoubleComplex s);

    hipblasStatus_t hipblasSrotgBatched(hipblasHandle_t handle, *const *mut float a, *const *mut float b, *const *mut float c, *const *mut float s, int batchCount);

    hipblasStatus_t hipblasDrotgBatched(hipblasHandle_t handle, *const *mut double a, *const *mut double b, *const *mut double c, *const *mut double s, int batchCount);

    hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t handle, *const *mut hipComplex a, *const *mut hipComplex b, *const *mut float c, *const *mut hipComplex s, int batchCount);

    hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t handle, *const *mut hipDoubleComplex a, *const *mut hipDoubleComplex b, *const *mut double c, *const *mut hipDoubleComplex s, int batchCount);

    hipblasStatus_t hipblasSrotgStridedBatched(hipblasHandle_t handle, *mut float a, hipblasStride stridea, *mut float b, hipblasStride strideb, *mut float c, hipblasStride stridec, *mut float s, hipblasStride strides, int batchCount);

    hipblasStatus_t hipblasDrotgStridedBatched(hipblasHandle_t handle, *mut double a, hipblasStride stridea, *mut double b, hipblasStride strideb, *mut double c, hipblasStride stridec, *mut double s, hipblasStride strides, int batchCount);

    hipblasStatus_t hipblasCrotgStridedBatched(hipblasHandle_t handle, *mut hipComplex a, hipblasStride stridea, *mut hipComplex b, hipblasStride strideb, *mut float c, hipblasStride stridec, *mut hipComplex s, hipblasStride strides, int batchCount);

    hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t handle, *mut hipDoubleComplex a, hipblasStride stridea, *mut hipDoubleComplex b, hipblasStride strideb, *mut double c, hipblasStride stridec, *mut hipDoubleComplex s, hipblasStride strides, int batchCount);

    hipblasStatus_t hipblasSrotm(hipblasHandle_t handle, int n, *mut float x, int incx, *mut float y, int incy, *const float param);

    hipblasStatus_t hipblasDrotm(hipblasHandle_t handle, int n, *mut double x, int incx, *mut double y, int incy, *const double param);

    hipblasStatus_t hipblasSrotmBatched(hipblasHandle_t handle, int n, *const *mut float x, int incx, *const *mut float y, int incy, *const *const float param, int batchCount);

    hipblasStatus_t hipblasDrotmBatched(hipblasHandle_t handle, int n, *const *mut double x, int incx, *const *mut double y, int incy, *const *const double param, int batchCount);

    hipblasStatus_t hipblasSrotmStridedBatched(hipblasHandle_t handle, int n, *mut float x, int incx, hipblasStride stridex, *mut float y, int incy, hipblasStride stridey, *const float param, hipblasStride strideParam, int batchCount);

    hipblasStatus_t hipblasDrotmStridedBatched(hipblasHandle_t handle, int n, *mut double x, int incx, hipblasStride stridex, *mut double y, int incy, hipblasStride stridey, *const double param, hipblasStride strideParam, int batchCount);

    hipblasStatus_t hipblasSrotmg(hipblasHandle_t handle, *mut float d1, *mut float d2, *mut float x1, *const float y1, *mut float param);

    hipblasStatus_t hipblasDrotmg(hipblasHandle_t handle, *mut double d1, *mut double d2, *mut double x1, *const double y1, *mut double param);

    hipblasStatus_t hipblasSrotmgBatched(hipblasHandle_t handle, *const *mut float d1, *const *mut float d2, *const *mut float x1, *const *const float y1, *const *mut float param, int batchCount);

    hipblasStatus_t hipblasDrotmgBatched(hipblasHandle_t handle, *const *mut double d1, *const *mut double d2, *const *mut double x1, *const *const double y1, *const *mut double param, int batchCount);

    hipblasStatus_t hipblasSrotmgStridedBatched(hipblasHandle_t handle, *mut float d1, hipblasStride strided1, *mut float d2, hipblasStride strided2, *mut float x1, hipblasStride stridex1, *const float y1, hipblasStride stridey1, *mut float param, hipblasStride strideParam, int batchCount);

    hipblasStatus_t hipblasDrotmgStridedBatched(hipblasHandle_t handle, *mut double d1, hipblasStride strided1, *mut double d2, hipblasStride strided2, *mut double x1, hipblasStride stridex1, *const double y1, hipblasStride stridey1, *mut double param, hipblasStride strideParam, int batchCount);

    hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, *const float alpha, *mut float x, int incx);

    hipblasStatus_t hipblasDscal(hipblasHandle_t handle, int n, *const double alpha, *mut double x, int incx);

    hipblasStatus_t hipblasCscal(hipblasHandle_t handle, int n, *const hipComplex alpha, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasCsscal(hipblasHandle_t handle, int n, *const float alpha, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasZscal(hipblasHandle_t handle, int n, *const hipDoubleComplex alpha, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasZdscal(hipblasHandle_t handle, int n, *const double alpha, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasSscalBatched(hipblasHandle_t handle, int n, *const float alpha, *const *mut float x, int incx, int batchCount);

    hipblasStatus_t hipblasDscalBatched(hipblasHandle_t handle, int n, *const double alpha, *const *mut double x, int incx, int batchCount);

    hipblasStatus_t hipblasCscalBatched(hipblasHandle_t handle, int n, *const hipComplex alpha, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZscalBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex alpha, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t handle, int n, *const float alpha, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t handle, int n, *const double alpha, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle, int n, *const float alpha, *mut float x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle, int n, *const double alpha, *mut double x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t handle, int n, *const hipComplex alpha, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t handle, int n, *const hipDoubleComplex alpha, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle, int n, *const float alpha, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t handle, int n, *const double alpha, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, *mut float x, int incx, *mut float y, int incy);

    hipblasStatus_t hipblasDswap(hipblasHandle_t handle, int n, *mut double x, int incx, *mut double y, int incy);

    hipblasStatus_t hipblasCswap(hipblasHandle_t handle, int n, *mut hipComplex x, int incx, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZswap(hipblasHandle_t handle, int n, *mut hipDoubleComplex x, int incx, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasSswapBatched(hipblasHandle_t handle, int n, *const *mut float x, int incx, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDswapBatched(hipblasHandle_t handle, int n, *const *mut double x, int incx, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasCswapBatched(hipblasHandle_t handle, int n, *const *mut hipComplex x, int incx, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZswapBatched(hipblasHandle_t handle, int n, *const *mut hipDoubleComplex x, int incx, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasSswapStridedBatched(hipblasHandle_t handle, int n, *mut float x, int incx, hipblasStride stridex, *mut float y, int incy, hipblasStride stridey, int batchCount);
    
    hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle, int n, *mut double x, int incx, hipblasStride stridex, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle, int n, *mut hipComplex x, int incx, hipblasStride stridex, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t handle, int n, *mut hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    // Level 2 BLAS
    hipblasStatus_t hipblasSgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const float alpha, *const float AP, int lda, *const float x, int incx, *const float beta, *mut float y, int incy);

    hipblasStatus_t hipblasDgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const double alpha, *const double AP, int lda, *const double x, int incx, *const double beta, *mut double y, int incy);

    hipblasStatus_t hipblasCgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const hipComplex alpha, *const hipComplex AP, int lda, *const hipComplex x, int incx, *const hipComplex beta, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasSgbmvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const float alpha, *const *const float AP, int lda, *const *const float x, int incx, *const float beta, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDgbmvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const double alpha, *const *const double AP, int lda, *const *const double x, int incx, *const double beta, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasCgbmvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const hipComplex alpha, *const *const hipComplex AP, int lda, *const *const hipComplex x, int incx, *const hipComplex beta, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZgbmvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const hipDoubleComplex alpha, *const *const hipDoubleComplex AP, int lda, *const *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasSgbmvStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const float alpha, *const float AP, int lda, hipblasStride strideA, *const float x, int incx, hipblasStride stridex, *const float beta, *mut float y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasDgbmvStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const double alpha, *const double AP, int lda, hipblasStride strideA, *const double x, int incx, hipblasStride stridex, *const double beta, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasCgbmvStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const hipComplex alpha, *const hipComplex AP, int lda, hipblasStride strideA, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex beta, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZgbmvStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const float alpha, *const float AP, int lda, *const float x, int incx, *const float beta, *mut float y, int incy);

    hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const double alpha, *const double AP, int lda, *const double x, int incx, *const double beta, *mut double y, int incy);

    hipblasStatus_t hipblasCgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const hipComplex alpha, *const hipComplex AP, int lda, *const hipComplex x, int incx, *const hipComplex beta, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const float alpha, *const *const float AP, int lda, *const *const float x, int incx, *const float beta, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const double alpha, *const *const double AP, int lda, *const *const double x, int incx, *const double beta, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasCgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const hipComplex alpha, *const *const hipComplex AP, int lda, *const *const hipComplex x, int incx, *const hipComplex beta, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex AP, int lda, *const *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasSgemvStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, int m, int n, *const float alpha, *const float AP, int lda, hipblasStride strideA, *const float x, int incx, hipblasStride stridex, *const float beta, *mut float y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, int m, int n, *const double alpha, *const double AP, int lda, hipblasStride strideA, *const double x, int incx, hipblasStride stridex, *const double beta, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, int m, int n, *const hipComplex alpha, *const hipComplex AP, int lda, hipblasStride strideA, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex beta, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, int m, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasSger(hipblasHandle_t handle, int m, int n, *const float alpha, *const float x, int incx, *const float y, int incy, *mut float AP, int lda);

    hipblasStatus_t hipblasDger(hipblasHandle_t handle, int m, int n, *const double alpha, *const double x, int incx, *const double y, int incy, *mut double AP, int lda);

    hipblasStatus_t hipblasCgeru(hipblasHandle_t handle, int m, int n, *const hipComplex alpha, *const hipComplex x, int incx, *const hipComplex y, int incy, *mut hipComplex AP, int lda);

    hipblasStatus_t hipblasCgerc(hipblasHandle_t handle, int m, int n, *const hipComplex alpha, *const hipComplex x, int incx, *const hipComplex y, int incy, *mut hipComplex AP, int lda);

    hipblasStatus_t hipblasZgeru(hipblasHandle_t handle, int m, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *const hipDoubleComplex y, int incy, *mut hipDoubleComplex AP, int lda);

    hipblasStatus_t hipblasZgerc(hipblasHandle_t handle, int m, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *const hipDoubleComplex y, int incy, *mut hipDoubleComplex AP, int lda);

    hipblasStatus_t hipblasSgerBatched(hipblasHandle_t handle, int m, int n, *const float alpha, *const *const float x, int incx, *const *const float y, int incy, *const *mut float AP, int lda, int batchCount);

    hipblasStatus_t hipblasDgerBatched(hipblasHandle_t handle, int m, int n, *const double alpha, *const *const double x, int incx, *const *const double y, int incy, *const *mut double AP, int lda, int batchCount);

    hipblasStatus_t hipblasCgeruBatched(hipblasHandle_t handle, int m, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *const hipComplex y, int incy, *const *mut hipComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasCgercBatched(hipblasHandle_t handle, int m, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *const hipComplex y, int incy, *const *mut hipComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasZgeruBatched(hipblasHandle_t handle, int m, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *const hipDoubleComplex y, int incy, *const *mut hipDoubleComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasZgercBatched(hipblasHandle_t handle, int m, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *const hipDoubleComplex y, int incy, *const *mut hipDoubleComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasSgerStridedBatched(hipblasHandle_t handle, int m, int n, *const float alpha, *const float x, int incx, hipblasStride stridex, *const float y, int incy, hipblasStride stridey, *mut float AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t handle, int m, int n, *const double alpha, *const double x, int incx, hipblasStride stridex, *const double y, int incy, hipblasStride stridey, *mut double AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasCgeruStridedBatched(hipblasHandle_t handle, int m, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex y, int incy, hipblasStride stridey, *mut hipComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasCgercStridedBatched(hipblasHandle_t handle, int m, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex y, int incy, hipblasStride stridey, *mut hipComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZgeruStridedBatched(hipblasHandle_t handle, int m, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex y, int incy, hipblasStride stridey, *mut hipDoubleComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZgercStridedBatched(hipblasHandle_t handle, int m, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex y, int incy, hipblasStride stridey, *mut hipDoubleComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasChbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const hipComplex alpha, *const hipComplex AP, int lda, *const hipComplex x, int incx, *const hipComplex beta, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZhbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasChbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const hipComplex alpha, *const *const hipComplex AP, int lda, *const *const hipComplex x, int incx, *const hipComplex beta, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZhbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const hipDoubleComplex alpha, *const *const hipDoubleComplex AP, int lda, *const *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasChbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const hipComplex alpha, *const hipComplex AP, int lda, hipblasStride strideA, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex beta, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZhbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasChemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex AP, int lda, *const hipComplex x, int incx, *const hipComplex beta, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZhemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasChemvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex AP, int lda, *const *const hipComplex x, int incx, *const hipComplex beta, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZhemvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex AP, int lda, *const *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasChemvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex AP, int lda, hipblasStride strideA, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex beta, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZhemvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasCher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const hipComplex x, int incx, *mut hipComplex AP, int lda);

    hipblasStatus_t hipblasZher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const hipDoubleComplex x, int incx, *mut hipDoubleComplex AP, int lda);

    hipblasStatus_t hipblasCherBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const hipComplex x, int incx, *const *mut hipComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasZherBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const hipDoubleComplex x, int incx, *const *mut hipDoubleComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasCherStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const hipComplex x, int incx, hipblasStride stridex, *mut hipComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZherStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasCher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, *const hipComplex y, int incy, *mut hipComplex AP, int lda);

    hipblasStatus_t hipblasZher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *const hipDoubleComplex y, int incy, *mut hipDoubleComplex AP, int lda);

    hipblasStatus_t hipblasCher2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *const hipComplex y, int incy, *const *mut hipComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasZher2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *const hipDoubleComplex y, int incy, *const *mut hipDoubleComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasCher2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex y, int incy, hipblasStride stridey, *mut hipComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZher2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex y, int incy, hipblasStride stridey, *mut hipDoubleComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasChpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex AP, *const hipComplex x, int incx, *const hipComplex beta, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZhpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasChpmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex AP, *const *const hipComplex x, int incx, *const hipComplex beta, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZhpmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex AP, *const *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasChpmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex AP, hipblasStride strideA, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex beta, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZhpmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, hipblasStride strideA, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasChpr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const hipComplex x, int incx, *mut hipComplex AP);

    hipblasStatus_t hipblasZhpr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const hipDoubleComplex x, int incx, *mut hipDoubleComplex AP);

    hipblasStatus_t hipblasChprBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const hipComplex x, int incx, *const *mut hipComplex AP, int batchCount);

    hipblasStatus_t hipblasZhprBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const hipDoubleComplex x, int incx, *const *mut hipDoubleComplex AP, int batchCount);

    hipblasStatus_t hipblasChprStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const hipComplex x, int incx, hipblasStride stridex, *mut hipComplex AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZhprStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasChpr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, *const hipComplex y, int incy, *mut hipComplex AP);

    hipblasStatus_t hipblasZhpr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *const hipDoubleComplex y, int incy, *mut hipDoubleComplex AP);

    hipblasStatus_t hipblasChpr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *const hipComplex y, int incy, *const *mut hipComplex AP, int batchCount);

    hipblasStatus_t hipblasZhpr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *const hipDoubleComplex y, int incy, *const *mut hipDoubleComplex AP, int batchCount);

    hipblasStatus_t hipblasChpr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex y, int incy, hipblasStride stridey, *mut hipComplex AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZhpr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex y, int incy, hipblasStride stridey, *mut hipDoubleComplex AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasSsbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const float alpha, *const float AP, int lda, *const float x, int incx, *const float beta, *mut float y, int incy);

    hipblasStatus_t hipblasDsbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const double alpha, *const double AP, int lda, *const double x, int incx, *const double beta, *mut double y, int incy);

    hipblasStatus_t hipblasSsbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const float alpha, *const *const float AP, int lda, *const *const float x, int incx, *const float beta, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDsbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const double alpha, *const *const double AP, int lda, *const *const double x, int incx, *const double beta, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasSsbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const float alpha, *const float AP, int lda, hipblasStride strideA, *const float x, int incx, hipblasStride stridex, *const float beta, *mut float y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasDsbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, *const double alpha, *const double AP, int lda, hipblasStride strideA, *const double x, int incx, hipblasStride stridex, *const double beta, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasSspmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float AP, *const float x, int incx, *const float beta, *mut float y, int incy);

    hipblasStatus_t hipblasDspmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double AP, *const double x, int incx, *const double beta, *mut double y, int incy);

    hipblasStatus_t hipblasSspmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const float AP, *const *const float x, int incx, *const float beta, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDspmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const double AP, *const *const double x, int incx, *const double beta, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasSspmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float AP, hipblasStride strideA, *const float x, int incx, hipblasStride stridex, *const float beta, *mut float y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasDspmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double AP, hipblasStride strideA, *const double x, int incx, hipblasStride stridex, *const double beta, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasSspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, *mut float AP);

    hipblasStatus_t hipblasDspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, *mut double AP);

    hipblasStatus_t hipblasCspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, *mut hipComplex AP);

    hipblasStatus_t hipblasZspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *mut hipDoubleComplex AP);

    hipblasStatus_t hipblasSsprBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const float x, int incx, *const *mut float AP, int batchCount);

    hipblasStatus_t hipblasDsprBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const double x, int incx, *const *mut double AP, int batchCount);

    hipblasStatus_t hipblasCsprBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *mut hipComplex AP, int batchCount);

    hipblasStatus_t hipblasZsprBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *mut hipDoubleComplex AP, int batchCount);

    hipblasStatus_t hipblasSsprStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, hipblasStride stridex, *mut float AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasDsprStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, hipblasStride stridex, *mut double AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasCsprStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *mut hipComplex AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZsprStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasSspr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, *const float y, int incy, *mut float AP);

    hipblasStatus_t hipblasDspr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, *const double y, int incy, *mut double AP);

    hipblasStatus_t hipblasSspr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const float x, int incx, *const *const float y, int incy, *const *mut float AP, int batchCount);

    hipblasStatus_t hipblasDspr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const double x, int incx, *const *const double y, int incy, *const *mut double AP, int batchCount);

    hipblasStatus_t hipblasSspr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, hipblasStride stridex, *const float y, int incy, hipblasStride stridey, *mut float AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasDspr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, hipblasStride stridex, *const double y, int incy, hipblasStride stridey, *mut double AP, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasSsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float AP, int lda, *const float x, int incx, *const float beta, *mut float y, int incy);

    hipblasStatus_t hipblasDsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double AP, int lda, *const double x, int incx, *const double beta, *mut double y, int incy);

    hipblasStatus_t hipblasCsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex AP, int lda, *const hipComplex x, int incx, *const hipComplex beta, *mut hipComplex y, int incy);

    hipblasStatus_t hipblasZsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy);

    hipblasStatus_t hipblasSsymvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const float AP, int lda, *const *const float x, int incx, *const float beta, *const *mut float y, int incy, int batchCount);

    hipblasStatus_t hipblasDsymvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const double AP, int lda, *const *const double x, int incx, *const double beta, *const *mut double y, int incy, int batchCount);

    hipblasStatus_t hipblasCsymvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex AP, int lda, *const *const hipComplex x, int incx, *const hipComplex beta, *const *mut hipComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasZsymvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex AP, int lda, *const *const hipDoubleComplex x, int incx, *const hipDoubleComplex beta, *const *mut hipDoubleComplex y, int incy, int batchCount);

    hipblasStatus_t hipblasSsymvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float AP, int lda, hipblasStride strideA, *const float x, int incx, hipblasStride stridex, *const float beta, *mut float y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasDsymvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double AP, int lda, hipblasStride strideA, *const double x, int incx, hipblasStride stridex, *const double beta, *mut double y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasCsymvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex AP, int lda, hipblasStride strideA, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex beta, *mut hipComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasZsymvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex beta, *mut hipDoubleComplex y, int incy, hipblasStride stridey, int batchCount);

    hipblasStatus_t hipblasSsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, *mut float AP, int lda);

    hipblasStatus_t hipblasDsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, *mut double AP, int lda);

    hipblasStatus_t hipblasCsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, *mut hipComplex AP, int lda);

    hipblasStatus_t hipblasZsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *mut hipDoubleComplex AP, int lda);

    hipblasStatus_t hipblasSsyrBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const float x, int incx, *const *mut float AP, int lda, int batchCount);

    hipblasStatus_t hipblasDsyrBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const double x, int incx, *const *mut double AP, int lda, int batchCount);

    hipblasStatus_t hipblasCsyrBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *mut hipComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasZsyrBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *mut hipDoubleComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasSsyrStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, hipblasStride stridex, *mut float AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasDsyrStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, hipblasStride stridex, *mut double AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasCsyrStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *mut hipComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZsyrStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *mut hipDoubleComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasSsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, *const float y, int incy, *mut float AP, int lda);

    hipblasStatus_t hipblasDsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, *const double y, int incy, *mut double AP, int lda);

    hipblasStatus_t hipblasCsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, *const hipComplex y, int incy, *mut hipComplex AP, int lda);

    hipblasStatus_t hipblasZsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, *const hipDoubleComplex y, int incy, *mut hipDoubleComplex AP, int lda);

    hipblasStatus_t hipblasSsyr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const *const float x, int incx, *const *const float y, int incy, *const *mut float AP, int lda, int batchCount);

    hipblasStatus_t hipblasDsyr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const *const double x, int incx, *const *const double y, int incy, *const *mut double AP, int lda, int batchCount);

    hipblasStatus_t hipblasCsyr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const *const hipComplex x, int incx, *const *const hipComplex y, int incy, *const *mut hipComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasZsyr2Batched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const *const hipDoubleComplex x, int incx, *const *const hipDoubleComplex y, int incy, *const *mut hipDoubleComplex AP, int lda, int batchCount);

    hipblasStatus_t hipblasSsyr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const float alpha, *const float x, int incx, hipblasStride stridex, *const float y, int incy, hipblasStride stridey, *mut float AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasDsyr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const double alpha, *const double x, int incx, hipblasStride stridex, *const double y, int incy, hipblasStride stridey, *mut double AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasCsyr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipComplex alpha, *const hipComplex x, int incx, hipblasStride stridex, *const hipComplex y, int incy, hipblasStride stridey, *mut hipComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasZsyr2StridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, *const hipDoubleComplex alpha, *const hipDoubleComplex x, int incx, hipblasStride stridex, *const hipDoubleComplex y, int incy, hipblasStride stridey, *mut hipDoubleComplex AP, int lda, hipblasStride strideA, int batchCount);

    hipblasStatus_t hipblasStbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const float AP, int lda, *mut float x, int incx);

    hipblasStatus_t hipblasDtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const double AP, int lda, *mut double x, int incx);

    hipblasStatus_t hipblasCtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipComplex AP, int lda, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasZtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipDoubleComplex AP, int lda, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasStbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const float AP, int lda, *const *mut float x, int incx, int batchCount);

    hipblasStatus_t hipblasDtbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const double AP, int lda, *const *mut double x, int incx, int batchCount);

    hipblasStatus_t hipblasCtbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const hipComplex AP, int lda, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZtbmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const hipDoubleComplex AP, int lda, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasStbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const float AP, int lda, hipblasStride strideA, *mut float x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasDtbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const double AP, int lda, hipblasStride strideA, *mut double x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCtbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipComplex AP, int lda, hipblasStride strideA, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);
    
    hipblasStatus_t hipblasZtbmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasStbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const float AP, int lda, *mut float x, int incx);

    hipblasStatus_t hipblasDtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const double AP, int lda, *mut double x, int incx);

    hipblasStatus_t hipblasCtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipComplex AP, int lda, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasZtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipDoubleComplex AP, int lda, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasStbsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const float AP, int lda, *const *mut float x, int incx, int batchCount);

    hipblasStatus_t hipblasDtbsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const double AP, int lda, *const *mut double x, int incx, int batchCount);

    hipblasStatus_t hipblasCtbsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const hipComplex AP, int lda, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZtbsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const *const hipDoubleComplex AP, int lda, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasStbsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const float AP, int lda, hipblasStride strideA, *mut float x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasDtbsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const double AP, int lda, hipblasStride strideA, *mut double x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCtbsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipComplex AP, int lda, hipblasStride strideA, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasZtbsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasStpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, *mut float x, int incx);

    hipblasStatus_t hipblasDtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, *mut double x, int incx);

    hipblasStatus_t hipblasCtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasZtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasStpmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const float AP, *const *mut float x, int incx, int batchCount);

    hipblasStatus_t hipblasDtpmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const double AP, *const *mut double x, int incx, int batchCount);

    hipblasStatus_t hipblasCtpmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipComplex AP, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZtpmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipDoubleComplex AP, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasStpmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, hipblasStride strideA, *mut float x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasDtpmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, hipblasStride strideA, *mut double x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCtpmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, hipblasStride strideA, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasZtpmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, hipblasStride strideA, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasStpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, *mut float x, int incx);

    hipblasStatus_t hipblasDtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, *mut double x, int incx);

    hipblasStatus_t hipblasCtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasZtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasStpsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const float AP, *const *mut float x, int incx, int batchCount);

    hipblasStatus_t hipblasDtpsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const double AP, *const *mut double x, int incx, int batchCount);

    hipblasStatus_t hipblasCtpsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipComplex AP, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZtpsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipDoubleComplex AP, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasStpsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, hipblasStride strideA, *mut float x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasDtpsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, hipblasStride strideA, *mut double x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCtpsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, hipblasStride strideA, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasZtpsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, hipblasStride strideA, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasStrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, int lda, *mut float x, int incx);

    hipblasStatus_t hipblasDtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, int lda, *mut double x, int incx);

    hipblasStatus_t hipblasCtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, int lda, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasZtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, int lda, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasStrmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const float AP, int lda, *const *mut float x, int incx, int batchCount);

    hipblasStatus_t hipblasDtrmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const double AP, int lda, *const *mut double x, int incx, int batchCount);

    hipblasStatus_t hipblasCtrmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipComplex AP, int lda, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZtrmvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipDoubleComplex AP, int lda, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasStrmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, int lda, hipblasStride strideA, *mut float x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasDtrmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, int lda, hipblasStride strideA, *mut double x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCtrmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, int lda, hipblasStride strideA, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasZtrmvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasStrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, int lda, *mut float x, int incx);

    hipblasStatus_t hipblasDtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, int lda, *mut double x, int incx);

    hipblasStatus_t hipblasCtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, int lda, *mut hipComplex x, int incx);

    hipblasStatus_t hipblasZtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, int lda, *mut hipDoubleComplex x, int incx);

    hipblasStatus_t hipblasStrsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const float AP, int lda, *const *mut float x, int incx, int batchCount);

    hipblasStatus_t hipblasDtrsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const double AP, int lda, *const *mut double x, int incx, int batchCount);

    hipblasStatus_t hipblasCtrsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipComplex AP, int lda, *const *mut hipComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasZtrsvBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const *const hipDoubleComplex AP, int lda, *const *mut hipDoubleComplex x, int incx, int batchCount);

    hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const float AP, int lda, hipblasStride strideA, *mut float x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const double AP, int lda, hipblasStride strideA, *mut double x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasCtrsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipComplex AP, int lda, hipblasStride strideA, *mut hipComplex x, int incx, hipblasStride stridex, int batchCount);

    hipblasStatus_t hipblasZtrsvStridedBatched(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, *const hipDoubleComplex AP, int lda, hipblasStride strideA, *mut hipDoubleComplex x, int incx, hipblasStride stridex, int batchCount);

    // Level 3 BLAS

    // BLAS extensions

    // SOLVER API
    hipblasStatus_t hipblasSgetrf(hipblasHandle_t handle, int n, *mut float A, int lda, *mut int ipiv, *mut int info);

    hipblasStatus_t hipblasDgetrf(hipblasHandle_t handle, int n, *mut double A, int lda, *mut int ipiv, *mut int info);

    hipblasStatus_t hipblasCgetrf(hipblasHandle_t handle, int n, *mut hipComplex A, int lda, *mut int ipiv, *mut int info);

    hipblasStatus_t hipblasZgetrf(hipblasHandle_t handle, int n, *mut hipDoubleComplex A, int lda, *mut int ipiv, *mut int info);

    hipblasStatus_t hipblasSgetrfBatched(hipblasHandle_t handle, int n, *const *mut float A, int lda, *mut int ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle, int n, *const *mut double A, int lda, *mut int ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasCgetrfBatched(hipblasHandle_t handle, int n, *const *mut hipComplex A, int lda, *mut int ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasZgetrfBatched(hipblasHandle_t handle, int n, *const *mut hipDoubleComplex A, int lda, *mut int ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasSgetrfStridedBatched(hipblasHandle_t handle, int n, *mut float A, int lda, hipblasStride strideA, *mut int ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasDgetrfStridedBatched(hipblasHandle_t handle, int n, *mut double A, int lda, hipblasStride strideA, *mut int ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasCgetrfStridedBatched(hipblasHandle_t handle, int n, *mut hipComplex A, int lda, hipblasStride strideA, *mut int ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasZgetrfStridedBatched(hipblasHandle_t handle, int n, *mut hipDoubleComplex A, int lda, hipblasStride strideA, *mut int ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasSgetrs(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut float A, int lda, *const int ipiv, *mut float B, int ldb, *mut int info);

    hipblasStatus_t hipblasDgetrs(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut double A, int lda, *const int ipiv, *mut double B, int ldb, *mut int info);

    hipblasStatus_t hipblasCgetrs(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut hipComplex A, int lda, *const int ipiv, *mut hipComplex B, int ldb, *mut int info);

    hipblasStatus_t hipblasZgetrs(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut hipDoubleComplex A, int lda, *const int ipiv, *mut hipDoubleComplex B, int ldb, *mut int info);

    hipblasStatus_t hipblasSgetrsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *const *mut float A, int lda, *const int ipiv, *const *mut float B, int ldb, *mut int info, int batchCount);

    hipblasStatus_t hipblasDgetrsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *const *mut double A, int lda, *const int ipiv, *const *mut double B, int ldb, *mut int info, int batchCount);

    hipblasStatus_t hipblasCgetrsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *const *mut hipComplex A, int lda, *const int ipiv, *const *mut hipComplex B, int ldb, *mut int info, int batchCount);

    hipblasStatus_t hipblasZgetrsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *const *mut hipDoubleComplex A, int lda, *const int ipiv, *const *mut hipDoubleComplex B, int ldb, *mut int info, int batchCount);

    hipblasStatus_t hipblasSgetrsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut float A, int lda, hipblasStride strideA, *const int ipiv, hipblasStride strideP, *mut float B, int ldb, hipblasStride strideB, *mut int info, int batchCount);

    hipblasStatus_t hipblasDgetrsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut double A, int lda, hipblasStride strideA, *const int ipiv, hipblasStride strideP, *mut double B, int ldb, hipblasStride strideB, *mut int info, int batchCount);

    hipblasStatus_t hipblasCgetrsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut hipComplex A, int lda, hipblasStride strideA, *const int ipiv, hipblasStride strideP, *mut hipComplex B, int ldb, hipblasStride strideB, *mut int info, int batchCount);

    hipblasStatus_t hipblasZgetrsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int n, int nrhs, *mut hipDoubleComplex A, int lda, hipblasStride strideA, *const int ipiv, hipblasStride strideP, *mut hipDoubleComplex B, int ldb, hipblasStride strideB, *mut int info, int batchCount);

    hipblasStatus_t hipblasSgetriBatched(hipblasHandle_t handle, int n, *const *mut float A, int lda, *mut int ipiv, *const *mut float C, int ldc, *mut int info, int batchCount);

    hipblasStatus_t hipblasDgetriBatched(hipblasHandle_t handle, int n, *const *mut double A, int lda, *mut int ipiv, *const *mut double C, int ldc, *mut int info, int batchCount);

    hipblasStatus_t hipblasCgetriBatched(hipblasHandle_t handle, int n, *const *mut hipComplex A, int lda, *mut int ipiv, *const *mut hipComplex C, int ldc, *mut int info, int batchCount);

    hipblasStatus_t hipblasZgetriBatched(hipblasHandle_t handle, int n, *const *mut hipDoubleComplex A, int lda, *mut int ipiv, *const *mut hipDoubleComplex C, int ldc, *mut int info, int batchCount);

    hipblasStatus_t hipblasSgeqrf(hipblasHandle_t handle, int m, int n, *mut float A, int lda, *mut float ipiv, *mut int info);

    hipblasStatus_t hipblasDgeqrf(hipblasHandle_t handle, int m, int n, *mut double A, int lda, *mut double ipiv, *mut int info);

    hipblasStatus_t hipblasCgeqrf(hipblasHandle_t handle, int m, int n, *mut hipComplex A, int lda, *mut hipComplex ipiv, *mut int info);

    hipblasStatus_t hipblasZgeqrf(hipblasHandle_t handle, int m, int n, *mut hipDoubleComplex A, int lda, *mut hipDoubleComplex ipiv, *mut int info);

    hipblasStatus_t hipblasSgeqrfBatched(hipblasHandle_t handle, int m, int n, *const *mut float A, int lda, *const *mut float ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasDgeqrfBatched(hipblasHandle_t handle, int m, int n, *const *mut double A, int lda, *const *mut double ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasCgeqrfBatched(hipblasHandle_t handle, int m, int n, *const *mut hipComplex A, int lda, *const *mut hipComplex ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasZgeqrfBatched(hipblasHandle_t handle, int m, int n, *const *mut hipDoubleComplex A, int lda, *const *mut hipDoubleComplex ipiv, *mut int info, int batchCount);

    hipblasStatus_t hipblasSgeqrfStridedBatched(hipblasHandle_t handle, int m, int n, *mut float A, int lda, hipblasStride strideA, *mut float ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasDgeqrfStridedBatched(hipblasHandle_t handle, int m, int n, *mut double A, int lda, hipblasStride strideA, *mut double ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasCgeqrfStridedBatched(hipblasHandle_t handle, int m, int n, *mut hipComplex A, int lda, hipblasStride strideA, *mut hipComplex ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasZgeqrfStridedBatched(hipblasHandle_t handle, int m, int n, *mut hipDoubleComplex A, int lda, hipblasStride strideA, *mut hipDoubleComplex ipiv, hipblasStride strideP, *mut int info, int batchCount);

    hipblasStatus_t hipblasSgels(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut float A, int lda, *mut float B, int ldb, *mut int info, *mut int deviceInfo);

    hipblasStatus_t hipblasDgels(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut double A, int lda, *mut double B, int ldb, *mut int info, *mut int deviceInfo);

    hipblasStatus_t hipblasCgels(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut hipComplex A, int lda, *mut hipComplex B, int ldb, *mut int info, *mut int deviceInfo);

    hipblasStatus_t hipblasZgels(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut hipDoubleComplex A, int lda, *mut hipDoubleComplex B, int ldb, *mut int info, *mut int deviceInfo);

    hipblasStatus_t hipblasSgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *const *mut float A, int lda, *const *mut float B, int ldb, *mut int info, *mut int deviceInfo, int batchCount);

    hipblasStatus_t hipblasDgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *const *mut double A, int lda, *const *mut double B, int ldb, *mut int info, *mut int deviceInfo, int batchCount);

    hipblasStatus_t hipblasCgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *const *mut hipComplex A, int lda, *const *mut hipComplex B, int ldb, *mut int info, *mut int deviceInfo, int batchCount);

    hipblasStatus_t hipblasZgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *const *mut hipDoubleComplex A, int lda, *const *mut hipDoubleComplex B, int ldb, *mut int info, *mut int deviceInfo, int batchCount);

    hipblasStatus_t hipblasSgelsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut float A, int lda, hipblasStride strideA, *mut float B, int ldb, hipblasStride strideB, *mut int info, *mut int deviceInfo, int batchCount);

    hipblasStatus_t hipblasDgelsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut double A, int lda, hipblasStride strideA, *mut double B, int ldb, hipblasStride strideB, *mut int info, *mut int deviceInfo, int batchCount);

    hipblasStatus_t hipblasCgelsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut hipComplex A, int lda, hipblasStride strideA, *mut hipComplex B, int ldb, hipblasStride strideB, *mut int info, *mut int deviceInfo, int batchCount);

    hipblasStatus_t hipblasZgelsStridedBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int nrhs, *mut hipDoubleComplex A, int lda, hipblasStride strideA, *mut hipDoubleComplex B, int ldb, hipblasStride strideB, *mut int info, *mut int deviceInfo, int batchCount);
}
