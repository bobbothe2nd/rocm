use super::types::*;
use crate::{hip::*, internal::*};

// #[cfg(feature = "dynamic-loading")]
// use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

// TODO: support hipFFTW, callback APIs, and memory management APIs

link! {
    hipfft : rocm_ver;

    hipfftResult hipfftPlan1d(*mut hipfftHandle plan, int nx, hipfftType type_, int batch);

    hipfftResult hipfftPlan2d(*mut hipfftHandle plan, int nx, int ny, hipfftType type_);

    hipfftResult hipfftPlan3d(*mut hipfftHandle plan, int nx, int ny, int nz, hipfftType type_);

    hipfftResult hipfftCreate(*mut hipfftHandle plan);

    hipfftResult hipfftDestroy(hipfftHandle plan);

    hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate);

    hipfftResult hipfftSetWorkArea(hipfftHandle plan, *mut void workArea);

    hipfftResult hipfftXtSetWorkArea(hipfftHandle plan, *mut *mut void workArea);

    hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type_, int batch, *mut size_t workSize);

    hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type_, *mut size_t workSize);

    hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type_, *mut size_t workSize);

    hipfftResult hipfftMakePlanMany(hipfftHandle plan, int rank, *mut int n, *mut int inembed, int istride, int idist, *mut int onembed, int ostride, int odist, hipfftType type_, int batch, *mut size_t workSize);

    hipfftResult hipfftXtMakePlanMany(hipfftHandle plan, int rank, *mut longlong n, *mut longlong inembed, longlong istride, longlong idist, hipDataType inputType, *mut longlong onembed, longlong ostride, longlong odist, hipDataType outputType, longlong batch, *mut size_t workSize, hipDataType executionType);

    hipfftResult hipfftEstimate1d(int nx, hipfftType type_, int batch, *mut size_t workSize);

    hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type_, *mut size_t workSize);

    hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type_, *mut size_t workSize);

    hipfftResult hipfftEstimateMany(int rank, *mut int n, *mut int inembed, int istride, int idist, *mut int onembed, int ostride, int odist, hipfftType type_, int batch, *mut size_t workSize);

    hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type_, int batch, *mut size_t workSize);

    hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type_, *mut size_t workSize);

    hipfftResult hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type_, *mut size_t workSize);

    hipfftResult hipfftGetSizeMany(hipfftHandle plan, int rank, *mut int n, *mut int inembed, int istride, int idist, *mut int onembed, int ostride, int odist, hipfftType type_, int batch, *mut size_t workSize);

    hipfftResult hipfftXtGetSizeMany(hipfftHandle plan, int rank, *mut longlong n, *mut longlong inembed, longlong istride, longlong idist, hipDataType inputType, *mut longlong onembed, longlong ostride, longlong odist, hipDataType outputType, longlong batch, *mut size_t workSize, hipDataType executionType);

    hipfftResult hipfftGetSize(hipfftHandle plan, *mut size_t workSize);

    hipfftResult hipfftExecC2C(hipfftHandle plan, *mut hipfftComplex idata, *mut hipfftComplex odata, int direction);

    hipfftResult hipfftExecR2C(hipfftHandle plan, *mut hipfftReal idata, *mut hipfftComplex odata);

    hipfftResult hipfftExecC2R(hipfftHandle plan, *mut hipfftComplex idata, *mut hipfftReal odata);

    hipfftResult hipfftExecZ2Z(hipfftHandle plan, *mut hipfftDoubleComplex idata, *mut hipfftDoubleComplex odata, int direction);

    hipfftResult hipfftExecD2Z(hipfftHandle plan, *mut hipfftDoubleReal idata, *mut hipfftDoubleComplex odata);

    hipfftResult hipfftExecZ2D(hipfftHandle plan, *mut hipfftDoubleComplex idata, *mut hipfftDoubleReal odata);

    hipfftResult hipfftXtExec(hipfftHandle plan, *mut void input, *mut void output, int direction);

    hipfftResult hipfftXtSetGPUs(hipfftHandle plan, int count, *mut int gpus);
}
