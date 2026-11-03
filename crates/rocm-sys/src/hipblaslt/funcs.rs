use super::types::*;
use crate::{internal::*, hip::*};

// #[cfg(feature = "dynamic-loading")]
// use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

link! {
    hipblaslt : [7, 8, 9, 10] : rocm_ver;

    hipblasStatus_t hipblasLtCreate(*mut hipblasLtHandle_t handle);

    hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle);

    hipblasStatus_t hipblasLtMatrixLayoutCreate(*mut hipblasLtMatrixLayout_t matLayout, hipDataType type_, uint64_t rows, uint64_t cols, int64_t ld);
}
