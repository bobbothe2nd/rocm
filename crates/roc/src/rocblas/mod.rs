use core::{error::Error, fmt};

use alloc::sync::Arc;
use rocm_sys::rocblas::{
    rocblas_create_handle, rocblas_handle, rocblas_pointer_mode, rocblas_set_pointer_mode,
    rocblas_set_stream, rocblas_status,
};

use crate::hip::stream::HipStream;

macro_rules! try_err {
    ($func:expr$(, $sucess:expr)?$(,)?) => {
        match $func {
            ::rocm_sys::rocblas::rocblas_status::rocblas_status_success => { $($sucess)? }
            err => return Err(<$crate::rocblas::RocblasError as ::core::convert::From::<::rocm_sys::rocblas::rocblas_status>>::from(err)),
        }
    };
}

#[derive(Debug, Clone)]
pub struct RocBlas {
    pub(crate) raw: rocblas_handle,
    pub(crate) stream: Arc<HipStream>,
}

unsafe impl Send for RocBlas {}
unsafe impl Sync for RocBlas {}

impl RocBlas {
    pub fn new(stream: Arc<HipStream>) -> Result<Self, RocblasError> {
        let raw: Result<*mut rocm_sys::rocblas::_rocblas_handle, RocblasError> =
            unsafe { wrap_sys_res!(|handle| rocblas_create_handle(handle.as_mut_ptr())) };
        let raw = raw?;
        unsafe { try_err!(rocblas_set_stream(raw, stream.raw)) };
        Ok(Self { raw, stream })
    }

    pub fn as_raw(&self) -> rocblas_handle {
        self.raw
    }

    pub fn set_stream(&mut self, stream: Arc<HipStream>) -> Result<(), RocblasError> {
        self.stream = stream;
        unsafe { try_err!(rocblas_set_stream(self.raw, self.stream.raw), Ok(())) }
    }

    pub fn set_pointer_mode(&self, pointer_mode: rocblas_pointer_mode) -> Result<(), RocblasError> {
        unsafe { try_err!(rocblas_set_pointer_mode(self.raw, pointer_mode), Ok(())) }
    }
}

#[repr(i32)]
// rocBLAS status codes definition.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum RocblasError {
    // Handle is uninitialized, invalid, or null.
    RocblasStatusInvalidHandle = 1,
    // Function is not implemented.
    RocblasStatusNotImplemented = 2,
    // Invalid pointer argument.
    RocblasStatusInvalidPointer = 3,
    // Invalid size argument.
    RocblasStatusInvalidSize = 4,
    // Failed internal memory allocation, copy, or dealloc.
    RocblasStatusMemoryError = 5,
    // Other internal library failure.
    RocblasStatusInternalError = 6,
    // Performance degraded due to low device memory.
    RocblasStatusPerfDegraded = 7,
    // Unmatched start/stop size query.
    RocblasStatusSizeQueryMismatch = 8,
    // Queried device memory size increased.
    RocblasStatusSizeIncreased = 9,
    // Queried device memory size unchanged.
    RocblasStatusSizeUnchanged = 10,
    // Passed argument not valid.
    RocblasStatusInvalidValue = 11,
    // Nothing preventing function from proceeding.
    RocblasStatusContinue = 12,
    RocblasStatusCheckNumericsFail = 13,
    RocblasStatusExcludedFromBuild = 14,
    RocblasStatusArchMismatch = 15,
}

impl fmt::Display for RocblasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cstr_ptr = unsafe { rocm_sys::rocblas::rocblas_status_to_string((*self).into()) };
        let cstr = unsafe { core::ffi::CStr::from_ptr(cstr_ptr) };
        let formatted = cstr.to_str().map_err(|_| fmt::Error)?;
        f.write_str(formatted)
    }
}

impl Error for RocblasError {}

impl From<rocblas_status> for RocblasError {
    fn from(value: rocblas_status) -> Self {
        match value {
            rocblas_status::rocblas_status_success => {
                panic!("converting rocblas_status_success to RocblasError")
            }
            err => unsafe { core::mem::transmute::<rocblas_status, Self>(err) },
        }
    }
}

impl From<RocblasError> for rocblas_status {
    fn from(value: RocblasError) -> Self {
        unsafe { core::mem::transmute::<RocblasError, Self>(value) }
    }
}
