use alloc::sync::Arc;
use rocm_sys::rocblas::{
    rocblas_create_handle, rocblas_get_pointer_mode, rocblas_handle, rocblas_pointer_mode, rocblas_set_pointer_mode, rocblas_set_stream, rocblas_status,
};

use crate::{aux::RocHandle, hip::stream::HipStream};

macro_rules! try_err {
    ($func:expr$(, $sucess:expr)?$(,)?) => {
        match $func {
            ::rocm_sys::rocblas::rocblas_status::rocblas_status_success => { $($sucess)? }
            err => return Err(<$crate::rocblas::RocblasError as ::core::convert::From::<::rocm_sys::rocblas::rocblas_status>>::from(err)),
        }
    };
}

mod ops;

#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug, Clone)]
pub struct RocblasHandle {
    pub(crate) raw: rocblas_handle,
    pub(crate) stream: Arc<HipStream>,
}

unsafe impl Send for RocblasHandle {}
unsafe impl Sync for RocblasHandle {}

impl RocblasHandle {
    #[inline]
    pub fn new(stream: &Arc<HipStream>) -> Result<Self, RocblasError> {
        let raw: Result<*mut rocm_sys::rocblas::_rocblas_handle, RocblasError> =
            unsafe { wrap_sys_res!(|handle| rocblas_create_handle(handle.as_mut_ptr())) };
        let raw = raw?;
        unsafe { try_err!(rocblas_set_stream(raw, stream.raw)) };
        Ok(Self {
            raw,
            stream: stream.clone(),
        })
    }

    #[inline(always)]
    pub fn as_raw(&self) -> rocblas_handle {
        self.raw
    }

    #[inline(always)]
    pub fn set_stream(&mut self, stream: Arc<HipStream>) -> Result<(), RocblasError> {
        self.stream = stream;
        unsafe { try_err!(rocblas_set_stream(self.raw, self.stream.raw), Ok(())) }
    }

    #[inline(always)]
    pub fn get_stream(&self) -> &HipStream {
        &self.stream
    }

    #[inline(always)]
    pub fn set_pointer_mode(&self, ptr_mode: RocblasPtrMode) -> Result<(), RocblasError> {
        unsafe { try_err!(rocblas_set_pointer_mode(self.raw, ptr_mode.as_raw()), Ok(())) }
    }

    #[inline(always)]
    pub fn get_pointer_mode(&self) -> Result<RocblasPtrMode, RocblasError> {
        unsafe { wrap_sys_res!(|ptr_mode: RocblasPtrMode| rocblas_get_pointer_mode(self.raw, ptr_mode.as_mut_ptr().cast())) }
    }
}

impl RocHandle for RocblasHandle {
    type Error = RocblasError;
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocblasPtrMode {
    /// Scalar values affected by this variable are located on the host.
    Host = 0,
    /// Scalar values affected by this variable are located on the device.
    Dev = 1,
}

impl RocblasPtrMode {
    pub const fn as_raw(self) -> rocblas_pointer_mode {
        match self {
            Self::Host => rocblas_pointer_mode::rocblas_pointer_mode_host,
            Self::Dev => rocblas_pointer_mode::rocblas_pointer_mode_device,
        }
    }
}

#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
/// rocBLAS status codes definition.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum RocblasError {
    /// Handle is uninitialized, invalid, or null.
    InvalidHandle = 1,
    /// Function is not implemented.
    NotImplemented = 2,
    /// Invalid pointer argument.
    InvalidPointer = 3,
    /// Invalid size argument.
    InvalidSize = 4,
    /// Failed internal memory allocation, copy, or dealloc.
    MemoryError = 5,
    /// Other internal library failure.
    InternalError = 6,
    /// Performance degraded due to low device memory.
    PerfDegraded = 7,
    /// Unmatched start/stop size query.
    SizeQueryMismatch = 8,
    /// Queried device memory size increased.
    SizeIncreased = 9,
    /// Queried device memory size unchanged.
    SizeUnchanged = 10,
    /// Passed argument not valid.
    InvalidValue = 11,
    /// Nothing preventing function from proceeding.
    Continue = 12,
    CheckNumericsFail = 13,
    ExcludedFromBuild = 14,
    ArchMismatch = 15,
}

err_ty!(type RocblasError = rocblas_status is !rocblas_status_success from rocm_sys::rocblas::rocblas_status_to_string;);

cvt_hip_err!(
    RocblasError {
        InvalidHandle => [
            InvalidHandle,
            InvalidContext,
            ContextAlreadyCurrent,
            ContextAlreadyInUse,
            ContextIsDestroyed,
            InvalidGraphicsContext,
            IllegalState,
        ],
        NotImplemented => [
            NotSupported,
            InvalidDeviceFunction,
            CooperativeLaunchTooLarge,
        ],
        InvalidPointer => [
            InvalidDevicePointer,
            IllegalAddress,
            HostMemoryNotRegistered,
        ],
        InvalidSize => [
            InvalidPitchValue,
            InvalidChannelDescriptor,
            InvalidConfiguration,
            InvalidTexture,
        ],
        MemoryError => [
            OutOfMemory,
            RuntimeMemory,
            LaunchOutOfResources,
            HostMemoryAlreadyRegistered,
        ],
        InternalError => [
            RuntimeOther,
            Unknown,
            OperatingSystem,
            SharedObjectInitFailed,
            SharedObjectSymbolNotFound,
            PriorLaunchFailure,
            LaunchFailure,
            ECCNotCorrectable,
            GraphExecUpdateFailure,
        ],
        PerfDegraded => [
            SetOnActiveProcess,
        ],
        SizeQueryMismatch => [
            MissingConfiguration,
        ],
        SizeUnchanged => [
            NotReady,
        ],
        InvalidValue => [
            InvalidValue,
            InvalidSymbol,
            InvalidMemcpyDirection,
            InvalidSource,
            InvalidImage,
            InvalidKernelFile,
        ],
        Continue => [
            Tbd,
        ],
        CheckNumericsFail => [
            Assert,
        ],
        ExcludedFromBuild => [
            NoBinaryForGpu,
        ],
        ArchMismatch => [
            InsufficientDriver,
            NoDevice,
            InvalidDevice,
        ],
    }
    reverse {
        InvalidHandle => InvalidHandle,
        NotImplemented => NotSupported,
        InvalidPointer => InvalidDevicePointer,
        InvalidSize => InvalidConfiguration,
        MemoryError => OutOfMemory,
        InternalError => RuntimeOther,
        PerfDegraded => SetOnActiveProcess,
        SizeQueryMismatch => MissingConfiguration,
        SizeIncreased => RuntimeMemory,
        SizeUnchanged => NotReady,
        InvalidValue => InvalidValue,
        Continue => Tbd,
        CheckNumericsFail => Assert,
        ExcludedFromBuild => NoBinaryForGpu,
        ArchMismatch => InsufficientDriver,
    }
    fallback InternalError
);
