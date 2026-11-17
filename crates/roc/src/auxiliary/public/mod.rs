pub trait MapToErr<E> {
    fn map_to_err(self) -> Result<(), E>;
}

pub trait RocHandle {
    type Error;
}

macro_rules! is_lib_present {
    ($lib_cfg:ident,$lib_lit:literal,$dyn_func:path,$new_func:ident) => {
        #[cfg(feature = $lib_lit)]
        pub fn $new_func() -> bool {
            cfg_select! {
                feature = "dynamic-loading" => $dyn_func(),
                _ => cfg!($lib_cfg),
            }
        }
    };
}

is_lib_present!(hip, "hip", rocm_sys::is_amdhip64_present, is_hip_present);
is_lib_present!(
    hiprtc,
    "hiprtc",
    rocm_sys::is_hiprtc_present,
    is_hiprtc_present
);
is_lib_present!(
    hipblas,
    "hipblas",
    rocm_sys::is_hipblas_present,
    is_hipblas_present
);
is_lib_present!(
    hipblaslt,
    "hipblaslt",
    rocm_sys::is_hipblaslt_present,
    is_hipblaslt_present
);
is_lib_present!(
    hipfft,
    "hipfft",
    rocm_sys::is_hipfft_present,
    is_hipfft_present
);
is_lib_present!(
    rocblas,
    "rocblas",
    rocm_sys::is_rocblas_present,
    is_rocblas_present
);
is_lib_present!(
    rocfft,
    "rocfft",
    rocm_sys::is_rocfft_present,
    is_rocfft_present
);
is_lib_present!(
    miopen,
    "miopen",
    rocm_sys::is_miopen_present,
    is_miopen_present
);
