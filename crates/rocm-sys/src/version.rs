#[cfg(feature = "dynamic-loading")]
mod dyn_ver {
    #![allow(unused_imports)]

    use super::Ver2;
    use std::sync::LazyLock;

    #[cfg(feature = "hip")]
    static ROCM_VER: LazyLock<Ver2> = LazyLock::new(|| {
        use crate::hip::hipError_t;

        let mut version = -1;

        let status = unsafe { crate::hip::hipRuntimeGetVersion(&raw mut version) };

        if status != hipError_t::hipSuccess {
            return Ver2::default();
        }

        let major = version as u32 / 10_000_000;
        let minor = (version as u32 / 100_000) % 100;

        Ver2 { major, minor }
    });

    #[cfg(feature = "hip")]
    pub static ROCM_MAJOR: LazyLock<u32> = LazyLock::new(|| ROCM_VER.major);

    #[cfg(feature = "hip")]
    pub static ROCM_MINOR: LazyLock<u32> = LazyLock::new(|| ROCM_VER.minor);

    #[cfg(feature = "rocblas")]
    static ROCBLAS_VER: LazyLock<Ver2> = LazyLock::new(|| {
        use core::ffi::CStr;

        use crate::rocblas::rocblas_status;

        let mut size = 0;

        let status = unsafe { crate::rocblas::rocblas_get_version_string_size(&raw mut size) };

        if status != rocblas_status::rocblas_status_success {
            return Ver2::default();
        }

        let mut version = vec![0; size];

        let status =
            unsafe { crate::rocblas::rocblas_get_version_string(version.as_mut_ptr(), size) };

        if status != rocblas_status::rocblas_status_success {
            return Ver2::default();
        }

        let ver_cstr = unsafe { CStr::from_ptr(version.as_ptr()) };
        let Ok(version_string) = ver_cstr.to_str() else {
            return Ver2::default();
        };

        super::parse_version(version_string).unwrap_or(Ver2::default())
    });

    #[cfg(feature = "rocblas")]
    pub static ROCBLAS_MAJOR: LazyLock<u32> = LazyLock::new(|| ROCBLAS_VER.major);

    #[cfg(feature = "rocblas")]
    pub static ROCBLAS_MINOR: LazyLock<u32> = LazyLock::new(|| ROCBLAS_VER.minor);
}

#[allow(unused_imports)]
#[cfg(feature = "dynamic-loading")]
pub use dyn_ver::*;

#[derive(Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ver2<T = u32> {
    pub major: T,
    pub minor: T,
}

pub fn parse_version(s: &str) -> Option<Ver2> {
    let mut nums = s
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().ok());

    Some(Ver2 {
        major: nums.next()??,
        minor: nums.next()??,
    })
}
