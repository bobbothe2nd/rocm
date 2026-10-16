#[cfg(feature = "dynamic-loading")]
mod dyn_ver {
    #[allow(unused_imports)]
    use std::sync::LazyLock;

    #[cfg(feature = "hip")]
    static ROCM_VER: LazyLock<(u32, u32)> = LazyLock::new(|| {
        use crate::hip::hipError_t;

        let mut version = -1;

        let status = unsafe {
            crate::hip::hipRuntimeGetVersion(&raw mut version)
        };

        if status != hipError_t::hipSuccess {
            return (0, 0);
        }

        let major = version / 10_000_000;
        let minor = (version / 100_000) % 100;

        (major as u32, minor as u32)
    });

    #[cfg(feature = "hip")]
    pub static ROCM_MAJOR: LazyLock<u32> = LazyLock::new(|| ROCM_VER.0);

    #[cfg(feature = "hip")]
    pub static ROCM_MINOR: LazyLock<u32> = LazyLock::new(|| ROCM_VER.1);

    #[cfg(feature = "rocblas")]
    static ROCBLAS_VER: LazyLock<(u32, u32)> = LazyLock::new(|| {
        use core::ffi::CStr;

        use crate::rocblas::rocblas_status;

        let mut size = 0;

        let status = unsafe {
            crate::rocblas::rocblas_get_version_string_size(&raw mut size)
        };

        if status != rocblas_status::rocblas_status_success {
            return (0, 0);
        }

        let mut version = vec![0; size];

        let status = unsafe {
            crate::rocblas::rocblas_get_version_string(version.as_mut_ptr(), size)
        };

        if status != rocblas_status::rocblas_status_success {
            return (0, 0);
        }

        let ver_cstr = unsafe { CStr::from_ptr(version.as_ptr()) };
        let Ok(version_string) = ver_cstr.to_str() else {
            return (0, 0);
        };

        parse_version(version_string).unwrap_or((0, 0))
    });

    #[cfg(feature = "rocblas")]
    pub static ROCBLAS_MAJOR: LazyLock<u32> = LazyLock::new(|| ROCBLAS_VER.0);

    #[cfg(feature = "rocblas")]
    pub static ROCBLAS_MINOR: LazyLock<u32> = LazyLock::new(|| ROCBLAS_VER.1);

    #[allow(dead_code)]
    fn parse_version(s: &str) -> Option<(u32, u32)> {
        let mut nums = s
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>().ok());

        Some((nums.next()??, nums.next()??))
    }
}

#[allow(unused_imports)]
#[cfg(feature = "dynamic-loading")]
pub use dyn_ver::*;
