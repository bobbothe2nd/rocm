#[cfg(feature = "dynamic-loading")]
mod dyn_ver {
    use std::sync::LazyLock;

    static ROCM_VER: LazyLock<(u32, u32)> = LazyLock::new(|| {
        let mut version = -1;

        unsafe {
            crate::hip::hipRuntimeGetVersion(&raw mut version);
        }

        let major = version / 10_000_000;
        let minor = (version / 100_000) % 100;

        (major as u32, minor as u32)
    });

    pub static ROCM_MAJOR: LazyLock<u32> = LazyLock::new(|| ROCM_VER.0);
    pub static ROCM_MINOR: LazyLock<u32> = LazyLock::new(|| ROCM_VER.1);
}

#[cfg(feature = "dynamic-loading")]
pub use dyn_ver::*;
