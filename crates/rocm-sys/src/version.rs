#[cfg(not(feature = "dynamic-loading"))]
mod static_ver {
    macro_rules! select_ver {
        ($ver_flag:ident = $($ver_str:literal:$ver:literal),*$(,)?) => {{
            cfg_select! {
                $(
                    $ver_flag = $ver_str => $ver,
                )*
                _ => 0,
            }
        }};
    }

    pub const ROCM_MAJOR: u32 = select_ver!(rocm_major = "5":5, "6":6, "7":7);
    pub const ROCM_MINOR: u32 = select_ver!(rocm_minor = "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7);

    pub const ROCBLAS_MAJOR: u32 = select_ver!(rocblas_major = "3":3, "4":4, "5":5);
    pub const ROCBLAS_MINOR: u32 = select_ver!(
        rocblas_minor = "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15, "16":16, "17":17,
        "18":18, "19":19, "20":20, "21":21, "22":22, "23":23, "24":24, "25":25, "26":26, "27":27,
        "28":28, "29":29, "30":30, "31":31, "32":32, "33":33, "34":34, "35":35, "36":36, "37":37,
        "38":38, "39":39, "40":40, "41":41, "42":42, "43":43, "44":44, "45":45, "46":46, "47":47,
    );

    pub const HIPBLAS_MAJOR: u32 = select_ver!(hipblas_major = "1":1, "2":2, "3":3);
    pub const HIPBLAS_MINOR: u32 = select_ver!(
        hipblas_minor = "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15, "16":16, "17":17,
        "18":18, "19":19, "20":20, "21":21, "22":22, "23":23, "24":24, "25":25, "26":26, "27":27,
        "28":28, "29":29, "30":30, "31":31, "32":32, "33":33, "34":34, "35":35, "36":36, "37":37,
        "38":38, "39":39, "40":40, "41":41, "42":42, "43":43, "44":44, "45":45, "46":46, "47":47,
        "48":48, "49":49, "50":50, "51":51, "52":52, "53":53, "54":54,
    );

    pub const HIPBLASLT_MAJOR: u32 = select_ver!(hipblaslt_major = "1":1);
    pub const HIPBLASLT_MINOR: u32 = select_ver!(hipblaslt_minor = "1":1, "2":2, "3":3, "4":4);

    pub const ROCFFT_MAJOR: u32 = select_ver!(rocfft_major = "1":1);
    pub const ROCFFT_MINOR: u32 = select_ver!(
        rocfft_minor = "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15, "16":16, "17":17,
        "18":18, "19":19, "20":20, "21":21, "22":22, "23":23, "24":24, "25":25, "36":36, "27":27,
        "28":28, "29":29, "30":30, "31":31, "32":32, "33":33, "34":34, "35":35,
    );
}

#[cfg(not(feature = "dynamic-loading"))]
pub use static_ver::*;

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
