use core::{
    fmt::{self, Display, Formatter},
    str::FromStr,
};

#[cfg(feature = "std")]
pub use super::dynamic::{ROCM_VERSION, find_rocm_version};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
}

impl FromStr for Version {
    type Err = VerParseErr;

    fn from_str(s: &str) -> Result<Version, Self::Err> {
        let mut nums = s
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>().ok());

        Ok(Version {
            major: nums.next().ok_or(VerParseErr)?.ok_or(VerParseErr)?,
            minor: nums.next().ok_or(VerParseErr)?.ok_or(VerParseErr)?,
        })
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

pub struct VerParseErr;

macro_rules! decl_known {
    ($($(#[$attr:meta])* $vis:vis const $name:ident = [$($major:literal:$minor:literal),*$(,)?];)+) => {
        $(
            $(#[$attr])*
            $vis const $name: &[&str] = &[$(concat!(stringify!($major), ".", stringify!($minor))),*];

            impl Version {
                $(#[$attr])*
                $vis const $name: &[Self] = &[$(Self { major: $major, minor: $minor }),*];
            }
        )+
    };
}

decl_known! {
    pub const KNOWN_ROCM_VERSIONS = [
        10:1, 10:0, 7:14, 7:2, 7:1, 7:0, 6:4, 6:3, 6:2, 6:1, 6:0, 5:7, 5:6,
        5:5, 5:4, 5:3, 5:2, 5:1, 5:0,
    ];

    pub const KNOWN_HIP_VERSIONS = [
        7:16, 7:15, 7:14, 7:13, 7:12, 7:11, 7:10, 7:2, 7:1, 7:9, 6:5, 6:4,
        6:3, 6:2, 6:1, 6:0, 5:7, 5:6, 5:5, 5:4, 5:3, 5:2, 5:1, 5:0, 4:5,
        4:4, 4:3,
    ];

    pub const KNOWN_ROCBLAS_VERSIONS = [
        7:16, 7:15, 7:14, 7:2, 5:2, 5:1, 5:0, 4:4, 4:3, 4:2, 4:1, 4:0, 3:1,
        2:47, 2:46, 2:45, 2:44, 2:43, 2:42, 2:41, 2:39, 2:38, 2:36, 2:32,
        2:30, 2:28, 2:26, 2:22, 2:24, 2:2, 2:1, 2:0, 14:3, 14:1, 14:0, 12:3,
        12:2, 0:12, 0:10, 0:4,
    ];

    pub const KNOWN_HIPBLAS_VERSIONS = [
        3:2, 3:1, 3:0, 2:4, 2:3, 2:2, 2:1, 2:0, 1:1, 1:0, 0:54, 0:53, 0:52,
        0:51, 0:50, 0:49, 0:48, 0:47, 0:46, 0:45, 0:44, 0:43, 0:42, 0:38, 0:36,
        0:34, 0:32, 0:30, 0:28, 12:2, 12:1, 12:0, 10:3, 10:1, 10:0, 0:10, 0:4,
    ];

    pub const KNOWN_HIPBLASLT_VERSIONS = [
        1:2, 1:1, 1:0, 0:12, 0:10, 0:8, 0:7, 0:6, 0:3, 0:2, 0:1,
    ];

    pub const KNOWN_ROCFFT_VERSIONS = [
        1:0, 0:9, 0:8, 0:7,
    ];
}
