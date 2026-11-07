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
