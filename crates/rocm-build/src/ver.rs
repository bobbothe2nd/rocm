use std::{
    fs, path::{Path, PathBuf},
};

pub const KNOWN_ROCM_VERSIONS: &[&str] = &[
    "10.1", "10.0", "7.15", "7.14", "7.13", "7.12", "7.11", "7.10", "7.2", "7.1", "7.9", "6.5",
    "6.4", "6.3", "6.2", "6.1", "6.0", "5.7", "5.6", "5.5", "5.4", "5.3", "5.2", "5.1", "5.0",
    "4.5", "4.4", "4.3",
];

pub const KNOWN_ROCBLAS_VERSIONS: &[&str] = &[
    "7.2", "5.2", "5.1", "5.0", "4.4", "4.3", "4.2", "4.1", "4.0", "3.1", "2.47", "2.46", "2.45",
    "2.44", "2.43", "2.42", "2.41", "2.39", "2.38", "2.36", "2.32", "2.32", "2.30", "2.28", "2.26",
    "2.22", "2.24", "2.2", "2.1", "2.0", "14.3", "14.1", "14.0", "12.3", "12.2", "0.12", "0.10",
    "0.4",
];

pub const KNOWN_HIPBLAS_VERSIONS: &[&str] = &[
    "3.2", "3.1", "3.0", "2.4", "2.3", "2.2", "2.1", "2.0", "1.1", "1.0", "0.54", "0.53", "0.52",
    "0.51", "0.50", "0.49", "0.48", "0.47", "0.46", "0.45", "0.44", "0.43", "0.42", "0.38", "0.36",
    "0.34", "0.32", "0.30", "0.28", "12.2", "12.1", "12.0", "10.3", "10.1", "10.0", "0.10", "0.4",
];

pub const KNOWN_HIPBLASLT_VERSIONS: &[&str] = &[
    "1.2", "1.1", "1.0", "0.12", "0.10", "0.8", "0.7", "0.6", "0.3", "0.2", "0.1",
];

pub const KNOWN_ROCFFT_VERSIONS: &[&str] = &["1.0", "0.9", "0.8", "0.7"];

#[derive(Debug, Clone, Copy)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
}

pub fn cfg_version(cfg: &str, version: Version, known: &[&str]) {
    let formatted = format!("{}.{}", version.major, version.minor);
    let index = known
        .iter()
        .position(|x| *x == formatted.as_str())
        .unwrap_or(known.len() - 1);

    for version in &known[index..] {
        println!("cargo:rustc-cfg={cfg}=\"{}\"", version);
    }
}

pub fn find_rocm_version(roots: &[PathBuf]) -> Option<Version> {
    for root in roots {
        let path = root.join(".info/version");

        let Ok(version) = fs::read_to_string(path) else {
            continue;
        };

        if let Some(version) = parse_version(version.trim()) {
            return Some(version);
        }
    }

    None
}

pub fn parse_version(s: &str) -> Option<Version> {
    let mut nums = s
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().ok());

    Some(Version {
        major: nums.next()??,
        minor: nums.next()??,
    })
}

pub fn hip_runtime_version(include: &Path) -> Option<Version> {
    let header = include.join("hip_version.h");

    println!("cargo:rerun-if-changed={}", header.display());

    let contents = fs::read_to_string(&header).ok()?;

    let major = parse_define(&contents, "HIP_VERSION_MAJOR")?;
    let minor = parse_define(&contents, "HIP_VERSION_MINOR")?;

    Some(Version { major, minor })
}

pub fn parse_define(contents: &str, name: &str) -> Option<u32> {
    contents
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();

            if parts.next()? == "#define" && parts.next()? == name {
                parts.next()?.parse().ok()
            } else {
                None
            }
        })
}
