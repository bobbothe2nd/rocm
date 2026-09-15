use libloading::Library;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const KNOWN_ROCM_VERSIONS: &[&str] = &[
    "10.0", "7.15", "7.14", "7.13", "7.12", "7.11", "7.10", "7.2", "7.1", "7.9", "6.5", "6.4",
    "6.3", "6.2", "6.1", "6.0", "5.7", "5.6", "5.5", "5.4", "5.3", "5.2", "5.1", "5.0", "4.5",
    "4.4", "4.3",
];

pub const KNOWN_ROCBLAS_VERSIONS: &[&str] = &[
    "7.2", "5.2", "5.1", "5.0", "4.4", "4.3", "4.2", "4.1", "4.0", "3.1", "2.47", "2.46", "2.45",
    "2.44", "2.43", "2.42", "2.41", "2.39", "2.38", "2.36", "2.32", "2.32", "2.30", "2.28", "2.26",
    "2.22", "2.24", "2.2", "2.1", "2.0", "14.3", "14.1", "14.0", "12.3", "12.2", "0.12", "0.10",
    "0.4",
];

pub const KNOWN_HIPBLAS_VERSONS: &[&str] = &[
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

    for version in &known[..index] {
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

/// Queries the HIP runtime API version.
///
/// # Safety
///
/// The library must be a valid `amdhip64` path.
pub unsafe fn hip_runtime_version(library: &Path) -> Option<Version> {
    type HipRuntimeGetVersion = unsafe extern "C" fn(*mut i32) -> i32;

    let lib = unsafe { Library::new(library).ok()? };

    let get_version = unsafe {
        lib.get::<HipRuntimeGetVersion>(b"hipRuntimeGetVersion\0")
            .ok()?
    };

    let mut version = 0i32;

    let status = unsafe { get_version(&mut version) };

    if status != 0 {
        return None;
    }

    let version = version as u32;

    Some(Version {
        major: version / 10_000_000,
        minor: (version / 100_000) % 100,
    })
}

/// Queries the rocBLAS API version.
///
/// # Safety
///
/// The library must be a valid rocBLAS path.
pub unsafe fn rocblas_version(library: &Path) -> Option<Version> {
    let lib = unsafe { Library::new(library).ok()? };

    type GetVersionString = unsafe extern "C" fn(*mut std::ffi::c_char, usize) -> i32;

    let get_version = unsafe {
        lib.get::<GetVersionString>(b"rocblas_get_version_string\0")
            .ok()?
    };

    let mut buffer = vec![0i8; 64];

    let status = unsafe { get_version(buffer.as_mut_ptr(), buffer.len()) };

    if status != 0 {
        return None;
    }

    let cstr = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) };
    let string = cstr.to_str().ok()?;

    parse_version(string)
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
