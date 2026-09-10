use libloading::Library;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const KNOWN_ROCM_VERSIONS: &[&str] = &[
    "10.0", "7.15", "7.14", "7.13", "7.12", "7.11", "7.10", "7.2", "7.1", "7.9", "6.5", "6.4",
    "6.3", "6.2", "6.1", "6.0", "5.7", "5.6", "5.5", "5.4", "5.3", "5.2", "5.1", "5.0", "4.5",
    "4.4", "4.3", "4.1", "4.0", "3.10", "3.9", "3.8", "3.7", "3.5", "3.1", "3.0", "2.10", "2.8",
    "2.7", "1.0",
];

const KNOWN_ROCBLAS_VERSIONS: &[&str] = &[
    "7.2", "5.2", "5.1", "5.0", "4.4", "4.3", "4.2", "4.1", "4.0", "3.1", "2.47", "2.46", "2.45",
    "2.44", "2.43", "2.42", "2.41", "2.39", "2.38", "2.36", "2.32", "2.32", "2.30", "2.28", "2.26",
    "2.22", "2.24", "2.2", "2.1", "2.0", "14.3", "14.1", "14.0", "12.3", "12.2", "0.12", "0.10",
    "0.4",
];

const KNOWN_HIPBLAS_VERSONS: &[&str] = &[
    "3.2", "3.1", "3.0", "2.4", "2.3", "2.2", "2.1", "2.0", "1.1", "1.0", "0.54", "0.53", "0.52",
    "0.51", "0.50", "0.49", "0.48", "0.47", "0.46", "0.45", "0.44", "0.43", "0.42", "0.38", "0.36",
    "0.34", "0.32", "0.30", "0.28", "12.2", "12.1", "12.0", "10.3", "10.1", "10.0", "0.10", "0.4",
];

const KNOWN_HIPBLASLT_VERSIONS: &[&str] = &[
    "1.2", "1.1", "1.0", "0.12", "0.10", "0.8", "0.7", "0.6", "0.3", "0.2", "0.1",
];

const KNOWN_ROCFFT_VERSIONS: &[&str] = &["1.0", "0.9", "0.8", "0.7"];

fn main() {
    check_cfg();

    let roots = rocm_roots();
    let mut found_rocm_ver = false;

    if let Some(version) = find_rocm_version(&roots) {
        found_rocm_ver = true;
        println!(
            "cargo:rustc-cfg=rocm_ver=\"{}.{}\"",
            version.major, version.minor
        );
    }

    if let Some(component) = find_component(
        &roots,
        &["hip_runtime_api.h", "hip_runtime.h"],
        &["amdhip64"],
    ) {
        if !found_rocm_ver {
            let version = unsafe {
                hip_runtime_version(&component.library_dir)
                    .expect("failed to determine HIP runtime version")
            };

            println!(
                "cargo:rustc-cfg=rocm_ver=\"{}.{}\"",
                version.major, version.minor
            );
        }

        link_component(&component);
        println!("cargo:rustc-cfg=hip");
    }

    if let Some(component) = find_component(&roots, &["rocblas.h"], &["rocblas"]) {
        let version = unsafe {
            rocblas_version(&component.library_dir)
                .expect("failed to determine HIP runtime version")
        };

        println!(
            "cargo:rustc-cfg=rocblas_ver=\"{}.{}\"",
            version.major, version.minor
        );

        link_component(&component);
        println!("cargo:rustc-cfg=rocblas");
    }

    if let Some(component) = find_component(&roots, &["hiprtc.h"], &["hiprtc"]) {
        link_component(&component);
        println!("cargo:rustc-cfg=hiprtc");
    }

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
}

#[derive(Debug, Clone, Copy)]
struct Version {
    major: u32,
    minor: u32,
}

fn find_rocm_version(roots: &[PathBuf]) -> Option<Version> {
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

unsafe fn hip_runtime_version(library: &Path) -> Option<Version> {
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

unsafe fn rocblas_version(library: &Path) -> Option<Version> {
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

fn parse_version(s: &str) -> Option<Version> {
    let mut nums = s
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().ok());

    Some(Version {
        major: nums.next()??,
        minor: nums.next()??,
    })
}

struct Component {
    include_dir: PathBuf,
    library_dir: PathBuf,
    library: String,
}

fn rocm_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    for variable in ["ROCM_PATH", "ROCM_HOME"] {
        if let Some(path) = env::var_os(variable) {
            roots.push(PathBuf::from(path));
        }
    }

    roots.extend([
        PathBuf::from("/opt/rocm"),
        PathBuf::from("/usr/local/rocm"),
        PathBuf::from("/usr"),
    ]);

    roots
}

fn find_component(roots: &[PathBuf], headers: &[&str], libraries: &[&str]) -> Option<Component> {
    for root in roots {
        let Some(header) = find_file(root, headers) else {
            continue;
        };

        let Some(library) = find_library(root, libraries) else {
            continue;
        };

        let include_dir = header.parent()?.to_path_buf();
        let library_dir = library.parent()?.to_path_buf();

        return Some(Component {
            include_dir,
            library_dir,
            library: libraries[0].to_string(),
        });
    }

    None
}

fn find_file(root: &Path, names: &[&str]) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).ok()?;

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                stack.push(path);
                continue;
            }

            if let Some(name) = path.file_name().and_then(|x| x.to_str())
                && names.contains(&name)
            {
                return Some(path);
            }
        }
    }

    None
}

fn find_library(root: &Path, names: &[&str]) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).ok()?;

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                stack.push(path);
                continue;
            }

            let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
                continue;
            };

            for library in names {
                if name == format!("lib{library}.so")
                    || name.starts_with(&format!("lib{library}.so."))
                {
                    return Some(path);
                }
            }
        }
    }

    None
}

fn link_component(component: &Component) {
    println!(
        "cargo:rustc-link-search=native={}",
        component.library_dir.display()
    );

    println!("cargo:rustc-link-lib=dylib={}", component.library);

    println!("cargo:include={}", component.include_dir.display());
}

fn check_cfg() {
    println!("cargo:rustc-check-cfg=cfg(hip)");
    println!("cargo:rustc-check-cfg=cfg(hiprtc)");
    println!("cargo:rustc-check-cfg=cfg(hipblas)");
    println!("cargo:rustc-check-cfg=cfg(hipblaslt)");
    println!("cargo:rustc-check-cfg=cfg(rocblas)");
    println!("cargo:rustc-check-cfg=cfg(rocfft)");

    let rocm_ver = KNOWN_ROCM_VERSIONS
        .iter()
        .map(|ver| format!("\"{ver}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocm_ver, values({rocm_ver}))");

    let rocblas_ver = KNOWN_ROCBLAS_VERSIONS
        .iter()
        .map(|ver| format!("\"{ver}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocblas_ver, values({rocblas_ver}))");

    let rocfft_ver = KNOWN_ROCFFT_VERSIONS
        .iter()
        .map(|ver| format!("\"{ver}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocfft_ver, values({rocfft_ver}))");

    let hipblas_ver = KNOWN_HIPBLAS_VERSONS
        .iter()
        .map(|ver| format!("\"{ver}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(hipblas_ver, values({hipblas_ver}))");

    let hipblaslt_ver = KNOWN_HIPBLASLT_VERSIONS
        .iter()
        .map(|ver| format!("\"{ver}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(hipblaslt_ver, values({hipblaslt_ver}))");
}
