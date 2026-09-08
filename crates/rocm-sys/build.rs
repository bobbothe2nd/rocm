use libloading::Library;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const KNOWN_ROCM_MAJOR: &[u32] = &[5, 6, 7, 10];
const KNOWN_ROCM_MINOR: &[u32] = &[0, 1, 2, 3, 4, 5, 6, 7, 14, 15];

const KNOWN_ROCBLAS_MAJOR: &[u32] = &[3, 4, 5];
const KNOWN_ROCBLAS_MINOR: &[u32] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
];

const KNOWN_HIPBLAS_MAJOR: &[u32] = &[0, 1, 2, 3];
const KNOWN_HIPBLAS_MINOR: &[u32] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54,
];

const KNOWN_HIPBLASLT_MAJOR: &[u32] = &[0, 1];
const KNOWN_HIPBLASLT_MINOR: &[u32] = &[0, 1, 2, 3, 4];

const KNOWN_ROCFFT_MAJOR: &[u32] = &[1];
const KNOWN_ROCFFT_MINOR: &[u32] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
];

fn main() {
    check_cfg();

    let roots = rocm_roots();
    let mut found_rocm_ver = false;

    if let Some(version) = find_rocm_version(&roots) {
        found_rocm_ver = true;
        println!("cargo:rustc-cfg=rocm_major=\"{}\"", version.major);
        println!("cargo:rustc-cfg=rocm_major=\"{}\"", version.minor);
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

            println!("cargo:rustc-cfg=rocm_major=\"{}\"", version.major);
            println!("cargo:rustc-cfg=rocm_major=\"{}\"", version.minor);
        }

        link_component(&component);
        println!("cargo:rustc-cfg=hip");
    }

    if let Some(component) = find_component(&roots, &["rocblas.h"], &["rocblas"]) {
        let version = unsafe {
            rocblas_version(&component.library_dir)
                .expect("failed to determine HIP runtime version")
        };

        println!("cargo:rustc-cfg=rocblas_major=\"{}\"", version.major);
        println!("cargo:rustc-cfg=rocblas_major=\"{}\"", version.minor);

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
                && names.contains(&name) {
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
    println!("cargo:rustc-check-cfg=cfg(rocblas)");
    println!("cargo:rustc-check-cfg=cfg(hiprtc)");

    let rocm_major = KNOWN_ROCM_MAJOR
        .iter()
        .map(|major| format!("\"{major}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocm_major, values({rocm_major}))");

    let rocm_minor = KNOWN_ROCM_MINOR
        .iter()
        .map(|major| format!("\"{major}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocm_minor, values({rocm_minor}))");

    let rocblas_major = KNOWN_ROCBLAS_MAJOR
        .iter()
        .map(|major| format!("\"{major}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocblas_major, values({rocblas_major}))");

    let rocblas_minor = KNOWN_ROCBLAS_MINOR
        .iter()
        .map(|minor| format!("\"{minor}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocblas_minor, values({rocblas_minor}))");

    let rocfft_major = KNOWN_ROCFFT_MAJOR
        .iter()
        .map(|major| format!("\"{major}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocfft_major, values({rocfft_major}))");

    let rocfft_minor = KNOWN_ROCFFT_MINOR
        .iter()
        .map(|minor| format!("\"{minor}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(rocfft_minor, values({rocfft_minor}))");

    let hipblas_major = KNOWN_HIPBLAS_MAJOR
        .iter()
        .map(|major| format!("\"{major}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(hipblas_major, values({hipblas_major}))");

    let hipblas_minor = KNOWN_HIPBLAS_MINOR
        .iter()
        .map(|minor| format!("\"{minor}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(hipblas_minor, values({hipblas_minor}))");

    let hipblaslt_major = KNOWN_HIPBLASLT_MAJOR
        .iter()
        .map(|major| format!("\"{major}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(hipblaslt_major, values({hipblaslt_major}))");

    let hipblaslt_minor = KNOWN_HIPBLASLT_MINOR
        .iter()
        .map(|minor| format!("\"{minor}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:rustc-check-cfg=cfg(hipblaslt_minor, values({hipblaslt_minor}))");
}
