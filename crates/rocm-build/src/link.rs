use crate::ver::{
    KNOWN_HIPBLAS_VERSONS, KNOWN_HIPBLASLT_VERSIONS, KNOWN_ROCBLAS_VERSIONS, KNOWN_ROCFFT_VERSIONS,
    KNOWN_ROCM_VERSIONS,
};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

pub struct Component {
    pub include_dir: PathBuf,
    pub library_dir: PathBuf,
    pub library: String,
}

pub fn rocm_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    for variable in ["ROCM_PATH", "ROCM_HOME"] {
        if let Some(path) = env::var_os(variable) {
            roots.push(PathBuf::from(path));
        }
    }

    roots.extend([
        PathBuf::from("/opt/rocm"),
        PathBuf::from("/usr/rocm"),
        PathBuf::from("/usr/local/rocm"),
    ]);

    roots
}

pub fn find_component(
    roots: &[PathBuf],
    headers: &[&str],
    libraries: &[&str],
) -> Option<Component> {
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

pub fn find_file(root: &Path, names: &[&str]) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };

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

pub fn find_library(root: &Path, names: &[&str]) -> Option<PathBuf> {
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

pub fn link_component(component: &Component) {
    println!(
        "cargo:rustc-link-search=native={}",
        component.library_dir.display()
    );

    println!("cargo:rustc-link-lib=dylib={}", component.library);

    println!("cargo:include={}", component.include_dir.display());
}

pub fn check_cfg() {
    println!("cargo:rustc-check-cfg=cfg(hip)");
    println!("cargo:rustc-check-cfg=cfg(hiprtc)");
    println!("cargo:rustc-check-cfg=cfg(hipblas)");
    println!("cargo:rustc-check-cfg=cfg(hipblaslt)");
    println!("cargo:rustc-check-cfg=cfg(hipcub)");
    println!("cargo:rustc-check-cfg=cfg(hipdnn)");
    println!("cargo:rustc-check-cfg=cfg(hipfft)");
    println!("cargo:rustc-check-cfg=cfg(hiprand)");
    println!("cargo:rustc-check-cfg=cfg(hipsolver)");
    println!("cargo:rustc-check-cfg=cfg(hipsparse)");
    println!("cargo:rustc-check-cfg=cfg(hipsparselt)");
    println!("cargo:rustc-check-cfg=cfg(hiptensor)");
    println!("cargo:rustc-check-cfg=cfg(hipthreads)");
    println!("cargo:rustc-check-cfg=cfg(rocalution)");
    println!("cargo:rustc-check-cfg=cfg(rocblas)");
    println!("cargo:rustc-check-cfg=cfg(rocfft)");
    println!("cargo:rustc-check-cfg=cfg(rocprim)");
    println!("cargo:rustc-check-cfg=cfg(rocrand)");
    println!("cargo:rustc-check-cfg=cfg(rocsolver)");
    println!("cargo:rustc-check-cfg=cfg(rocsparse)");
    println!("cargo:rustc-check-cfg=cfg(rocthrust)");
    println!("cargo:rustc-check-cfg=cfg(rocwmma)");

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
