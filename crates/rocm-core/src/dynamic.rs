use std::{env, fs, path::PathBuf, sync::LazyLock};

use crate::{c::parse_define_const, ver::Version};

pub static ROCM_VERSION: LazyLock<Option<Version>> = LazyLock::new(|| {
    let rocm_path = env::var("ROCM_PATH").ok();

    let mut headers = Vec::with_capacity(1 + rocm_path.is_some() as usize);
    headers.push("/opt/rocm/include/rocm-core/rocm_version.h".into());
    if let Some(rocm_path) = &rocm_path {
        headers.push(format!("{rocm_path}/include/rocm-core/rocm_version.h").into());
    }

    find_rocm_version(&headers)
});

pub fn find_rocm_version(roots: &[PathBuf]) -> Option<Version> {
    for root in roots {
        let path = root.join("include/rocm-core/rocm_version.h");

        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };

        let major = parse_define_const(&contents, "ROCM_VERSION_MAJOR")?;
        let minor = parse_define_const(&contents, "ROCM_VERSION_MINOR")?;

        return Some(Version { major, minor });
    }

    None
}
