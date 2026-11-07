pub mod link;

macro_rules! build {
    ($($link:ident)?) => {
        $crate::link::check_cfg();

        let roots = $crate::link::rocm_roots();
        let mut found_rocm_ver = false;

        if let Some(version) = ::rocm_include::ver::find_rocm_version(&roots) {
            found_rocm_ver = true;
            $crate::cfg_version("rocm_ver", version, ::rocm_include::consts::KNOWN_ROCM_VERSIONS);
        }

        if let Some(_component) = $crate::link::find_component(
            &roots,
            &["hip_runtime_api.h", "hip_runtime.h"],
            &["amdhip64"],
        ) {
            if !found_rocm_ver {
                let version = ::rocm_include::ver::ROCM_VERSION.expect("failed to determine HIP runtime version");
                $crate::cfg_version("rocm_ver", version, ::rocm_include::consts::KNOWN_ROCM_VERSIONS);
            }

            $(
                let $link = ();
                $crate::link::link_component(&_component);
            )?

            println!("cargo:rustc-cfg=hip");
        }
        build!(@component roots $([$link])? "hiprtc");

        build!(@component roots $([$link])? "hipblas");
        build!(@component roots $([$link])? "hipblaslt");
        build!(@component roots $([$link])? "hipdnn");
        build!(@component roots $([$link])? "hipfft");
        build!(@component roots $([$link])? "hiprand");
        build!(@component roots $([$link])? "hipsolver");
        build!(@component roots $([$link])? "hipsparse");
        build!(@component roots $([$link])? "hipsparselt");
        build!(@component roots $([$link])? "hiptensor");

        build!(@component roots $([$link])? "rocalution");
        build!(@component roots $([$link])? "rocblas");
        build!(@component roots $([$link])? "rocfft");
        build!(@component roots $([$link])? "rocrand");
        build!(@component roots $([$link])? "rocsolver");
        build!(@component roots $([$link])? "rocsparse");

        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_HOME");
    };

    (@component $roots:ident $([$link:ident])? $name:literal) => {
        $(
            let $link = ();
            if let Some(component) = $crate::link::find_component(&$roots, &[concat!($name, ".h")], &[$name]) {
                $crate::link::link_component(&component);

                println!(concat!("cargo:rustc-cfg=", $name));
            }
        )?
    };

    (@component $roots:ident $([$link:ident])? $name:literal $ver:path) => {
        if let Some(_component) = $crate::link::find_component(&$roots, &[concat!($name, ".h")], &[$name]) {
            $(
                let $link = ();
                $crate::link::link_component(&_component);
            )?

            println!(concat!("cargo:rustc-cfg=", $name));
        }
    };
}

/// Finds available ROCm versions on system and adds cargo `#[cfg]`.
pub fn build_version_cfg() {
    build!();
}

/// Adds cargo `#[cfg]`, links libraries, and finds available versions.
pub fn build_version_cfg_link() {
    build!(_link);
}

fn cfg_version(cfg: &str, version: rocm_include::ver::Version, known: &[&str]) {
    let formatted = format!("{}.{}", version.major, version.minor);
    let index = known
        .iter()
        .position(|x| *x == formatted.as_str())
        .unwrap_or(known.len() - 1);

    for version in &known[index..] {
        println!("cargo:rustc-cfg={cfg}=\"{}\"", version);
    }
}
