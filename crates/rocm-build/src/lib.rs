pub mod ver;
pub mod link;

macro_rules! build {
    ($($link:ident)?) => {
        $crate::link::check_cfg();

        let roots = $crate::link::rocm_roots();
        let mut found_rocm_ver = false;

        if let Some(version) = $crate::ver::find_rocm_version(&roots) {
            found_rocm_ver = true;
            $crate::ver::cfg_version("rocm_ver", version, $crate::ver::KNOWN_ROCM_VERSIONS);
        }

        if let Some(component) = $crate::link::find_component(
            &roots,
            &["hip_runtime_api.h", "hip_runtime.h"],
            &["amdhip64"],
        ) {
            if !found_rocm_ver {
                let version = unsafe {
                    $crate::ver::hip_runtime_version(&component.library_dir)
                        .expect("failed to determine HIP runtime version")
                };

                $crate::ver::cfg_version("rocm_ver", version, $crate::ver::KNOWN_ROCM_VERSIONS);
            }

            $(
                let $link = ();
                $crate::link::link_component(&component);
            )?

            println!("cargo:rustc-cfg=hip");
        }

        if let Some(component) = $crate::link::find_component(&roots, &["rocblas.h"], &["rocblas"]) {
            let version = unsafe {
                $crate::ver::rocblas_version(&component.library_dir)
                    .expect("failed to determine HIP runtime version")
            };

            $crate::ver::cfg_version("rocblas_ver", version, $crate::ver::KNOWN_ROCBLAS_VERSIONS);

            $(
                let $link = ();
                $crate::link::link_component(&component);
            )?

            println!("cargo:rustc-cfg=rocblas");
        }

        $(
            if let Some(component) = $crate::link::find_component(&roots, &["hiprtc.h"], &["hiprtc"]) {
                let $link = ();
                $crate::link::link_component(&component);

                println!("cargo:rustc-cfg=hiprtc");
            }
        )?

        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_HOME");
    };
}

/// Finds available ROCm versions on system and tell `rustc`.
pub fn build_version_cfg() {
    build!();
}

/// Tells `rustc` to link libraries and finds available versions.
pub fn build_version_cfg_link() {
    build!(_link);
}
