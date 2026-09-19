use core::{
    ffi::{CStr, c_int},
    ptr::null,
};

use alloc::{boxed::Box, ffi::CString, format, vec::Vec};

use rocm_sys::hiprtc::{
    hiprtcCompileProgram, hiprtcCreateProgram, hiprtcDestroyProgram, hiprtcGetCode,
    hiprtcGetCodeSize, hiprtcProgram,
};

use crate::{hiprtc::HiprtcError, shared::GfxVersion};

/// Represents a compiled HSACO binary. (like CUBIN)
pub struct Hsaco {
    pub(crate) code: Box<[u8]>,
}

impl Hsaco {
    /// Compiles HIP into a binary.
    pub fn compile<S: AsRef<str>>(src: S, opts: &CompileOptions<'_>) -> Result<Self, HiprtcError> {
        #[repr(transparent)]
        struct HiprtcProgram {
            raw: hiprtcProgram,
        }

        impl Drop for HiprtcProgram {
            fn drop(&mut self) {
                unsafe {
                    let _ = hiprtcDestroyProgram(&mut self.raw);
                }
            }
        }

        let src = CString::new(src.as_ref().as_bytes()).map_err(|_| HiprtcError::InvalidProgram)?;

        let prog: Result<HiprtcProgram, HiprtcError> = unsafe {
            wrap_sys_res!(|prog| hiprtcCreateProgram(
                (&raw mut prog).cast(),
                src.as_ptr(),
                opts.name.map(|n| n.as_ptr()).unwrap_or(null()),
                0,
                null(),
                null(),
            ))
        };
        let prog = prog?;

        {
            let c_strs = opts.build();
            let opts = c_strs
                .iter()
                .map(CString::as_c_str)
                .map(CStr::as_ptr)
                .collect::<Vec<_>>();

            unsafe {
                try_err!(hiprtcCompileProgram(
                    prog.raw,
                    opts.len() as c_int,
                    opts.as_ptr().cast(),
                ));
            }
        }

        let code = {
            let code_size: Result<usize, HiprtcError> = unsafe {
                wrap_sys_res!(|size| hiprtcGetCodeSize(prog.raw, (&raw mut size).cast()))
            };
            let code_size = code_size?;

            let mut code = Box::new_uninit_slice(code_size);

            unsafe {
                try_err!(hiprtcGetCode(prog.raw, code.as_mut_ptr().cast(),));

                code.assume_init()
            }
        };

        Ok(Self { code })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileOptions<'a> {
    pub opt_level: Option<u8>,
    pub fast_math: Option<bool>,
    pub name: Option<&'a CStr>,
    pub defines: &'a [&'a str],
    pub include_paths: &'a [&'a str],
    pub arch: GfxVersion,
    pub options: &'a [&'a str],
}

impl CompileOptions<'_> {
    pub(crate) fn build(&self) -> Vec<CString> {
        let mut out: Vec<CString> = Vec::new();

        if let Some(level) = self.opt_level {
            out.push(CString::new(format!("-O{level}")).unwrap());
        }

        match self.fast_math {
            Some(true) => out.push(c"-ffast-math".into()),
            Some(false) => out.push(c"-fno-fast-math".into()),
            None => {}
        }

        for def in self.defines {
            out.push(CString::new(format!("-D{def}")).unwrap());
        }

        {
            let rocm_include = {
                let root = {
                    #[cfg(feature = "std")]
                    {
                        std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string())
                    }

                    #[cfg(not(feature = "std"))]
                    {
                        "/opt/rocm"
                    }
                };

                format!("{root}/include")
            };

            if !self.include_paths.contains(&rocm_include.as_str()) {
                out.push(CString::new(format!("-I{rocm_include}")).unwrap());
            }
        }

        for path in self.include_paths {
            out.push(CString::new(format!("-I{path}")).unwrap());
        }

        out.push(CString::new(format!("--offload-arch={}", self.arch)).unwrap());

        for opt in self.options {
            out.push(CString::new(*opt).unwrap());
        }

        out
    }
}
