# `roc`

Pure-Rust safe ROCm wrappers around `rocm-sys` and `rocm-include`, using `rocm-build` to find available libraries.

## Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Advanced Micro Devices, Inc. (AMD).

ROCm and AMD are trademarks of Advanced Micro Devices, Inc.

## Usage

This API is very limited and only covers a small subset of HIP and hipRTC, but no higher level libraries (e.g., hipBLAS, rocFFT, rocSOLVER).

The example shown below compiles and launches a simple program:

```rustuse roc::{
    hip::{
        device::Device,
        module::LaunchShape,
        stream::{HipStream, StreamKind},
    },
    hiprtc::program::{CompileOptions, Hsaco},
};

const SRC: &str = r#"
extern "C" __global__
void vec_add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"#;

const LEN: usize = 1024;

const A_VAL: f32 = 3.0;
const B_VAL: f32 = 2.0;

let dev = Device::current().unwrap();
let arch = dev.gfx_version().unwrap();

let opts = CompileOptions {
    opt_level: Some(3),
    fast_math: Some(true),
    name: Some(c"vec_add"),
    defines: &[],
    include_paths: &[],
    arch,
    options: &[],
};

let func = {
    let hsaco = Hsaco::compile(SRC, &opts).unwrap();
    let module = hsaco.load().unwrap();
    module.get_func_c(c"vec_add").unwrap()
};

let stream = HipStream::create(StreamKind::Default).unwrap();

let mut out = stream.alloc_unified::<f32>(LEN, true).unwrap();
let mut a = stream.alloc(LEN).unwrap();
let mut b = stream.alloc(LEN).unwrap();

let a_host = stream.map_slice(&[A_VAL; LEN]).unwrap();
let b_host = stream.map_slice(&[B_VAL; LEN]).unwrap();

a.copy_from_async(&a_host, 0, 0, LEN).unwrap();
b.copy_from_async(&b_host, 0, 0, LEN).unwrap();

let mut n = LEN as i32;

let mut args = [
    (&raw mut out).cast(),
    (&raw mut a).cast(),
    (&raw mut b).cast(),
    (&raw mut n).cast(),
];

let conf = LaunchShape {
    grid: [LEN.div_ceil(128) as u32, 1, 1],
    block: [128, 1, 1],
};

unsafe {
    stream.launch(&func, &mut args, conf).unwrap();
}

stream.sync().unwrap();

for val in out.as_slice() {
    assert_eq!(*val, A_VAL + B_VAL);
}
```
