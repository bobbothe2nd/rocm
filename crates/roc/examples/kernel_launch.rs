use core::mem::transmute;

use roc::{
    hip::{device::Device, module::LaunchConfig, stream::HipStream},
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
const BYTES: usize = LEN * size_of::<f32>();

const A_VAL: f32 = 3.0;
const B_VAL: f32 = 2.0;

fn main() {
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

    let stream = HipStream::create().unwrap();

    let mut out = stream.alloc(BYTES).unwrap();
    let mut a = stream.alloc(BYTES).unwrap();
    let mut b = stream.alloc(BYTES).unwrap();

    {
        let a_host = {
            let buf = unsafe { transmute::<[f32; LEN], [u8; BYTES as usize]>([A_VAL; LEN]) };

            stream.alloc_host_init(&buf).unwrap()
        };
        let b_host = {
            let buf = unsafe { transmute::<[f32; LEN], [u8; BYTES as usize]>([B_VAL; LEN]) };

            stream.alloc_host_init(&buf).unwrap()
        };

        a.copy_from_async(&a_host, 0, 0, BYTES as u32).unwrap();
        b.copy_from_async(&b_host, 0, 0, BYTES as u32).unwrap();
    }

    let mut n = LEN as i32;

    let mut args = [
        (&raw mut out).cast(),
        (&raw mut a).cast(),
        (&raw mut b).cast(),
        (&raw mut n).cast(),
    ];

    let conf = LaunchConfig {
        grid: [LEN.div_ceil(128) as u32, 1, 1],
        block: [128, 1, 1],
    };

    unsafe {
        stream.launch(&func, &mut args, conf).unwrap();
    }

    let mut out_host = stream.alloc_host(BYTES as u32).unwrap();

    out.copy_to_async(&mut out_host, 0, 0, BYTES as u32).unwrap();

    stream.sync().unwrap();

    for chunk in out_host.chunks(size_of::<f32>()) {
        let u8x4: [u8; size_of::<f32>()] = chunk.try_into().unwrap();
        let float = f32::from_ne_bytes(u8x4);

        assert_eq!(float, A_VAL + B_VAL);
    }
}
