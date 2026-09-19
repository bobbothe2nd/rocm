use core::mem::transmute;

use rocm_rt::{
    hip::{
        device::Device,
        memory::DevMapped,
        module::LaunchConfig,
        stream::Stream,
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
const BYTES: usize = LEN * size_of::<f32>();

const A_VAL: f32 = 3.0;
const B_VAL: f32 = 2.0;

fn main() {
    let dev = Device::create(0).unwrap();
    dev.set_default().unwrap();

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

    let hsaco = Hsaco::compile(SRC, &opts).unwrap();

    let func = hsaco.load().unwrap().get_func_c(c"vec_add").unwrap();

    let stream = Stream::create().unwrap();

    let out = dev.alloc(BYTES as u32).unwrap();
    let a = dev.alloc(BYTES as u32).unwrap();
    let b = dev.alloc(BYTES as u32).unwrap();

    let a_host = {
        let buf = unsafe { transmute::<[f32; LEN], [u8; BYTES]>([A_VAL; LEN]) };

        DevMapped::new(&buf).unwrap()
    };
    let b_host = {
        let buf = unsafe { transmute::<[f32; LEN], [u8; BYTES]>([B_VAL; LEN]) };

        DevMapped::new(&buf).unwrap()
    };

    stream.copy_htod(&a_host, &a, 0, 0, BYTES).unwrap();
    stream.copy_htod(&b_host, &b, 0, 0, BYTES).unwrap();

    let mut out_ptr = out.as_ptr();
    let mut a_ptr = a.as_ptr();
    let mut b_ptr = b.as_ptr();
    let mut n = LEN as i32;

    let mut args = [
        (&raw mut out_ptr).cast(),
        (&raw mut a_ptr).cast(),
        (&raw mut b_ptr).cast(),
        (&raw mut n).cast(),
    ];

    let conf = LaunchConfig {
        grid: [LEN.div_ceil(128) as u32, 1, 1],
        block: [128, 1, 1],
    };

    unsafe {
        stream.launch(&func, &mut args, conf).unwrap();
    }

    let out_host = DevMapped::alloc(BYTES).unwrap();

    stream.sync().unwrap();

    out.copy_to_host(&out_host, 0, 0, BYTES).unwrap();

    for chunk in out_host.as_slice().chunks(size_of::<f32>()) {
        let u8x4: [u8; size_of::<f32>()] = chunk.try_into().unwrap();
        let float = f32::from_ne_bytes(u8x4);

        assert_eq!(float, A_VAL + B_VAL);
    }
}
