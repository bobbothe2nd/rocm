use core::mem::transmute;

use rocm_rt::{
    hip::{
        device::Device,
        graph::{Graph, GraphInstantiateFlags, KernelParams},
        module::LaunchConfig,
        stream::HipStream,
    },
    hiprtc::program::{CompileOptions, Hsaco},
};

const SRC: &str = r#"
extern "C" __global__
void vec_add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += a[i] + b[i];
}
"#;

const LEN: usize = 1024;
const BYTES: u32 = (LEN * size_of::<f32>()) as u32;

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

    let mut out = stream.alloc_zeroed(BYTES).unwrap();
    let mut a = stream.alloc(BYTES).unwrap();
    let mut b = stream.alloc(BYTES).unwrap();

    let a_host = {
        let buf = unsafe { transmute::<[f32; LEN], [u8; BYTES as usize]>([A_VAL; LEN]) };

        stream.alloc_host_init(&buf).unwrap()
    };

    let b_host = {
        let buf = unsafe { transmute::<[f32; LEN], [u8; BYTES as usize]>([B_VAL; LEN]) };

        stream.alloc_host_init(&buf).unwrap()
    };

    a.copy_from_host_async(&a_host, 0, 0, BYTES).unwrap();
    b.copy_from_host_async(&b_host, 0, 0, BYTES).unwrap();

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

    let params = KernelParams::new(&func, &mut args, conf);

    let mut graph = Graph::new(GraphInstantiateFlags::empty()).unwrap();

    let mut prev_nodes = Vec::with_capacity(10);

    for _ in 0..10 {
        let node = graph.add_kernel_node(&prev_nodes, &params).unwrap();
        prev_nodes.push(node);
    }

    let exec = graph.init(GraphInstantiateFlags::empty()).unwrap();

    unsafe {
        exec.upload(&stream).unwrap();
        stream.launch_graph(&exec).unwrap();
    }

    stream.sync().unwrap();

    let out_host = stream.alloc_host(BYTES).unwrap();

    out.copy_to_host(&out_host.borrowed(), 0, 0, BYTES).unwrap();

    for chunk in out_host.as_slice().chunks(size_of::<f32>()) {
        let bytes: [u8; size_of::<f32>()] = chunk.try_into().unwrap();
        let value = f32::from_ne_bytes(bytes);

        assert_eq!(value, 10.0 * (A_VAL + B_VAL));
    }
}
