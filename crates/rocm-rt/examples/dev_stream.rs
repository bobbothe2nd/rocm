use rocm_rt::hip::{device::Device, stream::Stream};

fn main() {
    let dev = Device::new(0).unwrap();
    dev.set_default().unwrap();

    let _ = dev.into_raw();

    let stream = Stream::new().unwrap();
    let _ = stream.into_raw();
}
