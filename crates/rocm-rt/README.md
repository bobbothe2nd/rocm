# `rocm-rt`

Safe ROCm wrappers around `rocm-sys`, using `rocm-build` to find available libraries.

## Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Advanced Micro Devices, Inc. (AMD).

ROCm and AMD are trademarks of Advanced Micro Devices, Inc.

## Usage

This API is very small right now, but will be expanded rapidly.

```rust
use rocm_rt::hip::{device::Device, stream::Stream};

fn main() {
    let dev = Device::new(0).unwrap();
    dev.set_default().unwrap();

    let _ = dev.into_raw();

    let stream = Stream::new().unwrap();
    let _ = stream.into_raw();
}
```

That creates a stream on device 0.
