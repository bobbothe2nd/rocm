# `rocm-build`

Runtime detection of ROCm libraries intended for use in build scripts.

## Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Advanced Micro Devices, Inc. (AMD).

ROCm and AMD are trademarks of Advanced Micro Devices, Inc.

## Usage

Create a `build.rs` for your crate and enter:

```rust
use rocm_build::build_version_cfg;

fn main() {
    build_version_cfg();
}
```

Or if you want to link the libraries:

```rust
use rocm_build::build_version_cfg_link;

fn main() {
    build_version_cfg_link();
}
```
