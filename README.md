# ROCm

This project consists of 4 Rust crates:

1. `rocm-include`: mostly compile-time version and C parsing
2. `rocm-build`: build-time ROCm library linking/detection
3. `rocm-sys`: low-level, raw ROCm bindings for various libraries
4. `roc`: a high-level abstraction around `rocm-sys`, offering a safe API

Each level builds off the previous ones.

## Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Advanced Micro Devices, Inc. (AMD).

ROCm and AMD are trademarks of Advanced Micro Devices, Inc.
