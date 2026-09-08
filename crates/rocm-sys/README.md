# `rocm-sys`

FFI bindings to C for ROCm *gated by version*. Explicitly checks for correct ROCm versions and function calls are guarnateed to be safe provided their prerequisites defined by AMDs documentation.

## Usage

This is a very unsafe and low-level API. For a safer API, use `rocm`.

```rust
let mut count = -1;

unsafe {
    hipGetDeviceCount(&mut count);
}

println!("device count = {count}");
```

* This will output `device count = -1` or `device count = 0` on failure or `device count = N` where `N >= 1` on success
* All `hip*` functions return a `hipError_t`

## Dynamic Loading vs Static Linking

The API is the same regardless of whether you use dynamic loading or static linking, but the functionality is different. If building for a generic target, use dynamic loading. If running on the build machine, static linking is faster.

|                          | Dynamic | Static |
|--------------------------|---------|--------|
| Runtime Checks           | Y       | N      |
| Portable Across Hardware | Y       | N      |
| Compile-Time Linking     | N       | Y      |

To enable dynamic loading, enable the cargo feature `dynamic-loading`.

When using static linking, all functions are just `extern "C"` functions wrapped in `extern "Rust"` functions with some compile-time ROCm version checks when applicable.

## Assumptions

This crate tries to assume as little about the target as possible.

- If `dynamic-loading` feature is disabled, the target is assumed to be the build machine
- The size of a Rust `usize` should be equal to the size of a C `size_t`
- `hipGetDeviceProperties` and `hipChooseDevice` are only correct profided `hipDeviceProp_tR0600` = `hipDeviceProp_t`

Try to avoid these. Device properties may get patched in the future.

## Contribution

All functions are hand-written (albeit using a macro) and types are auto-generated using bindgen.

Functions must be hand-written because they have version checks which bindgen doesnt make automatically. The API is as close to the C API as possible (template functions make this hard).

This is maintained separately from `rocm` because functions are all hand-written.
