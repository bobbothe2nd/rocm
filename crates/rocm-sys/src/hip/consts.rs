#![allow(clippy::excessive_precision, clippy::approx_constant)]

pub const HIP_INF_F: f32 = f32::from_bits(0x7f800000);
pub const HIP_NAN_F: f32 = f32::from_bits(0x7fffffff);
pub const HIP_MIN_DENORM_F: f32 = f32::from_bits(0x00000001);
pub const HIP_MAX_NORMAL_F: f32 = f32::from_bits(0x7f7fffff);
pub const HIP_NEG_ZERO_F: f32 = f32::from_bits(0x80000000);

pub const HIP_ZERO_F: f32 = 0.0;
pub const HIP_ONE_F: f32 = 1.0;
pub const HIP_SQRT_HALF_F: f32 = 0.707106781;
pub const HIP_SQRT_HALF_HI_F: f32 = 0.707106781;
pub const HIP_SQRT_HALF_LO_F: f32 = 1.210161749e-08;
pub const HIP_SQRT_TWO_F: f32 = 1.414213562;
pub const HIP_THIRD_F: f32 = 0.333333333;
pub const HIP_PIO4_F: f32 = 0.785398163;
pub const HIP_PIO2_F: f32 = 1.570796327;
pub const HIP_3PIO4_F: f32 = 2.356194490;
pub const HIP_2_OVER_PI_F: f32 = 0.636619772;
pub const HIP_SQRT_2_OVER_PI_F: f32 = 0.797884561;
pub const HIP_PI_F: f32 = 3.141592654;
pub const HIP_L2E_F: f32 = 1.442695041;
pub const HIP_L2T_F: f32 = 3.321928094;
pub const HIP_LG2_F: f32 = 0.301029996;
pub const HIP_LGE_F: f32 = 0.434294482;
pub const HIP_LN2_F: f32 = 0.693147181;
pub const HIP_LNT_F: f32 = 2.302585093;
pub const HIP_LNPI_F: f32 = 1.144729886;
pub const HIP_TWO_TO_M126_F: f32 = 1.175494351e-38;
pub const HIP_TWO_TO_126_F: f32 = 8.507059173e37;
pub const HIP_NORM_HUGE_F: f32 = 3.402823466e38;
pub const HIP_TWO_TO_23_F: f32 = 8388608.0;
pub const HIP_TWO_TO_24_F: f32 = 16777216.0;
pub const HIP_TWO_TO_31_F: f32 = 2147483648.0;
pub const HIP_TWO_TO_32_F: f32 = 4294967296.0;

pub const HIP_REMQUO_BITS_F: u32 = 3;
pub const HIP_REMQUO_MASK_F: u32 = !((!0u32) << HIP_REMQUO_BITS_F);

pub const HIP_TRIG_PLOSS_F: f32 = 105615.0;

pub const HIP_INF: f64 = f64::from_bits(0x7ff0000000000000);
pub const HIP_NAN: f64 = f64::from_bits(0xfff8000000000000);
pub const HIP_NEG_ZERO: f64 = f64::from_bits(0x8000000000000000);
pub const HIP_MIN_DENORM: f64 = f64::from_bits(0x0000000000000001);

pub const HIP_ZERO: f64 = 0.0;
pub const HIP_ONE: f64 = 1.0;
pub const HIP_SQRT_TWO: f64 = 1.4142135623730951e+0;
pub const HIP_SQRT_HALF: f64 = 7.0710678118654757e-1;
pub const HIP_SQRT_HALF_HI: f64 = 7.0710678118654757e-1;
pub const HIP_SQRT_HALF_LO: f64 = -4.8336466567264567e-17;
pub const HIP_THIRD: f64 = 3.3333333333333333e-1;
pub const HIP_TWOTHIRD: f64 = 6.6666666666666667e-1;
pub const HIP_PIO4: f64 = 7.8539816339744828e-1;
pub const HIP_PIO4_HI: f64 = 7.8539816339744828e-1;
pub const HIP_PIO4_LO: f64 = 3.0616169978683830e-17;
pub const HIP_PIO2: f64 = 1.5707963267948966e+0;
pub const HIP_PIO2_HI: f64 = 1.5707963267948966e+0;
pub const HIP_PIO2_LO: f64 = 6.1232339957367660e-17;
pub const HIP_3PIO4: f64 = 2.3561944901923448e+0;
pub const HIP_2_OVER_PI: f64 = 6.3661977236758138e-1;
pub const HIP_PI: f64 = 3.1415926535897931e+0;
pub const HIP_PI_HI: f64 = 3.1415926535897931e+0;
pub const HIP_PI_LO: f64 = 1.2246467991473532e-16;
pub const HIP_SQRT_2PI: f64 = 2.5066282746310007e+0;
pub const HIP_SQRT_2PI_HI: f64 = 2.5066282746310007e+0;
pub const HIP_SQRT_2PI_LO: f64 = -1.8328579980459167e-16;
pub const HIP_SQRT_PIO2: f64 = 1.2533141373155003e+0;
pub const HIP_SQRT_PIO2_HI: f64 = 1.2533141373155003e+0;
pub const HIP_SQRT_PIO2_LO: f64 = -9.1642899902295834e-17;
pub const HIP_SQRT_2OPI: f64 = 7.9788456080286536e-1;
pub const HIP_L2E: f64 = 1.4426950408889634e+0;
pub const HIP_L2E_HI: f64 = 1.4426950408889634e+0;
pub const HIP_L2E_LO: f64 = 2.0355273740931033e-17;
pub const HIP_L2T: f64 = 3.3219280948873622e+0;
pub const HIP_LG2: f64 = 3.0102999566398120e-1;
pub const HIP_LG2_HI: f64 = 3.0102999566398120e-1;
pub const HIP_LG2_LO: f64 = -2.8037281277851704e-18;
pub const HIP_LGE: f64 = 4.3429448190325182e-1;
pub const HIP_LGE_HI: f64 = 4.3429448190325182e-1;
pub const HIP_LGE_LO: f64 = 1.09831965021676510e-17;
pub const HIP_LN2: f64 = 6.9314718055994529e-1;
pub const HIP_LN2_HI: f64 = 6.9314718055994529e-1;
pub const HIP_LN2_LO: f64 = 2.3190468138462996e-17;
pub const HIP_LNT: f64 = 2.3025850929940459e+0;
pub const HIP_LNT_HI: f64 = 2.3025850929940459e+0;
pub const HIP_LNT_LO: f64 = -2.1707562233822494e-16;
pub const HIP_LNPI: f64 = 1.1447298858494002e+0;
pub const HIP_LN2_X_1024: f64 = 7.0978271289338397e+2;
pub const HIP_LN2_X_1025: f64 = 7.1047586007394398e+2;
pub const HIP_LN2_X_1075: f64 = 7.4513321910194122e+2;
pub const HIP_LG2_X_1024: f64 = 3.0825471555991675e+2;
pub const HIP_LG2_X_1075: f64 = 3.2360724533877976e+2;
pub const HIP_TWO_TO_23: f64 = 8388608.0;
pub const HIP_TWO_TO_52: f64 = 4503599627370496.0;
pub const HIP_TWO_TO_53: f64 = 9007199254740992.0;
pub const HIP_TWO_TO_54: f64 = 18014398509481984.0;
pub const HIP_TWO_TO_M54: f64 = 5.5511151231257827e-17;
pub const HIP_TWO_TO_M1022: f64 = 2.22507385850720140e-308;
pub const HIP_TRIG_PLOSS: f64 = 2147483648.0;
pub const HIP_DBL2INT_CVT: f64 = 6755399441055744.0;

/// Default flag. Equivalent to `HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM` if compiled with
/// `-fgpu-default-stream=per-thread flag` or `HIP_API_PER_THREAD_DEFAULT_STREAM` macro is
/// defined.
pub const HIP_GET_PROC_ADDRESS_DEFAULT: u32 = 0;
/// Search for all symbols except the corresponding per-thread versions.
pub const HIP_GET_PROC_ADDRESS_LEGACY_STREAM: u32 = 1;
/// Search for all symbols including the per-thread versions. If a per-thread version cannot be
/// found, returns the legacy version.
pub const HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM: u32 = 2;
/// The hipTexRefSetArray function flags parameter override format value
pub const HIP_TRSA_OVERRIDE_FORMAT: u32 = 1;
/// The `hipTexRefSetFlags` function flags parameter read as integer value
pub const HIP_TRSF_READ_AS_INTEGER: u32 = 1;
/// The `hipTexRefSetFlags` function flags parameter normalized coordinate value
pub const HIP_TRSF_NORMALIZED_COORDINATES: u32 = 2;
/// The `hipTexRefSetFlags` function flags parameter srgb value
pub const HIP_TRSF_SRGB: u32 = 16;
pub const hipTextureType1D: u32 = 1;
pub const hipTextureType2D: u32 = 2;
pub const hipTextureType3D: u32 = 3;
pub const hipTextureTypeCubemap: u32 = 12;
pub const hipTextureType1DLayered: u32 = 241;
pub const hipTextureType2DLayered: u32 = 242;
pub const hipTextureTypeCubemapLayered: u32 = 252;
/// Should be same as `HSA_IMAGE_OBJECT_SIZE_DWORD`/`HSA_SAMPLER_OBJECT_SIZE_DWORD`
pub const HIP_IMAGE_OBJECT_SIZE_DWORD: u32 = 12;
pub const HIP_SAMPLER_OBJECT_SIZE_DWORD: u32 = 8;
pub const HIP_SAMPLER_OBJECT_OFFSET_DWORD: u32 = 12;
pub const HIP_TEXTURE_OBJECT_SIZE_DWORD: u32 = 20;
pub const HIP_DEPRECATED_MSG : & [u8 ; 189] = b"This API is marked as deprecated and might not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md\0" ;
pub const HIP_RESOURCE_ABI_BYTES: u32 = 40;
pub const hipIpcMemLazyEnablePeerAccess: u32 = 1;
pub const HIP_IPC_HANDLE_SIZE: u32 = 64;
/// Default stream creation flags. These are used with `hipStreamCreate()`.
pub const hipStreamDefault: u32 = 0;
/// Stream does not implicitly synchronize with null stream.
pub const hipStreamNonBlocking: u32 = 1;
/// Default flags.
pub const hipEventDefault: u32 = 0;
/// Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.
pub const hipEventBlockingSync: u32 = 1;
/// Disable event's capability to record timing information. May improve performance.
pub const hipEventDisableTiming: u32 = 2;
/// Event can support IPC. hipEventDisableTiming also must be set.
pub const hipEventInterprocess: u32 = 4;
/// Default flag.
pub const hipEventRecordDefault: u32 = 0;
/// Event is captured in the graph as an external event node when performing stream capture.
pub const hipEventRecordExternal: u32 = 1;
/// Default flag.
pub const hipEventWaitDefault: u32 = 0;
/// Wait is captured in the graph as an external event node when performing stream capture.
pub const hipEventWaitExternal: u32 = 1;
/// Disable performing a system scope sequentially consistent memory fence when the event
/// transitions from recording to recorded.  This can be used for events that are only being
/// used to measure timing, and do not require the event inspection operations
/// (see `::hipEventSynchronize`, `::hipEventQuery`, and `::hipEventElapsedTime`) to synchronize-with
/// the work on which the recorded event (see `::hipEventRecord`) is waiting.
/// On some AMD GPU devices this can improve the accuracy of timing measurements by avoiding the
/// cost of cache writeback and invalidation, and the performance impact of those actions on the
/// execution of following work.
pub const hipEventDisableSystemFence: u32 = 536870912;
/// Use a device-scope release when recording this event. This flag is useful to obtain more
/// precise timings of commands between events.  The flag is a no-op on CUDA platforms.
pub const hipEventReleaseToDevice: u32 = 1073741824;
/// Use a system-scope release when recording this event. This flag is useful to make
/// non-coherent host memory visible to the host. The flag is a no-op on CUDA platforms.
pub const hipEventReleaseToSystem: u32 = 2147483648;
/// Default flag.
///
/// Equivalent to hipEnablePerThreadDefaultStream if compiled with
/// `-fgpu-default-stream=per-thread` flag or `HIP_API_PER_THREAD_DEFAULT_STREAM` macro is
/// defined.
pub const hipEnableDefault: u32 = 0;
/// Search for all symbols except the corresponding per-thread versions.
pub const hipEnableLegacyStream: u32 = 1;
/// Search for all symbols including the per-thread versions. If a per-thread version cannot be
/// found, returns the legacy version.
pub const hipEnablePerThreadDefaultStream: u32 = 2;
/// Default pinned memory allocation on the host.
pub const hipHostAllocDefault: u32 = 0;
/// Default pinned memory allocation on the host.
///
/// This is the same definition as #hipHostAllocPortable.
pub const hipHostMallocDefault: u32 = 0;
/// Memory is considered allocated by all contexts.
pub const hipHostAllocPortable: u32 = 1;
/// Memory is considered allocated by all contexts.
///
/// This is the same definition as #hipHostAllocPortable.
pub const hipHostMallocPortable: u32 = 1;
/// Map the allocation into the address space for the current device. The device pointer
/// can be obtained with #hipHostGetDevicePointer.
pub const hipHostAllocMapped: u32 = 2;
/// Map the allocation into the address space for the current device. The device pointer
/// can be obtained with #hipHostGetDevicePointer.
///
/// This is the same #hipHostMallocMapped.
pub const hipHostMallocMapped: u32 = 2;
/// Allocates the memory as write-combined. On some system configurations, write-combined allocation
/// may be transferred faster across the PCI Express bus, however, could have low read efficiency by
/// most CPUs. It's a good option for data transfer from host to device via mapped pinned memory.
///
///  This flag is only for CUDA source compatibility but not functional within HIP runtime,
/// because the allocation path is currently not supported on the AMD platform.
pub const hipHostAllocWriteCombined: u32 = 4;
/// Allocates the memory as write-combined. On some system configurations, write-combined allocation
/// may be transferred faster across the PCI Express bus, however, could have low read efficiency by
/// most CPUs. It's a good option for data transfer from host to device via mapped pinned memory.
///
///  This flag is the same definition as #hipHostAllocWriteCombined which is equivalent to
/// cudaHostAllocWriteCombined. It is only for CUDA source compatibility but not functional within
/// HIP runtime, because the allocation path is currently not supported on the AMD platform.
pub const hipHostMallocWriteCombined: u32 = 4;
/// Host memory will be forcedly allocated on extended fine grained system memory
/// pool which is with MTYPE_UC.
///
///  This allocation flag is applicable on AMD devices, except for Navi4X, in Linux only.
pub const hipHostMallocUncached: u32 = 268435456;
pub const hipHostAllocUncached: u32 = 268435456;
/// Host memory allocation will follow numa policy set by user.
///
///  This numa allocation flag is applicable on Linux, under development on Windows.
pub const hipHostMallocNumaUser: u32 = 536870912;
/// Allocate coherent memory. Overrides HIP_HOST_COHERENT for specific allocation.
pub const hipHostMallocCoherent: u32 = 1073741824;
/// Allocate non-coherent memory. Overrides HIP_HOST_COHERENT for specific allocation.
pub const hipHostMallocNonCoherent: u32 = 2147483648;
/// Memory can be accessed by any stream on any device
pub const hipMemAttachGlobal: u32 = 1;
/// Memory cannot be accessed by any stream on any device.
pub const hipMemAttachHost: u32 = 2;
/// Memory can only be accessed by a single stream on the associated device.
pub const hipMemAttachSingle: u32 = 4;
pub const hipDeviceMallocDefault: u32 = 0;
/// Memory is allocated in fine grained region of device.
pub const hipDeviceMallocFinegrained: u32 = 1;
/// Memory represents a HSA signal.
pub const hipMallocSignalMemory: u32 = 2;
/// Memory allocated will be uncached.
pub const hipDeviceMallocUncached: u32 = 3;
/// Memory allocated will be contiguous.
pub const hipDeviceMallocContiguous: u32 = 4;
/// Memory is Mapped and Portable.
pub const hipHostRegisterDefault: u32 = 0;
/// Memory is considered registered by all contexts.
pub const hipHostRegisterPortable: u32 = 1;
/// Map the allocation into the address space for the current device. The device pointer
/// can be obtained with #hipHostGetDevicePointer.
pub const hipHostRegisterMapped: u32 = 2;
/// The passed memory pointer is treated as pointing to some memory-mapped I/O space, e.g.
/// belonging to a third-party PCIe device, and it will be marked as non cache-coherent and
/// contiguous.
pub const hipHostRegisterIoMemory: u32 = 4;
/// This flag is ignored On AMD devices.
pub const hipHostRegisterReadOnly: u32 = 8;
/// Coarse Grained host memory lock.
pub const hipExtHostRegisterCoarseGrained: u32 = 8;
/// Map host memory onto extended fine grained access host memory pool when enabled.
/// It is applicable on AMD devices, except for Navi4X, in Linux only.
pub const hipExtHostRegisterUncached: u32 = 2147483648;
/// Automatically select between Spin and Yield.
pub const hipDeviceScheduleAuto: u32 = 0;
/// Dedicate a CPU core to spin-wait. Provides lowest latency, but burns a CPU core and may
/// consume more power.
pub const hipDeviceScheduleSpin: u32 = 1;
/// Yield the CPU to the operating system when waiting. May increase latency, but lowers power
/// and is friendlier to other threads in the system.
pub const hipDeviceScheduleYield: u32 = 2;
pub const hipDeviceScheduleBlockingSync: u32 = 4;
pub const hipDeviceScheduleMask: u32 = 7;
pub const hipDeviceMapHost: u32 = 8;
pub const hipDeviceLmemResizeToMax: u32 = 16;
/// Indicates that the deviceFlags passed to hipInitDevice() are valid and should be applied.
pub const hipInitDeviceFlagsAreValid: u32 = 1;
/// Default HIP array allocation flag.
pub const hipArrayDefault: u32 = 0;
pub const hipArrayLayered: u32 = 1;
pub const hipArraySurfaceLoadStore: u32 = 2;
pub const hipArrayCubemap: u32 = 4;
pub const hipArrayTextureGather: u32 = 8;
pub const hipOccupancyDefault: u32 = 0;
pub const hipOccupancyDisableCachingOverride: u32 = 1;
pub const hipCooperativeLaunchMultiDeviceNoPreSync: u32 = 1;
pub const hipCooperativeLaunchMultiDeviceNoPostSync: u32 = 2;
pub const hipCpuDeviceId: i32 = -1;
pub const hipInvalidDeviceId: i32 = -2;
/// AnyOrderLaunch of kernels.
pub const hipExtAnyOrderLaunch: u32 = 1;
pub const hipStreamWaitValueGte: u32 = 0;
pub const hipStreamWaitValueEq: u32 = 1;
pub const hipStreamWaitValueAnd: u32 = 2;
pub const hipStreamWaitValueNor: u32 = 3;
pub const hipStreamWriteValueDefault: u32 = 0;
pub const hipExtStreamWriteValueIncrement: u32 = 4096;
pub const hipExtStreamWriteValueDecrement: u32 = 4097;
pub const hipExternalMemoryDedicated: u32 = 1;
/// Compile-time upper bound for the number of prefetch regions in
/// [`hipExtDynDataPrefetchConfig`](`crate::hip::hipExtDynDataPrefetchConfig`). The actual device limit is queried at
/// runtime via [`hipDeviceAttributeMaxDynDataPrefetchRegions`].
pub const HIP_EXT_DYN_DATA_PREFETCH_MAX_REGIONS: u32 = 2;
/// This port activates when the kernel has finished executing.
pub const hipGraphKernelNodePortDefault: u32 = 0;
/// This port activates when all blocks of the kernel have begun execution.
pub const hipGraphKernelNodePortLaunchCompletion: u32 = 2;
/// This port activates when all blocks of the kernel have performed
/// `hipTriggerProgrammaticLaunchCompletion()` or have terminated.
/// It must be used with edge type hipGraphDependencyTypeProgrammatic.
pub const hipGraphKernelNodePortProgrammatic: u32 = 1;
