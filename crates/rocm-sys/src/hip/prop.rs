use crate::hip::{hipDeviceArch_t, hipUUID};

#[cfg(any(rocm_ver = "6.0", feature = "dynamic-loading"))]
pub type hipDeviceProp_t = hipDeviceProp_tR0600;
#[cfg(not(any(rocm_ver = "6.0", feature = "dynamic-loading")))]
pub type hipDeviceProp_t = hipDeviceProp_tR0000;

#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipDeviceProp_tR0000 {
    /// Device name.
    pub name: [::core::ffi::c_char; 256],

    /// Size of the global memory region, in bytes.
    pub totalGlobalMem: usize,

    /// Size of the shared memory region, in bytes.
    pub sharedMemPerBlock: usize,

    /// Number of registers per block.
    pub regsPerBlock: ::core::ffi::c_int,

    /// Warp size.
    pub warpSize: ::core::ffi::c_int,

    /// Maximum number of work items per work group.
    pub maxThreadsPerBlock: ::core::ffi::c_int,

    /// Maximum number of threads in each block dimension (X, Y, Z).
    pub maxThreadsDim: [::core::ffi::c_int; 3],

    /// Maximum grid dimensions (X, Y, Z).
    pub maxGridSize: [::core::ffi::c_int; 3],

    /// Maximum multiprocessor clock frequency, in kHz.
    pub clockRate: ::core::ffi::c_int,

    /// Maximum global memory clock frequency, in kHz.
    pub memoryClockRate: ::core::ffi::c_int,

    /// Global memory bus width, in bits.
    pub memoryBusWidth: ::core::ffi::c_int,

    /// Size of the constant memory region, in bytes.
    pub totalConstMem: usize,

    /// Major compute capability version.
    ///
    /// On HCC, this is an approximation, and supported features may differ
    /// from CUDA compute capability. Use architectural feature flags to
    /// query feature capabilities portably.
    pub major: ::core::ffi::c_int,

    /// Minor compute capability version.
    ///
    /// On HCC, this is an approximation, and supported features may differ
    /// from CUDA compute capability. Use architectural feature flags to
    /// query feature capabilities portably.
    pub minor: ::core::ffi::c_int,

    /// Number of multiprocessors.
    ///
    /// In Compute Unit (CU) mode, this equals the number of CUs. In
    /// Workgroup Processor (WGP) mode, this equals half the number of CUs,
    /// since each WGP contains two CUs.
    pub multiProcessorCount: ::core::ffi::c_int,

    /// L2 cache size.
    pub l2CacheSize: ::core::ffi::c_int,

    /// Maximum number of resident threads per multiprocessor.
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,

    /// Compute mode.
    pub computeMode: ::core::ffi::c_int,

    /// Frequency, in kHz, of the timer used by device-side `clock*`
    /// instructions.
    ///
    /// Introduced in HIP.
    pub clockInstructionRate: ::core::ffi::c_int,

    /// Architectural feature flags.
    ///
    /// Introduced in HIP.
    pub arch: hipDeviceArch_t,

    /// Whether the device can potentially execute multiple kernels
    /// concurrently.
    pub concurrentKernels: ::core::ffi::c_int,

    /// PCI domain ID.
    pub pciDomainID: ::core::ffi::c_int,

    /// PCI bus ID.
    pub pciBusID: ::core::ffi::c_int,

    /// PCI device ID.
    pub pciDeviceID: ::core::ffi::c_int,

    /// Maximum shared memory per multiprocessor, in bytes.
    pub maxSharedMemoryPerMultiProcessor: usize,

    /// Whether the device is on a multi-GPU board (1 if true, 0 otherwise).
    pub isMultiGpuBoard: ::core::ffi::c_int,

    /// Whether HIP can map host memory.
    pub canMapHostMemory: ::core::ffi::c_int,

    /// Deprecated: use `gcnArchName` instead.
    pub gcnArch: ::core::ffi::c_int,

    /// AMD GCN architecture name.
    pub gcnArchName: [::core::ffi::c_char; 256],

    /// Whether the device is an APU (1) or a discrete GPU (0).
    pub integrated: ::core::ffi::c_int,

    /// Whether the device supports cooperative launch.
    pub cooperativeLaunch: ::core::ffi::c_int,

    /// Whether the device supports cooperative launch across multiple
    /// devices.
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,

    /// Maximum size of 1D textures bound to linear memory.
    pub maxTexture1DLinear: ::core::ffi::c_int,

    /// Maximum number of elements in 1D images.
    pub maxTexture1D: ::core::ffi::c_int,

    /// Maximum dimensions (width, height) of 2D images, in image elements.
    pub maxTexture2D: [::core::ffi::c_int; 2],

    /// Maximum dimensions (width, height, depth) of 3D images,
    /// in image elements.
    pub maxTexture3D: [::core::ffi::c_int; 3],

    /// Address of the `HDP_MEM_COHERENCY_FLUSH_CNTL` register.
    pub hdpMemFlushCntl: *mut ::core::ffi::c_uint,

    /// Address of the `HDP_REG_COHERENCY_FLUSH_CNTL` register.
    pub hdpRegFlushCntl: *mut ::core::ffi::c_uint,

    /// Maximum pitch, in bytes, allowed by memory copies.
    pub memPitch: usize,

    /// Texture alignment requirement.
    pub textureAlignment: usize,

    /// Pitch alignment requirement for texture references bound to pitched
    /// memory.
    pub texturePitchAlignment: usize,

    /// Whether kernel execution on the device has a runtime limit.
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,

    /// Whether error-correcting code (ECC) is enabled on the device.
    pub ECCEnabled: ::core::ffi::c_int,

    /// Whether the device is a Tesla device using the TCC driver (1 if true,
    /// 0 otherwise).
    pub tccDriver: ::core::ffi::c_int,

    /// Whether cooperative multi-device launch supports functions that
    /// do not match across devices.
    pub cooperativeMultiDeviceUnmatchedFunc: ::core::ffi::c_int,

    /// Whether cooperative multi-device launch supports grid dimensions
    /// that do not match across devices.
    pub cooperativeMultiDeviceUnmatchedGridDim: ::core::ffi::c_int,

    /// Whether cooperative multi-device launch supports block dimensions
    /// that do not match across devices.
    pub cooperativeMultiDeviceUnmatchedBlockDim: ::core::ffi::c_int,

    /// Whether cooperative multi-device launch supports shared memory sizes
    /// that do not match across devices.
    pub cooperativeMultiDeviceUnmatchedSharedMem: ::core::ffi::c_int,

    /// Whether the device has a large PCI BAR (1 if true, 0 otherwise).
    pub isLargeBar: ::core::ffi::c_int,

    /// GPU ASIC revision.
    pub asicRevision: ::core::ffi::c_int,

    /// Whether the device supports managed memory allocation on this system.
    pub managedMemory: ::core::ffi::c_int,

    /// Whether the host can directly access device managed memory without
    /// migration.
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,

    /// Whether the device can coherently access managed memory concurrently
    /// with the CPU.
    pub concurrentManagedAccess: ::core::ffi::c_int,

    /// Whether the device supports coherent access to pageable memory
    /// without registering it with `hipHostRegister`.
    pub pageableMemoryAccess: ::core::ffi::c_int,

    /// Whether the device accesses pageable memory through the host's
    /// page tables.
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
}

#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipDeviceProp_tR0600 {
    /// Name of the device.
    pub name: [::core::ffi::c_char; 256],

    /// UUID identifying the device.
    pub uuid: hipUUID,

    /// 8-byte locally unique identifier (LUID). Only valid on Windows.
    pub luid: crate::platform::windows::LUID,

    /// LUID device node mask.
    pub luidDeviceNodeMask: ::core::ffi::c_uint,

    /// Total size of global memory, in bytes.
    pub totalGlobalMem: usize,

    /// Size of shared memory available per block, in bytes.
    pub sharedMemPerBlock: usize,

    /// Number of registers available per block.
    pub regsPerBlock: ::core::ffi::c_int,

    /// Warp size.
    pub warpSize: ::core::ffi::c_int,

    /// Maximum memory pitch, in bytes, supported by pitched memory copies.
    pub memPitch: usize,

    /// Maximum number of threads per block.
    pub maxThreadsPerBlock: ::core::ffi::c_int,

    /// Maximum number of threads in each block dimension (X, Y, Z).
    pub maxThreadsDim: [::core::ffi::c_int; 3],

    /// Maximum grid dimensions (X, Y, Z).
    pub maxGridSize: [::core::ffi::c_int; 3],

    /// Maximum multiprocessor clock frequency, in kHz.
    pub clockRate: ::core::ffi::c_int,

    /// Size of the device's constant memory region, in bytes.
    pub totalConstMem: usize,

    /// Major GPU architecture or compute capability version.
    ///
    /// For example, a value of 11 corresponds to Navi III (RDNA3).
    /// Refer to architecture feature flags for portable capability queries.
    pub major: ::core::ffi::c_int,

    /// Minor GPU architecture or compute capability version.
    ///
    /// Identifies a specific configuration or feature variation within
    /// the major architecture version. Different GPU models may have
    /// different feature support and optimization capabilities.
    pub minor: ::core::ffi::c_int,

    /// Required texture alignment, in bytes.
    pub textureAlignment: usize,

    /// Required pitch alignment for texture references bound to memory.
    pub texturePitchAlignment: usize,

    /// Deprecated. Use `asyncEngineCount` instead.
    pub deviceOverlap: ::core::ffi::c_int,

    /// Number of multiprocessors on the device.
    ///
    /// In Compute Unit (CU) mode, this equals the number of CUs.
    /// In Workgroup Processor (WGP) mode, this equals half the number
    /// of CUs because each WGP contains two CUs.
    pub multiProcessorCount: ::core::ffi::c_int,

    /// Indicates whether kernel execution on the device has a time limit.
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,

    /// Indicates whether the device is an APU (integrated GPU) or a dGPU.
    pub integrated: ::core::ffi::c_int,

    /// Indicates whether HIP can map host memory.
    pub canMapHostMemory: ::core::ffi::c_int,

    /// Device compute mode.
    pub computeMode: ::core::ffi::c_int,

    /// Maximum number of elements in a 1D texture.
    pub maxTexture1D: ::core::ffi::c_int,

    /// Maximum size of a 1D mipmapped texture.
    pub maxTexture1DMipmap: ::core::ffi::c_int,

    /// Maximum size of a 1D texture bound to linear memory.
    pub maxTexture1DLinear: ::core::ffi::c_int,

    /// Maximum dimensions (width, height) of a 2D texture.
    pub maxTexture2D: [::core::ffi::c_int; 2],

    /// Maximum dimensions of a 2D mipmapped texture.
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2],

    /// Maximum dimensions of a 2D texture bound to pitched memory.
    pub maxTexture2DLinear: [::core::ffi::c_int; 3],

    /// Maximum dimensions of a 2D texture when using gather operations.
    pub maxTexture2DGather: [::core::ffi::c_int; 2],

    /// Maximum dimensions (width, height, depth) of a 3D texture.
    pub maxTexture3D: [::core::ffi::c_int; 3],

    /// Maximum alternate dimensions of a 3D texture.
    pub maxTexture3DAlt: [::core::ffi::c_int; 3],

    /// Maximum dimensions of a cubemap texture.
    pub maxTextureCubemap: ::core::ffi::c_int,

    /// Maximum dimensions of a layered 1D texture.
    pub maxTexture1DLayered: [::core::ffi::c_int; 2],

    /// Maximum dimensions of a layered 2D texture.
    pub maxTexture2DLayered: [::core::ffi::c_int; 3],

    /// Maximum dimensions of a layered cubemap texture.
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2],

    /// Maximum size of a 1D surface.
    pub maxSurface1D: ::core::ffi::c_int,

    /// Maximum dimensions of a 2D surface.
    pub maxSurface2D: [::core::ffi::c_int; 2],

    /// Maximum dimensions of a 3D surface.
    pub maxSurface3D: [::core::ffi::c_int; 3],

    /// Maximum dimensions of a layered 1D surface.
    pub maxSurface1DLayered: [::core::ffi::c_int; 2],

    /// Maximum dimensions of a layered 2D surface.
    pub maxSurface2DLayered: [::core::ffi::c_int; 3],

    /// Maximum size of a cubemap surface.
    pub maxSurfaceCubemap: ::core::ffi::c_int,

    /// Maximum dimensions of a layered cubemap surface.
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2],

    /// Required alignment for surface memory, in bytes.
    pub surfaceAlignment: usize,

    /// Indicates whether the device can execute multiple kernels concurrently.
    pub concurrentKernels: ::core::ffi::c_int,

    /// Indicates whether error-correcting code (ECC) is enabled.
    pub ECCEnabled: ::core::ffi::c_int,

    /// PCI bus identifier.
    pub pciBusID: ::core::ffi::c_int,

    /// PCI device identifier.
    pub pciDeviceID: ::core::ffi::c_int,

    /// PCI domain identifier.
    pub pciDomainID: ::core::ffi::c_int,

    /// Indicates whether the device is a Tesla device using the TCC driver.
    pub tccDriver: ::core::ffi::c_int,

    /// Number of asynchronous engines available on the device.
    pub asyncEngineCount: ::core::ffi::c_int,

    /// Indicates whether the device and host share a unified address space.
    pub unifiedAddressing: ::core::ffi::c_int,

    /// Maximum global memory clock frequency, in kHz.
    pub memoryClockRate: ::core::ffi::c_int,

    /// Global memory bus width, in bits.
    pub memoryBusWidth: ::core::ffi::c_int,

    /// L2 cache size, in bytes.
    pub l2CacheSize: ::core::ffi::c_int,

    /// Maximum L2 cache size available for persisting accesses, in bytes.
    ///
    /// Zero indicates that L2 cache persistence is unsupported.
    /// Check this value before configuring `hipLimitPersistingL2CacheSize`
    /// through `hipDeviceSetLimit`.
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,

    /// Maximum number of resident threads per multiprocessor.
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,

    /// Indicates whether stream priorities are supported.
    pub streamPrioritiesSupported: ::core::ffi::c_int,

    /// Indicates whether global memory accesses are cached in L1.
    pub globalL1CacheSupported: ::core::ffi::c_int,

    /// Indicates whether local memory accesses are cached in L1.
    pub localL1CacheSupported: ::core::ffi::c_int,

    /// Amount of shared memory available per multiprocessor, in bytes.
    pub sharedMemPerMultiprocessor: usize,

    /// Number of registers available per multiprocessor.
    pub regsPerMultiprocessor: ::core::ffi::c_int,

    /// Indicates whether managed memory allocation is supported.
    pub managedMemory: ::core::ffi::c_int,

    /// Indicates whether the device belongs to a multi-GPU board.
    pub isMultiGpuBoard: ::core::ffi::c_int,

    /// Identifier shared by devices belonging to the same multi-GPU board.
    pub multiGpuBoardGroupID: ::core::ffi::c_int,

    /// Indicates whether native atomic operations are supported
    /// between the host and device.
    pub hostNativeAtomicSupported: ::core::ffi::c_int,

    /// Deprecated. CUDA-specific field.
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,

    /// Indicates whether the device can coherently access pageable host memory
    /// without registering it through `hipHostRegister`.
    pub pageableMemoryAccess: ::core::ffi::c_int,

    /// Indicates whether the device can coherently access managed memory
    /// concurrently with the CPU.
    pub concurrentManagedAccess: ::core::ffi::c_int,

    /// Indicates whether compute preemption is supported.
    pub computePreemptionSupported: ::core::ffi::c_int,

    /// Indicates whether the device can access registered host memory
    /// using the same virtual address as the host.
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,

    /// Indicates whether cooperative kernel launches are supported.
    pub cooperativeLaunch: ::core::ffi::c_int,

    /// Indicates whether cooperative launches across multiple devices
    /// are supported.
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,

    /// Maximum shared memory per block when opt-in is enabled, in bytes.
    pub sharedMemPerBlockOptin: usize,

    /// Indicates whether pageable memory accesses use the host's page tables.
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,

    /// Indicates whether the host can directly access managed device memory
    /// without triggering migration.
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,

    /// Maximum number of blocks per multiprocessor.
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,

    /// Maximum supported access policy window size.
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,

    /// Amount of shared memory reserved by the driver per block, in bytes.
    pub reservedSharedMemPerBlock: usize,

    /// Indicates whether host memory registration through `hipHostRegister`
    /// is supported.
    pub hostRegisterSupported: ::core::ffi::c_int,

    /// Indicates whether sparse HIP arrays are supported.
    pub sparseHipArraySupported: ::core::ffi::c_int,

    /// Indicates whether read-only host memory registration
    /// through `hipHostRegisterReadOnly` is supported.
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,

    /// Indicates whether external timeline semaphore interoperability
    /// is supported.
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,

    /// Indicates whether asynchronous memory allocation and memory pool APIs
    /// (`hipMallocAsync` and `hipMemPool`) are supported.
    pub memoryPoolsSupported: ::core::ffi::c_int,

    /// Indicates whether GPU Direct RDMA operations are supported.
    pub gpuDirectRDMASupported: ::core::ffi::c_int,

    /// Options bitmask controlling GPU Direct RDMA write flushing.
    ///
    /// Interpreted according to `hipFlushGPUDirectRDMAWritesOptions`.
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,

    /// GPU Direct RDMA write ordering mode.
    ///
    /// Values are defined by `hipGPUDirectRDMAWritesOrdering`.
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,

    /// Bitmask of supported handle types for memory-pool-based IPC.
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,

    /// Indicates whether deferred mapping of HIP arrays
    /// and mipmapped HIP arrays is supported.
    pub deferredMappingHipArraySupported: ::core::ffi::c_int,

    /// Indicates whether inter-process communication (IPC) events are supported.
    pub ipcEventSupported: ::core::ffi::c_int,

    /// Indicates whether cluster launches are supported.
    pub clusterLaunch: ::core::ffi::c_int,

    /// Indicates whether unified function pointers are supported.
    pub unifiedFunctionPointers: ::core::ffi::c_int,

    /// Reserved fields for CUDA compatibility.
    pub reserved: [::core::ffi::c_int; 63],

    /// Reserved fields for future HIP/CUDA extensions.
    pub hipReserved: [::core::ffi::c_int; 32],

    // HIP-specific fields.
    /// Name of the AMD GPU architecture (GCN architecture name).
    pub gcnArchName: [::core::ffi::c_char; 256],

    /// Maximum shared memory available per multiprocessor (CU), in bytes.
    pub maxSharedMemoryPerMultiProcessor: usize,

    /// Frequency, in kHz, of the timer used by device-side `clock*` instructions.
    pub clockInstructionRate: ::core::ffi::c_int,

    /// GPU architectural feature flags.
    pub arch: hipDeviceArch_t,

    /// Pointer to the HDP memory coherency flush control register.
    pub hdpMemFlushCntl: *mut ::core::ffi::c_uint,

    /// Pointer to the HDP register coherency flush control register.
    pub hdpRegFlushCntl: *mut ::core::ffi::c_uint,

    /// Indicates whether cooperative multi-device launches support
    /// functions that do not match across devices.
    pub cooperativeMultiDeviceUnmatchedFunc: ::core::ffi::c_int,

    /// Indicates whether cooperative multi-device launches support
    /// different grid dimensions across devices.
    pub cooperativeMultiDeviceUnmatchedGridDim: ::core::ffi::c_int,

    /// Indicates whether cooperative multi-device launches support
    /// different block dimensions across devices.
    pub cooperativeMultiDeviceUnmatchedBlockDim: ::core::ffi::c_int,

    /// Indicates whether cooperative multi-device launches support
    /// different shared memory configurations across devices.
    pub cooperativeMultiDeviceUnmatchedSharedMem: ::core::ffi::c_int,

    /// Indicates whether the device has a large PCI Base Address Register (BAR).
    pub isLargeBar: ::core::ffi::c_int,

    /// GPU ASIC revision identifier.
    pub asicRevision: ::core::ffi::c_int,
}
