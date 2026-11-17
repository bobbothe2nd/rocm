use core::{
    ffi::{CStr, c_int, c_uint},
    mem::transmute,
};

use alloc::string::{String, ToString};

use rocm_sys::hip::{
    hipDevice_t, hipDeviceAttribute_t, hipDeviceGetAttribute, hipDeviceGetDefaultMemPool,
    hipDeviceGetLimit, hipDeviceGetMemPool, hipDeviceProp_t, hipDeviceReset, hipDeviceSetLimit,
    hipDeviceSynchronize, hipGetDevice, hipGetDeviceCount, hipGetDeviceFlags,
    hipGetDevicePropertiesR0600, hipLimit_t, hipSetDevice, hipSetDeviceFlags,
};

use crate::{
    aux::RocHandle,
    hip::{HipError, gfx::GfxVersion, memory::MemPool},
};

#[repr(C)]
#[cfg_attr(feature = "stable_auto", stable_auto::auto_opaque)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    dev: hipDevice_t,
    pub(crate) has_mem_pools: bool,
    pub(crate) concurrent_managed_access: bool,
    pub(crate) has_managed_memory: bool,
}

impl Default for Device {
    fn default() -> Self {
        Self::current().unwrap()
    }
}

impl Device {
    /// Recreates the current device
    pub fn current() -> Result<Self, HipError> {
        let raw: Result<_, HipError> =
            unsafe { wrap_sys_res!(|dev: hipDevice_t| hipGetDevice(dev.as_mut_ptr().cast())) };
        Self::create(raw?)
    }

    /// Creates a device from the ID
    pub fn create(dev: i32) -> Result<Self, HipError> {
        let dev = dev as c_int;
        let count = Self::get_dev_count()? as c_int;

        if dev >= count || dev.is_negative() {
            return Err(HipError::InvalidDevice);
        }

        unsafe { Self::from_raw(dev) }
    }

    /// Converts the device into the raw one used by `rocm-sys`.
    #[inline(always)]
    pub fn into_raw(self) -> hipDevice_t {
        self.dev
    }

    /// Converts the device into the raw one used by `rocm-sys`.
    #[inline(always)]
    pub fn as_raw(self) -> hipDevice_t {
        self.dev
    }

    /// Constructs a safe device from the raw one used by `rocm-sys`.
    #[inline(always)]
    pub unsafe fn from_raw(dev: hipDevice_t) -> Result<Self, HipError> {
        fn has_attr(attr: hipDeviceAttribute_t, dev: hipDevice_t) -> Result<bool, HipError> {
            unsafe { wrap_sys_res!(|val| hipDeviceGetAttribute(val.as_mut_ptr(), attr, dev,)) }
                .map(|x| x > 0)
        }

        let has_mem_pools = has_attr(
            hipDeviceAttribute_t::hipDeviceAttributeMemoryPoolsSupported,
            dev,
        )?;
        let concurrent_managed_access = has_attr(
            hipDeviceAttribute_t::hipDeviceAttributeConcurrentManagedAccess,
            dev,
        )?;
        let has_managed_memory =
            has_attr(hipDeviceAttribute_t::hipDeviceAttributeManagedMemory, dev)?;

        Ok(Self {
            dev,
            has_mem_pools,
            concurrent_managed_access,
            has_managed_memory,
        })
    }

    /// Sets the current default device
    pub fn set_default(self) -> Result<(), HipError> {
        unsafe { try_err!(hipSetDevice(self.dev), Ok(())) }
    }

    /// Gets the specified attribute of this device as a signed integer
    pub fn get_attr(self, attr: DeviceAttr) -> Result<i32, HipError> {
        unsafe {
            let attr = transmute::<DeviceAttr, hipDeviceAttribute_t>(attr);
            wrap_sys_res!(|val| hipDeviceGetAttribute(val.as_mut_ptr(), attr, self.dev))
        }
    }

    /// Gets default memory pool for the device
    pub fn get_default_mem_pool(self) -> Result<MemPool, HipError> {
        unsafe {
            wrap_sys_res!(|mem_pool: MemPool| hipDeviceGetDefaultMemPool(
                mem_pool.as_mut_ptr().cast(),
                self.dev
            ))
        }
    }

    /// Gets active memory pool for the device
    pub fn get_mem_pool(self) -> Result<MemPool, HipError> {
        unsafe {
            wrap_sys_res!(|mem_pool: MemPool| hipDeviceGetMemPool(
                self.dev,
                mem_pool.as_mut_ptr().cast()
            ))
        }
    }

    /// Gets properties of the device
    pub fn properties(self) -> Result<DeviceProp, HipError> {
        unsafe { wrap_sys_res!(|prop| hipGetDevicePropertiesR0600(prop.as_mut_ptr(), self.dev)) }
    }

    /// Gets the GFX/GCN architecture name.
    pub fn gfx_arch(&self) -> Result<String, HipError> {
        let props = self.properties()?;
        let cstr = unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) };
        let raw = cstr.to_string_lossy();
        Ok(raw.split(':').next().unwrap_or(&raw).to_string())
    }

    /// Gets the GFX/GCN architecture version.
    pub fn gfx_version(&self) -> Result<GfxVersion, HipError> {
        let props = self.properties()?;
        let cstr = unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) };
        let arch = cstr.to_string_lossy();
        arch.parse().map_err(|_| HipError::InvalidValue)
    }

    /// Gets a count of available ROCm devices.
    pub fn get_dev_count() -> Result<i32, HipError> {
        unsafe { wrap_sys_res!(|count| hipGetDeviceCount(count.as_mut_ptr())) }
            .map(|count: c_int| count as i32)
    }

    pub fn sync(self) -> Result<(), HipError> {
        self.set_default()?;
        sync()
    }

    pub fn reset(self) -> Result<(), HipError> {
        self.set_default()?;
        reset()
    }

    pub fn get_limit(self, limit: DeviceLimit) -> Result<usize, HipError> {
        self.set_default()?;
        get_limit(limit)
    }

    pub fn set_limit(self, limit: DeviceLimit, val: usize) -> Result<(), HipError> {
        self.set_default()?;
        set_limit(limit, val)
    }

    pub fn get_flags(self) -> Result<u32, HipError> {
        self.set_default()?;
        get_flags()
    }

    pub fn set_flags(self, val: u32) -> Result<(), HipError> {
        self.set_default()?;
        set_flags(val)
    }
}

impl RocHandle for Device {
    type Error = HipError;
}

/// Synchronizes the current default device
///
/// To set the current default device, use [`Device::set_default`].
pub fn sync() -> Result<(), HipError> {
    unsafe { try_err!(hipDeviceSynchronize(), Ok(())) }
}

/// Resets the current default device
///
/// To set the current default device, use [`Device::set_default`].
pub fn reset() -> Result<(), HipError> {
    unsafe { try_err!(hipDeviceReset(), Ok(())) }
}

/// Queries the specified limit of the current device as an unsigned integer
pub fn get_limit(limit: DeviceLimit) -> Result<usize, HipError> {
    unsafe {
        let limit = transmute::<DeviceLimit, hipLimit_t>(limit);
        wrap_sys_res!(|val| hipDeviceGetLimit(val.as_mut_ptr(), limit))
    }
}

/// Sets the specified limit of the current device to a new value
pub fn set_limit(limit: DeviceLimit, val: usize) -> Result<(), HipError> {
    unsafe {
        let limit = transmute::<DeviceLimit, hipLimit_t>(limit);
        try_err!(hipDeviceSetLimit(limit, val), Ok(()))
    }
}

/// Gets flags set for current device
pub fn get_flags() -> Result<u32, HipError> {
    unsafe { wrap_sys_res!(|flags| hipGetDeviceFlags(flags.as_mut_ptr())) }
        .map(|flags: c_uint| flags as u32)
}

/// Sets flags set for current device
pub fn set_flags(flags: u32) -> Result<(), HipError> {
    unsafe { try_err!(hipSetDeviceFlags(flags as c_uint), Ok(())) }
}

pub type DeviceProp = hipDeviceProp_t;

#[repr(u32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceLimit {
    StackSize = 0,
    PrintfFifoSize = 1,
    MallocHeapSize = 2,
    ScratchMin = 4096,
    ScratchMax = 4097,
    ScratchCurrent = 4098,
    Range = 4099,
}

#[repr(u32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceAttr {
    CudaCompatibleBegin = 0,
    /// Cuda only. The maximum size of the window
    /// policy in bytes.
    AccessPolicyMaxWindowSize = 1,
    /// Asynchronous engines number.
    AsyncEngineCount = 2,
    /// Whether host memory can be mapped into device address
    /// space
    CanMapHostMemory = 3,
    /// Device can access host registered
    /// memory at the same virtual address as
    /// the CPU
    CanUseHostPointerForRegisteredMem = 4,
    /// Peak clock frequency in kilohertz.
    ClockRate = 5,
    /// Compute mode that device is currently in.
    ComputeMode = 6,
    /// Device supports Compute Preemption.
    ComputePreemptionSupported = 7,
    /// Device can possibly execute multiple kernels
    /// concurrently.
    ConcurrentKernels = 8,
    /// Device can coherently access managed memory
    /// concurrently with the CPU
    ConcurrentManagedAccess = 9,
    /// Support cooperative launch
    CooperativeLaunch = 10,
    /// Support cooperative launch on multiple
    /// devices
    CooperativeMultiDeviceLaunch = 11,
    /// Device can concurrently copy memory and execute a kernel.
    /// Deprecated. Use instead asyncEngineCount.
    DeviceOverlap = 12,
    /// Host can directly access managed memory on
    /// the device without migration
    DirectManagedMemAccessFromHost = 13,
    /// Device supports caching globals in L1
    GlobalL1CacheSupported = 14,
    /// Link between the device and the host supports
    /// native atomic operations
    HostNativeAtomicSupported = 15,
    /// Device is integrated GPU
    Integrated = 16,
    /// Multiple GPU devices.
    IsMultiGpuBoard = 17,
    /// Run time limit for kernels executed on the device
    KernelExecTimeout = 18,
    /// Size of L2 cache in bytes. 0 if the device doesn't have L2
    /// cache.
    L2CacheSize = 19,
    /// caching locals in L1 is supported
    LocalL1CacheSupported = 20,
    /// 8-byte locally unique identifier in 8 bytes. Undefined on TCC and
    /// non-Windows platforms
    #[cfg(windows)]
    Luid = 21,
    /// Luid device node mask. Undefined on TCC and
    /// non-Windows platforms
    #[cfg(windows)]
    LuidDeviceNodeMask = 22,
    /// Major compute capability version number.
    ComputeCapabilityMajor = 23,
    /// Device supports allocating managed memory on this system
    ManagedMemory = 24,
    /// Max block size per multiprocessor
    MaxBlocksPerMultiProcessor = 25,
    /// Max block size in width.
    MaxBlockDimX = 26,
    /// Max block size in height.
    MaxBlockDimY = 27,
    /// Max block size in depth.
    MaxBlockDimZ = 28,
    /// Max grid size  in width.
    MaxGridDimX = 29,
    /// Max grid size  in height.
    MaxGridDimY = 30,
    /// Max grid size  in depth.
    MaxGridDimZ = 31,
    /// Maximum size of 1D surface.
    MaxSurface1D = 32,
    /// Cuda only. Maximum dimensions of 1D layered surface.
    MaxSurface1DLayered = 33,
    /// Maximum dimension (width, height) of 2D surface.
    MaxSurface2D = 34,
    /// Cuda only. Maximum dimensions of 2D layered surface.
    MaxSurface2DLayered = 35,
    /// Maximum dimension (width, height, depth) of 3D surface.
    MaxSurface3D = 36,
    /// Cuda only. Maximum dimensions of Cubemap surface.
    MaxSurfaceCubemap = 37,
    /// Cuda only. Maximum dimension of Cubemap layered
    /// surface.
    MaxSurfaceCubemapLayered = 38,
    /// Maximum size of 1D texture.
    MaxTexture1DWidth = 39,
    /// Maximum dimensions of 1D layered texture.
    MaxTexture1DLayered = 40,
    /// Maximum number of elements allocatable in a 1D linear
    /// texture. Use cudaDeviceGetTexture1DLinearMaxWidth()
    /// instead on Cuda.
    MaxTexture1DLinear = 41,
    /// Maximum size of 1D mipmapped texture.
    MaxTexture1DMipmap = 42,
    /// Maximum dimension width of 2D texture.
    MaxTexture2DWidth = 43,
    /// Maximum dimension hight of 2D texture.
    MaxTexture2DHeight = 44,
    /// Maximum dimensions of 2D texture if gather operations
    /// performed.
    MaxTexture2DGather = 45,
    /// Maximum dimensions of 2D layered texture.
    MaxTexture2DLayered = 46,
    /// Maximum dimensions (width, height, pitch) of 2D
    /// textures bound to pitched memory.
    MaxTexture2DLinear = 47,
    /// Maximum dimensions of 2D mipmapped texture.
    MaxTexture2DMipmap = 48,
    /// Maximum dimension width of 3D texture.
    MaxTexture3DWidth = 49,
    /// Maximum dimension height of 3D texture.
    MaxTexture3DHeight = 50,
    /// Maximum dimension depth of 3D texture.
    MaxTexture3DDepth = 51,
    /// Maximum dimensions of alternate 3D texture.
    MaxTexture3DAlt = 52,
    /// Maximum dimensions of Cubemap texture
    MaxTextureCubemap = 53,
    /// Maximum dimensions of Cubemap layered texture.
    MaxTextureCubemapLayered = 54,
    /// Maximum dimension of a block
    MaxThreadsDim = 55,
    /// Maximum number of threads per block.
    MaxThreadsPerBlock = 56,
    /// Maximum resident threads per multiprocessor.
    MaxThreadsPerMultiProcessor = 57,
    /// Maximum pitch in bytes allowed by memory copies
    MaxPitch = 58,
    /// Global memory bus width in bits.
    MemoryBusWidth = 59,
    /// Peak memory clock frequency in kilohertz.
    MemoryClockRate = 60,
    /// Minor compute capability version number.
    ComputeCapabilityMinor = 61,
    /// Unique ID of device group on the same multi-GPU
    /// board
    MultiGpuBoardGroupID = 62,
    /// Number of multi-processors. When the GPU works in Compute
    /// Unit (CU) mode, this value equals the number of CUs;
    /// when in Workgroup Processor (WGP) mode, this value equels
    /// half of CUs, because a single WGP contains two CUs.
    MultiprocessorCount = 63,
    /// Previously hipDeviceAttributeName
    Unused1 = 64,
    /// Device supports coherently accessing pageable memory
    /// without calling hipHostRegister on it
    PageableMemoryAccess = 65,
    /// Device accesses pageable memory
    /// via the host's page tables
    PageableMemoryAccessUsesHostPageTables = 66,
    /// PCI Bus ID.
    PciBusId = 67,
    /// PCI Device ID. Returns pcie slot id
    PciDeviceId = 68,
    /// PCI Domain Id.
    PciDomainId = 69,
    /// Maximum l2 persisting lines capacity in bytes
    PersistingL2CacheMaxSize = 70,
    /// 32-bit registers available to a thread block. This
    /// number is shared by all thread blocks simultaneously
    /// resident on a multiprocessor.
    MaxRegistersPerBlock = 71,
    /// 32-bit registers available per block.
    MaxRegistersPerMultiprocessor = 72,
    /// Shared memory reserved by CUDA driver per
    /// block.
    ReservedSharedMemPerBlock = 73,
    /// Maximum shared memory available per block in
    /// bytes.
    MaxSharedMemoryPerBlock = 74,
    /// Maximum shared memory per block usable by special
    /// opt in.
    SharedMemPerBlockOptin = 75,
    /// Shared memory available per multiprocessor.
    SharedMemPerMultiprocessor = 76,
    /// Cuda only. Performance ratio of single
    /// precision to double precision.
    SingleToDoublePrecisionPerfRatio = 77,
    /// Whether to support stream priorities.
    StreamPrioritiesSupported = 78,
    /// Alignment requirement for surfaces
    SurfaceAlignment = 79,
    /// Cuda only. Whether device is a Tesla device using TCC driver
    TccDriver = 80,
    /// Alignment requirement for textures
    TextureAlignment = 81,
    /// Pitch alignment requirement for 2D texture
    /// references bound to pitched memory;
    TexturePitchAlignment = 82,
    /// Constant memory size in bytes.
    TotalConstantMemory = 83,
    /// Global memory available on devicice.
    TotalGlobalMem = 84,
    /// Cuda only. An unified address space shared with the
    /// host.
    UnifiedAddressing = 85,
    /// Previously hipDeviceAttributeUuid
    Unused2 = 86,
    /// Warp size in threads.
    WarpSize = 87,
    /// Device supports HIP Stream Ordered Memory Allocator
    MemoryPoolsSupported = 88,
    /// Device supports HIP virtual memory
    /// management
    VirtualMemoryManagementSupported = 89,
    /// Can device support host memory registration via
    /// hipHostRegister
    HostRegisterSupported = 90,
    /// Supported handle mask for HIP Stream
    /// Ordered Memory Allocator
    MemoryPoolSupportedHandleTypes = 91,
    /// NUMA ID of the cpu node closest to the device,
    /// or -1 when NUMA isn't supported
    HostNumaId = 92,
    /// Device supports DMABuf buffer sharing
    DmaBufSupported = 93,
    /// GPU Direct RDMA with HIP VMM is supported
    /// (requires DMA-Buf and HIP virtual memory
    /// management)
    GPUDirectRDMAWithHipVMMSupported = 94,
    /// Device supports exporting memory to a fabric handle
    HandleTypeFabricSupported = 95,
    /// Device supports host-allocated DMABuf buffer sharing
    HostAllocDmaBufSupported = 96,
    /// Device supports GPUDirect RDMA APIs
    GPUDirectRDMASupported = 97,
    /// Bitmask of
    /// hipFlushGPUDirectRDMAWritesOptions
    /// describing the flush paths the device
    /// supports
    GPUDirectRDMAFlushWritesOptions = 98,
    /// A hipGPUDirectRDMAWritesOrdering value
    /// giving the scope at which GPUDirect RDMA
    /// writes are naturally ordered, i.e. visible
    /// without an explicit flush
    GPUDirectRDMAWritesOrdering = 99,
    CudaCompatibleEnd = 9999,
    AmdSpecificBegin = 10000,
    /// Previously hipDeviceAttributeArch
    Unused3 = 10001,
    /// Maximum Shared Memory PerMultiprocessor.
    MaxSharedMemoryPerMultiprocessor = 10002,
    /// Previously hipDeviceAttributeGcnArch
    Unused4 = 10003,
    /// Previously hipDeviceAttributeGcnArchName
    Unused5 = 10004,
    /// Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    HdpMemFlushCntl = 10005,
    /// Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    HdpRegFlushCntl = 10006,
    /// Supports cooperative launch on
    /// multiple devices with unmatched
    /// functions
    CooperativeMultiDeviceUnmatchedFunc = 10007,
    /// Supports cooperative launch on
    /// multiple devices with unmatched
    /// grid dimensions
    CooperativeMultiDeviceUnmatchedGridDim = 10008,
    /// Supports cooperative launch on
    /// multiple devices with unmatched
    /// block dimensions
    CooperativeMultiDeviceUnmatchedBlockDim = 10009,
    /// Supports cooperative launch on
    /// multiple devices with unmatched
    /// shared memories
    CooperativeMultiDeviceUnmatchedSharedMem = 10010,
    /// Whether it is LargeBar
    IsLargeBar = 10011,
    /// Revision of the GPU in this device
    AsicRevision = 10012,
    /// '1' if Device supports `hipStreamWaitValue32()` and
    /// `hipStreamWaitValue64()`, '0' otherwise.
    CanUseStreamWaitValue = 10013,
    /// '1' if Device supports image, '0' otherwise.
    ImageSupport = 10014,
    /// All available physical compute
    /// units for the device
    PhysicalMultiProcessorCount = 10015,
    /// '1' if Device supports fine grain, '0' otherwise
    FineGrainSupport = 10016,
    /// Constant frequency of wall clock in kilohertz.
    WallClockRate = 10017,
    /// Number of XCC(s) in the device’s current
    /// compute partition(XCP).
    /// An XCC is AMD’s fundamental GPU compute tile.
    /// Devices like MI300X include multiple XCCs
    /// per package. Value varies by boot‑time
    /// partition mode(compute or memory).
    /// Partition modes can be set using amd‑smi.
    NumberOfXccs = 10018,
    /// Max number of available (directly or
    /// indirectly addressable) VGPRs per thread in
    /// DWORDs.
    MaxAvailableVgprsPerThread = 10019,
    /// GPU Manufacturer device id
    PciChipId = 10020,
    /// '1' if Device supports expert scheduling mode,
    /// '0' otherwise.
    ExpertSchedMode = 10021,
    /// Maximum number of dynamic data prefetch regions
    /// per kernel launch (0 if unsupported).
    MaxDynDataPrefetchRegions = 10022,
    AmdSpecificEnd = 19999,
    VendorSpecificBegin = 20000,
}
