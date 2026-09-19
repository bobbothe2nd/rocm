use core::{
    ffi::{CStr, c_int, c_uint},
    mem::transmute,
};

#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};

use rocm_sys::hip::{
    hipDevice_t, hipDeviceAttribute_t, hipDeviceGetAttribute, hipDeviceGetDefaultMemPool,
    hipDeviceGetLimit, hipDeviceGetMemPool, hipDeviceProp_tR0600, hipDeviceReset,
    hipDeviceSetLimit, hipDeviceSynchronize, hipGetDevice, hipGetDeviceCount, hipGetDeviceFlags,
    hipGetDevicePropertiesR0600, hipLimit_t, hipSetDevice, hipSetDeviceFlags,
};

use crate::{
    hip::{HipError, memory::MemPool},
    shared::GfxVersion,
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    dev: hipDevice_t,
}

impl Device {
    /// Recreates the current device
    pub fn current() -> Result<Self, HipError> {
        unsafe { wrap_sys_res!(|dev| hipGetDevice((&raw mut dev).cast())) }
    }

    /// Creates a device from the ID
    pub fn create(dev: i32) -> Result<Self, HipError> {
        let dev = dev as c_int;
        let count = Self::get_dev_count()? as c_int;

        if dev >= count || dev.is_negative() {
            return Err(HipError::InvalidDevice);
        }

        Ok(Self { dev })
    }

    /// Converts the device into the raw one used by `rocm-sys`.
    #[inline(always)]
    pub fn into_raw(self) -> hipDevice_t {
        self.dev
    }

    /// Constructs a safe device from the raw one used by `rocm-sys`.
    #[inline(always)]
    pub unsafe fn from_raw(dev: hipDevice_t) -> Self {
        Self { dev }
    }

    /// Sets the current default device
    pub fn set_default(self) -> Result<(), HipError> {
        unsafe { try_err!(hipSetDevice(self.dev), Ok(())) }
    }

    /// Gets the specified attribute of this device as a signed integer
    pub fn get_attr(self, attr: DeviceAttr) -> Result<i32, HipError> {
        unsafe {
            let attr = transmute::<DeviceAttr, hipDeviceAttribute_t>(attr);
            wrap_sys_res!(|val| hipDeviceGetAttribute((&raw mut val).cast(), attr, self.dev))
        }
    }

    /// Gets default memory pool for the device
    pub fn get_default_mem_pool(self) -> Result<MemPool, HipError> {
        unsafe {
            wrap_sys_res!(|mem_pool| hipDeviceGetDefaultMemPool(
                (&raw mut mem_pool).cast(),
                self.dev
            ))
        }
    }

    /// Gets active memory pool for the device
    pub fn get_mem_pool(self) -> Result<MemPool, HipError> {
        unsafe {
            wrap_sys_res!(|mem_pool| hipDeviceGetMemPool(self.dev, (&raw mut mem_pool).cast()))
        }
    }

    /// Gets properties of the device
    pub fn properties(self) -> Result<DeviceProp, HipError> {
        let mut prop = core::mem::MaybeUninit::<DeviceProp>::uninit();

        let err = unsafe { hipGetDevicePropertiesR0600(prop.as_mut_ptr(), self.dev) };

        match err {
            rocm_sys::hip::hipError_t::hipSuccess => Ok(unsafe { prop.assume_init() }),
            err => Err(HipError::from(err)),
        }
    }

    /// Gets the GFX/GCN architecture name.
    #[cfg(feature = "alloc")]
    pub fn gfx_arch(&self) -> Result<String, HipError> {
        let props = self.properties()?;
        let cstr = unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) };
        let raw = cstr.to_string_lossy();
        Ok(raw.split(':').next().unwrap_or(&raw).to_string())
    }

    /// Gets the GFX/GCN architecture version.
    #[cfg(feature = "alloc")]
    pub fn gfx_version(&self) -> Result<GfxVersion, HipError> {
        let props = self.properties()?;
        let cstr = unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) };
        let raw = cstr.to_string_lossy();
        let arch = raw.split(':').next().unwrap_or(&raw);
        arch.parse().map_err(|_| HipError::InvalidValue )
    }

    /// Gets a count of available ROCm devices.
    pub fn get_dev_count() -> Result<i32, HipError> {
        unsafe { wrap_sys_res!(|count| hipGetDeviceCount((&raw mut count).cast())) }
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
        wrap_sys_res!(|val| hipDeviceGetLimit((&raw mut val).cast(), limit))
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
    unsafe { wrap_sys_res!(|flags| hipGetDeviceFlags((&raw mut flags).cast())) }
        .map(|flags: c_uint| flags as u32)
}

/// Sets flags set for current device
pub fn set_flags(flags: u32) -> Result<(), HipError> {
    unsafe { try_err!(hipSetDeviceFlags(flags as c_uint), Ok(())) }
}

pub type DeviceProp = hipDeviceProp_tR0600;

#[repr(u32)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceAttr {
    CudaCompatibleBegin = 0,
    AccessPolicyMaxWindowSize = 1,
    AsyncEngineCount = 2,
    CanMapHostMemory = 3,
    CanUseHostPointerForRegisteredMem = 4,
    ClockRate = 5,
    ComputeMode = 6,
    ComputePreemptionSupported = 7,
    ConcurrentKernels = 8,
    ConcurrentManagedAccess = 9,
    CooperativeLaunch = 10,
    CooperativeMultiDeviceLaunch = 11,
    DeviceOverlap = 12,
    DirectManagedMemAccessFromHost = 13,
    GlobalL1CacheSupported = 14,
    HostNativeAtomicSupported = 15,
    Integrated = 16,
    IsMultiGpuBoard = 17,
    KernelExecTimeout = 18,
    L2CacheSize = 19,
    LocalL1CacheSupported = 20,
    Luid = 21,
    LuidDeviceNodeMask = 22,
    ComputeCapabilityMajor = 23,
    ManagedMemory = 24,
    MaxBlocksPerMultiProcessor = 25,
    MaxBlockDimX = 26,
    MaxBlockDimY = 27,
    MaxBlockDimZ = 28,
    MaxGridDimX = 29,
    MaxGridDimY = 30,
    MaxGridDimZ = 31,
    MaxSurface1D = 32,
    MaxSurface1DLayered = 33,
    MaxSurface2D = 34,
    MaxSurface2DLayered = 35,
    MaxSurface3D = 36,
    MaxSurfaceCubemap = 37,
    MaxSurfaceCubemapLayered = 38,
    MaxTexture1DWidth = 39,
    MaxTexture1DLayered = 40,
    MaxTexture1DLinear = 41,
    MaxTexture1DMipmap = 42,
    MaxTexture2DWidth = 43,
    MaxTexture2DHeight = 44,
    MaxTexture2DGather = 45,
    MaxTexture2DLayered = 46,
    MaxTexture2DLinear = 47,
    MaxTexture2DMipmap = 48,
    MaxTexture3DWidth = 49,
    MaxTexture3DHeight = 50,
    MaxTexture3DDepth = 51,
    MaxTexture3DAlt = 52,
    MaxTextureCubemap = 53,
    MaxTextureCubemapLayered = 54,
    MaxThreadsDim = 55,
    MaxThreadsPerBlock = 56,
    MaxThreadsPerMultiProcessor = 57,
    MaxPitch = 58,
    MemoryBusWidth = 59,
    MemoryClockRate = 60,
    ComputeCapabilityMinor = 61,
    MultiGpuBoardGroupID = 62,
    MultiprocessorCount = 63,
    Unused1 = 64,
    PageableMemoryAccess = 65,
    PageableMemoryAccessUsesHostPageTables = 66,
    PciBusId = 67,
    PciDeviceId = 68,
    PciDomainId = 69,
    PersistingL2CacheMaxSize = 70,
    MaxRegistersPerBlock = 71,
    MaxRegistersPerMultiprocessor = 72,
    ReservedSharedMemPerBlock = 73,
    MaxSharedMemoryPerBlock = 74,
    SharedMemPerBlockOptin = 75,
    SharedMemPerMultiprocessor = 76,
    SingleToDoublePrecisionPerfRatio = 77,
    StreamPrioritiesSupported = 78,
    SurfaceAlignment = 79,
    TccDriver = 80,
    TextureAlignment = 81,
    TexturePitchAlignment = 82,
    TotalConstantMemory = 83,
    TotalGlobalMem = 84,
    UnifiedAddressing = 85,
    Unused2 = 86,
    WarpSize = 87,
    MemoryPoolsSupported = 88,
    VirtualMemoryManagementSupported = 89,
    HostRegisterSupported = 90,
    MemoryPoolSupportedHandleTypes = 91,
    HostNumaId = 92,
    GPUDirectRDMAWithHipVMMSupported = 94,
    HandleTypeFabricSupported = 95,
    CudaCompatibleEnd = 9999,
    AmdSpecificBegin = 10000,
    Unused3 = 10001,
    MaxSharedMemoryPerMultiprocessor = 10002,
    Unused4 = 10003,
    Unused5 = 10004,
    HdpMemFlushCntl = 10005,
    HdpRegFlushCntl = 10006,
    CooperativeMultiDeviceUnmatchedFunc = 10007,
    CooperativeMultiDeviceUnmatchedGridDim = 10008,
    CooperativeMultiDeviceUnmatchedBlockDim = 10009,
    CooperativeMultiDeviceUnmatchedSharedMem = 10010,
    IsLargeBar = 10011,
    AsicRevision = 10012,
    CanUseStreamWaitValue = 10013,
    ImageSupport = 10014,
    PhysicalMultiProcessorCount = 10015,
    FineGrainSupport = 10016,
    WallClockRate = 10017,
    NumberOfXccs = 10018,
    MaxAvailableVgprsPerThread = 10019,
    PciChipId = 10020,
    ExpertSchedMode = 10021,
    AmdSpecificEnd = 19999,
    VendorSpecificBegin = 20000,
}
