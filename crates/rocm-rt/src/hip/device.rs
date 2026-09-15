use rocm_sys::hip::{
    hipDevice_t, hipDeviceAttribute_t, hipDeviceGetAttribute, hipDeviceGetDefaultMemPool,
    hipDeviceGetLimit, hipDeviceGetMemPool, hipDeviceProp_tR0600, hipDeviceReset,
    hipDeviceSetLimit, hipDeviceSynchronize, hipGetDevice, hipGetDeviceCount, hipGetDeviceFlags,
    hipLimit_t, hipMalloc, hipMemPool_t, hipSetDevice, hipSetDeviceFlags,
};

use crate::hip::HipError;

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
    pub fn new(dev: i32) -> Result<Self, HipError> {
        let count = get_dev_count()?;

        if dev >= count {
            return Err(HipError::HipErrorInvalidDevice);
        }

        Ok(Self { dev })
    }

    pub fn into_raw(self) -> hipDevice_t {
        self.dev
    }

    pub unsafe fn from_raw(dev: hipDevice_t) -> Self {
        Self { dev }
    }

    /// Sets the current default device
    pub fn set_default(self) -> Result<(), HipError> {
        unsafe {
            try_err!(hipSetDevice(self.dev));
        }

        Ok(())
    }

    /// Gets the specified attribute of this device as a signed integer
    pub fn get_attr(self, attr: DeviceAttr) -> Result<i32, HipError> {
        unsafe { wrap_sys_res!(|val| hipDeviceGetAttribute((&raw mut val).cast(), attr, self.dev)) }
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
    pub fn get_properties(self) -> Result<DeviceProp, HipError> {
        unsafe {
            wrap_sys_res!(|mem_pool| hipDeviceGetMemPool(self.dev, (&raw mut mem_pool).cast()))
        }
    }

    pub fn malloc(self, size: usize) -> Result<Buffer, HipError> {
        let ptr: Result<*mut u8, HipError> =
            unsafe { wrap_sys_res!(|ptr| hipMalloc((&raw mut ptr).cast(), size)) };

        Ok(Buffer { ptr: ptr?, size })
    }
}

pub type DeviceAttr = hipDeviceAttribute_t;
pub type DeviceProp = hipDeviceProp_tR0600;
pub type DeviceLimit = hipLimit_t;

#[derive(Debug, Clone, Copy, Hash)]
pub struct Buffer {
    ptr: *mut u8,
    size: usize,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemPool {
    mem_pool: hipMemPool_t,
}

/// Synchronizes the current default device
///
/// To set the current default device, use [`Device::set_default`].
pub fn sync() {
    unsafe {
        hipDeviceSynchronize();
    }
}

/// Resets the current default device
///
/// To set the current default device, use [`Device::set_default`].
pub fn reset() {
    unsafe {
        hipDeviceReset();
    }
}

/// Counts the available ROCm devices.
pub fn get_dev_count() -> Result<i32, HipError> {
    unsafe { wrap_sys_res!(|count| hipGetDeviceCount((&raw mut count).cast())) }
}

/// Queries the specified limit of the current device as an unsigned integer
pub fn get_limit(limit: DeviceLimit) -> Result<usize, HipError> {
    unsafe { wrap_sys_res!(|val| hipDeviceGetLimit((&raw mut val).cast(), limit)) }
}

/// Sets the specified limit of the current device to a new value
pub fn set_limit(limit: DeviceLimit, val: usize) -> Result<(), HipError> {
    unsafe {
        try_err!(hipDeviceSetLimit(limit, val));
    }

    Ok(())
}

/// Gets flags set for current device
pub fn get_flags() -> Result<u32, HipError> {
    unsafe { wrap_sys_res!(|flags| hipGetDeviceFlags((&raw mut flags).cast())) }
}

/// Sets flags set for current device
pub fn set_flags(flags: u32) -> Result<(), HipError> {
    unsafe {
        try_err!(hipSetDeviceFlags(flags));
    }

    Ok(())
}
