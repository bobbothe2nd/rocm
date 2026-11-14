#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct __BindgenBitfieldUnit<Storage> {
    storage: Storage,
}
impl<Storage> __BindgenBitfieldUnit<Storage> {
    #[inline]
    pub const fn new(storage: Storage) -> Self {
        Self { storage }
    }
}
impl<Storage> __BindgenBitfieldUnit<Storage>
where
    Storage: AsRef<[u8]> + AsMut<[u8]>,
{
    #[inline]
    fn extract_bit(byte: u8, index: usize) -> bool {
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        byte & mask == mask
    }
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = self.storage.as_ref()[byte_index];
        Self::extract_bit(byte, index)
    }
    #[inline]
    pub unsafe fn raw_get_bit(this: *const Self, index: usize) -> bool {
        debug_assert!(index / 8 < core::mem::size_of::<Storage>());
        let byte_index = index / 8;
        let byte = unsafe {
            *(core::ptr::addr_of!((*this).storage) as *const u8).offset(byte_index as isize)
        };
        Self::extract_bit(byte, index)
    }
    #[inline]
    fn change_bit(byte: u8, index: usize, val: bool) -> u8 {
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        if val { byte | mask } else { byte & !mask }
    }
    #[inline]
    pub fn set_bit(&mut self, index: usize, val: bool) {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = &mut self.storage.as_mut()[byte_index];
        *byte = Self::change_bit(*byte, index, val);
    }
    #[inline]
    pub unsafe fn raw_set_bit(this: *mut Self, index: usize, val: bool) {
        debug_assert!(index / 8 < core::mem::size_of::<Storage>());
        let byte_index = index / 8;
        let byte = unsafe {
            (core::ptr::addr_of_mut!((*this).storage) as *mut u8).offset(byte_index as isize)
        };
        unsafe { *byte = Self::change_bit(*byte, index, val) };
    }
    #[inline]
    pub fn get(&self, bit_offset: usize, bit_width: u8) -> u64 {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize) + 7) / 8 <= self.storage.as_ref().len());
        if bit_width == 0 {
            return 0;
        }
        let mut val = 0u64;
        let storage = self.storage.as_ref();
        let start_byte = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let bytes_needed = (bit_width as usize + bit_shift + 7) / 8;
        if cfg!(target_endian = "big") {
            for i in 0..bytes_needed {
                val |= (storage[start_byte + i].reverse_bits() as u64) << (i * 8);
            }
        } else {
            for i in 0..bytes_needed {
                val |= (storage[start_byte + i] as u64) << (i * 8);
            }
        }
        val >>= bit_shift;
        if bit_width < 64 {
            val &= (1u64 << bit_width) - 1;
        }
        if cfg!(target_endian = "big") {
            val = val.reverse_bits() >> (64 - bit_width as usize);
        }
        val
    }
    #[inline]
    pub unsafe fn raw_get(this: *const Self, bit_offset: usize, bit_width: u8) -> u64 {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < core::mem::size_of::<Storage>());
        debug_assert!(
            (bit_offset + (bit_width as usize) + 7) / 8 <= core::mem::size_of::<Storage>()
        );
        if bit_width == 0 {
            return 0;
        }
        let mut val = 0u64;
        let start_byte = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let bytes_needed = (bit_width as usize + bit_shift + 7) / 8;
        let storage_ptr = unsafe { core::ptr::addr_of!((*this).storage) as *const u8 };
        if cfg!(target_endian = "big") {
            for i in 0..bytes_needed {
                let byte = unsafe { *storage_ptr.add(start_byte + i) };
                val |= (byte.reverse_bits() as u64) << (i * 8);
            }
        } else {
            for i in 0..bytes_needed {
                let byte = unsafe { *storage_ptr.add(start_byte + i) };
                val |= (byte as u64) << (i * 8);
            }
        }
        val >>= bit_shift;
        if bit_width < 64 {
            val &= (1u64 << bit_width) - 1;
        }
        if cfg!(target_endian = "big") {
            val = val.reverse_bits() >> (64 - bit_width as usize);
        }
        val
    }
    #[inline]
    pub fn set(&mut self, bit_offset: usize, bit_width: u8, val: u64) {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize) + 7) / 8 <= self.storage.as_ref().len());
        if bit_width == 0 {
            return;
        }
        let mut val = val;
        if bit_width < 64 {
            val &= (1u64 << bit_width) - 1;
        }
        if cfg!(target_endian = "big") {
            val = val.reverse_bits() >> (64 - bit_width as usize);
        }
        let storage = self.storage.as_mut();
        let start_byte = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let bytes_needed = (bit_width as usize + bit_shift + 7) / 8;
        val <<= bit_shift;
        let field_mask = if bit_width as usize + bit_shift >= 64 {
            !0u64 << bit_shift
        } else {
            ((1u64 << bit_width) - 1) << bit_shift
        };
        for i in 0..bytes_needed {
            let byte_val = (val >> (i * 8)) as u8;
            let byte_mask = (field_mask >> (i * 8)) as u8;
            if cfg!(target_endian = "big") {
                let byte = storage[start_byte + i].reverse_bits();
                let new_byte = (byte & !byte_mask) | (byte_val & byte_mask);
                storage[start_byte + i] = new_byte.reverse_bits();
            } else {
                storage[start_byte + i] =
                    (storage[start_byte + i] & !byte_mask) | (byte_val & byte_mask);
            }
        }
    }
    #[inline]
    pub unsafe fn raw_set(this: *mut Self, bit_offset: usize, bit_width: u8, val: u64) {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < core::mem::size_of::<Storage>());
        debug_assert!(
            (bit_offset + (bit_width as usize) + 7) / 8 <= core::mem::size_of::<Storage>()
        );
        if bit_width == 0 {
            return;
        }
        let mut val = val;
        if bit_width < 64 {
            val &= (1u64 << bit_width) - 1;
        }
        if cfg!(target_endian = "big") {
            val = val.reverse_bits() >> (64 - bit_width as usize);
        }
        let start_byte = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let bytes_needed = (bit_width as usize + bit_shift + 7) / 8;
        val <<= bit_shift;
        let field_mask = if bit_width as usize + bit_shift >= 64 {
            !0u64 << bit_shift
        } else {
            ((1u64 << bit_width) - 1) << bit_shift
        };
        let storage_ptr = unsafe { core::ptr::addr_of_mut!((*this).storage) as *mut u8 };
        for i in 0..bytes_needed {
            let byte_val = (val >> (i * 8)) as u8;
            let byte_mask = (field_mask >> (i * 8)) as u8;
            let byte_ptr = unsafe { storage_ptr.add(start_byte + i) };
            if cfg!(target_endian = "big") {
                let byte = unsafe { (*byte_ptr).reverse_bits() };
                let new_byte = (byte & !byte_mask) | (byte_val & byte_mask);
                unsafe { *byte_ptr = new_byte.reverse_bits() };
            } else {
                unsafe { *byte_ptr = (*byte_ptr & !byte_mask) | (byte_val & byte_mask) };
            }
        }
    }
}
/// Const-generic methods for efficient bitfield access when offset and width
/// are known at compile time.
impl<const N: usize> __BindgenBitfieldUnit<[u8; N]> {
    /// Get a field using const generics for compile-time optimization.
    /// Uses native word size operations when the field fits in usize.
    #[inline]
    pub const fn get_const<const BIT_OFFSET: usize, const BIT_WIDTH: u8>(&self) -> u64 {
        debug_assert!(BIT_WIDTH <= 64);
        debug_assert!(BIT_OFFSET / 8 < N);
        debug_assert!((BIT_OFFSET + (BIT_WIDTH as usize) + 7) / 8 <= N);
        if BIT_WIDTH == 0 {
            return 0;
        }
        let start_byte = BIT_OFFSET / 8;
        let bit_shift = BIT_OFFSET % 8;
        let bytes_needed = (BIT_WIDTH as usize + bit_shift + 7) / 8;
        if BIT_WIDTH as usize + bit_shift <= usize::BITS as usize {
            let mut val = 0usize;
            if cfg!(target_endian = "big") {
                let mut i = 0;
                while i < bytes_needed {
                    val |= (self.storage[start_byte + i].reverse_bits() as usize) << (i * 8);
                    i += 1;
                }
            } else {
                let mut i = 0;
                while i < bytes_needed {
                    val |= (self.storage[start_byte + i] as usize) << (i * 8);
                    i += 1;
                }
            }
            val >>= bit_shift;
            if (BIT_WIDTH as u32) < usize::BITS {
                val &= (1usize << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (usize::BITS as usize - BIT_WIDTH as usize);
            }
            val as u64
        } else {
            let mut val = 0u64;
            if cfg!(target_endian = "big") {
                let mut i = 0;
                while i < bytes_needed {
                    val |= (self.storage[start_byte + i].reverse_bits() as u64) << (i * 8);
                    i += 1;
                }
            } else {
                let mut i = 0;
                while i < bytes_needed {
                    val |= (self.storage[start_byte + i] as u64) << (i * 8);
                    i += 1;
                }
            }
            val >>= bit_shift;
            if BIT_WIDTH < 64 {
                val &= (1u64 << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (64 - BIT_WIDTH as usize);
            }
            val
        }
    }
    /// Set a field using const generics for compile-time optimization.
    /// Uses native word size operations when the field fits in usize.
    #[inline]
    pub fn set_const<const BIT_OFFSET: usize, const BIT_WIDTH: u8>(&mut self, val: u64) {
        debug_assert!(BIT_WIDTH <= 64);
        debug_assert!(BIT_OFFSET / 8 < N);
        debug_assert!((BIT_OFFSET + (BIT_WIDTH as usize) + 7) / 8 <= N);
        if BIT_WIDTH == 0 {
            return;
        }
        let start_byte = BIT_OFFSET / 8;
        let bit_shift = BIT_OFFSET % 8;
        let bytes_needed = (BIT_WIDTH as usize + bit_shift + 7) / 8;
        if BIT_WIDTH as usize + bit_shift <= usize::BITS as usize {
            let mut val = val as usize;
            if (BIT_WIDTH as u32) < usize::BITS {
                val &= (1usize << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (usize::BITS as usize - BIT_WIDTH as usize);
            }
            val <<= bit_shift;
            let field_mask = if BIT_WIDTH as usize + bit_shift >= usize::BITS as usize {
                !0usize << bit_shift
            } else {
                ((1usize << BIT_WIDTH) - 1) << bit_shift
            };
            let mut i = 0;
            while i < bytes_needed {
                let byte_val = (val >> (i * 8)) as u8;
                let byte_mask = (field_mask >> (i * 8)) as u8;
                if cfg!(target_endian = "big") {
                    let byte = self.storage[start_byte + i].reverse_bits();
                    let new_byte = (byte & !byte_mask) | (byte_val & byte_mask);
                    self.storage[start_byte + i] = new_byte.reverse_bits();
                } else {
                    self.storage[start_byte + i] =
                        (self.storage[start_byte + i] & !byte_mask) | (byte_val & byte_mask);
                }
                i += 1;
            }
        } else {
            let mut val = val;
            if BIT_WIDTH < 64 {
                val &= (1u64 << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (64 - BIT_WIDTH as usize);
            }
            val <<= bit_shift;
            let field_mask = if BIT_WIDTH as usize + bit_shift >= 64 {
                !0u64 << bit_shift
            } else {
                ((1u64 << BIT_WIDTH) - 1) << bit_shift
            };
            let mut i = 0;
            while i < bytes_needed {
                let byte_val = (val >> (i * 8)) as u8;
                let byte_mask = (field_mask >> (i * 8)) as u8;
                if cfg!(target_endian = "big") {
                    let byte = self.storage[start_byte + i].reverse_bits();
                    let new_byte = (byte & !byte_mask) | (byte_val & byte_mask);
                    self.storage[start_byte + i] = new_byte.reverse_bits();
                } else {
                    self.storage[start_byte + i] =
                        (self.storage[start_byte + i] & !byte_mask) | (byte_val & byte_mask);
                }
                i += 1;
            }
        }
    }
    /// Raw pointer get using const generics for compile-time optimization.
    /// Uses native word size operations when the field fits in usize.
    #[inline]
    pub const unsafe fn raw_get_const<const BIT_OFFSET: usize, const BIT_WIDTH: u8>(
        this: *const Self,
    ) -> u64 {
        debug_assert!(BIT_WIDTH <= 64);
        debug_assert!(BIT_OFFSET / 8 < N);
        debug_assert!((BIT_OFFSET + (BIT_WIDTH as usize) + 7) / 8 <= N);
        if BIT_WIDTH == 0 {
            return 0;
        }
        let start_byte = BIT_OFFSET / 8;
        let bit_shift = BIT_OFFSET % 8;
        let bytes_needed = (BIT_WIDTH as usize + bit_shift + 7) / 8;
        let storage_ptr = unsafe { core::ptr::addr_of!((*this).storage) as *const u8 };
        if BIT_WIDTH as usize + bit_shift <= usize::BITS as usize {
            let mut val = 0usize;
            if cfg!(target_endian = "big") {
                let mut i = 0;
                while i < bytes_needed {
                    let byte = unsafe { *storage_ptr.add(start_byte + i) };
                    val |= (byte.reverse_bits() as usize) << (i * 8);
                    i += 1;
                }
            } else {
                let mut i = 0;
                while i < bytes_needed {
                    let byte = unsafe { *storage_ptr.add(start_byte + i) };
                    val |= (byte as usize) << (i * 8);
                    i += 1;
                }
            }
            val >>= bit_shift;
            if (BIT_WIDTH as u32) < usize::BITS {
                val &= (1usize << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (usize::BITS as usize - BIT_WIDTH as usize);
            }
            val as u64
        } else {
            let mut val = 0u64;
            if cfg!(target_endian = "big") {
                let mut i = 0;
                while i < bytes_needed {
                    let byte = unsafe { *storage_ptr.add(start_byte + i) };
                    val |= (byte.reverse_bits() as u64) << (i * 8);
                    i += 1;
                }
            } else {
                let mut i = 0;
                while i < bytes_needed {
                    let byte = unsafe { *storage_ptr.add(start_byte + i) };
                    val |= (byte as u64) << (i * 8);
                    i += 1;
                }
            }
            val >>= bit_shift;
            if BIT_WIDTH < 64 {
                val &= (1u64 << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (64 - BIT_WIDTH as usize);
            }
            val
        }
    }
    /// Raw pointer set using const generics for compile-time optimization.
    /// Uses native word size operations when the field fits in usize.
    #[inline]
    pub unsafe fn raw_set_const<const BIT_OFFSET: usize, const BIT_WIDTH: u8>(
        this: *mut Self,
        val: u64,
    ) {
        debug_assert!(BIT_WIDTH <= 64);
        debug_assert!(BIT_OFFSET / 8 < N);
        debug_assert!((BIT_OFFSET + (BIT_WIDTH as usize) + 7) / 8 <= N);
        if BIT_WIDTH == 0 {
            return;
        }
        let start_byte = BIT_OFFSET / 8;
        let bit_shift = BIT_OFFSET % 8;
        let bytes_needed = (BIT_WIDTH as usize + bit_shift + 7) / 8;
        let storage_ptr = this.cast::<[u8; N]>().cast::<u8>();
        if BIT_WIDTH as usize + bit_shift <= usize::BITS as usize {
            let mut val = val as usize;
            if (BIT_WIDTH as u32) < usize::BITS {
                val &= (1usize << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (usize::BITS as usize - BIT_WIDTH as usize);
            }
            val <<= bit_shift;
            let field_mask = if BIT_WIDTH as usize + bit_shift >= usize::BITS as usize {
                !0usize << bit_shift
            } else {
                ((1usize << BIT_WIDTH) - 1) << bit_shift
            };
            let mut i = 0;
            while i < bytes_needed {
                let byte_val = (val >> (i * 8)) as u8;
                let byte_mask = (field_mask >> (i * 8)) as u8;
                let byte_ptr = unsafe { storage_ptr.add(start_byte + i) };
                if cfg!(target_endian = "big") {
                    let byte = unsafe { (*byte_ptr).reverse_bits() };
                    let new_byte = (byte & !byte_mask) | (byte_val & byte_mask);
                    unsafe { *byte_ptr = new_byte.reverse_bits() };
                } else {
                    unsafe { *byte_ptr = (*byte_ptr & !byte_mask) | (byte_val & byte_mask) };
                }
                i += 1;
            }
        } else {
            let mut val = val;
            if BIT_WIDTH < 64 {
                val &= (1u64 << BIT_WIDTH) - 1;
            }
            if cfg!(target_endian = "big") {
                val = val.reverse_bits() >> (64 - BIT_WIDTH as usize);
            }
            val <<= bit_shift;
            let field_mask = if BIT_WIDTH as usize + bit_shift >= 64 {
                !0u64 << bit_shift
            } else {
                ((1u64 << BIT_WIDTH) - 1) << bit_shift
            };
            let mut i = 0;
            while i < bytes_needed {
                let byte_val = (val >> (i * 8)) as u8;
                let byte_mask = (field_mask >> (i * 8)) as u8;
                let byte_ptr = unsafe { storage_ptr.add(start_byte + i) };
                if cfg!(target_endian = "big") {
                    let byte = unsafe { (*byte_ptr).reverse_bits() };
                    let new_byte = (byte & !byte_mask) | (byte_val & byte_mask);
                    unsafe { *byte_ptr = new_byte.reverse_bits() };
                } else {
                    unsafe { *byte_ptr = (*byte_ptr & !byte_mask) | (byte_val & byte_mask) };
                }
                i += 1;
            }
        }
    }
}
/// Default flag. Equivalent to HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM if compiled with
/// ///  -fgpu-default-stream=per-thread flag or HIP_API_PER_THREAD_DEFAULT_STREAM macro is
/// /// defined.
pub const HIP_GET_PROC_ADDRESS_DEFAULT: u32 = 0;
/// Search for all symbols except the corresponding per-thread versions.
pub const HIP_GET_PROC_ADDRESS_LEGACY_STREAM: u32 = 1;
/// Search for all symbols including the per-thread versions. If a per-thread version cannot be
/// found, returns the legacy version.
pub const HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM: u32 = 2;
/// @brief The hipTexRefSetArray function flags parameter override format value
pub const HIP_TRSA_OVERRIDE_FORMAT: u32 = 1;
/// @brief The hipTexRefSetFlags function flags parameter read as integer value
pub const HIP_TRSF_READ_AS_INTEGER: u32 = 1;
/// @brief The hipTexRefSetFlags function flags parameter normalized coordinate value
pub const HIP_TRSF_NORMALIZED_COORDINATES: u32 = 2;
/// @brief The hipTexRefSetFlags function flags parameter srgb value
pub const HIP_TRSF_SRGB: u32 = 16;
pub const hipTextureType1D: u32 = 1;
pub const hipTextureType2D: u32 = 2;
pub const hipTextureType3D: u32 = 3;
pub const hipTextureTypeCubemap: u32 = 12;
pub const hipTextureType1DLayered: u32 = 241;
pub const hipTextureType2DLayered: u32 = 242;
pub const hipTextureTypeCubemapLayered: u32 = 252;
/// Should be same as HSA_IMAGE_OBJECT_SIZE_DWORD/HSA_SAMPLER_OBJECT_SIZE_DWORD
pub const HIP_IMAGE_OBJECT_SIZE_DWORD: u32 = 12;
pub const HIP_SAMPLER_OBJECT_SIZE_DWORD: u32 = 8;
pub const HIP_SAMPLER_OBJECT_OFFSET_DWORD: u32 = 12;
pub const HIP_TEXTURE_OBJECT_SIZE_DWORD: u32 = 20;
pub const HIP_DEPRECATED_MSG : & [u8 ; 189] = b"This API is marked as deprecated and might not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md\0" ;
pub const HIP_RESOURCE_ABI_BYTES: u32 = 40;
pub const hipIpcMemLazyEnablePeerAccess: u32 = 1;
pub const HIP_IPC_HANDLE_SIZE: u32 = 64;
/// Default stream creation flags. These are used with hipStreamCreate().
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
/// (see ::hipEventSynchronize, ::hipEventQuery, and ::hipEventElapsedTime) to synchronize-with
/// the work on which the recorded event (see ::hipEventRecord) is waiting.
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
/// Default flag. Equivalent to hipEnablePerThreadDefaultStream if compiled with
///  -fgpu-default-stream=per-thread flag or HIP_API_PER_THREAD_DEFAULT_STREAM macro is
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
/// This is the same definition as #hipHostAllocDefault.
pub const hipHostMallocDefault: u32 = 0;
/// Memory is considered allocated by all contexts.
pub const hipHostAllocPortable: u32 = 1;
/// Memory is considered allocated by all contexts.
/// This is the same definition as #hipHostAllocPortable.
pub const hipHostMallocPortable: u32 = 1;
/// Map the allocation into the address space for the current device. The device pointer
/// can be obtained with #hipHostGetDevicePointer.
pub const hipHostAllocMapped: u32 = 2;
/// Map the allocation into the address space for the current device. The device pointer
/// can be obtained with #hipHostGetDevicePointer.
/// This is the same #hipHostMallocMapped.
pub const hipHostMallocMapped: u32 = 2;
/// Allocates the memory as write-combined. On some system configurations, write-combined allocation
/// may be transferred faster across the PCI Express bus, however, could have low read efficiency by
/// most CPUs. It's a good option for data transfer from host to device via mapped pinned memory.
///  This flag is only for CUDA source compatibility but not functional within HIP runtime,
/// because the allocation path is currently not supported on the AMD platform.
pub const hipHostAllocWriteCombined: u32 = 4;
/// Allocates the memory as write-combined. On some system configurations, write-combined allocation
/// may be transferred faster across the PCI Express bus, however, could have low read efficiency by
/// most CPUs. It's a good option for data transfer from host to device via mapped pinned memory.
///  This flag is the same definition as #hipHostAllocWriteCombined which is equivalent to
/// cudaHostAllocWriteCombined. It is only for CUDA source compatibility but not functional within
/// HIP runtime, because the allocation path is currently not supported on the AMD platform.
pub const hipHostMallocWriteCombined: u32 = 4;
/// Host memory will be forcedly allocated on extended fine grained system memory
/// pool which is with MTYPE_UC.
///  This allocation flag is applicable on AMD devices, except for Navi4X, in Linux only.
pub const hipHostMallocUncached: u32 = 268435456;
pub const hipHostAllocUncached: u32 = 268435456;
/// Host memory allocation will follow numa policy set by user.
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
/// @ref hipExtDynDataPrefetchConfig.  The actual device limit is queried at
/// runtime via @ref hipDeviceAttributeMaxDynDataPrefetchRegions.
pub const HIP_EXT_DYN_DATA_PREFETCH_MAX_REGIONS: u32 = 2;
/// This port activates when the kernel has finished executing.
pub const hipGraphKernelNodePortDefault: u32 = 0;
/// This port activates when all blocks of the kernel have begun execution.
pub const hipGraphKernelNodePortLaunchCompletion: u32 = 2;
/// This port activates when all blocks of the kernel have performed
/// hipTriggerProgrammaticLaunchCompletion() or have terminated.
/// It must be used with edge type hipGraphDependencyTypeProgrammatic.
pub const hipGraphKernelNodePortProgrammatic: u32 = 1;
#[repr(i32)]
/// hipJitOption
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitOption {
    /// CUDA Only Maximum registers may be used in a thread,
    ///< passed to compiler
    hipJitOptionMaxRegisters = 0,
    /// CUDA Only Number of thread per block
    hipJitOptionThreadsPerBlock = 1,
    /// CUDA Only Value for total wall clock time
    hipJitOptionWallTime = 2,
    /// CUDA Only Pointer to the buffer with logged information
    hipJitOptionInfoLogBuffer = 3,
    /// CUDA Only Size of the buffer in bytes for logged info
    hipJitOptionInfoLogBufferSizeBytes = 4,
    /// CUDA Only Pointer to the buffer with logged error(s)
    hipJitOptionErrorLogBuffer = 5,
    /// CUDA Only Size of the buffer in bytes for logged
    ///< error(s)
    hipJitOptionErrorLogBufferSizeBytes = 6,
    /// Value of optimization level for generated codes, acceptable
    ///< options -O0, -O1, -O2, -O3
    hipJitOptionOptimizationLevel = 7,
    /// CUDA Only The target context, which is the default
    hipJitOptionTargetFromContext = 8,
    /// CUDA Only JIT target
    hipJitOptionTarget = 9,
    /// CUDA Only Fallback strategy
    hipJitOptionFallbackStrategy = 10,
    /// CUDA Only Generate debug information
    hipJitOptionGenerateDebugInfo = 11,
    /// CUDA Only Generate log verbose
    hipJitOptionLogVerbose = 12,
    /// CUDA Only Generate line number information
    hipJitOptionGenerateLineInfo = 13,
    /// CUDA Only Set cache mode
    hipJitOptionCacheMode = 14,
    /// @deprecated CUDA Only New SM3X option.
    hipJitOptionSm3xOpt = 15,
    /// CUDA Only Set fast compile
    hipJitOptionFastCompile = 16,
    /// CUDA Only Array of device symbol names to be relocated to the
    ///< host
    hipJitOptionGlobalSymbolNames = 17,
    /// CUDA Only Array of host addresses to be relocated to the
    ///< device
    hipJitOptionGlobalSymbolAddresses = 18,
    /// CUDA Only Number of symbol count.
    hipJitOptionGlobalSymbolCount = 19,
    /// @deprecated CUDA Only Enable link-time optimization for device code
    hipJitOptionLto = 20,
    /// @deprecated CUDA Only Set single-precision denormals.
    hipJitOptionFtz = 21,
    /// @deprecated CUDA Only Set single-precision floating-point division
    ///< and reciprocals
    hipJitOptionPrecDiv = 22,
    /// @deprecated CUDA Only Set single-precision floating-point square root
    hipJitOptionPrecSqrt = 23,
    /// @deprecated CUDA Only Enable floating-point multiplies and
    ///< adds/subtracts operations
    hipJitOptionFma = 24,
    /// CUDA Only Generates Position Independent code
    hipJitOptionPositionIndependentCode = 25,
    /// CUDA Only Hints to JIT compiler the minimum number of CTAs frin
    ///< kernel's grid to be mapped to SM
    hipJitOptionMinCTAPerSM = 26,
    /// CUDA only Maximum number of threads in a thread block
    hipJitOptionMaxThreadsPerBlock = 27,
    /// Cuda only Override Directive values
    hipJitOptionOverrideDirectiveValues = 28,
    /// Number of options
    hipJitOptionNumOptions = 29,
    /// Hip Only Linker options to be passed on to compiler
    hipJitOptionIRtoISAOptExt = 10000,
    /// Hip Only Count of linker options to be passed on to compiler
    hipJitOptionIRtoISAOptCountExt = 10001,
}
#[repr(i32)]
/// hipJitInputType
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitInputType {
    /// Cuda only Input cubin
    hipJitInputCubin = 0,
    /// Cuda only Input PTX
    hipJitInputPtx = 1,
    /// Cuda Only Input FAT Binary
    hipJitInputFatBinary = 2,
    /// Cuda Only Host Object with embedded device code
    hipJitInputObject = 3,
    /// Cuda Only Archive of Host Objects with embedded
    ///< device code
    hipJitInputLibrary = 4,
    /// @deprecated Cuda only High Level intermediate
    ///< code for LTO
    hipJitInputNvvm = 5,
    /// Count of Legacy Input Types
    hipJitNumLegacyInputTypes = 6,
    /// HIP Only LLVM Bitcode or IR assembly
    hipJitInputLLVMBitcode = 100,
    /// HIP Only LLVM Clang Bundled Code
    hipJitInputLLVMBundledBitcode = 101,
    /// HIP Only LLVM Archive of Bundled Bitcode
    hipJitInputLLVMArchivesOfBundledBitcode = 102,
    /// HIP Only SPIRV Code Object
    hipJitInputSpirv = 103,
    /// Count of Input Types
    hipJitNumInputTypes = 10,
}
#[repr(i32)]
/// hipJitCacheMode
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitCacheMode {
    hipJitCacheOptionNone = 0,
    hipJitCacheOptionCG = 1,
    hipJitCacheOptionCA = 2,
}
#[repr(i32)]
/// hipJitFallback
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipJitFallback {
    hipJitPreferPTX = 0,
    hipJitPreferBinary = 1,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipLibraryOption_e {
    hipLibraryHostUniversalFunctionAndDataTable = 0,
    hipLibraryBinaryIsPreserved = 1,
}
pub use self::hipLibraryOption_e as hipLibraryOption;
pub const HIP_SUCCESS: _bindgen_ty_15 = _bindgen_ty_15::HIP_SUCCESS;
pub const HIP_ERROR_INVALID_VALUE: _bindgen_ty_15 = _bindgen_ty_15::HIP_ERROR_INVALID_VALUE;
pub const HIP_ERROR_NOT_INITIALIZED: _bindgen_ty_15 = _bindgen_ty_15::HIP_ERROR_NOT_INITIALIZED;
pub const HIP_ERROR_LAUNCH_OUT_OF_RESOURCES: _bindgen_ty_15 =
    _bindgen_ty_15::HIP_ERROR_LAUNCH_OUT_OF_RESOURCES;
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum _bindgen_ty_15 {
    HIP_SUCCESS = 0,
    HIP_ERROR_INVALID_VALUE = 1,
    HIP_ERROR_NOT_INITIALIZED = 2,
    HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = 3,
}
/// @defgroup GlobalDefs Global enum and defines
/// @{
////
////**
/// hipDeviceArch_t
///
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipDeviceArch_t {
    pub _bindgen_align: [u32; 0],
    pub _bitfield_1: __BindgenBitfieldUnit<[u8; 3usize]>,
    pub __bindgen_padding_0: u8,
}
impl hipDeviceArch_t {
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasGlobalInt32Atomics(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<0usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasGlobalInt32Atomics(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<0usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasGlobalInt32Atomics_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<0usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasGlobalInt32Atomics_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<0usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasGlobalFloatAtomicExch(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<1usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasGlobalFloatAtomicExch(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<1usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasGlobalFloatAtomicExch_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<1usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasGlobalFloatAtomicExch_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<1usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasSharedInt32Atomics(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<2usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasSharedInt32Atomics(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<2usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasSharedInt32Atomics_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<2usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasSharedInt32Atomics_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<2usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasSharedFloatAtomicExch(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<3usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasSharedFloatAtomicExch(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<3usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasSharedFloatAtomicExch_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<3usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasSharedFloatAtomicExch_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<3usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasFloatAtomicAdd(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<4usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasFloatAtomicAdd(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<4usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasFloatAtomicAdd_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<4usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasFloatAtomicAdd_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<4usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasGlobalInt64Atomics(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<5usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasGlobalInt64Atomics(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<5usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasGlobalInt64Atomics_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<5usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasGlobalInt64Atomics_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<5usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasSharedInt64Atomics(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<6usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasSharedInt64Atomics(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<6usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasSharedInt64Atomics_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<6usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasSharedInt64Atomics_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<6usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasDoubles(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<7usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasDoubles(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<7usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasDoubles_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<7usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasDoubles_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<7usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasWarpVote(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<8usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasWarpVote(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<8usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasWarpVote_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<8usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasWarpVote_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<8usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasWarpBallot(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<9usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasWarpBallot(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<9usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasWarpBallot_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<9usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasWarpBallot_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<9usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasWarpShuffle(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<10usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasWarpShuffle(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<10usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasWarpShuffle_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<10usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasWarpShuffle_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<10usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasFunnelShift(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<11usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasFunnelShift(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<11usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasFunnelShift_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<11usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasFunnelShift_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<11usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasThreadFenceSystem(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<12usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasThreadFenceSystem(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<12usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasThreadFenceSystem_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<12usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasThreadFenceSystem_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<12usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasSyncThreadsExt(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<13usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasSyncThreadsExt(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<13usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasSyncThreadsExt_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<13usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasSyncThreadsExt_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<13usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasSurfaceFuncs(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<14usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasSurfaceFuncs(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<14usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasSurfaceFuncs_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<14usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasSurfaceFuncs_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<14usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn has3dGrid(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<15usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_has3dGrid(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<15usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn has3dGrid_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<15usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_has3dGrid_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<15usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn hasDynamicParallelism(&self) -> ::core::ffi::c_uint {
        unsafe { ::std::mem::transmute(self._bitfield_1.get_const::<16usize, 1u8>() as u32) }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn set_hasDynamicParallelism(&mut self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set_const::<16usize, 1u8>(val as u64)
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn hasDynamicParallelism_raw(this: *const Self) -> ::core::ffi::c_uint {
        unsafe {
            ::std::mem::transmute(
                <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get_const::<16usize, 1u8>(
                    ::std::ptr::addr_of!((*this)._bitfield_1),
                ) as u32,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub unsafe fn set_hasDynamicParallelism_raw(this: *mut Self, val: ::core::ffi::c_uint) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set_const::<16usize, 1u8>(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                val as u64,
            )
        }
    }
    #[inline]
    #[allow(unnecessary_transmutes)]
    pub fn new_bitfield_1(
        hasGlobalInt32Atomics: ::core::ffi::c_uint,
        hasGlobalFloatAtomicExch: ::core::ffi::c_uint,
        hasSharedInt32Atomics: ::core::ffi::c_uint,
        hasSharedFloatAtomicExch: ::core::ffi::c_uint,
        hasFloatAtomicAdd: ::core::ffi::c_uint,
        hasGlobalInt64Atomics: ::core::ffi::c_uint,
        hasSharedInt64Atomics: ::core::ffi::c_uint,
        hasDoubles: ::core::ffi::c_uint,
        hasWarpVote: ::core::ffi::c_uint,
        hasWarpBallot: ::core::ffi::c_uint,
        hasWarpShuffle: ::core::ffi::c_uint,
        hasFunnelShift: ::core::ffi::c_uint,
        hasThreadFenceSystem: ::core::ffi::c_uint,
        hasSyncThreadsExt: ::core::ffi::c_uint,
        hasSurfaceFuncs: ::core::ffi::c_uint,
        has3dGrid: ::core::ffi::c_uint,
        hasDynamicParallelism: ::core::ffi::c_uint,
    ) -> __BindgenBitfieldUnit<[u8; 3usize]> {
        let mut __bindgen_bitfield_unit: __BindgenBitfieldUnit<[u8; 3usize]> = Default::default();
        __bindgen_bitfield_unit.set_const::<0usize, 1u8>({
            let hasGlobalInt32Atomics: u32 =
                unsafe { ::std::mem::transmute(hasGlobalInt32Atomics) };
            hasGlobalInt32Atomics as u64
        });
        __bindgen_bitfield_unit.set_const::<1usize, 1u8>({
            let hasGlobalFloatAtomicExch: u32 =
                unsafe { ::std::mem::transmute(hasGlobalFloatAtomicExch) };
            hasGlobalFloatAtomicExch as u64
        });
        __bindgen_bitfield_unit.set_const::<2usize, 1u8>({
            let hasSharedInt32Atomics: u32 =
                unsafe { ::std::mem::transmute(hasSharedInt32Atomics) };
            hasSharedInt32Atomics as u64
        });
        __bindgen_bitfield_unit.set_const::<3usize, 1u8>({
            let hasSharedFloatAtomicExch: u32 =
                unsafe { ::std::mem::transmute(hasSharedFloatAtomicExch) };
            hasSharedFloatAtomicExch as u64
        });
        __bindgen_bitfield_unit.set_const::<4usize, 1u8>({
            let hasFloatAtomicAdd: u32 = unsafe { ::std::mem::transmute(hasFloatAtomicAdd) };
            hasFloatAtomicAdd as u64
        });
        __bindgen_bitfield_unit.set_const::<5usize, 1u8>({
            let hasGlobalInt64Atomics: u32 =
                unsafe { ::std::mem::transmute(hasGlobalInt64Atomics) };
            hasGlobalInt64Atomics as u64
        });
        __bindgen_bitfield_unit.set_const::<6usize, 1u8>({
            let hasSharedInt64Atomics: u32 =
                unsafe { ::std::mem::transmute(hasSharedInt64Atomics) };
            hasSharedInt64Atomics as u64
        });
        __bindgen_bitfield_unit.set_const::<7usize, 1u8>({
            let hasDoubles: u32 = unsafe { ::std::mem::transmute(hasDoubles) };
            hasDoubles as u64
        });
        __bindgen_bitfield_unit.set_const::<8usize, 1u8>({
            let hasWarpVote: u32 = unsafe { ::std::mem::transmute(hasWarpVote) };
            hasWarpVote as u64
        });
        __bindgen_bitfield_unit.set_const::<9usize, 1u8>({
            let hasWarpBallot: u32 = unsafe { ::std::mem::transmute(hasWarpBallot) };
            hasWarpBallot as u64
        });
        __bindgen_bitfield_unit.set_const::<10usize, 1u8>({
            let hasWarpShuffle: u32 = unsafe { ::std::mem::transmute(hasWarpShuffle) };
            hasWarpShuffle as u64
        });
        __bindgen_bitfield_unit.set_const::<11usize, 1u8>({
            let hasFunnelShift: u32 = unsafe { ::std::mem::transmute(hasFunnelShift) };
            hasFunnelShift as u64
        });
        __bindgen_bitfield_unit.set_const::<12usize, 1u8>({
            let hasThreadFenceSystem: u32 = unsafe { ::std::mem::transmute(hasThreadFenceSystem) };
            hasThreadFenceSystem as u64
        });
        __bindgen_bitfield_unit.set_const::<13usize, 1u8>({
            let hasSyncThreadsExt: u32 = unsafe { ::std::mem::transmute(hasSyncThreadsExt) };
            hasSyncThreadsExt as u64
        });
        __bindgen_bitfield_unit.set_const::<14usize, 1u8>({
            let hasSurfaceFuncs: u32 = unsafe { ::std::mem::transmute(hasSurfaceFuncs) };
            hasSurfaceFuncs as u64
        });
        __bindgen_bitfield_unit.set_const::<15usize, 1u8>({
            let has3dGrid: u32 = unsafe { ::std::mem::transmute(has3dGrid) };
            has3dGrid as u64
        });
        __bindgen_bitfield_unit.set_const::<16usize, 1u8>({
            let hasDynamicParallelism: u32 =
                unsafe { ::std::mem::transmute(hasDynamicParallelism) };
            hasDynamicParallelism as u64
        });
        __bindgen_bitfield_unit
    }
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipUUID_t {
    pub bytes: [::core::ffi::c_char; 16usize],
}
pub type hipUUID = hipUUID_t;
impl hipError_t {
    pub const hipErrorMemoryAllocation: hipError_t = hipError_t::hipErrorOutOfMemory;
}
impl hipError_t {
    pub const hipErrorInitializationError: hipError_t = hipError_t::hipErrorNotInitialized;
}
impl hipError_t {
    pub const hipErrorMapBufferObjectFailed: hipError_t = hipError_t::hipErrorMapFailed;
}
impl hipError_t {
    pub const hipErrorInvalidResourceHandle: hipError_t = hipError_t::hipErrorInvalidHandle;
}
#[repr(u32)]
/// HIP error type
///
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipError_t {
    /// Successful completion.
    hipSuccess = 0,
    /// One or more of the parameters passed to the API call is NULL
    ///< or not in an acceptable range.
    hipErrorInvalidValue = 1,
    /// out of memory range.
    hipErrorOutOfMemory = 2,
    /// Invalid not initialized
    hipErrorNotInitialized = 3,
    /// Deinitialized
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    /// Invalide configuration
    hipErrorInvalidConfiguration = 9,
    /// Invalid pitch value
    hipErrorInvalidPitchValue = 12,
    /// Invalid symbol
    hipErrorInvalidSymbol = 13,
    /// Invalid Device Pointer
    hipErrorInvalidDevicePointer = 17,
    /// Invalid memory copy direction
    hipErrorInvalidMemcpyDirection = 21,
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    /// Invalid device function
    hipErrorInvalidDeviceFunction = 98,
    /// Call to hipGetDeviceCount returned 0 devices
    hipErrorNoDevice = 100,
    /// DeviceID must be in range from 0 to compute-devices.
    hipErrorInvalidDevice = 101,
    /// Invalid image
    hipErrorInvalidImage = 200,
    /// Produced when input context is invalid.
    hipErrorInvalidContext = 201,
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    /// Unsupported limit
    hipErrorUnsupportedLimit = 215,
    /// The context is already in use
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    /// In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidKernelFile = 218,
    hipErrorInvalidGraphicsContext = 219,
    /// Invalid source.
    hipErrorInvalidSource = 300,
    /// the file is not found.
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    /// Failed to initialize shared object.
    hipErrorSharedObjectInitFailed = 303,
    /// Not the correct operating system
    hipErrorOperatingSystem = 304,
    /// Invalide handle
    hipErrorInvalidHandle = 400,
    /// Resource required is not in a valid state to perform operation.
    hipErrorIllegalState = 401,
    /// Not found
    hipErrorNotFound = 500,
    /// Indicates that asynchronous operations enqueued earlier are not
    ///< ready.  This is not actually an error, but is used to distinguish
    ///< from hipSuccess (which indicates completion).  APIs that return
    ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorNotReady = 600,
    hipErrorIllegalAddress = 700,
    /// Out of resources error.
    hipErrorLaunchOutOfResources = 701,
    /// Timeout for the launch.
    hipErrorLaunchTimeOut = 702,
    /// Peer access was already enabled from the current
    ///< device.
    hipErrorPeerAccessAlreadyEnabled = 704,
    /// Peer access was never enabled from the current device.
    hipErrorPeerAccessNotEnabled = 705,
    /// The process is active.
    hipErrorSetOnActiveProcess = 708,
    /// The context is already destroyed
    hipErrorContextIsDestroyed = 709,
    /// Produced when the kernel calls assert.
    hipErrorAssert = 710,
    /// Produced when trying to lock a page-locked
    ///< memory.
    hipErrorHostMemoryAlreadyRegistered = 712,
    /// Produced when trying to unlock a non-page-locked
    ///< memory.
    hipErrorHostMemoryNotRegistered = 713,
    /// An exception occurred on the device while executing a kernel.
    hipErrorLaunchFailure = 719,
    /// This error indicates that the number of blocks
    ///< launched per grid for a kernel that was launched
    ///< via cooperative launch APIs exceeds the maximum
    ///< number of allowed blocks for the current device.
    hipErrorCooperativeLaunchTooLarge = 720,
    /// The attempted operation is not permitted, typically because the
    ///< resource is owned by another object that is responsible for
    ///< releasing it.
    hipErrorNotPermitted = 800,
    /// Produced when the hip API is not supported/implemented
    hipErrorNotSupported = 801,
    /// The operation is not permitted when the stream
    ///< is capturing.
    hipErrorStreamCaptureUnsupported = 900,
    /// The current capture sequence on the stream
    ///< has been invalidated due to a previous error.
    hipErrorStreamCaptureInvalidated = 901,
    /// The operation would have resulted in a merge of
    ///< two independent capture sequences.
    hipErrorStreamCaptureMerge = 902,
    /// The capture was not initiated in this stream.
    hipErrorStreamCaptureUnmatched = 903,
    /// The capture sequence contains a fork that was not
    ///< joined to the primary stream.
    hipErrorStreamCaptureUnjoined = 904,
    /// A dependency would have been created which crosses
    ///< the capture sequence boundary. Only implicit
    ///< in-stream ordering dependencies  are allowed
    ///< to cross the boundary
    hipErrorStreamCaptureIsolation = 905,
    /// The operation would have resulted in a disallowed
    ///< implicit dependency on a current capture sequence
    ///< from hipStreamLegacy.
    hipErrorStreamCaptureImplicit = 906,
    /// The operation is not permitted on an event which was last
    ///< recorded in a capturing stream.
    hipErrorCapturedEvent = 907,
    /// A stream capture sequence not initiated with
    ///< the hipStreamCaptureModeRelaxed argument to
    ///< hipStreamBeginCapture was passed to
    ///< hipStreamEndCapture in a different thread.
    hipErrorStreamCaptureWrongThread = 908,
    /// This error indicates that the graph update
    ///< not performed because it included changes which
    ///< violated constraintsspecific to instantiated graph
    ///< update.
    hipErrorGraphExecUpdateFailure = 910,
    /// Invalid channel descriptor.
    hipErrorInvalidChannelDescriptor = 911,
    /// Invalid texture.
    hipErrorInvalidTexture = 912,
    /// Resource type is not valid for the operation.
    hipErrorInvalidResourceType = 914,
    /// Resource configuration is not valid for
    ///< the operation.
    hipErrorInvalidResourceConfiguration = 915,
    /// The stream is detached.
    hipErrorStreamDetached = 916,
    /// Unknown error.
    hipErrorUnknown = 999,
    /// HSA runtime memory call returned error.  Typically not seen
    ///< in production systems.
    hipErrorRuntimeMemory = 1052,
    /// HSA runtime call other than memory returned error.  Typically
    ///< not seen in production systems.
    hipErrorRuntimeOther = 1053,
    /// The specified cluster size is invalid, for instance
    ///< when passing launch configurations to occupancy
    ///< calculations
    hipErrorInvalidClusterSize = 1054,
    /// Marker that more error codes are needed.
    hipErrorTbd = 1055,
}
impl hipError_t {
    pub const fn is_err(self) -> bool {
        !matches!(self, Self::hipSuccess | Self::hipErrorNotReady)
    }
}
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
    pub luid: [::core::ffi::c_char; 8],

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
#[repr(i32)]
/// hipMemoryType (for pointer attributes)
/// hipMemoryType enum values are combination of cudaMemoryType and cuMemoryType and AMD
/// specific enum values.
///
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemoryType {
    /// Unregistered memory
    hipMemoryTypeUnregistered = 0,
    /// Memory is physically located on host
    hipMemoryTypeHost = 1,
    /// Memory is physically located on device. (see deviceId for
    ///< specific device)
    hipMemoryTypeDevice = 2,
    /// Managed memory, automaticallly managed by the unified
    ///< memory system
    ///< place holder for new values.
    hipMemoryTypeManaged = 3,
    /// Array memory, physically located on device. (see deviceId for
    ///< specific device)
    hipMemoryTypeArray = 10,
    /// unified address space
    hipMemoryTypeUnified = 11,
}
/// Pointer attributes
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipPointerAttribute_t {
    pub type_: hipMemoryType,
    pub device: ::core::ffi::c_int,
    pub devicePointer: *mut ::core::ffi::c_void,
    pub hostPointer: *mut ::core::ffi::c_void,
    pub isManaged: ::core::ffi::c_int,
    pub allocationFlags: ::core::ffi::c_uint,
}
impl hipDeviceAttribute_t {
    pub const hipDeviceAttributeEccEnabled: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeCudaCompatibleBegin;
}
impl hipDeviceAttribute_t {
    pub const hipDeviceAttributePciDomainID: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributePciDomainId;
}
impl hipDeviceAttribute_t {
    pub const hipDeviceAttributeClockInstructionRate: hipDeviceAttribute_t =
        hipDeviceAttribute_t::hipDeviceAttributeAmdSpecificBegin;
}
#[repr(i32)]
/// hipDeviceAttribute_t
/// hipDeviceAttributeUnused number: 5
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,
    /// Cuda only. The maximum size of the window
    ///< policy in bytes.
    hipDeviceAttributeAccessPolicyMaxWindowSize = 1,
    /// Asynchronous engines number.
    hipDeviceAttributeAsyncEngineCount = 2,
    /// Whether host memory can be mapped into device address
    ///< space
    hipDeviceAttributeCanMapHostMemory = 3,
    /// Device can access host registered
    ///< memory at the same virtual address as
    ///< the CPU
    hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4,
    /// Peak clock frequency in kilohertz.
    hipDeviceAttributeClockRate = 5,
    /// Compute mode that device is currently in.
    hipDeviceAttributeComputeMode = 6,
    /// Device supports Compute Preemption.
    hipDeviceAttributeComputePreemptionSupported = 7,
    /// Device can possibly execute multiple kernels
    ///< concurrently.
    hipDeviceAttributeConcurrentKernels = 8,
    /// Device can coherently access managed memory
    ///< concurrently with the CPU
    hipDeviceAttributeConcurrentManagedAccess = 9,
    /// Support cooperative launch
    hipDeviceAttributeCooperativeLaunch = 10,
    /// Support cooperative launch on multiple
    ///< devices
    hipDeviceAttributeCooperativeMultiDeviceLaunch = 11,
    /// Device can concurrently copy memory and execute a kernel.
    ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDeviceOverlap = 12,
    /// Host can directly access managed memory on
    ///< the device without migration
    hipDeviceAttributeDirectManagedMemAccessFromHost = 13,
    /// Device supports caching globals in L1
    hipDeviceAttributeGlobalL1CacheSupported = 14,
    /// Link between the device and the host supports
    ///< native atomic operations
    hipDeviceAttributeHostNativeAtomicSupported = 15,
    /// Device is integrated GPU
    hipDeviceAttributeIntegrated = 16,
    /// Multiple GPU devices.
    hipDeviceAttributeIsMultiGpuBoard = 17,
    /// Run time limit for kernels executed on the device
    hipDeviceAttributeKernelExecTimeout = 18,
    /// Size of L2 cache in bytes. 0 if the device doesn't have L2
    ///< cache.
    hipDeviceAttributeL2CacheSize = 19,
    /// caching locals in L1 is supported
    hipDeviceAttributeLocalL1CacheSupported = 20,
    /// 8-byte locally unique identifier in 8 bytes. Undefined on TCC and
    ///< non-Windows platforms
    hipDeviceAttributeLuid = 21,
    /// Luid device node mask. Undefined on TCC and
    ///< non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask = 22,
    /// Major compute capability version number.
    hipDeviceAttributeComputeCapabilityMajor = 23,
    /// Device supports allocating managed memory on this system
    hipDeviceAttributeManagedMemory = 24,
    /// Max block size per multiprocessor
    hipDeviceAttributeMaxBlocksPerMultiProcessor = 25,
    /// Max block size in width.
    hipDeviceAttributeMaxBlockDimX = 26,
    /// Max block size in height.
    hipDeviceAttributeMaxBlockDimY = 27,
    /// Max block size in depth.
    hipDeviceAttributeMaxBlockDimZ = 28,
    /// Max grid size  in width.
    hipDeviceAttributeMaxGridDimX = 29,
    /// Max grid size  in height.
    hipDeviceAttributeMaxGridDimY = 30,
    /// Max grid size  in depth.
    hipDeviceAttributeMaxGridDimZ = 31,
    /// Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1D = 32,
    /// Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface1DLayered = 33,
    /// Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2D = 34,
    /// Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface2DLayered = 35,
    /// Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurface3D = 36,
    /// Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemap = 37,
    /// Cuda only. Maximum dimension of Cubemap layered
    ///< surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered = 38,
    /// Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DWidth = 39,
    /// Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLayered = 40,
    /// Maximum number of elements allocatable in a 1D linear
    ///< texture. Use cudaDeviceGetTexture1DLinearMaxWidth()
    ///< instead on Cuda.
    hipDeviceAttributeMaxTexture1DLinear = 41,
    /// Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture1DMipmap = 42,
    /// Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DWidth = 43,
    /// Maximum dimension hight of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight = 44,
    /// Maximum dimensions of 2D texture if gather operations
    ///< performed.
    hipDeviceAttributeMaxTexture2DGather = 45,
    /// Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLayered = 46,
    /// Maximum dimensions (width, height, pitch) of 2D
    ///< textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DLinear = 47,
    /// Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture2DMipmap = 48,
    /// Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DWidth = 49,
    /// Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight = 50,
    /// Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth = 51,
    /// Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTexture3DAlt = 52,
    /// Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemap = 53,
    /// Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxTextureCubemapLayered = 54,
    /// Maximum dimension of a block
    hipDeviceAttributeMaxThreadsDim = 55,
    /// Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerBlock = 56,
    /// Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 57,
    /// Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMaxPitch = 58,
    /// Global memory bus width in bits.
    hipDeviceAttributeMemoryBusWidth = 59,
    /// Peak memory clock frequency in kilohertz.
    hipDeviceAttributeMemoryClockRate = 60,
    /// Minor compute capability version number.
    hipDeviceAttributeComputeCapabilityMinor = 61,
    /// Unique ID of device group on the same multi-GPU
    ///< board
    hipDeviceAttributeMultiGpuBoardGroupID = 62,
    /// Number of multi-processors. When the GPU works in Compute
    ///< Unit (CU) mode, this value equals the number of CUs;
    ///< when in Workgroup Processor (WGP) mode, this value equels
    ///< half of CUs, because a single WGP contains two CUs.
    hipDeviceAttributeMultiprocessorCount = 63,
    /// Previously hipDeviceAttributeName
    hipDeviceAttributeUnused1 = 64,
    /// Device supports coherently accessing pageable memory
    ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccess = 65,
    /// Device accesses pageable memory
    ///< via the host's page tables
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66,
    /// PCI Bus ID.
    hipDeviceAttributePciBusId = 67,
    /// PCI Device ID. Returns pcie slot id
    hipDeviceAttributePciDeviceId = 68,
    /// PCI Domain Id.
    hipDeviceAttributePciDomainId = 69,
    /// Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributePersistingL2CacheMaxSize = 70,
    /// 32-bit registers available to a thread block. This
    ///< number is shared by all thread blocks simultaneously
    ///< resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerBlock = 71,
    /// 32-bit registers available per block.
    hipDeviceAttributeMaxRegistersPerMultiprocessor = 72,
    /// Shared memory reserved by CUDA driver per
    ///< block.
    hipDeviceAttributeReservedSharedMemPerBlock = 73,
    /// Maximum shared memory available per block in
    ///< bytes.
    hipDeviceAttributeMaxSharedMemoryPerBlock = 74,
    /// Maximum shared memory per block usable by special
    ///< opt in.
    hipDeviceAttributeSharedMemPerBlockOptin = 75,
    /// Shared memory available per multiprocessor.
    hipDeviceAttributeSharedMemPerMultiprocessor = 76,
    /// Cuda only. Performance ratio of single
    ///< precision to double precision.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77,
    /// Whether to support stream priorities.
    hipDeviceAttributeStreamPrioritiesSupported = 78,
    /// Alignment requirement for surfaces
    hipDeviceAttributeSurfaceAlignment = 79,
    /// Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTccDriver = 80,
    /// Alignment requirement for textures
    hipDeviceAttributeTextureAlignment = 81,
    /// Pitch alignment requirement for 2D texture
    ///< references bound to pitched memory;
    hipDeviceAttributeTexturePitchAlignment = 82,
    /// Constant memory size in bytes.
    hipDeviceAttributeTotalConstantMemory = 83,
    /// Global memory available on devicice.
    hipDeviceAttributeTotalGlobalMem = 84,
    /// Cuda only. An unified address space shared with the
    ///< host.
    hipDeviceAttributeUnifiedAddressing = 85,
    /// Previously hipDeviceAttributeUuid
    hipDeviceAttributeUnused2 = 86,
    /// Warp size in threads.
    hipDeviceAttributeWarpSize = 87,
    /// Device supports HIP Stream Ordered Memory Allocator
    hipDeviceAttributeMemoryPoolsSupported = 88,
    /// Device supports HIP virtual memory
    ///< management
    hipDeviceAttributeVirtualMemoryManagementSupported = 89,
    /// Can device support host memory registration via
    ///< hipHostRegister
    hipDeviceAttributeHostRegisterSupported = 90,
    /// Supported handle mask for HIP Stream
    ///< Ordered Memory Allocator
    hipDeviceAttributeMemoryPoolSupportedHandleTypes = 91,
    /// NUMA ID of the cpu node closest to the device,
    ///< or -1 when NUMA isn't supported
    hipDeviceAttributeHostNumaId = 92,
    /// Device supports DMABuf buffer sharing
    hipDeviceAttributeDmaBufSupported = 93,
    /// GPU Direct RDMA with HIP VMM is supported
    ///< (requires DMA-Buf and HIP virtual memory
    ///< management)
    hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported = 94,
    /// Device supports exporting memory to a fabric handle
    hipDeviceAttributeHandleTypeFabricSupported = 95,
    /// Device supports host-allocated DMABuf buffer sharing
    hipDeviceAttributeHostAllocDmaBufSupported = 96,
    /// Device supports GPUDirect RDMA APIs
    hipDeviceAttributeGPUDirectRDMASupported = 97,
    /// Bitmask of
    ///< hipFlushGPUDirectRDMAWritesOptions
    ///< describing the flush paths the device
    ///< supports
    hipDeviceAttributeGPUDirectRDMAFlushWritesOptions = 98,
    /// A hipGPUDirectRDMAWritesOrdering value
    ///< giving the scope at which GPUDirect RDMA
    ///< writes are naturally ordered, i.e. visible
    ///< without an explicit flush
    hipDeviceAttributeGPUDirectRDMAWritesOrdering = 99,
    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,
    /// Previously hipDeviceAttributeArch
    hipDeviceAttributeUnused3 = 10001,
    /// Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002,
    /// Previously hipDeviceAttributeGcnArch
    hipDeviceAttributeUnused4 = 10003,
    /// Previously hipDeviceAttributeGcnArchName
    hipDeviceAttributeUnused5 = 10004,
    /// Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpMemFlushCntl = 10005,
    /// Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl = 10006,
    /// Supports cooperative launch on
    ///< multiple devices with unmatched
    ///< functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 10007,
    /// Supports cooperative launch on
    ///< multiple devices with unmatched
    ///< grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 10008,
    /// Supports cooperative launch on
    ///< multiple devices with unmatched
    ///< block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 10009,
    /// Supports cooperative launch on
    ///< multiple devices with unmatched
    ///< shared memories
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 10010,
    /// Whether it is LargeBar
    hipDeviceAttributeIsLargeBar = 10011,
    /// Revision of the GPU in this device
    hipDeviceAttributeAsicRevision = 10012,
    /// '1' if Device supports hipStreamWaitValue32() and
    ///< hipStreamWaitValue64(), '0' otherwise.
    hipDeviceAttributeCanUseStreamWaitValue = 10013,
    /// '1' if Device supports image, '0' otherwise.
    hipDeviceAttributeImageSupport = 10014,
    /// All available physical compute
    ///< units for the device
    hipDeviceAttributePhysicalMultiProcessorCount = 10015,
    /// '1' if Device supports fine grain, '0' otherwise
    hipDeviceAttributeFineGrainSupport = 10016,
    /// Constant frequency of wall clock in kilohertz.
    hipDeviceAttributeWallClockRate = 10017,
    /// Number of XCC(s) in the device’s current
    ///< compute partition(XCP).
    ///< An XCC is AMD’s fundamental GPU compute tile.
    ///< Devices like MI300X include multiple XCCs
    ///< per package. Value varies by boot‑time
    ///< partition mode(compute or memory).
    ///< Partition modes can be set using amd‑smi.
    hipDeviceAttributeNumberOfXccs = 10018,
    /// Max number of available (directly or
    ///< indirectly addressable) VGPRs per thread in
    ///< DWORDs.
    hipDeviceAttributeMaxAvailableVgprsPerThread = 10019,
    /// GPU Manufacturer device id
    hipDeviceAttributePciChipId = 10020,
    /// '1' if Device supports expert scheduling mode,
    ///< '0' otherwise.
    hipDeviceAttributeExpertSchedMode = 10021,
    /// Maximum number of dynamic data prefetch regions
    ///< per kernel launch (0 if unsupported).
    hipDeviceAttributeMaxDynDataPrefetchRegions = 10022,
    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDriverProcAddressQueryResult {
    HIP_GET_PROC_ADDRESS_SUCCESS = 0,
    HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipComputeMode {
    hipComputeModeDefault = 0,
    hipComputeModeExclusive = 1,
    hipComputeModeProhibited = 2,
    hipComputeModeExclusiveProcess = 3,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipFlushGPUDirectRDMAWritesOptions {
    hipFlushGPUDirectRDMAWritesOptionHost = 1,
    hipFlushGPUDirectRDMAWritesOptionMemOps = 2,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGPUDirectRDMAWritesOrdering {
    hipGPUDirectRDMAWritesOrderingNone = 0,
    hipGPUDirectRDMAWritesOrderingOwner = 100,
    hipGPUDirectRDMAWritesOrderingAllDevices = 200,
}
#[repr(i32)]
/// The target of a hipDeviceFlushGPUDirectRDMAWrites operation.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipFlushGPUDirectRDMAWritesTarget {
    /// Memory of the current HIP device
    hipFlushGPUDirectRDMAWritesTargetCurrentDevice = 0,
}
#[repr(i32)]
/// The scope at which hipDeviceFlushGPUDirectRDMAWrites makes pending remote writes visible.
/// The enumerator values match hipGPUDirectRDMAWritesOrdering, so a device whose
/// hipDeviceAttributeGPUDirectRDMAWritesOrdering is greater than or equal to the requested
/// scope already orders those writes and needs no explicit flush.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipFlushGPUDirectRDMAWritesScope {
    /// Visible to the device owning the memory
    hipFlushGPUDirectRDMAWritesToOwner = 100,
    /// Visible to all HIP devices
    hipFlushGPUDirectRDMAWritesToAllDevices = 200,
}
///  @defgroup DriverTypes Driver Types
///  @{
///  This section describes the driver data types.
///
pub type hipDeviceptr_t = *mut ::core::ffi::c_void;
#[repr(i32)]
/// HIP channel format kinds
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipChannelFormatKind {
    /// Signed channel format
    hipChannelFormatKindSigned = 0,
    /// Unsigned channel format
    hipChannelFormatKindUnsigned = 1,
    /// Float channel format
    hipChannelFormatKindFloat = 2,
    /// No channel format
    hipChannelFormatKindNone = 3,
}
/// HIP channel format descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipChannelFormatDesc {
    pub x: ::core::ffi::c_int,
    pub y: ::core::ffi::c_int,
    pub z: ::core::ffi::c_int,
    pub w: ::core::ffi::c_int,
    /// Channel format kind
    pub f: hipChannelFormatKind,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct hipArray {
    _unused: [u8; 0],
}
pub type hipArray_t = *mut hipArray;
pub type hipArray_const_t = *const hipArray;
#[repr(i32)]
/// HIP array format
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipArray_Format {
    /// Unsigned 8-bit array format
    HIP_AD_FORMAT_UNSIGNED_INT8 = 1,
    /// Unsigned 16-bit array format
    HIP_AD_FORMAT_UNSIGNED_INT16 = 2,
    /// Unsigned 32-bit array format
    HIP_AD_FORMAT_UNSIGNED_INT32 = 3,
    /// Signed 8-bit array format
    HIP_AD_FORMAT_SIGNED_INT8 = 8,
    /// Signed 16-bit array format
    HIP_AD_FORMAT_SIGNED_INT16 = 9,
    /// Signed 32-bit array format
    HIP_AD_FORMAT_SIGNED_INT32 = 10,
    /// Half array format
    HIP_AD_FORMAT_HALF = 16,
    /// Float array format
    HIP_AD_FORMAT_FLOAT = 32,
}
/// HIP array descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_ARRAY_DESCRIPTOR {
    /// Width of the array
    pub Width: usize,
    /// Height of the array
    pub Height: usize,
    /// Format of the array
    pub Format: hipArray_Format,
    /// Number of channels of the array
    pub NumChannels: ::core::ffi::c_uint,
}
/// HIP 3D array descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_ARRAY3D_DESCRIPTOR {
    /// Width of the array
    pub Width: usize,
    /// Height of the array
    pub Height: usize,
    /// Depth of the array
    pub Depth: usize,
    /// Format of the array
    pub Format: hipArray_Format,
    /// Number of channels of the array
    pub NumChannels: ::core::ffi::c_uint,
    /// Flags of the array
    pub Flags: ::core::ffi::c_uint,
}
/// HIP 2D memory copy parameters
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hip_Memcpy2D {
    /// Source width in bytes
    pub srcXInBytes: usize,
    /// Source height
    pub srcY: usize,
    /// Source memory type
    pub srcMemoryType: hipMemoryType,
    /// Source pointer
    pub srcHost: *const ::core::ffi::c_void,
    /// Source device
    pub srcDevice: hipDeviceptr_t,
    /// Source array
    pub srcArray: hipArray_t,
    /// Source pitch
    pub srcPitch: usize,
    /// Destination width in bytes
    pub dstXInBytes: usize,
    /// Destination height
    pub dstY: usize,
    /// Destination memory type
    pub dstMemoryType: hipMemoryType,
    /// Destination pointer
    pub dstHost: *mut ::core::ffi::c_void,
    /// Destination device
    pub dstDevice: hipDeviceptr_t,
    /// Destination array
    pub dstArray: hipArray_t,
    /// Destination pitch
    pub dstPitch: usize,
    /// Width in bytes of the 2D memory copy
    pub WidthInBytes: usize,
    /// Height of the 2D memory copy
    pub Height: usize,
}
/// HIP mipmapped array
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMipmappedArray {
    /// Data pointer of the mipmapped array
    pub data: *mut ::core::ffi::c_void,
    /// Description of the mipmapped array
    pub desc: hipChannelFormatDesc,
    /// Type of the mipmapped array
    pub type_: ::core::ffi::c_uint,
    /// Width of the mipmapped array
    pub width: ::core::ffi::c_uint,
    /// Height of the mipmapped array
    pub height: ::core::ffi::c_uint,
    /// Depth of the mipmapped array
    pub depth: ::core::ffi::c_uint,
    /// Minimum level of the mipmapped array
    pub min_mipmap_level: ::core::ffi::c_uint,
    /// Maximum level of the mipmapped array
    pub max_mipmap_level: ::core::ffi::c_uint,
    /// Flags of the mipmapped array
    pub flags: ::core::ffi::c_uint,
    /// Format of the mipmapped array
    pub format: hipArray_Format,
    /// Number of channels of the mipmapped array
    pub num_channels: ::core::ffi::c_uint,
}
/// HIP mipmapped array pointer
pub type hipMipmappedArray_t = *mut hipMipmappedArray;
pub type hipmipmappedArray = hipMipmappedArray_t;
pub type hipMipmappedArray_const_t = *const hipMipmappedArray;
#[repr(i32)]
/// HIP resource types
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipResourceType {
    /// Array resource
    hipResourceTypeArray = 0,
    /// Mipmapped array resource
    hipResourceTypeMipmappedArray = 1,
    /// Linear resource
    hipResourceTypeLinear = 2,
    /// Pitch 2D resource
    hipResourceTypePitch2D = 3,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HIPresourcetype_enum {
    /// Array resource
    HIP_RESOURCE_TYPE_ARRAY = 0,
    /// Mipmapped array resource
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    /// Linear resource
    HIP_RESOURCE_TYPE_LINEAR = 2,
    /// Pitch 2D resource
    HIP_RESOURCE_TYPE_PITCH2D = 3,
}
pub use self::HIPresourcetype_enum as HIPresourcetype;
pub use self::HIPresourcetype_enum as hipResourcetype;
#[repr(i32)]
/// HIP texture address modes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HIPaddress_mode_enum {
    /// Wrap address mode
    HIP_TR_ADDRESS_MODE_WRAP = 0,
    /// Clamp address mode
    HIP_TR_ADDRESS_MODE_CLAMP = 1,
    /// Mirror address mode
    HIP_TR_ADDRESS_MODE_MIRROR = 2,
    /// Border address mode
    HIP_TR_ADDRESS_MODE_BORDER = 3,
}
/// HIP texture address modes
pub use self::HIPaddress_mode_enum as HIPaddress_mode;
#[repr(i32)]
/// HIP filter modes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HIPfilter_mode_enum {
    /// Filter mode point
    HIP_TR_FILTER_MODE_POINT = 0,
    /// Filter mode linear
    HIP_TR_FILTER_MODE_LINEAR = 1,
}
/// HIP filter modes
pub use self::HIPfilter_mode_enum as HIPfilter_mode;
/// HIP texture descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_TEXTURE_DESC_st {
    /// Address modes
    pub addressMode: [HIPaddress_mode; 3usize],
    /// Filter mode
    pub filterMode: HIPfilter_mode,
    /// Flags
    pub flags: ::core::ffi::c_uint,
    /// Maximum anisotropy ratio
    pub maxAnisotropy: ::core::ffi::c_uint,
    /// Mipmap filter mode
    pub mipmapFilterMode: HIPfilter_mode,
    /// Mipmap level bias
    pub mipmapLevelBias: f32,
    /// Mipmap minimum level clamp
    pub minMipmapLevelClamp: f32,
    /// Mipmap maximum level clamp
    pub maxMipmapLevelClamp: f32,
    /// Border Color
    pub borderColor: [f32; 4usize],
    pub reserved: [::core::ffi::c_int; 12usize],
}
/// HIP texture descriptor
pub type HIP_TEXTURE_DESC = HIP_TEXTURE_DESC_st;
#[repr(i32)]
/// HIP texture resource view formats
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipResourceViewFormat {
    /// No resource view format (use underlying resource format)
    hipResViewFormatNone = 0,
    /// 1 channel, unsigned 8-bit integers
    hipResViewFormatUnsignedChar1 = 1,
    /// 2 channels, unsigned 8-bit integers
    hipResViewFormatUnsignedChar2 = 2,
    /// 4 channels, unsigned 8-bit integers
    hipResViewFormatUnsignedChar4 = 3,
    /// 1 channel, signed 8-bit integers
    hipResViewFormatSignedChar1 = 4,
    /// 2 channels, signed 8-bit integers
    hipResViewFormatSignedChar2 = 5,
    /// 4 channels, signed 8-bit integers
    hipResViewFormatSignedChar4 = 6,
    /// 1 channel, unsigned 16-bit integers
    hipResViewFormatUnsignedShort1 = 7,
    /// 2 channels, unsigned 16-bit integers
    hipResViewFormatUnsignedShort2 = 8,
    /// 4 channels, unsigned 16-bit integers
    hipResViewFormatUnsignedShort4 = 9,
    /// 1 channel, signed 16-bit integers
    hipResViewFormatSignedShort1 = 10,
    /// 2 channels, signed 16-bit integers
    hipResViewFormatSignedShort2 = 11,
    /// 4 channels, signed 16-bit integers
    hipResViewFormatSignedShort4 = 12,
    /// 1 channel, unsigned 32-bit integers
    hipResViewFormatUnsignedInt1 = 13,
    /// 2 channels, unsigned 32-bit integers
    hipResViewFormatUnsignedInt2 = 14,
    /// 4 channels, unsigned 32-bit integers
    hipResViewFormatUnsignedInt4 = 15,
    /// 1 channel, signed 32-bit integers
    hipResViewFormatSignedInt1 = 16,
    /// 2 channels, signed 32-bit integers
    hipResViewFormatSignedInt2 = 17,
    /// 4 channels, signed 32-bit integers
    hipResViewFormatSignedInt4 = 18,
    /// 1 channel, 16-bit floating point
    hipResViewFormatHalf1 = 19,
    /// 2 channels, 16-bit floating point
    hipResViewFormatHalf2 = 20,
    /// 4 channels, 16-bit floating point
    hipResViewFormatHalf4 = 21,
    /// 1 channel, 32-bit floating point
    hipResViewFormatFloat1 = 22,
    /// 2 channels, 32-bit floating point
    hipResViewFormatFloat2 = 23,
    /// 4 channels, 32-bit floating point
    hipResViewFormatFloat4 = 24,
    /// Block-compressed 1
    hipResViewFormatUnsignedBlockCompressed1 = 25,
    /// Block-compressed 2
    hipResViewFormatUnsignedBlockCompressed2 = 26,
    /// Block-compressed 3
    hipResViewFormatUnsignedBlockCompressed3 = 27,
    /// Block-compressed 4 unsigned
    hipResViewFormatUnsignedBlockCompressed4 = 28,
    /// Block-compressed 4 signed
    hipResViewFormatSignedBlockCompressed4 = 29,
    /// Block-compressed 5 unsigned
    hipResViewFormatUnsignedBlockCompressed5 = 30,
    /// Block-compressed 5 signed
    hipResViewFormatSignedBlockCompressed5 = 31,
    /// Block-compressed 6 unsigned half-float
    hipResViewFormatUnsignedBlockCompressed6H = 32,
    /// Block-compressed 6 signed half-float
    hipResViewFormatSignedBlockCompressed6H = 33,
    /// Block-compressed 7
    hipResViewFormatUnsignedBlockCompressed7 = 34,
}
#[repr(i32)]
/// HIP texture resource view formats
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HIPresourceViewFormat_enum {
    /// No resource view format (use underlying resource format)
    HIP_RES_VIEW_FORMAT_NONE = 0,
    /// 1 channel, unsigned 8-bit integers
    HIP_RES_VIEW_FORMAT_UINT_1X8 = 1,
    /// 2 channels, unsigned 8-bit integers
    HIP_RES_VIEW_FORMAT_UINT_2X8 = 2,
    /// 4 channels, unsigned 8-bit integers
    HIP_RES_VIEW_FORMAT_UINT_4X8 = 3,
    /// 1 channel, signed 8-bit integers
    HIP_RES_VIEW_FORMAT_SINT_1X8 = 4,
    /// 2 channels, signed 8-bit integers
    HIP_RES_VIEW_FORMAT_SINT_2X8 = 5,
    /// 4 channels, signed 8-bit integers
    HIP_RES_VIEW_FORMAT_SINT_4X8 = 6,
    /// 1 channel, unsigned 16-bit integers
    HIP_RES_VIEW_FORMAT_UINT_1X16 = 7,
    /// 2 channels, unsigned 16-bit integers
    HIP_RES_VIEW_FORMAT_UINT_2X16 = 8,
    /// 4 channels, unsigned 16-bit integers
    HIP_RES_VIEW_FORMAT_UINT_4X16 = 9,
    /// 1 channel, signed 16-bit integers
    HIP_RES_VIEW_FORMAT_SINT_1X16 = 10,
    /// 2 channels, signed 16-bit integers
    HIP_RES_VIEW_FORMAT_SINT_2X16 = 11,
    /// 4 channels, signed 16-bit integers
    HIP_RES_VIEW_FORMAT_SINT_4X16 = 12,
    /// 1 channel, unsigned 32-bit integers
    HIP_RES_VIEW_FORMAT_UINT_1X32 = 13,
    /// 2 channels, unsigned 32-bit integers
    HIP_RES_VIEW_FORMAT_UINT_2X32 = 14,
    /// 4 channels, unsigned 32-bit integers
    HIP_RES_VIEW_FORMAT_UINT_4X32 = 15,
    /// 1 channel, signed 32-bit integers
    HIP_RES_VIEW_FORMAT_SINT_1X32 = 16,
    /// 2 channels, signed 32-bit integers
    HIP_RES_VIEW_FORMAT_SINT_2X32 = 17,
    /// 4 channels, signed 32-bit integers
    HIP_RES_VIEW_FORMAT_SINT_4X32 = 18,
    /// 1 channel, 16-bit floating point
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19,
    /// 2 channels, 16-bit floating point
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20,
    /// 4 channels, 16-bit floating point
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21,
    /// 1 channel, 32-bit floating point
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22,
    /// 2 channels, 32-bit floating point
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23,
    /// 4 channels, 32-bit floating point
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24,
    /// Block-compressed 1
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25,
    /// Block-compressed 2
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26,
    /// Block-compressed 3
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27,
    /// Block-compressed 4 unsigned
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28,
    /// Block-compressed 4 signed
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29,
    /// Block-compressed 5 unsigned
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30,
    /// Block-compressed 5 signed
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31,
    /// Block-compressed 6 unsigned half-float
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32,
    /// Block-compressed 6 signed half-float
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33,
    /// Block-compressed 7
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34,
}
/// HIP texture resource view formats
pub use self::HIPresourceViewFormat_enum as HIPresourceViewFormat;
/// HIP resource descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipResourceDesc {
    /// Resource type
    pub resType: hipResourceType,
    pub res: hipResourceDesc__bindgen_ty_1,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipResourceDesc__bindgen_ty_1 {
    pub array: hipResourceDesc__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: hipResourceDesc__bindgen_ty_1__bindgen_ty_2,
    pub linear: hipResourceDesc__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: hipResourceDesc__bindgen_ty_1__bindgen_ty_4,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipResourceDesc__bindgen_ty_1__bindgen_ty_1 {
    /// HIP array
    pub array: hipArray_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipResourceDesc__bindgen_ty_1__bindgen_ty_2 {
    /// HIP mipmapped array
    pub mipmap: hipMipmappedArray_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipResourceDesc__bindgen_ty_1__bindgen_ty_3 {
    /// Device pointer
    pub devPtr: *mut ::core::ffi::c_void,
    /// Channel format description
    pub desc: hipChannelFormatDesc,
    /// Size in bytes
    pub sizeInBytes: usize,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipResourceDesc__bindgen_ty_1__bindgen_ty_4 {
    /// Device pointer
    pub devPtr: *mut ::core::ffi::c_void,
    /// Channel format description
    pub desc: hipChannelFormatDesc,
    /// Width of the array in elements
    pub width: usize,
    /// Height of the array in elements
    pub height: usize,
    /// Pitch between two rows in bytes
    pub pitchInBytes: usize,
}
/// HIP resource view descriptor struct
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct HIP_RESOURCE_DESC_st {
    /// Resource type
    pub resType: HIPresourcetype,
    pub res: HIP_RESOURCE_DESC_st__bindgen_ty_1,
    /// Flags (must be zero)
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union HIP_RESOURCE_DESC_st__bindgen_ty_1 {
    pub array: HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2,
    pub linear: HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    pub reserved: HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    /// HIP array
    pub hArray: hipArray_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2 {
    /// HIP mipmapped array
    pub hMipmappedArray: hipMipmappedArray_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
    /// Device pointer
    pub devPtr: hipDeviceptr_t,
    /// Array format
    pub format: hipArray_Format,
    /// Channels per array element
    pub numChannels: ::core::ffi::c_uint,
    /// Size in bytes
    pub sizeInBytes: usize,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
    /// Device pointer
    pub devPtr: hipDeviceptr_t,
    /// Array format
    pub format: hipArray_Format,
    /// Channels per array element
    pub numChannels: ::core::ffi::c_uint,
    /// Width of the array in elements
    pub width: usize,
    /// Height of the array in elements
    pub height: usize,
    /// Pitch between two rows in bytes
    pub pitchInBytes: usize,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5 {
    pub reserved: [::core::ffi::c_int; 32usize],
}
/// HIP resource view descriptor struct
pub type HIP_RESOURCE_DESC = HIP_RESOURCE_DESC_st;
/// HIP resource view descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipResourceViewDesc {
    /// Resource view format
    pub format: hipResourceViewFormat,
    /// Width of the resource view
    pub width: usize,
    /// Height of the resource view
    pub height: usize,
    /// Depth of the resource view
    pub depth: usize,
    /// First defined mipmap level
    pub firstMipmapLevel: ::core::ffi::c_uint,
    /// Last defined mipmap level
    pub lastMipmapLevel: ::core::ffi::c_uint,
    /// First layer index
    pub firstLayer: ::core::ffi::c_uint,
    /// Last layer index
    pub lastLayer: ::core::ffi::c_uint,
}
/// Resource view descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_RESOURCE_VIEW_DESC_st {
    /// Resource view format
    pub format: HIPresourceViewFormat,
    /// Width of the resource view
    pub width: usize,
    /// Height of the resource view
    pub height: usize,
    /// Depth of the resource view
    pub depth: usize,
    /// First defined mipmap level
    pub firstMipmapLevel: ::core::ffi::c_uint,
    /// Last defined mipmap level
    pub lastMipmapLevel: ::core::ffi::c_uint,
    /// First layer index
    pub firstLayer: ::core::ffi::c_uint,
    /// Last layer index
    pub lastLayer: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
/// Resource view descriptor
pub type HIP_RESOURCE_VIEW_DESC = HIP_RESOURCE_VIEW_DESC_st;
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemcpyKind {
    /// Host-to-Host Copy
    hipMemcpyHostToHost = 0,
    /// Host-to-Device Copy
    hipMemcpyHostToDevice = 1,
    /// Device-to-Host Copy
    hipMemcpyDeviceToHost = 2,
    /// Device-to-Device Copy
    hipMemcpyDeviceToDevice = 3,
    /// Runtime will automatically determine
    ///< copy-kind based on virtual addresses.
    hipMemcpyDefault = 4,
    /// Device-to-Device Copy without using compute units
    hipMemcpyDeviceToDeviceNoCU = 1024,
}
/// HIP pithed pointer
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipPitchedPtr {
    /// Pointer to the allocated memory
    pub ptr: *mut ::core::ffi::c_void,
    /// Pitch in bytes
    pub pitch: usize,
    /// Logical size of the first dimension of allocation in elements
    pub xsize: usize,
    /// Logical size of the second dimension of allocation in elements
    pub ysize: usize,
}
/// HIP extent
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}
///  HIP position
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipPos {
    /// X coordinate
    pub x: usize,
    /// Y coordinate
    pub y: usize,
    /// Z coordinate
    pub z: usize,
}
/// HIP 3D memory copy parameters
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemcpy3DParms {
    /// Source array
    pub srcArray: hipArray_t,
    /// Source position
    pub srcPos: hipPos,
    /// Source pointer
    pub srcPtr: hipPitchedPtr,
    /// Destination array
    pub dstArray: hipArray_t,
    /// Destination position
    pub dstPos: hipPos,
    /// Destination pointer
    pub dstPtr: hipPitchedPtr,
    /// Extent of 3D memory copy
    pub extent: hipExtent,
    /// Kind of 3D memory copy
    pub kind: hipMemcpyKind,
}
/// HIP 3D memory copy
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_MEMCPY3D {
    /// Source X in bytes
    pub srcXInBytes: usize,
    /// Source Y
    pub srcY: usize,
    /// Source Z
    pub srcZ: usize,
    /// Source LOD
    pub srcLOD: usize,
    /// Source memory type
    pub srcMemoryType: hipMemoryType,
    /// Source host pointer
    pub srcHost: *const ::core::ffi::c_void,
    /// Source device
    pub srcDevice: hipDeviceptr_t,
    /// Source array
    pub srcArray: hipArray_t,
    /// Source pitch
    pub srcPitch: usize,
    /// Source height
    pub srcHeight: usize,
    /// Destination X in bytes
    pub dstXInBytes: usize,
    /// Destination Y
    pub dstY: usize,
    /// Destination Z
    pub dstZ: usize,
    /// Destination LOD
    pub dstLOD: usize,
    /// Destination memory type
    pub dstMemoryType: hipMemoryType,
    /// Destination host pointer
    pub dstHost: *mut ::core::ffi::c_void,
    /// Destination device
    pub dstDevice: hipDeviceptr_t,
    /// Destination array
    pub dstArray: hipArray_t,
    /// Destination pitch
    pub dstPitch: usize,
    /// Destination height
    pub dstHeight: usize,
    /// Width in bytes of 3D memory copy
    pub WidthInBytes: usize,
    /// Height in bytes of 3D memory copy
    pub Height: usize,
    /// Depth in bytes of 3D memory copy
    pub Depth: usize,
}
impl hipMemLocationType {
    pub const hipMemLocationTypeNone: hipMemLocationType =
        hipMemLocationType::hipMemLocationTypeInvalid;
}
#[repr(i32)]
/// Specifies the type of location
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemLocationType {
    hipMemLocationTypeInvalid = 0,
    /// Device location, thus it's HIP device ID
    hipMemLocationTypeDevice = 1,
    /// Host location, id is ignored
    hipMemLocationTypeHost = 2,
    /// Host NUMA node location, id is host NUMA node id
    hipMemLocationTypeHostNuma = 3,
    hipMemLocationTypeHostNumaCurrent = 4,
}
/// Specifies a memory location.
/// To specify a gpu, set type = @p hipMemLocationTypeDevice and set id = the gpu's device ID
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemLocation {
    /// Specifies the location type, which describes the meaning of id
    pub type_: hipMemLocationType,
    /// Identifier for the provided location type @p hipMemLocationType
    pub id: ::core::ffi::c_int,
}
#[repr(i32)]
/// Flags to specify for copies within a batch. Used with hipMemcpyBatchAsync
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemcpyFlags {
    /// Default flag
    hipMemcpyFlagDefault = 0,
    /// Tries to overlap copy with compute work.
    hipMemcpyFlagPreferOverlapWithCompute = 1,
    /// Prefer copy engine over compute engine.
    hipMemcpyFlagExtPreferCE = 256,
    /// Swap contents of src and dst.
    hipMemcpyFlagExtOpSwap = 512,
    /// The src pointer holds the address of the real
    ///< source pointer, read when the copy runs rather
    ///< than when it is submitted.
    hipMemcpyFlagExtOpIndirectSrc = 1024,
    /// Same as IndirectSrc, but for the dst pointer.
    hipMemcpyFlagExtOpIndirectDst = 2048,
    /// Prefer compute engine over copy engine.
    ///< Linear copies only.
    ///< hipExtMemcpyBatchAsync only.
    hipMemcpyFlagExtPreferComputeEngine = 4096,
    /// Prefer issuing each copy individually.
    ///< hipExtMemcpyBatchAsync only.
    hipMemcpyFlagExtPreferLinear = 8192,
    /// Prefer one broadcast for copies with the same
    ///< src and size. Linear copies only.
    ///< hipExtMemcpyBatchAsync only.
    hipMemcpyFlagExtPreferBroadcast = 16384,
}
#[repr(i32)]
/// Comparison function for a per-entry wait in hipExtMemcpyBatchAsync.
/// Numbered to match the underlying HSA wait function.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipExtMemcpyWaitOp {
    /// No wait; addr must be NULL
    hipExtMemcpyWaitNone = 0,
    /// *addr <  value
    hipExtMemcpyWaitLt = 1,
    /// *addr <= value
    hipExtMemcpyWaitLe = 2,
    /// *addr == value
    hipExtMemcpyWaitEq = 3,
    /// *addr != value
    hipExtMemcpyWaitNe = 4,
    /// *addr >= value
    hipExtMemcpyWaitGe = 5,
    /// *addr >  value
    hipExtMemcpyWaitGt = 6,
}
/// Per-entry wait condition for hipExtMemcpyBatchAsync. Reserved for future use;
/// layout may change before support is enabled.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExtMemcpyWait {
    /// GPU address to poll; NULL iff compareOp is None
    pub addr: *mut ::core::ffi::c_void,
    /// Reference value for comparison
    pub value: u64,
    /// AND mask applied before comparison (0 = full 64-bit word)
    pub mask: u64,
    /// Comparison function; see hipExtMemcpyWaitOp
    pub compareOp: u32,
    /// Reserved; must be 0
    pub reserved0: u32,
    /// Reserved; must be 0 (future use)
    pub reserved: [u64; 4usize],
}
#[repr(i32)]
/// Atomic operation for a per-entry signal in hipExtMemcpyBatchAsync.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipExtMemcpySignalOp {
    /// No signal; addr must be NULL
    hipExtMemcpySignalNone = 0,
    /// *addr  = data
    hipExtMemcpySignalWrite = 1,
    /// *addr += data
    hipExtMemcpySignalAdd = 2,
    /// *addr -= data
    hipExtMemcpySignalSub = 3,
}
/// Per-entry signal-after-copy for hipExtMemcpyBatchAsync. Reserved for future use;
/// layout may change before support is enabled.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExtMemcpySignal {
    /// GPU address to signal; NULL iff signalOp is None
    pub addr: *mut ::core::ffi::c_void,
    /// Data value for the atomic operation
    pub data: u64,
    /// Atomic operation; see hipExtMemcpySignalOp
    pub signalOp: u32,
    /// Reserved; must be 0
    pub reserved0: u32,
    /// Reserved; must be 0 (future use)
    pub reserved: [u64; 5usize],
}
#[repr(i32)]
/// Flags to specify order in which source pointer is accessed by Batch memcpy
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemcpySrcAccessOrder {
    /// Default Invalid.
    hipMemcpySrcAccessOrderInvalid = 0,
    /// Access to source pointer must be in stream order.
    hipMemcpySrcAccessOrderStream = 1,
    hipMemcpySrcAccessOrderDuringApiCall = 2,
    hipMemcpySrcAccessOrderAny = 3,
    hipMemcpySrcAccessOrderMax = 2147483647,
}
/// Attributes for copies within a batch.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemcpyAttributes {
    /// Source access ordering to be observed for copies with this attribute.
    pub srcAccessOrder: hipMemcpySrcAccessOrder,
    /// Location hint for src operand.
    pub srcLocHint: hipMemLocation,
    /// Location hint for destination operand.
    pub dstLocHint: hipMemLocation,
    /// Additional Flags for copies. See hipMemcpyFlags.
    pub flags: ::core::ffi::c_uint,
}
/// Attributes for copies within a batch submitted through hipExtMemcpyBatchAsync.
/// hipMemcpyAttributes plus a reserved tail; do not cast between the two.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExtMemcpyAttributes {
    /// Source access ordering; see hipMemcpySrcAccessOrder.
    pub srcAccessOrder: hipMemcpySrcAccessOrder,
    /// Location hint for src operand.
    pub srcLocHint: hipMemLocation,
    /// Location hint for destination operand.
    pub dstLocHint: hipMemLocation,
    /// Op selection and hints; see hipMemcpyFlags.
    pub flags: ::core::ffi::c_uint,
    /// Reserved; must be 0 (future use).
    pub reserved: [u32; 8usize],
}
#[repr(i32)]
/// Operand types for individual copies within a batch
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemcpy3DOperandType {
    /// Mempcy operand is a valid pointer.
    hipMemcpyOperandTypePointer = 1,
    /// Memcpy operand is a valid hipArray.
    hipMemcpyOperandTypeArray = 2,
    hipMemcpyOperandTypeMax = 2147483647,
}
/// Struct representing offset into a hipArray_t in elements.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipOffset3D {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
///  Struct representing an operand for copy with hipMemcpy3DBatchAsync.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipMemcpy3DOperand {
    pub type_: hipMemcpy3DOperandType,
    pub op: hipMemcpy3DOperand__bindgen_ty_1,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipMemcpy3DOperand__bindgen_ty_1 {
    pub ptr: hipMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1,
    pub array: hipMemcpy3DOperand__bindgen_ty_1__bindgen_ty_2,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1 {
    pub ptr: *mut ::core::ffi::c_void,
    /// Length of each row in elements.
    pub rowLength: usize,
    /// Height of each layer in elements.
    pub layerHeight: usize,
    /// Location Hint for the operand.
    pub locHint: hipMemLocation,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemcpy3DOperand__bindgen_ty_1__bindgen_ty_2 {
    /// Array struct for hipMemcpyOperandTypeArray.
    pub array: hipArray_t,
    /// Offset into array in elements.
    pub offset: hipOffset3D,
}
/// HIP 3D Batch Op
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipMemcpy3DBatchOp {
    pub src: hipMemcpy3DOperand,
    pub dst: hipMemcpy3DOperand,
    pub extent: hipExtent,
    pub srcAccessOrder: hipMemcpySrcAccessOrder,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemcpy3DPeerParms {
    /// Source memory address
    pub srcArray: hipArray_t,
    /// Source position offset
    pub srcPos: hipPos,
    /// Pitched source memory address
    pub srcPtr: hipPitchedPtr,
    /// Source device
    pub srcDevice: ::core::ffi::c_int,
    /// Destination memory address
    pub dstArray: hipArray_t,
    /// Destination position offset
    pub dstPos: hipPos,
    /// Pitched destination memory address
    pub dstPtr: hipPitchedPtr,
    /// Destination device
    pub dstDevice: ::core::ffi::c_int,
    /// Requested memory copy size
    pub extent: hipExtent,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipFunction_attribute {
    /// The maximum number of threads per block. Depends
    ///< on function and device.
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    /// The statically allocated shared memory size in bytes
    ///< per block required by the function.
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    /// The user-allocated constant memory by the function in
    ///< bytes.
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    /// The local memory usage of each thread by this function
    ///< in bytes.
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    /// The number of registers used by each thread of this function.
    HIP_FUNC_ATTRIBUTE_NUM_REGS = 4,
    /// PTX version
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    /// Binary version
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    /// Cache mode
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    /// The maximum dynamic shared memory per
    ///< block for this function in bytes.
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    /// The shared memory carveout preference
    ///< in percent of the maximum shared
    ///< memory.
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    /// the kernel must launch with a valid cluster size specified.
    HIP_FUNC_ATTRIBUTE_CLUSTER_DIM_MUST_BE_SET = 10,
    /// The required cluster width in blocks
    HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    /// The required cluster height in blocks
    HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    /// The required cluster depth in blocks
    HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    /// Is the function allowed to launch with non-portable cluster size.
    HIP_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    /// The block scheduling policy of a function.
    HIP_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
    HIP_FUNC_ATTRIBUTE_MAX = 16,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipPointer_attribute {
    /// The context on which a pointer was allocated
    ///< @warning This attribute is not supported in HIP
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1,
    /// memory type describing the location of a pointer
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    /// address at which the pointer is allocated on the
    ///< device
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    /// address at which the pointer is allocated on the host
    HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    /// A pair of tokens for use with Linux kernel interface
    ///< @warning This attribute is not supported in HIP
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    /// Synchronize every synchronous memory operation
    ///< initiated on this region
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    /// Unique ID for an allocated memory region
    HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    /// Indicates if the pointer points to managed memory
    HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    /// device ordinal of a device on which a pointer
    ///< was allocated or registered
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    /// if this pointer maps to an allocation
    ///< that is suitable for hipIpcGetMemHandle
    ///< @warning This attribute is not supported in
    ///< HIP
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10,
    /// Starting address for this requested pointer
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    /// Size of the address range for this requested pointer
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    /// tells if this pointer is in a valid address range
    ///< that is mapped to a backing allocation
    HIP_POINTER_ATTRIBUTE_MAPPED = 13,
    /// Bitmask of allowed hipmemAllocationHandleType
    ///< for this allocation @warning This attribute is
    ///< not supported in HIP
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    /// returns if the memory referenced by
    ///< this pointer can be used with the
    ///< GPUDirect RDMA API
    ///< @warning This attribute is not supported
    ///< in HIP
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    /// Returns the access flags the device associated with
    ///< for the corresponding memory referenced by the ptr
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    /// Returns the mempool handle for the allocation if
    ///< it was allocated from a mempool
    ///< @warning This attribute is not supported in HIP
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
}
/// An opaque value that represents a hip texture object
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct __hip_texture {
    _unused: [u8; 0],
}
pub type hipTextureObject_t = *mut __hip_texture;
#[repr(i32)]
/// hip texture address modes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipTextureAddressMode {
    hipAddressModeWrap = 0,
    hipAddressModeClamp = 1,
    hipAddressModeMirror = 2,
    hipAddressModeBorder = 3,
}
#[repr(i32)]
/// hip texture filter modes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipTextureFilterMode {
    hipFilterModePoint = 0,
    hipFilterModeLinear = 1,
}
#[repr(i32)]
/// hip texture read modes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipTextureReadMode {
    hipReadModeElementType = 0,
    hipReadModeNormalizedFloat = 1,
}
/// hip texture reference
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct textureReference {
    pub normalized: ::core::ffi::c_int,
    pub readMode: hipTextureReadMode,
    pub filterMode: hipTextureFilterMode,
    pub addressMode: [hipTextureAddressMode; 3usize],
    pub channelDesc: hipChannelFormatDesc,
    pub sRGB: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: hipTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub textureObject: hipTextureObject_t,
    pub numChannels: ::core::ffi::c_int,
    pub format: hipArray_Format,
}
/// hip texture descriptor
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipTextureDesc {
    pub addressMode: [hipTextureAddressMode; 3usize],
    pub filterMode: hipTextureFilterMode,
    pub readMode: hipTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: hipTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
}
pub type hipTexRef = *mut textureReference;
/// An opaque value that represents a hip surface object
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct __hip_surface {
    _unused: [u8; 0],
}
pub type hipSurfaceObject_t = *mut __hip_surface;
/// hip surface reference
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct surfaceReference {
    pub surfaceObject: hipSurfaceObject_t,
}
#[repr(i32)]
/// hip surface boundary modes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipSurfaceBoundaryMode {
    hipBoundaryModeZero = 0,
    hipBoundaryModeTrap = 1,
    hipBoundaryModeClamp = 2,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipCtx_t {
    _unused: [u8; 0],
}
pub type hipCtx_t = *mut ihipCtx_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipExecutionCtx_t {
    _unused: [u8; 0],
}
pub type hipExecutionCtx_t = *mut ihipExecutionCtx_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipDevResourceDesc_t {
    _unused: [u8; 0],
}
pub type hipDevResourceDesc_t = *mut ihipDevResourceDesc_t;
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDevResourceType {
    hipDevResourceTypeInvalid = 0,
    hipDevResourceTypeSm = 1,
    hipDevResourceTypeWorkqueueConfig = 1000,
    hipDevResourceTypeWorkqueue = 10000,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDevSmResourceGroup_flags {
    hipDevSmResourceGroupDefault = 0,
    hipDevSmResourceGroupBackfill = 1,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDevSmResourceSplitByCount_flags {
    hipDevSmResourceSplitIgnoreSmCoscheduling = 1,
    hipDevSmResourceSplitMaxPotentialClusterSize = 2,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDevWorkqueueConfigScope {
    hipDevWorkqueueConfigScopeDeviceCtx = 0,
    hipDevWorkqueueConfigScopeGreenCtxBalanced = 1,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipDevSmResource {
    pub smCount: ::core::ffi::c_uint,
    pub minSmPartitionSize: ::core::ffi::c_uint,
    pub smCoscheduledAlignment: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipDevWorkqueueConfigResource {
    pub device: ::core::ffi::c_int,
    pub wqConcurrencyLimit: ::core::ffi::c_uint,
    pub sharingScope: hipDevWorkqueueConfigScope,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipDevWorkqueueResource {
    pub reserved: [::core::ffi::c_uchar; 40usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipDevResource_st {
    pub type_: hipDevResourceType,
    pub _internal_padding: [::core::ffi::c_uchar; 92usize],
    pub __bindgen_anon_1: hipDevResource_st__bindgen_ty_1,
    pub nextResource: *mut hipDevResource_st,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipDevResource_st__bindgen_ty_1 {
    pub sm: hipDevSmResource,
    pub wqConfig: hipDevWorkqueueConfigResource,
    pub wq: hipDevWorkqueueResource,
    pub _oversize: [::core::ffi::c_uchar; 40usize],
}
pub type hipDevResource = hipDevResource_st;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipDevSmResourceGroupParams_st {
    pub smCount: ::core::ffi::c_uint,
    pub coscheduledSmCount: ::core::ffi::c_uint,
    pub preferredCoscheduledSmCount: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
pub type hipDevSmResourceGroupParams = hipDevSmResourceGroupParams_st;
pub type hipDevice_t = ::core::ffi::c_int;
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDeviceP2PAttr {
    hipDevP2PAttrPerformanceRank = 0,
    hipDevP2PAttrAccessSupported = 1,
    hipDevP2PAttrNativeAtomicSupported = 2,
    hipDevP2PAttrHipArrayAccessSupported = 3,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDriverEntryPointQueryResult {
    hipDriverEntryPointSuccess = 0,
    hipDriverEntryPointSymbolNotFound = 1,
    hipDriverEntryPointVersionNotSufficent = 2,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipStream_t {
    _unused: [u8; 0],
}
pub type hipStream_t = *mut ihipStream_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipIpcMemHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
pub type hipIpcMemHandle_t = hipIpcMemHandle_st;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipIpcEventHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
pub type hipIpcEventHandle_t = hipIpcEventHandle_st;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemFabricHandle_st {
    pub data: [::core::ffi::c_uchar; 64usize],
}
pub type hipMemFabricHandle_t = hipMemFabricHandle_st;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipModule_t {
    _unused: [u8; 0],
}
pub type hipModule_t = *mut ihipModule_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipModuleSymbol_t {
    _unused: [u8; 0],
}
pub type hipFunction_t = *mut ihipModuleSymbol_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipLinkState_t {
    _unused: [u8; 0],
}
pub type hipLinkState_t = *mut ihipLinkState_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipLibrary_t {
    _unused: [u8; 0],
}
pub type hipLibrary_t = *mut ihipLibrary_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipKernel_t {
    _unused: [u8; 0],
}
pub type hipKernel_t = *mut ihipKernel_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipMemPoolHandle_t {
    _unused: [u8; 0],
}
/// HIP memory pool
pub type hipMemPool_t = *mut ihipMemPoolHandle_t;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipFuncAttributes {
    pub binaryVersion: ::core::ffi::c_int,
    pub cacheModeCA: ::core::ffi::c_int,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxDynamicSharedSizeBytes: ::core::ffi::c_int,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub numRegs: ::core::ffi::c_int,
    pub preferredShmemCarveout: ::core::ffi::c_int,
    pub ptxVersion: ::core::ffi::c_int,
    pub sharedSizeBytes: usize,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipEvent_t {
    _unused: [u8; 0],
}
pub type hipEvent_t = *mut ihipEvent_t;
#[repr(i32)]
/// hipLimit
/// In HIP device limit-related APIs, any input limit value other than those defined in the
/// enum is treated as \"UnsupportedLimit\" by default.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipLimit_t {
    /// Limit of stack size in bytes on the current device, per
    ///< thread. The size is in units of 256 dwords, up to the
    ///< limit of (128K - 16)
    hipLimitStackSize = 0,
    /// Size limit in bytes of fifo used by printf call on the
    ///< device. Currently not supported
    hipLimitPrintfFifoSize = 1,
    /// Limit of heap size in bytes on the current device, should
    ///< be less than the global memory size on the device
    hipLimitMallocHeapSize = 2,
    /// Requested persisting L2 cache size in bytes for the
    ///< current device. This size can be set to a value up to
    ///< persistingL2CacheMaxSize reported in hipDeviceProp_t.
    hipLimitPersistingL2CacheSize = 3,
    /// Minimum allowed value in bytes for scratch limit on this
    ///< device. Valid only on Rocm device. This is read only.
    hipExtLimitScratchMin = 4096,
    /// Maximum allowed value in bytes for scratch limit on this
    ///< device. Valid only on Rocm device. This is read only.
    hipExtLimitScratchMax = 4097,
    /// Current scratch limit threshold in bytes on this
    ///< device. Must be between hipExtLimitScratchMin and
    ///< hipExtLimitScratchMaxValid values. Valid only on Rocm
    ///< device. This can be modified.
    hipExtLimitScratchCurrent = 4098,
    /// Supported limit range
    hipLimitRange = 4099,
}
#[repr(i32)]
/// Operations for hipStreamBatchMemOp
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipStreamBatchMemOpType {
    hipStreamMemOpWaitValue32 = 1,
    hipStreamMemOpWriteValue32 = 2,
    hipStreamMemOpWaitValue64 = 4,
    hipStreamMemOpWriteValue64 = 5,
    /// Currently not supported
    hipStreamMemOpBarrier = 6,
    /// Currently not supported
    hipStreamMemOpFlushRemoteWrites = 3,
}
/// @brief Union representing batch memory operation parameters for HIP streams.
/// hipStreamBatchMemOpParams is used to specify the parameters for batch memory
/// operations in a HIP stream. This union supports various operations including
/// waiting for a specific value, writing a value, and different flags for wait conditions.
/// @details
/// The union includes fields for different types of operations defined in the
/// enum hipStreamBatchMemOpType:
/// - hipStreamMemOpWaitValue32:  Wait for a 32-bit value.
/// - hipStreamMemOpWriteValue32: Write a 32-bit value.
/// - hipStreamMemOpWaitValue64:  Wait for a 64-bit value.
/// - hipStreamMemOpWriteValue64: Write a 64-bit value.
/// Each operation type includes an address, the value to wait for or write, flags, and an
/// optional alias that is not relevant on AMD GPUs. Flags can be used to specify different
/// wait conditions such as equality, bitwise AND, greater than or equal, and bitwise NOR.
/// Example usage:
/// @code
/// hipStreamBatchMemOpParams myArray[2];
/// myArray[0].operation = hipStreamMemOpWaitValue32;
/// myArray[0].waitValue.address = waitAddr1;
/// myArray[0].waitValue.value = 0x1;
/// myArray[0].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
/// myArray[1].operation = hipStreamMemOpWriteValue32;
/// myArray[1].writeValue.address = writeAddr1;
/// myArray[1].writeValue.value = 0x1;
/// myArray[1].writeValue.flags = 0x0;
/// result = hipStreamBatchMemOp(stream, 2, myArray, 0);
/// @endcode
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipStreamBatchMemOpParams_union {
    pub operation: hipStreamBatchMemOpType,
    pub waitValue: hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t,
    pub writeValue: hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t,
    /// Currently not supported on AMD
    pub flushRemoteWrites: hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t,
    /// Currently not supported on AMD
    pub memoryBarrier: hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t,
    pub pad: [u64; 6usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t {
    pub operation: hipStreamBatchMemOpType,
    pub address: hipDeviceptr_t,
    pub __bindgen_anon_1:
        hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    /// Not valid for AMD backend. Initial value is unimportant
    pub alias: hipDeviceptr_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t__bindgen_ty_1 {
    pub value: u32,
    pub value64: u64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t {
    pub operation: hipStreamBatchMemOpType,
    pub address: hipDeviceptr_t,
    pub __bindgen_anon_1:
        hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    /// Not valid for AMD backend. Initial value is unimportant
    pub alias: hipDeviceptr_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t__bindgen_ty_1 {
    pub value: u32,
    pub value64: u64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t {
    pub operation: hipStreamBatchMemOpType,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t {
    pub operation: hipStreamBatchMemOpType,
    pub flags: ::core::ffi::c_uint,
}
/// @brief Union representing batch memory operation parameters for HIP streams.
/// hipStreamBatchMemOpParams is used to specify the parameters for batch memory
/// operations in a HIP stream. This union supports various operations including
/// waiting for a specific value, writing a value, and different flags for wait conditions.
/// @details
/// The union includes fields for different types of operations defined in the
/// enum hipStreamBatchMemOpType:
/// - hipStreamMemOpWaitValue32:  Wait for a 32-bit value.
/// - hipStreamMemOpWriteValue32: Write a 32-bit value.
/// - hipStreamMemOpWaitValue64:  Wait for a 64-bit value.
/// - hipStreamMemOpWriteValue64: Write a 64-bit value.
/// Each operation type includes an address, the value to wait for or write, flags, and an
/// optional alias that is not relevant on AMD GPUs. Flags can be used to specify different
/// wait conditions such as equality, bitwise AND, greater than or equal, and bitwise NOR.
/// Example usage:
/// @code
/// hipStreamBatchMemOpParams myArray[2];
/// myArray[0].operation = hipStreamMemOpWaitValue32;
/// myArray[0].waitValue.address = waitAddr1;
/// myArray[0].waitValue.value = 0x1;
/// myArray[0].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
/// myArray[1].operation = hipStreamMemOpWriteValue32;
/// myArray[1].writeValue.address = writeAddr1;
/// myArray[1].writeValue.value = 0x1;
/// myArray[1].writeValue.flags = 0x0;
/// result = hipStreamBatchMemOp(stream, 2, myArray, 0);
/// @endcode
pub type hipStreamBatchMemOpParams = hipStreamBatchMemOpParams_union;
/// @brief Structure representing node parameters for batch memory operations in HIP graphs.
/// hipBatchMemOpNodeParams is used to specify the parameters for batch memory
/// operations in HIP graphs. This struct includes the context to use for the operations, the
/// number of operations, and an array of hipStreamBatchMemOpParams that describe the operations.
/// @details
/// The structure includes the following fields:
/// - ctx: The HIP context to use for the operations.
/// - count: The number of operations in the paramArray.
/// - paramArray: A pointer to an array of hipStreamBatchMemOpParams.
/// - flags: Flags to control the node.
/// Example usage:
/// @code
/// hipBatchMemOpNodeParams nodeParams;
/// nodeParams.ctx = context;
/// nodeParams.count = ARRAY_SIZE;
/// nodeParams.paramArray = myArray;
/// nodeParams.flags = 0;
/// Pass nodeParams to a HIP graph APIs hipGraphAddBatchMemOpNode, hipGraphBatchMemOpNodeGetParams,
/// hipGraphBatchMemOpNodeSetParams, hipGraphExecBatchMemOpNodeSetParams
/// @endcode
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipBatchMemOpNodeParams {
    pub ctx: hipCtx_t,
    pub count: ::core::ffi::c_uint,
    pub paramArray: *mut hipStreamBatchMemOpParams,
    pub flags: ::core::ffi::c_uint,
}
#[repr(i32)]
/// HIP Memory Advise values
/// This memory advise enumeration is used on Linux, not Windows.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemoryAdvise {
    /// Data will mostly be read and only occassionally
    ///< be written to
    hipMemAdviseSetReadMostly = 1,
    /// Undo the effect of hipMemAdviseSetReadMostly
    hipMemAdviseUnsetReadMostly = 2,
    /// Set the preferred location for the data as
    ///< the specified device
    hipMemAdviseSetPreferredLocation = 3,
    /// Clear the preferred location for the data
    hipMemAdviseUnsetPreferredLocation = 4,
    /// Data will be accessed by the specified device
    ///< so prevent page faults as much as possible
    hipMemAdviseSetAccessedBy = 5,
    /// Let HIP to decide on the page faulting policy
    ///< for the specified device
    hipMemAdviseUnsetAccessedBy = 6,
    /// The default memory model is fine-grain. That allows
    ///< coherent operations between host and device, while
    ///< executing kernels. The coarse-grain can be used
    ///< for data that only needs to be coherent at dispatch
    ///< boundaries for better performance
    hipMemAdviseSetCoarseGrain = 100,
    /// Restores cache coherency policy back to fine-grain
    hipMemAdviseUnsetCoarseGrain = 101,
}
#[repr(i32)]
/// HIP Coherency Mode
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemRangeCoherencyMode {
    /// Updates to memory with this attribute can be
    ///< done coherently from all devices
    hipMemRangeCoherencyModeFineGrain = 0,
    /// Writes to memory with this attribute can be
    ///< performed by a single device at a time
    hipMemRangeCoherencyModeCoarseGrain = 1,
    /// Memory region queried contains subregions with
    ///< both hipMemRangeCoherencyModeFineGrain and
    ///< hipMemRangeCoherencyModeCoarseGrain attributes
    hipMemRangeCoherencyModeIndeterminate = 2,
}
#[repr(i32)]
/// HIP range attributes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemRangeAttribute {
    /// Whether the range will mostly be read and
    ///< only occassionally be written to
    hipMemRangeAttributeReadMostly = 1,
    /// The preferred location of the range
    hipMemRangeAttributePreferredLocation = 2,
    /// Memory range has hipMemAdviseSetAccessedBy
    ///< set for the specified device
    hipMemRangeAttributeAccessedBy = 3,
    /// The last location to where the range was
    ///< prefetched
    hipMemRangeAttributeLastPrefetchLocation = 4,
    /// Returns coherency mode
    ///< @ref hipMemRangeCoherencyMode for the range
    hipMemRangeAttributeCoherencyMode = 100,
}
#[repr(i32)]
/// HIP memory pool attributes
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemPoolAttr {
    /// (value type = int)
    /// Allow @p hipMemAllocAsync to use memory asynchronously freed
    /// in another streams as long as a stream ordering dependency
    /// of the allocating stream on the free action exists.
    /// hip events and null stream interactions can create the required
    /// stream ordered dependencies. (default enabled)
    hipMemPoolReuseFollowEventDependencies = 1,
    /// (value type = int)
    /// Allow reuse of already completed frees when there is no dependency
    /// between the free and allocation. (default enabled)
    hipMemPoolReuseAllowOpportunistic = 2,
    /// (value type = int)
    /// Allow @p hipMemAllocAsync to insert new stream dependencies
    /// in order to establish the stream ordering required to reuse
    /// a piece of memory released by cuFreeAsync (default enabled).
    hipMemPoolReuseAllowInternalDependencies = 3,
    /// (value type = uint64_t)
    /// Amount of reserved memory in bytes to hold onto before trying
    /// to release memory back to the OS. When more than the release
    /// threshold bytes of memory are held by the memory pool, the
    /// allocator will try to release memory back to the OS on the
    /// next call to stream, event or context synchronize. (default 0)
    hipMemPoolAttrReleaseThreshold = 4,
    /// (value type = uint64_t)
    /// Amount of backing memory currently allocated for the mempool.
    hipMemPoolAttrReservedMemCurrent = 5,
    /// (value type = uint64_t)
    /// High watermark of backing memory allocated for the mempool since the
    /// last time it was reset. High watermark can only be reset to zero.
    hipMemPoolAttrReservedMemHigh = 6,
    /// (value type = uint64_t)
    /// Amount of memory from the pool that is currently in use by the application.
    hipMemPoolAttrUsedMemCurrent = 7,
    /// (value type = uint64_t)
    /// High watermark of the amount of memory from the pool that was in use by the application since
    /// the last time it was reset. High watermark can only be reset to zero.
    hipMemPoolAttrUsedMemHigh = 8,
}
#[repr(i32)]
/// Specifies the memory protection flags for mapping
///
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemAccessFlags {
    /// Default, make the address range not accessible
    hipMemAccessFlagsProtNone = 0,
    /// Set the address range read accessible
    hipMemAccessFlagsProtRead = 1,
    /// Set the address range read-write accessible
    hipMemAccessFlagsProtReadWrite = 3,
}
/// Memory access descriptor structure is used to specify memory access
/// permissions for a virtual memory region in Virtual Memory Management API.
/// This structure changes read, and write permissions for
/// specific memory regions.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemAccessDesc {
    /// Location on which the accessibility has to change
    pub location: hipMemLocation,
    /// Accessibility flags to set
    pub flags: hipMemAccessFlags,
}
#[repr(i32)]
/// Defines the allocation types
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemAllocationType {
    hipMemAllocationTypeInvalid = 0,
    /// This allocation type is 'pinned', i.e. cannot migrate from its current
    /// location while the application is actively using it
    hipMemAllocationTypePinned = 1,
    /// This allocation type is 'pinned', i.e. cannot migrate from its current
    /// location while the application is actively using it
    hipMemAllocationTypeManaged = 2,
    /// This allocation type is 'pinned', i.e. cannot migrate from its current
    /// location while the application is actively using it
    hipMemAllocationTypeUncached = 1073741824,
    /// This allocation type is 'pinned', i.e. cannot migrate from its current
    /// location while the application is actively using it
    hipMemAllocationTypeMax = 2147483647,
}
#[repr(i32)]
/// Flags for specifying handle types for memory pool allocations
///
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemAllocationHandleType {
    /// Does not allow any export mechanism
    hipMemHandleTypeNone = 0,
    hipMemHandleTypePosixFileDescriptor = 1,
    /// Allows a Win32 NT handle for exporting. (HANDLE)
    hipMemHandleTypeWin32 = 2,
    /// Allows a Win32 KMT handle for exporting. (D3DKMT_HANDLE)
    hipMemHandleTypeWin32Kmt = 4,
    /// Allows a fabric handle to be used for exporting.
    hipMemHandleTypeFabric = 8,
}
/// Specifies the properties of allocations made from the pool.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemPoolProps {
    /// Allocation type. Currently must be specified as @p hipMemAllocationTypePinned
    pub allocType: hipMemAllocationType,
    /// Handle types that will be supported by allocations from the pool
    pub handleTypes: hipMemAllocationHandleType,
    /// Location where allocations should reside
    pub location: hipMemLocation,
    /// Windows-specific LPSECURITYATTRIBUTES required when @p hipMemHandleTypeWin32 is specified
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    /// Maximum pool size. When set to 0, defaults to a system dependent value
    pub maxSize: usize,
    /// Reserved for future use, must be 0
    pub reserved: [::core::ffi::c_uchar; 56usize],
}
/// Opaque data structure for exporting a pool allocation
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemPoolPtrExportData {
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[repr(i32)]
/// @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipFuncAttribute {
    /// The maximum number of bytes requested for dynamically allocated shared memory
    hipFuncAttributeMaxDynamicSharedMemorySize = 8,
    /// Sets the percentage of total shared memory allocated as the shared memory carveout
    hipFuncAttributePreferredSharedMemoryCarveout = 9,
    /// The kernel must launch with a valid cluster size specified.
    hipFuncAttributeClusterDimMustBeSet = 10,
    /// The required cluster width in blocks
    hipFuncAttributeRequiredClusterWidth = 11,
    /// The required cluster height in blocks
    hipFuncAttributeRequiredClusterHeight = 12,
    /// The required cluster depth in blocks
    hipFuncAttributeRequiredClusterDepth = 13,
    /// Is the function allowed to launch with non-portable cluster size.
    hipFuncAttributeNonPortableClusterSizeAllowed = 14,
    /// The block scheduling policy of a function.
    hipFuncAttributeClusterSchedulingPolicyPreference = 15,
    hipFuncAttributeMax = 16,
}
#[repr(i32)]
/// @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipFuncCache_t {
    /// no preference for shared memory or L1 (default)
    hipFuncCachePreferNone = 0,
    /// prefer larger shared memory and smaller L1 cache
    hipFuncCachePreferShared = 1,
    /// prefer larger L1 cache and smaller shared memory
    hipFuncCachePreferL1 = 2,
    /// prefer equal size L1 cache and shared memory
    hipFuncCachePreferEqual = 3,
}
#[repr(i32)]
/// @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipSharedMemConfig {
    /// The compiler selects a device-specific value for the banking.
    hipSharedMemBankSizeDefault = 0,
    /// Shared mem is banked at 4-bytes intervals and performs best
    ///< when adjacent threads access data 4 bytes apart.
    hipSharedMemBankSizeFourByte = 1,
    /// Shared mem is banked at 8-byte intervals and performs best
    ///< when adjacent threads access data 4 bytes apart.
    hipSharedMemBankSizeEightByte = 2,
}
/// Struct for data in 3D
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct dim3 {
    /// x
    pub x: u32,
    /// y
    pub y: u32,
    /// z
    pub z: u32,
}
/// struct hipLaunchParams_t
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipLaunchParams_t {
    /// Device function symbol
    pub func: *mut ::core::ffi::c_void,
    /// Grid dimensions
    pub gridDim: dim3,
    /// Block dimensions
    pub blockDim: dim3,
    /// Arguments
    pub args: *mut *mut ::core::ffi::c_void,
    /// Shared memory
    pub sharedMem: usize,
    /// Stream identifier
    pub stream: hipStream_t,
}
/// struct hipLaunchParams_t
pub type hipLaunchParams = hipLaunchParams_t;
/// struct hipFunctionLaunchParams_t
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipFunctionLaunchParams_t {
    /// Kernel to launch
    pub function: hipFunction_t,
    /// Width(X) of grid in blocks
    pub gridDimX: ::core::ffi::c_uint,
    /// Height(Y) of grid in blocks
    pub gridDimY: ::core::ffi::c_uint,
    /// Depth(Z) of grid in blocks
    pub gridDimZ: ::core::ffi::c_uint,
    /// X dimension of each thread block
    pub blockDimX: ::core::ffi::c_uint,
    /// Y dimension of each thread block
    pub blockDimY: ::core::ffi::c_uint,
    /// Z dimension of each thread block
    pub blockDimZ: ::core::ffi::c_uint,
    /// Shared memory
    pub sharedMemBytes: ::core::ffi::c_uint,
    /// Stream identifier
    pub hStream: hipStream_t,
    /// Kernel parameters
    pub kernelParams: *mut *mut ::core::ffi::c_void,
}
/// struct hipFunctionLaunchParams_t
pub type hipFunctionLaunchParams = hipFunctionLaunchParams_t;
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipExternalMemoryHandleType_enum {
    hipExternalMemoryHandleTypeOpaqueFd = 1,
    hipExternalMemoryHandleTypeOpaqueWin32 = 2,
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
    hipExternalMemoryHandleTypeD3D12Heap = 4,
    hipExternalMemoryHandleTypeD3D12Resource = 5,
    hipExternalMemoryHandleTypeD3D11Resource = 6,
    hipExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    hipExternalMemoryHandleTypeNvSciBuf = 8,
}
pub use self::hipExternalMemoryHandleType_enum as hipExternalMemoryHandleType;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipExternalMemoryHandleDesc_st {
    pub type_: hipExternalMemoryHandleType,
    pub handle: hipExternalMemoryHandleDesc_st__bindgen_ty_1,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipExternalMemoryHandleDesc_st__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: hipExternalMemoryHandleDesc_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciBufObject: *const ::core::ffi::c_void,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalMemoryHandleDesc_st__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
pub type hipExternalMemoryHandleDesc = hipExternalMemoryHandleDesc_st;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalMemoryBufferDesc_st {
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
pub type hipExternalMemoryBufferDesc = hipExternalMemoryBufferDesc_st;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalMemoryMipmappedArrayDesc_st {
    pub offset: ::core::ffi::c_ulonglong,
    pub formatDesc: hipChannelFormatDesc,
    pub extent: hipExtent,
    pub flags: ::core::ffi::c_uint,
    pub numLevels: ::core::ffi::c_uint,
}
pub type hipExternalMemoryMipmappedArrayDesc = hipExternalMemoryMipmappedArrayDesc_st;
pub type hipExternalMemory_t = *mut ::core::ffi::c_void;
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipExternalSemaphoreHandleType_enum {
    hipExternalSemaphoreHandleTypeOpaqueFd = 1,
    hipExternalSemaphoreHandleTypeOpaqueWin32 = 2,
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    hipExternalSemaphoreHandleTypeD3D12Fence = 4,
    hipExternalSemaphoreHandleTypeD3D11Fence = 5,
    hipExternalSemaphoreHandleTypeNvSciSync = 6,
    hipExternalSemaphoreHandleTypeKeyedMutex = 7,
    hipExternalSemaphoreHandleTypeKeyedMutexKmt = 8,
    hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,
    hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10,
}
pub use self::hipExternalSemaphoreHandleType_enum as hipExternalSemaphoreHandleType;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipExternalSemaphoreHandleDesc_st {
    pub type_: hipExternalSemaphoreHandleType,
    pub handle: hipExternalSemaphoreHandleDesc_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipExternalSemaphoreHandleDesc_st__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: hipExternalSemaphoreHandleDesc_st__bindgen_ty_1__bindgen_ty_1,
    pub NvSciSyncObj: *const ::core::ffi::c_void,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalSemaphoreHandleDesc_st__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
pub type hipExternalSemaphoreHandleDesc = hipExternalSemaphoreHandleDesc_st;
pub type hipExternalSemaphore_t = *mut ::core::ffi::c_void;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipExternalSemaphoreSignalParams_st {
    pub params: hipExternalSemaphoreSignalParams_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipExternalSemaphoreSignalParams_st__bindgen_ty_1 {
    pub fence: hipExternalSemaphoreSignalParams_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: hipExternalSemaphoreSignalParams_st__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: hipExternalSemaphoreSignalParams_st__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalSemaphoreSignalParams_st__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipExternalSemaphoreSignalParams_st__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalSemaphoreSignalParams_st__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
}
pub type hipExternalSemaphoreSignalParams = hipExternalSemaphoreSignalParams_st;
/// External semaphore wait parameters, compatible with driver type
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipExternalSemaphoreWaitParams_st {
    pub params: hipExternalSemaphoreWaitParams_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipExternalSemaphoreWaitParams_st__bindgen_ty_1 {
    pub fence: hipExternalSemaphoreWaitParams_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: hipExternalSemaphoreWaitParams_st__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: hipExternalSemaphoreWaitParams_st__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 10usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalSemaphoreWaitParams_st__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipExternalSemaphoreWaitParams_st__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalSemaphoreWaitParams_st__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
    pub timeoutMs: ::core::ffi::c_uint,
}
/// External semaphore wait parameters, compatible with driver type
pub type hipExternalSemaphoreWaitParams = hipExternalSemaphoreWaitParams_st;
#[repr(i32)]
/// HIP Access falgs for Interop resources.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphicsRegisterFlags {
    hipGraphicsRegisterFlagsNone = 0,
    /// HIP will not write to this registered resource, read only
    hipGraphicsRegisterFlagsReadOnly = 1,
    hipGraphicsRegisterFlagsWriteDiscard = 2,
    /// HIP will bind this resource to a surface, read and write
    hipGraphicsRegisterFlagsSurfaceLoadStore = 4,
    hipGraphicsRegisterFlagsTextureGather = 8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct _hipGraphicsResource {
    _unused: [u8; 0],
}
pub type hipGraphicsResource = _hipGraphicsResource;
pub type hipGraphicsResource_t = *mut hipGraphicsResource;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipGraph {
    _unused: [u8; 0],
}
/// An opaque value that represents a hip graph
pub type hipGraph_t = *mut ihipGraph;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct hipGraphNode {
    _unused: [u8; 0],
}
/// An opaque value that represents a hip graph node
pub type hipGraphNode_t = *mut hipGraphNode;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct hipGraphExec {
    _unused: [u8; 0],
}
/// An opaque value that represents a hip graph Exec
pub type hipGraphExec_t = *mut hipGraphExec;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct hipUserObject {
    _unused: [u8; 0],
}
/// An opaque value that represents a user obj
pub type hipUserObject_t = *mut hipUserObject;
#[repr(i32)]
/// hipGraphNodeType
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphNodeType {
    /// GPU kernel node
    hipGraphNodeTypeKernel = 0,
    /// Memcpy node
    hipGraphNodeTypeMemcpy = 1,
    /// Memset node
    hipGraphNodeTypeMemset = 2,
    /// Host (executable) node
    hipGraphNodeTypeHost = 3,
    /// Node which executes an embedded graph
    hipGraphNodeTypeGraph = 4,
    /// Empty (no-op) node
    hipGraphNodeTypeEmpty = 5,
    /// External event wait node
    hipGraphNodeTypeWaitEvent = 6,
    /// External event record node
    hipGraphNodeTypeEventRecord = 7,
    /// External Semaphore signal node
    hipGraphNodeTypeExtSemaphoreSignal = 8,
    /// External Semaphore wait node
    hipGraphNodeTypeExtSemaphoreWait = 9,
    /// Memory alloc node
    hipGraphNodeTypeMemAlloc = 10,
    /// Memory free node
    hipGraphNodeTypeMemFree = 11,
    /// MemcpyFromSymbol node
    hipGraphNodeTypeMemcpyFromSymbol = 12,
    /// MemcpyToSymbol node
    hipGraphNodeTypeMemcpyToSymbol = 13,
    /// BatchMemOp node
    hipGraphNodeTypeBatchMemOp = 14,
    hipGraphNodeTypeCount = 15,
}
pub type hipHostFn_t =
    ::std::option::Option<unsafe extern "C" fn(userData: *mut ::core::ffi::c_void)>;
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipHostNodeParams {
    pub fn_: hipHostFn_t,
    pub userData: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipKernelNodeParams {
    pub blockDim: dim3,
    pub extra: *mut *mut ::core::ffi::c_void,
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub sharedMemBytes: ::core::ffi::c_uint,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemsetParams {
    pub dst: *mut ::core::ffi::c_void,
    pub elementSize: ::core::ffi::c_uint,
    pub height: usize,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub width: usize,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemAllocNodeParams {
    /// Pool properties, which contain where
    ///< the location should reside
    pub poolProps: hipMemPoolProps,
    /// The number of memory access descriptors.
    pub accessDescs: *const hipMemAccessDesc,
    /// The number of access descriptors.
    ///< Must not be bigger than the number of GPUs
    pub accessDescCount: usize,
    /// The size of the requested allocation in bytes
    pub bytesize: usize,
    /// Returned device address of the allocation
    pub dptr: *mut ::core::ffi::c_void,
}
#[repr(i32)]
/// Specifies performance hint with hipAccessPolicyWindow
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipAccessProperty {
    /// Normal cache persistence.
    hipAccessPropertyNormal = 0,
    /// Streaming access is less likely to persist from cache
    hipAccessPropertyStreaming = 1,
    /// Persisting access is more likely to persist in cache
    hipAccessPropertyPersisting = 2,
}
/// Specifies access policy for a window, a contiguous extent of memory
/// beginning at base_ptr and ending at base_ptr + num_bytes.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipAccessPolicyWindow {
    /// Starting address of the access policy window
    pub base_ptr: *mut ::core::ffi::c_void,
    /// hipAccessProperty set for hit
    pub hitProp: hipAccessProperty,
    /// hitRatio specifies percentage of lines assigned hitProp
    pub hitRatio: f32,
    /// hipAccessProperty set for miss
    pub missProp: hipAccessProperty,
    /// Size in bytes of the window policy.
    pub num_bytes: usize,
}
/// Memory Synchronization Domain map
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipLaunchMemSyncDomainMap {
    /// The default domain ID to use for designated kernels
    pub default_: ::core::ffi::c_uchar,
    /// The remote domain ID to use for designated kernels
    pub remote: ::core::ffi::c_uchar,
}
#[repr(i32)]
/// Memory Synchronization Domain
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipLaunchMemSyncDomain {
    /// Launch kernels in the default domain
    hipLaunchMemSyncDomainDefault = 0,
    /// Launch kernels in the remote domain
    hipLaunchMemSyncDomainRemote = 1,
}
#[repr(i32)]
/// Stream Synchronization Policy.
/// Can be set with hipStreamSetAttribute
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipSynchronizationPolicy {
    /// Default Synchronization Policy. Host thread waits actively
    hipSyncPolicyAuto = 1,
    /// Host thread spins in tight loop waiting for completition
    hipSyncPolicySpin = 2,
    /// Host spins but yields to other threads, reducing CPU usage
    hipSyncPolicyYield = 3,
    /// Host thread blocks (sleeps) until the stream completes
    hipSyncPolicyBlockingSync = 4,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipClusterSchedulingPolicy {
    /// the default scheduling policy
    hipClusterSchedulingPolicyDefault = 0,
    /// distribute blocks evenly across cluster's CUs
    hipClusterSchedulingPolicySpread = 1,
    hipClusterSchedulingPolicyLoadBalancing = 2,
}
#[repr(i32)]
/// @brief Temporal locality hint for dynamic data prefetch.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipExtDynDataPrefetchTemporal {
    /// Regular temporal locality
    hipExtDynDataPrefetchTemporalRegular = 0,
    /// High temporal locality (prefer caching)
    hipExtDynDataPrefetchTemporalHigh = 1,
}
/// @brief Describes one 2D memory region to prefetch into L2 cache.
/// @p address must be aligned to the device's L2 cache line size.
/// @p width is measured in bytes and must be a multiple of the cache line size.
/// @p height is the number of rows to prefetch.
/// @p stride is the byte stride between the start of consecutive rows.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExtDynDataPrefetchRegion {
    /// Base address (must be cache-line aligned)
    pub address: *mut ::core::ffi::c_void,
    /// Stride between row starts in bytes
    pub stride: usize,
    /// Width of each row in bytes (must be a multiple of cache line size)
    pub width: usize,
    /// Number of rows to prefetch
    pub height: usize,
}
/// @brief Configuration for dynamic data prefetch.
/// Pointed to by a launch attribute via @ref hipLaunchAttributeExtDynDataPrefetch.
/// @p numRegions must not exceed the device limit queried via
/// @ref hipDeviceAttributeMaxDynDataPrefetchRegions (use @ref hipDeviceGetAttribute).
/// Cooperative prefetch is always enabled internally; the user only controls
/// the temporal policy.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExtDynDataPrefetchConfig {
    /// Number of valid regions (1-max)
    pub numRegions: ::core::ffi::c_uint,
    /// Cache retention policy for prefetched data
    pub temporal: hipExtDynDataPrefetchTemporal,
    /// Prefetch regions
    pub regions: [hipExtDynDataPrefetchRegion; 2usize],
}
#[repr(i32)]
///  Launch Attribute ID
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipLaunchAttributeID {
    /// Ignored entry
    hipLaunchAttributeIgnore = 0,
    /// Valid for Streams, graph nodes, launches
    hipLaunchAttributeAccessPolicyWindow = 1,
    /// Valid for graph nodes, launches
    hipLaunchAttributeCooperative = 2,
    /// Valid for streams
    hipLaunchAttributeSynchronizationPolicy = 3,
    /// Valid for graph nodes, launches
    hipLaunchAttributeClusterDimension = 4,
    /// Valid for graph nodes, launches
    hipLaunchAttributeClusterSchedulingPolicyPreference = 5,
    /// Valid for graph node, streams, launches
    hipLaunchAttributePriority = 8,
    /// Valid for streams, graph nodes, launches
    hipLaunchAttributeMemSyncDomainMap = 9,
    /// Valid for streams, graph nodes, launches
    hipLaunchAttributeMemSyncDomain = 10,
    /// Valid for launches. Prefetch data into L2 before kernel execution.
    hipLaunchAttributeExtDynDataPrefetch = 1024,
    hipLaunchAttributeMax = 1025,
}
pub type hipStreamAttrID = hipLaunchAttributeID;
///  Launch Attribute Value
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipLaunchAttributeValue {
    /// 64 byte padding
    pub pad: [::core::ffi::c_char; 64usize],
    /// Value of launch attribute ::hipLaunchAttributeAccessPolicyWindow.
    pub accessPolicyWindow: hipAccessPolicyWindow,
    /// Value of launch attribute ::hipLaunchAttributeCooperative. Indicates
    ///< whether the kernel is cooperative.
    pub cooperative: ::core::ffi::c_int,
    /// Value of launch attribute :: hipLaunchAttributePriority. Execution priority of
    ///< kernel
    pub priority: ::core::ffi::c_int,
    /// Value of launch attribute :: hipLaunchAttributeSynchronizationPolicy. Used
    ///< to work queued up in stream
    pub syncPolicy: hipSynchronizationPolicy,
    /// Value of launch attribute hipLaunchAttributeMemSyncDomainMap
    pub memSyncDomainMap: hipLaunchMemSyncDomainMap,
    /// Value of launch attribute hipLaunchAttributeMemSyncDomain
    pub memSyncDomain: hipLaunchMemSyncDomain,
    pub clusterDim: hipLaunchAttributeValue__bindgen_ty_1,
    /// Value of launch attribute :: hipLaunchAttributeClusterSchedulingPolicyPreference
    ///< determines the preferred strategy for distributing blocks within a compute cluster
    pub clusterSchedulingPolicyPreference: hipClusterSchedulingPolicy,
    /// Value of launch attribute ::hipLaunchAttributeExtDynDataPrefetch
    pub dynDataPrefetch: *const hipExtDynDataPrefetchConfig,
}
/// @brief Specifies the desired cluster dimensions for a kernel launch.
/// This opaque type is used as the value for the launch attribute
/// ::hipLaunchAttributeClusterDimension. It defines the dimensions of the
/// compute cluster in terms of blocks, where each field must evenly divide
/// the corresponding grid dimension:
///  - \\p x: Number of blocks along the X-axis.
///  - \\p y: Number of blocks along the Y-axis.
///  - \\p z: Number of blocks along the Z-axis.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipLaunchAttributeValue__bindgen_ty_1 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[repr(i32)]
/// Graph execution update result
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphExecUpdateResult {
    /// The update succeeded
    hipGraphExecUpdateSuccess = 0,
    /// The update failed for an unexpected reason which is described
    ///< in the return value of the function
    hipGraphExecUpdateError = 1,
    /// The update failed because the topology changed
    hipGraphExecUpdateErrorTopologyChanged = 2,
    /// The update failed because a node type changed
    hipGraphExecUpdateErrorNodeTypeChanged = 3,
    hipGraphExecUpdateErrorFunctionChanged = 4,
    hipGraphExecUpdateErrorParametersChanged = 5,
    hipGraphExecUpdateErrorNotSupported = 6,
    hipGraphExecUpdateErrorUnsupportedFunctionChange = 7,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipStreamCaptureMode {
    hipStreamCaptureModeGlobal = 0,
    hipStreamCaptureModeThreadLocal = 1,
    hipStreamCaptureModeRelaxed = 2,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipStreamCaptureStatus {
    /// Stream is not capturing
    hipStreamCaptureStatusNone = 0,
    /// Stream is actively capturing
    hipStreamCaptureStatusActive = 1,
    /// Stream is part of a capture sequence that has been
    ///< invalidated, but not terminated
    hipStreamCaptureStatusInvalidated = 2,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipStreamUpdateCaptureDependenciesFlags {
    /// Add new nodes to the dependency set
    hipStreamAddCaptureDependencies = 0,
    /// Replace the dependency set with the new nodes
    hipStreamSetCaptureDependencies = 1,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphMemAttributeType {
    hipGraphMemAttrUsedMemCurrent = 0,
    /// High watermark of memory, in bytes, associated with graphs since
    ///< the last time.
    hipGraphMemAttrUsedMemHigh = 1,
    /// Amount of memory, in bytes, currently allocated for
    ///< graphs.
    hipGraphMemAttrReservedMemCurrent = 2,
    /// High watermark of memory, in bytes, currently allocated for
    ///< graphs
    hipGraphMemAttrReservedMemHigh = 3,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipUserObjectFlags {
    /// Destructor execution is not synchronized.
    hipUserObjectNoDestructorSync = 1,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipUserObjectRetainFlags {
    /// Add new reference or retain.
    hipGraphUserObjectMove = 1,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphInstantiateFlags {
    hipGraphInstantiateFlagAutoFreeOnLaunch = 1,
    /// Automatically upload the graph after instantiation.
    hipGraphInstantiateFlagUpload = 2,
    hipGraphInstantiateFlagDeviceLaunch = 4,
    hipGraphInstantiateFlagUseNodePriority = 8,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphDebugDotFlags {
    hipGraphDebugDotFlagsVerbose = 1,
    /// Adds hipKernelNodeParams to output
    hipGraphDebugDotFlagsKernelNodeParams = 4,
    /// Adds hipMemcpy3DParms to output
    hipGraphDebugDotFlagsMemcpyNodeParams = 8,
    /// Adds hipMemsetParams to output
    hipGraphDebugDotFlagsMemsetNodeParams = 16,
    /// Adds hipHostNodeParams to output
    hipGraphDebugDotFlagsHostNodeParams = 32,
    hipGraphDebugDotFlagsEventNodeParams = 64,
    hipGraphDebugDotFlagsExtSemasSignalNodeParams = 128,
    hipGraphDebugDotFlagsExtSemasWaitNodeParams = 256,
    hipGraphDebugDotFlagsKernelNodeAttributes = 512,
    hipGraphDebugDotFlagsHandles = 1024,
}
#[repr(i32)]
/// hipGraphInstantiateWithParams results
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphInstantiateResult {
    /// Instantiation Success
    hipGraphInstantiateSuccess = 0,
    /// Instantiation failed for an
    ///unexpected reason which is described in the return value of the function
    hipGraphInstantiateError = 1,
    /// Instantiation failed due
    ///to invalid structure, such as cycles
    hipGraphInstantiateInvalidStructure = 2,
    /// Instantiation for device launch failed
    ///because the graph contained an unsupported operation
    hipGraphInstantiateNodeOperationNotSupported = 3,
    /// Instantiation for device launch failed
    ///due to the nodes belonging to different contexts
    hipGraphInstantiateMultipleDevicesNotSupported = 4,
}
/// Graph Instantiation parameters
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipGraphInstantiateParams {
    /// The node which caused instantiation to fail, if any
    pub errNode_out: hipGraphNode_t,
    /// Instantiation flags
    pub flags: ::core::ffi::c_ulonglong,
    /// Whether instantiation was successful.
    ///If it failed, the reason why
    pub result_out: hipGraphInstantiateResult,
    /// Upload stream
    pub uploadStream: hipStream_t,
}
/// Memory allocation properties
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipMemAllocationProp {
    /// Memory allocation type
    pub type_: hipMemAllocationType,
    pub __bindgen_anon_1: hipMemAllocationProp__bindgen_ty_1,
    /// Memory location
    pub location: hipMemLocation,
    /// Metadata for Win32 handles
    pub win32HandleMetaData: *mut ::core::ffi::c_void,
    pub allocFlags: hipMemAllocationProp__bindgen_ty_2,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipMemAllocationProp__bindgen_ty_1 {
    /// Requested handle type
    pub requestedHandleType: hipMemAllocationHandleType,
    /// Requested handle types
    pub requestedHandleTypes: hipMemAllocationHandleType,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemAllocationProp__bindgen_ty_2 {
    /// Compression type
    pub compressionType: ::core::ffi::c_uchar,
    /// RDMA capable
    pub gpuDirectRDMACapable: ::core::ffi::c_uchar,
    /// Usage
    pub usage: ::core::ffi::c_ushort,
}
/// External semaphore signal node parameters
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalSemaphoreSignalNodeParams {
    pub extSemArray: *mut hipExternalSemaphore_t,
    pub paramsArray: *const hipExternalSemaphoreSignalParams,
    pub numExtSems: ::core::ffi::c_uint,
}
/// External semaphore wait node parameters
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipExternalSemaphoreWaitNodeParams {
    pub extSemArray: *mut hipExternalSemaphore_t,
    pub paramsArray: *const hipExternalSemaphoreWaitParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug)]
pub struct ihipMemGenericAllocationHandle {
    _unused: [u8; 0],
}
/// Generic handle for memory allocation
pub type hipMemGenericAllocationHandle_t = *mut ihipMemGenericAllocationHandle;
#[repr(i32)]
/// Flags for granularity
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemAllocationGranularity_flags {
    /// Minimum granularity
    hipMemAllocationGranularityMinimum = 0,
    /// Recommended granularity for performance
    hipMemAllocationGranularityRecommended = 1,
}
#[repr(i32)]
/// Memory handle type
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemHandleType {
    /// Generic handle type
    hipMemHandleTypeGeneric = 0,
}
#[repr(i32)]
/// Memory operation types
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemOperationType {
    /// Map operation
    hipMemOperationTypeMap = 1,
    /// Unmap operation
    hipMemOperationTypeUnmap = 2,
}
#[repr(i32)]
/// Subresource types for sparse arrays
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipArraySparseSubresourceType {
    /// Sparse level
    hipArraySparseSubresourceTypeSparseLevel = 0,
    /// Miptail
    hipArraySparseSubresourceTypeMiptail = 1,
}
/// Map info for arrays
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipArrayMapInfo {
    /// Resource type
    pub resourceType: hipResourceType,
    pub resource: hipArrayMapInfo__bindgen_ty_1,
    /// Sparse subresource type
    pub subresourceType: hipArraySparseSubresourceType,
    pub subresource: hipArrayMapInfo__bindgen_ty_2,
    /// Memory operation type
    pub memOperationType: hipMemOperationType,
    /// Memory handle type
    pub memHandleType: hipMemHandleType,
    pub memHandle: hipArrayMapInfo__bindgen_ty_3,
    /// Offset within the memory
    pub offset: ::core::ffi::c_ulonglong,
    /// Device ordinal bit mask
    pub deviceBitMask: ::core::ffi::c_uint,
    /// flags for future use, must be zero now.
    pub flags: ::core::ffi::c_uint,
    /// Reserved for future use, must be zero now.
    pub reserved: [::core::ffi::c_uint; 2usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipArrayMapInfo__bindgen_ty_1 {
    pub mipmap: hipMipmappedArray,
    pub array: hipArray_t,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipArrayMapInfo__bindgen_ty_2 {
    pub sparseLevel: hipArrayMapInfo__bindgen_ty_2__bindgen_ty_1,
    pub miptail: hipArrayMapInfo__bindgen_ty_2__bindgen_ty_2,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipArrayMapInfo__bindgen_ty_2__bindgen_ty_1 {
    /// For mipmapped arrays must be a valid mipmap level. For arrays must be zero
    pub level: ::core::ffi::c_uint,
    /// For layered arrays must be a valid layer index. Otherwise, must be zero
    pub layer: ::core::ffi::c_uint,
    /// X offset in elements
    pub offsetX: ::core::ffi::c_uint,
    /// Y offset in elements
    pub offsetY: ::core::ffi::c_uint,
    /// Z offset in elements
    pub offsetZ: ::core::ffi::c_uint,
    /// Width in elements
    pub extentWidth: ::core::ffi::c_uint,
    /// Height in elements
    pub extentHeight: ::core::ffi::c_uint,
    /// Depth in elements
    pub extentDepth: ::core::ffi::c_uint,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipArrayMapInfo__bindgen_ty_2__bindgen_ty_2 {
    /// For layered arrays must be a valid layer index. Otherwise, must be zero
    pub layer: ::core::ffi::c_uint,
    /// Offset within mip tail
    pub offset: ::core::ffi::c_ulonglong,
    /// Extent in bytes
    pub size: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipArrayMapInfo__bindgen_ty_3 {
    pub memHandle: hipMemGenericAllocationHandle_t,
}
/// Memcpy node params
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemcpyNodeParams {
    /// Must be zero.
    pub flags: ::core::ffi::c_int,
    /// Must be zero.
    pub reserved: [::core::ffi::c_int; 3usize],
    /// Params set for the memory copy.
    pub copyParams: hipMemcpy3DParms,
}
/// Child graph node params
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipChildGraphNodeParams {
    /// Either the child graph to clone into the node, or
    ///< a handle to the graph possesed by the node used during query
    pub graph: hipGraph_t,
}
/// Event record node params
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipEventWaitNodeParams {
    /// Event to wait on
    pub event: hipEvent_t,
}
/// Event record node params
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipEventRecordNodeParams {
    /// The event to be recorded when node executes
    pub event: hipEvent_t,
}
/// Memory free node params
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipMemFreeNodeParams {
    /// the pointer to be freed
    pub dptr: *mut ::core::ffi::c_void,
}
/// Params for different graph nodes
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipGraphNodeParams {
    pub type_: hipGraphNodeType,
    pub reserved0: [::core::ffi::c_int; 3usize],
    pub __bindgen_anon_1: hipGraphNodeParams__bindgen_ty_1,
    pub reserved2: ::core::ffi::c_longlong,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipGraphNodeParams__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: hipKernelNodeParams,
    pub memcpy: hipMemcpyNodeParams,
    pub memset: hipMemsetParams,
    pub host: hipHostNodeParams,
    pub graph: hipChildGraphNodeParams,
    pub eventWait: hipEventWaitNodeParams,
    pub eventRecord: hipEventRecordNodeParams,
    pub extSemSignal: hipExternalSemaphoreSignalNodeParams,
    pub extSemWait: hipExternalSemaphoreWaitNodeParams,
    pub alloc: hipMemAllocNodeParams,
    pub free: hipMemFreeNodeParams,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipGraphDependencyType {
    hipGraphDependencyTypeDefault = 0,
    hipGraphDependencyTypeProgrammatic = 1,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipGraphEdgeData {
    /// This indicates when the dependency is triggered from the upstream node on the
    ///< edge. The meaning is specfic to the node type. A value of 0 in all cases
    ///< means full completion of the upstream node, with memory visibility to the
    ///< downstream node or portion thereof (indicated by to_port). Only kernel nodes
    ///< define non-zero ports. A kernel node can use the following output port types:
    ///< hipGraphKernelNodePortDefault, hipGraphKernelNodePortProgrammatic, or
    ///< hipGraphKernelNodePortLaunchCompletion.
    pub from_port: ::core::ffi::c_uchar,
    /// These bytes are unused and must be zeroed
    pub reserved: [::core::ffi::c_uchar; 5usize],
    /// Currently no node types define non-zero ports. This field must be set to zero.
    pub to_port: ::core::ffi::c_uchar,
    /// This should be populated with a value from hipGraphDependencyType
    pub type_: ::core::ffi::c_uchar,
}
/// Used to specify custom attributes for launching kernels
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub struct hipLaunchAttribute_st {
    /// Identifier of the launch attribute
    pub id: hipLaunchAttributeID,
    /// Padding to align the structure to 8 bytes
    pub pad: [::core::ffi::c_char; 4usize],
    pub __bindgen_anon_1: hipLaunchAttribute_st__bindgen_ty_1,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Copy, Clone)]
pub union hipLaunchAttribute_st__bindgen_ty_1 {
    /// Value associated with the launch attribute
    pub val: hipLaunchAttributeValue,
    /// Value associated with the launch attribute
    pub value: hipLaunchAttributeValue,
}
/// Used to specify custom attributes for launching kernels
pub type hipLaunchAttribute = hipLaunchAttribute_st;
/// HIP extensible launch configuration
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipLaunchConfig_st {
    /// Grid dimensions
    pub gridDim: dim3,
    /// Block dimensions
    pub blockDim: dim3,
    /// Dynamic shared-memory size per thread block
    pub dynamicSmemBytes: usize,
    /// Stream identifier
    pub stream: hipStream_t,
    /// Attributes list
    pub attrs: *mut hipLaunchAttribute,
    /// Number of attributes
    pub numAttrs: ::core::ffi::c_uint,
}
/// HIP extensible launch configuration
pub type hipLaunchConfig_t = hipLaunchConfig_st;
/// HIP driver extensible launch configuration
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct HIP_LAUNCH_CONFIG_st {
    /// Grid width in blocks
    pub gridDimX: ::core::ffi::c_uint,
    /// Grid height in blocks
    pub gridDimY: ::core::ffi::c_uint,
    /// Grid depth in blocks
    pub gridDimZ: ::core::ffi::c_uint,
    /// Thread block dimension in X
    pub blockDimX: ::core::ffi::c_uint,
    /// Thread block dimension in Y
    pub blockDimY: ::core::ffi::c_uint,
    /// Thread block dimension in Z
    pub blockDimZ: ::core::ffi::c_uint,
    /// Dynamic shared-memory size in bytes per block
    pub sharedMemBytes: ::core::ffi::c_uint,
    /// HIP stream identifier
    pub hStream: hipStream_t,
    /// Attribute list
    pub attrs: *mut hipLaunchAttribute,
    /// Number of attributes
    pub numAttrs: ::core::ffi::c_uint,
}
/// HIP driver extensible launch configuration
pub type HIP_LAUNCH_CONFIG = HIP_LAUNCH_CONFIG_st;
/// Struct representing array memory requirements.
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipArrayMemoryRequirements {
    pub alignment: usize,
    pub size: usize,
}
#[repr(i32)]
/// Requested handle type for address range.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemRangeHandleType {
    hipMemRangeHandleTypeDmaBufFd = 1,
    hipMemRangeHandleTypeMax = 2147483647,
}
#[repr(i32)]
/// Mem Range Flags used in hipMemGetHandleForAddressRange.
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipMemRangeFlags {
    hipMemRangeFlagDmaBufMappingTypePcie = 1,
    hipMemRangeFlagsMax = 2147483647,
}
/// Stream CallBack struct
pub type hipStreamCallback_t = ::std::option::Option<
    unsafe extern "C" fn(
        stream: hipStream_t,
        status: hipError_t,
        userData: *mut ::core::ffi::c_void,
    ),
>;
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipDataType {
    HIP_R_32F = 0,
    HIP_R_64F = 1,
    HIP_R_16F = 2,
    HIP_R_8I = 3,
    HIP_C_32F = 4,
    HIP_C_64F = 5,
    HIP_C_16F = 6,
    HIP_C_8I = 7,
    HIP_R_8U = 8,
    HIP_C_8U = 9,
    HIP_R_32I = 10,
    HIP_C_32I = 11,
    HIP_R_32U = 12,
    HIP_C_32U = 13,
    HIP_R_16BF = 14,
    HIP_C_16BF = 15,
    HIP_R_4I = 16,
    HIP_C_4I = 17,
    HIP_R_4U = 18,
    HIP_C_4U = 19,
    HIP_R_16I = 20,
    HIP_C_16I = 21,
    HIP_R_16U = 22,
    HIP_C_16U = 23,
    HIP_R_64I = 24,
    HIP_C_64I = 25,
    HIP_R_64U = 26,
    HIP_C_64U = 27,
    HIP_R_8F_E4M3 = 28,
    HIP_R_8F_E5M2 = 29,
    HIP_R_8F_UE8M0 = 30,
    HIP_R_6F_E2M3 = 31,
    HIP_R_6F_E3M2 = 32,
    HIP_R_4F_E2M1 = 33,
    HIP_R_8F_E4M3_FNUZ = 1000,
    HIP_R_8F_E5M2_FNUZ = 1001,
}
#[repr(i32)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipLibraryPropertyType {
    HIP_LIBRARY_MAJOR_VERSION = 0,
    HIP_LIBRARY_MINOR_VERSION = 1,
    HIP_LIBRARY_PATCH_LEVEL = 2,
}
