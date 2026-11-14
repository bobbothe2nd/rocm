use rocm_sys::hip::hipMemPool_t;

mod copy;
mod devmem;
mod hostmem;
mod memset;

pub use copy::*;
pub use devmem::*;
pub use hostmem::*;
pub use memset::*;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemPool {
    mem_pool: hipMemPool_t,
}

pub trait EventSync {
    /// Records a synchronization checkpoint for the buffer, or a logical buffer usage.
    fn latesync(&self) -> Result<(), super::HipError>;

    /// Synchronizes the buffer up to it's last `latesync` or usage.
    ///
    /// These are logically equivalent (other than edge cases and errors):
    ///
    /// ```no_run
    /// # #[cfg(feature = "dynamic-loading")] {
    /// # use roc::hip::*;
    /// # let stream = stream::HipStream::create().unwrap();
    /// # let buf = stream.alloc(128).unwrap();
    /// buf.latesync().unwrap();
    /// buf.sync().unwrap();
    /// # }
    /// ```
    ///
    /// and
    ///
    /// ```no_run
    /// # #[cfg(feature = "dynamic-loading")] {
    /// # let stream = roc::hip::stream::HipStream::create().unwrap();
    /// stream.sync().unwrap();
    /// # }
    /// ```
    ///
    /// assuming `buf` was allocated on `stream`.
    fn sync(&self) -> Result<(), super::HipError>;
}

pub trait StreamBound {
    fn stream(&self) -> &super::stream::HipStream;
}

pub trait DevPtr {
    fn as_ptr(&self) -> *const u8;
    fn size(&self) -> usize;
    fn dev(&self) -> super::device::Device;
}

macro_rules! impl_buf {
    ($($ty:ty)*) => {$(
        impl StreamBound for $ty {
            #[inline(always)]
            fn stream(&self) -> &super::stream::HipStream {
                &self.stream
            }
        }

        impl EventSync for $ty {
            #[inline(always)]
            fn latesync(&self) -> Result<(), crate::hip::HipError> {
                self.event.record(&self.stream, None)
            }

            #[inline(always)]
            fn sync(&self) -> Result<(), crate::hip::HipError> {
                self.event.sync()
            }
        }

        impl DevPtr for $ty {
            #[inline(always)]
            fn as_ptr(&self) -> *const u8 {
                self.ptr
            }

            #[inline(always)]
            fn size(&self) -> usize {
                self.size
            }

            #[inline(always)]
            fn dev(&self) -> super::device::Device {
                self.stream
                    .get_dev()
                    .expect("valid HipStream should have device")
            }
        }
    )*};
}

impl_buf!(
    HipBuffer HipSlice<'_> HipSliceMut<'_>
    HostBuffer HostSlice<'_> HostSliceMut<'_>
);
