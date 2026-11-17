mod copy;
mod devmem;
mod extmem;
mod hostmem;
mod mempool;
mod memset;
mod unified;

pub use copy::*;
pub use devmem::*;
pub use extmem::*;
pub use hostmem::*;
pub use mempool::*;
pub use memset::*;
pub use unified::*;

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

pub trait DevPtr<T> {
    fn as_ptr(&self) -> *const T;
    fn len(&self) -> usize;
    fn get_dev(&self) -> super::device::Device;
    fn size(&self) -> usize {
        self.len() * size_of::<T>()
    }
}

macro_rules! impl_buf {
    ($($ty:ty)*) => {$(
        impl<T: super::DevRepr> StreamBound for $ty {
            #[inline(always)]
            fn stream(&self) -> &super::stream::HipStream {
                &self.stream
            }
        }

        impl<T: super::DevRepr> EventSync for $ty {
            #[inline(always)]
            fn latesync(&self) -> Result<(), crate::hip::HipError> {
                if self.stream.dev.has_mem_pools { return Ok(()); }
                self.event.record(&self.stream, None)
            }

            #[inline(always)]
            fn sync(&self) -> Result<(), crate::hip::HipError> {
                self.event.sync()
            }
        }
    )*};
}

impl_buf!(
    HipBuffer<T> HipSlice<'_, T> HipSliceMut<'_, T>
    HostBuffer<T> HostSlice<'_, T> HostSliceMut<'_, T>
    UnifiedBuffer<T> UnifiedSlice<'_, T> UnifiedSliceMut<'_, T>
);

macro_rules! impl_dev {
    ($($ty:ty)*) => {$(
        impl<T: super::DevRepr> DevPtr<T> for $ty {
            #[inline(always)]
            fn as_ptr(&self) -> *const T {
                self.ptr
            }

            #[inline(always)]
            fn len(&self) -> usize {
                self.len
            }

            #[inline(always)]
            fn get_dev(&self) -> super::device::Device {
                self.stream.dev
            }
        }
    )*};
}

impl_dev!(
    HipBuffer<T> HipSlice<'_, T> HipSliceMut<'_, T>
    UnifiedBuffer<T> UnifiedSlice<'_, T> UnifiedSliceMut<'_, T>
);

macro_rules! impl_host {
    ($($ty:ty)*) => {$(
        impl<T: super::DevRepr> DevPtr<T> for $ty {
            #[inline(always)]
            fn as_ptr(&self) -> *const T {
                self.as_dev_ptr().unwrap()
            }

            #[inline(always)]
            fn len(&self) -> usize {
                self.len
            }

            #[inline(always)]
            fn get_dev(&self) -> super::device::Device {
                self.stream.dev
            }
        }
    )*};
}

impl_host!(
    HostBuffer<T> HostSlice<'_, T> HostSliceMut<'_, T>
);
