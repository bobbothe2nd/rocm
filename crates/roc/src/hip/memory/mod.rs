mod copy;
mod devmem;
mod hostmem;
mod memset;
mod unified;

pub use copy::*;
pub use devmem::*;
pub use hostmem::*;
pub use memset::*;
pub use unified::*;

#[cfg(any(windows, unix))]
mod mempool;
#[cfg(any(windows, unix))]
pub use mempool::*;

#[cfg(all(feature = "std", any(windows, unix)))]
mod extmem;
#[cfg(all(feature = "std", any(windows, unix)))]
pub use extmem::*;

use crate::hip::event::HipEvent;

pub trait EventSync {
    fn event(&self) -> &HipEvent;

    /// Records a synchronization checkpoint for the buffer, or a logical buffer usage.
    fn latesync(&self) -> Result<(), super::HipError>;

    /// Synchronizes the buffer up to it's last `latesync` or usage.
    ///
    /// These are logically equivalent (other than edge cases and errors):
    ///
    /// ```no_run
    /// # #[cfg(feature = "dynamic-loading")] {
    /// # use roc::hip::{*, memory::EventSync as _};
    /// # let stream = stream::HipStream::create(stream::StreamKind::Default).unwrap();
    /// # let buf = stream.alloc::<u8>(128).unwrap();
    /// buf.latesync().unwrap();
    /// buf.sync().unwrap();
    /// # }
    /// ```
    ///
    /// and
    ///
    /// ```no_run
    /// # #[cfg(feature = "dynamic-loading")] {
    /// # use roc::hip::*;
    /// # let stream = stream::HipStream::create(stream::StreamKind::Default).unwrap();
    /// stream.sync().unwrap();
    /// # }
    /// ```
    ///
    /// assuming `buf` was allocated on `stream`.
    fn sync(&self) -> Result<(), super::HipError> {
        self.event().sync()
    }
}

pub trait StreamBound {
    fn stream(&self) -> &super::stream::HipStream;
}

pub unsafe trait MutPtr {}

/// # Safety
///
/// It is assumed that implementors of this trait will return a valid
/// pointer from [`Self::as_ptr`]. It must be valid for reads up to
/// `self.len() * size_of::<T>()`.
pub unsafe trait DevPtr<T> {
    fn as_ptr(&self) -> *const T;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get_dev(&self) -> super::device::Device;
    fn size(&self) -> usize {
        self.len() * size_of::<T>()
    }
}

pub unsafe trait HostPtr<T> {
    fn as_ptr(&self) -> *const T;
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
            fn event(&self) -> &HipEvent {
                &self.event
            }

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
        unsafe impl<T: super::DevRepr> DevPtr<T> for $ty {
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
);

unsafe impl<T: super::DevRepr> MutPtr for HipBuffer<T> {}
unsafe impl<T: super::DevRepr> MutPtr for HipSliceMut<'_, T> {}

macro_rules! impl_unified {
    ($($ty:ty)*) => {$(
        unsafe impl<T: super::DevRepr> DevPtr<T> for $ty {
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
        unsafe impl<T: super::DevRepr> HostPtr<T> for $ty {
            #[inline(always)]
            fn as_ptr(&self) -> *const T {
                self.ptr
            }
        }
    )*};
}

impl_unified!(
    UnifiedBuffer<T> UnifiedSlice<'_, T> UnifiedSliceMut<'_, T>
);

unsafe impl<T: super::DevRepr> MutPtr for UnifiedBuffer<T> {}
unsafe impl<T: super::DevRepr> MutPtr for UnifiedSliceMut<'_, T> {}

macro_rules! impl_host {
    ($($ty:ty)*) => {$(
        unsafe impl<T: super::DevRepr> DevPtr<T> for $ty {
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
        unsafe impl<T: super::DevRepr> HostPtr<T> for $ty {
            #[inline(always)]
            fn as_ptr(&self) -> *const T {
                self.ptr
            }
        }
    )*};
}

impl_host!(
    HostBuffer<T> HostSlice<'_, T> HostSliceMut<'_, T>
);

unsafe impl<T: super::DevRepr> MutPtr for HostBuffer<T> {}
unsafe impl<T: super::DevRepr> MutPtr for HostSliceMut<'_, T> {}
