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
    /// # use roc::hip::*;
    /// # let stream = stream::HipStream::create().unwrap();
    /// stream.sync().unwrap();
    /// # }
    /// ```
    /// 
    /// assuming `buf` was allocated on `stream`.
    fn sync(&self) -> Result<(), super::HipError>;
}
