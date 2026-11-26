use core::{cell::UnsafeCell, mem::MaybeUninit};

use crate::{
    aux::BUF_MAX, blas::Asum, hip::{
        DevRepr, DevWrRepr, event::HipEvent, memory::{DevPtr, EventSync}, sord::Sord,
    }, rocblas::{RocblasError, RocblasHandle, RocblasPtrMode},
};

mod asum;

impl RocblasHandle {
    pub fn asum<X, Xptr, Y>(&self, x: &Xptr) -> Result<Sord<Y>, RocblasError>
    where
        X: DevRepr,
        Y: DevWrRepr,
        Xptr: DevPtr<X> + EventSync,
        Self: Asum<X, Y>,
    {
        let n = x.len();
        #[cfg(not(feature = "ilp64"))]
        if n > BUF_MAX {
            return Err(RocblasError::InvalidSize);
        }

        let event = HipEvent::create(None)?;
        let mut sord = unsafe { Sord::new_async(UnsafeCell::new(MaybeUninit::<Y>::uninit()), event) };

        if self.get_pointer_mode()? == RocblasPtrMode::Dev {
            Sord::map(&sord)?;
        }

        if n > BUF_MAX {
            #[cfg(feature = "ilp64")]
            unsafe {
                self.blas_asum_64(x.len() as u64, x.as_ptr(), 1, Sord::as_mut_ptr(&mut sord))?;
            }
        } else {
            unsafe {
                self.blas_asum(x.len() as u32, x.as_ptr(), 1, Sord::as_mut_ptr(&mut sord))?;
            }
        }

        sync_ret_if_err!(self.stream, sord.inner.event.record(&self.stream, None));
        sync_ret_if_err!(self.stream, x.latesync());

        Ok(sord)
    }
}
