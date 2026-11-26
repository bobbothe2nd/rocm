use core::{cell::UnsafeCell, mem::MaybeUninit};

use crate::{
    aux::BUF_MAX,
    blas::Asum,
    hip::{
        DevRepr, DevWrRepr,
        event::HipEvent,
        sord::Sord,
        memory::{DevPtr, EventSync},
    },
    rocblas::{RocblasError, RocblasHandle},
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
        if n > BUF_MAX {
            return Err(RocblasError::InvalidSize);
        }

        let event = HipEvent::create(None)?;
        let mut sord = unsafe { Sord::new_async(UnsafeCell::new(MaybeUninit::<Y>::uninit()), event) };

        unsafe {
            self.blas_asum(x.len() as u32, x.as_ptr(), 1, Sord::as_mut_ptr(&mut sord))?;
        }

        sync_ret_if_err!(self.stream, sord.inner.event.record(&self.stream, None));
        sync_ret_if_err!(self.stream, x.latesync());

        Ok(sord)
    }
}
