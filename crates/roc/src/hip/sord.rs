#![allow(clippy::partialeq_ne_impl)]

use core::{
    cell::{Cell, UnsafeCell}, fmt::{Debug, Display}, mem::MaybeUninit, ops::{Deref, DerefMut},
};

use alloc::{boxed::Box, fmt};
use rocm_sys::hip::{hipHostRegister, hipHostUnregister};

use crate::hip::{DevRepr, HipError, event::HipEvent};

/// Handle to asynchronously written to value (**S**ync **o**n **r**ea**d**)
///
/// This is an uninitialized value that is safe to read only after synchronizing a stream
#[repr(transparent)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
pub struct Sord<T: DevRepr> {
    pub(crate) inner: Box<SordInner<T>>,
}

impl<T: DevRepr + Default> Default for Sord<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: PartialEq + DevRepr> PartialEq for Sord<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Self::get(self).eq(Self::get(other))
    }
    fn ne(&self, other: &Self) -> bool {
        Self::get(self).ne(Self::get(other))
    }
}

impl<T: Eq + DevRepr> Eq for Sord<T> {}

impl<T: PartialEq + DevRepr> PartialEq<T> for Sord<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        Self::get(self).eq(other)
    }
    fn ne(&self, other: &T) -> bool {
        Self::get(self).ne(other)
    }
}

impl<T: PartialOrd + DevRepr> PartialOrd for Sord<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Self::get(self).partial_cmp(Self::get(other))
    }
    fn ge(&self, other: &Self) -> bool {
        Self::get(self).ge(Self::get(other))
    }
    fn gt(&self, other: &Self) -> bool {
        Self::get(self).gt(Self::get(other))
    }
    fn le(&self, other: &Self) -> bool {
        Self::get(self).le(Self::get(other))
    }
    fn lt(&self, other: &Self) -> bool {
        Self::get(self).lt(Self::get(other))
    }
}

impl<T: PartialOrd + DevRepr> PartialOrd<T> for Sord<T> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        Self::get(self).partial_cmp(other)
    }
    fn ge(&self, other: &T) -> bool {
        Self::get(self).ge(other)
    }
    fn gt(&self, other: &T) -> bool {
        Self::get(self).gt(other)
    }
    fn le(&self, other: &T) -> bool {
        Self::get(self).le(other)
    }
    fn lt(&self, other: &T) -> bool {
        Self::get(self).lt(other)
    }
}

impl<T: Ord + DevRepr> Ord for Sord<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        Self::get(self).cmp(Self::get(other))
    }
    fn clamp(self, min: Self, max: Self) -> Self {
        Self::new(Self::into_inner(self).clamp(Self::into_inner(min), Self::into_inner(max)))
    }
    fn max(self, other: Self) -> Self {
        Self::new(Self::into_inner(self).max(Self::into_inner(other)))
    }
    fn min(self, other: Self) -> Self {
        Self::new(Self::into_inner(self).min(Self::into_inner(other)))
    }
}

impl<T: Debug + DevRepr> Debug for Sord<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::get(self).fmt(f)
    }
}

impl<T: Display + DevRepr> Display for Sord<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::get(self).fmt(f)
    }
}

impl<T: DevRepr> Sord<T> {
    pub fn new(val: T) -> Self {
        Self {
            inner: Box::new(SordInner {
                inner: UnsafeCell::new(MaybeUninit::new(val)),
                event: unsafe { HipEvent::invalid() },
                synced: Cell::new(true),
                mapped: Cell::new(false),
            }),
        }
    }

    /// # Safety
    ///
    /// The inner value must be written to through [`Self::as_mut_ptr`] before being read
    ///
    /// The caller must guarantee that:
    /// - The value will be initialized before the event completes.
    /// - The value will be initialized exactly once.
    /// - No conflicting accesses occur while the GPU is writing.
    /// - The event represents completion of the operation writing the value.
    pub unsafe fn new_async(inner: UnsafeCell<MaybeUninit<T>>, event: HipEvent) -> Self {
        Self {
            inner: Box::new(SordInner {
                inner,
                event,
                synced: Cell::new(false),
                mapped: Cell::new(false),
            }),
        }
    }

    pub fn map(this: &Self) -> Result<(), HipError> {
        if !this.inner.mapped.get() {
            try_err!(unsafe {
                hipHostRegister(
                    this.inner.inner.get().cast(),
                    size_of::<T>(),
                    0,
                )
            });

            this.inner.mapped.set(true);
        }

        Ok(())
    }

    pub fn as_ptr(this: &Self) -> *const T {
        this.inner.inner.get() as *const _
    }

    pub fn as_mut_ptr(this: &mut Self) -> *mut T {
        this.inner.inner.get().cast()
    }

    pub fn sync(this: &Self) -> Result<(), HipError> {
        if !this.inner.synced.get() {
            this.inner.event.sync()?;

            this.inner.synced.set(true);
        }

        Ok(())
    }

    pub fn get(this: &Self) -> &T {
        Self::try_get(this).expect("failed to synchronize value")
    }

    pub unsafe fn get_unchecked(this: &Self) -> &T {
        unsafe { &*this.inner.inner.get().cast() }
    }

    pub fn try_get(this: &Self) -> Result<&T, HipError> {
        Self::sync(this)?;
        Ok(unsafe { Self::get_unchecked(this) })
    }

    pub fn get_mut(this: &mut Self) -> &mut T {
        Self::try_get_mut(this).expect("failed to synchronize value")
    }

    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        unsafe { &mut *this.inner.inner.get().cast() }
    }

    pub fn try_get_mut(this: &mut Self) -> Result<&mut T, HipError> {
        Self::sync(this)?;
        Ok(unsafe { Self::get_mut_unchecked(this) })
    }

    pub fn into_inner(this: Self) -> T {
        Self::try_into_inner(this).expect("failed to synchronize value")
    }

    pub unsafe fn into_inner_unchecked(this: Self) -> T {
        unsafe { Self::as_ptr(&this).cast::<T>().read() }
    }

    pub fn try_into_inner(this: Self) -> Result<T, HipError> {
        Self::sync(&this)?;
        Ok(unsafe { Self::into_inner_unchecked(this) })
    }

    pub fn dbg(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    where 
        T: Debug,
    {
        f.debug_struct("Sord")
            .field("inner", &Self::get(self))
            .field("synced", &self.inner.synced.get())
            .field("mapped", &self.inner.mapped.get())
            .finish()
    }
}

impl<T: DevRepr> Deref for Sord<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        Self::get(self)
    }
}

impl<T: DevRepr> DerefMut for Sord<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Self::get_mut(self)
    }
}

impl<T: DevRepr> AsRef<T> for Sord<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: DevRepr> AsMut<T> for Sord<T> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: DevRepr> From<T> for Sord<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: DevRepr> Drop for Sord<T> {
    fn drop(&mut self) {
        log_if_err!("failed to synchronize value": Self::sync(self));

        if self.inner.mapped.get() {
            log_if_err!("failed to unregister Sord memory": unsafe {
                hipHostUnregister(self.inner.inner.get().cast())
            });
        }
    }
}

mod private {
    use super::*;

    #[repr(align(16))]
    pub struct SordInner<T> {
        pub(super) inner: UnsafeCell<MaybeUninit<T>>,
        pub(crate) event: HipEvent,
        pub(super) synced: Cell<bool>,
        pub(super) mapped: Cell<bool>,
    }

    unsafe impl<T: Send> Send for SordInner<T> {}
}

use private::SordInner;
