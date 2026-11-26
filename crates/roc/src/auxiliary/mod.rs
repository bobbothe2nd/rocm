use core::ops::{Bound, RangeBounds};

pub mod public;

#[allow(dead_code)]
pub struct ResolvedRange<T> {
    pub start: T,
    #[expect(dead_code)]
    pub end: T,
    pub len: T,
}

#[allow(dead_code)]
pub fn resolve_range(range: impl RangeBounds<usize>, total_len: usize) -> ResolvedRange<usize> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n + 1,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => total_len,
    };
    assert!(
        start <= end && end <= total_len,
        "slice range out of bounds"
    );
    ResolvedRange {
        start,
        end,
        len: end - start,
    }
}

#[expect(unused_imports)]
use ::rocm_sys;

#[cfg(feature = "hip")]
#[expect(unused_imports)]
use ::bitflags;

#[cfg(feature = "half")]
#[expect(unused_imports)]
use ::half;
#[cfg(feature = "lowp")]
#[expect(unused_imports)]
use ::lowp;
#[cfg(feature = "stable_auto")]
#[expect(unused_imports)]
use ::stable_auto;
#[cfg(feature = "log")]
#[expect(unused_imports)]
use ::log;

#[cfg(windows)]
#[expect(unused_imports)]
use ::windows_sys;
#[cfg(unix)]
#[expect(unused_imports)]
use ::libc;
