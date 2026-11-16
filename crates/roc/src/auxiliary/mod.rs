use core::ops::{Bound, RangeBounds};

pub struct ResolvedRange<T> {
    pub start: T,
    pub end: T,
    pub len: T,
}

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
    ResolvedRange { start, end, len: end - start }
}
