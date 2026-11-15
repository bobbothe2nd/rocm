use core::ops::{Bound, RangeBounds};

pub struct ResolvedRange<T> {
    pub start: T,
    pub end: T,
    pub size: T,
}

pub fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> ResolvedRange<usize> {
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Excluded(start) => start + 1,
        Bound::Included(start) => *start,
    };
    let end = match range.end_bound() {
        Bound::Unbounded => len,
        Bound::Excluded(end) => end - 1,
        Bound::Included(end) => *end,
    };
    let size = end - start;

    ResolvedRange { start, end, size }
}
