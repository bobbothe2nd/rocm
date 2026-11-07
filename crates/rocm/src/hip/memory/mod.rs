use rocm_sys::hip::hipMemPool_t;

mod devmem;
mod hostmem;
mod memset;

pub use devmem::*;
pub use hostmem::*;
pub use memset::*;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemPool {
    mem_pool: hipMemPool_t,
}
