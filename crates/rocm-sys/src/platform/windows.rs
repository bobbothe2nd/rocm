use rocm_include::ctype::{long, uint, void};

#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct _LUID {
    LowPart: u32,
    HighPart: long,
}

pub type LUID = _LUID;
pub type PLUID = *mut _LUID;

pub type HANDLE = *mut void;
pub type D3DKMT_HANDLE = uint;

#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct SECURITY_ATTRIBUTES {
    pub nLength: u32,
    pub lpSecurityDescriptor: *mut core::ffi::c_void,
    pub bInheritHandle: i32,
}
pub type LPSECURITYATTRIBUTES = *mut SECURITY_ATTRIBUTES;
