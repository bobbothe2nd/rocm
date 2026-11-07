#![allow(non_camel_case_types)]

use core::ffi::*;

pub type rvoid = ();
pub type void = c_void;
pub type char = c_char;
pub type uchar = c_uchar;
pub type short = c_short;
pub type ushort = c_ushort;
pub type int = c_int;
pub type uint = c_uint;
pub type long = c_long;
pub type ulong = c_ulong;
pub type longlong = c_longlong;
pub type ulonglong = c_ulonglong;
pub type intptr_t = isize;
pub type uintptr_t = usize;
pub type size_t = usize;
pub type float = c_float;
pub type double = c_double;
pub type uint8_t = u8;
pub type uint16_t = u16;
pub type uint32_t = u32;
pub type uint64_t = u64;
pub type int8_t = i8;
pub type int16_t = i16;
pub type int32_t = i32;
pub type int64_t = i64;
