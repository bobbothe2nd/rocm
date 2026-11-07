use rocm_include::ctype::*;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uchar1 {
    pub x: uchar,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uchar2 {
    pub x: uchar,
    pub y: uchar,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uchar3 {
    pub x: uchar,
    pub y: uchar,
    pub z: uchar,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uchar4 {
    pub x: uchar,
    pub y: uchar,
    pub z: uchar,
    pub w: uchar,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct char1 {
    pub x: char,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct char2 {
    pub x: char,
    pub y: char,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct char3 {
    pub x: char,
    pub y: char,
    pub z: char,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct char4 {
    pub x: char,
    pub y: char,
    pub z: char,
    pub w: char,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ushort1 {
    pub x: ushort,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ushort2 {
    pub x: ushort,
    pub y: ushort,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ushort3 {
    pub x: ushort,
    pub y: ushort,
    pub z: ushort,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ushort4 {
    pub x: ushort,
    pub y: ushort,
    pub z: ushort,
    pub w: ushort,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct short1 {
    pub x: short,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct short2 {
    pub x: short,
    pub y: short,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct short3 {
    pub x: short,
    pub y: short,
    pub z: short,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct short4 {
    pub x: short,
    pub y: short,
    pub z: short,
    pub w: short,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uint1 {
    pub x: uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uint2 {
    pub x: uint,
    pub y: uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uint3 {
    pub x: uint,
    pub y: uint,
    pub z: uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct uint4 {
    pub x: uint,
    pub y: uint,
    pub z: uint,
    pub w: uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct int1 {
    pub x: int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct int2 {
    pub x: int,
    pub y: int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct int3 {
    pub x: int,
    pub y: int,
    pub z: int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct int4 {
    pub x: int,
    pub y: int,
    pub z: int,
    pub w: int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulong1 {
    pub x: ulong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulong2 {
    pub x: ulong,
    pub y: ulong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulong3 {
    pub x: ulong,
    pub y: ulong,
    pub z: ulong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulong4 {
    pub x: ulong,
    pub y: ulong,
    pub z: ulong,
    pub w: ulong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct long1 {
    pub x: long,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct long2 {
    pub x: long,
    pub y: long,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct long3 {
    pub x: long,
    pub y: long,
    pub z: long,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct long4 {
    pub x: long,
    pub y: long,
    pub z: long,
    pub w: long,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulonglong1 {
    pub x: ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulonglong2 {
    pub x: ulonglong,
    pub y: ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulonglong3 {
    pub x: ulonglong,
    pub y: ulonglong,
    pub z: ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ulonglong4 {
    pub x: ulonglong,
    pub y: ulonglong,
    pub z: ulonglong,
    pub w: ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct longlong1 {
    pub x: longlong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct longlong2 {
    pub x: longlong,
    pub y: longlong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct longlong3 {
    pub x: longlong,
    pub y: longlong,
    pub z: longlong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct longlong4 {
    pub x: longlong,
    pub y: longlong,
    pub z: longlong,
    pub w: longlong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct float1 {
    pub x: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct float2 {
    pub x: f32,
    pub y: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct double1 {
    pub x: f64,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct double2 {
    pub x: f64,
    pub y: f64,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct double3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct double4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

pub type hipFloatComplex = float2;
pub type hipComplex = hipFloatComplex;
pub type hipDoubleComplex = double2;
