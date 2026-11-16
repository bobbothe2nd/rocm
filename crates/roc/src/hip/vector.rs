#![allow(non_camel_case_types)]

#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u8x1 {
    pub x: u8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u8x2 {
    pub x: u8,
    pub y: u8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u8x3 {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u8x4 {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub w: u8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i8x1 {
    pub x: i8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i8x2 {
    pub x: i8,
    pub y: i8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i8x3 {
    pub x: i8,
    pub y: i8,
    pub z: i8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i8x4 {
    pub x: i8,
    pub y: i8,
    pub z: i8,
    pub w: i8,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u16x1 {
    pub x: u16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u16x2 {
    pub x: u16,
    pub y: u16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u16x3 {
    pub x: u16,
    pub y: u16,
    pub z: u16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u16x4 {
    pub x: u16,
    pub y: u16,
    pub z: u16,
    pub w: u16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i16x1 {
    pub x: i16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i16x2 {
    pub x: i16,
    pub y: i16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i16x3 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i16x4 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
    pub w: i16,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u32x1 {
    pub x: u32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u32x2 {
    pub x: u32,
    pub y: u32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u32x3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u32x4 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i32x1 {
    pub x: i32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i32x2 {
    pub x: i32,
    pub y: i32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i32x3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i32x4 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u64x1 {
    pub x: u64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u64x2 {
    pub x: u64,
    pub y: u64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u64x3 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct u64x4 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
    pub w: u64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i64x1 {
    pub x: i64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i64x2 {
    pub x: i64,
    pub y: i64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i64x3 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct i64x4 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub w: i64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f32x1 {
    pub x: f32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f32x2 {
    pub x: f32,
    pub y: f32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f32x3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f32x4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f64x1 {
    pub x: f64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f64x2 {
    pub x: f64,
    pub y: f64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f64x3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct f64x4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}
