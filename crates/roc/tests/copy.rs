#[test]
#[cfg(any(feature = "dynamic-loading", hip))]
fn htod_copy_preserves_bytes_outside_requested_range() {
    use roc::{
        hip::stream::{HipStream, StreamKind},
        is_amdhip64_present,
    };

    if !is_amdhip64_present() {
        return;
    }

    let stream = HipStream::create(StreamKind::Default).unwrap();

    let src_bytes: Vec<u8> = (0..64).collect();
    let initial_dst = [0xAA_u8; 64];

    let src = stream.alloc_host_init(&src_bytes).unwrap();
    let mut dst = stream.alloc(64).unwrap();
    let dst_host = stream.host_slice(&initial_dst).unwrap();

    dst.copy_from_async(&dst_host, 0, 0, 64).unwrap();
    dst.copy_from_async(&src, 8, 24, 16).unwrap();

    let mut result = stream.alloc_host(64).unwrap();
    dst.copy_to_async(&mut result, 0, 0, 64).unwrap();

    stream.sync().unwrap();

    assert_eq!(&result[..24], &[0xAA; 24]);
    assert_eq!(&result[24..40], &src_bytes[8..24]);
    assert_eq!(&result[40..], &[0xAA; 24]);
}
