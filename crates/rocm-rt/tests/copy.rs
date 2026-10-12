use rocm_rt::hip::stream::Stream;

#[test]
fn htod_copy_preserves_bytes_outside_requested_range() {
    let stream = Stream::create().unwrap();

    let src_bytes: Vec<u8> = (0..64).collect();
    let initial_dst = [0xAA_u8; 64];

    let src = stream.alloc_host_init(&src_bytes).unwrap();
    let dst = stream.alloc(64).unwrap();
    let dst_host = stream.alloc_host_init(&initial_dst).unwrap();

    stream
        .copy_htod(&dst_host.borrowed(), &dst, 0, 0, 64)
        .unwrap();

    stream.copy_htod(&src.borrowed(), &dst, 8, 24, 16).unwrap();

    stream.sync().unwrap();

    let result = stream.alloc_host(64).unwrap();
    dst.copy_to_host(&result.borrowed(), 0, 0, 64).unwrap();

    assert_eq!(&result.as_slice()[..24], &[0xAA; 24]);
    assert_eq!(&result.as_slice()[24..40], &src_bytes[8..24]);
    assert_eq!(&result.as_slice()[40..], &[0xAA; 24]);
}
