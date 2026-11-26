use roc::{
    hip::stream::{HipStream, StreamKind},
    rocblas::RocblasHandle,
};

fn main() {
    let stream = HipStream::create(StreamKind::Default).unwrap();
    let rocblas = RocblasHandle::new(&stream).unwrap();

    let mut input = stream.alloc::<f32>(64).unwrap();
    input.memset_async(3.0).unwrap();

    let val = rocblas.asum(&input).unwrap();

    assert_eq!(val, 3.0 * 64.0);
}
