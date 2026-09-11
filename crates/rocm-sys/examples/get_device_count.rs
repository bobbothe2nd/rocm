use rocm_sys::hip::{hipGetDeviceCount, hipInit};

fn main() {
    let mut count = -1;

    let err = unsafe { hipInit(0) };

    println!("hipInit = {err:?}");

    let err = unsafe { hipGetDeviceCount(&mut count) };

    println!("hipGetDeviceCount = {err:?}");
    println!("device count = {count}");
}
