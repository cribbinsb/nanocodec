//! libFuzzer entry for the H.264 decoder: `common::drive::h264`.
#![no_main]
#![forbid(unsafe_code)]

mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| common::drive::h264(data));
