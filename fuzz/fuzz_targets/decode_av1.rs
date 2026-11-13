//! libFuzzer entry for the AV1 decoder: `common::drive::av1`.
#![no_main]
#![forbid(unsafe_code)]

mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| common::drive::av1(data));
