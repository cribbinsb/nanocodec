//! libFuzzer entry for the VP9 decoder: `common::drive::vp9`.
#![no_main]
#![forbid(unsafe_code)]

mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| common::drive::vp9(data));
