//! libFuzzer entry for the HEVC decoder: `common::drive::hevc`.
#![no_main]
#![forbid(unsafe_code)]

mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| common::drive::hevc(data));
