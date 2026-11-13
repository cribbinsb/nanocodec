//! libFuzzer entry for the VVC decoder: `common::drive::vvc`.
#![no_main]
#![forbid(unsafe_code)]

mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| common::drive::vvc(data));
