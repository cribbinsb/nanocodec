//! Arbitrary packet input to the public HEVC subset decoder.
#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut decoder = nc_hevc::Decoder::new();
    let _ = decoder.decode(data);
});
