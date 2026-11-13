//! Arbitrary packet input to the public AV2 subset decoder.
#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut decoder = nc_av2::Decoder::default();
    let _ = decoder.decode(data);
});
