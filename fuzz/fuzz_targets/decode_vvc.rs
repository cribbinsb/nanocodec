//! Arbitrary packet and Annex-B sequence input to the public VVC subset decoder.
#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut decoder = nc_vvc::Decoder::new();
    let _ = decoder.decode(data);

    // Keep reference state between pictures so mutations exercise P syntax too.
    let mut decoder = nc_vvc::Decoder::new();
    let starts: Vec<_> = data
        .windows(3)
        .enumerate()
        .filter_map(|(i, w)| (w == [0, 0, 1]).then_some(i))
        .collect();
    let mut packet_start = 0;
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(data.len());
        if start + 5 <= end && matches!(data[start + 4] >> 3, 0 | 8) {
            let _ = decoder.decode(&data[packet_start..end]);
            packet_start = end;
        }
    }
});
