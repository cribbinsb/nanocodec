//! Arbitrary input to the public H264 decoder: as one access unit
//! (`Decoder::decode`), and as a stream of NAL units through
//! `Decoder::decode_nal` and the output queue (multi-slice pictures,
//! reference marking and output order).
#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;

/// The NAL units of Annex-B `data`: the bytes between `00 00 01` start
/// codes (trailing zeros stay with the unit; the decoder tolerates them).
fn nal_units(data: &[u8]) -> Vec<&[u8]> {
    let starts: Vec<usize> = (0..data.len().saturating_sub(2))
        .filter(|&i| data[i..i + 3] == [0, 0, 1])
        .map(|i| i + 3)
        .collect();
    let ends = starts.iter().skip(1).map(|&s| s - 3).chain([data.len()]);
    starts
        .iter()
        .zip(ends)
        .filter(|&(&start, end)| start < end)
        .map(|(&start, end)| &data[start..end])
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let mut decoder = nc_h264::Decoder::new();
    let _ = decoder.decode(data);

    let mut decoder = nc_h264::Decoder::new();
    let mut queue = nc_h264::OutputQueue::default();
    for nal in nal_units(data) {
        if let Ok(Some(frame)) = decoder.decode_nal(nal) {
            let _ = queue.push(frame);
        }
    }
    let _ = queue.flush();
});
