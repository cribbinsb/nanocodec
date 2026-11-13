//! Malformed inputs found by the decoder fuzz targets (`fuzz/`): each must
//! decode or fail with an error, never panic. Inputs are minimised fuzzer
//! outputs, given as the packets the browser path received.

#![forbid(unsafe_code)]

use nc_hevc::Decoder;

/// An IDR slice, then a slice whose `slice_pic_parameter_set_id` (96) is
/// beyond the 64 parameter set slots (§7.4.7.1): it indexed past them.
#[test]
fn slice_pps_id_beyond_range_is_an_error() {
    let packets: [&[u8]; 2] = [
        &[0x00, 0x00, 0x01, 0x26, 0x01, 0xaf],
        &[0x00, 0x00, 0x01, 0x00, 0x01, 0x81, 0x85],
    ];
    let mut decoder = Decoder::new();
    let _ = decoder.decode_shown(packets[0]);
    assert!(decoder.decode_shown(packets[1]).is_err());
}
