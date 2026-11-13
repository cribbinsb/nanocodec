//! Malformed and edge-case inputs found by the decoder fuzz targets
//! (`fuzz/`): each must decode or fail with an error, never panic.
//! The inputs are minimised fuzzer outputs under `tests/malformed/`.

#![forbid(unsafe_code)]

use nc_av1::Decoder;

/// A mutated prefix of the AOM conformance vector `av1-1-b8-01-size-18x32`
/// (BSD-2-Clause, Alliance for Open Media): blocks start beyond the 18-sample
/// frame edge (`MiCols` rounds up to 24), whose deblocking line ranges were
/// once negative (an arithmetic overflow).
#[test]
fn deblocking_skips_areas_beyond_the_frame() {
    let packet = include_bytes!("malformed/deblock_area_beyond_frame.obu");
    let mut decoder = Decoder::new();
    let _ = decoder.decode_shown(packet);
}

/// A key frame whose film grain scaling points do not increase
/// (§6.8.20): building the scaling lookup divided by their difference,
/// which was negative or zero.
#[test]
fn film_grain_points_must_increase() {
    let packet = include_bytes!("malformed/film_grain_points_not_increasing.obu");
    let mut decoder = Decoder::new();
    let _ = decoder.decode_shown(packet);
}
