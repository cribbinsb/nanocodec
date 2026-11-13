//! The CDEF strength model shared by the AV1-family backends (AV1, AV2):
//! frame strengths predicted from the quantiser, which backends use as a
//! search candidate or directly.
//!
//! Ported from SVT-AV1 (`svt_pick_cdef_from_qp`, camera content;
//! BSD-3-Clause-Clear, Alliance for Open Media).

/// Largest AC step the model was fitted on (AV1 8-bit `ac_qlookup[255]`).
/// Callers on a coarser quantiser scale saturate here: beyond it the fitted
/// polynomials fall back towards zero strength.
pub const CDEF_MODEL_MAX_Q: f32 = 1828.0;

/// Largest primary strength (4-bit syntax).
const MAX_PRIMARY: f32 = 15.0;
/// Largest coded secondary strength (2-bit syntax; 3 means 4).
const MAX_CODED_SECONDARY: f32 = 3.0;

/// (primary, coded secondary) strengths for luma and chroma from the AC
/// quantiser step `q` on AV1's 8-bit scale (`ac_qlookup`), for an intra or
/// an inter frame.
#[must_use]
pub fn cdef_strengths_from_q(q: f32, intra: bool) -> ((u8, u8), (u8, u8)) {
    let poly = |a: f32, b: f32, c: f32| (q * q * a + q * b + c).round();
    let (y1, y2, uv1, uv2) = if intra {
        (
            poly(0.000_003_373_197_4, 0.008_070_594, 0.018_763_4),
            poly(0.000_002_916_734_3, 0.002_779_862_4, 0.007_940_5),
            poly(-0.000_013_079_1, 0.012_892_405, -0.007_483_88),
            poly(0.000_003_265_178_3, 0.000_355_201_83, 0.002_280_92),
        )
    } else {
        (
            poly(-0.000_002_359_394_6, 0.006_861_518_6, 0.027_098_86),
            poly(-0.000_000_576_297_34, 0.001_399_334_5, 0.038_310_67),
            poly(-0.000_000_709_506_9, 0.003_462_884_6, 0.008_870_99),
            poly(0.000_000_238_740_85, 0.000_282_235_85, 0.055_763_07),
        )
    };
    let pri = |v: f32| v.clamp(0.0, MAX_PRIMARY) as u8;
    let sec = |v: f32| v.clamp(0.0, MAX_CODED_SECONDARY) as u8;
    ((pri(y1), sec(y2)), (pri(uv1), sec(uv2)))
}
