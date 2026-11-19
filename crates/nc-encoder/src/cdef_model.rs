//! The CDEF strength model shared by the AV1-family backends (AV1, AV2):
//! frame strengths predicted from the quantiser, which backends use as a
//! search candidate, and the short ladder of strengths a probe measures
//! beside it ([`cdef_probe_candidates`], [`cdef_probe_luma_refinement`]).
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

/// Luma primary strengths of the probe's grid, weakest first: the AV2
/// search's coarse grid over the whole 0..=15 range.
const PROBE_LUMA_PRIMARIES: [u8; 5] = [0, 3, 6, 10, 15];
/// Coded luma secondary strengths of the probe's grid.
const PROBE_LUMA_SECONDARIES: [u8; 2] = [0, 2];

/// Luma (primary, coded secondary) strengths a probe measures first,
/// besides off and the model's; [`cdef_probe_luma_refinement`] then
/// measures the grid's neighbours of the best.
///
/// The model alone is far too weak at real-time quantisers: it predicts
/// luma primary 1-2 with no secondary (off below AV1 step ~100), where
/// AV2's strength search picks primary 3-15, usually with secondary 2, on
/// every clip tried (CTC `FourPeople` and `Johnny`, MOT20-05, VC-1K phone
/// video), and dark phone video loses most without it. A primary of 1 is
/// also no filter at all on 8x8 units of variance below 1024, where the
/// variance adjustment (`cdef_adjust_strength`) rounds it to 0; the
/// ladder's primaries from 3 survive it on every unit, and its secondary
/// is not adjusted.
pub const CDEF_PROBE_LUMA: [(u8, u8); 3] = [(0, 2), (3, 2), (10, 2)];
/// Chroma strengths a probe measures besides off and the model's: the
/// search's chroma primaries (chroma strengths are not variance-adjusted).
pub const CDEF_PROBE_CHROMA: [(u8, u8); 3] = [(2, 0), (5, 0), (9, 0)];

/// The strengths a probe measures for one plane type: off first (so ties
/// keep the plane unfiltered), then the model's `model`, then `ladder`,
/// each once.
#[must_use]
pub fn cdef_probe_candidates(model: (u8, u8), ladder: &[(u8, u8)]) -> Vec<(u8, u8)> {
    let mut list = vec![(0, 0)];
    for &strength in std::iter::once(&model).chain(ladder) {
        if !list.contains(&strength) {
            list.push(strength);
        }
    }
    list
}

/// The luma strengths a probe measures after its first pass found `best`:
/// the next stronger primary of the grid with each grid secondary, less
/// those `measured` already. Where the first pass and the AV2 search
/// disagreed, the search mostly chose exactly these (10/2 against 15/2 or
/// 15/0, 3/2 against 6/2 or 6/0, 0/2 against 3/0; CTC, MOT20-05 and VC-1K,
/// 720 frames).
#[must_use]
pub fn cdef_probe_luma_refinement(best: (u8, u8), measured: &[(u8, u8)]) -> Vec<(u8, u8)> {
    let Some(&stronger) = PROBE_LUMA_PRIMARIES.iter().find(|&&p| p > best.0) else {
        return Vec::new();
    };
    PROBE_LUMA_SECONDARIES
        .iter()
        .map(|&secondary| (stronger, secondary))
        .filter(|strength| !measured.contains(strength))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_lists_off_then_model_then_ladder_once() {
        assert_eq!(
            cdef_probe_candidates((3, 2), &CDEF_PROBE_LUMA),
            vec![(0, 0), (3, 2), (0, 2), (10, 2)]
        );
        assert_eq!(
            cdef_probe_candidates((0, 0), &CDEF_PROBE_CHROMA),
            vec![(0, 0), (2, 0), (5, 0), (9, 0)]
        );
    }

    #[test]
    fn refinement_tries_the_next_stronger_primary_unmeasured() {
        let measured = cdef_probe_candidates((1, 0), &CDEF_PROBE_LUMA);
        assert_eq!(
            cdef_probe_luma_refinement((10, 2), &measured),
            vec![(15, 0), (15, 2)]
        );
        assert_eq!(cdef_probe_luma_refinement((0, 2), &measured), vec![(3, 0)]);
        assert_eq!(cdef_probe_luma_refinement((15, 2), &measured), Vec::new());
    }
}
