//! SAO offsets from statistics, after HM's `deriveOffsets` and
//! `estIterOffset` (BSD-3-Clause, ITU/ISO/IEC; `THIRD_PARTY.md`): each
//! category's (or band's) offset starts at the rounded mean of `source −
//! deblocked`, clipped to the coded range, and moves towards zero while
//! that lowers `distortion + λ·bits`; band offset then keeps the four
//! consecutive bands (modulo 32, HEVC §8.7.3.2) of least summed cost.
//!
//! Distortion is the change in squared error from adding offset `o` to `n`
//! samples whose errors sum to `s`: `n·o² − 2·o·s` (HM `estSaoDist`).

use nc_dsp::{SaoBandStats, SaoStats};

use super::{BANDS, MAX_OFFSET};

/// Offsets of one plane's type and their squared-error change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Derived {
    /// Offsets of edge categories 1..4, or of the four bands from the
    /// position.
    pub offsets: [i8; 4],
    /// Squared-error change of applying them (negative: better).
    pub distortion: i64,
}

/// Band-offset choice of one plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Band {
    /// `sao_band_position`.
    pub position: u8,
    /// The four bands' offsets and their distortion.
    pub offsets: Derived,
}

/// Squared-error change of adding `offset` to `count` samples whose
/// errors sum to `sum`.
pub(super) fn delta(count: i64, sum: i64, offset: i32) -> i64 {
    let o = i64::from(offset);
    count * o * o - 2 * o * sum
}

/// The mean error `sum / count` rounded half away from zero and clipped to
/// the coded range (HM `xRoundIbdi` at 8 bits).
fn rounded_mean(count: i64, sum: i64) -> i32 {
    if count == 0 {
        return 0;
    }
    let rounded = (2 * sum.abs() + count) / (2 * count);
    (sum.signum() * rounded).clamp(-i64::from(MAX_OFFSET), i64::from(MAX_OFFSET)) as i32
}

/// Bits of `sao_offset_abs` = `|offset|` (truncated unary, cMax
/// [`MAX_OFFSET`]), plus its sign for band offset.
pub(super) fn magnitude_bits(offset: i32, band: bool) -> u32 {
    let abs = offset.unsigned_abs();
    let unary = (abs + 1).min(MAX_OFFSET as u32);
    unary + u32::from(band && abs != 0)
}

/// The offset of least `distortion + λ·bits` from `start` towards zero
/// (zero costs one bit), its distortion and cost (HM `estIterOffset`).
fn refine(start: i32, count: i64, sum: i64, lambda: f64, band: bool) -> (i32, i64, f64) {
    let mut best = (0, 0, lambda * f64::from(magnitude_bits(0, band)));
    let mut offset = start;
    while offset != 0 {
        let distortion = delta(count, sum, offset);
        let cost = distortion as f64 + lambda * f64::from(magnitude_bits(offset, band));
        if cost < best.2 {
            best = (offset, distortion, cost);
        }
        offset -= offset.signum();
    }
    best
}

/// Edge offsets of one plane and class: categories 1 and 2 only brighten,
/// 3 and 4 only darken (§7.4.9.3.2), so a mean of the other sign gives 0.
pub(super) fn edge(stats: &SaoStats, lambda: f64) -> Derived {
    let mut out = Derived {
        offsets: [0; 4],
        distortion: 0,
    };
    for category in 0..4 {
        let (count, sum) = (stats.count[category], stats.sum[category]);
        let mean = rounded_mean(count, sum);
        let start = if category < 2 {
            mean.max(0)
        } else {
            mean.min(0)
        };
        let (offset, distortion, _) = refine(start, count, sum, lambda, false);
        out.offsets[category] = offset as i8;
        out.distortion += distortion;
    }
    out
}

/// The best band position and the four offsets from it. Positions wrap
/// (bands 30, 31, 0, 1 is legal), unlike HM's search, which stops at 28.
pub(super) fn band(stats: &SaoBandStats, lambda: f64) -> Band {
    let bands = usize::from(BANDS);
    let mut chosen = [(0i32, 0i64, 0f64); BANDS as usize];
    for (b, slot) in chosen.iter_mut().enumerate() {
        let (count, sum) = (i64::from(stats.count[b]), i64::from(stats.sum[b]));
        *slot = refine(rounded_mean(count, sum), count, sum, lambda, true);
    }
    let window = |position: usize| (0..4).map(move |k| (position + k) % bands);
    let cost = |position: usize| window(position).map(|b| chosen[b].2).sum::<f64>();
    let position = (0..bands)
        .min_by(|&a, &b| cost(a).total_cmp(&cost(b)))
        .expect("32 positions");
    let mut offsets = Derived {
        offsets: [0; 4],
        distortion: 0,
    };
    for (k, b) in window(position).enumerate() {
        offsets.offsets[k] = chosen[b].0 as i8;
        offsets.distortion += chosen[b].1;
    }
    Band {
        position: position as u8,
        offsets,
    }
}

/// Squared-error change of edge `offsets` on a plane's class statistics.
pub(super) fn edge_distortion(stats: &SaoStats, offsets: [i8; 4]) -> i64 {
    (0..4)
        .map(|c| delta(stats.count[c], stats.sum[c], i32::from(offsets[c])))
        .sum()
}

/// Squared-error change of band `offsets` from `position`.
pub(super) fn band_distortion(stats: &SaoBandStats, position: u8, offsets: [i8; 4]) -> i64 {
    offsets
        .iter()
        .enumerate()
        .map(|(k, &o)| {
            let b = (usize::from(position) + k) % usize::from(BANDS);
            delta(
                i64::from(stats.count[b]),
                i64::from(stats.sum[b]),
                i32::from(o),
            )
        })
        .sum()
}
