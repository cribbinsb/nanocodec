//! VP9 deblocking filters of one edge (VP9 §8.8.2; libvpx
//! `vpx_dsp/loopfilter.c` `filter4`/`filter8`/`filter16` and their masks,
//! BSD-3-Clause, see `THIRD_PARTY.md`). Edge scheduling and masks from the
//! mode info stay in nc-vp9.
//!
//! Scalar reference; SIMD versions belong beside it with a bit-exactness
//! test in nc-checkasm. Deeper samples run the same reference with the
//! thresholds, the flatness bound and `filter4`'s signed range scaled by
//! `BitDepth − 8` (`vpx_highbd_lpf_*`, `signed_char_clamp_high`).

use nc_common::{BitDepth, Sample};

/// Per-level thresholds (`loop_filter_thresh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9FilterLimits {
    /// `lim`: largest step inside either side.
    pub(crate) inside: i32,
    /// `mblim`: largest weighted step across the edge.
    pub(crate) boundary: i32,
    /// `hev_thr`: high edge variance.
    pub(crate) variance: i32,
}

impl Vp9FilterLimits {
    /// The thresholds of filter `level` (1..=63) at sharpness 0.
    #[must_use]
    pub fn new(level: u8) -> Self {
        Self::with_sharpness(level, 0)
    }

    /// The thresholds of filter `level` at `sharpness` (0..=7; libvpx
    /// `update_sharpness`): sharper filters shrink the inside limit, so
    /// textured edges are filtered less.
    #[must_use]
    pub fn with_sharpness(level: u8, sharpness: u8) -> Self {
        let shift = u32::from(sharpness > 0) + u32::from(sharpness > 4);
        let mut inside = i32::from(level >> shift);
        if sharpness > 0 {
            inside = inside.min(9 - i32::from(sharpness));
        }
        let inside = inside.max(1);
        Self {
            inside,
            boundary: 2 * (i32::from(level) + 2) + inside,
            variance: i32::from(level >> 4),
        }
    }
}

impl Vp9FilterLimits {
    /// The thresholds scaled to samples of `depth` (`limit16`, `blimit16`,
    /// `thresh16` of libvpx's high-bit-depth masks).
    pub(crate) fn at(self, depth: BitDepth) -> Self {
        let shift = depth.above_eight();
        Self {
            inside: self.inside << shift,
            boundary: self.boundary << shift,
            variance: self.variance << shift,
        }
    }
}

/// Filter width of one edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9FilterWidth {
    /// `filter4` only.
    Four,
    /// `filter8` where flat.
    Eight,
    /// `filter16` where flat out to 8 samples.
    Sixteen,
}

/// `filter_mask` over `p[4..8]` (p3..p0) and `q[0..4]` (q0..q3).
fn filter_mask(p: &[i32; 8], q: &[i32; 8], limits: Vp9FilterLimits) -> bool {
    let (p3, p2, p1, p0) = (p[4], p[5], p[6], p[7]);
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    let steps = [p3 - p2, p2 - p1, p1 - p0, q1 - q0, q2 - q1, q3 - q2];
    steps.iter().all(|d| d.abs() <= limits.inside)
        && 2 * (p0 - q0).abs() + (p1 - q1).abs() / 2 <= limits.boundary
}

/// Filter a line that passed [`filter_mask`], choosing the filter with
/// `flat_mask4` / `flat_mask5` (flat: every step at most `1 << (BitDepth −
/// 8)`) over taps `p[0..8]` (p7..p0) and `q[0..8]` (q0..q7); returns how
/// many samples each side may have changed.
fn filter_samples(
    p: &mut [i32; 8],
    q: &mut [i32; 8],
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
    depth: BitDepth,
) -> usize {
    let (p0, q0) = (p[7], q[0]);
    let flatness = 1 << depth.above_eight();
    let flat = width != Vp9FilterWidth::Four
        && p[4..7].iter().all(|&s| (s - p0).abs() <= flatness)
        && q[1..4].iter().all(|&s| (s - q0).abs() <= flatness);
    let flat2 = width == Vp9FilterWidth::Sixteen
        && flat
        && p[..4].iter().all(|&s| (s - p0).abs() <= flatness)
        && q[4..].iter().all(|&s| (s - q0).abs() <= flatness);
    if flat2 {
        filter16(p, q);
        7
    } else if flat {
        filter8(p, q);
        3
    } else {
        filter4(p, q, limits, depth);
        2
    }
}

/// 15-tap smoothing of 7 samples each side (`filter16`).
fn filter16(p: &mut [i32; 8], q: &mut [i32; 8]) {
    // Taps in order p7..p0, q0..q7.
    let taps: [i32; 16] = std::array::from_fn(|i| if i < 8 { p[i] } else { q[i - 8] });
    let mut out = taps;
    for (i, sample) in out.iter_mut().enumerate().take(15).skip(1) {
        // Window of 15 taps centred on i, replicating the outermost taps,
        // with the centre weighted twice: sum of 16 terms.
        let mut sum = taps[i];
        for offset in -7i32..=7 {
            let j = (i as i32 + offset).clamp(0, 15) as usize;
            sum += taps[j];
        }
        *sample = (sum + 8) >> 4;
    }
    p[1..].copy_from_slice(&out[1..8]);
    q[..7].copy_from_slice(&out[8..15]);
}

/// 7-tap smoothing of 3 samples each side (`filter8`).
fn filter8(p: &mut [i32; 8], q: &mut [i32; 8]) {
    let (p3, p2, p1, p0) = (p[4], p[5], p[6], p[7]);
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    p[5] = (3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3;
    p[6] = (2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3;
    p[7] = (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3;
    q[0] = (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3;
    q[1] = (p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3;
    q[2] = (p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3;
}

/// The narrow filter on p1..q1 (`filter4`), in signed-char arithmetic
/// scaled to the depth (`highbd_filter4`: the range `±128 << (BitDepth − 8)`).
fn filter4(p: &mut [i32; 8], q: &mut [i32; 8], limits: Vp9FilterLimits, depth: BitDepth) {
    let half = 128 << depth.above_eight();
    let clamp = |v: i32| v.clamp(-half, half - 1);
    let (p1, p0, q0, q1) = (p[6], p[7], q[0], q[1]);
    let high_variance = (p1 - p0).abs() > limits.variance || (q1 - q0).abs() > limits.variance;
    let outer = if high_variance { clamp(p1 - q1) } else { 0 };
    let filter = clamp(outer + 3 * (q0 - p0));
    let filter1 = clamp(filter + 4) >> 3;
    let filter2 = clamp(filter + 3) >> 3;
    q[0] = depth.clip(q0 - filter1);
    p[7] = depth.clip(p0 + filter2);
    if !high_variance {
        let adjust = (filter1 + 1) >> 1;
        q[1] = depth.clip(q1 - adjust);
        p[6] = depth.clip(p1 + adjust);
    }
}

/// Filter `lines` lines across one edge of `data`: sample `edge` is q0 of
/// the first line, samples across the edge are `step` apart and lines
/// `along` apart (`vpx_lpf_{horizontal,vertical}_{4,8,16}_c`). The outer
/// taps of 16-wide edges are read only for lines that pass the mask.
///
/// # Panics
///
/// If a tap lies outside `data`.
pub fn vp9_filter_edge(
    data: &mut [u8],
    edge: usize,
    step: usize,
    along: usize,
    lines: usize,
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
) {
    vp9_filter_edge_at(
        data,
        (edge, step, along),
        lines,
        width,
        limits,
        BitDepth::EIGHT,
    );
}

/// [`vp9_filter_edge`] on samples of `depth`, with `limits` as at 8 bits
/// (scaled here).
pub(crate) fn vp9_filter_edge_at<T: Sample>(
    data: &mut [T],
    (edge, step, along): (usize, usize, usize),
    lines: usize,
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
    depth: BitDepth,
) {
    let limits = limits.at(depth);
    let outer = if width == Vp9FilterWidth::Sixteen {
        8
    } else {
        4
    };
    for line in 0..lines {
        let q0 = edge + line * along;
        let mut p = [0; 8];
        let mut q = [0; 8];
        for k in 0..4 {
            p[7 - k] = data[q0 - (k + 1) * step].to_i32();
            q[k] = data[q0 + k * step].to_i32();
        }
        if !filter_mask(&p, &q, limits) {
            continue;
        }
        for k in 4..outer {
            p[7 - k] = data[q0 - (k + 1) * step].to_i32();
            q[k] = data[q0 + k * step].to_i32();
        }
        let written = filter_samples(&mut p, &mut q, width, limits, depth);
        for k in 0..written {
            data[q0 - (k + 1) * step] = T::from_i32(p[7 - k]);
            data[q0 + k * step] = T::from_i32(q[k]);
        }
    }
}

/// Lines of one edge filtered together by [`filter_lines_at`].
const LANES: usize = 8;
/// One value per line of a group.
type Lanes = [i32; LANES];

/// [`vp9_filter_edge_at`] written over groups of [`LANES`] lines, every
/// filter computed for every line and the result selected per line, so
/// that an ISA's instance vectorises it across lines (`high_depth::vector`);
/// the result equals the per-line form. It reads the whole footprint
/// (4 taps each side, 8 for 16-wide edges) of every line: the caller checks
/// that it lies inside `data`.
#[allow(clippy::inline_always, reason = "inlined into the ISA instances")]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "lane loops over parallel arrays"
)]
pub(crate) fn filter_lines_at<T: Sample>(
    data: &mut [T],
    (edge, step, along): (usize, usize, usize),
    lines: usize,
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
    depth: BitDepth,
) {
    let limits = limits.at(depth);
    let reach = if width == Vp9FilterWidth::Sixteen {
        8
    } else {
        4
    };
    for first in (0..lines).step_by(LANES) {
        let count = (lines - first).min(LANES);
        // Taps p7..p0 (0..8) then q0..q7 (8..16), one lane per line.
        let mut t: [Lanes; 16] = [[0; LANES]; 16];
        for lane in 0..count {
            let q0 = edge + (first + lane) * along;
            for k in 0..reach {
                t[7 - k][lane] = data[q0 - (k + 1) * step].to_i32();
                t[8 + k][lane] = data[q0 + k * step].to_i32();
            }
        }
        let out = filter_group(&t, width, limits, depth);
        for lane in 0..count {
            let q0 = edge + (first + lane) * along;
            for k in 0..reach - 1 {
                data[q0 - (k + 1) * step] = T::from_i32(out[7 - k][lane]);
                data[q0 + k * step] = T::from_i32(out[8 + k][lane]);
            }
        }
    }
}

/// The filtered taps (p7..p0, q0..q7) of one group of lines: each
/// quantity is a loop over the lanes (no branches, no closures), so that
/// the instance vectorises it; each line takes its own filter's outputs.
#[allow(clippy::inline_always, reason = "inlined into the ISA instances")]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "lane loops over parallel arrays"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one vectorised group: split into a function per filter it ran 23 % more instructions"
)]
fn filter_group(
    t: &[Lanes; 16],
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
    depth: BitDepth,
) -> [Lanes; 16] {
    let flatness = 1 << depth.above_eight();
    let half = 128 << depth.above_eight();
    let max = depth.max();
    let (p3, p2, p1, p0) = (&t[4], &t[5], &t[6], &t[7]);
    let (q0, q1, q2, q3) = (&t[8], &t[9], &t[10], &t[11]);
    let mut mask = [false; LANES];
    let mut flat = [false; LANES];
    let mut flat2 = [false; LANES];
    let mut high_variance = [false; LANES];
    for l in 0..LANES {
        let inside = (p3[l] - p2[l])
            .abs()
            .max((p2[l] - p1[l]).abs())
            .max((p1[l] - p0[l]).abs())
            .max((q1[l] - q0[l]).abs())
            .max((q2[l] - q1[l]).abs())
            .max((q3[l] - q2[l]).abs());
        mask[l] = inside <= limits.inside
            && 2 * (p0[l] - q0[l]).abs() + (p1[l] - q1[l]).abs() / 2 <= limits.boundary;
        let near = (p1[l] - p0[l])
            .abs()
            .max((p2[l] - p0[l]).abs())
            .max((p3[l] - p0[l]).abs())
            .max((q1[l] - q0[l]).abs())
            .max((q2[l] - q0[l]).abs())
            .max((q3[l] - q0[l]).abs());
        let far = (t[0][l] - p0[l])
            .abs()
            .max((t[1][l] - p0[l]).abs())
            .max((t[2][l] - p0[l]).abs())
            .max((t[3][l] - p0[l]).abs())
            .max((t[12][l] - q0[l]).abs())
            .max((t[13][l] - q0[l]).abs())
            .max((t[14][l] - q0[l]).abs())
            .max((t[15][l] - q0[l]).abs());
        flat[l] = width != Vp9FilterWidth::Four && near <= flatness;
        flat2[l] = width == Vp9FilterWidth::Sixteen && flat[l] && far <= flatness;
        high_variance[l] =
            (p1[l] - p0[l]).abs() > limits.variance || (q1[l] - q0[l]).abs() > limits.variance;
    }
    let mut out = *t;
    // filter4 everywhere it applies.
    for l in 0..LANES {
        let clamp = |v: i32| v.clamp(-half, half - 1);
        let outer = if high_variance[l] {
            clamp(p1[l] - q1[l])
        } else {
            0
        };
        let filter = clamp(outer + 3 * (q0[l] - p0[l]));
        let filter1 = clamp(filter + 4) >> 3;
        let filter2 = clamp(filter + 3) >> 3;
        let adjust = if high_variance[l] {
            0
        } else {
            (filter1 + 1) >> 1
        };
        let narrow = mask[l] && !flat[l];
        out[6][l] = if narrow {
            (p1[l] + adjust).clamp(0, max)
        } else {
            out[6][l]
        };
        out[7][l] = if narrow {
            (p0[l] + filter2).clamp(0, max)
        } else {
            out[7][l]
        };
        out[8][l] = if narrow {
            (q0[l] - filter1).clamp(0, max)
        } else {
            out[8][l]
        };
        out[9][l] = if narrow {
            (q1[l] - adjust).clamp(0, max)
        } else {
            out[9][l]
        };
    }
    if width == Vp9FilterWidth::Four {
        return out;
    }
    for l in 0..LANES {
        let eight = mask[l] && flat[l] && !flat2[l];
        let (p3, p2, p1, p0, q0, q1, q2, q3) =
            (p3[l], p2[l], p1[l], p0[l], q0[l], q1[l], q2[l], q3[l]);
        let values = [
            (3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3,
            (2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3,
            (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3,
            (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3,
            (p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3,
            (p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3,
        ];
        for (i, value) in (5..11).zip(values) {
            out[i][l] = if eight { value } else { out[i][l] };
        }
    }
    if !flat2.iter().zip(&mask).any(|(&f, &m)| f && m) {
        return out;
    }
    // filter16: 15 taps around each output, the centre twice, the
    // outermost replicated; a running sum as libvpx's `filter16`.
    for l in 0..LANES {
        let wide = mask[l] && flat2[l];
        let tap = |i: usize| t[i][l];
        let mut sum = 7 * tap(0) + 2 * tap(1) + (2..=8).map(tap).sum::<i32>() + 8;
        let mut values = [0; 16];
        values[1] = sum >> 4;
        for i in 2..15 {
            // Slide the window one tap: the far end gains, the near end
            // (or the replicated p7) loses; the doubled centre moves.
            sum += tap((i + 7).min(15)) - tap(i.saturating_sub(8)) + tap(i) - tap(i - 1);
            values[i] = sum >> 4;
        }
        for i in 1..15 {
            out[i][l] = if wide { values[i] } else { out[i][l] };
        }
    }
    out
}

/// Whether the whole footprint of [`filter_lines_at`] lies inside `len`
/// samples.
pub(crate) fn footprint_fits(
    len: usize,
    (edge, step, along): (usize, usize, usize),
    lines: usize,
    width: Vp9FilterWidth,
) -> bool {
    let reach: usize = if width == Vp9FilterWidth::Sixteen {
        8
    } else {
        4
    };
    let before = reach.checked_mul(step).is_some_and(|before| edge >= before);
    let last = lines
        .checked_sub(1)
        .and_then(|l| l.checked_mul(along))
        .and_then(|x| edge.checked_add(x))
        .and_then(|x| (reach - 1).checked_mul(step).and_then(|y| x.checked_add(y)))
        .is_some_and(|last| last < len);
    before && last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_filter_matches_reference_weights() {
        // libvpx filter16 op6 and oq0 on a ramp.
        let mut p: [i32; 8] = std::array::from_fn(|i| 100 + i as i32);
        let mut q: [i32; 8] = std::array::from_fn(|i| 108 + i as i32);
        let (pp, qq) = (p, q);
        filter16(&mut p, &mut q);
        let op6 =
            (pp[0] * 7 + pp[1] * 2 + pp[2] + pp[3] + pp[4] + pp[5] + pp[6] + pp[7] + qq[0] + 8)
                >> 4;
        let oq0 = (pp[1..].iter().sum::<i32>() + qq[0] * 2 + qq[1..].iter().sum::<i32>() + 8) >> 4;
        assert_eq!((p[1], q[0]), (op6, oq0));
        assert_eq!((p[0], q[7]), (pp[0], qq[7]));
    }
}
