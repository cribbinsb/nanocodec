//! VP9 deblocking filters of one edge (VP9 §8.8.2; libvpx
//! `vpx_dsp/loopfilter.c` `filter4`/`filter8`/`filter16` and their masks,
//! BSD-3-Clause, see `THIRD_PARTY.md`). Edge scheduling and masks from the
//! mode info stay in nc-vp9.
//!
//! Scalar reference; SIMD versions belong beside it with a bit-exactness
//! test in nc-checkasm.

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
/// `flat_mask4` / `flat_mask5` over taps `p[0..8]` (p7..p0) and `q[0..8]`
/// (q0..q7); returns how many samples each side may have changed.
fn filter_samples(
    p: &mut [i32; 8],
    q: &mut [i32; 8],
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
) -> usize {
    let (p0, q0) = (p[7], q[0]);
    let flat = width != Vp9FilterWidth::Four
        && p[4..7].iter().all(|&s| (s - p0).abs() <= 1)
        && q[1..4].iter().all(|&s| (s - q0).abs() <= 1);
    let flat2 = width == Vp9FilterWidth::Sixteen
        && flat
        && p[..4].iter().all(|&s| (s - p0).abs() <= 1)
        && q[4..].iter().all(|&s| (s - q0).abs() <= 1);
    if flat2 {
        filter16(p, q);
        7
    } else if flat {
        filter8(p, q);
        3
    } else {
        filter4(p, q, limits);
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

/// The narrow filter on p1..q1 (`filter4`), in signed-char arithmetic.
fn filter4(p: &mut [i32; 8], q: &mut [i32; 8], limits: Vp9FilterLimits) {
    let clamp = |v: i32| v.clamp(-128, 127);
    let (p1, p0, q0, q1) = (p[6], p[7], q[0], q[1]);
    let high_variance = (p1 - p0).abs() > limits.variance || (q1 - q0).abs() > limits.variance;
    let outer = if high_variance { clamp(p1 - q1) } else { 0 };
    let filter = clamp(outer + 3 * (q0 - p0));
    let filter1 = clamp(filter + 4) >> 3;
    let filter2 = clamp(filter + 3) >> 3;
    q[0] = (q0 - filter1).clamp(0, 255);
    p[7] = (p0 + filter2).clamp(0, 255);
    if !high_variance {
        let adjust = (filter1 + 1) >> 1;
        q[1] = (q1 - adjust).clamp(0, 255);
        p[6] = (p1 + adjust).clamp(0, 255);
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
            p[7 - k] = i32::from(data[q0 - (k + 1) * step]);
            q[k] = i32::from(data[q0 + k * step]);
        }
        if !filter_mask(&p, &q, limits) {
            continue;
        }
        for k in 4..outer {
            p[7 - k] = i32::from(data[q0 - (k + 1) * step]);
            q[k] = i32::from(data[q0 + k * step]);
        }
        let written = filter_samples(&mut p, &mut q, width, limits);
        for k in 0..written {
            data[q0 - (k + 1) * step] = p[7 - k] as u8;
            data[q0 + k * step] = q[k] as u8;
        }
    }
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
