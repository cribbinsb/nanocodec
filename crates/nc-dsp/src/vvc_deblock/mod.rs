//! Scalar H.266 deblocking kernels for 8-bit square DCT-II blocks, and the
//! in-place luma/chroma edges in `inplace` and `chroma_edge`.
//! Equations checked against `VVenC` `LoopFilter` (BSD-3-Clause-Clear;
//! `THIRD_PARTY.md`). Codec control derives strengths and gathers edge samples.

fn clip(value: i32, original: i32, limit: i32) -> u8 {
    value
        .clamp(original - limit, original + limit)
        .clamp(0, 255) as u8
}
fn derivative(side: &[i32]) -> i32 {
    (side[2] - 2 * side[1] + side[0]).abs()
}
fn sides(row: &[u8; 16]) -> ([i32; 8], [i32; 8]) {
    (
        std::array::from_fn(|i| i32::from(row[7 - i])),
        std::array::from_fn(|i| i32::from(row[8 + i])),
    )
}
/// The long filter's weights by distance from the edge, for a side of 3,
/// 5 or 7 samples (§8.8.3.6.7 `fP`/`fQ`).
fn long_weights(length: usize) -> &'static [i32] {
    match length {
        7 => &[59, 50, 41, 32, 23, 14, 5],
        5 => &[58, 45, 32, 19, 6],
        _ => &[53, 32, 11],
    }
}
/// The long filter's clipping multiples of `tc` (halved) by distance from
/// the edge (§8.8.3.6.7 `tCPD`/`tCQD`): a 5-sample side takes the first
/// five of the 7-sample side's.
fn long_limits(length: usize) -> &'static [i32] {
    if length == 3 {
        &[6, 4, 2]
    } else {
        &[6, 5, 4, 3, 2, 1, 1]
    }
}
/// `refMiddle` of §8.8.3.6.7 for P and Q reaches `lengths`.
fn long_middle(p: &[i32; 8], q: &[i32; 8], lengths: [usize; 2]) -> i32 {
    let sum = |side: &[i32; 8], range: std::ops::Range<usize>| side[range].iter().sum::<i32>();
    match (lengths[0].max(lengths[1]), lengths[0].min(lengths[1])) {
        (7, 7) => (2 * (p[0] + q[0]) + sum(p, 1..7) + sum(q, 1..7) + 8) >> 4,
        (5, 5) => (2 * (sum(p, 0..3) + sum(q, 0..3)) + p[3] + q[3] + p[4] + q[4] + 8) >> 4,
        (7, 5) => (2 * (sum(p, 0..2) + sum(q, 0..2)) + sum(p, 2..6) + sum(q, 2..6) + 8) >> 4,
        (5, 3) => (sum(p, 0..4) + sum(q, 0..4) + 4) >> 3,
        _ => {
            let (large, small) = if lengths[0] == 7 { (p, q) } else { (q, p) };
            (2 * (large[0] + small[0])
                + small[0]
                + 2 * (small[1] + small[2])
                + large[1]
                + small[1]
                + sum(large, 2..7)
                + 8)
                >> 4
        }
    }
}
fn long_filter(row: &mut [u8; 16], tc: i32, lengths: [usize; 2]) {
    let (p, q) = sides(row);
    let middle = long_middle(&p, &q, lengths);
    for (which, side) in [p, q].iter().enumerate() {
        let length = lengths[which];
        let far = (side[length - 1] + side[length] + 1) >> 1;
        let (weights, limits) = (long_weights(length), long_limits(length));
        for i in 0..length {
            let value = (middle * weights[i] + far * (64 - weights[i]) + 32) >> 6;
            let index = if which == 0 { 7 - i } else { 8 + i };
            row[index] = clip(value, side[i], (tc * limits[i]) >> 1);
        }
    }
}
fn short_filter(row: &mut [u8; 16], tc: i32, strong: bool, second: [bool; 2]) {
    let (p, q) = sides(row);
    if strong {
        for (which, (near, far)) in [(&p, &q), (&q, &p)].into_iter().enumerate() {
            let values = [
                (near[2] + 2 * near[1] + 2 * near[0] + 2 * far[0] + far[1] + 4) >> 3,
                (near[2] + near[1] + near[0] + far[0] + 2) >> 2,
                (2 * near[3] + 3 * near[2] + near[1] + near[0] + far[0] + 4) >> 3,
            ];
            for i in 0..3 {
                row[if which == 0 { 7 - i } else { 8 + i }] =
                    clip(values[i], near[i], (3 - i as i32) * tc);
            }
        }
    } else {
        let delta = (9 * (q[0] - p[0]) - 3 * (q[1] - p[1]) + 8) >> 4;
        if delta.abs() >= 10 * tc {
            return;
        }
        let delta = delta.clamp(-tc, tc);
        row[7] = (p[0] + delta).clamp(0, 255) as u8;
        row[8] = (q[0] - delta).clamp(0, 255) as u8;
        for (which, side) in [p, q].iter().enumerate() {
            if second[which] {
                let correction = ((((side[2] + side[0] + 1) >> 1) - side[1]
                    + if which == 0 { delta } else { -delta })
                    >> 1)
                    .clamp(-(tc >> 1), tc >> 1);
                row[if which == 0 { 6 } else { 9 }] = (side[1] + correction).clamp(0, 255) as u8;
            }
        }
    }
}

/// Filter four rows across a luma edge. Each row is p7..p0,q0..q7.
/// `lengths` contains the maximum P/Q reach (§8.8.3.3 `maxFilterLengthP/Q`):
/// 3, 5 or 7 samples on each side of a transform block edge (5 beside an
/// affine CU), or the same 1 or 2 on both sides: 1 beside a transform block
/// 4 samples across (only the weak filter of p0 and q0, which reads
/// p2..q2), 1 or 2 on affine sub-block edges near a transform edge.
/// # Panics
/// If the reaches are not such a pair.
pub fn vvc_deblock_luma(edge: &mut [[u8; 16]; 4], beta: i32, tc: i32, lengths: [usize; 2]) {
    match decision(edge, beta, tc, lengths) {
        LumaFilter::None => {}
        LumaFilter::Long => {
            for row in edge {
                long_filter(row, tc, lengths);
            }
        }
        LumaFilter::Short { strong, second } => {
            for row in edge {
                short_filter(row, tc, strong, second);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LumaFilter {
    None,
    Long,
    Short { strong: bool, second: [bool; 2] },
}

fn decision(edge: &[[u8; 16]; 4], beta: i32, tc: i32, lengths: [usize; 2]) -> LumaFilter {
    assert!(
        lengths == [1, 1] || lengths == [2, 2] || lengths.iter().all(|&n| matches!(n, 3 | 5 | 7)),
        "luma deblocking reaches {lengths:?}"
    );
    let terms = [&edge[0], &edge[3]].map(|row| {
        let (p, q) = sides(row);
        LineTerms::new(&p, &q, lengths)
    });
    decide(&terms, beta, tc, lengths)
}

/// What the luma decision (§8.8.3.6.2) needs from one of a segment's two
/// decision lines (lines 0 and 3). Every term fits 16 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LineTerms {
    /// Second derivative at p0..p2 and at q0..q2.
    dp: i32,
    dq: i32,
    /// The same averaged with p3..p5 (q3..q5) on a side longer than 3,
    /// summed over both sides: the long filter's activity.
    long_activity: i32,
    /// `|p0 − q0|`.
    step: i32,
    /// `|p3 − p0| + |q3 − q0|`: flatness for the strong short filter.
    spread: i32,
    /// Flatness out to the reach on sides longer than 3, for the long filter.
    long_spread: i32,
}

impl LineTerms {
    fn new(p: &[i32; 8], q: &[i32; 8], lengths: [usize; 2]) -> Self {
        let long_derivative = |side: &[i32], length: usize| {
            if length > 3 {
                (derivative(side) + derivative(&side[3..]) + 1) >> 1
            } else {
                derivative(side)
            }
        };
        let spread = |side: &[i32], length: usize| {
            let mut value = (side[3] - side[0]).abs();
            if length > 3 {
                if length == 7 {
                    value += (side[4] - side[5] - side[6] + side[7]).abs();
                }
                value = (value + (side[3] - side[length]).abs() + 1) >> 1;
            }
            value
        };
        Self {
            dp: derivative(p),
            dq: derivative(q),
            long_activity: long_derivative(p, lengths[0]) + long_derivative(q, lengths[1]),
            step: (p[0] - q[0]).abs(),
            spread: spread(p, 3) + spread(q, 3),
            long_spread: spread(p, lengths[0]) + spread(q, lengths[1]),
        }
    }

    /// Whether the line is flat enough for the long (`long`) or the strong
    /// short filter, at decision activity `activity`.
    fn smooth(&self, activity: i32, long: bool, beta: i32, tc: i32) -> bool {
        let (spread, limit) = if long {
            (self.long_spread, (beta * 3) >> 5)
        } else {
            (self.spread, beta >> 3)
        };
        activity < beta >> if long { 4 } else { 2 }
            && self.step < (5 * tc + 1) >> 1
            && spread < limit
    }
}

/// A segment's filter from the terms of its lines 0 and 3.
#[allow(
    clippy::inline_always,
    reason = "two scalar calls per SIMD edge spill live sample vectors; keep decisions in the kernel"
)]
#[inline(always)]
fn decide(lines: &[LineTerms; 2], beta: i32, tc: i32, lengths: [usize; 2]) -> LumaFilter {
    if lengths.iter().any(|&n| n > 3)
        && lines.iter().map(|l| l.long_activity).sum::<i32>() < beta
        && lines
            .iter()
            .all(|l| l.smooth(2 * l.long_activity, true, beta, tc))
    {
        return LumaFilter::Long;
    }
    let dp = lines[0].dp + lines[1].dp;
    let dq = lines[0].dq + lines[1].dq;
    if dp + dq >= beta {
        return LumaFilter::None;
    }
    let reach = lengths[0].min(lengths[1]);
    let strong = reach > 2
        && lines
            .iter()
            .all(|l| l.smooth(2 * (l.dp + l.dq), false, beta, tc));
    let threshold = (beta + (beta >> 1)) >> 3;
    LumaFilter::Short {
        strong,
        second: [reach > 1 && dp < threshold, reach > 1 && dq < threshold],
    }
}

/// Filter two rows of a 4:2:0 chroma edge, p3..p0,q0..q3.
/// `large` requires at least 8 chroma samples on both sides. `ctu_boundary`
/// restricts the P-side reads/writes at a horizontal CTU boundary.
pub fn vvc_deblock_chroma(
    edge: &mut [[u8; 8]; 2],
    beta: i32,
    tc: i32,
    large: bool,
    ctu_boundary: bool,
) {
    let endpoint = edge.map(|row| {
        let p = std::array::from_fn::<_, 4, _>(|i| i32::from(row[3 - i]));
        let q = std::array::from_fn::<_, 4, _>(|i| i32::from(row[4 + i]));
        (p, q)
    });
    let strong = large && chroma_strong(edge, beta, tc, ctu_boundary);
    for (row, (p, q)) in edge.iter_mut().zip(endpoint) {
        if !strong {
            let delta = ((4 * (q[0] - p[0]) + p[1] - q[1] + 4) >> 3).clamp(-tc, tc);
            row[3] = (p[0] + delta).clamp(0, 255) as u8;
            row[4] = (q[0] - delta).clamp(0, 255) as u8;
        } else if ctu_boundary {
            let values = [
                (3 * p[1] + 2 * p[0] + q[0] + q[1] + q[2] + 4) >> 3,
                (2 * p[1] + p[0] + 2 * q[0] + q[1] + q[2] + q[3] + 4) >> 3,
                (p[1] + p[0] + q[0] + 2 * q[1] + q[2] + 2 * q[3] + 4) >> 3,
                (p[0] + q[0] + q[1] + 2 * q[2] + 3 * q[3] + 4) >> 3,
            ];
            for i in 0..4 {
                row[3 + i] = clip(values[i], i32::from(row[3 + i]), tc);
            }
        } else {
            for (which, (near, far)) in [(&p, &q), (&q, &p)].into_iter().enumerate() {
                let values = [
                    (near[3] + near[2] + near[1] + 2 * near[0] + far[0] + far[1] + far[2] + 4) >> 3,
                    (2 * near[3] + near[2] + 2 * near[1] + near[0] + far[0] + far[1] + 4) >> 3,
                    (3 * near[3] + 2 * near[2] + near[1] + near[0] + far[0] + 4) >> 3,
                ];
                for i in 0..3 {
                    row[if which == 0 { 3 - i } else { 4 + i }] = clip(values[i], near[i], tc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Validate coordinates before unsigned kernel addressing, including in
    /// release builds where unchecked coordinate addition would wrap.
    #[test]
    fn deblocking_rejects_overflowing_coordinates_without_writes() {
        let dsp = crate::Dsp::new();
        for at in [
            (usize::MAX, 8),
            (8, usize::MAX),
            (usize::MAX - 3, 8),
            (8, usize::MAX - 3),
        ] {
            for vertical in [false, true] {
                for kernel in 0..4 {
                    let mut pixels = [127u8; 32 * 32];
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut view =
                            nc_common::PlaneViewMut::new(&mut pixels, 32, 32, 32).unwrap();
                        match kernel {
                            0 => dsp.hevc_deblock_luma_edge(&mut view, at, vertical, [16; 2], 64),
                            1 => dsp.vvc_deblock_luma_edge(&mut view, at, vertical, 88, 16, [7; 2]),
                            2 => dsp
                                .vvc_deblock_luma_segment(&mut view, at, vertical, 88, 16, [5; 2]),
                            _ => dsp.vvc_deblock_chroma_edge(
                                &mut view, at, vertical, 2, 88, 16, true, false,
                            ),
                        }
                    }));
                    assert!(result.is_err());
                    assert_eq!(pixels, [127; 32 * 32]);
                }
            }
        }
    }

    #[test]
    fn flat_fields_and_zero_threshold_are_unchanged() {
        for value in [0, 64, 128, 255] {
            let mut luma = [[value; 16]; 4];
            let original = luma;
            vvc_deblock_luma(&mut luma, 88, 100, [7, 3]);
            assert_eq!(luma, original);
            let mut chroma = [[value; 8]; 2];
            let original = chroma;
            vvc_deblock_chroma(&mut chroma, 88, 100, true, true);
            assert_eq!(chroma, original);
        }
        let mut edge = std::array::from_fn(|y| std::array::from_fn(|x| (x * 7 + y * 11) as u8));
        let original = edge;
        vvc_deblock_luma(&mut edge, 0, 0, [3, 3]);
        assert_eq!(edge, original);
    }
}

mod chroma_edge;
mod generic;
pub(crate) use chroma_edge::ChromaEdge;
mod inplace;

/// Eight luma lines across one edge, in the plane (two 4-line segments
/// between the same two coded blocks).
#[derive(Debug, Clone, Copy)]
pub(crate) struct VvcLumaEdge {
    /// Index of q0 of the first line.
    pub base: usize,
    /// Number of lines: one or two four-line segments.
    pub lines: usize,
    /// Index step across the edge (p0 to q0).
    pub across: usize,
    /// Index step from one line to the next.
    pub along: usize,
    pub beta: i32,
    pub tc: i32,
    /// Largest P and Q reach: 3, 5 or 7, or matching narrow reaches 1/2.
    pub lengths: [usize; 2],
}
/// Lines of a [`VvcLumaEdge`].
pub(crate) const EDGE_LINES: usize = 8;
/// Samples read on each side of a luma edge (p7..p0).
pub(crate) const LUMA_REACH: usize = 8;

/// Scalar reference of the in-place edge: each segment gathered, filtered
/// by [`vvc_deblock_luma`] and written back.
fn luma_edge(plane: &mut [u8], e: &VvcLumaEdge) {
    for segment in 0..e.lines / 4 {
        let at = |line: usize, pos: usize| {
            (e.base + (4 * segment + line) * e.along + pos * e.across) - LUMA_REACH * e.across
        };
        let mut rows: [[u8; 16]; 4] =
            std::array::from_fn(|line| std::array::from_fn(|pos| plane[at(line, pos)]));
        vvc_deblock_luma(&mut rows, e.beta, e.tc, e.lengths);
        for (line, row) in rows.iter().enumerate() {
            for (pos, &value) in row.iter().enumerate() {
                plane[at(line, pos)] = value;
            }
        }
    }
}

pub(crate) type LumaEdgeFn = unsafe fn(&mut [u8], &VvcLumaEdge);
pub(crate) type LumaFn = unsafe fn(&mut [[u8; 16]; 4], i32, i32, [usize; 2]);
pub(crate) type ChromaEdgeFn = unsafe fn(&mut [u8], &ChromaEdge);
pub(crate) type ChromaFn = unsafe fn(&mut [[u8; 8]; 2], i32, i32, bool, bool);
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    pub luma: LumaFn,
    pub luma_edge: LumaEdgeFn,
    pub luma_segment: LumaEdgeFn,
    pub luma_extended: LumaEdgeFn,
    pub chroma: ChromaFn,
    pub chroma_edge: ChromaEdgeFn,
}
pub(crate) fn kernels(isa: crate::Isa) -> Kernels {
    match isa {
        crate::Isa::Scalar => Kernels {
            luma: vvc_deblock_luma,
            luma_edge,
            luma_segment: luma_edge,
            luma_extended: luma_edge,
            chroma: vvc_deblock_chroma,
            chroma_edge: chroma_edge::scalar,
        },
        #[cfg(target_arch = "x86_64")]
        crate::Isa::Avx2 => avx2::kernels(),
        #[cfg(target_arch = "aarch64")]
        crate::Isa::Neon => neon::kernels(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        crate::Isa::Simd128 => simd128::kernels(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("unsupported ISA"),
    }
}
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2 {
    use crate::simd::avx2::{I16x8, I32x4};
    super::generic::kernels!(#[target_feature(enable = "avx2")]; I32x4, I16x8);
}
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use crate::simd::neon::{I16x8, I32x4};
    super::generic::kernels!(#[target_feature(enable = "neon")]; I32x4, I16x8);
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128 {
    use crate::simd::simd128::{I16x8, I32x4};
    super::generic::kernels!(#[target_feature(enable = "simd128")]; I32x4, I16x8);
}

fn chroma_strong(edge: &[[u8; 8]; 2], beta: i32, tc: i32, boundary: bool) -> bool {
    let activity = edge.map(|row| {
        let p = std::array::from_fn::<_, 4, _>(|i| i32::from(row[3 - i]));
        let q = std::array::from_fn::<_, 4, _>(|i| i32::from(row[4 + i]));
        let d = if boundary {
            (p[1] - p[0]).abs()
        } else {
            derivative(&p)
        } + derivative(&q);
        let spread = (p[if boundary { 1 } else { 3 }] - p[0]).abs() + (q[3] - q[0]).abs();
        (
            d,
            2 * d < beta >> 2 && (p[0] - q[0]).abs() < (5 * tc + 1) >> 1 && spread < beta >> 3,
        )
    });
    activity.iter().map(|&(d, _)| d).sum::<i32>() < beta && activity.iter().all(|&(_, ok)| ok)
}
