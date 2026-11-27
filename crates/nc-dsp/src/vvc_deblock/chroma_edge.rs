//! H.266 §8.8.3.6.5 chroma deblocking in place, up to eight lines.
//! Each adjacent pair independently chooses strong or weak filtering. Plane
//! access shares HEVC's row loads/transposes; every arithmetic intermediate
//! fits i16 up to 10 bits (the largest weighted sum is `8 * 1023 + 4 = 8188`).
#![allow(unsafe_code, clippy::inline_always)]

use nc_common::{BitDepth, Sample};

use crate::deblock::samples::{self, Edge, LANES, POSITIONS, REACH};
use crate::simd::{I16FilterLanes, I16x8Shuffles};

/// Lines of a segment where chroma is subsampled along the edge.
pub(crate) const SEGMENT: usize = 2;
/// Exchange the two i16 lanes of each decision segment.
const SWAP_PAIR: [u8; 16] = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13];

#[derive(Debug, Clone, Copy)]
pub(crate) struct ChromaEdge {
    pub samples: Edge,
    pub beta: i32,
    pub tc: i32,
    pub large: bool,
    pub boundary: bool,
}

/// Scalar reference: original two-line kernel, independently per segment.
pub(super) fn scalar(plane: &mut [u8], e: &ChromaEdge) {
    scalar_at(plane, e, BitDepth::EIGHT);
}

/// [`scalar`] at any sample type and depth (`e.beta`, `e.tc` scaled to it).
pub(crate) fn scalar_at<T: Sample>(plane: &mut [T], e: &ChromaEdge, depth: BitDepth) {
    segments_at::<T, SEGMENT>(plane, e, depth);
}

/// The edge in four-line segments, each decided on its lines 0 and 3
/// (chroma not subsampled along the edge; scalar only).
pub(crate) fn quads_at<T: Sample>(plane: &mut [T], e: &ChromaEdge, depth: BitDepth) {
    segments_at::<T, QUAD>(plane, e, depth);
}

/// Lines of a segment where chroma is not subsampled along the edge.
pub(crate) const QUAD: usize = 4;

/// The edge in `N`-line segments, each decided on its first and last line.
fn segments_at<T: Sample, const N: usize>(plane: &mut [T], e: &ChromaEdge, depth: BitDepth) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let edge = e.samples;
    for segment in (0..edge.lines).step_by(N) {
        let at = |line: usize, pos: usize| {
            edge.base + (segment + line) * edge.along + pos * edge.across - REACH * edge.across
        };
        let mut rows: [[T; POSITIONS]; N] =
            std::array::from_fn(|line| std::array::from_fn(|pos| plane[at(line, pos)]));
        super::chroma_at(&mut rows, e.beta, e.tc, e.large, e.boundary, depth);
        for (line, row) in rows.iter().enumerate() {
            for (pos, &value) in row.iter().enumerate() {
                plane[at(line, pos)] = value;
            }
        }
    }
}

#[inline(always)]
fn times<V: I16FilterLanes>(x: V, k: i16) -> V {
    x.mul(V::splat(k))
}

#[inline(always)]
fn round<V: I16FilterLanes>(x: V) -> V {
    x.add(V::splat(4)).sra(3)
}

#[inline(always)]
fn clip<V: I16FilterLanes>(value: V, original: V, tc: V) -> V {
    value.max(original.sub(tc)).min(original.add(tc))
}

#[inline(always)]
fn side_derivative<V: I16FilterLanes>(s: &[V; 4]) -> V {
    s[2].sub(times(s[1], 2)).add(s[0]).abs()
}

/// One all-ones/zero decision mask repeated across each two-line segment.
#[inline(always)]
fn strong_mask<V: I16x8Shuffles + I16FilterLanes>(p: &[V; 4], q: &[V; 4], e: &ChromaEdge) -> V {
    let activity = (if e.boundary {
        p[1].sub(p[0]).abs()
    } else {
        side_derivative(p)
    })
    .add(side_derivative(q));
    let spread = p[if e.boundary { 1 } else { 3 }]
        .sub(p[0])
        .abs()
        .add(q[3].sub(q[0]).abs());
    let smooth = V::splat((e.beta >> 2) as i16)
        .gt_mask(times(activity, 2))
        .and(V::splat(((5 * e.tc + 1) >> 1) as i16).gt_mask(p[0].sub(q[0]).abs()))
        .and(V::splat((e.beta >> 3) as i16).gt_mask(spread));
    smooth
        .and(smooth.shuffle_bytes(&SWAP_PAIR))
        .and(V::splat(e.beta as i16).gt_mask(activity.add(activity.shuffle_bytes(&SWAP_PAIR))))
}

#[inline(always)]
fn strong<V: I16FilterLanes>(
    p: &[V; 4],
    q: &[V; 4],
    values: &[V; POSITIONS],
    boundary: bool,
    tc: V,
) -> [V; POSITIONS] {
    let mut out = *values;
    if boundary {
        let filtered = [
            times(p[1], 3)
                .add(times(p[0], 2))
                .add(q[0])
                .add(q[1])
                .add(q[2]),
            times(p[1], 2)
                .add(p[0])
                .add(times(q[0], 2))
                .add(q[1])
                .add(q[2])
                .add(q[3]),
            p[1].add(p[0])
                .add(q[0])
                .add(times(q[1], 2))
                .add(q[2])
                .add(times(q[3], 2)),
            p[0].add(q[0])
                .add(q[1])
                .add(times(q[2], 2))
                .add(times(q[3], 3)),
        ];
        for (i, value) in filtered.into_iter().enumerate() {
            out[3 + i] = clip(round(value), values[3 + i], tc);
        }
    } else {
        for (side, (near, far)) in [(p, q), (q, p)].into_iter().enumerate() {
            let filtered = [
                near[3]
                    .add(near[2])
                    .add(near[1])
                    .add(times(near[0], 2))
                    .add(far[0])
                    .add(far[1])
                    .add(far[2]),
                times(near[3], 2)
                    .add(near[2])
                    .add(times(near[1], 2))
                    .add(near[0])
                    .add(far[0])
                    .add(far[1]),
                times(near[3], 3)
                    .add(times(near[2], 2))
                    .add(near[1])
                    .add(near[0])
                    .add(far[0]),
            ];
            for (i, value) in filtered.into_iter().enumerate() {
                out[if side == 0 { 3 - i } else { 4 + i }] = clip(round(value), near[i], tc);
            }
        }
    }
    out
}

#[inline(always)]
fn compute_chroma<V: I16x8Shuffles + I16FilterLanes>(
    values: [V; POSITIONS],
    e: &ChromaEdge,
) -> ([V; POSITIONS], (usize, usize)) {
    let p = [values[3], values[2], values[1], values[0]];
    let q = [values[4], values[5], values[6], values[7]];
    let tc = V::splat(e.tc as i16);
    let delta = round(times(q[0].sub(p[0]), 4).add(p[1]).sub(q[1]))
        .max(V::splat(0).sub(tc))
        .min(tc);
    let mut out = values;
    out[3] = p[0].add(delta);
    out[4] = q[0].sub(delta);
    let mut range = (3, 5);
    if e.large {
        let mask = strong_mask(&p, &q, e);
        let mut lanes = [0; LANES];
        mask.store(&mut lanes);
        if lanes[..e.samples.lines].iter().any(|&lane| lane != 0) {
            let filtered = strong(&p, &q, &values, e.boundary, tc);
            for (value, filtered) in out.iter_mut().zip(filtered) {
                *value = mask.blend(filtered, *value);
            }
            range = (if e.boundary { 3 } else { 1 }, 7);
        }
    }
    (out, range)
}

/// # Safety
/// The caller enables V's ISA and validates all p3..q3 samples.
#[inline(always)]
pub(super) unsafe fn simd<V: I16x8Shuffles + I16FilterLanes>(plane: &mut [u8], e: &ChromaEdge) {
    let values = samples::load::<V>(plane, &e.samples);
    let (out, range) = compute_chroma(values, e);
    samples::store(plane, &e.samples, out, range);
}

/// # Safety
/// The caller enables V's ISA and validates all p3..q3 samples.
#[inline(always)]
pub(super) unsafe fn simd_u16<V: I16x8Shuffles + I16FilterLanes>(
    plane: &mut [u16],
    e: &ChromaEdge,
    depth: BitDepth,
) {
    if depth.bits() > 10 {
        scalar_at(plane, e, depth);
        return;
    }
    let values = samples::load_u16::<V>(plane, &e.samples);
    let (out, range) = compute_chroma(values, e);
    samples::store_u16(plane, &e.samples, out, range, depth.max() as i16);
}
