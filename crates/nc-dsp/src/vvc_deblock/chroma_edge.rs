//! H.266 §8.8.3.6.5 chroma deblocking in place, up to eight lines.
//! Each adjacent pair independently chooses strong or weak filtering. Plane
//! access shares HEVC's row loads/transposes; every arithmetic intermediate
//! fits i16 (the largest weighted sum is 8 * 255 before rounding).
#![allow(unsafe_code, clippy::inline_always)]

use crate::deblock::samples::{self, Edge, LANES, POSITIONS, REACH};
use crate::simd::{I16FilterLanes, I16x8Shuffles};

const SEGMENT: usize = 2;
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
    let edge = e.samples;
    for segment in (0..edge.lines).step_by(SEGMENT) {
        let at = |line: usize, pos: usize| {
            edge.base + (segment + line) * edge.along + pos * edge.across - REACH * edge.across
        };
        let mut rows: [[u8; POSITIONS]; SEGMENT] =
            std::array::from_fn(|line| std::array::from_fn(|pos| plane[at(line, pos)]));
        super::vvc_deblock_chroma(&mut rows, e.beta, e.tc, e.large, e.boundary);
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

/// One all-ones/zero decision mask repeated across each two-line segment.
#[inline(always)]
fn strong_mask<V: I16x8Shuffles + I16FilterLanes>(p: &[V; 4], q: &[V; 4], e: &ChromaEdge) -> V {
    let derivative = |s: &[V; 4]| s[2].sub(times(s[1], 2)).add(s[0]).abs();
    let activity = (if e.boundary {
        p[1].sub(p[0]).abs()
    } else {
        derivative(p)
    })
    .add(derivative(q));
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

/// # Safety
/// The caller enables V's ISA and validates all p3..q3 samples.
#[inline(always)]
pub(super) unsafe fn simd<V: I16x8Shuffles + I16FilterLanes>(plane: &mut [u8], e: &ChromaEdge) {
    let values = samples::load::<V>(plane, &e.samples);
    let p: [V; 4] = std::array::from_fn(|i| values[3 - i]);
    let q: [V; 4] = std::array::from_fn(|i| values[4 + i]);
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
    samples::store(plane, &e.samples, out, range);
}
