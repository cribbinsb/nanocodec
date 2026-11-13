//! H.266 luma deblocking of four or eight lines in place, written once over
//! [`I16x8Shuffles`]: one lane per line across the edge, two 4-line
//! segments per full call. Both segments share `β`, `tC` and filter lengths;
//! each still decides its own filter from its lines 0 and 3 (§8.8.3.6.2:
//! the per-line terms on lanes, then the scalar [`super::decide`]). Every value fits 16 bits: the largest
//! intermediate is a long-filter blend, `64 * 255`.
//!
//! A horizontal edge's lines are contiguous columns: one 8-sample load per
//! position across the edge. A vertical edge's lines are rows: each row's
//! p7..p0 and q0..q7 are loaded as two vectors and transposed (twice 8x8).
#![allow(unsafe_code, clippy::inline_always)]

use super::{EDGE_LINES, LUMA_REACH, LumaFilter, VvcLumaEdge};
use crate::deblock::samples;
use crate::deblock::samples::transpose;
use crate::simd::{I16FilterLanes, I16x8Shuffles};

/// Lines per call, one per lane.
const LANES: usize = 8;
/// Positions across the edge: p7..p0, q0..q7.
const POSITIONS: usize = 16;
/// Lines per segment (each decides its own filter).
const SEGMENT: usize = 4;
/// Samples each side the strong short filter changes (p2..q2), and the
/// normal one (p1..q1).
const STRONG_REACH: usize = 3;
const WEAK_REACH: usize = 2;

/// Positions read on either side: the long filter needs p7..q7;
/// the short filters need only p3..q3.
#[inline(always)]
fn reach(e: &VvcLumaEdge) -> usize {
    if e.lengths.iter().any(|&n| n > 3) {
        LUMA_REACH
    } else {
        SEGMENT
    }
}

/// Reuse HEVC's partial-line access and transpose for each eight-position
/// slab. Four-line calls touch only four lines, including at picture edges.
#[inline(always)]
fn slab(e: &VvcLumaEdge, start: usize) -> samples::Edge {
    samples::Edge {
        base: e.base + start * e.across - samples::REACH * e.across,
        across: e.across,
        along: e.along,
        lines: SEGMENT,
    }
}

#[inline(always)]
fn load<V: I16x8Shuffles>(plane: &[u8], e: &VvcLumaEdge) -> [V; POSITIONS] {
    let mut out = [V::splat(0); POSITIONS];
    for start in (LUMA_REACH - reach(e)..LUMA_REACH + reach(e)).step_by(LANES) {
        out[start..start + LANES].copy_from_slice(&samples::load::<V>(plane, &slab(e, start)));
    }
    out
}

#[inline(always)]
fn store<V: I16x8Shuffles>(plane: &mut [u8], e: &VvcLumaEdge, values: &[V; POSITIONS]) {
    for start in (LUMA_REACH - reach(e)..LUMA_REACH + reach(e)).step_by(LANES) {
        let block = values[start..start + LANES]
            .try_into()
            .expect("eight positions");
        samples::store(plane, &slab(e, start), block, (0, LANES));
    }
}

/// Index of position `k` (p7 = 0 … q7 = 15) of line `line`, from the
/// first line's p7 (`origin`): plain arithmetic, the range was checked once
/// by [`checked_origin`].
#[inline(always)]
fn index(e: &VvcLumaEdge, origin: usize, line: usize, k: usize) -> usize {
    origin + line * e.along + k * e.across
}

/// The first line's p7, after one check that every sample a call touches
/// (p7..q7 of all eight lines, and each 8-sample row load) lies in `plane`.
#[inline(always)]
fn checked_origin(plane: &[u8], e: &VvcLumaEdge) -> usize {
    let origin = e.base - LUMA_REACH * e.across;
    let last = origin + (EDGE_LINES - 1) * e.along + (POSITIONS - 1) * e.across;
    assert!(last < plane.len(), "deblocking sample inside the plane");
    origin
}

/// Positions p7..q7 of the 8 lines, one vector per position (index `k` is
/// position `k − 8`); positions beyond [`reach`] stay zero.
#[inline(always)]
fn load_full<V: I16x8Shuffles>(plane: &[u8], e: &VvcLumaEdge, origin: usize) -> [V; POSITIONS] {
    let mut out = [V::splat(0); POSITIONS];
    let reach = reach(e);
    let centre = POSITIONS / 2;
    if e.along == 1 {
        for k in centre - reach..centre + reach {
            out[k] = V::load_u8(&plane[index(e, origin, 0, k)..]);
        }
    } else {
        // Transposed blocks of 8 positions, from the first position read.
        for start in (centre - reach..centre + reach).step_by(LANES) {
            // Keep loads inside the ISA-enabled caller: array::from_fn
            // outlined this closure on x86 and called the widening intrinsic
            // separately for each row, losing the caller's target features.
            let mut rows = [V::splat(0); LANES];
            for (line, row) in rows.iter_mut().enumerate() {
                *row = V::load_u8(&plane[index(e, origin, line, start)..]);
            }
            let block = transpose(rows);
            out[start..start + LANES].copy_from_slice(&block);
        }
    }
    out
}

/// Store positions p7..q7 (clamped to 8 bits) of the 8 lines, within
/// [`reach`].
#[inline(always)]
fn store_full<V: I16x8Shuffles>(
    plane: &mut [u8],
    e: &VvcLumaEdge,
    origin: usize,
    values: &[V; POSITIONS],
) {
    let reach = reach(e);
    let centre = POSITIONS / 2;
    if e.along == 1 {
        for k in centre - reach..centre + reach {
            values[k].store_u8_sat(&mut plane[index(e, origin, 0, k)..]);
        }
    } else {
        for start in (centre - reach..centre + reach).step_by(LANES) {
            let block = &values[start..start + LANES];
            let rows = transpose::<V>(block.try_into().expect("8 positions"));
            for (line, row) in rows.iter().enumerate() {
                let at = index(e, origin, line, start);
                let mut bytes = [0u8; LANES];
                row.store_u8_sat(&mut bytes);
                plane[at..at + LANES].copy_from_slice(&bytes);
            }
        }
    }
}

/// `|a − 2b + c|` per line.
#[inline(always)]
fn derivative<V: I16FilterLanes>(a: V, b: V, c: V) -> V {
    a.sub(times(b, 2)).add(c).abs()
}

/// `(a + b + 1) >> 1` per line.
#[inline(always)]
fn mean<V: I16FilterLanes>(a: V, b: V) -> V {
    round(a.add(b), 1)
}

/// A side's flatness (`|s3 − s0|`, out to s7 on a side longer than 3).
#[inline(always)]
fn spread<V: I16FilterLanes, const EXTENDED: bool>(side: &[V; 8], length: usize) -> V {
    let near = side[3].sub(side[0]).abs();
    if length <= 3 {
        return near;
    }
    if EXTENDED && length == 5 {
        return mean(near, side[3].sub(side[5]).abs());
    }
    let far = side[4].sub(side[5]).sub(side[6]).add(side[7]).abs();
    mean(near.add(far), side[3].sub(side[7]).abs())
}

/// Each segment's filter: [`super::LineTerms`] of every line on lanes,
/// then the scalar [`super::decide`] on each segment's lines 0 and 3.
fn decide<V: I16x8Shuffles + I16FilterLanes, const LINES: usize, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    e: &VvcLumaEdge,
) -> [LumaFilter; 2] {
    let long = e.lengths.map(|n| n > 3);
    let dp = derivative(p[0], p[1], p[2]);
    let dq = derivative(q[0], q[1], q[2]);
    let long_derivative = |side: &[V; 8], d: V, long: bool| {
        if long {
            mean(d, derivative(side[3], side[4], side[5]))
        } else {
            d
        }
    };
    let terms = [
        dp,
        dq,
        long_derivative(p, dp, long[0]).add(long_derivative(q, dq, long[1])),
        p[0].sub(q[0]).abs(),
        spread::<V, EXTENDED>(p, 3).add(spread::<V, EXTENDED>(q, 3)),
        spread::<V, EXTENDED>(p, e.lengths[0]).add(spread::<V, EXTENDED>(q, e.lengths[1])),
    ]
    .map(|v| {
        let mut lanes = [0i16; LANES];
        v.store(&mut lanes);
        lanes
    });
    let [dp, dq, long_activity, step, spread, long_spread] = terms;
    std::array::from_fn(|segment| {
        if segment * SEGMENT >= LINES {
            return LumaFilter::None;
        }
        let line = |offset: usize| {
            let l = segment * SEGMENT + offset;
            super::LineTerms {
                dp: dp[l].into(),
                dq: dq[l].into(),
                long_activity: long_activity[l].into(),
                step: step[l].into(),
                spread: spread[l].into(),
                long_spread: long_spread[l].into(),
            }
        };
        super::decide(&[line(0), line(SEGMENT - 1)], e.beta, e.tc, e.lengths)
    })
}

/// −1 in the lanes of the segments where `taken`, else 0.
#[inline(always)]
fn mask<V: I16x8Shuffles>(filters: &[LumaFilter; 2], taken: impl Fn(LumaFilter) -> bool) -> V {
    let mut lanes = [0i16; LANES];
    for (segment, &filter) in filters.iter().enumerate() {
        lanes[segment * SEGMENT..(segment + 1) * SEGMENT].fill(-i16::from(taken(filter)));
    }
    V::load(&lanes)
}

/// `round_shift(value, bits)`.
#[inline(always)]
fn round<V: I16FilterLanes>(value: V, bits: u32) -> V {
    value.add(V::splat(1 << (bits - 1))).sra(bits)
}

/// `value` within `original ± limit`.
#[inline(always)]
fn clip<V: I16FilterLanes>(value: V, original: V, limit: i32) -> V {
    let limit = V::splat(limit as i16);
    value.max(original.sub(limit)).min(original.add(limit))
}

#[inline(always)]
fn sum<V: I16FilterLanes>(values: &[V]) -> V {
    values.iter().fold(V::splat(0), |s, &v| s.add(v))
}

#[inline(always)]
fn times<V: I16FilterLanes>(v: V, k: i16) -> V {
    v.mul(V::splat(k))
}

/// # Safety
/// V's ISA is enabled (dispatch contract); every index is bounds checked.
#[inline(always)]
pub(super) unsafe fn luma<
    V: I16x8Shuffles + I16FilterLanes,
    const LINES: usize,
    const EXTENDED: bool,
>(
    plane: &mut [u8],
    e: &VvcLumaEdge,
) {
    let values = if LINES == LANES {
        load_full::<V>(plane, e, checked_origin(plane, e))
    } else {
        load::<V>(plane, e)
    };
    let p: [V; 8] = std::array::from_fn(|i| values[7 - i]);
    let q: [V; 8] = std::array::from_fn(|i| values[8 + i]);
    let filters = decide::<V, LINES, EXTENDED>(&p, &q, e);
    if filters.iter().all(|f| matches!(f, LumaFilter::None)) {
        return;
    }
    let out = if LINES == SEGMENT || filters[0] == filters[1] {
        uniform::<V, EXTENDED>(&p, &q, &values, e, filters[0])
    } else {
        mixed::<V, EXTENDED>(&p, &q, &values, e, &filters)
    };
    if LINES == LANES {
        store_full(plane, e, checked_origin(plane, e), &out);
    } else {
        store(plane, e, &out);
    }
}

/// Every line's filter is `filter`: its output as is, without lane masks
/// (the segments of most edges agree).
#[inline(always)]
fn uniform<V: I16x8Shuffles + I16FilterLanes, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    values: &[V; POSITIONS],
    e: &VvcLumaEdge,
    filter: LumaFilter,
) -> [V; POSITIONS] {
    match filter {
        LumaFilter::None => *values,
        LumaFilter::Long => long::<V, EXTENDED>(p, q, values, e.tc, e.lengths),
        LumaFilter::Short { strong: true, .. } => strong(p, q, values, e.tc),
        LumaFilter::Short {
            strong: false,
            second,
        } => weak(p, q, values, e.tc, second.map(|s| V::splat(-i16::from(s)))),
    }
}

/// The two segments' different filters: each filter's output in the lanes
/// of the segments that take it, at the positions that filter changes.
#[inline(always)]
fn mixed<V: I16x8Shuffles + I16FilterLanes, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    values: &[V; POSITIONS],
    e: &VvcLumaEdge,
    filters: &[LumaFilter; 2],
) -> [V; POSITIONS] {
    let mut out = *values;
    let mut apply = |filtered: [V; POSITIONS], taken: V, reach: usize| {
        let changed = POSITIONS / 2 - reach..POSITIONS / 2 + reach;
        for k in changed {
            out[k] = taken.blend(filtered[k], out[k]);
        }
    };
    if filters.contains(&LumaFilter::Long) {
        apply(
            long::<V, EXTENDED>(p, q, values, e.tc, e.lengths),
            mask(filters, |f| f == LumaFilter::Long),
            LUMA_REACH,
        );
    }
    if filters
        .iter()
        .any(|f| matches!(f, LumaFilter::Short { strong: true, .. }))
    {
        apply(
            strong(p, q, values, e.tc),
            mask(filters, |f| {
                matches!(f, LumaFilter::Short { strong: true, .. })
            }),
            STRONG_REACH,
        );
    }
    if filters
        .iter()
        .any(|f| matches!(f, LumaFilter::Short { strong: false, .. }))
    {
        let side = |which: usize| {
            mask::<V>(
                filters,
                |f| matches!(f, LumaFilter::Short { strong: false, second } if second[which]),
            )
        };
        apply(
            weak(p, q, values, e.tc, [side(0), side(1)]),
            mask(filters, |f| {
                matches!(f, LumaFilter::Short { strong: false, .. })
            }),
            WEAK_REACH,
        );
    }
    out
}

/// §8.8.3.6.7: the long filter of `lengths` (3, 5 or 7 per side).
#[inline(always)]
fn long<V: I16FilterLanes, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    values: &[V; POSITIONS],
    tc: i32,
    lengths: [usize; 2],
) -> [V; POSITIONS] {
    let pair = if EXTENDED {
        (lengths[0].max(lengths[1]), lengths[0].min(lengths[1]))
    } else {
        (lengths[0], lengths[1])
    };
    let middle = match pair {
        (7, 7) => round(
            times(p[0].add(q[0]), 2)
                .add(sum(&p[1..7]))
                .add(sum(&q[1..7])),
            4,
        ),
        (5, 5) if EXTENDED => round(
            times(sum(&p[..3]).add(sum(&q[..3])), 2)
                .add(p[3])
                .add(q[3])
                .add(p[4])
                .add(q[4]),
            4,
        ),
        (7, 5) if EXTENDED => round(
            times(sum(&p[..2]).add(sum(&q[..2])), 2)
                .add(sum(&p[2..6]))
                .add(sum(&q[2..6])),
            4,
        ),
        (5, 3) if EXTENDED => round(sum(&p[..4]).add(sum(&q[..4])), 3),
        _ => {
            let (large, small) = if lengths[0] == 7 { (p, q) } else { (q, p) };
            round(
                times(large[0].add(small[0]), 2)
                    .add(small[0])
                    .add(times(small[1].add(small[2]), 2))
                    .add(large[1])
                    .add(small[1])
                    .add(sum(&large[2..7])),
                4,
            )
        }
    };
    let mut out = *values;
    for (which, side) in [p, q].into_iter().enumerate() {
        let length = lengths[which];
        let far = round(side[length - 1].add(side[length]), 1);
        let (weights, limits): (&[i16], &[i32]) = if length == 7 {
            (&[59, 50, 41, 32, 23, 14, 5], &[6, 5, 4, 3, 2, 1, 1])
        } else if EXTENDED && length == 5 {
            (&[58, 45, 32, 19, 6], &[6, 5, 4, 3, 2])
        } else {
            (&[53, 32, 11], &[6, 4, 2])
        };
        for i in 0..length {
            let value = round(
                times(middle, weights[i]).add(times(far, 64 - weights[i])),
                6,
            );
            out[if which == 0 { 7 - i } else { 8 + i }] =
                clip(value, side[i], (tc * limits[i]) >> 1);
        }
    }
    out
}

/// §8.8.3.6.6, strong: three samples per side.
#[inline(always)]
fn strong<V: I16FilterLanes>(
    p: &[V; 8],
    q: &[V; 8],
    values: &[V; POSITIONS],
    tc: i32,
) -> [V; POSITIONS] {
    let mut out = *values;
    for (which, (near, far)) in [(p, q), (q, p)].into_iter().enumerate() {
        let filtered = [
            round(
                near[2]
                    .add(times(near[1].add(near[0]).add(far[0]), 2))
                    .add(far[1]),
                3,
            ),
            round(near[2].add(near[1]).add(near[0]).add(far[0]), 2),
            round(
                times(near[3], 2)
                    .add(times(near[2], 3))
                    .add(near[1])
                    .add(near[0])
                    .add(far[0]),
                3,
            ),
        ];
        for i in 0..3 {
            out[if which == 0 { 7 - i } else { 8 + i }] =
                clip(filtered[i], near[i], (3 - i as i32) * tc);
        }
    }
    out
}

/// §8.8.3.6.6, normal: p0/q0 per line where `|Δ| < 10 tC`, and p1 / q1
/// in the lanes of `second` (per-segment masks).
#[inline(always)]
fn weak<V: I16FilterLanes>(
    p: &[V; 8],
    q: &[V; 8],
    values: &[V; POSITIONS],
    tc: i32,
    second: [V; 2],
) -> [V; POSITIONS] {
    let mut out = *values;
    let delta = round(times(q[0].sub(p[0]), 9).sub(times(q[1].sub(p[1]), 3)), 4);
    let enabled = V::splat((10 * tc) as i16).gt_mask(delta.abs());
    let delta = delta.max(V::splat(-tc as i16)).min(V::splat(tc as i16));
    out[7] = enabled.blend(clip(p[0].add(delta), p[0], tc), p[0]);
    out[8] = enabled.blend(clip(q[0].sub(delta), q[0], tc), q[0]);
    let half = (tc >> 1) as i16;
    for (which, side) in [p, q].into_iter().enumerate() {
        let signed = if which == 0 {
            delta
        } else {
            V::splat(0).sub(delta)
        };
        let correction = round(side[2].add(side[0]), 1)
            .sub(side[1])
            .add(signed)
            .sra(1)
            .max(V::splat(-half))
            .min(V::splat(half));
        let filtered = clip(side[1].add(correction), side[1], tc >> 1);
        out[if which == 0 { 6 } else { 9 }] = enabled.and(second[which]).blend(filtered, side[1]);
    }
    out
}
