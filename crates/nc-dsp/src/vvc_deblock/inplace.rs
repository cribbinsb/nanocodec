//! H.266 luma deblocking of four or eight lines in place, written once over
//! [`I16x8Shuffles`]: one lane per line across the edge, two 4-line
//! segments per full call. Both segments share `β`, `tC` and filter lengths;
//! each still decides its own filter from its lines 0 and 3 (§8.8.3.6.2:
//! the per-line terms on lanes, then the scalar [`super::decide`]). Up to 10
//! bits, decisions and filter values fit 16 bits. The long filter uses
//! an exact Q15 weighted difference to keep the blend in 16-bit lanes.
//!
//! Short-only edges keep p3..q3 in the shared HEVC/VVC sample layout,
//! following the separate short/long arithmetic in VVenC/VVdeC `LoopFilterX86`
//! (BSD-3-Clause-Clear; see `THIRD_PARTY.md`).
#![allow(unsafe_code, clippy::inline_always)]

use nc_common::BitDepth;

use super::{EDGE_LINES, LUMA_REACH, LumaFilter, VvcLumaEdge, long_limits, long_weights};
use crate::deblock::samples::transpose;
use crate::deblock::{samples, short_luma};
use crate::simd::{I16FilterLanes, I16Lanes, I16x8Shuffles};

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
fn load<V: I16x8Shuffles>(plane: &[u8], e: &VvcLumaEdge, extent: usize) -> [V; POSITIONS] {
    let mut out = [V::splat(0); POSITIONS];
    for start in (LUMA_REACH - extent..LUMA_REACH + extent).step_by(LANES) {
        out[start..start + LANES].copy_from_slice(&samples::load::<V>(plane, &slab(e, start)));
    }
    out
}

#[inline(always)]
fn store<V: I16x8Shuffles>(
    plane: &mut [u8],
    e: &VvcLumaEdge,
    extent: usize,
    values: &[V; POSITIONS],
) {
    for start in (LUMA_REACH - extent..LUMA_REACH + extent).step_by(LANES) {
        let block = values[start..start + LANES]
            .try_into()
            .expect("eight positions");
        samples::store(plane, &slab(e, start), block, (0, LANES));
    }
}

#[inline(always)]
fn load_u16<V: I16x8Shuffles>(plane: &[u16], e: &VvcLumaEdge) -> [V; POSITIONS] {
    let mut out = [V::splat(0); POSITIONS];
    for start in (LUMA_REACH - reach(e)..LUMA_REACH + reach(e)).step_by(LANES) {
        out[start..start + LANES].copy_from_slice(&samples::load_u16::<V>(plane, &slab(e, start)));
    }
    out
}

#[inline(always)]
fn store_u16<V: I16x8Shuffles>(
    plane: &mut [u16],
    e: &VvcLumaEdge,
    values: &[V; POSITIONS],
    max_val: i16,
) {
    for start in (LUMA_REACH - reach(e)..LUMA_REACH + reach(e)).step_by(LANES) {
        let block = values[start..start + LANES]
            .try_into()
            .expect("eight positions");
        samples::store_u16(plane, &slab(e, start), block, (0, LANES), max_val);
    }
}

#[inline(always)]
fn index(e: &VvcLumaEdge, origin: usize, line: usize, k: usize) -> usize {
    origin + line * e.along + k * e.across
}

#[inline(always)]
fn checked_origin<T>(plane: &[T], e: &VvcLumaEdge) -> usize {
    let origin = e.base - LUMA_REACH * e.across;
    let last = origin + (EDGE_LINES - 1) * e.along + (POSITIONS - 1) * e.across;
    assert!(last < plane.len(), "deblocking sample inside the plane");
    origin
}

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
        for start in (centre - reach..centre + reach).step_by(LANES) {
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

#[inline(always)]
fn load_full_u16<V: I16x8Shuffles>(
    plane: &[u16],
    e: &VvcLumaEdge,
    origin: usize,
) -> [V; POSITIONS] {
    let mut out = [V::splat(0); POSITIONS];
    let reach = reach(e);
    let centre = POSITIONS / 2;
    if e.along == 1 {
        for k in centre - reach..centre + reach {
            out[k] = V::load_u16(&plane[index(e, origin, 0, k)..]);
        }
    } else {
        for start in (centre - reach..centre + reach).step_by(LANES) {
            let mut rows = [V::splat(0); LANES];
            for (line, row) in rows.iter_mut().enumerate() {
                *row = V::load_u16(&plane[index(e, origin, line, start)..]);
            }
            let block = transpose(rows);
            out[start..start + LANES].copy_from_slice(&block);
        }
    }
    out
}

#[inline(always)]
fn store_full_u16<V: I16x8Shuffles>(
    plane: &mut [u16],
    e: &VvcLumaEdge,
    origin: usize,
    values: &[V; POSITIONS],
    max_val: i16,
) {
    let reach = reach(e);
    let centre = POSITIONS / 2;
    let (zero, hi) = (V::splat(0), V::splat(max_val));
    if e.along == 1 {
        for k in centre - reach..centre + reach {
            values[k]
                .max(zero)
                .min(hi)
                .store_u16(&mut plane[index(e, origin, 0, k)..]);
        }
    } else {
        for start in (centre - reach..centre + reach).step_by(LANES) {
            let block = &values[start..start + LANES];
            let rows = transpose::<V>(block.try_into().expect("8 positions"));
            for (line, row) in rows.iter().enumerate() {
                let at = index(e, origin, line, start);
                row.max(zero).min(hi).store_u16(&mut plane[at..at + LANES]);
            }
        }
    }
}

#[inline(always)]
pub(super) fn derivative<V: I16FilterLanes>(a: V, b: V, c: V) -> V {
    a.sub(times(b, 2)).add(c).abs()
}

#[inline(always)]
pub(super) fn mean<V: I16FilterLanes>(a: V, b: V) -> V {
    round(a.add(b), 1)
}

#[inline(always)]
pub(super) fn spread<V: I16FilterLanes, const EXTENDED: bool>(side: &[V; 8], length: usize) -> V {
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

#[inline(always)]
pub(super) fn long_derivative<V: I16FilterLanes>(side: &[V; 8], d: V, is_long: bool) -> V {
    if is_long {
        mean(d, derivative(side[3], side[4], side[5]))
    } else {
        d
    }
}

#[inline(always)]
fn store_lanes<V: I16x8Shuffles>(v: V) -> [i16; LANES] {
    let mut lanes = [0i16; LANES];
    v.store(&mut lanes);
    lanes
}

#[inline(always)]
fn decide<V: I16x8Shuffles + I16FilterLanes, const LINES: usize, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    e: &VvcLumaEdge,
) -> [LumaFilter; 2] {
    let dp_v = derivative(p[0], p[1], p[2]);
    let dq_v = derivative(q[0], q[1], q[2]);
    let dp = store_lanes(dp_v);
    let dq = store_lanes(dq_v);
    let long_activity = store_lanes(
        long_derivative(p, dp_v, e.lengths[0] > 3).add(long_derivative(q, dq_v, e.lengths[1] > 3)),
    );
    let step = store_lanes(p[0].sub(q[0]).abs());
    let near_spread = store_lanes(spread::<V, EXTENDED>(p, 3).add(spread::<V, EXTENDED>(q, 3)));
    let long_spread = store_lanes(
        spread::<V, EXTENDED>(p, e.lengths[0]).add(spread::<V, EXTENDED>(q, e.lengths[1])),
    );
    let mut out = [LumaFilter::None; 2];
    for (segment, dst) in out.iter_mut().enumerate() {
        if segment * SEGMENT >= LINES {
            continue;
        }
        let l0 = segment * SEGMENT;
        let l3 = l0 + SEGMENT - 1;
        *dst = super::decide(
            &[
                super::LineTerms {
                    dp: dp[l0].into(),
                    dq: dq[l0].into(),
                    long_activity: long_activity[l0].into(),
                    step: step[l0].into(),
                    spread: near_spread[l0].into(),
                    long_spread: long_spread[l0].into(),
                },
                super::LineTerms {
                    dp: dp[l3].into(),
                    dq: dq[l3].into(),
                    long_activity: long_activity[l3].into(),
                    step: step[l3].into(),
                    spread: near_spread[l3].into(),
                    long_spread: long_spread[l3].into(),
                },
            ],
            e.beta,
            e.tc,
            e.lengths,
        );
    }
    out
}

#[inline(always)]
fn seg_mask<V: I16x8Shuffles>(seg0: bool, seg1: bool) -> V {
    let (m0, m1) = (-i16::from(seg0), -i16::from(seg1));
    V::load(&[m0, m0, m0, m0, m1, m1, m1, m1])
}

#[inline(always)]
fn round<V: I16FilterLanes>(value: V, bits: u32) -> V {
    value.add(V::splat(1 << (bits - 1))).sra(bits)
}

#[inline(always)]
fn clip<V: I16FilterLanes>(value: V, original: V, limit: i32) -> V {
    let limit = V::splat(limit as i16);
    value.max(original.sub(limit)).min(original.add(limit))
}

#[inline(always)]
fn sum<V: I16FilterLanes>(values: &[V]) -> V {
    let mut s = V::splat(0);
    for &v in values {
        s = s.add(v);
    }
    s
}

#[inline(always)]
fn times<V: I16FilterLanes>(v: V, k: i16) -> V {
    v.mul(V::splat(k))
}

#[inline(always)]
fn blend_range<V: I16FilterLanes, const N: usize>(
    out: &mut [V; N],
    filtered: &[V; N],
    taken: V,
    reach: usize,
) {
    for k in N / 2 - reach..N / 2 + reach {
        out[k] = taken.blend(filtered[k], out[k]);
    }
}

#[inline(always)]
fn compute_luma<V: I16x8Shuffles + I16FilterLanes, const LINES: usize, const EXTENDED: bool>(
    values: &[V; POSITIONS],
    e: &VvcLumaEdge,
) -> Option<[V; POSITIONS]> {
    let p = [
        values[7], values[6], values[5], values[4], values[3], values[2], values[1], values[0],
    ];
    let q = [
        values[8], values[9], values[10], values[11], values[12], values[13], values[14],
        values[15],
    ];
    let filters = decide::<V, LINES, EXTENDED>(&p, &q, e);
    if matches!(filters, [LumaFilter::None, LumaFilter::None]) {
        return None;
    }
    Some(if LINES == SEGMENT || filters[0] == filters[1] {
        uniform::<V, EXTENDED>(&p, &q, values, e, filters[0])
    } else {
        mixed::<V, EXTENDED>(&p, &q, values, e, &filters)
    })
}

#[inline(always)]
fn compute_short_edge<V: I16x8Shuffles + I16FilterLanes>(
    values: [V; samples::POSITIONS],
    e: &VvcLumaEdge,
) -> Option<[V; samples::POSITIONS]> {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = values;
    let (p, q) = ([p0, p1, p2, p3], [q0, q1, q2, q3]);
    let dp = store_lanes(derivative(p0, p1, p2));
    let dq = store_lanes(derivative(q0, q1, q2));
    let step = store_lanes(p0.sub(q0).abs());
    let spread = store_lanes(p3.sub(p0).abs().add(q3.sub(q0).abs()));
    let mut filters = [LumaFilter::None; 2];
    for (segment, filter) in filters.iter_mut().enumerate() {
        let l0 = segment * SEGMENT;
        let l3 = l0 + SEGMENT - 1;
        *filter = super::decide(
            &[
                super::LineTerms {
                    dp: dp[l0].into(),
                    dq: dq[l0].into(),
                    step: step[l0].into(),
                    spread: spread[l0].into(),
                    long_activity: 0,
                    long_spread: 0,
                },
                super::LineTerms {
                    dp: dp[l3].into(),
                    dq: dq[l3].into(),
                    step: step[l3].into(),
                    spread: spread[l3].into(),
                    long_activity: 0,
                    long_spread: 0,
                },
            ],
            e.beta,
            e.tc,
            [STRONG_REACH; 2],
        );
    }
    if matches!(filters, [LumaFilter::None, LumaFilter::None]) {
        return None;
    }
    Some(if filters[0] == filters[1] {
        match filters[0] {
            LumaFilter::Short { strong: true, .. } => strong(&p, &q, &values, e.tc),
            LumaFilter::Short {
                strong: false,
                second,
            } => weak(
                &p,
                &q,
                &values,
                e.tc,
                [
                    V::splat(-i16::from(second[0])),
                    V::splat(-i16::from(second[1])),
                ],
            ),
            _ => values,
        }
    } else {
        short_filters(&p, &q, &values, e, &filters)
    })
}

#[inline(always)]
fn filter_long_tail<V: crate::simd::I16x8Tiles>(
    values: [V; POSITIONS],
    edge: &VvcLumaEdge,
) -> Option<[V; POSITIONS]> {
    let tc = V::splat_segments(&[edge.tc as i16, 0, 0, 0, 0, 0, 0, 0]);
    let beta = V::splat(edge.beta as i16);
    super::long_luma::filter::<V, true>(values, beta, tc, edge.lengths)
}

/// # Safety
/// V's ISA is enabled (dispatch contract); every index is bounds checked.
#[inline(always)]
pub(super) unsafe fn luma<
    V: I16x8Shuffles + crate::simd::I16x8Tiles,
    const LINES: usize,
    const EXTENDED: bool,
>(
    plane: &mut [u8],
    e: &VvcLumaEdge,
) {
    if LINES == LANES && e.lengths == [STRONG_REACH; 2] {
        let edge = samples::Edge {
            base: e.base,
            across: e.across,
            along: e.along,
            lines: LANES,
        };
        if let Some(out) = compute_short_edge::<V>(samples::load::<V>(plane, &edge), e) {
            samples::store(plane, &edge, out, (1, samples::POSITIONS - 1));
        }
        return;
    }
    let extent = reach(e);
    if extent == LUMA_REACH {
        if LINES == LANES {
            // SAFETY: the caller enables V's ISA and the checked tile is one edge.
            unsafe { super::long_luma::single::<V>(plane, e) };
        } else {
            let values = load::<V>(plane, e, LUMA_REACH);
            if let Some(filtered) = filter_long_tail(values, e) {
                store(plane, e, LUMA_REACH, &filtered);
            }
        }
        return;
    }
    let values = if LINES == LANES {
        load_full::<V>(plane, e, checked_origin(plane, e))
    } else {
        load::<V>(plane, e, extent)
    };
    if let Some(out) = compute_luma::<V, LINES, EXTENDED>(&values, e) {
        if LINES == LANES {
            store_full(plane, e, checked_origin(plane, e), &out);
        } else {
            store(plane, e, extent, &out);
        }
    }
}

/// # Safety
/// V's ISA is enabled (dispatch contract); every index is bounds checked.
#[inline(always)]
pub(super) unsafe fn luma_u16<
    V: I16x8Shuffles + I16FilterLanes + crate::simd::I16x8Tiles,
    const LINES: usize,
    const EXTENDED: bool,
>(
    plane: &mut [u16],
    e: &VvcLumaEdge,
    depth: BitDepth,
) {
    if depth.bits() > 10 {
        super::luma_edge_at(plane, e, depth);
        return;
    }
    let max_val = depth.max() as i16;
    if LINES == LANES && e.lengths == [STRONG_REACH; 2] {
        let edge = samples::Edge {
            base: e.base,
            across: e.across,
            along: e.along,
            lines: LANES,
        };
        if let Some(out) = compute_short_edge::<V>(samples::load_u16::<V>(plane, &edge), e) {
            samples::store_u16(plane, &edge, out, (1, samples::POSITIONS - 1), max_val);
        }
        return;
    }
    if LINES == SEGMENT && reach(e) == LUMA_REACH {
        if let Some(out) = filter_long_tail(load_u16::<V>(plane, e), e) {
            store_u16(plane, e, &out, max_val);
        }
        return;
    }
    let values = if LINES == LANES {
        load_full_u16::<V>(plane, e, checked_origin(plane, e))
    } else {
        load_u16::<V>(plane, e)
    };
    if let Some(out) = compute_luma::<V, LINES, EXTENDED>(&values, e) {
        if LINES == LANES {
            store_full_u16(plane, e, checked_origin(plane, e), &out, max_val);
        } else {
            store_u16(plane, e, &out, max_val);
        }
    }
}

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
        } => weak(
            p,
            q,
            values,
            e.tc,
            [
                V::splat(-i16::from(second[0])),
                V::splat(-i16::from(second[1])),
            ],
        ),
    }
}

#[inline(always)]
fn mixed<V: I16x8Shuffles + I16FilterLanes, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    values: &[V; POSITIONS],
    e: &VvcLumaEdge,
    filters: &[LumaFilter; 2],
) -> [V; POSITIONS] {
    let mut out = *values;
    if filters[0] == LumaFilter::Long || filters[1] == LumaFilter::Long {
        let filtered = long::<V, EXTENDED>(p, q, values, e.tc, e.lengths);
        let taken = seg_mask::<V>(
            filters[0] == LumaFilter::Long,
            filters[1] == LumaFilter::Long,
        );
        blend_range(&mut out, &filtered, taken, LUMA_REACH);
    }
    short_filters(p, q, &out, e, filters)
}

#[inline(always)]
fn short_filters<V: I16x8Shuffles + I16FilterLanes, const SIDE: usize, const N: usize>(
    p: &[V; SIDE],
    q: &[V; SIDE],
    values: &[V; N],
    e: &VvcLumaEdge,
    filters: &[LumaFilter; 2],
) -> [V; N] {
    let mut out = *values;
    let s0 = matches!(filters[0], LumaFilter::Short { strong: true, .. });
    let s1 = matches!(filters[1], LumaFilter::Short { strong: true, .. });
    if s0 || s1 {
        let filtered = strong(p, q, values, e.tc);
        blend_range(&mut out, &filtered, seg_mask::<V>(s0, s1), STRONG_REACH);
    }
    let w0 = matches!(filters[0], LumaFilter::Short { strong: false, .. });
    let w1 = matches!(filters[1], LumaFilter::Short { strong: false, .. });
    if w0 || w1 {
        let sec = [
            seg_mask::<V>(
                matches!(filters[0], LumaFilter::Short { strong: false, second } if second[0]),
                matches!(filters[1], LumaFilter::Short { strong: false, second } if second[0]),
            ),
            seg_mask::<V>(
                matches!(filters[0], LumaFilter::Short { strong: false, second } if second[1]),
                matches!(filters[1], LumaFilter::Short { strong: false, second } if second[1]),
            ),
        ];
        let filtered = weak(p, q, values, e.tc, sec);
        blend_range(&mut out, &filtered, seg_mask::<V>(w0, w1), WEAK_REACH);
    }
    out
}

/// §8.8.3.6.7: common reference for both sides of the edge.
#[inline(always)]
pub(super) fn long_middle<V: I16FilterLanes, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    lengths: [usize; 2],
) -> V {
    let pair = if EXTENDED {
        (lengths[0].max(lengths[1]), lengths[0].min(lengths[1]))
    } else {
        (lengths[0], lengths[1])
    };
    match pair {
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
    }
}

/// `(middle * weight + far * (64 - weight) + 32) >> 6`, with
/// `difference = middle - far`. The difference fits in signed i16 at
/// all supported depths, and the largest Q15 weight is 59 * 512.
#[inline(always)]
pub(super) fn long_blend<V: I16Lanes>(far: V, difference: V, weight: i32) -> V {
    far.add(difference.mul_q15_round(V::splat((weight as i16) << 9)))
}

#[inline(always)]
fn long<V: I16FilterLanes, const EXTENDED: bool>(
    p: &[V; 8],
    q: &[V; 8],
    values: &[V; POSITIONS],
    tc: i32,
    lengths: [usize; 2],
) -> [V; POSITIONS] {
    let middle = long_middle::<V, EXTENDED>(p, q, lengths);
    let mut out = *values;
    for (which, side) in [p, q].into_iter().enumerate() {
        let length = lengths[which];
        let far = round(side[length - 1].add(side[length]), 1);
        let difference = middle.sub(far);
        let (weights, limits) = (long_weights(length), long_limits(length));
        for i in 0..length {
            let value = long_blend(far, difference, weights[i]);
            out[if which == 0 { 7 - i } else { 8 + i }] =
                clip(value, side[i], (tc * limits[i]) >> 1);
        }
    }
    out
}

#[inline(always)]
fn strong<V: I16FilterLanes, const SIDE: usize, const N: usize>(
    p: &[V; SIDE],
    q: &[V; SIDE],
    values: &[V; N],
    tc: i32,
) -> [V; N] {
    let mut out = *values;
    let limits = [
        V::splat((3 * tc) as i16),
        V::splat((2 * tc) as i16),
        V::splat(tc as i16),
    ];
    for (which, (near, far)) in [(p, q), (q, p)].into_iter().enumerate() {
        let filtered = short_luma::strong_side(
            [near[3], near[2], near[1], near[0]],
            [far[0], far[1]],
            limits,
        );
        for (i, value) in filtered.into_iter().enumerate() {
            out[if which == 0 { N / 2 - 1 - i } else { N / 2 + i }] = value;
        }
    }
    out
}

#[inline(always)]
fn weak<V: I16FilterLanes, const SIDE: usize, const N: usize>(
    p: &[V; SIDE],
    q: &[V; SIDE],
    values: &[V; N],
    tc: i32,
    second: [V; 2],
) -> [V; N] {
    let mut out = *values;
    let n = short_luma::normal([p[2], p[1], p[0]], [q[0], q[1], q[2]], V::splat(tc as i16));
    out[N / 2 - 2] = n.enabled.and(second[0]).blend(n.p1, p[1]);
    out[N / 2 - 1] = n.enabled.blend(n.p0, p[0]);
    out[N / 2] = n.enabled.blend(n.q0, q[0]);
    out[N / 2 + 1] = n.enabled.and(second[1]).blend(n.q1, q[1]);
    out
}
