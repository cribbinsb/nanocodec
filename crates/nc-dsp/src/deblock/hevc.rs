//! HEVC deblocking written once over [`I16x8Shuffles`]: one lane per line
//! across the edge, 8 lines per call. Every value fits 16 bits (at most
//! `12 * 255` before a shift).
//!
//! A horizontal edge's lines are contiguous columns: one vector load per
//! position across the edge. A vertical edge's lines are rows: each row's
//! eight samples p3..q3 are loaded as one vector and the eight rows
//! transposed into position vectors (and back for the store).
//!
//! The luma decisions of a 4-line segment read its lines 0 and 3: their
//! inputs are computed in every lane, the two lanes' values are taken out to
//! decide per segment, and the decisions are spread back over the
//! segment's lanes as all-ones masks, applied with `blend` (no multiplies).
//! The strong filter is only computed when a segment chose it.

use super::{HEVC_LINES, HEVC_SEGMENT, HevcEdge};
use crate::simd::{I16FilterLanes, I16x8Shuffles};

/// Lanes per vector, lines per call.
const LANES: usize = HEVC_LINES;
/// Positions loaded across the edge: p3..q3.
const POSITIONS: usize = 8;
/// Segments per call.
#[cfg(not(target_arch = "wasm32"))]
const SEGMENTS: usize = HEVC_LINES / HEVC_SEGMENT;

/// −1 in the lanes where `a < b`, else 0.
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
fn less<V: I16FilterLanes>(a: V, b: V) -> V {
    b.gt_mask(a)
}

/// The lane mask value of a decision: −1 (all bits set) if taken, else 0.
#[cfg(not(target_arch = "wasm32"))]
fn lane_mask(taken: bool) -> i16 {
    -i16::from(taken)
}

/// `Clip3(−t, t, x)` per lane.
#[allow(clippy::inline_always)]
#[inline(always)]
fn clip<V: I16x8Shuffles>(x: V, t: V) -> V {
    x.max(V::splat(0).sub(t)).min(t)
}

pub(crate) use super::samples::transpose;

#[allow(clippy::inline_always)]
#[inline(always)]
fn sample_edge(e: &HevcEdge) -> super::samples::Edge {
    super::samples::Edge {
        base: e.base,
        across: e.across,
        along: e.along,
        lines: e.lines,
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn load<V: I16x8Shuffles>(plane: &[u8], e: &HevcEdge) -> [V; POSITIONS] {
    super::samples::load(plane, &sample_edge(e))
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn store<V: I16x8Shuffles>(
    plane: &mut [u8],
    e: &HevcEdge,
    values: [V; POSITIONS],
    range: (usize, usize),
) {
    super::samples::store(plane, &sample_edge(e), values, range);
}

/// The segments' luma decisions as per-lane masks (−1 taken, 0 not):
/// strong filter, normal filter, normal filter of p1, of q1; and `tC`.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Default)]
struct Decisions {
    /// Some segment takes the strong filter.
    any_strong: bool,
    strong: [i16; LANES],
    normal: [i16; LANES],
    filter_p1: [i16; LANES],
    filter_q1: [i16; LANES],
    tc: [i16; LANES],
}

/// Per-lane inputs of the luma decisions (§8.7.2.5.3).
#[cfg(not(target_arch = "wasm32"))]
struct DecisionInputs {
    /// `|p2 − 2p1 + p0|`.
    dp: [i16; LANES],
    /// `|q2 − 2q1 + q0|`.
    dq: [i16; LANES],
    /// `|p3 − p0| + |q0 − q3|`.
    ends: [i16; LANES],
    /// `|p0 − q0|`.
    step: [i16; LANES],
}

/// Decide every segment; `None` if no segment is filtered.
#[cfg(not(target_arch = "wasm32"))]
fn decide(e: &HevcEdge, inputs: &DecisionInputs) -> Option<Decisions> {
    let beta = i32::from(e.beta);
    let mut out = Decisions::default();
    let mut any = false;
    for segment in 0..SEGMENTS {
        let tc = i32::from(e.tc[segment]);
        if tc == 0 {
            continue;
        }
        let (a, b) = (
            segment * HEVC_SEGMENT,
            segment * HEVC_SEGMENT + HEVC_SEGMENT - 1,
        );
        let value = |v: &[i16; LANES], lane: usize| i32::from(v[lane]);
        let (dp0, dp3) = (value(&inputs.dp, a), value(&inputs.dp, b));
        let (dq0, dq3) = (value(&inputs.dq, a), value(&inputs.dq, b));
        let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
        if dpq0 + dpq3 >= beta {
            continue;
        }
        // Both endpoints must pass the same three strict thresholds. Maxima
        // express those paired tests without a short-circuit branch tree;
        // the segment's tC/curvature early exits above remain unchanged.
        let strong = (2 * dpq0.max(dpq3) < (beta >> 2))
            & (value(&inputs.ends, a).max(value(&inputs.ends, b)) < (beta >> 3))
            & (value(&inputs.step, a).max(value(&inputs.step, b)) < ((5 * tc + 1) >> 1));
        let side_threshold = (beta + (beta >> 1)) >> 3;
        let lanes = a..=b;
        out.any_strong |= strong;
        out.strong[lanes.clone()].fill(lane_mask(strong));
        out.normal[lanes.clone()].fill(lane_mask(!strong));
        out.filter_p1[lanes.clone()].fill(lane_mask(dp0 + dp3 < side_threshold));
        out.filter_q1[lanes.clone()].fill(lane_mask(dq0 + dq3 < side_threshold));
        out.tc[lanes].fill(e.tc[segment]);
        any = true;
    }
    any.then_some(out)
}

/// The strong filter's output of one side (§8.7.2.5.7): (x0, x1, x2) for
/// x = p (or q, with the sides swapped), each clipped to `±2tC` of its input.
#[allow(clippy::inline_always)]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn strong_side<V: I16FilterLanes>([x3, x2, x1, x0]: [V; 4], [y0, y1]: [V; 2], tc2: V) -> [V; 3] {
    let (two, four) = (V::splat(2), V::splat(4));
    let limit = |v: V, orig: V| v.max(orig.sub(tc2)).min(orig.add(tc2));
    [
        limit(
            x2.add(x1.mul(two))
                .add(x0.mul(two))
                .add(y0.mul(two))
                .add(y1)
                .add(four)
                .sra(3),
            x0,
        ),
        limit(x2.add(x1).add(x0).add(y0).add(two).sra(2), x1),
        limit(
            x3.mul(two)
                .add(x2.mul(V::splat(3)))
                .add(x1)
                .add(x0)
                .add(y0)
                .add(four)
                .sra(3),
            x2,
        ),
    ]
}

/// HEVC luma: decisions (§8.7.2.5.3) and filtering (§8.7.2.5.6, .7) of the
/// two segments of an 8-line edge.
#[allow(clippy::inline_always)]
#[inline(always)]
#[allow(clippy::similar_names)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) unsafe fn luma<V: I16x8Shuffles + I16FilterLanes>(plane: &mut [u8], e: &HevcEdge) {
    let samples = load::<V>(plane, e);
    let [p3, p2, p1, p0, q0, q1, q2, q3] = samples;
    let two = V::splat(2);
    let mut inputs = DecisionInputs {
        dp: [0; LANES],
        dq: [0; LANES],
        ends: [0; LANES],
        step: [0; LANES],
    };
    p2.sub(p1.mul(two)).add(p0).abs().store(&mut inputs.dp);
    q2.sub(q1.mul(two)).add(q0).abs().store(&mut inputs.dq);
    p3.sub(p0)
        .abs()
        .add(q0.sub(q3).abs())
        .store(&mut inputs.ends);
    p0.sub(q0).abs().store(&mut inputs.step);
    let Some(d) = decide(e, &inputs) else {
        return;
    };
    let masks = DecisionMasks {
        any_strong: d.any_strong,
        strong: V::load(&d.strong),
        normal: V::load(&d.normal),
        filter_p1: V::load(&d.filter_p1),
        filter_q1: V::load(&d.filter_q1),
        tc: V::load(&d.tc),
    };
    apply(plane, e, samples, masks);
}

/// Decision masks already expanded to the segment's four lanes.
#[derive(Clone, Copy)]
struct DecisionMasks<V> {
    any_strong: bool,
    strong: V,
    normal: V,
    filter_p1: V,
    filter_q1: V,
    tc: V,
}

/// The unchanged filter arithmetic shared by scalar and vector decisions.
#[allow(clippy::inline_always, clippy::similar_names)]
#[inline(always)]
fn apply<V: I16x8Shuffles + I16FilterLanes>(
    plane: &mut [u8],
    e: &HevcEdge,
    samples: [V; POSITIONS],
    d: DecisionMasks<V>,
) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = samples;
    let DecisionMasks {
        any_strong,
        strong,
        normal,
        filter_p1,
        filter_q1,
        tc,
    } = d;
    let raw = q0
        .sub(p0)
        .mul(V::splat(9))
        .sub(q1.sub(p1).mul(V::splat(3)))
        .add(V::splat(8))
        .sra(4);
    let normal = normal.and(less(raw.abs(), tc.mul(V::splat(10))));
    let delta = clip(raw, tc);
    let half = tc.sra(1);
    let np1 = p1.add(clip(
        p2.add(p0).add(V::splat(1)).sra(1).sub(p1).add(delta).sra(1),
        half,
    ));
    let nq1 = q1.add(clip(
        q2.add(q0).add(V::splat(1)).sra(1).sub(q1).sub(delta).sra(1),
        half,
    ));
    let (normal_p1, normal_q1) = (normal.and(filter_p1), normal.and(filter_q1));
    let mut out = [
        p3,
        p2,
        normal_p1.blend(np1, p1),
        normal.blend(p0.add(delta), p0),
        normal.blend(q0.sub(delta), q0),
        normal_q1.blend(nq1, q1),
        q2,
        q3,
    ];
    if any_strong {
        let [sp0, sp1, sp2] = strong_side([p3, p2, p1, p0], [q0, q1], tc.add(tc));
        let [sq0, sq1, sq2] = strong_side([q3, q2, q1, q0], [p0, p1], tc.add(tc));
        for (k, filtered) in [(1, sp2), (2, sp1), (3, sp0), (4, sq0), (5, sq1), (6, sq2)] {
            out[k] = strong.blend(filtered, out[k]);
        }
    }
    // Positions p2..q2 (1..7) change.
    store(plane, e, out, (1, POSITIONS - 1));
}

/// Keep the two four-line decisions in vectors on SIMD128. The existing
/// arithmetic computes every lane already; endpoint shuffles avoid spilling
/// four input arrays and rebuilding five mask/tC arrays through scalar loops.
///
/// # Safety
/// V's ISA is enabled and the caller validates the eight-line HEVC footprint.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::inline_always, clippy::similar_names)]
#[inline(always)]
pub(crate) unsafe fn luma_vector_decisions<V: I16x8Shuffles + I16FilterLanes>(
    plane: &mut [u8],
    e: &HevcEdge,
) {
    let samples = load::<V>(plane, e);
    let [p3, p2, p1, p0, q0, q1, q2, q3] = samples;
    let two = V::splat(2);
    let dp = p2.sub(p1.mul(two)).add(p0).abs();
    let dq = q2.sub(q1.mul(two)).add(q0).abs();
    let ends = p3.sub(p0).abs().add(q0.sub(q3).abs());
    let step = p0.sub(q0).abs();
    let Some(d) = vector_decisions(e, dp, dq, ends, step) else {
        return;
    };
    apply(plane, e, samples, d);
}

/// Replicate each segment's first/last endpoint over its four lanes.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::inline_always)]
#[inline(always)]
fn endpoints<V: I16x8Shuffles>(x: V) -> (V, V) {
    const FIRST: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9, 8, 9, 8, 9];
    const LAST: [u8; 16] = [6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15];
    (x.shuffle_bytes(&FIRST), x.shuffle_bytes(&LAST))
}

/// Only lanes 0 and 4 are needed because each decision fills a whole segment.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::inline_always)]
#[inline(always)]
fn any_segment<V: I16x8Shuffles>(mask: V) -> bool {
    let mut lanes = [0; LANES];
    mask.store(&mut lanes);
    lanes[0] != 0 || lanes[HEVC_SEGMENT] != 0
}

/// H.265 §8.7.2.5.3, the same strict tests as `decide`, without scalar
/// decision arrays. Byte inputs imply dp/dq <= 510, each endpoint's sum <= 1020,
/// and the two-endpoint sum / doubled maximum <= 2040. Public beta/tC are u8:
/// 5*tC + 1 <= 1276, beta + beta/2 <= 382. Every intermediate fits signed16.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::inline_always, clippy::similar_names)]
#[inline(always)]
fn vector_decisions<V: I16x8Shuffles + I16FilterLanes>(
    e: &HevcEdge,
    dp: V,
    dq: V,
    ends: V,
    step: V,
) -> Option<DecisionMasks<V>> {
    let zero = V::splat(0);
    let beta = V::splat(e.beta);
    let tc = V::load(&[
        e.tc[0], e.tc[0], e.tc[0], e.tc[0], e.tc[1], e.tc[1], e.tc[1], e.tc[1],
    ]);
    let (d0, d3) = endpoints(dp.add(dq));
    let active = tc.gt_mask(zero).and(less(d0.add(d3), beta));
    if !any_segment(active) {
        return None;
    }
    let (end0, end3) = endpoints(ends);
    let (step0, step3) = endpoints(step);
    let maximum = d0.max(d3);
    let strong = active
        .and(less(maximum.add(maximum), beta.sra(2)))
        .and(less(end0.max(end3), beta.sra(3)))
        .and(less(
            step0.max(step3),
            tc.mul(V::splat(5)).add(V::splat(1)).sra(1),
        ));
    let threshold = beta.add(beta.sra(1)).sra(3);
    let (dp0, dp3) = endpoints(dp);
    let (dq0, dq3) = endpoints(dq);
    Some(DecisionMasks {
        any_strong: any_segment(strong),
        strong,
        normal: active.and(strong.eq_mask(zero)),
        filter_p1: less(dp0.add(dp3), threshold),
        filter_q1: less(dq0.add(dq3), threshold),
        tc,
    })
}
