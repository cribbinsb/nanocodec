//! H.266 §8.8.3.6.7, the long luma filter across independent 8-line
//! tiles. Each four-line segment retains its own endpoint decision.
//! The short fallback and independent tile transposes are shared with HEVC.
#![allow(unsafe_code, clippy::inline_always)]

use super::inplace::{derivative, long_blend, long_derivative, long_middle, spread};
use super::{VvcLumaEdge, long_limits, long_weights};
use crate::deblock::batch::{self, Geometry, Tiles};
use crate::simd::I16x8Tiles;

const TAPS: usize = 16;
const SIDE: usize = TAPS / 2;

/// An eight-line edge with its shared thresholds.
#[inline(always)]
pub(super) unsafe fn single<V: I16x8Tiles>(data: &mut [u8], edge: &VvcLumaEdge) {
    let geometry = Geometry {
        base: edge.base,
        across: edge.across,
        along: edge.along,
    };
    // SAFETY: one tile is disjoint by construction, and the caller enables V.
    unsafe {
        tiles::<V>(
            data,
            &[geometry; 4],
            V::splat(edge.beta as i16),
            V::splat(edge.tc as i16),
            edge.lengths,
        );
    }
}

/// Filter independent complete tiles that share their P/Q reaches.
/// # Safety
/// V's ISA is enabled and the first `V::LANES / 8` tiles do not overlap.
#[inline(always)]
pub(crate) unsafe fn tiles<V: I16x8Tiles>(
    data: &mut [u8],
    edges: &[Geometry; 4],
    beta: V,
    tc: V,
    lengths: [usize; 2],
) {
    let batch = Tiles::new::<V, TAPS>(data.len(), edges);
    // SAFETY: the tile bounds were checked above, the caller enables V's ISA.
    let values = unsafe { batch.load::<V, TAPS>(data) };
    if let Some(out) = filter::<V, false>(values, beta, tc, lengths) {
        // SAFETY: bounds were checked above and the caller established disjointness.
        unsafe { batch.store(data, out) };
    }
}

#[inline(always)]
fn max_at_ends<V: I16x8Tiles>(v: V) -> V {
    let (first, last) = batch::endpoints(v);
    first.max(last)
}

#[inline(always)]
pub(super) fn filter<V: I16x8Tiles, const PARTIAL: bool>(
    values: [V; TAPS],
    beta: V,
    tc: V,
    lengths: [usize; 2],
) -> Option<[V; TAPS]> {
    let [
        p7,
        p6,
        p5,
        p4,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        q7,
    ] = values;
    let p = [p0, p1, p2, p3, p4, p5, p6, p7];
    let q = [q0, q1, q2, q3, q4, q5, q6, q7];
    let dp = long_derivative(&p, derivative(p0, p1, p2), lengths[0] > 3);
    let dq = long_derivative(&q, derivative(q0, q1, q2), lengths[1] > 3);
    let activity = max_at_ends(dp.add(dq));

    // If both endpoint activities are below (beta >> 4) / 2,
    // their sum is already below beta.
    let long = beta
        .sra(4)
        .gt_mask(activity.add(activity))
        .and(
            tc.mul(V::splat(5))
                .add(V::splat(1))
                .sra(1)
                .gt_mask(max_at_ends(p0.sub(q0).abs())),
        )
        .and(beta.mul(V::splat(3)).sra(5).gt_mask(max_at_ends(
            spread::<V, true>(&p, lengths[0]).add(spread::<V, true>(&q, lengths[1])),
        )));
    let zero = V::splat(0);
    const { assert!(!PARTIAL || V::LANES == 8) };
    // Every lane of a four-line segment has the same decision. A partial
    // tile disables its second segment with zero tC.
    let all_long = if PARTIAL {
        long.any_nonzero()
    } else {
        !long.eq_mask(zero).any_nonzero()
    };
    let mut out = values;
    if !all_long {
        let short_inputs = [p3, p2, p1, p0, q0, q1, q2, q3];
        if let Some(short) = batch::filter::<V, true>(short_inputs, beta, long.blend(zero, tc)) {
            out[4..12].copy_from_slice(&short);
        } else if PARTIAL || !long.any_nonzero() {
            return None;
        }
    }
    if if PARTIAL {
        all_long
    } else {
        long.any_nonzero()
    } {
        let middle = long_middle::<V, true>(&p, &q, lengths);
        let (p_out, q_out) = out.split_at_mut(SIDE);
        match lengths[0] {
            7 => side::<V, 7>(p_out.iter_mut().rev(), &p, middle, tc, long, all_long),
            5 => side::<V, 5>(p_out.iter_mut().rev(), &p, middle, tc, long, all_long),
            _ => side::<V, 3>(p_out.iter_mut().rev(), &p, middle, tc, long, all_long),
        }
        match lengths[1] {
            7 => side::<V, 7>(q_out.iter_mut(), &q, middle, tc, long, all_long),
            5 => side::<V, 5>(q_out.iter_mut(), &q, middle, tc, long, all_long),
            _ => side::<V, 3>(q_out.iter_mut(), &q, middle, tc, long, all_long),
        }
    }
    Some(out)
}

#[inline(always)]
fn side<'a, V: I16x8Tiles + 'a, const LENGTH: usize>(
    out: impl Iterator<Item = &'a mut V>,
    side: &[V; SIDE],
    middle: V,
    tc: V,
    long: V,
    all_long: bool,
) {
    let near = side[LENGTH - 1];
    let far = near.add(side[LENGTH]).add(V::splat(1)).sra(1);
    let difference = middle.sub(far);
    for (i, dst) in out.take(LENGTH).enumerate() {
        let value = long_blend(far, difference, long_weights(LENGTH)[i]);
        let multiple = long_limits(LENGTH)[i] as i16;
        let limit = tc.mul(V::splat(multiple)).sra(1);
        let original = side[i];
        let clipped = value.max(original.sub(limit)).min(original.add(limit));
        *dst = if all_long {
            clipped
        } else {
            long.blend(clipped, *dst)
        };
    }
}
