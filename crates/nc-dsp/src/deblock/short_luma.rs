//! The short luma filters HEVC (§8.7.2.5.7) and H.266 (§8.8.3.6.6) share,
//! written once over [`I16FilterLanes`] (one lane per line across the
//! edge): the strong filter of p2..q2 and the normal filter of p1..q1.
//!
//! The two standards compute the same values. The normal filter is equal:
//! H.266's extra `Clip3(−tC, tC, …)` of p0/q0 and `Clip3(−tC/2, tC/2, …)` of
//! p1/q1 are no-ops, since `|Δ| ≤ tC` and the p1/q1 correction is already
//! clipped to `±(tC >> 1)`. The strong filter differs only in its clip:
//! HEVC keeps every output within `±2tC` of its input, H.266 within
//! `±3tC`, `±2tC`, `±tC` (x0, x1, x2), so the caller passes the limit of each
//! position. Every intermediate fits 16 bits (at most `12 * 255` before a
//! shift; `tC` is at most 99 at 8 bits in either standard).

use crate::simd::I16FilterLanes;

/// The strong filter's output of one side: (x0, x1, x2) for x = p (or q,
/// with the sides swapped), each within `limits[i]` of its input.
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
pub(crate) fn strong_side<V: I16FilterLanes>(
    [x3, x2, x1, x0]: [V; 4],
    [y0, y1]: [V; 2],
    limits: [V; 3],
) -> [V; 3] {
    let (two, four) = (V::splat(2), V::splat(4));
    let limit = |v: V, orig: V, l: V| v.max(orig.sub(l)).min(orig.add(l));
    [
        limit(
            x2.add(x1.mul(two))
                .add(x0.mul(two))
                .add(y0.mul(two))
                .add(y1)
                .add(four)
                .sra(3),
            x0,
            limits[0],
        ),
        limit(x2.add(x1).add(x0).add(y0).add(two).sra(2), x1, limits[1]),
        limit(
            x3.mul(two)
                .add(x2.mul(V::splat(3)))
                .add(x1)
                .add(x0)
                .add(y0)
                .add(four)
                .sra(3),
            x2,
            limits[2],
        ),
    ]
}

/// The normal filter's outputs, before the caller's per-segment masks.
#[derive(Clone, Copy)]
pub(crate) struct Normal<V> {
    /// −1 in the lines where `|Δ| < 10 tC` (the filter applies), else 0.
    pub enabled: V,
    pub p1: V,
    pub p0: V,
    pub q0: V,
    pub q1: V,
}

/// `Clip3(−t, t, x)` per lane.
#[allow(clippy::inline_always)]
#[inline(always)]
fn clip<V: I16FilterLanes>(x: V, t: V) -> V {
    x.max(V::splat(0).sub(t)).min(t)
}

/// The normal filter of every line: p0/q0 moved by `Δ`, and the p1/q1
/// candidates (the caller applies them where its side decision allows).
#[allow(clippy::inline_always, clippy::similar_names)]
#[inline(always)]
pub(crate) fn normal<V: I16FilterLanes>(
    [p2, p1, p0]: [V; 3],
    [q0, q1, q2]: [V; 3],
    tc: V,
) -> Normal<V> {
    let raw = q0
        .sub(p0)
        .mul(V::splat(9))
        .sub(q1.sub(p1).mul(V::splat(3)))
        .add(V::splat(8))
        .sra(4);
    let enabled = tc.mul(V::splat(10)).gt_mask(raw.abs());
    let delta = clip(raw, tc);
    let half = tc.sra(1);
    let one = V::splat(1);
    Normal {
        enabled,
        p1: p1.add(clip(
            p2.add(p0).add(one).sra(1).sub(p1).add(delta).sra(1),
            half,
        )),
        p0: p0.add(delta),
        q0: q0.sub(delta),
        q1: q1.add(clip(
            q2.add(q0).add(one).sra(1).sub(q1).sub(delta).sra(1),
            half,
        )),
    }
}
