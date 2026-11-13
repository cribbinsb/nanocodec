//! The SIMD 2x2-mean residual of `vvc_tx64`, generic over [`I16Pairs`]:
//! the two rows of each pair are summed (source minus prediction) in
//! 16-bit lanes (|sum| <= 510), a pairwise multiply-add with ones adds the
//! horizontal neighbours into 32-bit lanes, which are rounded, shifted and
//! narrowed back. Integer and exact: the scalar reference's results.
//!
//! The function is `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs): call it from a `#[target_feature]`
//! function of that ISA.

use super::{BLOCK_SIDE, CODED_SIDE, HALVE_SHIFT, row_pair};
use crate::simd::{I16Pairs, I32Lanes};

/// [`super::HalveFn`] on `V` (whose lanes divide 64).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) unsafe fn halve<V: I16Pairs>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
) {
    let ones = V::splat_pair(1, 1);
    for (y, line) in out[..CODED_SIDE * CODED_SIDE]
        .as_chunks_mut::<CODED_SIDE>()
        .0
        .iter_mut()
        .enumerate()
    {
        let (s0, s1) = row_pair(src, src_stride, y);
        let (p0, p1) = row_pair(pred, pred_stride, y);
        let means = |x: usize| {
            V::load_u8(&s0[x..])
                .add(V::load_u8(&s1[x..]))
                .sub(V::load_u8(&p0[x..]))
                .sub(V::load_u8(&p1[x..]))
                .pair_dot(ones)
                .round_shift(HALVE_SHIFT)
        };
        for x in (0..BLOCK_SIDE).step_by(2 * V::LANES) {
            V::narrow_pairs(means(x), means(x + V::LANES)).store(&mut line[x / 2..]);
        }
    }
}
