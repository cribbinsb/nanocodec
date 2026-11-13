//! The weighted prediction kernel written once over [`I16Lanes`]: the
//! widest vectors (16 lanes on AVX2, as `VVdeC`'s `addAvg` uses),
//! then 8 lanes, then 4, then the scalar reference for the last samples of
//! each row (chroma blocks may be 2 or 6 wide).

use super::HevcWeights;
use super::reference;
use crate::simd::I16Lanes;

/// `V::LANES` samples of a row from `x`.
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
fn chunk<V: I16Lanes>(weights: &HevcWeights, a: &[i16], b: &[i16], dst: &mut [u8], x: usize) {
    let acc = V::acc_mul_add2(
        V::acc_zero(),
        V::load(&a[x..]),
        weights.w0,
        V::load(&b[x..]),
        weights.w1,
    );
    V::acc_round_shift_store_u8(acc, weights.round, weights.shift, &mut dst[x..]);
}

/// One row of `w` samples on vectors `W` (widest), `V` (8 lanes) and `H`
/// (4 lanes).
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
fn row<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    dst: &mut [u8],
) {
    let w = dst.len();
    let mut x = 0;
    while x + W::LANES <= w {
        chunk::<W>(weights, a, b, dst, x);
        x += W::LANES;
    }
    while x + V::LANES <= w {
        chunk::<V>(weights, a, b, dst, x);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        chunk::<H>(weights, a, b, dst, x);
        x += H::LANES;
    }
    reference::row(weights, &a[x..w], &b[x..w], &mut dst[x..]);
}

/// The kernel ([`super::WeightFn`]).
///
/// # Safety
///
/// `V`'s and `H`'s ISA must be available; the slices cover the block.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) unsafe fn weight<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    for y in 0..h {
        row::<W, V, H>(
            weights,
            &a[y * w..(y + 1) * w],
            &b[y * w..(y + 1) * w],
            &mut dst[y * dst_stride..y * dst_stride + w],
        );
    }
}
