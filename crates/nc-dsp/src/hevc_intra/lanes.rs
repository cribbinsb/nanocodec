//! The HEVC planar kernel written once over [`I16Lanes`]; each ISA module
//! instantiates them inside a `#[target_feature]` function. `V::LANES` must
//! divide the block side.

use super::{MAX_SIDE, PlanarEdges};
use crate::simd::I16Lanes;

/// Planar (eq. 8-48) on 16-bit lanes, also used for 64x64 VVC luma.
/// The two axis sums plus rounding are at most `2 * 64 * 255 + 64`,
/// which fits signed 16-bit lanes.
///
/// # Safety
///
/// `V`'s ISA must be available; the edges hold `n` samples, `out` `n * n`,
/// and `V::LANES` divides `n`. The side is at most `CAPACITY` and 64.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn planar<V: I16Lanes, const CAPACITY: usize>(
    edges: &PlanarEdges<'_>,
    n: usize,
    out: &mut [u8],
) {
    let shift = n.trailing_zeros() + 1;
    let side = n as i16;
    // Column weights of the left sample (n − 1 − x) and of top-right (x + 1).
    let mut left_weight = [0i16; CAPACITY];
    let mut right_weight = [0i16; CAPACITY];
    for x in 0..n {
        left_weight[x] = side - 1 - x as i16;
        right_weight[x] = x as i16 + 1;
    }
    let top_right = V::splat(i16::from(edges.top_right));
    let bottom_left = i16::from(edges.bottom_left);
    for (row, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let left = V::splat(i16::from(edges.left[row]));
        let above_weight = V::splat(side - 1 - row as i16);
        let constant = V::splat((row as i16 + 1) * bottom_left + side);
        for k in (0..n).step_by(V::LANES) {
            let value = V::load(&left_weight[k..])
                .mul(left)
                .add(V::load(&right_weight[k..]).mul(top_right))
                .add(V::load_u8(&edges.above[k..]).mul(above_weight))
                .add(constant)
                .sra(shift);
            value.store_u8_sat(&mut line[k..]);
        }
    }
}

/// [`planar`] for `u16` samples at any bit depth `8..=12`, accumulating in
/// 32-bit lanes so `64 * 4095 + 32` never overflows.
///
/// # Safety
///
/// `V`'s ISA must be available; the edges hold `n` samples, `out` `n * n`,
/// and `V::LANES` divides `n`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn planar_u16<V: I16Lanes>(
    edges: &PlanarEdges<'_, u16>,
    n: usize,
    out: &mut [u16],
) {
    let shift = n.trailing_zeros() + 1;
    let side = n as i16;
    let mut left_weight = [0i16; MAX_SIDE];
    let mut right_weight = [0i16; MAX_SIDE];
    for x in 0..n {
        left_weight[x] = side - 1 - x as i16;
        right_weight[x] = x as i16 + 1;
    }
    let top_right = edges.top_right as i16;
    let bottom_left = i32::from(edges.bottom_left);
    for (row, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let left = edges.left[row] as i16;
        let above_weight = side - 1 - row as i16;
        let constant = (row as i32 + 1) * bottom_left + n as i32;
        for k in (0..n).step_by(V::LANES) {
            let mut acc = V::acc_mul_add2(
                V::acc_zero(),
                V::load(&left_weight[k..]),
                left,
                V::load(&right_weight[k..]),
                top_right,
            );
            acc = V::acc_mul_add(acc, V::load_u16(&edges.above[k..]), above_weight);
            V::acc_round_shift_narrow(acc, constant, shift).store_u16(&mut line[k..]);
        }
    }
}
