//! The HEVC planar kernel written once over [`I16Lanes`]; each ISA module
//! instantiates them inside a `#[target_feature]` function. `V::LANES` must
//! divide the block side.

use super::{MAX_SIDE, PlanarEdges};
use crate::simd::I16Lanes;

/// Planar (eq. 8-48) on 16-bit lanes: every term is at most `32 · 255` and
/// their sum below 2^15.
///
/// # Safety
///
/// `V`'s ISA must be available; the edges hold `n` samples, `out` `n * n`,
/// and `V::LANES` divides `n`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) unsafe fn planar<V: I16Lanes>(edges: &PlanarEdges<'_>, n: usize, out: &mut [u8]) {
    let shift = n.trailing_zeros() + 1;
    let side = n as i16;
    // Column weights of the left sample (n − 1 − x) and of top-right (x + 1).
    let mut left_weight = [0i16; MAX_SIDE];
    let mut right_weight = [0i16; MAX_SIDE];
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
