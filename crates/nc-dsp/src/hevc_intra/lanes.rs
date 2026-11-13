//! The HEVC intra kernels written once over [`I16Lanes`]; each ISA module
//! instantiates them inside a `#[target_feature]` function. `V::LANES` must
//! divide the block side.

use super::{ANGLE_FRACTION_BITS, MAX_SIDE, PlanarEdges};
use crate::simd::I16Lanes;

/// Angular lines (module docs of [`super`]): one 2-tap blend per vector of
/// `V::LANES` samples (weights sum to 32, so every lane stays below 2^13).
///
/// # Safety
///
/// `V`'s ISA must be available; `refs` holds `3n + 1` samples, `out` `n * n`,
/// `angle` is in −32..=32 and `V::LANES` divides `n`.
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
pub(super) unsafe fn angular<V: I16Lanes>(refs: &[u8], n: usize, angle: i32, out: &mut [u8]) {
    let one = 1 << ANGLE_FRACTION_BITS;
    let round = V::splat(one as i16 / 2);
    for (j, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let pos = (j as i32 + 1) * angle;
        let (idx, fact) = (pos >> ANGLE_FRACTION_BITS, pos & (one - 1));
        let start = (n as i32 + idx + 1) as usize;
        if fact == 0 {
            // A whole-sample line is a copy, done in vectors: a call to the
            // generic memcpy per line was 0.6% of an HEVC Fast encode on
            // the Jetson. Bytes are within 0..=255, so the saturating
            // store is exact.
            let source = &refs[start..start + n];
            for k in (0..n).step_by(V::LANES) {
                // SAFETY: k + LANES <= n, inside `source` and `line`.
                unsafe {
                    V::load_u8_at(source.as_ptr().add(k)).store_u8_sat_at(line.as_mut_ptr().add(k));
                }
            }
            continue;
        }
        // A fractional position has idx < n, so ref[idx + n + 1] exists.
        let pair = &refs[start..=start + n];
        let (wa, wb) = ((one - fact) as u8, fact as u8);
        for k in (0..n).step_by(V::LANES) {
            // SAFETY: k + LANES <= n, so both loads lie inside `pair` (n + 1
            // samples) and the store inside `line`.
            unsafe {
                V::mul_add_u8_pair_at(pair.as_ptr().add(k), pair.as_ptr().add(k + 1), wa, wb)
                    .add(round)
                    .sra(ANGLE_FRACTION_BITS)
                    .store_u8_sat_at(line.as_mut_ptr().add(k));
            }
        }
    }
}

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
