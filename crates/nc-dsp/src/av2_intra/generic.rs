//! The AV2 smooth kernel generic over [`I16Lanes`]: `V::LANES` outputs of a
//! block row per step. The ramps stay within `±255 · 63` before their
//! rounding shift. The interpolation runs are in `runs.rs` (and NEON's
//! `neon.rs`).
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use crate::intra_prediction::av2::{
    AV2_INTRA_MAX_SIDE, BLEND_SHIFT, SmoothEdges, SmoothKind, smooth_weight,
};
use crate::simd::I16Lanes;

/// `(x + 2^(bits − 1)) >> bits`, arithmetic, per lane.
#[inline(always)]
fn round_shift<V: I16Lanes>(x: V, bits: u32) -> V {
    x.add(V::splat(1 << (bits - 1))).sra(bits)
}

/// A ramp from `end` to `edge` over `distance` (of `side − 1`), blended
/// towards `edge` by `weight` (the scalar `blend` in `smooth`).
#[inline(always)]
fn blend<V: I16Lanes>(edge: V, end: V, distance: V, weight: V, log2: u32) -> V {
    let ramp = end.add(round_shift(edge.sub(end).mul(distance), log2));
    ramp.add(round_shift(edge.sub(ramp).mul(weight), BLEND_SHIFT))
}

/// Smooth prediction of a block whose side is a multiple of `V::LANES`.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn smooth<V: I16Lanes>(edges: &SmoothEdges<'_>, dst: &mut [u8]) {
    let side = edges.side();
    debug_assert!(side.is_multiple_of(V::LANES));
    let log2 = side.trailing_zeros();
    let below = V::splat(i16::from(edges.left[side]));
    let right = V::splat(i16::from(edges.above[side]));
    // Per column: distance from the far corner and blend weight.
    let mut distances = [0; AV2_INTRA_MAX_SIDE];
    let mut weights = [0; AV2_INTRA_MAX_SIDE];
    for (i, (distance, weight)) in distances[..side]
        .iter_mut()
        .zip(&mut weights[..side])
        .enumerate()
    {
        *distance = (side - 1 - i) as i16;
        *weight = smooth_weight(i, log2) as i16;
    }
    for (r, row) in dst.chunks_exact_mut(side).enumerate() {
        let left = V::splat(i16::from(edges.left[r]));
        let row_distance = V::splat(distances[r]);
        let row_weight = V::splat(weights[r]);
        // No closures here: they would not inherit the kernel's target
        // features, leaving every lane operation out of line.
        for c0 in (0..side).step_by(V::LANES) {
            let value = match edges.kind {
                SmoothKind::Vertical => {
                    let top = V::load_u8(&edges.above[c0..]);
                    blend(top, below, row_distance, row_weight, log2)
                }
                SmoothKind::Horizontal => {
                    let (distance, weight) = (V::load(&distances[c0..]), V::load(&weights[c0..]));
                    blend(left, right, distance, weight, log2)
                }
                SmoothKind::Both => {
                    let top = V::load_u8(&edges.above[c0..]);
                    let column = blend(top, below, row_distance, row_weight, log2);
                    let (distance, weight) = (V::load(&distances[c0..]), V::load(&weights[c0..]));
                    let line = blend(left, right, distance, weight, log2);
                    round_shift(column.add(line), 1)
                }
            };
            value.store_u8_sat(&mut row[c0..]);
        }
    }
}

/// Instantiates one ISA's `smooth`: on `$v4` for 4×4 blocks, `$narrow` for
/// 8×8 and `$wide` above.
macro_rules! av2_smooth_kernel {
    ($(#[$attr:meta])*; $v4:ty, $narrow:ty, $wide:ty) => {
        $(#[$attr])*
        unsafe fn smooth(
            edges: &$crate::intra_prediction::av2::SmoothEdges<'_>,
            dst: &mut [u8],
        ) {
            use $crate::simd::I16Lanes;
            let side = edges.side();
            // SAFETY: see the module docs; each lane type divides its sides.
            unsafe {
                if side < <$narrow as I16Lanes>::LANES {
                    $crate::av2_intra::generic::smooth::<$v4>(edges, dst);
                } else if side < <$wide as I16Lanes>::LANES {
                    $crate::av2_intra::generic::smooth::<$narrow>(edges, dst);
                } else {
                    $crate::av2_intra::generic::smooth::<$wide>(edges, dst);
                }
            }
        }
    };
}

pub(crate) use av2_smooth_kernel;
