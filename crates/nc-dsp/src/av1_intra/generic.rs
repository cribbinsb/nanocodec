//! Filter intra patches generic over [`I16Lanes`] with eight lanes: one
//! patch's eight outputs per vector, as the sum over its seven inputs of
//! the input broadcast times that input's weights for every output. Sums
//! stay within 16 bits (`|taps| · 255 ≤ 34 · 255`). And the directional
//! tile transpose generic over the loop filters' 8x8 byte transpose
//! ([`Bytes`]). And Paeth prediction (AV1 §7.11.2.2) generic over
//! [`I16FilterLanes`]: one block row of `V::LANES` outputs per step (4x4:
//! two rows per eight lanes). Smooth needs unsigned 16-bit sums, so each
//! ISA writes its own (`smooth`).
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use crate::intra_prediction::filter::{
    FILTER_TAP_COLUMNS, FilterGrid, FilterIntraMode, PATCH_SAMPLES, SCALE_BITS, TAPS, patch_inputs,
    store_patch,
};
use nc_common::PlaneViewMut;

use crate::intra_prediction::{IntraEdges, directional};
use crate::lpf_bytes::{self, Bytes, LANES};
use crate::simd::{I16FilterLanes, I16Lanes};

/// Side of the blocks whose Paeth rows go two to a vector.
const PAIRED_SIDE: usize = 4;

/// Patch width and height (`filter`'s 4x2 patches).
const PATCH_W: usize = 4;
const PATCH_H: usize = 2;

/// As `filter::predict_patches`, a patch per vector.
///
/// # Safety
///
/// `V`'s ISA must be available; `V::LANES` is [`PATCH_SAMPLES`].
#[inline(always)]
pub(crate) unsafe fn filter_patches<V: I16Lanes>(
    grid: &mut FilterGrid,
    width: usize,
    height: usize,
    mode: FilterIntraMode,
) {
    assert_eq!(V::LANES, PATCH_SAMPLES);
    let weights: [V; TAPS] = FILTER_TAP_COLUMNS[mode.symbol()].map(|column| V::load(&column));
    let round = V::splat(1 << (SCALE_BITS - 1));
    for r in (1..=height).step_by(PATCH_H) {
        for c in (1..=width).step_by(PATCH_W) {
            let inputs = patch_inputs(grid, r, c);
            let mut sum = round;
            for (&input, &weight) in inputs.iter().zip(&weights) {
                sum = sum.add(V::splat(i16::from(input)).mul(weight));
            }
            // Clip1(Round2Signed(sum, 4)) as in the scalar code: the
            // saturating store clips.
            let mut values = [0; PATCH_SAMPLES];
            sum.sra(SCALE_BITS).store_u8_sat(&mut values);
            store_patch(grid, r, c, values);
        }
    }
}

/// `directional::store_transposed` by 8x8 byte tiles in registers (sides
/// below 8 by sample). One bounds check up front: per-tile slicing cost
/// more than the transposes.
///
/// # Safety
///
/// `B`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn store_transposed<B: Bytes>(
    columns: &[u8],
    side: usize,
    dst: &mut PlaneViewMut<'_>,
) {
    if side < LANES {
        directional::store_transposed(columns, side, dst);
        return;
    }
    assert!(side.is_multiple_of(LANES) && dst.width() >= side && dst.height() >= side);
    let stride = dst.stride();
    let out = dst.data_mut();
    // The view holds `(height − 1) · stride + width` bytes (its contract);
    // these checks make every offset below provably in bounds.
    assert!(columns.len() >= side * side && out.len() >= (side - 1) * stride + side);
    let (src, out) = (columns.as_ptr(), out.as_mut_ptr());
    for i0 in (0..side).step_by(LANES) {
        for j0 in (0..side).step_by(LANES) {
            let mut rows = [B::splat(0); LANES];
            for (k, row) in rows.iter_mut().enumerate() {
                // SAFETY: column `j0 + k`, samples `i0..i0 + 8`, inside the
                // checked `side²` bytes; the ISA is the caller's.
                *row = unsafe { B::load(src.add((j0 + k) * side + i0)) };
            }
            for (k, row) in lpf_bytes::transpose(rows).into_iter().enumerate() {
                // SAFETY: row `i0 + k`, samples `j0..j0 + 8`, inside the
                // checked view.
                unsafe { row.store(out.add((i0 + k) * stride + j0)) };
            }
        }
    }
}

/// The Paeth choice per lane: `left` where its distance `to_left` is at
/// most both others, else `top` where `to_top ≤ to_corner`, else `corner`
/// (the scalar `paeth`'s order of ties).
#[inline(always)]
fn paeth_select<V: I16FilterLanes>(
    (left, top, corner): (V, V, V),
    (to_left, to_top, to_corner): (V, V, V),
) -> V {
    let top_or_corner = to_top.gt_mask(to_corner).blend(corner, top);
    to_left
        .gt_mask(to_top.min(to_corner))
        .blend(top_or_corner, left)
}

/// Paeth's distances from `left + top − corner` to each candidate:
/// `|top − corner|`, `|left − corner|`, `|left + top − 2 · corner|`.
#[inline(always)]
fn paeth_distances<V: I16FilterLanes>(left: V, top: V, corner: V) -> (V, V, V) {
    (
        top.sub(corner).abs(),
        left.sub(corner).abs(),
        left.add(top).sub(corner).sub(corner).abs(),
    )
}

/// AV1 Paeth prediction on a side that is a multiple of `V::LANES`.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn paeth<V: I16FilterLanes>(edges: &IntraEdges, dst: &mut PlaneViewMut<'_>) {
    let n = edges.size();
    assert!(n.is_multiple_of(V::LANES) && dst.width() == n && dst.height() == n);
    let (above, left) = (edges.above_row(), edges.left_column());
    let corner = V::splat(i16::from(edges.corner()));
    for (y, &l) in left.iter().enumerate() {
        let l = V::splat(i16::from(l));
        let row = &mut dst.row_mut(y)[..n];
        for x0 in (0..n).step_by(V::LANES) {
            let top = V::load_u8(&above[x0..]);
            let distances = paeth_distances(l, top, corner);
            paeth_select((l, top, corner), distances).store_u8_sat(&mut row[x0..]);
        }
    }
}

/// AV1 Paeth prediction of a 4x4 block, two rows per eight-lane vector.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn paeth_4x4<V: I16FilterLanes>(edges: &IntraEdges, dst: &mut PlaneViewMut<'_>) {
    assert!(
        edges.size() == PAIRED_SIDE && dst.width() == PAIRED_SIDE && dst.height() == PAIRED_SIDE
    );
    let widen =
        |bytes: &[u8]| -> [i16; PAIRED_SIDE] { core::array::from_fn(|i| i16::from(bytes[i])) };
    let above = widen(edges.above_row());
    let top = V::load_halves(&above, &above);
    let corner = V::splat(i16::from(edges.corner()));
    for (pair, left) in edges.left_column().as_chunks::<2>().0.iter().enumerate() {
        let l = V::load_halves(
            &[i16::from(left[0]); PAIRED_SIDE],
            &[i16::from(left[1]); PAIRED_SIDE],
        );
        let distances = paeth_distances(l, top, corner);
        let (mut first, mut second) = ([0; PAIRED_SIDE], [0; PAIRED_SIDE]);
        paeth_select((l, top, corner), distances).store_u8_sat_halves(&mut first, &mut second);
        dst.row_mut(2 * pair)[..PAIRED_SIDE].copy_from_slice(&first);
        dst.row_mut(2 * pair + 1)[..PAIRED_SIDE].copy_from_slice(&second);
    }
}

/// Instantiates one ISA's kernels: filter intra on `$v8`; Paeth on `$v8`
/// (4x4 two rows per vector) and `$wide`; smooth through the ISA module's
/// own `smooth` from 8x8 (4x4 is the scalar code, which measured faster).
macro_rules! av1_intra_kernels {
    ($(#[$attr:meta])*; $v8:ty, $wide:ty) => {
        use $crate::intra_prediction::filter::{FilterGrid, FilterIntraMode};

        $(#[$attr])*
        pub(crate) unsafe fn filter_patches(
            grid: &mut FilterGrid,
            width: usize,
            height: usize,
            mode: FilterIntraMode,
        ) {
            // SAFETY: see the module docs.
            unsafe { super::generic::filter_patches::<$v8>(grid, width, height, mode) }
        }

        $(#[$attr])*
        pub(crate) unsafe fn smooth_paeth(
            edges: &$crate::intra_prediction::IntraEdges,
            mode: $crate::intra_prediction::IntraMode,
            dst: &mut nc_common::PlaneViewMut<'_>,
        ) {
            use $crate::intra_prediction::IntraMode;
            use $crate::simd::I16Lanes;
            let n = edges.size();
            let narrow = n < <$v8 as I16Lanes>::LANES;
            // SAFETY: see the module docs; each lane type divides its sides.
            unsafe {
                match mode {
                    IntraMode::Paeth if narrow => super::generic::paeth_4x4::<$v8>(edges, dst),
                    IntraMode::Paeth if n >= <$wide as I16Lanes>::LANES => {
                        super::generic::paeth::<$wide>(edges, dst);
                    }
                    IntraMode::Paeth => super::generic::paeth::<$v8>(edges, dst),
                    _ if narrow => edges.predict(mode, dst),
                    _ => smooth(edges, mode, dst),
                }
            }
        }
    };
}

pub(crate) use av1_intra_kernels;
