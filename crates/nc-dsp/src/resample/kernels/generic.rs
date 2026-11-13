//! Generic SIMD polyphase Lanczos-3 resampling over [`I16Lanes`],
//! [`I16Pairs`] and [`I16x8Shuffles`], bit-identical to the scalar pass.

#![allow(unsafe_code, clippy::inline_always)]

use nc_common::{PlaneView, PlaneViewMut};

use crate::resample::{
    HorizontalPlan, PlaneResampler, VerticalTap, WEIGHT_BITS, WEIGHT_ONE, to_sample,
};
use crate::simd::{I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes};

/// Resample `src` into `dst` using the precomputed `plan` and its internal
/// scratch buffers (zero heap allocations per call).
///
/// # Safety
///
/// The caller must enable the target ISA of `V4`, `V8`, and `VWide`.
#[inline(always)]
pub(super) unsafe fn resample<V4, V8, VWide>(
    plan: &mut PlaneResampler,
    src: &PlaneView<'_>,
    dst: &mut PlaneViewMut<'_>,
) where
    V4: I16Lanes,
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
    VWide: I16Lanes,
{
    let (in_w, in_h) = plan.in_size();
    let (out_w, out_h) = plan.out_size();
    let Some(vert) = &plan.vertical else {
        match &plan.horizontal {
            Some(horiz) => {
                let padded = &mut plan.scratch_padded[..horiz.left_pad + in_w + horiz.right_pad];
                for y in 0..out_h {
                    horizontal_row::<V8>(
                        horiz,
                        &src.row(y)[..in_w],
                        in_w,
                        out_w,
                        padded,
                        &mut dst.row_mut(y)[..out_w],
                    );
                }
            }
            None => {
                for y in 0..out_h {
                    dst.row_mut(y)[..out_w].copy_from_slice(&src.row(y)[..out_w]);
                }
            }
        }
        return;
    };

    let rows = &mut plan.scratch_rows[..out_w * in_h];
    match &plan.horizontal {
        Some(horiz) => {
            let padded = &mut plan.scratch_padded[..horiz.left_pad + in_w + horiz.right_pad];
            for (y, out_row) in rows.chunks_exact_mut(out_w).enumerate() {
                horizontal_row::<V8>(horiz, &src.row(y)[..in_w], in_w, out_w, padded, out_row);
            }
        }
        None => {
            for (y, row) in rows.chunks_exact_mut(out_w).enumerate() {
                row.copy_from_slice(&src.row(y)[..out_w]);
            }
        }
    }
    vertical_pass::<V4, V8, VWide>(vert, rows, out_w, out_h, dst);
}

#[inline(always)]
fn horizontal_row<V8>(
    horiz: &HorizontalPlan,
    src_row: &[u8],
    in_w: usize,
    out_w: usize,
    padded: &mut [u8],
    out_row: &mut [u8],
) where
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
{
    let groups = horiz.groups;
    let w8 = out_w & !7;
    padded[..horiz.left_pad].fill(src_row[0]);
    padded[horiz.left_pad..horiz.left_pad + in_w].copy_from_slice(src_row);
    padded[horiz.left_pad + in_w..].fill(src_row[in_w - 1]);

    let mut x0 = 0usize;
    match groups {
        1 => {
            while x0 < w8 {
                // SAFETY: `x0 + 8 <= out_w`, `horiz.starts.len() == out_w`,
                // `horiz.padded_weights.len() == out_w`, and `start + 8 <= padded.len()`.
                let (lo, hi) = unsafe {
                    (
                        dot4_g1::<V8>(padded, &horiz.starts, &horiz.padded_weights, x0),
                        dot4_g1::<V8>(padded, &horiz.starts, &horiz.padded_weights, x0 + 4),
                    )
                };
                V8::narrow_pairs(lo, hi).store_u8_sat(&mut out_row[x0..]);
                x0 += 8;
            }
            if x0 + 4 <= out_w {
                // SAFETY: `x0 + 4 <= out_w`.
                let lo = unsafe { dot4_g1::<V8>(padded, &horiz.starts, &horiz.padded_weights, x0) };
                let mut tmp = [0u8; 8];
                V8::narrow_pairs(lo, lo).store_u8_sat(&mut tmp);
                out_row[x0..x0 + 4].copy_from_slice(&tmp[..4]);
                x0 += 4;
            }
        }
        2 => {
            while x0 < w8 {
                // SAFETY: `x0 + 8 <= out_w`, `horiz.starts.len() == out_w`,
                // `horiz.padded_weights.len() == 2 * out_w`, and `start + 16 <= padded.len()`.
                let (lo, hi) = unsafe {
                    (
                        dot4_g2::<V8>(padded, &horiz.starts, &horiz.padded_weights, x0),
                        dot4_g2::<V8>(padded, &horiz.starts, &horiz.padded_weights, x0 + 4),
                    )
                };
                V8::narrow_pairs(lo, hi).store_u8_sat(&mut out_row[x0..]);
                x0 += 8;
            }
            if x0 + 4 <= out_w {
                // SAFETY: `x0 + 4 <= out_w`.
                let lo = unsafe { dot4_g2::<V8>(padded, &horiz.starts, &horiz.padded_weights, x0) };
                let mut tmp = [0u8; 8];
                V8::narrow_pairs(lo, lo).store_u8_sat(&mut tmp);
                out_row[x0..x0 + 4].copy_from_slice(&tmp[..4]);
                x0 += 4;
            }
        }
        _ => {
            while x0 < w8 {
                let lo = dot4::<V8>(
                    padded,
                    &horiz.starts[x0..],
                    &horiz.padded_weights[x0 * groups..],
                    groups,
                );
                let hi = dot4::<V8>(
                    padded,
                    &horiz.starts[x0 + 4..],
                    &horiz.padded_weights[(x0 + 4) * groups..],
                    groups,
                );
                V8::narrow_pairs(lo, hi).store_u8_sat(&mut out_row[x0..]);
                x0 += 8;
            }
            if x0 + 4 <= out_w {
                let lo = dot4::<V8>(
                    padded,
                    &horiz.starts[x0..],
                    &horiz.padded_weights[x0 * groups..],
                    groups,
                );
                let mut tmp = [0u8; 8];
                V8::narrow_pairs(lo, lo).store_u8_sat(&mut tmp);
                out_row[x0..x0 + 4].copy_from_slice(&tmp[..4]);
                x0 += 4;
            }
        }
    }
    for (x, out) in out_row.iter_mut().enumerate().skip(x0) {
        let start = horiz.starts[x];
        let chunks = &horiz.padded_weights[x * groups..(x + 1) * groups];
        let samples = &padded[start..start + groups * 8];
        let sum: i32 = chunks
            .iter()
            .flatten()
            .zip(samples)
            .map(|(&w, &s)| i32::from(w) * i32::from(s))
            .sum();
        *out = to_sample(sum);
    }
}

#[inline(always)]
unsafe fn dot4_g1<V8: I16x8Shuffles>(
    padded: &[u8],
    starts: &[usize],
    weights: &[[i16; 8]],
    x0: usize,
) -> V8::Wide {
    // SAFETY: Caller guarantees `x0 + 4 <= out_w`.
    unsafe {
        V8::sum_lanes4([
            col_dot_g1::<V8>(padded, starts, weights, x0),
            col_dot_g1::<V8>(padded, starts, weights, x0 + 1),
            col_dot_g1::<V8>(padded, starts, weights, x0 + 2),
            col_dot_g1::<V8>(padded, starts, weights, x0 + 3),
        ])
        .round_shift(WEIGHT_BITS)
    }
}

#[inline(always)]
unsafe fn col_dot_g1<V8: I16x8Shuffles>(
    padded: &[u8],
    starts: &[usize],
    weights: &[[i16; 8]],
    x: usize,
) -> V8::Wide {
    // SAFETY: `build_horizontal` guarantees `starts[x] + 8 <= padded.len()`.
    unsafe {
        let s = *starts.get_unchecked(x);
        debug_assert!(s + 8 <= padded.len());
        V8::load_u8(padded.get_unchecked(s..)).madd(V8::load(weights.get_unchecked(x)))
    }
}

#[inline(always)]
unsafe fn dot4_g2<V8: I16x8Shuffles>(
    padded: &[u8],
    starts: &[usize],
    weights: &[[i16; 8]],
    x0: usize,
) -> V8::Wide {
    // SAFETY: Caller guarantees `x0 + 4 <= out_w`.
    unsafe {
        V8::sum_lanes4([
            col_dot_g2::<V8>(padded, starts, weights, x0),
            col_dot_g2::<V8>(padded, starts, weights, x0 + 1),
            col_dot_g2::<V8>(padded, starts, weights, x0 + 2),
            col_dot_g2::<V8>(padded, starts, weights, x0 + 3),
        ])
        .round_shift(WEIGHT_BITS)
    }
}

#[inline(always)]
unsafe fn col_dot_g2<V8: I16x8Shuffles>(
    padded: &[u8],
    starts: &[usize],
    weights: &[[i16; 8]],
    x: usize,
) -> V8::Wide {
    // SAFETY: `build_horizontal` guarantees `starts[x] + 16 <= padded.len()`
    // and `2 * x + 1 < weights.len()`.
    unsafe {
        let s = *starts.get_unchecked(x);
        debug_assert!(s + 16 <= padded.len());
        let a0 =
            V8::load_u8(padded.get_unchecked(s..)).madd(V8::load(weights.get_unchecked(2 * x)));
        let a1 = V8::load_u8(padded.get_unchecked(s + 8..))
            .madd(V8::load(weights.get_unchecked(2 * x + 1)));
        a0.add(a1)
    }
}

#[inline(always)]
fn dot4<V8: I16x8Shuffles>(
    padded: &[u8],
    starts: &[usize],
    weights: &[[i16; 8]],
    groups: usize,
) -> V8::Wide {
    let column = |k: usize| (starts[k], &weights[k * groups..(k + 1) * groups]);
    let ((s0, w0), (s1, w1), (s2, w2), (s3, w3)) = (column(0), column(1), column(2), column(3));
    V8::sum_lanes4([
        column_dot::<V8>(padded, s0, w0),
        column_dot::<V8>(padded, s1, w1),
        column_dot::<V8>(padded, s2, w2),
        column_dot::<V8>(padded, s3, w3),
    ])
    .round_shift(WEIGHT_BITS)
}

/// One output column's `8 * groups`-tap dot product from `padded[start..]`.
///
/// A plain `#[inline(always)]` function rather than a closure body: a closure
/// is its own function, does not inherit the caller's `#[target_feature]`,
/// and so compiled every vector intrinsic as an out-of-line call
/// (`dot4::{closure#0}` was 20% of a 480p30 + 720p5 encode's instructions in
/// callgrind). The `column` closure above only slices, with no vector code.
#[inline(always)]
fn column_dot<V8: I16x8Shuffles>(padded: &[u8], start: usize, weights: &[[i16; 8]]) -> V8::Wide {
    let mut acc = V8::load_u8(&padded[start..]).madd(V8::load(&weights[0]));
    for (g, w) in weights.iter().enumerate().skip(1) {
        acc = acc.add(V8::load_u8(&padded[start + 8 * g..]).madd(V8::load(w)));
    }
    acc
}

#[inline(always)]
fn vertical_pass<V4, V8, VWide>(
    vert: &[VerticalTap],
    rows: &[u8],
    out_w: usize,
    out_h: usize,
    dst: &mut PlaneViewMut<'_>,
) where
    V4: I16Lanes,
    V8: I16Lanes,
    VWide: I16Lanes,
{
    let round = WEIGHT_ONE >> 1;
    for (y, tap) in vert.iter().enumerate().take(out_h) {
        let dst_row = &mut dst.row_mut(y)[..out_w];
        let mut x0 = 0usize;
        while x0 + VWide::LANES <= out_w {
            // SAFETY: `x0 + VWide::LANES <= out_w`.
            let acc = unsafe { vertical_acc::<VWide>(rows, x0, &tap.pairs) };
            VWide::acc_round_shift_store_u8(acc, round, WEIGHT_BITS, &mut dst_row[x0..]);
            x0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && x0 + V8::LANES <= out_w {
            // SAFETY: `x0 + V8::LANES <= out_w`.
            let acc = unsafe { vertical_acc::<V8>(rows, x0, &tap.pairs) };
            V8::acc_round_shift_store_u8(acc, round, WEIGHT_BITS, &mut dst_row[x0..]);
            x0 += V8::LANES;
        }
        if x0 + V4::LANES <= out_w {
            // SAFETY: `x0 + V4::LANES <= out_w`.
            let acc = unsafe { vertical_acc::<V4>(rows, x0, &tap.pairs) };
            V4::acc_round_shift_store_u8(acc, round, WEIGHT_BITS, &mut dst_row[x0..]);
            x0 += V4::LANES;
        }
        for (x, out) in dst_row.iter_mut().enumerate().skip(x0) {
            let sum: i32 = tap
                .pairs
                .iter()
                .map(|&(off_a, wa, off_b, wb)| {
                    i32::from(wa) * i32::from(rows[off_a + x])
                        + i32::from(wb) * i32::from(rows[off_b + x])
                })
                .sum();
            *out = to_sample(sum);
        }
    }
}

#[inline(always)]
unsafe fn vertical_acc<V: I16Lanes>(
    rows: &[u8],
    x0: usize,
    pairs: &[(usize, i16, usize, i16)],
) -> V::Acc {
    let mut acc = V::acc_zero();
    for &(off_a, wa, off_b, wb) in pairs {
        // SAFETY: `build_vertical` precomputes `off = row * out_w` with `row < in_h`,
        // `rows.len() == out_w * in_h`, and the caller guarantees `x0 + V::LANES <= out_w`.
        let (a, b) = unsafe {
            debug_assert!(off_a + x0 + V::LANES <= rows.len());
            debug_assert!(off_b + x0 + V::LANES <= rows.len());
            (
                V::load_u8(rows.get_unchecked(off_a + x0..)),
                V::load_u8(rows.get_unchecked(off_b + x0..)),
            )
        };
        acc = V::acc_mul_add2(acc, a, wa, b, wb);
    }
    acc
}
