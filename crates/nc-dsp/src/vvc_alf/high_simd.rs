//! High-bit-depth (`u16`) VVC ALF and CC-ALF row kernels over [`I16FilterLanes`].
//!
//! At up to [`super::VVC_ALF_MAX_BIT_DEPTH`] (10 bits), samples lie in
//! `0..=1023`, symmetric tap differences (`a + b - 2c` or clipped sums) lie in
//! `-2046..=2046`, and CC-ALF differences (`s - c`) lie in `-1023..=1023`.
//! Every tap difference therefore fits signed 16-bit lanes across all four
//! SIMD tiers (`I16x16` AVX2, `I16x32` AVX-512, `I16x8` NEON and SIMD128),
//! accumulating exact 32-bit products before rounding and clamping to `depth`.
//!
//! Arithmetic follows BSD-3-Clause-Clear `VVenC` / `VVdeC`
//! `AdaptiveLoopFilter` (`THIRD_PARTY.md`).

use nc_common::BitDepth;

use crate::simd::I16FilterLanes;

use super::clipped::{
    CHROMA_TAP_OFFSETS, CHROMA_TAPS, assert_coeffs_and_limits, chroma_row_at, clipped_row_at,
};
use super::cross::cross_row_at;
use super::{VVC_ALF_BLOCK, VVC_ALF_REACH, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, filter_row_at};

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn clipped_diff_ptr<V: I16FilterLanes>(
    ptrs: &[*const u16; 7],
    x: usize,
    tap: usize,
    centre: V,
    lim: V,
    neg_lim: V,
) -> V {
    let (dx, dy) = VVC_ALF_TAP_OFFSETS[tap];
    let ax = (x as isize + VVC_ALF_REACH as isize + dx) as usize;
    let bx = (x as isize + VVC_ALF_REACH as isize - dx) as usize;
    // SAFETY: caller validated every row slice covers `whole + 2 * VVC_ALF_REACH`.
    let (a, b) = unsafe {
        (
            V::load_u16_at(ptrs[(3 + dy) as usize].add(ax)),
            V::load_u16_at(ptrs[(3 - dy) as usize].add(bx)),
        )
    };
    a.sub(centre)
        .max(neg_lim)
        .min(lim)
        .add(b.sub(centre).max(neg_lim).min(lim))
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn linear_diff_ptr<V: I16FilterLanes>(
    ptrs: &[*const u16; 7],
    x: usize,
    tap: usize,
    twice: V,
) -> V {
    let (dx, dy) = VVC_ALF_TAP_OFFSETS[tap];
    let ax = (x as isize + VVC_ALF_REACH as isize + dx) as usize;
    let bx = (x as isize + VVC_ALF_REACH as isize - dx) as usize;
    // SAFETY: caller validated every row slice covers `whole + 2 * VVC_ALF_REACH`.
    let (a, b) = unsafe {
        (
            V::load_u16_at(ptrs[(3 + dy) as usize].add(ax)),
            V::load_u16_at(ptrs[(3 - dy) as usize].add(bx)),
        )
    };
    a.add(b).sub(twice)
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn chroma_diff_ptr<V: I16FilterLanes>(
    ptrs: &[*const u16; 7],
    x: usize,
    tap: usize,
    centre: V,
    lim: V,
    neg_lim: V,
) -> V {
    let (dx, dy) = CHROMA_TAP_OFFSETS[tap];
    let ax = (x as isize + VVC_ALF_REACH as isize + dx) as usize;
    let bx = (x as isize + VVC_ALF_REACH as isize - dx) as usize;
    // SAFETY: caller validated every row slice covers `whole + 2 * VVC_ALF_REACH`.
    let (a, b) = unsafe {
        (
            V::load_u16_at(ptrs[(3 + dy) as usize].add(ax)),
            V::load_u16_at(ptrs[(3 - dy) as usize].add(bx)),
        )
    };
    a.sub(centre)
        .max(neg_lim)
        .min(lim)
        .add(b.sub(centre).max(neg_lim).min(lim))
}

#[inline(always)]
#[allow(clippy::inline_always)]
fn row_ptrs(rows: &[&[u16]; 7], min_len: usize) -> [*const u16; 7] {
    let mut ptrs = [core::ptr::null::<u16>(); 7];
    for k in 0..7 {
        assert!(rows[k].len() >= min_len);
        ptrs[k] = rows[k].as_ptr();
    }
    ptrs
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn clipped_row_lanes<V: I16FilterLanes>(
    rows: &[&[u16]; 7],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shift: u32,
    out: &mut [u16],
    depth: BitDepth,
) {
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let round = 1i32 << (shift - 1);
    let blocks_per_vec = V::LANES / VVC_ALF_BLOCK;
    let steps = out.len() / V::LANES;
    let whole = steps * V::LANES;
    assert!(coefficients.len() >= steps * blocks_per_vec && limits.len() >= steps * blocks_per_vec);
    let ptrs = row_ptrs(rows, whole + 2 * VVC_ALF_REACH);
    let dst = out.as_mut_ptr();
    for step in 0..steps {
        let x = step * V::LANES;
        let b0 = step * blocks_per_vec;
        let coeffs = V::expand_alf_blocks(&coefficients[b0..]);
        let lims = V::expand_alf_blocks(&limits[b0..]);
        // SAFETY: `ptrs` and `dst` were bounds-checked once before the loop.
        unsafe {
            let centre = V::load_u16_at(ptrs[3].add(x + VVC_ALF_REACH));
            let mut acc = V::acc_zero();
            for p in 0..VVC_ALF_TAPS / 2 {
                let (lim0, lim1) = (lims[2 * p], lims[2 * p + 1]);
                let d0 = clipped_diff_ptr::<V>(&ptrs, x, 2 * p, centre, lim0, zero.sub(lim0));
                let d1 = clipped_diff_ptr::<V>(&ptrs, x, 2 * p + 1, centre, lim1, zero.sub(lim1));
                acc = V::acc_vec_fma2(acc, d0, coeffs[2 * p], d1, coeffs[2 * p + 1]);
            }
            let offset = V::acc_round_shift_narrow(acc, round, shift);
            centre
                .add_sat(offset)
                .max(zero)
                .min(max_val)
                .store_u16_at(dst.add(x));
        }
    }
    if whole < out.len() {
        let tail_rows = [
            &rows[0][whole..],
            &rows[1][whole..],
            &rows[2][whole..],
            &rows[3][whole..],
            &rows[4][whole..],
            &rows[5][whole..],
            &rows[6][whole..],
        ];
        clipped_row_at(
            &tail_rows,
            &coefficients[whole / VVC_ALF_BLOCK..],
            &limits[whole / VVC_ALF_BLOCK..],
            shift,
            &mut out[whole..],
            depth,
        );
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn clipped_block_row_lanes<V: I16FilterLanes>(
    rows: &[[&[u16]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u16]; VVC_ALF_BLOCK],
    depth: BitDepth,
) {
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let blocks_per_vec = V::LANES / VVC_ALF_BLOCK;
    let width = out[0].len();
    let total_blocks = width.div_ceil(VVC_ALF_BLOCK);
    assert_coeffs_and_limits(
        &coefficients[..total_blocks],
        &limits[..total_blocks],
        depth,
    );
    let steps = width / V::LANES;
    let whole = steps * V::LANES;
    let min_in = whole + 2 * VVC_ALF_REACH;
    let mut ptrs = [[core::ptr::null::<u16>(); 7]; VVC_ALF_BLOCK];
    for r in 0..VVC_ALF_BLOCK {
        assert!(out[r].len() >= whole);
        ptrs[r] = row_ptrs(&rows[r], min_in);
    }
    for step in 0..steps {
        let x = step * V::LANES;
        let b0 = step * blocks_per_vec;
        let coeffs = V::expand_alf_blocks(&coefficients[b0..]);
        let lims = V::expand_alf_blocks(&limits[b0..]);
        // SAFETY: `ptrs` and `out[r]` were bounds-checked once before the loop.
        unsafe {
            for r_base in [0, 2] {
                let c0 = V::load_u16_at(ptrs[r_base][3].add(x + VVC_ALF_REACH));
                let c1 = V::load_u16_at(ptrs[r_base + 1][3].add(x + VVC_ALF_REACH));
                let (mut acc0, mut acc1) = (V::acc_zero(), V::acc_zero());
                for p in 0..VVC_ALF_TAPS / 2 {
                    let (lim0, lim1) = (lims[2 * p], lims[2 * p + 1]);
                    let (neg0, neg1) = (zero.sub(lim0), zero.sub(lim1));
                    let (w0, w1) = (coeffs[2 * p], coeffs[2 * p + 1]);
                    let d0_0 = clipped_diff_ptr::<V>(&ptrs[r_base], x, 2 * p, c0, lim0, neg0);
                    let d1_0 = clipped_diff_ptr::<V>(&ptrs[r_base], x, 2 * p + 1, c0, lim1, neg1);
                    acc0 = V::acc_vec_fma2(acc0, d0_0, w0, d1_0, w1);
                    let d0_1 = clipped_diff_ptr::<V>(&ptrs[r_base + 1], x, 2 * p, c1, lim0, neg0);
                    let d1_1 =
                        clipped_diff_ptr::<V>(&ptrs[r_base + 1], x, 2 * p + 1, c1, lim1, neg1);
                    acc1 = V::acc_vec_fma2(acc1, d0_1, w0, d1_1, w1);
                }
                for (dr, c, acc) in [(0, c0, acc0), (1, c1, acc1)] {
                    let r = r_base + dr;
                    let shift = shifts[r];
                    let offset = V::acc_round_shift_narrow(acc, 1i32 << (shift - 1), shift);
                    c.add_sat(offset)
                        .max(zero)
                        .min(max_val)
                        .store_u16_at(out[r].as_mut_ptr().add(x));
                }
            }
        }
    }
    if whole < width {
        for r in 0..VVC_ALF_BLOCK {
            let tail_rows = [
                &rows[r][0][whole..],
                &rows[r][1][whole..],
                &rows[r][2][whole..],
                &rows[r][3][whole..],
                &rows[r][4][whole..],
                &rows[r][5][whole..],
                &rows[r][6][whole..],
            ];
            clipped_row_at(
                &tail_rows,
                &coefficients[whole / VVC_ALF_BLOCK..],
                &limits[whole / VVC_ALF_BLOCK..],
                shifts[r],
                &mut out[r][whole..],
                depth,
            );
        }
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn filter_block_row_lanes<V: I16FilterLanes>(
    rows: &[[&[u16]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u16]; VVC_ALF_BLOCK],
    depth: BitDepth,
) {
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let blocks_per_vec = V::LANES / VVC_ALF_BLOCK;
    let width = out[0].len();
    let steps = width / V::LANES;
    let whole = steps * V::LANES;
    assert!(coefficients.len() >= steps * blocks_per_vec);
    let min_in = whole + 2 * VVC_ALF_REACH;
    let mut ptrs = [[core::ptr::null::<u16>(); 7]; VVC_ALF_BLOCK];
    for r in 0..VVC_ALF_BLOCK {
        assert!(out[r].len() >= whole);
        ptrs[r] = row_ptrs(&rows[r], min_in);
    }
    for step in 0..steps {
        let x = step * V::LANES;
        let b0 = step * blocks_per_vec;
        let coeffs = V::expand_alf_blocks(&coefficients[b0..]);
        // SAFETY: `ptrs` and `out[r]` were bounds-checked once before the loop.
        unsafe {
            for r_base in [0, 2] {
                let c0 = V::load_u16_at(ptrs[r_base][3].add(x + VVC_ALF_REACH));
                let c1 = V::load_u16_at(ptrs[r_base + 1][3].add(x + VVC_ALF_REACH));
                let (t0, t1) = (c0.add(c0), c1.add(c1));
                let (mut acc0, mut acc1) = (V::acc_zero(), V::acc_zero());
                for p in 0..VVC_ALF_TAPS / 2 {
                    let (w0, w1) = (coeffs[2 * p], coeffs[2 * p + 1]);
                    let d0_0 = linear_diff_ptr::<V>(&ptrs[r_base], x, 2 * p, t0);
                    let d1_0 = linear_diff_ptr::<V>(&ptrs[r_base], x, 2 * p + 1, t0);
                    acc0 = V::acc_vec_fma2(acc0, d0_0, w0, d1_0, w1);
                    let d0_1 = linear_diff_ptr::<V>(&ptrs[r_base + 1], x, 2 * p, t1);
                    let d1_1 = linear_diff_ptr::<V>(&ptrs[r_base + 1], x, 2 * p + 1, t1);
                    acc1 = V::acc_vec_fma2(acc1, d0_1, w0, d1_1, w1);
                }
                for (dr, c, acc) in [(0, c0, acc0), (1, c1, acc1)] {
                    let r = r_base + dr;
                    let shift = shifts[r];
                    let offset = V::acc_round_shift_narrow(acc, 1i32 << (shift - 1), shift);
                    c.add_sat(offset)
                        .max(zero)
                        .min(max_val)
                        .store_u16_at(out[r].as_mut_ptr().add(x));
                }
            }
        }
    }
    if whole < width {
        for r in 0..VVC_ALF_BLOCK {
            let tail_rows = [
                &rows[r][0][whole..],
                &rows[r][1][whole..],
                &rows[r][2][whole..],
                &rows[r][3][whole..],
                &rows[r][4][whole..],
                &rows[r][5][whole..],
                &rows[r][6][whole..],
            ];
            filter_row_at(
                &tail_rows,
                &coefficients[whole / VVC_ALF_BLOCK..],
                shifts[r],
                &mut out[r][whole..],
                depth,
            );
        }
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn chroma_row_lanes<V: I16FilterLanes>(
    rows: &[&[u16]; 7],
    coefficients: [i16; CHROMA_TAPS],
    limits: [i16; CHROMA_TAPS],
    shift: u32,
    out: &mut [u16],
    depth: BitDepth,
) {
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let round = 1i32 << (shift - 1);
    let lims = [
        V::splat(limits[0]),
        V::splat(limits[1]),
        V::splat(limits[2]),
        V::splat(limits[3]),
        V::splat(limits[4]),
        V::splat(limits[5]),
    ];
    let neg_lims = [
        zero.sub(lims[0]),
        zero.sub(lims[1]),
        zero.sub(lims[2]),
        zero.sub(lims[3]),
        zero.sub(lims[4]),
        zero.sub(lims[5]),
    ];
    let steps = out.len() / V::LANES;
    let whole = steps * V::LANES;
    let ptrs = row_ptrs(rows, whole + 2 * VVC_ALF_REACH);
    let dst = out.as_mut_ptr();
    for step in 0..steps {
        let x = step * V::LANES;
        // SAFETY: `ptrs` and `dst` were bounds-checked once before the loop.
        unsafe {
            let centre = V::load_u16_at(ptrs[3].add(x + VVC_ALF_REACH));
            let d0 = chroma_diff_ptr::<V>(&ptrs, x, 0, centre, lims[0], neg_lims[0]);
            let d1 = chroma_diff_ptr::<V>(&ptrs, x, 1, centre, lims[1], neg_lims[1]);
            let d2 = chroma_diff_ptr::<V>(&ptrs, x, 2, centre, lims[2], neg_lims[2]);
            let d3 = chroma_diff_ptr::<V>(&ptrs, x, 3, centre, lims[3], neg_lims[3]);
            let d4 = chroma_diff_ptr::<V>(&ptrs, x, 4, centre, lims[4], neg_lims[4]);
            let d5 = chroma_diff_ptr::<V>(&ptrs, x, 5, centre, lims[5], neg_lims[5]);
            let mut acc = V::acc_zero();
            acc = V::acc_mul_add2(acc, d0, coefficients[0], d1, coefficients[1]);
            acc = V::acc_mul_add2(acc, d2, coefficients[2], d3, coefficients[3]);
            acc = V::acc_mul_add2(acc, d4, coefficients[4], d5, coefficients[5]);
            let offset = V::acc_round_shift_narrow(acc, round, shift);
            centre
                .add_sat(offset)
                .max(zero)
                .min(max_val)
                .store_u16_at(dst.add(x));
        }
    }
    if whole < out.len() {
        let tail_rows = [
            &rows[0][whole..],
            &rows[1][whole..],
            &rows[2][whole..],
            &rows[3][whole..],
            &rows[4][whole..],
            &rows[5][whole..],
            &rows[6][whole..],
        ];
        chroma_row_at(
            &tail_rows,
            coefficients,
            limits,
            shift,
            &mut out[whole..],
            depth,
        );
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn cross_row_lanes<V: I16FilterLanes>(
    rows: &[&[u16]; 4],
    c: [i16; 7],
    out: &mut [u16],
    depth: BitDepth,
) {
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let half = depth.mid() as i16;
    let (min_corr, max_corr) = (V::splat(-half), V::splat(half - 1));
    let steps = out.len() / V::LANES;
    let whole = steps * V::LANES;
    for step in 0..steps {
        let x = step * V::LANES;
        let lx = 2 * x;
        let (_, r0_odd) = V::load_u16_deinterleaved(&rows[0][lx..]);
        let (r1_even, centre) = V::load_u16_deinterleaved(&rows[1][lx..]);
        let (r1_even_next, _) = V::load_u16_deinterleaved(&rows[1][lx + 2..]);
        let (r2_even, r2_odd) = V::load_u16_deinterleaved(&rows[2][lx..]);
        let (r2_even_next, _) = V::load_u16_deinterleaved(&rows[2][lx + 2..]);
        let (_, r3_odd) = V::load_u16_deinterleaved(&rows[3][lx..]);
        let mut acc = V::acc_zero();
        acc = V::acc_mul_add2(acc, r0_odd.sub(centre), c[0], r1_even.sub(centre), c[1]);
        acc = V::acc_mul_add2(
            acc,
            r1_even_next.sub(centre),
            c[2],
            r2_even.sub(centre),
            c[3],
        );
        acc = V::acc_mul_add2(
            acc,
            r2_odd.sub(centre),
            c[4],
            r2_even_next.sub(centre),
            c[5],
        );
        acc = V::acc_mul_add(acc, r3_odd.sub(centre), c[6]);
        let corr = V::acc_round_shift_narrow(acc, 64, 7)
            .max(min_corr)
            .min(max_corr);
        let cur = V::load_u16(&out[x..]);
        cur.add_sat(corr)
            .max(zero)
            .min(max_val)
            .store_u16(&mut out[x..]);
    }
    if whole < out.len() {
        let tail_rows = [
            &rows[0][2 * whole..],
            &rows[1][2 * whole..],
            &rows[2][2 * whole..],
            &rows[3][2 * whole..],
        ];
        cross_row_at(&tail_rows, c, &mut out[whole..], depth);
    }
}

macro_rules! isa_alf_simd {
    ($mod_name:ident, $lanes:ty, $feature:literal) => {
        pub(crate) mod $mod_name {
            use nc_common::BitDepth;

            use super::super::{VVC_ALF_BLOCK, VVC_ALF_TAPS};
            use super::{
                chroma_row_lanes, clipped_block_row_lanes, clipped_row_lanes, cross_row_lanes,
                filter_block_row_lanes,
            };

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn clipped_block_row(
                rows: &[[&[u16]; 7]; VVC_ALF_BLOCK],
                coefficients: &[[i16; VVC_ALF_TAPS]],
                limits: &[[i16; VVC_ALF_TAPS]],
                shifts: &[u32; VVC_ALF_BLOCK],
                out: &mut [&mut [u16]; VVC_ALF_BLOCK],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    clipped_block_row_lanes::<$lanes>(
                        rows,
                        coefficients,
                        limits,
                        shifts,
                        out,
                        depth,
                    );
                }
            }

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn clipped_row(
                rows: &[&[u16]; 7],
                coefficients: &[[i16; VVC_ALF_TAPS]],
                limits: &[[i16; VVC_ALF_TAPS]],
                shift: u32,
                out: &mut [u16],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    clipped_row_lanes::<$lanes>(rows, coefficients, limits, shift, out, depth);
                }
            }

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn filter_block_row(
                rows: &[[&[u16]; 7]; VVC_ALF_BLOCK],
                coefficients: &[[i16; VVC_ALF_TAPS]],
                shifts: &[u32; VVC_ALF_BLOCK],
                out: &mut [&mut [u16]; VVC_ALF_BLOCK],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    filter_block_row_lanes::<$lanes>(rows, coefficients, shifts, out, depth);
                }
            }

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn chroma_row(
                rows: &[&[u16]; 7],
                coefficients: [i16; 6],
                limits: [i16; 6],
                shift: u32,
                out: &mut [u16],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    chroma_row_lanes::<$lanes>(rows, coefficients, limits, shift, out, depth);
                }
            }

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn cross_row(
                rows: &[&[u16]; 4],
                coefficients: [i16; 7],
                out: &mut [u16],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    cross_row_lanes::<$lanes>(rows, coefficients, out, depth);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_alf_simd!(avx2, crate::simd::avx2::I16x16, "avx2");

#[cfg(target_arch = "x86_64")]
isa_alf_simd!(
    avx512,
    crate::simd::avx512_i16::I16x32,
    "avx512f,avx512bw,avx512vl"
);

#[cfg(target_arch = "aarch64")]
isa_alf_simd!(neon, crate::simd::neon::I16x8, "neon");

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_alf_simd!(simd128, crate::simd::simd128::I16x8, "simd128");
