//! Shared (`u8` and `u16`) VVC ALF and CC-ALF row kernels over [`I16FilterLanes`].
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

#[cfg(target_arch = "aarch64")]
mod neon_clipped;

use nc_common::BitDepth;

use crate::simd::{I16FilterLanes, SimdSample};

use super::clipped::{
    CHROMA_TAP_OFFSETS, CHROMA_TAPS, assert_coeffs_and_limits, chroma_row_at, clipped_row_at,
};
use super::cross::cross_row_at;
use super::{VVC_ALF_BLOCK, VVC_ALF_REACH, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, filter_row_at};

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn load_luma_tap_ptr<S: SimdSample, V: I16FilterLanes>(
    ptrs: &[*const S; 7],
    x: usize,
    tap: usize,
) -> (V, V) {
    let (dx, dy) = VVC_ALF_TAP_OFFSETS[tap];
    let ax = (x as isize + VVC_ALF_REACH as isize + dx) as usize;
    let bx = (x as isize + VVC_ALF_REACH as isize - dx) as usize;
    // SAFETY: caller validated every row slice covers `whole + 2 * VVC_ALF_REACH`.
    unsafe {
        (
            S::load_lanes::<V>(ptrs[(3 + dy) as usize].add(ax)),
            S::load_lanes::<V>(ptrs[(3 - dy) as usize].add(bx)),
        )
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn linear_diff_ptr<V: I16FilterLanes>(
    ptrs: &[*const u16; 7],
    x: usize,
    tap: usize,
    twice: V,
) -> V {
    // SAFETY: forwarded from `filter_block_row_lanes` after bounds validation.
    let (a, b) = unsafe { load_luma_tap_ptr::<u16, V>(ptrs, x, tap) };
    a.add(b).sub(twice)
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn load_chroma_tap_ptr<S: SimdSample, V: I16FilterLanes>(
    ptrs: &[*const S; 7],
    x: usize,
    tap: usize,
) -> (V, V) {
    let (dx, dy) = CHROMA_TAP_OFFSETS[tap];
    let ax = (x as isize + VVC_ALF_REACH as isize + dx) as usize;
    let bx = (x as isize + VVC_ALF_REACH as isize - dx) as usize;
    // SAFETY: caller validated every row slice covers `whole + 2 * VVC_ALF_REACH`.
    unsafe {
        (
            S::load_lanes::<V>(ptrs[(3 + dy) as usize].add(ax)),
            S::load_lanes::<V>(ptrs[(3 - dy) as usize].add(bx)),
        )
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
fn row_ptrs<S>(rows: &[&[S]; 7], min_len: usize) -> [*const S; 7] {
    let mut ptrs = [core::ptr::null::<S>(); 7];
    for k in 0..7 {
        assert!(rows[k].len() >= min_len);
        ptrs[k] = rows[k].as_ptr();
    }
    ptrs
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn clipped_row_lanes<S: SimdSample, V: I16FilterLanes>(
    rows: &[&[S]; 7],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shift: u32,
    out: &mut [S],
    depth: BitDepth,
) {
    let depth = depth.of::<S>();
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
        let w_params = V::expand_alf_params(&coefficients[b0..]);
        let l_params = V::expand_alf_params(&limits[b0..]);
        // SAFETY: `ptrs` and `dst` were bounds-checked once before the loop.
        unsafe {
            let centre = S::load_lanes::<V>(ptrs[3].add(x + VVC_ALF_REACH));
            let mut acc = V::acc_zero();
            for p in 0..VVC_ALF_TAPS / 2 {
                let s0 = load_luma_tap_ptr::<S, V>(&ptrs, x, 2 * p);
                let s1 = load_luma_tap_ptr::<S, V>(&ptrs, x, 2 * p + 1);
                acc = V::acc_alf_clipped_pair(acc, s0, s1, centre, l_params[p], w_params[p]);
            }
            let offset = V::acc_round_shift_narrow(acc, round, shift);
            S::store_clamped(centre.add_sat(offset), zero, max_val, dst.add(x));
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
unsafe fn clipped_block_row_vectors<S: SimdSample, V: I16FilterLanes>(
    ptrs: &[[*const S; 7]; VVC_ALF_BLOCK],
    dsts: &[*mut S; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    depth: BitDepth,
    start: usize,
    end: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if V::LANES == 8 {
        // SAFETY: the containing dispatcher enables NEON and checked the
        // source halos, parameter arrays and destination lengths.
        unsafe {
            neon_clipped::filter_vectors(
                ptrs,
                dsts,
                coefficients,
                limits,
                shifts,
                depth,
                start,
                end,
            );
        }
        return;
    }

    let (zero, max_val) = (V::splat(0), V::splat(depth.max() as i16));
    let blocks_per_vec = V::LANES / VVC_ALF_BLOCK;
    let steps = (end - start) / V::LANES;
    for step in 0..steps {
        let x = start + step * V::LANES;
        let b0 = x / VVC_ALF_BLOCK;
        let w_params = V::expand_alf_params(&coefficients[b0..b0 + blocks_per_vec]);
        let l_params = V::expand_alf_params(&limits[b0..b0 + blocks_per_vec]);
        // SAFETY: `ptrs` and `dsts` were bounds-checked once before the loop.
        unsafe {
            for r in 0..VVC_ALF_BLOCK {
                let centre = S::load_lanes::<V>(ptrs[r][3].add(x + VVC_ALF_REACH));
                let mut acc = V::acc_zero();
                for p in 0..VVC_ALF_TAPS / 2 {
                    let s0 = load_luma_tap_ptr::<S, V>(&ptrs[r], x, 2 * p);
                    let s1 = load_luma_tap_ptr::<S, V>(&ptrs[r], x, 2 * p + 1);
                    acc = V::acc_alf_clipped_pair(acc, s0, s1, centre, l_params[p], w_params[p]);
                }
                let shift = shifts[r];
                let offset = if shift == 7 {
                    V::acc_round_shift_narrow(acc, 64, 7)
                } else {
                    V::acc_round_shift_narrow(acc, 1i32 << (shift - 1), shift)
                };
                S::store_clamped(centre.add_sat(offset), zero, max_val, dsts[r].add(x));
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn clipped_block_row_lanes<S, V, M, N>(
    rows: &[[&[S]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [S]; VVC_ALF_BLOCK],
    depth: BitDepth,
) where
    S: SimdSample,
    V: I16FilterLanes,
    M: I16FilterLanes,
    N: I16FilterLanes,
{
    let depth = depth.of::<S>();
    let width = out[0].len();
    let total_blocks = width.div_ceil(VVC_ALF_BLOCK);
    assert_coeffs_and_limits(
        &coefficients[..total_blocks],
        &limits[..total_blocks],
        depth,
    );
    let ptrs = rows.map(|row| row.map(<[S]>::as_ptr));
    let dsts = out.each_mut().map(|row| row.as_mut_ptr());

    let wide_end = width / V::LANES * V::LANES;
    let mut whole = wide_end;
    // SAFETY: the dispatcher checked every source halo and the four output
    // lengths. Each range ends at or before width, starts on a four-sample
    // block boundary, and is processed only in complete vectors.
    unsafe {
        clipped_block_row_vectors::<S, V>(
            &ptrs,
            &dsts,
            coefficients,
            limits,
            shifts,
            depth,
            0,
            whole,
        );
        if M::LANES < V::LANES && width - whole >= M::LANES {
            let end = whole + (width - whole) / M::LANES * M::LANES;
            clipped_block_row_vectors::<S, M>(
                &ptrs,
                &dsts,
                coefficients,
                limits,
                shifts,
                depth,
                whole,
                end,
            );
            whole = end;
        }
        if N::LANES < M::LANES && width - whole >= N::LANES {
            let end = whole + (width - whole) / N::LANES * N::LANES;
            clipped_block_row_vectors::<S, N>(
                &ptrs,
                &dsts,
                coefficients,
                limits,
                shifts,
                depth,
                whole,
                end,
            );
            whole = end;
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
    let vec_blocks = steps * blocks_per_vec;
    let coeff_vec = &coefficients[..vec_blocks];
    let shifts = *shifts;
    let min_in = whole + 2 * VVC_ALF_REACH;
    let mut ptrs = [[core::ptr::null::<u16>(); 7]; VVC_ALF_BLOCK];
    let mut dsts = [core::ptr::null_mut::<u16>(); VVC_ALF_BLOCK];
    for r in 0..VVC_ALF_BLOCK {
        assert!(out[r].len() >= whole);
        ptrs[r] = row_ptrs(&rows[r], min_in);
        dsts[r] = out[r].as_mut_ptr();
    }
    for step in 0..steps {
        let x = step * V::LANES;
        let b0 = step * blocks_per_vec;
        let w_params = V::expand_alf_params(&coeff_vec[b0..b0 + blocks_per_vec]);
        // SAFETY: `ptrs` and `dsts` were bounds-checked once before the loop.
        unsafe {
            for r in 0..VVC_ALF_BLOCK {
                let c = V::load_u16_at(ptrs[r][3].add(x + VVC_ALF_REACH));
                let twice = c.add(c);
                let mut acc = V::acc_zero();
                for (p, &wp) in w_params.iter().enumerate() {
                    let d0 = linear_diff_ptr::<V>(&ptrs[r], x, 2 * p, twice);
                    let d1 = linear_diff_ptr::<V>(&ptrs[r], x, 2 * p + 1, twice);
                    acc = V::acc_alf_linear_pair(acc, d0, d1, wp);
                }
                let shift = shifts[r];
                let offset = if shift == 7 {
                    V::acc_round_shift_narrow(acc, 64, 7)
                } else {
                    V::acc_round_shift_narrow(acc, 1i32 << (shift - 1), shift)
                };
                c.add_sat(offset)
                    .max(zero)
                    .min(max_val)
                    .store_u16_at(dsts[r].add(x));
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
unsafe fn chroma_row_lanes<S: SimdSample, V: I16FilterLanes>(
    rows: &[&[S]; 7],
    coefficients: [i16; CHROMA_TAPS],
    limits: [i16; CHROMA_TAPS],
    shift: u32,
    out: &mut [S],
    depth: BitDepth,
) {
    let depth = depth.of::<S>();
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let round = 1i32 << (shift - 1);
    let w_params = [
        V::splat_alf_pair(coefficients[0], coefficients[1]),
        V::splat_alf_pair(coefficients[2], coefficients[3]),
        V::splat_alf_pair(coefficients[4], coefficients[5]),
    ];
    let l_params = [
        V::splat_alf_pair(limits[0], limits[1]),
        V::splat_alf_pair(limits[2], limits[3]),
        V::splat_alf_pair(limits[4], limits[5]),
    ];
    let steps = out.len() / V::LANES;
    let whole = steps * V::LANES;
    let ptrs = row_ptrs(rows, whole + 2 * VVC_ALF_REACH);
    let dst = out.as_mut_ptr();
    for step in 0..steps {
        let x = step * V::LANES;
        // SAFETY: `ptrs` and `dst` were bounds-checked once before the loop.
        unsafe {
            let centre = S::load_lanes::<V>(ptrs[3].add(x + VVC_ALF_REACH));
            let mut acc = V::acc_zero();
            for p in 0..CHROMA_TAPS / 2 {
                let s0 = load_chroma_tap_ptr::<S, V>(&ptrs, x, 2 * p);
                let s1 = load_chroma_tap_ptr::<S, V>(&ptrs, x, 2 * p + 1);
                acc = V::acc_alf_clipped_pair(acc, s0, s1, centre, l_params[p], w_params[p]);
            }
            let offset = V::acc_round_shift_narrow(acc, round, shift);
            S::store_clamped(centre.add_sat(offset), zero, max_val, dst.add(x));
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
unsafe fn cross_row_lanes<S: SimdSample, V: I16FilterLanes>(
    rows: &[&[S]; 4],
    c: [i16; 7],
    out: &mut [S],
    depth: BitDepth,
) {
    let depth = depth.of::<S>();
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let half = depth.mid() as i16;
    let (min_corr, max_corr) = (V::splat(-half), V::splat(half - 1));
    let steps = out.len() / V::LANES;
    let whole = steps * V::LANES;
    for step in 0..steps {
        let x = step * V::LANES;
        let lx = 2 * x;
        let (_, r0_odd) = S::load_deinterleaved::<V>(&rows[0][lx..]);
        let (r1_even, centre) = S::load_deinterleaved::<V>(&rows[1][lx..]);
        let (r1_even_next, _) = S::load_deinterleaved::<V>(&rows[1][lx + 2..]);
        let (r2_even, r2_odd) = S::load_deinterleaved::<V>(&rows[2][lx..]);
        let (r2_even_next, _) = S::load_deinterleaved::<V>(&rows[2][lx + 2..]);
        let (_, r3_odd) = S::load_deinterleaved::<V>(&rows[3][lx..]);
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
        let cur = S::load_slice::<V>(&out[x..]);
        S::store_clamped_slice(cur.add_sat(corr), zero, max_val, &mut out[x..]);
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
    ($mod_name:ident, $lanes:ty, $mid:ty, $narrow:ty, $feature:literal) => {
        pub(crate) mod $mod_name {
            use nc_common::BitDepth;

            use crate::simd::SimdSample;

            use super::super::{VVC_ALF_BLOCK, VVC_ALF_TAPS};
            use super::{
                chroma_row_lanes, clipped_block_row_lanes, clipped_row_lanes, cross_row_lanes,
                filter_block_row_lanes,
            };

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn clipped_block_row<S: SimdSample>(
                rows: &[[&[S]; 7]; VVC_ALF_BLOCK],
                coefficients: &[[i16; VVC_ALF_TAPS]],
                limits: &[[i16; VVC_ALF_TAPS]],
                shifts: &[u32; VVC_ALF_BLOCK],
                out: &mut [&mut [S]; VVC_ALF_BLOCK],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    clipped_block_row_lanes::<S, $lanes, $mid, $narrow>(
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
            pub(crate) unsafe fn clipped_row<S: SimdSample>(
                rows: &[&[S]; 7],
                coefficients: &[[i16; VVC_ALF_TAPS]],
                limits: &[[i16; VVC_ALF_TAPS]],
                shift: u32,
                out: &mut [S],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    clipped_row_lanes::<S, $lanes>(rows, coefficients, limits, shift, out, depth);
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
            pub(crate) unsafe fn chroma_row<S: SimdSample>(
                rows: &[&[S]; 7],
                coefficients: [i16; 6],
                limits: [i16; 6],
                shift: u32,
                out: &mut [S],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    chroma_row_lanes::<S, $lanes>(rows, coefficients, limits, shift, out, depth);
                }
            }

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn cross_row<S: SimdSample>(
                rows: &[&[S]; 4],
                coefficients: [i16; 7],
                out: &mut [S],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked slice lengths.
                unsafe {
                    cross_row_lanes::<S, $lanes>(rows, coefficients, out, depth);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_alf_simd!(
    avx2,
    crate::simd::avx2::I16x16,
    crate::simd::avx2::I16x16,
    crate::simd::avx2::I16x8,
    "avx2"
);

#[cfg(target_arch = "x86_64")]
isa_alf_simd!(
    avx512,
    crate::simd::avx512_i16::I16x32,
    crate::simd::avx2::I16x16,
    crate::simd::avx2::I16x8,
    "avx2,avx512f,avx512bw,avx512vl"
);

#[cfg(target_arch = "aarch64")]
isa_alf_simd!(
    neon,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8,
    "neon"
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_alf_simd!(
    simd128,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8,
    "simd128"
);
