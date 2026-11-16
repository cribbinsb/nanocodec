//! High-bit-depth (`u16`, 10- and 12-bit) and shared rectangular (`u8`/`u16`)
//! AV1 intra prediction & `CfL` SIMD kernels across AVX2, AVX-512, and NEON.
//!
//! Attribution: directional run / 8x8 tile transpose, 1-pass `pmaddwd` smooth
//! formulation, Paeth lane selection, filter-intra 4x2 patch dot products, and
//! `pmulhrsw`/`sqrdmulh` `CfL` scaling follow dav1d 1.5.4 (`src/ipred_tmpl.c`,
//! `src/x86/ipred16_avx2.asm`, `src/x86/ipred16_avx512.asm`,
//! `src/arm/64/ipred16.S`, BSD-2-Clause; see `THIRD_PARTY.md`).

#![allow(unsafe_code, clippy::inline_always)]
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code, unused_imports)
)]

use core::mem::MaybeUninit;

use nc_common::{BitDepth, PlaneView, PlaneViewMut};

use super::generic::{paeth_distances, paeth_select};
use crate::Isa;
use crate::av1_mc::high_blend::BlendSample;
use crate::deblock::samples::transpose;
use crate::intra_prediction::av1_block::cfl::{CFL_MAX_SIDE, cfl_predict_scalar};
use crate::intra_prediction::av1_block::{BlockEdges, IntraKind};
use crate::intra_prediction::directional::{self, DirectionalKernels};
use crate::intra_prediction::filter::{
    FILTER_TAP_COLUMNS, FilterGrid, FilterIntraMode, PATCH_SAMPLES, SCALE_BITS, TAPS, patch_inputs,
    predict_patches_at, store_patch,
};
use crate::intra_prediction::{IntraEdges, SMOOTH_SCALE, SMOOTH_SHIFT, smooth_weights};
use crate::simd::{I16FilterLanes, I16Lanes, I16x8Shuffles};

const PATCH_W: usize = 4;
const PATCH_H: usize = 2;

pub(crate) type FilterPatches16Fn = unsafe fn(
    grid: &mut FilterGrid<u16>,
    width: usize,
    height: usize,
    mode: FilterIntraMode,
    depth: BitDepth,
);
pub(crate) type Directional16Fn =
    unsafe fn(edges: &IntraEdges<u16>, angle: u16, dst: &mut PlaneViewMut<'_, u16>);
pub(crate) type RunFn<T> = unsafe fn(window: &[T], weight: usize, out: &mut [T]);
pub(crate) type PaethBlockFn<T> = unsafe fn(edges: &BlockEdges<T>, dst: &mut PlaneViewMut<'_, T>);
pub(crate) type SmoothBlockFn<T> =
    unsafe fn(edges: &BlockEdges<T>, kind: IntraKind, dst: &mut PlaneViewMut<'_, T>);
pub(crate) type CflPredictFn<T> = unsafe fn(
    luma: &PlaneView<'_, T>,
    luma_extent: (usize, usize),
    dc: T,
    alpha_q3: i32,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
);

/// High-bit-depth (`u16`) and shared rectangular (`u8`) AV1 intra + `CfL` kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HighAv1IntraFns {
    pub filter_patches: FilterPatches16Fn,
    pub directional: Directional16Fn,
    pub run_u16: RunFn<u16>,
    pub run_u8: RunFn<u8>,
    pub paeth_u16: PaethBlockFn<u16>,
    pub paeth_u8: PaethBlockFn<u8>,
    pub smooth_u16: SmoothBlockFn<u16>,
    pub smooth_u8: SmoothBlockFn<u8>,
    pub cfl_u16: CflPredictFn<u16>,
    pub cfl_u8: CflPredictFn<u8>,
}

fn scalar_patches_u16(
    grid: &mut FilterGrid<u16>,
    width: usize,
    height: usize,
    mode: FilterIntraMode,
    depth: BitDepth,
) {
    predict_patches_at(grid, width, height, mode, depth);
}

fn scalar_directional_u16(edges: &IntraEdges<u16>, angle: u16, dst: &mut PlaneViewMut<'_, u16>) {
    edges.predict_angle(angle, dst);
}

fn scalar_run<T: nc_common::Sample>(window: &[T], weight: usize, out: &mut [T]) {
    directional::interpolate_run(window, weight, out);
}

fn scalar_paeth<T: nc_common::Sample>(edges: &BlockEdges<T>, dst: &mut PlaneViewMut<'_, T>) {
    edges.paeth_scalar(dst);
}

fn scalar_smooth<T: nc_common::Sample>(
    edges: &BlockEdges<T>,
    kind: IntraKind,
    dst: &mut PlaneViewMut<'_, T>,
) {
    edges.smooth(kind, dst);
}

impl HighAv1IntraFns {
    pub const SCALAR: Self = Self {
        filter_patches: scalar_patches_u16,
        directional: scalar_directional_u16,
        run_u16: scalar_run::<u16>,
        run_u8: scalar_run::<u8>,
        paeth_u16: scalar_paeth::<u16>,
        paeth_u8: scalar_paeth::<u8>,
        smooth_u16: scalar_smooth::<u16>,
        smooth_u8: scalar_smooth::<u8>,
        cfl_u16: cfl_predict_scalar::<u16>,
        cfl_u8: cfl_predict_scalar::<u8>,
    };

    pub fn for_isa(isa: Isa) -> Self {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86_avx2::FNS,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86_avx512::FNS,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon_high::FNS,
            _ => Self::SCALAR,
        }
    }
}

// ----------------------------------------- Filter Intra (u16) ----

#[inline(always)]
unsafe fn filter_patches_u16<V8: I16Lanes>(
    grid: &mut FilterGrid<u16>,
    width: usize,
    height: usize,
    mode: FilterIntraMode,
    depth: BitDepth,
) {
    if depth.bits() > 12 {
        predict_patches_at(grid, width, height, mode, depth);
        return;
    }
    assert_eq!(V8::LANES, PATCH_SAMPLES);
    let w: [V8; TAPS] = FILTER_TAP_COLUMNS[mode.symbol()].map(|col| V8::load(&col));
    let round = 1_i32 << (SCALE_BITS - 1);
    let zero = V8::splat(0);
    let max_v = V8::splat(depth.max() as i16);
    for r in (1..=height).step_by(PATCH_H) {
        for c in (1..=width).step_by(PATCH_W) {
            let p = patch_inputs(grid, r, c).map(|v| v as i16);
            let mut acc = V8::acc_mul_add2(V8::acc_zero(), w[0], p[0], w[1], p[1]);
            acc = V8::acc_mul_add2(acc, w[2], p[2], w[3], p[3]);
            acc = V8::acc_mul_add2(acc, w[4], p[4], w[5], p[5]);
            acc = V8::acc_mul_add(acc, w[6], p[6]);
            let v = V8::acc_round_shift_narrow(acc, round, SCALE_BITS)
                .max(zero)
                .min(max_v);
            let mut raw = [0i16; PATCH_SAMPLES];
            v.store(&mut raw);
            store_patch(grid, r, c, raw.map(|s| s as u16));
        }
    }
}

// ----------------------------------------- Directional (u16) ----

#[inline(always)]
unsafe fn run_u16_lanes<V8: I16Lanes, VWide: I16Lanes>(
    window: &[u16],
    weight: usize,
    out: &mut [u16],
) {
    assert_eq!(window.len(), out.len() + 1);
    let len = out.len();
    if weight == 0 {
        out.copy_from_slice(&window[..len]);
        return;
    }
    if len < 8 || window.iter().any(|&v| v > 4095) {
        directional::interpolate_run(window, weight, out);
        return;
    }
    let w1 = weight as i16;
    let w0 = 32 - w1;
    let src = window.as_ptr().cast::<i16>();
    let dst = out.as_mut_ptr().cast::<i16>();
    let mut done = 0;
    // SAFETY: `window.len() == len + 1` and `len >= 8` checked above.
    unsafe {
        while done + VWide::LANES <= len {
            let a = VWide::load(core::slice::from_raw_parts(src.add(done), VWide::LANES));
            let b = VWide::load(core::slice::from_raw_parts(src.add(done + 1), VWide::LANES));
            let acc = VWide::acc_mul_add2(VWide::acc_zero(), a, w0, b, w1);
            VWide::acc_round_shift_narrow(acc, 16, 5)
                .store(core::slice::from_raw_parts_mut(dst.add(done), VWide::LANES));
            done += VWide::LANES;
        }
        while done < len {
            let at = done.min(len - 8);
            let a = V8::load(core::slice::from_raw_parts(src.add(at), 8));
            let b = V8::load(core::slice::from_raw_parts(src.add(at + 1), 8));
            let acc = V8::acc_mul_add2(V8::acc_zero(), a, w0, b, w1);
            V8::acc_round_shift_narrow(acc, 16, 5)
                .store(core::slice::from_raw_parts_mut(dst.add(at), 8));
            done = at + 8;
        }
    }
}

#[inline(always)]
unsafe fn store_transposed_u16<V8: I16x8Shuffles>(
    columns: &[u16],
    side: usize,
    dst: &mut PlaneViewMut<'_, u16>,
) {
    if side < 8 {
        directional::store_transposed(columns, side, dst);
        return;
    }
    assert!(side.is_multiple_of(8) && dst.width() >= side && dst.height() >= side);
    let stride = dst.stride();
    let out = dst.data_mut();
    assert!(columns.len() >= side * side && out.len() >= (side - 1) * stride + side);
    let (src_p, out_p) = (
        columns.as_ptr().cast::<i16>(),
        out.as_mut_ptr().cast::<i16>(),
    );
    for i0 in (0..side).step_by(8) {
        for j0 in (0..side).step_by(8) {
            let mut rows = [V8::splat(0); 8];
            for (k, row) in rows.iter_mut().enumerate() {
                // SAFETY: `j0 + k < side` and `i0 + 8 <= side` inside checked `side * side`.
                *row = V8::load(unsafe {
                    core::slice::from_raw_parts(src_p.add((j0 + k) * side + i0), 8)
                });
            }
            for (k, row) in transpose(rows).into_iter().enumerate() {
                // SAFETY: `i0 + k < side` and `j0 + 8 <= side` inside checked destination.
                row.store(unsafe {
                    core::slice::from_raw_parts_mut(out_p.add((i0 + k) * stride + j0), 8)
                });
            }
        }
    }
}

// ----------------------------------------- Shared Paeth (u8 & u16) ----

#[inline(always)]
unsafe fn paeth_block_lanes<S: BlendSample, V8: I16FilterLanes, VWide: I16FilterLanes>(
    edges: &BlockEdges<S>,
    dst: &mut PlaneViewMut<'_, S>,
) {
    let (w, h) = edges.shape();
    assert!(dst.width() == w && dst.height() == h);
    let (above, left, corner_s) = edges.edges();
    if corner_s.to_i32() > 4095 {
        edges.paeth_scalar(dst);
        return;
    }
    if w == 4 {
        let a4: [i16; 4] = core::array::from_fn(|i| above[i].to_i32() as i16);
        let top = V8::load_halves(&a4, &a4);
        let corner = V8::splat(corner_s.to_i32() as i16);
        for pair in 0..h / 2 {
            let l0 = [left[2 * pair].to_i32() as i16; 4];
            let l1 = [left[2 * pair + 1].to_i32() as i16; 4];
            let l = V8::load_halves(&l0, &l1);
            let sel = paeth_select((l, top, corner), paeth_distances(l, top, corner));
            let (mut r0, mut r1) = ([0i16; 4], [0i16; 4]);
            sel.store_halves(&mut r0, &mut r1);
            for c in 0..4 {
                dst.row_mut(2 * pair)[c] = S::from_i32(i32::from(r0[c]));
                dst.row_mut(2 * pair + 1)[c] = S::from_i32(i32::from(r1[c]));
            }
        }
        return;
    }
    let corner = VWide::splat(corner_s.to_i32() as i16);
    let corner8 = V8::splat(corner_s.to_i32() as i16);
    for (y, &ls) in left[..h].iter().enumerate() {
        let lv = ls.to_i32() as i16;
        let (lw, l8) = (VWide::splat(lv), V8::splat(lv));
        let row = &mut dst.row_mut(y)[..w];
        let mut x0 = 0;
        while x0 + VWide::LANES <= w {
            // SAFETY: `x0 + VWide::LANES <= w <= EDGE_LEN`.
            unsafe {
                let top = S::load_i16::<VWide>(&above[x0..]);
                let sel = paeth_select((lw, top, corner), paeth_distances(lw, top, corner));
                S::store_i16(sel, &mut row[x0..]);
            }
            x0 += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && x0 + 8 <= w {
            // SAFETY: `x0 + 8 <= w <= EDGE_LEN`.
            unsafe {
                let top = S::load_i16::<V8>(&above[x0..]);
                let sel = paeth_select((l8, top, corner8), paeth_distances(l8, top, corner8));
                S::store_i16(sel, &mut row[x0..]);
            }
            x0 += 8;
        }
    }
}

// ----------------------------------------- Shared Smooth (u8 & u16) ----

#[inline(always)]
unsafe fn smooth_vec<V: I16Lanes>(
    kind: IntraKind,
    top: V,
    wx: V,
    wy: i16,
    slope: i16,
    round: i32,
    shift: u32,
) -> V {
    let acc = match kind {
        IntraKind::Smooth => V::acc_mul_add2(V::acc_zero(), top, wy, wx, slope),
        IntraKind::SmoothVertical => V::acc_mul_add(V::acc_zero(), top, wy),
        _ => V::acc_mul_add(V::acc_zero(), wx, slope),
    };
    V::acc_round_shift_narrow(acc, round, shift)
}

#[inline(always)]
unsafe fn smooth_block_lanes<S: BlendSample, V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    edges: &BlockEdges<S>,
    kind: IntraKind,
    dst: &mut PlaneViewMut<'_, S>,
) {
    let (w, h) = edges.shape();
    assert!(dst.width() == w && dst.height() == h);
    let (above, left, _) = edges.edges();
    let bottom = left[h - 1].to_i32();
    let right = above[w - 1].to_i32();
    if bottom > 4095 || right > 4095 {
        edges.smooth(kind, dst);
        return;
    }
    let weights_x = smooth_weights(w);
    let weights_y = smooth_weights(h);
    let shift = if kind == IntraKind::Smooth {
        SMOOTH_SHIFT + 1
    } else {
        SMOOTH_SHIFT
    };
    for (r, &wy_u8) in weights_y.iter().enumerate() {
        let wy = i32::from(wy_u8);
        let l = left[r].to_i32();
        let slope = (l - right) as i16;
        let round = match kind {
            IntraKind::Smooth => {
                (SMOOTH_SCALE as i32 - wy) * bottom
                    + (SMOOTH_SCALE as i32) * right
                    + SMOOTH_SCALE as i32
            }
            IntraKind::SmoothVertical => {
                (SMOOTH_SCALE as i32 - wy) * bottom + (SMOOTH_SCALE as i32 / 2)
            }
            _ => (SMOOTH_SCALE as i32) * right + (SMOOTH_SCALE as i32 / 2),
        };
        let wy_i16 = wy as i16;
        let out = &mut dst.row_mut(r)[..w];
        let mut x0 = 0;
        while x0 + VWide::LANES <= w {
            // SAFETY: `x0 + VWide::LANES <= w`.
            unsafe {
                let top = S::load_i16::<VWide>(&above[x0..]);
                let wx = VWide::load_u8(&weights_x[x0..]);
                let v = smooth_vec(kind, top, wx, wy_i16, slope, round, shift);
                S::store_i16(v, &mut out[x0..]);
            }
            x0 += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && x0 + 8 <= w {
            // SAFETY: `x0 + 8 <= w`.
            unsafe {
                let top = S::load_i16::<V8>(&above[x0..]);
                let wx = V8::load_u8(&weights_x[x0..]);
                let v = smooth_vec(kind, top, wx, wy_i16, slope, round, shift);
                S::store_i16(v, &mut out[x0..]);
            }
            x0 += 8;
        }
        if x0 + 4 <= w {
            // SAFETY: `x0 + 4 <= w`.
            unsafe {
                let top = S::load_i16::<V4>(&above[x0..]);
                let wx = V4::load_u8(&weights_x[x0..]);
                let v = smooth_vec(kind, top, wx, wy_i16, slope, round, shift);
                S::store_i16(v, &mut out[x0..]);
            }
        }
    }
}

// ----------------------------------------- Shared One-Pass CfL (u8 & u16) ----

const fn is_cfl_side(n: usize) -> bool {
    n.is_power_of_two() && n >= 4 && n <= CFL_MAX_SIDE
}

#[inline(always)]
fn scale_cfl<V: I16Lanes>(raw: V, avg: V, alpha_q15: V, neg: bool) -> V {
    let ac = raw.sub(avg);
    let signed = if neg { V::splat(0).sub(ac) } else { ac };
    ac.abs().mul_q15_round(alpha_q15).with_sign_of(signed)
}

#[inline(always)]
unsafe fn cfl_predict_lanes<S: BlendSample, V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    luma: &PlaneView<'_, S>,
    (luma_width, luma_height): (usize, usize),
    dc: S,
    alpha_q3: i32,
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
) {
    let (w, h) = (dst.width(), dst.height());
    assert!(is_cfl_side(w) && is_cfl_side(h), "CfL block {w}x{h}");
    for (extent, side) in [(luma_width, w), (luma_height, h)] {
        assert!(
            extent > 0 && extent.is_multiple_of(2) && extent <= 2 * side,
            "CfL luma extent {extent} for chroma side {side}"
        );
    }
    assert!(luma.width() >= luma_width && luma.height() >= luma_height);
    if alpha_q3 == 0 {
        dst.fill(dc);
        return;
    }
    if alpha_q3.abs() > 16 || depth.bits() > 12 {
        cfl_predict_scalar(luma, (luma_width, luma_height), dc, alpha_q3, dst, depth);
        return;
    }
    let (stored_w, stored_h) = (luma_width / 2, luma_height / 2);
    let area = w * h;
    let mut uninit = [MaybeUninit::<i16>::uninit(); CFL_MAX_SIDE * CFL_MAX_SIDE];
    let mut last_row = [0i16; CFL_MAX_SIDE];
    for r in 0..h {
        if r < stored_h {
            let (top, bot) = (luma.row(2 * r), luma.row(2 * r + 1));
            for c in 0..stored_w {
                let sum = top[2 * c].to_i32()
                    + top[2 * c + 1].to_i32()
                    + bot[2 * c].to_i32()
                    + bot[2 * c + 1].to_i32();
                last_row[c] = (sum << 1) as i16;
            }
            let pad = last_row[stored_w - 1];
            last_row[stored_w..w].fill(pad);
        }
        for (slot, &val) in uninit[r * w..(r + 1) * w].iter_mut().zip(&last_row[..w]) {
            *slot = MaybeUninit::new(val);
        }
    }
    // SAFETY: `0..area` elements of `uninit` were initialized above.
    let samples = unsafe { core::slice::from_raw_parts(uninit.as_ptr().cast::<i16>(), area) };
    let mut acc8 = V8::acc_zero();
    let mut idx = 0;
    while idx + 8 <= area {
        acc8 = V8::acc_add(acc8, V8::load(&samples[idx..]));
        idx += 8;
    }
    let mut total = V8::acc_sum(acc8);
    for &rem in &samples[idx..] {
        total += i64::from(rem);
    }
    let avg = ((total + (area / 2) as i64) >> area.trailing_zeros()) as i16;
    let neg = alpha_q3 < 0;
    let q15 = (alpha_q3.abs() as i16) << 9;
    let dc_i16 = dc.to_i32() as i16;
    let max_i16 = depth.max() as i16;
    let (avg_w, q15_w, dc_w, z_w, hi_w) = (
        VWide::splat(avg),
        VWide::splat(q15),
        VWide::splat(dc_i16),
        VWide::splat(0),
        VWide::splat(max_i16),
    );
    let (avg_8, q15_8, dc_8, z_8, hi_8) = (
        V8::splat(avg),
        V8::splat(q15),
        V8::splat(dc_i16),
        V8::splat(0),
        V8::splat(max_i16),
    );
    let (avg_4, q15_4, dc_4, z_4, hi_4) = (
        V4::splat(avg),
        V4::splat(q15),
        V4::splat(dc_i16),
        V4::splat(0),
        V4::splat(max_i16),
    );
    for r in 0..h {
        let row_ac = &samples[r * w..(r + 1) * w];
        let out = &mut dst.row_mut(r)[..w];
        let mut c0 = 0;
        while c0 + VWide::LANES <= w {
            let s = scale_cfl(VWide::load(&row_ac[c0..]), avg_w, q15_w, neg);
            // SAFETY: `c0 + VWide::LANES <= w`.
            unsafe { S::clamp_and_store(dc_w.add(s), z_w, hi_w, &mut out[c0..]) };
            c0 += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && c0 + 8 <= w {
            let s = scale_cfl(V8::load(&row_ac[c0..]), avg_8, q15_8, neg);
            // SAFETY: `c0 + 8 <= w`.
            unsafe { S::clamp_and_store(dc_8.add(s), z_8, hi_8, &mut out[c0..]) };
            c0 += 8;
        }
        if c0 + 4 <= w {
            let s = scale_cfl(V4::load(&row_ac[c0..]), avg_4, q15_4, neg);
            // SAFETY: `c0 + 4 <= w`.
            unsafe { S::clamp_and_store(dc_4.add(s), z_4, hi_4, &mut out[c0..]) };
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! high_intra_isa {
    ($mod_name:ident, $(#[$feature:meta])*, $v4:ty, $v8:ty, $vwide:ty, $run_u8:path) => {
        mod $mod_name {
            use nc_common::{BitDepth, PlaneView, PlaneViewMut};

            use super::{
                BlockEdges, DirectionalKernels, FilterGrid, FilterIntraMode, HighAv1IntraFns,
                IntraEdges, IntraKind, cfl_predict_lanes, filter_patches_u16, paeth_block_lanes,
                run_u16_lanes, smooth_block_lanes, store_transposed_u16,
            };

            struct Dir;

            impl DirectionalKernels<u16> for Dir {
                #[inline(always)]
                fn run(window: &[u16], weight: usize, out: &mut [u16]) {
                    // SAFETY: `Dir` is instantiated only inside `directional` below.
                    unsafe { run_u16_lanes::<$v8, $vwide>(window, weight, out) };
                }

                #[inline(always)]
                fn store_transposed(
                    columns: &[u16],
                    side: usize,
                    dst: &mut PlaneViewMut<'_, u16>,
                ) {
                    // SAFETY: `Dir` is instantiated only inside `directional` below.
                    unsafe { store_transposed_u16::<$v8>(columns, side, dst) };
                }
            }

            $(#[$feature])*
            unsafe fn filter_patches(
                grid: &mut FilterGrid<u16>,
                width: usize,
                height: usize,
                mode: FilterIntraMode,
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { filter_patches_u16::<$v8>(grid, width, height, mode, depth) };
            }

            $(#[$feature])*
            unsafe fn directional(
                edges: &IntraEdges<u16>,
                angle: u16,
                dst: &mut PlaneViewMut<'_, u16>,
            ) {
                edges.predict_angle_with::<Dir>(angle, dst);
            }

            $(#[$feature])*
            unsafe fn run_u16(window: &[u16], weight: usize, out: &mut [u16]) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { run_u16_lanes::<$v8, $vwide>(window, weight, out) };
            }

            $(#[$feature])*
            unsafe fn paeth_u16(edges: &BlockEdges<u16>, dst: &mut PlaneViewMut<'_, u16>) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { paeth_block_lanes::<u16, $v8, $vwide>(edges, dst) };
            }

            $(#[$feature])*
            unsafe fn paeth_u8(edges: &BlockEdges<u8>, dst: &mut PlaneViewMut<'_, u8>) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { paeth_block_lanes::<u8, $v8, $vwide>(edges, dst) };
            }

            $(#[$feature])*
            unsafe fn smooth_u16(
                edges: &BlockEdges<u16>,
                kind: IntraKind,
                dst: &mut PlaneViewMut<'_, u16>,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { smooth_block_lanes::<u16, $v4, $v8, $vwide>(edges, kind, dst) };
            }

            $(#[$feature])*
            unsafe fn smooth_u8(
                edges: &BlockEdges<u8>,
                kind: IntraKind,
                dst: &mut PlaneViewMut<'_, u8>,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { smooth_block_lanes::<u8, $v4, $v8, $vwide>(edges, kind, dst) };
            }

            $(#[$feature])*
            unsafe fn cfl_u16(
                luma: &PlaneView<'_, u16>,
                luma_extent: (usize, usize),
                dc: u16,
                alpha_q3: i32,
                dst: &mut PlaneViewMut<'_, u16>,
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe {
                    cfl_predict_lanes::<u16, $v4, $v8, $vwide>(
                        luma,
                        luma_extent,
                        dc,
                        alpha_q3,
                        dst,
                        depth,
                    );
                }
            }

            $(#[$feature])*
            unsafe fn cfl_u8(
                luma: &PlaneView<'_, u8>,
                luma_extent: (usize, usize),
                dc: u8,
                alpha_q3: i32,
                dst: &mut PlaneViewMut<'_, u8>,
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe {
                    cfl_predict_lanes::<u8, $v4, $v8, $vwide>(
                        luma,
                        luma_extent,
                        dc,
                        alpha_q3,
                        dst,
                        depth,
                    );
                }
            }

            pub(super) const FNS: HighAv1IntraFns = HighAv1IntraFns {
                filter_patches,
                directional,
                run_u16,
                run_u8: $run_u8,
                paeth_u16,
                paeth_u8,
                smooth_u16,
                smooth_u8,
                cfl_u16,
                cfl_u8,
            };
        }
    };
}

#[cfg(target_arch = "x86_64")]
high_intra_isa!(
    x86_avx2,
    #[target_feature(enable = "avx2")],
    crate::simd::avx2::I16x4,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16,
    super::super::avx2::run_u8
);

#[cfg(target_arch = "x86_64")]
high_intra_isa!(
    x86_avx512,
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")],
    crate::simd::avx2::I16x4,
    crate::simd::avx2::I16x8,
    crate::simd::avx512_i16::I16x32,
    super::super::avx2::run_u8
);

#[cfg(target_arch = "aarch64")]
high_intra_isa!(
    neon_high,
    #[target_feature(enable = "neon")],
    crate::simd::neon::I16x4,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8,
    super::super::neon::run_u8
);
