//! High-bit-depth (`u16`, 10 and 12 bpc) AV1 motion-compensation SIMD kernels:
//! unscaled `put`/`prep`, scaled `predict`/`predict_compound`, and 8x8 affine
//! `warp`, across AVX2, AVX-512, NEON, and SIMD128.
//!
//! Attribution: fast paths for `(0, 0)` copy/shift, 1-D horizontal/vertical
//! filtering, 2/4/6/8-tap span specialization, paired `pmaddwd` / NEON
//! widening MACs, and 8x8 warp follow dav1d 1.5.4 (`src/mc_tmpl.c`,
//! `src/x86/mc16_avx2.asm`, `src/x86/mc16_avx512.asm`, `src/arm/64/mc16.S`,
//! BSD-2-Clause; see `THIRD_PARTY.md`).

#![allow(unsafe_code, clippy::inline_always)]
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, unused_imports)
)]

use core::mem::MaybeUninit;

use nc_common::{BitDepth, PlaneView, PlaneViewMut};

use super::convolve::{
    AV1_MAX_PREDICTION_EDGE, Av1McAxis, KERNEL_EDGE, check_prediction, predict_compound_scalar,
    predict_scalar, unscaled_prep_at, unscaled_put_at,
};
use super::filter::{AV1_FILTER_BEFORE, AV1_FILTER_TAPS, Av1Filters, Kernel};
use super::high_blend::{HighBlendFns, SCALAR_BLENDS};
use super::warp::warp_taps_i16;
use super::warp_kernels::{
    WARP_BLOCK, WINDOW_SIDE, WINDOW_STRIDE, WarpOut, WarpPhases, warp8x8_scalar_at,
};
use super::{
    FILTER_BITS, INTER_ROUND1_COMPOUND, av1_compound_bias, inter_post_round, inter_round0,
    inter_round1_final, round2,
};
use crate::Isa;
use crate::high_depth::{Av1Prep16Fn, Av1Put16Fn, Av1Taps, av1_widths};
use crate::simd::{I16FilterLanes, I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes};

const IDENTITY_TAPS: [i16; AV1_FILTER_TAPS] = [0, 0, 0, 128, 0, 0, 0, 0];
const FILTER_CENTER: usize = 4;
const INLINE_SCALED_SAMPLES: usize = 128 * 64;

/// Scaled `u16` single-reference prediction kernel.
pub(crate) type ScaledPut16Fn = unsafe fn(
    filters: Av1Filters,
    window: &PlaneView<'_, u16>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_, u16>,
    depth: BitDepth,
);

/// Scaled `u16` compound-precision prediction kernel.
pub(crate) type ScaledPrep16Fn = unsafe fn(
    filters: Av1Filters,
    window: &PlaneView<'_, u16>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: (&mut [i16], usize),
    depth: BitDepth,
);

/// High-bit-depth 8x8 warp kernel (`rounds = (InterRound0, InterRound1)`).
pub(crate) type Warp16Fn =
    unsafe fn(window: (&[u16], usize), phases: &WarpPhases, rounds: (u32, u32), out: &mut WarpOut);

/// All `u16` AV1 motion-compensation and blend kernels for an ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HighAv1McFns {
    pub put: [Av1Put16Fn; 5],
    pub prep: [Av1Prep16Fn; 5],
    pub scaled_put: ScaledPut16Fn,
    pub scaled_prep: ScaledPrep16Fn,
    pub warp8x8: Warp16Fn,
    pub blend: HighBlendFns,
}

impl HighAv1McFns {
    pub const SCALAR: Self = Self {
        put: av1_widths!(scalar_put),
        prep: av1_widths!(scalar_prep),
        scaled_put: scalar_scaled_put,
        scaled_prep: scalar_scaled_prep,
        warp8x8: warp8x8_scalar_at::<u16>,
        blend: SCALAR_BLENDS,
    };

    pub fn for_isa(isa: Isa) -> Self {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86::AVX2_FNS,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86::AVX512_FNS,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon::NEON_FNS,
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            Isa::Sve2 => neon::SVE2_FNS,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => simd128::SIMD128_FNS,
            _ => Self::SCALAR,
        }
    }
}

fn scalar_put<const W: usize>(
    src: (&[u16], usize),
    dst: (&mut [u16], usize),
    h: usize,
    taps: Av1Taps<'_>,
    depth: BitDepth,
) {
    unscaled_put_at::<u16, W>(src, dst, h, taps, depth);
}

fn scalar_prep<const W: usize>(
    src: (&[u16], usize),
    dst: (&mut [i16], usize),
    h: usize,
    taps: Av1Taps<'_>,
    depth: BitDepth,
) {
    unscaled_prep_at::<u16, W>(src, dst, h, taps, depth);
}

fn scalar_scaled_put(
    filters: Av1Filters,
    window: &PlaneView<'_, u16>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_, u16>,
    depth: BitDepth,
) {
    predict_scalar(filters, window, size, axes, dst, depth);
}

fn scalar_scaled_prep(
    filters: Av1Filters,
    window: &PlaneView<'_, u16>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    (dst, stride): (&mut [i16], usize),
    depth: BitDepth,
) {
    predict_compound_scalar(filters, window, size, axes, dst, stride, depth);
}

/// Number of symmetric tap pairs around the filter centre (`1`..=`4`, i.e. 2..=8 taps);
/// AV1 and VP9 (`crate::vp9_mc_high`) share the 8-tap layout.
#[inline(always)]
pub(crate) fn tap_pair_count(taps: &[i16; AV1_FILTER_TAPS]) -> usize {
    if taps[0] == 0 && taps[7] == 0 {
        if taps[1] == 0 && taps[6] == 0 {
            if taps[2] == 0 && taps[5] == 0 {
                return 1;
            }
            return 2;
        }
        return 3;
    }
    4
}

#[inline(always)]
pub(crate) fn tap_pairs<const PAIRS: usize>(taps: &[i16; AV1_FILTER_TAPS]) -> [(i16, i16); PAIRS] {
    let first = FILTER_CENTER - PAIRS;
    let mut out = [(0i16, 0i16); PAIRS];
    let mut p = 0;
    while p < PAIRS {
        let t = first + 2 * p;
        out[p] = (taps[t], taps[t + 1]);
        p += 1;
    }
    out
}

#[inline(always)]
pub(crate) unsafe fn load_i16<V: I16Lanes>(ptr: *const u16) -> V {
    // SAFETY: caller validated `V::LANES` readable `u16` words (`0..=4095`).
    V::load(unsafe { core::slice::from_raw_parts(ptr.cast::<i16>(), V::LANES) })
}

#[inline(always)]
pub(crate) unsafe fn store_u16<V: I16Lanes>(v: V, ptr: *mut u16) {
    // SAFETY: caller validated `V::LANES` writable `u16` words.
    v.store(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<i16>(), V::LANES) });
}

#[inline(always)]
unsafe fn store_i16<V: I16Lanes>(v: V, ptr: *mut i16) {
    // SAFETY: caller validated `V::LANES` writable `i16` words.
    v.store(unsafe { core::slice::from_raw_parts_mut(ptr, V::LANES) });
}

#[inline(always)]
unsafe fn load_halves_u16<V: I16FilterLanes>(lo: *const u16, hi: *const u16) -> V {
    let half = V::LANES / 2;
    // SAFETY: caller validated `half` readable `u16` words (`0..=4095`) at both pointers.
    unsafe {
        V::load_halves(
            core::slice::from_raw_parts(lo.cast::<i16>(), half),
            core::slice::from_raw_parts(hi.cast::<i16>(), half),
        )
    }
}

#[inline(always)]
unsafe fn store_halves_u16<V: I16FilterLanes>(v: V, lo: *mut u16, hi: *mut u16) {
    let half = V::LANES / 2;
    // SAFETY: caller validated `half` writable `u16` words at both pointers.
    unsafe {
        v.store_halves(
            core::slice::from_raw_parts_mut(lo.cast::<i16>(), half),
            core::slice::from_raw_parts_mut(hi.cast::<i16>(), half),
        );
    }
}

#[inline(always)]
unsafe fn store_halves_i16<V: I16FilterLanes>(v: V, lo: *mut i16, hi: *mut i16) {
    let half = V::LANES / 2;
    // SAFETY: caller validated `half` writable `i16` words at both pointers.
    unsafe {
        v.store_halves(
            core::slice::from_raw_parts_mut(lo, half),
            core::slice::from_raw_parts_mut(hi, half),
        );
    }
}

#[inline(always)]
pub(crate) unsafe fn h_dot_lanes<V: I16Lanes, const PAIRS: usize>(
    row: *const u16,
    fx: [(i16, i16); PAIRS],
) -> V::Acc {
    let first = FILTER_CENTER - PAIRS;
    let mut acc = V::acc_zero();
    let mut p = 0;
    while p < PAIRS {
        let t = first + 2 * p;
        let (c0, c1) = fx[p];
        // SAFETY: `row` covers `V::LANES + 7` words inside the validated window.
        let (a, b) = unsafe { (load_i16::<V>(row.add(t)), load_i16::<V>(row.add(t + 1))) };
        acc = V::acc_mul_add2(acc, a, c0, b, c1);
        p += 1;
    }
    acc
}

/// Vertical counterpart of [`h_dot_lanes`] down `PAIRS` tap pairs of a column
/// with stride `ss`. VP9's one-pass vertical filter uses it; AV1's passes slide
/// rows through registers instead (`v_strip_1`..`v_strip_4`).
#[inline(always)]
pub(crate) unsafe fn v_dot_lanes<V: I16Lanes, const PAIRS: usize>(
    col: *const u16,
    ss: usize,
    fy: [(i16, i16); PAIRS],
) -> V::Acc {
    let first = FILTER_CENTER - PAIRS;
    let mut acc = V::acc_zero();
    let mut p = 0;
    while p < PAIRS {
        let t = first + 2 * p;
        let (c0, c1) = fy[p];
        // SAFETY: `col` covers the 8 rows of the tap window inside the validated window.
        let (a, b) = unsafe {
            (
                load_i16::<V>(col.add(t * ss)),
                load_i16::<V>(col.add((t + 1) * ss)),
            )
        };
        acc = V::acc_mul_add2(acc, a, c0, b, c1);
        p += 1;
    }
    acc
}

#[inline(always)]
unsafe fn h_dot_halves<V: I16FilterLanes, const PAIRS: usize>(
    row_a: *const u16,
    row_b: *const u16,
    fx: [(i16, i16); PAIRS],
) -> V::Acc {
    let first = FILTER_CENTER - PAIRS;
    let mut acc = V::acc_zero();
    let mut p = 0;
    while p < PAIRS {
        let t = first + 2 * p;
        let (c0, c1) = fx[p];
        // SAFETY: each row covers `V::LANES / 2 + 7` words inside the validated window.
        let (a, b) = unsafe {
            (
                load_halves_u16::<V>(row_a.add(t), row_b.add(t)),
                load_halves_u16::<V>(row_a.add(t + 1), row_b.add(t + 1)),
            )
        };
        acc = V::acc_mul_add2(acc, a, c0, b, c1);
        p += 1;
    }
    acc
}

#[inline(always)]
fn splat_fx_pairs<V: I16Pairs, const PAIRS: usize>(fx: [(i16, i16); PAIRS]) -> [V; PAIRS] {
    let z = V::splat(0);
    let mut out = [z; PAIRS];
    let mut p = 0;
    while p < PAIRS {
        out[p] = V::splat_pair(fx[p].0, fx[p].1);
        p += 1;
    }
    out
}

#[inline(always)]
unsafe fn h_convolve_lanes<V: I16Pairs, const PAIRS: usize>(
    row: *const u16,
    fx: [(i16, i16); PAIRS],
    fx_vec: [V; PAIRS],
    round: i32,
    shift: u32,
) -> V {
    if PAIRS == 4 {
        let taps = [
            fx[0].0, fx[0].1, fx[1].0, fx[1].1, fx[2].0, fx[2].1, fx[3].0, fx[3].1,
        ];
        // SAFETY: the caller validated this horizontal row and enables V's ISA.
        if let Some(out) = unsafe { V::horizontal_dot_8tap(row.cast(), taps, round, shift) } {
            return out;
        }
    }
    if !V::FAST_PAIR_DOT {
        // SAFETY: `row` covers `V::LANES + 7` words inside the validated window.
        let acc = unsafe { h_dot_lanes::<V, PAIRS>(row, fx) };
        return V::acc_round_shift_narrow(acc, round, shift);
    }
    let first = FILTER_CENTER - PAIRS;
    // SAFETY: `first + 1 < 8` and `row` covers `V::LANES + 7` words.
    let (mut even, mut odd) = unsafe {
        (
            load_i16::<V>(row.add(first)).pair_dot(fx_vec[0]),
            load_i16::<V>(row.add(first + 1)).pair_dot(fx_vec[0]),
        )
    };
    let mut p = 1;
    while p < PAIRS {
        let t = first + 2 * p;
        // SAFETY: `t + 1 <= 7`, so `row + t + 1` reads up to `V::LANES + 7` words.
        unsafe {
            even = even.add(load_i16::<V>(row.add(t)).pair_dot(fx_vec[p]));
            odd = odd.add(load_i16::<V>(row.add(t + 1)).pair_dot(fx_vec[p]));
        }
        p += 1;
    }
    let r = <V::Pairs as I32Lanes>::splat(round);
    V::interleave_pairs(even.add(r).sra(shift), odd.add(r).sra(shift))
}

#[inline(always)]
unsafe fn h_convolve_halves<V: I16FilterLanes + I16Pairs, const PAIRS: usize>(
    row_a: *const u16,
    row_b: *const u16,
    fx: [(i16, i16); PAIRS],
    fx_vec: [V; PAIRS],
    round: i32,
    shift: u32,
) -> V {
    if !V::FAST_PAIR_DOT {
        // SAFETY: each row covers `V::LANES / 2 + 7` words inside the validated window.
        let acc = unsafe { h_dot_halves::<V, PAIRS>(row_a, row_b, fx) };
        return V::acc_round_shift_narrow(acc, round, shift);
    }
    let first = FILTER_CENTER - PAIRS;
    // SAFETY: `first + 1 < 8` and each row covers `V::LANES / 2 + 7` words.
    let (mut even, mut odd) = unsafe {
        (
            load_halves_u16::<V>(row_a.add(first), row_b.add(first)).pair_dot(fx_vec[0]),
            load_halves_u16::<V>(row_a.add(first + 1), row_b.add(first + 1)).pair_dot(fx_vec[0]),
        )
    };
    let mut p = 1;
    while p < PAIRS {
        let t = first + 2 * p;
        // SAFETY: `t + 1 <= 7`, reading up to `V::LANES / 2 + 7` words per row.
        unsafe {
            even = even.add(load_halves_u16::<V>(row_a.add(t), row_b.add(t)).pair_dot(fx_vec[p]));
            odd = odd
                .add(load_halves_u16::<V>(row_a.add(t + 1), row_b.add(t + 1)).pair_dot(fx_vec[p]));
        }
        p += 1;
    }
    let r = <V::Pairs as I32Lanes>::splat(round);
    V::interleave_pairs(even.add(r).sra(shift), odd.add(r).sra(shift))
}

trait McSink<V: I16Lanes>: Copy {
    fn round_shift(self) -> (i32, u32);
    unsafe fn write_narrow(self, out: V, r: usize, c0: usize);
    #[inline(always)]
    unsafe fn write_vec(self, acc: V::Acc, r: usize, c0: usize) {
        let (round, shift) = self.round_shift();
        // SAFETY: forwarded from caller.
        unsafe { self.write_narrow(V::acc_round_shift_narrow(acc, round, shift), r, c0) };
    }
}

#[derive(Clone, Copy)]
struct PutSink<V, const SHIFT: u32> {
    dst_p: *mut u16,
    ds: usize,
    round: i32,
    zero: V,
    max_v: V,
}

impl<V: I16Lanes, const SHIFT: u32> McSink<V> for PutSink<V, SHIFT> {
    #[inline(always)]
    fn round_shift(self) -> (i32, u32) {
        (self.round, SHIFT)
    }

    #[inline(always)]
    unsafe fn write_narrow(self, out: V, r: usize, c0: usize) {
        let clamped = out.max(self.zero).min(self.max_v);
        // SAFETY: caller validated `dst` covers `r * ds + c0 + V::LANES`.
        unsafe { store_u16(clamped, self.dst_p.add(r * self.ds + c0)) };
    }
}

#[derive(Clone, Copy)]
struct PrepSink<const SHIFT: u32> {
    dst_p: *mut i16,
    ds: usize,
    round: i32,
}

impl<V: I16Lanes, const SHIFT: u32> McSink<V> for PrepSink<SHIFT> {
    #[inline(always)]
    fn round_shift(self) -> (i32, u32) {
        (self.round, SHIFT)
    }

    #[inline(always)]
    unsafe fn write_narrow(self, out: V, r: usize, c0: usize) {
        // SAFETY: caller validated `dst` covers `r * ds + c0 + V::LANES`.
        unsafe { store_i16(out, self.dst_p.add(r * self.ds + c0)) };
    }
}

trait RowSource<V: I16Lanes>: Copy {
    unsafe fn row(self, y: usize, c0: usize) -> V;
}

#[derive(Clone, Copy)]
struct DirectRow {
    src_p: *const u16,
    ss: usize,
}

impl<V: I16Lanes> RowSource<V> for DirectRow {
    #[inline(always)]
    unsafe fn row(self, y: usize, c0: usize) -> V {
        // SAFETY: `y < h + 7` and `c0 + 3 + V::LANES <= W + 3 <= W + 7`.
        unsafe { load_i16::<V>(self.src_p.add(y * self.ss + c0 + AV1_FILTER_BEFORE)) }
    }
}

#[derive(Clone, Copy)]
struct HorizRow<V, const HP: usize, const R0: u32> {
    src_p: *const u16,
    ss: usize,
    fx: [(i16, i16); HP],
    fx_vec: [V; HP],
}

impl<V: I16Pairs, const HP: usize, const R0: u32> RowSource<V> for HorizRow<V, HP, R0> {
    #[inline(always)]
    unsafe fn row(self, y: usize, c0: usize) -> V {
        // SAFETY: `y < h + 7` and `c0 + V::LANES + 7 <= W + 7`.
        unsafe {
            h_convolve_lanes::<V, HP>(
                self.src_p.add(y * self.ss + c0),
                self.fx,
                self.fx_vec,
                1 << (R0 - 1),
                R0,
            )
        }
    }
}

#[inline(always)]
unsafe fn h_pass_lanes<V: I16Pairs, S: McSink<V>, const W: usize, const PAIRS: usize>(
    src_p: *const u16,
    ss: usize,
    fx: &[i16; AV1_FILTER_TAPS],
    h: usize,
    sink: S,
) {
    let pairs = tap_pairs::<PAIRS>(fx);
    let fx_vec = splat_fx_pairs::<V, PAIRS>(pairs);
    let (round, shift) = sink.round_shift();
    for r in 0..h {
        // SAFETY: `r + 3 < h + 7`, inside the validated window.
        let row = unsafe { src_p.add((r + AV1_FILTER_BEFORE) * ss) };
        for c0 in (0..W).step_by(V::LANES) {
            // SAFETY: `c0 + V::LANES + 7 <= W + 7` and `dst` covers `W`.
            unsafe {
                let out = h_convolve_lanes::<V, PAIRS>(row.add(c0), pairs, fx_vec, round, shift);
                sink.write_narrow(out, r, c0);
            }
        }
    }
}

#[inline(always)]
unsafe fn v_strip_1<V: I16Lanes, R: RowSource<V>, S: McSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 1],
    h: usize,
    c0: usize,
) {
    let (f0, f1) = fy[0];
    // SAFETY: reads rows `3..h + 4` inside `0..h + 7`.
    unsafe {
        let mut s0 = rows.row(3, c0);
        for r in 0..h {
            let s1 = rows.row(r + 4, c0);
            let acc = V::acc_mul_add2(V::acc_zero(), s0, f0, s1, f1);
            sink.write_vec(acc, r, c0);
            s0 = s1;
        }
    }
}

#[inline(always)]
unsafe fn v_strip_2<V: I16Lanes, R: RowSource<V>, S: McSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 2],
    h: usize,
    c0: usize,
) {
    let ((f0, f1), (f2, f3)) = (fy[0], fy[1]);
    // SAFETY: reads rows `2..h + 5` inside `0..h + 7`.
    unsafe {
        let (mut s0, mut s1, mut s2) = (rows.row(2, c0), rows.row(3, c0), rows.row(4, c0));
        for r in 0..h {
            let s3 = rows.row(r + 5, c0);
            let acc = V::acc_mul_add2(V::acc_zero(), s0, f0, s1, f1);
            let acc = V::acc_mul_add2(acc, s2, f2, s3, f3);
            sink.write_vec(acc, r, c0);
            (s0, s1, s2) = (s1, s2, s3);
        }
    }
}

#[inline(always)]
unsafe fn v_strip_3<V: I16Lanes, R: RowSource<V>, S: McSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 3],
    h: usize,
    c0: usize,
) {
    let ((f0, f1), (f2, f3), (f4, f5)) = (fy[0], fy[1], fy[2]);
    // SAFETY: reads rows `1..h + 6` inside `0..h + 7`.
    unsafe {
        let (mut s0, mut s1, mut s2, mut s3, mut s4) = (
            rows.row(1, c0),
            rows.row(2, c0),
            rows.row(3, c0),
            rows.row(4, c0),
            rows.row(5, c0),
        );
        for r in 0..h {
            let s5 = rows.row(r + 6, c0);
            let acc = V::acc_mul_add2(V::acc_zero(), s0, f0, s1, f1);
            let acc = V::acc_mul_add2(acc, s2, f2, s3, f3);
            let acc = V::acc_mul_add2(acc, s4, f4, s5, f5);
            sink.write_vec(acc, r, c0);
            (s0, s1, s2, s3, s4) = (s1, s2, s3, s4, s5);
        }
    }
}

#[inline(always)]
unsafe fn v_strip_4<V: I16Lanes, R: RowSource<V>, S: McSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 4],
    h: usize,
    c0: usize,
) {
    let ((f0, f1), (f2, f3), (f4, f5), (f6, f7)) = (fy[0], fy[1], fy[2], fy[3]);
    // SAFETY: reads rows `0..h + 7` inside `0..h + 7`.
    unsafe {
        let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6) = (
            rows.row(0, c0),
            rows.row(1, c0),
            rows.row(2, c0),
            rows.row(3, c0),
            rows.row(4, c0),
            rows.row(5, c0),
            rows.row(6, c0),
        );
        for r in 0..h {
            let s7 = rows.row(r + 7, c0);
            let acc = V::acc_mul_add2(V::acc_zero(), s0, f0, s1, f1);
            let acc = V::acc_mul_add2(acc, s2, f2, s3, f3);
            let acc = V::acc_mul_add2(acc, s4, f4, s5, f5);
            let acc = V::acc_mul_add2(acc, s6, f6, s7, f7);
            sink.write_vec(acc, r, c0);
            (s0, s1, s2, s3, s4, s5, s6) = (s1, s2, s3, s4, s5, s6, s7);
        }
    }
}

#[inline(always)]
unsafe fn v_pass_lanes<V: I16Lanes, R: RowSource<V>, S: McSink<V>, const W: usize>(
    rows: R,
    sink: S,
    fy: &[i16; AV1_FILTER_TAPS],
    h: usize,
) {
    match tap_pair_count(fy) {
        1 => {
            let pairs = tap_pairs::<1>(fy);
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: validated window and destination cover strip `c0..c0 + V::LANES`.
                unsafe { v_strip_1(rows, sink, pairs, h, c0) };
            }
        }
        2 => {
            let pairs = tap_pairs::<2>(fy);
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: validated window and destination cover strip `c0..c0 + V::LANES`.
                unsafe { v_strip_2(rows, sink, pairs, h, c0) };
            }
        }
        3 => {
            let pairs = tap_pairs::<3>(fy);
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: validated window and destination cover strip `c0..c0 + V::LANES`.
                unsafe { v_strip_3(rows, sink, pairs, h, c0) };
            }
        }
        _ => {
            let pairs = tap_pairs::<4>(fy);
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: validated window and destination cover strip `c0..c0 + V::LANES`.
                unsafe { v_strip_4(rows, sink, pairs, h, c0) };
            }
        }
    }
}

#[inline(always)]
unsafe fn hv_pass_lanes<V: I16Pairs, S: McSink<V>, const W: usize, const R0: u32>(
    src_p: *const u16,
    ss: usize,
    (fx, fy): Av1Taps<'_>,
    h: usize,
    sink: S,
) {
    match tap_pair_count(fx) {
        1 => {
            let pairs = tap_pairs::<1>(fx);
            let hr = HorizRow::<V, 1, R0> {
                src_p,
                ss,
                fx: pairs,
                fx_vec: splat_fx_pairs::<V, 1>(pairs),
            };
            // SAFETY: window and destination validated by caller.
            unsafe { v_pass_lanes::<V, _, _, W>(hr, sink, fy, h) };
        }
        2 => {
            let pairs = tap_pairs::<2>(fx);
            let hr = HorizRow::<V, 2, R0> {
                src_p,
                ss,
                fx: pairs,
                fx_vec: splat_fx_pairs::<V, 2>(pairs),
            };
            // SAFETY: window and destination validated by caller.
            unsafe { v_pass_lanes::<V, _, _, W>(hr, sink, fy, h) };
        }
        3 => {
            let pairs = tap_pairs::<3>(fx);
            let hr = HorizRow::<V, 3, R0> {
                src_p,
                ss,
                fx: pairs,
                fx_vec: splat_fx_pairs::<V, 3>(pairs),
            };
            // SAFETY: window and destination validated by caller.
            unsafe { v_pass_lanes::<V, _, _, W>(hr, sink, fy, h) };
        }
        _ => {
            let pairs = tap_pairs::<4>(fx);
            let hr = HorizRow::<V, 4, R0> {
                src_p,
                ss,
                fx: pairs,
                fx_vec: splat_fx_pairs::<V, 4>(pairs),
            };
            // SAFETY: window and destination validated by caller.
            unsafe { v_pass_lanes::<V, _, _, W>(hr, sink, fy, h) };
        }
    }
}

/// Unscaled `put` (`u16` output, clipped to `depth`) over `V: I16Pairs` (`W >= V::LANES`).
#[inline(always)]
pub(crate) unsafe fn unscaled_put_lanes<V: I16Pairs, const W: usize, const ICL: bool>(
    (src, ss): (&[u16], usize),
    (dst, ds): (&mut [u16], usize),
    h: usize,
    (fx, fy): Av1Taps<'_>,
    depth: BitDepth,
) {
    assert!(
        h > 0
            && h <= KERNEL_EDGE
            && src.len() >= (h + AV1_FILTER_TAPS - 2) * ss + W + AV1_FILTER_TAPS - 1
            && dst.len() >= (h - 1) * ds + W,
        "AV1 u16 put window or destination too small"
    );
    debug_assert!(W.is_multiple_of(V::LANES));
    let (src_p, dst_p) = (src.as_ptr(), dst.as_mut_ptr());
    let (hx_id, vy_id) = (*fx == IDENTITY_TAPS, *fy == IDENTITY_TAPS);
    if hx_id && vy_id {
        for r in 0..h {
            // SAFETY: validated window and destination cover `W` words per row.
            let row = unsafe { src_p.add((r + AV1_FILTER_BEFORE) * ss + AV1_FILTER_BEFORE) };
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: `c0 + V::LANES <= W`.
                unsafe { store_u16(load_i16::<V>(row.add(c0)), dst_p.add(r * ds + c0)) };
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if ICL
        // SAFETY: the entry validated the padded source and output.
        && unsafe {
            try_icl::<W, true>(
                (src_p, ss), (dst_p.cast(), ds), h, (fx, fy), (hx_id, vy_id), depth,
            )
        }
    {
        return;
    }
    let (zero, max_v) = (V::splat(0), V::splat(depth.max() as i16));
    let r0 = inter_round0(depth);
    if vy_id {
        let sink = PutSink::<V, FILTER_BITS> {
            dst_p,
            ds,
            round: (1 << (FILTER_BITS - 1)) + (1 << (r0 - 1)),
            zero,
            max_v,
        };
        // SAFETY: window and destination validated above.
        unsafe {
            match tap_pair_count(fx) {
                1 => h_pass_lanes::<V, _, W, 1>(src_p, ss, fx, h, sink),
                2 => h_pass_lanes::<V, _, W, 2>(src_p, ss, fx, h, sink),
                3 => h_pass_lanes::<V, _, W, 3>(src_p, ss, fx, h, sink),
                _ => h_pass_lanes::<V, _, W, 4>(src_p, ss, fx, h, sink),
            }
        }
        return;
    }
    if hx_id {
        let sink = PutSink::<V, FILTER_BITS> {
            dst_p,
            ds,
            round: 1 << (FILTER_BITS - 1),
            zero,
            max_v,
        };
        // SAFETY: window and destination validated above.
        unsafe { v_pass_lanes::<V, _, _, W>(DirectRow { src_p, ss }, sink, fy, h) };
        return;
    }

    // SAFETY: window and destination validated above.
    unsafe {
        if depth.bits() == 12 {
            let sink = PutSink::<V, 9> {
                dst_p,
                ds,
                round: 1 << 8,
                zero,
                max_v,
            };
            hv_pass_lanes::<V, _, W, 5>(src_p, ss, (fx, fy), h, sink);
        } else {
            let sink = PutSink::<V, 11> {
                dst_p,
                ds,
                round: 1 << 10,
                zero,
                max_v,
            };
            hv_pass_lanes::<V, _, W, 3>(src_p, ss, (fx, fy), h, sink);
        }
    }
}

#[inline(always)]
unsafe fn h_dispatch_lanes<V: I16Pairs, S: McSink<V>, const W: usize>(
    src_p: *const u16,
    ss: usize,
    fx: &[i16; AV1_FILTER_TAPS],
    h: usize,
    sink: S,
) {
    // SAFETY: forwarded from caller.
    unsafe {
        match tap_pair_count(fx) {
            1 => h_pass_lanes::<V, _, W, 1>(src_p, ss, fx, h, sink),
            2 => h_pass_lanes::<V, _, W, 2>(src_p, ss, fx, h, sink),
            3 => h_pass_lanes::<V, _, W, 3>(src_p, ss, fx, h, sink),
            _ => h_pass_lanes::<V, _, W, 4>(src_p, ss, fx, h, sink),
        }
    }
}

/// Unscaled `prep` (`i16` compound output less bias) over `V: I16Pairs` (`W >= V::LANES`).
#[inline(always)]
pub(crate) unsafe fn unscaled_prep_lanes<V: I16Pairs, const W: usize, const ICL: bool>(
    (src, ss): (&[u16], usize),
    (dst, ds): (&mut [i16], usize),
    h: usize,
    (fx, fy): Av1Taps<'_>,
    depth: BitDepth,
) {
    assert!(
        h > 0
            && h <= KERNEL_EDGE
            && src.len() >= (h + AV1_FILTER_TAPS - 2) * ss + W + AV1_FILTER_TAPS - 1
            && dst.len() >= (h - 1) * ds + W,
        "AV1 u16 prep window or destination too small"
    );
    debug_assert!(W.is_multiple_of(V::LANES));
    let (src_p, dst_p) = (src.as_ptr(), dst.as_mut_ptr());
    let bias = av1_compound_bias(depth);
    let post = inter_post_round(depth);
    let (hx_id, vy_id) = (*fx == IDENTITY_TAPS, *fy == IDENTITY_TAPS);
    if hx_id && vy_id {
        let (scale_v, bias_v) = (V::splat(1 << post), V::splat(bias as i16));
        for r in 0..h {
            // SAFETY: validated window and destination cover `W` words per row.
            let row = unsafe { src_p.add((r + AV1_FILTER_BEFORE) * ss + AV1_FILTER_BEFORE) };
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: `c0 + V::LANES <= W`.
                unsafe {
                    let v = load_i16::<V>(row.add(c0)).mul(scale_v).sub(bias_v);
                    store_i16(v, dst_p.add(r * ds + c0));
                }
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if ICL
        // SAFETY: the entry validated the padded source and output.
        && unsafe {
            try_icl::<W, false>(
                (src_p, ss), (dst_p.cast(), ds), h, (fx, fy), (hx_id, vy_id), depth,
            )
        }
    {
        return;
    }
    let r0 = inter_round0(depth);
    let round = (1 << (r0 - 1)) - (bias << r0);
    if vy_id {
        // SAFETY: window and destination validated above.
        unsafe {
            if depth.bits() == 12 {
                h_dispatch_lanes::<V, _, W>(src_p, ss, fx, h, PrepSink::<5> { dst_p, ds, round });
            } else {
                h_dispatch_lanes::<V, _, W>(src_p, ss, fx, h, PrepSink::<3> { dst_p, ds, round });
            }
        }
        return;
    }
    if hx_id {
        // SAFETY: window and destination validated above.
        unsafe {
            if depth.bits() == 12 {
                let s = PrepSink::<5> { dst_p, ds, round };
                v_pass_lanes::<V, _, _, W>(DirectRow { src_p, ss }, s, fy, h);
            } else {
                let s = PrepSink::<3> { dst_p, ds, round };
                v_pass_lanes::<V, _, _, W>(DirectRow { src_p, ss }, s, fy, h);
            }
        }
        return;
    }

    let sink_2d = PrepSink::<INTER_ROUND1_COMPOUND> {
        dst_p,
        ds,
        round: (1 << (INTER_ROUND1_COMPOUND - 1)) - (bias << INTER_ROUND1_COMPOUND),
    };
    // SAFETY: window and destination validated above.
    unsafe {
        if depth.bits() == 12 {
            hv_pass_lanes::<V, _, W, 5>(src_p, ss, (fx, fy), h, sink_2d);
        } else {
            hv_pass_lanes::<V, _, W, 3>(src_p, ss, (fx, fy), h, sink_2d);
        }
    }
}

// ----------------------------------------- 2-rows-per-vector (`2 * W == LANES`) --

trait McHalvesSink<V: I16FilterLanes>: Copy {
    fn round_shift(self) -> (i32, u32);
    unsafe fn write_pair_narrow(self, out: V, r: usize);
    #[inline(always)]
    unsafe fn write_pair(self, acc: V::Acc, r: usize) {
        let (round, shift) = self.round_shift();
        // SAFETY: forwarded from caller.
        unsafe { self.write_pair_narrow(V::acc_round_shift_narrow(acc, round, shift), r) };
    }
}

#[derive(Clone, Copy)]
struct PutHalvesSink<V, const W: usize, const SHIFT: u32> {
    dst_p: *mut u16,
    ds: usize,
    round: i32,
    zero: V,
    max_v: V,
}

impl<V: I16FilterLanes, const W: usize, const SHIFT: u32> McHalvesSink<V>
    for PutHalvesSink<V, W, SHIFT>
{
    #[inline(always)]
    fn round_shift(self) -> (i32, u32) {
        (self.round, SHIFT)
    }

    #[inline(always)]
    unsafe fn write_pair_narrow(self, out: V, r: usize) {
        let clamped = out.max(self.zero).min(self.max_v);
        // SAFETY: caller validated `dst` covers rows `r` and `r + 1` of width `W`.
        unsafe {
            if self.ds == W {
                store_u16(clamped, self.dst_p.add(r * W));
            } else {
                store_halves_u16(
                    clamped,
                    self.dst_p.add(r * self.ds),
                    self.dst_p.add((r + 1) * self.ds),
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct PrepHalvesSink<const W: usize, const SHIFT: u32> {
    dst_p: *mut i16,
    ds: usize,
    round: i32,
}

impl<V: I16FilterLanes, const W: usize, const SHIFT: u32> McHalvesSink<V>
    for PrepHalvesSink<W, SHIFT>
{
    #[inline(always)]
    fn round_shift(self) -> (i32, u32) {
        (self.round, SHIFT)
    }

    #[inline(always)]
    unsafe fn write_pair_narrow(self, out: V, r: usize) {
        // SAFETY: caller validated `dst` covers rows `r` and `r + 1` of width `W`.
        unsafe {
            if self.ds == W {
                store_i16(out, self.dst_p.add(r * W));
            } else {
                store_halves_i16(
                    out,
                    self.dst_p.add(r * self.ds),
                    self.dst_p.add((r + 1) * self.ds),
                );
            }
        }
    }
}

trait PairRowSource<V: I16FilterLanes>: Copy {
    unsafe fn pair(self, ya: usize, yb: usize) -> V;
}

#[derive(Clone, Copy)]
struct DirectPairRow {
    src_p: *const u16,
    ss: usize,
}

impl<V: I16FilterLanes> PairRowSource<V> for DirectPairRow {
    #[inline(always)]
    unsafe fn pair(self, ya: usize, yb: usize) -> V {
        // SAFETY: `ya, yb < h + 7` and `3 + V::LANES / 2 <= W + 7`.
        unsafe {
            load_halves_u16::<V>(
                self.src_p.add(ya * self.ss + AV1_FILTER_BEFORE),
                self.src_p.add(yb * self.ss + AV1_FILTER_BEFORE),
            )
        }
    }
}

#[derive(Clone, Copy)]
struct HorizPairRow<V, const HP: usize, const R0: u32> {
    src_p: *const u16,
    ss: usize,
    fx: [(i16, i16); HP],
    fx_vec: [V; HP],
}

impl<V: I16FilterLanes + I16Pairs, const HP: usize, const R0: u32> PairRowSource<V>
    for HorizPairRow<V, HP, R0>
{
    #[inline(always)]
    unsafe fn pair(self, ya: usize, yb: usize) -> V {
        // SAFETY: `ya, yb < h + 7` and `V::LANES / 2 + 7 == W + 7`.
        unsafe {
            h_convolve_halves::<V, HP>(
                self.src_p.add(ya * self.ss),
                self.src_p.add(yb * self.ss),
                self.fx,
                self.fx_vec,
                1 << (R0 - 1),
                R0,
            )
        }
    }
}

#[inline(always)]
unsafe fn h_pass_halves<V: I16FilterLanes + I16Pairs, S: McHalvesSink<V>, const PAIRS: usize>(
    src_p: *const u16,
    ss: usize,
    fx: &[i16; AV1_FILTER_TAPS],
    h: usize,
    sink: S,
) {
    let pairs = tap_pairs::<PAIRS>(fx);
    let fx_vec = splat_fx_pairs::<V, PAIRS>(pairs);
    let (round, shift) = sink.round_shift();
    for r in (0..h).step_by(2) {
        // SAFETY: `r + 4 <= h + 2 < h + 7`, inside the validated window.
        unsafe {
            let row_a = src_p.add((r + AV1_FILTER_BEFORE) * ss);
            let row_b = row_a.add(ss);
            let out = h_convolve_halves::<V, PAIRS>(row_a, row_b, pairs, fx_vec, round, shift);
            sink.write_pair_narrow(out, r);
        }
    }
}

#[inline(always)]
unsafe fn v_halves_1<V: I16FilterLanes, R: PairRowSource<V>, S: McHalvesSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 1],
    h: usize,
) {
    let (f0, f1) = fy[0];
    // SAFETY: reads rows `3..=h + 3` inside `0..h + 7`.
    unsafe {
        let mut prev = rows.pair(3, 3);
        for r in (0..h).step_by(2) {
            let new = rows.pair(r + 4, r + 5);
            let u0 = prev.combine_hi_lo(new);
            sink.write_pair(V::acc_mul_add2(V::acc_zero(), u0, f0, new, f1), r);
            prev = new;
        }
    }
}

#[inline(always)]
unsafe fn v_halves_2<V: I16FilterLanes, R: PairRowSource<V>, S: McHalvesSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 2],
    h: usize,
) {
    let ((f0, f1), (f2, f3)) = (fy[0], fy[1]);
    // SAFETY: reads rows `2..=h + 4` inside `0..h + 7`.
    unsafe {
        let (mut u0, mut prev) = (rows.pair(2, 3), rows.pair(4, 4));
        for r in (0..h).step_by(2) {
            let new = rows.pair(r + 5, r + 6);
            let u1 = prev.combine_hi_lo(new);
            let v0 = u0.combine_hi_lo(u1);
            let acc = V::acc_mul_add2(V::acc_zero(), u0, f0, v0, f1);
            let acc = V::acc_mul_add2(acc, u1, f2, new, f3);
            sink.write_pair(acc, r);
            (u0, prev) = (u1, new);
        }
    }
}

#[inline(always)]
unsafe fn v_halves_3<V: I16FilterLanes, R: PairRowSource<V>, S: McHalvesSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 3],
    h: usize,
) {
    let ((f0, f1), (f2, f3), (f4, f5)) = (fy[0], fy[1], fy[2]);
    // SAFETY: reads rows `1..=h + 5` inside `0..h + 7`.
    unsafe {
        let (mut u0, mut u1, mut prev) = (rows.pair(1, 2), rows.pair(3, 4), rows.pair(5, 5));
        let mut v0 = u0.combine_hi_lo(u1);
        for r in (0..h).step_by(2) {
            let new = rows.pair(r + 6, r + 7);
            let u2 = prev.combine_hi_lo(new);
            let v1 = u1.combine_hi_lo(u2);
            let acc = V::acc_mul_add2(V::acc_zero(), u0, f0, v0, f1);
            let acc = V::acc_mul_add2(acc, u1, f2, v1, f3);
            let acc = V::acc_mul_add2(acc, u2, f4, new, f5);
            sink.write_pair(acc, r);
            (u0, v0, u1, prev) = (u1, v1, u2, new);
        }
    }
}

#[inline(always)]
unsafe fn v_halves_4<V: I16FilterLanes, R: PairRowSource<V>, S: McHalvesSink<V>>(
    rows: R,
    sink: S,
    fy: [(i16, i16); 4],
    h: usize,
) {
    let ((f0, f1), (f2, f3), (f4, f5), (f6, f7)) = (fy[0], fy[1], fy[2], fy[3]);
    // SAFETY: reads rows `0..=h + 6` inside `0..h + 7`.
    unsafe {
        let (mut u0, mut u1, mut u2) = (rows.pair(0, 1), rows.pair(2, 3), rows.pair(4, 5));
        let (mut v0, mut v1) = (u0.combine_hi_lo(u1), u1.combine_hi_lo(u2));
        let mut prev = rows.pair(6, 6);
        for r in (0..h).step_by(2) {
            let new = rows.pair(r + 7, r + 8);
            let u3 = prev.combine_hi_lo(new);
            let v2 = u2.combine_hi_lo(u3);
            let acc = V::acc_mul_add2(V::acc_zero(), u0, f0, v0, f1);
            let acc = V::acc_mul_add2(acc, u1, f2, v1, f3);
            let acc = V::acc_mul_add2(acc, u2, f4, v2, f5);
            let acc = V::acc_mul_add2(acc, u3, f6, new, f7);
            sink.write_pair(acc, r);
            (u0, v0, u1, v1, u2, prev) = (u1, v1, u2, v2, u3, new);
        }
    }
}

#[inline(always)]
unsafe fn v_pass_halves<V: I16FilterLanes, R: PairRowSource<V>, S: McHalvesSink<V>>(
    rows: R,
    sink: S,
    fy: &[i16; AV1_FILTER_TAPS],
    h: usize,
) {
    // SAFETY: caller validated window and even height `h`.
    unsafe {
        match tap_pair_count(fy) {
            1 => v_halves_1(rows, sink, tap_pairs::<1>(fy), h),
            2 => v_halves_2(rows, sink, tap_pairs::<2>(fy), h),
            3 => v_halves_3(rows, sink, tap_pairs::<3>(fy), h),
            _ => v_halves_4(rows, sink, tap_pairs::<4>(fy), h),
        }
    }
}

#[inline(always)]
unsafe fn hv_pass_halves<V: I16FilterLanes + I16Pairs, S: McHalvesSink<V>, const R0: u32>(
    src_p: *const u16,
    ss: usize,
    (fx, fy): Av1Taps<'_>,
    h: usize,
    sink: S,
) {
    // SAFETY: caller validated window and even height `h`.
    unsafe {
        match tap_pair_count(fx) {
            1 => {
                let pairs = tap_pairs::<1>(fx);
                v_pass_halves::<V, _, _>(
                    HorizPairRow::<V, 1, R0> {
                        src_p,
                        ss,
                        fx: pairs,
                        fx_vec: splat_fx_pairs::<V, 1>(pairs),
                    },
                    sink,
                    fy,
                    h,
                );
            }
            2 => {
                let pairs = tap_pairs::<2>(fx);
                v_pass_halves::<V, _, _>(
                    HorizPairRow::<V, 2, R0> {
                        src_p,
                        ss,
                        fx: pairs,
                        fx_vec: splat_fx_pairs::<V, 2>(pairs),
                    },
                    sink,
                    fy,
                    h,
                );
            }
            3 => {
                let pairs = tap_pairs::<3>(fx);
                v_pass_halves::<V, _, _>(
                    HorizPairRow::<V, 3, R0> {
                        src_p,
                        ss,
                        fx: pairs,
                        fx_vec: splat_fx_pairs::<V, 3>(pairs),
                    },
                    sink,
                    fy,
                    h,
                );
            }
            _ => {
                let pairs = tap_pairs::<4>(fx);
                v_pass_halves::<V, _, _>(
                    HorizPairRow::<V, 4, R0> {
                        src_p,
                        ss,
                        fx: pairs,
                        fx_vec: splat_fx_pairs::<V, 4>(pairs),
                    },
                    sink,
                    fy,
                    h,
                );
            }
        }
    }
}

/// Unscaled `put` over `V: I16FilterLanes + I16Pairs` packing two rows per vector (`2 * W == V::LANES`).
#[inline(always)]
pub(crate) unsafe fn unscaled_put_halves<
    V: I16FilterLanes + I16Pairs,
    const W: usize,
    const ICL: bool,
>(
    (src, ss): (&[u16], usize),
    (dst, ds): (&mut [u16], usize),
    h: usize,
    (fx, fy): Av1Taps<'_>,
    depth: BitDepth,
) {
    assert!(
        h > 0
            && h <= KERNEL_EDGE
            && src.len() >= (h + AV1_FILTER_TAPS - 2) * ss + W + AV1_FILTER_TAPS - 1
            && dst.len() >= (h - 1) * ds + W,
        "AV1 u16 put window or destination too small"
    );
    debug_assert_eq!(2 * W, V::LANES);
    if !h.is_multiple_of(2) {
        unscaled_put_at::<u16, W>((src, ss), (dst, ds), h, (fx, fy), depth);
        return;
    }
    let (src_p, dst_p) = (src.as_ptr(), dst.as_mut_ptr());
    let (hx_id, vy_id) = (*fx == IDENTITY_TAPS, *fy == IDENTITY_TAPS);
    if hx_id && vy_id {
        for r in (0..h).step_by(2) {
            // SAFETY: validated window and destination cover rows `r` and `r + 1` of width `W`.
            unsafe {
                let row_a = src_p.add((r + AV1_FILTER_BEFORE) * ss + AV1_FILTER_BEFORE);
                let v = load_halves_u16::<V>(row_a, row_a.add(ss));
                if ds == W {
                    store_u16(v, dst_p.add(r * W));
                } else {
                    store_halves_u16(v, dst_p.add(r * ds), dst_p.add((r + 1) * ds));
                }
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if ICL
        // SAFETY: the entry validated the padded source and output.
        && unsafe {
            try_icl::<W, true>(
                (src_p, ss), (dst_p.cast(), ds), h, (fx, fy), (hx_id, vy_id), depth,
            )
        }
    {
        return;
    }
    let (zero, max_v) = (V::splat(0), V::splat(depth.max() as i16));
    let r0 = inter_round0(depth);
    if vy_id {
        let sink = PutHalvesSink::<V, W, FILTER_BITS> {
            dst_p,
            ds,
            round: (1 << (FILTER_BITS - 1)) + (1 << (r0 - 1)),
            zero,
            max_v,
        };
        // SAFETY: window and even height `h` validated above.
        unsafe {
            match tap_pair_count(fx) {
                1 => h_pass_halves::<V, _, 1>(src_p, ss, fx, h, sink),
                2 => h_pass_halves::<V, _, 2>(src_p, ss, fx, h, sink),
                3 => h_pass_halves::<V, _, 3>(src_p, ss, fx, h, sink),
                _ => h_pass_halves::<V, _, 4>(src_p, ss, fx, h, sink),
            }
        }
        return;
    }
    if hx_id {
        let sink = PutHalvesSink::<V, W, FILTER_BITS> {
            dst_p,
            ds,
            round: 1 << (FILTER_BITS - 1),
            zero,
            max_v,
        };
        // SAFETY: window and even height `h` validated above.
        unsafe { v_pass_halves::<V, _, _>(DirectPairRow { src_p, ss }, sink, fy, h) };
        return;
    }

    // SAFETY: window and even height `h` validated above.
    unsafe {
        if depth.bits() == 12 {
            let sink = PutHalvesSink::<V, W, 9> {
                dst_p,
                ds,
                round: 1 << 8,
                zero,
                max_v,
            };
            hv_pass_halves::<V, _, 5>(src_p, ss, (fx, fy), h, sink);
        } else {
            let sink = PutHalvesSink::<V, W, 11> {
                dst_p,
                ds,
                round: 1 << 10,
                zero,
                max_v,
            };
            hv_pass_halves::<V, _, 3>(src_p, ss, (fx, fy), h, sink);
        }
    }
}

#[inline(always)]
unsafe fn h_dispatch_halves<V: I16FilterLanes + I16Pairs, S: McHalvesSink<V>>(
    src_p: *const u16,
    ss: usize,
    fx: &[i16; AV1_FILTER_TAPS],
    h: usize,
    sink: S,
) {
    // SAFETY: forwarded from caller.
    unsafe {
        match tap_pair_count(fx) {
            1 => h_pass_halves::<V, _, 1>(src_p, ss, fx, h, sink),
            2 => h_pass_halves::<V, _, 2>(src_p, ss, fx, h, sink),
            3 => h_pass_halves::<V, _, 3>(src_p, ss, fx, h, sink),
            _ => h_pass_halves::<V, _, 4>(src_p, ss, fx, h, sink),
        }
    }
}

/// Unscaled `prep` over `V: I16FilterLanes + I16Pairs` packing two rows per vector (`2 * W == V::LANES`).
#[inline(always)]
pub(crate) unsafe fn unscaled_prep_halves<
    V: I16FilterLanes + I16Pairs,
    const W: usize,
    const ICL: bool,
>(
    (src, ss): (&[u16], usize),
    (dst, ds): (&mut [i16], usize),
    h: usize,
    (fx, fy): Av1Taps<'_>,
    depth: BitDepth,
) {
    assert!(
        h > 0
            && h <= KERNEL_EDGE
            && src.len() >= (h + AV1_FILTER_TAPS - 2) * ss + W + AV1_FILTER_TAPS - 1
            && dst.len() >= (h - 1) * ds + W,
        "AV1 u16 prep window or destination too small"
    );
    debug_assert_eq!(2 * W, V::LANES);
    if !h.is_multiple_of(2) {
        unscaled_prep_at::<u16, W>((src, ss), (dst, ds), h, (fx, fy), depth);
        return;
    }
    let (src_p, dst_p) = (src.as_ptr(), dst.as_mut_ptr());
    let bias = av1_compound_bias(depth);
    let post = inter_post_round(depth);
    let (hx_id, vy_id) = (*fx == IDENTITY_TAPS, *fy == IDENTITY_TAPS);
    if hx_id && vy_id {
        let (scale_v, bias_v) = (V::splat(1 << post), V::splat(bias as i16));
        for r in (0..h).step_by(2) {
            // SAFETY: validated window and destination cover rows `r` and `r + 1` of width `W`.
            unsafe {
                let row_a = src_p.add((r + AV1_FILTER_BEFORE) * ss + AV1_FILTER_BEFORE);
                let v = load_halves_u16::<V>(row_a, row_a.add(ss))
                    .mul(scale_v)
                    .sub(bias_v);
                if ds == W {
                    store_i16(v, dst_p.add(r * W));
                } else {
                    store_halves_i16(v, dst_p.add(r * ds), dst_p.add((r + 1) * ds));
                }
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if ICL
        // SAFETY: the entry validated the padded source and output.
        && unsafe {
            try_icl::<W, false>(
                (src_p, ss), (dst_p.cast(), ds), h, (fx, fy), (hx_id, vy_id), depth,
            )
        }
    {
        return;
    }
    let r0 = inter_round0(depth);
    let round = (1 << (r0 - 1)) - (bias << r0);
    if vy_id {
        // SAFETY: window and even height `h` validated above.
        unsafe {
            if depth.bits() == 12 {
                let s = PrepHalvesSink::<W, 5> { dst_p, ds, round };
                h_dispatch_halves::<V, _>(src_p, ss, fx, h, s);
            } else {
                let s = PrepHalvesSink::<W, 3> { dst_p, ds, round };
                h_dispatch_halves::<V, _>(src_p, ss, fx, h, s);
            }
        }
        return;
    }
    if hx_id {
        // SAFETY: window and even height `h` validated above.
        unsafe {
            if depth.bits() == 12 {
                let s = PrepHalvesSink::<W, 5> { dst_p, ds, round };
                v_pass_halves::<V, _, _>(DirectPairRow { src_p, ss }, s, fy, h);
            } else {
                let s = PrepHalvesSink::<W, 3> { dst_p, ds, round };
                v_pass_halves::<V, _, _>(DirectPairRow { src_p, ss }, s, fy, h);
            }
        }
        return;
    }

    let sink_2d = PrepHalvesSink::<W, INTER_ROUND1_COMPOUND> {
        dst_p,
        ds,
        round: (1 << (INTER_ROUND1_COMPOUND - 1)) - (bias << INTER_ROUND1_COMPOUND),
    };
    // SAFETY: window and even height `h` validated above.
    unsafe {
        if depth.bits() == 12 {
            hv_pass_halves::<V, _, 5>(src_p, ss, (fx, fy), h, sink_2d);
        } else {
            hv_pass_halves::<V, _, 3>(src_p, ss, (fx, fy), h, sink_2d);
        }
    }
}

// ------------------------------------------------------------ Scaled u16 ----

#[inline(always)]
fn zeroed_prefix(buf: &mut [MaybeUninit<i16>], len: usize) -> &mut [i16] {
    let prefix = &mut buf[..len];
    prefix.fill(MaybeUninit::new(0));
    // SAFETY: `prefix` was initialized above and has the layout of `[i16]`.
    unsafe { &mut *(core::ptr::from_mut::<[MaybeUninit<i16>]>(prefix) as *mut [i16]) }
}

/// Four horizontal 8-tap outputs: sample starts `st[..4]`, taps `tp[..4]`.
/// A function, not a closure: closures do not inline into the caller's
/// `#[target_feature]` context, and the outlined closure called its
/// horizontal adds out of line.
#[inline(always)]
fn dot4<V8>(src_i16: &[i16], round0: u32, st: &[usize], tp: &[V8]) -> V8::Wide
where
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
{
    let p0 = V8::load(&src_i16[st[0]..]).madd(tp[0]);
    let p1 = V8::load(&src_i16[st[1]..]).madd(tp[1]);
    let p2 = V8::load(&src_i16[st[2]..]).madd(tp[2]);
    let p3 = V8::load(&src_i16[st[3]..]).madd(tp[3]);
    V8::sum_lanes4([p0, p1, p2, p3]).round_shift(round0)
}

#[inline(always)]
fn scaled_h_pass<V8>(
    kernel_x: &Kernel,
    window: &PlaneView<'_, u16>,
    w: usize,
    rows: usize,
    x: Av1McAxis,
    round0: u32,
    inter: &mut [i16],
) where
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
{
    let mut starts_u = [MaybeUninit::<usize>::uninit(); AV1_MAX_PREDICTION_EDGE];
    let mut phases_u = [MaybeUninit::<u8>::uninit(); AV1_MAX_PREDICTION_EDGE];
    let mut taps_u = [MaybeUninit::<V8>::uninit(); AV1_MAX_PREDICTION_EDGE];
    for c in 0..w {
        let (start, phase) = x.sample(c);
        starts_u[c].write(start);
        phases_u[c].write(phase as u8);
        taps_u[c].write(V8::load(&kernel_x[phase]));
    }
    // SAFETY: indices `0..w` were initialized above.
    let (starts, phases, taps_x) = unsafe {
        (
            &*(core::ptr::from_ref::<[MaybeUninit<usize>]>(&starts_u[..w]) as *const [usize]),
            &*(core::ptr::from_ref::<[MaybeUninit<u8>]>(&phases_u[..w]) as *const [u8]),
            &*(core::ptr::from_ref::<[MaybeUninit<V8>]>(&taps_u[..w]) as *const [V8]),
        )
    };
    let w8 = w & !7;
    for (r, line) in inter.chunks_exact_mut(w).take(rows).enumerate() {
        let source = window.row(r);
        // SAFETY: `u16` samples in `0..=4095` fit positive `i16`.
        let src_i16 =
            unsafe { core::slice::from_raw_parts(source.as_ptr().cast::<i16>(), source.len()) };
        let mut c0 = 0usize;
        while c0 < w8 {
            let lo = dot4::<V8>(src_i16, round0, &starts[c0..], &taps_x[c0..]);
            let hi = dot4::<V8>(src_i16, round0, &starts[c0 + 4..], &taps_x[c0 + 4..]);
            V8::narrow_pairs(lo, hi).store(&mut line[c0..]);
            c0 += 8;
        }
        if c0 + 4 <= w {
            let lo = dot4::<V8>(src_i16, round0, &starts[c0..], &taps_x[c0..]);
            let mut tmp = [0i16; 8];
            V8::narrow_pairs(lo, lo).store(&mut tmp);
            line[c0..c0 + 4].copy_from_slice(&tmp[..4]);
            c0 += 4;
        }
        for c in c0..w {
            let start = starts[c];
            let phase = usize::from(phases[c]);
            let sum: i32 = kernel_x[phase]
                .iter()
                .zip(&source[start..start + AV1_FILTER_TAPS])
                .map(|(&t, &s)| i32::from(t) * i32::from(s))
                .sum();
            line[c] = round2(sum, round0) as i16;
        }
    }
}

#[inline(always)]
fn scaled_v_acc<V: I16Lanes>(
    inter: &[i16],
    offset: usize,
    stride: usize,
    fy: &[i16; AV1_FILTER_TAPS],
) -> V::Acc {
    let mut acc = V::acc_zero();
    for t in (0..AV1_FILTER_TAPS).step_by(2) {
        let a = V::load(&inter[offset + t * stride..]);
        let b = V::load(&inter[offset + (t + 1) * stride..]);
        acc = V::acc_mul_add2(acc, a, fy[t], b, fy[t + 1]);
    }
    acc
}

#[inline(always)]
pub(crate) unsafe fn scaled_put_lanes<V4, V8, VWide>(
    filters: Av1Filters,
    window: &PlaneView<'_, u16>,
    (w, h): (usize, usize),
    (axis_x, axis_y): (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_, u16>,
    depth: BitDepth,
) where
    V4: I16Lanes,
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
    VWide: I16Lanes,
{
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 prediction destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    check_prediction(window, (w, h), (axis_x, axis_y));
    let (r0, r1) = (inter_round0(depth), inter_round1_final(depth));
    let (kernel_x, kernel_y) = (filters.x.kernel(w), filters.y.kernel(h));
    let rows = axis_y.span(h);
    let needed = rows * w;
    let mut stack_buf = [MaybeUninit::<i16>::uninit(); INLINE_SCALED_SAMPLES];
    let mut heap_buf = Vec::new();
    let inter: &mut [i16] = if needed <= INLINE_SCALED_SAMPLES {
        zeroed_prefix(&mut stack_buf, needed)
    } else {
        heap_buf.resize(needed, 0);
        &mut heap_buf
    };
    scaled_h_pass::<V8>(kernel_x, window, w, rows, axis_x, r0, inter);

    let round = 1i32 << (r1 - 1);
    let (zero_w, max_w) = (VWide::splat(0), VWide::splat(depth.max() as i16));
    let (zero_8, max_8) = (V8::splat(0), V8::splat(depth.max() as i16));
    let (zero_4, max_4) = (V4::splat(0), V4::splat(depth.max() as i16));
    for r in 0..h {
        let (start, phase) = axis_y.sample(r);
        let fy = &kernel_y[phase];
        let base = start * w;
        let out_row = &mut dst.row_mut(r)[..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let acc = scaled_v_acc::<VWide>(inter, base + c0, w, fy);
            let clamped = VWide::acc_round_shift_narrow(acc, round, r1)
                .max(zero_w)
                .min(max_w);
            // SAFETY: `out_row[c0..]` holds at least `VWide::LANES` words.
            unsafe { store_u16(clamped, out_row[c0..].as_mut_ptr()) };
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let acc = scaled_v_acc::<V8>(inter, base + c0, w, fy);
            let clamped = V8::acc_round_shift_narrow(acc, round, r1)
                .max(zero_8)
                .min(max_8);
            // SAFETY: `out_row[c0..]` holds at least `V8::LANES` words.
            unsafe { store_u16(clamped, out_row[c0..].as_mut_ptr()) };
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let acc = scaled_v_acc::<V4>(inter, base + c0, w, fy);
            let clamped = V4::acc_round_shift_narrow(acc, round, r1)
                .max(zero_4)
                .min(max_4);
            // SAFETY: `out_row[c0..]` holds at least `V4::LANES` words.
            unsafe { store_u16(clamped, out_row[c0..].as_mut_ptr()) };
            c0 += V4::LANES;
        }
        for (c, out) in out_row.iter_mut().enumerate().skip(c0) {
            let sum: i32 = fy
                .iter()
                .enumerate()
                .map(|(t, &tap)| i32::from(tap) * i32::from(inter[(start + t) * w + c]))
                .sum();
            *out = depth.clip_to(round2(sum, r1));
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn scaled_prep_lanes<V4, V8, VWide>(
    filters: Av1Filters,
    window: &PlaneView<'_, u16>,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
    (dst, dst_stride): (&mut [i16], usize),
    depth: BitDepth,
) where
    V4: I16Lanes,
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
    VWide: I16Lanes,
{
    check_prediction(window, (w, h), (x, y));
    let r0 = inter_round0(depth);
    let (kernel_x, kernel_y) = (filters.x.kernel(w), filters.y.kernel(h));
    let rows = y.span(h);
    let needed = rows * w;
    let mut stack_buf = [MaybeUninit::<i16>::uninit(); INLINE_SCALED_SAMPLES];
    let mut heap_buf = Vec::new();
    let inter: &mut [i16] = if needed <= INLINE_SCALED_SAMPLES {
        zeroed_prefix(&mut stack_buf, needed)
    } else {
        heap_buf.resize(needed, 0);
        &mut heap_buf
    };
    scaled_h_pass::<V8>(kernel_x, window, w, rows, x, r0, inter);

    let bias = av1_compound_bias(depth);
    let round = (1i32 << (INTER_ROUND1_COMPOUND - 1)) - (bias << INTER_ROUND1_COMPOUND);
    for r in 0..h {
        let (start, phase) = y.sample(r);
        let fy = &kernel_y[phase];
        let base = start * w;
        let out_row = &mut dst[r * dst_stride..r * dst_stride + w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let acc = scaled_v_acc::<VWide>(inter, base + c0, w, fy);
            VWide::acc_round_shift_narrow(acc, round, INTER_ROUND1_COMPOUND)
                .store(&mut out_row[c0..]);
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let acc = scaled_v_acc::<V8>(inter, base + c0, w, fy);
            V8::acc_round_shift_narrow(acc, round, INTER_ROUND1_COMPOUND).store(&mut out_row[c0..]);
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let acc = scaled_v_acc::<V4>(inter, base + c0, w, fy);
            V4::acc_round_shift_narrow(acc, round, INTER_ROUND1_COMPOUND).store(&mut out_row[c0..]);
            c0 += V4::LANES;
        }
        for (c, out) in out_row.iter_mut().enumerate().skip(c0) {
            let sum: i32 = fy
                .iter()
                .enumerate()
                .map(|(t, &tap)| i32::from(tap) * i32::from(inter[(start + t) * w + c]))
                .sum();
            *out = (round2(sum, INTER_ROUND1_COMPOUND) - bias) as i16;
        }
    }
}

// -------------------------------------------------------------- Warp u16 ----

#[inline(always)]
fn warp_products<V: I16x8Shuffles>(
    line: &[i16],
    offset: usize,
    base: i32,
    step: i32,
) -> [V::Wide; 4] {
    [
        V::load(&line[offset..]).madd(V::load(warp_taps_i16(base))),
        V::load(&line[offset + 1..]).madd(V::load(warp_taps_i16(base + step))),
        V::load(&line[offset + 2..]).madd(V::load(warp_taps_i16(base + 2 * step))),
        V::load(&line[offset + 3..]).madd(V::load(warp_taps_i16(base + 3 * step))),
    ]
}

#[inline(always)]
pub(crate) unsafe fn warp8x8_u16_lanes<V: I16x8Shuffles>(
    (window, stride): (&[u16], usize),
    phases: &WarpPhases,
    (round0, round1): (u32, u32),
    out: &mut WarpOut,
) {
    let mut columns = [[0i16; WINDOW_STRIDE]; WARP_BLOCK];
    let (start_x, alpha, beta) = phases.x;
    for i in 0..WINDOW_SIDE {
        let source = &window[i * stride..][..WINDOW_STRIDE];
        // SAFETY: `u16` samples in `0..=4095` fit positive `i16`.
        let row =
            unsafe { core::slice::from_raw_parts(source.as_ptr().cast::<i16>(), WINDOW_STRIDE) };
        let row_phase = start_x + beta * (i as i32 - AV1_FILTER_BEFORE as i32);
        for half in [0, 4] {
            let products = warp_products::<V>(row, half, row_phase + alpha * half as i32, alpha);
            let mut sums = [0i32; 4];
            V::sum_lanes4(products).round_shift(round0).store(&mut sums);
            for (q, &sum) in sums.iter().enumerate() {
                columns[half + q][i] = sum as i16;
            }
        }
    }
    let (start_y, gamma, delta) = phases.y;
    for (c, column) in columns.iter().enumerate() {
        let col_phase = start_y + gamma * c as i32;
        for half in [0, 4] {
            let products = warp_products::<V>(column, half, col_phase + delta * half as i32, delta);
            let mut sums = [0i32; 4];
            V::sum_lanes4(products).round_shift(round1).store(&mut sums);
            for (q, &sum) in sums.iter().enumerate() {
                out[(half + q) * WARP_BLOCK + c] = sum;
            }
        }
    }
}

// ------------------------------------------------------ ISA Instantiations --

#[cfg(target_arch = "x86_64")]
mod horizontal_icl;

#[cfg(target_arch = "x86_64")]
mod separable_icl;

#[cfg(target_arch = "x86_64")]
mod vertical_icl;

/// Use the dedicated Ice Lake pass for the active axes where available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn try_icl<const W: usize, const PUT: bool>(
    src: (*const u16, usize),
    dst: (*mut u16, usize),
    height: usize,
    (fx, fy): Av1Taps<'_>,
    (horizontal_identity, vertical_identity): (bool, bool),
    depth: BitDepth,
) -> bool {
    // SAFETY: the caller checked both spans and selected the complete tier.
    unsafe {
        match (horizontal_identity, vertical_identity) {
            (false, true) => horizontal_icl::try_filter::<W, PUT>(src, dst, height, fx, depth),
            (true, false) => vertical_icl::try_filter::<W, PUT>(src, dst, height, fy, depth),
            (false, false) => {
                separable_icl::try_filter::<W, PUT>(src, dst, height, (fx, fy), depth)
            }
            (true, true) => false,
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{
        Av1Filters, Av1McAxis, Av1Taps, BitDepth, HighAv1McFns, PlaneView, PlaneViewMut, WarpOut,
        WarpPhases, scaled_prep_lanes, scaled_put_lanes, unscaled_prep_halves, unscaled_prep_lanes,
        unscaled_put_halves, unscaled_put_lanes, warp8x8_u16_lanes,
    };
    use crate::av1_mc::high_blend::x86::{AVX2_BLENDS, AVX512_BLENDS};
    use crate::simd::avx2::{I16x4, I16x8, I16x16};
    use crate::simd::avx512_i16::I16x32;

    macro_rules! avx2_unscaled_halves {
        ($put:ident, $prep:ident, $v:ty, $w:literal) => {
            #[target_feature(enable = "avx2")]
            unsafe fn $put(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: AVX2 is enabled by this wrapper.
                unsafe { unscaled_put_halves::<$v, $w, false>(src, dst, h, taps, depth) }
            }
            #[target_feature(enable = "avx2")]
            unsafe fn $prep(
                src: (&[u16], usize),
                dst: (&mut [i16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: AVX2 is enabled by this wrapper.
                unsafe { unscaled_prep_halves::<$v, $w, false>(src, dst, h, taps, depth) }
            }
        };
    }

    macro_rules! avx2_unscaled {
        ($put:ident, $prep:ident, $v:ty, $w:literal) => {
            #[target_feature(enable = "avx2")]
            unsafe fn $put(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: AVX2 is enabled by this wrapper.
                unsafe { unscaled_put_lanes::<$v, $w, false>(src, dst, h, taps, depth) }
            }
            #[target_feature(enable = "avx2")]
            unsafe fn $prep(
                src: (&[u16], usize),
                dst: (&mut [i16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: AVX2 is enabled by this wrapper.
                unsafe { unscaled_prep_lanes::<$v, $w, false>(src, dst, h, taps, depth) }
            }
        };
    }

    avx2_unscaled_halves!(put_w4, prep_w4, I16x8, 4);
    avx2_unscaled_halves!(put_w8, prep_w8, I16x16, 8);
    avx2_unscaled!(put_w16, prep_w16, I16x16, 16);
    avx2_unscaled!(put_w32, prep_w32, I16x16, 32);
    avx2_unscaled!(put_w64, prep_w64, I16x16, 64);

    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
    unsafe fn put_w8_512(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: this entry uses the complete Ice Lake feature set.
        unsafe { unscaled_put_halves::<I16x16, 8, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
    unsafe fn prep_w4_512(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: this entry uses the complete Ice Lake feature set.
        unsafe { unscaled_prep_halves::<I16x8, 4, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn put_w16_512(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe { unscaled_put_halves::<I16x32, 16, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn put_w32_512(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe { unscaled_put_lanes::<I16x32, 32, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn put_w64_512(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe { unscaled_put_lanes::<I16x32, 64, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn prep_w16_512(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe { unscaled_prep_halves::<I16x32, 16, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn prep_w32_512(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe { unscaled_prep_lanes::<I16x32, 32, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn prep_w64_512(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe { unscaled_prep_lanes::<I16x32, 64, true>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn scaled_put_avx2(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, u16>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX2 is enabled by this wrapper.
        unsafe {
            scaled_put_lanes::<I16x4, I16x8, I16x16>(filters, window, size, axes, dst, depth);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn scaled_prep_avx2(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: (&mut [i16], usize),
        depth: BitDepth,
    ) {
        // SAFETY: AVX2 is enabled by this wrapper.
        unsafe {
            scaled_prep_lanes::<I16x4, I16x8, I16x16>(filters, window, size, axes, dst, depth);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn scaled_put_avx512(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, u16>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe {
            scaled_put_lanes::<I16x4, I16x8, I16x32>(filters, window, size, axes, dst, depth);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn scaled_prep_avx512(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: (&mut [i16], usize),
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe {
            scaled_prep_lanes::<I16x4, I16x8, I16x32>(filters, window, size, axes, dst, depth);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn warp8x8_avx2(
        window: (&[u16], usize),
        phases: &WarpPhases,
        rounds: (u32, u32),
        out: &mut WarpOut,
    ) {
        // SAFETY: AVX2 is enabled by this wrapper.
        unsafe { warp8x8_u16_lanes::<I16x8>(window, phases, rounds, out) }
    }

    pub(super) const AVX2_FNS: HighAv1McFns = HighAv1McFns {
        put: [put_w4, put_w8, put_w16, put_w32, put_w64],
        prep: [prep_w4, prep_w8, prep_w16, prep_w32, prep_w64],
        scaled_put: scaled_put_avx2,
        scaled_prep: scaled_prep_avx2,
        warp8x8: warp8x8_avx2,
        blend: AVX2_BLENDS,
    };

    pub(super) const AVX512_FNS: HighAv1McFns = HighAv1McFns {
        put: [put_w4, put_w8_512, put_w16_512, put_w32_512, put_w64_512],
        prep: [
            prep_w4_512,
            prep_w8,
            prep_w16_512,
            prep_w32_512,
            prep_w64_512,
        ],
        scaled_put: scaled_put_avx512,
        scaled_prep: scaled_prep_avx512,
        warp8x8: warp8x8_avx2,
        blend: AVX512_BLENDS,
    };
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{
        Av1Filters, Av1McAxis, Av1Taps, BitDepth, HighAv1McFns, PlaneView, PlaneViewMut, WarpOut,
        WarpPhases, scaled_prep_lanes, scaled_put_lanes, unscaled_prep_halves, unscaled_prep_lanes,
        unscaled_put_halves, unscaled_put_lanes, warp8x8_u16_lanes,
    };
    use crate::av1_mc::high_blend::neon::NEON_BLENDS;
    #[cfg(target_os = "linux")]
    use crate::simd::neon::I16x8Sve2;
    use crate::simd::neon::{I16x4, I16x8};

    #[target_feature(enable = "neon")]
    unsafe fn put_w4(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: NEON is enabled by this wrapper.
        unsafe { unscaled_put_halves::<I16x8, 4, false>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn prep_w4(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: NEON is enabled by this wrapper.
        unsafe { unscaled_prep_halves::<I16x8, 4, false>(src, dst, h, taps, depth) }
    }

    macro_rules! neon_unscaled {
        ($put:ident, $prep:ident, $w:literal) => {
            #[target_feature(enable = "neon")]
            unsafe fn $put(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: NEON is enabled by this wrapper.
                unsafe { unscaled_put_lanes::<I16x8, $w, false>(src, dst, h, taps, depth) }
            }

            #[target_feature(enable = "neon")]
            unsafe fn $prep(
                src: (&[u16], usize),
                dst: (&mut [i16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: NEON is enabled by this wrapper.
                unsafe { unscaled_prep_lanes::<I16x8, $w, false>(src, dst, h, taps, depth) }
            }
        };
    }

    neon_unscaled!(put_w8, prep_w8, 8);
    neon_unscaled!(put_w16, prep_w16, 16);
    neon_unscaled!(put_w32, prep_w32, 32);
    neon_unscaled!(put_w64, prep_w64, 64);

    // Keep the SVE2 loop's stack frame out of the width entry point so the
    // common short filters branch straight to the existing NEON functions.
    #[cfg(target_os = "linux")]
    #[inline(never)]
    #[target_feature(enable = "neon")]
    unsafe fn put_full_taps_sve2<const W: usize>(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: these functions are only reached from the SVE2 table.
        unsafe { unscaled_put_lanes::<I16x8Sve2, W, false>(src, dst, h, taps, depth) }
    }

    #[cfg(target_os = "linux")]
    #[inline(never)]
    #[target_feature(enable = "neon")]
    unsafe fn prep_full_taps_sve2<const W: usize>(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: these functions are only reached from the SVE2 table.
        unsafe { unscaled_prep_lanes::<I16x8Sve2, W, false>(src, dst, h, taps, depth) }
    }

    #[cfg(target_os = "linux")]
    macro_rules! neon_sve2_unscaled {
        ($put:ident, $prep:ident, $w:literal, $neon_put:ident, $neon_prep:ident) => {
            unsafe fn $put(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: the SVE2 table also requires NEON.
                unsafe {
                    if taps.0[0] == 0 && taps.0[7] == 0 {
                        $neon_put(src, dst, h, taps, depth)
                    } else {
                        put_full_taps_sve2::<$w>(src, dst, h, taps, depth)
                    }
                }
            }

            unsafe fn $prep(
                src: (&[u16], usize),
                dst: (&mut [i16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: the SVE2 table also requires NEON.
                unsafe {
                    if taps.0[0] == 0 && taps.0[7] == 0 {
                        $neon_prep(src, dst, h, taps, depth)
                    } else {
                        prep_full_taps_sve2::<$w>(src, dst, h, taps, depth)
                    }
                }
            }
        };
    }

    #[cfg(target_os = "linux")]
    neon_sve2_unscaled!(put_sve2_w8, prep_sve2_w8, 8, put_w8, prep_w8);
    #[cfg(target_os = "linux")]
    neon_sve2_unscaled!(put_sve2_w16, prep_sve2_w16, 16, put_w16, prep_w16);
    #[cfg(target_os = "linux")]
    neon_sve2_unscaled!(put_sve2_w32, prep_sve2_w32, 32, put_w32, prep_w32);
    #[cfg(target_os = "linux")]
    neon_sve2_unscaled!(put_sve2_w64, prep_sve2_w64, 64, put_w64, prep_w64);

    #[target_feature(enable = "neon")]
    unsafe fn scaled_put_neon(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, u16>,
        depth: BitDepth,
    ) {
        // SAFETY: NEON is enabled by this wrapper.
        unsafe { scaled_put_lanes::<I16x4, I16x8, I16x8>(filters, window, size, axes, dst, depth) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn scaled_prep_neon(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: (&mut [i16], usize),
        depth: BitDepth,
    ) {
        // SAFETY: NEON is enabled by this wrapper.
        unsafe { scaled_prep_lanes::<I16x4, I16x8, I16x8>(filters, window, size, axes, dst, depth) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn warp8x8_neon(
        window: (&[u16], usize),
        phases: &WarpPhases,
        rounds: (u32, u32),
        out: &mut WarpOut,
    ) {
        // SAFETY: NEON is enabled by this wrapper.
        unsafe { warp8x8_u16_lanes::<I16x8>(window, phases, rounds, out) }
    }

    pub(super) const NEON_FNS: HighAv1McFns = HighAv1McFns {
        put: [put_w4, put_w8, put_w16, put_w32, put_w64],
        prep: [prep_w4, prep_w8, prep_w16, prep_w32, prep_w64],
        scaled_put: scaled_put_neon,
        scaled_prep: scaled_prep_neon,
        warp8x8: warp8x8_neon,
        blend: NEON_BLENDS,
    };

    #[cfg(target_os = "linux")]
    pub(super) const SVE2_FNS: HighAv1McFns = HighAv1McFns {
        put: [
            put_w4,
            put_sve2_w8,
            put_sve2_w16,
            put_sve2_w32,
            put_sve2_w64,
        ],
        prep: [
            prep_w4,
            prep_sve2_w8,
            prep_sve2_w16,
            prep_sve2_w32,
            prep_sve2_w64,
        ],
        ..NEON_FNS
    };
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use super::{
        Av1Filters, Av1McAxis, Av1Taps, BitDepth, HighAv1McFns, PlaneView, PlaneViewMut, WarpOut,
        WarpPhases, scaled_prep_lanes, scaled_put_lanes, unscaled_prep_halves, unscaled_prep_lanes,
        unscaled_put_halves, unscaled_put_lanes, warp8x8_u16_lanes,
    };
    use crate::av1_mc::high_blend::simd128::SIMD128_BLENDS;
    use crate::simd::simd128::{I16x4, I16x8};

    #[target_feature(enable = "simd128")]
    unsafe fn put_w4(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: SIMD128 is enabled by this wrapper.
        unsafe { unscaled_put_halves::<I16x8, 4, false>(src, dst, h, taps, depth) }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn prep_w4(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: SIMD128 is enabled by this wrapper.
        unsafe { unscaled_prep_halves::<I16x8, 4, false>(src, dst, h, taps, depth) }
    }

    macro_rules! simd128_unscaled {
        ($put:ident, $prep:ident, $v:ty, $w:literal) => {
            #[target_feature(enable = "simd128")]
            unsafe fn $put(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: SIMD128 is enabled by this wrapper.
                unsafe { unscaled_put_lanes::<$v, $w, false>(src, dst, h, taps, depth) }
            }
            #[target_feature(enable = "simd128")]
            unsafe fn $prep(
                src: (&[u16], usize),
                dst: (&mut [i16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: SIMD128 is enabled by this wrapper.
                unsafe { unscaled_prep_lanes::<$v, $w, false>(src, dst, h, taps, depth) }
            }
        };
    }

    simd128_unscaled!(put_w8, prep_w8, I16x8, 8);
    simd128_unscaled!(put_w16, prep_w16, I16x8, 16);
    simd128_unscaled!(put_w32, prep_w32, I16x8, 32);
    simd128_unscaled!(put_w64, prep_w64, I16x8, 64);

    #[target_feature(enable = "simd128")]
    unsafe fn scaled_put_simd128(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, u16>,
        depth: BitDepth,
    ) {
        // SAFETY: SIMD128 is enabled by this wrapper.
        unsafe { scaled_put_lanes::<I16x4, I16x8, I16x8>(filters, window, size, axes, dst, depth) }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn scaled_prep_simd128(
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: (&mut [i16], usize),
        depth: BitDepth,
    ) {
        // SAFETY: SIMD128 is enabled by this wrapper.
        unsafe { scaled_prep_lanes::<I16x4, I16x8, I16x8>(filters, window, size, axes, dst, depth) }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn warp8x8_simd128(
        window: (&[u16], usize),
        phases: &WarpPhases,
        rounds: (u32, u32),
        out: &mut WarpOut,
    ) {
        // SAFETY: SIMD128 is enabled by this wrapper.
        unsafe { warp8x8_u16_lanes::<I16x8>(window, phases, rounds, out) }
    }

    pub(super) const SIMD128_FNS: HighAv1McFns = HighAv1McFns {
        put: [put_w4, put_w8, put_w16, put_w32, put_w64],
        prep: [prep_w4, prep_w8, prep_w16, prep_w32, prep_w64],
        scaled_put: scaled_put_simd128,
        scaled_prep: scaled_prep_simd128,
        warp8x8: warp8x8_simd128,
        blend: SIMD128_BLENDS,
    };
}
