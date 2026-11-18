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
use crate::simd::{I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes};

const IDENTITY_TAPS: [i16; AV1_FILTER_TAPS] = [0, 0, 0, 128, 0, 0, 0, 0];
const UNSCALED_ROWS: usize = KERNEL_EDGE + AV1_FILTER_TAPS - 1;
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

/// Non-zero tap span of an 8-tap AV1 filter (2, 4, 6, or 8 taps).
#[derive(Clone, Copy)]
struct TapSpan {
    first: usize,
    pairs: usize,
}

impl TapSpan {
    #[inline(always)]
    fn of(taps: &[i16; AV1_FILTER_TAPS]) -> Self {
        if taps[0] == 0 && taps[7] == 0 {
            if taps[1] == 0 && taps[6] == 0 {
                if taps[2] == 0 && taps[5] == 0 {
                    return Self { first: 3, pairs: 1 };
                }
                return Self { first: 2, pairs: 2 };
            }
            return Self { first: 1, pairs: 3 };
        }
        Self { first: 0, pairs: 4 }
    }

    #[inline(always)]
    const fn count(self) -> usize {
        self.pairs * 2
    }
}

#[inline(always)]
unsafe fn load_i16<V: I16Lanes>(ptr: *const u16) -> V {
    // SAFETY: caller validated `V::LANES` readable `u16` words (`0..=4095`).
    V::load(unsafe { core::slice::from_raw_parts(ptr.cast::<i16>(), V::LANES) })
}

#[inline(always)]
unsafe fn store_u16<V: I16Lanes>(v: V, ptr: *mut u16) {
    // SAFETY: caller validated `V::LANES` writable `u16` words.
    v.store(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<i16>(), V::LANES) });
}

#[inline(always)]
unsafe fn store_i16<V: I16Lanes>(v: V, ptr: *mut i16) {
    // SAFETY: caller validated `V::LANES` writable `i16` words.
    v.store(unsafe { core::slice::from_raw_parts_mut(ptr, V::LANES) });
}

#[inline(always)]
unsafe fn h_dot<V: I16Lanes>(
    row: *const u16,
    fx: &[i16; AV1_FILTER_TAPS],
    span: TapSpan,
) -> V::Acc {
    let mut acc = V::acc_zero();
    for p in 0..span.pairs {
        let t = span.first + 2 * p;
        // SAFETY: `row` covers `V::LANES + 7` words inside the validated window.
        let (a, b) = unsafe { (load_i16::<V>(row.add(t)), load_i16::<V>(row.add(t + 1))) };
        acc = V::acc_mul_add2(acc, a, fx[t], b, fx[t + 1]);
    }
    acc
}

#[inline(always)]
unsafe fn v_dot_src<V: I16Lanes>(
    col_base: *const u16,
    ss: usize,
    fy: &[i16; AV1_FILTER_TAPS],
    span: TapSpan,
) -> V::Acc {
    let mut acc = V::acc_zero();
    for p in 0..span.pairs {
        let t = span.first + 2 * p;
        // SAFETY: `col_base` covers `7` rows below inside the validated window.
        let (a, b) = unsafe {
            (
                load_i16::<V>(col_base.add(t * ss)),
                load_i16::<V>(col_base.add((t + 1) * ss)),
            )
        };
        acc = V::acc_mul_add2(acc, a, fy[t], b, fy[t + 1]);
    }
    acc
}

#[inline(always)]
unsafe fn fill_inter<V: I16Lanes, const W: usize>(
    src_p: *const u16,
    ss: usize,
    fx: &[i16; AV1_FILTER_TAPS],
    h_span: TapSpan,
    first_row: usize,
    inter_rows: usize,
    r0: u32,
    inter: &mut [MaybeUninit<[i16; W]>; UNSCALED_ROWS],
) {
    let round0 = 1 << (r0 - 1);
    for (i, slot) in inter.iter_mut().take(inter_rows).enumerate() {
        // SAFETY: `first_row + i < h + 7`, inside the validated window.
        let row = unsafe { src_p.add((first_row + i) * ss) };
        let line_ptr = slot.as_mut_ptr().cast::<i16>();
        for c0 in (0..W).step_by(V::LANES) {
            // SAFETY: writes all `W` words of `slot`.
            unsafe {
                let acc = h_dot::<V>(row.add(c0), fx, h_span);
                store_i16(V::acc_round_shift_narrow(acc, round0, r0), line_ptr.add(c0));
            }
        }
    }
}

#[inline(always)]
unsafe fn v_dot_inter<V: I16Lanes, const W: usize>(
    inter_base: *const i16,
    r: usize,
    c0: usize,
    fy: &[i16; AV1_FILTER_TAPS],
    v_span: TapSpan,
) -> V::Acc {
    let mut acc = V::acc_zero();
    for p in 0..v_span.pairs {
        let t = 2 * p;
        let fy_idx = v_span.first + t;
        // SAFETY: caller initialized rows `0..h + v_span.count() - 1` of `inter`.
        let (lo, hi) = unsafe {
            (
                V::load(core::slice::from_raw_parts(
                    inter_base.add((r + t) * W + c0),
                    V::LANES,
                )),
                V::load(core::slice::from_raw_parts(
                    inter_base.add((r + t + 1) * W + c0),
                    V::LANES,
                )),
            )
        };
        acc = V::acc_mul_add2(acc, lo, fy[fy_idx], hi, fy[fy_idx + 1]);
    }
    acc
}

/// Unscaled `put` (`u16` output, clipped to `depth`) over `V: I16Lanes`.
#[inline(always)]
pub(crate) unsafe fn unscaled_put_lanes<V: I16Lanes, const W: usize>(
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
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src_p.add((r + AV1_FILTER_BEFORE) * ss + AV1_FILTER_BEFORE),
                    dst_p.add(r * ds),
                    W,
                );
            }
        }
        return;
    }
    let (zero, max_v) = (V::splat(0), V::splat(depth.max() as i16));
    let (r0, r1) = (inter_round0(depth), inter_round1_final(depth));
    if vy_id {
        let span = TapSpan::of(fx);
        let round = (1 << (FILTER_BITS - 1)) + (1 << (r0 - 1));
        for r in 0..h {
            // SAFETY: row `r + 3` lies inside the validated `h + 7` window rows.
            let row = unsafe { src_p.add((r + AV1_FILTER_BEFORE) * ss) };
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: `c0 + V::LANES + 7` lies inside `W + 7`, and `dst` covers `W`.
                unsafe {
                    let acc = h_dot::<V>(row.add(c0), fx, span);
                    let out = V::acc_round_shift_narrow(acc, round, FILTER_BITS)
                        .max(zero)
                        .min(max_v);
                    store_u16(out, dst_p.add(r * ds + c0));
                }
            }
        }
        return;
    }
    if hx_id {
        let span = TapSpan::of(fy);
        let round = 1 << (FILTER_BITS - 1);
        for r in 0..h {
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: `r..r + 8` and `c0 + 3..c0 + 3 + V::LANES` lie inside the window.
                unsafe {
                    let acc =
                        v_dot_src::<V>(src_p.add(r * ss + c0 + AV1_FILTER_BEFORE), ss, fy, span);
                    let out = V::acc_round_shift_narrow(acc, round, FILTER_BITS)
                        .max(zero)
                        .min(max_v);
                    store_u16(out, dst_p.add(r * ds + c0));
                }
            }
        }
        return;
    }
    let (h_span, v_span) = (TapSpan::of(fx), TapSpan::of(fy));
    let inter_rows = h + v_span.count() - 1;
    let mut inter = [MaybeUninit::<[i16; W]>::uninit(); UNSCALED_ROWS];
    // SAFETY: window validated above.
    unsafe {
        fill_inter::<V, W>(
            src_p,
            ss,
            fx,
            h_span,
            v_span.first,
            inter_rows,
            r0,
            &mut inter,
        );
    }
    let round1 = 1 << (r1 - 1);
    let inter_base = inter.as_ptr().cast::<i16>();
    for r in 0..h {
        for c0 in (0..W).step_by(V::LANES) {
            // SAFETY: rows `0..inter_rows` initialized above; destination covers `r * ds + c0 + V::LANES`.
            unsafe {
                let acc = v_dot_inter::<V, W>(inter_base, r, c0, fy, v_span);
                let out = V::acc_round_shift_narrow(acc, round1, r1)
                    .max(zero)
                    .min(max_v);
                store_u16(out, dst_p.add(r * ds + c0));
            }
        }
    }
}

/// Unscaled `prep` (`i16` compound output less bias) over `V: I16Lanes`.
#[inline(always)]
pub(crate) unsafe fn unscaled_prep_lanes<V: I16Lanes, const W: usize>(
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
    let r0 = inter_round0(depth);
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
    let round_1d = (1 << (r0 - 1)) - (bias << r0);
    if vy_id {
        let span = TapSpan::of(fx);
        for r in 0..h {
            // SAFETY: row `r + 3` lies inside the validated window.
            let row = unsafe { src_p.add((r + AV1_FILTER_BEFORE) * ss) };
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: `c0 + V::LANES + 7 <= W + 7`.
                unsafe {
                    let acc = h_dot::<V>(row.add(c0), fx, span);
                    store_i16(
                        V::acc_round_shift_narrow(acc, round_1d, r0),
                        dst_p.add(r * ds + c0),
                    );
                }
            }
        }
        return;
    }
    if hx_id {
        let span = TapSpan::of(fy);
        for r in 0..h {
            for c0 in (0..W).step_by(V::LANES) {
                // SAFETY: `r..r + 8` and `c0 + 3..c0 + 3 + V::LANES` lie inside the window.
                unsafe {
                    let acc =
                        v_dot_src::<V>(src_p.add(r * ss + c0 + AV1_FILTER_BEFORE), ss, fy, span);
                    store_i16(
                        V::acc_round_shift_narrow(acc, round_1d, r0),
                        dst_p.add(r * ds + c0),
                    );
                }
            }
        }
        return;
    }
    let (h_span, v_span) = (TapSpan::of(fx), TapSpan::of(fy));
    let inter_rows = h + v_span.count() - 1;
    let mut inter = [MaybeUninit::<[i16; W]>::uninit(); UNSCALED_ROWS];
    // SAFETY: window validated above.
    unsafe {
        fill_inter::<V, W>(
            src_p,
            ss,
            fx,
            h_span,
            v_span.first,
            inter_rows,
            r0,
            &mut inter,
        );
    }
    let round1 = (1 << (INTER_ROUND1_COMPOUND - 1)) - (bias << INTER_ROUND1_COMPOUND);
    let inter_base = inter.as_ptr().cast::<i16>();
    for r in 0..h {
        for c0 in (0..W).step_by(V::LANES) {
            // SAFETY: rows `0..inter_rows` initialized above; destination covers `r * ds + c0 + V::LANES`.
            unsafe {
                let acc = v_dot_inter::<V, W>(inter_base, r, c0, fy, v_span);
                store_i16(
                    V::acc_round_shift_narrow(acc, round1, INTER_ROUND1_COMPOUND),
                    dst_p.add(r * ds + c0),
                );
            }
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
pub(crate) unsafe fn warp8x8_u16_lanes<V: I16x8Shuffles>(
    (window, stride): (&[u16], usize),
    phases: &WarpPhases,
    (round0, round1): (u32, u32),
    out: &mut WarpOut,
) {
    let mut columns = [[0i16; WINDOW_STRIDE]; WARP_BLOCK];
    for i in 0..WINDOW_SIDE {
        let source = &window[i * stride..][..WINDOW_STRIDE];
        // SAFETY: `u16` samples in `0..=4095` fit positive `i16`.
        let row =
            unsafe { core::slice::from_raw_parts(source.as_ptr().cast::<i16>(), WINDOW_STRIDE) };
        for half in [0, 4] {
            let products: [V::Wide; 4] = core::array::from_fn(|q| {
                let c = half + q;
                let (start, alpha, beta) = phases.x;
                let phase = start + alpha * c as i32 + beta * (i as i32 - 3);
                V::load(&row[c..]).madd(V::load(warp_taps_i16(phase)))
            });
            let mut sums = [0i32; 4];
            V::sum_lanes4(products).round_shift(round0).store(&mut sums);
            for (q, &sum) in sums.iter().enumerate() {
                columns[half + q][i] = sum as i16;
            }
        }
    }
    for (c, column) in columns.iter().enumerate() {
        for half in [0, 4] {
            let products: [V::Wide; 4] = core::array::from_fn(|q| {
                let r = half + q;
                let (start, gamma, delta) = phases.y;
                let phase = start + gamma * c as i32 + delta * r as i32;
                V::load(&column[r..]).madd(V::load(warp_taps_i16(phase)))
            });
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
mod x86 {
    use super::{
        Av1Filters, Av1McAxis, Av1Taps, BitDepth, HighAv1McFns, PlaneView, PlaneViewMut, WarpOut,
        WarpPhases, scaled_prep_lanes, scaled_put_lanes, unscaled_prep_lanes, unscaled_put_lanes,
        warp8x8_u16_lanes,
    };
    use crate::av1_mc::high_blend::x86::{AVX2_BLENDS, AVX512_BLENDS};
    use crate::simd::avx2::{I16x4, I16x8, I16x16};
    use crate::simd::avx512_i16::I16x32;

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
                unsafe { unscaled_put_lanes::<$v, $w>(src, dst, h, taps, depth) }
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
                unsafe { unscaled_prep_lanes::<$v, $w>(src, dst, h, taps, depth) }
            }
        };
    }

    avx2_unscaled!(put_w4, prep_w4, I16x4, 4);
    avx2_unscaled!(put_w8, prep_w8, I16x8, 8);
    avx2_unscaled!(put_w16, prep_w16, I16x16, 16);
    avx2_unscaled!(put_w32, prep_w32, I16x16, 32);
    avx2_unscaled!(put_w64, prep_w64, I16x16, 64);

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn put_w32_512(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL is enabled by this wrapper.
        unsafe { unscaled_put_lanes::<I16x32, 32>(src, dst, h, taps, depth) }
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
        unsafe { unscaled_put_lanes::<I16x32, 64>(src, dst, h, taps, depth) }
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
        unsafe { unscaled_prep_lanes::<I16x32, 32>(src, dst, h, taps, depth) }
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
        unsafe { unscaled_prep_lanes::<I16x32, 64>(src, dst, h, taps, depth) }
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
        put: [put_w4, put_w8, put_w16, put_w32_512, put_w64_512],
        prep: [prep_w4, prep_w8, prep_w16, prep_w32_512, prep_w64_512],
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
        WarpPhases, scaled_prep_lanes, scaled_put_lanes, unscaled_prep_lanes, unscaled_put_lanes,
        warp8x8_u16_lanes,
    };
    use crate::av1_mc::high_blend::neon::NEON_BLENDS;
    use crate::simd::neon::{I16x4, I16x8};

    macro_rules! neon_unscaled {
        ($put:ident, $prep:ident, $v:ty, $w:literal) => {
            #[target_feature(enable = "neon")]
            unsafe fn $put(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: Av1Taps<'_>,
                depth: BitDepth,
            ) {
                // SAFETY: NEON is enabled by this wrapper.
                unsafe { unscaled_put_lanes::<$v, $w>(src, dst, h, taps, depth) }
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
                unsafe { unscaled_prep_lanes::<$v, $w>(src, dst, h, taps, depth) }
            }
        };
    }

    neon_unscaled!(put_w4, prep_w4, I16x4, 4);
    neon_unscaled!(put_w8, prep_w8, I16x8, 8);
    neon_unscaled!(put_w16, prep_w16, I16x8, 16);
    neon_unscaled!(put_w32, prep_w32, I16x8, 32);
    neon_unscaled!(put_w64, prep_w64, I16x8, 64);

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
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use super::{
        Av1Filters, Av1McAxis, Av1Taps, BitDepth, HighAv1McFns, PlaneView, PlaneViewMut, WarpOut,
        WarpPhases, scaled_prep_lanes, scaled_put_lanes, unscaled_prep_lanes, unscaled_put_lanes,
        warp8x8_u16_lanes,
    };
    use crate::av1_mc::high_blend::simd128::SIMD128_BLENDS;
    use crate::simd::simd128::{I16x4, I16x8};

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
                unsafe { unscaled_put_lanes::<$v, $w>(src, dst, h, taps, depth) }
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
                unsafe { unscaled_prep_lanes::<$v, $w>(src, dst, h, taps, depth) }
            }
        };
    }

    simd128_unscaled!(put_w4, prep_w4, I16x4, 4);
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
