//! AV1 compound blends, difference-weighted masks, inter-intra blends, and
//! OBMC blends shared across `u8` (8 bpc) and `u16` (10 and 12 bpc), on AVX2,
//! AVX-512, NEON, and SIMD128.
//!
//! Attribution: paired 16-bit weight-prediction products (`pmaddwd` / NEON
//! widening MACs) and unsigned difference-mask rounding follow dav1d 1.5.4
//! (`src/mc_tmpl.c`, `src/x86/mc16_avx2.asm`, `src/x86/mc16_avx512.asm`,
//! `src/arm/64/mc16.S`, BSD-2-Clause; see `THIRD_PARTY.md`).

#![allow(unsafe_code, clippy::inline_always)]
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, unused_imports)
)]

use nc_common::{BitDepth, PlaneView, PlaneViewMut, Sample};

use super::blend::{
    AV1_MASK_BITS, AV1_MASK_MAX, Av1CompoundPred, Av1Mask, Av1ObmcEdge, av1_blend_average_at,
    av1_blend_distance_at, av1_blend_interintra, av1_blend_mask_at, av1_obmc_blend, obmc_mask,
};
use super::mask::av1_difference_weight_mask_at;
use super::{av1_compound_bias, inter_post_round, round2};
use crate::simd::I16Lanes;

const MAX_BLOCK: usize = 128;
const DISTANCE_BITS: u32 = 4;
const DIFFERENCE_BASE: i16 = 38;

pub(crate) type BlendAvgFn<S> = unsafe fn(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
);

pub(crate) type BlendDistFn<S> = unsafe fn(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    weights: [u8; 2],
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
);

pub(crate) type BlendMaskFn<S> = unsafe fn(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    mask: Av1Mask<'_>,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
);

pub(crate) type DiffMaskFn = unsafe fn(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    inverse: bool,
    size: (usize, usize),
    mask: &mut [u8],
    depth: BitDepth,
);

pub(crate) type BlendInterIntraFn<S> = unsafe fn(
    intra: &PlaneView<'_, S>,
    mask: Av1Mask<'_>,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
);

pub(crate) type BlendObmcFn<S> = unsafe fn(
    obmc: &PlaneView<'_, S>,
    edge: Av1ObmcEdge,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
);

#[derive(Debug, Clone, Copy)]
pub(crate) struct HighBlendFns {
    pub average: BlendAvgFn<u16>,
    pub distance: BlendDistFn<u16>,
    pub mask: BlendMaskFn<u16>,
    pub diff_mask: DiffMaskFn,
    pub interintra: BlendInterIntraFn<u16>,
    pub obmc: BlendObmcFn<u16>,
    pub distance_u8: BlendDistFn<u8>,
    pub mask_u8: BlendMaskFn<u8>,
    pub interintra_u8: BlendInterIntraFn<u8>,
    pub obmc_u8: BlendObmcFn<u8>,
}

pub(crate) const SCALAR_BLENDS: HighBlendFns = HighBlendFns {
    average: scalar_avg::<u16>,
    distance: scalar_dist::<u16>,
    mask: scalar_mask::<u16>,
    diff_mask: scalar_diff_mask,
    interintra: scalar_interintra::<u16>,
    obmc: scalar_obmc::<u16>,
    distance_u8: scalar_dist::<u8>,
    mask_u8: scalar_mask::<u8>,
    interintra_u8: scalar_interintra::<u8>,
    obmc_u8: scalar_obmc::<u8>,
};

fn scalar_avg<S: Sample>(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
) {
    av1_blend_average_at(p0, p1, size, dst, depth);
}

fn scalar_dist<S: Sample>(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    weights: [u8; 2],
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
) {
    av1_blend_distance_at(p0, p1, weights, size, dst, depth);
}

fn scalar_mask<S: Sample>(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    mask: Av1Mask<'_>,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
) {
    av1_blend_mask_at(p0, p1, mask, size, dst, depth);
}

fn scalar_diff_mask(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    inverse: bool,
    size: (usize, usize),
    mask: &mut [u8],
    depth: BitDepth,
) {
    av1_difference_weight_mask_at(p0, p1, inverse, size, mask, depth);
}

fn scalar_interintra<S: Sample>(
    intra: &PlaneView<'_, S>,
    mask: Av1Mask<'_>,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
) {
    av1_blend_interintra(intra, mask, size, dst);
}

fn scalar_obmc<S: Sample>(
    obmc: &PlaneView<'_, S>,
    edge: Av1ObmcEdge,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
) {
    av1_obmc_blend(obmc, edge, size, dst);
}

/// Sample load/store adapter so `u8` and `u16` share one set of SIMD blend drivers.
pub(crate) trait BlendSample: Sample {
    unsafe fn load_i16<V: I16Lanes>(src: &[Self]) -> V;
    unsafe fn store_i16<V: I16Lanes>(v: V, dst: &mut [Self]);
    unsafe fn clamp_and_store<V: I16Lanes>(v: V, zero: V, hi: V, dst: &mut [Self]);
}

impl BlendSample for u8 {
    #[inline(always)]
    unsafe fn load_i16<V: I16Lanes>(src: &[u8]) -> V {
        V::load_u8(src)
    }

    #[inline(always)]
    unsafe fn store_i16<V: I16Lanes>(v: V, dst: &mut [u8]) {
        v.store_u8_sat(dst);
    }

    #[inline(always)]
    unsafe fn clamp_and_store<V: I16Lanes>(v: V, _zero: V, _hi: V, dst: &mut [u8]) {
        // `store_u8_sat` (`packuswb` / `vqmovun`) saturates `i16 -> 0..=255` directly.
        v.store_u8_sat(dst);
    }
}

impl BlendSample for u16 {
    #[inline(always)]
    unsafe fn load_i16<V: I16Lanes>(src: &[u16]) -> V {
        let src = &src[..V::LANES];
        // SAFETY: `src` holds `V::LANES` `u16` samples (`0..=4095`, positive `i16`).
        V::load(unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<i16>(), V::LANES) })
    }

    #[inline(always)]
    unsafe fn store_i16<V: I16Lanes>(v: V, dst: &mut [u16]) {
        let dst = &mut dst[..V::LANES];
        // SAFETY: `dst` holds `V::LANES` writable `u16` words.
        v.store(unsafe {
            core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<i16>(), V::LANES)
        });
    }

    #[inline(always)]
    unsafe fn clamp_and_store<V: I16Lanes>(v: V, zero: V, hi: V, dst: &mut [u16]) {
        // SAFETY: forwarded caller contract.
        unsafe { Self::store_i16(v.max(zero).min(hi), dst) };
    }
}

#[inline(always)]
unsafe fn avg_step<V: I16Lanes>(
    p0: *const i16,
    p1: *const i16,
    dst: *mut u16,
    rnd: V,
    zero: V,
    shift: u32,
) {
    // SAFETY: caller validated `V::LANES` readable `i16`s and writable `u16`s.
    unsafe {
        let a = V::load(core::slice::from_raw_parts(p0, V::LANES));
        let b = V::load(core::slice::from_raw_parts(p1, V::LANES));
        let out = a.add_sat(b).add_sat(rnd).max(zero).sra(shift);
        out.store(core::slice::from_raw_parts_mut(dst.cast::<i16>(), V::LANES));
    }
}

/// Fast 16-bit saturating compound average (`(p0 + p1 + 2 * 8192 + round) >> shift`):
/// `i16::MAX >> shift` equals `1023` at 10-bit (`shift = 5`) and `4095` at 12-bit
/// (`shift = 3`), so two `add_sat` + `max(0)` + `sra(shift)` is exact across
/// every `i16` input pair without widening to 32 bits.
#[inline(always)]
unsafe fn average_u16_lanes<V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_, u16>,
    depth: BitDepth,
) {
    let shift = 1 + inter_post_round(depth);
    if depth.bits() != 10 && depth.bits() != 12 {
        // SAFETY: forwarded ISA contract.
        unsafe {
            blend_weighted_lanes::<u16, V4, V8, VWide>(p0, p1, (1, 1), shift, (w, h), dst, depth);
        }
        return;
    }
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 blend destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    if w == 0 || h == 0 {
        return;
    }
    assert!(
        p0.data.len() >= (h - 1) * p0.stride + w && p1.data.len() >= (h - 1) * p1.stride + w,
        "AV1 compound predictions smaller than {w}x{h}"
    );
    let rnd_scalar = (16384 + (1_i32 << (shift - 1))) as i16;
    let (rnd_w, zero_w) = (VWide::splat(rnd_scalar), VWide::splat(0));
    let (rnd_8, zero_8) = (V8::splat(rnd_scalar), V8::splat(0));
    let (rnd_4, zero_4) = (V4::splat(rnd_scalar), V4::splat(0));
    let ds = dst.stride();
    let (p0_base, p1_base, dst_base) = (
        p0.data.as_ptr(),
        p1.data.as_ptr(),
        dst.data_mut().as_mut_ptr(),
    );
    for r in 0..h {
        // SAFETY: validated `(h - 1) * stride + w` spans above.
        let (r0, r1, out) = unsafe {
            (
                p0_base.add(r * p0.stride),
                p1_base.add(r * p1.stride),
                dst_base.add(r * ds),
            )
        };
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            // SAFETY: `c0 + VWide::LANES <= w`.
            unsafe { avg_step(r0.add(c0), r1.add(c0), out.add(c0), rnd_w, zero_w, shift) };
            c0 += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            // SAFETY: `c0 + V8::LANES <= w`.
            unsafe { avg_step(r0.add(c0), r1.add(c0), out.add(c0), rnd_8, zero_8, shift) };
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            // SAFETY: `c0 + V4::LANES <= w`.
            unsafe { avg_step(r0.add(c0), r1.add(c0), out.add(c0), rnd_4, zero_4, shift) };
            c0 += V4::LANES;
        }
        for c in c0..w {
            // SAFETY: `c < w`.
            unsafe {
                let sum = i32::from(*r0.add(c)) + i32::from(*r1.add(c)) + 16384;
                *out.add(c) = depth.clip_to(round2(sum, shift));
            }
        }
    }
}

/// Vectorized `w0 * p0 + w1 * p1 + (w0 + w1) * bias`, rounded by `shift` and
/// clipped to `depth`: shared by compound average (`[1, 1]`) and distance
/// average (`[w0, w1]`).
#[inline(always)]
unsafe fn blend_weighted_lanes<S: BlendSample, V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    (w0, w1): (i16, i16),
    shift: u32,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
) {
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 blend destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    let bias_total = (i32::from(w0) + i32::from(w1)) * av1_compound_bias(depth);
    let round = bias_total + (1 << (shift - 1));
    let max_val = depth.max() as i16;
    let (zero_w, hi_w) = (VWide::splat(0), VWide::splat(max_val));
    let (zero_8, hi_8) = (V8::splat(0), V8::splat(max_val));
    let (zero_4, hi_4) = (V4::splat(0), V4::splat(max_val));
    for r in 0..h {
        let row0 = &p0.data[r * p0.stride..][..w];
        let row1 = &p1.data[r * p1.stride..][..w];
        let out = &mut dst.row_mut(r)[..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let acc = VWide::acc_mul_add2(
                VWide::acc_zero(),
                VWide::load(&row0[c0..]),
                w0,
                VWide::load(&row1[c0..]),
                w1,
            );
            let narrowed = VWide::acc_round_shift_narrow(acc, round, shift);
            // SAFETY: `c0 + VWide::LANES <= w`.
            unsafe { S::clamp_and_store(narrowed, zero_w, hi_w, &mut out[c0..]) };
            c0 += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let acc = V8::acc_mul_add2(
                V8::acc_zero(),
                V8::load(&row0[c0..]),
                w0,
                V8::load(&row1[c0..]),
                w1,
            );
            let narrowed = V8::acc_round_shift_narrow(acc, round, shift);
            // SAFETY: `c0 + V8::LANES <= w`.
            unsafe { S::clamp_and_store(narrowed, zero_8, hi_8, &mut out[c0..]) };
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let acc = V4::acc_mul_add2(
                V4::acc_zero(),
                V4::load(&row0[c0..]),
                w0,
                V4::load(&row1[c0..]),
                w1,
            );
            let narrowed = V4::acc_round_shift_narrow(acc, round, shift);
            // SAFETY: `c0 + V4::LANES <= w`.
            unsafe { S::clamp_and_store(narrowed, zero_4, hi_4, &mut out[c0..]) };
            c0 += V4::LANES;
        }
        for c in c0..w {
            let sum = i32::from(w0) * i32::from(row0[c])
                + i32::from(w1) * i32::from(row1[c])
                + bias_total;
            out[c] = depth.clip_to(round2(sum, shift));
        }
    }
}

/// Extract row `row` of `mask` into `out` (`w` weights in `0..=64`), using
/// SIMD byte widening / averaging on full-rate and vertically subsampled masks.
#[inline(always)]
fn mask_weights_row<V8: I16Lanes, VWide: I16Lanes>(
    mask: &Av1Mask<'_>,
    row: usize,
    out: &mut [i16],
) {
    let n = out.len();
    match (mask.sub_x, mask.sub_y) {
        (false, false) => {
            let src = &mask.data[row * mask.stride..][..n];
            let mut c0 = 0usize;
            while c0 + VWide::LANES <= n {
                VWide::load_u8(&src[c0..]).store(&mut out[c0..]);
                c0 += VWide::LANES;
            }
            if V8::LANES < VWide::LANES && c0 + V8::LANES <= n {
                V8::load_u8(&src[c0..]).store(&mut out[c0..]);
                c0 += V8::LANES;
            }
            for c in c0..n {
                out[c] = i16::from(src[c]);
            }
        }
        (false, true) => {
            let top = &mask.data[(2 * row) * mask.stride..][..n];
            let bot = &mask.data[(2 * row + 1) * mask.stride..][..n];
            let mut c0 = 0usize;
            while c0 + VWide::LANES <= n {
                VWide::mul_add_u8_pair(&top[c0..], &bot[c0..], 1, 1)
                    .add(VWide::splat(1))
                    .sra(1)
                    .store(&mut out[c0..]);
                c0 += VWide::LANES;
            }
            if V8::LANES < VWide::LANES && c0 + V8::LANES <= n {
                V8::mul_add_u8_pair(&top[c0..], &bot[c0..], 1, 1)
                    .add(V8::splat(1))
                    .sra(1)
                    .store(&mut out[c0..]);
                c0 += V8::LANES;
            }
            for c in c0..n {
                out[c] = ((u16::from(top[c]) + u16::from(bot[c]) + 1) >> 1) as i16;
            }
        }
        _ => mask.weights_row(row, out),
    }
}

#[inline(always)]
unsafe fn blend_mask_lanes<S: BlendSample, V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    mask: Av1Mask<'_>,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
    depth: BitDepth,
) {
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 blend destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    let max_m = i32::from(AV1_MASK_MAX);
    let shift = AV1_MASK_BITS + inter_post_round(depth);
    let bias = max_m * av1_compound_bias(depth);
    let round = bias + (1 << (shift - 1));
    let max_val = depth.max() as i16;
    let (zero_w, hi_w, m64_w) = (VWide::splat(0), VWide::splat(max_val), VWide::splat(64));
    let (zero_8, hi_8, m64_8) = (V8::splat(0), V8::splat(max_val), V8::splat(64));
    let (zero_4, hi_4, m64_4) = (V4::splat(0), V4::splat(max_val), V4::splat(64));
    let mut weights_buf = [0i16; MAX_BLOCK];
    let weights = &mut weights_buf[..w];
    for r in 0..h {
        mask_weights_row::<V8, VWide>(&mask, r, weights);
        let row0 = &p0.data[r * p0.stride..][..w];
        let row1 = &p1.data[r * p1.stride..][..w];
        let out = &mut dst.row_mut(r)[..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let wt = VWide::load(&weights[c0..]);
            let acc = VWide::acc_vec_mul_add2(
                VWide::load(&row0[c0..]),
                wt,
                VWide::load(&row1[c0..]),
                m64_w.sub(wt),
            );
            let narrowed = VWide::acc_round_shift_narrow(acc, round, shift);
            // SAFETY: `c0 + VWide::LANES <= w`.
            unsafe { S::clamp_and_store(narrowed, zero_w, hi_w, &mut out[c0..]) };
            c0 += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let wt = V8::load(&weights[c0..]);
            let acc = V8::acc_vec_mul_add2(
                V8::load(&row0[c0..]),
                wt,
                V8::load(&row1[c0..]),
                m64_8.sub(wt),
            );
            let narrowed = V8::acc_round_shift_narrow(acc, round, shift);
            // SAFETY: `c0 + V8::LANES <= w`.
            unsafe { S::clamp_and_store(narrowed, zero_8, hi_8, &mut out[c0..]) };
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let wt = V4::load(&weights[c0..]);
            let acc = V4::acc_vec_mul_add2(
                V4::load(&row0[c0..]),
                wt,
                V4::load(&row1[c0..]),
                m64_4.sub(wt),
            );
            let narrowed = V4::acc_round_shift_narrow(acc, round, shift);
            // SAFETY: `c0 + V4::LANES <= w`.
            unsafe { S::clamp_and_store(narrowed, zero_4, hi_4, &mut out[c0..]) };
            c0 += V4::LANES;
        }
        for c in c0..w {
            let wt = i32::from(weights[c]);
            let sum = wt * i32::from(row0[c]) + (max_m - wt) * i32::from(row1[c]) + bias;
            out[c] = depth.clip_to(round2(sum, shift));
        }
    }
}

#[inline(always)]
fn diff_mask_step<V: I16Lanes>(x: V, y: V, shift: u32, inverse: bool) -> V {
    debug_assert!(shift >= 2);
    // Exact unsigned 16-bit `|x - y|` in `0..=65535` via `max(x, y) - min(x, y)`.
    // For all `u >= 0`, `Round2(u, shift) / 16 == ((u >> (shift - 1)) + 1) >> 5`.
    let abs_u16 = x.max(y).sub(x.min(y));
    let div16 = abs_u16.srl(shift - 1).add(V::splat(1)).srl(5);
    let wt = div16.add(V::splat(DIFFERENCE_BASE)).min(V::splat(64));
    if inverse { V::splat(64).sub(wt) } else { wt }
}

#[inline(always)]
unsafe fn diff_mask_lanes<V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    inverse: bool,
    (w, h): (usize, usize),
    mask: &mut [u8],
    depth: BitDepth,
) {
    let shift = depth.above_eight() + inter_post_round(depth);
    let max_m = i32::from(AV1_MASK_MAX);
    for (r, row) in mask[..w * h].chunks_exact_mut(w).enumerate() {
        let row0 = &p0.data[r * p0.stride..][..w];
        let row1 = &p1.data[r * p1.stride..][..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            diff_mask_step(
                VWide::load(&row0[c0..]),
                VWide::load(&row1[c0..]),
                shift,
                inverse,
            )
            .store_u8_sat(&mut row[c0..]);
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            diff_mask_step(V8::load(&row0[c0..]), V8::load(&row1[c0..]), shift, inverse)
                .store_u8_sat(&mut row[c0..]);
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            diff_mask_step(V4::load(&row0[c0..]), V4::load(&row1[c0..]), shift, inverse)
                .store_u8_sat(&mut row[c0..]);
            c0 += V4::LANES;
        }
        for c in c0..w {
            let diff = round2((i32::from(row0[c]) - i32::from(row1[c])).abs(), shift);
            let wt = (i32::from(DIFFERENCE_BASE) + diff / 16).clamp(0, max_m);
            row[c] = if inverse { max_m - wt } else { wt } as u8;
        }
    }
}

#[inline(always)]
unsafe fn interintra_lanes<S: BlendSample, V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    intra: &PlaneView<'_, S>,
    mask: Av1Mask<'_>,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
) {
    assert!(
        dst.width() >= w && dst.height() >= h && intra.width() >= w && intra.height() >= h,
        "AV1 blend of {w}x{h}: inputs too small"
    );
    let round = 1i32 << (AV1_MASK_BITS - 1);
    let (m64_w, m64_8, m64_4) = (VWide::splat(64), V8::splat(64), V4::splat(64));
    let mut weights_buf = [0i16; MAX_BLOCK];
    let weights = &mut weights_buf[..w];
    for r in 0..h {
        mask_weights_row::<V8, VWide>(&mask, r, weights);
        let src = &intra.row(r)[..w];
        let out = &mut dst.row_mut(r)[..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let wt = VWide::load(&weights[c0..]);
            // SAFETY: `c0 + VWide::LANES <= w`.
            unsafe {
                let (src_v, dst_v) = (
                    S::load_i16::<VWide>(&src[c0..]),
                    S::load_i16::<VWide>(&out[c0..]),
                );
                let acc = VWide::acc_vec_mul_add2(src_v, wt, dst_v, m64_w.sub(wt));
                S::store_i16(
                    VWide::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let wt = V8::load(&weights[c0..]);
            // SAFETY: `c0 + V8::LANES <= w`.
            unsafe {
                let (src_v, dst_v) = (S::load_i16::<V8>(&src[c0..]), S::load_i16::<V8>(&out[c0..]));
                let acc = V8::acc_vec_mul_add2(src_v, wt, dst_v, m64_8.sub(wt));
                S::store_i16(
                    V8::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let wt = V4::load(&weights[c0..]);
            // SAFETY: `c0 + V4::LANES <= w`.
            unsafe {
                let (src_v, dst_v) = (S::load_i16::<V4>(&src[c0..]), S::load_i16::<V4>(&out[c0..]));
                let acc = V4::acc_vec_mul_add2(src_v, wt, dst_v, m64_4.sub(wt));
                S::store_i16(
                    V4::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += V4::LANES;
        }
        for c in c0..w {
            let wt = i32::from(weights[c]);
            out[c] = S::from_i32(round2(
                wt * src[c].to_i32() + (64 - wt) * out[c].to_i32(),
                AV1_MASK_BITS,
            ));
        }
    }
}

#[inline(always)]
unsafe fn obmc_above_lanes<S: BlendSample, V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    obmc: &PlaneView<'_, S>,
    mask: &[u8],
    w: usize,
    dst: &mut PlaneViewMut<'_, S>,
) {
    let round = 1i32 << (AV1_MASK_BITS - 1);
    for (r, &m_dst) in mask.iter().enumerate() {
        if m_dst == AV1_MASK_MAX {
            continue;
        }
        let (wd, wo) = (i16::from(m_dst), 64 - i16::from(m_dst));
        let src = &obmc.row(r)[..w];
        let out = &mut dst.row_mut(r)[..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            // SAFETY: `c0 + VWide::LANES <= w`.
            unsafe {
                let (dst_v, src_v) = (
                    S::load_i16::<VWide>(&out[c0..]),
                    S::load_i16::<VWide>(&src[c0..]),
                );
                let acc = VWide::acc_mul_add2(VWide::acc_zero(), dst_v, wd, src_v, wo);
                S::store_i16(
                    VWide::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            // SAFETY: `c0 + V8::LANES <= w`.
            unsafe {
                let (dst_v, src_v) = (S::load_i16::<V8>(&out[c0..]), S::load_i16::<V8>(&src[c0..]));
                let acc = V8::acc_mul_add2(V8::acc_zero(), dst_v, wd, src_v, wo);
                S::store_i16(
                    V8::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            // SAFETY: `c0 + V4::LANES <= w`.
            unsafe {
                let (dst_v, src_v) = (S::load_i16::<V4>(&out[c0..]), S::load_i16::<V4>(&src[c0..]));
                let acc = V4::acc_mul_add2(V4::acc_zero(), dst_v, wd, src_v, wo);
                S::store_i16(
                    V4::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += V4::LANES;
        }
        for c in c0..w {
            out[c] = S::from_i32(round2(
                i32::from(wd) * out[c].to_i32() + i32::from(wo) * src[c].to_i32(),
                AV1_MASK_BITS,
            ));
        }
    }
}

#[inline(always)]
unsafe fn obmc_lanes<S: BlendSample, V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    obmc: &PlaneView<'_, S>,
    edge: Av1ObmcEdge,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_, S>,
) {
    assert!(
        dst.width() >= w && dst.height() >= h && obmc.width() >= w && obmc.height() >= h,
        "AV1 OBMC blend of {w}x{h}: inputs too small"
    );
    let mask = obmc_mask(if edge == Av1ObmcEdge::Above { h } else { w });
    if edge == Av1ObmcEdge::Above {
        // SAFETY: forwarded ISA and validated plane dimensions.
        unsafe { obmc_above_lanes::<S, V4, V8, VWide>(obmc, mask, w, dst) };
        return;
    }
    let round = 1i32 << (AV1_MASK_BITS - 1);
    let mut wd_row = [0i16; 32];
    for (o, &m) in wd_row[..w].iter_mut().zip(mask) {
        *o = i16::from(m);
    }
    let (m64_w, m64_8, m64_4) = (VWide::splat(64), V8::splat(64), V4::splat(64));
    for r in 0..h {
        let src = &obmc.row(r)[..w];
        let out = &mut dst.row_mut(r)[..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let wd = VWide::load(&wd_row[c0..]);
            // SAFETY: `c0 + VWide::LANES <= w`.
            unsafe {
                let (dst_v, src_v) = (
                    S::load_i16::<VWide>(&out[c0..]),
                    S::load_i16::<VWide>(&src[c0..]),
                );
                let acc = VWide::acc_vec_mul_add2(dst_v, wd, src_v, m64_w.sub(wd));
                S::store_i16(
                    VWide::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let wd = V8::load(&wd_row[c0..]);
            // SAFETY: `c0 + V8::LANES <= w`.
            unsafe {
                let (dst_v, src_v) = (S::load_i16::<V8>(&out[c0..]), S::load_i16::<V8>(&src[c0..]));
                let acc = V8::acc_vec_mul_add2(dst_v, wd, src_v, m64_8.sub(wd));
                S::store_i16(
                    V8::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let wd = V4::load(&wd_row[c0..]);
            // SAFETY: `c0 + V4::LANES <= w`.
            unsafe {
                let (dst_v, src_v) = (S::load_i16::<V4>(&out[c0..]), S::load_i16::<V4>(&src[c0..]));
                let acc = V4::acc_vec_mul_add2(dst_v, wd, src_v, m64_4.sub(wd));
                S::store_i16(
                    V4::acc_round_shift_narrow(acc, round, AV1_MASK_BITS),
                    &mut out[c0..],
                );
            }
            c0 += V4::LANES;
        }
        for c in c0..w {
            let wd = i32::from(wd_row[c]);
            out[c] = S::from_i32(round2(
                wd * out[c].to_i32() + (64 - wd) * src[c].to_i32(),
                AV1_MASK_BITS,
            ));
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! isa_blends {
    ($mod_name:ident, $const_name:ident, $feat:literal, $v4:ty, $v8:ty, $vw:ty) => {
        pub(super) mod $mod_name {
            use super::super::{
                Av1CompoundPred, Av1Mask, Av1ObmcEdge, BitDepth, BlendSample, DISTANCE_BITS,
                HighBlendFns, PlaneView, PlaneViewMut, average_u16_lanes, blend_mask_lanes,
                blend_weighted_lanes, diff_mask_lanes, inter_post_round, interintra_lanes,
                obmc_lanes,
            };

            #[target_feature(enable = $feat)]
            unsafe fn average(
                p0: Av1CompoundPred<'_>,
                p1: Av1CompoundPred<'_>,
                size: (usize, usize),
                dst: &mut PlaneViewMut<'_, u16>,
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper.
                unsafe { average_u16_lanes::<$v4, $v8, $vw>(p0, p1, size, dst, depth) }
            }

            #[target_feature(enable = $feat)]
            unsafe fn distance<S: BlendSample>(
                p0: Av1CompoundPred<'_>,
                p1: Av1CompoundPred<'_>,
                [w0, w1]: [u8; 2],
                size: (usize, usize),
                dst: &mut PlaneViewMut<'_, S>,
                depth: BitDepth,
            ) {
                let shift = DISTANCE_BITS + inter_post_round(depth);
                // SAFETY: target feature enabled by this wrapper.
                unsafe {
                    blend_weighted_lanes::<S, $v4, $v8, $vw>(
                        p0,
                        p1,
                        (i16::from(w0), i16::from(w1)),
                        shift,
                        size,
                        dst,
                        depth,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn mask_fn<S: BlendSample>(
                p0: Av1CompoundPred<'_>,
                p1: Av1CompoundPred<'_>,
                mask: Av1Mask<'_>,
                size: (usize, usize),
                dst: &mut PlaneViewMut<'_, S>,
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper.
                unsafe { blend_mask_lanes::<S, $v4, $v8, $vw>(p0, p1, mask, size, dst, depth) }
            }

            #[target_feature(enable = $feat)]
            unsafe fn diff_mask(
                p0: Av1CompoundPred<'_>,
                p1: Av1CompoundPred<'_>,
                inverse: bool,
                size: (usize, usize),
                mask: &mut [u8],
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper.
                unsafe { diff_mask_lanes::<$v4, $v8, $vw>(p0, p1, inverse, size, mask, depth) }
            }

            #[target_feature(enable = $feat)]
            unsafe fn interintra<S: BlendSample>(
                intra: &PlaneView<'_, S>,
                mask: Av1Mask<'_>,
                size: (usize, usize),
                dst: &mut PlaneViewMut<'_, S>,
            ) {
                // SAFETY: target feature enabled by this wrapper.
                unsafe { interintra_lanes::<S, $v4, $v8, $vw>(intra, mask, size, dst) }
            }

            #[target_feature(enable = $feat)]
            unsafe fn obmc<S: BlendSample>(
                obmc: &PlaneView<'_, S>,
                edge: Av1ObmcEdge,
                size: (usize, usize),
                dst: &mut PlaneViewMut<'_, S>,
            ) {
                // SAFETY: target feature enabled by this wrapper.
                unsafe { obmc_lanes::<S, $v4, $v8, $vw>(obmc, edge, size, dst) }
            }

            pub(crate) const $const_name: HighBlendFns = HighBlendFns {
                average,
                distance: distance::<u16>,
                mask: mask_fn::<u16>,
                diff_mask,
                interintra: interintra::<u16>,
                obmc: obmc::<u16>,
                distance_u8: distance::<u8>,
                mask_u8: mask_fn::<u8>,
                interintra_u8: interintra::<u8>,
                obmc_u8: obmc::<u8>,
            };
        }
    };
}

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    isa_blends!(
        avx2_impl,
        AVX2_BLENDS,
        "avx2",
        crate::simd::avx2::I16x4,
        crate::simd::avx2::I16x8,
        crate::simd::avx2::I16x16
    );
    isa_blends!(
        avx512_impl,
        AVX512_BLENDS,
        "avx512f,avx512bw,avx512vl",
        crate::simd::avx2::I16x4,
        crate::simd::avx2::I16x8,
        crate::simd::avx512_i16::I16x32
    );
    pub(crate) use avx2_impl::AVX2_BLENDS;
    pub(crate) use avx512_impl::AVX512_BLENDS;
}

#[cfg(target_arch = "aarch64")]
pub(super) mod neon {
    isa_blends!(
        neon_impl,
        NEON_BLENDS,
        "neon",
        crate::simd::neon::I16x4,
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x8
    );
    pub(crate) use neon_impl::NEON_BLENDS;
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) mod simd128 {
    isa_blends!(
        simd128_impl,
        SIMD128_BLENDS,
        "simd128",
        crate::simd::simd128::I16x4,
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x8
    );
    // Preserve the existing u8 function table, including the shared mask.
    pub(crate) const SIMD128_BLENDS: super::HighBlendFns = super::HighBlendFns {
        diff_mask: super::SCALAR_BLENDS.diff_mask,
        distance_u8: super::SCALAR_BLENDS.distance_u8,
        mask_u8: super::SCALAR_BLENDS.mask_u8,
        interintra_u8: super::SCALAR_BLENDS.interintra_u8,
        obmc_u8: super::SCALAR_BLENDS.obmc_u8,
        ..simd128_impl::SIMD128_BLENDS
    };
}
