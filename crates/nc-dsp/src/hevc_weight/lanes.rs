//! The weighted prediction kernel written once over [`I16Lanes`] /
//! [`I16FilterLanes`]: widest vectors first (16 lanes on AVX2, 32 on AVX-512),
//! then 8 lanes, then 4, then the scalar reference for narrow chroma tails
//! (widths 2 or 6).
//!
//! Unweighted and equal-power-of-two weighted cases (uni-prediction and
//! default bi-prediction, both uncentered and HEVC-centered) stay in 16-bit
//! lanes via rounding Q15 multiply-high (`mul_q15_round`), avoiding `i32`
//! widening/packing and skipping the unused second source on uni-prediction.

use nc_common::BitDepth;

use super::HevcWeights;
use super::reference;
use crate::simd::{I16FilterLanes, I16Lanes};

/// Q15 exponent of [`I16Lanes::mul_q15_round`] (`(a * b + 2^14) >> 15`).
const Q15_BITS: u32 = 15;
/// Bounds on post-shift integer offset so `q + offset` (`|q| <= 16384`) never
/// saturates in `i16::add_sat`.
const MIN_FAST_OFFSET: i64 = -16_384;
const MAX_FAST_OFFSET: i64 = 16_383;

#[derive(Clone, Copy)]
enum FastMode {
    Uni {
        second: bool,
        scale: i16,
        offset: i16,
    },
    BiEqual {
        scale: i16,
        offset: i16,
    },
    General,
}

#[inline]
fn classify(weights: &HevcWeights) -> FastMode {
    let shift = weights.shift;
    if (1..Q15_BITS).contains(&shift) && weights.round == (1i32 << (shift - 1)) {
        if weights.w0 == 1 && weights.w1 == 1 && shift >= 2 {
            return FastMode::BiEqual {
                scale: 1i16 << (Q15_BITS + 1 - shift),
                offset: 0,
            };
        }
        if weights.w0 == 1 && weights.w1 == 0 {
            return FastMode::Uni {
                second: false,
                scale: 1i16 << (Q15_BITS - shift),
                offset: 0,
            };
        }
        if weights.w0 == 0 && weights.w1 == 1 {
            return FastMode::Uni {
                second: true,
                scale: 1i16 << (Q15_BITS - shift),
                offset: 0,
            };
        }
    }
    classify_extended(weights)
}

#[cold]
fn classify_extended(weights: &HevcWeights) -> FastMode {
    let shift = weights.shift;
    if !(1..Q15_BITS).contains(&shift) {
        return FastMode::General;
    }
    let half = 1i64 << (shift - 1);
    let step_mask = (1i64 << shift) - 1;
    let rem = i64::from(weights.round) - half;
    if rem & step_mask != 0 {
        return FastMode::General;
    }
    let offset = rem >> shift;
    if !(MIN_FAST_OFFSET..=MAX_FAST_OFFSET).contains(&offset) {
        return FastMode::General;
    }
    let offset = offset as i16;
    if (weights.w1 == 0 && weights.w0 > 0 && (weights.w0 as u16).is_power_of_two())
        || (weights.w0 == 0 && weights.w1 > 0 && (weights.w1 as u16).is_power_of_two())
    {
        let (active, second) = if weights.w1 == 0 {
            (weights.w0 as u16, false)
        } else {
            (weights.w1 as u16, true)
        };
        let exp = active.trailing_zeros();
        if shift > exp {
            let scale = 1i16 << (Q15_BITS - (shift - exp));
            return FastMode::Uni {
                second,
                scale,
                offset,
            };
        }
    }
    if weights.w0 == weights.w1 && weights.w0 > 0 && (weights.w0 as u16).is_power_of_two() {
        let exp = (weights.w0 as u16).trailing_zeros();
        if shift >= exp + 2 {
            let eff_shift = shift - exp;
            let scale = 1i16 << (Q15_BITS + 1 - eff_shift);
            return FastMode::BiEqual { scale, offset };
        }
    }
    FastMode::General
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn floor_avg_filter<V: I16FilterLanes>(a: V, b: V) -> V {
    a.sra(1).add(b.sra(1)).add(a.and(b).and(V::splat(1)))
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn floor_avg_lanes<V: I16Lanes>(a: V, b: V) -> V {
    let ha = a.sra(1);
    let hb = b.sra(1);
    let odd_a = a.sub(ha.add(ha));
    let odd_b = b.sub(hb.add(hb));
    ha.add(hb).add(odd_a.mul(odd_b))
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn uni_vec<V: I16Lanes>(src: &[i16], scale: V, offset: V) -> V {
    V::load(src).mul_q15_round(scale).add_sat(offset)
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn row_uni_u8<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    weights: &HevcWeights,
    src: &[i16],
    dst: &mut [u8],
    scale: i16,
    offset: i16,
) {
    let w = dst.len();
    let (sw, sv, sh) = (W::splat(scale), V::splat(scale), H::splat(scale));
    let (ow, ov, oh) = (W::splat(offset), V::splat(offset), H::splat(offset));
    let mut x = 0;
    while x + W::LANES <= w {
        uni_vec::<W>(&src[x..], sw, ow).store_u8_sat(&mut dst[x..]);
        x += W::LANES;
    }
    while x + V::LANES <= w {
        uni_vec::<V>(&src[x..], sv, ov).store_u8_sat(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        uni_vec::<H>(&src[x..], sh, oh).store_u8_sat(&mut dst[x..]);
        x += H::LANES;
    }
    if x < w {
        reference::row(
            weights,
            &src[x..w],
            &src[x..w],
            &mut dst[x..],
            BitDepth::EIGHT,
        );
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn row_bi_u8<W: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    dst: &mut [u8],
    scale: i16,
    offset: i16,
) {
    let w = dst.len();
    let (sw, sv, sh) = (W::splat(scale), V::splat(scale), H::splat(scale));
    let (ow, ov, oh) = (W::splat(offset), V::splat(offset), H::splat(offset));
    let mut x = 0;
    while x + W::LANES <= w {
        floor_avg_filter(W::load(&a[x..]), W::load(&b[x..]))
            .mul_q15_round(sw)
            .add_sat(ow)
            .store_u8_sat(&mut dst[x..]);
        x += W::LANES;
    }
    while x + V::LANES <= w {
        floor_avg_filter(V::load(&a[x..]), V::load(&b[x..]))
            .mul_q15_round(sv)
            .add_sat(ov)
            .store_u8_sat(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        floor_avg_lanes(H::load(&a[x..]), H::load(&b[x..]))
            .mul_q15_round(sh)
            .add_sat(oh)
            .store_u8_sat(&mut dst[x..]);
        x += H::LANES;
    }
    if x < w {
        reference::row(weights, &a[x..w], &b[x..w], &mut dst[x..], BitDepth::EIGHT);
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn chunk<V: I16Lanes>(weights: &HevcWeights, a: &[i16], b: &[i16], dst: &mut [u8], x: usize) {
    let acc = V::acc_mul_add2(
        V::acc_zero(),
        V::load(&a[x..]),
        weights.w0,
        V::load(&b[x..]),
        weights.w1,
    );
    V::acc_round_shift_store_u8(acc, weights.round, weights.shift, &mut dst[x..]);
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn row_gen_u8<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    dst: &mut [u8],
) {
    let w = dst.len();
    let mut x = 0;
    while x + W::LANES <= w {
        chunk::<W>(weights, a, b, dst, x);
        x += W::LANES;
    }
    while x + V::LANES <= w {
        chunk::<V>(weights, a, b, dst, x);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        chunk::<H>(weights, a, b, dst, x);
        x += H::LANES;
    }
    if x < w {
        reference::row(weights, &a[x..w], &b[x..w], &mut dst[x..], BitDepth::EIGHT);
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn strided_uni_u8<W: I16Lanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    (sp, dp): (*const i16, *mut u8),
    (w, h, dst_stride): (usize, usize, usize),
    (scale, offset): (i16, i16),
) {
    let mut y = 0;
    if w == 4 && V::LANES == 8 {
        let (sv, ov) = (V::splat(scale), V::splat(offset));
        while y + 2 <= h {
            // SAFETY: `y + 2 <= h`, `w == 4`, and slices were validated by `weight`.
            unsafe {
                let s8 = core::slice::from_raw_parts(sp.add(y * 4), 8);
                let r0 = core::slice::from_raw_parts_mut(dp.add(y * dst_stride), 4);
                let r1 = core::slice::from_raw_parts_mut(dp.add((y + 1) * dst_stride), 4);
                uni_vec::<V>(s8, sv, ov).store_u8_sat_halves(r0, r1);
            }
            y += 2;
        }
    }
    while y < h {
        // SAFETY: `y < h`, validated by `weight`.
        unsafe {
            row_uni_u8::<W, V, H>(
                weights,
                core::slice::from_raw_parts(sp.add(y * w), w),
                core::slice::from_raw_parts_mut(dp.add(y * dst_stride), w),
                scale,
                offset,
            );
        }
        y += 1;
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn strided_bi_u8<W: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    (ap, bp, dp): (*const i16, *const i16, *mut u8),
    (w, h, dst_stride): (usize, usize, usize),
    (scale, offset): (i16, i16),
) {
    let mut y = 0;
    if w == 4 && V::LANES == 8 {
        let (sv, ov) = (V::splat(scale), V::splat(offset));
        while y + 2 <= h {
            // SAFETY: `y + 2 <= h`, `w == 4`, and slices were validated by `weight`.
            unsafe {
                let a8 = core::slice::from_raw_parts(ap.add(y * 4), 8);
                let b8 = core::slice::from_raw_parts(bp.add(y * 4), 8);
                let r0 = core::slice::from_raw_parts_mut(dp.add(y * dst_stride), 4);
                let r1 = core::slice::from_raw_parts_mut(dp.add((y + 1) * dst_stride), 4);
                floor_avg_filter(V::load(a8), V::load(b8))
                    .mul_q15_round(sv)
                    .add_sat(ov)
                    .store_u8_sat_halves(r0, r1);
            }
            y += 2;
        }
    }
    while y < h {
        // SAFETY: `y < h`, validated by `weight`.
        unsafe {
            row_bi_u8::<W, V, H>(
                weights,
                core::slice::from_raw_parts(ap.add(y * w), w),
                core::slice::from_raw_parts(bp.add(y * w), w),
                core::slice::from_raw_parts_mut(dp.add(y * dst_stride), w),
                scale,
                offset,
            );
        }
        y += 1;
    }
}

/// The 8-bit kernel ([`super::WeightFn`]).
///
/// # Safety
///
/// `W`, `V` and `H`'s ISA must be available; the slices cover the block.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) unsafe fn weight<W: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    if w == 0 || h == 0 {
        return;
    }
    let total = w * h;
    let (a, b) = (&a[..total], &b[..total]);
    let mode = classify(weights);
    if dst_stride == w {
        let dst = &mut dst[..total];
        match mode {
            FastMode::Uni {
                second,
                scale,
                offset,
            } => row_uni_u8::<W, V, H>(weights, if second { b } else { a }, dst, scale, offset),
            FastMode::BiEqual { scale, offset } => {
                row_bi_u8::<W, V, H>(weights, a, b, dst, scale, offset);
            }
            FastMode::General => row_gen_u8::<W, V, H>(weights, a, b, dst),
        }
        return;
    }
    let dst = &mut dst[..(h - 1) * dst_stride + w];
    let (ap, bp, dp) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    // SAFETY: `a`, `b`, and `dst` were validated above, and the caller enabled the ISA.
    unsafe {
        match mode {
            FastMode::Uni {
                second,
                scale,
                offset,
            } => strided_uni_u8::<W, V, H>(
                weights,
                (if second { bp } else { ap }, dp),
                (w, h, dst_stride),
                (scale, offset),
            ),
            FastMode::BiEqual { scale, offset } => {
                strided_bi_u8::<W, V, H>(
                    weights,
                    (ap, bp, dp),
                    (w, h, dst_stride),
                    (scale, offset),
                );
            }
            FastMode::General => {
                for y in 0..h {
                    row_gen_u8::<W, V, H>(
                        weights,
                        core::slice::from_raw_parts(ap.add(y * w), w),
                        core::slice::from_raw_parts(bp.add(y * w), w),
                        core::slice::from_raw_parts_mut(dp.add(y * dst_stride), w),
                    );
                }
            }
        }
    }
}

#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
fn row_uni_u16<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    weights: &HevcWeights,
    src: &[i16],
    dst: &mut [u16],
    depth: BitDepth,
    (scale, offset): (i16, i16),
    (max_w, max_v, max_h): (W, V, H),
) {
    let w = dst.len();
    let (zw, zv, zh) = (W::splat(0), V::splat(0), H::splat(0));
    let (sw, sv, sh) = (W::splat(scale), V::splat(scale), H::splat(scale));
    let (ow, ov, oh) = (W::splat(offset), V::splat(offset), H::splat(offset));
    let mut x = 0;
    while x + W::LANES <= w {
        uni_vec::<W>(&src[x..], sw, ow)
            .max(zw)
            .min(max_w)
            .store_u16(&mut dst[x..]);
        x += W::LANES;
    }
    while x + V::LANES <= w {
        uni_vec::<V>(&src[x..], sv, ov)
            .max(zv)
            .min(max_v)
            .store_u16(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        uni_vec::<H>(&src[x..], sh, oh)
            .max(zh)
            .min(max_h)
            .store_u16(&mut dst[x..]);
        x += H::LANES;
    }
    if x < w {
        reference::row(weights, &src[x..w], &src[x..w], &mut dst[x..], depth);
    }
}

#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
fn row_bi_u16<W: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    dst: &mut [u16],
    depth: BitDepth,
    (scale, offset): (i16, i16),
    (max_w, max_v, max_h): (W, V, H),
) {
    let w = dst.len();
    let (zw, zv, zh) = (W::splat(0), V::splat(0), H::splat(0));
    let (sw, sv, sh) = (W::splat(scale), V::splat(scale), H::splat(scale));
    let (ow, ov, oh) = (W::splat(offset), V::splat(offset), H::splat(offset));
    let mut x = 0;
    while x + W::LANES <= w {
        floor_avg_filter(W::load(&a[x..]), W::load(&b[x..]))
            .mul_q15_round(sw)
            .add_sat(ow)
            .max(zw)
            .min(max_w)
            .store_u16(&mut dst[x..]);
        x += W::LANES;
    }
    while x + V::LANES <= w {
        floor_avg_filter(V::load(&a[x..]), V::load(&b[x..]))
            .mul_q15_round(sv)
            .add_sat(ov)
            .max(zv)
            .min(max_v)
            .store_u16(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        floor_avg_lanes(H::load(&a[x..]), H::load(&b[x..]))
            .mul_q15_round(sh)
            .add_sat(oh)
            .max(zh)
            .min(max_h)
            .store_u16(&mut dst[x..]);
        x += H::LANES;
    }
    if x < w {
        reference::row(weights, &a[x..w], &b[x..w], &mut dst[x..], depth);
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn chunk_u16<V: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    dst: &mut [u16],
    x: usize,
    max_val: V,
) {
    let acc = V::acc_mul_add2(
        V::acc_zero(),
        V::load(&a[x..]),
        weights.w0,
        V::load(&b[x..]),
        weights.w1,
    );
    V::acc_round_shift_narrow(acc, weights.round, weights.shift)
        .max(V::splat(0))
        .min(max_val)
        .store_u16(&mut dst[x..]);
}

#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
fn row_gen_u16<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    dst: &mut [u16],
    depth: BitDepth,
    (max_w, max_v, max_h): (W, V, H),
) {
    let w = dst.len();
    let mut x = 0;
    while x + W::LANES <= w {
        chunk_u16::<W>(weights, a, b, dst, x, max_w);
        x += W::LANES;
    }
    while x + V::LANES <= w {
        chunk_u16::<V>(weights, a, b, dst, x, max_v);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        chunk_u16::<H>(weights, a, b, dst, x, max_h);
        x += H::LANES;
    }
    if x < w {
        reference::row(weights, &a[x..w], &b[x..w], &mut dst[x..], depth);
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn strided_w4_uni_u16<W: I16FilterLanes, V: I16FilterLanes>(
    (sp, dp): (*const i16, *mut u16),
    (h, dst_stride): (usize, usize),
    (scale, offset): (i16, i16),
    (max_w, max_v): (W, V),
) -> usize {
    let mut y = 0;
    if W::LANES == 16 {
        let (zw, sw, ow) = (W::splat(0), W::splat(scale), W::splat(offset));
        while y + 4 <= h {
            // SAFETY: `y + 4 <= h`, `w == 4`, validated by `weight_u16`.
            unsafe {
                let v = W::load_u16_at(sp.add(y * 4).cast())
                    .mul_q15_round(sw)
                    .add_sat(ow)
                    .max(zw)
                    .min(max_w);
                let r0 = core::slice::from_raw_parts_mut(dp.add(y * dst_stride).cast(), 4);
                let r1 = core::slice::from_raw_parts_mut(dp.add((y + 1) * dst_stride).cast(), 4);
                let r2 = core::slice::from_raw_parts_mut(dp.add((y + 2) * dst_stride).cast(), 4);
                let r3 = core::slice::from_raw_parts_mut(dp.add((y + 3) * dst_stride).cast(), 4);
                v.store_quarters([r0, r1, r2, r3]);
            }
            y += 4;
        }
    }
    if V::LANES == 8 {
        let (zv, sv, ov) = (V::splat(0), V::splat(scale), V::splat(offset));
        while y + 2 <= h {
            // SAFETY: `y + 2 <= h`, `w == 4`, validated by `weight_u16`.
            unsafe {
                let s8 = core::slice::from_raw_parts(sp.add(y * 4), 8);
                let r0 = core::slice::from_raw_parts_mut(dp.add(y * dst_stride).cast(), 4);
                let r1 = core::slice::from_raw_parts_mut(dp.add((y + 1) * dst_stride).cast(), 4);
                uni_vec::<V>(s8, sv, ov)
                    .max(zv)
                    .min(max_v)
                    .store_halves(r0, r1);
            }
            y += 2;
        }
    }
    y
}

#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
unsafe fn strided_uni_u16<W: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    (mut sp, mut dp): (*const i16, *mut u16),
    (w, h, dst_stride): (usize, usize, usize),
    depth: BitDepth,
    (scale, offset): (i16, i16),
    max_vecs: (W, V, H),
) {
    if w == 8 && V::LANES == 8 {
        let (zv, sv, ov) = (V::splat(0), V::splat(scale), V::splat(offset));
        for _ in 0..h {
            // SAFETY: `w == 8` and slices were validated by `weight_u16`.
            unsafe {
                V::load_u16_at(sp.cast())
                    .mul_q15_round(sv)
                    .add_sat(ov)
                    .max(zv)
                    .min(max_vecs.1)
                    .store_u16_at(dp);
                sp = sp.add(8);
                dp = dp.add(dst_stride);
            }
        }
        return;
    }
    if w >= W::LANES && w.is_multiple_of(W::LANES) && matches!(w, 16 | 32 | 64) {
        let (zw, sw, ow) = (W::splat(0), W::splat(scale), W::splat(offset));
        for _ in 0..h {
            let mut x = 0;
            while x < w {
                // SAFETY: `x + W::LANES <= w` and slices were validated by `weight_u16`.
                unsafe {
                    W::load_u16_at(sp.add(x).cast())
                        .mul_q15_round(sw)
                        .add_sat(ow)
                        .max(zw)
                        .min(max_vecs.0)
                        .store_u16_at(dp.add(x));
                }
                x += W::LANES;
            }
            // SAFETY: `h` rows validated by `weight_u16`.
            unsafe {
                sp = sp.add(w);
                dp = dp.add(dst_stride);
            }
        }
        return;
    }
    let mut y = if w == 4 {
        // SAFETY: forwarded from `weight_u16`.
        unsafe {
            strided_w4_uni_u16::<W, V>(
                (sp, dp),
                (h, dst_stride),
                (scale, offset),
                (max_vecs.0, max_vecs.1),
            )
        }
    } else {
        0
    };
    while y < h {
        // SAFETY: `y < h`, validated by `weight_u16`.
        unsafe {
            row_uni_u16::<W, V, H>(
                weights,
                core::slice::from_raw_parts(sp.add(y * w), w),
                core::slice::from_raw_parts_mut(dp.add(y * dst_stride), w),
                depth,
                (scale, offset),
                max_vecs,
            );
        }
        y += 1;
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn strided_w4_bi_u16<W: I16FilterLanes, V: I16FilterLanes>(
    (ap, bp, dp): (*const i16, *const i16, *mut u16),
    (h, dst_stride): (usize, usize),
    (scale, offset): (i16, i16),
    (max_w, max_v): (W, V),
) -> usize {
    let mut y = 0;
    if W::LANES == 16 {
        let (zw, sw, ow) = (W::splat(0), W::splat(scale), W::splat(offset));
        while y + 4 <= h {
            // SAFETY: `y + 4 <= h`, `w == 4`, validated by `weight_u16`.
            unsafe {
                let v = floor_avg_filter(
                    W::load_u16_at(ap.add(y * 4).cast()),
                    W::load_u16_at(bp.add(y * 4).cast()),
                )
                .mul_q15_round(sw)
                .add_sat(ow)
                .max(zw)
                .min(max_w);
                let r0 = core::slice::from_raw_parts_mut(dp.add(y * dst_stride).cast(), 4);
                let r1 = core::slice::from_raw_parts_mut(dp.add((y + 1) * dst_stride).cast(), 4);
                let r2 = core::slice::from_raw_parts_mut(dp.add((y + 2) * dst_stride).cast(), 4);
                let r3 = core::slice::from_raw_parts_mut(dp.add((y + 3) * dst_stride).cast(), 4);
                v.store_quarters([r0, r1, r2, r3]);
            }
            y += 4;
        }
    }
    if V::LANES == 8 {
        let (zv, sv, ov) = (V::splat(0), V::splat(scale), V::splat(offset));
        while y + 2 <= h {
            // SAFETY: `y + 2 <= h`, `w == 4`, validated by `weight_u16`.
            unsafe {
                let a8 = core::slice::from_raw_parts(ap.add(y * 4), 8);
                let b8 = core::slice::from_raw_parts(bp.add(y * 4), 8);
                let r0 = core::slice::from_raw_parts_mut(dp.add(y * dst_stride).cast(), 4);
                let r1 = core::slice::from_raw_parts_mut(dp.add((y + 1) * dst_stride).cast(), 4);
                floor_avg_filter(V::load(a8), V::load(b8))
                    .mul_q15_round(sv)
                    .add_sat(ov)
                    .max(zv)
                    .min(max_v)
                    .store_halves(r0, r1);
            }
            y += 2;
        }
    }
    y
}

#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
unsafe fn strided_bi_u16<W: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    (mut ap, mut bp, mut dp): (*const i16, *const i16, *mut u16),
    (w, h, dst_stride): (usize, usize, usize),
    depth: BitDepth,
    (scale, offset): (i16, i16),
    max_vecs: (W, V, H),
) {
    if w == 8 && V::LANES == 8 {
        let (zv, sv, ov) = (V::splat(0), V::splat(scale), V::splat(offset));
        for _ in 0..h {
            // SAFETY: `w == 8` and slices were validated by `weight_u16`.
            unsafe {
                floor_avg_filter(V::load_u16_at(ap.cast()), V::load_u16_at(bp.cast()))
                    .mul_q15_round(sv)
                    .add_sat(ov)
                    .max(zv)
                    .min(max_vecs.1)
                    .store_u16_at(dp);
                ap = ap.add(8);
                bp = bp.add(8);
                dp = dp.add(dst_stride);
            }
        }
        return;
    }
    if w >= W::LANES && w.is_multiple_of(W::LANES) && matches!(w, 16 | 32 | 64) {
        let (zw, sw, ow) = (W::splat(0), W::splat(scale), W::splat(offset));
        for _ in 0..h {
            let mut x = 0;
            while x < w {
                // SAFETY: `x + W::LANES <= w` and slices were validated by `weight_u16`.
                unsafe {
                    floor_avg_filter(
                        W::load_u16_at(ap.add(x).cast()),
                        W::load_u16_at(bp.add(x).cast()),
                    )
                    .mul_q15_round(sw)
                    .add_sat(ow)
                    .max(zw)
                    .min(max_vecs.0)
                    .store_u16_at(dp.add(x));
                }
                x += W::LANES;
            }
            // SAFETY: `h` rows validated by `weight_u16`.
            unsafe {
                ap = ap.add(w);
                bp = bp.add(w);
                dp = dp.add(dst_stride);
            }
        }
        return;
    }
    let mut y = if w == 4 {
        // SAFETY: forwarded from `weight_u16`.
        unsafe {
            strided_w4_bi_u16::<W, V>(
                (ap, bp, dp),
                (h, dst_stride),
                (scale, offset),
                (max_vecs.0, max_vecs.1),
            )
        }
    } else {
        0
    };
    while y < h {
        // SAFETY: `y < h`, validated by `weight_u16`.
        unsafe {
            row_bi_u16::<W, V, H>(
                weights,
                core::slice::from_raw_parts(ap.add(y * w), w),
                core::slice::from_raw_parts(bp.add(y * w), w),
                core::slice::from_raw_parts_mut(dp.add(y * dst_stride), w),
                depth,
                (scale, offset),
                max_vecs,
            );
        }
        y += 1;
    }
}

/// High-bit-depth (`u16`) weighted prediction ([`crate::high_depth::Weight16Fn`]).
///
/// # Safety
///
/// `W`, `V` and `H`'s ISA must be available; the slices cover the block.
#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
pub(super) unsafe fn weight_u16<W: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
) {
    if w == 0 || h == 0 {
        return;
    }
    let total = w * h;
    let (a, b) = (&a[..total], &b[..total]);
    let max = depth.max() as i16;
    let max_vecs = (W::splat(max), V::splat(max), H::splat(max));
    let mode = classify(weights);
    if dst_stride == w {
        let dst = &mut dst[..total];
        match mode {
            FastMode::Uni {
                second,
                scale,
                offset,
            } => row_uni_u16::<W, V, H>(
                weights,
                if second { b } else { a },
                dst,
                depth,
                (scale, offset),
                max_vecs,
            ),
            FastMode::BiEqual { scale, offset } => {
                row_bi_u16::<W, V, H>(weights, a, b, dst, depth, (scale, offset), max_vecs);
            }
            FastMode::General => row_gen_u16::<W, V, H>(weights, a, b, dst, depth, max_vecs),
        }
        return;
    }
    let dst = &mut dst[..(h - 1) * dst_stride + w];
    let (ap, bp, dp) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    // SAFETY: `a`, `b`, and `dst` were validated above, and the caller enabled the ISA.
    unsafe {
        match mode {
            FastMode::Uni {
                second,
                scale,
                offset,
            } => strided_uni_u16::<W, V, H>(
                weights,
                (if second { bp } else { ap }, dp),
                (w, h, dst_stride),
                depth,
                (scale, offset),
                max_vecs,
            ),
            FastMode::BiEqual { scale, offset } => strided_bi_u16::<W, V, H>(
                weights,
                (ap, bp, dp),
                (w, h, dst_stride),
                depth,
                (scale, offset),
                max_vecs,
            ),
            FastMode::General => {
                for y in 0..h {
                    row_gen_u16::<W, V, H>(
                        weights,
                        core::slice::from_raw_parts(ap.add(y * w), w),
                        core::slice::from_raw_parts(bp.add(y * w), w),
                        core::slice::from_raw_parts_mut(dp.add(y * dst_stride), w),
                        depth,
                        max_vecs,
                    );
                }
            }
        }
    }
}
