//! Eight-column GDF on NEON, SIMD128 and narrow x86 stripes. The stripe schedule follows
//! dav2d 237229e's AVX2 preparation kernel (BSD-2-Clause, Copyright © 2026
//! `VideoLAN` and dav2d authors and Two Orioles, LLC, see `THIRD_PARTY.md`).
//! Eight 16-bit lanes reuse gradients for pairs of adjacent output rows.
//! The shared accumulators use widening MACs on NEON and word dots on WASM.

#![allow(unsafe_code, clippy::inline_always)]

use super::{
    GDF_BORDER, GDF_ERROR_STRIDE, GDF_MAX_SIDE, GRADIENT_DIRECTIONS, GdfParams, GdfWindow,
};
use crate::simd::{I16FilterLanes, I16x8Shuffles};

const LANES: usize = 8;
const GRADIENT_COLUMNS: usize = GDF_MAX_SIDE + 2;
type Gradients = [[i16; GRADIENT_COLUMNS]; 4];

#[inline(always)]
unsafe fn grad_columns<V: I16FilterLanes>(
    first: *const u16,
    second: *const u16,
    taps: [isize; 4],
    out: &mut Gradients,
    x: usize,
) {
    // SAFETY: the two rows plus the ±tap neighbourhood are inside the
    // validated window for all eight columns.
    unsafe {
        let a = first.add(x);
        let b = second.add(x);
        let ca = V::load_u16_at(a);
        let ca = ca.add(ca);
        let cb = V::load_u16_at(b);
        let cb = cb.add(cb);
        for (dst, tap) in out.iter_mut().zip(taps) {
            let grad = |p: *const u16, twice: V| {
                let ends = V::load_u16_at(p.offset(-tap)).add(V::load_u16_at(p.offset(tap)));
                twice.sub(ends).abs()
            };
            grad(a, ca).add(grad(b, cb)).store(&mut dst[x..]);
        }
    }
}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
unsafe fn gradient_pair<V: I16FilterLanes>(window: &GdfWindow<'_>, y: isize, out: &mut Gradients) {
    let stride = window.stride as isize;
    let start = (y + GDF_BORDER as isize) * stride + GDF_BORDER as isize - 1;
    // SAFETY: the border covers the two rows over columns -1..=width
    // and their one-sample gradient taps.
    unsafe {
        let first = window.samples.as_ptr().offset(start);
        let second = first.offset(stride);
        let taps = GRADIENT_DIRECTIONS.map(|(dy, dx)| dy * stride + dx);
        for x in (0..window.width).step_by(LANES) {
            grad_columns::<V>(first, second, taps, out, x);
        }
        // The last two values are needed by the final four-column sum.
        // Overlap the preceding six gradient columns to keep this load in SIMD.
        grad_columns::<V>(first, second, taps, out, window.width + 2 - LANES);
    }
}

#[inline(always)]
fn bias<V: I16FilterLanes>(value: i16) -> V::Acc {
    V::acc_mul_add(V::acc_zero(), V::splat(1), value)
}

#[inline(always)]
fn dot<V: I16FilterLanes>(acc: V::Acc, a: V, b: V, [wa, wb]: [i16; 2]) -> V::Acc {
    V::acc_mul_add2(acc, a, wa, b, wb)
}

#[derive(Clone, Copy)]
struct Classes<V: I16FilterLanes> {
    transpose: V,
    mirror: V,
    bias: V::Acc,
}

#[inline(always)]
fn classify<V: I16FilterLanes + I16x8Shuffles>(
    (top, bottom): (&Gradients, &Gradients),
    x: usize,
    params: &GdfParams,
) -> Classes<V> {
    const SWAP: [u8; 16] = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13];
    let mut total = [V::splat(0); 4];
    let mut cap = total;
    for direction in 0..4 {
        let a = &top[direction];
        let b = &bottom[direction];
        let columns = V::load(&a[x..])
            .add(V::load(&b[x..]))
            .add(V::load(&a[x + 2..]))
            .add(V::load(&b[x + 2..]));
        let sum = columns.add(columns.shuffle_bytes(&SWAP));
        total[direction] = sum;
        cap[direction] = sum
            .srl(4)
            .min(V::splat(params.gradient_alpha[direction] as i16));
    }
    let [w0, w1, w2, w3] = params.gradient_weight;
    let third = dot(bias::<V>(params.bias[2]), cap[0], cap[1], [w0, w1]);
    Classes {
        // A strict greater-than mask selects the unswapped pair.
        transpose: total[0].gt_mask(total[1]),
        mirror: total[2].gt_mask(total[3]),
        bias: dot(third, cap[2], cap[3], [w2, w3]),
    }
}

#[inline(always)]
fn pair<V: I16FilterLanes>(a: V, b: V, cap: u16) -> V {
    let hi = V::splat(cap as i16);
    let lo = V::splat(-(cap as i16));
    a.max(lo)
        .min(hi)
        .add(b.max(lo).min(hi))
        .max(V::splat(-512))
        .min(V::splat(511))
}

#[inline(always)]
fn swap<V: I16FilterLanes>(keep: V, a: V, b: V) -> (V, V) {
    (keep.blend(a, b), keep.blend(b, a))
}

#[inline(always)]
fn pairs<V: I16FilterLanes>(
    [a0, a1]: [V; 2],
    [b0, b1]: [V; 2],
    keep: V,
    [cap0, cap1]: [u16; 2],
) -> (V, V) {
    let (a0, b0) = swap(keep, a0, b0);
    let (a1, b1) = swap(keep, a1, b1);
    (pair(a0, a1, cap0), pair(b0, b1, cap1))
}

#[inline(always)]
fn digit<V: I16FilterLanes>(words: V, scale: V, largest: V) -> V {
    let zero = V::splat(0);
    let signed_scale = words.gt_mask(zero).blend(scale, zero.sub(scale));
    words
        .mul_q15_round(signed_scale)
        .with_sign_of(words)
        .min(largest)
        .add(scale)
}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
unsafe fn row<V: I16FilterLanes>(
    pixels: *const u16,
    stride: isize,
    classes: Classes<V>,
    params: &GdfParams,
    out: &mut [i8],
) {
    // SAFETY: the checked stripe has eight lanes at each tap up to six rows
    // and columns from its centre. All output indices fit the validated LUT.
    unsafe {
        let centre = V::load_u16_at(pixels);
        let taps = |offset: isize| {
            [
                V::load_u16_at(pixels.offset(-offset)).sub(centre),
                V::load_u16_at(pixels.offset(offset)).sub(centre),
            ]
        };
        let axis = |k: usize| {
            let step = k as isize + 1;
            pairs(
                taps(step * stride),
                taps(step),
                classes.transpose,
                params.alpha[k],
            )
        };
        let mut first = bias::<V>(params.bias[0]);
        let mut second = bias::<V>(params.bias[1]);
        let mut third = classes.bias;
        for k in 0..2 {
            let (v, h) = axis(k);
            first = dot(first, v, h, params.weight1[3 * k]);
            second = dot(second, v, h, params.weight1[3 * k + 1]);
            third = dot(third, v, h, params.weight1[3 * k + 2]);
        }
        let (v3, h3) = axis(2);
        third = dot(third, v3, h3, params.weight1[6]);
        let third = V::acc_round_shift_narrow(third, 0, 0);
        let (v4, h4) = axis(3);
        let [w3, w4, w5, w6] = params.weight2[0];
        let [u3, u4, u5, u6] = params.weight2[1];
        first = dot(first, v3, v4, [w3, w4]);
        second = dot(second, h3, h4, [u3, u4]);
        let (v5, h5) = axis(4);
        let (v6, h6) = axis(5);
        first = dot(first, v5, v6, [w5, w6]);
        second = dot(second, h5, h6, [u5, u6]);

        let (a, b) = pairs(
            taps(stride + 1),
            taps(stride - 1),
            classes.mirror,
            params.alpha[6],
        );
        first = dot(first, a, b, params.weight3[0]);
        second = dot(second, a, b, params.weight3[1]);

        let [td0, td1] = taps(2 * stride + 1);
        let [wd0, wd1] = taps(stride + 2);
        let [tu0, tu1] = taps(2 * stride - 1);
        let [wu0, wu1] = taps(stride - 2);
        let keep = classes.transpose;
        let (td0, wd0) = swap(keep, td0, wd0);
        let (td1, wd1) = swap(keep, td1, wd1);
        let (tu0, wu0) = swap(keep, tu0, wu0);
        let (tu1, wu1) = swap(keep, tu1, wu1);
        let (a, b) = pairs([td0, td1], [tu0, tu1], classes.mirror, params.alpha[7]);
        first = dot(first, a, b, params.weight4[0]);
        let (a, b) = pairs([wd0, wd1], [wu0, wu1], classes.mirror, params.alpha[8]);
        second = dot(second, a, b, params.weight4[1]);

        let scale = V::splat(params.scale as i16);
        let largest = V::splat(params.scale as i16 - 1);
        let base = scale.add(scale);
        let a = digit(V::acc_round_shift_narrow(first, 0, 0), scale, largest);
        let b = digit(V::acc_round_shift_narrow(second, 0, 0), scale, largest);
        let c = digit(third, scale, largest);
        let mut indices = [0i16; LANES];
        a.mul(base).add(b).mul(base).add(c).store(&mut indices);
        let lut = params.error_lut.as_ptr();
        for (dst, index) in out.iter_mut().zip(indices) {
            *dst = *lut.add(usize::from(index as u16));
        }
    }
}

/// # Safety
///
/// The caller enables V's tier and validates the padded window and LUT.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
pub(super) unsafe fn error<V: I16FilterLanes + I16x8Shuffles>(
    window: &GdfWindow<'_>,
    params: &GdfParams,
    out: &mut [i8],
) {
    let mut previous = [[0; GRADIENT_COLUMNS]; 4];
    let mut next = previous;
    let stride = window.stride;
    // SAFETY: the stripe contains its full six-tap border.
    unsafe { gradient_pair::<V>(window, -1, &mut previous) };
    for y in (0..window.height).step_by(2) {
        // SAFETY: the next two gradient rows require only one more context row.
        unsafe { gradient_pair::<V>(window, y as isize + 1, &mut next) };
        for x in (0..window.width).step_by(LANES) {
            let offset = y * GDF_ERROR_STRIDE + x;
            let base = (y + GDF_BORDER) * stride + x + GDF_BORDER;
            let classes = classify::<V>((&previous, &next), x, params);
            // SAFETY: both output vectors and their tap windows are checked.
            unsafe {
                let p = window.samples.as_ptr().add(base);
                row(
                    p,
                    stride as isize,
                    classes,
                    params,
                    &mut out[offset..offset + LANES],
                );
                let offset = offset + GDF_ERROR_STRIDE;
                row(
                    p.add(stride),
                    stride as isize,
                    classes,
                    params,
                    &mut out[offset..offset + LANES],
                );
            }
        }
        core::mem::swap(&mut previous, &mut next);
    }
}
