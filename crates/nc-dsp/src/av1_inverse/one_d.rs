//! The non-DCT 1-D inverse transforms of AV1 (§7.13.2.5–7 ADST4/8/16,
//! §7.13.2.15 identity), applied independently to every lane of their input
//! vectors like the DCT graphs in `txfm::dct`. ADST8/16 and identity are
//! generic over `InverseLanes`, like those graphs, so the 32-bit generic
//! driver and the guarded 16-bit drivers (`txfm::narrow_inverse`) share
//! them; ADST4 (no stage clamps) stays on 32-bit lanes.
//!
//! Butterfly structure and constants follow libaom's `av1_iadst4/8/16` and
//! `av1_iidentity*_c` (`av1/common/av1_inv_txfm1d.c`, BSD-2-Clause,
//! Copyright (c) 2016, Alliance for Open Media; see `THIRD_PARTY.md`),
//! which equal the spec's flow graphs. ADST8/16 are written as their three
//! building blocks: an input rotation stage, then alternating
//! clamped add/sub butterflies and rotations on ever smaller blocks, then an
//! output permutation with alternating signs.
//!
//! # Ranges
//!
//! Inputs are clamped to 16 bits by the 2-D driver and the add/sub stages
//! clamp to 16 bits, so every `half_btf` product sum stays below `2^30`
//! (the [`I32Lanes::half_btf`] contract). ADST4 has no stage clamps; its
//! largest sum is `2^15 * (3803 + 5124 + 6285 + 3344) < 2^30`, so 32-bit
//! lanes compute it exactly. Identity outputs stay below `2^29`.

#![allow(clippy::inline_always)]

use super::{SQRT2, SQRT2_BITS};
use crate::simd::I32Lanes;
use crate::txfm::inverse_lanes::InverseLanes;

/// libaom `av1_sinpi_arr_data` at 12 bits: `round(2^12 * (2/3) * sqrt(2) *
/// sin(j * pi / 9))` for `j = 1..4`, adjusted so that `[1] + [2] == [4]`
/// (§7.13.2.6 `SINPI_1_9` .. `SINPI_4_9`).
const SINPI: [i32; 5] = [0, 1321, 2482, 3344, 3803];

/// Rotate the pair `x[i], x[i + 1]` by `angle` (in `pi / 128`; §7.13.2.1
/// `B( a, b, angle, 0 )` with `cospi[64 - angle] = sinpi`).
#[inline(always)]
fn rotate<V: InverseLanes>(x: &mut [V], i: usize, angle: usize, cospi: &[i32; 64], bit: u32) {
    let (a, b) = (x[i], x[i + 1]);
    x[i] = V::half_btf(cospi[angle], a, cospi[64 - angle], b, bit);
    x[i + 1] = V::half_btf(cospi[64 - angle], a, -cospi[angle], b, bit);
}

/// The mirrored rotation libaom uses for the second half of a rotation
/// stage: `(-sin * a + cos * b, cos * a + sin * b)`.
#[inline(always)]
fn rotate_mirrored<V: InverseLanes>(
    x: &mut [V],
    i: usize,
    angle: usize,
    cospi: &[i32; 64],
    bit: u32,
) {
    let (a, b) = (x[i], x[i + 1]);
    x[i] = V::half_btf(-cospi[64 - angle], a, cospi[angle], b, bit);
    x[i + 1] = V::half_btf(cospi[angle], a, cospi[64 - angle], b, bit);
}

/// Clamped add/sub butterflies within each block of `block` values: the
/// first half becomes the sums, the second half the differences.
#[inline(always)]
fn butterflies<V: InverseLanes>(x: &mut [V], block: usize) {
    let half = block / 2;
    for start in (0..x.len()).step_by(block) {
        for i in start..start + half {
            let (a, b) = (x[i], x[i + half]);
            x[i] = a.add(b).clamp16();
            x[i + half] = a.sub(b).clamp16();
        }
    }
}

/// The ADST input permutation: even positions take the inputs from the end
/// backwards, odd positions the even inputs forwards.
#[inline(always)]
fn permute_input<V: InverseLanes, const N: usize>(input: &[V; N]) -> [V; N] {
    let mut x = [V::splat(0); N];
    for k in 0..N / 2 {
        x[2 * k] = input[N - 1 - 2 * k];
        x[2 * k + 1] = input[2 * k];
    }
    x
}

/// The ADST output stage: `output[i] = ±x[order[i]]`, negated at odd `i`.
#[inline(always)]
fn permute_output<V: InverseLanes, const N: usize>(
    x: &[V; N],
    order: &[usize; N],
    output: &mut [V; N],
) {
    for (i, (out, &from)) in output.iter_mut().zip(order).enumerate() {
        *out = if i.is_multiple_of(2) {
            x[from]
        } else {
            V::splat(0).sub(x[from])
        };
    }
}

/// Output order of the 8-point ADST (libaom `av1_iadst8` stage 7).
const ADST8_OUTPUT: [usize; 8] = [0, 4, 6, 2, 3, 7, 5, 1];
/// Output order of the 16-point ADST (libaom `av1_iadst16` stage 9).
const ADST16_OUTPUT: [usize; 16] = [0, 8, 12, 4, 6, 14, 10, 2, 3, 11, 15, 7, 5, 13, 9, 1];
/// The final rotation of every ADST: by `pi / 4`.
const QUARTER_PI: usize = 32;

/// Inverse ADST4 (§7.13.2.6, libaom `av1_iadst4`).
#[inline(always)]
pub(super) fn iadst4<V: I32Lanes>(input: &[V; 4], output: &mut [V; 4], bit: u32) {
    let [x0, x1, x2, x3] = *input;
    let s0 = x0.mul_scalar(SINPI[1]);
    let s1 = x0.mul_scalar(SINPI[2]);
    let s2 = x1.mul_scalar(SINPI[3]);
    let s3 = x2.mul_scalar(SINPI[4]);
    let s4 = x2.mul_scalar(SINPI[1]);
    let s5 = x3.mul_scalar(SINPI[2]);
    let s6 = x3.mul_scalar(SINPI[4]);
    let s7 = x0.sub(x2).add(x3);

    let s0 = s0.add(s3).add(s5);
    let s1 = s1.sub(s4).sub(s6);
    let s3 = s2;
    let s2 = s7.mul_scalar(SINPI[3]);

    output[0] = s0.add(s3).round_shift(bit);
    output[1] = s1.add(s3).round_shift(bit);
    output[2] = s2.round_shift(bit);
    output[3] = s0.add(s1).sub(s3).round_shift(bit);
}

/// Inverse ADST8 (§7.13.2.7, libaom `av1_iadst8`).
#[inline(always)]
pub(crate) fn iadst8<V: InverseLanes>(
    input: &[V; 8],
    output: &mut [V; 8],
    cospi: &[i32; 64],
    bit: u32,
) {
    let mut x = permute_input(input);
    for (k, angle) in [4, 20, 36, 52].into_iter().enumerate() {
        rotate(&mut x, 2 * k, angle, cospi, bit);
    }
    butterflies(&mut x, 8);
    rotate(&mut x, 4, 16, cospi, bit);
    rotate_mirrored(&mut x, 6, 16, cospi, bit);
    butterflies(&mut x, 4);
    for i in [2, 6] {
        rotate(&mut x, i, QUARTER_PI, cospi, bit);
    }
    permute_output(&x, &ADST8_OUTPUT, output);
}

/// Inverse ADST16 (§7.13.2.7, libaom `av1_iadst16`).
#[inline(always)]
pub(crate) fn iadst16<V: InverseLanes>(
    input: &[V; 16],
    output: &mut [V; 16],
    cospi: &[i32; 64],
    bit: u32,
) {
    let mut x = permute_input(input);
    for (k, angle) in [2, 10, 18, 26, 34, 42, 50, 58].into_iter().enumerate() {
        rotate(&mut x, 2 * k, angle, cospi, bit);
    }
    butterflies(&mut x, 16);
    rotate(&mut x, 8, 8, cospi, bit);
    rotate(&mut x, 10, 40, cospi, bit);
    rotate_mirrored(&mut x, 12, 8, cospi, bit);
    rotate_mirrored(&mut x, 14, 40, cospi, bit);
    butterflies(&mut x, 8);
    for i in [4, 12] {
        rotate(&mut x, i, 16, cospi, bit);
        rotate_mirrored(&mut x, i + 2, 16, cospi, bit);
    }
    butterflies(&mut x, 4);
    for i in [2, 6, 10, 14] {
        rotate(&mut x, i, QUARTER_PI, cospi, bit);
    }
    permute_output(&x, &ADST16_OUTPUT, output);
}

/// Inverse identity of `N` points (§7.13.2.15): a scale by `sqrt(2)` (4),
/// 2 (8), `2 * sqrt(2)` (16) or 4 (32).
#[inline(always)]
pub(crate) fn identity<V: InverseLanes, const N: usize>(input: &[V; N], output: &mut [V; N]) {
    for (out, &x) in output.iter_mut().zip(input) {
        *out = match N {
            4 => V::mul_round(SQRT2, x, SQRT2_BITS),
            8 => x.add(x),
            16 => V::mul_round(2 * SQRT2, x, SQRT2_BITS),
            _ => {
                let twice = x.add(x);
                twice.add(twice)
            }
        };
    }
}
