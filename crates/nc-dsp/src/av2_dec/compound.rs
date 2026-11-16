//! Compound and blended predictions of the general AV2 decoder: the
//! average ([`avg`]), distance / CWP weights ([`w_avg`]), a given mask
//! ([`mask`]), the difference-weighted mask written as it is applied
//! ([`w_mask`]), and the inter-intra / OBMC-style blend of final-precision
//! samples ([`blend`]).
//!
//! Ported from dav2d `src/mc_tmpl.c` (`avg_c`, `w_avg_c`, `mask_c`,
//! `w_mask_{444,422,420}_c`, `blend_c`; commit 237229e, Copyright ©
//! 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see
//! `docs/licenses/dav2d`). The inputs `tmp1`, `tmp2` are compound
//! predictions ([`super::mc_8tap::prep_8tap`] and friends) of `w` x `h`
//! values stored contiguously (stride `w`); so are the masks, 6-bit
//! weights of `tmp1`. At 8 bits the average and mask equal AV1's
//! (`av1_mc::blend`, `InterPostRound` = 4 = dav2d's intermediate bits),
//! but those take AV1's 8-bit plane views; these keep dav2d's contract,
//! generic over the sample type with its `PREP_BIAS`.

use nc_common::{BitDepth, Sample};

use super::mc::{Precision, assert_compound_size};

/// Fraction bits of a mask weight.
const MASK_BITS: u32 = 6;
/// The full weight of a mask.
const MASK_ONE: i32 = 1 << MASK_BITS;
/// Fraction bits of a [`w_avg`] weight.
const WEIGHT_BITS: u32 = 4;
/// The full weight of a [`w_avg`] pair.
const WEIGHT_ONE: i32 = 1 << WEIGHT_BITS;
/// The smallest weight of the difference-weighted mask (AV1 / AV2
/// `DIFFWTD_38`).
const DIFF_WEIGHT_BASE: i32 = 38;
/// Bits of a 4-bit-scale difference: the difference's shift is
/// `depth + intermediate bits - 4`.
const DIFF_SCALE_BITS: u32 = 4;
/// The difference is rounded at 5 bits below its shift.
const DIFF_ROUND_BITS: u32 = 5;

/// The chroma layout a [`w_mask`] writes its mask for (dav2d
/// `w_mask[0..3]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskLayout {
    /// 4:4:4: one weight per sample.
    I444,
    /// 4:2:2: one weight per horizontal pair.
    I422,
    /// 4:2:0: one weight per 2x2 quad.
    I420,
}

impl MaskLayout {
    /// Horizontal and vertical subsampling.
    fn subsampling(self) -> (bool, bool) {
        match self {
            Self::I444 => (false, false),
            Self::I422 => (true, false),
            Self::I420 => (true, true),
        }
    }
}

/// Write `value(row_index, row of tmp1, row of tmp2, out row)` over a
/// `w` x `h` block.
fn combine<T: Sample>(
    (dst, dst_offset, dst_stride): (&mut [T], usize, usize),
    tmp1: &[i16],
    tmp2: &[i16],
    (w, h): (usize, usize),
    mut row: impl FnMut(usize, &[i16], &[i16], &mut [T]),
) {
    assert_compound_size(w, h);
    let lines = dst[dst_offset..].chunks_mut(dst_stride).take(h);
    let inputs = tmp1[..w * h]
        .chunks_exact(w)
        .zip(tmp2[..w * h].chunks_exact(w));
    for (y, (line, (a, b))) in lines.zip(inputs).enumerate() {
        row(y, a, b, &mut line[..w]);
    }
}

/// `(tmp1 + tmp2)` averaged to samples (dav2d `avg_c`).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
pub fn avg<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    size: (usize, usize),
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let shift = precision.bits + 1;
    let rounding = (1 << precision.bits) + 2 * precision.bias;
    combine(
        (dst, dst_offset, dst_stride),
        tmp1,
        tmp2,
        size,
        |_, a, b, out| {
            for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                *o = depth.clip_to((i32::from(a) + i32::from(b) + rounding) >> shift);
            }
        },
    );
}

/// `weight * tmp1 + (16 - weight) * tmp2` to samples, `weight` 0 to 16
/// (dav2d `w_avg_c`).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
#[allow(clippy::too_many_arguments, reason = "dav2d's kernel contract")]
pub fn w_avg<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    size: (usize, usize),
    weight: i32,
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let shift = precision.bits + WEIGHT_BITS;
    let rounding = ((WEIGHT_ONE / 2) << precision.bits) + WEIGHT_ONE * precision.bias;
    combine(
        (dst, dst_offset, dst_stride),
        tmp1,
        tmp2,
        size,
        |_, a, b, out| {
            for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                let sum = i32::from(a) * weight + i32::from(b) * (WEIGHT_ONE - weight);
                *o = depth.clip_to((sum + rounding) >> shift);
            }
        },
    );
}

/// `m * tmp1 + (64 - m) * tmp2` to samples with `m` from `mask` (`w` x `h`
/// weights 0 to 64, stride `w`; dav2d `mask_c`).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
#[allow(clippy::too_many_arguments, reason = "dav2d's kernel contract")]
pub fn mask<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    (w, h): (usize, usize),
    mask: &[u8],
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let shift = precision.bits + MASK_BITS;
    let rounding = ((MASK_ONE / 2) << precision.bits) + MASK_ONE * precision.bias;
    let mask = &mask[..w * h];
    combine(
        (dst, dst_offset, dst_stride),
        tmp1,
        tmp2,
        (w, h),
        |y, a, b, out| {
            let weights = &mask[y * w..][..w];
            for (((o, &a), &b), &m) in out.iter_mut().zip(a).zip(b).zip(weights) {
                let m = i32::from(m);
                let sum = i32::from(a) * m + i32::from(b) * (MASK_ONE - m);
                *o = depth.clip_to((sum + rounding) >> shift);
            }
        },
    );
}

/// Blend `tmp1` and `tmp2` by the difference-weighted mask
/// (`38 + |tmp1 - tmp2|` scaled, at most 64, as `tmp1`'s weight) and write
/// that mask subsampled for the chroma `layout` to `mask` (rows
/// `mask_stride` apart; 4:2:2 and 4:2:0 average the pair or quad, rounded
/// toward `sign`'s side; dav2d `w_mask_{444,422,420}_c`). The 4:2:0 mask's
/// even rows hold a partial sum until the odd row completes them.
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
#[allow(clippy::too_many_arguments, reason = "dav2d's kernel contract")]
pub fn w_mask<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    size: (usize, usize),
    (mask, mask_stride): (&mut [u8], usize),
    layout: MaskLayout,
    sign: bool,
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let shift = precision.bits + MASK_BITS;
    let rounding = ((MASK_ONE / 2) << precision.bits) + MASK_ONE * precision.bias;
    let mask_shift = depth.bits() + precision.bits - DIFF_SCALE_BITS;
    let mask_rounding = 1 << (mask_shift - DIFF_ROUND_BITS);
    let sign = i32::from(sign);
    let (sub_x, sub_y) = layout.subsampling();
    combine(
        (dst, dst_offset, dst_stride),
        tmp1,
        tmp2,
        size,
        |y, a, b, out| {
            let weight_and_blend = |x: usize, out: &mut T| {
                let diff = i32::from(a[x]) - i32::from(b[x]);
                let m =
                    (DIFF_WEIGHT_BASE + ((diff.abs() + mask_rounding) >> mask_shift)).min(MASK_ONE);
                *out = depth.clip_to((diff * m + i32::from(b[x]) * MASK_ONE + rounding) >> shift);
                m
            };
            let mask_row = &mut mask[(y >> usize::from(sub_y)) * mask_stride..];
            if !sub_x {
                for (x, o) in out.iter_mut().enumerate() {
                    mask_row[x] = weight_and_blend(x, o) as u8;
                }
                return;
            }
            for (pair, outs) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let m = weight_and_blend(2 * pair, &mut outs[0]);
                let n = weight_and_blend(2 * pair + 1, &mut outs[1]);
                let stored = &mut mask_row[pair];
                *stored = match (sub_y, y % 2 == 1) {
                    (false, _) => (m + n + 1 - sign) >> 1,
                    (true, false) => m + n,
                    (true, true) => (m + n + i32::from(*stored) + 2 - sign) >> 2,
                } as u8;
            }
        },
    );
}

/// Blend final-precision `tmp` (`w` x `h`, stride `w`) into `dst` in
/// place: `dst = (dst * (64 - m) + tmp * m + 32) >> 6` with `m` from `mask`
/// (stride `w`; dav2d `blend_c`, inter-intra and wedge inter-intra).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
pub fn blend<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    tmp: &[T],
    (w, h): (usize, usize),
    mask: &[u8],
) {
    assert_compound_size(w, h);
    let lines = dst[dst_offset..].chunks_mut(dst_stride).take(h);
    let inputs = tmp[..w * h]
        .chunks_exact(w)
        .zip(mask[..w * h].chunks_exact(w));
    for (line, (source, weights)) in lines.zip(inputs) {
        for ((o, &s), &m) in line[..w].iter_mut().zip(source).zip(weights) {
            let m = i32::from(m);
            let sum = o.to_i32() * (MASK_ONE - m) + s.to_i32() * m;
            *o = T::from_i32((sum + MASK_ONE / 2) >> MASK_BITS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equal predictions average to themselves; full weights pick one.
    #[test]
    fn averages_and_weights() {
        let eight = BitDepth::EIGHT;
        let a = [100_i16 << 4; 16];
        let b = [20_i16 << 4; 16];
        let mut dst = [0_u8; 16];
        avg(&mut dst, 0, 4, &a, &a, (4, 4), eight);
        assert_eq!(dst, [100; 16]);
        avg(&mut dst, 0, 4, &a, &b, (4, 4), eight);
        assert_eq!(dst, [60; 16]);
        w_avg(&mut dst, 0, 4, &a, &b, (4, 4), 16, eight);
        assert_eq!(dst, [100; 16]);
        mask(&mut dst, 0, 4, &a, &b, (4, 4), &[0; 16], eight);
        assert_eq!(dst, [20; 16]);
        let mut inter = [10_u8; 16];
        blend(&mut inter, 0, 4, &[250_u8; 16], (4, 4), &[32; 16]);
        assert_eq!(inter, [130; 16]);
    }

    /// Values captured from dav2d's `w_mask_420_c` (8-bit, sign 1).
    #[test]
    fn w_mask_420_matches_dav2d() {
        let a: Vec<i16> = (0..16).map(|i| i * 600 - 3000).collect();
        let b: Vec<i16> = (0..16).map(|i| 4000 - i * 450).collect();
        let mut dst = [0_u8; 16];
        let mut weights = [0_u8; 4];
        w_mask(
            &mut dst,
            0,
            4,
            &a,
            &b,
            (4, 4),
            (&mut weights, 2),
            MaskLayout::I420,
            true,
            BitDepth::EIGHT,
        );
        assert_eq!(
            dst,
            [
                0, 0, 0, 0, 6, 34, 54, 66, 84, 109, 143, 185, 235, 255, 255, 255
            ]
        );
        assert_eq!(weights, [54, 47, 53, 58]);
    }
}
