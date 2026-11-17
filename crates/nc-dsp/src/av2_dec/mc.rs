//! Bilinear motion compensation of the general AV2 decoder (intra block
//! copy and the optical-flow refinement windows) at final precision
//! ([`put_bilinear_at`]) and compound precision ([`prep_bilinear`]), the
//! linear scaling of a prediction by block-adaptive weights ([`morph`]),
//! and the intermediate precision every compound kernel shares
//! ([`Precision`]).
//!
//! Ported from dav2d `src/mc_tmpl.c` (`put_bilin_c`, `prep_bilin_c`,
//! `morph_c`; commit 237229e, Copyright © 2018-2026 `VideoLAN` and dav2d
//! authors, BSD-2-Clause, see `docs/licenses/dav2d`), which is bit-exact
//! with AVM v1.0.0. A C `pixel *` with a stride becomes a slice, an offset
//! and a stride.

use nc_common::{BitDepth, Sample};

/// Sub-sample positions are in 1/16 samples: the bilinear weights sum to
/// this.
const BILINEAR_ONE: i32 = 16;
/// Fraction bits of the bilinear weights.
const BILINEAR_BITS: u32 = 4;
/// Fraction bits of the block-adaptive scale.
const MORPH_SHIFT: u32 = 8;
/// Largest block side.
pub(crate) const MAX_SIDE: usize = 64;
/// The one block height that is not a power of two: a 16-row optical-flow
/// refinement slice with 8 rows of padding (dav2d `put_bilin_c`).
const REFINE_SLICE_HEIGHT: usize = 24;
/// Smallest side of a compound-precision block.
pub(crate) const MIN_COMPOUND_SIDE: usize = 4;
/// Intermediate bits of 8-bit content (dav2d `get_intermediate_bits`).
const EIGHT_BIT_INTERMEDIATE: u32 = 4;
/// Deeper content keeps `COMPOUND_PRECISION - depth` intermediate bits.
const COMPOUND_PRECISION: u32 = 14;
/// What deeper content subtracts from compound values so they fit `i16`
/// (dav2d `PREP_BIAS`; 8-bit values fit as they are).
const DEEP_PREP_BIAS: i32 = 8192;

/// dav2d's intermediate precision at one bit depth: compound
/// predictions (`prep`, `int16_t *tmp`) are samples scaled up by `bits`
/// minus `bias`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Precision {
    /// `get_intermediate_bits`: 4 at 8 bits, `14 - depth` above.
    pub(crate) bits: u32,
    /// `PREP_BIAS`: 0 at 8 bits, 8192 above.
    pub(crate) bias: i32,
}

impl Precision {
    /// The precision of `depth`.
    pub(crate) fn of(depth: BitDepth) -> Self {
        if depth.bits() == BitDepth::EIGHT.bits() {
            Self {
                bits: EIGHT_BIT_INTERMEDIATE,
                bias: 0,
            }
        } else {
            Self {
                bits: COMPOUND_PRECISION - depth.bits(),
                bias: DEEP_PREP_BIAS,
            }
        }
    }
}

/// `value` rounded down by `shift` bits (half up; `shift` may be 0).
pub(crate) fn round_shift(value: i32, shift: u32) -> i32 {
    (value + ((1 << shift) >> 1)) >> shift
}

/// One bilinear tap pair at sub-sample `phase` (1/16 samples):
/// `16 a + phase (b - a)`.
fn bilinear(a: i32, b: i32, phase: i32) -> i32 {
    BILINEAR_ONE * a + phase * (b - a)
}

/// Check a final-precision block size: powers of two from 2 to 64, or the
/// refinement slice height.
fn assert_put_size(w: usize, h: usize) {
    assert!(w.is_power_of_two() && (2..=MAX_SIDE).contains(&w));
    assert!((h.is_power_of_two() && (2..=MAX_SIDE).contains(&h)) || h == REFINE_SLICE_HEIGHT);
}

/// Check a compound-precision block size: powers of two from 4 to 64.
pub(crate) fn assert_compound_size(w: usize, h: usize) {
    for side in [w, h] {
        assert!(side.is_power_of_two() && (MIN_COMPOUND_SIDE..=MAX_SIDE).contains(&side));
    }
}

/// The horizontal bilinear pass of `rows` rows of `w` samples, kept at
/// `16 << intermediate bits` scale (shift `4 - bits`, 0 below 12 bits).
fn horizontal_pass<T: Sample>(
    src: &[T],
    (src_offset, src_stride): (usize, usize),
    (w, rows): (usize, usize),
    mx: i32,
    shift: u32,
) -> [[i16; MAX_SIDE]; MAX_SIDE + 1] {
    let mut mid = [[0_i16; MAX_SIDE]; MAX_SIDE + 1];
    for (y, line) in mid.iter_mut().take(rows).enumerate() {
        let row = &src[src_offset + y * src_stride..][..=w];
        for (x, value) in line[..w].iter_mut().enumerate() {
            let sum = bilinear(row[x].to_i32(), row[x + 1].to_i32(), mx);
            *value = round_shift(sum, shift) as i16;
        }
    }
    mid
}

/// Write `values` to `line` clipped to samples of `depth`.
pub(crate) fn store_pixels<T: Sample>(
    line: &mut [T],
    values: impl Iterator<Item = i32>,
    depth: BitDepth,
) {
    for (out, value) in line.iter_mut().zip(values) {
        *out = depth.clip_to(value);
    }
}

/// Write `values` to `line` as compound values (minus the bias; they fit
/// `i16` by dav2d's range analysis of every filter).
pub(crate) fn store_compound(
    line: &mut [i16],
    values: impl Iterator<Item = i32>,
    precision: Precision,
) {
    for (out, value) in line.iter_mut().zip(values) {
        *out = (value - precision.bias) as i16;
    }
}

/// [`put_bilinear_at`] for 8-bit samples.
pub fn put_bilinear(
    dst: &mut [u8],
    dst_offset: usize,
    dst_stride: usize,
    src: &[u8],
    src_offset: usize,
    src_stride: usize,
    size: (usize, usize),
    phase: (i32, i32),
) {
    put_bilinear_at(
        dst,
        dst_offset,
        dst_stride,
        src,
        src_offset,
        src_stride,
        size,
        phase,
        BitDepth::EIGHT,
    );
}

/// Predict a `w` x `h` block (each a power of two, 2 to 64, or `h` = 24)
/// at `dst[dst_offset..]` from `src[src_offset..]` with the bilinear
/// filter at sub-sample position (`mx`, `my`) in 1/16 samples (dav2d
/// `put_bilin_c`). The source must hold `w + 1` columns and `h + 1` rows
/// (the extra column and row weigh in only at nonzero phases).
///
/// # Panics
///
/// If either slice is too short for that at its stride, or a size is out
/// of range.
pub fn put_bilinear_at<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    (w, h): (usize, usize),
    (mx, my): (i32, i32),
    depth: BitDepth,
) {
    assert_put_size(w, h);
    let precision = Precision::of(depth);
    let source = |y: usize| &src[src_offset + y * src_stride..][..=w];
    let rows = dst[dst_offset..].chunks_mut(dst_stride);
    let first_shift = BILINEAR_BITS - precision.bits.min(BILINEAR_BITS);
    match (mx != 0, my != 0) {
        (true, true) => {
            // The horizontal pass of h + 1 rows at intermediate precision,
            // then the vertical pass with one rounding.
            let mid = horizontal_pass(src, (src_offset, src_stride), (w, h + 1), mx, first_shift);
            for (y, line) in rows.take(h).enumerate() {
                store_pixels(
                    &mut line[..w],
                    (0..w).map(|x| {
                        let sum = bilinear(i32::from(mid[y][x]), i32::from(mid[y + 1][x]), my);
                        round_shift(sum, BILINEAR_BITS + precision.bits)
                    }),
                    depth,
                );
            }
        }
        (true, false) => {
            for (y, line) in rows.take(h).enumerate() {
                let row = source(y);
                store_pixels(
                    &mut line[..w],
                    (0..w).map(|x| {
                        let sum = bilinear(row[x].to_i32(), row[x + 1].to_i32(), mx);
                        round_shift(round_shift(sum, first_shift), precision.bits)
                    }),
                    depth,
                );
            }
        }
        (false, true) => {
            for (y, line) in rows.take(h).enumerate() {
                let (row, below) = (source(y), source(y + 1));
                store_pixels(
                    &mut line[..w],
                    (0..w).map(|x| {
                        let sum = bilinear(row[x].to_i32(), below[x].to_i32(), my);
                        round_shift(sum, BILINEAR_BITS)
                    }),
                    depth,
                );
            }
        }
        (false, false) => {
            for (y, line) in rows.take(h).enumerate() {
                line[..w].copy_from_slice(&source(y)[..w]);
            }
        }
    }
}

/// Predict a `w` x `h` block (powers of two, 4 to 64) at compound
/// precision into `tmp` (rows `tmp_stride` apart) from `src[src_offset..]`
/// with the bilinear filter at (`mx`, `my`) in 1/16 samples (dav2d
/// `prep_bilin_c`). The source holds `w + 1` columns and `h + 1` rows.
///
/// # Panics
///
/// If a slice is too short or a size is out of range.
pub fn prep_bilinear<T: Sample>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    (w, h): (usize, usize),
    (mx, my): (i32, i32),
    depth: BitDepth,
) {
    assert_compound_size(w, h);
    let precision = Precision::of(depth);
    let first_shift = BILINEAR_BITS - precision.bits.min(BILINEAR_BITS);
    let source = |y: usize| &src[src_offset + y * src_stride..][..=w];
    let rows = tmp.chunks_mut(tmp_stride);
    match (mx != 0, my != 0) {
        (true, true) => {
            let mid = horizontal_pass(src, (src_offset, src_stride), (w, h + 1), mx, first_shift);
            for (y, line) in rows.take(h).enumerate() {
                store_compound(
                    &mut line[..w],
                    (0..w).map(|x| {
                        let sum = bilinear(i32::from(mid[y][x]), i32::from(mid[y + 1][x]), my);
                        round_shift(sum, BILINEAR_BITS)
                    }),
                    precision,
                );
            }
        }
        (true, false) => {
            for (y, line) in rows.take(h).enumerate() {
                let row = source(y);
                store_compound(
                    &mut line[..w],
                    (0..w).map(|x| {
                        let sum = bilinear(row[x].to_i32(), row[x + 1].to_i32(), mx);
                        round_shift(sum, first_shift)
                    }),
                    precision,
                );
            }
        }
        (false, true) => {
            for (y, line) in rows.take(h).enumerate() {
                let (row, below) = (source(y), source(y + 1));
                store_compound(
                    &mut line[..w],
                    (0..w).map(|x| {
                        let sum = bilinear(row[x].to_i32(), below[x].to_i32(), my);
                        round_shift(sum, first_shift)
                    }),
                    precision,
                );
            }
        }
        (false, false) => prep_copy(
            tmp,
            tmp_stride,
            src,
            (src_offset, src_stride),
            (w, h),
            depth,
        ),
    }
}

/// A whole-sample compound prediction: samples scaled to intermediate
/// precision minus the bias (dav2d `prep_c`).
pub(crate) fn prep_copy<T: Sample>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[T],
    (src_offset, src_stride): (usize, usize),
    (w, h): (usize, usize),
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    crate::block_copy::prep_samples(
        src.get(src_offset..).expect("prep source offset"),
        src_stride,
        tmp,
        tmp_stride,
        (w, h),
        (precision.bits, precision.bias),
    );
}

/// Scale a `w` x `h` prediction at `dst[offset..]` in place by `alpha`
/// (8 fraction bits) and add `beta` (dav2d `morph_c`).
///
/// # Panics
///
/// If `dst` is too short for the block at its stride.
pub fn morph(
    dst: &mut [u8],
    offset: usize,
    stride: usize,
    (alpha, beta): (i32, i32),
    (w, h): (usize, usize),
) {
    for line in dst[offset..].chunks_mut(stride).take(h) {
        for sample in &mut line[..w] {
            let value = (alpha * i32::from(*sample) + beta) >> MORPH_SHIFT;
            *sample = value.clamp(0, i32::from(u8::MAX)) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A copy (zero phases) and a half-sample average agree with the
    /// definitions.
    #[test]
    fn copy_and_half_sample() {
        let src: Vec<u8> = (0..9 * 9_u32).map(|v| (v * 7 % 251) as u8).collect();
        let mut dst = [0_u8; 8 * 8];
        put_bilinear(&mut dst, 0, 8, &src, 0, 9, (8, 8), (0, 0));
        for y in 0..8 {
            assert_eq!(dst[y * 8..][..8], src[y * 9..][..8]);
        }
        put_bilinear(&mut dst, 0, 8, &src, 0, 9, (8, 8), (8, 0));
        for y in 0..8 {
            for x in 0..8 {
                let (a, b) = (u16::from(src[y * 9 + x]), u16::from(src[y * 9 + x + 1]));
                assert_eq!(u16::from(dst[y * 8 + x]), (a + b + 1) >> 1);
            }
        }
    }

    /// Compound bilinear values captured from dav2d's `prep_bilin_c`
    /// (8-bit) for a 4x4 block at phase (5, 11).
    #[test]
    fn prep_matches_dav2d() {
        let src: Vec<u8> = (0..5 * 5_u32).map(|v| (v * 37 % 256) as u8).collect();
        let mut tmp = [0_i16; 16];
        prep_bilinear(&mut tmp, 4, &src, 0, 5, (4, 4), (5, 11), BitDepth::EIGHT);
        assert_eq!(tmp[..8], [2220, 1932, 588, 1180, 2364, 2556, 2268, 1980]);
    }

    /// The identity scale leaves a block unchanged; beta shifts it.
    #[test]
    fn morph_identity_and_offset() {
        let mut block = [10_u8, 20, 250, 0];
        morph(&mut block, 0, 4, (256, 0), (4, 1));
        assert_eq!(block, [10, 20, 250, 0]);
        morph(&mut block, 0, 4, (256, 5 << 8), (4, 1));
        assert_eq!(block, [15, 25, 255, 5]);
    }
}
