//! Motion compensation kernels of the general AV2 decoder, 8-bit: the
//! bilinear prediction intra block copy uses ([`put_bilinear`]) and the
//! linear scaling of a prediction by block-adaptive weights ([`morph`]).
//!
//! Ported from dav2d `src/mc_tmpl.c` (`put_bilin_c`, `morph_c`; commit
//! 237229e, Copyright © 2018-2026 `VideoLAN` and dav2d authors,
//! BSD-2-Clause, see `docs/licenses/dav2d`), which is bit-exact with AVM
//! v1.0.0. A C `pixel *` with a stride becomes a slice, an offset and a
//! stride.

/// Sub-sample positions are in 1/16 samples: the bilinear weights sum to
/// this.
const BILINEAR_ONE: i32 = 16;
/// Bits of the horizontal pass's intermediate precision at 8 bits (dav2d
/// `get_intermediate_bits`): the first pass keeps all four of its
/// fraction bits.
const INTERMEDIATE_BITS: u32 = 4;
/// Fraction bits of the block-adaptive scale.
const MORPH_SHIFT: u32 = 8;
/// Largest block side.
const MAX_SIDE: usize = 64;

/// One bilinear tap pair at sub-sample `phase` (1/16 samples):
/// `16 a + phase (b - a)`.
fn bilinear(a: i32, b: i32, phase: i32) -> i32 {
    BILINEAR_ONE * a + phase * (b - a)
}

/// `value` rounded down by `shift` bits, clamped to a sample.
fn round_to_pixel(value: i32, shift: u32) -> u8 {
    ((value + ((1 << shift) >> 1)) >> shift).clamp(0, i32::from(u8::MAX)) as u8
}

/// Predict a `w` x `h` block (each a power of two, 2 to 64) at
/// `dst[dst_offset..]` from `src[src_offset..]` with the bilinear filter at
/// sub-sample position (`mx`, `my`) in 1/16 samples (dav2d
/// `put_bilin_c`). The source must hold `w + 1` columns and `h + 1` rows
/// (the extra column and row weigh in only at nonzero phases).
///
/// # Panics
///
/// If either slice is too short for that at its stride, or a size is out
/// of range.
#[allow(clippy::too_many_arguments, reason = "dav2d's kernel contract")]
pub fn put_bilinear(
    dst: &mut [u8],
    dst_offset: usize,
    dst_stride: usize,
    src: &[u8],
    src_offset: usize,
    src_stride: usize,
    (w, h): (usize, usize),
    (mx, my): (i32, i32),
) {
    assert!(w.is_power_of_two() && (2..=MAX_SIDE).contains(&w));
    assert!(h.is_power_of_two() && (2..=MAX_SIDE).contains(&h));
    let source = |y: usize| &src[src_offset + y * src_stride..][..=w];
    let rows = dst[dst_offset..].chunks_mut(dst_stride);
    match (mx != 0, my != 0) {
        (true, true) => {
            // The horizontal pass of h + 1 rows at full precision, then
            // the vertical pass with one rounding.
            let mut mid = [[0_i16; MAX_SIDE]; MAX_SIDE + 1];
            for (y, line) in mid.iter_mut().take(h + 1).enumerate() {
                let row = source(y);
                for (x, value) in line.iter_mut().take(w).enumerate() {
                    *value = bilinear(i32::from(row[x]), i32::from(row[x + 1]), mx) as i16;
                }
            }
            for (y, line) in rows.take(h).enumerate() {
                for (x, out) in line[..w].iter_mut().enumerate() {
                    let value = bilinear(i32::from(mid[y][x]), i32::from(mid[y + 1][x]), my);
                    *out = round_to_pixel(value, 2 * INTERMEDIATE_BITS);
                }
            }
        }
        (true, false) => {
            for (y, line) in rows.take(h).enumerate() {
                let row = source(y);
                for (x, out) in line[..w].iter_mut().enumerate() {
                    let value = bilinear(i32::from(row[x]), i32::from(row[x + 1]), mx);
                    *out = round_to_pixel(value, INTERMEDIATE_BITS);
                }
            }
        }
        (false, true) => {
            for (y, line) in rows.take(h).enumerate() {
                let (row, below) = (source(y), source(y + 1));
                for (x, out) in line[..w].iter_mut().enumerate() {
                    let value = bilinear(i32::from(row[x]), i32::from(below[x]), my);
                    *out = round_to_pixel(value, INTERMEDIATE_BITS);
                }
            }
        }
        (false, false) => {
            for (y, line) in rows.take(h).enumerate() {
                line[..w].copy_from_slice(&source(y)[..w]);
            }
        }
    }
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
