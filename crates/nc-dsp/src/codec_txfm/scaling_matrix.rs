//! Inverse scaling of one coefficient with an explicit scaling factor
//! (H.265 §8.6.3 eq. 8-309, H.266 §8.7.3 eq. 1158): the HEVC and VVC
//! decoders' scaling-list path, one shared implementation. Flat scaling
//! (factor 16) has vector kernels; only streams with scaling lists come here.

/// `Clip3(CoeffMin, CoeffMax, (level · m · levelScale · 2^per + 2^(bdShift − 1)) >> bdShift)`
/// with `CoeffMin`/`CoeffMax` the 16-bit range: `level` the parsed level,
/// `factor` the scaling factor `m`, `level_scale` `levelScale[qP % 6]`,
/// `per` `qP / 6` and `bd_shift` the codec's `bdShift` (it depends on the
/// bit depth and the block size). Exact in 64 bits for every 32-bit level.
#[must_use]
pub fn scale_coefficient(
    level: i32,
    factor: i32,
    level_scale: i32,
    per: u32,
    bd_shift: u32,
) -> i16 {
    // |level| < 2^31, m <= 255, levelScale <= 72: the product is below 2^46,
    // so a shift by up to 17 stays in 64 bits; larger per saturate anyway.
    let product = i64::from(level) * i64::from(factor) * i64::from(level_scale);
    let shifted = product.saturating_mul(1i64 << per.min(17));
    let value = shifted.saturating_add(1 << (bd_shift - 1)) >> bd_shift;
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_flat_scaling_equation() {
        for level in [-40_000, -300, -1, 0, 1, 7, 300, 40_000] {
            for (factor, per, shift) in [(16, 0, 5), (16, 3, 7), (255, 5, 9), (1, 10, 12)] {
                let exact =
                    (((i64::from(level) * factor * 57) << per) + (1 << (shift - 1))) >> shift;
                let expected = exact.clamp(-32_768, 32_767) as i16;
                assert_eq!(
                    scale_coefficient(level, factor as i32, 57, per, shift),
                    expected
                );
            }
        }
    }
}
