//! VP9 4×4 hybrid DCT/ADST transforms for directional intra prediction.
//!
//! Integer sine stages and pass order follow libvpx v1.16.0
//! `vp9/encoder/vp9_dct.c`, `vp9/common/vp9_idct.c` and
//! `vpx_dsp/inv_txfm.c` (`WebM` Project, BSD-3-Clause; see `THIRD_PARTY.md`).
//! `DCT_DCT` is the 4x4 DCT; lossless uses the WHT path.

const COS8: i32 = 15137;
const COS16: i32 = 11585;
const COS24: i32 = 6270;
const DCT_FRACTION_BITS: u32 = 14;

fn round_dct(value: i32) -> i32 {
    (value + (1 << (DCT_FRACTION_BITS - 1))) >> DCT_FRACTION_BITS
}

fn forward_1d([a, b, c, d]: [i32; 4]) -> [i32; 4] {
    let (sum0, sum1, diff0, diff1) = (a + d, b + c, b - c, a - d);
    [
        round_dct((sum0 + sum1) * COS16),
        round_dct(diff0 * COS24 + diff1 * COS8),
        round_dct((sum0 - sum1) * COS16),
        round_dct(-diff0 * COS8 + diff1 * COS24),
    ]
}

/// Products are formed in 64 bits: coefficients of a corrupt stream can
/// be far outside the 16-bit range of valid ones (the results then wrap as
/// libvpx's `WRAPLOW` does, and valid streams are unaffected).
fn inverse_1d(input: [i32; 4]) -> [i32; 4] {
    let [a, b, c, d] = input.map(i64::from);
    let [cos8, cos16, cos24] = [COS8, COS16, COS24].map(i64::from);
    let even0 = wrap(round((a + c) * cos16));
    let even1 = wrap(round((a - c) * cos16));
    let odd0 = wrap(round(b * cos24 - d * cos8));
    let odd1 = wrap(round(b * cos8 + d * cos24));
    [
        wrap(even0 + odd1),
        wrap(even1 + odd0),
        wrap(even1 - odd0),
        wrap(even0 - odd1),
    ]
}

const SIN1: i64 = 5283;
const SIN2: i64 = 9929;
const SIN3: i64 = 13377;
const SIN4: i64 = 15212;

/// Column/row transform pair, in VP9's normative transform-type order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transform {
    /// DC intra, chroma, or inter prediction.
    DctDct,
    /// Vertical intra prediction: sine columns and cosine rows.
    AdstDct,
    /// Horizontal intra prediction: cosine columns and sine rows.
    DctAdst,
    /// True-motion prediction: sine in both dimensions.
    AdstAdst,
}

fn round(value: i64) -> i32 {
    ((value + (1 << 13)) >> 14) as i32
}

fn wrap(value: i32) -> i32 {
    i32::from(value as i16)
}

fn forward_adst(input: [i32; 4]) -> [i32; 4] {
    let [a, b, c, d] = input.map(i64::from);
    let first = SIN1 * a + SIN2 * b + SIN4 * d;
    let middle = SIN3 * (a + b - d);
    let last = SIN4 * a - SIN1 * b + SIN2 * d;
    let third = SIN3 * c;
    [
        round(first + third),
        round(middle),
        round(last - third),
        round(last - first + third),
    ]
}

fn inverse_adst(input: [i32; 4]) -> [i32; 4] {
    let [a, b, c, d] = input.map(i64::from);
    let first = SIN1 * a + SIN4 * c + SIN2 * d;
    let second = SIN2 * a - SIN1 * c - SIN4 * d;
    let middle = SIN3 * b;
    let difference = i64::from(wrap((a - c + d) as i32));
    [
        wrap(round(first + middle)),
        wrap(round(second + middle)),
        wrap(round(SIN3 * difference)),
        wrap(round(first + second - middle)),
    ]
}

impl Transform {
    pub(crate) fn columns_adst(self) -> bool {
        matches!(self, Self::AdstDct | Self::AdstAdst)
    }
    pub(crate) fn rows_adst(self) -> bool {
        matches!(self, Self::DctAdst | Self::AdstAdst)
    }
}

/// Forward 4x4 transform of a raster residual (libvpx `vp9_fht4x4_c`).
pub(crate) fn forward(input: &[i32; 16], transform: Transform) -> [i32; 16] {
    let mut columns = [0; 16];
    for x in 0..4 {
        let mut column = std::array::from_fn(|y| input[y * 4 + x] * 16);
        if x == 0 && column[0] != 0 {
            column[0] += 1;
        }
        let transformed = if transform.columns_adst() {
            forward_adst(column)
        } else {
            forward_1d(column)
        };
        for y in 0..4 {
            columns[y * 4 + x] = transformed[y];
        }
    }
    let mut output = [0; 16];
    for y in 0..4 {
        let row = std::array::from_fn(|x| columns[y * 4 + x]);
        let transformed = if transform.rows_adst() {
            forward_adst(row)
        } else {
            forward_1d(row)
        };
        for x in 0..4 {
            output[y * 4 + x] = (transformed[x] + 1) >> 2;
        }
    }
    output
}

/// Normative inverse 4x4 transform of dequantized coefficients
/// (libvpx `vp9_iht4x4_16_add_c`), before adding the prediction.
pub(crate) fn inverse(input: &[i32; 16], transform: Transform) -> [i32; 16] {
    let mut rows = [0; 16];
    for y in 0..4 {
        let row = std::array::from_fn(|x| input[y * 4 + x]);
        let transformed = if transform.rows_adst() {
            inverse_adst(row)
        } else {
            inverse_1d(row)
        };
        rows[y * 4..y * 4 + 4].copy_from_slice(&transformed);
    }
    let mut output = [0; 16];
    for x in 0..4 {
        let column = std::array::from_fn(|y| rows[y * 4 + x]);
        let transformed = if transform.columns_adst() {
            inverse_adst(column)
        } else {
            inverse_1d(column)
        };
        for y in 0..4 {
            output[y * 4 + x] = (transformed[y] + 8) >> 4;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corrupt streams can carry any 32-bit coefficient: the inverse must
    /// wrap, not overflow (fuzzing found the overflow).
    #[test]
    fn inverse_wraps_extreme_coefficients() {
        for value in [i32::MAX, i32::MIN, 1 << 20, -(1 << 24)] {
            for transform in [
                Transform::DctDct,
                Transform::AdstDct,
                Transform::DctAdst,
                Transform::AdstAdst,
            ] {
                let output = inverse(&[value; 16], transform);
                assert!(output.iter().all(|v| (-(1 << 12)..1 << 12).contains(v)));
            }
        }
    }

    #[test]
    fn matches_original_libvpx_forward_and_inverse() {
        // Original vp9_fht4x4_c and vp9_iht4x4_16_add_c from pinned v1.16.0.
        // Inverse outputs use a 128 predictor; none clips, so subtracting 128
        // gives the exact residual and tests both axis order and rounding.
        let input = [
            17, -23, 55, -101, 37, 0, -7, 85, -92, 41, 5, -64, 3, 72, -18, 29,
        ];
        let coefficients = [
            45, -31, 17, -9, 83, -22, 6, 33, -66, 14, -20, 7, 11, -3, 24, -40,
        ];
        let forward_expected = [
            [
                78, 101, -422, -126, -117, 209, 113, 807, 58, 405, -130, 263, -738, 215, -967, 251,
            ],
            [
                122, 0, -409, -399, -53, 89, 81, 606, 160, 413, 7, 385, -724, 294, -984, 353,
            ],
            [
                69, 245, -342, -169, -199, 72, 7, 822, -72, 406, -60, 282, -712, 245, -1006, 88,
            ],
            [
                147, 179, -303, -442, -95, 3, -10, 612, 11, 395, 65, 429, -725, 320, -1019, 193,
            ],
        ];
        let inverse_expected = [
            [2, 1, 4, 5, 3, 1, 7, 6, 2, 3, -2, 8, -5, -2, -5, -5],
            [1, 0, 2, 2, 4, 1, 9, 6, 3, 4, 1, 10, -4, -2, -4, -3],
            [1, 0, 4, 6, 2, 0, 7, 7, 0, 3, -2, 8, -3, -2, -5, -6],
            [0, 0, 1, 3, 2, -1, 8, 7, 1, 4, 0, 11, -3, -1, -4, -4],
        ];
        for (index, transform) in [
            Transform::DctDct,
            Transform::AdstDct,
            Transform::DctAdst,
            Transform::AdstAdst,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(forward(&input, transform), forward_expected[index]);
            assert_eq!(inverse(&coefficients, transform), inverse_expected[index]);
        }
    }
}
