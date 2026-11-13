//! The butterfly kernels against a plain matrix-product statement of AVM's
//! passes (and the typed matrix kernels at `DCT_DCT` against the
//! butterflies), and the derived matrices against AVM rows. End-to-end goldens from
//! the original AVM C live with the quantiser in `nc-av2`'s `transform.rs`.

use super::{Av2TxSize, DCT_BASIS, MAX_POINTS};
use crate::{Dsp, Isa};

/// Full `n`-point matrix entry `(k, j)`, mirroring [`DCT_BASIS`].
fn matrix(n: usize, k: usize, j: usize) -> i64 {
    let row = &DCT_BASIS[k * (MAX_POINTS / n)];
    let value = if j < n / 2 {
        row[j]
    } else if k.is_multiple_of(2) {
        row[n - 1 - j]
    } else {
        -row[n - 1 - j]
    };
    i64::from(value)
}

fn round_shift(value: i64, shift: u32) -> i64 {
    (value + (1 << (shift - 1))) >> shift
}

/// AVM `fwd_txfm_c` as two plain matrix products.
fn forward_reference(size: Av2TxSize, residual: &[i16]) -> Vec<i32> {
    let (n, c) = (size.side(), size.coded_side());
    let (first, second) = size.forward_shifts();
    let mut columns = vec![0i64; c * n];
    for k in 0..c {
        for x in 0..n {
            let sum: i64 = (0..n)
                .map(|y| matrix(n, k, y) * i64::from(residual[y * n + x]))
                .sum();
            columns[k * n + x] = round_shift(sum, first);
        }
    }
    let mut out = vec![0; c * c];
    for v in 0..c {
        for u in 0..c {
            let sum: i64 = (0..n).map(|x| matrix(n, u, x) * columns[v * n + x]).sum();
            out[v * c + u] = round_shift(sum, second) as i32;
        }
    }
    out
}

/// AVM `inv_txfm_c` as two plain matrix products with clamps, plus the 2x2
/// repetition of 64x64.
fn inverse_reference(size: Av2TxSize, coeffs: &[i32]) -> Vec<i32> {
    let c = size.coded_side();
    let (first, second) = size.inverse_shifts();
    let input: Vec<i64> = coeffs
        .iter()
        .map(|&v| i64::from(v.clamp(-32768, 32767)))
        .collect();
    let mut rows = vec![0i64; c * c];
    for r in 0..c {
        for x in 0..c {
            let sum: i64 = (0..c).map(|k| matrix(c, k, x) * input[r * c + k]).sum();
            rows[r * c + x] = round_shift(sum, first).clamp(-32768, 32767);
        }
    }
    let n = size.side();
    let scale = n / c;
    let mut out = vec![0; n * n];
    for y in 0..c {
        for x in 0..c {
            let sum: i64 = (0..c).map(|k| matrix(c, k, y) * rows[k * c + x]).sum();
            let sample = round_shift(sum, second).clamp(-256, 255) as i32;
            for dy in 0..scale {
                for dx in 0..scale {
                    out[(y * scale + dy) * n + x * scale + dx] = sample;
                }
            }
        }
    }
    out
}

/// Deterministic xorshift values.
struct Values(u32);

impl Values {
    fn below(&mut self, bound: u32) -> i32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 % bound) as i32
    }
}

#[test]
fn matrices_reproduce_avm_rows() {
    let row = |n: usize, k: usize| -> Vec<i64> { (0..n).map(|j| matrix(n, k, j)).collect() };
    assert_eq!(row(4, 1), [83, 35, -35, -83]);
    assert_eq!(row(4, 3), [35, -83, 83, -35]);
    assert_eq!(row(8, 1), [89, 75, 50, 18, -18, -50, -75, -89]);
    assert_eq!(row(16, 1)[..8], [90, 87, 80, 70, 57, 43, 26, 9]);
    assert_eq!(
        row(32, 1)[..16],
        [
            90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4
        ]
    );
    assert_eq!(row(64, 63)[..4], [2, -7, 11, -15]);
    for n in [4, 8, 16, 32, 64] {
        assert!(row(n, 0).iter().all(|&w| w == 64));
    }
}

#[test]
fn butterflies_equal_matrix_products() {
    let dsp = Dsp::with_isa(Isa::Scalar).unwrap();
    let mut values = Values(0x2545_f491);
    for size in Av2TxSize::ALL {
        let (n, c) = (size.side(), size.coded_side());
        for mode in 0..6 {
            let residual: Vec<i16> = (0..n * n)
                .map(|i| match mode {
                    1 => -255,
                    2 if !(i % n + i / n).is_multiple_of(2) => -255,
                    0 | 2 => 255,
                    _ => (values.below(511) - 255) as i16,
                })
                .collect();
            let mut forward = vec![0; c * c];
            dsp.av2_forward(size, &residual, n, &mut forward);
            assert_eq!(
                forward,
                forward_reference(size, &residual),
                "{size:?} {mode}"
            );

            let coeffs: Vec<i32> = (0..c * c)
                .map(|i| match mode {
                    0 => 40_000,
                    1 if i % 2 == 0 => 32767,
                    1 => -32768,
                    2 => values.below(1 << 16) - (1 << 15),
                    3 if values.below(8) == 0 => values.below(4001) - 2000,
                    3 => 0,
                    _ => forward[i],
                })
                .collect();
            let mut inverse = vec![0; n * n];
            dsp.av2_inverse(size, &coeffs, &mut inverse);
            assert_eq!(inverse, inverse_reference(size, &coeffs), "{size:?} {mode}");
        }
    }
}

#[test]
fn typed_dct_matrices_equal_the_butterflies() {
    use super::typed::{self, Av2TxType};
    let dsp = Dsp::with_isa(Isa::Scalar).unwrap();
    let mut values = Values(0x9e37_79b9);
    for size in &Av2TxSize::ALL[..4] {
        let n = size.side();
        let residual: Vec<i16> = (0..n * n)
            .map(|_| (values.below(511) - 255) as i16)
            .collect();
        let (mut butterfly, mut matrix) = (vec![0; n * n], vec![0; n * n]);
        dsp.av2_forward(*size, &residual, n, &mut butterfly);
        typed::forward(*size, Av2TxType::DCT, &residual, n, &mut matrix);
        assert_eq!(butterfly, matrix, "forward {size:?}");
        let coeffs: Vec<i32> = (0..n * n)
            .map(|_| values.below(1 << 16) - (1 << 15))
            .collect();
        dsp.av2_inverse(*size, &coeffs, &mut butterfly);
        typed::inverse(*size, Av2TxType::DCT, &coeffs, &mut matrix);
        assert_eq!(butterfly, matrix, "inverse {size:?}");
    }
}

#[test]
fn dc_residual_equals_the_full_inverse() {
    let dsp = Dsp::with_isa(Isa::Scalar).unwrap();
    let mut values = Values(0x1234_5679);
    for size in Av2TxSize::ALL {
        let (n, c) = (size.side(), size.coded_side());
        let extremes = [
            0,
            1,
            -1,
            7,
            -8,
            32767,
            -32768,
            40_000,
            -40_000,
            i32::MAX,
            i32::MIN,
        ];
        let random = (0..200).map(|_| values.below(1 << 17) - (1 << 16));
        for dc in extremes.into_iter().chain(random) {
            let mut coeffs = vec![0; c * c];
            coeffs[0] = dc;
            let mut inverse = vec![0; n * n];
            dsp.av2_inverse(size, &coeffs, &mut inverse);
            let expected = size.inverse_dc_residual(dc);
            assert!(inverse.iter().all(|&r| r == expected), "{size:?} dc {dc}");
        }
    }
}
