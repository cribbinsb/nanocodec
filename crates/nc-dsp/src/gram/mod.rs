//! Gram matrices of short 16-bit rows: `Σ_x rows[i][x] · rows[j][x]` for
//! every pair `i ≤ j`, added into a packed upper triangle of 64-bit sums.
//!
//! The statistics of least-squares filter design: each row holds one
//! feature (or the target error) over a run of samples, and the Gram matrix
//! is the normal equations' autocorrelation and cross-correlation. The AV1
//! encoder's Wiener restoration search (`nc-av1` `restoration`) uses it;
//! any codec fitting a symmetric filter by least squares can share it.
//!
//! Kernels widen 16-bit products into 32-bit lanes and reduce each pair's
//! lanes once per call: [`GRAM_MAX_MAGNITUDE`] and [`GRAM_ROW_MAX_LEN`]
//! keep every 32-bit lane exact.

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use crate::cpu::Isa;

/// Most rows (features plus target) one call may take.
pub const GRAM_MAX_ROWS: usize = 16;

/// Row lengths are a multiple of this (callers pad with zeros, which add
/// nothing to the sums).
pub const GRAM_ROW_ALIGN: usize = 16;

/// Longest row one call may take: with every value within
/// [`GRAM_MAX_MAGNITUDE`], a 32-bit lane of eight-value vectors (two
/// products per lane and vector) sums at most 2^10 products of at most 2^20,
/// below 2^31.
pub const GRAM_ROW_MAX_LEN: usize = 4096;

/// Largest magnitude of a row value: four 8-bit samples less four times a
/// fifth (a folded Wiener feature) stay within ±1020.
pub const GRAM_MAX_MAGNITUDE: i16 = 1024;

/// Entries of the packed upper triangle of an `n`-row Gram matrix.
#[must_use]
pub const fn gram_terms(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Where entry `(i, j)`, `i ≤ j < n`, lies in the packed upper triangle of
/// an `n`-row Gram matrix (row by row).
#[must_use]
pub const fn gram_index(n: usize, i: usize, j: usize) -> usize {
    i * n - i * (i + 1) / 2 + j
}

/// Add `Σ_x rows[i][x] · rows[j][x]` into `out[gram_index(n, i, j)]` for
/// every `i ≤ j < n = rows.len()`. Rows share one length, a multiple of
/// [`GRAM_ROW_ALIGN`] up to [`GRAM_ROW_MAX_LEN`], values within
/// ±[`GRAM_MAX_MAGNITUDE`]; `out` holds at least [`gram_terms`]`(n)`
/// entries. The dispatch wrapper checks the lengths.
pub(crate) type GramRowsFn = unsafe fn(rows: &[&[i16]], out: &mut [i64]);

/// The kernel of `isa`.
pub(crate) fn kernel(isa: Isa) -> GramRowsFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::gram_rows,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::gram_rows,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::gram_rows,
        _ => gram_rows,
    }
}

/// Scalar reference [`GramRowsFn`].
fn gram_rows(rows: &[&[i16]], out: &mut [i64]) {
    let n = rows.len();
    for i in 0..n {
        for j in i..n {
            let sum: i64 = rows[i]
                .iter()
                .zip(rows[j])
                .map(|(&a, &b)| i64::from(a) * i64::from(b))
                .sum();
            out[gram_index(n, i, j)] += sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_index_walks_the_upper_triangle_in_order() {
        for n in 1..=GRAM_MAX_ROWS {
            let mut next = 0;
            for i in 0..n {
                for j in i..n {
                    assert_eq!(gram_index(n, i, j), next);
                    next += 1;
                }
            }
            assert_eq!(next, gram_terms(n));
        }
    }

    #[test]
    fn scalar_sums_products_and_accumulates() {
        let a = [1i16; GRAM_ROW_ALIGN];
        let b: Vec<i16> = (0..GRAM_ROW_ALIGN as i16).collect();
        let mut out = [5i64; 3];
        gram_rows(&[&a, &b], &mut out);
        assert_eq!(out, [5 + 16, 5 + 120, 5 + 1240]);
    }
}
