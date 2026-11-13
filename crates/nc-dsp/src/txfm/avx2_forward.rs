//! AVX2 staged HEVC forward rows on sixteen 16-bit lanes, as the NEON
//! (`neon_forward::hevc_rows`) and SIMD128 (`simd128_forward::hevc_rows`)
//! ones: the staged rows' graph (`codec_txfm::generic::staged_hevc_pass`)
//! keeps every stage of 8-bit residuals scaled by 4 within 16 bits, so one
//! vector carries sixteen lines through it. A 16-point block is then one
//! pass of the graph instead of two 8-line passes in 32-bit lanes, which
//! halves both the instructions and the code (the 32-bit passes unrolled to
//! most of the 16 KB `hevc_forward16_bounded`, the largest instruction-cache
//! consumer of HEVC Faster). Rotations multiply interleaved pairs with
//! `vpmaddwd` (exact in 32 bits) and round back to 16; equal-weight ones add
//! or subtract first and take the rounding Q15 multiply (`vpmulhrsw`). The
//! results equal the generic staged rows (checkasm).
//!
//! Kernel contract: see the `txfm` module docs.

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_cvtepi16_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_mulhrs_epi16, _mm256_mullo_epi32, _mm256_packs_epi32, _mm256_permute2x128_si256,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_slli_epi16, _mm256_sra_epi32, _mm256_storeu_si256,
    _mm256_sub_epi16, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64,
    _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};
use core::mem::MaybeUninit;

use super::forward_lanes::ForwardLanes;
use super::{cospi, dct};

/// Lines per vector.
const LANES: usize = 16;
/// Cosine bits of the staged graph.
const STAGED_COS_BIT: u32 = 12;
/// Input scale of the staged rows.
const ROW_INPUT_SHIFT: i32 = crate::codec_txfm::generic::ROW_INPUT_SHIFT;
/// The staged rows' output scale, `181 / 2^(8 + 1/2)`: about `1/sqrt(2)`
/// relative to the graph's orthonormal scaling.
const OUTPUT_SCALE: i32 = 181;
/// Bits of the Q15 multiplier of `vpmulhrsw`.
const Q15_BITS: u32 = 15;

/// Sixteen lines' samples or butterfly values in 16-bit lanes.
#[derive(Clone, Copy)]
struct Column(__m256i);

impl ForwardLanes for Column {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: this type exists only inside the AVX2 kernel below.
        unsafe { Self(_mm256_set1_epi16(value as i16)) }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // Staged-row stages fit i16 (module docs).
        // SAFETY: as `splat`.
        unsafe { Self(_mm256_add_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: as `splat`.
        unsafe { Self(_mm256_sub_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // SAFETY: as `splat`.
        unsafe {
            // Equal weights: the sum or difference fits i16 (as NEON), and
            // the rounding Q15 multiply by `w << (15 - bit)` is libaom's
            // `round_shift(x * w, bit)`.
            if (w0 == w1 || w0 == -w1) && w0.abs() < (1 << bit) {
                let sum = if w0 == w1 {
                    _mm256_add_epi16(a.0, b.0)
                } else {
                    _mm256_sub_epi16(a.0, b.0)
                };
                let weight = _mm256_set1_epi16((w0 << (Q15_BITS - bit)) as i16);
                return Self(_mm256_mulhrs_epi16(sum, weight));
            }
            // Interleaved (a, b) pairs of lines 0-3 / 8-11 and 4-7 / 12-15
            // (per 128-bit half); the saturating pack restores line order
            // and never clamps: the graph keeps every rounded result in 16
            // bits.
            let weights = _mm256_set1_epi32((w0 & 0xFFFF) | (w1 << 16));
            let round = _mm256_set1_epi32(1 << (bit - 1));
            let count = _mm_cvtsi32_si128(bit as i32);
            let low = _mm256_madd_epi16(_mm256_unpacklo_epi16(a.0, b.0), weights);
            let high = _mm256_madd_epi16(_mm256_unpackhi_epi16(a.0, b.0), weights);
            Self(_mm256_packs_epi32(
                _mm256_sra_epi32(_mm256_add_epi32(low, round), count),
                _mm256_sra_epi32(_mm256_add_epi32(high, round), count),
            ))
        }
    }
}

/// `vperm2i128` selectors: the low halves of both operands, or the high.
const LOW_HALVES: i32 = 0x20;
const HIGH_HALVES: i32 = 0x31;

/// Transpose a 16x16 tile of 16-bit values: `out[c]` lane `r` is `rows[r]`
/// lane `c`. Words, then dwords, then qwords are interleaved within each
/// 128-bit half, and the halves are finally exchanged across rows 0-7 and
/// 8-15.
#[inline(always)]
fn transpose16(rows: [__m256i; LANES]) -> [__m256i; LANES] {
    // SAFETY: called only from the AVX2 kernel below.
    unsafe {
        // `words[2p + h]`: rows 2p, 2p + 1 interleaved, columns 4h..4h + 4
        // of each half.
        let words: [__m256i; LANES] = core::array::from_fn(|k| {
            let (x, y) = (rows[k & !1], rows[k | 1]);
            if k % 2 == 0 {
                _mm256_unpacklo_epi16(x, y)
            } else {
                _mm256_unpackhi_epi16(x, y)
            }
        });
        // `dwords[4g + q]`: rows 4g..4g + 4, columns 2q, 2q + 1 of each half.
        let dwords: [__m256i; LANES] = core::array::from_fn(|k| {
            let (quad, pair) = (k / 4, k % 4);
            let half = pair / 2;
            let (first, second) = (words[4 * quad + half], words[4 * quad + 2 + half]);
            if pair % 2 == 1 {
                _mm256_unpackhi_epi32(first, second)
            } else {
                _mm256_unpacklo_epi32(first, second)
            }
        });
        // `qwords[8o + c]`: rows 8o..8o + 8, column c of each half.
        let qwords: [__m256i; LANES] = core::array::from_fn(|k| {
            let (o, c) = (k / 8, k % 8);
            let (x, y) = (dwords[8 * o + c / 2], dwords[8 * o + 4 + c / 2]);
            if c % 2 == 1 {
                _mm256_unpackhi_epi64(x, y)
            } else {
                _mm256_unpacklo_epi64(x, y)
            }
        });
        core::array::from_fn(|c| {
            let (x, y) = (qwords[c % 8], qwords[8 + c % 8]);
            if c < 8 {
                _mm256_permute2x128_si256::<LOW_HALVES>(x, y)
            } else {
                _mm256_permute2x128_si256::<HIGH_HALVES>(x, y)
            }
        })
    }
}

/// Staged HEVC rows of an `N`-point block (`N` 16 or 32) into the row-major
/// intermediate `out` (`N * N`), as `generic::staged_hevc_pass` with the
/// first-pass shift `log2(N) - 1` and an exact DC.
///
/// # Safety
///
/// AVX2 must be available. `residual` covers `N` rows of `N` 8-bit
/// differences at `stride`; `out` holds `N * N` values.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hevc_rows<const N: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [MaybeUninit<i32>],
) {
    assert!(matches!(N, 16 | 32));
    // The first pass shifts by log2(N) - 1, plus 1 for the output scale and
    // the input scale.
    let output_shift = _mm_cvtsi32_si128(N.ilog2() as i32 + ROW_INPUT_SHIFT);
    let output_round = _mm256_set1_epi32(1 << (N.ilog2() as i32 + ROW_INPUT_SHIFT - 1));
    let scale = _mm256_set1_epi32(OUTPUT_SCALE);
    let mut input = [Column::splat(0); N];
    let mut output = input;
    for r0 in (0..N).step_by(LANES) {
        for c0 in (0..N).step_by(LANES) {
            let rows = core::array::from_fn(|i| {
                let row = &residual[(r0 + i) * stride + c0..][..LANES];
                // SAFETY: `row` holds sixteen samples (32 bytes).
                unsafe { _mm256_loadu_si256(row.as_ptr().cast()) }
            });
            for (i, column) in transpose16(rows).into_iter().enumerate() {
                input[c0 + i] = Column(_mm256_slli_epi16::<ROW_INPUT_SHIFT>(column));
            }
        }
        // The scaled inputs sum to at most 32 · 4 · 255 = 32640: i16.
        let dc = input.iter().fold(Column::splat(0), |sum, &v| sum.add(v));
        match N {
            16 => dct::fdct16::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(STAGED_COS_BIT),
                STAGED_COS_BIT,
            ),
            32 => dct::fdct32::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(STAGED_COS_BIT),
                STAGED_COS_BIT,
            ),
            _ => unreachable!(),
        }
        for k in (0..N).step_by(LANES) {
            let tile = transpose16(core::array::from_fn(|i| output[k + i].0));
            for (i, row) in tile.into_iter().enumerate() {
                let halves = [
                    _mm256_castsi256_si128(row),
                    _mm256_extracti128_si256::<1>(row),
                ];
                let dest = &mut out[(r0 + i) * N + k..][..LANES];
                for (half, dest) in halves
                    .into_iter()
                    .zip(dest.as_chunks_mut::<{ LANES / 2 }>().0)
                {
                    let product = _mm256_mullo_epi32(_mm256_cvtepi16_epi32(half), scale);
                    let value =
                        _mm256_sra_epi32(_mm256_add_epi32(product, output_round), output_shift);
                    // SAFETY: `dest` holds eight `i32` slots (32 bytes).
                    unsafe { _mm256_storeu_si256(dest.as_mut_ptr().cast(), value) };
                }
            }
        }
        // The exact DC: 64 · sum >> (log2(N) - 1) of the unscaled inputs,
        // i.e. the scaled sum << (5 - log2(N)).
        let mut dc_lanes = [0_i16; LANES];
        // SAFETY: `dc_lanes` holds sixteen `i16`s (32 bytes).
        unsafe { _mm256_storeu_si256(dc_lanes.as_mut_ptr().cast(), dc.0) };
        for (i, value) in dc_lanes.into_iter().enumerate() {
            out[(r0 + i) * N].write(i32::from(value) << (5 - N.ilog2()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LANES, transpose16};
    use core::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};

    #[test]
    fn transpose16_transposes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let tile: [[i16; LANES]; LANES] =
            core::array::from_fn(|r| core::array::from_fn(|c| (r * LANES + c) as i16));
        // SAFETY: AVX2 is available (checked above); every row holds sixteen
        // values.
        let out = unsafe {
            let rows = core::array::from_fn(|r| _mm256_loadu_si256(tile[r].as_ptr().cast()));
            transpose16(rows).map(|v| {
                let mut row = [0i16; LANES];
                _mm256_storeu_si256(row.as_mut_ptr().cast(), v);
                row
            })
        };
        for (c, row) in out.iter().enumerate() {
            for (r, &value) in row.iter().enumerate() {
                assert_eq!(value, tile[r][c], "row {r} column {c}");
            }
        }
    }
}
