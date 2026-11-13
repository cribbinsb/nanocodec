//! AV1 transform and quantisation kernels: `DCT_DCT` forward and inverse
//! for 4x4 to 32x32, and quantise / dequantise.
//!
//! - **Forward DCT** (encoder only): libaom's integer `fwd_txfm2d` for square
//!   `DCT_DCT` (`av1/encoder/av1_fwd_txfm2d.c`): input shifted left by 2,
//!   column transform, rounding shift, row transform. Coefficients have
//!   exactly the scale AV1's quantiser tables assume.
//! - **Inverse DCT + add** (normative, AV1 §7.13.3): libaom's
//!   `inv_txfm2d_add_c`: row transform on 16-bit-clamped input, rounding
//!   shift, column transform on 16-bit-clamped input, `round_shift(.., 4)`,
//!   added to the prediction with 8-bit clipping. Every butterfly stage is
//!   clamped to 16 bits (the 8-bit stage range).
//! - **Quantise** (encoder): `level = (|c| * 2^log_scale * 128 +
//!   step * rounding_q7) / (step * 128)`, capped at `max_level`, signed like
//!   `c`. With `log_scale = 0` this is `nc-av1`'s quantiser. SIMD versions
//!   divide exactly by a multiply and a shift (`Reciprocal`), which needs
//!   `|c| < 2^22` (transform outputs are far smaller).
//! - **Dequantise** (normative, AV1 §7.12.3): the magnitude
//!   `((|level| * step) & 0xFFFFFF) >> log_scale`, signed, clamped to 16
//!   bits. `log_scale` is 0 up to 16x16 and 1 for 32x32 (`dqDenom`).
//!
//! Coefficients use libaom's layout throughout: coefficient (vertical
//! frequency `r`, horizontal frequency `c`) of an `n`x`n` block is at
//! `c * n + r`. Step 0 is the DC coefficient.
//!
//! The 1-D butterflies (`dct.rs`) are generated from libaom and generic over
//! small lane traits (`ForwardLanes`, `InverseLanes`) that every
//! [`I32Lanes`](crate::simd::I32Lanes) type implements: each ISA runs them on
//! 4, 8 or 16 columns at once; the scalar reference runs the same code on one
//! `i32` lane, computing `half_btf` in 64 bits as libaom does. The NEON 32-
//! and 64-point drivers also run them on eight 16-bit lanes, behind input L1
//! guards whose range proofs are in `range_probe.rs`.
//!
//! # Ranges
//!
//! SIMD implementations compute `half_btf` in 32 bits. That is exact because
//! every product sum stays below `2^30`: inverse inputs are clamped to 16
//! bits and weights are at most `2^12`; forward inputs are residuals of 8-bit
//! samples (`|r| <= 255`, a contract of the forward kernel), so after the
//! input shift and the growth of an `n`-point transform the largest sum is
//! about `2^28.5` (32x32 rows: `32 * 2040 * 4096`).
//!
//! # Kernel contract
//!
//! Every kernel is an `unsafe fn`, called only from [`crate::Dsp`], which
//! guarantees the CPU supports the kernel's ISA. Slice arguments are indexed
//! with bounds checks inside the kernels, so short slices panic rather than
//! read out of bounds; `Dsp` checks lengths up front for clear messages.

pub(crate) mod dct;
mod forward_lanes;
#[allow(unsafe_code)]
pub(crate) mod generic;
pub(crate) mod inverse_lanes;
#[cfg(test)]
mod range_probe;
#[allow(unsafe_code)]
pub(crate) mod scalar;
pub(crate) mod sparse_lanes;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512_quant;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2_forward;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon_forward;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon_inverse;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2_inverse;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
pub(crate) mod narrow_inverse;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128_inverse;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128_forward;

/// Square DCT sizes the transform kernels support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DctSize {
    /// 4x4.
    N4 = 0,
    /// 8x8.
    N8 = 1,
    /// 16x16.
    N16 = 2,
    /// 32x32.
    N32 = 3,
}

impl DctSize {
    /// Every size, smallest first.
    pub const ALL: [DctSize; 4] = [DctSize::N4, DctSize::N8, DctSize::N16, DctSize::N32];

    /// The size with side `n`, or `None` if `n` is not 4, 8, 16 or 32.
    #[must_use]
    pub const fn from_side(n: usize) -> Option<Self> {
        match n {
            4 => Some(Self::N4),
            8 => Some(Self::N8),
            16 => Some(Self::N16),
            32 => Some(Self::N32),
            _ => None,
        }
    }

    /// Side length in samples.
    #[must_use]
    pub const fn side(self) -> usize {
        4 << self as usize
    }

    /// Coefficients in the block (`side^2`).
    #[must_use]
    pub const fn area(self) -> usize {
        self.side() * self.side()
    }
}

/// Parameters of AV1 quantisation and dequantisation for one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1Quantizer {
    /// Quantiser step of coefficient 0 (`dc_q`).
    pub dc_step: i32,
    /// Quantiser step of every other coefficient (`ac_q`).
    pub ac_step: i32,
    /// Encoder rounding offset in 1/128 of a step (64 rounds to nearest;
    /// smaller values form a deadzone). Ignored by dequantisation.
    pub rounding_q7: i32,
    /// Largest level magnitude to emit. Ignored by dequantisation.
    pub max_level: i32,
    /// `log2(dqDenom)`: 0 up to 16x16, 1 for 32x32.
    pub log_scale: u32,
}

impl Av1Quantizer {
    fn step(&self, index: usize) -> i32 {
        if index == 0 {
            self.dc_step
        } else {
            self.ac_step
        }
    }
}

/// Dequantised magnitudes wrap at 24 bits (AV1 §7.12.3).
pub(crate) const DEQUANT_MASK: i64 = 0xff_ffff;

/// Transpose an `n`x`n` coefficient block `src` into `dst` (both `n^2`).
pub(crate) type TransposeFn = unsafe fn(src: &[i32], dst: &mut [i32]);
/// Forward DCT of an `n`x`n` residual at `stride` into `out` (`n^2`).
pub(crate) type ForwardDctFn = unsafe fn(residual: &[i16], stride: usize, out: &mut [i32]);
/// [`ForwardDctFn`] that returns false, leaving `out` untouched, when every
/// coefficient provably has magnitude at most `zero_limit`.
pub(crate) type ForwardBoundedFn =
    unsafe fn(residual: &[i16], stride: usize, zero_limit: u32, out: &mut [i32]) -> bool;
/// Inverse DCT of `coeffs` (`n^2`), added to `dst` (`n` rows at `stride`).
pub(crate) type InverseDctAddFn = unsafe fn(coeffs: &[i32], dst: &mut [u8], dst_stride: usize);
/// Sparse large inverse; only a low8/16/32 coefficient square is nonzero.
pub(crate) type SparseInverseDctAddFn = unsafe fn(&[i32], &mut [u8], usize, usize);
/// Exact unnormalized gain with `|original| < 2^22`, signed16-bit dequantized values.
pub(crate) type CoefficientGainFn = unsafe fn(&[i32], &[i32]) -> i64;
/// Quantise in place; returns the number of non-zero levels.
pub(crate) type QuantizeFn = unsafe fn(coeffs: &mut [i32], q: &Av1Quantizer) -> u32;
/// Quantise `coeffs` into `out` (at least as long), keeping `coeffs`;
/// returns the number of non-zero levels.
pub(crate) type QuantizeFromFn =
    unsafe fn(coeffs: &[i32], q: &Av1Quantizer, out: &mut [i32]) -> u32;
/// Dequantise `levels` into `out` (same length).
pub(crate) type DequantizeFn = unsafe fn(levels: &[i32], q: &Av1Quantizer, out: &mut [i32]);

/// One ISA's transform kernels, indexed by [`DctSize`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct TxfmFns {
    pub forward_dct: [ForwardDctFn; 4],
    pub inverse_dct_add: [InverseDctAddFn; 4],
    /// 64x64, keeping (forward) or taking (inverse) only the top-left 32x32
    /// coefficients: not a [`DctSize`], whose sizes keep every coefficient.
    pub forward_dct_64: ForwardDctFn,
    pub inverse_dct_add_64: InverseDctAddFn,
    pub inverse_dct_add_sparse: [SparseInverseDctAddFn; 2],
    pub coefficient_gain: CoefficientGainFn,
    pub quantize: QuantizeFn,
    pub quantize_from: QuantizeFromFn,
    pub dequantize: DequantizeFn,
    /// Column-major <-> raster coefficient layout, indexed by [`DctSize`].
    pub transpose: [TransposeFn; 4],
    /// Forward DCTs storing raster coefficients directly, where an ISA has
    /// them (otherwise the forward kernel and [`Self::transpose`]).
    pub forward_dct_raster: [Option<ForwardDctFn>; 4],
    /// Low-frequency forward DCTs of 16x16, 32x32 and 64x64 blocks (only
    /// the [`low_keep`] lowest frequencies per dimension), where an ISA has
    /// them (otherwise the full forward with the rest zeroed).
    pub forward_dct_low: [Option<ForwardDctFn>; 3],
    /// The same for 16x16 and 32x32 with raster coefficients.
    pub forward_dct_raster_low: [Option<ForwardDctFn>; 2],
    /// Raster 16x16 and 32x32 forwards with a proven-zero test
    /// (`generic::dct_bounded!`).
    pub forward_dct_raster_bounded: [ForwardBoundedFn; 2],
}

/// Accumulated rounding of the 16- and 32-point forward row graphs (12-bit
/// cosines), in output units: a row output is at most its input line's L1
/// norm plus this (`range_probe` checks both the coefficient and the
/// rounding bound). The bounded forwards' proven-zero test.
pub(crate) const FORWARD_ROW_ROUNDING: [u32; 2] = [3, 7];

/// Frequencies per dimension a low-frequency forward of an `n`-point block
/// keeps (an encoder-side approximation: AV1 coefficients beyond them are
/// coded as zero, SVT's TX shortcut): 8 of 16, 16 of 32 and of 64.
#[must_use]
pub const fn low_keep(n: usize) -> usize {
    if n == 16 { 8 } else { 16 }
}

/// libaom `av1_cospi_arr_data` for 10, 12 and 13 bits:
/// `round(2^bits * cos(i * pi / 128))`.
pub(crate) const COSPI_10: [i32; 64] = [
    1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980, 972, 964, 955, 946,
    936, 926, 915, 903, 891, 878, 865, 851, 837, 822, 807, 792, 775, 759, 742, 724, 706, 688, 669,
    650, 630, 610, 590, 569, 548, 526, 505, 483, 460, 438, 415, 392, 369, 345, 321, 297, 273, 249,
    224, 200, 175, 150, 125, 100, 75, 50, 25,
];
pub(crate) const COSPI_12: [i32; 64] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101,
];
pub(crate) const COSPI_13: [i32; 64] = [
    8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946, 7895, 7839, 7779, 7713, 7643,
    7568, 7489, 7405, 7317, 7225, 7128, 7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933,
    5793, 5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038, 3862, 3683, 3503, 3320,
    3135, 2948, 2760, 2570, 2378, 2185, 1990, 1795, 1598, 1401, 1202, 1003, 803, 603, 402, 201,
];

/// The cosine table for `cos_bit` (10, 12 or 13).
pub(crate) fn cospi(cos_bit: u32) -> &'static [i32; 64] {
    match cos_bit {
        10 => &COSPI_10,
        12 => &COSPI_12,
        _ => &COSPI_13,
    }
}

/// Exact division by a constant `divisor` (not a power of two, at least 512)
/// as a multiply and a shift, for numerators below `2^31`
/// (Granlund–Montgomery): with `shift = 31 + ceil(log2 divisor)` and
/// `multiplier = floor(2^shift / divisor) + 1`, the error
/// `multiplier * divisor - 2^shift` lies in `(0, divisor]`, so
/// `n * error < 2^31 * divisor <= 2^shift` and
/// `(n * multiplier) >> shift == n / divisor` for every `n < 2^31`.
/// For divisors of at least 512 that are not powers of two the multiplier
/// fits 32 bits, so SIMD can use 32x32->64-bit multiplies. Quantiser
/// numerators stay below `2^31` under the coefficient contract `|c| < 2^22`
/// (`2^22 * 128 * 2 + 1828 * 127 < 2^31`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reciprocal {
    pub multiplier: u32,
    pub shift: u32,
}

impl Reciprocal {
    /// The reciprocal of `divisor`, or `None` for powers of two and divisors
    /// below 512 (callers use the reference division there).
    pub(crate) fn new(divisor: u32) -> Option<Self> {
        if divisor < 512 || divisor.is_power_of_two() {
            return None;
        }
        let shift = 31 + (32 - (divisor - 1).leading_zeros());
        let multiplier = (1u64 << shift) / u64::from(divisor) + 1;
        Some(Self {
            multiplier: u32::try_from(multiplier).ok()?,
            shift,
        })
    }
}

/// The quantiser numerator and divisor of a coefficient magnitude at step
/// `step`: `level = numerator / divisor`.
#[inline]
pub(crate) fn quant_terms(q: &Av1Quantizer, step: i32) -> (u32, u32) {
    let offset = (step * q.rounding_q7) as u32;
    (offset, (step * 128) as u32)
}

/// Quantise one coefficient: the reference every ISA matches.
#[inline]
pub(crate) fn quantize_one(coeff: i32, index: usize, q: &Av1Quantizer) -> i32 {
    let step = i64::from(q.step(index));
    let scaled = i64::from(coeff.unsigned_abs()) << q.log_scale;
    let magnitude = (scaled * 128 + step * i64::from(q.rounding_q7)) / (step * 128);
    let level = magnitude.min(i64::from(q.max_level)) as i32;
    if coeff < 0 { -level } else { level }
}

/// Dequantise one level: the reference every ISA matches (AV1 §7.12.3).
#[inline]
pub(crate) fn dequantize_one(level: i32, index: usize, q: &Av1Quantizer) -> i32 {
    let product = i64::from(level.unsigned_abs()) * i64::from(q.step(index));
    let magnitude = ((product & DEQUANT_MASK) >> q.log_scale) as i32;
    let signed = if level < 0 { -magnitude } else { magnitude };
    signed.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

#[cfg(test)]
mod sparse_tests {
    use super::{COSPI_12, dct};

    #[test]
    fn gain_dispatch_matches_original_formula_at_extremes_and_tails() {
        use crate::{Dsp, Isa};
        let reference = Dsp::with_isa(Isa::Scalar).unwrap();
        let cs = [-(1 << 22) + 1, -32768, -1, 0, 1, 32767, (1 << 22) - 1];
        let ds = [-32768, -32767, -1, 0, 1, 32766, 32767];
        for n in [
            0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 255, 256, 1023, 1024,
        ] {
            for offset in 0..8 {
                let mut c = vec![0_i32; n + offset];
                let mut d = vec![0_i32; n + offset];
                for seed in 0..8_u32 {
                    let mut state = seed + 17;
                    for i in 0..n {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        c[offset + i] = if seed < 4 {
                            cs[(i + seed as usize) % cs.len()]
                        } else {
                            ((state >> 8) % ((1 << 23) - 1)) as i32 - ((1 << 22) - 1)
                        };
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        d[offset + i] = if seed < 4 {
                            ds[(i + offset + seed as usize) % ds.len()]
                        } else {
                            (state >> 16) as i32 - 32768
                        };
                    }
                    let expected = reference.av1_coefficient_gain(&c[offset..], &d[offset..]);
                    for isa in Isa::ALL {
                        if let Some(dsp) = Dsp::with_isa(isa) {
                            assert_eq!(
                                dsp.av1_coefficient_gain(&c[offset..], &d[offset..]),
                                expected,
                                "{isa:?} len{n} offset{offset} seed{seed}"
                            );
                        }
                    }
                }
            }
        }
        for isa in Isa::ALL {
            if let Some(dsp) = Dsp::with_isa(isa) {
                let original = vec![-(1 << 22) + 1; 1024];
                let d = vec![32767; 1024];
                let gain = dsp.av1_coefficient_gain(&original, &d);
                assert!(gain < 0);
                assert_eq!(gain, reference.av1_coefficient_gain(&original, &d));
            }
        }
    }

    #[test]
    fn specialized_graphs_equal_full_before_pixel_clipping() {
        type Inverse<const N: usize> = fn(&[i32; N], &mut [i32; N], &[i32; 64], u32);
        fn check<const N: usize>(full: Inverse<N>, sparse: Inverse<N>, limit: usize) {
            let values = [-32_768, -32_767, -1, 0, 1, 32_766, 32_767];
            for seed in 0..128_u32 {
                let mut state = seed + 3;
                let mut input = [0_i32; N];
                for (i, value) in input[..limit].iter_mut().enumerate() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = match seed % 4 {
                        0 => values[(i + seed as usize) % values.len()],
                        1 => {
                            if i + 1 == limit {
                                values[seed as usize % values.len()]
                            } else {
                                0
                            }
                        }
                        2 => (state >> 16) as i32 - 32_768,
                        _ => (state >> 24) as i32 - 127,
                    };
                }
                let mut expected = [0x1234; N];
                let mut actual = [0x5678; N];
                full(&input, &mut expected, &COSPI_12, 12);
                sparse(&input, &mut actual, &COSPI_12, 12);
                assert_eq!(actual, expected, "N{N} low{limit} seed{seed}");
            }
        }
        check(dct::idct32, dct::idct32_low8, 8);
        check(dct::idct32, dct::idct32_low16, 16);
        check(dct::idct64, dct::idct64_low8, 8);
        check(dct::idct64, dct::idct64_low16, 16);
        check(dct::idct64, dct::idct64_low32, 32);
    }
}
