//! AV1 square transforms for 4x4 to 64x64: `DCT_DCT` at every size, and
//! the other transform types where they exist (intra blocks up to 16x16).
//!
//! Every size runs on nc-dsp's kernels (SIMD, bit-exact with libaom's
//! `av1_fwd_txfm2d.c` / `av1_inv_txfm2d.c`). The inverse is normative (AV1
//! §7.13.3); the forward is encoder-only but uses libaom's scale, which the
//! quantiser tables assume.
//!
//! A 64x64 transform only ever codes its top-left 32x32 coefficients (AV1
//! §5.11.39: the rest are zero), so callers see 64x64 transforms as 32x32
//! arrays of coefficients ([`TxSize::coded_size`]).
//!
//! Coefficient layout follows libaom throughout this crate: coefficient
//! (row `r`, column `c`) of an `n`×`n` block is stored at `c * n + r`.

use nc_dsp::{Av1Quantizer, Av1TxType, DctSize, Dsp};

/// AV1's 12-bit `cospi[32]`, used once in each DC-only inverse pass.
const INVERSE_COS_PI_OVER_FOUR: i32 = 2896;
const INVERSE_COS_BITS: u32 = 12;
const INVERSE_COLUMN_SHIFT: u32 = 4;

pub use crate::tx_size::TxSize;

/// Forward 2-D DCT of the `n`×`n` residual (row-major at `stride`) into
/// `out` in libaom's column-major layout. For 64x64 only the coded
/// top-left 32x32 coefficients are returned (`out` holds `coded_size()²`).
pub fn forward_dct(dsp: &Dsp, tx: TxSize, residual: &[i16], stride: usize, out: &mut [i16]) {
    match DctSize::from_side(tx.size()) {
        Some(size) => dsp.av1_forward_dct(size, residual, stride, out),
        None => dsp.av1_forward_dct_64(residual, stride, out),
    }
}

/// Which blocks the encoder transforms with the low-frequency forward
/// (`av1.partial_forward`, `av1.partial_forward16_min_q`): only the
/// [`nc_dsp::av1_low_keep`] lowest frequencies per dimension are computed,
/// the rest are coded as zero. Encoder-side only: the decoder inverts what
/// was coded, exactly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForwardPolicy {
    /// 32x32 and 64x64 blocks keep their low 16x16.
    pub large: bool,
    /// 16x16 blocks keep their low 8x8.
    pub medium: bool,
}

impl ForwardPolicy {
    /// Whether `tx` uses the low-frequency forward (a square DCT
    /// approximation: never on rectangles).
    #[must_use]
    pub fn low(self, tx: TxSize) -> bool {
        if !tx.is_square() {
            return false;
        }
        match tx.size() {
            32 | 64 => self.large,
            16 => self.medium,
            _ => false,
        }
    }

    /// Leading coefficients of `tx`'s coded block (libaom layout, one line
    /// of `coded_size()` per horizontal frequency) that can be nonzero under
    /// this policy: all of them, or with the low-frequency forward the first
    /// [`nc_dsp::av1_low_keep`] lines (the later ones transform to zero).
    #[must_use]
    pub fn coded_prefix(self, tx: TxSize) -> usize {
        if self.low(tx) {
            nc_dsp::av1_low_keep(tx.size()) * tx.coded_size()
        } else {
            tx.coded_area()
        }
    }
}

/// Quantise `tx`'s forward coefficients in place (`levels`, the whole coded
/// block) where `policy` lets them be nonzero; the rest are zero already and
/// stay zero. Returns the number of nonzero levels. Exact: a low-frequency
/// 32x32 forward quantised only its first 16 of 32 lines halves the work
/// (2026-11-11, x86 AV1 Faster).
pub fn quantize_coded(
    dsp: &Dsp,
    policy: ForwardPolicy,
    tx: TxSize,
    levels: &mut [i16],
    quantizer: &Av1Quantizer,
) -> u32 {
    dsp.av1_quantize(&mut levels[..policy.coded_prefix(tx)], quantizer)
}

/// [`quantize_coded`] out of place: `coeffs` (the whole coded block, kept)
/// into `levels`, whose lines beyond the policy's prefix are zeroed.
pub fn quantize_coded_from(
    dsp: &Dsp,
    policy: ForwardPolicy,
    tx: TxSize,
    coeffs: &[i16],
    quantizer: &Av1Quantizer,
    levels: &mut [i16],
) -> u32 {
    let prefix = policy.coded_prefix(tx);
    levels[prefix..coeffs.len()].fill(0);
    dsp.av1_quantize_from(&coeffs[..prefix], quantizer, &mut levels[..prefix])
}

/// Leading dequantised coefficients [`inverse_dct_add`] reads for a
/// block with end of block `eob` (at least 1): the lines of the EOB-bounded
/// square for 32x32 and 64x64, else the whole coded block. Every later
/// level is zero, so it dequantises to zero and adds nothing to
/// [`coefficient_gain`] either.
#[must_use]
pub fn dequantized_extent(tx: TxSize, eob: usize) -> usize {
    if matches!(tx, TxSize::TX_32X32 | TxSize::TX_64X64) {
        nc_dsp::av1_dct_nonzero_side(eob) * tx.coded_size()
    } else {
        tx.coded_area()
    }
}

/// [`forward_dct`], or the low-frequency forward where `policy` says so.
pub fn forward_dct_with(
    dsp: &Dsp,
    policy: ForwardPolicy,
    tx: TxSize,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    if policy.low(tx) {
        dsp.av1_forward_dct_low(tx.size(), residual, stride, out);
    } else {
        forward_dct(dsp, tx, residual, stride, out);
    }
}

/// Forward transform of type `kind`: [`forward_dct_with`] for a square
/// `DCT_DCT`, the full forward of the size and type otherwise (the
/// low-frequency forward of `policy` is a square-DCT approximation only).
pub fn forward_with(
    dsp: &Dsp,
    policy: ForwardPolicy,
    tx: TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    if kind == Av1TxType::DctDct && tx.is_square() {
        forward_dct_with(dsp, policy, tx, residual, stride, out);
    } else {
        dsp.av1_forward(tx.dsp(), kind, residual, stride, out);
    }
}

/// Inverse transform of type `kind` added to the prediction in `dst`, as
/// [`inverse_dct_add`] for a square `DCT_DCT`; other sizes and types run
/// the normative inverse of every type (`Dsp::av1_inverse_add_i16`), which
/// may zero `coeffs`.
pub fn inverse_add(
    dsp: &Dsp,
    tx: TxSize,
    kind: Av1TxType,
    coeffs: &mut [i16],
    eob: usize,
    dst: &mut nc_common::PlaneViewMut<'_>,
) {
    if kind == Av1TxType::DctDct && tx.is_square() {
        inverse_dct_add(dsp, tx, coeffs, eob, dst);
    } else {
        dsp.av1_inverse_add_i16(tx.dsp(), kind, coeffs, eob, dst);
    }
}

/// Inverse 2-D DCT of `coeffs` (dequantised, 16-bit as AV1 §7.12.3
/// clamps them, `coded_size()²`, libaom layout) whose nonzero values all
/// lie in the first `eob` scan positions, added to the prediction already
/// in `dst` with 8-bit clipping; bit-exact with AV1 §7.13.3. The default
/// 16x16 and 32x32 scans first reach coordinate 8 at position 36, and
/// coordinate 16 at 136 (`nc_dsp::av1_dct_nonzero_side`), so the EOB
/// prefixes fit low 8/16 squares (also for 64x64's coded 32x32), whose
/// sparse graphs then run; pinned SVT similarly dispatches low1/8/16/32
/// kernels from its EOB-derived coordinate bounds. Only that square is read.
pub fn inverse_dct_add(
    dsp: &Dsp,
    tx: TxSize,
    coeffs: &[i16],
    eob: usize,
    dst: &mut nc_common::PlaneViewMut<'_>,
) {
    if matches!(tx, TxSize::TX_16X16 | TxSize::TX_32X32 | TxSize::TX_64X64) {
        let limit = nc_dsp::av1_dct_nonzero_side(eob).min(tx.size());
        dsp.av1_inverse_dct_add_sparse(tx.size(), coeffs, limit, dst);
    } else {
        let size = DctSize::from_side(tx.size()).expect("4x4 and 8x8 are DCT sizes");
        dsp.av1_inverse_dct_add(size, coeffs, dst);
    }
}

/// Estimated spatial SSE gain from coding a quantized residual. Forward
/// coefficients have spatial energy gain `2^(6-2*dequant_shift)`; dequantized
/// coefficients are in that same forward domain. Keep the actual prediction
/// SSE at the caller, so uncoded64x64 high-frequency energy is retained.
/// Forward coefficients and dequantized inputs are 16-bit.
pub fn coefficient_gain(dsp: &Dsp, tx: TxSize, original: &[i16], dequantized: &[i16]) -> f64 {
    let gain = dsp.av1_coefficient_gain(original, dequantized);
    gain as f64 / f64::from(1_u32 << (6 - 2 * tx.dequant_shift()))
}

/// Inverse `DCT_DCT` when DC is the only nonzero dequantised coefficient.
/// The two 1-D inverses produce a constant block (AV1 §7.13.3); retain their
/// input clamps and each rounding step, then use the existing SIMD add.
/// Isolated legal-rate probes: AV1 P instructions −2.3%, key −1.5%;
/// byte-identical parity gives zero BD-rate change.
/// The shared constant-add kernel writes only the transform's square.
pub fn inverse_dct_dc_add(
    dsp: &Dsp,
    tx: TxSize,
    dc: i16,
    dst: &mut nc_common::PlaneViewMut<'_>,
    _scratch: &mut [i16],
) {
    let n = tx.size();
    let residual = i32::from(inverse_dc_value(tx, dc));
    if dst.width() == n && dst.height() == n {
        dsp.add_constant(dst, residual);
    } else {
        let mut block = dst
            .sub_view_mut(0, 0, n, n)
            .expect("transform inside destination");
        dsp.add_constant(&mut block, residual);
    }
}

fn inverse_dc_value(tx: TxSize, dc: i16) -> i16 {
    // The square inverses' row shift (§7.13.3, `Transform_Row_Shift`).
    let row_shift = match tx.log2() {
        2 => 0,
        3 => 1,
        _ => 2,
    };
    // Each pass's input is 16-bit (the second clamped). Products then fit
    // i32, and the final residual is within ±1024, safely inside
    // add_residual's range.
    let dc = i32::from(dc);
    let row = round_shift(dc * INVERSE_COS_PI_OVER_FOUR, INVERSE_COS_BITS);
    let row = round_shift(row, row_shift).clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    let column = round_shift(row * INVERSE_COS_PI_OVER_FOUR, INVERSE_COS_BITS);
    round_shift(column, INVERSE_COLUMN_SHIFT) as i16
}

fn round_shift(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        value
    } else {
        (value + (1 << (shift - 1))) >> shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;
    use nc_dsp::{AV1_LOW8_MAX_EOB as LOW8_MAX_EOB, AV1_LOW16_MAX_EOB as LOW16_MAX_EOB};

    /// The backend's zero-residual limits assume coefficients are 8 times
    /// orthonormal once `dqDenom` is applied: a constant residual `r` has
    /// orthonormal DC `r · n`.
    #[test]
    fn forward_dct_is_eight_times_orthonormal_after_dq_denominator() {
        let dsp = Dsp::new();
        for log2 in 2..=6 {
            let tx = TxSize::from_log2(log2).unwrap();
            let n = tx.size();
            let r = 5_i16;
            let residual = vec![r; n * n];
            let mut out = vec![0; tx.coded_size() * tx.coded_size()];
            forward_dct(&dsp, tx, &residual, n, &mut out);
            let scaled = i32::from(out[0]) << tx.dequant_shift();
            let expected = 8 * i32::from(r) * n as i32;
            assert!(
                (scaled - expected).abs() <= expected / 64,
                "{n}: {scaled} vs {expected}"
            );
        }
    }

    #[test]
    fn coefficient_gain_uses_spatial_scale_and_keeps_uncoded_energy() {
        let dsp = Dsp::new();
        for log2 in 2..=6 {
            let tx = TxSize::from_log2(log2).unwrap();
            let n = tx.size();
            for value in [-64_i16, -7, -1, 0, 1, 7, 64] {
                let residual = vec![value; n * n];
                let mut coefficients = vec![0; tx.coded_size() * tx.coded_size()];
                forward_dct(&dsp, tx, &residual, n, &mut coefficients);
                let energy = (n * n) as f64 * f64::from(value).powi(2);
                let ideal_dc = ((8 * n) >> tx.dequant_shift()) as i32 * i32::from(value);
                let ideal = i16::try_from(ideal_dc).unwrap();
                assert_eq!(coefficient_gain(&dsp, tx, &[ideal], &[ideal]), energy);
                assert_eq!(
                    coefficient_gain(&dsp, tx, &[ideal], &[ideal / 2]),
                    energy * 0.75
                );
                let gain = coefficient_gain(&dsp, tx, &coefficients, &coefficients);
                // Integer FDCT rounding matters at unit residuals, but this
                // tolerance is far smaller than a 16x normalization mistake.
                assert!(
                    (gain - energy).abs() <= (n * n) as f64 * (0.5 * f64::from(value).abs() + 0.25),
                    "{tx:?} value{value}: gain{gain}, SSE{energy}"
                );
                assert_eq!(
                    coefficient_gain(&dsp, tx, &coefficients, &vec![0; coefficients.len()]),
                    0.0
                );
            }
        }
        let tx = TxSize::TX_64X64;
        let residual: Vec<i16> = (0..4096)
            .map(|i| if (i / 64 + i % 64) % 2 == 0 { 20 } else { -20 })
            .collect();
        let mut coefficients = vec![0; 1024];
        forward_dct(&dsp, tx, &residual, 64, &mut coefficients);
        let prediction_sse = 4096.0 * 400.0;
        let gain = coefficient_gain(&dsp, tx, &coefficients, &coefficients);
        assert!(
            gain < prediction_sse * 0.1,
            "uncoded three-quadrant energy must remain in actual prediction SSE"
        );
        let original = vec![i16::MAX; 1024];
        let dequant = vec![-32768; 1024];
        let expected = 1024.0 * (2.0 * f64::from(i16::MAX) * -32768.0 - 32768.0 * 32768.0) / 4.0;
        assert_eq!(coefficient_gain(&dsp, tx, &original, &dequant), expected);
    }

    /// The 16x16 low-8 bound the sparse inverse relies on: the scan's first
    /// `LOW8_MAX_EOB` positions lie in the low 8x8 and the next does not.
    #[test]
    fn scan_16x16_prefix_fits_the_low_8x8() {
        const SIDE: usize = 16;
        let scan = &crate::tables::DEFAULT_SCAN_16X16;
        let inside = |p: u16| usize::from(p) / SIDE < 8 && usize::from(p) % SIDE < 8;
        assert!(scan[..LOW8_MAX_EOB].iter().all(|&p| inside(p)));
        assert!(!inside(scan[LOW8_MAX_EOB]));
    }

    #[test]
    fn sparse_inverse_matches_full_scalar_at_scan_boundaries_and_extremes() {
        use nc_common::PlaneViewMut;
        use nc_dsp::Isa;
        let scalar = Dsp::with_isa(Isa::Scalar).unwrap();
        let extremes: [i16; 7] = [i16::MIN, -32_767, -1, 0, 1, 32_766, i16::MAX];
        for tx in [TxSize::TX_32X32, TxSize::TX_64X64] {
            let n = tx.size();
            let stride = n + 7;
            let prediction: Vec<u8> = (0..stride * n + 16).map(|i| (i * 37 + 53) as u8).collect();
            let scan = &crate::tables::DEFAULT_SCAN_32X32;
            for (eob, limit) in [(LOW8_MAX_EOB, 8), (LOW16_MAX_EOB, 16)] {
                let first_outside = usize::from(scan[eob]);
                assert!(first_outside / 32 >= limit || first_outside % 32 >= limit);
            }
            for eob in [
                0,
                1,
                2,
                7,
                8,
                9,
                LOW8_MAX_EOB - 1,
                LOW8_MAX_EOB,
                LOW8_MAX_EOB + 1,
                63,
                64,
                65,
                127,
                128,
                LOW16_MAX_EOB - 1,
                LOW16_MAX_EOB,
                LOW16_MAX_EOB + 1,
                255,
                511,
                1024,
            ] {
                let limit = if eob <= LOW8_MAX_EOB {
                    8
                } else if eob <= LOW16_MAX_EOB {
                    16
                } else {
                    32
                };
                assert!(
                    scan[..eob]
                        .iter()
                        .all(|&p| usize::from(p) / 32 < limit && usize::from(p) % 32 < limit)
                );
                for mode in 0..4 {
                    let mut coefficients = vec![0; 1024];
                    let mut state = 19_u32;
                    for (i, &position) in scan[..eob].iter().enumerate() {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let value = match mode {
                            0 => (state >> 24) as i16 - 127,
                            1 => extremes[i % extremes.len()],
                            2 => {
                                if i + 1 == eob {
                                    extremes[eob % extremes.len()]
                                } else {
                                    0
                                }
                            }
                            _ => (state >> 16) as i16,
                        };
                        coefficients[usize::from(position)] = value;
                    }
                    // The reference reads every coefficient (a full EOB).
                    let mut expected = prediction.clone();
                    inverse_dct_add(
                        &scalar,
                        tx,
                        &coefficients,
                        coefficients.len(),
                        &mut PlaneViewMut::new(&mut expected, stride, n, n).unwrap(),
                    );
                    for isa in Isa::ALL {
                        let Some(dsp) = Dsp::with_isa(isa) else {
                            continue;
                        };
                        let mut actual = prediction.clone();
                        inverse_dct_add(
                            &dsp,
                            tx,
                            &coefficients,
                            eob,
                            &mut PlaneViewMut::new(&mut actual, stride, n, n).unwrap(),
                        );
                        assert_eq!(actual, expected, "{tx:?} {isa:?} eob{eob}, mode{mode}");
                    }
                }
            }
        }
    }

    #[test]
    fn dc_shortcut_matches_full_inverse_across_sizes_and_supported_isas() {
        use nc_common::PlaneViewMut;
        use nc_dsp::Isa;

        let scalar = Dsp::with_isa(Isa::Scalar).unwrap();
        for log2 in 2..=6 {
            let tx = TxSize::from_log2(log2).unwrap();
            let n = tx.size();
            let stride = n + 7;
            let prediction: Vec<u8> = (0..stride * n + 16).map(|i| (i * 37 + 53) as u8).collect();
            for dc in [
                i16::MIN,
                -32_767,
                -1024,
                -65,
                -33,
                -17,
                -1,
                0,
                1,
                17,
                33,
                65,
                1024,
                32_766,
                i16::MAX,
            ] {
                let mut coefficients = vec![0; tx.coded_size() * tx.coded_size()];
                coefficients[0] = dc;
                let mut expected = prediction.clone();
                let eob = coefficients.len();
                inverse_dct_add(
                    &scalar,
                    tx,
                    &coefficients,
                    eob,
                    &mut PlaneViewMut::new(&mut expected, stride, n, n).unwrap(),
                );
                for isa in Isa::ALL {
                    let Some(dsp) = Dsp::with_isa(isa) else {
                        continue;
                    };
                    let mut actual = prediction.clone();
                    let mut scratch = vec![0; n * n];
                    inverse_dct_dc_add(
                        &dsp,
                        tx,
                        dc,
                        &mut PlaneViewMut::new(&mut actual, stride, n, n).unwrap(),
                        &mut scratch,
                    );
                    assert_eq!(actual, expected, "{tx:?} {isa:?} dc={dc}");
                }
            }
        }
    }

    /// A 64x64 transform keeps only its low frequencies: a smooth residual
    /// survives the round trip closely.
    /// A low-frequency forward leaves every coded position from its
    /// `coded_prefix` on zero, and the prefix is the whole block otherwise.
    #[test]
    fn low_forward_is_zero_past_its_coded_prefix() {
        let dsp = Dsp::new();
        let policy = ForwardPolicy {
            large: true,
            medium: true,
        };
        for log2 in 4..=6 {
            let tx = TxSize::from_log2(log2).unwrap();
            let n = tx.size();
            let residual: Vec<i16> = (0..n * n).map(|i| ((i * 37 % 61) as i16) - 30).collect();
            let mut out = vec![0; tx.coded_size() * tx.coded_size()];
            forward_dct_with(&dsp, policy, tx, &residual, n, &mut out);
            let prefix = policy.coded_prefix(tx);
            assert!(out[prefix..].iter().all(|&c| c == 0), "{n}");
            assert!(out[..prefix].iter().any(|&c| c != 0), "{n}");
            let side = tx.coded_size();
            assert_eq!(ForwardPolicy::default().coded_prefix(tx), side * side);
        }
    }

    #[test]
    fn transform_64_round_trip_of_smooth_residual() {
        let residual: Vec<i16> = (0..64 * 64)
            .map(|i| ((i / 64) as i16 - 32) / 2 + ((i % 64) as i16 - 32) / 4)
            .collect();
        let mut coeffs = vec![0i16; 32 * 32];
        forward_dct(&Dsp::new(), TxSize::TX_64X64, &residual, 64, &mut coeffs);
        let mut plane = Plane::new(64, 64, 0).unwrap();
        plane.view_mut().fill(128);
        let eob = coeffs.len();
        inverse_dct_add(
            &Dsp::new(),
            TxSize::TX_64X64,
            &coeffs,
            eob,
            &mut plane.view_mut(),
        );
        let view = plane.view();
        let worst = (0..64 * 64)
            .map(|i| (i32::from(view.row(i / 64)[i % 64]) - 128 - i32::from(residual[i])).abs())
            .max()
            .unwrap();
        assert!(worst <= 2, "worst error {worst}");
    }

    /// The dsp kernels and the 64-point path agree with libaom's scale: a
    /// DC-only 8x8 coefficient produces a flat block.
    #[test]
    fn dc_coefficient_is_flat() {
        let mut coeffs = vec![0i16; 64];
        coeffs[0] = 8 * 8 * 10; // orthonormal DC of a flat +10 block, ×8
        let mut plane = Plane::new(8, 8, 0).unwrap();
        plane.view_mut().fill(100);
        inverse_dct_add(
            &Dsp::new(),
            TxSize::TX_8X8,
            &coeffs,
            1,
            &mut plane.view_mut(),
        );
        assert!(plane.view().rows().all(|row| row.iter().all(|&v| v == 110)));
    }
}
