//! AV1 `DCT_DCT` transforms for 4x4 to 64x64.
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

use nc_dsp::{Av1Quantizer, BlockSize, DctSize, Dsp};

/// AV1's 12-bit `cospi[32]`, used once in each DC-only inverse pass.
const INVERSE_COS_PI_OVER_FOUR: i32 = 2896;
const INVERSE_COS_BITS: u32 = 12;
const INVERSE_COLUMN_SHIFT: u32 = 4;

/// Square transform sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxSize {
    /// 4x4.
    Tx4x4,
    /// 8x8.
    Tx8x8,
    /// 16x16.
    Tx16x16,
    /// 32x32.
    Tx32x32,
    /// 64x64 (only the top-left 32x32 coefficients are coded).
    Tx64x64,
}

impl TxSize {
    /// The transform for an `n`×`n` block, `n = 1 << log2`.
    #[must_use]
    pub fn from_log2(log2: u8) -> Option<Self> {
        match log2 {
            2 => Some(Self::Tx4x4),
            3 => Some(Self::Tx8x8),
            4 => Some(Self::Tx16x16),
            5 => Some(Self::Tx32x32),
            6 => Some(Self::Tx64x64),
            _ => None,
        }
    }

    /// log2 of the side length.
    #[must_use]
    pub fn log2(self) -> u8 {
        match self {
            Self::Tx4x4 => 2,
            Self::Tx8x8 => 3,
            Self::Tx16x16 => 4,
            Self::Tx32x32 => 5,
            Self::Tx64x64 => 6,
        }
    }

    /// Side length in samples.
    #[must_use]
    pub fn size(self) -> usize {
        1 << self.log2()
    }

    /// Index of the square size in AV1's `TX_SIZES` order (`TX_4X4` = 0):
    /// also the transform-size context of the coefficient CDFs.
    #[must_use]
    pub fn index(self) -> usize {
        usize::from(self.log2() - 2)
    }

    /// Side length of the coded coefficient array: 64x64 transforms code
    /// only their top-left 32x32 (libaom `av1_get_adjusted_tx_size`).
    #[must_use]
    pub fn coded_size(self) -> usize {
        self.size().min(32)
    }

    /// log2 of [`TxSize::coded_size`].
    #[must_use]
    pub fn coded_log2(self) -> u8 {
        self.log2().min(5)
    }

    /// Dequantisation shift (AV1 `dqDenom` = 1, 2, 4; libaom
    /// `av1_get_tx_scale`): transforms of more than 256 coefficients are
    /// scaled down by the forward transform and back up here.
    #[must_use]
    pub fn dequant_shift(self) -> u32 {
        let pels = self.size() * self.size();
        u32::from(pels > 256) + u32::from(pels > 1024)
    }
}

/// Forward 2-D DCT of the `n`×`n` residual (row-major at `stride`) into
/// `out` in libaom's column-major layout. For 64x64 only the coded
/// top-left 32x32 coefficients are returned (`out` holds `coded_size()²`).
pub fn forward_dct(dsp: &Dsp, tx: TxSize, residual: &[i16], stride: usize, out: &mut [i32]) {
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
    /// Whether `tx` uses the low-frequency forward.
    #[must_use]
    pub fn low(self, tx: TxSize) -> bool {
        match tx.size() {
            32 | 64 => self.large,
            16 => self.medium,
            _ => false,
        }
    }

    /// Scan positions of `tx` that can hold a nonzero level under this
    /// policy: the whole coded block, or with the low-frequency forward the
    /// default-scan prefix that ends at the last position of the kept square
    /// (every later position transforms, and so quantises, to zero). End of
    /// block searches stop there instead of walking the dropped frequencies.
    #[must_use]
    pub fn scan_len(self, tx: TxSize) -> usize {
        if !self.low(tx) {
            return tx.coded_size() * tx.coded_size();
        }
        match tx.size() {
            16 => LOW_SCAN_LEN_16X16,
            _ => LOW_SCAN_LEN_32X32,
        }
    }

    /// Leading coefficients of `tx`'s coded block (libaom layout, one line
    /// of `coded_size()` per horizontal frequency) that can be nonzero under
    /// this policy: all of them, or with the low-frequency forward the first
    /// [`nc_dsp::av1_low_keep`] lines (the later ones transform to zero).
    #[must_use]
    pub fn coded_prefix(self, tx: TxSize) -> usize {
        let coded = tx.coded_size();
        if self.low(tx) {
            nc_dsp::av1_low_keep(tx.size()) * coded
        } else {
            coded * coded
        }
    }
}

/// Quantise `tx`'s forward coefficients in place (`levels`, the whole coded
/// block) where `policy` lets them be nonzero; the rest are zero already and
/// stay zero. Returns the number of nonzero levels. Exact: a low-frequency
/// 32x32 forward quantised only its first 16 of 32 lines halves the work
/// (2026-11-11, x86 av1-faster).
pub fn quantize_coded(
    dsp: &Dsp,
    policy: ForwardPolicy,
    tx: TxSize,
    levels: &mut [i32],
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
    coeffs: &[i32],
    quantizer: &Av1Quantizer,
    levels: &mut [i32],
) -> u32 {
    let prefix = policy.coded_prefix(tx);
    levels[prefix..coeffs.len()].fill(0);
    dsp.av1_quantize_from(&coeffs[..prefix], quantizer, &mut levels[..prefix])
}

/// Leading dequantised coefficients [`inverse_dct_add_eob`] reads for a
/// block with end of block `eob` (at least 1): the lines of the EOB-bounded
/// square for 32x32 and 64x64, else the whole coded block. Every later
/// level is zero, so it dequantises to zero and adds nothing to
/// [`coefficient_gain`] either.
#[must_use]
pub fn dequantized_extent(tx: TxSize, eob: usize) -> usize {
    let coded = tx.coded_size();
    if matches!(tx, TxSize::Tx32x32 | TxSize::Tx64x64) {
        nc_dsp::av1_dct_nonzero_side(eob) * coded
    } else {
        coded * coded
    }
}

/// [`ForwardPolicy::scan_len`] of a low-frequency 16x16 forward.
const LOW_SCAN_LEN_16X16: usize = kept_scan_len(
    &crate::tables::DEFAULT_SCAN_16X16,
    4,
    nc_dsp::av1_low_keep(16),
);
/// [`ForwardPolicy::scan_len`] of a low-frequency 32x32 or 64x64 forward
/// (64x64 codes its top-left 32x32 with the 32x32 scan).
const LOW_SCAN_LEN_32X32: usize = kept_scan_len(
    &crate::tables::DEFAULT_SCAN_32X32,
    5,
    nc_dsp::av1_low_keep(32),
);

/// One past the last position of `scan` (over a `1 << log2` square, libaom
/// layout) whose row and column are both below `keep`.
const fn kept_scan_len(scan: &[u16], log2: u32, keep: usize) -> usize {
    let mask = (1 << log2) - 1;
    let mut len = 0;
    let mut c = 0;
    while c < scan.len() {
        let pos = scan[c] as usize;
        if pos & mask < keep && pos >> log2 < keep {
            len = c + 1;
        }
        c += 1;
    }
    len
}

/// [`forward_dct`], or the low-frequency forward where `policy` says so.
pub fn forward_dct_with(
    dsp: &Dsp,
    policy: ForwardPolicy,
    tx: TxSize,
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    if policy.low(tx) {
        dsp.av1_forward_dct_low(tx.size(), residual, stride, out);
    } else {
        forward_dct(dsp, tx, residual, stride, out);
    }
}

/// Inverse 2-D DCT of `coeffs` (dequantised, `coded_size()²`, libaom
/// layout), added to the prediction already in `dst` (rows at `stride`)
/// with 8-bit clipping. Bit-exact with AV1 §7.13.3.
pub fn inverse_dct_add(
    dsp: &Dsp,
    tx: TxSize,
    coeffs: &[i32],
    dst: &mut nc_common::PlaneViewMut<'_>,
) {
    match DctSize::from_side(tx.size()) {
        Some(size) => dsp.av1_inverse_dct_add(size, coeffs, dst),
        None => dsp.av1_inverse_dct_add_64(coeffs, dst),
    }
}

/// EOB-bounded inverse: the default 32x32 scan first reaches coordinate 8
/// at position 36, and coordinate 16 at 136 (`nc_dsp::av1_dct_nonzero_side`).
/// Thus the preceding EOB prefixes fit low8/16 squares (also for 64x64's
/// coded 32x32). Pinned SVT similarly dispatches low1/8/16/32 kernels using
/// its EOB-derived coordinate bounds.
pub fn inverse_dct_add_eob(
    dsp: &Dsp,
    tx: TxSize,
    coeffs: &[i32],
    eob: usize,
    dst: &mut nc_common::PlaneViewMut<'_>,
) {
    if matches!(tx, TxSize::Tx32x32 | TxSize::Tx64x64) {
        let limit = nc_dsp::av1_dct_nonzero_side(eob);
        dsp.av1_inverse_dct_add_sparse(tx.size(), coeffs, limit, dst);
    } else {
        inverse_dct_add(dsp, tx, coeffs, dst);
    }
}

/// Estimated spatial SSE gain from coding a quantized residual. Forward
/// coefficients have spatial energy gain `2^(6-2*dequant_shift)`; dequantized
/// coefficients are in that same forward domain. Keep the actual prediction
/// SSE at the caller, so uncoded64x64 high-frequency energy is retained.
/// Valid forward coefficients fit22 bits; dequantized inputs are clamped16-bit.
pub fn coefficient_gain(dsp: &Dsp, tx: TxSize, original: &[i32], dequantized: &[i32]) -> f64 {
    let gain = dsp.av1_coefficient_gain(original, dequantized);
    gain as f64 / f64::from(1_u32 << (6 - 2 * tx.dequant_shift()))
}

/// Inverse `DCT_DCT` when DC is the only nonzero dequantised coefficient.
/// The two 1-D inverses produce a constant block (AV1 §7.13.3); retain their
/// input clamps and each rounding step, then use the existing SIMD add.
/// Isolated legal-rate probes: AV1 P instructions −2.3%, key −1.5%;
/// byte-identical parity gives zero BD-rate change.
/// `scratch` must hold `tx.size()²` residual samples.
pub fn inverse_dct_dc_add(
    dsp: &Dsp,
    tx: TxSize,
    dc: i32,
    dst: &mut nc_common::PlaneViewMut<'_>,
    scratch: &mut [i16],
) {
    let n = tx.size();
    let residual = &mut scratch[..n * n];
    residual.fill(inverse_dc_value(tx, dc));
    dsp.add_residual(
        BlockSize::new(n, n).expect("transform size is a block size"),
        dst,
        residual,
        n,
    );
}

fn inverse_dc_value(tx: TxSize, dc: i32) -> i16 {
    let row_shift = match tx {
        TxSize::Tx4x4 => 0,
        TxSize::Tx8x8 => 1,
        TxSize::Tx16x16 | TxSize::Tx32x32 | TxSize::Tx64x64 => 2,
    };
    // Each pass clamps its input to 16 bits. Products then fit i32, and
    // the final residual is within ±1024, safely inside add_residual's range.
    let dc = dc.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
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
            let scaled = out[0] << tx.dequant_shift();
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
                assert_eq!(coefficient_gain(&dsp, tx, &[ideal_dc], &[ideal_dc]), energy);
                assert_eq!(
                    coefficient_gain(&dsp, tx, &[ideal_dc], &[ideal_dc / 2]),
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
        let tx = TxSize::Tx64x64;
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
        let original = vec![(1 << 22) - 1; 1024];
        let dequant = vec![-32768; 1024];
        let expected =
            1024.0 * (2.0 * f64::from((1 << 22) - 1) * -32768.0 - 32768.0 * 32768.0) / 4.0;
        assert_eq!(coefficient_gain(&dsp, tx, &original, &dequant), expected);
    }

    #[test]
    fn sparse_inverse_matches_full_scalar_at_scan_boundaries_and_extremes() {
        use nc_common::PlaneViewMut;
        use nc_dsp::Isa;
        let scalar = Dsp::with_isa(Isa::Scalar).unwrap();
        let extremes = [
            -8_388_608, -32_769, -32_768, -32_767, -1, 0, 1, 32_767, 32_768, 8_388_607,
        ];
        for tx in [TxSize::Tx32x32, TxSize::Tx64x64] {
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
                            0 => (state >> 24) as i32 - 127,
                            1 => extremes[i % extremes.len()],
                            2 => {
                                if i + 1 == eob {
                                    extremes[eob % extremes.len()]
                                } else {
                                    0
                                }
                            }
                            _ => (state >> 8) as i32 - 8_388_608,
                        };
                        coefficients[usize::from(position)] = value;
                    }
                    let mut expected = prediction.clone();
                    inverse_dct_add(
                        &scalar,
                        tx,
                        &coefficients,
                        &mut PlaneViewMut::new(&mut expected, stride, n, n).unwrap(),
                    );
                    for isa in Isa::ALL {
                        let Some(dsp) = Dsp::with_isa(isa) else {
                            continue;
                        };
                        let mut actual = prediction.clone();
                        inverse_dct_add_eob(
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
                -8_388_608, -32_769, -32_768, -1024, -65, -33, -17, -1, 0, 1, 17, 33, 65, 1024,
                32_767, 32_768, 8_388_607,
            ] {
                let mut coefficients = vec![0; tx.coded_size() * tx.coded_size()];
                coefficients[0] = dc;
                let mut expected = prediction.clone();
                inverse_dct_add(
                    &scalar,
                    tx,
                    &coefficients,
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
    /// A low-frequency forward leaves every scan position from
    /// `scan_len` on zero, and `scan_len` is no longer than it must be.
    #[test]
    fn low_forward_is_zero_past_its_scan_len() {
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
            let scan = crate::coeff::coefficient_scan(tx);
            let len = policy.scan_len(tx);
            assert!(scan[len..].iter().all(|&p| out[usize::from(p)] == 0), "{n}");
            let keep = nc_dsp::av1_low_keep(n);
            let last = usize::from(scan[len - 1]);
            let side = tx.coded_size();
            assert!(last % side < keep && last / side < keep, "{n}");
            assert_eq!(ForwardPolicy::default().scan_len(tx), side * side);
        }
    }

    #[test]
    fn transform_64_round_trip_of_smooth_residual() {
        let residual: Vec<i16> = (0..64 * 64)
            .map(|i| ((i / 64) as i16 - 32) / 2 + ((i % 64) as i16 - 32) / 4)
            .collect();
        let mut coeffs = vec![0i32; 32 * 32];
        forward_dct(&Dsp::new(), TxSize::Tx64x64, &residual, 64, &mut coeffs);
        let mut plane = Plane::new(64, 64, 0).unwrap();
        plane.view_mut().fill(128);
        inverse_dct_add(&Dsp::new(), TxSize::Tx64x64, &coeffs, &mut plane.view_mut());
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
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = 8 * 8 * 10; // orthonormal DC of a flat +10 block, ×8
        let mut plane = Plane::new(8, 8, 0).unwrap();
        plane.view_mut().fill(100);
        inverse_dct_add(&Dsp::new(), TxSize::Tx8x8, &coeffs, &mut plane.view_mut());
        assert!(plane.view().rows().all(|row| row.iter().all(|&v| v == 110)));
    }
}
