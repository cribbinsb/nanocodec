//! VP9 transform sizes and types, forward transforms, quantization,
//! dequantization and the normative inverse (VP9 §8.7; libvpx
//! `vp9_idct.c`, `vp9_detokenize.c`; BSD-3-Clause, see `THIRD_PARTY.md`).
//!
//! Coefficients are at VP9's scale: 8x the orthonormal DCT up to 16x16 and
//! 4x for 32x32, whose dequantizer halves (`dq_shift`). Lossless frames use
//! the reversible 4x4 Walsh-Hadamard transform, whose forward output is
//! already in quantized units.

use crate::hybrid;
use crate::tables::{AC_QUANT, DC_QUANT};

pub(crate) use hybrid::Transform as TxType;

/// Square transform sizes (`TX_SIZE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TxSize {
    Tx4 = 0,
    Tx8 = 1,
    Tx16 = 2,
    Tx32 = 3,
}

impl TxSize {
    /// log2 of the side in samples.
    pub(crate) const fn log2(self) -> u8 {
        self as u8 + 2
    }

    /// Side in samples.
    pub(crate) const fn side(self) -> usize {
        4 << self as usize
    }

    /// Samples (and coefficients) in one transform block.
    pub(crate) const fn area(self) -> usize {
        self.side() * self.side()
    }

    /// Side in 4x4 units (entropy-context entries it covers).
    pub(crate) const fn units(self) -> usize {
        1 << self as usize
    }

    pub(crate) fn from_log2(log2: u8) -> Self {
        [Self::Tx4, Self::Tx8, Self::Tx16, Self::Tx32][usize::from(log2 - 2)]
    }

    /// The dequantizer's right shift (`dq_shift`): 32x32 halves.
    pub(crate) const fn dequant_shift(self) -> u32 {
        (self as u32 == Self::Tx32 as u32) as u32
    }
}

/// Quantizer steps of one plane type at one qindex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Steps {
    pub dc: i32,
    pub ac: i32,
}

impl Steps {
    /// Steps at `qindex` with the frame's DC/AC deltas (VP9 §8.6.1:
    /// `dc_q(qindex + delta)`, clamped to the table).
    pub(crate) fn new(qindex: u8, dc_delta: i8, ac_delta: i8) -> Self {
        let at = |delta: i8| (i32::from(qindex) + i32::from(delta)).clamp(0, 255) as usize;
        Self {
            dc: DC_QUANT[at(dc_delta)],
            ac: AC_QUANT[at(ac_delta)],
        }
    }

    pub(crate) fn step(self, index: usize) -> i32 {
        if index == 0 { self.dc } else { self.ac }
    }
}

/// Rounding offset of the deadzone quantizer, as a fraction of the step.
/// Nearest-integer rounding (1/2) keeps many expensive +-1 levels; these
/// are the classic HM intra/inter offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rounding {
    /// Intra blocks: offset 1/3 of the step.
    Intra,
    /// Inter residuals: offset 1/6 of the step.
    Inter,
    /// Nearest integer: offset 1/2 of the step.
    Nearest,
    /// Offset 1/n of the step (a tuned inter rounding).
    Fraction(u8),
}

impl Rounding {
    pub(crate) const fn divisor(self) -> i32 {
        match self {
            Self::Intra => 3,
            Self::Inter => 6,
            Self::Nearest => 2,
            Self::Fraction(n) => n as i32,
        }
    }
}

/// The column and row passes of an 8x8 or larger transform (32x32 is DCT
/// only: `get_tx_type` never gives it an ADST).
pub(crate) fn passes(ty: TxType) -> (nc_dsp::Vp9Pass, nc_dsp::Vp9Pass) {
    let pass = |adst: bool| {
        if adst {
            nc_dsp::Vp9Pass::Adst
        } else {
            nc_dsp::Vp9Pass::Dct
        }
    };
    (pass(ty.columns_adst()), pass(ty.rows_adst()))
}

/// Forward transform of a raster residual (`tx.area()` samples). The
/// 8..32-point DCT runs on the shared AV1 kernels (SIMD where available):
/// libaom's forward DCT has VP9's coefficient scale at every size (8x the
/// orthonormal transform, 4x at 32x32), only its rounding differs, which
/// any encoder may choose. It returns libaom's column-major layout, so the
/// result is transposed.
pub(crate) fn forward(
    dsp: &nc_dsp::Dsp,
    tx: TxSize,
    ty: TxType,
    lossless: bool,
    residual: &[i16],
    out: &mut [i32],
) {
    let area = tx.area();
    if lossless || tx == TxSize::Tx4 {
        let input: [i32; 16] = std::array::from_fn(|i| i32::from(residual[i]));
        let coefficients = if lossless {
            nc_dsp::reversible_wht4::forward(&input)
        } else {
            hybrid::forward(&input, ty)
        };
        out[..16].copy_from_slice(&coefficients);
        return;
    }
    if ty == TxType::DctDct {
        let side = tx.side();
        let size = [
            nc_dsp::DctSize::N8,
            nc_dsp::DctSize::N16,
            nc_dsp::DctSize::N32,
        ][tx as usize - 1];
        dsp.av1_forward_dct_raster(size, &residual[..area], side, &mut out[..area]);
        return;
    }
    let (columns, rows) = passes(ty);
    nc_dsp::vp9_forward_transform(
        tx.log2(),
        columns,
        rows,
        &residual[..area],
        &mut out[..area],
    );
}

/// The largest coefficient magnitude that quantizes below `min_level` (so
/// is zero after the planned step's dead zone) with the smaller of the DC
/// and AC steps: a level reaches `min_level` only from `min_level` steps
/// less the rounding offset, compared after the 32x32 doubling shift.
pub(crate) fn zero_coefficient_limit(
    tx: TxSize,
    steps: Steps,
    (rounding, min_level): (Rounding, i32),
) -> u32 {
    let step = steps.dc.min(steps.ac);
    let threshold = min_level * step - step / rounding.divisor();
    (threshold.max(1) - 1) as u32 >> tx.dequant_shift()
}

/// The levels of a transform block that codes nothing, for callers whose
/// level buffer was left unspecified ([`quantize`]).
pub(crate) static ZERO_LEVELS: [i32; TxSize::Tx32.area()] = [0; TxSize::Tx32.area()];

/// Margin for the forward transform's integer rounding in
/// [`dct_levels_are_zero`] (each stage rounds; a few units at most).
const ROUNDING_MARGIN: u32 = 16;

/// Whether a DCT block with residual absolute sum `sad` certainly codes no
/// level of at least `min_level` (1, or the planned step's dead zone,
/// [`planned_min_level`]), so the transform can be skipped. Every 2-D DCT
/// basis function is at most 2/N in magnitude, and coefficients are 8x the
/// orthonormal transform (32x32: 4x, but quantized at twice the value), so
/// no coefficient exceeds 16 * sad / N (plus rounding); a level reaches
/// `min_level` only from `min_level` steps less the rounding offset.
///
/// `scale` above 1 (`vp9.pre_transform_zero_scale`, inter blocks only)
/// widens the bound into a speed heuristic: residual up to `scale` times
/// the proof's limit is dropped without a transform.
pub(crate) fn dct_levels_are_zero(
    tx: TxSize,
    sad: u32,
    steps: Steps,
    (rounding, min_level): (Rounding, i32),
    scale: f32,
) -> bool {
    let step = steps.dc.min(steps.ac);
    let threshold = (min_level * step - step / rounding.divisor()) as u32;
    let bound = 16 * sad / tx.side() as u32 + ROUNDING_MARGIN;
    if scale <= 1.0 {
        bound < threshold
    } else {
        (bound as f32) < scale * threshold as f32
    }
}

/// Smallest level magnitude a frame planned past qindex 255 keeps
/// (`FramePlan::lambda_scale`: its planned step is √scale times the coded
/// one): a level survives only if its coefficient reaches `(1 − 1/divisor)
/// · √scale` coded steps. 1 (every level) at or below the planned step.
pub(crate) fn planned_min_level(rounding: Rounding, lambda_scale: f32) -> i32 {
    if lambda_scale <= 1.0 {
        return 1;
    }
    let keep = 1.0 - 1.0 / rounding.divisor() as f32;
    ((keep * lambda_scale.sqrt()).ceil() as i32).max(1)
}

/// Zero the levels below `min_level` ([`planned_min_level`]), the ones a
/// quantiser at the planned step past qindex 255 would zero. Returns
/// whether any level is left.
pub(crate) fn apply_planned_dead_zone(tx: TxSize, min_level: i32, levels: &mut [i32]) -> bool {
    let mut any = false;
    for level in &mut levels[..tx.area()] {
        if level.abs() < min_level {
            *level = 0;
        }
        any |= *level != 0;
    }
    any
}

/// Quantize raster `coefficients` into `levels`; returns whether any level
/// of at least `min_level` ([`planned_min_level`]; 1 counts every nonzero
/// level) may be left; when it returns false `levels` is unspecified (the
/// caller codes [`ZERO_LEVELS`]): zero-filling every skipped block cost a
/// memset per block. Lossless coefficients are copied (already quantized).
pub(crate) fn quantize(
    dsp: &nc_dsp::Dsp,
    tx: TxSize,
    lossless: bool,
    coefficients: &[i32],
    steps: Steps,
    (rounding, min_level): (Rounding, i32),
    levels: &mut [i32],
) -> bool {
    let area = tx.area();
    if lossless {
        levels[..area].copy_from_slice(&coefficients[..area]);
        return levels[..area].iter().any(|&l| l != 0);
    }
    let shift = tx.dequant_shift();
    // Exact shortcut: below the smallest magnitude that rounds to
    // `min_level` with the smaller step, every level is zero or below the
    // dead zone that zeroes it next. On the Jetson this scan
    // is cheaper than the vector quantizer for the many all-zero blocks
    // (without it: +2.3% encode CPU, crowd 720p q230), although x86
    // callgrind counts it, with the zero fill, as a loss.
    let largest = dsp.vp9_largest_magnitude(&coefficients[..area]);
    if largest <= zero_coefficient_limit(tx, steps, (rounding, min_level)) {
        return false;
    }
    let dc = nc_dsp::Vp9QuantStep::new(steps.dc, rounding.divisor(), shift);
    let ac = nc_dsp::Vp9QuantStep::new(steps.ac, rounding.divisor(), shift);
    dsp.vp9_quantize(&coefficients[..area], &dc, &ac, &mut levels[..area])
}

/// Dequantize one coefficient level as the reference decoder does
/// (`(|level| * step) >> dq_shift`, sign applied after).
pub(crate) fn dequantize_level(tx: TxSize, index: usize, level: i32, steps: Steps) -> i32 {
    let magnitude = (level.unsigned_abs() as i32 * steps.step(index)) >> tx.dequant_shift();
    if level < 0 { -magnitude } else { magnitude }
}

/// Squared error of quantizing `coefficients` to `levels`, in residual
/// sample units: coefficients are 8x orthonormal (32x32: 4x and quantized
/// at twice their value), so the error divides by the squared scale.
pub(crate) fn coefficient_sse(
    tx: TxSize,
    coefficients: &[i32],
    levels: &[i32],
    steps: Steps,
) -> u64 {
    let area = tx.area();
    let error: u64 = coefficients[..area]
        .iter()
        .zip(&levels[..area])
        .enumerate()
        .map(|(index, (&c, &level))| {
            let d = i64::from(dequantize_level(tx, index, level, steps));
            (i64::from(c) - d).unsigned_abs().pow(2)
        })
        .sum();
    let scale_log2 = if tx == TxSize::Tx32 { 2 } else { 3 };
    error >> (2 * scale_log2)
}

/// `cospi_16_64` and the DCT's 14-bit fixed point.
const COSPI_16_64: i64 = 11_585;
const DCT_ROUND_BITS: u32 = 14;

/// The constant residual of a DCT block whose only coefficient is DC
/// (libvpx `vpx_idctNxN_1_add`): every stage of both passes carries the
/// DC product unchanged, so the full transform gives this value
/// everywhere. Exact; most coded blocks at low rates are DC-only.
pub(crate) fn dc_only_residual(tx: TxSize, dc: i32) -> i32 {
    let round = |value: i64| ((value + (1 << (DCT_ROUND_BITS - 1))) >> DCT_ROUND_BITS) as i32;
    let wrap = |value: i32| i32::from(value as i16);
    let rows = wrap(round(i64::from(dc as i16) * COSPI_16_64));
    let columns = wrap(round(i64::from(rows) * COSPI_16_64));
    let shift = inverse_output_shift(tx);
    (columns + (1 << (shift - 1))) >> shift
}

/// Final rounding shift of the 8x8 (5) and larger (6) inverse transforms.
fn inverse_output_shift(tx: TxSize) -> u32 {
    if tx == TxSize::Tx8 { 5 } else { 6 }
}

/// Normative inverse transform of dequantized raster coefficients into a
/// residual, before adding the prediction and clipping.
pub(crate) fn inverse(
    dsp: &nc_dsp::Dsp,
    tx: TxSize,
    ty: TxType,
    lossless: bool,
    dequantized: &[i32],
    residual: &mut [i32],
) {
    let area = tx.area();
    if lossless {
        // The reference dequantizer multiplies by 4 and its WHT shifts it
        // back out (`UNIT_QUANT_SHIFT`).
        let levels: [i32; 16] = std::array::from_fn(|i| dequantized[i] >> 2);
        residual[..16].copy_from_slice(&nc_dsp::reversible_wht4::inverse(&levels));
        return;
    }
    if tx == TxSize::Tx4 {
        let input: [i32; 16] = std::array::from_fn(|i| dequantized[i]);
        residual[..16].copy_from_slice(&hybrid::inverse(&input, ty));
        return;
    }
    if ty == TxType::DctDct && dequantized[1..area].iter().all(|&c| c == 0) {
        residual[..area].fill(dc_only_residual(tx, dequantized[0]));
        return;
    }
    let (columns, rows) = passes(ty);
    dsp.vp9_inverse_transform(
        tx.log2(),
        columns,
        rows,
        &dequantized[..area],
        &mut residual[..area],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_shortcut_equals_the_full_inverse() {
        let dsp = nc_dsp::Dsp::new();
        for tx in [TxSize::Tx8, TxSize::Tx16, TxSize::Tx32] {
            let area = tx.area();
            for dc in (i32::from(i16::MIN)..=i32::from(i16::MAX)).step_by(13) {
                let mut input = vec![0; area];
                input[0] = dc;
                let mut full = vec![0; area];
                dsp.vp9_inverse_transform(
                    tx.log2(),
                    nc_dsp::Vp9Pass::Dct,
                    nc_dsp::Vp9Pass::Dct,
                    &input,
                    &mut full,
                );
                assert!(
                    full.iter().all(|&r| r == dc_only_residual(tx, dc)),
                    "{tx:?} {dc}"
                );
            }
        }
    }

    #[test]
    fn dc_only_blocks_round_trip_close_to_the_residual() {
        for tx in [TxSize::Tx4, TxSize::Tx8, TxSize::Tx16, TxSize::Tx32] {
            let area = tx.area();
            let residual = vec![37i16; area];
            let mut coefficients = vec![0; area];
            forward(
                &nc_dsp::Dsp::new(),
                tx,
                TxType::DctDct,
                false,
                &residual,
                &mut coefficients,
            );
            let steps = Steps::new(1, 0, 0);
            let mut levels = vec![0; area];
            assert!(quantize(
                &nc_dsp::Dsp::new(),
                tx,
                false,
                &coefficients,
                steps,
                (Rounding::Intra, 1),
                &mut levels
            ));
            let dequantized: Vec<i32> = levels
                .iter()
                .enumerate()
                .map(|(i, &l)| dequantize_level(tx, i, l, steps))
                .collect();
            let mut output = vec![0; area];
            inverse(
                &nc_dsp::Dsp::new(),
                tx,
                TxType::DctDct,
                false,
                &dequantized,
                &mut output,
            );
            assert!(
                output.iter().all(|&r| (r - 37).abs() <= 1),
                "{tx:?}: {output:?}"
            );
        }
    }

    #[test]
    fn levels_never_overflow_the_sixteen_bit_dequantizer() {
        for tx in [TxSize::Tx8, TxSize::Tx32] {
            let area = tx.area();
            let mut coefficients = vec![0; area];
            forward(
                &nc_dsp::Dsp::new(),
                tx,
                TxType::DctDct,
                false,
                &vec![255; area],
                &mut coefficients,
            );
            let steps = Steps::new(1, 0, 0);
            let mut levels = vec![0; area];
            quantize(
                &nc_dsp::Dsp::new(),
                tx,
                false,
                &coefficients,
                steps,
                (Rounding::Intra, 1),
                &mut levels,
            );
            assert!(dequantize_level(tx, 0, levels[0], steps) <= i32::from(i16::MAX));
        }
    }

    #[test]
    fn lossless_is_exact() {
        let residual: Vec<i16> = (0..16).map(|i| (i * 37 % 511 - 255) as i16).collect();
        let mut coefficients = [0; 16];
        forward(
            &nc_dsp::Dsp::new(),
            TxSize::Tx4,
            TxType::DctDct,
            true,
            &residual,
            &mut coefficients,
        );
        let mut levels = [0; 16];
        quantize(
            &nc_dsp::Dsp::new(),
            TxSize::Tx4,
            true,
            &coefficients,
            Steps::new(0, 0, 0),
            (Rounding::Intra, 1),
            &mut levels,
        );
        let dequantized: Vec<i32> = levels.iter().map(|&l| l * 4).collect();
        let mut output = [0; 16];
        inverse(
            &nc_dsp::Dsp::new(),
            TxSize::Tx4,
            TxType::DctDct,
            true,
            &dequantized,
            &mut output,
        );
        assert_eq!(
            output.to_vec(),
            residual.iter().map(|&r| i32::from(r)).collect::<Vec<_>>()
        );
    }
}
