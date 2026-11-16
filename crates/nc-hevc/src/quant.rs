//! Quantisation (encoder choice) and scaling (normative, HEVC §8.6.2–8.6.3).
//!
//! The encoder quantises and scales with nc-dsp's kernels
//! (`Dsp::hevc_quantize`, `Dsp::hevc_dequantize`); the scalar versions here
//! are their test-only reference. What stays in use is what the kernels do
//! not cover: chroma QP mapping, the all-zero bound and sign hiding.
//!
//! Scaling lists are off, so the scaling factor `m` is 16 everywhere. The
//! forward quantiser is the HM dead-zone quantiser: the rounding offset is
//! 171/512 of a step for intra blocks and 128/512 for inter blocks (both
//! tunable, `hevc.intra_rounding` / `hevc.inter_rounding`).

use nc_encoder::CodedGroups;

use crate::residual::SIGN_HIDING_THRESHOLD;
use crate::scan::{SUB_BLOCK_COEFFS, Scan, trim_sub_blocks};
#[cfg(test)]
use crate::transform::{COEFF_MAX, COEFF_MIN};
use nc_dsp::HevcQuantizer;

/// Bit depth of every sample (Main profile).
const BIT_DEPTH: i32 = 8;
#[cfg(test)]
/// `levelScale` of HEVC §8.6.3, indexed by `qP % 6`.
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];
#[cfg(test)]
/// The flat scaling factor `m` when scaling lists are disabled (§8.6.3).
const FLAT_SCALING_FACTOR: i64 = 16;
/// Dynamic range of transform coefficients (HM `MAX_TR_DYNAMIC_RANGE`).
const TRANSFORM_RANGE: i32 = 15;
/// Intra rounding offset, in 1/512 of a quantisation step (HM; the default
/// of `hevc.intra_rounding`).
pub const INTRA_ROUNDING_512THS: i32 = 171;
/// Inter rounding offset (the default of `hevc.inter_rounding`). HM uses 85
/// (a wide dead zone: residuals after motion compensation are mostly
/// noise); 128 keeps the small texture coefficients SVT-AV1's quantiser
/// keeps (zbin about 0.66 step), and with `hevc.shave_gap` = 8 dropping the
/// isolated high-frequency ±1s that are noise, VMAF-NEG vs svtav1-m12 went
/// +13.0% → +8.3% for −5.3% → −4.9% PSNR-YUV (quick clips, 10 s IP).
pub const INTER_ROUNDING_512THS: i32 = 128;

/// Chroma QP for `qPi` = luma QP + chroma QP offset when
/// `ChromaArrayType == 1` (HEVC Table 8-10).
#[must_use]
pub fn chroma_qp(luma_qp: i32) -> i32 {
    chroma_qp_table(luma_qp.clamp(0, MAX_CHROMA_QPI))
}

/// The largest `qPi` of Table 8-10 (the clip of eq. 8-257).
pub const MAX_CHROMA_QPI: i32 = 57;

/// `QpC` of Table 8-10 for any `qPi`, unclipped: below 30 it is `qPi`
/// itself, negative at high bit depth (`qPi >= −QpBdOffsetC`) and in the
/// deblocking filter's average (§8.7.2.5.5, which does not clip `qPi`).
#[must_use]
pub fn chroma_qp_table(qpi: i32) -> i32 {
    const TABLE_30_TO_43: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
    match qpi {
        ..30 => qpi,
        30..=43 => TABLE_30_TO_43[(qpi - 30) as usize],
        _ => qpi - 6,
    }
}

/// The largest coefficient magnitude that [`quantize`] maps to level 0.
#[must_use]
pub fn largest_zero_coefficient(qp: i32, log2: u8, rounding: i32) -> u64 {
    u64::from(HevcQuantizer::with_rounding(qp as u8, log2, rounding).largest_zero_coefficient())
}

/// The largest orthonormal-DCT coefficient (residual sample units) that
/// quantises to level 0 at `qp` with `rounding`: the core transform of a
/// `2^log2` block scales orthonormal coefficients by
/// `2^(TRANSFORM_RANGE − BIT_DEPTH − log2)` (HEVC §8.6.4.2), the same
/// limit for every size up to integer rounding; taken at 8x8.
#[must_use]
pub fn orthonormal_zero_limit(qp: i32, rounding: i32) -> f32 {
    const LOG2: u8 = 3;
    let gain = 1i64 << (TRANSFORM_RANGE - BIT_DEPTH - i32::from(LOG2));
    largest_zero_coefficient(qp, LOG2, rounding) as f32 / gain as f32
}

/// `(qbits, scale, rounding offset)` of the forward quantiser (nc-dsp's
/// [`HevcQuantizer`]), for a rounding offset of `rounding` 512ths of a step.
fn quant_params(qp: i32, log2: u8, rounding: i32) -> (i32, i64, i64) {
    let (qbits, scale, offset) =
        HevcQuantizer::with_rounding(qp as u8, log2, rounding).forward_terms();
    (qbits as i32, i64::from(scale), i64::from(offset))
}

/// `(qbits, scale)` of the forward quantiser at `qp` for a `2^log2` block:
/// `|c| · scale / 2^qbits` is a coefficient's magnitude in quantisation
/// steps (HM `quantScales`, `QUANT_SHIFT`).
#[must_use]
pub fn step_terms(qp: i32, log2: u8) -> (i32, i64) {
    let (qbits, scale, _) = quant_params(qp, log2, 0);
    (qbits, scale)
}

#[cfg(test)]
/// Quantise transform coefficients in place into levels.
pub fn quantize(coeffs: &mut [i32], qp: i32, log2: u8, intra: bool) {
    let rounding = if intra {
        INTRA_ROUNDING_512THS
    } else {
        INTER_ROUNDING_512THS
    };
    let (qbits, scale, offset) = quant_params(qp, log2, rounding);
    for c in coeffs {
        let magnitude = (i64::from(c.unsigned_abs()) * scale + offset) >> qbits;
        let level = magnitude.min(i64::from(COEFF_MAX)) as i32;
        *c = if *c < 0 { -level } else { level };
    }
}

/// Drop trailing ±1 levels that stand at least `min_gap` scan positions after
/// the previous non-zero level, from the end of the block backwards
/// (SVT-AV1's `shave_coeff`: an isolated high-frequency ±1 costs its last
/// position and significance bits for almost no distortion gain; it is
/// mostly noise). `sub_blocks` are the block's coded sub-blocks; returns
/// them after shaving (the block's first level always remains).
///
/// The walk visits only coded sub-blocks: testing every sub-block of the
/// block again cost about as much as the levels it reads. Exact; per P
/// frame HEVC Fast instructions -1.0 % / -0.1 % (callgrind, `WestWindEasy` /
/// `Johnny`, QP 37).
pub fn shave_isolated_ones(
    levels: &mut [i16],
    scan: &Scan,
    sub_blocks: CodedGroups,
    min_gap: usize,
) -> CodedGroups {
    // A shaved level may empty its sub-block, leaving a stale bit; the walk
    // only moves down and then finds nothing there, and the stale bits (all
    // above the last remaining level) are trimmed after it.
    nc_encoder::shave_isolated_ones(levels, &scan.positions, min_gap, |levels, end| {
        previous_nonzero(levels, scan, sub_blocks, end)
    });
    trim_sub_blocks(sub_blocks, levels, scan)
}

/// The highest scan index below `end` whose level is non-zero, searched in
/// the sub-blocks `sub_blocks` may hold levels in.
fn previous_nonzero(
    levels: &[i16],
    scan: &Scan,
    sub_blocks: CodedGroups,
    end: usize,
) -> Option<usize> {
    let mut end = end;
    while let Some(sub_block) = sub_blocks.last_below(end.div_ceil(SUB_BLOCK_COEFFS)) {
        let start = sub_block * SUB_BLOCK_COEFFS;
        let stop = end.min(start + SUB_BLOCK_COEFFS);
        if let Some(i) = (start..stop)
            .rev()
            .find(|&i| levels[scan.positions[i]] != 0)
        {
            return Some(i);
        }
        end = start;
    }
    None
}

/// Make every sub-block that hides a sign (`SIGN_HIDING_THRESHOLD`) carry
/// the right parity: the sum of its absolute levels must be odd exactly when
/// its first (lowest scan position) level is negative (HEVC §7.4.9.11).
/// Where it is not, the one-level change that adds the least squared error
/// is applied (the HM "sign bit hiding" adjustment, distortion only):
/// raising or lowering a level inside the sub-block's non-zero span by one,
/// never zeroing the first or last level of the span (that would move the
/// span), so `sub_blocks` (the coded sub-blocks) stay as they are. `coeffs`
/// are the transform coefficients before quantisation.
pub fn hide_signs(
    coeffs: &[i16],
    (levels, sub_blocks): (&mut [i16], CodedGroups),
    scan: &Scan,
    qp: i32,
) {
    let (qbits, scale, _) = quant_params(qp, scan.log2, 0);
    // Unquantised magnitude in quantisation steps.
    let value =
        |p: usize| (i64::from(coeffs[p].unsigned_abs()) * scale) as f64 / (1i64 << qbits) as f64;
    let spans = scan.positions.as_chunks::<SUB_BLOCK_COEFFS>().0;
    for span in sub_blocks.iter().map(|i| &spans[i]) {
        let nonzero = |n: &usize| levels[span[*n]] != 0;
        let (Some(first), Some(last)) = (
            (0..SUB_BLOCK_COEFFS).find(nonzero),
            (0..SUB_BLOCK_COEFFS).rfind(nonzero),
        ) else {
            continue;
        };
        if last - first <= SIGN_HIDING_THRESHOLD {
            continue;
        }
        let sum: u32 = span
            .iter()
            .map(|&p| u32::from(levels[p].unsigned_abs()))
            .sum();
        if (sum % 2 == 1) == (levels[span[first]] < 0) {
            continue;
        }
        // (cost increase, position, change of magnitude)
        let mut best: Option<(f64, usize, i32)> = None;
        for (n, &p) in span.iter().enumerate().take(last + 1).skip(first) {
            let magnitude = levels[p].unsigned_abs();
            let (level, v) = (f64::from(magnitude), value(p));
            let mut consider = |cost: f64, delta: i32| {
                if best.is_none_or(|(c, _, _)| cost < c) {
                    best = Some((cost, p, delta));
                }
            };
            consider(2.0 * (level - v) + 1.0, 1);
            let can_lower = magnitude > 1 || (magnitude == 1 && n != first && n != last);
            if can_lower {
                consider(2.0 * (v - level) + 1.0, -1);
            }
        }
        let (_, p, delta) = best.expect("the span has at least one level");
        // At most 2^14 + 1 (the backend's `store_levels`): fits 16 bits.
        let magnitude = (i32::from(levels[p].unsigned_abs()) + delta) as i16;
        // A new level takes the coefficient's sign (positive for zero).
        let negative = if levels[p] != 0 {
            levels[p] < 0
        } else {
            coeffs[p] < 0
        };
        levels[p] = if negative { -magnitude } else { magnitude };
    }
}

#[cfg(test)]
/// Scale levels back to transform coefficients (HEVC §8.6.3, eq. 8-309).
pub fn dequantize(levels: &[i32], qp: i32, log2: u8, out: &mut [i32]) {
    let bd_shift = BIT_DEPTH + i32::from(log2) - 5;
    let scale = FLAT_SCALING_FACTOR * LEVEL_SCALE[(qp % 6) as usize];
    let round = 1i64 << (bd_shift - 1);
    for (o, &l) in out.iter_mut().zip(levels) {
        let v = ((i64::from(l) * scale) << (qp / 6)) + round;
        *o = (v >> bd_shift).clamp(i64::from(COEFF_MIN), i64::from(COEFF_MAX)) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::{ScanKind, coded_sub_blocks};

    #[test]
    fn shaving_drops_only_isolated_trailing_ones() {
        let scan = Scan::new(3, ScanKind::Diagonal);
        let mut levels = vec![0; 64];
        let at = |i: usize| scan.positions[i];
        levels[at(0)] = 5;
        levels[at(3)] = -1;
        levels[at(20)] = 1; // 17 positions after the -1: isolated
        levels[at(40)] = -1; // 20 after: isolated
        let coded = coded_sub_blocks(&levels, &scan);
        let coded = shave_isolated_ones(&mut levels, &scan, coded, 8);
        assert_eq!(coded, coded_sub_blocks(&levels, &scan));
        assert_eq!((levels[at(20)], levels[at(40)]), (0, 0));
        assert_eq!((levels[at(0)], levels[at(3)]), (5, -1)); // gap 3 < 8: kept
        // A 2 at the end is never dropped.
        levels[at(50)] = 2;
        let coded = coded_sub_blocks(&levels, &scan);
        shave_isolated_ones(&mut levels, &scan, coded, 8);
        assert_eq!(levels[at(50)], 2);
    }

    #[test]
    fn chroma_qp_follows_table_8_10() {
        assert_eq!(chroma_qp(29), 29);
        assert_eq!(chroma_qp(30), 29);
        assert_eq!(chroma_qp(37), 34);
        assert_eq!(chroma_qp(43), 37);
        assert_eq!(chroma_qp(51), 45);
    }

    #[test]
    fn quantise_then_scale_errs_by_less_than_one_step() {
        // QP 4, 8x8: the step in the coefficient domain is 2^(15-8-3) = 16,
        // and the dead zone (offset 171/512) bounds the error below one step.
        let mut c = vec![100, -37, 0, 5];
        let orig = c.clone();
        quantize(&mut c, 4, 3, true);
        let mut d = vec![0; 4];
        dequantize(&c, 4, 3, &mut d);
        for (a, b) in orig.iter().zip(&d) {
            assert!((a - b).abs() < 16, "{orig:?} -> {d:?}");
        }
    }

    #[test]
    fn hidden_signs_get_the_right_parity() {
        let scan = Scan::new(3, ScanKind::Diagonal);
        let mut seed = 0x0bad_5eed_1234_5678u64;
        for _ in 0..200 {
            let coeffs: Vec<i16> = (0..64)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed % 4001) as i16 - 2000
                })
                .collect();
            let mut wide: Vec<i32> = coeffs.iter().map(|&c| i32::from(c)).collect();
            quantize(&mut wide, 22, 3, false);
            let mut levels: Vec<i16> = wide.iter().map(|&l| l as i16).collect();
            let sub_blocks = coded_sub_blocks(&levels, &scan);
            hide_signs(&coeffs, (&mut levels, sub_blocks), &scan, 22);
            for span in scan.positions.as_chunks::<SUB_BLOCK_COEFFS>().0 {
                let nz: Vec<usize> = (0..SUB_BLOCK_COEFFS)
                    .filter(|&n| levels[span[n]] != 0)
                    .collect();
                if let (Some(&first), Some(&last)) = (nz.first(), nz.last())
                    && last - first > SIGN_HIDING_THRESHOLD
                {
                    let sum: u32 = span
                        .iter()
                        .map(|&p| u32::from(levels[p].unsigned_abs()))
                        .sum();
                    assert_eq!(sum % 2 == 1, levels[span[first]] < 0);
                }
            }
        }
    }
}
