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

use crate::residual::SIGN_HIDING_THRESHOLD;
use crate::scan::{SUB_BLOCK_COEFFS, Scan, sub_block_coded};
#[cfg(test)]
use crate::transform::{COEFF_MAX, COEFF_MIN};

/// Bit depth of every sample (Main profile).
const BIT_DEPTH: i32 = 8;
#[cfg(test)]
/// `levelScale` of HEVC §8.6.3, indexed by `qP % 6`.
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];
/// Forward scale matching `LEVEL_SCALE` (`quantScales` of HM): the product
/// of the two is `2^20` within rounding.
const QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
#[cfg(test)]
/// The flat scaling factor `m` when scaling lists are disabled (§8.6.3).
const FLAT_SCALING_FACTOR: i64 = 16;
/// Base shift of the forward quantiser (HM `QUANT_SHIFT`).
const QUANT_SHIFT: i32 = 14;
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
    const TABLE_30_TO_43: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
    let qpi = luma_qp.clamp(0, 57);
    match qpi {
        ..30 => qpi,
        30..=43 => TABLE_30_TO_43[(qpi - 30) as usize],
        _ => qpi - 6,
    }
}

/// The largest coefficient magnitude that [`quantize`] maps to level 0.
#[must_use]
pub fn largest_zero_coefficient(qp: i32, log2: u8, rounding: i32) -> u64 {
    let (qbits, scale, offset) = quant_params(qp, log2, rounding);
    // Level 0 while c * scale + offset < 2^qbits.
    (((1i64 << qbits) - offset - 1) / scale).max(0) as u64
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

/// `(qbits, scale, rounding offset)` of the forward quantiser, for a
/// rounding offset of `rounding` 512ths of a step.
fn quant_params(qp: i32, log2: u8, rounding: i32) -> (i32, i64, i64) {
    let transform_shift = TRANSFORM_RANGE - BIT_DEPTH - i32::from(log2);
    let qbits = QUANT_SHIFT + qp / 6 + transform_shift;
    (
        qbits,
        QUANT_SCALE[(qp % 6) as usize],
        i64::from(rounding) << (qbits - 9),
    )
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
/// mostly noise). Returns whether any level remains.
pub fn shave_isolated_ones(levels: &mut [i32], scan: &Scan, min_gap: usize) -> bool {
    let positions = &scan.positions;
    let Some(block_last) = previous_nonzero(levels, scan, positions.len()) else {
        return false;
    };
    // `last` is dropped while it is an isolated ±1; the first level kept
    // ends the block. The first non-zero level of the block is always kept.
    let (mut last, mut keep) = (block_last, block_last);
    while let Some(previous) = previous_nonzero(levels, scan, last) {
        if levels[positions[last]].abs() == 1 && last - previous >= min_gap {
            keep = previous;
            last = previous;
        } else {
            break;
        }
    }
    // Levels after the block's last non-zero one are already zero.
    for &p in &positions[keep + 1..=block_last] {
        levels[p] = 0;
    }
    true
}

/// The highest scan index below `end` whose level is non-zero. Whole empty
/// sub-blocks are skipped by testing their four raster rows, which is much
/// cheaper than visiting their sixteen scan positions one by one.
fn previous_nonzero(levels: &[i32], scan: &Scan, end: usize) -> Option<usize> {
    let size = 1usize << scan.log2;
    let mut end = end;
    while end > 0 {
        let sub_block = (end - 1) / SUB_BLOCK_COEFFS;
        let coded = sub_block_coded(levels, size, scan.sub_blocks[sub_block]);
        let start = sub_block * SUB_BLOCK_COEFFS;
        if coded && let Some(i) = (start..end).rev().find(|&i| levels[scan.positions[i]] != 0) {
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
/// span). `coeffs` are the transform coefficients before quantisation.
pub fn hide_signs(coeffs: &[i32], levels: &mut [i32], scan: &Scan, qp: i32) {
    let (qbits, scale, _) = quant_params(qp, scan.log2, 0);
    // Unquantised magnitude in quantisation steps.
    let value =
        |p: usize| (i64::from(coeffs[p].unsigned_abs()) * scale) as f64 / (1i64 << qbits) as f64;
    let size = 1usize << scan.log2;
    let spans = scan.positions.as_chunks::<SUB_BLOCK_COEFFS>().0;
    for (span, &(xs, ys)) in spans.iter().zip(&scan.sub_blocks) {
        // Most sub-blocks of a coded block are empty: test the 4x4 in raster
        // order (four contiguous rows) before walking it in scan order.
        if !sub_block_coded(levels, size, (xs, ys)) {
            continue;
        }
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
        let sum: u32 = span.iter().map(|&p| levels[p].unsigned_abs()).sum();
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
        let magnitude = levels[p].unsigned_abs() as i32 + delta;
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
    use crate::scan::ScanKind;

    #[test]
    fn shaving_drops_only_isolated_trailing_ones() {
        let scan = Scan::new(3, ScanKind::Diagonal);
        let mut levels = vec![0; 64];
        let at = |i: usize| scan.positions[i];
        levels[at(0)] = 5;
        levels[at(3)] = -1;
        levels[at(20)] = 1; // 17 positions after the -1: isolated
        levels[at(40)] = -1; // 20 after: isolated
        assert!(shave_isolated_ones(&mut levels, &scan, 8));
        assert_eq!((levels[at(20)], levels[at(40)]), (0, 0));
        assert_eq!((levels[at(0)], levels[at(3)]), (5, -1)); // gap 3 < 8: kept
        // A 2 at the end is never dropped.
        levels[at(50)] = 2;
        shave_isolated_ones(&mut levels, &scan, 8);
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
            let coeffs: Vec<i32> = (0..64)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed % 4001) as i32 - 2000
                })
                .collect();
            let mut levels = coeffs.clone();
            quantize(&mut levels, 22, 3, false);
            hide_signs(&coeffs, &mut levels, &scan, 22);
            for span in scan.positions.as_chunks::<SUB_BLOCK_COEFFS>().0 {
                let nz: Vec<usize> = (0..SUB_BLOCK_COEFFS)
                    .filter(|&n| levels[span[n]] != 0)
                    .collect();
                if let (Some(&first), Some(&last)) = (nz.first(), nz.last())
                    && last - first > SIGN_HIDING_THRESHOLD
                {
                    let sum: u32 = span.iter().map(|&p| levels[p].unsigned_abs()).sum();
                    assert_eq!(sum % 2 == 1, levels[span[first]] < 0);
                }
            }
        }
    }
}
