//! The H.264 CABAC coefficient rate model of the shared trellis
//! ([`nc_encoder::CoeffRateModel`]) and the trellis quantiser built on it.
//!
//! Prices levels as `residual_block_cabac()` codes them (H.264 §7.3.5.3.3,
//! contexts §9.3.3.1.1.9 and §9.3.3.1.3) with the slice's context states as
//! they stand. The significance map is coded forward but its contexts
//! depend on the scan position only, so each position's flags are priced
//! where its level is decided: a zero pays `significant_coeff_flag` = 0, a
//! nonzero level before the last pays 1 and `last_significant_coeff_flag`
//! = 0, and the last level's flags (1, 1) go with the end of block. The
//! levels are coded in reverse scan order with contexts from the counts of
//! ones and of larger levels coded so far (`numDecodAbsLevelEq1`,
//! `numDecodAbsLevelGt1`): that is the whole state, which merges exactly,
//! as in x264's CABAC trellis.

use nc_encoder::{Candidates, CoeffRateModel, LoneLevel, Trellis, lone_level_kept};

use super::{
    BlockCat, CODED_BLOCK_FLAG_OFFSET, COEFF_ABS_LEVEL_8X8_OFFSET, COEFF_ABS_LEVEL_OFFSET,
    LAST_SIGNIFICANT_COEFF_8X8_OFFSET, LAST_SIGNIFICANT_COEFF_OFFSET, LEVEL_PREFIX_MAX,
    SIGNIFICANT_COEFF_8X8_OFFSET, SIGNIFICANT_COEFF_OFFSET, significance_ctx_inc,
};
use crate::cabac::BitCounter;
use crate::tables::{LAST_CTX_INC_8X8, SIGNIFICANT_CTX_INC_8X8};

/// `ctxIdxInc` of the first `coeff_abs_level_minus1` bin saturates at 4,
/// reached after three levels of one.
const EQ1_CAP: u8 = 3;
/// `numDecodAbsLevelGt1` saturates at 4 in the later bins' `ctxIdxInc`.
const GT1_CAP: u8 = 4;
/// `ctxIdxInc` offset of the later `coeff_abs_level_minus1` bins.
const LATER_BINS_INC: usize = 5;

/// The level counts the remaining contexts depend on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LevelState {
    /// A level has been coded (the block's last).
    started: bool,
    /// `numDecodAbsLevelEq1`, saturated.
    eq1: u8,
    /// `numDecodAbsLevelGt1`, saturated.
    gt1: u8,
}

/// The kind of block a model prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrellisBlock {
    /// A 4x4 (or AC) block of category `cat`, with the
    /// `coded_block_flag` context increment from its neighbours.
    Small { cat: BlockCat, cbf_ctx_inc: usize },
    /// A luma 8x8 block (`ctxBlockCat` 5; no `coded_block_flag`).
    Luma8x8,
}

/// Prices one block's levels.
pub(crate) struct RateModel<'a> {
    bins: &'a BitCounter<'a>,
    block: TrellisBlock,
    significant: usize,
    last: usize,
    level: usize,
    /// Scan positions of the block (`maxNumCoeff`).
    positions: usize,
}

impl<'a> RateModel<'a> {
    pub(crate) fn new(bins: &'a BitCounter<'a>, block: TrellisBlock) -> Self {
        let (significant, last, level, positions) = match block {
            TrellisBlock::Small { cat, .. } => (
                SIGNIFICANT_COEFF_OFFSET + cat.significance_offset(),
                LAST_SIGNIFICANT_COEFF_OFFSET + cat.significance_offset(),
                COEFF_ABS_LEVEL_OFFSET + cat.level_offset(),
                cat.max_num_coeff(),
            ),
            TrellisBlock::Luma8x8 => (
                SIGNIFICANT_COEFF_8X8_OFFSET,
                LAST_SIGNIFICANT_COEFF_8X8_OFFSET,
                COEFF_ABS_LEVEL_8X8_OFFSET,
                64,
            ),
        };
        Self {
            bins,
            block,
            significant,
            last,
            level,
            positions,
        }
    }

    /// `ctxIdx` of `significant_coeff_flag` and `last_significant_coeff_flag`
    /// at scan position `i`.
    #[inline]
    fn significance_contexts(&self, i: usize) -> (usize, usize) {
        match self.block {
            TrellisBlock::Small { cat, .. } => {
                let inc = significance_ctx_inc(cat, i);
                (self.significant + inc, self.last + inc)
            }
            TrellisBlock::Luma8x8 => (
                self.significant + usize::from(SIGNIFICANT_CTX_INC_8X8[i]),
                self.last + usize::from(LAST_CTX_INC_8X8[i]),
            ),
        }
    }

    fn cbf_context(&self) -> Option<usize> {
        match self.block {
            TrellisBlock::Small { cat, cbf_ctx_inc } => {
                Some(CODED_BLOCK_FLAG_OFFSET + cat.cbf_offset() + cbf_ctx_inc)
            }
            TrellisBlock::Luma8x8 => None,
        }
    }

    /// Bits of `coeff_abs_level_minus1` (UEG0, §9.3.2.3) and the sign of a
    /// nonzero `level` given precomputed `first_ctx` and `later_ctx`.
    #[inline]
    fn level_bits_ctx(&self, first_ctx: usize, later_ctx: usize, level: u32) -> f32 {
        let minus1 = level - 1;
        let prefix = minus1.min(LEVEL_PREFIX_MAX);
        let mut bits = self.bins.bin_bits(first_ctx, prefix > 0);
        if prefix > 0 {
            bits += (prefix - 1) as f32 * self.bins.bin_bits(later_ctx, true);
            if prefix < LEVEL_PREFIX_MAX {
                bits += self.bins.bin_bits(later_ctx, false);
            }
        }
        if minus1 >= LEVEL_PREFIX_MAX {
            bits += exp_golomb_bits(minus1 - LEVEL_PREFIX_MAX) as f32;
        }
        bits + 1.0 // coeff_sign_flag
    }
}

/// Bypass bins of the k = 0 Exp-Golomb suffix of `value`
/// (`write_exp_golomb_bypass`).
#[inline]
fn exp_golomb_bits(value: u32) -> u32 {
    let x = value + 1;
    2 * (u32::BITS - x.leading_zeros()) - 1
}

impl CoeffRateModel for RateModel<'_> {
    type State = LevelState;

    fn initial_state(&self) -> LevelState {
        LevelState {
            started: false,
            eq1: 0,
            gt1: 0,
        }
    }

    /// `coded_block_flag` = 1 and the last level's significance flags (not
    /// coded at the final scan position).
    fn eob_rate(&mut self, scan_pos: usize) -> f32 {
        let cbf = self
            .cbf_context()
            .map_or(0.0, |ctx| self.bins.bin_bits(ctx, true));
        if scan_pos + 1 == self.positions {
            return cbf;
        }
        let (significant, last) = self.significance_contexts(scan_pos);
        cbf + self.bins.bin_bits(significant, true) + self.bins.bin_bits(last, true)
    }

    fn level_rates(
        &mut self,
        state: &LevelState,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32],
    ) {
        let n = levels.len().min(bits.len());
        if !state.started {
            let first_ctx = self.level
                + if state.gt1 != 0 {
                    0
                } else {
                    usize::from(1 + state.eq1)
                };
            let later_ctx = self.level + LATER_BINS_INC + usize::from(state.gt1);
            // The last level: its flags went with the end of block.
            for i in 0..n {
                bits[i] = self.level_bits_ctx(first_ctx, later_ctx, levels[i]);
            }
            return;
        }
        // Not the last level, so not the final scan position, which has
        // no significance flags.
        let (significant, last) = self.significance_contexts(scan_pos);
        let zero_bits = self.bins.bin_bits(significant, false);
        if n == 1 && levels[0] == 0 {
            bits[0] = zero_bits;
            return;
        }
        let first_ctx = self.level
            + if state.gt1 != 0 {
                0
            } else {
                usize::from(1 + state.eq1)
            };
        let later_ctx = self.level + LATER_BINS_INC + usize::from(state.gt1);
        let nz_flags = self.bins.bin_bits(significant, true) + self.bins.bin_bits(last, false);
        for i in 0..n {
            let level = levels[i];
            bits[i] = if level == 0 {
                zero_bits
            } else {
                nz_flags + self.level_bits_ctx(first_ctx, later_ctx, level)
            };
        }
    }

    fn zero_step(&mut self, state: &mut LevelState, scan_pos: usize) -> f32 {
        debug_assert!(state.started);
        let (significant, _) = self.significance_contexts(scan_pos);
        self.bins.bin_bits(significant, false)
    }

    fn advance(&self, state: &mut LevelState, _scan_pos: usize, level: u32) {
        state.started = true;
        match level {
            0 => {}
            1 => state.eq1 = (state.eq1 + 1).min(EQ1_CAP),
            _ => state.gt1 = (state.gt1 + 1).min(GT1_CAP),
        }
    }

    fn empty_rate(&mut self) -> f32 {
        self.cbf_context()
            .map_or(0.0, |ctx| self.bins.bin_bits(ctx, false))
    }
}

/// One block's forward quantiser, as the trellis needs it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockScale<'a> {
    /// Forward multipliers by raster position and `qbits`
    /// (`H264Quantizer::forward_scale_4x4`).
    pub mf: &'a [i32],
    pub qbits: u32,
    /// Squared error of one level step, in sample-SSE units (`Qstep²`).
    pub step_sse: f32,
}

/// Trellis quantisation of H.264 blocks, with reusable buffers.
#[derive(Debug, Default)]
pub(crate) struct TrellisQuantizer {
    search: Trellis<LevelState>,
    candidates: Vec<Candidates>,
    chosen: Vec<u32>,
    /// A lone level's zero distortions ([`Self::lone_level`]).
    zero_distortion: Vec<f32>,
}

/// A block's coefficients against its quantiser: each one's level rounded
/// to the nearest and the distortion of each candidate level.
struct Rounding<'a> {
    scale: BlockScale<'a>,
    /// 2^-qbits: exact, so multiplying equals dividing by 2^qbits.
    unit: f64,
    half: i64,
}

impl<'a> Rounding<'a> {
    fn of(scale: BlockScale<'a>) -> Self {
        Self {
            scale,
            unit: 1.0 / f64::from(1_u32 << scale.qbits),
            half: 1_i64 << (scale.qbits - 1),
        }
    }

    /// `|c| · mf` of the coefficient at raster position `p`: below 2^53, so
    /// its value in levels is exact and rounding it to the nearest (half
    /// away from zero) is this integer rounding.
    #[inline]
    fn scaled(&self, coeffs: &[i16], p: usize) -> i64 {
        i64::from(coeffs[p].unsigned_abs()) * i64::from(self.scale.mf[p])
    }

    /// The level nearest a coefficient of `scaled` ([`Self::scaled`]), at
    /// least its plain level `plain`.
    #[inline]
    fn nearest(&self, scaled: i64, plain: i16) -> u32 {
        (((scaled + self.half) >> self.scale.qbits) as u32).max(u32::from(plain.unsigned_abs()))
    }

    /// The coefficient's exact value in levels.
    #[inline]
    fn value(&self, scaled: i64) -> f64 {
        scaled as f64 * self.unit
    }

    /// The distortion of coding a coefficient of `value` levels as `level`.
    #[inline]
    fn distortion(&self, value: f64, level: u32) -> f32 {
        let error = value - f64::from(level);
        (error * error) as f32 * self.scale.step_sse
    }
}

impl TrellisQuantizer {
    /// Choose the levels of one block from its raster transform
    /// coefficients `coeffs`, coded in `scan` order (raster positions), by
    /// distortion plus `lambda` (sample SSE per bit) times `model`'s rate
    /// with at most `max_states` survivors per position. On entry `levels`
    /// (raster) holds the plain quantiser's levels, which bound the search
    /// (their last level in scan order); candidates sit around each
    /// level rounded to the nearest. Returns whether any level is left.
    pub(crate) fn quantize(
        &mut self,
        model: &mut RateModel<'_>,
        coeffs: &[i16],
        scan: &[usize],
        (scale, lambda, max_states): (BlockScale<'_>, f32, usize),
        levels: &mut [i16],
    ) -> bool {
        let Some(last) = scan.iter().rposition(|&p| levels[p] != 0) else {
            return false;
        };
        let rounding = Rounding::of(scale);
        if let Some(lone) = self.lone_level(coeffs, scan, &rounding, last, levels) {
            let kept = lone_level_kept(model, lone, lambda);
            if !kept {
                levels[scan[last]] = 0;
            }
            return kept;
        }
        self.full_search(
            model,
            coeffs,
            (scan, last),
            (&rounding, lambda, max_states),
            levels,
        )
    }

    /// The [`LoneLevel`] of a block whose plain levels are one ±1 at scan
    /// position `last` and whose every other coefficient rounds to zero
    /// (its candidates are zero alone), else `None`: the search's only
    /// choice is then keeping or dropping that level.
    fn lone_level(
        &mut self,
        coeffs: &[i16],
        scan: &[usize],
        rounding: &Rounding<'_>,
        last: usize,
        levels: &[i16],
    ) -> Option<LoneLevel<'_>> {
        let p = scan[last];
        let scaled = rounding.scaled(coeffs, p);
        if levels[p].unsigned_abs() != 1 || rounding.nearest(scaled, levels[p]) != 1 {
            return None;
        }
        self.zero_distortion.clear();
        for &q in &scan[..last] {
            let scaled = rounding.scaled(coeffs, q);
            if rounding.nearest(scaled, levels[q]) != 0 {
                return None;
            }
            self.zero_distortion
                .push(rounding.distortion(rounding.value(scaled), 0));
        }
        let value = rounding.value(scaled);
        self.zero_distortion.push(rounding.distortion(value, 0));
        Some(LoneLevel {
            pos: last,
            zero_distortion: &self.zero_distortion,
            level_distortion: rounding.distortion(value, 1),
        })
    }

    /// [`Self::quantize`] by the shared trellis search.
    fn full_search(
        &mut self,
        model: &mut RateModel<'_>,
        coeffs: &[i16],
        (scan, last): (&[usize], usize),
        (rounding, lambda, max_states): (&Rounding<'_>, f32, usize),
        levels: &mut [i16],
    ) -> bool {
        self.candidates.clear();
        self.candidates.reserve(last + 1);
        for &p in &scan[..=last] {
            let scaled = rounding.scaled(coeffs, p);
            let value = rounding.value(scaled);
            if levels[p] == 0 && scaled < rounding.half {
                self.candidates.push(Candidates {
                    levels: [0, 0, 0],
                    distortion: [(value * value) as f32 * rounding.scale.step_sse, 0.0, 0.0],
                    count: 1,
                });
                continue;
            }
            self.candidates.push(Candidates::around(
                rounding.nearest(scaled, levels[p]),
                |level| rounding.distortion(value, level),
            ));
        }
        self.chosen.resize(last + 1, 0);
        let coded = self.search.quantize(
            model,
            &self.candidates,
            lambda,
            max_states,
            &mut self.chosen,
        );
        let mut any = false;
        for (i, &p) in scan[..=last].iter().enumerate() {
            // At most one above the nearest level of a 16-bit coefficient
            // (below 2^13 for 8-bit residuals, `scaled_ac_fits_i16` in
            // nc-dsp), so it fits 16 bits.
            let level = if i < coded { self.chosen[i] as i16 } else { 0 };
            levels[p] = if coeffs[p] < 0 { -level } else { level };
            any |= level != 0;
        }
        any
    }
}

#[cfg(test)]
mod tests {
    use super::super::{SCAN_4X4, SCAN_4X4_AC, write_luma_8x8_block, write_residual_block};
    use super::*;
    use crate::cabac::{BinCosts, CabacEncoder, SliceKind};
    use crate::tables::{ZIGZAG_4X4, ZIGZAG_8X8};
    use nc_common::BitWriter;
    use nc_dsp::Dsp;

    /// The lone-level fast path decides as the full search on every block
    /// it takes: many random blocks near the quantiser's thresholds, 4x4 and
    /// 8x8, at several QPs, λs and beam widths.
    #[test]
    fn lone_level_equals_the_full_search() {
        let costs = BinCosts::new();
        let cabac = CabacEncoder::new(BitWriter::new(), SliceKind::P, 0, 30);
        let bins = cabac.bit_counter(&costs);
        let mut seed = 7_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        let mut search = TrellisQuantizer::default();
        let mut taken = 0;
        for qp in [18_u8, 28, 36, 44, 51] {
            let quantizer = nc_dsp::H264Quantizer::new(qp, false);
            for (block, scale, scan) in [
                (
                    TrellisBlock::Luma8x8,
                    crate::macroblock::scale_8x8(&quantizer),
                    &ZIGZAG_8X8[..],
                ),
                (
                    TrellisBlock::Small {
                        cat: BlockCat::Luma4x4,
                        cbf_ctx_inc: 1,
                    },
                    crate::macroblock::scale_4x4(&quantizer),
                    &ZIGZAG_4X4[..],
                ),
            ] {
                let n = scan.len();
                let deadzone = (1_i64 << scale.qbits) / 6;
                for _ in 0..4000 {
                    // Coefficients mostly below half a step, one near a step.
                    let step = (1_i64 << scale.qbits) as f64;
                    let mut coeffs = vec![0_i16; n];
                    let lone = scan[next() as usize % n];
                    for (p, c) in coeffs.iter_mut().enumerate() {
                        let mf = f64::from(scale.mf[p]);
                        let fraction = if p == lone {
                            0.6 + f64::from(next() % 100) / 120.0
                        } else {
                            f64::from(next() % 100) / 210.0
                        };
                        let magnitude = (fraction * step / mf) as i16;
                        *c = if next() % 2 == 0 {
                            magnitude
                        } else {
                            -magnitude
                        };
                    }
                    let plain: Vec<i16> = (0..n)
                        .map(|p| {
                            let scaled =
                                i64::from(coeffs[p].unsigned_abs()) * i64::from(scale.mf[p]);
                            let level = ((scaled + deadzone) >> scale.qbits) as i16;
                            if coeffs[p] < 0 { -level } else { level }
                        })
                        .collect();
                    let Some(last) = scan.iter().rposition(|&p| plain[p] != 0) else {
                        continue;
                    };
                    let lambda = [0.3_f32, 1.0, 4.0][next() as usize % 3]
                        * 0.85
                        * 2f32.powf((f32::from(qp) - 12.0) / 3.0);
                    let max_states = [1_usize, 4][next() as usize % 2];
                    let rounding = Rounding::of(scale);
                    if search
                        .lone_level(&coeffs, scan, &rounding, last, &plain)
                        .is_none()
                    {
                        continue;
                    }
                    taken += 1;
                    let (mut fast, mut full) = (plain.clone(), plain.clone());
                    let mut model = RateModel::new(&bins, block);
                    let kept_fast = search.quantize(
                        &mut model,
                        &coeffs,
                        scan,
                        (scale, lambda, max_states),
                        &mut fast,
                    );
                    let mut model = RateModel::new(&bins, block);
                    let kept_full = search.full_search(
                        &mut model,
                        &coeffs,
                        (scan, last),
                        (&rounding, lambda, max_states),
                        &mut full,
                    );
                    assert_eq!((kept_fast, &fast), (kept_full, &full), "qp {qp} {block:?}");
                }
            }
        }
        assert!(taken > 2000, "only {taken} lone blocks drawn");
    }

    /// Summed rates equal the bit counter's price of the written block.
    #[test]
    fn rates_follow_the_residual_coder() {
        let costs = BinCosts::new();
        let dsp = Dsp::new();
        let cabac = CabacEncoder::new(BitWriter::new(), SliceKind::P, 0, 30);
        let mut seed = 3_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        for block in [
            TrellisBlock::Small {
                cat: BlockCat::Luma4x4,
                cbf_ctx_inc: 1,
            },
            TrellisBlock::Small {
                cat: BlockCat::ChromaAc,
                cbf_ctx_inc: 3,
            },
            TrellisBlock::Small {
                cat: BlockCat::LumaAc,
                cbf_ctx_inc: 2,
            },
            TrellisBlock::Luma8x8,
        ] {
            for _ in 0..20 {
                let (scan, n): (&[usize], usize) = match block {
                    TrellisBlock::Luma8x8 => (&ZIGZAG_8X8, 64),
                    TrellisBlock::Small {
                        cat: BlockCat::ChromaAc | BlockCat::LumaAc,
                        ..
                    } => (&ZIGZAG_4X4[1..], 16),
                    TrellisBlock::Small { .. } => (&ZIGZAG_4X4, 16),
                };
                let mut levels = vec![0_i16; n];
                for &p in scan {
                    let v = match next() % 8 {
                        0..=3 => 0,
                        4 | 5 => 1,
                        6 => 2 + (next() % 30) as i16,
                        _ => 2,
                    };
                    levels[p] = if next() % 2 == 0 { v } else { -v };
                }
                levels[scan[1]] = 3;
                let mut counter = cabac.bit_counter(&costs);
                match block {
                    TrellisBlock::Luma8x8 => {
                        write_luma_8x8_block(
                            &mut counter,
                            &dsp,
                            levels.as_slice().try_into().unwrap(),
                        );
                    }
                    TrellisBlock::Small { cat, cbf_ctx_inc } => {
                        let scan = if matches!(cat, BlockCat::ChromaAc | BlockCat::LumaAc) {
                            &SCAN_4X4_AC
                        } else {
                            &SCAN_4X4
                        };
                        write_residual_block(&mut counter, &dsp, cat, &levels, scan, cbf_ctx_inc);
                    }
                }
                let expected = counter.bits();
                let bins = cabac.bit_counter(&costs);
                let mut model = RateModel::new(&bins, block);
                let last = scan.iter().rposition(|&p| levels[p] != 0).unwrap();
                let mut bits = model.eob_rate(last);
                let mut state = model.initial_state();
                for i in (0..=last).rev() {
                    let level = u32::from(levels[scan[i]].unsigned_abs());
                    let mut b = [0.0];
                    model.level_rates(&state, i, &[level], &mut b);
                    bits += b[0];
                    model.advance(&mut state, i, level);
                }
                assert!(
                    (bits - expected).abs() < 0.05,
                    "{block:?}: {bits} vs {expected}"
                );
            }
        }
    }
}
