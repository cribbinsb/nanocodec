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

use nc_encoder::{Candidates, CoeffRateModel, Trellis};

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
    /// nonzero `level` in `state`.
    fn level_bits(&self, state: LevelState, level: u32) -> f32 {
        let minus1 = level - 1;
        let first_inc = if state.gt1 != 0 {
            0
        } else {
            usize::from(1 + state.eq1)
        };
        let later = self.level + LATER_BINS_INC + usize::from(state.gt1);
        let prefix = minus1.min(LEVEL_PREFIX_MAX);
        let mut bits = self.bins.bin_bits(self.level + first_inc, prefix > 0);
        if prefix > 0 {
            bits += (prefix - 1) as f32 * self.bins.bin_bits(later, true);
            if prefix < LEVEL_PREFIX_MAX {
                bits += self.bins.bin_bits(later, false);
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
        if !state.started {
            // The last level: its flags went with the end of block.
            for (&level, bits) in levels.iter().zip(bits) {
                *bits = self.level_bits(*state, level);
            }
            return;
        }
        // Not the last level, so not the final scan position, which has
        // no significance flags.
        let (significant, last) = self.significance_contexts(scan_pos);
        for (&level, bits) in levels.iter().zip(bits) {
            *bits = if level == 0 {
                self.bins.bin_bits(significant, false)
            } else {
                self.bins.bin_bits(significant, true)
                    + self.bins.bin_bits(last, false)
                    + self.level_bits(*state, level)
            };
        }
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
        coeffs: &[i32],
        scan: &[usize],
        (scale, lambda, max_states): (BlockScale<'_>, f32, usize),
        levels: &mut [i32],
    ) -> bool {
        let Some(last) = scan.iter().rposition(|&p| levels[p] != 0) else {
            return false;
        };
        // 2^-qbits: exact, so multiplying equals dividing by 2^qbits.
        let unit = 1.0 / f64::from(1_u32 << scale.qbits);
        let half = 1_i64 << (scale.qbits - 1);
        self.candidates.clear();
        self.candidates.extend(scan[..=last].iter().map(|&p| {
            // |c| · mf < 2^53, so `value` is exact and rounding it to the
            // nearest (half away from zero) is this integer rounding.
            let scaled = i64::from(coeffs[p].unsigned_abs()) * i64::from(scale.mf[p]);
            let value = scaled as f64 * unit;
            let nearest = (((scaled + half) >> scale.qbits) as u32).max(levels[p].unsigned_abs());
            Candidates::around(nearest, |level| {
                let error = value - f64::from(level);
                (error * error) as f32 * scale.step_sse
            })
        }));
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
            let level = if i < coded { self.chosen[i] as i32 } else { 0 };
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
                let mut levels = vec![0_i32; n];
                for &p in scan {
                    let v = match next() % 8 {
                        0..=3 => 0,
                        4 | 5 => 1,
                        6 => 2 + (next() % 30) as i32,
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
                    let level = levels[scan[i]].unsigned_abs();
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
