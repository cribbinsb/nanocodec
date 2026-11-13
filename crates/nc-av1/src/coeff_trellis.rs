//! The AV1 coefficient rate model of the shared trellis
//! ([`nc_encoder::CoeffRateModel`]): prices levels with the frame's adapted
//! coefficient CDFs under the exact contexts of `CoeffCdfs::write`
//! (AV1 §5.11.39 `coeffs()`, contexts §8.3.2; libaom `txb_common.h`).
//!
//! Levels are coded in reverse scan order and each `coeff_base` /
//! `coeff_br` context reads only neighbours to the right and below
//! (`(r, c+1)`, `(r+1, c)`, `(r+1, c+1)`, `(r, c+2)`, `(r+2, c)`), which lie
//! on the next two anti-diagonals. The default scans visit anti-diagonals in
//! order, so the state the trellis carries is the saturated magnitudes of
//! the last three diagonals ([`ContextState`]): the two that the current
//! diagonal reads and the current one, which the next diagonal reads. That
//! state is exact for every context. Paths whose levels differ only on
//! diagonals that have left the window merge (an exact dynamic program);
//! the rest compete for a bounded beam of the cheapest states.

use nc_dsp::Av1Quantizer;
use nc_encoder::symbol_cost::{BitCounter, COST_UNITS_PER_BIT};
use nc_encoder::{Candidates, CoeffRateModel, Trellis};

use super::{
    BASE_EOB_SYMBOLS, BR_CDF_SIZE, COEFF_BASE_RANGE, CoeffCdfs, EOB_GROUP_START, EOB_OFFSET_BITS,
    LevelMap, MAP_STRIDE, MAX_BASE_BR_RANGE, NUM_BASE_LEVELS, PlaneType, TxbContext,
    base_eob_context, nz_map_ctx_offset, scan,
};
use crate::transform::TxSize;

/// Spatial SSE of a unit coefficient error at dequantisation shift 0: AV1
/// forward transforms scale orthonormal coefficients by 8, and `dqDenom`
/// (`2^shift`) halves that per step (`transform::coefficient_gain`).
const COEFFICIENT_GAIN_LOG2: u32 = 6;

/// Trellis quantisation of AV1 transform blocks, with reusable buffers.
#[derive(Debug, Default)]
pub struct TrellisQuantizer {
    search: Trellis<ContextState>,
    candidates: Vec<Candidates>,
    chosen: Vec<u32>,
}

impl TrellisQuantizer {
    /// Choose the levels of a block of `model`'s transform size from its
    /// forward coefficients `coeffs` (libaom layout): distortion plus
    /// `lambda` (sample SSE per bit) times `model`'s rate, with at most
    /// `max_states` survivors per scan position. On entry `levels` holds
    /// the plain quantiser's levels (`quantizer`, whose rounding is at most
    /// half a step) and `eob` their end of block; on return the chosen
    /// levels. Each coefficient's candidates sit around its level rounded
    /// to the nearest. The search spans the plain end of block only: a
    /// coefficient past it is under the plain rounding threshold, and coding
    /// it as the new last level rarely pays for the longer end of block,
    /// while scanning to the round-to-nearest end of block cost as much
    /// again on large transforms.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &mut self,
        model: &mut RateModel<'_>,
        coeffs: &[i32],
        quantizer: &Av1Quantizer,
        lambda: f32,
        max_states: usize,
        levels: &mut [i32],
        eob: usize,
    ) -> u32 {
        let scan = &scan(model.tx)[..eob];
        let log_scale = quantizer.log_scale;
        // Coefficient-domain squared error to sample SSE.
        let to_sse = 1.0 / f64::from(1_u32 << (COEFFICIENT_GAIN_LOG2 - 2 * log_scale));
        self.candidates.clear();
        self.candidates.extend(scan.iter().map(|&p| {
            let pos = usize::from(p);
            let step = i64::from(if pos == 0 {
                quantizer.dc_step
            } else {
                quantizer.ac_step
            });
            let magnitude = i64::from(coeffs[pos].unsigned_abs());
            let plain = levels[pos].unsigned_abs();
            let past_half = 2 * (magnitude << log_scale) >= (2 * i64::from(plain) + 1) * step;
            let nearest = if past_half && (plain as i32) < quantizer.max_level {
                plain + 1
            } else {
                plain
            };
            Candidates::around(nearest, |level| {
                let error = (magnitude - ((i64::from(level) * step) >> log_scale)) as f64;
                (error * error * to_sse) as f32
            })
        }));
        self.chosen.resize(eob, 0);
        let coded = self.search.quantize(
            model,
            &self.candidates,
            lambda,
            max_states,
            &mut self.chosen,
        );
        let mut nonzero = 0;
        for (i, &p) in scan.iter().enumerate() {
            let pos = usize::from(p);
            let level = if i < coded { self.chosen[i] as i32 } else { 0 };
            levels[pos] = if coeffs[pos] < 0 { -level } else { level };
            nonzero += u32::from(level != 0);
        }
        nonzero
    }
}

/// Diagonals the context state holds: the current one and the two its
/// contexts read.
const STATE_DIAGONALS: usize = 3;
/// `diagonal` of a state in which no level is coded yet.
const NO_DIAGONAL: u16 = u16::MAX;

/// Saturated magnitudes (at `MAX_BASE_BR_RANGE`, as in `LevelMap`) of the
/// last [`STATE_DIAGONALS`] anti-diagonals, by row. Rows run to
/// `MAP_STRIDE` (the largest coded side plus the context reach), so
/// neighbours past the block read zeros without a test. Diagonals after the
/// last coded level are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextState {
    /// Anti-diagonal (`row + col`) of the last coded level.
    diagonal: u16,
    /// Diagonal `d` lives in slot `d % STATE_DIAGONALS`.
    magnitudes: [[u8; MAP_STRIDE]; STATE_DIAGONALS],
}

impl ContextState {
    /// `(coeff_base sum, coeff_br sum)` of the coefficient at `row` on
    /// `diagonal`, as `LevelMap::base_sum` and
    /// `LevelMap::range_sum`.
    fn neighbour_sums(&self, row: usize, diagonal: usize) -> (u32, u32) {
        let next = &self.magnitudes[(diagonal + 1) % STATE_DIAGONALS];
        let after = &self.magnitudes[(diagonal + 2) % STATE_DIAGONALS];
        // (r, c+1), (r+1, c) on the next diagonal; (r+1, c+1) on the one
        // after; and the farther (r, c+2), (r+2, c) read by `coeff_base`.
        let near = [next[row], next[row + 1], after[row + 1]].map(u32::from);
        let far = [after[row], after[row + 2]].map(u32::from);
        let cap = |m: u32| m.min(u32::from(super::BASE_CONTEXT_LEVEL_MAX));
        let base = near.iter().chain(&far).map(|&m| cap(m)).sum();
        let br = near.iter().sum();
        (base, br)
    }
}

/// Prices one transform block's levels for the trellis.
pub struct RateModel<'a> {
    cdfs: &'a mut CoeffCdfs,
    tx: TxSize,
    plane: PlaneType,
    ctx: TxbContext,
    /// Bits of the coded block flag and the luma transform type.
    coded_block_bits: f32,
    /// Whether the DC coefficient is negative (its sign is context coded).
    dc_negative: bool,
    /// The transform's scan and `nz_map_ctx_offset`, looked up once.
    scan: &'static [u16],
    nz_offsets: &'static [u8],
}

impl<'a> RateModel<'a> {
    /// The model of a block coded with `cdfs` as they stand; the other
    /// arguments are those `CoeffCdfs::write` will get. `dc_negative` is the
    /// sign of the unquantised DC coefficient.
    pub fn new(
        cdfs: &'a mut CoeffCdfs,
        tx: TxSize,
        plane: PlaneType,
        is_inter: bool,
        intra_mode: usize,
        ctx: TxbContext,
        dc_negative: bool,
    ) -> Self {
        let mut counter = BitCounter::default();
        let txs = tx.index();
        counter.add_q9(cdfs.txb_skip[txs][ctx.skip].cost_q9(0));
        if plane == PlaneType::Luma {
            if is_inter {
                cdfs.write_inter_tx_type_dct(&mut counter, tx);
            } else {
                cdfs.write_intra_tx_type_dct(&mut counter, tx, intra_mode);
            }
        }
        Self {
            cdfs,
            tx,
            plane,
            ctx,
            coded_block_bits: counter.bits(),
            dc_negative,
            scan: scan(tx),
            nz_offsets: nz_map_ctx_offset(tx),
        }
    }

    /// Coefficient index, row and column of scan position `scan_pos`.
    fn coordinates(&self, scan_pos: usize) -> (usize, usize, usize) {
        let pos = usize::from(self.scan[scan_pos]);
        let row = pos & (self.tx.coded_size() - 1);
        let col = pos >> self.tx.coded_log2();
        (pos, row, col)
    }

    /// `coeff_br` symbols of `level` (> `NUM_BASE_LEVELS`), in 1/512 bit.
    fn range_cost_q9(&mut self, br_ctx: usize, level: u32) -> u32 {
        let cdf = &mut self.cdfs.br[self.tx.index().min(3)][self.plane as usize][br_ctx];
        let base_range = level as i32 - 1 - NUM_BASE_LEVELS;
        let mut cost = 0;
        let mut idx = 0;
        while idx < COEFF_BASE_RANGE {
            let k = (base_range - idx).min(BR_CDF_SIZE - 1);
            cost += cdf.cost_q9(k as usize);
            if k < BR_CDF_SIZE - 1 {
                break;
            }
            idx += BR_CDF_SIZE - 1;
        }
        cost
    }
}

/// Bits of the Exp-Golomb remainder of `level` (`write_golomb`).
fn golomb_bits(level: u32) -> u32 {
    if level < MAX_BASE_BR_RANGE as u32 {
        return 0;
    }
    let x = level - MAX_BASE_BR_RANGE as u32 + 1;
    2 * (32 - x.leading_zeros()) - 1
}

fn q9_to_bits(q9: u32) -> f32 {
    q9 as f32 / COST_UNITS_PER_BIT as f32
}

impl CoeffRateModel for RateModel<'_> {
    type State = ContextState;

    fn initial_state(&self) -> ContextState {
        ContextState {
            diagonal: NO_DIAGONAL,
            magnitudes: [[0; MAP_STRIDE]; STATE_DIAGONALS],
        }
    }

    /// `all_zero` = 0, the luma transform type and the end of block
    /// (`eob_pt_*`, `eob_extra` and its literal bits, priced as
    /// `CoeffCdfs::write_eob` codes them).
    fn eob_rate(&mut self, scan_pos: usize) -> f32 {
        let eob = scan_pos + 1;
        let eob_pt = EOB_GROUP_START
            .iter()
            .rposition(|&s| s <= eob)
            .expect("eob >= 1");
        let pt = self.plane as usize;
        let symbol = eob_pt - 1;
        // `eob_multi_ctx` is 0 for transform class 2D.
        let mut cost = match self.tx {
            TxSize::Tx4x4 => self.cdfs.eob_pt_16[pt][0].cost_q9(symbol),
            TxSize::Tx8x8 => self.cdfs.eob_pt_64[pt][0].cost_q9(symbol),
            TxSize::Tx16x16 => self.cdfs.eob_pt_256[pt][0].cost_q9(symbol),
            TxSize::Tx32x32 | TxSize::Tx64x64 => self.cdfs.eob_pt_1024[pt][0].cost_q9(symbol),
        };
        let bits = EOB_OFFSET_BITS[eob_pt];
        if bits > 0 {
            let top = ((eob - EOB_GROUP_START[eob_pt]) >> (bits - 1)) & 1;
            cost += self.cdfs.eob_extra[self.tx.index()][pt][eob_pt - 3].cost_q9(top);
            cost += (bits - 1) * COST_UNITS_PER_BIT;
        }
        self.coded_block_bits + q9_to_bits(cost)
    }

    fn level_rates(
        &mut self,
        state: &ContextState,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32],
    ) {
        let (pos, row, col) = self.coordinates(scan_pos);
        let (txs, pt) = (self.tx.index(), self.plane as usize);
        let last = state.diagonal == NO_DIAGONAL;
        // The contexts are shared by every candidate.
        let (base_ctx, br_sum) = if last {
            // The last coded level: `coeff_base_eob`, no neighbours.
            (base_eob_context(scan_pos, self.tx.coded_size()), 0)
        } else {
            let (base_sum, br_sum) = state.neighbour_sums(row, row + col);
            let offset = self.nz_offsets[pos];
            (LevelMap::base_context(pos, base_sum, offset), br_sum)
        };
        for (i, &level) in levels.iter().enumerate() {
            let base_symbol = level.min(3) as usize;
            let mut cost = if last {
                debug_assert!((1..=BASE_EOB_SYMBOLS).contains(&base_symbol));
                self.cdfs.base_eob[txs][pt][base_ctx].cost_q9(base_symbol - 1)
            } else {
                self.cdfs.base[txs][pt][base_ctx].cost_q9(base_symbol)
            };
            if level == 0 {
                bits[i] = q9_to_bits(cost);
                continue;
            }
            if level as i32 > NUM_BASE_LEVELS {
                cost += self.range_cost_q9(LevelMap::br_context(pos, row, col, br_sum), level);
            }
            cost += if pos == 0 {
                self.cdfs.dc_sign[pt][self.ctx.dc_sign].cost_q9(usize::from(self.dc_negative))
            } else {
                COST_UNITS_PER_BIT
            };
            bits[i] = q9_to_bits(cost) + golomb_bits(level) as f32;
        }
    }

    fn advance(&self, state: &mut ContextState, scan_pos: usize, level: u32) {
        let (_, row, col) = self.coordinates(scan_pos);
        let diagonal = row + col;
        let slot = &mut state.magnitudes[diagonal % STATE_DIAGONALS];
        if usize::from(state.diagonal) != diagonal {
            // Entering a new diagonal: its slot held the one three further
            // on, which no remaining context reads.
            slot.fill(0);
            state.diagonal = diagonal as u16;
        }
        slot[row] = level.min(MAX_BASE_BR_RANGE as u32) as u8;
    }

    fn empty_rate(&mut self) -> f32 {
        self.cdfs.txb_skip[self.tx.index()][self.ctx.skip].bits(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZES: [TxSize; 5] = [
        TxSize::Tx4x4,
        TxSize::Tx8x8,
        TxSize::Tx16x16,
        TxSize::Tx32x32,
        TxSize::Tx64x64,
    ];

    /// The state is only exact if scans visit anti-diagonals in order.
    #[test]
    fn default_scans_visit_anti_diagonals_in_order() {
        for tx in SIZES {
            let n = tx.coded_size();
            let diagonals: Vec<usize> = scan(tx)
                .iter()
                .map(|&p| usize::from(p) % n + usize::from(p) / n)
                .collect();
            assert!(diagonals.windows(2).all(|w| w[0] <= w[1]), "{tx:?}");
        }
    }

    /// Summed level rates equal the block cost `CoeffCdfs::cost` reports.
    #[test]
    fn rates_add_up_to_the_coded_block_cost() {
        let mut seed = 7_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        for tx in SIZES {
            for plane in [PlaneType::Luma, PlaneType::Chroma] {
                let area = tx.coded_size() * tx.coded_size();
                let mut levels = vec![0_i32; area];
                for &p in &scan(tx)[..area.min(40)] {
                    let v = match next() % 8 {
                        0..=3 => 0,
                        4 | 5 => 1,
                        6 => 3 + (next() % 20) as i32,
                        _ => 2,
                    };
                    levels[usize::from(p)] = if next() % 2 == 0 { v } else { -v };
                }
                levels[usize::from(scan(tx)[5])] = 4;
                let eob = super::super::end_of_block(tx, &levels);
                let ctx = TxbContext::derive(plane, true, &[0x9], &[0x3]);
                let mut cdfs = CoeffCdfs::new(1);
                let expected = cdfs.cost(tx, plane, true, 0, ctx, &levels, eob);
                let mut model = RateModel::new(&mut cdfs, tx, plane, true, 0, ctx, levels[0] < 0);
                let mut bits = model.eob_rate(eob - 1);
                let mut state = model.initial_state();
                for c in (0..eob).rev() {
                    let level = levels[usize::from(scan(tx)[c])].unsigned_abs();
                    let mut level_bits = [0.0];
                    model.level_rates(&state, c, &[level], &mut level_bits);
                    bits += level_bits[0];
                    model.advance(&mut state, c, level);
                }
                assert!(
                    (bits - expected).abs() < 0.05,
                    "{tx:?} {plane:?}: {bits} vs {expected}"
                );
                let empty = cdfs.cost(tx, plane, true, 0, ctx, &levels, 0);
                let mut model = RateModel::new(&mut cdfs, tx, plane, true, 0, ctx, false);
                assert!((model.empty_rate() - empty).abs() < 0.01);
            }
        }
    }
}
