//! Parity hiding of the luma DC level (AVM v1.0.0 `av2_read_coeffs_txb` /
//! `av2_write_coeffs_txb` with `allow_parity_hiding`, `read_coeff_hidden`,
//! `get_base_ctx_ph`; encoder side `parity_hiding_trellis_off` in
//! `av2/encoder/encodemb.c`).
//!
//! A luma transform of a type other than IDTX whose end of block exceeds
//! [`THRESHOLD`] and which has at least [`THRESHOLD`] nonzero levels after
//! DC does not code its DC level's parity: the decoder takes it from the
//! sum of those levels' base and low-range parts. The DC level is coded as
//! a four-symbol index (`level >> 1`, 0..=3) and, from index 3, a high-range
//! remainder in steps of two; its sign uses the hidden group of the DC sign
//! models. The encoder makes every such block's DC parity match
//! ([`conform`]). See `THIRD_PARTY.md` for reference terms.
use super::levels::{self, Neighbours};
use super::symbols::SymbolCoder;
use super::{Block, LevelCoding, PLANE_Y, TxSize, TxType, models};
use crate::transform;
use nc_common::decode::Result;

/// AVM `PHTHRESH`: the end of block must exceed it, and the levels after DC
/// must count at least this many nonzero ones.
const THRESHOLD: usize = 4;
/// Largest hidden DC index symbol (AVM `MAX_BASE_BR_RANGE` cap of the
/// four-symbol `coeff_base_ph_cdf`).
const MAX_INDEX: u32 = 3;
/// Hidden DC levels from here code a high-range remainder (AVM
/// `(NUM_BASE_LEVELS + 1) << 1`).
const HIGH_RANGE_START: u32 = 2 * MAX_INDEX;
/// Each level after DC enters the parity sum capped here (AVM
/// `MAX_BASE_BR_RANGE`, 6: below the low-frequency region's saturation of
/// 8, so those levels' 7 and 8 both count as 6).
const SUM_CAP: u32 = 6;

/// Whether a block of `block`'s frame and plane, of type `tx_type` and end
/// of block `eob`, may hide its DC parity.
pub(super) fn applies(block: Block, tx_type: TxType, eob: usize) -> bool {
    block.levels == LevelCoding::ParityHiding
        && block.ctx.plane == PLANE_Y
        && !block.lossless
        && tx_type != TxType::Idtx
        && eob > THRESHOLD
}

/// The base and low-range levels coded after DC: their nonzero count and
/// capped sum (AVM `num_nz`, `sum_abs1`).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Tally {
    nonzero: usize,
    sum: u32,
}

impl Tally {
    /// Count a base and low-range level after DC.
    pub(super) fn add(&mut self, level: u32) {
        self.nonzero += usize::from(level != 0);
        self.sum += level.min(SUM_CAP);
    }

    /// Enough nonzero levels to hide the DC parity.
    pub(super) fn hides(self) -> bool {
        self.nonzero >= THRESHOLD
    }

    /// The hidden DC parity.
    fn parity(self) -> u32 {
        self.sum & 1
    }
}

/// Code a hidden-parity DC level index (the writer's from `target`, whose
/// parity must be the tally's) and return the level it gives,
/// `2 · index + parity`.
pub(super) fn code_hidden_level<C: SymbolCoder>(
    coder: &mut C,
    block: Block,
    neighbours: Neighbours,
    target: u32,
    tally: Tally,
) -> Result<u32> {
    let context = levels::hidden_dc_context(neighbours);
    let symbol = (target >> 1).min(MAX_INDEX) as usize;
    let index = models::hidden_level(coder, symbol, block, context)? as u32;
    Ok(2 * index + tally.parity())
}

/// The hidden DC level after its high range: `coded` plus twice the coded
/// remainder (the writer's from `target`) once the index reaches its last
/// symbol (AVM: `read_adaptive_hr` in the context `hr_level_avg >> 1`).
pub(super) fn code_hidden_high_range<C: SymbolCoder>(
    coder: &mut C,
    coded: u32,
    target: u32,
    average: u32,
) -> Result<u32> {
    if coded < HIGH_RANGE_START {
        return Ok(coded);
    }
    let remainder = target.saturating_sub(coded) / 2;
    Ok(coded + 2 * coder.code_high_range(remainder, average >> 1)?)
}

/// The AC levels eligible to change parity have a small fixed alphabet.
const AC_LEVELS: usize = SUM_CAP as usize + 1;
const QUANTIZERS: usize = 1 << u8::BITS;

const fn ac_dequantizers(size: TxSize) -> [[i16; AC_LEVELS]; QUANTIZERS] {
    let mut table = [[0; AC_LEVELS]; QUANTIZERS];
    let mut q = 0;
    while q < table.len() {
        let mut level = 1;
        while level < AC_LEVELS {
            table[q][level] = transform::dequantized_magnitude(size, q as u8, level as u32) as i16;
            level += 1;
        }
        q += 1;
    }
    table
}

static AC_DEQUANT: [[[i16; AC_LEVELS]; QUANTIZERS]; 3] = [
    ac_dequantizers(TxSize::Tx16),
    ac_dequantizers(TxSize::Tx32),
    ac_dequantizers(TxSize::Tx64),
];

/// The rate model charges 2.5 extra bits for a level that becomes nonzero.
const CROSS_ZERO_BITS: f64 = 3.5;

type Move = (f64, usize, u32);

#[inline]
fn retain_move(best: &mut Option<Move>, added: f64, position: usize, level: u32) {
    // Positions and directions are visited in the original scan order.
    if best.is_none_or(|(least, ..)| added < least) {
        *best = Some((added, position, level));
    }
}

/// Count and sum a dense block's AC levels in raster order. Levels after
/// the scan's end are zero. The 20-bit levels are clipped to six and the
/// block contains at most 1024 of them, so both accumulators fit u32.
fn raster_tally(levels: &[i32]) -> Tally {
    let mut nonzero = 0_u32;
    let mut sum = 0;
    for &level in &levels[1..] {
        let level = level.unsigned_abs();
        nonzero += u32::from(level != 0);
        sum += level.min(SUM_CAP);
    }
    Tally {
        nonzero: nonzero as usize,
        sum,
    }
}

/// Make the quantized luma `levels` of a `tx` transform of type `tx_type`
/// with end of block `eob` decodable under parity hiding, returning the
/// end of block after it. The input `eob` ends at the last nonzero level
/// in scan order. Where the block hides its DC parity and the DC
/// level disagrees, one level within the end of block moves by one, the
/// move of least added squared error plus `lambda` (sample SSE per bit)
/// times its approximate rate (AVM's
/// `parity_hiding_trellis_off` moves toward the coefficient by least
/// error, which spends rate on raising levels): a DC move fixes the DC
/// parity, any other move the hidden parity, since every sum term below
/// [`SUM_CAP`] changes by one. A level at the cap may only move down and
/// one above it not at all (AVM's `tunable`). A move that drops the block
/// below the hiding condition leaves nothing to fix. `qindices` are the DC
/// and AC quantizers of the unquantized `coefficients`.
pub(crate) fn conform(
    (levels, coefficients): (&mut [i32], &[i32]),
    tx: TxSize,
    (tx_type, eob): (TxType, usize),
    lambda: f64,
    qindices: (u8, u8),
) -> usize {
    if tx_type == TxType::Idtx || eob <= THRESHOLD {
        return eob;
    }
    // Every luma region saturates at or above the sum's cap, so a level's
    // capped partial value is its capped magnitude.
    let positions = &levels::scan_positions(tx)[..eob];
    // At half the coded area or more, the contiguous pass can process
    // several levels per vector. Sparse blocks visit only their scan prefix.
    let tally = if eob >= levels.len() / 2 {
        raster_tally(levels)
    } else {
        let mut tally = Tally::default();
        for at in &positions[1..] {
            tally.add(levels[at.pos()].unsigned_abs());
        }
        tally
    };
    if !tally.hides() || levels[0].unsigned_abs() & 1 == tally.parity() {
        return eob;
    }
    let error_scale = transform::coefficient_error_scale(tx);
    let cross_zero_cost = lambda * CROSS_ZERO_BITS;
    let mut best = None;
    let dc_step = transform::dequantized_magnitude(tx, qindices.0, 1);
    let (magnitude, level) = (coefficients[0].unsigned_abs(), levels[0].unsigned_abs());
    if level != 0 || 2 * i64::from(magnitude) >= i64::from(dc_step) {
        let step = f64::from(dc_step);
        let error = f64::from(magnitude)
            - f64::from(transform::dequantized_magnitude(tx, qindices.0, level));
        let shift_error = (2.0 * step) * error;
        if level != 0 {
            let rate = if level == 1 {
                -cross_zero_cost
            } else {
                -lambda
            };
            let added = (step * step + shift_error) * error_scale + rate;
            retain_move(&mut best, added, 0, level - 1);
        }
        let rate = if level == 0 { cross_zero_cost } else { lambda };
        let added = (step * step - shift_error) * error_scale + rate;
        retain_move(&mut best, added, 0, level + 1);
    }

    // The transform shift and quantizer are fixed across the AC scan.
    let shift = transform::dequant_shift(tx) as usize;
    let dequant = &AC_DEQUANT[shift][usize::from(qindices.1)];
    let first_level = i32::from(dequant[1]);
    let step = f64::from(first_level);
    let step_squared = step * step;
    let twice_step = 2.0 * step;
    for at in &positions[1..] {
        let pos = at.pos();
        let level = levels[pos].unsigned_abs();
        let Some(&reconstructed) = dequant.get(level as usize) else {
            continue;
        };
        let magnitude = coefficients[pos].unsigned_abs();
        if level == 0 && 2 * i64::from(magnitude) < i64::from(first_level) {
            continue;
        }
        let error = f64::from(magnitude) - f64::from(reconstructed);
        let shift_error = twice_step * error;
        if level != 0 {
            let rate = if level == 1 {
                -cross_zero_cost
            } else {
                -lambda
            };
            let added = (step_squared + shift_error) * error_scale + rate;
            retain_move(&mut best, added, pos, level - 1);
        }
        if level < SUM_CAP {
            let rate = if level == 0 { cross_zero_cost } else { lambda };
            let added = (step_squared - shift_error) * error_scale + rate;
            retain_move(&mut best, added, pos, level + 1);
        }
    }
    let Some((_, pos, moved)) = best else {
        return eob;
    };
    let negative = coefficients[pos] < 0 || coefficients[pos] == 0 && levels[pos] < 0;
    levels[pos] = if negative {
        -(moved as i32)
    } else {
        moved as i32
    };
    if moved != 0 || pos != positions[eob - 1].pos() {
        return eob;
    }
    positions[..eob - 1]
        .iter()
        .rposition(|at| levels[at.pos()] != 0)
        .map_or(0, |index| index + 1)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
    use super::*;
    use crate::coeff::levels::Region;

    /// The raster tally in [`conform`] relies on every luma region
    /// saturating at or above the sum's cap.
    #[test]
    fn luma_regions_saturate_above_the_sum_cap() {
        for region in [Region::LumaLow, Region::LumaDefault] {
            assert!(region.saturation() >= SUM_CAP);
        }
    }

    /// A disagreeing block conforms: afterwards the DC parity is the
    /// tally's, or the block no longer hides it.
    #[test]
    fn conform_fixes_every_disagreeing_block() {
        let tx = TxSize::Tx8;
        let mut seed = 7_u32;
        for _ in 0..500 {
            let mut coefficients = vec![0_i32; tx.coded_len()];
            for c in &mut coefficients {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *c = ((seed >> 16) % 2001) as i32 - 1000;
            }
            let mut levels: Vec<i32> = coefficients.iter().map(|c| c / 200).collect();
            let eob = super::super::end_of_block(&nc_dsp::Dsp::new(), &levels, tx);
            let eob = conform(
                (&mut levels, &coefficients),
                tx,
                (TxType::DctDct, eob),
                10.0,
                (100, 100),
            );
            assert_eq!(
                eob,
                super::super::end_of_block(&nc_dsp::Dsp::new(), &levels, tx)
            );
            let mut tally = Tally::default();
            for &level in &levels[1..] {
                tally.add(level.unsigned_abs());
            }
            assert!(
                eob <= THRESHOLD
                    || !tally.hides()
                    || levels[0].unsigned_abs() & 1 == tally.parity()
            );
        }
    }
}
