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

/// Approximate bits of a level of magnitude `level`, for choosing a
/// parity fix: a nonzero level pays its significance and sign, then about
/// a bit per unit of magnitude.
fn level_bits(level: u32) -> f64 {
    /// Significance and sign of a nonzero level.
    const NONZERO_BITS: f64 = 2.5;
    if level == 0 {
        0.0
    } else {
        NONZERO_BITS + f64::from(level)
    }
}

/// Make the quantized luma `levels` of a `tx` transform of type `tx_type`
/// with end of block `eob` decodable under parity hiding, returning the
/// end of block after it. Where the block hides its DC parity and the DC
/// level disagrees, one level within the end of block moves by one, the
/// move of least added squared error plus `lambda` (sample SSE per bit)
/// times its approximate rate ([`level_bits`]; AVM's
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
    let mut tally = Tally::default();
    for at in &positions[1..] {
        tally.add(levels[at.pos()].unsigned_abs());
    }
    if !tally.hides() || levels[0].unsigned_abs() & 1 == tally.parity() {
        return eob;
    }
    let error_scale = transform::coefficient_error_scale(tx);
    let first_level = [qindices.0, qindices.1].map(|q| transform::dequantized_magnitude(tx, q, 1));
    // (added cost, position, new magnitude)
    let mut best: Option<(f64, usize, u32)> = None;
    for (index, at) in positions.iter().enumerate() {
        let pos = at.pos();
        let (magnitude, level) = (coefficients[pos].unsigned_abs(), levels[pos].unsigned_abs());
        let is_ac = usize::from(index != 0);
        // Speed: a zero level whose coefficient is under half the first
        // level's reconstruction gains nothing by rising (more error and
        // more rate), and zeros are most positions; skipping them only
        // narrows the choice. Crowd 720p key frame: conform's work about
        // halves.
        if level == 0 && 2 * i64::from(magnitude) < i64::from(first_level[is_ac]) {
            continue;
        }
        let qindex = if is_ac == 0 { qindices.0 } else { qindices.1 };
        // A move by one changes the reconstruction by about one first-level
        // step `s`: from error `e` the squared error changes by
        // `s² ∓ 2·s·e` (one dequantization per position instead of three).
        let step = f64::from(first_level[is_ac]);
        let error =
            f64::from(magnitude) - f64::from(transform::dequantized_magnitude(tx, qindex, level));
        for (moved, direction) in [(level.checked_sub(1), -1.0), (Some(level + 1), 1.0)] {
            let Some(moved) = moved else { continue };
            // A level after DC flips the hidden parity only if its capped
            // value changes.
            if is_ac == 1 && moved.min(SUM_CAP) == level.min(SUM_CAP) {
                continue;
            }
            let added = (step * step - 2.0 * direction * step * error) * error_scale
                + lambda * (level_bits(moved) - level_bits(level));
            if best.is_none_or(|(least, ..)| added < least) {
                best = Some((added, pos, moved));
            }
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
    positions
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
            let eob = super::super::end_of_block(&levels, tx);
            let eob = conform(
                (&mut levels, &coefficients),
                tx,
                (TxType::DctDct, eob),
                10.0,
                (100, 100),
            );
            assert_eq!(eob, super::super::end_of_block(&levels, tx));
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
