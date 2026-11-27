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
use nc_encoder::{Candidates, CoeffRateModel, MAX_CANDIDATES, PackedTrellisDiagonal, Trellis};

use super::{
    BASE_EOB_SYMBOLS, BR_CDF_SIZE, COEFF_BASE_RANGE, CoeffCdfs, EOB_GROUP_START, EOB_OFFSET_BITS,
    MAX_BASE_BR_RANGE, NUM_BASE_LEVELS, PlaneType, TxbContext, TypeSyntax, base_context,
    base_eob_context, eob_group, scan, trellis_positions,
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
    /// levels and their end of block. Each coefficient's candidates sit around its level rounded
    /// to the nearest. The search spans the plain end of block only: a
    /// coefficient past it is under the plain rounding threshold, and coding
    /// it as the new last level rarely pays for the longer end of block,
    /// while scanning to the round-to-nearest end of block cost as much
    /// again on large transforms.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &mut self,
        model: &mut RateModel<'_>,
        coeffs: &[i16],
        quantizer: &Av1Quantizer,
        lambda: f32,
        max_states: usize,
        levels: &mut [i16],
        eob: usize,
    ) -> usize {
        let scan = &scan(model.tx)[..eob];
        let log_scale = quantizer.log_scale;
        // Coefficient-domain squared error to sample SSE.
        let to_sse = 1.0 / f64::from(1_u32 << (COEFFICIENT_GAIN_LOG2 - 2 * log_scale));
        model.beam_rates = None;
        self.candidates.clear();
        self.candidates.extend(scan.iter().map(|&p| {
            let pos = usize::from(p);
            let step = i64::from(if pos == 0 {
                quantizer.dc_step
            } else {
                quantizer.ac_step
            });
            let magnitude = i64::from(coeffs[pos].unsigned_abs());
            let plain = u32::from(levels[pos].unsigned_abs());
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
        // The search writes zeros beyond its chosen end of block.
        for (&p, &level) in scan.iter().zip(&self.chosen) {
            let pos = usize::from(p);
            // Candidates never exceed the nearest level, which the capped
            // `max_level` keeps within 16 bits (`QuantSteps::quantizer`).
            let level = level as i16;
            levels[pos] = if coeffs[pos] < 0 { -level } else { level };
        }
        coded
    }
}

/// Diagonals the context state holds: the current one and the two its
/// contexts read.
const STATE_DIAGONALS: usize = 3;

/// The scan's geometry and context offsets, prepared with the encoder's
/// magnitude walk. Slot 0 is the current anti-diagonal's storage.
#[derive(Debug, Clone, Copy)]
pub(super) struct Position {
    slots: [u8; STATE_DIAGONALS],
    starts_diagonal: bool,
    shift: u8,
    base_offset: u8,
    range_offset: u8,
    eob_context: u8,
}

impl Position {
    pub(super) const DC: Self = Self::new(0, 16, 0, 1, 2, 0);

    pub(super) const fn new(
        scan: usize,
        area: usize,
        pos: usize,
        previous_pos: usize,
        height: u32,
        base_offset: u8,
    ) -> Self {
        let row = pos & ((1 << height) - 1);
        let col = pos >> height;
        let diagonal = row + col;
        let previous_diagonal = (previous_pos & ((1 << height) - 1)) + (previous_pos >> height);
        let slots = [
            (diagonal % STATE_DIAGONALS) as u8,
            ((diagonal + 1) % STATE_DIAGONALS) as u8,
            ((diagonal + 2) % STATE_DIAGONALS) as u8,
        ];
        let range_offset = if pos == 0 {
            0
        } else if row < 2 && col < 2 {
            7
        } else {
            14
        };
        Self {
            slots,
            starts_diagonal: previous_diagonal != diagonal,
            shift: (row * 4) as u8,
            base_offset,
            range_offset,
            eob_context: base_eob_context(scan, area) as u8,
        }
    }
}

/// Saturated magnitudes (at `MAX_BASE_BR_RANGE = 15`, 4 bits per row) of the
/// last [`STATE_DIAGONALS`] anti-diagonals, packed 32 rows into one `u128`
/// per diagonal. Shifting right by `row * 4` naturally zero-extends past row
/// 31, avoids `MAP_STRIDE` padding, halves state copies, and replaces the
/// 108-byte `memcmp` in `Trellis::select_survivors` with three integer
/// comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextState {
    /// Diagonal `d` lives in slot `d % STATE_DIAGONALS`; row `r` (`0..32`)
    /// occupies bits `4 * r .. 4 * r + 4`.
    magnitudes: [PackedTrellisDiagonal; STATE_DIAGONALS],
}

const _: () = assert!(MAX_BASE_BR_RANGE <= 15);

impl ContextState {
    /// `(coeff_base sum, coeff_br sum)` of the coefficient at `row` on
    /// `diagonal`, as `LevelMap::base_sum` and
    /// `LevelMap::range_sum`.
    #[inline]
    fn neighbour_sums(&self, at: Position) -> (u32, u32) {
        const _: () = assert!(super::BASE_CONTEXT_LEVEL_MAX == 3);
        let next = self.magnitudes[usize::from(at.slots[1])].window(at.shift);
        let after = self.magnitudes[usize::from(at.slots[2])].window(at.shift);
        // (r, c+1), (r+1, c) on the next diagonal; (r+1, c+1) on the one
        // after; and the farther (r, c+2), (r+2, c) read by `coeff_base`.
        let next = next & 0xFF;
        let five = next | ((after & 0xFFF) << 8);
        // A nibble with either upper bit set saturates to 3. A multiplier
        // with one bit in each nibble sums these five capped magnitudes
        // into the fifth nibble without any carry (5 * 3 <= 15).
        let above_three = ((five >> 2) | (five >> 3)) & 0x11111;
        let saturated = (five | (above_three * 3)) & 0x33333;
        let base = (saturated.wrapping_mul(0x11111) >> 16) & 15;
        let br = (next & 15) + (next >> 4) + ((after >> 4) & 15);
        (base, br)
    }

    /// Enter the coefficient's diagonal, clearing the one that fell out
    /// of the context window, and return the current slot.
    #[inline]
    fn enter(&mut self, at: Position) -> &mut PackedTrellisDiagonal {
        let slot = &mut self.magnitudes[usize::from(at.slots[0])];
        // Paths visit every position after starting, and a new path
        // starts with all slots empty even if it starts mid-diagonal.
        if at.starts_diagonal {
            *slot = PackedTrellisDiagonal::EMPTY;
        }
        slot
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
    /// `dc_sign` cost of the DC coefficient's sign (context coded), in
    /// 1/512 bit: the same for every candidate, so priced once.
    dc_sign_q9: u32,
    context: usize,
    positions: &'static [Position],
    /// Rates for the most recently priced path context within this block.
    beam_rates: Option<(u32, [f32; MAX_CANDIDATES])>,
}

impl<'a> RateModel<'a> {
    /// The model of a block coded with `cdfs` as they stand; the other
    /// arguments are those `CoeffCdfs::write` will get. `dc_negative` is the
    /// sign of the unquantised DC coefficient.
    pub fn new(
        cdfs: &'a mut CoeffCdfs,
        tx: TxSize,
        plane: PlaneType,
        syntax: TypeSyntax,
        ctx: TxbContext,
        dc_negative: bool,
    ) -> Self {
        let mut counter = BitCounter::default();
        let txs = tx.context();
        counter.add_q9(cdfs.txb_skip[txs][ctx.skip].cost_q9(0));
        if plane == PlaneType::Luma {
            cdfs.write_tx_type(&mut counter, tx, syntax);
        }
        let dc_sign_q9 =
            cdfs.dc_sign[plane as usize][ctx.dc_sign].cost_q9(usize::from(dc_negative));
        Self {
            cdfs,
            tx,
            plane,
            ctx,
            coded_block_bits: counter.bits(),
            dc_sign_q9,
            context: txs,
            positions: trellis_positions(tx),
            beam_rates: None,
        }
    }

    /// `coeff_br` symbols of `level` (> `NUM_BASE_LEVELS`), in 1/512 bit.
    fn range_cost_q9(&mut self, br_ctx: usize, level: u32) -> u32 {
        let cdf = &mut self.cdfs.br[self.context.min(3)][self.plane as usize][br_ctx];
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

    #[inline]
    fn contexts(&self, state: &ContextState, scan_pos: usize) -> (usize, usize) {
        let at = self.positions[scan_pos];
        let (base_sum, br_sum) = state.neighbour_sums(at);
        let base = base_context(scan_pos, base_sum, at.base_offset);
        let br = (((br_sum + 1) >> 1).min(6) as usize) + usize::from(at.range_offset);
        (base, br)
    }

    fn price_levels(
        &mut self,
        scan_pos: usize,
        base_ctx: usize,
        br: usize,
        levels: &[u32],
        bits: &mut [f32],
    ) {
        let (txs, pt) = (self.context, self.plane as usize);
        let dc = scan_pos == 0;
        for (i, &level) in levels.iter().enumerate() {
            let base_symbol = level.min(3) as usize;
            let mut cost = self.cdfs.base[txs][pt][base_ctx].cost_q9(base_symbol);
            if level == 0 {
                bits[i] = q9_to_bits(cost);
                continue;
            }
            if level as i32 > NUM_BASE_LEVELS {
                cost += self.range_cost_q9(br, level);
            }
            cost += if dc {
                self.dc_sign_q9
            } else {
                COST_UNITS_PER_BIT
            };
            bits[i] = q9_to_bits(cost) + golomb_bits(level) as f32;
        }
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
            magnitudes: [PackedTrellisDiagonal::EMPTY; STATE_DIAGONALS],
        }
    }

    /// `all_zero` = 0, the luma transform type and the end of block
    /// (`eob_pt_*`, `eob_extra` and its literal bits, priced as
    /// `CoeffCdfs::write_eob` codes them).
    fn eob_rate(&mut self, scan_pos: usize) -> f32 {
        let eob = scan_pos + 1;
        let eob_pt = eob_group(eob);
        let pt = self.plane as usize;
        let symbol = eob_pt - 1;
        // `eob_multi_ctx` is 0 for transform class 2D.
        let mut cost = self.cdfs.eob_pt_cost_q9(self.tx, pt, symbol);
        let bits = EOB_OFFSET_BITS[eob_pt];
        if bits > 0 {
            let top = ((eob - usize::from(EOB_GROUP_START[eob_pt])) >> (bits - 1)) & 1;
            cost += self.cdfs.eob_extra[self.context][pt][eob_pt - 3].cost_q9(top);
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
        let (base, br) = self.contexts(state, scan_pos);
        self.price_levels(scan_pos, base, br, levels, bits);
    }

    fn beam_level_rates(
        &mut self,
        state: &ContextState,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32; MAX_CANDIDATES],
    ) {
        const _: () = assert!(super::SIG_COEF_CONTEXTS <= 64 && super::LEVEL_CONTEXTS <= 32);
        let (base, br) = self.contexts(state, scan_pos);
        let key = ((scan_pos as u32) << 11) | ((base as u32) << 5) | br as u32;
        if let Some((cached_key, cached)) = self.beam_rates
            && cached_key == key
        {
            *bits = cached;
        } else {
            self.price_levels(scan_pos, base, br, levels, bits);
            self.beam_rates = Some((key, *bits));
        }
    }

    fn last_level_rates(
        &mut self,
        _initial: &ContextState,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32],
    ) {
        let at = self.positions[scan_pos];
        let pt = self.plane as usize;
        let sign = if scan_pos == 0 {
            self.dc_sign_q9
        } else {
            COST_UNITS_PER_BIT
        };
        for (&level, bits) in levels.iter().zip(bits) {
            let symbol = level.min(3) as usize;
            debug_assert!((1..=BASE_EOB_SYMBOLS).contains(&symbol));
            let mut cost = self.cdfs.base_eob[self.context][pt][usize::from(at.eob_context)]
                .cost_q9(symbol - 1);
            if level as i32 > NUM_BASE_LEVELS {
                cost += self.range_cost_q9(usize::from(at.range_offset), level);
            }
            *bits = q9_to_bits(cost + sign) + golomb_bits(level) as f32;
        }
    }

    /// [`Self::level_rates`] of a zero (only its `coeff_base` symbol, under
    /// the survivor's context: a survivor has coded a level, so its state is
    /// never the initial one), then [`Self::advance`].
    fn zero_step(&mut self, state: &mut ContextState, scan_pos: usize) -> f32 {
        let at = self.positions[scan_pos];
        let (base_sum, _) = state.neighbour_sums(at);
        let ctx = base_context(scan_pos, base_sum, at.base_offset);
        let cost = self.cdfs.base[self.context][self.plane as usize][ctx].cost_q9(0);
        // A scan visits a row only once per diagonal. This nibble is
        // already zero, including after a slot is cleared on entry.
        state.enter(at);
        q9_to_bits(cost)
    }

    fn advance(&self, state: &mut ContextState, scan_pos: usize, level: u32) {
        let at = self.positions[scan_pos];
        state
            .enter(at)
            .insert(at.shift, level.min(MAX_BASE_BR_RANGE as u32));
    }

    fn empty_rate(&mut self) -> f32 {
        self.cdfs.txb_skip[self.context][self.ctx.skip].bits(1)
    }
}

/// The shared rate-aware trimming (`nc_encoder::trim_levels`) of a block's
/// plain `levels` (end of block `eob`) from its forward coefficients
/// `coeffs` (libaom layout), at `lambda` (sample SSE per bit), with
/// distortion in coefficient units as [`TrellisQuantizer::quantize`] prices
/// it. Returns the new end of block.
pub(crate) fn rate_trim(
    trim: &nc_encoder::RateTrim,
    lambda: f64,
    tx: TxSize,
    quantizer: &Av1Quantizer,
    coeffs: &[i16],
    levels: &mut [i16],
    eob: usize,
) -> usize {
    let log_scale = quantizer.log_scale;
    let to_sse = 1.0 / f64::from(1_u32 << (COEFFICIENT_GAIN_LOG2 - 2 * log_scale));
    nc_encoder::trim_levels(trim, levels, scan(tx), eob, lambda, |pos, magnitude| {
        let step = i64::from(if pos == 0 {
            quantizer.dc_step
        } else {
            quantizer.ac_step
        });
        let dequantized = (i64::from(magnitude) * step) >> log_scale;
        nc_encoder::dequantized_removal_sse(
            i64::from(coeffs[pos].unsigned_abs()),
            dequantized,
            to_sse,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 2:1 rectangles (the coder's other shapes).
    fn rectangles() -> Vec<TxSize> {
        (2..=6_u8)
            .flat_map(|w| [w + 1, w.wrapping_sub(1)].map(|h| TxSize::new(w, h)))
            .flatten()
            .collect()
    }

    const SIZES: [TxSize; 5] = [
        TxSize::TX_4X4,
        TxSize::TX_8X8,
        TxSize::TX_16X16,
        TxSize::TX_32X32,
        TxSize::TX_64X64,
    ];

    /// The state is only exact if scans visit anti-diagonals in order.
    #[test]
    fn default_scans_visit_anti_diagonals_in_order() {
        for tx in SIZES.into_iter().chain(rectangles()) {
            let n = 1 << tx.coded_height_log2();
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
        for tx in SIZES.into_iter().chain(rectangles()) {
            for plane in [PlaneType::Luma, PlaneType::Chroma] {
                let area = tx.coded_area();
                let mut levels = vec![0_i16; area];
                for &p in &scan(tx)[..area.min(40)] {
                    let v = match next() % 8 {
                        0..=3 => 0,
                        4 | 5 => 1,
                        6 => 3 + (next() % 20) as i16,
                        _ => 2,
                    };
                    levels[usize::from(p)] = if next() % 2 == 0 { v } else { -v };
                }
                levels[usize::from(scan(tx)[5])] = 4;
                let eob = super::super::end_of_block(tx, &levels);
                let ctx = TxbContext::derive(plane, true, &[0x9], &[0x3]);
                let mut cdfs = CoeffCdfs::new(1);
                let expected = cdfs.cost(tx, plane, TypeSyntax::Inter, ctx, &levels, eob);
                let mut model =
                    RateModel::new(&mut cdfs, tx, plane, TypeSyntax::Inter, ctx, levels[0] < 0);
                let mut bits = model.eob_rate(eob - 1);
                let mut state = model.initial_state();
                for c in (0..eob).rev() {
                    let level = u32::from(levels[usize::from(scan(tx)[c])].unsigned_abs());
                    let mut level_bits = [0.0];
                    if c == eob - 1 {
                        model.last_level_rates(&state, c, &[level], &mut level_bits);
                    } else {
                        model.level_rates(&state, c, &[level], &mut level_bits);
                    }
                    bits += level_bits[0];
                    model.advance(&mut state, c, level);
                }
                assert!(
                    (bits - expected).abs() < 0.05,
                    "{tx:?} {plane:?}: {bits} vs {expected}"
                );
                let empty = cdfs.cost(tx, plane, TypeSyntax::Inter, ctx, &levels, 0);
                let mut model = RateModel::new(&mut cdfs, tx, plane, TypeSyntax::Inter, ctx, false);
                assert!((model.empty_rate() - empty).abs() < 0.01);
            }
        }
    }
}
