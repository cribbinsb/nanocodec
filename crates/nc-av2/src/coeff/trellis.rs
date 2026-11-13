//! The AV2 coefficient rate model of the shared trellis
//! ([`nc_encoder::CoeffRateModel`]): prices levels with the tile's current
//! probabilities under the exact contexts of [`super::write_typed`] (AVM
//! v1.0.0 `av2_write_coeffs_txb`; contexts `txb_common.h`
//! `get_lower_levels_ctx_*` and `get_br_ctx_*`).
//!
//! Levels are coded in reverse scan order and every base and low-range
//! context reads only the template neighbours to the right and below
//! (`levels::NEIGHBOURS`: `(r, c+1)`, `(r+1, c)`, `(r+1, c+1)`, `(r, c+2)`,
//! `(r+2, c)`), which lie on the next two anti-diagonals. AV2's default
//! scans visit anti-diagonals in order (tested in `levels`), so the state
//! the trellis carries is the levels of the last three diagonals
//! ([`ContextState`]): the two the current diagonal reads and the current
//! one, which the next diagonal reads. Levels are saturated at the largest
//! neighbour clip (`levels::CONTEXT_LEVEL_CAP`), so the state is exact for
//! every context while paths that differ only in levels no context can
//! tell apart merge.
//!
//! Symbols are priced through the coder's own model selection
//! ([`super::symbols::Pricer`] over the shared walk's functions), at the
//! tile's probabilities as they stand: adaptation within the block is not
//! modelled, as in nc-av1's trellis. Since the probabilities do not move
//! during a search, each base and low-range model is priced once per block
//! and context, for its whole alphabet ([`CostCache`]). One approximation remains: signs and
//! high-range remainders are coded in a second, forward pass whose adaptive
//! Rice parameter follows the remainders before them (AVM
//! `get_adaptive_param`). The reverse search cannot know those, so every
//! remainder is priced at the starting parameter (average zero), which is
//! exact for the first one and for blocks with at most one level past its
//! region's saturation (nearly all at the low rates we target).

use nc_encoder::symbol_cost::COST_UNITS_PER_BIT;
use nc_encoder::{Candidates, CoeffRateModel, Trellis};

use super::levels::{
    self, BASE_CONTEXTS, CONTEXT_LEVEL_CAP, LOW_RANGE_CONTEXTS, MAP_STRIDE, MAX_BASE_SYMBOLS,
    MAX_LOW_RANGE, NEIGHBOURS, Neighbours, Region, ScanPosition,
};
use super::models::BaseFamily;
use super::symbols::{Pricer, high_range_bits};
use super::tcq::MAX_CANDIDATES as MAX_TCQ_LEVELS;
use super::{
    Block, Context, Kind, MAX_LEVEL, PLANE_Y, TxSize, TxType, code_eob, code_tx_type, models,
};
use crate::entropy::Writer;
use crate::transform::{self, Qindices};

/// Diagonals the context state holds: the current one and the two its
/// contexts read.
const STATE_DIAGONALS: usize = 3;
/// The [`ContextState`] slots of the two diagonals after each diagonal
/// (`(d + 1) % 3`, `(d + 2) % 3`), looked up rather than divided.
const SLOT_AFTER: [[usize; 2]; 2 * MAP_STRIDE] = {
    let mut slots = [[0; 2]; 2 * MAP_STRIDE];
    let mut d = 0;
    while d < slots.len() {
        slots[d] = [(d + 1) % STATE_DIAGONALS, (d + 2) % STATE_DIAGONALS];
        d += 1;
    }
    slots
};
/// `diagonal` of a state in which no level is coded yet.
const NO_DIAGONAL: u16 = u16::MAX;
/// The running high-range average before the first remainder (AVM
/// `av2_write_coeffs_txb` starts it at zero).
const FIRST_HIGH_RANGE_AVERAGE: u32 = 0;

// Every neighbour lies on one of the two diagonals after its coefficient's,
// in the order `ContextState::neighbours` reads them.
const _: () = {
    assert!(matches!(
        NEIGHBOURS,
        [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
    ));
    let mut i = 0;
    while i < NEIGHBOURS.len() {
        assert!(NEIGHBOURS[i].0 + NEIGHBOURS[i].1 < STATE_DIAGONALS);
        i += 1;
    }
};

/// One transform block as it will be coded: its size, contexts, quantizer
/// and transform type (the arguments [`super::write_typed`] will get).
#[derive(Debug, Clone, Copy)]
pub(crate) struct CodedBlock {
    pub tx: TxSize,
    pub context: Context,
    /// The frame's base qindex, which selects the coefficient models.
    pub qindex: u8,
    /// The quantizer of the block's plane, which sets the dequantized
    /// magnitude of each level.
    pub steps: Qindices,
    pub kind: Kind,
}

/// Trellis quantisation of AV2 transform blocks, with reusable buffers.
#[derive(Debug, Default)]
pub(crate) struct TrellisQuantizer {
    search: Trellis<ContextState>,
    candidates: Vec<Candidates>,
    chosen: Vec<u32>,
    costs: CostCache,
}

impl TrellisQuantizer {
    /// Choose the levels of `block` from its forward coefficients
    /// `coefficients` (the coded region, raster): squared error plus
    /// `lambda` (sample SSE per bit) times its rate through `writer` as it
    /// stands ([`RateModel`]), with at most `max_states` survivors per scan
    /// position. On entry `levels` holds the plain (dead-zone) quantiser's
    /// levels, whose end of block is `eob`; on return the chosen ones, and
    /// the result is their end of block. Each coefficient's candidates sit
    /// around its level rounded to the nearest ([`Candidates::around`]). The
    /// search spans the plain end of block only, as nc-av1's: a coefficient
    /// past it is under the dead-zone threshold, and making it the new last
    /// level rarely pays for the longer end of block.
    pub(crate) fn quantize(
        &mut self,
        writer: &mut Writer,
        block: CodedBlock,
        (lambda, max_states): (f32, usize),
        coefficients: &[i32],
        (levels, eob): (&mut [i32], usize),
    ) -> usize {
        let (tx, steps) = (block.tx, block.steps);
        debug_assert_eq!(eob, super::end_of_block(levels, tx));
        if eob == 0 {
            return 0;
        }
        let scan = &super::scan(tx)[..eob];
        let error_scale = transform::coefficient_error_scale(tx);
        let distortion = |qindex: u8, magnitude: i64, level: u32| {
            let error = magnitude - i64::from(transform::dequantized_magnitude(tx, qindex, level));
            ((error * error) as f64 * error_scale) as f32
        };
        self.candidates.clear();
        self.candidates.extend(scan.iter().map(|&p| {
            let pos = usize::from(p);
            let magnitude = i64::from(coefficients[pos].unsigned_abs());
            let plain = levels[pos].unsigned_abs();
            let qindex = steps.at(pos);
            // The dead zone rounds down by less than one step, so the
            // nearest level is the plain one or the next. Their distortions
            // are kept for the candidates (one of them is the nearest, the
            // other often the level below it).
            let at_plain = distortion(qindex, magnitude, plain);
            let above = (plain < MAX_LEVEL).then(|| distortion(qindex, magnitude, plain + 1));
            let nearest = match above {
                Some(above) if above < at_plain => plain + 1,
                _ => plain,
            };
            Candidates::around(nearest, |level| match above {
                _ if level == plain => at_plain,
                Some(above) if level == plain + 1 => above,
                _ => distortion(qindex, magnitude, level),
            })
        }));
        self.chosen.resize(eob, 0);
        let mut model = RateModel::new(writer, &mut self.costs, block, coefficients[0] < 0);
        let coded = self.search.quantize(
            &mut model,
            &self.candidates,
            lambda,
            max_states,
            &mut self.chosen,
        );
        for (i, &p) in scan.iter().enumerate() {
            let pos = usize::from(p);
            let level = if i < coded { self.chosen[i] as i32 } else { 0 };
            levels[pos] = if coefficients[pos] < 0 { -level } else { level };
        }
        coded
    }
}

/// The levels of the last [`STATE_DIAGONALS`] anti-diagonals by row,
/// saturated at [`CONTEXT_LEVEL_CAP`]. Rows run to [`MAP_STRIDE`] (the
/// largest coded side plus the template's reach), so neighbours below the
/// block read zeros without a test; neighbours right of it lie on rows of a
/// diagonal no coded position fills, which are zero too. Diagonals after
/// the last coded level are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextState {
    /// Anti-diagonal of the last coded level.
    diagonal: u16,
    /// Diagonal `d` lives in slot `d % STATE_DIAGONALS`.
    levels: [[u8; MAP_STRIDE]; STATE_DIAGONALS],
}

impl ContextState {
    /// The template neighbours of `at`, as `LevelMap::neighbours` gives
    /// them to the coder.
    fn neighbours(&self, at: ScanPosition) -> Neighbours {
        let (row, diagonal) = (at.row(), at.diagonal());
        // NEIGHBOURS in order: two on the next diagonal (rows r, r + 1),
        // three on the one after (rows r + 1, r, r + 2); one slice of each
        // bounds-checks them all.
        let next = &self.levels[SLOT_AFTER[diagonal][0]][row..row + 2];
        let after = &self.levels[SLOT_AFTER[diagonal][1]][row..row + 3];
        [next[0], next[1], after[1], after[0], after[2]]
    }
}

/// A TCQ position's candidate levels under one quantizer, as the symbols
/// that code them ([`RateModel::tcq_symbols`]): what pricing them in any
/// path's contexts needs besides the context's model costs.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TcqSymbols {
    /// Whether the levels are the block's last coded one.
    last: bool,
    count: usize,
    /// Each level's base symbol.
    base: [u8; MAX_TCQ_LEVELS],
    /// Each level's low-range increment, if its low part reaches the low
    /// range.
    low_range: [Option<u8>; MAX_TCQ_LEVELS],
    /// Whether any level's does.
    reaches_low_range: bool,
    /// Each level's sign and high-range remainder, in 1/512 bit.
    extra_q9: [u32; MAX_TCQ_LEVELS],
}

/// A [`CachedCosts`] symbol not priced yet in the entry's block.
const UNPRICED: u32 = u32::MAX;

/// The costs of the symbols of one model in one context, in 1/512 bit
/// ([`UNPRICED`] for symbols not priced yet), stamped with the block they
/// were priced for.
#[derive(Debug, Clone, Copy, Default)]
struct CachedCosts {
    block: u32,
    costs: [u32; MAX_BASE_SYMBOLS],
}

/// Base and low-range symbol costs by region and context, priced on first
/// use in each block. Entries stamped with an earlier block are stale, so
/// a new block costs no clearing.
#[derive(Debug)]
pub(super) struct CostCache {
    /// The current block's stamp (0 marks entries never priced).
    block: u32,
    /// By TCQ quantizer set, region, last-coefficient family and base
    /// context.
    base: Vec<CachedCosts>,
    /// By region and low-range context.
    low_range: Vec<CachedCosts>,
}

impl Default for CostCache {
    fn default() -> Self {
        Self {
            block: 0,
            base: vec![CachedCosts::default(); Region::COUNT * 2 * 2 * BASE_CONTEXTS],
            low_range: vec![CachedCosts::default(); Region::COUNT * LOW_RANGE_CONTEXTS],
        }
    }
}

/// Prices one transform block's levels for the trellis.
pub(crate) struct RateModel<'a> {
    pricer: Pricer<'a>,
    costs: &'a mut CostCache,
    block: Block,
    tx_type: TxType,
    /// Cost of a nonzero level's sign, in 1/512 bit: the luma DC sign is
    /// model coded (for the unquantised DC's sign), every other a bypass
    /// bit.
    dc_sign_q9: u32,
    /// The transform's scan positions, looked up once.
    positions: &'static [ScanPosition],
}

impl<'a> RateModel<'a> {
    /// The model of `block` coded through `writer` as it stands, caching
    /// symbol costs in `costs`. `dc_negative` is the sign of the
    /// unquantised DC coefficient.
    pub(super) fn new(
        writer: &'a mut Writer,
        costs: &'a mut CostCache,
        block: CodedBlock,
        dc_negative: bool,
    ) -> Self {
        costs.block = costs
            .block
            .checked_add(1)
            .expect("a tile prices far fewer than 2^32 blocks");
        let mut pricer = Pricer::new(writer);
        let coded = Block::new(
            block.tx,
            block.context,
            block.qindex,
            (block.kind.set, block.kind.levels),
        );
        // The trellis prices DC signs in the plain group: whether the
        // block will hide its parity depends on the levels it chooses.
        let dc_sign_q9 = if block.context.plane == PLANE_Y {
            models::dc_sign(&mut pricer, usize::from(dc_negative), coded, false)
                .expect("pricing codes nothing and cannot fail");
            pricer.take_q9()
        } else {
            COST_UNITS_PER_BIT
        };
        Self {
            pricer,
            costs,
            block: coded,
            tx_type: block.kind.tx_type,
            dc_sign_q9,
            positions: levels::scan_positions(block.tx),
        }
    }

    /// The rate priced since the last call, in bits.
    fn take_bits(&mut self) -> f32 {
        self.pricer.take_q9() as f32 / COST_UNITS_PER_BIT as f32
    }

    /// The cost of base `symbol` in `family`'s model in `context`, priced
    /// on first use in the block (candidates use only a few of a model's
    /// symbols).
    #[allow(clippy::inline_always, reason = "once per trellised candidate")]
    #[inline(always)]
    fn base_cost(&mut self, family: BaseFamily, context: usize, symbol: usize) -> u32 {
        let BaseFamily {
            region,
            last,
            upper,
        } = family;
        // TCQ's second-quantizer entries form a second half, so the plain
        // trellis keeps its compact half (interleaved they cost the default
        // encoder about 0.3% Jetson cycles).
        let key = ((usize::from(upper) * Region::COUNT + region.index()) * 2 + usize::from(last))
            * BASE_CONTEXTS
            + context;
        let entry = &mut self.costs.base[key];
        if entry.block != self.costs.block {
            entry.block = self.costs.block;
            entry.costs = [UNPRICED; MAX_BASE_SYMBOLS];
        }
        if entry.costs[symbol] == UNPRICED {
            models::base(&mut self.pricer, symbol, self.block, family, context)
                .expect("pricing codes nothing and cannot fail");
            entry.costs[symbol] = self.pricer.take_q9();
        }
        entry.costs[symbol]
    }

    /// The symbols of a TCQ position's candidate `levels` at `scan_pos`
    /// under one quantizer ([`super::tcq`]), for pricing in any path's
    /// context ([`Self::tcq_rates`]): the base and low-range symbols of the
    /// low part that keeps the level's parity (the `last` coefficient's
    /// base alphabet omits zero), the sign, and the pair count of the
    /// high-range remainder, priced at the starting Rice parameter as for
    /// plain levels.
    pub(super) fn tcq_symbols(&self, scan_pos: usize, last: bool, levels: &[u32]) -> TcqSymbols {
        let at = self.positions[scan_pos];
        let region = at.region(PLANE_Y);
        let saturation = region.saturation();
        let base_limit = region.base_levels() + 1;
        let sign_q9 = if at.pos() == 0 {
            self.dc_sign_q9
        } else {
            COST_UNITS_PER_BIT
        };
        let offset = u32::from(last);
        let mut symbols = TcqSymbols {
            last,
            count: levels.len(),
            ..TcqSymbols::default()
        };
        for (i, &level) in levels.iter().enumerate() {
            debug_assert!(level >= offset, "the last coded level is nonzero");
            let low = super::tcq::low_part(level, saturation);
            symbols.base[i] = (low.min(base_limit) - offset) as u8;
            if low >= base_limit {
                symbols.low_range[i] = Some((low - base_limit).min(MAX_LOW_RANGE) as u8);
                symbols.reaches_low_range = true;
            }
            if level != 0 {
                let remainder = if super::tcq::has_remainder(low, saturation) {
                    high_range_bits((level - low) / 2, FIRST_HIGH_RANGE_AVERAGE)
                } else {
                    0
                };
                symbols.extra_q9[i] = sign_q9 + remainder * COST_UNITS_PER_BIT;
            }
        }
        symbols
    }

    /// What decides the price of the TCQ levels `symbols` describes at
    /// `scan_pos` (not the last) on the path whose contexts are `state`,
    /// when one number does: levels short of the low range are priced by
    /// their base context alone, which the neighbours' magnitude
    /// ([`levels::luma_base_magnitude`]) sets. Equal keys at a position
    /// give equal [`Self::tcq_rates`].
    pub(super) fn tcq_key(
        &self,
        state: &ContextState,
        scan_pos: usize,
        symbols: &TcqSymbols,
    ) -> Option<usize> {
        debug_assert!(!symbols.last);
        if symbols.reaches_low_range {
            return None;
        }
        let at = self.positions[scan_pos];
        Some(levels::luma_base_magnitude(
            state.neighbours(at),
            at,
            at.region(PLANE_Y),
        ))
    }

    /// [`Self::tcq_rates`] for a path whose price key ([`Self::tcq_key`])
    /// is `key`: the levels' base context follows from it without the
    /// path's neighbours.
    pub(super) fn tcq_rates_by_key(
        &mut self,
        (scan_pos, upper): (usize, bool),
        key: usize,
        symbols: &TcqSymbols,
        bits: &mut [f32],
    ) {
        debug_assert!(!symbols.last && !symbols.reaches_low_range);
        let at = self.positions[scan_pos];
        let region = at.region(PLANE_Y);
        let family = BaseFamily {
            region,
            last: false,
            upper,
        };
        let context = key + levels::luma_base_offset(at, region);
        for (i, out) in bits[..symbols.count].iter_mut().enumerate() {
            let q9 =
                self.base_cost(family, context, usize::from(symbols.base[i])) + symbols.extra_q9[i];
            *out = q9 as f32 / COST_UNITS_PER_BIT as f32;
        }
    }

    /// The bits of the TCQ levels `symbols` describes at `scan_pos` on the
    /// path whose contexts are `state`, under the quantizer `upper` selects
    /// (which also selects the base models, except for the last
    /// coefficient's).
    pub(super) fn tcq_rates(
        &mut self,
        state: &ContextState,
        (scan_pos, upper): (usize, bool),
        symbols: &TcqSymbols,
        bits: &mut [f32],
    ) {
        let at = self.positions[scan_pos];
        let region = at.region(PLANE_Y);
        let last = symbols.last;
        debug_assert_eq!(last, state.diagonal == NO_DIAGONAL);
        let neighbours = state.neighbours(at);
        let base_context = if last {
            levels::last_base_context(scan_pos, self.block.tx)
        } else {
            levels::base_context(neighbours, at, region, PLANE_Y)
        };
        let family = BaseFamily {
            region,
            last,
            upper: upper && !last,
        };
        let low_range = symbols.reaches_low_range.then(|| {
            let context = levels::low_range_context(neighbours, at, region);
            self.low_range_costs(region, context)
        });
        for (i, out) in bits[..symbols.count].iter_mut().enumerate() {
            let mut q9 = self.base_cost(family, base_context, usize::from(symbols.base[i]))
                + symbols.extra_q9[i];
            if let (Some(costs), Some(increment)) = (low_range, symbols.low_range[i]) {
                q9 += costs[usize::from(increment)];
            }
            *out = q9 as f32 / COST_UNITS_PER_BIT as f32;
        }
    }

    /// The low-range increment costs of `region` in `context`.
    fn low_range_costs(&mut self, region: Region, context: usize) -> [u32; MAX_BASE_SYMBOLS] {
        let key = region.index() * LOW_RANGE_CONTEXTS + context;
        let entry = &mut self.costs.low_range[key];
        if entry.block != self.costs.block {
            entry.block = self.costs.block;
            for increment in 0..=MAX_LOW_RANGE as usize {
                models::low_range(&mut self.pricer, increment, self.block.q, region, context)
                    .expect("pricing codes nothing and cannot fail");
                entry.costs[increment] = self.pricer.take_q9();
            }
        }
        entry.costs
    }
}

impl CoeffRateModel for RateModel<'_> {
    type State = ContextState;

    fn initial_state(&self) -> ContextState {
        ContextState {
            diagonal: NO_DIAGONAL,
            levels: [[0; MAP_STRIDE]; STATE_DIAGONALS],
        }
    }

    /// The all-zero flag (0), the end of block and the luma transform type,
    /// which depends on it. A type the block cannot signal at this end of
    /// block (a non-DCT type that a DC-only block infers as DCT) is not a
    /// valid coding: it costs infinitely many bits.
    fn eob_rate(&mut self, scan_pos: usize) -> f32 {
        let eob = scan_pos + 1;
        let block = self.block;
        let priced = models::all_zero(&mut self.pricer, 0, block)
            .and_then(|_| code_eob(&mut self.pricer, eob, block))
            .and_then(|_| code_tx_type(&mut self.pricer, eob, block, self.tx_type));
        let bits = self.take_bits();
        if priced.is_ok() { bits } else { f32::INFINITY }
    }

    fn level_rates(
        &mut self,
        state: &ContextState,
        scan_pos: usize,
        candidates: &[u32],
        bits: &mut [f32],
    ) {
        let at = self.positions[scan_pos];
        let plane = self.block.ctx.plane;
        let region = at.region(plane);
        let last = state.diagonal == NO_DIAGONAL;
        // The contexts are shared by every candidate. The last level's
        // neighbours are all zero, as in the coder.
        let neighbours = state.neighbours(at);
        let base_context = if last {
            levels::last_base_context(scan_pos, self.block.tx)
        } else {
            levels::base_context(neighbours, at, region, plane)
        };
        let family = BaseFamily::plain(region, last);
        let base_limit = region.base_levels() + 1;
        let low_range = (region.has_low_range() && candidates.iter().any(|&l| l >= base_limit))
            .then(|| {
                let context = levels::low_range_context(neighbours, at, region);
                self.low_range_costs(region, context)
            });
        let sign_q9 = if at.pos() == 0 {
            self.dc_sign_q9
        } else {
            COST_UNITS_PER_BIT
        };
        let saturation = region.saturation();
        // The symbols `code_level` codes: the base symbol (the last
        // coefficient's alphabet omits zero), then past the base levels the
        // low-range increment, then the sign and past saturation the
        // high-range remainder.
        let offset = u32::from(last);
        for (&level, out) in candidates.iter().zip(bits) {
            debug_assert!(level >= offset, "the last coded level is nonzero");
            let mut q9 = self.base_cost(
                family,
                base_context,
                (level.min(base_limit) - offset) as usize,
            );
            if let Some(low_range) = low_range.filter(|_| level >= base_limit) {
                q9 += low_range[(level - base_limit).min(MAX_LOW_RANGE) as usize];
            }
            let mut extra_bits = 0;
            if level != 0 {
                q9 += sign_q9;
                if level >= saturation {
                    extra_bits = high_range_bits(level - saturation, FIRST_HIGH_RANGE_AVERAGE);
                }
            }
            *out = (q9 + extra_bits * COST_UNITS_PER_BIT) as f32 / COST_UNITS_PER_BIT as f32;
        }
    }

    fn advance(&self, state: &mut ContextState, scan_pos: usize, level: u32) {
        let at = self.positions[scan_pos];
        let diagonal = at.diagonal();
        let slot = &mut state.levels[diagonal % STATE_DIAGONALS];
        if usize::from(state.diagonal) != diagonal {
            // Entering a new diagonal: its slot held the one three further
            // on, which no remaining context reads.
            slot.fill(0);
            state.diagonal = diagonal as u16;
        }
        slot[at.row()] = level.min(CONTEXT_LEVEL_CAP) as u8;
    }

    fn empty_rate(&mut self) -> f32 {
        models::all_zero(&mut self.pricer, 1, self.block)
            .expect("pricing codes nothing and cannot fail");
        self.take_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{LevelCoding, PLANE_U, PLANE_V, TxSetMode, write_typed};
    use super::*;

    /// Largest relative difference the block's own CDF adaptation makes.
    const ADAPTATION_TOLERANCE: f32 = 0.2;

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0 >> 8
        }
    }

    /// Random levels over a scan prefix, mostly small, with at most one
    /// level past its region's saturation (the remainder pricing is exact
    /// for the first one).
    fn random_levels(rng: &mut Lcg, tx: TxSize, plane: u8) -> Vec<i32> {
        let positions = levels::scan_positions(tx);
        let mut block = vec![0; tx.coded_len()];
        let eob = 1 + rng.next() as usize % positions.len().min(80);
        let mut high = false;
        for &at in &positions[..eob] {
            let saturation = at.region(plane).saturation();
            let mut magnitude = match rng.next() % 10 {
                0..=4 => 0,
                5 | 6 => 1,
                7 => 2,
                8 => 3 + rng.next() % 4,
                _ => 1 + rng.next() % 30,
            };
            if magnitude >= saturation {
                if high {
                    magnitude = saturation - 1;
                }
                high = true;
            }
            block[at.pos()] = if rng.next().is_multiple_of(2) {
                magnitude as i32
            } else {
                -(magnitude as i32)
            };
        }
        let last = positions[eob - 1].pos();
        if block[last] == 0 {
            block[last] = 1;
        }
        block
    }

    /// The coded size of `levels` priced by the writer's estimate.
    fn estimated_bits(
        writer: &mut Writer,
        levels: &[i32],
        tx: TxSize,
        ctx: Context,
        qindex: u8,
        kind: Kind,
    ) -> f32 {
        writer.start_estimate();
        write_typed(writer, levels, tx, ctx, qindex, kind).unwrap();
        writer.finish_estimate() as f32
    }

    /// The model's rate of `levels`: the end of block, then each level in
    /// reverse scan order from the state the levels after it leave.
    fn model_bits(model: &mut RateModel<'_>, levels: &[i32], tx: TxSize) -> f32 {
        let eob = super::super::end_of_block(levels, tx);
        if eob == 0 {
            return model.empty_rate();
        }
        let scan = super::super::scan(tx);
        let mut bits = model.eob_rate(eob - 1);
        let mut state = model.initial_state();
        for c in (0..eob).rev() {
            let level = levels[usize::from(scan[c])].unsigned_abs();
            let mut level_bits = [0.0];
            model.level_rates(&state, c, &[level], &mut level_bits);
            bits += level_bits[0];
            model.advance(&mut state, c, level);
        }
        bits
    }

    /// Without adaptation every symbol is priced alike by the model and by
    /// the writer's estimate, so summed level rates equal the block's coded
    /// rate; with adaptation they differ only by the block's own
    /// adaptation, which the model leaves out (on these dense random blocks
    /// of up to 80 levels it lowers the coded rate by up to about 15%).
    #[test]
    fn rates_add_up_to_the_coded_block_rate() {
        let mut rng = Lcg(5);
        let mut cache = CostCache::default();
        for adaptive in [false, true] {
            for qindex in [60_u8, 120, 180, 230] {
                for tx in TxSize::ALL {
                    for plane in [PLANE_Y, PLANE_U, PLANE_V] {
                        if plane != PLANE_Y && tx == TxSize::Tx64 {
                            continue;
                        }
                        for is_inter in [false, true] {
                            let mut writer = Writer::new();
                            if adaptive {
                                writer.enable_adaptation(qindex);
                            }
                            let ctx = Context {
                                is_inter,
                                plane,
                                skip: (rng.next() % 4) as u8,
                                dc_sign: (rng.next() % 3) as u8,
                            };
                            let kind = Kind {
                                set: TxSetMode::IntraDct,
                                tx_type: TxType::DctDct,
                                levels: LevelCoding::Plain,
                            };
                            let levels = random_levels(&mut rng, tx, plane);
                            let expected =
                                estimated_bits(&mut writer, &levels, tx, ctx, qindex, kind);
                            let empty = vec![0; tx.coded_len()];
                            let expected_empty =
                                estimated_bits(&mut writer, &empty, tx, ctx, qindex, kind);
                            let block = CodedBlock {
                                tx,
                                context: ctx,
                                qindex,
                                steps: Qindices::uniform(qindex),
                                kind,
                            };
                            let mut model =
                                RateModel::new(&mut writer, &mut cache, block, levels[0] < 0);
                            let bits = model_bits(&mut model, &levels, tx);
                            let empty_bits = model_bits(&mut model, &empty, tx);
                            let tolerance = if adaptive {
                                ADAPTATION_TOLERANCE * expected
                            } else {
                                0.01
                            };
                            let case = format!("{tx:?} plane {plane} inter {is_inter} q {qindex}");
                            assert!(
                                (bits - expected).abs() <= tolerance,
                                "{case}: {bits} vs {expected}"
                            );
                            assert!(
                                (empty_bits - expected_empty).abs() < 0.01,
                                "{case} empty: {empty_bits} vs {expected_empty}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A non-DCT luma type that a DC-only block cannot signal makes that
    /// end of block invalid.
    #[test]
    fn unsignallable_type_prices_the_end_of_block_out() {
        let mut writer = Writer::new();
        let ctx = Context {
            is_inter: false,
            plane: PLANE_Y,
            skip: 0,
            dc_sign: 0,
        };
        let kind = Kind {
            set: TxSetMode::IntraDctAdst,
            tx_type: TxType::AdstAdst,
            levels: LevelCoding::Plain,
        };
        let mut cache = CostCache::default();
        let block = CodedBlock {
            tx: TxSize::Tx8,
            context: ctx,
            qindex: 120,
            steps: Qindices::uniform(120),
            kind,
        };
        let mut model = RateModel::new(&mut writer, &mut cache, block, false);
        assert!(model.eob_rate(0).is_infinite());
        assert!(model.eob_rate(5).is_finite());
    }

    /// At λ zero the trellis keeps the nearest levels; at a large λ it
    /// keeps only the cheapest coding.
    #[test]
    fn trellis_trades_rate_for_distortion() {
        let tx = TxSize::Tx8;
        let qindex = 120;
        let one = transform::dequantized_magnitude(tx, qindex, 1);
        let mut coefficients = vec![0; tx.coded_len()];
        coefficients[0] = one + one / 5;
        coefficients[9] = -(one * 3 / 4);
        coefficients[17] = one * 2 / 3;
        let plain = |coefficients: &[i32]| -> Vec<i32> {
            coefficients
                .iter()
                .map(|&c| (c.abs() + one / 4) / one * c.signum())
                .collect()
        };
        let ctx = Context {
            is_inter: true,
            plane: PLANE_Y,
            skip: 0,
            dc_sign: 0,
        };
        let kind = Kind {
            set: TxSetMode::IntraDct,
            tx_type: TxType::DctDct,
            levels: LevelCoding::Plain,
        };
        let mut search = TrellisQuantizer::default();
        let mut writer = Writer::new();
        let mut levels = plain(&coefficients);
        let block = CodedBlock {
            tx,
            context: ctx,
            qindex,
            steps: Qindices::uniform(qindex),
            kind,
        };
        let eob = super::super::end_of_block(&levels, tx);
        search.quantize(
            &mut writer,
            block,
            (0.0, 1),
            &coefficients,
            (&mut levels, eob),
        );
        // Position 17 is past the plain end of block: never searched.
        assert_eq!((levels[0], levels[9], levels[17]), (1, -1, 0));
        let mut levels = plain(&coefficients);
        let eob = super::super::end_of_block(&levels, tx);
        let huge_lambda = 1.0e9;
        search.quantize(
            &mut writer,
            block,
            (huge_lambda, 1),
            &coefficients,
            (&mut levels, eob),
        );
        // A full-block inter luma transform rarely codes no levels (the
        // block skip flag covers that), so its all-zero flag is dear: the
        // cheapest coding keeps the DC level alone.
        assert_eq!(levels.iter().filter(|&&l| l != 0).count(), 1);
        assert_eq!(levels[0], 1);
    }
}
