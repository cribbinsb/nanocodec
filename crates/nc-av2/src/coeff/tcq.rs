//! Trellis-coded quantisation (TCQ) of luma levels: the decoder's state
//! machine and dequantiser, and the encoder's 8-state Viterbi search.
//!
//! In a TCQ frame every lossy luma transform (all our types are of the 2D
//! class) codes its levels with two interleaved scalar quantizers. A state
//! machine walks the coefficients in coding order (reverse scan, from the
//! last nonzero one): states 0, 1, 4 and 5 use Q0, whose reconstructions
//! are the even multiples of half a step, states 2, 3, 6 and 7 use Q1, the
//! odd ones (and zero); each level's parity picks the next state. The union
//! is a grid of half steps, while each coefficient pays for a level of the
//! full step. The current quantizer also selects the luma base models
//! (`coeff_base*_cdf[..][q_i]`), and high-range remainders count in steps
//! of two from one level earlier, so a level's parity is always coded with
//! its base and low-range part.
//!
//! Reimplemented from AVM v1.0.0 `av2/common/quant_common.c`
//! (`tcq_init_state`, `tcq_next_state`, `tcq_quant`), `quant_common.h`
//! (`tcq_enable`) and `av2/decoder/decodetxb.c` (`read_high_range`, the
//! TCQ dequantisation in `av2_read_coeffs_txb`); see `THIRD_PARTY.md`.

use nc_encoder::CoeffRateModel;

use super::TxSize;
use super::levels::{self, MAX_LUMA_BASE_MAGNITUDE};
use super::trellis::{CodedBlock, ContextState, CostCache, RateModel, TcqSymbols};
use crate::entropy::Writer;
use crate::transform::{self, Qindices};

/// States of the 8-state machine (AVM `TCQ_MAX_STATES`).
pub(crate) const STATES: usize = 8;
/// The state a block's first coded level (its last nonzero one) starts in.
pub(crate) const INITIAL_STATE: u8 = 0;
/// AVM `next_state_lut_8st`: the state after a level of even and odd parity.
const NEXT_STATE: [[u8; 2]; STATES] = [
    [0, 4],
    [4, 0],
    [1, 5],
    [5, 1],
    [6, 2],
    [2, 6],
    [7, 3],
    [3, 7],
];

/// The state after coding a level of magnitude `level` in `state`.
#[must_use]
pub(crate) const fn next_state(state: u8, level: u32) -> u8 {
    NEXT_STATE[state as usize][(level & 1) as usize]
}

/// Whether `state` quantizes with Q1 (AVM `tcq_quant`: `state & 2`), which
/// also selects the second set of luma base models.
#[must_use]
pub(crate) const fn upper(state: u8) -> bool {
    state & 2 != 0
}

/// The reconstruction index of a level in half steps: `2 · level` under Q0,
/// `2 · level − 1` under Q1 (zero stays zero).
#[must_use]
pub(crate) const fn half_steps(level: u32, upper: bool) -> u32 {
    (2 * level).saturating_sub(upper as u32)
}

/// Dequantize the levels of a TCQ transform in place of
/// [`transform::dequantize`]: walking the coefficients in coding order, each
/// nonzero level's half-step index times the quantizer's step, masked to 24
/// bits as AVM does, rounded from Q3 and shifted one bit more than a plain
/// level. `eob` is the levels' end of block. The caller has checked the
/// 20-bit level limit.
pub(crate) fn dequantize(
    size: TxSize,
    (levels, eob): (&[i32], usize),
    qindex: Qindices,
    out: &mut [i32],
) {
    out[..size.coded_len()].fill(0);
    let mut state = INITIAL_STATE;
    for at in levels::scan_positions(size)[..eob].iter().rev() {
        let pos = at.pos();
        let level = levels[pos];
        let magnitude = level.unsigned_abs();
        if magnitude != 0 {
            let index = half_steps(magnitude, upper(state));
            let value = transform::dequantized_half_steps(size, qindex.at(pos), index);
            out[pos] = transform::clamp_coefficient(value * level.signum());
        }
        state = next_state(state, magnitude);
    }
}

/// The low part a TCQ level codes before its high-range remainder: from one
/// below `saturation` the remainder counts pairs, so the low part keeps the
/// level's parity (AVM `read_high_range` with `tcq_mode`).
#[must_use]
pub(crate) const fn low_part(level: u32, saturation: u32) -> u32 {
    let start = saturation - 1;
    if level < start {
        level
    } else {
        start + ((level - start) & 1)
    }
}

/// Whether a TCQ low part `coded` is followed by a high-range remainder.
#[must_use]
pub(crate) const fn has_remainder(coded: u32, saturation: u32) -> bool {
    coded + 1 >= saturation
}

/// The change in squared coefficient error when a TCQ transform's
/// `coefficients` (forward transform, coded area) are coded as `levels`
/// (end of block `eob`) instead of all zero, as
/// [`transform::coefficient_error_change`] forms it for plain levels.
#[must_use]
pub(crate) fn error_change(
    size: TxSize,
    (levels, eob): (&[i32], usize),
    qindex: Qindices,
    coefficients: &[i32],
) -> i64 {
    let mut state = INITIAL_STATE;
    let mut total = 0;
    for at in levels::scan_positions(size)[..eob].iter().rev() {
        let pos = at.pos();
        let magnitude = levels[pos].unsigned_abs();
        if magnitude != 0 {
            let index = half_steps(magnitude, upper(state));
            let value = transform::dequantized_half_steps(size, qindex.at(pos), index);
            let dequantized = i64::from(transform::clamp_coefficient(value * levels[pos].signum()));
            total += dequantized * (dequantized - 2 * i64::from(coefficients[pos]));
        }
        state = next_state(state, magnitude);
    }
    total
}

/// Largest candidate levels per quantizer and coefficient: the two around
/// the coefficient and zero.
pub(super) const MAX_CANDIDATES: usize = 3;
/// Nearest levels up to this one still have zero as a candidate (as the
/// shared trellis's `ZERO_CANDIDATE_MAX_LEVEL`).
const ZERO_CANDIDATE_MAX_LEVEL: u32 = 1;
/// Marks a path that starts at its own position (the block's last level).
const NO_PARENT: u32 = u32::MAX;

/// The levels one quantizer may give a coefficient, with their squared
/// error (sample SSE units).
#[derive(Debug, Clone, Copy, Default)]
struct Candidates {
    levels: [u32; MAX_CANDIDATES],
    distortion: [f32; MAX_CANDIDATES],
    count: usize,
}

impl Candidates {
    /// The levels of quantizer `upper` around `magnitude`: the one whose
    /// reconstruction lies at or below it, the next, and zero for small
    /// ones. `step` is the reconstruction of one plain level and
    /// `distortion(level)` prices each.
    fn new(magnitude: u32, step: i32, upper: bool, distortion: impl Fn(u32) -> f32) -> Self {
        // Q0 reconstructs L·step, Q1 (L − ½)·step: the level at or below
        // the magnitude is ⌊(2m + step·[Q1]) / 2step⌋.
        let step = u64::from(step.max(1).unsigned_abs());
        let below = ((2 * u64::from(magnitude) + step * u64::from(upper)) / (2 * step))
            .min(u64::from(super::MAX_LEVEL - 1)) as u32;
        // The two around the magnitude, then zero when it is not one of
        // them already.
        let (levels, count) = match below {
            0 => ([0, 1, 0], 2),
            b if b <= ZERO_CANDIDATE_MAX_LEVEL => ([b, b + 1, 0], 3),
            b => ([b, b + 1, 0], 2),
        };
        let mut out = Self {
            levels,
            count,
            ..Self::default()
        };
        for (level, slot) in levels[..count].iter().zip(&mut out.distortion) {
            *slot = distortion(*level);
        }
        out
    }

    /// Drop the zero candidate (a block's last level is nonzero).
    fn retain_nonzero(&mut self) {
        let mut kept = 0;
        for i in 0..self.count {
            if self.levels[i] != 0 {
                self.levels[kept] = self.levels[i];
                self.distortion[kept] = self.distortion[i];
                kept += 1;
            }
        }
        self.count = kept;
    }

    fn levels(&self) -> &[u32] {
        &self.levels[..self.count]
    }

    fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.levels[..self.count]
            .iter()
            .copied()
            .zip(self.distortion[..self.count].iter().copied())
    }
}

/// One coefficient of the search: each quantizer's candidates and their
/// symbols, and the coefficient's magnitude.
#[derive(Debug, Clone, Copy)]
struct Position {
    candidates: [Candidates; 2],
    symbols: [TcqSymbols; 2],
    /// The nonzero Q0 candidates as the block's last level, when the
    /// coefficient may be it: at least half a step (Q0's nearest level is
    /// nonzero).
    first: Option<(Candidates, TcqSymbols)>,
    magnitude: u32,
}

/// One decided level and the decision before it (towards the last level).
#[derive(Debug, Clone, Copy)]
struct Step {
    parent: u32,
    level: u32,
}

/// The survivors of the search at one position: per TCQ state the cost of
/// the cheapest path into it (infinite for none), its coefficient
/// contexts and its latest decision.
#[derive(Debug, Clone)]
struct Survivors {
    cost: [f64; STATES],
    context: [ContextState; STATES],
    step: [u32; STATES],
}

impl Survivors {
    fn new(initial: ContextState) -> Self {
        Self {
            cost: [f64::INFINITY; STATES],
            context: [initial; STATES],
            step: [NO_PARENT; STATES],
        }
    }
}

/// The best way found so far into one state: its cost, the state it
/// leaves (or [`FROM_TAIL`]) and its level.
#[derive(Debug, Clone, Copy)]
struct Choice {
    cost: f64,
    from: usize,
    level: u32,
}

/// [`Choice::from`] of a path that starts at its own position.
const FROM_TAIL: usize = STATES;

/// The rates priced at the current position, by quantizer and price key
/// ([`RateModel::tcq_key`]): the eight paths read only a few distinct
/// contexts at a position, so each is priced once.
#[derive(Debug)]
struct RateMemo {
    /// The current position's stamp; entries with another are stale.
    position: u64,
    entries: [[(u64, [f32; MAX_CANDIDATES]); MAX_LUMA_BASE_MAGNITUDE + 1]; 2],
}

impl Default for RateMemo {
    fn default() -> Self {
        Self {
            position: 0,
            entries: [[(0, [0.0; MAX_CANDIDATES]); MAX_LUMA_BASE_MAGNITUDE + 1]; 2],
        }
    }
}

impl RateMemo {
    /// Move to the next position, making every entry stale.
    fn advance(&mut self) {
        self.position += 1;
    }

    /// The rates of quantizer `upper` under `key`, priced by `price` on
    /// first use at this position.
    fn rates(
        &mut self,
        (upper, key): (bool, usize),
        price: impl FnOnce(&mut [f32; MAX_CANDIDATES]),
    ) -> [f32; MAX_CANDIDATES] {
        let entry = &mut self.entries[usize::from(upper)][key];
        if entry.0 != self.position {
            entry.0 = self.position;
            price(&mut entry.1);
        }
        entry.1
    }
}

/// How one transform's TCQ search weighs and prunes its paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TcqSearch {
    /// Sample SSE per bit.
    pub lambda: f32,
    /// Paths whose cost trails the position's cheapest path by more than
    /// this many bits (times λ) are dropped (`av2.tcq_prune_bits`), or
    /// `None` to keep every state's.
    pub prune_bits: Option<f32>,
}

/// The TCQ search of one transform: a Viterbi over the eight states (paths
/// into the same state are priced alike from then on, except for their
/// coefficient contexts, of which the cheaper path's are kept, as in VVC's
/// dependent-quantisation search), with a separate all-zero tail from
/// which any position may become the last coded level.
#[derive(Debug, Default)]
pub(crate) struct TcqQuantizer {
    steps: Vec<Step>,
    costs: CostCache,
    positions: Vec<Position>,
    memo: Box<RateMemo>,
}

impl TcqQuantizer {
    /// Choose the TCQ levels of `block` from its forward `coefficients`
    /// (coded region, raster): squared error plus `search.lambda` times
    /// their rate through `writer` as it stands, in each path's exact
    /// contexts. On entry
    /// `levels` holds the plain quantiser's levels, whose end of block
    /// `eob` bounds the search (as the plain trellis's); on return the
    /// chosen levels, and the result is their end of block.
    pub(crate) fn quantize(
        &mut self,
        writer: &mut Writer,
        block: CodedBlock,
        search: TcqSearch,
        coefficients: &[i32],
        (levels, eob): (&mut [i32], usize),
    ) -> usize {
        let (tx, steps) = (block.tx, block.steps);
        let scan = &levels::scan_positions(tx)[..eob];
        let error_scale = transform::coefficient_error_scale(tx);
        let model = RateModel::new(writer, &mut self.costs, block, coefficients[0] < 0);
        self.positions.clear();
        self.positions
            .extend(scan.iter().enumerate().map(|(index, at)| {
                let pos = at.pos();
                let magnitude = coefficients[pos].unsigned_abs();
                let qindex = steps.at(pos);
                let rebuild = transform::HalfStepDequantizer::new(tx, qindex);
                let distortion = |upper: bool| {
                    move |level: u32| {
                        let error = i64::from(magnitude)
                            - i64::from(rebuild.magnitude(half_steps(level, upper)));
                        // Exact in f64 (|error| < 2^26), as the squared
                        // f64 difference.
                        ((error * error) as f64 * error_scale) as f32
                    }
                };
                let step = transform::dequantized_magnitude(tx, qindex, 1);
                let candidates = [false, true]
                    .map(|upper| Candidates::new(magnitude, step, upper, distortion(upper)));
                let symbols = candidates.map(|c| model.tcq_symbols(index, false, c.levels()));
                // A new last level starts in the initial (Q0) state.
                let first = (2 * magnitude >= step.unsigned_abs())
                    .then(|| {
                        let mut first = candidates[usize::from(upper(INITIAL_STATE))];
                        first.retain_nonzero();
                        (first.count > 0)
                            .then(|| (first, model.tcq_symbols(index, true, first.levels())))
                    })
                    .flatten();
                Position {
                    candidates,
                    symbols,
                    first,
                    magnitude,
                }
            }));
        levels[..tx.coded_len()].fill(0);
        let best = viterbi(
            &self.positions,
            (&mut self.steps, &mut self.memo),
            model,
            (search, error_scale),
        );
        let Some(mut step) = best else {
            return 0;
        };
        let mut index = 0;
        loop {
            let Step { parent, level } = self.steps[step as usize];
            let pos = scan[index].pos();
            let magnitude = level as i32;
            levels[pos] = if coefficients[pos] < 0 {
                -magnitude
            } else {
                magnitude
            };
            if parent == NO_PARENT {
                return index + 1;
            }
            step = parent;
            index += 1;
        }
    }
}

/// The Viterbi pass over `positions`, pricing through `model` (each
/// distinct price once per position, in `memo`) and recording decisions
/// in `steps`: the best path's final step (at scan index 0), or `None`
/// when coding no level is cheapest. With `search.prune_bits`, paths far
/// above the cheapest at a position are dropped there.
fn viterbi(
    positions: &[Position],
    (steps, memo): (&mut Vec<Step>, &mut RateMemo),
    mut model: RateModel<'_>,
    (search, error_scale): (TcqSearch, f64),
) -> Option<u32> {
    let lambda = f64::from(search.lambda);
    let prune = search.prune_bits.map(|bits| f64::from(bits) * lambda);
    let initial = model.initial_state();
    steps.clear();
    // Two buffers whose roles swap at each position (by reference: the
    // contexts are large).
    let mut buffers = [Survivors::new(initial), Survivors::new(initial)];
    let [first, second] = &mut buffers;
    let (mut survivors, mut next) = (first, second);
    let mut tail = 0.0_f64;
    let mut rates = [0.0_f32; MAX_CANDIDATES];
    for (index, position) in positions.iter().enumerate().rev() {
        let mut best = [Choice {
            cost: f64::INFINITY,
            from: FROM_TAIL,
            level: 0,
        }; STATES];
        let mut relax = |cost: f64, from: usize, level: u32, state: u8| {
            let slot = &mut best[usize::from(next_state(state, level))];
            if cost < slot.cost {
                *slot = Choice { cost, from, level };
            }
        };
        memo.advance();
        for state in 0..STATES {
            let cost = survivors.cost[state];
            if cost == f64::INFINITY {
                continue;
            }
            let quantizer = upper(state as u8);
            let candidates = &position.candidates[usize::from(quantizer)];
            let symbols = &position.symbols[usize::from(quantizer)];
            let context = &survivors.context[state];
            match model.tcq_key(context, index, symbols) {
                Some(key) => {
                    rates = memo.rates((quantizer, key), |bits| {
                        model.tcq_rates_by_key((index, quantizer), key, symbols, bits);
                    });
                }
                None => model.tcq_rates(context, (index, quantizer), symbols, &mut rates),
            }
            for ((level, distortion), &rate) in candidates.iter().zip(&rates) {
                let total = cost + f64::from(distortion) + lambda * f64::from(rate);
                relax(total, state, level, state as u8);
            }
        }
        if let Some((first, symbols)) = &position.first {
            let end = f64::from(model.eob_rate(index));
            model.tcq_rates(&initial, (index, false), symbols, &mut rates);
            for ((level, distortion), &rate) in first.iter().zip(&rates) {
                let total = tail + f64::from(distortion) + lambda * (end + f64::from(rate));
                relax(total, FROM_TAIL, level, INITIAL_STATE);
            }
        }
        let magnitude = f64::from(position.magnitude);
        tail += magnitude * magnitude * error_scale;
        if let Some(margin) = prune {
            prune_paths(&mut best, margin);
        }
        for (state, choice) in best.iter().enumerate() {
            next.cost[state] = choice.cost;
            if choice.cost == f64::INFINITY {
                continue;
            }
            let parent = if choice.from == FROM_TAIL {
                next.context[state] = initial;
                NO_PARENT
            } else {
                next.context[state] = survivors.context[choice.from];
                survivors.step[choice.from]
            };
            model.advance(&mut next.context[state], index, choice.level);
            steps.push(Step {
                parent,
                level: choice.level,
            });
            next.step[state] = steps.len() as u32 - 1;
        }
        (survivors, next) = (next, survivors);
    }
    let empty = tail + lambda * f64::from(model.empty_rate());
    // The first of equally cheap states, as `Iterator::min_by` picks.
    let (best, cost) = survivors.cost.iter().enumerate().fold(
        (None, f64::INFINITY),
        |(best, low), (state, &cost)| {
            if cost < low {
                (Some(state), cost)
            } else {
                (best, low)
            }
        },
    );
    best.filter(|_| cost < empty)
        .map(|state| survivors.step[state])
}

/// Drop the paths of `best` whose cost exceeds the cheapest one's by more
/// than `margin` (sample SSE).
///
/// Speed heuristic (`av2.tcq_prune_bits`): a pruned state is not priced
/// or extended at the following positions. Not exact: a dropped path
/// could still have won through cheap later levels, which is rare when it
/// trails by several bits. Measured (`crowd_mot17_09` / `pan_jaad_0101` /
/// `lowlight_pp22_049` 60 f CBR, luma at equal rate vs TCQ off; exact
/// search +0.15 / +0.23 / +0.18 dB): 8 bits +0.13 / +0.23 / +0.17 dB,
/// 5 bits +0.12 / +0.20 / +0.16 dB, 3 bits +0.12 / +0.13 / +0.10 dB,
/// 0 (one survivor) none of the gain; crowd 11 f P-frame instructions
/// (vs TCQ off) +39% / +32% / +25% against +44% exact.
fn prune_paths(best: &mut [Choice; STATES], margin: f64) {
    let low = best.iter().map(|c| c.cost).fold(f64::INFINITY, f64::min);
    for choice in best {
        if choice.cost > low + margin {
            choice.cost = f64::INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn states_follow_avm_and_split_into_two_quantizers() {
        // Both successors of a state use the same quantizer; four states
        // use each.
        for state in 0..STATES as u8 {
            assert_eq!(upper(next_state(state, 0)), upper(next_state(state, 1)));
        }
        assert_eq!((0..STATES as u8).filter(|&s| upper(s)).count(), STATES / 2);
        assert_eq!(next_state(0, 3), 4);
        assert_eq!(next_state(6, 2), 7);
        assert!(!upper(INITIAL_STATE));
    }

    #[test]
    fn low_parts_keep_parity_from_one_below_saturation() {
        let saturation = 6;
        for level in 0..40 {
            let low = low_part(level, saturation);
            assert!(low <= saturation);
            assert_eq!(low & 1, level & 1);
            if has_remainder(low, saturation) {
                assert_eq!((level - low) % 2, 0);
            } else {
                assert_eq!(low, level);
            }
        }
    }

    #[test]
    fn upper_quantizer_reconstructs_odd_half_steps() {
        assert_eq!(half_steps(0, true), 0);
        assert_eq!(half_steps(1, true), 1);
        assert_eq!(half_steps(3, false), 6);
    }
}
