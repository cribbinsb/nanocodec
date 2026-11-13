//! The HEVC coefficient rate model of the shared trellis
//! ([`nc_encoder::CoeffRateModel`]) and the trellis quantiser built on it.
//!
//! Prices levels as `residual::write_residual` codes them (HEVC §7.3.8.11,
//! contexts §9.3.4.2.4–9.3.4.2.7), with the slice's CABAC context states
//! as they stand. Levels are decided in reverse scan order, the order
//! `residual_coding()` codes them in, and the state carries what the
//! contexts of the remaining levels depend on: which 4x4 sub-blocks are
//! coded (`csbfCtx`, the `sig_coeff_flag` pattern), and, in the current
//! sub-block, the greater-1 context state, its context set, the greater-2
//! flag, the Rice parameter and how many levels it holds. A sub-block that
//! ends with no level pays `coded_sub_block_flag` = 0 instead of its
//! `sig_coeff_flag`s.
//!
//! Approximations (each costs at most a bit or so per block):
//! - sign data hiding is not priced: every level pays its sign bit, and the
//!   parity fix-up (`quant::hide_signs`) runs after the trellis;
//! - bins of one syntax element are priced with the context states before
//!   it, without the adaptation between them (`last_sig_coeff_*_prefix`);
//! - the coded-block flag (`cbf_luma` / `cbf_cb` / `cbf_cr`) belongs to the
//!   transform tree and is left to the backend's own decisions.

use nc_encoder::{Candidates, CoeffRateModel, Trellis};

use crate::cabac::Context;
use crate::contexts::{
    CODED_SUB_BLOCK_CHROMA_OFFSET, Contexts, GREATER1_CHROMA_OFFSET, GREATER2_CHROMA_OFFSET,
    LAST_CHROMA_OFFSET, SIG_CHROMA_OFFSET,
};
use crate::rate::BinCosts;
use crate::residual::{
    GROUP_IDX, MAX_GREATER1_FLAGS, MAX_RICE_PARAM, RICE_PREFIX_LIMIT, sig_ctx_inc,
};
use crate::scan::{SUB_BLOCK_COEFFS, SUB_BLOCK_LOG2, Scan, ScanKind};

/// `coeff_abs_level_greater1_flag` contexts per context set.
const GREATER1_CONTEXTS_PER_SET: usize = 4;
/// Initial greater-1 context state of a block (`greater1Ctx` = 1).
const GREATER1_INITIAL: u8 = 1;
/// Greater-1 context states that move up after a level of one.
const GREATER1_RISING: std::ops::Range<u8> = 1..3;

/// What the contexts of the levels still to be decided depend on.
#[derive(Debug, Clone, Copy)]
pub struct ContextState {
    /// Coded sub-blocks so far, one bit per sub-block in raster order.
    coded: u64,
    /// Scan index of the current sub-block; `None` before the block's last
    /// level is coded.
    sub_block: Option<u8>,
    /// Whether the current sub-block holds the block's last level (its
    /// `coded_sub_block_flag` is inferred).
    in_last_sub_block: bool,
    /// Non-zero levels in the current sub-block.
    count: u8,
    /// `greater1Ctx` (carried from sub-block to sub-block, §9.3.4.2.6).
    greater1: u8,
    /// `ctxSet` of the current sub-block.
    context_set: u8,
    /// Whether the current sub-block coded its `greater2` flag.
    greater2_coded: bool,
    /// `firstGreater2Pending` of `coeff_abs_level_remaining`'s base level.
    greater2_pending: bool,
    /// `cRiceParam`.
    rice: u8,
    /// Bits of the current sub-block's zero `sig_coeff_flag`s, refunded if
    /// the sub-block ends uncoded. Not part of the state's identity: it
    /// matters only for an all-zero sub-block, whose zeros are the same in
    /// every path.
    zero_bits: f32,
}

impl ContextState {
    /// Equality including `zero_bits` (which `==` leaves out).
    fn same_as(&self, other: &Self) -> bool {
        self == other && self.zero_bits.to_bits() == other.zero_bits.to_bits()
    }
}

impl PartialEq for ContextState {
    fn eq(&self, other: &Self) -> bool {
        self.coded == other.coded
            && self.sub_block == other.sub_block
            && self.in_last_sub_block == other.in_last_sub_block
            && self.count == other.count
            && self.greater1 == other.greater1
            && self.context_set == other.context_set
            && self.greater2_coded == other.greater2_coded
            && self.greater2_pending == other.greater2_pending
            && self.rice == other.rice
    }
}

/// A position `state` has entered: the bits of closing the previous
/// sub-block, and the `sig_coeff_flag` context (`None` where the flag is
/// implied or inferred).
#[derive(Debug, Clone, Copy)]
struct Position {
    bits: f32,
    significance: Option<Context>,
    sub_block: usize,
}

/// The last position [`RateModel::level_rates`] entered: the trellis
/// advances a state it has just priced, so `advance` reuses the entry
/// instead of deriving it again.
#[derive(Debug, Clone, Copy)]
struct Entered {
    scan_pos: usize,
    /// The state before entering (compared exactly, `zero_bits` included).
    from: ContextState,
    to: ContextState,
    position: Position,
}

/// Prices one transform block's levels.
pub struct RateModel<'a> {
    contexts: &'a Contexts,
    costs: &'a BinCosts,
    scan: &'a Scan,
    luma: bool,
    entered: Option<Entered>,
}

impl<'a> RateModel<'a> {
    /// The model of a block coded in `scan` with the context states
    /// `contexts`, priced by `costs`.
    #[must_use]
    pub fn new(contexts: &'a Contexts, costs: &'a BinCosts, scan: &'a Scan, luma: bool) -> Self {
        Self {
            contexts,
            costs,
            scan,
            luma,
            entered: None,
        }
    }

    fn bin(&self, ctx: Context, bin: bool) -> f32 {
        self.costs.bin(ctx, bin)
    }

    fn side(&self) -> usize {
        1 << (self.scan.log2 - SUB_BLOCK_LOG2)
    }

    fn is_coded(&self, state: &ContextState, xs: usize, ys: usize) -> bool {
        let side = self.side();
        xs < side && ys < side && state.coded & (1 << (ys * side + xs)) != 0
    }

    /// Close the current sub-block: record whether it is coded and price
    /// its `coded_sub_block_flag` (inferred for the first and the last).
    fn close_sub_block(&self, state: &mut ContextState) -> f32 {
        let Some(index) = state.sub_block.map(usize::from) else {
            return 0.0;
        };
        let (xs, ys) = self.scan.sub_blocks[index];
        let coded = state.count > 0;
        if coded {
            state.coded |= 1 << (ys * self.side() + xs);
        }
        if state.in_last_sub_block || index == 0 {
            return 0.0;
        }
        let neighbour = self.is_coded(state, xs + 1, ys) || self.is_coded(state, xs, ys + 1);
        let base = if self.luma {
            0
        } else {
            CODED_SUB_BLOCK_CHROMA_OFFSET
        };
        let flag = self.bin(
            self.contexts.coded_sub_block_flag[base + usize::from(neighbour)],
            coded,
        );
        if coded { flag } else { flag - state.zero_bits }
    }

    /// Code `level` at `scan_pos` in `state`: its bits, and `state` moved
    /// past it.
    fn step(&self, state: &mut ContextState, scan_pos: usize, level: u32) -> f32 {
        let position = self.enter(state, scan_pos);
        position.bits + self.code_level(state, &position, level)
    }

    /// Move `state` to `scan_pos` before its level is chosen: close the
    /// previous sub-block on entering a new one, and find the
    /// `sig_coeff_flag` context, which every candidate level shares.
    fn enter(&self, state: &mut ContextState, scan_pos: usize) -> Position {
        let sub_block = scan_pos / SUB_BLOCK_COEFFS;
        let is_last = state.sub_block.is_none();
        let mut bits = 0.0;
        if is_last {
            state.in_last_sub_block = true;
            state.sub_block = Some(sub_block as u8);
        } else if state.sub_block != Some(sub_block as u8) {
            bits += self.close_sub_block(state);
            state.sub_block = Some(sub_block as u8);
            state.in_last_sub_block = false;
            state.count = 0;
            state.zero_bits = 0.0;
        }

        // sig_coeff_flag: implied for the last level, and inferred for the
        // DC of a coded sub-block (not the first) with no other level.
        let inferred =
            scan_pos.is_multiple_of(SUB_BLOCK_COEFFS) && sub_block > 0 && state.count == 0;
        let significance = (!is_last && !inferred).then(|| {
            let p = self.scan.positions[scan_pos];
            let log2 = self.scan.log2;
            let (xs, ys) = self.scan.sub_blocks[sub_block];
            let pattern = (self.side() > 1).then(|| {
                usize::from(self.is_coded(state, xs + 1, ys))
                    | (usize::from(self.is_coded(state, xs, ys + 1)) << 1)
            });
            let (x, y) = (p & ((1 << log2) - 1), p >> log2);
            let ctx = sig_ctx_inc(pattern, x, y, log2, self.luma, self.scan.kind);
            let base = if self.luma { 0 } else { SIG_CHROMA_OFFSET };
            self.contexts.sig_coeff_flag[base + ctx]
        });
        Position {
            bits,
            significance,
            sub_block,
        }
    }

    /// Code `level` at the position `state` has entered: its bits after
    /// the transition's, and `state` moved past it.
    fn code_level(&self, state: &mut ContextState, position: &Position, level: u32) -> f32 {
        let mut bits = 0.0;
        if let Some(ctx) = position.significance {
            let sig = self.bin(ctx, level != 0);
            bits += sig;
            if level == 0 {
                state.zero_bits += sig;
            }
        }
        if level == 0 {
            return bits;
        }
        let sub_block = position.sub_block;

        if state.count == 0 {
            // First level of the sub-block: its context set (§9.3.4.2.6).
            let mut set = if sub_block > 0 && self.luma { 2 } else { 0 };
            if state.greater1 == 0 {
                set += 1;
            }
            state.context_set = set;
            state.greater1 = GREATER1_INITIAL;
            state.greater2_coded = false;
            state.greater2_pending = true;
            state.rice = 0;
        }
        let index = usize::from(state.count);
        state.count += 1;
        let set = usize::from(state.context_set);
        if index < MAX_GREATER1_FLAGS {
            let base = if self.luma { 0 } else { GREATER1_CHROMA_OFFSET };
            let ctx = base + GREATER1_CONTEXTS_PER_SET * set + usize::from(state.greater1);
            bits += self.bin(self.contexts.greater1_flag[ctx], level > 1);
            if level > 1 {
                state.greater1 = 0;
                if !state.greater2_coded {
                    let base = if self.luma { 0 } else { GREATER2_CHROMA_OFFSET };
                    bits += self.bin(self.contexts.greater2_flag[base + set], level > 2);
                    state.greater2_coded = true;
                }
            } else if GREATER1_RISING.contains(&state.greater1) {
                state.greater1 += 1;
            }
        }
        // The sign, then coeff_abs_level_remaining above what the flags
        // implied.
        bits += 1.0;
        let base = if index < MAX_GREATER1_FLAGS {
            2 + u32::from(state.greater2_pending)
        } else {
            1
        };
        if level >= base {
            let rice = u32::from(state.rice);
            bits += remaining_bits(level - base, rice) as f32;
            if level > 3 * (1 << rice) {
                state.rice = (state.rice + 1).min(MAX_RICE_PARAM as u8);
            }
        }
        if level >= 2 {
            state.greater2_pending = false;
        }
        bits
    }
}

/// Bypass bins of `coeff_abs_level_remaining` = `value` with Rice
/// parameter `rice` (HEVC §9.3.3.11; `residual::write_remaining`).
fn remaining_bits(value: u32, rice: u32) -> u32 {
    if value < (RICE_PREFIX_LIMIT << rice) {
        return (value >> rice) + 1 + rice;
    }
    let mut length = rice;
    let mut rest = value - (RICE_PREFIX_LIMIT << rice);
    while rest >= (1 << length) {
        rest -= 1 << length;
        length += 1;
    }
    RICE_PREFIX_LIMIT + length + 1 - rice + length
}

impl CoeffRateModel for RateModel<'_> {
    type State = ContextState;

    fn initial_state(&self) -> ContextState {
        ContextState {
            coded: 0,
            sub_block: None,
            in_last_sub_block: false,
            count: 0,
            greater1: GREATER1_INITIAL,
            context_set: 0,
            greater2_coded: false,
            greater2_pending: true,
            rice: 0,
            zero_bits: 0.0,
        }
    }

    /// `last_sig_coeff_{x,y}_{prefix,suffix}` (HEVC §7.3.8.11, §9.3.4.2.3).
    fn eob_rate(&mut self, scan_pos: usize) -> f32 {
        let log2 = self.scan.log2;
        let size = 1usize << log2;
        let p = self.scan.positions[scan_pos];
        // A vertical scan codes the last position with x and y swapped.
        let (x, y) = match self.scan.kind {
            ScanKind::Vertical => (p / size, p % size),
            _ => (p % size, p / size),
        };
        let log2 = u32::from(log2);
        let (offset, shift) = if self.luma {
            (
                (3 * (log2 - 2) + ((log2 - 1) >> 2)) as usize,
                (log2 + 1) >> 2,
            )
        } else {
            (LAST_CHROMA_OFFSET, log2 - 2)
        };
        let max_group = GROUP_IDX[size - 1];
        let mut bits = 0.0;
        for (coord, contexts) in [
            (x, &self.contexts.last_x_prefix),
            (y, &self.contexts.last_y_prefix),
        ] {
            let group = GROUP_IDX[coord];
            for bin in 0..group {
                bits += self.bin(contexts[offset + (bin >> shift) as usize], true);
            }
            if group < max_group {
                bits += self.bin(contexts[offset + (group >> shift) as usize], false);
            }
            if group > 3 {
                bits += ((group - 2) / 2) as f32;
            }
        }
        bits
    }

    fn level_rates(
        &mut self,
        state: &ContextState,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32],
    ) {
        let mut entered = *state;
        let position = self.enter(&mut entered, scan_pos);
        for (i, &level) in levels.iter().enumerate() {
            bits[i] = position.bits + self.code_level(&mut { entered }, &position, level);
        }
        self.entered = Some(Entered {
            scan_pos,
            from: *state,
            to: entered,
            position,
        });
    }

    fn advance(&self, state: &mut ContextState, scan_pos: usize, level: u32) {
        match &self.entered {
            Some(e) if e.scan_pos == scan_pos && e.from.same_as(state) => {
                *state = e.to;
                self.code_level(state, &e.position, level);
            }
            _ => {
                self.step(state, scan_pos, level);
            }
        }
    }

    fn final_rate(&mut self, state: &ContextState) -> f32 {
        self.close_sub_block(&mut state.clone())
    }

    fn empty_rate(&mut self) -> f32 {
        0.0
    }
}

/// Trellis quantisation of HEVC transform blocks, with reusable buffers.
#[derive(Debug, Default)]
pub struct TrellisQuantizer {
    search: Trellis<ContextState>,
    candidates: Vec<Candidates>,
    chosen: Vec<u32>,
}

/// One block's quantiser, as the trellis needs it.
#[derive(Debug, Clone, Copy)]
pub struct BlockQuantizer {
    /// Squared error of one quantisation step, in sample-SSE units.
    pub step_sse: f32,
    /// Forward scale and shift: `|c| * scale / 2^qbits` is a coefficient's
    /// magnitude in steps (`quant::quant_params`).
    pub scale: i64,
    pub qbits: i32,
}

impl TrellisQuantizer {
    /// Choose the levels of a block from its transform coefficients
    /// `coeffs` (row-major), by distortion plus `lambda` (sample SSE per
    /// bit) times `model`'s rate, keeping at most `max_states` survivors per
    /// scan position. On entry `levels` holds the plain quantiser's levels
    /// (rounding below half a step); the search spans their last level and
    /// each coefficient's candidates sit around its level rounded to the
    /// nearest. Returns the number of non-zero levels.
    pub fn quantize(
        &mut self,
        model: &mut RateModel<'_>,
        coeffs: &[i32],
        quantizer: BlockQuantizer,
        (lambda, max_states): (f32, usize),
        levels: &mut [i32],
    ) -> u32 {
        let positions = &model.scan.positions;
        let Some(last) = positions.iter().rposition(|&p| levels[p] != 0) else {
            return 0;
        };
        // 2^-qbits: exact, so multiplying equals dividing by 2^qbits.
        let unit = 1.0 / (1_i64 << quantizer.qbits) as f64;
        self.candidates.clear();
        self.candidates.extend(positions[..=last].iter().map(|&p| {
            let value = (i64::from(coeffs[p].unsigned_abs()) * quantizer.scale) as f64 * unit;
            let plain = levels[p].unsigned_abs();
            let nearest = (value.round() as u32).max(plain);
            Candidates::around(nearest, |level| {
                let error = value - f64::from(level);
                (error * error) as f32 * quantizer.step_sse
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
        let mut nonzero = 0;
        for (i, &p) in positions[..=last].iter().enumerate() {
            let level = if i < coded { self.chosen[i] as i32 } else { 0 };
            levels[p] = if coeffs[p] < 0 { -level } else { level };
            nonzero += u32::from(level != 0);
        }
        nonzero
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate::BitEstimator;
    use crate::residual::write_residual;

    /// Summed level rates equal `write_residual`'s estimate (signs not
    /// hidden), up to the adaptation between bins of one element.
    #[test]
    fn rates_follow_the_residual_coder() {
        let costs = BinCosts::new();
        let mut seed = 99_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        for log2 in 2..=5u8 {
            for kind in [ScanKind::Diagonal, ScanKind::Horizontal, ScanKind::Vertical] {
                if log2 > 3 && kind != ScanKind::Diagonal {
                    continue;
                }
                let scan = Scan::new(log2, kind);
                // 4:2:0 chroma transforms stop at 16x16, and only its 4x4
                // ones take mode-dependent scans (§7.4.9.11 scanIdx).
                let chroma_ok = log2 < 5 && (log2 == 2 || kind == ScanKind::Diagonal);
                for luma in [true, false].into_iter().filter(|&l| l || chroma_ok) {
                    let n = 1usize << log2;
                    let mut levels = vec![0_i32; n * n];
                    for &p in &scan.positions[..(n * n).min(48)] {
                        let v = match next() % 9 {
                            0..=4 => 0,
                            5 | 6 => 1,
                            7 => 2,
                            _ => 3 + (next() % 30) as i32,
                        };
                        levels[p] = if next() % 2 == 0 { v } else { -v };
                    }
                    levels[scan.positions[3]] = 5;
                    if log2 > 2 {
                        // An uncoded sub-block between coded ones.
                        for &p in &scan.positions[SUB_BLOCK_COEFFS..2 * SUB_BLOCK_COEFFS] {
                            levels[p] = 0;
                        }
                    }
                    let contexts = Contexts::for_slice(crate::contexts::SliceKind::Predicted, 30);
                    let mut estimate = BitEstimator::new(&costs);
                    write_residual(
                        &mut estimate,
                        &mut contexts.clone(),
                        &scan,
                        &levels,
                        luma,
                        false,
                    );
                    let mut model = RateModel::new(&contexts, &costs, &scan, luma);
                    let last = scan
                        .positions
                        .iter()
                        .rposition(|&p| levels[p] != 0)
                        .unwrap();
                    let mut bits = model.eob_rate(last);
                    let mut state = model.initial_state();
                    for pos in (0..=last).rev() {
                        let level = levels[scan.positions[pos]].unsigned_abs();
                        let mut b = [0.0];
                        model.level_rates(&state, pos, &[level], &mut b);
                        bits += b[0];
                        model.advance(&mut state, pos, level);
                    }
                    bits += model.final_rate(&state);
                    // Context adaptation inside the block moves the
                    // estimate a little; the model prices static states.
                    let tolerance = 0.05 * estimate.bits + 1.0;
                    assert!(
                        (bits - estimate.bits).abs() < tolerance,
                        "log2 {log2} {kind:?} luma {luma}: {bits} vs {}",
                        estimate.bits
                    );
                }
            }
        }
    }

    #[test]
    fn remaining_bits_match_the_binarisation() {
        // value < 3 << rice: unary prefix + stop + rice suffix bits.
        assert_eq!(remaining_bits(0, 0), 1);
        assert_eq!(remaining_bits(2, 0), 3);
        assert_eq!(remaining_bits(5, 1), 2 + 1 + 1);
        // Escape: 3 << 0 = 3 → prefix 4 bins + 0 suffix bits for value 3.
        assert_eq!(remaining_bits(3, 0), 4);
    }
}
