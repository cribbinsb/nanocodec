//! The VVC coefficient rate model of the shared trellis
//! ([`nc_encoder::CoeffRateModel`]) and the trellis quantiser built on it.
//!
//! Prices levels as `coeff::code` codes them (H.266 §7.3.11.11, contexts
//! §9.3.4.2.4–9.3.4.2.8, without dependent quantisation), with the
//! picture's planning context states as they stand. Levels are decided in
//! reverse scan order, the order `residual_coding()` codes them in. A
//! position's contexts read the template of five positions to its right and
//! below ([`TEMPLATE`]), which in VVC's grouped diagonal scan may lie in any
//! earlier-coded 4x4 group, so the levels decided so far are kept whole, in
//! a raster buffer ([`Decided`]), rather than in the state. That makes the model exact for the one-survivor search only (the
//! shared trellis' `quantize_single`, every adopted codec's setting): the
//! buffer holds the survivor's levels, and the state's `last` says which of
//! them belong to its path. The state carries the rest: the coded groups
//! (`sb_coded_flag` contexts), the current group's level count (the
//! inferred significance of its first position), the context-coded bin
//! budget (`remBinsPass1`), and the current group's zero-flag bits and
//! bins, refunded when the group ends empty and its flag says so instead.
//!
//! Approximations (each at most a bit or so per block):
//! - sign data hiding is not priced: every level pays its sign bit, and the
//!   parity fix-up (`recon::hide_signs`) runs after the trellis;
//! - bins of one syntax element are priced with the context states before
//!   it, without adaptation in between (`last_sig_coeff_*_prefix`);
//! - the coded-block flags (`tu_y/cb/cr_coded_flag`, `cu_coded_flag`)
//!   belong to the coding unit and are left to the backend's decisions.
use std::cell::Cell;

use nc_dsp::VvcQuantizer;
use nc_encoder::{Candidates, CoeffRateModel, QuantParams, Trellis};

use crate::coeff::{
    MIN_REGULAR_BUDGET, TEMPLATE, context_bin_budget, last_coordinate_bits, pass_one_magnitude,
    remainder_bits, rice_parameter, template_contexts,
};
use crate::coeff_contexts::CoeffContexts;
use crate::recon::CodedGroups;
use crate::scan::GROUP_SIZE;

/// Side of the largest transform block.
const MAX_SIDE: usize = 32;
/// Magnitude the first pass's flags imply at most: above it a remainder
/// follows (`abs_remainder`, §7.3.11.11).
const PASS_ONE_LIMIT: u32 = 4;
/// The QP whose quantisation step is one sample (`2^((qp − 4) / 6)`, H.266
/// §8.7.3 with flat scaling, as HEVC).
const QSTEP_UNIT_QP: f32 = 4.;

/// What the contexts of the levels still to be decided depend on, besides
/// the decided levels themselves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ContextState {
    /// Scan index of the block's last level; `None` before it is coded.
    last: Option<u16>,
    /// Coded groups so far, one bit per group in raster order.
    coded_groups: u64,
    /// Nonzero levels in the current group.
    group_levels: u8,
    /// Context-coded bins left (`remBinsPass1`).
    budget: u16,
    /// Bins and bits of the current group's zero levels, refunded if the
    /// group ends with none and is skipped by its `sb_coded_flag`.
    zero_bins: u16,
    zero_bits: f32,
}

/// A position a state has entered: the bits of closing the previous
/// group, and how its level is coded.
#[derive(Debug, Clone, Copy)]
struct Position {
    bits: f32,
    /// The position's [`Decided`] cell.
    cell: usize,
    coding: Coding,
}

#[derive(Debug, Clone, Copy)]
enum Coding {
    /// Context-coded flags: the significance context (`None` where the flag
    /// is implied or inferred), the level context and the Rice parameter of
    /// a remainder above the flags.
    Regular {
        significance: Option<usize>,
        level: usize,
        rice: u32,
    },
    /// The budget is spent: the whole magnitude is a bypass remainder.
    Bypass { rice: u32 },
}

/// The last position [`RateModel::level_rates`] entered: the trellis
/// advances a state it has just priced, so `advance` reuses the entry.
#[derive(Debug, Clone, Copy)]
struct Entered {
    scan_pos: usize,
    from: ContextState,
    to: ContextState,
    position: Position,
}

/// Prices one transform block's levels.
pub(crate) struct RateModel<'a> {
    contexts: &'a CoeffContexts,
    /// The coded region's width and height.
    size: (usize, usize),
    /// The transform block's width and height (the last position's
    /// contexts, §9.3.4.2.4).
    sides: (usize, usize),
    /// Channel type: 0 luma, 1 chroma.
    ch: usize,
    /// Each scan position's [`Decided`] cell and diagonal.
    places: &'static [Place],
    groups: &'static [usize],
    decided: &'a Decided,
    entered: Option<Entered>,
    /// Bits of each last-position coordinate, x then y, priced on first use
    /// (`None` until then): every position the all-zero tail reaches asks.
    last_bits: [[Option<f32>; MAX_SIDE]; 2],
}

impl<'a> RateModel<'a> {
    fn new(
        contexts: &'a CoeffContexts,
        sides: (usize, usize),
        ch: usize,
        decided: &'a Decided,
    ) -> Self {
        let size = coded_size(sides);
        Self {
            contexts,
            size,
            sides,
            ch,
            places: places(size),
            groups: crate::scan::rect_groups(size.0, size.1),
            decided,
            entered: None,
            last_bits: [[None; MAX_SIDE]; 2],
        }
    }

    /// The template of raster position `raster` on `state`'s path: the
    /// sum of its first-pass magnitudes, its nonzero count and the sum of
    /// its whole magnitudes.
    fn template(&self, state: &ContextState, cell: usize) -> (u32, u32, u32) {
        if state.last.is_none() {
            return (0, 0, 0);
        }
        let (mut pass_one, mut count, mut whole) = (0, 0, 0);
        for (dx, dy) in TEMPLATE {
            let magnitude = self.decided.get(cell + dy * DECIDED_STRIDE + dx);
            pass_one += pass_one_magnitude(magnitude);
            count += u32::from(magnitude != 0);
            whole += magnitude;
        }
        (pass_one, count, whole)
    }

    /// The block's coefficient-group grid: columns and rows.
    fn group_grid(&self) -> (usize, usize) {
        crate::scan::group_grid(self.size.0, self.size.1)
    }

    fn group_is_coded(&self, state: &ContextState, gx: usize, gy: usize) -> bool {
        let (columns, rows) = self.group_grid();
        gx < columns && gy < rows && state.coded_groups >> (gy * columns + gx) & 1 != 0
    }

    /// Close group `index` (scan order): record whether it is coded and
    /// price its `sb_coded_flag`, inferred for the first and the last group.
    fn close_group(&self, state: &mut ContextState, index: usize) -> f32 {
        let raster = self.groups[index];
        let (columns, _) = self.group_grid();
        let coded = state.group_levels > 0;
        if coded {
            state.coded_groups |= 1 << raster;
        }
        let (zero_bins, zero_bits) = (state.zero_bins, state.zero_bits);
        state.group_levels = 0;
        state.zero_bins = 0;
        state.zero_bits = 0.;
        let last_group = state.last.map(|l| usize::from(l) / GROUP_SIZE);
        if index == 0 || last_group == Some(index) {
            return 0.;
        }
        let (gx, gy) = (raster % columns, raster / columns);
        let neighbour =
            self.group_is_coded(state, gx + 1, gy) || self.group_is_coded(state, gx, gy + 1);
        let flag = self.contexts.group[self.ch][usize::from(neighbour)].bits(coded);
        if coded {
            flag
        } else {
            state.budget += zero_bins;
            flag - zero_bits
        }
    }

    /// Move `state` to `scan_pos` before its level is chosen: close the
    /// previous group on entering a new one, and find the contexts every
    /// candidate level shares.
    fn enter(&self, state: &mut ContextState, scan_pos: usize) -> Position {
        let mut bits = 0.;
        if state.last.is_some() && scan_pos % GROUP_SIZE == GROUP_SIZE - 1 {
            bits += self.close_group(state, (scan_pos + 1) / GROUP_SIZE);
        }
        let place = self.places[scan_pos];
        let cell = usize::from(place.cell);
        let (pass_one, count, whole) = self.template(state, cell);
        let coding = if usize::from(state.budget) < MIN_REGULAR_BUDGET {
            Coding::Bypass {
                rice: rice_parameter(whole, 0),
            }
        } else {
            let is_last = state.last.is_none();
            let group = scan_pos / GROUP_SIZE;
            let inferred = is_last
                || (scan_pos.is_multiple_of(GROUP_SIZE) && group != 0 && state.group_levels == 0);
            let diag = usize::from(place.diag);
            let (significance, level) = template_contexts(pass_one, count, diag, self.ch);
            Coding::Regular {
                significance: (!inferred).then_some(significance),
                level: if is_last { 0 } else { level },
                rice: rice_parameter(whole, PASS_ONE_LIMIT),
            }
        };
        Position { bits, cell, coding }
    }

    /// Code `level` at the position `state` has entered: its bits after
    /// the transition's, and `state` moved past it.
    fn code_level(&self, state: &mut ContextState, position: &Position, level: u32) -> f32 {
        let c = self.contexts;
        let ch = self.ch;
        let mut bits = 0.;
        match position.coding {
            Coding::Regular {
                significance,
                level: ctx,
                rice,
            } => {
                if let Some(sig) = significance {
                    let flag = c.sig[ch][sig].bits(level != 0);
                    bits += flag;
                    state.budget -= 1;
                    if level == 0 {
                        state.zero_bits += flag;
                        state.zero_bins += 1;
                    }
                }
                if level == 0 {
                    return bits;
                }
                bits += c.gt1[ch][ctx].bits(level > 1);
                state.budget -= 1;
                if level > 1 {
                    bits += c.parity[ch][ctx].bits(level & 1 != 0) + c.gt2[ch][ctx].bits(level > 3);
                    state.budget -= 2;
                }
                if level >= PASS_ONE_LIMIT {
                    bits += remainder_bits((level - PASS_ONE_LIMIT) >> 1, rice);
                }
            }
            Coding::Bypass { rice } => {
                // dec_abs_level: zero maps to 1 << rice (dependent
                // quantisation off), the levels at or below it one down.
                let zero = 1 << rice;
                let value = match level {
                    0 => zero,
                    l if l <= zero => l - 1,
                    l => l,
                };
                let remainder = remainder_bits(value, rice);
                bits += remainder;
                if level == 0 {
                    state.zero_bits += remainder;
                    return bits;
                }
            }
        }
        state.group_levels += 1;
        bits + 1. // the sign
    }

    fn step(&self, state: &mut ContextState, scan_pos: usize, level: u32) -> f32 {
        let position = self.enter(state, scan_pos);
        position.bits + self.record(state, &position, scan_pos, level)
    }

    /// [`Self::code_level`], keeping `level` as the survivor's decision.
    fn record(
        &self,
        state: &mut ContextState,
        position: &Position,
        scan_pos: usize,
        level: u32,
    ) -> f32 {
        let bits = self.code_level(state, position, level);
        if state.last.is_none() {
            // A new path starts here: what an older one decided past it is gone.
            self.decided.restart(scan_pos, self.places);
            state.last = Some(scan_pos as u16);
        }
        self.decided.set(position.cell, scan_pos, level);
        bits
    }
}

impl CoeffRateModel for RateModel<'_> {
    type State = ContextState;

    fn initial_state(&self) -> ContextState {
        ContextState {
            last: None,
            coded_groups: 0,
            group_levels: 0,
            budget: context_bin_budget(self.size.0 * self.size.1) as u16,
            zero_bins: 0,
            zero_bits: 0.,
        }
    }

    /// `last_sig_coeff_{x,y}_{prefix,suffix}` (§7.3.11.11, §9.3.4.2.4).
    fn eob_rate(&mut self, scan_pos: usize) -> f32 {
        let cell = usize::from(self.places[scan_pos].cell);
        let coordinates = [cell % DECIDED_STRIDE, cell / DECIDED_STRIDE];
        let mut bits = 0.;
        for (axis, value) in coordinates.into_iter().enumerate() {
            let (contexts, side) = if axis == 0 {
                (&self.contexts.last_x[self.ch], self.sides.0)
            } else {
                (&self.contexts.last_y[self.ch], self.sides.1)
            };
            let ch = self.ch;
            bits += *self.last_bits[axis][value]
                .get_or_insert_with(|| last_coordinate_bits(contexts, value, side, ch));
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
        for (bits, &level) in bits.iter_mut().zip(levels) {
            *bits = position.bits + self.code_level(&mut { entered }, &position, level);
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
            Some(e) if e.scan_pos == scan_pos && e.from == *state => {
                *state = e.to;
                self.record(state, &e.position, scan_pos, level);
            }
            _ => {
                self.step(state, scan_pos, level);
            }
        }
    }

    fn zero_step(&mut self, state: &mut ContextState, scan_pos: usize) -> f32 {
        self.step(state, scan_pos, 0)
    }

    fn empty_rate(&mut self) -> f32 {
        0.
    }
}

/// Trellis quantisation of VVC transform blocks, with reusable buffers.
#[derive(Debug, Default)]
pub(crate) struct TrellisQuantizer {
    search: Trellis<ContextState>,
    candidates: Vec<Candidates>,
    chosen: Vec<u32>,
    decided: Decided,
}

/// [`Decided`]'s row stride: a block's rows padded by the template's reach
/// (two columns right, two rows below), whose cells stay zero, so template
/// reads need no bounds checks.
const DECIDED_STRIDE: usize = MAX_SIDE + 2;

/// The coded region of a transform block of `sides` (§7.3.11.11 zero-out).
fn coded_size((width, height): (usize, usize)) -> (usize, usize) {
    (width.min(MAX_SIDE), height.min(MAX_SIDE))
}
const DECIDED_CELLS: usize = DECIDED_STRIDE * DECIDED_STRIDE;

/// A scan position of a transform block: its [`Decided`] cell and its
/// anti-diagonal `x + y` (the significance and level contexts' band),
/// tabulated so the search's inner step does no division by the width.
#[derive(Debug, Clone, Copy)]
struct Place {
    cell: u16,
    diag: u8,
}

/// The [`Place`]s of a `width` x `height` block in scan order.
const fn place_table<const N: usize>(width: usize, height: usize) -> [Place; N] {
    let scan = crate::scan::rect_positions(width, height);
    let mut out = [Place { cell: 0, diag: 0 }; N];
    let mut i = 0;
    while i < N {
        let (x, y) = (scan[i] % width, scan[i] / width);
        out[i] = Place {
            cell: Decided::cell(x, y) as u16,
            diag: (x + y) as u8,
        };
        i += 1;
    }
    out
}

/// Defines the [`Place`] table of every transform size and [`places`].
macro_rules! place_tables {
    ($(($w:literal, $h:literal, $name:ident)),* $(,)?) => {
        $(const $name: [Place; $w * $h] = place_table($w, $h);)*

        /// [`place_table`] of a `width` x `height` block.
        fn places((width, height): (usize, usize)) -> &'static [Place] {
            match (width, height) {
                $(($w, $h) => &$name,)*
                _ => panic!("unsupported VVC trellis block size"),
            }
        }
    };
}

place_tables!(
    (1, 16, P1X16),
    (1, 32, P1X32),
    (2, 8, P2X8),
    (2, 16, P2X16),
    (2, 32, P2X32),
    (8, 2, P8X2),
    (16, 1, P16X1),
    (16, 2, P16X2),
    (32, 1, P32X1),
    (32, 2, P32X2),
    (4, 4, P4X4),
    (4, 8, P4X8),
    (4, 16, P4X16),
    (4, 32, P4X32),
    (8, 4, P8X4),
    (8, 8, P8X8),
    (8, 16, P8X16),
    (8, 32, P8X32),
    (16, 4, P16X4),
    (16, 8, P16X8),
    (16, 16, P16X16),
    (16, 32, P16X32),
    (32, 4, P32X4),
    (32, 8, P32X8),
    (32, 16, P32X16),
    (32, 32, P32X32),
);

/// The survivor's levels (see the module comment), zero everywhere else
/// and between blocks.
#[derive(Debug)]
struct Decided {
    cells: Vec<Cell<u16>>,
    /// The highest scan position holding a level, if any.
    highest: Cell<Option<usize>>,
}

impl Default for Decided {
    fn default() -> Self {
        Self {
            cells: vec![Cell::new(0); DECIDED_CELLS],
            highest: Cell::new(None),
        }
    }
}

impl Decided {
    const fn cell(x: usize, y: usize) -> usize {
        y * DECIDED_STRIDE + x
    }

    fn get(&self, cell: usize) -> u32 {
        u32::from(self.cells[cell].get())
    }

    fn set(&self, cell: usize, scan_pos: usize, level: u32) {
        self.cells[cell].set(u16::try_from(level).unwrap_or(u16::MAX));
        if self.highest.get().is_none_or(|h| h < scan_pos) {
            self.highest.set(Some(scan_pos));
        }
    }

    /// Zero every level past `scan_pos` (`places` is the block's scan).
    fn restart(&self, scan_pos: usize, places: &[Place]) {
        self.zero_from(scan_pos + 1, places);
    }

    /// Zero every level, for the next block.
    fn clear(&self, places: &[Place]) {
        self.zero_from(0, places);
    }

    fn zero_from(&self, start: usize, places: &[Place]) {
        if let Some(highest) = self.highest.get() {
            for place in places.get(start..=highest).unwrap_or_default() {
                self.cells[usize::from(place.cell)].set(0);
            }
        }
        self.highest.set(None);
    }
}

/// One transform block's trellis inputs besides its coefficients.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockTrellis<'a> {
    pub(crate) contexts: &'a CoeffContexts,
    /// Channel type: 0 luma, 1 chroma.
    pub(crate) ch: usize,
    /// Sample SSE per bit.
    pub(crate) lambda: f32,
    /// Whether a level may rise above the dead-zone quantiser's.
    pub(crate) round_up: bool,
}

impl TrellisQuantizer {
    /// Choose the levels of a transform block of `sides` (width, height)
    /// from the transform coefficients `coefficients` of its coded region
    /// (raster order), by distortion plus λ times the rate.
    /// On entry `levels` holds the plain quantiser's levels, whose last
    /// nonzero one is at scan index `last`; the search spans up to it and
    /// each coefficient's candidates sit below its level rounded to the
    /// nearest (`block.round_up`) or its plain level. Returns the coefficient
    /// groups holding the chosen levels, read off as they are written (the
    /// caller would otherwise rescan the block for them), or `None` when no
    /// level remains. One survivor (see the module comment).
    pub(crate) fn quantize(
        &mut self,
        block: BlockTrellis<'_>,
        coefficients: &[i32],
        quantizer: &VvcQuantizer,
        qp: u32,
        levels: &mut [i32],
        sides: (usize, usize),
        last: usize,
    ) -> Option<CodedGroups> {
        let size = coded_size(sides);
        let positions = crate::scan::rect_positions(size.0, size.1);
        let step_sse = 2f32.powf((qp as f32 - QSTEP_UNIT_QP) / 3.);
        self.candidates.clear();
        self.candidates.extend(positions[..=last].iter().map(|&p| {
            let value = quantizer.magnitude_in_steps(coefficients[p]);
            let plain = levels[p].unsigned_abs();
            let nearest = if block.round_up {
                (value.round() as u32).max(plain)
            } else {
                plain
            };
            Candidates::around(nearest, |level| {
                let error = value - f64::from(level);
                (error * error) as f32 * step_sse
            })
        }));
        self.chosen.resize(last + 1, 0);
        let mut model = RateModel::new(block.contexts, sides, block.ch, &self.decided);
        let coded = self.search.quantize(
            &mut model,
            &self.candidates,
            block.lambda,
            1,
            &mut self.chosen,
        );
        self.decided.clear(places(size));
        // Levels past `last` were zero on entry and stay so.
        let mut groups = 0u64;
        for (i, &p) in positions[..=last].iter().enumerate() {
            let level = if i < coded { self.chosen[i] as i32 } else { 0 };
            groups |= u64::from(level != 0) << (i / GROUP_SIZE);
            levels[p] = if coefficients[p] < 0 { -level } else { level };
        }
        (coded > 0).then_some(CodedGroups::from_scan_mask(groups))
    }
}

/// The registered trellis controls (`nc_encoder::QuantParams`) and the
/// search's buffers.
#[derive(Debug)]
pub(crate) struct TrellisQuant {
    /// Multiple of the mode-decision λ.
    lambda_scale: f32,
    /// Further multiple of chroma's λ; `None` leaves chroma unoptimised.
    chroma_scale: Option<f32>,
    /// Whether key (IDR) pictures are trellised too.
    key_frames: bool,
    /// Further multiple of λ on key pictures.
    key_scale: f32,
    /// [`BlockTrellis::round_up`].
    round_up: bool,
    search: TrellisQuantizer,
}

impl TrellisQuant {
    /// The trellis `quant` enables, if any, rounding levels up past the
    /// dead zone where `round_up`. `quant.trellis_states` does not apply:
    /// the model is exact for one survivor only.
    /// `key_scale` further scales λ on key pictures.
    pub(crate) fn from_params(quant: &QuantParams, round_up: bool, key_scale: f32) -> Option<Self> {
        (quant.trellis >= 0.5).then(|| Self {
            round_up,
            lambda_scale: quant.trellis_lambda_scale,
            chroma_scale: (quant.trellis_chroma > 0.).then_some(quant.trellis_chroma),
            key_frames: quant.trellis_key >= 0.5,
            key_scale,
            search: TrellisQuantizer::default(),
        })
    }

    /// The trellis of a block of `plane` in a picture that is `inter` or
    /// not, at the plane's λ (`lambdas`: luma, chroma; sample SSE per bit)
    /// times the registered multiples; `None` where it does not run.
    pub(crate) fn for_block<'a>(
        &'a mut self,
        contexts: &'a CoeffContexts,
        plane: nc_common::PlaneIndex,
        inter: bool,
        lambdas: [f32; 2],
    ) -> Option<crate::recon::Trellis<'a>> {
        let ch = usize::from(plane != nc_common::PlaneIndex::Y);
        let lambda = lambdas[ch];
        if !inter && !self.key_frames {
            return None;
        }
        let picture_scale = if inter { 1. } else { self.key_scale };
        let scale = if ch == 0 {
            self.lambda_scale * picture_scale
        } else {
            self.lambda_scale * picture_scale * self.chroma_scale?
        };
        Some(crate::recon::Trellis {
            search: &mut self.search,
            block: BlockTrellis {
                contexts,
                ch,
                lambda: lambda * scale,
                round_up: self.round_up,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coeff::{Bins, code_rect};
    use crate::contexts::Context;

    /// Prices bins at the context states as they stand, as the model does.
    #[derive(Default)]
    struct StaticCost(f32);
    impl Bins for StaticCost {
        fn bin(&mut self, c: &mut Context, value: bool) -> nc_common::decode::Result<bool> {
            self.0 += c.bits(value);
            Ok(value)
        }
        fn bypass(&mut self, value: u32, bits: u32) -> nc_common::decode::Result<u32> {
            self.0 += bits as f32;
            Ok(value)
        }
    }

    /// Transform block sides (each of width and height).
    const TB_SIDES: [usize; 4] = [4, 8, 16, 32];

    /// Summed level rates equal `coeff::code`'s cost without adaptation,
    /// less the signs it hides.
    #[test]
    fn rates_follow_the_coefficient_coder() {
        let mut seed = 7_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        for size in TB_SIDES.into_iter().flat_map(|w| TB_SIDES.map(|h| (w, h))) {
            let (width, height) = size;
            for ch in [0, 1] {
                for density in [3, 6, 9] {
                    let positions = crate::scan::rect_positions(width, height);
                    let mut levels = vec![0_i32; width * height];
                    let span = (width * height).min(80);
                    for &p in &positions[..span] {
                        let v = match next() % 12 {
                            n if n >= density => 0,
                            0..=2 => 1,
                            3 | 4 => 2,
                            5 => 3,
                            _ => 4 + (next() % 40) as i32,
                        };
                        levels[p] = v;
                    }
                    levels[positions[2]] = 5;
                    if width * height > GROUP_SIZE {
                        // An empty group between coded ones.
                        for &p in &positions[GROUP_SIZE..2 * GROUP_SIZE] {
                            levels[p] = 0;
                        }
                    }
                    let contexts = CoeffContexts::new(32, true);
                    let mut cost = StaticCost::default();
                    code_rect(&mut cost, &mut contexts.clone(), &levels, size, ch).unwrap();
                    // `code` charges a sign per level; hidden ones save it.
                    let hidden = hidden_signs(&levels, size) as f32;
                    let decided = Decided::default();
                    let mut model = RateModel::new(&contexts, size, ch, &decided);
                    let last = positions.iter().rposition(|&p| levels[p] != 0).unwrap();
                    let mut bits = model.eob_rate(last);
                    let mut state = model.initial_state();
                    for pos in (0..=last).rev() {
                        let level = levels[positions[pos]].unsigned_abs();
                        let mut b = [0.];
                        model.level_rates(&state, pos, &[level], &mut b);
                        bits += b[0];
                        model.advance(&mut state, pos, level);
                    }
                    bits += model.final_rate(&state) - hidden;
                    assert!(
                        (bits - cost.0).abs() < 1e-3 * cost.0,
                        "size {size:?} ch {ch} density {density}: {bits} vs {}",
                        cost.0
                    );
                }
            }
        }
    }

    /// Groups whose first sign `code` infers from the level parity.
    fn hidden_signs(levels: &[i32], (width, height): (usize, usize)) -> usize {
        crate::scan::rect_positions(width, height)
            .as_chunks::<GROUP_SIZE>()
            .0
            .iter()
            .filter(|span| {
                let nonzero: Vec<usize> =
                    (0..GROUP_SIZE).filter(|&n| levels[span[n]] != 0).collect();
                nonzero.len() > 1
                    && nonzero[nonzero.len() - 1] - nonzero[0] > crate::coeff::SIGN_HIDING_THRESHOLD
            })
            .count()
    }

    /// With λ 0 the trellis keeps the nearest levels; with a huge λ it
    /// empties the block.
    #[test]
    fn lambda_extremes() {
        let contexts = CoeffContexts::new(32, true);
        let quantizer = VvcQuantizer::new(32, 3);
        // Coefficients of 1.4, 0.9 and 1.6 steps: zero is a candidate of each.
        let step = 1. / quantizer.magnitude_in_steps(1);
        let mut coefficients = [0_i32; 64];
        coefficients[0] = (1.4 * step) as i32;
        coefficients[1] = -(0.9 * step) as i32;
        coefficients[8] = (1.6 * step) as i32;
        let nearest: Vec<i32> = coefficients
            .iter()
            .map(|&c| {
                let v = quantizer.magnitude_in_steps(c).round() as i32;
                if c < 0 { -v } else { v }
            })
            .collect();
        assert!(nearest.iter().filter(|&&l| l != 0).count() >= 2);
        let mut trellis = TrellisQuantizer::default();
        for (lambda, expect_empty) in [(0., false), (1e9, true)] {
            let mut levels = nearest.clone();
            let block = BlockTrellis {
                contexts: &contexts,
                ch: 0,
                lambda,
                round_up: true,
            };
            let positions = crate::scan::positions(8);
            let last = positions.iter().rposition(|&p| levels[p] != 0).unwrap();
            let coded = trellis.quantize(
                block,
                &coefficients,
                &quantizer,
                32,
                &mut levels,
                (8, 8),
                last,
            );
            assert_eq!(coded.is_some(), !expect_empty);
            if !expect_empty {
                assert_eq!(levels, nearest);
            }
        }
    }
}
