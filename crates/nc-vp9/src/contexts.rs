//! VP9 above/left syntax contexts: coefficient nonzero flags per 4x4 unit,
//! partition contexts, and the neighbour-derived contexts of the skip,
//! intra/inter, reference and key-frame mode syntax (libvpx
//! `vp9_pred_common.h/.c`, `vp9_onyxc_int.h`, `vp9_blockd.c`; BSD-3-Clause,
//! see `THIRD_PARTY.md`).
//!
//! Above contexts span the frame width and reset per frame; left contexts
//! span one superblock and reset at the start of every superblock row.

use nc_common::PlaneIndex;

use crate::probs::SWITCHABLE_FILTERS;
use crate::transform::TxSize;
use crate::types::{IntraMode, MI_PER_SUPERBLOCK, ModeInfo, ModeInfoGrid};

/// `partition_context_lookup[square bsize]` (above == left): 64x64 .. 8x8.
const PARTITION_CONTEXT: [u8; 4] = [0, 8, 12, 14];

/// Above and left coefficient contexts of every plane, in 4x4 units.
#[derive(Debug, Clone)]
pub(crate) struct EntropyContexts {
    above: [Vec<bool>; 3],
    left: [Vec<bool>; 3],
    partition_above: Vec<u8>,
    partition_left: [u8; MI_PER_SUPERBLOCK],
}

/// Shift from luma to `plane` samples in 4:2:0.
fn subsampling(plane: PlaneIndex) -> usize {
    usize::from(plane != PlaneIndex::Y)
}

impl EntropyContexts {
    /// Contexts of a frame of `mi_cols` 8x8 columns.
    pub(crate) fn new(mi_cols: usize) -> Self {
        let units = |plane: usize| (mi_cols * 2) >> usize::from(plane != 0);
        let left_units = |plane: usize| (MI_PER_SUPERBLOCK * 2) >> usize::from(plane != 0);
        Self {
            above: std::array::from_fn(|p| vec![false; units(p)]),
            left: std::array::from_fn(|p| vec![false; left_units(p)]),
            partition_above: vec![0; mi_cols],
            partition_left: [0; MI_PER_SUPERBLOCK],
        }
    }

    /// Clear everything (start of a frame).
    pub(crate) fn reset_frame(&mut self) {
        self.above.iter_mut().for_each(|a| a.fill(false));
        self.partition_above.fill(0);
        self.reset_left();
    }

    /// Clear the left contexts (start of a superblock row).
    pub(crate) fn reset_left(&mut self) {
        self.left.iter_mut().for_each(|l| l.fill(false));
        self.partition_left.fill(0);
    }

    /// Entropy context (0..=2) of a transform block at plane 4x4 unit
    /// `(ux, uy)`: whether any covered above / left unit was nonzero.
    pub(crate) fn coefficient(&self, plane: PlaneIndex, ux: usize, uy: usize, tx: TxSize) -> usize {
        let p = plane as usize;
        let units = tx.units();
        let left_row = uy % self.left[p].len();
        let any = |flags: &[bool]| usize::from(flags.iter().any(|&f| f));
        any(&self.above[p][ux..(ux + units).min(self.above[p].len())])
            + any(&self.left[p][left_row..left_row + units])
    }

    /// Record a coded transform block's nonzero flag over its units.
    pub(crate) fn set_coefficient(
        &mut self,
        plane: PlaneIndex,
        ux: usize,
        uy: usize,
        tx: TxSize,
        nonzero: bool,
    ) {
        let p = plane as usize;
        let units = tx.units();
        let end = (ux + units).min(self.above[p].len());
        self.above[p][ux..end].fill(nonzero);
        let left_row = uy % self.left[p].len();
        self.left[p][left_row..left_row + units].fill(nonzero);
    }

    /// Clear the coefficient contexts a skipped block covers
    /// (`dec_reset_skip_context`).
    pub(crate) fn clear_block(&mut self, x: usize, y: usize, size: usize) {
        for plane in PlaneIndex::ALL {
            let shift = 2 + subsampling(plane);
            let (ux, uy, units) = (x >> shift, y >> shift, size >> shift);
            let p = plane as usize;
            let end = (ux + units).min(self.above[p].len());
            self.above[p][ux..end].fill(false);
            let left_row = uy % self.left[p].len();
            self.left[p][left_row..left_row + units].fill(false);
        }
    }

    /// Partition context of a node of `log2` at 8x8 position `(row, col)`
    /// (`partition_plane_context`).
    pub(crate) fn partition(&self, row: usize, col: usize, log2: u8) -> usize {
        let level = usize::from(log2 - 3);
        let above = usize::from((self.partition_above[col] >> level) & 1);
        let left = usize::from((self.partition_left[row % MI_PER_SUPERBLOCK] >> level) & 1);
        level * 4 + left * 2 + above
    }

    /// Record an unsplit node (`update_partition_context`).
    pub(crate) fn set_partition(&mut self, row: usize, col: usize, log2: u8) {
        let value = PARTITION_CONTEXT[usize::from(6 - log2)];
        let units = 1usize << (log2 - 3);
        let end = (col + units).min(self.partition_above.len());
        self.partition_above[col..end].fill(value);
        let start = row % MI_PER_SUPERBLOCK;
        self.partition_left[start..start + units].fill(value);
    }
}

/// Skip context: number of skipped neighbours (`vp9_get_skip_context`).
pub(crate) fn skip(grid: &ModeInfoGrid, row: usize, col: usize) -> usize {
    let (above, left) = grid.neighbours(row, col);
    usize::from(above.is_some_and(|m| m.skip)) + usize::from(left.is_some_and(|m| m.skip))
}

/// `switchable_interp` symbol of a filter (`EIGHTTAP`, `_SMOOTH`, `_SHARP`).
pub(crate) fn filter_symbol(filter: nc_dsp::Vp9Filter) -> usize {
    match filter {
        nc_dsp::Vp9Filter::Regular => 0,
        nc_dsp::Vp9Filter::Smooth => 1,
        nc_dsp::Vp9Filter::Sharp => 2,
        nc_dsp::Vp9Filter::Bilinear => unreachable!("bilinear is never switchable"),
    }
}

/// The filter of a `switchable_interp` symbol.
pub(crate) fn symbol_filter(symbol: usize) -> nc_dsp::Vp9Filter {
    [
        nc_dsp::Vp9Filter::Regular,
        nc_dsp::Vp9Filter::Smooth,
        nc_dsp::Vp9Filter::Sharp,
    ][symbol]
}

/// Filter context (`get_pred_context_switchable_interp`): the filter the
/// inter neighbours above and left agree on, else 3.
pub(crate) fn switchable_interp(grid: &ModeInfoGrid, row: usize, col: usize) -> usize {
    let (above, left) = grid.neighbours(row, col);
    let kind = |m: Option<&ModeInfo>| {
        m.filter(|m| m.is_inter())
            .map_or(SWITCHABLE_FILTERS, |m| filter_symbol(m.filter))
    };
    match (kind(left), kind(above)) {
        (l, a) if l == a => l,
        (SWITCHABLE_FILTERS, a) => a,
        (l, SWITCHABLE_FILTERS) => l,
        _ => SWITCHABLE_FILTERS,
    }
}

/// Intra/inter context (`get_intra_inter_context`).
pub(crate) fn intra_inter(grid: &ModeInfoGrid, row: usize, col: usize) -> usize {
    match grid.neighbours(row, col) {
        (Some(a), Some(l)) => {
            let (a, l) = (!a.is_inter(), !l.is_inter());
            if a && l { 3 } else { usize::from(a || l) }
        }
        (Some(edge), None) | (None, Some(edge)) => 2 * usize::from(!edge.is_inter()),
        (None, None) => 0,
    }
}

/// Context of the first single-reference bit
/// (`vp9_get_pred_context_single_ref_p1`, `VP9 §6.4.18`): `0` codes
/// `LAST_FRAME`, `1` codes `GOLDEN_FRAME` or `ALTREF_FRAME`.
pub(crate) fn single_ref_p1(grid: &ModeInfoGrid, row: usize, col: usize) -> usize {
    let last = |m: &ModeInfo| usize::from(m.is_inter() && !m.secondary_ref);
    match grid.neighbours(row, col) {
        (Some(a), Some(l)) => match (a.is_inter(), l.is_inter()) {
            (false, false) => 2,
            (true, false) => 4 * last(a),
            (false, true) => 4 * last(l),
            (true, true) => 2 * last(a) + 2 * last(l),
        },
        (Some(edge), None) | (None, Some(edge)) => {
            if edge.is_inter() {
                4 * last(edge)
            } else {
                2
            }
        }
        (None, None) => 2,
    }
}

/// Context of the second single-reference bit
/// (`vp9_get_pred_context_single_ref_p2`, `VP9 §6.4.18`) when
/// `single_ref_p1` is `1`: `0` codes `GOLDEN_FRAME`, `1` codes
/// `ALTREF_FRAME`. In our subset every non-`LAST` inter block uses
/// `GOLDEN_FRAME` (`secondary_ref == true`).
pub(crate) fn single_ref_p2(grid: &ModeInfoGrid, row: usize, col: usize) -> usize {
    let is_last = |m: &ModeInfo| m.is_inter() && !m.secondary_ref;
    let is_golden = |m: &ModeInfo| m.is_inter() && m.secondary_ref;
    let edge = |e: &ModeInfo| {
        if is_last(e) {
            3
        } else {
            4 * usize::from(is_golden(e))
        }
    };
    match grid.neighbours(row, col) {
        (Some(a), Some(l)) => match (a.is_inter(), l.is_inter()) {
            (false, false) => 2,
            (true, false) => edge(a),
            (false, true) => edge(l),
            (true, true) => {
                if is_last(a) && is_last(l) {
                    3
                } else if is_last(a) {
                    4 * usize::from(is_golden(l))
                } else if is_last(l) {
                    4 * usize::from(is_golden(a))
                } else {
                    2 * usize::from(is_golden(a)) + 2 * usize::from(is_golden(l))
                }
            }
        },
        (Some(e), None) | (None, Some(e)) => {
            if !e.is_inter() || is_last(e) {
                2
            } else {
                4 * usize::from(is_golden(e))
            }
        }
        (None, None) => 2,
    }
}

/// Transform-size context (`get_tx_size_context`): whether the neighbours'
/// transforms (the largest for skipped or missing ones) sum above `max`.
pub(crate) fn tx_size(grid: &ModeInfoGrid, row: usize, col: usize, max: TxSize) -> usize {
    let size = |m: Option<&ModeInfo>| m.filter(|m| !m.skip).map(|m| m.tx_size as usize);
    let (above, left) = grid.neighbours(row, col);
    let (mut a, mut l) = (size(above), size(left));
    let max_size = max as usize;
    if above.is_some() && a.is_none() {
        a = Some(max_size);
    }
    if left.is_some() && l.is_none() {
        l = Some(max_size);
    }
    let (a, l) = match (a, l) {
        (Some(a), Some(l)) => (a, l),
        (Some(a), None) => (a, a),
        (None, Some(l)) => (l, l),
        (None, None) => (max_size, max_size),
    };
    usize::from(a + l > max_size)
}

/// Above and left luma modes for key-frame mode probabilities
/// (`vp9_above_block_mode` / `vp9_left_block_mode`): DC where missing.
pub(crate) fn key_frame_modes(
    grid: &ModeInfoGrid,
    row: usize,
    col: usize,
) -> (IntraMode, IntraMode) {
    let (above, left) = grid.neighbours(row, col);
    let mode = |m: Option<&ModeInfo>| {
        m.filter(|m| !m.is_inter())
            .map_or(IntraMode::Dc, |m| m.y_mode)
    };
    (mode(above), mode(left))
}
