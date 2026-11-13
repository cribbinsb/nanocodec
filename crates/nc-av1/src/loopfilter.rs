//! AV1 loop filter (deblocking, AV1 §7.14): which edges of the frame are
//! filtered, with which length and thresholds, and the frame-header syntax
//! (`loop_filter_params()`, §5.9.11). The per-line filters are nc-dsp's
//! [`Dsp::av1_loop_filter_edge`].
//!
//! In our subset every transform covers its whole block (`TX_MODE_LARGEST`,
//! square blocks of 8x8 and up), so every transform edge is a block edge
//! and the "both sides skipped" exemption (which applies only inside a
//! block) never fires. There are no segments, no `delta_lf` and
//! `loop_filter_delta_enabled = 0`, so a plane/direction has one level for
//! the whole frame. The filter runs on the reconstruction before CDEF; for
//! each plane all vertical edges are filtered first, then all horizontal
//! ones (§7.14.1).
//!
//! The edge rules follow the specification; libaom's
//! `av1/common/av1_loopfilter.c` (BSD-2-Clause, see `THIRD_PARTY.md`) served
//! as the readable reference.

use nc_common::{BitWriter, Picture, PlaneIndex};
use nc_dsp::{AV1_EDGE_LINES, Av1EdgeFilter, Av1FilterLength, Dsp};

use crate::modeinfo::{MI_LOG2, ModeInfo, ModeInfoGrid};

/// `MAX_LOOP_FILTER`: largest filter level.
pub const MAX_LOOP_FILTER: u8 = 63;
/// Smallest transform side (4x4) and the edge spacing the filter visits.
const EDGE_STEP: usize = 4;
/// Level bits (`loop_filter_level[i]`, f(6)) and sharpness bits (f(3)).
const LEVEL_BITS: u32 = 6;
const SHARPNESS_BITS: u32 = 3;
/// `blimit = 2 · (level + 2) + limit` (AV1 §7.14.4).
const BLIMIT_LEVEL_OFFSET: u32 = 2;
/// `thresh = level >> 4`.
const THRESH_SHIFT: u32 = 4;
/// Sharpness above which the inner limit is shifted by two, not one.
const SHARPNESS_DOUBLE_SHIFT: u8 = 4;
/// `limit` is capped at `9 − sharpness` when sharpness is nonzero.
const SHARPNESS_LIMIT_BASE: u8 = 9;

/// The frame's loop filter levels and sharpness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoopFilterParams {
    /// `loop_filter_level[0..4]`: luma vertical edges, luma horizontal
    /// edges, U, V.
    pub level: [u8; 4],
    /// `loop_filter_sharpness` (0..=7).
    pub sharpness: u8,
}

impl LoopFilterParams {
    /// No filtering.
    pub const OFF: Self = Self {
        level: [0; 4],
        sharpness: 0,
    };

    /// With both luma levels zero nothing is filtered, chroma included
    /// (the chroma levels are not even coded).
    #[must_use]
    pub fn is_on(self) -> bool {
        self.level[0] != 0 || self.level[1] != 0
    }

    /// `loop_filter_params()` (AV1 §5.9.11) for a 4:2:0 stream, with
    /// `loop_filter_delta_enabled = 0`.
    pub fn write(self, w: &mut BitWriter) {
        w.put_bits(u32::from(self.level[0]), LEVEL_BITS);
        w.put_bits(u32::from(self.level[1]), LEVEL_BITS);
        if self.is_on() {
            w.put_bits(u32::from(self.level[2]), LEVEL_BITS);
            w.put_bits(u32::from(self.level[3]), LEVEL_BITS);
        }
        w.put_bits(u32::from(self.sharpness), SHARPNESS_BITS);
        w.put_bit(false); // loop_filter_delta_enabled
    }

    /// The level of `plane` for edges of direction `vertical`.
    fn plane_level(self, plane: PlaneIndex, vertical: bool) -> u8 {
        match plane {
            PlaneIndex::Y if vertical => self.level[0],
            PlaneIndex::Y => self.level[1],
            PlaneIndex::U => self.level[2],
            PlaneIndex::V => self.level[3],
        }
    }

    /// `limit`, `blimit` and `thresh` of `level` (AV1 §7.14.4).
    fn thresholds(self, level: u8, length: Av1FilterLength) -> Av1EdgeFilter {
        let shift =
            u32::from(self.sharpness > 0) + u32::from(self.sharpness > SHARPNESS_DOUBLE_SHIFT);
        let mut limit = level >> shift;
        if self.sharpness > 0 {
            limit = limit.min(SHARPNESS_LIMIT_BASE - self.sharpness);
        }
        let limit = limit.max(1);
        Av1EdgeFilter {
            length,
            limit,
            blimit: (2 * (u32::from(level) + BLIMIT_LEVEL_OFFSET) + u32::from(limit)) as u8,
            thresh: level >> THRESH_SHIFT,
        }
    }
}

/// libaom's `LPF_PICK_FROM_Q` level fit for 8-bit video
/// (`av1/encoder/picklpf.c`, BSD-2-Clause): `round(q · m + c)` in 18-bit
/// fixed point, from the frame's AC quantiser step `q`. Inter frames use
/// libaom's real-time slope (0.0459, "boosted").
const PICK_FROM_Q_SHIFT: u32 = 18;
const KEY_SLOPE: i64 = 17_563;
const KEY_OFFSET: i64 = -421_574;
const INTER_SLOPE: i64 = 12_034;
const INTER_OFFSET: i64 = 650_707;

/// Filter level for a frame whose AC quantiser step is `ac_q`.
#[must_use]
pub fn level_from_q(ac_q: i32, key: bool) -> u8 {
    let (slope, offset) = if key {
        (KEY_SLOPE, KEY_OFFSET)
    } else {
        (INTER_SLOPE, INTER_OFFSET)
    };
    let fixed = i64::from(ac_q) * slope + offset;
    let level = (fixed + (1 << (PICK_FROM_Q_SHIFT - 1))) >> PICK_FROM_Q_SHIFT;
    level.clamp(0, i64::from(MAX_LOOP_FILTER)) as u8
}

/// Filter `recon` in place. `width` x `height` is the frame size: edges
/// at or beyond it are not filtered.
pub fn apply(
    dsp: &Dsp,
    recon: &mut Picture,
    grid: &ModeInfoGrid,
    params: LoopFilterParams,
    width: usize,
    height: usize,
) {
    if !params.is_on() {
        return;
    }
    // The transforms' positions depend only on the subsampling: found once
    // for luma and once for both chroma planes, and used by both passes.
    let mut chroma = None;
    let luma = transform_starts(grid, 0, (width, height));
    for plane in PlaneIndex::ALL {
        if plane != PlaneIndex::Y && params.plane_level(plane, true) == 0 {
            continue;
        }
        let starts = if plane == PlaneIndex::Y {
            &luma
        } else {
            chroma.get_or_insert_with(|| transform_starts(grid, 1, (width, height)))
        };
        for vertical in [true, false] {
            filter_plane_edges(
                dsp,
                recon,
                grid,
                params,
                plane,
                vertical,
                starts,
                (width, height),
            );
        }
    }
}

/// The top-left sample and side of every transform of a plane with
/// subsampling `ss`, in the filtering order: 4-line stripes from the top,
/// left to right, each transform at the stripe holding its top row.
/// Transforms tile the plane and are aligned to their size, so the walk
/// meets each one once at its top-left stripe.
fn transform_starts(
    grid: &ModeInfoGrid,
    ss: usize,
    (width, height): (usize, usize),
) -> Vec<(usize, usize, usize)> {
    let (plane_width, plane_height) = ((width + ss) >> ss, (height + ss) >> ss);
    let mut starts = Vec::new();
    for y in (0..plane_height).step_by(EDGE_STEP) {
        let mut x = 0;
        while x < plane_width {
            let side = transform_side(block_at(grid, x, y, ss), ss);
            if y % side == 0 {
                starts.push((x, y, side));
            }
            x += side;
        }
    }
    starts
}

/// One pass (all vertical or all horizontal edges) over one plane: the
/// left (top) side of every transform in `starts`, filtered whole, in runs
/// of equal filter length, which lets the kernel process many lines at once.
#[allow(clippy::too_many_arguments)]
fn filter_plane_edges(
    dsp: &Dsp,
    recon: &mut Picture,
    grid: &ModeInfoGrid,
    params: LoopFilterParams,
    plane: PlaneIndex,
    vertical: bool,
    starts: &[(usize, usize, usize)],
    (width, height): (usize, usize),
) {
    let ss = usize::from(plane != PlaneIndex::Y);
    let (plane_width, plane_height) = ((width + ss) >> ss, (height + ss) >> ss);
    let level = params.plane_level(plane, vertical);
    if level == 0 {
        return;
    }
    let mut view = recon.plane_mut(plane).view_mut();
    for &(x, y, side) in starts {
        let on_edge = if vertical { x > 0 } else { y > 0 };
        if !on_edge {
            continue;
        }
        let (start, end) = if vertical {
            (y, (y + side).min(plane_height))
        } else {
            (x, (x + side).min(plane_width))
        };
        // The transform side across the edge at position `along`.
        let previous_side = |along: usize| {
            let previous = if vertical {
                block_at(grid, x - 1, along, ss)
            } else {
                block_at(grid, along, y - 1, ss)
            };
            transform_side(previous, ss)
        };
        let mut filter_run = |run_start: usize, run_end: usize, length| {
            let origin = if vertical {
                (x, run_start)
            } else {
                (run_start, y)
            };
            dsp.av1_loop_filter_edge(
                &mut view,
                origin,
                vertical,
                (run_end - run_start).next_multiple_of(EDGE_STEP),
                params.thresholds(level, length),
            );
        };
        for_each_run(start..end, side, plane, previous_side, &mut filter_run);
    }
}

/// Call `filter_run(start, end, length)` for each run of equal filter
/// length along the edge `span` of a transform of side `side`. The length
/// depends on the smaller transform side across the edge, and neighbouring
/// transforms are aligned to their size, so it can only change where a
/// neighbour ends: one lookup per neighbouring transform, not per 4 lines.
fn for_each_run(
    span: core::ops::Range<usize>,
    side: usize,
    plane: PlaneIndex,
    previous_side: impl Fn(usize) -> usize,
    filter_run: &mut impl FnMut(usize, usize, Av1FilterLength),
) {
    let length_of = |previous: usize| filter_length(plane, side.min(previous));
    let mut previous = previous_side(span.start);
    let (mut run_start, mut run_length) = (span.start, length_of(previous));
    let mut along = span.start;
    loop {
        // The end of the neighbouring transform covering `along`.
        let next = ((along / previous + 1) * previous).min(span.end);
        if next == span.end {
            filter_run(run_start, span.end, run_length);
            return;
        }
        previous = previous_side(next);
        let length = length_of(previous);
        if length != run_length {
            filter_run(run_start, next, run_length);
            (run_start, run_length) = (next, length);
        }
        along = next;
    }
}

/// The mode info covering plane sample (`x`, `y`). Chroma reads the
/// bottom-right 4x4 of the co-located 8x8 (§7.14.2: `MiRow | ssy`).
fn block_at(grid: &ModeInfoGrid, x: usize, y: usize, ss: usize) -> &ModeInfo {
    let mi_row = ss | ((y << ss) >> MI_LOG2);
    let mi_col = ss | ((x << ss) >> MI_LOG2);
    grid.at(mi_row.min(grid.mi_rows - 1), mi_col.min(grid.mi_cols - 1))
}

/// Side of the block in plane samples.
fn block_side(info: &ModeInfo, ss: usize) -> usize {
    (info.size_mi() << MI_LOG2) >> ss
}

/// Transform side in plane samples: the cell's luma transform (the block,
/// or a leaf of its inter transform tree), and for 4:2:0 chroma half the
/// block (`get_max_uv_txsize`), never below 4.
fn transform_side(info: &ModeInfo, ss: usize) -> usize {
    if ss == 0 {
        1 << info.tx_log2
    } else {
        block_side(info, ss).max(EDGE_STEP)
    }
}

/// Filter length from the smaller transform side across the edge
/// (AV1 §7.14.3): luma 4 / 8 / 14 taps for 4 / 8 / 16+ samples, chroma
/// 4 / 6 for 4 / 8+.
fn filter_length(plane: PlaneIndex, side: usize) -> Av1FilterLength {
    const SIDE_4: usize = 4;
    const SIDE_8: usize = 8;
    match (plane == PlaneIndex::Y, side) {
        (_, SIDE_4) => Av1FilterLength::Four,
        (false, _) => Av1FilterLength::Six,
        (true, SIDE_8) => Av1FilterLength::Eight,
        (true, _) => Av1FilterLength::Fourteen,
    }
}

/// Lines one edge call filters, as the plane-walk step.
const _: () = assert!(AV1_EDGE_LINES == EDGE_STEP);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_from_q_matches_the_libaom_fit() {
        // Inter: round(140 * 12034 + 650707 >> 18) = round(8.91) = 9.
        assert_eq!(level_from_q(140, false), 9);
        // Key frames at fine quantisers round down to zero.
        assert_eq!(level_from_q(20, true), 0);
        assert_eq!(level_from_q(4000, false), MAX_LOOP_FILTER);
    }

    #[test]
    fn thresholds_follow_the_sharpness_rules() {
        let p = LoopFilterParams {
            level: [10; 4],
            sharpness: 0,
        };
        let t = p.thresholds(10, Av1FilterLength::Four);
        assert_eq!((t.limit, t.blimit, t.thresh), (10, 34, 0));
        let p = LoopFilterParams {
            level: [40; 4],
            sharpness: 5,
        };
        // 40 >> 2 = 10, capped at 9 - 5 = 4.
        let t = p.thresholds(40, Av1FilterLength::Four);
        assert_eq!((t.limit, t.blimit, t.thresh), (4, 88, 2));
        let t = p.thresholds(1, Av1FilterLength::Four);
        assert_eq!(t.limit, 1);
    }

    #[test]
    fn chroma_levels_are_coded_only_when_luma_filters() {
        let mut off = BitWriter::new();
        LoopFilterParams::OFF.write(&mut off);
        let mut on = BitWriter::new();
        LoopFilterParams {
            level: [5, 6, 7, 8],
            sharpness: 1,
        }
        .write(&mut on);
        assert_eq!(off.bit_len(), 6 + 6 + 3 + 1);
        assert_eq!(on.bit_len(), 4 * 6 + 3 + 1);
    }
}
