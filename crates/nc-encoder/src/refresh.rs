//! Intra refresh: a column of intra blocks sweeping across the picture, so
//! a decoder that lost a frame (or joined mid-stream) recovers without a key
//! frame (`docs/ARCHITECTURE.md` §6.1, GDR-style).
//!
//! Each inter frame codes one *intra range* of columns intra; it advances
//! left to right by a whole number of the codec's smallest blocks per frame
//! (nodes across the range's edges split down to them) and restarts
//! after reaching the right edge. Everything left of the range (the *clean*
//! region) was refreshed earlier in the sweep, and stays clean only if it
//! never reads the not-yet-refreshed (*dirty*) part of its reference:
//!
//! - motion vectors of clean blocks keep their whole interpolation
//!   footprint, luma and chroma, left of the reference's clean edge
//!   ([`RefreshWindow::allows`]);
//! - that edge is pulled in by the backend's loop-filter reach
//!   ([`CodecBackend::loop_filter_reach`](crate::CodecBackend::loop_filter_reach)):
//!   deblocking, SAO and CDEF across the clean/dirty boundary leave dirty
//!   samples just inside it;
//! - so the intra range starts that far before the previous frame's range
//!   ended, which keeps the zero vector (a copy, without interpolation
//!   reach) legal for every clean block: re-coding more than that as intra
//!   only adds intra area;
//! - a sweep takes exactly `period` frames (fewer only when the picture has
//!   fewer block columns), its column boundaries spread evenly: a
//!   sweep that finished early would start the next one early, coding more
//!   intra area per frame than the period asks for.
//! - intra blocks must not read above-right samples past the range's
//!   right edge, which are dirty ([`RefreshWindow::reads_dirty_above_right`];
//!   H.264's macroblocks: [`RefreshWindow::on_dirty_edge`]);
//! - backends must not use motion predicted from the reference's motion
//!   field (HEVC TMVP): after a loss the decoder's field differs, and so
//!   would the vector a merge index selects.
//!
//! The dirty region is unconstrained. The frame starting a sweep is marked
//! as a recovery point (H.264 and HEVC recovery-point SEI, VVC GDR
//! pictures; [`RefreshWindow::recovery_frames`]), so a decoder joining there
//! knows when its pictures become correct. `VVdeC` (as VTM) withholds the
//! output of the first GDR picture's recovering pictures even in a stream
//! that started with an IDR picture (`DecLibParser.cpp` `m_gdrRecovered`,
//! d261469), where H.266 §8.1.1/§8.1.2 output them
//! (`NoOutputBeforeRecoveryFlag` 0): `VVdeC`-based players skip the first
//! sweep after a stream start (owner decision 2026-11-27: GDR kept, the
//! behaviour documented).

use std::ops::Range;

use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
use crate::partition::Block;

/// Luma samples an interpolated prediction reads beyond the block on either
/// side, in luma units, over every codec's luma and (doubled) chroma filter:
/// AV1 chroma uses 8-tap filters, 4 samples after the block in chroma = 8 in
/// luma.
pub const MAX_INTERPOLATION_REACH: usize = 8;
/// Granularity the intra range's start is aligned down to (the smallest
/// leaf block of any codec).
const RANGE_ALIGNMENT: usize = 8;
/// Fewest frames a sweep may take.
pub const MIN_REFRESH_PERIOD: u32 = 2;

/// Intra-refresh settings ([`crate::Encoder::set_intra_refresh`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraRefresh {
    /// Frames one sweep across the picture takes (at least
    /// [`MIN_REFRESH_PERIOD`]): a decoder recovers at most this many frames
    /// after a loss. The sweep's block-column boundaries are spread
    /// evenly over the period.
    pub period: u32,
}

/// Where a block lies relative to this frame's refresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshRegion {
    /// Refreshed earlier in the sweep: inter prediction is constrained.
    Clean,
    /// Overlaps the intra range: coded intra.
    Intra,
    /// Not yet refreshed in this sweep: unconstrained.
    Dirty,
}

/// One frame's refresh geometry, in luma samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshWindow {
    /// Columns coded intra in this frame.
    pub intra: Range<usize>,
    /// Clean blocks' predictions may read reference columns left of this.
    pub clean_limit: usize,
    /// Picture width (coded).
    pub width: usize,
    /// This frame starts a sweep (its left neighbours were refreshed in the
    /// previous sweep); recovery completes when the sweep ends.
    pub sweep_start: bool,
    /// Coded frames from this one to the one that completes its sweep, in
    /// output order (the sweep's frames are this far apart in a layered
    /// structure): a decoder starting at a sweep start shows correct
    /// pictures from that frame on.
    pub recovery_frames: u32,
}

impl RefreshWindow {
    /// The region `block` lies in.
    #[must_use]
    pub fn region(&self, block: &Block) -> RefreshRegion {
        let right = block.x + block.size();
        if block.x < self.intra.end && right > self.intra.start {
            RefreshRegion::Intra
        } else if right <= self.intra.start {
            RefreshRegion::Clean
        } else {
            RefreshRegion::Dirty
        }
    }

    /// Whether `node` straddles an edge of the intra range (a partition
    /// walk splits it rather than coding clean or dirty blocks intra).
    #[must_use]
    pub fn straddles_intra_edge(&self, node: &Block) -> bool {
        let crosses = |edge: usize| node.x < edge && node.x + node.size() > edge;
        crosses(self.intra.start) || crosses(self.intra.end)
    }

    /// Whether the `width`-wide area at `x` may be predicted with `mv`: always
    /// outside the clean region; inside it, the prediction's footprint must
    /// end left of [`RefreshWindow::clean_limit`].
    #[must_use]
    pub fn allows_area(&self, x: usize, width: usize, mv: Mv) -> bool {
        if x + width > self.intra.start {
            return true;
        }
        let units = MV_UNITS_PER_SAMPLE;
        // Whole-sample in luma and in (half-resolution) chroma: a copy.
        let copy = i32::from(mv.x).rem_euclid(2 * units) == 0;
        let reach = if copy {
            0
        } else {
            MAX_INTERPOLATION_REACH as i64
        };
        let (dx, _) = mv.full_pel();
        let right = (x + width) as i64 + i64::from(dx) + reach;
        right <= self.clean_limit as i64
    }

    /// [`RefreshWindow::allows_area`] for a whole block.
    #[must_use]
    pub fn allows(&self, block: &Block, mv: Mv) -> bool {
        self.allows_area(block.x, block.size(), mv)
    }

    /// Whether intra `block` ends at the intra range's right edge with dirty
    /// samples beyond it: it must not predict from above-right samples.
    /// For H.264, whose intra predictions read at most 8 samples past the
    /// macroblock (`Intra_8x8`, §8.3.2.2), so only the edge macroblock reads
    /// dirty ones; codecs whose top reference row is twice the block's
    /// width use [`RefreshWindow::reads_dirty_above_right`].
    #[must_use]
    pub fn on_dirty_edge(&self, block: &Block) -> bool {
        block.x + block.size() == self.intra.end && self.intra.end < self.width
    }

    /// Whether intra `block` (of any shape) has above-right samples in the
    /// dirty region: codecs whose intra prediction reads a top reference
    /// row twice the block's width (H.266 §8.4.5.2.8: `refW = 2 * nTbW`)
    /// read dirty samples past the range's right edge also from blocks
    /// that end before it.
    #[must_use]
    pub fn reads_dirty_above_right(&self, block: &Block) -> bool {
        self.intra.end < self.width
            && block.x < self.intra.end
            && block.x + 2 * block.width() > self.intra.end
    }
}

/// The sweep's progress between frames.
#[derive(Debug, Clone)]
pub(crate) struct RefreshSweep {
    settings: IntraRefresh,
    width: usize,
    /// Width of the columns the range is made of: the smallest block.
    unit: usize,
    /// Frames one sweep takes.
    frames: usize,
    /// Coded frames from one frame of the sweep to the next.
    spacing: u64,
    /// Frames of this sweep coded so far.
    done: usize,
    /// Luma samples the intra range starts before the previous range's end.
    overlap: usize,
    /// Luma samples loop filtering contaminates inside the clean edge.
    filter_reach: usize,
}

impl RefreshSweep {
    /// A sweep over a `width`-wide picture of `unit`-wide columns, for
    /// a codec whose loop filters reach `filter_reach` samples.
    pub fn new(settings: IntraRefresh, width: usize, unit: usize, filter_reach: usize) -> Self {
        let overlap = filter_reach.next_multiple_of(RANGE_ALIGNMENT);
        Self {
            settings,
            width,
            unit,
            frames: 0,
            spacing: 1,
            done: 0,
            overlap,
            filter_reach,
        }
        .with_frame_spacing(1)
    }

    /// This sweep, advancing on every `spacing`-th frame only (the base
    /// layer of a layered structure): it takes as many of those frames as
    /// cover the period, at least [`MIN_REFRESH_PERIOD`].
    pub fn with_frame_spacing(self, spacing: u64) -> Self {
        let columns = self.width.div_ceil(self.unit);
        let period = u64::from(self.settings.period).div_ceil(spacing.max(1));
        let frames = usize::try_from(period).unwrap_or(usize::MAX);
        Self {
            frames: frames.max(MIN_REFRESH_PERIOD as usize).min(columns),
            spacing: spacing.max(1),
            ..self
        }
    }

    /// The settings in use.
    pub fn settings(&self) -> IntraRefresh {
        self.settings
    }

    /// A key frame refreshes everything: the next frame starts a sweep.
    pub fn restart(&mut self) {
        self.done = 0;
    }

    /// The window of the next inter frame, advancing the sweep.
    pub fn next_window(&mut self) -> RefreshWindow {
        if self.done >= self.frames {
            self.done = 0;
        }
        let columns = self.width.div_ceil(self.unit);
        // Columns refreshed after `frames` frames of the sweep.
        let boundary = |frames: usize| (frames * columns).div_ceil(self.frames) * self.unit;
        let previous_end = boundary(self.done);
        self.done += 1;
        let end = boundary(self.done).min(self.width);
        let start = if previous_end == 0 {
            0
        } else {
            previous_end.saturating_sub(self.overlap) / RANGE_ALIGNMENT * RANGE_ALIGNMENT
        };
        RefreshWindow {
            intra: start..end,
            clean_limit: previous_end.saturating_sub(self.filter_reach),
            width: self.width,
            sweep_start: previous_end == 0,
            recovery_frames: u32::try_from((self.frames as u64 - 1) * self.spacing)
                .unwrap_or(u32::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IntraRefresh, RefreshRegion, RefreshSweep};
    use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
    use crate::partition::Block;

    const UNITS: i16 = MV_UNITS_PER_SAMPLE as i16;

    #[test]
    fn sweeps_columns_with_an_overlap() {
        // 320 wide, 64-sample columns (5), 3 frames per sweep:
        // two columns per frame.
        let mut sweep = RefreshSweep::new(IntraRefresh { period: 3 }, 320, 64, 16);
        let first = sweep.next_window();
        assert_eq!((first.intra.clone(), first.sweep_start), (0..128, true));
        assert_eq!(first.recovery_frames, 2);
        let second = sweep.next_window();
        // Starts the filter reach (16) before 128.
        assert_eq!((second.intra.clone(), second.clean_limit), (112..256, 112));
        let third = sweep.next_window();
        assert_eq!(third.intra, 240..320);
        assert!(sweep.next_window().sweep_start);
    }

    #[test]
    fn sweeps_take_exactly_the_period() {
        // 80 macroblock columns in 30 frames: 2 or 3 per frame, never ending
        // early (which would start the next sweep early).
        let mut sweep = RefreshSweep::new(IntraRefresh { period: 30 }, 1280, 16, 8);
        let mut end = 0;
        for frame in 0..30 {
            let window = sweep.next_window();
            assert_eq!(window.sweep_start, frame == 0);
            let advance = window.intra.end - end;
            assert!(advance == 32 || advance == 48, "frame {frame}: {advance}");
            end = window.intra.end;
        }
        assert_eq!(end, 1280);
        assert!(sweep.next_window().sweep_start);
    }

    #[test]
    fn base_layer_sweeps_cover_the_period() {
        // Period 12 on a base layer every 4 frames: 3 base frames a sweep.
        let mut sweep =
            RefreshSweep::new(IntraRefresh { period: 12 }, 320, 64, 16).with_frame_spacing(4);
        assert!(sweep.next_window().sweep_start);
        sweep.next_window();
        assert_eq!(sweep.next_window().intra.end, 320);
        assert!(sweep.next_window().sweep_start);
        // Never fewer than two frames a sweep.
        let mut short =
            RefreshSweep::new(IntraRefresh { period: 4 }, 320, 64, 16).with_frame_spacing(4);
        assert_eq!(short.next_window().intra.end, 192);
    }

    #[test]
    fn clean_blocks_keep_their_footprint_left_of_the_limit() {
        let mut sweep = RefreshSweep::new(IntraRefresh { period: 3 }, 320, 64, 16);
        sweep.next_window();
        let window = sweep.next_window(); // intra 112..256, limit 112
        let block = |x| Block::square(x, 0, 4);
        assert_eq!(window.region(&block(80)), RefreshRegion::Clean);
        assert_eq!(window.region(&block(104)), RefreshRegion::Intra);
        assert_eq!(window.region(&block(256)), RefreshRegion::Dirty);
        // Zero is always legal for a clean block: 96 + 0 <= 112.
        assert!(window.allows(&block(80), Mv::ZERO));
        // 16 whole samples right: 96 + 16 = 112, still legal.
        assert!(window.allows(
            &block(80),
            Mv {
                x: 16 * UNITS,
                y: 0
            }
        ));
        // A half-sample vector adds the interpolation reach.
        assert!(window.allows(
            &block(80),
            Mv {
                x: 2 * UNITS + 2,
                y: 0
            }
        ));
        assert!(!window.allows(
            &block(80),
            Mv {
                x: 10 * UNITS + 2,
                y: 0
            }
        ));
        // Odd whole samples need chroma interpolation.
        assert!(!window.allows(
            &block(80),
            Mv {
                x: 13 * UNITS,
                y: 0
            }
        ));
        assert!(window.allows(&block(256), Mv { x: 500, y: 0 }));
        assert!(window.on_dirty_edge(&block(240)));
        // A 16-wide block ending 16 before the range's end (256) reads
        // above-right samples up to it; one ending 8 before reads past it.
        assert!(!window.reads_dirty_above_right(&block(224)));
        assert!(window.reads_dirty_above_right(&block(232)));
        assert!(window.reads_dirty_above_right(&block(240)));
    }
}
