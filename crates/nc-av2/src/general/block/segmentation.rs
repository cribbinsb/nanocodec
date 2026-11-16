//! Segment ids, and the per-4x4-unit segment maps they are predicted from
//! and the loop filters read.
//!
//! Ported from dav2d (commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`): `decode_b`'s segment id reading before and after
//! the residual skip flag, `get_prev_frame_segid`, `neg_deinterleave`
//! (`src/decode.c`), `get_cur_frame_segid` (`src/env.h`).
//!
//! A frame with segmentation gives every block a segment id: read before
//! the skip flag when a segment skips or forces global motion (the id then
//! gates the block's syntax), else after it; taken from the primary
//! reference's map when the frame does not update its map, or when a block
//! signals temporal prediction; otherwise coded as a difference from the
//! spatial prediction of its above, left and above-left neighbours. A
//! chroma-tree block takes the id its luma block recorded.

use super::Tile;
use super::context::left_index;
use super::size::BlockSize;
use crate::general::frame_header::{FrameHeader, Segmentation};
use crate::general::sequence::MAX_SEGMENTS;
use nc_common::decode::{DecodeError, Result};

/// Segment-id differences the extended alphabet adds above the first
/// eight (dav2d `ext_flag << 3`).
const EXTENDED_STEP: usize = 8;
/// Map sides are padded to 256x256 superblocks (64 units), as dav2d's
/// `b4_stride` and `sb256h`, so blocks overhanging the frame stay inside.
const PADDING_UNITS: usize = 64;

/// A segment id per 4x4 unit (luma, or chroma for the loop filters' chroma
/// map), padded to whole 256x256 superblocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentMap {
    /// The unpadded size in units, which a later frame's must equal to
    /// predict from it (dav2d `ref_w == f->bw && ref_h == f->bh`).
    columns: usize,
    rows: usize,
    stride: usize,
    ids: Vec<u8>,
}

impl SegmentMap {
    /// Segment 0 everywhere, for a plane of `columns` x `rows` units.
    #[must_use]
    pub fn new(columns: usize, rows: usize) -> Self {
        let stride = columns.next_multiple_of(PADDING_UNITS);
        Self {
            columns,
            rows,
            stride,
            ids: vec![0; stride * rows.next_multiple_of(PADDING_UNITS)],
        }
    }

    /// The segment of unit (`row`, `column`).
    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> u8 {
        self.ids[row * self.stride + column]
    }

    /// Set `columns` x `rows` units from (`row`, `column`) to `id`.
    fn fill(&mut self, (row, column): (usize, usize), (columns, rows): (usize, usize), id: u8) {
        for y in row..row + rows {
            let start = y * self.stride + column;
            self.ids[start..start + columns].fill(id);
        }
    }

    /// The lowest segment over `columns` x `rows` units from (`row`,
    /// `column`) (dav2d `get_prev_frame_segid`).
    fn lowest(&self, (row, column): (usize, usize), (columns, rows): (usize, usize)) -> u8 {
        (row..row + rows)
            .flat_map(|y| {
                let start = y * self.stride + column;
                self.ids[start..start + columns].iter().copied()
            })
            .min()
            .unwrap_or(0)
    }

    /// Whether `other` covers a frame of this map's size.
    #[must_use]
    pub fn same_size(&self, other: &Self) -> bool {
        (self.columns, self.rows) == (other.columns, other.rows)
    }
}

/// The frame's luma and chroma segment maps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentMaps {
    /// Per luma 4x4 unit: the map later frames predict from (dav2d
    /// `cur_segmap`).
    pub luma: SegmentMap,
    /// Per chroma 4x4 unit, for the chroma loop filters (dav2d
    /// `lf.segmap_uv`).
    pub chroma: SegmentMap,
}

impl SegmentMaps {
    /// The maps of a frame of `columns` x `rows` luma units, when it has
    /// segmentation.
    #[must_use]
    pub fn of_frame(frame: &FrameHeader, (columns, rows): (u32, u32)) -> Option<Self> {
        frame
            .segmentation
            .is_some()
            .then(|| Self::new(columns as usize, rows as usize))
    }

    /// The reference map `previous` a frame with `maps` predicts from:
    /// only one of the frame's size (dav2d `ref_w == f->bw`).
    #[must_use]
    pub fn predicting<'a>(
        maps: Option<&Self>,
        previous: Option<&'a SegmentMap>,
    ) -> Option<&'a SegmentMap> {
        previous.filter(|previous| maps.is_some_and(|maps| previous.same_size(&maps.luma)))
    }

    /// Segment 0 everywhere, for a frame of `columns` x `rows` luma units.
    #[must_use]
    pub fn new(columns: usize, rows: usize) -> Self {
        Self {
            luma: SegmentMap::new(columns, rows),
            chroma: SegmentMap::new(columns / 2, rows / 2),
        }
    }
}

/// The segment a coded difference stands for, around the predicted
/// segment `reference` among `max` segments (dav2d `neg_deinterleave`:
/// differences alternate below and above the prediction while both
/// exist).
fn neg_deinterleave(diff: i32, reference: i32, max: i32) -> i32 {
    if reference == 0 {
        return diff;
    }
    if reference >= max - 1 {
        return max - diff - 1;
    }
    let around = |diff: i32| {
        if diff & 1 != 0 {
            reference + ((diff + 1) >> 1)
        } else {
            reference - (diff >> 1)
        }
    };
    if 2 * reference < max {
        if diff <= 2 * reference {
            around(diff)
        } else {
            diff
        }
    } else if diff <= 2 * (max - reference - 1) {
        around(diff)
    } else {
        max - (diff + 1)
    }
}

impl Tile<'_, '_> {
    /// The segment id of a block known before its skip flag: a chroma-tree
    /// block's, an id from the reference map, or one the frame reads here
    /// (`preskip`). Returns whether the id is read after the skip flag
    /// instead ([`Tile::segment_after_skip`]).
    pub(super) fn segment_before_skip(
        &mut self,
        depth: usize,
        size: BlockSize,
        has_luma: bool,
    ) -> Result<bool> {
        self.segment = 0;
        self.segment_predicted = false;
        let Some(segmentation) = self.frame.segmentation else {
            return Ok(false);
        };
        if !has_luma {
            if let Some(maps) = &self.filters.segments {
                self.segment = maps.luma.get(self.row as usize, self.column as usize);
            }
            return Ok(false);
        }
        if !segmentation.update_map {
            self.segment = self.previous_segment(size);
            return Ok(false);
        }
        if !segmentation.preskip {
            return Ok(true);
        }
        self.read_segment(depth, size, &segmentation, None)?;
        Ok(false)
    }

    /// The segment id read after the skip flag `skip_txfm`.
    pub(super) fn segment_after_skip(
        &mut self,
        depth: usize,
        size: BlockSize,
        skip_txfm: bool,
    ) -> Result<()> {
        let Some(segmentation) = self.frame.segmentation else {
            return Ok(());
        };
        self.read_segment(depth, size, &segmentation, Some(skip_txfm))
    }

    /// Read a segment id: temporally predicted, or a difference from the
    /// spatial prediction. `skip_txfm` is the skip flag when read after it:
    /// a skipped block of a frame without lossless segments then takes the
    /// prediction, and only unskipped blocks may predict temporally.
    fn read_segment(
        &mut self,
        depth: usize,
        size: BlockSize,
        segmentation: &Segmentation,
        skip_txfm: Option<bool>,
    ) -> Result<()> {
        let skipped = skip_txfm == Some(true);
        let context = usize::from(self.contexts.above.segment_predicted[self.column as usize])
            + usize::from(self.contexts.left.segment_predicted[left_index(self.row)]);
        if !skipped && segmentation.temporal_update && self.bool(|m| &mut m.seg_pred[context])? {
            self.segment_predicted = true;
            self.segment = self.previous_segment(size);
        } else {
            let (prediction, context) = self.spatial_segment();
            self.segment = if skipped && self.frame.lossless.segments == 0 {
                prediction
            } else {
                self.read_segment_difference(segmentation, prediction, context)?
            };
        }
        let range = self.reader.range();
        trace!(self, depth, "Post-segid[{}]: r={range}", self.segment);
        Ok(())
    }

    /// A segment coded as a difference from `prediction`, its models picked
    /// by the spatial `context`.
    fn read_segment_difference(
        &mut self,
        segmentation: &Segmentation,
        prediction: u8,
        context: usize,
    ) -> Result<u8> {
        let extended =
            self.sequence.segmentation.extended && self.bool(|m| &mut m.seg_id_ext[context])?;
        let diff = if extended {
            EXTENDED_STEP + self.symbol(|m| &mut m.seg_id_upper[context])?
        } else {
            self.symbol(|m| &mut m.seg_id[context])?
        };
        let max = segmentation
            .last_active
            .map_or(0, |last| i32::from(last) + 1);
        let segment = neg_deinterleave(diff as i32, i32::from(prediction), max);
        let in_range = segmentation
            .last_active
            .is_none_or(|last| segment <= i32::from(last));
        u8::try_from(segment)
            .ok()
            .filter(|&segment| in_range && usize::from(segment) < MAX_SEGMENTS)
            .ok_or(DecodeError::Malformed("AV2 segment id"))
    }

    /// The spatial segment prediction and its context: the above or left
    /// neighbour's, whichever the above-left agrees with (dav2d
    /// `get_cur_frame_segid`).
    fn spatial_segment(&self) -> (u8, usize) {
        let Some(maps) = &self.filters.segments else {
            return (0, 0);
        };
        let (row, column) = (self.row as usize, self.column as usize);
        let have_left = self.column > self.bounds.column_start;
        let have_top = self.row > self.bounds.row_start;
        match (have_top, have_left) {
            (true, true) => {
                let left = maps.luma.get(row, column - 1);
                let above = maps.luma.get(row - 1, column);
                let above_left = maps.luma.get(row - 1, column - 1);
                let context = if left == above && above_left == left {
                    2
                } else {
                    usize::from(left == above || above_left == left || above == above_left)
                };
                (if above == above_left { above } else { left }, context)
            }
            (_, true) => (maps.luma.get(row, column - 1), 0),
            (true, _) => (maps.luma.get(row - 1, column), 0),
            _ => (0, 0),
        }
    }

    /// The lowest segment of the primary reference's map under the block
    /// (clipped to the frame), or 0 without one.
    fn previous_segment(&self, size: BlockSize) -> u8 {
        let Some(previous) = self.previous_segments else {
            return 0;
        };
        let (bw4, bh4) = size.units();
        let columns = bw4.min(self.geometry.columns - self.column);
        let rows = bh4.min(self.geometry.rows - self.row);
        previous.lowest(
            (self.row as usize, self.column as usize),
            (columns as usize, rows as usize),
        )
    }

    /// Whether the block's segment skips or uses global motion: no skip
    /// mode, no compound, reference 0 and global motion (dav2d
    /// `(globalmv_mask | skip_mask) & (1 << seg_id)`).
    pub(super) fn segment_forces_global(&self) -> bool {
        self.frame.segmentation.is_some_and(|segmentation| {
            let features = segmentation.features;
            (features.global_mv_mask | features.skip_mask) >> self.segment & 1 != 0
        })
    }

    /// Whether the block's segment skips its residual.
    pub(super) fn segment_skips(&self) -> bool {
        self.frame
            .segmentation
            .is_some_and(|segmentation| segmentation.features.skip_mask >> self.segment & 1 != 0)
    }

    /// Record the block's segment over its luma and chroma extents (dav2d
    /// `decode_b`'s `cur_segmap` and `segmap_uv` splats).
    pub(super) fn record_segment(&mut self, luma: Option<BlockSize>, chroma: Option<BlockSize>) {
        let segment = self.segment;
        let Some(maps) = &mut self.filters.segments else {
            return;
        };
        if let Some(size) = luma {
            let (bw4, bh4) = size.units();
            maps.luma.fill(
                (self.row as usize, self.column as usize),
                (bw4 as usize, bh4 as usize),
                segment,
            );
        }
        if let Some(size) = chroma {
            let (bw4, bh4) = size.units();
            let units = |luma: u32| (luma >> 1).max(1) as usize;
            maps.chroma.fill(
                (
                    (self.chroma_row >> 1) as usize,
                    (self.chroma_column >> 1) as usize,
                ),
                (units(bw4), units(bh4)),
                segment,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterleaving_alternates_around_the_prediction() {
        // Prediction 2 of 8: 2, 3, 1, 4, 0, then the rest in order.
        let order: Vec<i32> = (0..8).map(|diff| neg_deinterleave(diff, 2, 8)).collect();
        assert_eq!(order, [2, 3, 1, 4, 0, 5, 6, 7]);
        // Prediction 6 of 8: 6, 7, 5, then downwards.
        let order: Vec<i32> = (0..8).map(|diff| neg_deinterleave(diff, 6, 8)).collect();
        assert_eq!(order, [6, 7, 5, 4, 3, 2, 1, 0]);
        // The top segment counts down.
        let order: Vec<i32> = (0..4).map(|diff| neg_deinterleave(diff, 3, 4)).collect();
        assert_eq!(order, [3, 2, 1, 0]);
    }

    #[test]
    fn maps_take_the_lowest_segment_under_a_block() {
        let mut map = SegmentMap::new(8, 8);
        map.fill((0, 0), (4, 4), 5);
        map.fill((2, 2), (2, 2), 3);
        assert_eq!(map.lowest((0, 0), (4, 4)), 3);
        assert_eq!(map.lowest((0, 0), (2, 2)), 5);
        assert_eq!(map.get(3, 3), 3);
    }
}
