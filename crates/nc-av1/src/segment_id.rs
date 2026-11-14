//! Segment ids of AV1 segmentation (AV1 §5.11.9 `read_segment_id`, §8.3.2
//! `get_segment_id` and the neighbour prediction): the per-4x4 map of a
//! frame, its spatial prediction and context, and the interleaving that
//! codes an id relative to its prediction. Shared by the encoder (which
//! writes ids) and the decoder (which reads them), so both derive the same
//! prediction from the same map.
//!
//! A frame may also keep its reference's map (`segmentation_update_map`
//! 0) or predict ids from it (`segmentation_temporal_update`), as libaom's
//! cyclic refresh does; [`SegmentMap::min_over`] and
//! [`SegmentMap::copy_from`] serve those.
//!
//! Reimplemented from libaom's `av1_get_spatial_seg_pred` and
//! `av1_neg_deinterleave` / `neg_interleave` (BSD-2-Clause, Copyright (c)
//! 2016, Alliance for Open Media; see `THIRD_PARTY.md`).

/// `MAX_SEGMENTS`: ids 0..8 (AV1 §3).
pub(crate) const MAX_SEGMENTS: usize = 8;

/// The segment id of every 4x4 unit of a frame.
#[derive(Debug, Clone, Default)]
pub(crate) struct SegmentMap {
    ids: Vec<u8>,
    mi_cols: usize,
    mi_rows: usize,
}

/// A block's spatial prediction (`pred`) and the CDF context of its coded
/// id (`ctx`, 0..3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpatialPrediction {
    pub pred: u8,
    pub ctx: usize,
}

impl SegmentMap {
    /// Clear the map for a frame of `mi_cols` x `mi_rows` units (every id 0).
    pub fn reset(&mut self, mi_cols: usize, mi_rows: usize) {
        self.mi_cols = mi_cols;
        self.mi_rows = mi_rows;
        self.ids.clear();
        self.ids.resize(mi_cols * mi_rows, 0);
    }

    /// The id of unit (`mi_col`, `mi_row`).
    fn at(&self, mi_col: usize, mi_row: usize) -> u8 {
        self.ids[mi_row * self.mi_cols + mi_col]
    }

    /// The prediction of the block at (`mi_col`, `mi_row`) whose above row
    /// and left column are (`up`, `left`) available inside its tile: the
    /// above-left, above and left units' ids; two or three alike give the
    /// prediction, else the left one (libaom `av1_get_spatial_seg_pred`).
    pub fn predict(
        &self,
        (mi_col, mi_row): (usize, usize),
        (up, left): (bool, bool),
    ) -> SpatialPrediction {
        let above_left = (up && left).then(|| self.at(mi_col - 1, mi_row - 1));
        let above = up.then(|| self.at(mi_col, mi_row - 1));
        let left_id = left.then(|| self.at(mi_col - 1, mi_row));
        let ctx = match (above_left, above, left_id) {
            (Some(ul), Some(u), Some(l)) if ul == u && ul == l => 2,
            (Some(ul), Some(u), Some(l)) if ul == u || ul == l || u == l => 1,
            _ => 0,
        };
        let pred = match (above_left, above, left_id) {
            (_, None, l) => l.unwrap_or(0),
            (_, Some(u), None) => u,
            (ul, Some(u), Some(l)) => {
                if ul == Some(u) {
                    u
                } else {
                    l
                }
            }
        };
        SpatialPrediction { pred, ctx }
    }

    /// Whether the map covers a frame of `mi_cols` x `mi_rows` units (a
    /// reference's map serves only a frame of its own size).
    pub fn covers(&self, (mi_cols, mi_rows): (usize, usize)) -> bool {
        (self.mi_cols, self.mi_rows) == (mi_cols, mi_rows)
    }

    /// The smallest id of the `width` x `height` units from (`mi_col`,
    /// `mi_row`), clipped to the frame (libaom `dec_get_segment_id`: a
    /// block's id predicted from a reference's map).
    pub fn min_over(
        &self,
        (mi_col, mi_row): (usize, usize),
        (width, height): (usize, usize),
    ) -> u8 {
        let cols = mi_col..(mi_col + width).min(self.mi_cols);
        (mi_row..(mi_row + height).min(self.mi_rows))
            .flat_map(|row| {
                self.ids[row * self.mi_cols + cols.start..row * self.mi_cols + cols.end].iter()
            })
            .copied()
            .min()
            .unwrap_or(0)
    }

    /// Copy the `width` x `height` units from (`mi_col`, `mi_row`) of
    /// `previous` (a map of the same size), or zero them without one.
    pub fn copy_from(
        &mut self,
        previous: Option<&Self>,
        (mi_col, mi_row): (usize, usize),
        (width, height): (usize, usize),
    ) {
        let cols = mi_col.min(self.mi_cols)..(mi_col + width).min(self.mi_cols);
        for row in mi_row..(mi_row + height).min(self.mi_rows) {
            let span = row * self.mi_cols + cols.start..row * self.mi_cols + cols.end;
            match previous {
                Some(previous) => self.ids[span.clone()].copy_from_slice(&previous.ids[span]),
                None => self.ids[span].fill(0),
            }
        }
    }

    /// The id of unit (`mi_col`, `mi_row`), for neighbour contexts.
    pub fn get(&self, mi_col: usize, mi_row: usize) -> u8 {
        self.at(mi_col, mi_row)
    }

    /// Set the `width` x `height` units from (`mi_col`, `mi_row`), clipped
    /// to the frame, to `id`.
    pub fn fill(
        &mut self,
        (mi_col, mi_row): (usize, usize),
        (width, height): (usize, usize),
        id: u8,
    ) {
        let cols = mi_col.min(self.mi_cols)..(mi_col + width).min(self.mi_cols);
        for row in mi_row..(mi_row + height).min(self.mi_rows) {
            let start = row * self.mi_cols;
            self.ids[start + cols.start..start + cols.end].fill(id);
        }
    }
}

/// The id coded as `diff` relative to prediction `pred` among `count` ids
/// (`LastActiveSegId + 1`): libaom `av1_neg_deinterleave`, the inverse of
/// [`neg_interleave`].
pub(crate) fn neg_deinterleave(diff: usize, pred: usize, count: usize) -> usize {
    let (d, r, max) = (diff as i64, pred as i64, count as i64);
    let near = || {
        if d & 1 == 1 {
            r + (d + 1) / 2
        } else {
            r - d / 2
        }
    };
    let id = if r == 0 {
        d
    } else if r >= max - 1 {
        max - d - 1
    } else if 2 * r < max {
        if d <= 2 * r { near() } else { d }
    } else if d <= 2 * (max - r - 1) {
        near()
    } else {
        max - (d + 1)
    };
    // Out of range only in a corrupt stream; the caller rejects it.
    usize::try_from(id).unwrap_or(usize::MAX)
}

/// The coded symbol of `id` relative to prediction `pred` among `count`
/// ids: ids near the prediction get small symbols, alternating above and
/// below it (libaom `av1_neg_interleave`).
pub(crate) fn neg_interleave(id: usize, pred: usize, count: usize) -> usize {
    let (x, r, max) = (id as i64, pred as i64, count as i64);
    let diff = x - r;
    let near = || if diff > 0 { 2 * diff - 1 } else { -2 * diff };
    let coded = if r == 0 {
        x
    } else if r >= max - 1 {
        max - 1 - x
    } else if 2 * r < max {
        if diff.abs() <= r { near() } else { x }
    } else if diff.abs() < max - r {
        near()
    } else {
        max - 1 - x
    };
    coded as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_round_trips_for_every_prediction_and_count() {
        for count in 1..=MAX_SEGMENTS {
            for pred in 0..count {
                let mut seen = vec![false; count];
                for id in 0..count {
                    let diff = neg_interleave(id, pred, count);
                    assert!(diff < count, "count {count} pred {pred} id {id}: {diff}");
                    assert!(!seen[diff], "count {count} pred {pred}: {diff} twice");
                    seen[diff] = true;
                    assert_eq!(
                        neg_deinterleave(diff, pred, count),
                        id,
                        "count {count} pred {pred}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_prediction_codes_as_zero() {
        for count in 1..=MAX_SEGMENTS {
            for pred in 0..count {
                assert_eq!(neg_interleave(pred, pred, count), 0);
            }
        }
    }

    #[test]
    fn spatial_prediction_follows_the_neighbours() {
        let mut map = SegmentMap::default();
        map.reset(4, 4);
        assert_eq!(
            map.predict((0, 0), (false, false)),
            SpatialPrediction { pred: 0, ctx: 0 }
        );
        map.fill((0, 0), (1, 1), 3);
        map.fill((1, 0), (1, 1), 3);
        map.fill((0, 1), (1, 1), 5);
        // Above-left 3, above 3, left 5: two alike (above-left, above).
        assert_eq!(
            map.predict((1, 1), (true, true)),
            SpatialPrediction { pred: 3, ctx: 1 }
        );
        map.fill((0, 1), (1, 1), 3);
        assert_eq!(
            map.predict((1, 1), (true, true)),
            SpatialPrediction { pred: 3, ctx: 2 }
        );
        // Only left available.
        assert_eq!(
            map.predict((1, 1), (false, true)),
            SpatialPrediction { pred: 3, ctx: 0 }
        );
    }
}
