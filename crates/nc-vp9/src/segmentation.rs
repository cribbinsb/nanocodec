//! VP9 segmentation as the encoder writes it: `segmentation_params` of the
//! uncompressed header (VP9 §6.2.11 / §7.2.10; libvpx
//! `vp9_decodeframe.c` `setup_segmentation`, BSD-3-Clause, see
//! `THIRD_PARTY.md`) and the per-block segment id.
//!
//! Written: a segment map coded every enabled frame (`update_map` = 1),
//! each id either explicit or predicted from the last segmented frame's
//! map (`temporal_update`), the per-segment quantizer (`SEG_LVL_ALT_Q`:
//! block-level QP, `block_qp.rs`) and loop-filter level (`SEG_LVL_ALT_LF`),
//! each absolute or a delta from the frame value. Features and the last map
//! persist across frames until a key frame resets them.
//! The decoder parses every segmentation feature (`decoder/header.rs`).

use nc_common::BitWriter;

/// `MAX_SEGMENTS`.
pub(crate) const MAX_SEGMENTS: usize = 8;
/// `SEG_TREE_PROBS`: node probabilities of the segment-id tree.
pub(crate) const TREE_PROBS: usize = MAX_SEGMENTS - 1;
/// `vp9_segment_tree`: a balanced tree over the eight ids.
pub(crate) const SEGMENT_TREE: &crate::tree::Tree =
    &crate::tree::Tree::new(&[2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7]);
/// `PREDICTION_PROBS`: probabilities of the "predicted" flag by context
/// (how many of the above and left blocks were predicted).
pub(crate) const PREDICTION_PROBS: usize = 3;
/// A tree probability left at `MAX_PROB` is not sent.
const MAX_PROB: u8 = 255;
const PROB_BITS: u32 = 8;
/// Magnitude bits of `SEG_LVL_ALT_Q` data (`get_unsigned_bits(255)`).
/// Features are coded in the order `SEG_LVL_ALT_Q`, `_ALT_LF`,
/// `_REF_FRAME`, `_SKIP`; the last two are never used.
const ALT_Q_BITS: u32 = 8;
/// Magnitude bits of `SEG_LVL_ALT_LF` data (`get_unsigned_bits(63)`).
const ALT_LF_BITS: u32 = 6;
/// Largest filter level.
const MAX_FILTER_LEVEL: i32 = 63;

/// The segment features in force (decoder state across frames).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct SegmentFeatures {
    /// `abs_delta`: feature data replaces the frame value instead of
    /// adjusting it.
    pub absolute: bool,
    /// `SEG_LVL_ALT_Q` data of each segment (-255..=255).
    pub quantizer: [Option<i16>; MAX_SEGMENTS],
    /// `SEG_LVL_ALT_LF` data of each segment.
    pub filter: [Option<i8>; MAX_SEGMENTS],
}

impl SegmentFeatures {
    /// Filter level of every segment for frame level `level`
    /// (`vp9_loop_filter_frame_init`).
    pub(crate) fn filter_levels(&self, level: u8) -> [u8; MAX_SEGMENTS] {
        self.filter.map(|data| match data {
            None => level,
            Some(data) => {
                let base = if self.absolute { 0 } else { i32::from(level) };
                (base + i32::from(data)).clamp(0, MAX_FILTER_LEVEL) as u8
            }
        })
    }
}

/// One frame's `segmentation_params`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segmentation {
    pub enabled: bool,
    /// Segment-id tree probabilities of this frame's map.
    pub tree_probs: [u8; TREE_PROBS],
    /// `temporal_update`: ids may be predicted from the last map, with
    /// these flag probabilities.
    pub temporal: Option<[u8; PREDICTION_PROBS]>,
    /// Whether this frame sends `features` (otherwise they carry over).
    pub update_data: bool,
    /// The features in force after this header.
    pub features: SegmentFeatures,
}

impl Segmentation {
    /// Segmentation off; `features` carry over unchanged.
    pub(crate) fn disabled(features: SegmentFeatures) -> Self {
        Self {
            enabled: false,
            tree_probs: [MAX_PROB; TREE_PROBS],
            temporal: None,
            update_data: false,
            features,
        }
    }

    /// The same segmentation decodable whatever frame came before: when
    /// enabled, ids are coded explicitly (no `temporal_update`) and the
    /// features are sent (`update_data`). A spatial layer's frames need it:
    /// a receiver of both layers decoded the other layer's frame last, a
    /// base-only receiver its own layer's.
    pub(crate) fn standalone(self) -> Self {
        Self {
            temporal: None,
            update_data: self.enabled,
            ..self
        }
    }

    /// The filter level of each segment, if the frame codes segment ids.
    pub(crate) fn active_filter_levels(&self, level: u8) -> Option<[u8; MAX_SEGMENTS]> {
        self.enabled.then(|| self.features.filter_levels(level))
    }

    pub(crate) fn write(&self, bits: &mut BitWriter) {
        bits.put_bit(self.enabled);
        if !self.enabled {
            return;
        }
        bits.put_bit(true); // update_map
        for &prob in &self.tree_probs {
            write_prob(bits, prob);
        }
        bits.put_bit(self.temporal.is_some());
        for &prob in self.temporal.iter().flatten() {
            write_prob(bits, prob);
        }
        bits.put_bit(self.update_data);
        if !self.update_data {
            return;
        }
        bits.put_bit(self.features.absolute);
        for (quantizer, filter) in self
            .features
            .quantizer
            .into_iter()
            .zip(self.features.filter)
        {
            bits.put_bit(quantizer.is_some());
            if let Some(data) = quantizer {
                bits.put_bits(u32::from(data.unsigned_abs()), ALT_Q_BITS);
                bits.put_bit(data < 0);
            }
            bits.put_bit(filter.is_some());
            if let Some(data) = filter {
                bits.put_bits(u32::from(data.unsigned_abs()), ALT_LF_BITS);
                bits.put_bit(data < 0);
            }
            bits.put_bit(false); // SEG_LVL_REF_FRAME
            bits.put_bit(false); // SEG_LVL_SKIP
        }
    }
}

/// A header probability: a flag, then 8 bits unless it is `MAX_PROB`.
fn write_prob(bits: &mut BitWriter, prob: u8) {
    bits.put_bit(prob != MAX_PROB);
    if prob != MAX_PROB {
        bits.put_bits(u32::from(prob), PROB_BITS);
    }
}

/// The segment ids of the last segmented frame, one per 8x8 unit
/// (libvpx `last_frame_seg_map`): zero after a key frame, replaced by each
/// frame that enables segmentation.
#[derive(Debug, Clone, Default)]
pub(crate) struct SegmentMap {
    cols: usize,
    ids: Vec<u8>,
}

impl SegmentMap {
    /// All zero, for a frame of `cols` x `rows` units.
    pub(crate) fn cleared(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            ids: vec![0; cols * rows],
        }
    }

    /// `dec_get_segment_id`: the smallest id under the block's units that
    /// lie inside the frame.
    pub(crate) fn predicted(&self, row: usize, col: usize, units: usize) -> u8 {
        let rows = self.ids.len() / self.cols.max(1);
        (row..(row + units).min(rows))
            .flat_map(|r| {
                let start = r * self.cols + col;
                self.ids[start..start + units.min(self.cols.saturating_sub(col))].iter()
            })
            .copied()
            .min()
            .unwrap_or(0)
    }

    /// Take the ids of a segmented frame.
    pub(crate) fn update(&mut self, grid: &crate::types::ModeInfoGrid) {
        self.cols = grid.cols;
        self.ids.clear();
        self.ids.extend(
            (0..grid.rows).flat_map(|r| (0..grid.cols).map(move |c| grid.at(r, c).segment_id)),
        );
    }
}

/// The `(node, bit)` decisions that code segment `id` in `SEGMENT_TREE`,
/// root first. The tree is balanced: node 0 splits on bit 2 of the id,
/// nodes 1 and 2 on bit 1, nodes 3 to 6 on bit 0.
pub(crate) fn tree_path(id: u8) -> [(usize, bool); 3] {
    let id = usize::from(id);
    [
        (0, id & 4 != 0),
        (1 + (id >> 2), id & 2 != 0),
        (3 + (id >> 1), id & 1 != 0),
    ]
}

/// The branches taken at each segment-tree node by one frame's explicitly
/// coded ids, which give that frame's own tree probabilities (the header
/// sends them, VP9 §6.2.11, and the tile is written after the frame).
#[derive(Debug, Default)]
pub(crate) struct SegmentStats {
    branches: [[u32; 2]; TREE_PROBS],
}

impl SegmentStats {
    /// Count one explicitly coded id.
    pub(crate) fn count(&mut self, id: u8) {
        for (node, bit) in tree_path(id) {
            self.branches[node][usize::from(bit)] += 1;
        }
    }

    /// The segment-tree node probabilities (of the 0 branch) these counts
    /// give, each inside 1..=255; a node no id reached is `MAX_PROB`, which
    /// costs nothing to send.
    pub(crate) fn tree_probs(&self) -> [u8; TREE_PROBS] {
        self.branches.map(|[zeros, ones]| {
            if zeros + ones == 0 {
                MAX_PROB
            } else {
                (256.0 * zeros as f32 / (zeros + ones) as f32)
                    .round()
                    .clamp(1.0, 255.0) as u8
            }
        })
    }
}

/// `vp9_get_pred_context_seg_id`: predicted flags of the above and left
/// blocks.
pub(crate) fn prediction_context(
    grid: &crate::types::ModeInfoGrid,
    row: usize,
    col: usize,
) -> usize {
    let above = row > 0 && grid.at(row - 1, col).segment_predicted;
    let left = col > 0 && grid.at(row, col - 1).segment_predicted;
    usize::from(above) + usize::from(left)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_levels_clamp() {
        let features = SegmentFeatures {
            absolute: false,
            filter: [None, Some(-63), Some(40), None, None, None, None, None],
            ..SegmentFeatures::default()
        };
        let levels = features.filter_levels(30);
        assert_eq!(levels[..3], [30, 0, 63]);
    }

    #[test]
    fn tree_probabilities_follow_the_id_counts() {
        let mut stats = SegmentStats::default();
        assert_eq!(stats.tree_probs(), [MAX_PROB; TREE_PROBS]);
        for (id, count) in [(0, 3), (1, 1), (4, 4)] {
            for _ in 0..count {
                stats.count(id);
            }
        }
        let probs = stats.tree_probs();
        // Node 0: ids 0-3 (4) against 4-7 (4); node 3: id 0 (3) against 1 (1).
        assert_eq!(probs[0], 128);
        assert_eq!(probs[1], 255);
        assert_eq!(probs[3], 192);
        assert_eq!(probs[5], 255);
        assert_eq!(probs[4], MAX_PROB);
    }
}
