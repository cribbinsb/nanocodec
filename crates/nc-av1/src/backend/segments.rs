//! Block quantizers through segmentation (`av1.segment_qp`, AV1 §5.9.14,
//! §5.11.9, §7.12.2): the engine's per-block offsets at 8x8 granularity,
//! instead of one `delta_qindex` per 64x64 superblock (`delta_q.rs`).
//!
//! A superblock delta gives every block of the superblock one quantizer, so
//! an offset meant for a few still cells is averaged over moving ones (the
//! hold and planned refinements of `nc_encoder`'s reuse term leaked
//! through that pooling on AV1;
//! `docs/analysis/persistent-region-coding-2026-11-14.md`). Segment ids are
//! coded per block, as libaom's cyclic refresh does.
//!
//! - **Segments.** Segment 0 codes the frame quantizer; the others take the
//!   frame's first distinct offsets (`nc_encoder::segment_quantizers`, the
//!   same snapping VP9 uses). Segments 1..7 always carry `SEG_LVL_ALT_Q`
//!   (0 when unused), so `LastActiveSegId` is 7 and every id is coded
//!   among eight, whatever the frame assigns later.
//! - **Ids.** After `skip` (§5.11.8 / §5.11.19 without pre-skip features):
//!   a skipped block takes its spatial prediction (it has no residual, so
//!   its quantizer does not matter), the others code their id against it
//!   (`crate::segment_id`, shared with the decoder). Every frame with
//!   segmentation codes a fresh map (`segmentation_update_map` 1, no
//!   temporal prediction), so no frame depends on a reference's map.
//! - Frames without block offsets disable segmentation (every id 0).

use crate::cdf::Cdf;
use crate::range_coder::SymbolWriter;
use crate::segment_id::{MAX_SEGMENTS, SegmentMap, neg_interleave};
use nc_encoder::segment_quantizers::SegmentQuantizers;

/// Smallest and largest qindex a segment codes (qindex 0 would make the
/// segment lossless).
const MIN_QINDEX: i32 = 1;
const MAX_QINDEX: i32 = 255;
/// log2 of the 4x4 mode-info unit in luma samples.
const MI_LOG2: usize = 2;

/// The segmentation of the frame being coded.
#[derive(Debug, Clone)]
struct FrameSegments {
    quantizers: SegmentQuantizers,
    map: SegmentMap,
    base_q_idx: u32,
}

/// Segment-id block quantizers of one stream.
#[derive(Debug, Clone, Default)]
pub(super) struct SegmentCoding {
    /// Whether frames may code segment ids (`enable_block_qp` with
    /// `av1.segment_qp`).
    enabled: bool,
    frame: Option<FrameSegments>,
}

impl SegmentCoding {
    /// Code segment ids from now on in frames whose blocks may carry
    /// offsets.
    pub(super) fn enable(&mut self) {
        self.enabled = true;
    }

    /// Whether the frame being coded codes segment ids.
    pub(super) fn active(&self) -> bool {
        self.frame.is_some()
    }

    /// Start a frame at `base_q_idx` of `width` x `height` coded luma
    /// samples; `block_qp`: its blocks may carry offsets.
    pub(super) fn start_frame(
        &mut self,
        base_q_idx: u32,
        block_qp: bool,
        (width, height): (usize, usize),
    ) {
        self.frame = (self.enabled && block_qp).then(|| {
            let mut map = SegmentMap::default();
            // MiCols / MiRows of the 8-aligned coded size, as the decoder's.
            map.reset(width.div_ceil(8) * 2, height.div_ceil(8) * 2);
            FrameSegments {
                quantizers: SegmentQuantizers::new(base_q_idx as u8),
                map,
                base_q_idx,
            }
        });
    }

    /// The offset a segment codes for a block the engine wants `requested`
    /// steps from the frame quantizer.
    pub(super) fn assign(&mut self, requested: i16) -> i16 {
        self.frame
            .as_mut()
            .map_or(requested, |frame| frame.quantizers.assign(requested).1)
    }

    /// The qindex of a block at `offset` (one [`Self::assign`] returned).
    pub(super) fn block_qindex(&self, offset: i16) -> Option<u32> {
        let frame = self.frame.as_ref()?;
        Some((frame.base_q_idx as i32 + i32::from(offset)).clamp(MIN_QINDEX, MAX_QINDEX) as u32)
    }

    /// `intra_segment_id()` / `inter_segment_id()` (§5.11.8, §5.11.19) of
    /// the block at luma (`x`, `y`) of `width` x `height` samples, coded at
    /// `q_idx`, after its `skip`.
    pub(super) fn write(
        &mut self,
        writer: &mut SymbolWriter,
        cdfs: &mut [Cdf<MAX_SEGMENTS>; 3],
        (x, y, width, height): (usize, usize, usize, usize),
        q_idx: u32,
        skip: bool,
    ) {
        let Some(frame) = self.frame.as_mut() else {
            return;
        };
        let at = (x >> MI_LOG2, y >> MI_LOG2);
        let units = ((width >> MI_LOG2).max(1), (height >> MI_LOG2).max(1));
        let prediction = frame.map.predict(at, (at.1 > 0, at.0 > 0));
        let id = if skip {
            prediction.pred
        } else {
            let offset = (q_idx as i32 - frame.base_q_idx as i32) as i16;
            let id = frame.quantizers.segment_of(offset);
            let symbol =
                neg_interleave(usize::from(id), usize::from(prediction.pred), MAX_SEGMENTS);
            writer.write_symbol(symbol, &mut cdfs[prediction.ctx], MAX_SEGMENTS);
            id
        };
        frame.map.fill(at, units, id);
    }

    /// `FeatureData[segment][SEG_LVL_ALT_Q]` of the frame's segments 1..7
    /// (segment 0 codes no feature), or `None` when the frame codes no
    /// segmentation.
    pub(super) fn header(&self) -> Option<[i16; MAX_SEGMENTS]> {
        let frame = self.frame.as_ref()?;
        let data = frame.quantizers.feature_data(false);
        Some(std::array::from_fn(|segment| data[segment].unwrap_or(0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_streams_code_nothing() {
        let mut segments = SegmentCoding::default();
        segments.start_frame(100, true, (64, 64));
        assert_eq!(segments.assign(-12), -12);
        assert!(segments.header().is_none());
    }

    #[test]
    fn offsets_take_segments_and_skipped_blocks_their_prediction() {
        let mut segments = SegmentCoding::default();
        segments.enable();
        segments.start_frame(100, true, (64, 64));
        assert_eq!(segments.assign(-12), -12);
        assert_eq!(segments.block_qindex(-12), Some(88));
        let mut writer = SymbolWriter::new();
        let mut cdfs = crate::tables::SPATIAL_PRED_SEG_TREE_CDF.map(Cdf::new);
        segments.write(&mut writer, &mut cdfs, (0, 0, 16, 16), 88, false);
        // A skipped neighbour takes the prediction (segment 1 to its left).
        segments.write(&mut writer, &mut cdfs, (16, 0, 16, 16), 100, true);
        let header = segments.header().unwrap();
        assert_eq!(header[1], -12);
        assert_eq!(header[2], 0);
    }
}
