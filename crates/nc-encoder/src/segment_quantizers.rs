//! Block-level quantizers through segmentation: codecs whose per-block
//! quantizer is a segment's (VP9, which has no block delta; AV1 at 8x8
//! granularity, finer than its superblock delta) carry a frame's block
//! offsets by up to eight segments, each with its own `SEG_LVL_ALT_Q` (VP9
//! §6.2.11, §7.2.10; AV1 §5.9.14, §7.12.2).
//!
//! Segment 0 always codes the frame quantizer. The others are handed out in
//! coding order to the first distinct offsets the engine asks for; once all
//! are taken, an offset takes the nearest one already in use. The engine
//! asks before coding each block (`CodecBackend::block_qp_offset`), so the
//! offset it ranks the block with and budgets in rate control is the one
//! the segment codes. A segment's qindex never reaches 0, which would
//! quantize with the lossless frame's smallest step.

/// Segments a frame codes (VP9 and AV1 `MAX_SEGMENTS`).
pub const MAX_SEGMENTS: usize = 8;

/// Smallest qindex a segment codes.
const MIN_SEGMENT_QINDEX: i32 = 1;
/// Largest qindex.
const MAX_QINDEX: i32 = 255;

/// The quantizer offset (from the frame's `base_q_idx`) of every segment in
/// use this frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentQuantizers {
    base: u8,
    /// Offset of each segment handed out so far; segment 0 is the frame's.
    offsets: [Option<i16>; MAX_SEGMENTS],
}

impl SegmentQuantizers {
    /// Only segment 0 (the frame quantizer `base`) in use.
    #[must_use]
    pub fn new(base: u8) -> Self {
        let mut offsets = [None; MAX_SEGMENTS];
        offsets[0] = Some(0);
        Self { base, offsets }
    }

    /// The segment that codes `requested` (an offset from the frame
    /// quantizer) and the offset it actually codes: a segment with that
    /// offset, a new one, or once all are in use the nearest.
    pub fn assign(&mut self, requested: i16) -> (u8, i16) {
        let base = i32::from(self.base);
        let wanted = (base + i32::from(requested)).clamp(MIN_SEGMENT_QINDEX, MAX_QINDEX) - base;
        let wanted = wanted as i16;
        let used = self.offsets.iter().take_while(|o| o.is_some()).count();
        if let Some(segment) = self.offsets[..used].iter().position(|&o| o == Some(wanted)) {
            return (segment as u8, wanted);
        }
        if used < MAX_SEGMENTS {
            self.offsets[used] = Some(wanted);
            return (used as u8, wanted);
        }
        let (segment, offset) = self.offsets[..used]
            .iter()
            .enumerate()
            .filter_map(|(segment, offset)| offset.map(|o| (segment, o)))
            .min_by_key(|&(_, offset)| (offset - wanted).abs())
            .expect("segment 0 is always in use");
        (segment as u8, offset)
    }

    /// The segment that codes `offset`, an offset [`Self::assign`] returned
    /// this frame (segment 0, the frame quantizer's, for any other).
    #[must_use]
    pub fn segment_of(&self, offset: i16) -> u8 {
        self.offsets
            .iter()
            .position(|&o| o == Some(offset))
            .map_or(0, |segment| segment as u8)
    }

    /// The `SEG_LVL_ALT_Q` data of every segment: a delta, or with
    /// `absolute` the segment's qindex; `None` where it is the frame's.
    #[must_use]
    pub fn feature_data(&self, absolute: bool) -> [Option<i16>; MAX_SEGMENTS] {
        self.offsets.map(|offset| match offset {
            Some(0) | None => None,
            Some(offset) if absolute => Some(i16::from(self.base) + offset),
            Some(offset) => Some(offset),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_take_segments_in_order_then_the_nearest() {
        let mut segments = SegmentQuantizers::new(100);
        assert_eq!(segments.assign(0), (0, 0));
        for (i, offset) in [-10, 10, -20, 20, -30, 30, 40].into_iter().enumerate() {
            assert_eq!(segments.assign(offset), (i as u8 + 1, offset));
        }
        assert_eq!(segments.assign(-10), (1, -10));
        // All eight in use: -12 codes as -10, 33 as 30.
        assert_eq!(segments.assign(-12), (1, -10));
        assert_eq!(segments.assign(33), (6, 30));
        let data = segments.feature_data(false);
        assert_eq!(data[0], None);
        assert_eq!(data[1], Some(-10));
        assert_eq!(segments.feature_data(true)[1], Some(90));
    }

    #[test]
    fn a_segment_never_reaches_qindex_zero() {
        let mut segments = SegmentQuantizers::new(5);
        assert_eq!(segments.assign(-20), (1, -4));
        assert_eq!(SegmentQuantizers::new(250).assign(30), (1, 5));
    }
}
