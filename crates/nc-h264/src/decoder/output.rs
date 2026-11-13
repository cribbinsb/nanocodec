//! Output order of decoded frames (§C.4.5.3 "bumping", simplified): frames
//! wait until more than the stream's reorder depth are pending, then leave
//! in picture order count order; an IDR picture or memory management
//! operation 5 first outputs every earlier frame.
//!
//! Streams without B slices normally have their order counts rising in
//! decoding order, so this only delays output by the reorder depth, which
//! is 0 when the VUI says so (`max_num_reorder_frames`).

use std::sync::Arc;

use nc_common::{Picture, ShownPicture};

/// One decoded frame with what its output order depends on. It shares the
/// decoder's picture (no copy); the decoder reuses the buffer once every
/// frame and reference holding it is dropped.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// The decoded picture, with the stream's padding and border; 4:0:0
    /// streams have mid-grey chroma.
    pub picture: Arc<Picture>,
    /// Top-left luma sample of the cropping window.
    pub origin: (usize, usize),
    /// Luma size of the cropping window.
    pub size: (usize, usize),
    /// `PicOrderCnt()` of the frame (0 after memory management operation 5).
    pub poc: i64,
    /// Whether the frame starts a new output sequence (IDR or MMCO 5).
    pub new_sequence: bool,
    /// Frames that may precede this one in decoding order but follow it in
    /// output order (`max_num_reorder_frames`, or the DPB size).
    pub reorder_depth: usize,
}

impl DecodedFrame {
    /// The visible (cropped) samples; `None` for a cropping window outside
    /// the picture or of odd size (rejected when the SPS is parsed).
    #[must_use]
    pub fn view(&self) -> Option<ShownPicture<'_>> {
        ShownPicture::window(&self.picture, self.origin, self.size)
    }
}

/// Frames waiting for output.
#[derive(Debug, Default)]
pub struct OutputQueue {
    pending: Vec<DecodedFrame>,
}

impl OutputQueue {
    /// Add a decoded frame; returns the frames now due, in output order.
    pub fn push(&mut self, frame: DecodedFrame) -> Vec<DecodedFrame> {
        let mut due = if frame.new_sequence {
            self.flush()
        } else {
            Vec::new()
        };
        let reorder_depth = frame.reorder_depth;
        self.pending.push(frame);
        while self.pending.len() > reorder_depth {
            due.push(self.pop_first());
        }
        due
    }

    /// Every pending frame, in output order (end of stream).
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        let mut due = Vec::with_capacity(self.pending.len());
        while !self.pending.is_empty() {
            due.push(self.pop_first());
        }
        due
    }

    /// The pending frame with the smallest order count (the earliest
    /// decoded one among equals).
    fn pop_first(&mut self) -> DecodedFrame {
        let index = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(i, frame)| (frame.poc, *i))
            .map(|(i, _)| i)
            .expect("a pending frame");
        self.pending.remove(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(poc: i64, new_sequence: bool, value: u8) -> DecodedFrame {
        let mut picture = Picture::new(2, 2, 0).unwrap();
        picture
            .plane_mut(nc_common::PlaneIndex::Y)
            .view_mut()
            .fill(value);
        DecodedFrame {
            picture: Arc::new(picture),
            origin: (0, 0),
            size: (2, 2),
            poc,
            new_sequence,
            reorder_depth: 1,
        }
    }

    fn values(frames: &[DecodedFrame]) -> Vec<u8> {
        frames
            .iter()
            .map(|f| f.view().unwrap().plane(nc_common::PlaneIndex::Y).row(0)[0])
            .collect()
    }

    #[test]
    fn frames_leave_in_order_count_order() {
        let mut queue = OutputQueue::default();
        assert!(queue.push(frame(0, true, 0)).is_empty());
        assert_eq!(values(&queue.push(frame(4, false, 2))), [0]);
        assert_eq!(values(&queue.push(frame(2, false, 1))), [1]);
        assert_eq!(values(&queue.push(frame(0, true, 3))), [2]);
        assert_eq!(values(&queue.flush()), [3]);
    }
}
