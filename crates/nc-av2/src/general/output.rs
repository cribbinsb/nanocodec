//! The order in which decoded frames are output: a frame shown now first
//! releases the hidden frames (shown later, `implicit_output_frame`) due
//! before it in display order, then itself, then the hidden frames that
//! follow it without a gap; at the end of the stream every hidden frame
//! still due is released in display order (dav2d `lib.c`
//! `dav2d_queue_output` and `queue_flush`).
//!
//! Only headers decide the order, so the header walk of
//! `examples/dump_headers.rs` uses this as the decoder does.
use super::frame_header::{MAX_SLOTS, SlotHeader};
use super::references;

/// A picture to output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Release {
    /// The frame being shown.
    Current,
    /// The frame held in this slot.
    Slot(usize),
}

/// The output order's state: the display order of the last picture
/// output.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutputOrder {
    /// dav2d `dpb_poc`: 0 before any output (so a first frame of order 0
    /// is never taken for a repeat).
    last: u32,
    /// Whether the sequence codes order hints (without them every order
    /// difference is zero and nothing is shown later).
    ordered: bool,
}

impl OutputOrder {
    /// The state for a sequence with `order_hint_bits`.
    #[must_use]
    pub fn new(order_hint_bits: u8) -> Self {
        Self {
            last: 0,
            ordered: order_hint_bits != 0,
        }
    }

    /// The pictures to output when a frame of display order `order` is
    /// shown, with `slots` holding the frames decoded before it; `None`
    /// when it repeats the last output's order (AVM rejects the stream;
    /// dav2d drops the picture).
    pub fn show(&mut self, slots: &[Option<SlotHeader>], order: u32) -> Option<Vec<Release>> {
        let mut released = Vec::new();
        if self.last != 0 && order == self.last {
            return None;
        }
        let mut taken = 0_u32;
        while let Some(slot) = self.earliest_hidden(slots, taken, Some(order)) {
            taken |= 1 << slot;
            self.last = hidden_order(slots, slot, 0).expect("a hidden frame");
            released.push(Release::Slot(slot));
        }
        released.push(Release::Current);
        self.last = order;
        while let Some(slot) = (0..slots.len().min(MAX_SLOTS)).find(|&slot| {
            hidden_order(slots, slot, taken)
                .is_some_and(|hidden| self.difference(hidden, self.last) == 1)
        }) {
            taken |= 1 << slot;
            self.last = hidden_order(slots, slot, 0).expect("a hidden frame");
            released.push(Release::Slot(slot));
        }
        Some(released)
    }

    /// The hidden frames still due at the end of the stream, in display
    /// order.
    pub fn flush(&mut self, slots: &[Option<SlotHeader>]) -> Vec<usize> {
        let mut released = Vec::new();
        let mut taken = 0_u32;
        while let Some(slot) = self.earliest_hidden(slots, taken, None) {
            taken |= 1 << slot;
            self.last = hidden_order(slots, slot, 0).expect("a hidden frame");
            released.push(slot);
        }
        released
    }

    /// The slot of the earliest hidden frame after the last output and
    /// before `before`, among the slots not in `taken`.
    fn earliest_hidden(
        self,
        slots: &[Option<SlotHeader>],
        taken: u32,
        before: Option<u32>,
    ) -> Option<usize> {
        let mut earliest: Option<(usize, u32)> = None;
        for slot in 0..slots.len().min(MAX_SLOTS) {
            let Some(order) = hidden_order(slots, slot, taken) else {
                continue;
            };
            let bound = earliest.map(|(_, earliest)| earliest).or(before);
            if self.difference(order, self.last) > 0
                && bound.is_none_or(|bound| self.difference(order, bound) < 0)
            {
                earliest = Some((slot, order));
            }
        }
        earliest.map(|(slot, _)| slot)
    }

    /// `a - b` in display order (dav2d `get_poc_diff`).
    fn difference(self, a: u32, b: u32) -> i32 {
        if self.ordered {
            references::order_difference(a, b)
        } else {
            0
        }
    }
}

/// The display order of the hidden frame in `slot`, unless the slot is
/// empty, in `taken`, or holds a frame not shown later.
fn hidden_order(slots: &[Option<SlotHeader>], slot: usize, taken: u32) -> Option<u32> {
    if taken & (1 << slot) != 0 {
        return None;
    }
    let held = slots.get(slot)?.as_ref()?;
    held.header.show_implicit.then_some(held.header.order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::general::frame_header::FrameHeader;
    use std::rc::Rc;

    #[allow(clippy::unnecessary_wraps, reason = "an occupied slot")]
    fn hidden(order: u32) -> Option<SlotHeader> {
        Some(SlotHeader {
            header: Rc::new(FrameHeader {
                order,
                show_implicit: true,
                ..FrameHeader::default()
            }),
            frame: order,
        })
    }

    /// Random access as avmenc codes it: key frame 0, hidden 8 and 4, then
    /// 1..3 shown at once release nothing hidden until 4 is due.
    #[test]
    fn hidden_frames_come_out_in_display_order() {
        let mut order = OutputOrder::new(4);
        let mut slots = vec![None; MAX_SLOTS];
        assert_eq!(order.show(&slots, 0).unwrap(), [Release::Current]);
        slots[1] = hidden(8);
        slots[2] = hidden(4);
        assert_eq!(order.show(&slots, 1).unwrap(), [Release::Current]);
        assert_eq!(order.show(&slots, 2).unwrap(), [Release::Current]);
        // 3 is shown, then 4 follows it without a gap.
        assert_eq!(
            order.show(&slots, 3).unwrap(),
            [Release::Current, Release::Slot(2)]
        );
        // 6 is shown: 5 was never coded, so nothing before it is due.
        assert_eq!(order.show(&slots, 6).unwrap(), [Release::Current]);
        assert_eq!(order.flush(&slots), [1]);
        // A repeat of the last order is dropped.
        assert_eq!(order.show(&slots, 8), None);
    }

    #[test]
    fn a_later_frame_releases_the_hidden_ones_before_it() {
        let mut order = OutputOrder::new(4);
        let mut slots = vec![None; MAX_SLOTS];
        slots[0] = hidden(2);
        slots[3] = hidden(1);
        assert_eq!(
            order.show(&slots, 5).unwrap(),
            [Release::Slot(3), Release::Slot(0), Release::Current]
        );
    }
}
