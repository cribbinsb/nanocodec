//! The order in which decoded frames are output: a frame shown now first
//! releases the hidden frames (shown later, `implicit_output_frame`) due
//! before it in display order, then itself, then the hidden frames that
//! follow it without a gap; at the end of the stream every hidden frame
//! still due is released in display order (dav2d `lib.c`
//! `dav2d_queue_output` and `queue_flush`). Frames of one display order in
//! several embedded layers come out lowest layer first (dav2d after 237229e,
//! `get_output_order_diff`; AVM `derive_output_order_idx`).
//!
//! Only headers decide the order, so the header walk of
//! `examples/dump_headers.rs` uses this as the decoder does.
use super::frame_header::{FrameHeader, MAX_SLOTS, SlotHeader};
use super::references;

/// A picture to output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Release {
    /// The frame being shown.
    Current,
    /// The frame held in this slot.
    Slot(usize),
}

/// A frame's place in output order: its display order, then its embedded
/// layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    /// Display order.
    pub order: u32,
    /// Embedded layer.
    pub embedded_layer: u8,
}

impl Position {
    /// The position of the frame of `header`.
    #[must_use]
    pub fn of(header: &FrameHeader) -> Self {
        Self {
            order: header.order,
            embedded_layer: header.embedded_layer,
        }
    }
}

/// The output order's state: the position of the last picture output.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutputOrder {
    /// dav2d `dpb_poc` and `dpb_mlayer`: order 0 before any output (so a
    /// first frame of order 0 is never taken for a repeat).
    last: Position,
    /// Whether the sequence codes order hints (without them every order
    /// difference is zero and nothing is shown later).
    ordered: bool,
    /// Embedded layers of the sequence (`max_mlayer_id + 1`).
    layers: i32,
}

impl OutputOrder {
    /// The state for a sequence with `order_hint_bits` and embedded layers
    /// up to `max_embedded_layer`.
    #[must_use]
    pub fn new(order_hint_bits: u8, max_embedded_layer: u8) -> Self {
        Self {
            last: Position::default(),
            ordered: order_hint_bits != 0,
            layers: i32::from(max_embedded_layer) + 1,
        }
    }

    /// Forget the last output, as at the start of a sequence (a
    /// closed-loop key frame starting a temporal unit).
    pub fn restart(&mut self) {
        self.last = Position::default();
    }

    /// The pictures to output when a frame at `position` is shown, with
    /// `slots` holding the frames decoded before it; `None` when it
    /// repeats the last output's position (AVM rejects the stream; dav2d
    /// drops the picture).
    pub fn show(
        &mut self,
        slots: &[Option<SlotHeader>],
        position: Position,
    ) -> Option<Vec<Release>> {
        let mut released = Vec::new();
        if self.last.order != 0 && position == self.last {
            return None;
        }
        let mut taken = 0_u32;
        while let Some(slot) = self.earliest_hidden(slots, taken, Some(position)) {
            taken |= 1 << slot;
            self.last = hidden_order(slots, slot, 0).expect("a hidden frame");
            released.push(Release::Slot(slot));
        }
        released.push(Release::Current);
        self.last = position;
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
        before: Option<Position>,
    ) -> Option<usize> {
        let mut earliest: Option<(usize, Position)> = None;
        for slot in 0..slots.len().min(MAX_SLOTS) {
            let Some(position) = hidden_order(slots, slot, taken) else {
                continue;
            };
            let bound = earliest.map(|(_, earliest)| earliest).or(before);
            if self.difference(position, self.last) > 0
                && bound.is_none_or(|bound| self.difference(position, bound) < 0)
            {
                earliest = Some((slot, position));
            }
        }
        earliest.map(|(slot, _)| slot)
    }

    /// `a - b` in output order: display order difference (dav2d
    /// `get_poc_diff`) in steps of the layer count, plus the layer
    /// difference (dav2d `get_output_order_diff`).
    fn difference(self, a: Position, b: Position) -> i32 {
        let order = if self.ordered {
            references::order_difference(a.order, b.order)
        } else {
            0
        };
        order * self.layers + i32::from(a.embedded_layer) - i32::from(b.embedded_layer)
    }
}

/// The position of the hidden frame in `slot`, unless the slot is empty,
/// in `taken`, or holds a frame not shown later.
fn hidden_order(slots: &[Option<SlotHeader>], slot: usize, taken: u32) -> Option<Position> {
    if taken & (1 << slot) != 0 {
        return None;
    }
    let held = slots.get(slot)?.as_ref()?;
    held.header
        .show_implicit
        .then(|| Position::of(&held.header))
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

    fn at(order: u32) -> Position {
        Position {
            order,
            embedded_layer: 0,
        }
    }

    /// Random access as avmenc codes it: key frame 0, hidden 8 and 4, then
    /// 1..3 shown at once release nothing hidden until 4 is due.
    #[test]
    fn hidden_frames_come_out_in_display_order() {
        let mut order = OutputOrder::new(4, 0);
        let mut slots = vec![None; MAX_SLOTS];
        assert_eq!(order.show(&slots, at(0)).unwrap(), [Release::Current]);
        slots[1] = hidden(8);
        slots[2] = hidden(4);
        assert_eq!(order.show(&slots, at(1)).unwrap(), [Release::Current]);
        assert_eq!(order.show(&slots, at(2)).unwrap(), [Release::Current]);
        // 3 is shown, then 4 follows it without a gap.
        assert_eq!(
            order.show(&slots, at(3)).unwrap(),
            [Release::Current, Release::Slot(2)]
        );
        // 6 is shown: 5 was never coded, so nothing before it is due.
        assert_eq!(order.show(&slots, at(6)).unwrap(), [Release::Current]);
        assert_eq!(order.flush(&slots), [1]);
        // A repeat of the last order is dropped.
        assert_eq!(order.show(&slots, at(8)), None);
    }

    #[test]
    fn a_later_frame_releases_the_hidden_ones_before_it() {
        let mut order = OutputOrder::new(4, 0);
        let mut slots = vec![None; MAX_SLOTS];
        slots[0] = hidden(2);
        slots[3] = hidden(1);
        assert_eq!(
            order.show(&slots, at(5)).unwrap(),
            [Release::Slot(3), Release::Slot(0), Release::Current]
        );
    }
}
