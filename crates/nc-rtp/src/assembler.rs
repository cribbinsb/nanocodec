//! The receiver's reorder window and unit assembly: packets in, in any
//! order, units (all packets of one timestamp, in sequence order) out, with
//! the gaps around and inside them marked.
//!
//! A packet waits until every earlier sequence number has arrived, or until
//! the window overflows (the newest packet is `window` or more ahead of the
//! oldest missing one) or the caller flushes, when the missing ones count
//! as lost. A unit ends at its marker bit or at the first packet of a later
//! timestamp (AV1 RTP §4.2: a receiver must cope with a lost marker). What
//! to make of a unit with gaps is the depacketizer's decision.

use std::collections::BTreeMap;

use crate::payload::ParsedPayload;
use crate::seq::{SequenceUnwrapper, TimestampUnwrapper};

/// One received packet, parsed, keyed by its unwrapped sequence number.
#[derive(Debug, Clone)]
pub(crate) struct Stored {
    /// Unwrapped RTP timestamp.
    pub timestamp: u64,
    /// The RTP timestamp as sent.
    pub raw_timestamp: u32,
    pub marker: bool,
    pub parsed: ParsedPayload,
}

/// The packets of one unit.
#[derive(Debug, Clone)]
pub(crate) struct AssembledUnit {
    /// The unit's packets with their unwrapped sequence numbers, in order.
    pub packets: Vec<(u64, Stored)>,
    /// Packets just before the unit were lost: they may have been its
    /// beginning, or whole units.
    pub gap_before: bool,
    /// Packets between the unit's first and last, or after its last (no
    /// marker seen), were lost.
    pub gap_inside: bool,
}

/// Reorder window and unit boundaries.
#[derive(Debug, Clone)]
pub(crate) struct Assembler {
    window: u64,
    packets: BTreeMap<u64, Stored>,
    /// The lowest sequence number not yet consumed.
    next: Option<u64>,
    sequence: SequenceUnwrapper,
    timestamp: TimestampUnwrapper,
    /// The last consumed unit ended in a gap (the next one starts after it).
    pending_gap: bool,
    /// Packets declared lost.
    pub lost_packets: u64,
    /// Packets that arrived after their place was consumed (too late, or
    /// duplicates).
    pub late_packets: u64,
}

/// Where a unit's walk through the window ended.
enum Walk {
    /// The unit is decided: its packets are `first..end` (some missing),
    /// with or without gaps inside.
    Done { end: u64, gap_inside: bool },
    /// Missing packets may still arrive.
    Wait,
}

impl Assembler {
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1) as u64,
            packets: BTreeMap::new(),
            next: None,
            sequence: SequenceUnwrapper::new(),
            timestamp: TimestampUnwrapper::new(),
            pending_gap: false,
            lost_packets: 0,
            late_packets: 0,
        }
    }

    /// Store one packet.
    pub fn push(&mut self, sequence: u16, timestamp: u32, marker: bool, parsed: ParsedPayload) {
        let seq = self.sequence.unwrap(u64::from(sequence));
        if self.next.is_some_and(|next| seq < next) || self.packets.contains_key(&seq) {
            self.late_packets += 1;
            return;
        }
        let stored = Stored {
            timestamp: self.timestamp.unwrap(u64::from(timestamp)),
            raw_timestamp: timestamp,
            marker,
            parsed,
        };
        self.packets.insert(seq, stored);
    }

    /// True when waiting is over for missing packets below the newest.
    /// The window is measured from the oldest missing packet, so a unit
    /// longer than the window still waits for its own late packets.
    fn overflowed(&self, next: u64) -> bool {
        let mut missing = next;
        for &seq in self.packets.range(next..).map(|(seq, _)| seq) {
            if seq != missing {
                break;
            }
            missing += 1;
        }
        self.packets
            .last_key_value()
            .is_some_and(|(&newest, _)| newest >= missing + self.window)
    }

    /// The next decided unit, if any. `flush` declares every missing packet
    /// lost (end of stream).
    pub fn pop(&mut self, flush: bool) -> Option<AssembledUnit> {
        let (&oldest, first) = self.packets.first_key_value()?;
        let starts_clean = first
            .parsed
            .pieces
            .first()
            .is_some_and(|piece| !piece.continues_previous);
        let stream_start = self.next.is_none();
        let next = *self.next.get_or_insert(oldest);
        let give_up = flush || self.overflowed(next);
        let mut gap_before = std::mem::take(&mut self.pending_gap);
        if stream_start && !give_up {
            // Where the stream starts is unknown: earlier packets of the
            // first unit may still be on their way. Wait for a clean first
            // packet and for a later unit to begin (one unit of latency, at
            // the start only).
            let later_unit = self.packets.values().any(|p| p.timestamp > first.timestamp);
            if !starts_clean || !later_unit {
                self.next = None;
                return None;
            }
        } else if stream_start && !starts_clean {
            gap_before = true;
        }
        let mut start = next;
        if oldest > next {
            if !give_up {
                return None;
            }
            self.lost_packets += oldest - next;
            gap_before = true;
            start = oldest;
        }
        match self.walk(start, give_up, flush) {
            Walk::Wait => {
                self.pending_gap = gap_before;
                self.next = Some(start);
                None
            }
            Walk::Done { end, gap_inside } => {
                let packets: Vec<(u64, Stored)> = (start..end)
                    .filter_map(|seq| self.packets.remove(&seq).map(|p| (seq, p)))
                    .collect();
                self.lost_packets += (end - start) - packets.len() as u64;
                let marker = packets.last().is_some_and(|(_, p)| p.marker);
                self.pending_gap = gap_inside && !marker;
                self.next = Some(end);
                Some(AssembledUnit {
                    packets,
                    gap_before,
                    gap_inside,
                })
            }
        }
    }

    /// Find the end of the unit starting at `start` (present). `give_up`:
    /// missing packets below the newest are lost; `flush`: so are the
    /// ones after it (end of stream).
    fn walk(&self, start: u64, give_up: bool, flush: bool) -> Walk {
        let timestamp = self.packets[&start].timestamp;
        let mut seq = start;
        let mut gap_inside = false;
        loop {
            match self.packets.get(&seq) {
                Some(packet) if packet.timestamp != timestamp => {
                    return Walk::Done {
                        end: seq,
                        gap_inside,
                    };
                }
                Some(packet) => {
                    seq += 1;
                    if packet.marker {
                        return Walk::Done {
                            end: seq,
                            gap_inside,
                        };
                    }
                }
                None => {
                    if !give_up {
                        return Walk::Wait;
                    }
                    let Some((&after, packet)) = self.packets.range(seq..).next() else {
                        if !flush {
                            // The unit's next packet is not late, only not
                            // sent yet.
                            return Walk::Wait;
                        }
                        // End of stream: the unit's end was lost.
                        return Walk::Done {
                            end: seq,
                            gap_inside: true,
                        };
                    };
                    if packet.timestamp != timestamp {
                        // The gap ends the unit; the lost packets may also
                        // have begun the next one.
                        return Walk::Done {
                            end: seq,
                            gap_inside: true,
                        };
                    }
                    gap_inside = true;
                    seq = after;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::{PacketInfo, Piece};

    fn payload(byte: u8) -> ParsedPayload {
        ParsedPayload {
            pieces: vec![Piece::whole(vec![byte])],
            info: PacketInfo::default(),
        }
    }

    fn bytes(unit: &AssembledUnit) -> Vec<u8> {
        unit.packets
            .iter()
            .map(|(_, p)| p.parsed.pieces[0].data[0])
            .collect()
    }

    #[test]
    fn reordered_packets_form_whole_units() {
        let mut a = Assembler::new(16);
        a.push(1, 100, true, payload(2));
        a.push(0, 100, false, payload(1));
        a.push(3, 200, true, payload(4));
        let unit = a.pop(false).unwrap();
        assert_eq!(bytes(&unit), vec![1, 2]);
        assert!(!unit.gap_before && !unit.gap_inside);
        assert!(a.pop(false).is_none(), "waits for 2");
        a.push(2, 200, false, payload(3));
        assert_eq!(bytes(&a.pop(false).unwrap()), vec![3, 4]);
        assert_eq!(a.lost_packets, 0);
    }

    #[test]
    fn overflow_declares_loss_and_marks_gaps() {
        let mut a = Assembler::new(4);
        a.push(10, 100, false, payload(1));
        // 11 (the marker packet of ts 100) is lost.
        a.push(12, 200, true, payload(3));
        assert!(a.pop(false).is_none());
        for seq in 13..17 {
            a.push(seq, 300 + u32::from(seq), true, payload(seq as u8));
        }
        let first = a.pop(false).unwrap();
        assert_eq!(bytes(&first), vec![1]);
        assert!(first.gap_inside);
        let second = a.pop(false).unwrap();
        assert!(second.gap_before && !second.gap_inside);
        assert_eq!(a.lost_packets, 1);
    }

    #[test]
    fn the_stream_start_waits_for_its_first_packet_and_the_next_unit() {
        let mut a = Assembler::new(8);
        let mut tail = payload(3);
        tail.pieces[0].continues_previous = true;
        a.push(2, 100, true, tail);
        a.push(1, 100, false, payload(2));
        assert!(
            a.pop(false).is_none(),
            "1 starts cleanly, but 0 may precede it"
        );
        a.push(0, 100, false, payload(1));
        assert!(a.pop(false).is_none(), "no later unit yet");
        a.push(3, 200, true, payload(4));
        assert_eq!(bytes(&a.pop(false).unwrap()), vec![1, 2, 3]);
        assert_eq!(bytes(&a.pop(false).unwrap()), vec![4]);
    }

    #[test]
    fn a_unit_longer_than_the_window_waits_for_its_end() {
        let mut a = Assembler::new(4);
        for seq in 0..10 {
            a.push(seq, 100, false, payload(seq as u8));
            assert!(a.pop(false).is_none());
        }
        a.push(10, 100, true, payload(10));
        a.push(11, 200, true, payload(11)); // the next unit: the stream start is known
        let unit = a.pop(false).unwrap();
        assert_eq!(unit.packets.len(), 11);
        assert!(!unit.gap_inside);
    }

    #[test]
    fn late_packets_of_a_long_unit_are_waited_for() {
        let mut a = Assembler::new(4);
        for seq in (0..10).filter(|&s| s != 7) {
            a.push(seq, 100, false, payload(seq as u8));
        }
        assert!(a.pop(false).is_none(), "7 is only 2 packets late");
        a.push(7, 100, false, payload(7));
        a.push(10, 100, true, payload(10));
        a.push(11, 200, true, payload(11));
        let unit = a.pop(false).unwrap();
        assert!(!unit.gap_inside && unit.packets.len() == 11);
    }

    #[test]
    fn a_lost_marker_ends_the_unit_at_the_next_timestamp() {
        let mut a = Assembler::new(8);
        a.push(0, 100, false, payload(1));
        a.push(1, 200, true, payload(2));
        let unit = a.pop(false).unwrap();
        assert_eq!(bytes(&unit), vec![1]);
        assert!(!unit.gap_inside, "boundary by timestamp, nothing missing");
    }

    #[test]
    fn flush_releases_everything_and_late_packets_are_counted() {
        let mut a = Assembler::new(64);
        a.push(5, 1, false, payload(1));
        a.push(7, 1, true, payload(3));
        assert!(a.pop(false).is_none());
        let unit = a.pop(true).unwrap();
        assert_eq!(bytes(&unit), vec![1, 3]);
        assert!(unit.gap_inside);
        a.push(6, 1, false, payload(2));
        assert_eq!(a.late_packets, 1);
    }
}
