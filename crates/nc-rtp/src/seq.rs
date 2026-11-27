//! Unwrapping of RTP's wrapping counters (16-bit sequence numbers, 32-bit
//! timestamps, the Dependency Descriptor's 16-bit frame numbers) into
//! monotone 64-bit values, so the receiver can order and subtract them.
//!
//! Each new value is placed at the 64-bit position nearest the previous one
//! (RFC 3550 Appendix A.1's rule, without its probation logic): a packet up
//! to half the counter range late is "before", anything else "after".

/// Unwraps one counter of `BITS` bits (16 or 32).
#[derive(Debug, Clone, Copy, Default)]
pub struct Unwrapper<const BITS: u32> {
    last: Option<u64>,
}

impl<const BITS: u32> Unwrapper<BITS> {
    const RANGE: u64 = 1 << BITS;

    /// A fresh unwrapper; its first value maps to itself plus one range, so
    /// early packets that arrive late can still be placed below it.
    #[must_use]
    pub const fn new() -> Self {
        Self { last: None }
    }

    /// The 64-bit value of `value`, nearest the previous one.
    pub fn unwrap(&mut self, value: u64) -> u64 {
        let value = value & (Self::RANGE - 1);
        let unwrapped = match self.last {
            None => Self::RANGE + value,
            Some(last) => {
                let base = last & !(Self::RANGE - 1);
                let candidates = [
                    base.wrapping_sub(Self::RANGE) + value,
                    base + value,
                    base + Self::RANGE + value,
                ];
                *candidates
                    .iter()
                    .min_by_key(|&&c| c.abs_diff(last))
                    .expect("three candidates")
            }
        };
        if self.last.is_none_or(|last| unwrapped > last) {
            self.last = Some(unwrapped);
        }
        unwrapped
    }
}

/// 16-bit sequence numbers.
pub type SequenceUnwrapper = Unwrapper<16>;
/// 32-bit timestamps.
pub type TimestampUnwrapper = Unwrapper<32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_numbers_unwrap_across_the_wrap_and_late_arrivals() {
        let mut seq = SequenceUnwrapper::new();
        let a = seq.unwrap(65534);
        assert_eq!(seq.unwrap(65535), a + 1);
        assert_eq!(seq.unwrap(1), a + 3);
        assert_eq!(seq.unwrap(0), a + 2); // late, after the wrap
        assert_eq!(seq.unwrap(2), a + 4);
        assert_eq!(seq.unwrap(65533), a - 1); // late, before the wrap
    }

    #[test]
    fn timestamps_unwrap() {
        let mut ts = TimestampUnwrapper::new();
        let a = ts.unwrap(u64::from(u32::MAX) - 1000);
        assert_eq!(ts.unwrap(2000), a + 3001);
    }
}
