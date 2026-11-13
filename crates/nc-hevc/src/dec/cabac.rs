//! The HEVC decoder's view of the shared CABAC engine
//! ([`nc_common::cabac::CabacReader`], H.265 §9.3.4.3): decisions on the
//! crate's [`Context`] variables and the `EGk` bypass values of §9.3.3.3.
//!
//! The engine counts the bits it consumed exactly, which matters at the two
//! places the slice data leaves arithmetic coding: after `pcm_flag` and
//! after `end_of_subset_one_bit` the following data starts at the next byte
//! boundary (the bits consumed so far end with a 1 and alignment zeros,
//! §9.3.2.6). Reads past the end of the substream return zero bits, as a
//! conforming stream never depends on them; [`Engine::overrun`] reports how
//! far a malformed one went so the caller can reject it.

use crate::cabac::Context;
use nc_common::cabac::CabacReader;
use nc_common::decode::DecodeError;

/// Longest Exp-Golomb prefix a conforming bypass value can have.
const MAX_EG_PREFIX: u32 = 32;

/// One substream's arithmetic decoder.
#[derive(Debug, Clone)]
pub(super) struct Engine<'a>(CabacReader<'a>);

impl<'a> Engine<'a> {
    /// Initialise at byte `start` of `data` (§9.3.2.6).
    pub fn new(data: &'a [u8], start: usize) -> Result<Self, DecodeError> {
        CabacReader::new(data, start).map(Self)
    }

    /// The substream's bytes.
    pub fn data(&self) -> &'a [u8] {
        self.0.data()
    }

    /// The next byte after the consumed bits: where PCM samples or the next
    /// substream start.
    pub fn next_byte(&self) -> usize {
        self.0.next_byte()
    }

    /// Bits consumed so far, for counting what syntax costs (analysis).
    #[cfg(feature = "analysis")]
    pub fn bits_consumed(&self) -> usize {
        self.0.bits_consumed()
    }

    /// Whether decoding has read well past the end of its data.
    pub fn overrun(&self) -> bool {
        self.0.overrun()
    }

    /// `DecodeDecision` (§9.3.4.3.2).
    #[inline]
    pub fn bin(&mut self, ctx: &mut Context) -> bool {
        self.0.decision(&mut ctx.0)
    }

    /// `DecodeBypass` (§9.3.4.3.4).
    #[inline]
    pub fn bypass(&mut self) -> bool {
        self.0.bypass()
    }

    /// `count` bypass bins, most significant first.
    #[inline]
    pub fn bypass_bits(&mut self, count: u32) -> u32 {
        self.0.bypass_bits(count)
    }

    /// Up to `max` bypass bins of 1 ended by a 0 (not read at `max`).
    #[inline]
    pub fn bypass_unary(&mut self, max: u32) -> u32 {
        self.0.bypass_unary(max)
    }

    /// `DecodeTerminate` (§9.3.4.3.5).
    pub fn terminate(&mut self) -> bool {
        self.0.terminate()
    }

    /// A k-th order Exp-Golomb bypass value (`EGk`, §9.3.3.3).
    pub fn exp_golomb(&mut self, order: u32) -> Result<u32, DecodeError> {
        let prefix = self.0.bypass_unary(MAX_EG_PREFIX);
        let order = order + prefix;
        if prefix >= MAX_EG_PREFIX || order >= MAX_EG_PREFIX {
            return Err(DecodeError::Malformed("CABAC Exp-Golomb prefix"));
        }
        // The prefix contributes 2^order − 2^k: sum of 2^(k+i), i < prefix.
        let base = (1u32 << order) - (1u32 << (order - prefix));
        base.checked_add(self.0.bypass_bits(order))
            .ok_or(DecodeError::Malformed("CABAC Exp-Golomb value"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacWriter;

    #[test]
    fn bins_round_trip_through_the_encoder() {
        let mut writer = CabacWriter::new();
        let mut ctx = Context::new(154, 30);
        let pattern: Vec<bool> = (0..2000).map(|i| (i * 7 % 11) < 3).collect();
        for (i, &bin) in pattern.iter().enumerate() {
            if i % 5 == 0 {
                writer.encode_bypass(u32::from(bin), 1);
            } else {
                writer.encode_bin(&mut ctx, bin);
            }
        }
        writer.encode_terminate(true);
        let mut bits = writer.finish();
        bits.trailing_bits();
        let bytes = bits.into_bytes();
        let mut engine = Engine::new(&bytes, 0).unwrap();
        let mut ctx = Context::new(154, 30);
        for (i, &bin) in pattern.iter().enumerate() {
            let got = if i % 5 == 0 {
                engine.bypass()
            } else {
                engine.bin(&mut ctx)
            };
            assert_eq!(got, bin, "bin {i}");
        }
        assert!(engine.terminate());
    }
}
