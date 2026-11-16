//! The CABAC arithmetic encoding engine shared by the HEVC and H.266
//! encoders (HEVC §9.3.4.3, H.266 §9.3.4.3: the same range coder; each
//! codec keeps its own probability model and gives [`CabacEncoder::
//! encode_decision`] the LPS range it derives).
//!
//! The engine follows kvazaar's `cabac.c` (BSD-3-Clause, see
//! `THIRD_PARTY.md`), which is HM's: `low` keeps `bits_left` free bits below
//! a 32-bit window, completed bytes are held back while they are `0xff` (a
//! later carry could still change them) and released once a carry can no
//! longer reach them. Slice data starts byte-aligned and the coder emits
//! whole bytes, so they are pushed straight into a byte vector rather than
//! bit by bit through a [`BitWriter`].
//!
//! H.264 keeps its own engine (`nc-h264` `cabac`): a different register
//! layout and renormalisation, and a whole shared engine was measured slower
//! for HEVC (`docs/handoff/astra-shared-cabac.md`).

use crate::BitWriter;

/// Bits kept free below the output window when the encoder starts.
const INITIAL_BITS_LEFT: i32 = 23;
/// Output a byte once fewer than this many free bits remain.
const BITS_LEFT_REFILL: i32 = 12;
/// Initial `ivlCurrRange` (HEVC §9.3.5.2, H.266 §9.3.2.5).
const INITIAL_RANGE: u32 = 510;
/// Renormalisation is needed while the range is below this.
const RANGE_RENORM: u32 = 256;
/// Bits of `low` above the output window.
const WINDOW_TOP: i32 = 24;
/// A completed byte that a later carry could still change.
const CARRY_PENDING: u32 = 0xff;

/// Renormalisation shift after an LPS, indexed by `rangeLps >> 3`.
const RENORM_SHIFT: [u8; 32] = [
    6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// The arithmetic coder of one slice segment's data.
#[derive(Debug)]
pub struct CabacEncoder {
    /// Completed bytes.
    bytes: Vec<u8>,
    low: u32,
    range: u32,
    bits_left: i32,
    /// Completed bytes not yet written: the first is `buffered_byte`, the
    /// rest are `0xff` (only their count is kept).
    num_buffered_bytes: u32,
    buffered_byte: u32,
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CabacEncoder {
    /// A fresh encoder (HEVC §9.3.5.2, H.266 §9.3.2.5).
    #[must_use]
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            low: 0,
            range: INITIAL_RANGE,
            bits_left: INITIAL_BITS_LEFT,
            num_buffered_bytes: 0,
            buffered_byte: CARRY_PENDING,
        }
    }

    /// The current range (`ivlCurrRange`), from which the model derives
    /// the LPS range.
    #[inline]
    #[must_use]
    pub fn range(&self) -> u32 {
        self.range
    }

    /// The arithmetic part of `EncodeDecision` (HEVC §9.3.4.3.2, H.266
    /// §9.3.4.3.2): code the MPS (`mps`) or the LPS of an interval whose LPS
    /// range is `lps`, then renormalise. The model updates its context
    /// itself.
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "one call per context-coded bin, the encoders' hottest path"
    )]
    pub fn encode_decision(&mut self, lps: u32, mps: bool) {
        self.range -= lps;
        if mps {
            if self.range >= RANGE_RENORM {
                return;
            }
            self.low <<= 1;
            self.range <<= 1;
            self.bits_left -= 1;
        } else {
            let shift = RENORM_SHIFT[(lps >> 3) as usize];
            self.low = (self.low + self.range) << shift;
            self.range = lps << shift;
            self.bits_left -= i32::from(shift);
        }
        self.flush_if_needed();
    }

    /// Encode equiprobable bins (`EncodeBypass`, HEVC §9.3.4.3.4, H.266
    /// §9.3.4.3.4): the `count` low bits of `value`, most significant first.
    #[inline]
    pub fn encode_bypass(&mut self, value: u32, count: u32) {
        let mut remaining = count;
        while remaining > 0 {
            let chunk = remaining.min(8);
            remaining -= chunk;
            let bits = (value >> remaining) & ((1 << chunk) - 1);
            self.low = (self.low << chunk) + self.range * bits;
            self.bits_left -= chunk as i32;
            self.flush_if_needed();
        }
    }

    /// Encode a terminating bin (`EncodeTerminate`, HEVC §9.3.4.3.5, H.266
    /// §9.3.4.3.5), e.g. `end_of_slice_segment_flag`.
    pub fn encode_terminate(&mut self, bin: bool) {
        self.range -= 2;
        if bin {
            self.low += self.range;
            self.low <<= 7;
            self.range = 2 << 7;
            self.bits_left -= 7;
        } else if self.range >= RANGE_RENORM {
            return;
        } else {
            self.low <<= 1;
            self.range <<= 1;
            self.bits_left -= 1;
        }
        self.flush_if_needed();
    }

    /// Flush the encoder after the terminating bin of the slice segment
    /// (`EncodeFlush`) and return the bits written. The caller appends the
    /// slice's trailing bits.
    #[must_use]
    pub fn finish(mut self) -> BitWriter {
        if self.low >> (32 - self.bits_left) != 0 {
            self.put_byte(self.buffered_byte + 1);
            while self.num_buffered_bytes > 1 {
                self.put_byte(0);
                self.num_buffered_bytes -= 1;
            }
            self.low -= 1 << (32 - self.bits_left);
        } else {
            if self.num_buffered_bytes > 0 {
                self.put_byte(self.buffered_byte);
            }
            while self.num_buffered_bytes > 1 {
                self.put_byte(CARRY_PENDING);
                self.num_buffered_bytes -= 1;
            }
        }
        let bits = (WINDOW_TOP - self.bits_left) as u32;
        let mut out = BitWriter::new();
        out.put_bytes(&self.bytes);
        out.put_bits(self.low >> 8, bits);
        out
    }

    /// This coder moved out (its bytes taken, its registers copied), for a
    /// caller that codes a run of bins on a local copy and stores it back:
    /// the registers of a coder reached through a reference stay in memory
    /// (loaded and stored around every bin), a local coder's do not.
    #[must_use]
    pub fn take(&mut self) -> Self {
        Self {
            bytes: std::mem::take(&mut self.bytes),
            ..*self
        }
    }

    #[inline]
    fn flush_if_needed(&mut self) {
        if self.bits_left < BITS_LEFT_REFILL {
            self.write_byte();
        }
    }

    /// Move the top completed byte out of `low`, resolving carries.
    fn write_byte(&mut self) {
        let lead = self.low >> (WINDOW_TOP - self.bits_left);
        self.bits_left += 8;
        self.low &= u32::MAX >> self.bits_left;
        if lead == CARRY_PENDING {
            self.num_buffered_bytes += 1;
        } else if self.num_buffered_bytes > 0 {
            let carry = lead >> 8;
            let byte = self.buffered_byte + carry;
            self.buffered_byte = lead & 0xff;
            self.put_byte(byte);
            let filler = (CARRY_PENDING + carry) & 0xff;
            while self.num_buffered_bytes > 1 {
                self.put_byte(filler);
                self.num_buffered_bytes -= 1;
            }
        } else {
            self.num_buffered_bytes = 1;
            self.buffered_byte = lead;
        }
    }

    fn put_byte(&mut self, byte: u32) {
        self.bytes.push((byte & 0xff) as u8);
    }
}
