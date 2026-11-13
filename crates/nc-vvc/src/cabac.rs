//! H.266 §9.3 binary arithmetic encoder with dual probability estimates.
//!
//! Carry/output machinery adapted from our HEVC coder (kvazaar, BSD-3-Clause).
//! VVC probability model follows `VVenC` (BSD-3-Clause-Clear); see `THIRD_PARTY.md`.
use crate::contexts::Context;
use nc_common::BitWriter;

/// Bits kept free below the output window when the encoder starts.
const INITIAL_BITS_LEFT: i32 = 23;
/// Output a byte once fewer than this many free bits remain.
const BITS_LEFT_REFILL: i32 = 12;
/// Initial `ivlCurrRange` (H.266 §9.3.2.5).
const INITIAL_RANGE: u32 = 510;
/// Renormalisation is needed while the range is below this.
const RANGE_RENORM: u32 = 256;

/// Encodes bins into one slice segment's `slice_data()`.
#[derive(Debug)]
pub struct CabacWriter {
    out: BitWriter,
    low: u32,
    range: u32,
    bits_left: i32,
    /// Completed bytes not yet written: the first is `buffered_byte`, the
    /// rest are `0xff` (only their count is kept).
    num_buffered_bytes: u32,
    buffered_byte: u32,
}

impl Default for CabacWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CabacWriter {
    /// A fresh encoder (H.266 §9.3.2.5, encoder side).
    #[must_use]
    pub fn new() -> Self {
        Self {
            out: BitWriter::new(),
            low: 0,
            range: INITIAL_RANGE,
            bits_left: INITIAL_BITS_LEFT,
            num_buffered_bytes: 0,
            buffered_byte: 0xff,
        }
    }

    /// Encode one context-coded bin (`EncodeDecision`, H.266 §9.3.4.3.2).
    pub fn encode_bin(&mut self, ctx: &mut Context, bin: bool) {
        let lps = ctx.lps(self.range);
        self.range -= lps;
        if bin == ctx.mps() {
            ctx.update(bin);
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
            ctx.update(bin);
            self.bits_left -= i32::from(shift);
        }
        self.flush_if_needed();
    }

    /// Encode equiprobable bins (`EncodeBypass`, H.266 §9.3.4.3.4): the
    /// `count` low bits of `value`, most significant first.
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

    /// Encode a terminating bin (`EncodeTerminate`, H.266 §9.3.4.3.5), e.g.
    /// `end_of_slice_flag`.
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
    /// (`EncodeFlush`, H.266 §9.3.4.3.5) and return the bits written. The
    /// caller appends `rbsp_slice_trailing_bits()`.
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
                self.put_byte(0xff);
                self.num_buffered_bytes -= 1;
            }
        }
        let bits = (24 - self.bits_left) as u32;
        self.out.put_bits(self.low >> 8, bits);
        self.out
    }

    fn flush_if_needed(&mut self) {
        if self.bits_left < BITS_LEFT_REFILL {
            self.write_byte();
        }
    }

    /// Move the top completed byte out of `low`, resolving carries.
    fn write_byte(&mut self) {
        let lead = self.low >> (24 - self.bits_left);
        self.bits_left += 8;
        self.low &= u32::MAX >> self.bits_left;
        if lead == 0xff {
            self.num_buffered_bytes += 1;
        } else if self.num_buffered_bytes > 0 {
            let carry = lead >> 8;
            let byte = self.buffered_byte + carry;
            self.buffered_byte = lead & 0xff;
            self.put_byte(byte);
            let filler = (0xff + carry) & 0xff;
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
        self.out.put_bits(byte & 0xff, 8);
    }
}

/// Renormalisation shift after an LPS, indexed by `rangeLps >> 3`.
const RENORM_SHIFT: [u8; 32] = [
    6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];
