//! MSB-first reader of one RBSP (emulation prevention already removed):
//! fixed-length fields, exp-Golomb codes (§9.1) and the peek/skip access
//! the CAVLC tables need (§9.2), with `more_rbsp_data()` (§7.2).
//!
//! Reads past the end return zero bits from `peek` but fail in `skip`, so
//! truncated data is reported as [`DecodeError::Truncated`] at the first
//! field that needs a missing bit.

use super::{DecodeError, Result};

/// Bytes loaded per peek: 32 bits at any bit offset need at most 5.
const WORD_BYTES: usize = 8;
/// Longest exp-Golomb prefix whose value fits 32 bits (§9.1).
const MAX_EXP_GOLOMB_ZEROS: u32 = 31;
/// Longest exp-Golomb prefix whose whole code fits one 32-bit peek.
const MAX_SHORT_EXP_GOLOMB_ZEROS: u32 = 15;

/// The reader over one RBSP.
#[derive(Debug, Clone)]
pub(super) struct Reader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
    /// Bit position of the `rbsp_stop_one_bit` (the last 1 bit of `data`),
    /// or the data length when there is none.
    stop_bit: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let stop_bit = data
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(data.len() * 8, |index| {
                index * 8 + 7 - data[index].trailing_zeros() as usize
            });
        Self {
            data,
            pos: 0,
            stop_bit,
        }
    }

    /// The next 32 bits, zero-filled past the end of the data.
    #[inline]
    pub fn peek32(&self) -> u32 {
        let byte = self.pos / 8;
        let word = match self.data.get(byte..byte + WORD_BYTES) {
            Some(bytes) => u64::from_be_bytes(bytes.try_into().expect("a word of bytes")),
            None => self.tail_word(byte),
        };
        (word << (self.pos % 8) >> 32) as u32
    }

    /// The bytes from `byte` on as a word, zero-filled past the end (the
    /// last few bytes of the data only).
    #[cold]
    fn tail_word(&self, byte: usize) -> u64 {
        let mut word = [0u8; WORD_BYTES];
        let rest = self.data.get(byte..).unwrap_or(&[]);
        word[..rest.len()].copy_from_slice(rest);
        u64::from_be_bytes(word)
    }

    /// The next `count` (1..=32) bits without consuming them.
    #[inline]
    pub fn peek(&self, count: u32) -> u32 {
        debug_assert!((1..=32).contains(&count));
        ((u64::from(self.peek32())) >> (32 - count)) as u32
    }

    /// Consume `count` bits.
    #[inline]
    pub fn skip(&mut self, count: usize) -> Result<()> {
        let end = self.pos + count;
        if end > self.data.len() * 8 {
            return Err(DecodeError::Truncated);
        }
        self.pos = end;
        Ok(())
    }

    /// Consume `count` bits without checking the end of the data: callers
    /// that parse a whole block from [`Reader::peek32`] words check it
    /// once afterwards ([`Reader::check_end`]).
    #[inline]
    pub fn consume(&mut self, count: u32) {
        self.pos += count as usize;
    }

    /// Fails if more bits were consumed than the data holds.
    #[inline]
    pub fn check_end(&self) -> Result<()> {
        if self.pos > self.data.len() * 8 {
            return Err(DecodeError::Truncated);
        }
        Ok(())
    }

    /// `u(n)` for `count` in 0..=32.
    #[inline]
    pub fn bits(&mut self, count: u32) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        let value = self.peek(count);
        self.skip(count as usize)?;
        Ok(value)
    }

    /// `u(1)` as a flag.
    #[inline]
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.bits(1)? == 1)
    }

    /// `ue(v)` (§9.1).
    #[inline]
    pub fn ue(&mut self) -> Result<u32> {
        let word = self.peek32();
        let zeros = word.leading_zeros();
        if zeros <= MAX_SHORT_EXP_GOLOMB_ZEROS {
            // The whole `2 * zeros + 1`-bit code is in `word`, and its value
            // plus one is that code read as a number: no branch on whether
            // there is a suffix.
            let length = 2 * zeros + 1;
            self.skip(length as usize)?;
            return Ok((word >> (u32::BITS - length)) - 1);
        }
        if zeros > MAX_EXP_GOLOMB_ZEROS {
            return Err(DecodeError::Malformed("exp-Golomb overflow"));
        }
        self.skip(zeros as usize + 1)?;
        let suffix = self.bits(zeros)?;
        Ok(((1u64 << zeros) - 1 + u64::from(suffix)) as u32)
    }

    /// `ue(v)` that must not exceed `max`.
    pub fn ue_max(&mut self, max: u32, what: &'static str) -> Result<u32> {
        let value = self.ue()?;
        if value > max {
            return Err(DecodeError::Malformed(what));
        }
        Ok(value)
    }

    /// `se(v)` (§9.1.1).
    #[inline]
    pub fn se(&mut self) -> Result<i32> {
        let code = i64::from(self.ue()?);
        let magnitude = (code + 1) / 2;
        Ok(if code % 2 == 0 { -magnitude } else { magnitude } as i32)
    }

    /// `se(v)` that must lie in `min..=max`.
    pub fn se_range(&mut self, min: i32, max: i32, what: &'static str) -> Result<i32> {
        let value = self.se()?;
        if !(min..=max).contains(&value) {
            return Err(DecodeError::Malformed(what));
        }
        Ok(value)
    }

    /// `te(v)` with range `max` (§9.1: one inverted bit when the range is 1).
    pub fn te(&mut self, max: u32) -> Result<u32> {
        if max == 1 {
            Ok(u32::from(!self.flag()?))
        } else {
            self.ue_max(max, "te(v) out of range")
        }
    }

    /// `more_rbsp_data()` (§7.2): whether any bit precedes the stop bit.
    pub fn more_rbsp_data(&self) -> bool {
        self.pos < self.stop_bit
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.pos.is_multiple_of(8)
    }

    /// The bytes from the current (byte-aligned) position on.
    pub fn remaining_bytes(&self) -> &'a [u8] {
        &self.data[(self.pos / 8).min(self.data.len())..]
    }

    /// Skip `pcm_alignment_zero_bit`s / `cabac_alignment_one_bit`s up to the
    /// next byte boundary; each must equal `bit` (§7.3.4, §7.3.5).
    pub fn align(&mut self, bit: bool) -> Result<()> {
        while !self.is_byte_aligned() {
            if self.flag()? != bit {
                return Err(DecodeError::Malformed("alignment bit"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_and_fields() {
        // 1 | 010 | 011 | 00100 | 0001000 | 101 | 1 (stop)
        // ue: 0, 1, 2, 3, 7; u(3) = 5.
        let data = [0b1010_0110, 0b0100_0001, 0b0001_0110];
        let mut r = Reader::new(&data);
        assert_eq!(r.ue().unwrap(), 0);
        assert_eq!(r.ue().unwrap(), 1);
        assert_eq!(r.se().unwrap(), -1);
        assert_eq!(r.ue().unwrap(), 3);
        assert_eq!(r.ue().unwrap(), 7);
        assert_eq!(r.bits(3).unwrap(), 5);
        assert!(!r.more_rbsp_data());
    }

    #[test]
    fn truncation_and_stop_bit() {
        let data = [0b0000_0001];
        let mut r = Reader::new(&data);
        assert!(r.more_rbsp_data());
        assert!(r.ue().is_err());
        let data = [0x80, 0x00];
        let r = Reader::new(&data);
        assert!(!r.more_rbsp_data());
    }
}
