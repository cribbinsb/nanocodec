//! VP9 boolean arithmetic coding, specification §9.2 (Boolean decoding).
//!
//! The byte-emission algorithm is reimplemented from libvpx v1.16.0,
//! `vpx_dsp/bitwriter.{h,c}` (`WebM` Project, BSD-3-Clause). See `THIRD_PARTY.md`.
//! The decoder keeps the specification's arithmetic in a word-wide window.

use nc_common::decode::{DecodeError, Result};

/// Boolean arithmetic encoder with VP9's initial zero marker and final padding.
#[derive(Debug, Clone)]
pub struct BoolWriter {
    bytes: Vec<u8>,
    low: u32,
    range: u32,
    pending: i32,
}

impl Default for BoolWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolWriter {
    /// Start a new compressed header or tile partition, including its zero marker.
    #[must_use]
    pub fn new() -> Self {
        let mut writer = Self {
            bytes: Vec::new(),
            low: 0,
            range: 255,
            pending: -24,
        };
        writer.write(false, 128);
        writer
    }

    /// Encode a bit with an eight-bit probability of zero (0..255).
    pub fn write(&mut self, bit: bool, probability: u8) {
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
        if bit {
            self.low += split;
            self.range -= split;
        } else {
            self.range = split;
        }
        let mut shift = self.range.leading_zeros() as i32 - 24;
        self.range <<= shift;
        self.pending += shift;
        if self.pending >= 0 {
            let offset = shift - self.pending;
            if self.low.wrapping_shl((offset - 1) as u32) & 0x8000_0000 != 0 {
                // Carry through any run of FF bytes. The initial zero marker
                // ensures there is a preceding byte in which the carry fits.
                let mut index = self.bytes.len();
                loop {
                    index = index.checked_sub(1).expect("initial marker absorbs carry");
                    if self.bytes[index] != 255 {
                        self.bytes[index] += 1;
                        break;
                    }
                    self.bytes[index] = 0;
                }
            }
            self.bytes.push((self.low >> (24 - offset)) as u8);
            self.low = self.low.wrapping_shl(offset as u32) & 0x00ff_ffff;
            shift = self.pending;
            self.pending -= 8;
        }
        self.low = self.low.wrapping_shl(shift as u32);
    }

    /// Finalize the partition; also prevent collision with a superframe marker.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write(false, 128);
        }
        if self.bytes.last().is_some_and(|byte| byte & 0xe0 == 0xc0) {
            self.bytes.push(0);
        }
        self.bytes
    }
}

/// Anything that takes boolean-coded bits: the writer itself, or a tile
/// recorder that writes them later.
pub(crate) trait BitSink {
    /// Encode `bit` with probability-of-zero `prob` (/256).
    fn write(&mut self, bit: bool, prob: u8);

    /// Encode `bit` at mode/mv probability slot `slot`, defaulting to [`Self::write`].
    fn write_mode(&mut self, bit: bool, prob: u8, _slot: usize) {
        self.write(bit, prob);
    }
}

impl BitSink for BoolWriter {
    fn write(&mut self, bit: bool, prob: u8) {
        BoolWriter::write(self, bit, prob);
    }
}

/// Bits of the decoder window ([`BoolReader::value`]).
const WINDOW_BITS: i32 = u64::BITS as i32;
/// Bits of the active arithmetic-decoder value at the top of the window.
const ACTIVE_BITS: i32 = 8;

/// Bounded VP9 boolean decoder. Each instance reads exactly one partition
/// slice.
///
/// The arithmetic is §9.2's, kept in a 64-bit window so that one refill
/// serves several symbols (the layout of libvpx `vpx_dsp/bitreader.{h,c}`,
/// BSD-3-Clause, see `THIRD_PARTY.md`): the top [`ACTIVE_BITS`] are the
/// specification's `BoolValue`, the `buffered` bits below them are the next
/// partition bits, and a normalization shifts both together. Past the end
/// of the partition the window fills with zero bits; whether a read needed
/// one of them (the specification's failure) is reported once, by
/// [`BoolReader::finish`], so that reads need no error path.
#[derive(Debug, Clone)]
pub struct BoolReader<'a> {
    data: &'a [u8],
    /// The next unread byte of `data`.
    position: usize,
    value: u64,
    /// Valid bits below the active ones (negative between a normalization
    /// and the refill the next read makes).
    buffered: i32,
    range: u32,
    /// Zero bits the window took in beyond the end of `data`.
    padding: i32,
}

impl<'a> BoolReader<'a> {
    /// Initialize and consume the required zero marker.
    /// # Errors
    /// Rejects an empty partition or a nonzero marker.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(DecodeError::Truncated);
        }
        let mut reader = Self {
            data,
            position: 0,
            value: 0,
            buffered: -ACTIVE_BITS,
            range: 255,
            padding: 0,
        };
        reader = reader.refilled();
        if reader.read(128) {
            return Err(DecodeError::Malformed("VP9 boolean marker must be zero"));
        }
        Ok(reader)
    }

    /// The reader with its window topped up with whole bytes (zero bytes
    /// past the end). By value, so that a reader copied into locals for a
    /// hot loop (`coeff::read`) stays in registers across this rare call.
    #[inline(never)]
    #[must_use]
    fn refilled(mut self) -> Self {
        // Bit offset (from the bottom) of the first free window bit.
        let mut free = WINDOW_BITS - ACTIVE_BITS - (self.buffered + ACTIVE_BITS);
        if let Some(bytes) = self.data.get(self.position..self.position + 8) {
            let next = u64::from_be_bytes(bytes.try_into().expect("eight bytes"));
            // Whole bytes that fit: the free bits rounded down, plus the
            // byte that straddles into the active bits' vacated space.
            let bits = (free & !7) + 8;
            self.value |= (next >> (WINDOW_BITS - bits)) << (free & 7);
            self.buffered += bits;
            self.position += (bits >> 3) as usize;
            return self;
        }
        while free >= 0 {
            match self.data.get(self.position) {
                Some(&byte) => {
                    self.value |= u64::from(byte) << free;
                    self.position += 1;
                }
                None => self.padding += 8,
            }
            self.buffered += 8;
            free -= 8;
        }
        self
    }

    /// Decode a bit with an eight-bit probability of zero.
    #[inline]
    pub fn read(&mut self, probability: u8) -> bool {
        if self.buffered < 0 {
            *self = self.clone().refilled();
        }
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
        let big_split = u64::from(split) << (WINDOW_BITS - ACTIVE_BITS);
        let bit = self.value >= big_split;
        let (range, value) = if bit {
            (self.range - split, self.value - big_split)
        } else {
            (split, self.value)
        };
        // `range` is 1..=255: shift its top bit to bit 7.
        let shift = range.leading_zeros() - (u32::BITS - ACTIVE_BITS as u32);
        self.range = range << shift;
        self.value = value << shift;
        self.buffered -= shift as i32;
        bit
    }

    /// Decode an unsigned `bits`-bit literal, most significant bit first.
    #[inline]
    pub fn read_literal(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |value, _| (value << 1) | u32::from(self.read(128)))
    }

    /// Whether any read needed a bit beyond the partition.
    #[must_use]
    pub fn overran(&self) -> bool {
        // The window holds `buffered + ACTIVE_BITS` unconsumed bits; the
        // active ones have been read. A read went past the data when more
        // padding bits entered than remain unread below the active bits.
        self.padding > 0 && self.padding > self.buffered
    }

    /// End reading the partition.
    /// # Errors
    /// `Truncated` if a read needed bits beyond the partition.
    pub fn finish(&self) -> Result<()> {
        if self.overran() {
            Err(DecodeError::Truncated)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diverse_probabilities_and_carry_runs() {
        let mut writer = BoolWriter::new();
        let mut symbols = Vec::new();
        let mut state = 0x9173_5aefu32;
        for i in 0..20000u32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let probability = i as u8;
            let bit = state & 1 != 0;
            writer.write(bit, probability);
            symbols.push((bit, probability));
        }
        let encoded = writer.finish();
        // Independent pinned libvpx writer, generated by gen_vp9_bool_vector.py.
        assert_eq!(
            encoded,
            include_bytes!("../tests/data/libvpx-bool.bin").as_slice()
        );
        let mut reader = BoolReader::new(&encoded).unwrap();
        for (bit, probability) in symbols {
            assert_eq!(reader.read(probability), bit);
        }
        assert!(reader.finish().is_ok());
    }

    #[test]
    fn rejects_bad_marker_and_exhaustion() {
        assert!(BoolReader::new(&[]).is_err());
        assert!(BoolReader::new(&[0xff, 0xff]).is_err());
        let mut reader = BoolReader::new(&[0, 0]).unwrap();
        for _ in 0..8 {
            reader.read(128);
        }
        assert!(reader.finish().is_ok(), "eight bits of the second byte");
        for _ in 0..24 {
            reader.read(128);
        }
        assert!(reader.finish().is_err());
    }
}
