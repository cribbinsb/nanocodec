//! Bounded header reading and NAL framing shared by the subset decoders.

use std::fmt;

/// A decoder rejects incomplete, invalid or unsupported syntax without panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The packet ended before a required field was available.
    Truncated,
    /// A syntax constraint was violated.
    Malformed(&'static str),
    /// Valid syntax outside nanocodec's supported subset.
    Unsupported(&'static str),
    /// A detailed allocation or configuration error.
    InvalidData(String),
    /// The picture predicts from a reference picture the decoder does not
    /// hold: an earlier picture was lost (H.264 §8.2.5.2 "non-existing"
    /// frames, HEVC §8.3.2 "no reference picture"). The stream itself may
    /// be intact; the picture is not decoded and the caller keeps showing
    /// the last good one until a picture it can decode arrives.
    MissingReference,
    /// Nothing is decodable yet: the key frame that starts the stream (and
    /// the parameter sets it carries) was lost. Decoding starts at the next
    /// key frame.
    MissingKey,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated bitstream"),
            Self::Malformed(message) => write!(f, "malformed bitstream: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported bitstream: {message}"),
            Self::InvalidData(message) => f.write_str(message),
            Self::MissingReference => {
                f.write_str("reference picture missing (an earlier picture was lost)")
            }
            Self::MissingKey => f.write_str("no key frame yet (the first key frame was lost)"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<crate::Error> for DecodeError {
    fn from(error: crate::Error) -> Self {
        Self::InvalidData(error.to_string())
    }
}

/// Result type for bounded bitstream parsing.
pub type Result<T> = std::result::Result<T, DecodeError>;

/// An MSB-first header reader. Entropy decoders use codec-specific readers.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    /// Read a packet from its first bit.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    /// Current position in bits from the start of the packet.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.bit
    }

    /// Unconsumed bits.
    #[must_use]
    pub fn remaining_bits(&self) -> usize {
        self.data.len().saturating_mul(8).saturating_sub(self.bit)
    }

    /// Whether the next field starts at a byte boundary.
    #[must_use]
    pub const fn is_byte_aligned(&self) -> bool {
        self.bit.is_multiple_of(8)
    }

    /// Read a single bit.
    /// # Errors
    /// Returns `Truncated` when no bit remains.
    pub fn read_bit(&mut self) -> Result<bool> {
        let byte = self.data.get(self.bit / 8).ok_or(DecodeError::Truncated)?;
        let value = byte & (1 << (7 - self.bit % 8)) != 0;
        self.bit += 1;
        Ok(value)
    }

    /// Read up to 32 bits, most significant first.
    /// # Errors
    /// Rejects oversized fields and truncated input before consuming any bits.
    pub fn read_bits(&mut self, count: u32) -> Result<u32> {
        if count > 32 {
            return Err(DecodeError::Malformed("fixed field exceeds 32 bits"));
        }
        if count as usize > self.remaining_bits() {
            return Err(DecodeError::Truncated);
        }
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// Read unsigned exponential Golomb syntax (H.264 §9.1 / HEVC §9.2).
    /// # Errors
    /// Rejects truncation and values outside `u32`.
    pub fn read_ue(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while !self.read_bit()? {
            zeros += 1;
            if zeros > 32 {
                return Err(DecodeError::Malformed("exp-Golomb overflow"));
            }
        }
        let suffix = u64::from(self.read_bits(zeros)?);
        u32::try_from(((1_u64 << zeros) - 1) + suffix)
            .map_err(|_| DecodeError::Malformed("exp-Golomb overflow"))
    }

    /// Read `ue(v)` syntax whose semantics bound it to `0..=max`, so the
    /// arithmetic that follows (`+ 1`, sizes, shifts) cannot overflow.
    /// # Errors
    /// Rejects truncation and values above `max`, reported as `what`.
    pub fn read_ue_max(&mut self, max: u32, what: &'static str) -> Result<u32> {
        let value = self.read_ue()?;
        if value > max {
            return Err(DecodeError::Malformed(what));
        }
        Ok(value)
    }

    /// Read signed exponential Golomb syntax.
    /// # Errors
    /// Rejects truncation and values outside `i32`.
    pub fn read_se(&mut self) -> Result<i32> {
        let code = self.read_ue()?;
        let magnitude = i64::from(code.div_ceil(2));
        let value = if code.is_multiple_of(2) {
            -magnitude
        } else {
            magnitude
        };
        i32::try_from(value).map_err(|_| DecodeError::Malformed("signed exp-Golomb overflow"))
    }

    /// Read AV1 `uvlc()` (§4.10.3), whose 32-zero sentinel differs from `ue(v)`.
    /// # Errors
    /// Returns `Truncated` when a field is incomplete.
    pub fn read_uvlc(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while !self.read_bit()? {
            zeros += 1;
            if zeros == 32 {
                return Ok(u32::MAX);
            }
        }
        Ok(((1_u32 << zeros) - 1) + self.read_bits(zeros)?)
    }

    /// Read byte-aligned AV1 `leb128()` (§4.10.5), at most eight bytes.
    /// # Errors
    /// Rejects misalignment, truncation and an unterminated eighth byte.
    pub fn read_leb128(&mut self) -> Result<u64> {
        if !self.is_byte_aligned() {
            return Err(DecodeError::Malformed("unaligned LEB128"));
        }
        let mut value = 0;
        for index in 0..8 {
            let byte = self.read_bits(8)?;
            value |= u64::from(byte & 127) << (7 * index);
            if byte & 128 == 0 {
                return Ok(value);
            }
        }
        Err(DecodeError::Malformed("unterminated LEB128"))
    }

    /// Skip a bounded field.
    /// # Errors
    /// Returns `Truncated` without advancing if the field is too long.
    pub fn skip_bits(&mut self, count: usize) -> Result<()> {
        if count > self.remaining_bits() {
            return Err(DecodeError::Truncated);
        }
        self.bit += count;
        Ok(())
    }

    /// Consume zero alignment bits (AV1 `byte_alignment()`).
    /// # Errors
    /// Rejects a nonzero alignment bit or truncated input.
    pub fn align_zero(&mut self) -> Result<()> {
        while !self.is_byte_aligned() {
            if self.read_bit()? {
                return Err(DecodeError::Malformed("nonzero alignment bit"));
            }
        }
        Ok(())
    }

    /// Consume a stop bit and zero padding to the byte boundary.
    /// # Errors
    /// Rejects missing stop bits, nonzero padding and truncation.
    pub fn trailing_bits(&mut self) -> Result<()> {
        if !self.read_bit()? {
            return Err(DecodeError::Malformed("missing trailing stop bit"));
        }
        self.align_zero()
    }

    /// Remaining byte-aligned packet data, without advancing.
    /// # Errors
    /// Rejects an unaligned position.
    pub fn remaining_bytes(&self) -> Result<&'a [u8]> {
        if !self.is_byte_aligned() {
            return Err(DecodeError::Malformed("unaligned byte payload"));
        }
        Ok(&self.data[self.bit / 8..])
    }
}

/// Split Annex-B start codes; returned slices include NAL headers but not prefixes.
/// # Errors
/// Rejects missing start codes, nonzero leading garbage and empty NAL units.
pub fn annex_b_units(data: &[u8]) -> Result<Vec<&[u8]>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        let length = if data[i..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if data[i..].starts_with(&[0, 0, 1]) {
            3
        } else {
            0
        };
        if length == 0 {
            i += 1;
        } else {
            starts.push((i, i + length));
            i += length;
        }
    }
    let first = starts
        .first()
        .ok_or(DecodeError::Malformed("missing Annex-B start code"))?;
    if data[..first.0].iter().any(|&b| b != 0) {
        return Err(DecodeError::Malformed("garbage before Annex-B start code"));
    }
    let mut units = Vec::with_capacity(starts.len());
    for (index, &(_, begin)) in starts.iter().enumerate() {
        let end = starts
            .get(index + 1)
            .map_or(data.len(), |&(start, _)| start);
        if begin == end {
            return Err(DecodeError::Malformed("empty NAL unit"));
        }
        units.push(&data[begin..end]);
    }
    Ok(units)
}

/// Remove NAL emulation-prevention bytes from payload bytes, excluding NAL headers.
/// # Errors
/// Rejects a dangling prevention byte or an invalid following byte.
pub fn unescape_rbsp(data: &[u8]) -> Result<Vec<u8>> {
    let mut result = Vec::with_capacity(data.len());
    // Copy the runs between `00 00 03` sequences whole. Counting zeros
    // restarts after each removed 03, so the next search starts there.
    let mut start = 0;
    while let Some(found) = data[start..]
        .windows(3)
        .position(|w| w == EMULATION_PREVENTION)
    {
        let three = start + found + 2;
        if data.get(three + 1).is_none_or(|&next| next > 3) {
            return Err(DecodeError::Malformed(
                "invalid emulation-prevention sequence",
            ));
        }
        result.extend_from_slice(&data[start..three]);
        start = three + 1;
    }
    result.extend_from_slice(&data[start..]);
    Ok(result)
}

/// `00 00 03`: two zero bytes and an `emulation_prevention_three_byte`.
const EMULATION_PREVENTION: [u8; 3] = [0, 0, 3];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitWriter;

    #[test]
    fn golomb_roundtrip_and_bounds() {
        let values = [0, 1, 2, 31, 255, 65535, u32::MAX];
        let mut writer = BitWriter::new();
        for value in values {
            writer.put_ue(value);
        }
        writer.trailing_bits();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        for value in values {
            assert_eq!(reader.read_ue().unwrap(), value);
        }
        reader.trailing_bits().unwrap();
        assert_eq!(reader.remaining_bits(), 0);
        assert_eq!(reader.read_bit(), Err(DecodeError::Truncated));
        assert!(BitReader::new(&[0; 9]).read_ue().is_err());
        assert_eq!(
            BitReader::new(&[]).read_bits(33),
            Err(DecodeError::Malformed("fixed field exceeds 32 bits"))
        );
    }

    #[test]
    fn signed_alignment_and_leb128() {
        let mut writer = BitWriter::new();
        for value in [-511, -1, 0, 1, 511] {
            writer.put_se(value);
        }
        writer.trailing_bits();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        for value in [-511, -1, 0, 1, 511] {
            assert_eq!(reader.read_se().unwrap(), value);
        }
        reader.trailing_bits().unwrap();
        assert_eq!(BitReader::new(&[0x81, 0x02]).read_leb128().unwrap(), 257);
        assert!(BitReader::new(&[0x80; 8]).read_leb128().is_err());
        assert_eq!(BitReader::new(&[0; 4]).read_uvlc().unwrap(), u32::MAX);
    }

    #[test]
    fn nal_framing_and_escape_errors() {
        let bytes = [0, 0, 0, 1, 0x67, 0x80, 0, 0, 1, 0x68, 0x80];
        assert_eq!(
            annex_b_units(&bytes).unwrap(),
            vec![&bytes[4..6], &bytes[9..]]
        );
        assert!(annex_b_units(&[0, 0, 1]).is_err());
        assert!(annex_b_units(&[9, 0, 0, 1, 7]).is_err());
        assert_eq!(
            unescape_rbsp(&[0, 0, 3, 0, 0, 3, 3]).unwrap(),
            [0, 0, 0, 0, 3]
        );
        assert!(unescape_rbsp(&[0, 0, 3]).is_err());
        assert!(unescape_rbsp(&[0, 0, 3, 4]).is_err());
    }
}
