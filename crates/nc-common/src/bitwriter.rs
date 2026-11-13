//! MSB-first bit writer for bitstream headers.
//!
//! All three codecs write their headers (H.264/HEVC parameter sets and slice
//! headers, AV1 OBU headers, sequence and frame headers) as fixed- and
//! variable-length bit fields, most significant bit first. This writer
//! covers those field types:
//!
//! - fixed-width unsigned fields: `f(n)` in AV1, `u(n)` in H.264/HEVC;
//! - exp-Golomb `ue(v)` / `se(v)` (H.264 §9.1, HEVC §9.2);
//! - AV1 `uvlc()` (AV1 §4.10.3) and `leb128()` (AV1 §4.10.5);
//! - byte alignment and RBSP / AV1 trailing bits.
//!
//! Entropy-coded payloads (CABAC, the AV1 range coder) have their own
//! writers in the codec crates; their output is appended here byte-aligned.

/// Accumulates bits MSB-first into a byte vector.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits of the partially filled final byte, left-aligned in `cur`.
    cur: u8,
    /// Number of valid bits in `cur` (0..8).
    used: u32,
}

impl BitWriter {
    /// An empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of bits written so far.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.used as usize
    }

    /// True if the next bit starts a new byte.
    #[must_use]
    pub fn is_byte_aligned(&self) -> bool {
        self.used == 0
    }

    /// Write one bit.
    pub fn put_bit(&mut self, bit: bool) {
        self.cur |= u8::from(bit) << (7 - self.used);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.used = 0;
        }
    }

    /// Write the low `n` bits of `value`, most significant first (`n` ≤ 32).
    /// Whole runs of bits go into the partial byte at once: H.264 CAVLC
    /// slice data is written through here, a few hundred bits per coded
    /// macroblock.
    pub fn put_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        debug_assert!(
            n == 32 || value < (1u32 << n),
            "value {value} does not fit {n} bits"
        );
        let mut remaining = n;
        while remaining > 0 {
            let free = 8 - self.used;
            let take = free.min(remaining);
            remaining -= take;
            let chunk = (u64::from(value) >> remaining) & ((1u64 << take) - 1);
            self.cur |= (chunk as u8) << (free - take);
            self.used += take;
            if self.used == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.used = 0;
            }
        }
    }

    /// Unsigned exp-Golomb `ue(v)` (H.264 §9.1, HEVC §9.2): `len - 1` zeros,
    /// then the `len` bits of `value + 1`.
    pub fn put_ue(&mut self, value: u32) {
        let code = u64::from(value) + 1;
        let len = 64 - code.leading_zeros(); // bits in `code`
        self.put_bits(0, len - 1);
        if len <= 32 {
            self.put_bits(code as u32, len);
        } else {
            // Only `u32::MAX` itself: a leading one, then 32 zero bits.
            self.put_bit(true);
            self.put_bits(0, 32);
        }
    }

    /// The number of bits [`Self::put_ue`] writes for `value`.
    #[must_use]
    pub fn ue_len(value: u32) -> u32 {
        let code = u64::from(value) + 1;
        2 * (64 - code.leading_zeros()) - 1
    }

    /// The number of bits [`Self::put_se`] writes for `value`.
    #[must_use]
    pub fn se_len(value: i32) -> u32 {
        Self::ue_len(se_code_num(value))
    }

    /// Signed exp-Golomb `se(v)`: 0, 1, −1, 2, −2, … map to 0, 1, 2, 3, 4, ….
    pub fn put_se(&mut self, value: i32) {
        self.put_ue(se_code_num(value));
    }

    /// AV1 `uvlc()` (AV1 §4.10.3): `leadingZeros` zero bits, a one, then the
    /// value's low bits. Identical to exp-Golomb for values below 2^32 − 1.
    pub fn put_uvlc(&mut self, value: u32) {
        self.put_ue(value);
    }

    /// Pad with zero bits to the next byte boundary (AV1 `byte_alignment()`).
    pub fn byte_align_zero(&mut self) {
        while !self.is_byte_aligned() {
            self.put_bit(false);
        }
    }

    /// A one bit, then zero bits to the byte boundary: AV1 `trailing_bits()`
    /// (§5.3.4) and H.264/HEVC `rbsp_trailing_bits()`.
    pub fn trailing_bits(&mut self) {
        self.put_bit(true);
        self.byte_align_zero();
    }

    /// Append whole bytes; the writer must be byte aligned.
    pub fn put_bytes(&mut self, bytes: &[u8]) {
        assert!(self.is_byte_aligned(), "put_bytes requires byte alignment");
        self.bytes.extend_from_slice(bytes);
    }

    /// The written bytes. The writer must be byte aligned, so no bits are
    /// silently dropped.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        assert!(self.is_byte_aligned(), "into_bytes requires byte alignment");
        self.bytes
    }
}

/// The `codeNum` of signed exp-Golomb `value` (H.264 Table 9-3).
fn se_code_num(value: i32) -> u32 {
    if value > 0 {
        (value as u32) * 2 - 1
    } else {
        value.unsigned_abs() * 2
    }
}

/// Encode `value` as AV1 `leb128()` (AV1 §4.10.5): little-endian base-128,
/// seven bits per byte, high bit set on every byte but the last.
#[must_use]
pub fn leb128(mut value: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(4);
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_fields_are_msb_first() {
        let mut w = BitWriter::new();
        w.put_bits(0b101, 3);
        w.put_bits(0b11110, 5);
        assert_eq!(w.into_bytes(), vec![0b1011_1110]);
    }

    #[test]
    fn exp_golomb_matches_spec_table() {
        // ue(v) codewords: 0 -> 1, 1 -> 010, 2 -> 011, 3 -> 00100.
        let mut w = BitWriter::new();
        for v in 0..4 {
            w.put_ue(v);
        }
        w.byte_align_zero();
        // 1 010 011 00100 + 0000 padding
        assert_eq!(w.into_bytes(), vec![0b1010_0110, 0b0100_0000]);
    }

    #[test]
    fn signed_exp_golomb_mapping() {
        let mut a = BitWriter::new();
        a.put_se(-2);
        let mut b = BitWriter::new();
        b.put_ue(4);
        a.byte_align_zero();
        b.byte_align_zero();
        assert_eq!(a.into_bytes(), b.into_bytes());
    }

    #[test]
    fn multi_bit_fields_match_single_bits() {
        let fields = [
            (0x2b, 6),
            (1, 1),
            (0xdead_beef, 32),
            (0, 3),
            (0x1ffff, 17),
            (5, 3),
        ];
        let mut fast = BitWriter::new();
        let mut slow = BitWriter::new();
        for (value, n) in fields {
            fast.put_bits(value, n);
            for i in (0..n).rev() {
                slow.put_bit((value >> i) & 1 != 0);
            }
        }
        fast.byte_align_zero();
        slow.byte_align_zero();
        assert_eq!(fast.into_bytes(), slow.into_bytes());
    }

    #[test]
    fn exp_golomb_lengths() {
        for value in [0, 1, 2, 3, 6, 7, 254, 65_535, u32::MAX] {
            let mut w = BitWriter::new();
            w.put_ue(value);
            assert_eq!(w.bit_len() as u32, BitWriter::ue_len(value));
        }
        for value in [0, 1, -1, 2, -7, 300] {
            let mut w = BitWriter::new();
            w.put_se(value);
            assert_eq!(w.bit_len() as u32, BitWriter::se_len(value));
        }
    }

    #[test]
    fn trailing_bits_pad_with_one_then_zeros() {
        let mut w = BitWriter::new();
        w.put_bits(0, 3);
        w.trailing_bits();
        assert_eq!(w.into_bytes(), vec![0b0001_0000]);
    }

    #[test]
    fn leb128_examples() {
        assert_eq!(leb128(0), vec![0]);
        assert_eq!(leb128(127), vec![127]);
        assert_eq!(leb128(128), vec![0x80, 0x01]);
        assert_eq!(leb128(300), vec![0xac, 0x02]);
    }
}
