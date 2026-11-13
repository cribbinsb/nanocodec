//! Table-driven decoding of the CAVLC prefix codes (§9.2), built once per
//! decoder from `(code, length, symbol)` lists.
//!
//! Every CAVLC code is a run of leading zeros, a one, and at most a few
//! more bits (or, for a few codes, zeros only). One lookup therefore
//! decodes any code without a data-dependent branch: the table is indexed
//! by the leading-zero count of the next 32 bits (capped at the longest
//! run of zeros that starts a code) and the bits after the first one. This
//! is the leading-zero-count form of openh264's (BSD-2-Clause, Copyright
//! (c) 2013, Cisco Systems) table-driven CAVLC decoding, which looks codes
//! up by their first eight bits and then by the bits that follow.

/// An entry: `symbol << LENGTH_BITS | code length`.
type Entry = u16;
const LENGTH_BITS: u32 = 5;
const LENGTH_MASK: u16 = (1 << LENGTH_BITS) - 1;
/// No code starts with these bits.
const INVALID: Entry = u16::MAX;
/// Longest code of the CAVLC tables.
const MAX_CODE_BITS: u32 = 16;

/// A prefix code decoder.
#[derive(Debug, Clone)]
pub(super) struct VlcTable {
    entries: Vec<Entry>,
    /// Clamp of the leading-zero count: at the clamp only the zeros are
    /// known, below it a one follows them.
    max_zeros: u32,
    /// Bits after the first one that index the table.
    suffix_bits: u32,
}

impl VlcTable {
    /// Build the decoder of `codes`: `(code value, code length, symbol)`,
    /// lengths 1..=16 (entries of length 0 are ignored). An empty list
    /// gives the zero-length code: every lookup decodes symbol 0 from no
    /// bits (`run_before` once no zeros are left).
    ///
    /// # Panics
    ///
    /// If two codes overlap or a code is too long: the tables are
    /// constants, so this is a bug.
    pub fn new(codes: impl IntoIterator<Item = (u32, u8, u16)>) -> Self {
        let codes: Vec<(u32, u32, u16)> = codes
            .into_iter()
            .filter(|&(_, length, _)| length != 0)
            .map(|(code, length, symbol)| (code, u32::from(length), symbol))
            .collect();
        assert!(
            codes
                .iter()
                .all(|&(_, length, symbol)| length <= MAX_CODE_BITS
                    && u32::from(symbol) < 1 << (u16::BITS - LENGTH_BITS)),
            "CAVLC code out of range"
        );
        if codes.is_empty() {
            return Self {
                entries: vec![0],
                max_zeros: 0,
                suffix_bits: 0,
            };
        }
        // A code with `z` leading zeros is resolved by the clamped count
        // and the `length - z - 1` bits after its first one.
        let zeros = |code: u32, length: u32| length - (u32::BITS - code.leading_zeros());
        // Counts below the clamp have a known one after the zeros, so the
        // clamp is one past the longest zero run of a code with a one (or
        // the length of an all-zero code).
        let max_zeros = codes
            .iter()
            .map(|&(c, l, _)| if c == 0 { l } else { zeros(c, l) + 1 })
            .max()
            .unwrap_or(0);
        let suffix_bits = codes
            .iter()
            .map(|&(c, l, _)| l.saturating_sub(zeros(c, l) + 1))
            .max()
            .unwrap_or(0);
        let mut entries = vec![INVALID; ((max_zeros + 1) << suffix_bits) as usize];
        for (index, entry) in entries.iter_mut().enumerate() {
            let z = index as u32 >> suffix_bits;
            let suffix = index as u32 & ((1 << suffix_bits) - 1);
            *entry = Self::resolve(&codes, z, suffix, max_zeros, suffix_bits);
        }
        Self {
            entries,
            max_zeros,
            suffix_bits,
        }
    }

    /// The entry of the bit pattern that index (`z`, `suffix`) stands for.
    fn resolve(
        codes: &[(u32, u32, u16)],
        z: u32,
        suffix: u32,
        max_zeros: u32,
        suffix_bits: u32,
    ) -> Entry {
        // The known bits, MSB-aligned in a u64: `z` zeros, then (unless the
        // count was clamped) a one and the suffix.
        let (pattern, known) = if z == max_zeros {
            (0u64, z)
        } else {
            let tail = (1u64 << suffix_bits) | u64::from(suffix);
            (tail << (64 - (z + 1 + suffix_bits)), z + 1 + suffix_bits)
        };
        let mut found = INVALID;
        for &(code, length, symbol) in codes {
            let compared = length.min(known);
            let code_bits = (u64::from(code) << (64 - length)) >> (64 - compared);
            if code_bits != pattern >> (64 - compared) {
                continue;
            }
            assert!(
                length <= known,
                "CAVLC code not resolved by its table index"
            );
            assert_eq!(found, INVALID, "overlapping CAVLC codes");
            found = (symbol << LENGTH_BITS) | length as u16;
        }
        found
    }

    /// Decode one symbol from `bits`, the next 32 bits of the stream:
    /// `(symbol, code length)`, or `None` if no code matches.
    #[inline]
    pub fn lookup(&self, bits: u32) -> Option<(u16, u32)> {
        let z = bits.leading_zeros().min(self.max_zeros);
        // The suffix bits after the first one; `>> 1 >> (63 - n)` is a
        // shift by `64 - n` that stays defined for `n == 0`.
        let after_one = (u64::from(bits) << 32) << z << 1;
        let suffix = (after_one >> 1 >> (63 - self.suffix_bits)) as u32;
        let index = ((z << self.suffix_bits) | suffix) as usize;
        let entry = self.entries[index];
        (entry != INVALID).then_some((entry >> LENGTH_BITS, u32::from(entry & LENGTH_MASK)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The next 32 bits of `data` from bit `pos`.
    fn peek(data: &[u8], pos: u32) -> u32 {
        let mut word = [0u8; 8];
        word[..data.len()].copy_from_slice(data);
        (u64::from_be_bytes(word) << pos >> 32) as u32
    }

    #[test]
    fn short_and_long_codes() {
        // "1" -> 0, "01" -> 1, "0000 0000 0000 001" (15 bits) -> 2.
        let table = VlcTable::new([(1, 1, 0), (1, 2, 1), (1, 15, 2)]);
        let data = [0b1010_0000, 0b0000_0000, 0b0100_0000];
        assert_eq!(table.lookup(peek(&data, 0)), Some((0, 1)));
        assert_eq!(table.lookup(peek(&data, 1)), Some((1, 2)));
        assert_eq!(table.lookup(peek(&data, 3)), Some((2, 15)));
        assert_eq!(table.lookup(0), None);
    }

    #[test]
    fn all_zero_codes_and_empty_table() {
        // "0" -> 0, "1" -> 1 (total_zeros with TotalCoeff 15).
        let table = VlcTable::new([(0, 1, 0), (1, 1, 1)]);
        assert_eq!(table.lookup(0), Some((0, 1)));
        assert_eq!(table.lookup(0x4000_0000), Some((0, 1)));
        assert_eq!(table.lookup(0x8000_0000), Some((1, 1)));
        // "11" -> 0, "10" -> 1, "01" -> 2, "001" -> 3, "000" -> 4.
        let table = VlcTable::new([(3, 2, 0), (2, 2, 1), (1, 2, 2), (1, 3, 3), (0, 3, 4)]);
        assert_eq!(table.lookup(0xC000_0000), Some((0, 2)));
        assert_eq!(table.lookup(0x8000_0000), Some((1, 2)));
        assert_eq!(table.lookup(0x4000_0000), Some((2, 2)));
        assert_eq!(table.lookup(0x2000_0000), Some((3, 3)));
        assert_eq!(table.lookup(0x1FFF_FFFF), Some((4, 3)));
        assert_eq!(VlcTable::new([]).lookup(0xFFFF_FFFF), Some((0, 0)));
    }
}
