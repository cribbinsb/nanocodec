//! The CABAC arithmetic decoding engine shared by the H.264 and HEVC
//! decoders (H.264 §9.3.3.2, H.265 §9.3.4.3: the same engine, the same
//! 64-state probability model and tables), with an entry point for decoders
//! that compute the LPS range themselves (H.266's two-rate model).
//!
//! `ivlOffset` (`codIOffset`) lives in the top [`OFFSET_BITS`] bits of a
//! 64-bit window whose lower bits are the next bits of the data, so a
//! decision compares and subtracts against the range shifted up to the same
//! place and renormalisation is one shift of the window: bytes are loaded
//! eight at a time, only when the lookahead runs out. A decision has no
//! data-dependent branch (the outcome selects the new range and window by
//! mask; one table gives the next context for either outcome), and `n`
//! bypass bins are one division: decoding them bit by bit is the long
//! division of the offset followed by the next `n` data bits by the range
//! (`VVdeC`'s `decodeBinsEP` does the same division a bit at a time). The
//! window design is the H.264 decoder's (adfb92b1); the branch-free
//! decision follows `VVdeC`'s `BinDecoder::decodeBin` (BSD-3-Clause-Clear).
//!
//! Past the end of the data the engine reads zero bits; [`CabacReader::
//! overrun`] reports how far a malformed stream went so the caller can
//! reject it.

use crate::decode::DecodeError;

/// `ivlCurrRange` / `codIRange` after initialisation.
const INITIAL_RANGE: u32 = 510;
/// Bits of the offset (and of the range after renormalisation).
const OFFSET_BITS: u32 = 9;
/// Position of the lowest offset bit in the window.
const OFFSET_SHIFT: u32 = u64::BITS - OFFSET_BITS;
/// Bytes loaded by one refill (fewer near the end of the data).
const REFILL_BYTES: usize = 8;
/// Most bypass bins [`CabacReader::bypass_bits`] decodes at once.
pub const MAX_BYPASS_BITS: u32 = 32;
/// Zero bytes read past the end before the data counts as overrun.
const MAX_OVERRUN: usize = 4;

/// Probability states of the 64-state model (`pStateIdx` 0..=63); a
/// context byte is `pStateIdx << 1 | valMps`.
const STATES: usize = 64;
/// Context byte values; tables are indexed by the whole byte so that no
/// lookup needs a bounds check (only the first 128 values are reachable).
const CONTEXT_BYTES: usize = 256;

/// `rangeTabLps` (H.264 Table 9-44, H.265 Table 9-52) by `pStateIdx`.
#[rustfmt::skip]
const RANGE_TAB_LPS: [[u8; 4]; STATES] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [95, 116, 137, 158], [90, 110, 130, 150], [85, 104, 123, 142], [81, 99, 117, 135],
    [77, 94, 111, 128], [73, 89, 105, 122], [69, 85, 100, 116], [66, 80, 95, 110],
    [62, 76, 90, 104], [59, 72, 86, 99], [56, 69, 81, 94], [53, 65, 77, 89],
    [51, 62, 73, 85], [48, 59, 69, 80], [46, 56, 66, 76], [43, 53, 63, 72],
    [41, 50, 59, 69], [39, 48, 56, 65], [37, 45, 54, 62], [35, 43, 51, 59],
    [33, 41, 48, 56], [32, 39, 46, 53], [30, 37, 43, 50], [29, 35, 41, 48],
    [27, 33, 39, 45], [26, 31, 37, 43], [24, 30, 35, 41], [23, 28, 33, 39],
    [22, 27, 32, 37], [21, 26, 30, 35], [20, 24, 29, 33], [19, 23, 27, 31],
    [18, 22, 26, 30], [17, 21, 25, 28], [16, 20, 23, 27], [15, 19, 22, 25],
    [14, 18, 21, 24], [14, 17, 20, 23], [13, 16, 19, 22], [12, 15, 18, 21],
    [12, 14, 17, 20], [11, 14, 16, 19], [11, 13, 15, 18], [10, 12, 15, 17],
    [10, 12, 14, 16], [9, 11, 13, 15], [9, 11, 12, 14], [8, 10, 12, 14],
    [8, 9, 11, 13], [7, 9, 11, 12], [7, 9, 10, 12], [7, 8, 10, 11],
    [6, 8, 9, 11], [6, 7, 9, 10], [6, 7, 8, 9], [2, 2, 2, 2],
];

/// `transIdxLps` (H.264 Table 9-45, H.265 Table 9-53) by `pStateIdx`;
/// `transIdxMps` is `min(pStateIdx + 1, 62)` (state 63 is terminate-only).
#[rustfmt::skip]
const TRANS_IDX_LPS: [u8; STATES] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];
/// The last state an MPS can reach.
const MAX_MPS_STATE: usize = 62;

/// `rangeTabLps` by context byte and `qRangeIdx`.
const LPS_RANGE: [[u8; 4]; CONTEXT_BYTES] = {
    let mut table = [[0; 4]; CONTEXT_BYTES];
    let mut ctx = 0;
    while ctx < 2 * STATES {
        table[ctx] = RANGE_TAB_LPS[ctx >> 1];
        ctx += 1;
    }
    table
};

/// The context byte after an MPS (index 0) or an LPS (index 1), by context
/// byte. An LPS in state 0 swaps `valMps` (H.264 §9.3.3.2.1.1, H.265
/// §9.3.4.3.2.2).
const NEXT_CONTEXT: [[u8; CONTEXT_BYTES]; 2] = {
    let mut table = [[0; CONTEXT_BYTES]; 2];
    let mut ctx = 0;
    while ctx < 2 * STATES {
        let (state, mps) = (ctx >> 1, (ctx & 1) as u8);
        let next_mps = if state < MAX_MPS_STATE {
            state + 1
        } else {
            state
        };
        table[0][ctx] = ((next_mps as u8) << 1) | mps;
        let flipped = if state == 0 { 1 - mps } else { mps };
        table[1][ctx] = (TRANS_IDX_LPS[state] << 1) | flipped;
        ctx += 1;
    }
    table
};

/// The context byte `pStateIdx << 1 | valMps` of the 64-state model from a
/// `preCtxState` in `1..=126` (H.264 eq. 9-5, H.265 eq. 9-6).
#[must_use]
pub const fn context_from_pre_state(pre: i32) -> u8 {
    const MPS_SPLIT: i32 = 63;
    if pre <= MPS_SPLIT {
        ((MPS_SPLIT - pre) << 1) as u8
    } else {
        (((pre - MPS_SPLIT - 1) << 1) | 1) as u8
    }
}

/// One arithmetic-coded (sub)stream being decoded.
#[derive(Debug, Clone)]
pub struct CabacReader<'a> {
    data: &'a [u8],
    /// First byte of `data` not yet counted in `window`.
    next: usize,
    /// The offset in the top [`OFFSET_BITS`] bits, then `lookahead` valid
    /// bits of the data. Bits below those may already hold the following
    /// data bits (a refill loads whole words; reloading them is harmless).
    window: u64,
    /// Valid data bits below the offset; briefly negative after a
    /// renormalisation, until the refill that follows it.
    lookahead: i32,
    range: u32,
}

impl<'a> CabacReader<'a> {
    /// Initialise at byte `start` of `data` (H.264 §9.3.1.2, H.265
    /// §9.3.2.5): the range is 510 and the offset the next 9 bits.
    ///
    /// # Errors
    /// `Truncated` when `start` is past the data, `Malformed` when the
    /// initial offset is out of range.
    pub fn new(data: &'a [u8], start: usize) -> Result<Self, DecodeError> {
        if start >= data.len() {
            return Err(DecodeError::Truncated);
        }
        let mut reader = Self {
            data,
            next: start,
            window: 0,
            lookahead: -(OFFSET_BITS as i32),
            range: INITIAL_RANGE,
        };
        reader.refill();
        if reader.window >> OFFSET_SHIFT >= u64::from(INITIAL_RANGE) {
            return Err(DecodeError::Malformed("CABAC initial offset"));
        }
        Ok(reader)
    }

    /// The data being decoded.
    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The byte after the bits consumed so far, rounded up to a whole byte:
    /// where data that follows a terminating bin starts (PCM samples, the
    /// next substream). The offset's 9 bits count as consumed, its
    /// lookahead does not.
    #[must_use]
    pub fn next_byte(&self) -> usize {
        (self.next * 8 - self.lookahead as usize).div_ceil(8)
    }

    /// Whether decoding has consumed well past the end of the data.
    #[must_use]
    pub fn overrun(&self) -> bool {
        self.next_byte() > self.data.len() + MAX_OVERRUN
    }

    /// The current range (`ivlCurrRange`), for models that derive the LPS
    /// range themselves.
    #[inline]
    #[must_use]
    pub fn range(&self) -> u32 {
        self.range
    }

    /// Load the next whole bytes below the valid bits (`lookahead` must be
    /// at least `-OFFSET_BITS`).
    #[inline]
    fn refill(&mut self) {
        let mut word = [0u8; REFILL_BYTES];
        if let Some(bytes) = self.data.get(self.next..self.next + REFILL_BYTES) {
            word.copy_from_slice(bytes);
        } else {
            let rest = self.data.get(self.next..).unwrap_or(&[]);
            word[..rest.len()].copy_from_slice(rest);
        }
        // The first new bit goes just below the valid ones.
        let shift = (OFFSET_BITS as i32 + self.lookahead) as u32;
        self.window |= u64::from_be_bytes(word) >> shift;
        let bytes = (OFFSET_SHIFT as i32 - self.lookahead) as usize / 8;
        self.next += bytes;
        self.lookahead += 8 * bytes as i32;
    }

    /// The decision with LPS range `lps` (`rangeTabLps` or the caller's own
    /// model): returns whether the LPS was decoded, with renormalisation.
    #[inline]
    pub fn decide_lps(&mut self, lps: u32) -> bool {
        let mps_range = self.range - lps;
        let scaled = u64::from(mps_range) << OFFSET_SHIFT;
        let is_lps = self.window >= scaled;
        // Selected by mask and conditional move: the outcome is a coin toss
        // for the branch predictor.
        self.window -= scaled & 0u64.wrapping_sub(u64::from(is_lps));
        let range = if is_lps { lps } else { mps_range };
        let shift = range.leading_zeros() - (u32::BITS - OFFSET_BITS);
        self.range = range << shift;
        self.window <<= shift;
        self.lookahead -= shift as i32;
        if self.lookahead < 0 {
            self.refill();
        }
        is_lps
    }

    /// `DecodeDecision` with a 64-state context byte (H.264 §9.3.3.2.1,
    /// H.265 §9.3.4.3.2), updating the context.
    #[inline]
    pub fn decision(&mut self, ctx: &mut u8) -> bool {
        let state = *ctx;
        let lps = u32::from(LPS_RANGE[usize::from(state)][((self.range >> 6) & 3) as usize]);
        let is_lps = self.decide_lps(lps);
        *ctx = NEXT_CONTEXT[usize::from(is_lps)][usize::from(state)];
        (state & 1 != 0) != is_lps
    }

    /// `DecodeBypass` (H.264 §9.3.3.2.3, H.265 §9.3.4.3.4): the offset
    /// doubled with the next bit is compared with the range, i.e. the
    /// window against the range one bit lower.
    #[inline]
    pub fn bypass(&mut self) -> bool {
        if self.lookahead <= 0 {
            self.refill();
        }
        let scaled = u64::from(self.range) << (OFFSET_SHIFT - 1);
        let bin = self.window >= scaled;
        self.window -= scaled & 0u64.wrapping_sub(u64::from(bin));
        self.window <<= 1;
        self.lookahead -= 1;
        bin
    }

    /// `count` (at most [`MAX_BYPASS_BITS`]) bypass bins, most significant
    /// first, as one division: the offset followed by the next `count`
    /// data bits, divided by the range, gives the bins as quotient and the
    /// new offset as remainder (the quotient fits `count` bits because the
    /// offset is below the range).
    #[inline]
    pub fn bypass_bits(&mut self, count: u32) -> u32 {
        debug_assert!(count <= MAX_BYPASS_BITS);
        if self.lookahead < count as i32 {
            self.refill();
        }
        let consumed = OFFSET_BITS + count;
        // `consumed` is at most 41, so both shifts stay below 64.
        let dividend = self.window >> (u64::BITS - consumed);
        let range = u64::from(self.range);
        let bins = dividend / range;
        let rest = dividend - bins * range;
        self.window = (rest << OFFSET_SHIFT) | ((self.window << consumed) >> OFFSET_BITS);
        self.lookahead -= count as i32;
        bins as u32
    }

    /// A unary run of 1-valued bypass bins ended by a 0, of at most `max`
    /// ones (the 0 is then not read): the prefixes of Rice / Exp-Golomb
    /// binarisations.
    #[inline]
    pub fn bypass_unary(&mut self, max: u32) -> u32 {
        let mut ones = 0;
        while ones < max && self.bypass() {
            ones += 1;
        }
        ones
    }

    /// `DecodeTerminate` (H.264 §9.3.3.2.2.3, H.265 §9.3.4.3.5).
    pub fn terminate(&mut self) -> bool {
        self.range -= 2;
        if self.window >= u64::from(self.range) << OFFSET_SHIFT {
            return true;
        }
        let shift = self.range.leading_zeros() - (u32::BITS - OFFSET_BITS);
        self.range <<= shift;
        self.window <<= shift;
        self.lookahead -= shift as i32;
        if self.lookahead < 0 {
            self.refill();
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bit-serial model of the engine (the specification's procedures,
    /// one bit per renormalisation step).
    struct Model<'a> {
        data: &'a [u8],
        bit: usize,
        range: u32,
        offset: u32,
    }

    impl Model<'_> {
        fn read(&mut self) -> u32 {
            let byte = self.data.get(self.bit / 8).copied().unwrap_or(0);
            let b = (byte >> (7 - self.bit % 8)) & 1;
            self.bit += 1;
            u32::from(b)
        }
        fn renorm(&mut self) {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.read();
            }
        }
        fn decision(&mut self, ctx: &mut u8) -> bool {
            let lps =
                u32::from(RANGE_TAB_LPS[usize::from(*ctx >> 1)][((self.range >> 6) & 3) as usize]);
            self.range -= lps;
            let mps = *ctx & 1 != 0;
            let is_lps = self.offset >= self.range;
            if is_lps {
                self.offset -= self.range;
                self.range = lps;
            }
            *ctx = NEXT_CONTEXT[usize::from(is_lps)][usize::from(*ctx)];
            self.renorm();
            mps != is_lps
        }
        fn bypass(&mut self) -> bool {
            self.offset = (self.offset << 1) | self.read();
            let bin = self.offset >= self.range;
            if bin {
                self.offset -= self.range;
            }
            bin
        }
    }

    /// Pseudo-random bytes (xorshift), deterministic.
    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect()
    }

    #[test]
    fn every_operation_matches_the_bit_serial_engine() {
        for seed in 1..40u64 {
            let mut data = noise(600, seed);
            data[0] &= 0x7f; // initial offset below 510
            let mut reader = CabacReader::new(&data, 0).unwrap();
            let mut model = Model {
                data: &data,
                bit: 9,
                range: 510,
                offset: (u32::from(data[0]) << 1) | u32::from(data[1] >> 7),
            };
            let (mut c1, mut c2) = (context_from_pre_state(70), context_from_pre_state(20));
            let (mut m1, mut m2) = (c1, c2);
            let ops = noise(2000, seed * 77);
            for (i, &op) in ops.iter().enumerate() {
                match op % 5 {
                    0 => assert_eq!(
                        reader.decision(&mut c1),
                        model.decision(&mut m1),
                        "{seed}/{i}"
                    ),
                    1 => assert_eq!(
                        reader.decision(&mut c2),
                        model.decision(&mut m2),
                        "{seed}/{i}"
                    ),
                    2 => assert_eq!(reader.bypass(), model.bypass(), "{seed}/{i}"),
                    3 => {
                        let n = u32::from(op / 5) % (MAX_BYPASS_BITS + 1);
                        let want = (0..n).fold(0, |v, _| (v << 1) | u32::from(model.bypass()));
                        assert_eq!(reader.bypass_bits(n), want, "{seed}/{i} n {n}");
                    }
                    _ => {
                        if reader.next_byte() + 8 < data.len() {
                            assert_eq!(reader.decision(&mut c1), model.decision(&mut m1));
                        }
                    }
                }
                assert_eq!(reader.range(), model.range);
                assert_eq!(reader.next_byte(), model.bit.div_ceil(8), "{seed}/{i}");
            }
        }
    }

    #[test]
    fn terminate_reports_the_end() {
        // 0xff 0xff: the offset (511 >> ...) is at the top of the range.
        let data = [0xfe, 0x00, 0x00];
        let mut reader = CabacReader::new(&data, 0).unwrap();
        assert!(reader.terminate());
        assert_eq!(reader.next_byte(), 2);
    }

    #[test]
    fn the_next_context_table_matches_the_specification() {
        // MPS climbs to 62 and stays; LPS in state 0 flips valMps.
        assert_eq!(NEXT_CONTEXT[0][62 << 1], 62 << 1);
        assert_eq!(NEXT_CONTEXT[1][0], 1);
        assert_eq!(NEXT_CONTEXT[1][1], 0);
        assert_eq!(NEXT_CONTEXT[1][(10 << 1) | 1], (TRANS_IDX_LPS[10] << 1) | 1);
    }
}
