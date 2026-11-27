//! The H.264 CABAC engine of one slice (§9.3.1, §9.3.3.2): the shared
//! arithmetic decoder ([`nc_common::cabac::CabacReader`], the same engine
//! as HEVC's) with the slice's context variables, initialised from Tables
//! 9-12..9-33 (§9.3.1.1).
//!
//! Context variables 460..=1023 are those of the Cb and Cr blocks of 4:4:4
//! (`ctxBlockCat` 5..13, Table 9-34): each starts from the same `(m, n)` as
//! its luma twin (Tables 9-24..9-33 repeat the luma values), so they are
//! initialised from the luma entries ([`CONTEXT_TWINS`]).
//!
//! Past the end of the slice data the engine reads zero bits: the
//! macroblock count and end-of-slice checks catch truncated slices.

use super::cavlc::read_pcm;
use super::reader::Reader;
use super::slice::SliceType;
use super::{DecodeError, Result};
use crate::tables::CABAC_INIT_MN;

/// Context variables of every chroma format (4:4:4 uses 460..=1023).
pub(super) const CONTEXT_COUNT_444: usize = 1024;

/// The Cb / Cr context ranges beyond [`crate::tables::CONTEXT_COUNT`] as `(first ctxIdx,
/// length, first ctxIdx of the luma twin)` (Table 9-34: coded block flags,
/// then frame and field significance maps, last flags and levels of
/// `ctxBlockCat` 6..8, 10..12, 9 and 13).
const CONTEXT_TWINS: [(usize, usize, usize); 25] = [
    (460, 12, 85),
    (472, 12, 85),
    (484, 44, 105),
    (528, 44, 105),
    (572, 44, 166),
    (616, 44, 166),
    (660, 15, 402),
    (675, 15, 436),
    (690, 9, 417),
    (699, 9, 451),
    (708, 10, 426),
    (718, 15, 402),
    (733, 15, 436),
    (748, 9, 417),
    (757, 9, 451),
    (766, 10, 426),
    (776, 44, 277),
    (820, 44, 277),
    (864, 44, 338),
    (908, 44, 338),
    (952, 30, 227),
    (982, 30, 227),
    // coded_block_flag of the 8x8 blocks of Y, Cb and Cr: as Luma4x4's.
    (1012, 4, 93),
    (1016, 4, 93),
    (1020, 4, 93),
];
use nc_common::cabac::{CabacReader, context_from_pre_state};

/// `preCtxState` bounds (eq. 9-5).
const MIN_PRE_STATE: i32 = 1;
const MAX_PRE_STATE: i32 = 126;
/// Largest exp-Golomb order a bypass suffix may reach before the value
/// would overflow (coefficient levels and motion vector differences).
const MAX_BYPASS_ORDER: u32 = 24;

/// The arithmetic decoder and context variables of one slice.
pub(super) struct Cabac<'a> {
    reader: CabacReader<'a>,
    /// `pStateIdx << 1 | valMPS` of every context variable.
    contexts: [u8; CONTEXT_COUNT_444],
}

impl<'a> Cabac<'a> {
    /// Initialise contexts for `slice_type` / `cabac_init_idc` at `qp`
    /// (§9.3.1.1; those of 4:4:4's Cb and Cr blocks when `chroma_444`) and
    /// the engine on `data` (the byte-aligned slice data).
    pub fn new(
        data: &'a [u8],
        (slice_type, cabac_init_idc): (SliceType, u8),
        qp: i32,
        chroma_444: bool,
    ) -> Result<Self> {
        let column = match slice_type {
            SliceType::I | SliceType::Si => 0,
            SliceType::P | SliceType::B => 1 + usize::from(cabac_init_idc),
        };
        let state = |(m, n): (i8, i8)| {
            let pre = (((i32::from(m) * qp.clamp(0, 51)) >> 4) + i32::from(n))
                .clamp(MIN_PRE_STATE, MAX_PRE_STATE);
            context_from_pre_state(pre)
        };
        let mut contexts = [0u8; CONTEXT_COUNT_444];
        for (ctx, mn) in contexts.iter_mut().zip(CABAC_INIT_MN.iter()) {
            *ctx = state(mn[column]);
        }
        if chroma_444 {
            for (first, length, twin) in CONTEXT_TWINS {
                contexts.copy_within(twin..twin + length, first);
            }
        }
        Ok(Self {
            reader: CabacReader::new(data, 0)?,
            contexts,
        })
    }

    /// `DecodeDecision` with context `ctx_idx` (§9.3.3.2.1).
    // Always inlined: it is the hottest call of CABAC slices (tens of
    // millions per 1080p second); out of line, the call and its spills cost
    // 3.6 % of all instructions on CABAC streams.
    #[allow(clippy::inline_always, reason = "per-bin work of CABAC parsing")]
    #[inline(always)]
    pub fn decision(&mut self, ctx_idx: usize) -> bool {
        self.reader.decision(&mut self.contexts[ctx_idx])
    }

    /// `DecodeBypass` (§9.3.3.2.3).
    #[inline]
    pub fn bypass(&mut self) -> bool {
        self.reader.bypass()
    }

    /// Bits consumed since the engine (re)started (analysis).
    #[cfg(feature = "analysis")]
    pub fn bits_consumed(&self) -> usize {
        self.reader.bits_consumed()
    }

    /// `DecodeTerminate` (§9.3.3.2.4): `end_of_slice_flag` and the `I_PCM`
    /// bin of `mb_type`.
    pub fn terminate(&mut self) -> bool {
        self.reader.terminate()
    }

    /// The `k`-th order exp-Golomb bypass suffix of `UEGk` binarisations
    /// (§9.3.2.3): the unary prefix bin by bin, the suffix as one run.
    pub fn exp_golomb(&mut self, mut order: u32) -> Result<u32> {
        let mut value = 0u32;
        while self.reader.bypass() {
            if order >= MAX_BYPASS_ORDER {
                return Err(DecodeError::Malformed("CABAC bypass magnitude"));
            }
            value += 1 << order;
            order += 1;
        }
        Ok(value + self.reader.bypass_bits(order))
    }

    /// The `pcm_sample_luma` / `pcm_sample_chroma` bytes after an `I_PCM`
    /// `mb_type`: they start at the byte boundary following the bits the
    /// engine consumed, and the engine restarts after them (§9.3.1.2).
    /// The `I_PCM` samples (`bits` per luma and chroma sample), which start
    /// at the byte after the bits the engine consumed; the engine restarts
    /// after them (§9.3.1.2).
    pub fn pcm_samples(&mut self, out: &mut [u16], bits: [u32; 2]) -> Result<()> {
        let data = self.reader.data();
        let start = self.reader.next_byte();
        let mut reader = Reader::new(data.get(start..).ok_or(DecodeError::Truncated)?);
        read_pcm(&mut reader, out, bits)?;
        // Whole bytes: 256 luma and 0, 128, 256 or 512 chroma samples of any
        // depth.
        let bytes = reader.position().div_ceil(8);
        self.reader = CabacReader::new(data, start + bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::CONTEXT_COUNT;

    /// The twin ranges tile 460..=1023 exactly, each twin inside the
    /// generated luma table.
    #[test]
    fn twins_cover_the_444_contexts() {
        let mut next = CONTEXT_COUNT;
        let mut ranges = CONTEXT_TWINS;
        ranges.sort_unstable();
        for (first, length, twin) in ranges {
            assert_eq!(first, next);
            assert!(twin + length <= CONTEXT_COUNT);
            next = first + length;
        }
        assert_eq!(next, CONTEXT_COUNT_444);
    }
}
