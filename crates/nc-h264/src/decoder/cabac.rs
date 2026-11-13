//! The H.264 CABAC engine of one slice (§9.3.1, §9.3.3.2): the shared
//! arithmetic decoder ([`nc_common::cabac::CabacReader`], the same engine
//! as HEVC's) with the slice's context variables, initialised from Tables
//! 9-12..9-33 (§9.3.1.1).
//!
//! Past the end of the slice data the engine reads zero bits: the
//! macroblock count and end-of-slice checks catch truncated slices.

use super::slice::SliceType;
use super::{DecodeError, Result};
use crate::tables::{CABAC_INIT_MN, CONTEXT_COUNT};
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
    contexts: [u8; CONTEXT_COUNT],
}

impl<'a> Cabac<'a> {
    /// Initialise contexts for `slice_type` / `cabac_init_idc` at `qp`
    /// (§9.3.1.1) and the engine on `data` (the byte-aligned slice data).
    pub fn new(data: &'a [u8], slice_type: SliceType, cabac_init_idc: u8, qp: i32) -> Result<Self> {
        let column = match slice_type {
            SliceType::I => 0,
            SliceType::P => 1 + usize::from(cabac_init_idc),
        };
        let mut contexts = [0u8; CONTEXT_COUNT];
        for (ctx, mn) in contexts.iter_mut().zip(CABAC_INIT_MN.iter()) {
            let (m, n) = mn[column];
            let pre = (((i32::from(m) * qp.clamp(0, 51)) >> 4) + i32::from(n))
                .clamp(MIN_PRE_STATE, MAX_PRE_STATE);
            *ctx = context_from_pre_state(pre);
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
    pub fn pcm_samples(&mut self, out: &mut [u8]) -> Result<()> {
        let data = self.reader.data();
        let start = self.reader.next_byte();
        let samples = data
            .get(start..start + out.len())
            .ok_or(DecodeError::Truncated)?;
        out.copy_from_slice(samples);
        self.reader = CabacReader::new(data, start + out.len())?;
        Ok(())
    }
}
