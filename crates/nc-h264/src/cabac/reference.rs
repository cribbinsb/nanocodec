//! The literal §9.3.4 encoding engine (10-bit `codILow`, bit-serial
//! `PutBit`, `bitsOutstanding`): the reference the byte-oriented engine in
//! the parent module is fuzzed against.

use nc_common::BitWriter;

use super::{Context, HALF, QUARTER, RENORM_THRESHOLD, SliceKind};
use crate::tables::{CABAC_INIT_MN, CONTEXT_COUNT, RANGE_TAB_LPS, TRANS_IDX};

/// The spec engine for one slice.
#[derive(Debug)]
pub struct SpecEncoder {
    contexts: [Context; CONTEXT_COUNT],
    low: u32,
    range: u32,
    first_bit: bool,
    bits_outstanding: u32,
    out: BitWriter,
}

impl SpecEncoder {
    /// Start a slice: `out` holds the slice header and must be byte aligned
    /// (after `cabac_alignment_one_bit`s, H.264 §7.3.4). Contexts are
    /// initialised for a `kind` slice at `slice_qp`.
    pub fn new(out: BitWriter, kind: SliceKind, slice_qp: u8) -> Self {
        assert!(
            out.is_byte_aligned(),
            "CABAC slice data must start byte aligned"
        );
        let mut contexts = [Context::default(); CONTEXT_COUNT];
        for (ctx, mn) in contexts.iter_mut().zip(CABAC_INIT_MN.iter()) {
            let (m, n) = mn[kind.init_column(0)];
            *ctx = Context::init(m, n, slice_qp);
        }
        // §9.3.1.2: initialisation of the arithmetic encoding engine.
        Self {
            contexts,
            low: 0,
            range: 510,
            first_bit: true,
            bits_outstanding: 0,
            out,
        }
    }

    /// `EncodeDecision` (§9.3.4.2) of `bin` with context `ctx_idx`.
    pub fn encode_decision(&mut self, ctx_idx: usize, bin: bool) {
        let ctx = &mut self.contexts[ctx_idx];
        let (p_state_idx, val_mps) = (ctx.p_state_idx(), ctx.val_mps());
        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let range_lps = u32::from(RANGE_TAB_LPS[usize::from(p_state_idx)][q_range_idx]);
        self.range -= range_lps;
        let (trans_lps, trans_mps) = TRANS_IDX[usize::from(p_state_idx)];
        *ctx = if bin == val_mps {
            Context::new(trans_mps, val_mps)
        } else {
            self.low += self.range;
            self.range = range_lps;
            Context::new(trans_lps, if p_state_idx == 0 { !val_mps } else { val_mps })
        };
        self.renormalize();
    }

    /// `EncodeBypass` (§9.3.4.4) of `bin`.
    pub fn encode_bypass(&mut self, bin: bool) {
        self.low <<= 1;
        if bin {
            self.low += self.range;
        }
        if self.low >= 2 * HALF {
            self.put_bit(true);
            self.low -= 2 * HALF;
        } else if self.low < HALF {
            self.put_bit(false);
        } else {
            self.low -= HALF;
            self.bits_outstanding += 1;
        }
    }

    /// `EncodeTerminate` (§9.3.4.5) of `bin` (ctxIdx 276). A `true` bin ends
    /// the slice and flushes the engine; the flush writes the
    /// `rbsp_stop_one_bit` itself.
    pub fn encode_terminate(&mut self, bin: bool) {
        self.range -= 2;
        if bin {
            self.low += self.range;
            self.flush();
        } else {
            self.renormalize();
        }
    }

    /// End the slice data: byte-align with zero bits (the stop bit was
    /// written by the final `encode_terminate(true)`) and return the RBSP.
    pub fn finish(mut self) -> Vec<u8> {
        self.out.byte_align_zero();
        self.out.into_bytes()
    }

    /// `EncodeFlush` (§9.3.4.5).
    fn flush(&mut self) {
        self.range = 2;
        self.renormalize();
        self.put_bit((self.low >> 9) & 1 != 0);
        // The last of these two bits is 1: the rbsp_stop_one_bit.
        self.out.put_bits(((self.low >> 7) & 3) | 1, 2);
    }

    /// `RenormE` (§9.3.4.3).
    fn renormalize(&mut self) {
        while self.range < RENORM_THRESHOLD {
            if self.low < QUARTER {
                self.put_bit(false);
            } else if self.low >= HALF {
                self.low -= HALF;
                self.put_bit(true);
            } else {
                self.low -= QUARTER;
                self.bits_outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    /// `PutBit` (§9.3.4.3): emit `bit` and any outstanding opposite bits.
    fn put_bit(&mut self, bit: bool) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.out.put_bit(bit);
        }
        while self.bits_outstanding > 0 {
            self.out.put_bit(!bit);
            self.bits_outstanding -= 1;
        }
    }
}
