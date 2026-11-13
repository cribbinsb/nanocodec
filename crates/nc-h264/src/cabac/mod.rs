//! The CABAC arithmetic encoder and its context variables (H.264 §9.3).
//!
//! The engine is the byte-oriented form of §9.3.4 used by x264 and
//! openh264: `low` keeps the spec's 10-bit `codILow` in its low bits with
//! the not yet written code bits above it, renormalisation shifts by a
//! table-free leading-zero count instead of looping bit by bit, and whole
//! bytes are written once 8 code bits are final. A carry out of `low` is
//! added to the last written byte; runs of `0xff` bytes that a carry could
//! still change are held back (`outstanding`), which is what the spec's
//! `bitsOutstanding` does bit by bit. The output is bit-identical to the
//! literal flowcharts, which `reference.rs` keeps for the fuzz test below.
//! It is appended to a [`BitWriter`] that already holds the slice header and
//! `cabac_alignment_one_bit`s.
//!
//! Macroblocks are coded through a [`CabacWriter`], which holds the coder's
//! registers by value and borrows the contexts and the output.

use nc_common::BitWriter;

use crate::tables::{CABAC_INIT_MN, CONTEXT_COUNT, RANGE_TAB_LPS, TRANS_IDX};

mod cost;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

/// Number of `cabac_init_idc` values (§7.4.3: 0..=2), each selecting one
/// of the P-slice context initialisation tables.
pub const CABAC_INIT_IDC_COUNT: u8 = 3;

/// The slice type, as far as CABAC cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceKind {
    /// I slice.
    I,
    /// P slice.
    P,
}

impl SliceKind {
    /// Which `(m, n)` column of [`CABAC_INIT_MN`] initialises the contexts
    /// (H.264 §9.3.1.1): I slices have their own; P slices use the table
    /// `cabac_init_idc` selects (ignored for I slices, where it is absent).
    pub(crate) const fn init_column(self, cabac_init_idc: u8) -> usize {
        match self {
            Self::I => 0,
            Self::P => 1 + cabac_init_idc as usize,
        }
    }
}

pub use cost::{BinCosts, BitCounter};

/// The part of a macroblock's syntax that the following bins belong to, so
/// that a [`BitCounter`] can price the parts of a residual separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbPart {
    /// Skip flag, type, prediction modes and motion.
    Header,
    /// Bins only present when some residual is coded (`mb_qp_delta`,
    /// `transform_size_8x8_flag`).
    ResidualShared,
    /// One luma 8x8 quadrant (raster index): its `coded_block_pattern` bin
    /// and residual blocks.
    Luma8x8(u8),
    /// Chroma `coded_block_pattern` bins and residual blocks.
    Chroma,
}

impl MbPart {
    /// Number of parts.
    pub(crate) const COUNT: usize = 7;

    /// Index of the part in `0..COUNT`.
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Header => 0,
            Self::ResidualShared => 1,
            Self::Luma8x8(b8) => 2 + b8 as usize,
            Self::Chroma => 6,
        }
    }
}

/// Where the syntax writers put their bins: the arithmetic encoder, or a
/// [`BitCounter`] that prices them without coding.
pub trait BinSink {
    /// A context-coded bin (`EncodeDecision`, §9.3.4.2).
    fn encode_decision(&mut self, ctx_idx: usize, bin: bool);
    /// A bypass bin (`EncodeBypass`, §9.3.4.4).
    fn encode_bypass(&mut self, bin: bool);
    /// The terminate bin (`EncodeTerminate`, §9.3.4.5).
    fn encode_terminate(&mut self, bin: bool);
    /// The bins that follow belong to `part` (only counters care).
    #[inline]
    fn begin_part(&mut self, _part: MbPart) {}
    /// Code `run` (e.g. one block's levels) from a local copy of the
    /// sink's state where that keeps it in machine registers (see
    /// [`CabacWriter`]).
    #[inline]
    fn code_run<T: BinRun>(&mut self, run: T) -> T::Output
    where
        Self: Sized,
    {
        run.code(self)
    }
}

/// A run of bins written to whatever sink [`BinSink::code_run`] hands it.
pub trait BinRun {
    /// What the run returns.
    type Output;
    /// Write the run's bins to `sink`.
    fn code(self, sink: &mut impl BinSink) -> Self::Output;
}

/// `codIRange` below this value triggers renormalisation (§9.3.4.3).
pub(crate) const RENORM_THRESHOLD: u32 = 256;
/// Leading zeros of a renormalised `codIRange` (256..=510) as a `u32`.
const RANGE_LEADING_ZEROS: u32 = u32::BITS - 9;
/// The two halves of the 10-bit `codILow` register.
#[cfg(test)]
pub(crate) const QUARTER: u32 = 256;
#[cfg(test)]
pub(crate) const HALF: u32 = 512;

/// One context variable: probability state and most probable symbol,
/// packed as `pStateIdx · 2 + valMPS` so the coder's tables are indexed by
/// one byte (no bounds checks in the per-bin path).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Context {
    state: u8,
}

/// `rangeTabLPS` by packed state (`pStateIdx · 2 + valMPS`), padded to 256
/// rows so that any `u8` index is in bounds.
const RANGE_LPS_BY_STATE: [[u8; 4]; 256] = {
    let mut table = [[0u8; 4]; 256];
    let mut packed = 0;
    while packed < 2 * RANGE_TAB_LPS.len() {
        table[packed] = RANGE_TAB_LPS[packed / 2];
        packed += 1;
    }
    table
};

/// The packed state after coding a bin: indexed by `state · 2 + bin` (as a
/// byte: packed states are below 128), the transitions of §9.3.4.2 (MPS:
/// `transIdxMPS`; LPS: `transIdxLPS`, swapping `valMPS` at `pStateIdx` 0).
const NEXT_STATE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut packed = 0;
    while packed < 2 * TRANS_IDX.len() {
        let (p_state, mps) = (packed / 2, packed % 2);
        let (trans_lps, trans_mps) = TRANS_IDX[p_state];
        // bin == mps
        table[packed * 2 + mps] = trans_mps * 2 + mps as u8;
        // bin != mps
        let next_mps = if p_state == 0 { 1 - mps } else { mps };
        table[packed * 2 + (1 - mps)] = trans_lps * 2 + next_mps as u8;
        packed += 1;
    }
    table
};

impl Context {
    /// A context in probability state `p_state_idx` with MPS `val_mps`.
    const fn new(p_state_idx: u8, val_mps: bool) -> Self {
        Self {
            state: p_state_idx * 2 + val_mps as u8,
        }
    }

    /// `pStateIdx`.
    pub(crate) const fn p_state_idx(self) -> u8 {
        self.state / 2
    }

    /// `valMPS`.
    pub(crate) const fn val_mps(self) -> bool {
        self.state % 2 == 1
    }

    /// `rangeTabLPS` of this state at quarter `q_range_idx` (0..=3).
    #[allow(
        clippy::inline_always,
        reason = "once per bin: the hot path of the entropy coder"
    )]
    #[inline(always)]
    pub(crate) fn range_lps(self, q_range_idx: usize) -> u32 {
        u32::from(RANGE_LPS_BY_STATE[usize::from(self.state)][q_range_idx & 3])
    }

    /// This context after coding `bin` (§9.3.4.2).
    #[allow(
        clippy::inline_always,
        reason = "once per bin: the hot path of the entropy coder"
    )]
    #[inline(always)]
    pub(crate) fn next(self, bin: bool) -> Self {
        Self {
            state: NEXT_STATE[usize::from(self.state.wrapping_shl(1) | u8::from(bin))],
        }
    }

    /// Initialisation from `(m, n)` at `SliceQPY` (H.264 §9.3.1.1, eq. 9-5).
    pub(crate) fn init(m: i8, n: i8, slice_qp: u8) -> Self {
        let qp = i32::from(slice_qp.min(51));
        let pre_ctx_state = (((i32::from(m) * qp) >> 4) + i32::from(n)).clamp(1, 126);
        if pre_ctx_state <= 63 {
            Self::new((63 - pre_ctx_state) as u8, false)
        } else {
            Self::new((pre_ctx_state - 64) as u8, true)
        }
    }
}

impl BinSink for CabacEncoder {
    fn encode_decision(&mut self, ctx_idx: usize, bin: bool) {
        self.write(|writer| writer.encode_decision(ctx_idx, bin));
    }

    fn encode_bypass(&mut self, bin: bool) {
        self.write(|writer| writer.encode_bypass(bin));
    }

    fn encode_terminate(&mut self, bin: bool) {
        self.write(|writer| writer.encode_terminate(bin));
    }
}

/// Bits of `low` below the code bits: the spec's 10-bit `codILow`.
const LOW_BITS: i32 = 10;

/// A CABAC encoding engine for one slice.
#[derive(Debug)]
pub struct CabacEncoder {
    contexts: [Context; CONTEXT_COUNT],
    registers: Registers,
    out: Output,
}

/// The arithmetic coder's registers. They are copied into a
/// [`CabacWriter`] for each macroblock, and into a local writer for each
/// block's levels ([`BinSink::code_run`]), where the compiler can keep them
/// in machine registers; reached through the encoder, every bin loaded and
/// stored them (Jetson cycles -1.0% on crowd).
#[derive(Debug, Clone, Copy)]
struct Registers {
    /// The spec's `codILow` in bits 9..0, preceded by `queue + 8` code
    /// bits that are not written yet (plus a possible carry above them).
    low: u32,
    range: u32,
    /// Code bits waiting in `low`, minus 8: a byte is written when it
    /// reaches 0. Starts at -9, which drops the spec's suppressed first bit.
    queue: i32,
}

/// The slice data written so far.
#[derive(Debug)]
struct Output {
    /// `0xff` bytes held back until it is known whether a carry turns them
    /// into `0x00`s (and increments the byte before them).
    outstanding: u32,
    /// Slice data bytes written so far.
    bytes: Vec<u8>,
    /// The slice header, before the slice data.
    header: BitWriter,
}

impl CabacEncoder {
    /// Start a slice: `out` holds the slice header and must be byte aligned
    /// (after `cabac_alignment_one_bit`s, H.264 §7.3.4). Contexts are
    /// initialised for a `kind` slice with `cabac_init_idc` at `slice_qp`.
    pub fn new(out: BitWriter, kind: SliceKind, cabac_init_idc: u8, slice_qp: u8) -> Self {
        assert!(
            out.is_byte_aligned(),
            "CABAC slice data must start byte aligned"
        );
        let mut contexts = [Context::default(); CONTEXT_COUNT];
        for (ctx, mn) in contexts.iter_mut().zip(CABAC_INIT_MN.iter()) {
            let (m, n) = mn[kind.init_column(cabac_init_idc)];
            *ctx = Context::init(m, n, slice_qp);
        }
        // §9.3.1.2: initialisation of the arithmetic encoding engine.
        Self {
            contexts,
            registers: Registers {
                low: 0,
                range: 510,
                queue: -9,
            },
            out: Output {
                outstanding: 0,
                bytes: Vec::with_capacity(4096),
                header: out,
            },
        }
    }

    /// A [`BitCounter`] pricing bins with the current context states.
    pub fn bit_counter<'a>(&'a self, costs: &'a BinCosts) -> BitCounter<'a> {
        BitCounter::new(&self.contexts, costs)
    }

    /// Code the bins `f` writes (e.g. one macroblock's) through a
    /// [`CabacWriter`].
    pub fn write<R>(&mut self, f: impl FnOnce(&mut CabacWriter<'_>) -> R) -> R {
        let mut writer = CabacWriter {
            registers: self.registers,
            contexts: &mut self.contexts,
            out: &mut self.out,
        };
        let result = f(&mut writer);
        self.registers = writer.registers;
        result
    }

    /// End the slice data: byte-align with zero bits (the stop bit was
    /// written by the final `encode_terminate(true)`) and return the RBSP.
    pub fn finish(self) -> Vec<u8> {
        let Output {
            bytes, mut header, ..
        } = self.out;
        header.put_bytes(&bytes);
        header.byte_align_zero();
        header.into_bytes()
    }
}

/// The encoder while it codes a run of bins: its registers by value (so
/// that they stay in machine registers), its contexts and output borrowed.
#[derive(Debug)]
pub struct CabacWriter<'a> {
    registers: Registers,
    contexts: &'a mut [Context; CONTEXT_COUNT],
    out: &'a mut Output,
}

impl BinSink for CabacWriter<'_> {
    /// The registers of a writer reached through a reference stay in memory
    /// (loaded and stored around every bin); a local writer's do not.
    #[inline]
    fn code_run<T: BinRun>(&mut self, run: T) -> T::Output {
        let mut local = CabacWriter {
            registers: self.registers,
            contexts: &mut *self.contexts,
            out: &mut *self.out,
        };
        let result = run.code(&mut local);
        self.registers = local.registers;
        result
    }

    /// `EncodeDecision` (§9.3.4.2) of `bin` with context `ctx_idx`.
    #[allow(
        clippy::inline_always,
        reason = "once per bin: the hot path of the entropy coder"
    )]
    #[inline(always)]
    fn encode_decision(&mut self, ctx_idx: usize, bin: bool) {
        let ctx = &mut self.contexts[ctx_idx];
        let r = &mut self.registers;
        let range_lps = ctx.range_lps((r.range >> 6) as usize);
        let range_mps = r.range - range_lps;
        // Branch-free: whether a bin is the MPS is close to a coin toss for
        // the significance and level bins, so selects beat a branch.
        let is_lps = bin != ctx.val_mps();
        r.low += if is_lps { range_mps } else { 0 };
        r.range = if is_lps { range_lps } else { range_mps };
        *ctx = ctx.next(bin);
        // `RenormE` (§9.3.4.3) unconditionally: the range stays below 512,
        // so its leading zeros beyond 23 are the shift, 0 when it is at
        // least 256 (most MPS bins).
        let shift = r.range.leading_zeros() - RANGE_LEADING_ZEROS;
        r.range <<= shift;
        r.low <<= shift;
        r.queue += shift as i32;
        if r.queue >= 0 {
            (r.low, r.queue) = self.out.put_byte(r.low, r.queue);
        }
    }

    /// `EncodeBypass` (§9.3.4.4) of `bin`.
    #[allow(
        clippy::inline_always,
        reason = "once per bin: the hot path of the entropy coder"
    )]
    #[inline(always)]
    fn encode_bypass(&mut self, bin: bool) {
        let r = &mut self.registers;
        r.low <<= 1;
        if bin {
            r.low += r.range;
        }
        r.queue += 1;
        if r.queue >= 0 {
            (r.low, r.queue) = self.out.put_byte(r.low, r.queue);
        }
    }

    /// `EncodeTerminate` (§9.3.4.5) of `bin` (ctxIdx 276). A `true` bin ends
    /// the slice and flushes the engine; the flush writes the
    /// `rbsp_stop_one_bit` itself.
    fn encode_terminate(&mut self, bin: bool) {
        let r = &mut self.registers;
        r.range -= 2;
        if bin {
            r.low += r.range;
            self.flush();
        } else if r.range < RENORM_THRESHOLD {
            // As for decisions: a range of 256 or more needs no
            // renormalisation (one end_of_slice_flag per macroblock).
            self.registers = self.out.renormalize(self.registers);
        }
    }
}

impl CabacWriter<'_> {
    /// `EncodeFlush` (§9.3.4.5): `codIRange = 2`, `RenormE` (a shift by 7),
    /// then `PutBit(codILow >> 9 & 1)` and `WriteBits(((codILow >> 7) & 3) |
    /// 1, 2)`. In this layout that writes every code bit down to bit 7 of
    /// `low`, with bit 7 forced to 1 (the `rbsp_stop_one_bit`); the bits
    /// below are dropped and the last byte is padded with zeros.
    fn flush(&mut self) {
        self.registers.range = 2;
        let r = self.out.renormalize(self.registers);
        self.registers = r;
        let low = (r.low | 0x80) >> 7;
        // `low` now holds the remaining code bits (`queue + 11` of them)
        // with a possible carry above.
        let count = (r.queue + LOW_BITS + 1) as u32;
        let out = &mut *self.out;
        out.resolve(low >> count);
        let pad = (8 - count % 8) % 8;
        let mut bits = u64::from(low & ((1 << count) - 1)) << pad;
        let mut remaining = count + pad;
        while remaining > 0 {
            remaining -= 8;
            out.bytes.push((bits >> remaining) as u8);
            bits &= (1 << remaining) - 1;
        }
    }
}

impl Output {
    /// `RenormE` (§9.3.4.3) in one step: `codIRange` is doubled until it is
    /// at least 256, i.e. shifted by the number of leading zeros it has
    /// beyond the 9-bit range.
    fn renormalize(&mut self, mut r: Registers) -> Registers {
        let shift = r.range.leading_zeros().saturating_sub(RANGE_LEADING_ZEROS);
        r.range <<= shift;
        r.low <<= shift;
        r.queue += shift as i32;
        if r.queue >= 0 {
            (r.low, r.queue) = self.put_byte(r.low, r.queue);
        }
        r
    }

    /// Write the top 8 code bits of `low` (resolving a carry into what was
    /// written before) once there are at least 8 of them, and return the
    /// new `(low, queue)`. Out of line: it runs about once per 8 code bits.
    /// The registers go by value (not as one struct, which would be passed
    /// through memory), so that the callers keep them in machine registers.
    #[inline(never)]
    fn put_byte(&mut self, low: u32, queue: i32) -> (u32, i32) {
        let shift = queue + LOW_BITS;
        let out = low >> shift;
        if out & 0xff == 0xff {
            self.outstanding += 1;
        } else {
            self.resolve(out >> 8);
            self.bytes.push(out as u8);
        }
        (low & ((1 << shift) - 1), queue - 8)
    }

    /// Settle held-back bytes: add `carry` (0 or 1) to the last written byte
    /// and write the `0xff`s (which a carry turns into `0x00`s).
    fn resolve(&mut self, carry: u32) {
        if carry != 0
            && let Some(last) = self.bytes.last_mut()
        {
            *last = last.wrapping_add(1);
        }
        let held = if carry != 0 { 0x00 } else { 0xff };
        for _ in 0..self.outstanding {
            self.bytes.push(held);
        }
        self.outstanding = 0;
    }
}
