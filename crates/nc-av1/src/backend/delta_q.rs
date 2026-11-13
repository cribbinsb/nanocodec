//! Superblock quantizer deltas (`delta_q`, AV1 §5.9.17, §5.11.12): how the
//! engine's per-block quantizer offsets (`BlockDecision::qp_offset`) reach
//! the bitstream.
//!
//! AV1 codes one `delta_qindex` per 64x64 superblock, at its first block,
//! moving `CurrentQIndex`, which every later block of the tile inherits until
//! the next delta. So the quantizer of a superblock is fixed by its first
//! coded block, and the engine plans offsets per superblock
//! (`CodecBackend::BLOCK_QP_LOG2`). A superblock coded as one skipped block
//! codes no delta and keeps the previous `CurrentQIndex`; it has no residual,
//! so its quantizer does not matter.
//!
//! Frames code the syntax only when the engine may send offsets
//! (`CodecBackend::enable_block_qp`) and the frame's blocks may carry them
//! (`FramePlan::block_qp`), so streams without AQ are unchanged and frames
//! without offsets pay no zero deltas.

use crate::cdf::Cdf;
use crate::range_coder::SymbolWriter;

/// `DELTA_Q_SMALL`: `delta_q_abs` values coded directly by the symbol; the
/// largest escapes to `delta_q_rem_bits` and `delta_q_abs_bits`.
const DELTA_Q_SMALL: usize = 3;
/// Bits of `delta_q_rem_bits`.
const DELTA_Q_REM_BITS: u32 = 3;
/// `CurrentQIndex` range (§5.11.12 `Clip3(1, 255, ...)`); qindex 0 would
/// make a block lossless, a different tool set.
const MIN_QINDEX: i32 = 1;
const MAX_QINDEX: i32 = 255;

/// The superblock quantizer state of the frame being coded.
#[derive(Debug, Clone, Default)]
pub(super) struct DeltaQ {
    /// Whether frames may code deltas (set once by `enable_block_qp`).
    enabled: bool,
    /// `delta_q_present` of the frame being coded.
    present: bool,
    /// `CurrentQIndex`: the quantizer of the block being coded.
    current: u32,
    /// `ReadDeltas`: the superblock's first block is still to be coded.
    pending: bool,
}

impl DeltaQ {
    /// Code `delta_q_present` = 1 from now on in frames whose blocks may
    /// carry offsets.
    pub(super) fn enable(&mut self) {
        self.enabled = true;
    }

    /// `delta_q_present` of the frame header.
    pub(super) fn present(&self) -> bool {
        self.present
    }

    /// A tile starts at the frame's `base_q_idx` (§5.11.1); `block_qp`:
    /// the frame's blocks may carry offsets.
    pub(super) fn start_frame(&mut self, base_q_idx: u32, block_qp: bool) {
        self.present = self.enabled && block_qp;
        self.current = base_q_idx;
        self.pending = false;
    }

    /// `ReadDeltas = delta_q_present` at each superblock (§5.11.2).
    pub(super) fn start_superblock(&mut self) {
        self.pending = self.present;
    }

    /// The quantizer of the next block: `frame_q_idx + offset` for a
    /// superblock's first block, else the superblock's.
    pub(super) fn block_qindex(&self, frame_q_idx: u32, offset: i16) -> u32 {
        if self.pending {
            (frame_q_idx as i32 + i32::from(offset)).clamp(MIN_QINDEX, MAX_QINDEX) as u32
        } else {
            self.current
        }
    }

    /// `read_delta_qindex()` (§5.11.12) after the block's `skip` and
    /// `cdef_idx`: codes the move to `qindex` at a superblock's first block,
    /// unless that block is the whole superblock and skipped.
    pub(super) fn write(
        &mut self,
        writer: &mut SymbolWriter,
        cdf: &mut Cdf<4>,
        qindex: u32,
        whole_superblock_skip: bool,
    ) {
        if !std::mem::take(&mut self.pending) || whole_superblock_skip {
            return;
        }
        // delta_q_res = 0: deltas are coded in qindex units.
        let delta = qindex as i32 - self.current as i32;
        let abs = delta.unsigned_abs();
        writer.write_symbol((abs as usize).min(DELTA_Q_SMALL), cdf, DELTA_Q_SMALL + 1);
        if abs as usize >= DELTA_Q_SMALL {
            // abs = delta_q_abs_bits + (1 << rem_bits) + 1.
            let rem_bits = (abs - 1).ilog2();
            write_bits(writer, rem_bits - 1, DELTA_Q_REM_BITS);
            write_bits(writer, abs - 1 - (1 << rem_bits), rem_bits);
        }
        if abs != 0 {
            writer.write_literal_bit(delta < 0); // delta_q_sign_bit
        }
        self.current = qindex;
    }
}

/// `L(n)`: `value` in `bits` literal bits, most significant first.
fn write_bits(writer: &mut SymbolWriter, value: u32, bits: u32) {
    for bit in (0..bits).rev() {
        writer.write_literal_bit((value >> bit) & 1 == 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_block_sets_the_superblock_quantizer() {
        let mut dq = DeltaQ::default();
        dq.enable();
        dq.start_frame(100, true);
        dq.start_superblock();
        assert_eq!(dq.block_qindex(100, -120), 1);
        assert_eq!(dq.block_qindex(100, 200), 255);
        let q = dq.block_qindex(100, -12);
        let mut writer = SymbolWriter::new();
        let mut cdf = Cdf::new(crate::tables::DELTA_Q_CDF);
        dq.write(&mut writer, &mut cdf, q, false);
        // Later blocks of the superblock inherit it and code nothing.
        assert_eq!(dq.block_qindex(100, 30), 88);
        dq.start_superblock();
        // A whole skipped superblock keeps the previous quantizer.
        dq.write(&mut writer, &mut cdf, 130, true);
        assert_eq!(dq.block_qindex(100, 0), 88);
    }

    #[test]
    fn disabled_frames_code_no_deltas() {
        let mut dq = DeltaQ::default();
        dq.start_frame(60, true);
        dq.start_superblock();
        assert_eq!(dq.block_qindex(60, -10), 60);
        assert!(!dq.present());
    }
}
