//! Superblock quantizer deltas (`delta_q_present`, AVM
//! `read_delta_q_params` / `read_delta_qindex` in `decodemv.c`): how the
//! engine's per-block quantizer offsets reach the bitstream, and how the
//! decoder follows them.
//!
//! As in AV1, a delta is coded only at the first (top-left) block of each
//! superblock, after its `skip_txfm` and CDEF index, unless that block is the
//! whole superblock and skipped. It moves `current_base_qindex`, which every
//! later block of the tile inherits until the next delta, so the engine plans
//! offsets per superblock (`CodecBackend::BLOCK_QP_LOG2`). Unlike AV1, AV2
//! deblocking reads each block's own qindex (`final_qindex_ac`), so a block
//! that codes no delta must record the inherited one.
//!
//! The encoder codes `delta_q_res` 0 (deltas in qindex units); the decoder
//! follows any. Frames code the syntax only when the engine may send
//! offsets, so streams without block AQ are unchanged.

use nc_common::decode::Result;

use crate::entropy::{Reader, Writer};

/// `DELTA_Q_SMALL`: `delta_q_abs` values coded directly by the symbol; the
/// largest symbol escapes to a literal (AVM `enums.h`).
const DELTA_Q_SMALL: u32 = 7;
/// Symbols of the `delta_q_abs` CDF (`DELTA_Q_PROBS + 1`).
const DELTA_Q_SYMBOLS: usize = DELTA_Q_SMALL as usize + 1;
/// `DELTA_Q_SMALL_MINUS_2`: offset of the escaped magnitudes.
const DELTA_Q_ESCAPE_OFFSET: u32 = DELTA_Q_SMALL - 2;
/// Bits of the escape's `rem_bits - 1` literal.
const DELTA_Q_REM_BITS: u32 = 3;
/// `current_base_qindex` range (`Clip3(1, MAXQ_8_BITS, ...)`): qindex 0
/// would make a block lossless, a different tool set.
const MIN_QINDEX: i32 = 1;
const MAX_QINDEX: i32 = 255;
/// `default_delta_q_cdf` as an inverse CDF (the generated
/// `TileCdfs::delta_q` default), for tiles that do not adapt.
const DELTA_Q_FIXED: [u16; DELTA_Q_SYMBOLS] = [16174, 9443, 6344, 4543, 3410, 2669, 2155, 0];

/// Bits of the frame header's `log2(delta_q_res)`.
pub(crate) const DELTA_Q_RES_BITS: u32 = 2;

/// The superblock quantizer state of one tile.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SuperblockQindex {
    /// The frame codes `delta_q_present` = 1.
    enabled: bool,
    /// `log2(delta_q_res)`: coded deltas are in units of `1 << res_log2`.
    res_log2: u8,
    /// `current_base_qindex`: the quantizer later blocks inherit.
    current: u8,
    /// The superblock's first block is still to be coded.
    pending: bool,
}

impl SuperblockQindex {
    /// A tile of a frame at `base_qindex`; `delta_q` is the frame header's
    /// `log2(delta_q_res)` when it codes `delta_q_present`, `None` without.
    pub(crate) fn new(base_qindex: u8, delta_q: Option<u8>) -> Self {
        Self {
            enabled: delta_q.is_some(),
            res_log2: delta_q.unwrap_or(0),
            current: base_qindex,
            pending: false,
        }
    }

    /// A superblock starts: its first block may code a delta.
    pub(crate) fn start_superblock(&mut self) {
        self.pending = self.enabled;
    }

    /// `current_base_qindex` (the qindex of a block that codes no delta).
    pub(crate) fn current(self) -> u8 {
        self.current
    }

    /// The quantizer of the next block: `frame_qindex + offset` (clamped)
    /// for a superblock's first block, the inherited one otherwise.
    pub(crate) fn block_qindex(self, frame_qindex: u8, offset: i16) -> u8 {
        if self.pending {
            (i32::from(frame_qindex) + i32::from(offset)).clamp(MIN_QINDEX, MAX_QINDEX) as u8
        } else {
            self.current
        }
    }

    /// Whether the block (`whole`: the whole superblock, `skip`: its
    /// `skip_txfm`) codes a delta: the first block of a superblock unless it
    /// is the whole superblock and skipped.
    pub(crate) fn codes_delta(self, whole: bool, skip: bool) -> bool {
        self.pending && !(whole && skip)
    }

    /// Write the move to `qindex` for a block that [`Self::codes_delta`]
    /// (encoder frames code `delta_q_res` 1, so any move is exact).
    /// The state is left unchanged, so rate trials can price it; the final
    /// coding calls [`Self::commit`].
    /// # Errors
    /// Propagates symbol coding errors.
    pub(crate) fn write(self, writer: &mut Writer, qindex: u8) -> Result<()> {
        debug_assert_eq!(self.res_log2, 0, "the encoder codes delta_q_res 1");
        write_delta(writer, i32::from(qindex) - i32::from(self.current))
    }

    /// The block's coding is final: the superblock's first block is done
    /// and, when it coded a delta (`coded`), `qindex` becomes current.
    pub(crate) fn commit(&mut self, qindex: u8, coded: bool) {
        if std::mem::take(&mut self.pending) && coded {
            self.current = qindex;
        }
    }

    /// Read a block's delta where it [`Self::codes_delta`] and return the
    /// block's qindex.
    /// # Errors
    /// Propagates truncated or malformed symbols.
    pub(crate) fn read(&mut self, reader: &mut Reader<'_>, whole: bool, skip: bool) -> Result<u8> {
        if self.codes_delta(whole, skip) {
            let delta = read_delta(reader)? << self.res_log2;
            self.current = (i32::from(self.current) + delta).clamp(MIN_QINDEX, MAX_QINDEX) as u8;
        }
        self.pending = false;
        Ok(self.current)
    }
}

/// AVM `read_delta_qindex`, encoder side: `delta_q_abs` (escaping to
/// `rem_bits` and `abs - thr`), then the sign when nonzero.
fn write_delta(writer: &mut Writer, delta: i32) -> Result<()> {
    let abs = delta.unsigned_abs();
    let symbol = abs.min(DELTA_Q_SMALL) as usize;
    writer.model(symbol, &DELTA_Q_FIXED, |m| &mut m.delta_q)?;
    if abs >= DELTA_Q_SMALL {
        // abs = literal(rem_bits) + (1 << rem_bits) + DELTA_Q_SMALL - 2.
        let rem_bits = (abs - DELTA_Q_ESCAPE_OFFSET).ilog2();
        writer.literal(rem_bits - 1, DELTA_Q_REM_BITS)?;
        writer.literal(abs - DELTA_Q_ESCAPE_OFFSET - (1 << rem_bits), rem_bits)?;
    }
    if abs != 0 {
        writer.literal(u32::from(delta < 0), 1)?;
    }
    Ok(())
}

/// AVM `read_delta_qindex`.
fn read_delta(reader: &mut Reader<'_>) -> Result<i32> {
    let mut abs = reader.model(&DELTA_Q_FIXED, |m| &mut m.delta_q)? as u32;
    if abs >= DELTA_Q_SMALL {
        let rem_bits = reader.literal(DELTA_Q_REM_BITS)? + 1;
        abs = reader.literal(rem_bits)? + (1 << rem_bits) + DELTA_Q_ESCAPE_OFFSET;
    }
    let negative = abs != 0 && reader.literal(1)? == 1;
    let abs = abs as i32;
    Ok(if negative { -abs } else { abs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltas_roundtrip_through_the_escape() {
        for adaptive in [false, true] {
            let deltas = [0, 1, -1, 6, -6, 7, -7, 8, 37, -100, 254, -254];
            let mut writer = Writer::new();
            if adaptive {
                writer.enable_adaptation(100);
            }
            for &delta in &deltas {
                write_delta(&mut writer, delta).unwrap();
            }
            let bytes = writer.finish();
            let mut reader = Reader::new(&bytes).unwrap();
            if adaptive {
                reader.enable_adaptation(100);
            }
            for &delta in &deltas {
                assert_eq!(read_delta(&mut reader).unwrap(), delta);
            }
        }
    }

    #[test]
    fn first_block_sets_the_superblock_quantizer() {
        let mut sb = SuperblockQindex::new(100, Some(0));
        sb.start_superblock();
        assert_eq!(sb.block_qindex(100, -120), 1);
        assert_eq!(sb.block_qindex(100, 200), 255);
        let q = sb.block_qindex(100, -12);
        assert!(sb.codes_delta(false, true));
        sb.commit(q, true);
        // Later blocks inherit it and code nothing.
        assert!(!sb.codes_delta(false, false));
        assert_eq!(sb.block_qindex(100, 30), 88);
        // A whole skipped superblock keeps the previous quantizer.
        sb.start_superblock();
        assert!(!sb.codes_delta(true, true));
        sb.commit(130, false);
        assert_eq!(sb.current(), 88);
        // Disabled frames never code deltas.
        let mut off = SuperblockQindex::new(60, None);
        off.start_superblock();
        assert!(!off.codes_delta(false, false));
        assert_eq!(off.block_qindex(60, -10), 60);
    }
}
