//! The tile decoder's analysis records (feature `analysis`): one coding
//! and one prediction record per block (`decode_block()`, AV1 §5.11.5)
//! and one transform record per luma transform block (§5.11.35), pushed
//! into [`StreamStats`]. Without the feature this module and every hook
//! are compiled out, so the decoder contains no analysis code.
//!
//! AV1 in the codec-neutral records: `qp` is the block's qindex
//! (`CurrentQIndex`, 0..=255); vectors are in 1/8 sample; a block coded
//! with `skip_mode` is `Skip`, `NEWMV`-family modes are `Inter` (explicit
//! difference) and the other inter modes `Merge`; `mode` holds the y mode
//! (intra, `IntraMode` symbol order) or the inter mode (`InterMode`
//! order); reference indices are reference names (`LAST_FRAME` = 1).

use nc_common::stream_stats::{
    BlockRecord, IntraClass, Pred, RecordKind, Shape, StreamStats, facts, flags,
};

use super::super::subset::FrameParameters;
use super::super::transform::TxSize;
use super::inter::InterModes;
use super::intra::IntraModes;
use super::{BlockPos, FrameDecoder};
use crate::backend::FrameKind;
use crate::decoder_range::SymbolReader;
use crate::modeinfo::Av1Mv;
use crate::tables::Partition;

/// `YMode` symbols (§6.10.22): `DC_PRED`, the directional `V_PRED` ..
/// `D67_PRED`, then `SMOOTH_PRED`, `SMOOTH_V_PRED`, `SMOOTH_H_PRED`,
/// `PAETH_PRED`.
const DC_PRED: u8 = 0;
const LAST_DIRECTIONAL: u8 = 8;
const LAST_SMOOTH: u8 = 11;
/// Luma samples per 4x4 mode-info unit.
const MI_SIZE: usize = 4;

/// What a block's records collect between `decode_block()`'s start and
/// its end.
#[derive(Debug, Clone, Copy, Default)]
pub(in super::super) struct BlockScratch {
    /// The partition the block came from.
    pub shape: Shape,
    /// Bits consumed when the block started.
    pub start: usize,
    /// Whether the block is inter (its transform records' `pred`).
    pub inter: bool,
}

/// Analysis facts of a frame (§5.9): starts its records. `slice_qp` is
/// `base_q_idx`, `slice_type` 2 for key frames, `slice_bytes` the tiles'.
pub(super) fn frame_facts(s: &mut StreamStats, params: &FrameParameters, tiles: &[&[u8]]) {
    /// AV1 vectors are in 1/8 sample.
    const MV_UNITS_PER_SAMPLE: i64 = 8;
    const KEY: i64 = 2;
    const INTER: i64 = 0;
    s.begin_frame();
    s.fact(facts::MV_UNITS_PER_SAMPLE, MV_UNITS_PER_SAMPLE, 0);
    let kind = if params.kind == FrameKind::Inter {
        INTER
    } else {
        KEY
    };
    s.fact("slice_type", kind, 0);
    s.fact("slice_qp", i64::from(params.base_q_idx), 0);
    let bytes: usize = tiles.iter().map(|t| t.len()).sum();
    s.fact("slice_bytes", 0, bytes as u64);
}

/// The neutral shape of a partition type (§6.10.4).
pub(super) fn shape(partition: Partition) -> Shape {
    match partition {
        Partition::None | Partition::Split => Shape::Whole,
        Partition::Horz => Shape::HalfHorizontal,
        Partition::Vert => Shape::HalfVertical,
        Partition::HorzA => Shape::QuarterTop,
        Partition::HorzB => Shape::QuarterBottom,
        Partition::VertA => Shape::QuarterLeft,
        Partition::VertB => Shape::QuarterRight,
        Partition::Horz4 | Partition::Vert4 => Shape::Sixteenth,
    }
}

/// The neutral class of a `YMode` symbol.
fn intra_class(mode: u8) -> IntraClass {
    match mode {
        DC_PRED => IntraClass::Dc,
        m if m <= LAST_DIRECTIONAL => IntraClass::Angular,
        m if m <= LAST_SMOOTH => IntraClass::Planar,
        _ => IntraClass::Other,
    }
}

/// A vector as `[x, y]`.
fn vector(mv: Av1Mv) -> [i16; 2] {
    [mv.col, mv.row]
}

impl FrameDecoder<'_> {
    /// `decode_block()` starts: remember where (the bits) and from which
    /// partition.
    pub(super) fn start_block_record(&mut self, reader: &SymbolReader<'_>, inter: bool) {
        self.maps.analysis.start = reader.bits_consumed();
        self.maps.analysis.inter = inter;
    }

    /// A record of `kind` placed on the block (luma samples).
    fn placed(kind: RecordKind, block: BlockPos) -> BlockRecord {
        BlockRecord {
            kind,
            x: (block.mi_col * MI_SIZE) as u16,
            y: (block.mi_row * MI_SIZE) as u16,
            w: block.size.width() as u16,
            h: block.size.height() as u16,
            ..BlockRecord::default()
        }
    }

    fn coding_record(
        &mut self,
        reader: &SymbolReader<'_>,
        block: BlockPos,
        pred: Pred,
        skip: bool,
    ) {
        let scratch = self.maps.analysis;
        let coded = if skip {
            0
        } else {
            flags::CODED_Y | flags::CODED_CB | flags::CODED_CR
        };
        let record = BlockRecord {
            pred,
            shape: scratch.shape,
            flags: coded,
            qp: self.block_q as i16,
            bits: (reader.bits_consumed() - scratch.start) as u32,
            ..Self::placed(RecordKind::Coding, block)
        };
        if let Some(s) = &mut self.maps.stats {
            s.push(record);
        }
    }

    /// An intra block's coding and prediction records.
    pub(super) fn intra_records(
        &mut self,
        reader: &SymbolReader<'_>,
        block: BlockPos,
        modes: &IntraModes,
        skip: bool,
    ) {
        self.coding_record(reader, block, Pred::Intra, skip);
        let mode = modes.luma.mode.symbol() as u8;
        let record = BlockRecord {
            pred: Pred::Intra,
            mode,
            mode2: modes.uv as u8,
            class: intra_class(mode) as u8,
            ref_idx: [-1; 2],
            ..Self::placed(RecordKind::Prediction, block)
        };
        if let Some(s) = &mut self.maps.stats {
            s.push(record);
        }
    }

    /// An inter (or intra block copy) block's coding and prediction
    /// records.
    pub(super) fn inter_records(
        &mut self,
        reader: &SymbolReader<'_>,
        block: BlockPos,
        modes: &InterModes,
        (skip, skip_mode): (bool, bool),
    ) {
        let pred = if skip_mode || skip && !modes.mode.has_new_mv() {
            Pred::Skip
        } else if modes.mode.has_new_mv() {
            Pred::Inter
        } else {
            Pred::Merge
        };
        self.coding_record(reader, block, pred, skip);
        let compound = modes.compound.is_some();
        let record = BlockRecord {
            pred,
            mode: modes.mode as u8,
            flags: if compound { flags::BI } else { 0 },
            ref_idx: [modes.refs[0], if compound { modes.refs[1] } else { -1 }],
            mv: modes.mv.map(vector),
            ..Self::placed(RecordKind::Prediction, block)
        };
        if let Some(s) = &mut self.maps.stats {
            s.push(record);
        }
    }

    /// A luma transform block's record: `eob` levels coded (0: none).
    pub(super) fn transform_record(&mut self, (px, py): (usize, usize), tx: TxSize, eob: usize) {
        let pred = if self.maps.analysis.inter {
            Pred::Inter
        } else {
            Pred::Intra
        };
        let record = BlockRecord {
            kind: RecordKind::Transform,
            pred,
            x: px as u16,
            y: py as u16,
            w: tx.width() as u16,
            h: tx.height() as u16,
            flags: if eob > 0 { flags::CODED_Y } else { 0 },
            qp: self.block_q as i16,
            nonzero: eob.min(usize::from(u16::MAX)) as u16,
            ..BlockRecord::default()
        };
        if let Some(s) = &mut self.maps.stats {
            s.push(record);
        }
    }
}
