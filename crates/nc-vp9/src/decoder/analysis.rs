//! The frame decoder's analysis records (feature `analysis`): per block
//! (`decode_block`, VP9 §6.4.3) one coding record, its prediction records
//! (one, or one per 4x4 of a sub-8x8 block) and its transform blocks, pushed
//! into [`StreamStats`]. Without the feature this module and every hook are
//! compiled out, so the decoder contains no analysis code.
//!
//! VP9 in the codec-neutral records: `qp` is the segment's qindex
//! (0..=255); vectors are in 1/8 sample; an inter block without residual
//! and without `NEWMV` is `Skip`, `NEWMV` is `Inter` (explicit difference),
//! `NEARESTMV` / `NEARMV` / `ZEROMV` are `Merge`; `mode` holds the intra
//! mode (`PREDICTION_MODE` order) or the inter mode (0 nearest, 1 near,
//! 2 zero, 3 new). Transform blocks tile the block at its `tx_size`; their
//! coded flag is the block's (VP9 codes no per-transform flag to read back
//! cheaply).

use nc_common::stream_stats::{
    BlockRecord, IntraClass, Pred, RecordKind, Shape, StreamStats, facts, flags,
};

use super::block::{BlockInfo, BlockSize, Partition, PredMode};
use super::frame::{BlockPosition, FrameDecoder};
use super::header::FrameHeader;
use crate::types::{IntraMode, MI_SIZE, Mv};

/// VP9 vectors are in 1/8 sample.
const MV_UNITS_PER_SAMPLE: i64 = 8;
/// `slice_type` facts: key (and intra-only) frames, inter frames.
const KEY: i64 = 2;
const INTER: i64 = 0;
/// Luma samples of a 4x4 sub-block.
const SUB_BLOCK: usize = 4;

/// What a block's records collect between `decode_block`'s start and end.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BlockScratch {
    /// The partition the block came from.
    pub shape: Shape,
}

/// The neutral shape of a partition type.
pub(super) fn shape(partition: Partition) -> Shape {
    match partition {
        Partition::None | Partition::Split => Shape::Whole,
        Partition::Horizontal => Shape::HalfHorizontal,
        Partition::Vertical => Shape::HalfVertical,
    }
}

/// Start a frame's records with its facts (§6.2): `slice_qp` is
/// `base_q_idx`, `slice_bytes` the frame's.
pub(super) fn frame_facts(s: &mut StreamStats, header: &FrameHeader, bytes: usize) {
    s.begin_frame();
    s.fact(facts::MV_UNITS_PER_SAMPLE, MV_UNITS_PER_SAMPLE, 0);
    s.fact("slice_type", if header.is_intra() { KEY } else { INTER }, 0);
    s.fact("slice_qp", i64::from(header.quantizer.base), 0);
    s.fact("slice_bytes", 0, bytes as u64);
}

fn intra_class(mode: IntraMode) -> IntraClass {
    match mode {
        IntraMode::Dc => IntraClass::Dc,
        IntraMode::Tm => IntraClass::Other,
        _ => IntraClass::Angular,
    }
}

fn vector(mv: Mv) -> [i16; 2] {
    let clamp = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    [clamp(mv.col), clamp(mv.row)]
}

/// The inter mode as the records number it.
fn inter_mode(mode: PredMode) -> u8 {
    match mode {
        PredMode::NearestMv => 0,
        PredMode::NearMv => 1,
        PredMode::ZeroMv => 2,
        PredMode::NewMv | PredMode::Intra(_) => 3,
    }
}

fn side(log2_4x4: u8) -> u16 {
    (SUB_BLOCK << log2_4x4) as u16
}

impl FrameDecoder<'_> {
    /// The records of a decoded block; `bits` it cost, `coded` whether it
    /// has residual.
    pub(super) fn block_records(
        &mut self,
        info: &BlockInfo,
        position: &BlockPosition,
        bits: u32,
        coded: bool,
    ) {
        let size = info.size;
        let header = self.inputs.header;
        let qindex = header
            .segmentation
            .qindex(info.segment_id, header.quantizer.base);
        let placed = BlockRecord {
            x: (position.mi_col * MI_SIZE) as u16,
            y: (position.mi_row * MI_SIZE) as u16,
            w: side(size.width_log2()),
            h: side(size.height_log2()),
            qp: i16::from(qindex),
            ..BlockRecord::default()
        };
        let pred = match (info.is_inter(), coded, info.mode == PredMode::NewMv) {
            (false, _, _) => Pred::Intra,
            (true, false, false) => Pred::Skip,
            (true, _, true) => Pred::Inter,
            (true, true, false) => Pred::Merge,
        };
        let shape = match size {
            BlockSize::B4x4 => Shape::Quarter,
            BlockSize::B4x8 => Shape::HalfVertical,
            BlockSize::B8x4 => Shape::HalfHorizontal,
            _ => self.scratch.analysis.shape,
        };
        let coded_flags = if coded {
            flags::CODED_Y | flags::CODED_CB | flags::CODED_CR
        } else {
            0
        };
        let frame = (
            (self.mi_cols * MI_SIZE) as u16,
            (self.mi_rows * MI_SIZE) as u16,
        );
        let Some(records) = self.scratch.stats.as_mut() else {
            return;
        };
        records.push(BlockRecord {
            kind: RecordKind::Coding,
            pred,
            shape,
            flags: coded_flags,
            bits,
            ..placed
        });
        prediction_records(records, info, pred, placed);
        transform_records(records, info, placed, frame, coded);
    }
}

/// The prediction records of a block: the whole block, or each 4x4 of a
/// sub-8x8 one.
fn prediction_records(
    records: &mut StreamStats,
    info: &BlockInfo,
    pred: Pred,
    placed: BlockRecord,
) {
    let sub8x8 = info.size.is_sub8x8();
    let parts: &[(u16, u16)] = if sub8x8 {
        &[(0, 0), (4, 0), (0, 4), (4, 4)]
    } else {
        &[(0, 0)]
    };
    let (width, height) = if sub8x8 {
        (SUB_BLOCK as u16, SUB_BLOCK as u16)
    } else {
        (placed.w, placed.h)
    };
    for (index, &(dx, dy)) in parts.iter().enumerate() {
        if dx >= placed.w || dy >= placed.h {
            continue;
        }
        let (mode, mvs) = if sub8x8 {
            (info.sub_modes[index], info.sub_mvs[index])
        } else {
            (info.mode, info.mv)
        };
        let base = BlockRecord {
            kind: RecordKind::Prediction,
            x: placed.x + dx,
            y: placed.y + dy,
            w: width,
            h: height,
            ..placed
        };
        records.push(if info.is_inter() {
            let compound = info.is_compound();
            BlockRecord {
                pred,
                mode: inter_mode(mode),
                flags: if compound { flags::BI } else { 0 },
                ref_idx: [
                    info.ref_frame[0],
                    if compound { info.ref_frame[1] } else { -1 },
                ],
                mv: mvs.map(vector),
                ..base
            }
        } else {
            let intra = mode.intra();
            BlockRecord {
                pred: Pred::Intra,
                mode: intra as u8,
                mode2: info.uv_mode as u8,
                class: intra_class(intra) as u8,
                ref_idx: [-1; 2],
                ..base
            }
        });
    }
}

/// The transform records of a block: its area at `tx_size`, inside the
/// frame (`frame`: its size in samples).
fn transform_records(
    records: &mut StreamStats,
    info: &BlockInfo,
    placed: BlockRecord,
    (frame_w, frame_h): (u16, u16),
    coded: bool,
) {
    let tx_side = 1u16 << info.tx_size.log2();
    let pred = if info.is_inter() {
        Pred::Inter
    } else {
        Pred::Intra
    };
    for ty in (0..placed.h).step_by(usize::from(tx_side)) {
        for tx in (0..placed.w).step_by(usize::from(tx_side)) {
            if placed.x + tx >= frame_w || placed.y + ty >= frame_h {
                continue;
            }
            records.push(BlockRecord {
                kind: RecordKind::Transform,
                pred,
                x: placed.x + tx,
                y: placed.y + ty,
                w: tx_side,
                h: tx_side,
                flags: if coded { flags::CODED_Y } else { 0 },
                ..placed
            });
        }
    }
}
