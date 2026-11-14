//! The tile decoder's analysis records (feature `analysis`): per block
//! (`tile_decoder.rs`, AVM `decode_block`) one coding record, one prediction
//! record and its luma transform blocks, pushed into [`StreamStats`].
//! Without the feature this module and every hook are compiled out, so the
//! decoder contains no analysis code.
//!
//! AV2 in the codec-neutral records: `qp` is the block's luma AC qindex;
//! vectors are in 1/8 sample; NEWMV blocks are `Inter`, NEARMV / GLOBALMV
//! `Merge`, skip-mode and residual-free non-NEWMV blocks `Skip`; intra
//! `mode` follows AV1's y-mode numbering (0 DC, 1..=8 directional from
//! vertical, 9..=11 smooth, 12 Paeth). Transform blocks carry the block's
//! coded flag.

use nc_common::stream_stats::{
    BlockRecord, IntraClass, Pred, RecordKind, StreamStats, facts, flags,
};

use crate::intra_modes::{Direction, Mode};
use crate::mv_refs::BlockMotion;
use crate::tile_decoder::TileParams;
use crate::tx_partition::Layout;

/// AV2 vectors are in 1/8 sample.
const MV_UNITS_PER_SAMPLE: i64 = 8;
/// `slice_type` facts: key and inter frames.
const KEY: i64 = 2;
const INTER: i64 = 0;
/// AV1 y-mode numbers of the non-directional modes (§6.10.22).
const DC_PRED: u8 = 0;
const SMOOTH_PRED: u8 = 9;
const SMOOTH_V_PRED: u8 = 10;
const SMOOTH_H_PRED: u8 = 11;
const PAETH_PRED: u8 = 12;

/// A frame's analysis: where its records go (if anywhere) and the bits
/// consumed when the current block started.
#[derive(Debug)]
pub(crate) struct Recorder<'a> {
    stats: Option<&'a mut StreamStats>,
    pub start: usize,
}

impl<'a> Recorder<'a> {
    /// Start a frame's records (when `stats` collects them) with its facts.
    /// `inter`: the frame predicts from a reference; `bytes` its tile data.
    pub(crate) fn new(
        stats: Option<&'a mut StreamStats>,
        params: &TileParams,
        (inter, bytes): (bool, usize),
    ) -> Self {
        let mut stats = stats;
        if let Some(s) = stats.as_deref_mut() {
            frame_facts(s, params.qindex, inter, bytes);
        }
        Self { stats, start: 0 }
    }

    pub(crate) fn on(&self) -> bool {
        self.stats.is_some()
    }

    pub(crate) fn intra_block(&mut self, block: PlacedBlock, modes: (Mode, Mode), layout: Layout) {
        if let Some(s) = self.stats.as_deref_mut() {
            intra_block(s, block, modes, layout);
        }
    }

    pub(crate) fn inter_block(
        &mut self,
        block: PlacedBlock,
        motion: &BlockMotion,
        layout: Option<Layout>,
    ) {
        if let Some(s) = self.stats.as_deref_mut() {
            inter_block(s, block, motion, layout);
        }
    }
}

/// Start a frame's records with its facts: `slice_qp` is the base qindex,
/// `slice_bytes` the tile data's.
fn frame_facts(s: &mut StreamStats, qindex: u8, inter: bool, bytes: usize) {
    s.begin_frame();
    s.fact(facts::MV_UNITS_PER_SAMPLE, MV_UNITS_PER_SAMPLE, 0);
    s.fact("slice_type", if inter { INTER } else { KEY }, 0);
    s.fact("slice_qp", i64::from(qindex), 0);
    s.fact("slice_bytes", 0, bytes as u64);
}

/// AV1's y-mode number of a direction (`V_PRED` 1 .. `D67_PRED` 8).
fn direction_number(direction: Direction) -> u8 {
    match direction {
        Direction::Vertical => 1,
        Direction::Horizontal => 2,
        Direction::D45 => 3,
        Direction::D135 => 4,
        Direction::D113 => 5,
        Direction::D157 => 6,
        Direction::D203 => 7,
        Direction::D67 => 8,
    }
}

/// A mode's number and class.
fn intra_mode(mode: Mode) -> (u8, IntraClass) {
    match mode {
        Mode::Dc => (DC_PRED, IntraClass::Dc),
        Mode::Smooth => (SMOOTH_PRED, IntraClass::Planar),
        Mode::SmoothVertical => (SMOOTH_V_PRED, IntraClass::Planar),
        Mode::SmoothHorizontal => (SMOOTH_H_PRED, IntraClass::Planar),
        Mode::Paeth => (PAETH_PRED, IntraClass::Other),
        Mode::Directional { direction, .. } => (direction_number(direction), IntraClass::Angular),
    }
}

/// One block as the records place it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlacedBlock {
    pub origin: (usize, usize),
    pub size: usize,
    pub qindex: u8,
    pub bits: u32,
}

impl PlacedBlock {
    fn record(self, kind: RecordKind) -> BlockRecord {
        BlockRecord {
            kind,
            x: self.origin.0 as u16,
            y: self.origin.1 as u16,
            w: self.size as u16,
            h: self.size as u16,
            qp: i16::from(self.qindex),
            ..BlockRecord::default()
        }
    }
}

fn transforms(s: &mut StreamStats, block: PlacedBlock, layout: Option<Layout>, pred: Pred) {
    let Some(layout) = layout else {
        return;
    };
    let side = layout.luma.width() as u16;
    for (x, y) in layout.positions(0, block.origin) {
        s.push(BlockRecord {
            pred,
            x: x as u16,
            y: y as u16,
            w: side,
            h: side,
            flags: flags::CODED_Y,
            ..block.record(RecordKind::Transform)
        });
    }
}

/// An intra block's records; `layout` its transforms.
pub(crate) fn intra_block(
    s: &mut StreamStats,
    block: PlacedBlock,
    (luma, chroma): (Mode, Mode),
    layout: Layout,
) {
    s.push(BlockRecord {
        pred: Pred::Intra,
        flags: flags::CODED_Y | flags::CODED_CB | flags::CODED_CR,
        bits: block.bits,
        ..block.record(RecordKind::Coding)
    });
    let (mode, class) = intra_mode(luma);
    s.push(BlockRecord {
        pred: Pred::Intra,
        mode,
        mode2: intra_mode(chroma).0,
        class: class as u8,
        ref_idx: [-1; 2],
        ..block.record(RecordKind::Prediction)
    });
    transforms(s, block, Some(layout), Pred::Intra);
}

/// An inter block's records; `layout` its transforms (`None` when skipped).
pub(crate) fn inter_block(
    s: &mut StreamStats,
    block: PlacedBlock,
    motion: &BlockMotion,
    layout: Option<Layout>,
) {
    let (pred, mv, compound) = match *motion {
        BlockMotion::Single { vector, is_new } => {
            let pred = match (is_new, layout.is_some()) {
                (true, _) => Pred::Inter,
                (false, true) => Pred::Merge,
                (false, false) => Pred::Skip,
            };
            (pred, [vector_xy(vector), [0, 0]], false)
        }
        BlockMotion::Compound((first, second)) => {
            (Pred::Skip, [vector_xy(first), vector_xy(second)], true)
        }
    };
    let coded = if layout.is_some() {
        flags::CODED_Y | flags::CODED_CB | flags::CODED_CR
    } else {
        0
    };
    s.push(BlockRecord {
        pred,
        flags: coded,
        bits: block.bits,
        ..block.record(RecordKind::Coding)
    });
    s.push(BlockRecord {
        pred,
        flags: if compound { flags::BI } else { 0 },
        ref_idx: [0, if compound { 0 } else { -1 }],
        mv,
        ..block.record(RecordKind::Prediction)
    });
    transforms(s, block, layout, Pred::Inter);
}

/// A `(row, col)` vector as `[x, y]`.
fn vector_xy((row, col): (i32, i32)) -> [i16; 2] {
    let clamp = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    [clamp(col), clamp(row)]
}
