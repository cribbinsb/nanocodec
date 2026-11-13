//! The slice data parser's analysis records (feature `analysis`): what
//! `ctu.rs` collects at its parse points (§7.3.8) and the block records it
//! pushes into [`StreamStats`]. Without the feature this module and every
//! hook are compiled out, so the decoder contains no analysis code.

use nc_common::stream_stats::{
    BlockRecord, IntraClass, Pred, RecordKind, Shape, StreamStats, flags,
};
use nc_encoder::Mv;

use super::{Cu, Part, SegmentDecoder, TuNode};
use crate::dec::cabac::Engine;
use crate::dec::picture::PuMotion;
use crate::dec::sao::{self, SaoSyntax};
use crate::predict;

/// What the records of the current coding, prediction and transform unit
/// collect while they are parsed.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Analysis {
    /// Bits consumed when the coding unit started.
    pub cu_start: usize,
    /// Coded planes of the coding unit ([`flags`]).
    pub cu_flags: u8,
    /// `CuQpDeltaVal` when coded in this coding unit.
    pub qp_delta: Option<i8>,
    /// The current prediction unit's signalling.
    pub pu: PuSignal,
    /// The current transform unit's residual.
    pub tu: Residual,
}

/// How a prediction unit's motion was coded: the merge index or list-0
/// predictor index, the motion vector differences and their bits.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PuSignal {
    pub index: usize,
    pub mvd: [Mv; 2],
    pub mvd_bits: u32,
}

/// A transform unit's residual: the bits consumed when it started, its
/// non-zero levels and whether any block skipped the transform.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Residual {
    pub start: usize,
    pub nonzero: u16,
    pub transform_skip: bool,
}

impl Part {
    /// The codec-neutral partition shape.
    fn shape(self) -> Shape {
        match self {
            Part::P2Nx2N => Shape::Whole,
            Part::P2NxN => Shape::HalfHorizontal,
            Part::PNx2N => Shape::HalfVertical,
            Part::PNxN => Shape::Quarter,
            Part::P2NxnU => Shape::QuarterTop,
            Part::P2NxnD => Shape::QuarterBottom,
            Part::PnLx2N => Shape::QuarterLeft,
            Part::PnRx2N => Shape::QuarterRight,
        }
    }
}

/// The class of an `IntraPredModeY` (§8.4.2): 0 planar, 1 DC, angular.
fn intra_class(mode: u8) -> IntraClass {
    match mode {
        predict::PLANAR => IntraClass::Planar,
        predict::DC => IntraClass::Dc,
        _ => IntraClass::Angular,
    }
}

/// A block position or size as a record field (pictures are below 64K).
fn coord(v: usize) -> u16 {
    v as u16
}

/// The bits consumed since `start`.
fn bits_since(e: &Engine<'_>, start: usize) -> u32 {
    (e.bits_consumed() - start) as u32
}

impl SegmentDecoder<'_, '_> {
    fn stats(&mut self) -> Option<&mut StreamStats> {
        self.s.stats.as_mut()
    }

    fn record(&mut self, record: BlockRecord) {
        if let Some(s) = self.stats() {
            s.push(record);
        }
    }

    /// A CTB's SAO record (`start`: the bits consumed before `sao()`).
    pub(super) fn sao_record(
        &mut self,
        e: &Engine<'_>,
        rs: usize,
        (x0, y0): (usize, usize),
        syntax: SaoSyntax,
        start: usize,
    ) {
        let sps = self.cx.sps;
        let size = 1usize << self.pic.layout.log2_ctb;
        let ctb = self.pic.sao[rs];
        self.record(BlockRecord {
            kind: RecordKind::Filter,
            x: coord(x0),
            y: coord(y0),
            w: coord(size.min(sps.width - x0)),
            h: coord(size.min(sps.height - y0)),
            mode: sao::kind_index(ctb[0].kind),
            mode2: sao::kind_index(ctb[1].kind),
            class: syntax as u8,
            bits: bits_since(e, start),
            ..BlockRecord::default()
        });
    }

    /// Start a coding unit's record.
    pub(super) fn start_cu(&mut self, e: &Engine<'_>) {
        self.analysis = Analysis {
            cu_start: e.bits_consumed(),
            ..Analysis::default()
        };
    }

    /// The intra prediction blocks' records of a coding unit.
    pub(super) fn intra_records(&mut self, cu: &Cu, parts: usize, pb: usize) {
        for (i, &mode) in cu.luma_modes[..parts].iter().enumerate() {
            self.record(BlockRecord {
                kind: RecordKind::Prediction,
                pred: Pred::Intra,
                x: coord(cu.x + (i & 1) * pb),
                y: coord(cu.y + (i >> 1) * pb),
                w: coord(pb),
                h: coord(pb),
                mode,
                mode2: cu.chroma_mode,
                class: intra_class(mode) as u8,
                ref_idx: [-1; 2],
                ..BlockRecord::default()
            });
        }
    }

    /// A prediction unit's record: how its motion was signalled (skip,
    /// merge, and `self.analysis.pu`) and its vectors.
    pub(super) fn pu_record(
        &mut self,
        (x, y, w, h): (usize, usize, usize, usize),
        (skip, merge): (bool, bool),
        motion: &PuMotion,
    ) {
        let pred = match (skip, merge) {
            (true, _) => Pred::Skip,
            (false, true) => Pred::Merge,
            (false, false) => Pred::Inter,
        };
        let signal = self.analysis.pu;
        let bi = motion.uses(0) && motion.uses(1);
        let vector = |mv: Mv| [mv.x, mv.y];
        self.record(BlockRecord {
            kind: RecordKind::Prediction,
            pred,
            x: coord(x),
            y: coord(y),
            w: coord(w),
            h: coord(h),
            mode: signal.index as u8,
            flags: if bi { flags::BI } else { 0 },
            ref_idx: [0, 1].map(|l| {
                if motion.uses(l) {
                    motion.ref_idx[l]
                } else {
                    -1
                }
            }),
            mv: motion.mv.map(vector),
            mvd: signal.mvd.map(vector),
            bits: signal.mvd_bits,
            ..BlockRecord::default()
        });
        self.analysis.pu = PuSignal::default();
    }

    /// A coding unit's record.
    pub(super) fn cu_record(&mut self, e: &Engine<'_>, cu: &Cu, skip: bool, depth: u8) {
        let size = 1usize << cu.log2;
        let pred = match (cu.intra, skip) {
            (true, _) => Pred::Intra,
            (false, true) => Pred::Skip,
            (false, false) => Pred::Inter,
        };
        let a = self.analysis;
        let qp_flag = if a.qp_delta.is_some() {
            flags::QP_DELTA_CODED
        } else {
            0
        };
        self.record(BlockRecord {
            kind: RecordKind::Coding,
            pred,
            shape: cu.part.shape(),
            depth,
            x: coord(cu.x),
            y: coord(cu.y),
            w: coord(size),
            h: coord(size),
            flags: a.cu_flags | qp_flag,
            qp: self.qp as i16,
            qp_delta: a.qp_delta.unwrap_or(0),
            bits: bits_since(e, a.cu_start),
            ..BlockRecord::default()
        });
    }

    /// Start collecting a transform unit's residual.
    pub(super) fn start_tu(&mut self, e: &Engine<'_>) {
        self.analysis.tu = Residual {
            start: e.bits_consumed(),
            ..Residual::default()
        };
    }

    /// Add a coded block's residual (its levels are in the scratch
    /// positions) to its transform unit's.
    pub(super) fn note_block(&mut self, transform_skip: bool) {
        let tu = &mut self.analysis.tu;
        tu.nonzero = tu.nonzero.saturating_add(self.s.positions.len() as u16);
        tu.transform_skip |= transform_skip;
    }

    /// A transform unit's record. A 4x4 luma block's chroma is coded with
    /// the fourth of its quadrant (§7.3.8.10), so only that one carries the
    /// chroma flags.
    pub(super) fn tu_record(
        &mut self,
        e: &Engine<'_>,
        cu: &Cu,
        node: TuNode,
        (luma_cbf, cbf): (bool, [bool; 2]),
    ) {
        let chroma = node.log2 > 2 || node.blk_idx == 3;
        let mut coded = if luma_cbf { flags::CODED_Y } else { 0 };
        if chroma && cbf[0] {
            coded |= flags::CODED_CB;
        }
        if chroma && cbf[1] {
            coded |= flags::CODED_CR;
        }
        self.analysis.cu_flags |= coded;
        let residual = self.analysis.tu;
        let size = coord(1 << node.log2);
        let skip = if residual.transform_skip {
            flags::TRANSFORM_SKIP
        } else {
            0
        };
        self.record(BlockRecord {
            kind: RecordKind::Transform,
            pred: if cu.intra { Pred::Intra } else { Pred::Inter },
            depth: node.depth,
            x: coord(node.x),
            y: coord(node.y),
            w: size,
            h: size,
            flags: coded | skip,
            qp: self.qp as i16,
            nonzero: residual.nonzero,
            bits: bits_since(e, residual.start),
            ..BlockRecord::default()
        });
    }
}
