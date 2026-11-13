//! The decoder's analysis records (feature `analysis`) for pictures of our
//! own streams (the canonical path, `decoder.rs` `decode_slice`): per coding
//! unit (H.266 §7.3.11.5) one coding record, one prediction record and one
//! transform record per transform unit, pushed into [`StreamStats`]. The
//! general path's units are recorded by `decoder_blocks::analysis`. Without
//! the feature these modules and every hook are compiled out, so the decoder
//! contains no analysis code.
//!
//! VVC in the codec-neutral records: `qp` is `QpY`; vectors are in 1/16
//! sample (an affine CU records its first control point); merge and
//! sub-block merge CUs are `Merge` (`Skip` without residual), AMVP and
//! affine AMVP CUs `Inter`, and `mode` holds the merge index or the
//! predictor index; intra `mode` is `IntraPredModeY` (0 planar, 1 DC,
//! angular 2..=66).

use nc_common::stream_stats::{
    BlockRecord, IntraClass, Pred, RecordKind, StreamStats, facts, flags,
};
use nc_encoder::Block;

use crate::intra::{DC, PLANAR};
use crate::motion::{Motion, Mv16};
use crate::syntax::{Coding, InterSyntax};
use crate::transform_tree::CodedUnits;

/// VVC vectors are in 1/16 sample.
const MV_UNITS_PER_SAMPLE: i64 = 16;
/// `slice_type` facts: intra and predicted pictures.
const KEY: i64 = 2;
const INTER: i64 = 0;

/// Start a picture's records with its facts: `slice_qp` is `SliceQpY`,
/// `slice_bytes` the slice data's.
pub(crate) fn picture_facts(s: &mut StreamStats, inter: bool, qp: u32, bytes: usize) {
    s.begin_frame();
    s.fact(facts::MV_UNITS_PER_SAMPLE, MV_UNITS_PER_SAMPLE, 0);
    s.fact("slice_type", if inter { INTER } else { KEY }, 0);
    s.fact("slice_qp", i64::from(qp), 0);
    s.fact("slice_bytes", 0, bytes as u64);
}

/// The neutral class of an `IntraPredModeY`.
pub(crate) fn intra_class(mode: u8) -> IntraClass {
    match mode {
        PLANAR => IntraClass::Planar,
        DC => IntraClass::Dc,
        _ => IntraClass::Angular,
    }
}

/// A vector as `[x, y]`, saturated to the record's range.
pub(crate) fn vector(mv: Mv16) -> [i16; 2] {
    let clamp = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    [clamp(mv.x), clamp(mv.y)]
}

/// A record of `kind` placed on `block`.
fn placed(kind: RecordKind, block: &Block, qp: u32) -> BlockRecord {
    BlockRecord {
        kind,
        x: block.x as u16,
        y: block.y as u16,
        w: 1 << block.log2_width,
        h: 1 << block.log2_height,
        qp: qp as i16,
        ..BlockRecord::default()
    }
}

/// The records of one coding unit of our own streams: `coding` its syntax,
/// `qp` its `QpY`, `units` its transform units.
pub(crate) fn coding_unit(
    s: &mut StreamStats,
    block: &Block,
    coding: &Coding,
    qp: u32,
    units: &CodedUnits,
) {
    let coded = units.any();
    let (pred, mode, mv) = match &coding.inter {
        None => (Pred::Intra, coding.intra_mode, [0, 0]),
        Some(inter) => {
            let mv = match inter.motion {
                Motion::Vector(mv) => vector(mv),
                Motion::Affine(model) => vector(model.cp[0]),
            };
            let (explicit, index) = match inter.syntax {
                InterSyntax::Merge(index) | InterSyntax::SubblockMerge(index) => (false, index),
                InterSyntax::AffineAmvp { mvp, .. } | InterSyntax::Amvp { mvp, .. } => (true, mvp),
            };
            let pred = match (explicit, coded) {
                (true, _) => Pred::Inter,
                (false, true) => Pred::Merge,
                (false, false) => Pred::Skip,
            };
            (pred, index as u8, mv)
        }
    };
    let mut coded_flags = 0;
    for (_, planes) in units.as_slice() {
        for (plane, &plane_coded) in planes.iter().enumerate() {
            if plane_coded {
                coded_flags |= 1 << plane;
            }
        }
    }
    if coding.qp_delta.is_some() {
        coded_flags |= flags::QP_DELTA_CODED;
    }
    s.push(BlockRecord {
        pred,
        flags: coded_flags,
        qp_delta: coding.qp_delta.unwrap_or(0) as i8,
        ..placed(RecordKind::Coding, block, qp)
    });
    let intra = pred == Pred::Intra;
    s.push(BlockRecord {
        pred,
        mode,
        mode2: if intra { coding.chroma_mode } else { 0 },
        class: if intra { intra_class(mode) as u8 } else { 0 },
        ref_idx: [if intra { -1 } else { 0 }, -1],
        mv: [mv, [0, 0]],
        ..placed(RecordKind::Prediction, block, qp)
    });
    let kind = if intra { Pred::Intra } else { Pred::Inter };
    for (unit, planes) in units.as_slice() {
        s.push(BlockRecord {
            pred: kind,
            flags: u8::from(planes[0]) * flags::CODED_Y,
            ..placed(RecordKind::Transform, unit, qp)
        });
    }
}
