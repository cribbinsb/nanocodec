//! What the H.264 decoder records into [`StreamStats`] when analysis is
//! enabled: picture-level facts from the parameter sets and slice headers
//! (§7.3.2, §7.3.3) and, per macroblock, the codec-neutral block records
//! built from its decoded [`MbState`] (§7.3.5): one coding record, its
//! prediction blocks (intra modes or partitions with motion) and its luma
//! transform blocks. Nothing here affects decoding.
//!
//! H.264 limits: motion vector differences are recorded as magnitudes
//! (saturated at 255 quarter samples, as kept for CABAC context
//! selection); chroma coded flags are per macroblock
//! (`CodedBlockPatternChroma`); bits are per macroblock only; a transform
//! block's non-zero count is `TotalCoeff` under CAVLC and not kept (0)
//! under CABAC.

use nc_common::stream_stats::{
    BlockRecord, IntraClass, Pred, RecordKind, Shape, StreamStats, facts, flags,
};

use super::mbstate::{DC_PRED_MODE, MbKind, MbState, Partition};
use super::pps::Pps;
use super::slice::{SliceHeader, SliceType};
use super::sps::Sps;

/// Luma samples per macroblock side, 4x4 block side and 8x8 block side.
const MB_SIZE: u16 = 16;
const B4: u16 = 4;
const B8: u16 = 8;
/// Quarter-sample motion vector units per sample (§8.4.2.2).
const MV_UNITS_PER_SAMPLE: i64 = 4;
/// `QP_Y` range: deltas wrap modulo 52 (eq. 7-37).
const QP_RANGE: i32 = 52;
/// `Intra16x16PredMode` 2 is DC, 3 plane (Table 8-4).
const I16X16_DC: u8 = 2;
const I16X16_PLANE: u8 = 3;

/// Record the facts of a new picture's parameter sets.
pub(super) fn picture(s: &mut StreamStats, sps: &Sps, pps: &Pps) {
    s.begin_frame();
    s.fact(facts::MV_UNITS_PER_SAMPLE, MV_UNITS_PER_SAMPLE, 0);
    let fields = [
        ("sps_profile_idc", i64::from(sps.profile_idc)),
        ("sps_max_num_ref_frames", sps.max_num_ref_frames as i64),
        ("pps_cabac", i64::from(pps.cabac)),
        ("pps_transform_8x8_mode", i64::from(pps.transform_8x8_mode)),
        ("pps_weighted_pred", i64::from(pps.weighted_pred)),
        (
            "pps_constrained_intra_pred",
            i64::from(pps.constrained_intra_pred),
        ),
        ("pps_chroma_qp_offset", i64::from(pps.chroma_qp_offset[0])),
        (
            "pps_scaling_lists",
            i64::from(pps.scaling.is_some() || sps.scaling_present),
        ),
    ];
    for (what, value) in fields {
        s.fact(what, value, 0);
    }
}

/// Record a slice header and its size.
pub(super) fn slice(s: &mut StreamStats, h: &SliceHeader, bytes: usize) {
    let slice_type = match h.slice_type {
        SliceType::P => 0,
        SliceType::I => 2,
    };
    let fields = [
        ("slice_type", slice_type),
        ("slice_qp", i64::from(h.qp)),
        ("slice_num_ref_l0", h.num_ref_idx_active as i64),
        (
            "slice_deblock_disable_idc",
            i64::from(h.deblock.disable_idc),
        ),
        (
            "slice_alpha_offset_div2",
            i64::from(h.deblock.alpha_offset_div2),
        ),
        (
            "slice_beta_offset_div2",
            i64::from(h.deblock.beta_offset_div2),
        ),
        ("slice_weighted", i64::from(h.weights.is_some())),
    ];
    for (what, value) in fields {
        s.fact(what, value, 0);
    }
    s.fact("slice_bytes", 0, bytes as u64);
}

/// One decoded macroblock: its position in macroblocks, `QP_Y,PRED`, the
/// bits it cost and whether the slice is CABAC-coded.
pub(super) struct MbContext {
    pub position: (usize, usize),
    pub qp_pred: i32,
    pub bits: usize,
    pub cabac: bool,
}

/// The records of one decoded macroblock. Cold and out of line: the
/// macroblock loop keeps only the call and its test when analysis is off.
#[cold]
#[inline(never)]
pub(super) fn macroblock(s: &mut StreamStats, cur: &MbState, mb: &MbContext) {
    let (mbx, mby) = mb.position;
    let (qp_pred, bits) = (mb.qp_pred, mb.bits);
    let (x0, y0) = (mbx as u16 * MB_SIZE, mby as u16 * MB_SIZE);
    let (pred, shape) = match cur.kind {
        MbKind::IntraNxN if cur.transform_8x8 => (Pred::Intra, Shape::Quarter),
        MbKind::IntraNxN => (Pred::Intra, Shape::Sixteenth),
        MbKind::Intra16x16(_) => (Pred::Intra, Shape::Whole),
        MbKind::IPcm => (Pred::Intra, Shape::Raw),
        MbKind::Inter(p) => (Pred::Inter, partition_shape(p)),
        MbKind::PSkip => (Pred::Skip, Shape::Whole),
    };
    let i16x16 = matches!(cur.kind, MbKind::Intra16x16(_));
    // mb_qp_delta is present in Intra_16x16 and in coded macroblocks
    // (§7.3.5).
    let delta_coded = !matches!(cur.kind, MbKind::IPcm | MbKind::PSkip) && (i16x16 || cur.cbp != 0);
    let qp_delta = (i32::from(cur.qp) - qp_pred + QP_RANGE / 2).rem_euclid(QP_RANGE) - QP_RANGE / 2;
    let mut coded = 0;
    if cur.cbp_luma() != 0 || i16x16 {
        coded |= flags::CODED_Y;
    }
    if cur.cbp_chroma() != 0 {
        coded |= flags::CODED_CB | flags::CODED_CR;
    }
    s.push(BlockRecord {
        kind: RecordKind::Coding,
        pred,
        shape,
        x: x0,
        y: y0,
        w: MB_SIZE,
        h: MB_SIZE,
        flags: coded
            | if delta_coded {
                flags::QP_DELTA_CODED
            } else {
                0
            },
        qp: i16::from(cur.qp),
        qp_delta: if delta_coded { qp_delta as i8 } else { 0 },
        bits: bits as u32,
        ..BlockRecord::default()
    });
    match cur.kind {
        MbKind::IntraNxN | MbKind::Intra16x16(_) => intra(s, cur, (x0, y0)),
        MbKind::Inter(_) | MbKind::PSkip => inter(s, cur, (x0, y0), pred),
        MbKind::IPcm => {}
    }
    if !matches!(cur.kind, MbKind::IPcm | MbKind::PSkip) {
        transforms(s, cur, (x0, y0), pred, mb.cabac);
    }
}

fn partition_shape(p: Partition) -> Shape {
    match p {
        Partition::P16x16 => Shape::Whole,
        Partition::P16x8 => Shape::HalfHorizontal,
        Partition::P8x16 => Shape::HalfVertical,
        Partition::P8x8 => Shape::Quarter,
    }
}

/// Intra prediction blocks: 16 4x4, 4 8x8 or one 16x16 (§8.3).
fn intra(s: &mut StreamStats, cur: &MbState, (x0, y0): (u16, u16)) {
    let block = |x, y, size, mode, class: IntraClass| BlockRecord {
        kind: RecordKind::Prediction,
        pred: Pred::Intra,
        x: x0 + x,
        y: y0 + y,
        w: size,
        h: size,
        mode,
        mode2: cur.chroma_pred_mode,
        class: class as u8,
        ref_idx: [-1; 2],
        ..BlockRecord::default()
    };
    // Intra 4x4 / 8x8 modes: 2 is DC, the others directional (Table 8-2).
    let nxn_class = |mode| {
        if mode == DC_PRED_MODE {
            IntraClass::Dc
        } else {
            IntraClass::Angular
        }
    };
    match cur.kind {
        MbKind::Intra16x16(mode) => {
            let class = match mode {
                I16X16_DC => IntraClass::Dc,
                I16X16_PLANE => IntraClass::Planar,
                _ => IntraClass::Angular,
            };
            s.push(block(0, 0, MB_SIZE, mode, class));
        }
        _ if cur.transform_8x8 => {
            for b8 in 0..4u16 {
                let (x, y) = ((b8 % 2) * B8, (b8 / 2) * B8);
                let mode = cur.intra_modes[usize::from((y / B4) * 4 + x / B4)];
                s.push(block(x, y, B8, mode, nxn_class(mode)));
            }
        }
        _ => {
            for (index, &mode) in cur.intra_modes.iter().enumerate() {
                let (x, y) = ((index % 4) as u16 * B4, (index / 4) as u16 * B4);
                s.push(block(x, y, B4, mode, nxn_class(mode)));
            }
        }
    }
}

/// Inter prediction blocks: the partitions, or sub-partitions of `P_8x8`,
/// with their vectors (§7.3.5.1, §7.3.5.2).
fn inter(stats: &mut StreamStats, cur: &MbState, (x0, y0): (u16, u16), pred: Pred) {
    // Partitions as (x4, y4, w4, h4).
    let mut parts = [(0usize, 0usize, 0usize, 0usize); 16];
    let count = match cur.kind {
        MbKind::Inter(Partition::P16x8) => {
            parts[..2].copy_from_slice(&[(0, 0, 4, 2), (0, 2, 4, 2)]);
            2
        }
        MbKind::Inter(Partition::P8x16) => {
            parts[..2].copy_from_slice(&[(0, 0, 2, 4), (2, 0, 2, 4)]);
            2
        }
        MbKind::Inter(Partition::P8x8) => {
            let mut filled = 0;
            for (b8, sub) in cur.sub_partitions.iter().enumerate() {
                let (w, h) = sub.size4();
                for index in 0..sub.count() {
                    let x = (b8 % 2) * 2 + (index * w) % 2;
                    let y = (b8 / 2) * 2 + (index * w / 2) * h;
                    parts[filled] = (x, y, w, h);
                    filled += 1;
                }
            }
            filled
        }
        _ => {
            parts[0] = (0, 0, 4, 4);
            1
        }
    };
    for &(x, y, w, h) in &parts[..count] {
        let block = y * 4 + x;
        let mv = cur.mvs[block];
        let mvd = cur.abs_mvd[block].map(i16::from);
        stats.push(BlockRecord {
            kind: RecordKind::Prediction,
            pred,
            x: x0 + x as u16 * B4,
            y: y0 + y as u16 * B4,
            w: w as u16 * B4,
            h: h as u16 * B4,
            ref_idx: [cur.ref_idx[(y / 2) * 2 + x / 2], -1],
            mv: [[mv.x, mv.y], [0, 0]],
            mvd: [mvd, [0, 0]],
            ..BlockRecord::default()
        });
    }
}

/// Luma transform blocks: 8x8 under `transform_size_8x8_flag`, else 4x4
/// (`Intra_16x16` adds its DC transform over the 4x4 blocks, §8.5.2).
fn transforms(s: &mut StreamStats, cur: &MbState, (x0, y0): (u16, u16), pred: Pred, cabac: bool) {
    // Under CABAC `total_coeff` holds coded-block flags, not counts.
    let count = |c: u8| if cabac { 0 } else { u16::from(c) };
    let pred = if pred == Pred::Intra {
        Pred::Intra
    } else {
        Pred::Inter
    };
    let block = |x, y, size, coded: bool, nonzero| BlockRecord {
        kind: RecordKind::Transform,
        pred,
        x: x0 + x,
        y: y0 + y,
        w: size,
        h: size,
        flags: if coded { flags::CODED_Y } else { 0 },
        qp: i16::from(cur.qp),
        nonzero,
        ..BlockRecord::default()
    };
    if cur.transform_8x8 {
        for b8 in 0..4u16 {
            let (x, y) = ((b8 % 2) * B8, (b8 / 2) * B8);
            let coded = cur.cbp_luma() >> b8 & 1 != 0;
            let nonzero = [0, 1, 4, 5]
                .iter()
                .map(|&i| count(cur.total_coeff[usize::from((y / B4) * 4 + x / B4) + i]))
                .sum();
            s.push(block(x, y, B8, coded, nonzero));
        }
    } else {
        for index in 0..16u16 {
            let (x, y) = ((index % 4) * B4, (index / 4) * B4);
            let coded = cur.nonzero >> index & 1 != 0;
            let nonzero = count(cur.total_coeff[usize::from(index)]);
            s.push(block(x, y, B4, coded, nonzero));
        }
    }
}
