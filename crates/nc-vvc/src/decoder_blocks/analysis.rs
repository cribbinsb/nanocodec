//! Analysis records (feature `analysis`) of a general-path picture: its
//! parsed coding units (H.266 §7.3.11.5, after motion derivation), as
//! `crate::analysis` records our own streams' CUs. One coding and one
//! prediction record per luma (or single-tree) CU and one transform record
//! per luma transform block; chroma-tree CUs of dual-tree pictures add none.
//! QP is the slice's (`SliceQpY`): per-CU QP deltas are not resolved here.

use nc_common::stream_stats::{BlockRecord, Pred, RecordKind, StreamStats, flags};

use super::{Unit, inter_syntax::Inter};
use crate::analysis::{intra_class, picture_facts, vector};
use crate::decoder_headers::SliceHeader;

/// `slice_type` of an I slice (§7.4.8.1).
const I_SLICE: u8 = 2;
/// The luma (or single) coding tree's channel.
const LUMA_TREE: usize = 0;

/// Record a general-path picture: its facts and every CU's records.
pub(crate) fn record(s: &mut StreamStats, units: &[Unit], header: &SliceHeader, bytes: usize) {
    let qp = header.qp.max(0) as u32;
    picture_facts(s, header.kind != I_SLICE, qp, bytes);
    for unit in units.iter().filter(|u| u.channel == LUMA_TREE) {
        coding_unit(s, unit, qp);
    }
    s.finish();
}

fn coding_unit(s: &mut StreamStats, unit: &Unit, qp: u32) {
    let node = unit.node;
    let placed = BlockRecord {
        x: node.x as u16,
        y: node.y as u16,
        w: node.width as u16,
        h: node.height as u16,
        qp: qp as i16,
        ..BlockRecord::default()
    };
    let coded = unit
        .transforms
        .iter()
        .flat_map(|t| &t.levels)
        .any(|levels| levels.iter().any(|&v| v != 0));
    let (pred, mode) = match unit.inter {
        None => (Pred::Intra, unit.luma_mode),
        Some(Inter::Amvp { lists, .. }) => (
            Pred::Inter,
            lists.iter().flatten().next().map_or(0, |p| u8::from(p.mvp)),
        ),
        Some(
            Inter::Merge(index)
            | Inter::Subblock(index)
            | Inter::Ciip(index)
            | Inter::Mmvd { base: index, .. },
        ) => (if unit.skip { Pred::Skip } else { Pred::Merge }, index),
        Some(Inter::Geometric { indices, .. }) => {
            (if unit.skip { Pred::Skip } else { Pred::Merge }, indices[0])
        }
    };
    s.push(BlockRecord {
        kind: RecordKind::Coding,
        pred,
        depth: node.qt_depth + node.mtt_depth,
        flags: if coded { flags::CODED_Y } else { 0 },
        ..placed
    });
    let motion = unit.motion.unwrap_or_default();
    let lists = motion
        .lists
        .map(|list| list.map(|v| (v.reference as i8, vector(v.vector))));
    let intra = pred == Pred::Intra;
    s.push(BlockRecord {
        kind: RecordKind::Prediction,
        pred,
        mode,
        mode2: if intra { unit.chroma_mode } else { 0 },
        class: if intra { intra_class(mode) as u8 } else { 0 },
        flags: if lists.iter().all(Option::is_some) {
            flags::BI
        } else {
            0
        },
        ref_idx: lists.map(|l| l.map_or(-1, |(r, _)| r)),
        mv: lists.map(|l| l.map_or([0, 0], |(_, mv)| mv)),
        ..placed
    });
    let kind = if intra { Pred::Intra } else { Pred::Inter };
    for transform in &unit.transforms {
        let block = transform.block;
        let luma_coded = transform.levels[0].iter().any(|&v| v != 0);
        s.push(BlockRecord {
            kind: RecordKind::Transform,
            pred: kind,
            x: block.x as u16,
            y: block.y as u16,
            w: 1 << block.log2_width,
            h: 1 << block.log2_height,
            flags: if luma_coded { flags::CODED_Y } else { 0 },
            ..placed
        });
    }
}
