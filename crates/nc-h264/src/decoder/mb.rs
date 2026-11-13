//! `macroblock_layer()` (§7.3.5) of I and P slices, written once against
//! [`MbSyntax`]: macroblock and sub-macroblock types, intra prediction
//! modes (§8.3.1.1, §8.3.2.1), reference indices and motion vectors
//! (§8.4.1), coded block patterns, `mb_qp_delta` and the residual
//! (§7.3.5.3), dequantised as it is parsed (`transform.rs`).

use nc_encoder::Mv;

use super::mbstate::{
    self, CBP_CHROMA_SHIFT, DC_PRED_MODE, MbKind, MbState, Neighbours, PCM_CBP, PCM_TOTAL_COEFF,
    Partition, SubPartition, partition_blocks,
};
use super::mvpred::MotionCache;
use super::slice::SliceType;
use super::syntax::{Block4x4, Coeffs, MbSyntax};
use super::transform::{Dequantiser, Residual};
use super::{DecodeError, Result};

/// Table 7-11: `mb_type` 0 is `I_NxN`, 1..=24 `Intra_16x16`, 25 `I_PCM`.
const MB_TYPE_I_NXN: u32 = 0;
const MB_TYPE_I_PCM: u32 = 25;
/// `Intra_16x16` `mb_type`s: 1 + mode + 4 * chroma cbp + 12 * (luma coded).
const I16X16_MODES: u32 = 4;
const I16X16_CHROMA_CBPS: u32 = 3;
/// Table 7-13: P `mb_type`s; 4 is `P_8x8ref0`, 5.. are intra.
const MB_TYPE_P_8X8: u32 = 3;
const MB_TYPE_P_8X8_REF0: u32 = 4;
const P_INTRA_BASE: u32 = 5;
/// `QP_Y` wraps modulo 52 (eq. 7-37).
const QP_RANGE: i32 = 52;
/// Samples of an `I_PCM` macroblock: 256 luma, 128 chroma (4:2:0).
pub(super) const PCM_LUMA_SAMPLES: usize = 256;
pub(super) const PCM_CHROMA_SAMPLES: usize = 128;

/// What a slice tells its macroblocks.
#[derive(Debug, Clone, Copy)]
pub(super) struct SliceParams<'a> {
    pub slice_type: SliceType,
    pub num_ref_idx_active: usize,
    /// Identity of each list entry's picture (`None`: no picture).
    pub ref_ids: &'a [Option<u32>],
    pub transform_8x8_mode: bool,
    pub constrained_intra_pred: bool,
    pub monochrome: bool,
    pub dequantiser: &'a Dequantiser,
}

/// Per-macroblock scratch beyond its [`MbState`], reused: the `I_PCM`
/// samples and the levels of the residual block being parsed.
#[derive(Debug)]
pub(super) struct MbExtra {
    pub pcm: [u8; PCM_LUMA_SAMPLES + PCM_CHROMA_SAMPLES],
    pub levels: Coeffs,
    /// `sub_mb_type` of each 8x8 block of the last `P_8x8` macroblock
    /// (read by analysis only for `P_8x8` macroblocks).
    pub sub_partitions: [SubPartition; 4],
}

impl MbExtra {
    pub fn new() -> Self {
        Self {
            pcm: [0; PCM_LUMA_SAMPLES + PCM_CHROMA_SAMPLES],
            levels: Coeffs::new(),
            sub_partitions: [SubPartition::P8x8; 4],
        }
    }
}

/// A `P_Skip` macroblock (§8.4.1.1).
pub(super) fn skip(
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    qp: i32,
) -> Result<()> {
    cur.kind = MbKind::PSkip;
    cur.qp = qp as u8;
    let mv = MotionCache::new(nb).skip_mv();
    cur.mvs = [mv; 16];
    let id = reference_id(params, 0)?;
    cur.ref_idx = [0; 4];
    cur.ref_pic = [id; 4];
    cur.single_motion = true;
    Ok(())
}

/// One coded macroblock; `qp` is `QP_Y`,PRED in and `QP_Y` out.
pub(super) fn decode<S: MbSyntax>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    qp: &mut i32,
    cur: &mut MbState,
    residual: &mut Residual,
    extra: &mut MbExtra,
) -> Result<()> {
    let mut mb_type = syntax.mb_type(params.slice_type, nb)?;
    let mut inter = None;
    if params.slice_type == SliceType::P {
        if mb_type < P_INTRA_BASE {
            inter = Some(mb_type);
        } else {
            mb_type -= P_INTRA_BASE;
        }
    }
    if let Some(p_type) = inter {
        let p8x8 = inter_prediction(syntax, params, nb, cur, p_type, &mut extra.sub_partitions)?;
        return coded_residual(
            syntax,
            params,
            nb,
            qp,
            cur,
            residual,
            &mut extra.levels,
            p8x8,
        );
    }
    cur.ref_idx.fill(-1);
    match mb_type {
        MB_TYPE_I_PCM => {
            cur.kind = MbKind::IPcm;
            cur.qp = *qp as u8;
            cur.cbp = PCM_CBP;
            cur.total_coeff = [PCM_TOTAL_COEFF; 16];
            cur.chroma_total_coeff = [[PCM_TOTAL_COEFF; 4]; 2];
            cur.dc_coded = [true; 3];
            cur.nonzero = u16::MAX;
            let samples = if params.monochrome {
                PCM_LUMA_SAMPLES
            } else {
                PCM_LUMA_SAMPLES + PCM_CHROMA_SAMPLES
            };
            syntax.pcm_samples(&mut extra.pcm[..samples])?;
            syntax.no_qp_delta();
            Ok(())
        }
        MB_TYPE_I_NXN => {
            cur.kind = MbKind::IntraNxN;
            if params.transform_8x8_mode {
                cur.transform_8x8 = syntax.transform_size_8x8_flag(nb)?;
            }
            intra_nxn_modes(syntax, params, nb, cur)?;
            if !params.monochrome {
                cur.chroma_pred_mode = syntax.intra_chroma_pred_mode(nb)?;
            }
            coded_residual(
                syntax,
                params,
                nb,
                qp,
                cur,
                residual,
                &mut extra.levels,
                false,
            )
        }
        _ => {
            // Intra_16x16: prediction mode and coded block patterns in mb_type.
            let index = mb_type - 1;
            cur.kind = MbKind::Intra16x16((index % I16X16_MODES) as u8);
            let chroma = (index / I16X16_MODES) % I16X16_CHROMA_CBPS;
            let luma = if index >= I16X16_MODES * I16X16_CHROMA_CBPS {
                0x0f
            } else {
                0
            };
            cur.cbp = luma | (chroma as u8) << CBP_CHROMA_SHIFT;
            if !params.monochrome {
                cur.chroma_pred_mode = syntax.intra_chroma_pred_mode(nb)?;
            }
            *qp = next_qp(*qp, syntax.mb_qp_delta()?);
            cur.qp = *qp as u8;
            residual_layer(syntax, params, nb, cur, residual, &mut extra.levels)
        }
    }
}

/// `coded_block_pattern`, `transform_size_8x8_flag`, `mb_qp_delta` and the
/// residual of a non-`Intra_16x16` macroblock. `p8x8_small` is whether a
/// `P_8x8` macroblock has sub-partitions below 8x8 (then no 8x8 transform).
fn coded_residual<S: MbSyntax>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    qp: &mut i32,
    cur: &mut MbState,
    residual: &mut Residual,
    levels: &mut Coeffs,
    p8x8_small: bool,
) -> Result<()> {
    cur.cbp = syntax.coded_block_pattern(nb, cur, params.monochrome)?;
    if !cur.is_intra() && cur.cbp_luma() != 0 && params.transform_8x8_mode && !p8x8_small {
        cur.transform_8x8 = syntax.transform_size_8x8_flag(nb)?;
    }
    if cur.cbp == 0 {
        cur.qp = *qp as u8;
        syntax.no_qp_delta();
        return Ok(());
    }
    *qp = next_qp(*qp, syntax.mb_qp_delta()?);
    cur.qp = *qp as u8;
    residual_layer(syntax, params, nb, cur, residual, levels)
}

/// `QP_Y` from `QP_Y`,PRED and `mb_qp_delta` (eq. 7-37).
fn next_qp(pred: i32, delta: i32) -> i32 {
    (pred + delta + QP_RANGE).rem_euclid(QP_RANGE)
}

/// `residual()` (§7.3.5.3) for 4:2:0 / 4:0:0, dequantised into `residual`.
fn residual_layer<S: MbSyntax>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    residual: &mut Residual,
    coeffs: &mut Coeffs,
) -> Result<()> {
    let dq = params.dequantiser;
    let intra = cur.is_intra();
    let qp = i32::from(cur.qp);
    let i16x16 = matches!(cur.kind, MbKind::Intra16x16(_));
    if i16x16 {
        syntax.luma_dc(nb, cur, coeffs)?;
        dq.luma_dc(coeffs, qp, residual);
    }
    for b8 in 0..4 {
        if (cur.cbp_luma() >> b8) & 1 == 0 {
            continue;
        }
        if cur.transform_8x8 {
            syntax.luma_8x8(nb, cur, b8, coeffs)?;
            dq.luma_8x8(coeffs, intra, qp, b8, residual);
            continue;
        }
        for i in 0..4 {
            let (x, y) = ((b8 % 2) * 2 + i % 2, (b8 / 2) * 2 + i / 2);
            let block = if i16x16 {
                Block4x4::LumaAc(x, y)
            } else {
                Block4x4::Luma(x, y)
            };
            syntax.block_4x4(nb, cur, block, coeffs)?;
            dq.luma_4x4(coeffs, i16x16, intra, qp, y * 4 + x, residual);
        }
    }
    cur.nonzero = residual.luma_nonzero();
    if params.monochrome || cur.cbp_chroma() == 0 {
        return Ok(());
    }
    for comp in 0..2 {
        syntax.chroma_dc(nb, cur, comp, coeffs)?;
        dq.chroma_dc(coeffs, intra, qp, comp, residual);
    }
    if cur.cbp_chroma() == 2 {
        for comp in 0..2 {
            for b in 0..4 {
                syntax.block_4x4(nb, cur, Block4x4::ChromaAc(comp, b), coeffs)?;
                dq.chroma_ac(coeffs, intra, qp, comp, b, residual);
            }
        }
    }
    Ok(())
}

/// `prev_intra*_pred_mode_flag` / `rem_intra*_pred_mode` of every 4x4 or
/// 8x8 block, resolved against the predicted modes (§8.3.1.1, §8.3.2.1).
fn intra_nxn_modes<S: MbSyntax>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
) -> Result<()> {
    let step = if cur.transform_8x8 { 4 } else { 1 };
    for index in (0..16).step_by(step) {
        let (x, y) = mbstate::block_position(index);
        let predicted = predicted_intra_mode(params, nb, cur, x, y);
        let mode = match syntax.intra_pred_mode()? {
            None => predicted,
            Some(rem) if rem < predicted => rem,
            Some(rem) => rem + 1,
        };
        if cur.transform_8x8 {
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                cur.intra_modes[(y + dy) * 4 + x + dx] = mode;
            }
        } else {
            cur.intra_modes[y * 4 + x] = mode;
        }
    }
    Ok(())
}

/// `predIntra4x4PredMode` / `predIntra8x8PredMode` of the block whose first
/// 4x4 block is (`x`, `y`): DC when a neighbour is missing (or inter under
/// `constrained_intra_pred_flag`), else the smaller neighbouring mode, with
/// non-`I_NxN` neighbours counting as DC.
fn predicted_intra_mode(
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &MbState,
    x: usize,
    y: usize,
) -> u8 {
    let mode = |n: Option<(&MbState, usize, usize)>| -> Option<u8> {
        let (m, bx, by) = n?;
        if !m.is_intra() && params.constrained_intra_pred {
            return None;
        }
        Some(if m.kind == MbKind::IntraNxN {
            m.intra_modes[by * 4 + bx]
        } else {
            DC_PRED_MODE
        })
    };
    match (mode(nb.left(cur, x, y)), mode(nb.above(cur, x, y))) {
        (Some(a), Some(b)) => a.min(b),
        _ => DC_PRED_MODE,
    }
}

/// The identity of the picture of list entry `ref_idx`. An entry without a
/// picture is "no reference picture" (§8.2.4.2) or a "non-existing" frame
/// of a `frame_num` gap (§8.2.5.2): a conforming stream never predicts from
/// either, so the picture it would have held was lost.
fn reference_id(params: &SliceParams<'_>, ref_idx: usize) -> Result<u32> {
    match params.ref_ids.get(ref_idx) {
        Some(Some(id)) => Ok(*id),
        Some(None) => Err(DecodeError::MissingReference),
        None => Err(DecodeError::Malformed("reference index outside the list")),
    }
}

/// The prediction part of a P macroblock (`mb_pred` / `sub_mb_pred`, §7.3.5.1,
/// §7.3.5.2) and its motion vectors; returns whether it is a `P_8x8`
/// macroblock with a sub-partition smaller than 8x8. A `P_8x8`
/// macroblock's `sub_mb_type`s go to `sub_partitions`.
fn inter_prediction<S: MbSyntax>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    p_type: u32,
    sub_partitions: &mut [SubPartition; 4],
) -> Result<bool> {
    let partition = match p_type {
        0 => Partition::P16x16,
        1 => Partition::P16x8,
        2 => Partition::P8x16,
        _ => Partition::P8x8,
    };
    cur.kind = MbKind::Inter(partition);
    let mut cache = MotionCache::new(nb);
    let refs_coded = params.num_ref_idx_active > 1 && p_type != MB_TYPE_P_8X8_REF0;
    if partition != Partition::P8x8 {
        // Partitions as (x4, y4, w4, h4).
        let parts: &[(usize, usize, usize, usize)] = match partition {
            Partition::P16x16 => &[(0, 0, 4, 4)],
            Partition::P16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
            _ => &[(0, 0, 2, 4), (2, 0, 2, 4)],
        };
        for &(x, y, w, h) in parts {
            let ref_idx = if refs_coded {
                syntax.ref_idx(nb, cur, x, y, params.num_ref_idx_active)?
            } else {
                0
            };
            set_ref(params, cur, x, y, w, h, ref_idx)?;
        }
        for (index, &(x, y, w, h)) in parts.iter().enumerate() {
            let mvd = syntax.mvd(nb, cur, x, y)?;
            let ref_idx = cur.ref_idx[(y / 2) * 2 + x / 2];
            let pred = cache.predict((x, y, w, h), partition, index, ref_idx);
            let mv = set_motion(cur, (x, y, w, h), pred, mvd);
            cache.record((x, y, w, h), mv, ref_idx);
        }
        cur.single_motion = partition == Partition::P16x16 || has_single_motion(cur);
        return Ok(false);
    }
    debug_assert!(p_type == MB_TYPE_P_8X8 || p_type == MB_TYPE_P_8X8_REF0);
    let mut subs = [SubPartition::P8x8; 4];
    for sub in &mut subs {
        *sub = SubPartition::from_code(syntax.sub_mb_type()?)
            .ok_or(DecodeError::Malformed("sub_mb_type"))?;
    }
    for b8 in 0..4 {
        let (x, y) = ((b8 % 2) * 2, (b8 / 2) * 2);
        let ref_idx = if refs_coded {
            syntax.ref_idx(nb, cur, x, y, params.num_ref_idx_active)?
        } else {
            0
        };
        set_ref(params, cur, x, y, 2, 2, ref_idx)?;
    }
    for (b8, sub) in subs.iter().enumerate() {
        let (w, h) = sub.size4();
        for index in 0..sub.count() {
            let x = (b8 % 2) * 2 + (index * w) % 2;
            let y = (b8 / 2) * 2 + (index * w / 2) * h;
            let mvd = syntax.mvd(nb, cur, x, y)?;
            let ref_idx = cur.ref_idx[b8];
            let pred = cache.predict((x, y, w, h), partition, b8, ref_idx);
            let mv = set_motion(cur, (x, y, w, h), pred, mvd);
            cache.record((x, y, w, h), mv, ref_idx);
        }
    }
    cur.single_motion = has_single_motion(cur);
    *sub_partitions = subs;
    Ok(subs.iter().any(|&s| s != SubPartition::P8x8))
}

/// Whether every 4x4 block of `cur` has the same vector and reference
/// index (the same index also means the same weights).
fn has_single_motion(cur: &MbState) -> bool {
    cur.mvs.iter().all(|&mv| mv == cur.mvs[0]) && cur.ref_idx.iter().all(|&r| r == cur.ref_idx[0])
}

/// Record `ref_idx` (and its picture) for the 8x8 blocks of a partition.
fn set_ref(
    params: &SliceParams<'_>,
    cur: &mut MbState,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    ref_idx: u8,
) -> Result<()> {
    let id = reference_id(params, usize::from(ref_idx))?;
    // The 8x8 blocks the partition covers, as a 2x2 grid of bits.
    let covered = partition_blocks(x / 2, y / 2, w.div_ceil(2), h.div_ceil(2), 2);
    for (b8, (slot_idx, slot_pic)) in cur.ref_idx.iter_mut().zip(&mut cur.ref_pic).enumerate() {
        let inside = (covered >> b8) & 1 != 0;
        *slot_idx = core::hint::select_unpredictable(inside, ref_idx as i8, *slot_idx);
        *slot_pic = core::hint::select_unpredictable(inside, id, *slot_pic);
    }
    Ok(())
}

/// `mvL0 = mvpL0 + mvdL0` (eq. 8-174) and the CABAC `absMvdComp` record;
/// returns the vector.
fn set_motion(
    cur: &mut MbState,
    (x, y, w, h): (usize, usize, usize, usize),
    pred: Mv,
    mvd: [i32; 2],
) -> Mv {
    let clamp = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    let mv = Mv {
        x: clamp(i32::from(pred.x) + mvd[0]),
        y: clamp(i32::from(pred.y) + mvd[1]),
    };
    let abs = mvd.map(|v| v.unsigned_abs().min(u32::from(u8::MAX)) as u8);
    // Every block, each kept or replaced by a select: a fixed trip count.
    let covered = partition_blocks(x, y, w, h, 4);
    for (block, (slot_mv, slot_abs)) in cur.mvs.iter_mut().zip(&mut cur.abs_mvd).enumerate() {
        let inside = (covered >> block) & 1 != 0;
        *slot_mv = core::hint::select_unpredictable(inside, mv, *slot_mv);
        *slot_abs = core::hint::select_unpredictable(inside, abs, *slot_abs);
    }
    mv
}
