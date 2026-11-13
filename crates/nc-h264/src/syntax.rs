//! Macroblock-layer syntax with CABAC (H.264 §7.3.4, §7.3.5) for the
//! macroblock types coded so far: `I_16x16` with any prediction mode and
//! `I_NxN` with `Intra_4x4` or `Intra_8x8` prediction (in I and P slices), `P_L0_16x16`,
//! `P_L0_L0_16x8`, `P_L0_L0_8x16` and `P_8x8` with `P_L0_8x8`
//! sub-macroblocks (4x4 or 8x8 transform), and `P_Skip`; binarizations
//! §9.3.2, context indices §9.3.3.1.
//!
//! Every function also returns, or fills in, the [`MbInfo`] later
//! macroblocks derive their contexts from.

use nc_dsp::Dsp;
use nc_encoder::Mv;

use crate::cabac::{BinSink, MbPart, SliceKind};
use crate::intra4x4::block_position;
use crate::intra8x8::modes_per_4x4;
use crate::macroblock::{
    CBP_CHROMA_AC, CBP_CHROMA_NONE, IntraModes, Residual, ResidualKind, blocks_of_8x8,
};
use crate::mbinfo::{MbInfo, MbKind, MbPartition, Neighbours};
use crate::mbqp::{QpTracker, delta_bin_ctx, mapped_delta};
use crate::mvpred::predict_mv;
use crate::predict::ChromaMode;
use crate::residual::{
    BlockCat, SCAN_4X4, SCAN_4X4_AC, SCAN_CHROMA_DC, write_luma_8x8_block, write_residual_block,
};

/// ctxIdxOffset values (Table 9-34).
const MB_TYPE_I_CTX: usize = 3; // mb_type, I slice
const MB_SKIP_P_CTX: usize = 11; // mb_skip_flag, P slice
const MB_TYPE_P_PREFIX_CTX: usize = 14; // mb_type prefix, P slice
const MB_TYPE_P_SUFFIX_CTX: usize = 17; // mb_type suffix (intra in P)
const SUB_MB_TYPE_P_CTX: usize = 21; // sub_mb_type, P slice
const REF_IDX_CTX: usize = 54; // ref_idx_l0
const MVD_CTX: [usize; 2] = [40, 47]; // mvd_l0[][][0], [1]
const INTRA_CHROMA_PRED_MODE_CTX: usize = 64;
const CBP_LUMA_CTX: usize = 73;
const CBP_CHROMA_CTX: usize = 77;
const TRANSFORM_SIZE_8X8_FLAG_CTX: usize = 399;
const PREV_INTRA_PRED_MODE_CTX: usize = 68;
const REM_INTRA_PRED_MODE_CTX: usize = 69;

/// `ctxIdxInc` of `ref_idx_l0` bin 1 and of the bins after it (Table 9-39).
const REF_IDX_BIN1_INC: usize = 4;
const REF_IDX_LATER_INC: usize = 5;

/// mvd binarization UEG3 (§9.3.2.3): `uCoff` and the Exp-Golomb order.
const MVD_PREFIX_MAX: u32 = 9;
const MVD_SUFFIX_ORDER: u32 = 3;

/// A macroblock ready to be written.
#[derive(Debug)]
pub enum CodedMb<'a> {
    /// `I_16x16`: its prediction modes and residual.
    Intra16x16 {
        /// Luma and chroma prediction modes.
        modes: IntraModes,
        /// The residual.
        residual: &'a Residual,
    },
    /// `I_NxN` with `Intra_4x4` prediction.
    Intra4x4 {
        /// `Intra4x4PredMode` per block, raster.
        luma_modes: [u8; 16],
        /// The chroma prediction mode.
        chroma: ChromaMode,
        /// The residual (`Intra4x4` kind).
        residual: &'a Residual,
    },
    /// `I_NxN` with `Intra_8x8` prediction (`transform_size_8x8_flag`).
    Intra8x8 {
        /// `Intra8x8PredMode` per 8x8 block, raster.
        luma_modes: [u8; 4],
        /// The chroma prediction mode.
        chroma: ChromaMode,
        /// The residual (`Intra8x8` kind).
        residual: &'a Residual,
    },
    /// An inter macroblock: its partitioning, the vector of each 8x8
    /// quadrant (equal within a partition), the reference index of every
    /// partition and the residual.
    Inter {
        /// The motion partitioning.
        partition: MbPartition,
        /// Vector of each 8x8 quadrant, raster.
        mvs: [Mv; 4],
        /// `ref_idx_l0` of every partition, or `None` when the slice has
        /// one active reference (then index 0, not coded).
        ref_idx: Option<u8>,
        /// The predicted vector of a `P_L0_16x16` macroblock when the caller
        /// already derived it (mode decision does), so it is not derived
        /// again (§8.4.1.3); `None` derives it.
        mvp: Option<Mv>,
        /// The residual.
        residual: &'a Residual,
    },
    /// `P_Skip` with its (implied) vector.
    Skip(Mv),
}

/// Write one macroblock (`mb_skip_flag` in P slices, then
/// `macroblock_layer()` unless skipped) and return its [`MbInfo`]. The
/// residual was quantized at `mb_qp`; a macroblock that carries
/// `mb_qp_delta` signals it, any other keeps `qp`'s prediction.
pub fn write_macroblock(
    cabac: &mut impl BinSink,
    dsp: &Dsp,
    slice: SliceKind,
    mb: &CodedMb<'_>,
    nb: &Neighbours<'_>,
    qp: &mut QpTracker,
    mb_qp: u8,
) -> MbInfo {
    let mut info = write_macroblock_layer(cabac, dsp, slice, mb, nb, qp, mb_qp);
    if !carries_qp_delta(&info) {
        info.qp = qp.uncoded();
    }
    info
}

/// Whether a written macroblock coded `mb_qp_delta` (§7.3.5: `I_16x16`, or
/// any other non-skipped macroblock with coded residual).
pub(crate) fn carries_qp_delta(info: &MbInfo) -> bool {
    match info.kind {
        MbKind::Skip => false,
        MbKind::Intra16x16 => true,
        _ => info.cbp_luma != 0 || info.cbp_chroma != CBP_CHROMA_NONE,
    }
}

/// `mb_qp_delta` taking the macroblock to `mb_qp` (§7.3.5): unary bins of
/// the mapped value (§9.3.2.7). Returns the macroblock's `QP_Y`.
fn write_mb_qp_delta(cabac: &mut impl BinSink, qp: &mut QpTracker, mb_qp: u8) -> u8 {
    cabac.begin_part(MbPart::ResidualShared);
    let delta = qp.delta_to(mb_qp);
    let mapped = mapped_delta(delta);
    let first_inc = qp.delta_ctx_inc();
    for bin in 0..=mapped {
        cabac.encode_decision(delta_bin_ctx(bin, first_inc), bin < mapped);
    }
    qp.coded(delta)
}

/// [`write_macroblock`] without the `QP_Y` of macroblocks lacking `mb_qp_delta`.
fn write_macroblock_layer(
    cabac: &mut impl BinSink,
    dsp: &Dsp,
    slice: SliceKind,
    mb: &CodedMb<'_>,
    nb: &Neighbours<'_>,
    qp: &mut QpTracker,
    mb_qp: u8,
) -> MbInfo {
    cabac.begin_part(MbPart::Header);
    if slice == SliceKind::P {
        // mb_skip_flag: ctxIdxInc counts available, non-skipped neighbours.
        let cond = |n: Option<&MbInfo>| usize::from(n.is_some_and(|m| !m.is_skip()));
        cabac.encode_decision(
            MB_SKIP_P_CTX + cond(nb.left) + cond(nb.above),
            matches!(mb, CodedMb::Skip(_)),
        );
    }
    match *mb {
        CodedMb::Skip(mv) => skip_info(mv),
        CodedMb::Intra4x4 {
            luma_modes,
            chroma,
            residual: res,
        } => {
            let header = IntraNxnHeader {
                transform_8x8: false,
                chroma,
            };
            write_intra_nxn(cabac, dsp, slice, nb, (qp, mb_qp), header, res, |cabac| {
                write_intra_4x4_modes(cabac, &luma_modes, nb);
                luma_modes
            })
        }
        CodedMb::Intra8x8 {
            luma_modes,
            chroma,
            residual: res,
        } => {
            let header = IntraNxnHeader {
                transform_8x8: true,
                chroma,
            };
            write_intra_nxn(cabac, dsp, slice, nb, (qp, mb_qp), header, res, |cabac| {
                write_intra_8x8_modes(cabac, luma_modes, nb)
            })
        }
        CodedMb::Intra16x16 {
            modes,
            residual: res,
        } => {
            write_mb_type_intra16(cabac, slice, res, modes.luma as u32, nb);
            write_intra_chroma_pred_mode(cabac, modes.chroma, nb);
            let mut info = new_info(MbKind::Intra16x16, [Mv::ZERO; 4], [[0; 2]; 4], res);
            info.qp = write_mb_qp_delta(cabac, qp, mb_qp);
            info.chroma_pred_mode = modes.chroma;
            write_residual(cabac, dsp, res, nb, &mut info);
            info
        }
        CodedMb::Inter {
            partition,
            mvs,
            ref_idx,
            mvp,
            residual,
        } => {
            write_mb_type_p(cabac, partition);
            if let Some(ref_idx) = ref_idx {
                write_ref_indices(cabac, partition, ref_idx, nb);
            }
            let ref_idx = ref_idx.unwrap_or(0);
            let abs_mvd = write_motion(cabac, partition, &mvs, (ref_idx, mvp), nb);
            let mut info = new_info(MbKind::Inter, mvs, abs_mvd, residual);
            info.ref_idx = ref_idx;
            write_coded_block_pattern(cabac, residual, nb);
            if residual.cbp_luma != 0 {
                // transform_size_8x8_flag (§7.3.5): present for inter
                // macroblocks with luma residual and no sub-8x8 partitions;
                // ctxIdxInc counts neighbours using it (§9.3.3.1.1.10).
                info.transform_8x8 = residual.kind == ResidualKind::Inter8x8;
                cabac.begin_part(MbPart::ResidualShared);
                let cond = |n: Option<&MbInfo>| usize::from(n.is_some_and(|m| m.transform_8x8));
                cabac.encode_decision(
                    TRANSFORM_SIZE_8X8_FLAG_CTX + cond(nb.left) + cond(nb.above),
                    info.transform_8x8,
                );
            }
            if residual.cbp_luma != 0 || residual.cbp_chroma != CBP_CHROMA_NONE {
                info.qp = write_mb_qp_delta(cabac, qp, mb_qp);
                write_residual(cabac, dsp, residual, nb, &mut info);
            }
            info
        }
    }
}

/// What the `macroblock_layer()` of an `I_NxN` macroblock codes besides its
/// luma modes and residual.
#[derive(Debug, Clone, Copy)]
struct IntraNxnHeader {
    /// `transform_size_8x8_flag`: `Intra_8x8` rather than `Intra_4x4`.
    transform_8x8: bool,
    chroma: ChromaMode,
}

/// An `I_NxN` macroblock (§7.3.5): `mb_type`, `transform_size_8x8_flag`,
/// the luma modes (`write_modes`, which returns the mode of every 4x4
/// block), `intra_chroma_pred_mode`, `coded_block_pattern`, and with coded
/// residual `mb_qp_delta` and the residual.
#[allow(clippy::too_many_arguments)]
fn write_intra_nxn<S: BinSink>(
    cabac: &mut S,
    dsp: &Dsp,
    slice: SliceKind,
    nb: &Neighbours<'_>,
    (qp, mb_qp): (&mut QpTracker, u8),
    header: IntraNxnHeader,
    res: &Residual,
    write_modes: impl FnOnce(&mut S) -> [u8; 16],
) -> MbInfo {
    write_mb_type_intra_nxn(cabac, slice, nb);
    // transform_size_8x8_flag: ctxIdxInc counts neighbours using the 8x8
    // transform (§9.3.3.1.1.10).
    let cond = |n: Option<&MbInfo>| usize::from(n.is_some_and(|m| m.transform_8x8));
    cabac.encode_decision(
        TRANSFORM_SIZE_8X8_FLAG_CTX + cond(nb.left) + cond(nb.above),
        header.transform_8x8,
    );
    let intra_modes = write_modes(cabac);
    write_intra_chroma_pred_mode(cabac, header.chroma, nb);
    let kind = if header.transform_8x8 {
        MbKind::Intra8x8
    } else {
        MbKind::Intra4x4
    };
    let mut info = new_info(kind, [Mv::ZERO; 4], [[0; 2]; 4], res);
    info.chroma_pred_mode = header.chroma;
    info.intra_modes = intra_modes;
    info.transform_8x8 = header.transform_8x8;
    write_coded_block_pattern(cabac, res, nb);
    if res.cbp_luma != 0 || res.cbp_chroma != CBP_CHROMA_NONE {
        info.qp = write_mb_qp_delta(cabac, qp, mb_qp);
        write_residual(cabac, dsp, res, nb, &mut info);
    }
    info
}

/// The [`MbInfo`] of a `P_Skip` macroblock with vector `mv`.
pub(crate) fn skip_info(mv: Mv) -> MbInfo {
    MbInfo {
        kind: MbKind::Skip,
        cbp_luma: 0,
        cbp_chroma: 0,
        luma_dc_coded: false,
        luma_coded: 0,
        chroma_dc_coded: [false; 2],
        chroma_ac_coded: [0; 2],
        mvs: [mv; 4],
        abs_mvd: [[0; 2]; 4],
        transform_8x8: false,
        chroma_pred_mode: ChromaMode::Dc,
        intra_modes: [2; 16],
        qp: 0,
        ref_idx: 0,
    }
}

/// The [`MbInfo`] of a coded macroblock before its syntax fills in the
/// coded-block flags, modes and `QP_Y`.
pub(crate) fn new_info(
    kind: MbKind,
    mvs: [Mv; 4],
    abs_mvd: [[u16; 2]; 4],
    res: &Residual,
) -> MbInfo {
    MbInfo {
        kind,
        cbp_luma: res.cbp_luma,
        cbp_chroma: res.cbp_chroma,
        luma_dc_coded: false,
        luma_coded: 0,
        chroma_dc_coded: [false; 2],
        chroma_ac_coded: [0; 2],
        mvs,
        abs_mvd,
        transform_8x8: false,
        chroma_pred_mode: ChromaMode::Dc,
        intra_modes: [2; 16],
        qp: 0,
        ref_idx: 0,
    }
}

/// `mb_type` `I_NxN` (Table 9-36: the single bin "0"). In I slices bin 0's
/// ctxIdxInc counts available neighbours that are not `I_NxN`
/// (§9.3.3.1.1.3); in P slices it follows the prefix bin "1" and uses the
/// suffix context 17.
fn write_mb_type_intra_nxn(cabac: &mut impl BinSink, slice: SliceKind, nb: &Neighbours<'_>) {
    match slice {
        SliceKind::I => cabac.encode_decision(MB_TYPE_I_CTX + i_slice_mb_type_inc(nb), false),
        SliceKind::P => {
            cabac.encode_decision(MB_TYPE_P_PREFIX_CTX, true); // prefix: intra
            cabac.encode_decision(MB_TYPE_P_SUFFIX_CTX, false);
        }
    }
}

/// ctxIdxInc of bin 0 of an I-slice `mb_type`: available neighbours that
/// are not `I_NxN` (§9.3.3.1.1.3).
fn i_slice_mb_type_inc(nb: &Neighbours<'_>) -> usize {
    let cond = |n: Option<&MbInfo>| usize::from(n.is_some_and(|m| !m.is_intra_nxn()));
    cond(nb.left) + cond(nb.above)
}

/// `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode` of the 16
/// blocks in decoding order (§7.3.5.1; `rem` is 3 fixed-length bins, least
/// significant first, §9.3.2.5).
fn write_intra_4x4_modes(cabac: &mut impl BinSink, modes: &[u8; 16], nb: &Neighbours<'_>) {
    let mut decoded = [2u8; 16];
    for blk_idx in 0..16 {
        let (bx, by) = block_position(blk_idx);
        let mode = modes[by * 4 + bx];
        let predicted = nb.predicted_intra_4x4_mode(bx, by, &decoded);
        write_intra_4x4_mode(cabac, mode, predicted);
        decoded[by * 4 + bx] = mode;
    }
}

/// `prev_intra8x8_pred_mode_flag` / `rem_intra8x8_pred_mode` of the four
/// 8x8 blocks (raster = decoding order; §7.3.5.1, the same binarization and
/// contexts as `Intra_4x4`); returns the mode of every 4x4 block.
fn write_intra_8x8_modes(
    cabac: &mut impl BinSink,
    modes: [u8; 4],
    nb: &Neighbours<'_>,
) -> [u8; 16] {
    let mut decoded = [2u8; 4];
    for (b8, &mode) in modes.iter().enumerate() {
        let own = modes_per_4x4(decoded);
        let predicted = nb.predicted_intra_4x4_mode(2 * (b8 % 2), 2 * (b8 / 2), &own);
        write_intra_4x4_mode(cabac, mode, predicted);
        decoded[b8] = mode;
    }
    modes_per_4x4(modes)
}

/// `prev_intra4x4_pred_mode_flag` and `rem_intra4x4_pred_mode` of one block
/// coded in `mode` whose predicted mode is `predicted`.
pub(crate) fn write_intra_4x4_mode(cabac: &mut impl BinSink, mode: u8, predicted: u8) {
    cabac.encode_decision(PREV_INTRA_PRED_MODE_CTX, mode == predicted);
    if mode != predicted {
        let rem = if mode < predicted { mode } else { mode - 1 };
        for bit in 0..3 {
            cabac.encode_decision(REM_INTRA_PRED_MODE_CTX, (rem >> bit) & 1 != 0);
        }
    }
}

/// `mb_type` of an `I_16x16` macroblock with `Intra16x16PredMode`
/// `pred_mode` (Table 9-36). In I slices it uses ctxIdxOffset 3; in P slices
/// it follows the prefix bin "1" and uses the suffix offset 17 (Table 9-39).
fn write_mb_type_intra16(
    cabac: &mut impl BinSink,
    slice: SliceKind,
    res: &Residual,
    pred_mode: u32,
    nb: &Neighbours<'_>,
) {
    let chroma_nonzero = res.cbp_chroma != CBP_CHROMA_NONE;
    let (bin0_ctx, luma_ctx, chroma_ctx, chroma_ac_ctx, pred_ctx) = match slice {
        SliceKind::I => {
            let b = MB_TYPE_I_CTX;
            (
                b + i_slice_mb_type_inc(nb),
                b + 3,
                b + 4,
                b + 5,
                [b + 6, b + 7],
            )
        }
        SliceKind::P => {
            cabac.encode_decision(MB_TYPE_P_PREFIX_CTX, true); // prefix: intra
            let b = MB_TYPE_P_SUFFIX_CTX;
            (b, b + 1, b + 2, b + 2, [b + 3, b + 3])
        }
    };
    cabac.encode_decision(bin0_ctx, true); // not I_NxN
    cabac.encode_terminate(false); // not I_PCM
    cabac.encode_decision(luma_ctx, res.cbp_luma != 0);
    cabac.encode_decision(chroma_ctx, chroma_nonzero);
    if chroma_nonzero {
        cabac.encode_decision(chroma_ac_ctx, res.cbp_chroma == CBP_CHROMA_AC);
    }
    // Intra16x16PredMode: two bins, most significant first.
    cabac.encode_decision(pred_ctx[0], pred_mode & 2 != 0);
    cabac.encode_decision(pred_ctx[1], pred_mode & 1 != 0);
}

/// `intra_chroma_pred_mode` (§9.3.2: truncated unary, cMax 3). Bin 0's
/// ctxIdxInc counts available intra neighbours with a non-DC chroma mode
/// (§9.3.3.1.1.8); bins 1 and 2 use ctxIdxInc 3.
fn write_intra_chroma_pred_mode(cabac: &mut impl BinSink, mode: ChromaMode, nb: &Neighbours<'_>) {
    let cond = |n: Option<&MbInfo>| {
        usize::from(n.is_some_and(|m| m.is_intra() && m.chroma_pred_mode != ChromaMode::Dc))
    };
    let value = mode as usize;
    cabac.encode_decision(
        INTRA_CHROMA_PRED_MODE_CTX + cond(nb.left) + cond(nb.above),
        value > 0,
    );
    for bin in 1..3 {
        if value < bin {
            break;
        }
        cabac.encode_decision(INTRA_CHROMA_PRED_MODE_CTX + 3, value > bin);
    }
}

/// `mb_type` of a P-slice inter macroblock (Table 9-37 bin strings "000",
/// "011", "010", "001"; ctxIdxInc of bin 2 is 2 after a "0" and 3 after a
/// "1", Table 9-39).
fn write_mb_type_p(cabac: &mut impl BinSink, partition: MbPartition) {
    let (b1, b2) = match partition {
        MbPartition::P16x16 => (false, false),
        MbPartition::P16x8 => (true, true),
        MbPartition::P8x16 => (true, false),
        MbPartition::P8x8 => (false, true),
    };
    cabac.encode_decision(MB_TYPE_P_PREFIX_CTX, false);
    cabac.encode_decision(MB_TYPE_P_PREFIX_CTX + 1, b1);
    cabac.encode_decision(MB_TYPE_P_PREFIX_CTX + 2 + usize::from(b1), b2);
    if partition == MbPartition::P8x8 {
        // sub_mb_type P_L0_8x8 ("1") for each of the four sub-macroblocks.
        for _ in 0..4 {
            cabac.encode_decision(SUB_MB_TYPE_P_CTX, true);
        }
    }
}

/// `ref_idx_l0` of every partition in order (§7.3.5.1 / §7.3.5.2), all
/// `ref_idx`: unary bins (§9.3.2.1), the first with `ctxIdxInc`
/// `condTermFlagA + 2 · condTermFlagB` (§9.3.3.1.1.6: the partition left of
/// and above the partition's top-left block uses an index above 0; skipped,
/// intra and absent neighbours do not).
fn write_ref_indices(
    cabac: &mut impl BinSink,
    partition: MbPartition,
    ref_idx: u8,
    nb: &Neighbours<'_>,
) {
    let inc = ref_idx_first_inc(partition, ref_idx, nb);
    for &first_inc in &inc[..partition.count()] {
        for bin in 0..=usize::from(ref_idx) {
            cabac.encode_decision(ref_idx_ctx(bin, first_inc), bin < usize::from(ref_idx));
        }
    }
}

/// `ctxIdxInc` of the first `ref_idx_l0` bin of each partition of a
/// macroblock whose partitions all use `ref_idx` (§9.3.3.1.1.6).
pub fn ref_idx_first_inc(partition: MbPartition, ref_idx: u8, nb: &Neighbours<'_>) -> [usize; 4] {
    let outside = |m: Option<&MbInfo>| {
        usize::from(m.is_some_and(|m| !m.is_intra() && !m.is_skip() && m.ref_idx > 0))
    };
    let own = usize::from(ref_idx > 0);
    core::array::from_fn(|part| {
        if part >= partition.count() {
            return 0;
        }
        let (x8, y8, _, _) = partition.rect(part);
        let a = if x8 > 0 { own } else { outside(nb.left) };
        let b = if y8 > 0 { own } else { outside(nb.above) };
        a + 2 * b
    })
}

/// The context index of `ref_idx_l0` bin `bin` whose first bin has
/// `first_inc` (for pricing).
pub fn ref_idx_ctx(bin: usize, first_inc: usize) -> usize {
    REF_IDX_CTX
        + match bin {
            0 => first_inc,
            1 => REF_IDX_BIN1_INC,
            _ => REF_IDX_LATER_INC,
        }
}

/// `mvd_l0` of every partition in order (§7.3.5.1 / §7.3.5.2), each against
/// its predicted vector for reference index `ref_idx` (`known_mvp`, if
/// given, for a `P_L0_16x16` macroblock); returns `|mvd|` per quadrant for
/// later contexts.
fn write_motion(
    cabac: &mut impl BinSink,
    partition: MbPartition,
    mvs: &[Mv; 4],
    (ref_idx, known_mvp): (u8, Option<Mv>),
    nb: &Neighbours<'_>,
) -> [[u16; 2]; 4] {
    let mut own = [None; 4];
    let mut abs = [[0u16; 2]; 4];
    for part in 0..partition.count() {
        let (x8, y8, w8, h8) = partition.rect(part);
        let mv = mvs[y8 * 2 + x8];
        let mvp = match known_mvp {
            Some(mvp) if partition == MbPartition::P16x16 => {
                debug_assert_eq!(mvp, predict_mv(nb, &own, partition, part, ref_idx));
                mvp
            }
            _ => predict_mv(nb, &own, partition, part, ref_idx),
        };
        let mvd = Mv {
            x: mv.x - mvp.x,
            y: mv.y - mvp.y,
        };
        // Neighbouring partitions A (left) and B (above) of the partition's
        // top-left block (§9.3.3.1.1.7); zero for intra, skipped or absent.
        let abs_a = if x8 > 0 {
            abs[y8 * 2]
        } else {
            nb.left.map_or([0; 2], |m| m.abs_mvd[y8 * 2 + 1])
        };
        let abs_b = if y8 > 0 {
            abs[x8]
        } else {
            nb.above.map_or([0; 2], |m| m.abs_mvd[2 + x8])
        };
        write_mvd(cabac, mvd, abs_a, abs_b);
        let this = [mvd.x.unsigned_abs(), mvd.y.unsigned_abs()];
        for qy in y8..y8 + h8 {
            for qx in x8..x8 + w8 {
                own[qy * 2 + qx] = Some(mv);
                abs[qy * 2 + qx] = this;
            }
        }
    }
    abs
}

/// `mvd_l0` of one partition, both components (§9.3.2.3 UEG3, signed;
/// contexts §9.3.3.1.1.7 from the `|mvd|` of neighbours A and B).
fn write_mvd(cabac: &mut impl BinSink, mvd: Mv, abs_a: [u16; 2], abs_b: [u16; 2]) {
    for (comp, value) in [i32::from(mvd.x), i32::from(mvd.y)].into_iter().enumerate() {
        let sum = u32::from(abs_a[comp]) + u32::from(abs_b[comp]);
        let inc0 = if sum < 3 {
            0
        } else if sum > 32 {
            2
        } else {
            1
        };
        let base = MVD_CTX[comp];
        let abs = value.unsigned_abs();
        let prefix = abs.min(MVD_PREFIX_MAX);
        // Truncated unary prefix: `prefix` ones, then a zero unless at cMax.
        for bin_idx in 0..=prefix {
            if bin_idx == MVD_PREFIX_MAX {
                break;
            }
            let ctx = if bin_idx == 0 {
                base + inc0
            } else {
                base + (bin_idx as usize + 2).min(6)
            };
            cabac.encode_decision(ctx, bin_idx < prefix);
        }
        if abs >= MVD_PREFIX_MAX {
            write_exp_golomb_bypass(cabac, abs - MVD_PREFIX_MAX, MVD_SUFFIX_ORDER);
        }
        if value != 0 {
            cabac.encode_bypass(value < 0);
        }
    }
}

/// k-th order Exp-Golomb suffix in bypass bins (§9.3.2.3).
fn write_exp_golomb_bypass(cabac: &mut impl BinSink, value: u32, order: u32) {
    let mut remaining = value;
    let mut k = order;
    while remaining >= (1 << k) {
        cabac.encode_bypass(true);
        remaining -= 1 << k;
        k += 1;
    }
    cabac.encode_bypass(false);
    for bit in (0..k).rev() {
        cabac.encode_bypass((remaining >> bit) & 1 != 0);
    }
}

/// `coded_block_pattern` of a non-`I_16x16` macroblock (§9.3.2.6: four
/// luma bins, then the chroma value truncated unary; contexts §9.3.3.1.1.4).
fn write_coded_block_pattern(cabac: &mut impl BinSink, res: &Residual, nb: &Neighbours<'_>) {
    let cbp = res.cbp_luma;
    for b8 in 0..4usize {
        // condTermFlagN is 1 when the neighbouring 8x8 block exists and has
        // no residual (or its macroblock is skipped); 0 when unavailable.
        let neighbour_bit = |mb: Option<&MbInfo>, nb8: usize| {
            mb.map_or(0, |m| {
                usize::from(m.is_skip() || (m.cbp_luma >> nb8) & 1 == 0)
            })
        };
        let current_bit = |nb8: usize| usize::from((cbp >> nb8) & 1 == 0);
        cabac.begin_part(MbPart::Luma8x8(b8 as u8));
        let a = if b8 % 2 == 1 {
            current_bit(b8 - 1)
        } else {
            neighbour_bit(nb.left, b8 + 1)
        };
        let b = if b8 >= 2 {
            current_bit(b8 - 2)
        } else {
            neighbour_bit(nb.above, b8 + 2)
        };
        cabac.encode_decision(CBP_LUMA_CTX + a + 2 * b, (cbp >> b8) & 1 != 0);
    }
    let cond = |mb: Option<&MbInfo>, f: fn(u8) -> bool| {
        usize::from(mb.is_some_and(|m| !m.is_skip() && f(m.cbp_chroma)))
    };
    let nonzero: fn(u8) -> bool = |c| c != CBP_CHROMA_NONE;
    let ac: fn(u8) -> bool = |c| c == CBP_CHROMA_AC;
    let inc = cond(nb.left, nonzero) + 2 * cond(nb.above, nonzero);
    cabac.begin_part(MbPart::Chroma);
    cabac.encode_decision(CBP_CHROMA_CTX + inc, res.cbp_chroma != CBP_CHROMA_NONE);
    if res.cbp_chroma != CBP_CHROMA_NONE {
        let inc = cond(nb.left, ac) + 2 * cond(nb.above, ac) + 4;
        cabac.encode_decision(CBP_CHROMA_CTX + inc, res.cbp_chroma == CBP_CHROMA_AC);
    }
}

/// condTermFlagN of `coded_block_flag` (§9.3.3.1.1.9) for a block in a
/// neighbouring macroblock: unavailable → 1 for an intra current macroblock
/// and 0 for an inter one; skipped → 0; otherwise the neighbour block's flag
/// (`coded` returns false where that block is not coded at all).
fn cond_term(
    neighbour: Option<&MbInfo>,
    current_intra: bool,
    coded: impl Fn(&MbInfo) -> bool,
) -> usize {
    match neighbour {
        None => usize::from(current_intra),
        Some(m) if m.is_skip() => 0,
        Some(m) => usize::from(coded(m)),
    }
}

/// `residual()` (§7.3.5.3): luma, then chroma DC, then chroma AC.
fn write_residual(
    cabac: &mut impl BinSink,
    dsp: &Dsp,
    res: &Residual,
    nb: &Neighbours<'_>,
    info: &mut MbInfo,
) {
    let intra16 = res.kind == ResidualKind::Intra16x16;
    let intra = !res.kind.is_inter();
    if intra16 {
        let coded_dc = |m: &MbInfo| m.is_intra() && m.luma_dc_coded;
        let a = cond_term(nb.left, true, coded_dc);
        let b = cond_term(nb.above, true, coded_dc);
        info.luma_dc_coded = write_residual_block(
            cabac,
            dsp,
            BlockCat::LumaDc,
            &res.luma_dc,
            &SCAN_4X4,
            a + 2 * b,
        );
    }
    for b8 in 0..4 {
        if (res.cbp_luma >> b8) & 1 == 0 {
            continue;
        }
        cabac.begin_part(MbPart::Luma8x8(b8 as u8));
        if res.kind.uses_8x8_transform() {
            let block = &res.luma_8x8[b8];
            write_luma_8x8_block(cabac, dsp, block);
            for blk in blocks_of_8x8(b8) {
                info.luma_coded |= 1 << blk;
            }
            continue;
        }
        for blk in blocks_of_8x8(b8) {
            let (bx, by) = (blk % 4, blk / 4);
            let a = if bx > 0 {
                usize::from(info.luma_block_coded(bx - 1, by))
            } else {
                cond_term(nb.left, intra, |m| m.luma_block_coded(3, by))
            };
            let b = if by > 0 {
                usize::from(info.luma_block_coded(bx, by - 1))
            } else {
                cond_term(nb.above, intra, |m| m.luma_block_coded(bx, 3))
            };
            let block = &res.luma[blk];
            let coded = if intra16 {
                write_residual_block(cabac, dsp, BlockCat::LumaAc, block, &SCAN_4X4_AC, a + 2 * b)
            } else {
                write_residual_block(cabac, dsp, BlockCat::Luma4x4, block, &SCAN_4X4, a + 2 * b)
            };
            if coded {
                info.luma_coded |= 1 << blk;
            }
        }
    }

    if res.cbp_chroma == CBP_CHROMA_NONE {
        return;
    }
    cabac.begin_part(MbPart::Chroma);
    for c in 0..2 {
        let coded_dc = |m: &MbInfo| m.chroma_dc_coded[c];
        let a = cond_term(nb.left, intra, coded_dc);
        let b = cond_term(nb.above, intra, coded_dc);
        info.chroma_dc_coded[c] = write_residual_block(
            cabac,
            dsp,
            BlockCat::ChromaDc,
            &res.chroma_dc[c],
            &SCAN_CHROMA_DC,
            a + 2 * b,
        );
    }
    if res.cbp_chroma != CBP_CHROMA_AC {
        return;
    }
    for c in 0..2 {
        let flag = |bits: u8, bx: usize, by: usize| (bits >> (by * 2 + bx)) & 1 != 0;
        for blk in 0..4 {
            let (bx, by) = (blk % 2, blk / 2);
            let a = if bx > 0 {
                usize::from(flag(info.chroma_ac_coded[c], 0, by))
            } else {
                cond_term(nb.left, intra, |m| flag(m.chroma_ac_coded[c], 1, by))
            };
            let b = if by > 0 {
                usize::from(flag(info.chroma_ac_coded[c], bx, 0))
            } else {
                cond_term(nb.above, intra, |m| flag(m.chroma_ac_coded[c], bx, 1))
            };
            let block = &res.chroma_ac[c][blk];
            if write_residual_block(
                cabac,
                dsp,
                BlockCat::ChromaAc,
                block,
                &SCAN_4X4_AC,
                a + 2 * b,
            ) {
                info.chroma_ac_coded[c] |= 1 << blk;
            }
        }
    }
}
