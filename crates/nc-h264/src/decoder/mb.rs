//! `macroblock_layer()` (§7.3.5) of I, P, B, SP and SI slices, written
//! once against [`MbSyntax`]: macroblock and sub-macroblock types, intra
//! prediction modes (§8.3.1.1, §8.3.2.1), reference indices and motion
//! vectors (§8.4.1, direct prediction in `direct.rs`), coded block
//! patterns, `mb_qp_delta` and the residual (§7.3.5.3), dequantised as it
//! is parsed (`transform.rs`), or kept as levels for the macroblocks that
//! §8.6 reconstructs (`switching.rs`).

use nc_encoder::Mv;

use super::chroma::ChromaArray;
use super::direct::{DirectContext, Spatial};
use super::mbstate::{
    self, CBP_CHROMA_SHIFT, DC_PRED_MODE, LISTS, MbKind, MbState, Neighbours, PCM_CBP,
    PCM_TOTAL_COEFF, Partition, SubPartition, partition_blocks,
};
use super::mvpred::MotionCache;
use super::sample::Coefficient;
use super::slice::SliceType;
use super::switching::SwitchingLevels;
use super::syntax::{Block4x4, Coeffs, MbSyntax};
use super::transform::{Dequantiser, Residual, Scan};
use super::{DecodeError, Result};

/// Table 7-11: `mb_type` 0 is `I_NxN`, 1..=24 `Intra_16x16`, 25 `I_PCM`.
const MB_TYPE_I_NXN: u32 = 0;
const MB_TYPE_I_PCM: u32 = 25;
/// `Intra_16x16` `mb_type`s: 1 + mode + 4 * chroma cbp + 12 * (luma coded).
const I16X16_MODES: u32 = 4;
const I16X16_CHROMA_CBPS: u32 = 3;
/// Table 7-12: `mb_type` 0 of SI slices is `SI`, 1.. the I types.
const MB_TYPE_SI: u32 = 0;
const SI_INTRA_BASE: u32 = 1;
/// Table 7-13: P `mb_type`s; 4 is `P_8x8ref0`, 5.. are intra.
const MB_TYPE_P_8X8_REF0: u32 = 4;
const P_INTRA_BASE: u32 = 5;
/// `QP_Y` wraps modulo 52 (eq. 7-37).
const QP_RANGE: i32 = 52;
/// Samples of an `I_PCM` macroblock: 256 luma, 128 chroma (4:2:0).
pub(super) const PCM_LUMA_SAMPLES: usize = 256;
const MAX_PCM_CHROMA_SAMPLES: usize = 512;

/// What a slice tells its macroblocks.
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the flags mirror independent slice and parameter set properties"
)]
pub(super) struct SliceParams<'a> {
    pub slice_type: SliceType,
    pub num_ref_idx_active: [usize; LISTS],
    /// Identity of each list entry's picture (`None`: no picture).
    pub ref_ids: [&'a [Option<u32>]; LISTS],
    /// Direct prediction (B slices).
    pub direct: Option<&'a DirectContext<'a>>,
    pub transform_8x8_mode: bool,
    pub constrained_intra_pred: bool,
    /// `ChromaArrayType`.
    pub chroma: ChromaArray,
    /// An SP or SI slice: its P and SI macroblocks keep their levels for
    /// §8.6.
    pub switching: bool,
    pub dequantiser: &'a Dequantiser,
    /// `QpBdOffsetY` (eq. 7-4).
    pub qp_bd_offset: i32,
    /// `qpprime_y_zero_transform_bypass_flag`: macroblocks with `QP′Y` 0
    /// are coded without transform (lossless).
    pub transform_bypass: bool,
    /// Bits of a luma and a chroma `I_PCM` sample (the bit depths).
    pub pcm_bits: [u32; 2],
}

/// Per-macroblock scratch beyond its [`MbState`], reused: the `I_PCM`
/// samples, the levels of the residual block being parsed and those of a
/// macroblock §8.6 reconstructs.
#[derive(Debug)]
pub(super) struct MbExtra {
    pub pcm: [u16; PCM_LUMA_SAMPLES + MAX_PCM_CHROMA_SAMPLES],
    pub levels: Coeffs,
    pub switching: SwitchingLevels,
}

impl MbExtra {
    pub fn new() -> Self {
        Self {
            pcm: [0; PCM_LUMA_SAMPLES + MAX_PCM_CHROMA_SAMPLES],
            levels: Coeffs::new(),
            switching: SwitchingLevels::new(),
        }
    }
}

/// Where the parsed levels of a residual go: dequantised into the
/// macroblock's residual (§8.5), or kept for §8.6. The luma-like blocks
/// are those of component `PLANE` (Cb and Cr: 4:4:4).
trait LevelSink {
    /// Start a macroblock coded with (lossless) or without transform bypass.
    fn set_bypass(&mut self, bypass: bool);
    fn luma_dc<const PLANE: usize>(&mut self, levels: &Coeffs, qp: (i32, &Scan));
    /// 4x4 block `block` (raster index); `kind`: (`Intra16x16ACLevel`,
    /// intra).
    fn luma_4x4<const PLANE: usize>(
        &mut self,
        levels: &Coeffs,
        kind: (bool, bool),
        qp: (i32, &Scan),
        block: usize,
    );
    fn luma_8x8<const PLANE: usize>(
        &mut self,
        levels: &Coeffs,
        intra: bool,
        qp: (i32, &Scan),
        b8: usize,
    );
    fn chroma_dc(&mut self, levels: &Coeffs, intra: bool, qp: i32, comp: usize);
    fn chroma_ac(&mut self, levels: &Coeffs, intra: bool, qp: (i32, &Scan), block: (usize, usize));
    /// Luma 4x4 blocks with coefficients (deblocking, bit `y4 * 4 + x4`).
    fn luma_nonzero(&self) -> u16;
}

/// The dequantising sink of every macroblock but those of §8.6.
struct Dequantise<'a, C> {
    dequantiser: &'a Dequantiser,
    residual: &'a mut Residual<C>,
}

impl<C: Coefficient> LevelSink for Dequantise<'_, C> {
    #[inline]
    fn set_bypass(&mut self, bypass: bool) {
        self.residual.set_bypass(bypass);
    }

    #[inline]
    fn luma_dc<const PLANE: usize>(&mut self, levels: &Coeffs, qp: (i32, &Scan)) {
        self.dequantiser
            .luma_dc::<C, PLANE>(levels, qp, self.residual);
    }

    #[inline]
    fn luma_4x4<const PLANE: usize>(
        &mut self,
        levels: &Coeffs,
        kind: (bool, bool),
        qp: (i32, &Scan),
        block: usize,
    ) {
        self.dequantiser
            .luma_4x4::<C, PLANE>(levels, kind, qp, block, self.residual);
    }

    #[inline]
    fn luma_8x8<const PLANE: usize>(
        &mut self,
        levels: &Coeffs,
        intra: bool,
        qp: (i32, &Scan),
        b8: usize,
    ) {
        self.dequantiser
            .luma_8x8::<C, PLANE>(levels, intra, qp, b8, self.residual);
    }

    #[inline]
    fn chroma_dc(&mut self, levels: &Coeffs, intra: bool, qp: i32, comp: usize) {
        self.dequantiser
            .chroma_dc(levels, intra, qp, comp, self.residual);
    }

    #[inline]
    fn chroma_ac(&mut self, levels: &Coeffs, intra: bool, qp: (i32, &Scan), block: (usize, usize)) {
        self.dequantiser
            .chroma_ac(levels, intra, qp, block, self.residual);
    }

    #[inline]
    fn luma_nonzero(&self) -> u16 {
        self.residual.luma_nonzero()
    }
}

/// §8.6 macroblocks (P in SP slices, SI) are never `Intra_16x16` nor use
/// the 8x8 transform (no Extended-profile PPS enables it, `switching.rs`):
/// only 4x4 luma and chroma levels of 4:2:0 reach this sink, without
/// transform bypass (an Extended-profile SPS has neither).
impl LevelSink for SwitchingLevels {
    fn set_bypass(&mut self, _bypass: bool) {}

    fn luma_dc<const PLANE: usize>(&mut self, _levels: &Coeffs, _qp: (i32, &Scan)) {
        unreachable!("an Intra_16x16 macroblock reconstructed by §8.6");
    }

    fn luma_4x4<const PLANE: usize>(
        &mut self,
        levels: &Coeffs,
        _kind: (bool, bool),
        (_, scan): (i32, &Scan),
        block: usize,
    ) {
        Self::luma_4x4(self, levels, scan, block);
    }

    fn luma_8x8<const PLANE: usize>(
        &mut self,
        _levels: &Coeffs,
        _intra: bool,
        _qp: (i32, &Scan),
        _b8: usize,
    ) {
        unreachable!("an 8x8 transform in an SP or SI slice");
    }

    fn chroma_dc(&mut self, levels: &Coeffs, _intra: bool, _qp: i32, comp: usize) {
        Self::chroma_dc(self, levels, comp);
    }

    fn chroma_ac(
        &mut self,
        levels: &Coeffs,
        _intra: bool,
        (_, scan): (i32, &Scan),
        (comp, b): (usize, usize),
    ) {
        Self::chroma_ac(self, levels, scan, comp, b);
    }

    fn luma_nonzero(&self) -> u16 {
        self.luma_coded()
    }
}

/// A `P_Skip` macroblock (§8.4.1.1).
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "one copy per sample type keeps the 8-bit instance inlined (high-bit-depth handoff)"
)]
pub(super) fn skip<T>(
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    qp: i32,
) -> Result<()> {
    cur.kind = MbKind::PSkip;
    cur.qp = qp as i8;
    let mv = MotionCache::skip_mv_of(nb);
    cur.mvs[0] = [mv; 16];
    let id = reference_id(params, 0, 0)?;
    cur.ref_idx[0] = [0; 4];
    cur.ref_pic[0] = [id; 4];
    cur.single_motion = true;
    Ok(())
}

/// One coded macroblock `addr`; `qp` is `QP_Y`,PRED in and `QP_Y` out.
#[allow(
    clippy::too_many_arguments,
    reason = "the macroblock's syntax, context, state and scratch"
)]
pub(super) fn decode<S: MbSyntax, C: Coefficient>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    addr: usize,
    qp: &mut i32,
    cur: &mut MbState,
    residual: &mut Residual<C>,
    extra: &mut MbExtra,
) -> Result<()> {
    let mut mb_type = syntax.mb_type(params.slice_type, nb)?;
    let intra_base = match params.slice_type {
        SliceType::I => 0,
        SliceType::Si => SI_INTRA_BASE,
        SliceType::P => P_INTRA_BASE,
        SliceType::B => B_INTRA_BASE,
    };
    let mut sink = Dequantise {
        dequantiser: params.dequantiser,
        residual,
    };
    if params.slice_type == SliceType::Si && mb_type == MB_TYPE_SI {
        return si_macroblock(syntax, params, nb, qp, cur, extra);
    }
    if mb_type < intra_base {
        let no_8x8 = if params.slice_type == SliceType::B && mb_type == MB_TYPE_B_DIRECT {
            direct_16x16::<C>(params, nb, cur, addr, false)?;
            !params.direct.is_some_and(|d| d.inference_8x8)
        } else if params.slice_type == SliceType::B {
            b_inter_prediction(syntax, params, nb, cur, addr, mb_type)?
        } else {
            inter_prediction::<S, false>(syntax, params, nb, cur, addr, mb_type)?
        };
        let (levels, held) = (&mut extra.levels, &mut extra.switching);
        return if params.switching {
            // A P macroblock of an SP slice (§8.6.1, §8.6.2).
            coded_residual(syntax, params, nb, qp, cur, held, levels, no_8x8)
        } else {
            coded_residual(syntax, params, nb, qp, cur, &mut sink, levels, no_8x8)
        };
    }
    mb_type -= intra_base;
    match mb_type {
        MB_TYPE_I_PCM => {
            cur.kind = MbKind::IPcm;
            cur.qp = *qp as i8;
            cur.cbp = PCM_CBP;
            cur.total_coeff = [PCM_TOTAL_COEFF; 16];
            cur.chroma_total_coeff = [[PCM_TOTAL_COEFF; 16]; 2];
            cur.dc_coded = [true; 3];
            cur.nonzero = u16::MAX;
            let samples = PCM_LUMA_SAMPLES + 2 * params.chroma.pcm_chroma_samples();
            syntax.pcm_samples(&mut extra.pcm[..samples], params.pcm_bits)?;
            syntax.no_qp_delta();
            Ok(())
        }
        MB_TYPE_I_NXN => {
            cur.kind = MbKind::IntraNxN;
            if params.transform_8x8_mode {
                cur.transform_8x8 = syntax.transform_size_8x8_flag(nb)?;
            }
            intra_nxn_modes(syntax, params, nb, cur)?;
            if params.chroma.has_chroma_blocks() {
                cur.chroma_pred_mode = syntax.intra_chroma_pred_mode(nb)?;
            }
            coded_residual(
                syntax,
                params,
                nb,
                qp,
                cur,
                &mut sink,
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
            if params.chroma.has_chroma_blocks() {
                cur.chroma_pred_mode = syntax.intra_chroma_pred_mode(nb)?;
            }
            *qp = next_qp(*qp, syntax.mb_qp_delta()?, params.qp_bd_offset);
            cur.qp = *qp as i8;
            residual_layer(syntax, params, nb, cur, &mut sink, &mut extra.levels)
        }
    }
}

/// An `SI` macroblock (Table 7-12): `Intra_4x4` prediction modes, the
/// chroma prediction mode and a coded residual, kept as levels for §8.6.2.
fn si_macroblock<S: MbSyntax>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    qp: &mut i32,
    cur: &mut MbState,
    extra: &mut MbExtra,
) -> Result<()> {
    cur.kind = MbKind::Si;
    intra_nxn_modes(syntax, params, nb, cur)?;
    if params.chroma.has_chroma_blocks() {
        cur.chroma_pred_mode = syntax.intra_chroma_pred_mode(nb)?;
    }
    let levels = &mut extra.levels;
    coded_residual(
        syntax,
        params,
        nb,
        qp,
        cur,
        &mut extra.switching,
        levels,
        false,
    )
}

/// `coded_block_pattern`, `transform_size_8x8_flag`, `mb_qp_delta` and the
/// residual of a non-`Intra_16x16` macroblock. `no_8x8` is whether an inter
/// macroblock's partitioning rules the 8x8 transform out (§7.3.5).
fn coded_residual<S: MbSyntax, K: LevelSink>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    qp: &mut i32,
    cur: &mut MbState,
    sink: &mut K,
    levels: &mut Coeffs,
    no_8x8: bool,
) -> Result<()> {
    cur.cbp = syntax.coded_block_pattern(nb, cur)?;
    if !cur.is_intra() && cur.cbp_luma() != 0 && params.transform_8x8_mode && !no_8x8 {
        cur.transform_8x8 = syntax.transform_size_8x8_flag(nb)?;
    }
    if cur.cbp == 0 {
        cur.qp = *qp as i8;
        syntax.no_qp_delta();
        return Ok(());
    }
    *qp = next_qp(*qp, syntax.mb_qp_delta()?, params.qp_bd_offset);
    cur.qp = *qp as i8;
    residual_layer(syntax, params, nb, cur, sink, levels)
}

/// `QP_Y` from `QP_Y`,PRED and `mb_qp_delta` (eq. 7-37).
fn next_qp(pred: i32, delta: i32, qp_bd_offset: i32) -> i32 {
    // eq. 7-37: ((pred + delta + 52 + 2 · QpBdOffsetY) % (52 +
    // QpBdOffsetY)) − QpBdOffsetY.
    (pred + delta + qp_bd_offset).rem_euclid(QP_RANGE + qp_bd_offset) - qp_bd_offset
}

/// `residual()` (§7.3.5.3), into `sink`: the luma blocks, then the
/// chroma DC and AC blocks (4:2:0, 4:2:2) or the Cb and Cr blocks coded
/// like luma (4:4:4).
fn residual_layer<S: MbSyntax, K: LevelSink>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    sink: &mut K,
    coeffs: &mut Coeffs,
) -> Result<()> {
    // TransformBypassModeFlag (§8.5.15): QP′Y is 0.
    sink.set_bypass(params.transform_bypass && i32::from(cur.qp) == -params.qp_bd_offset);
    residual_luma::<S, K, 0>(syntax, nb, cur, sink, coeffs)?;
    cur.nonzero = sink.luma_nonzero();
    match params.chroma {
        ChromaArray::Mono => Ok(()),
        ChromaArray::Yuv444 => {
            residual_luma::<S, K, 1>(syntax, nb, cur, sink, coeffs)?;
            residual_luma::<S, K, 2>(syntax, nb, cur, sink, coeffs)
        }
        ChromaArray::Yuv420 | ChromaArray::Yuv422 => {
            residual_chroma(syntax, params, nb, cur, sink, coeffs)
        }
    }
}

/// `residual_luma()` (§7.3.5.3.1) of component `PLANE`: luma, or Cb and
/// Cr coded like luma (4:4:4).
fn residual_luma<S: MbSyntax, K: LevelSink, const PLANE: usize>(
    syntax: &mut S,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    sink: &mut K,
    coeffs: &mut Coeffs,
) -> Result<()> {
    let intra = cur.is_intra();
    let qp = i32::from(cur.qp);
    let i16x16 = matches!(cur.kind, MbKind::Intra16x16(_));
    let scan = Scan::of(cur.field);
    if i16x16 {
        syntax.luma_dc::<PLANE>(nb, cur, coeffs)?;
        sink.luma_dc::<PLANE>(coeffs, (qp, scan));
    }
    for b8 in 0..4 {
        if (cur.cbp_luma() >> b8) & 1 == 0 {
            continue;
        }
        if cur.transform_8x8 {
            syntax.luma_8x8::<PLANE>(nb, cur, b8, coeffs)?;
            sink.luma_8x8::<PLANE>(coeffs, intra, (qp, scan), b8);
            continue;
        }
        for i in 0..4 {
            let (x, y) = ((b8 % 2) * 2 + i % 2, (b8 / 2) * 2 + i / 2);
            let block = if i16x16 {
                Block4x4::LumaAc(x, y)
            } else {
                Block4x4::Luma(x, y)
            };
            syntax.block_4x4::<PLANE>(nb, cur, block, coeffs)?;
            sink.luma_4x4::<PLANE>(coeffs, (i16x16, intra), (qp, scan), y * 4 + x);
        }
    }
    Ok(())
}

/// The chroma DC and AC blocks of 4:2:0 and 4:2:2 (§7.3.5.3: `4 *
/// NumC8x8` blocks per component, in `chroma4x4BlkIdx` order).
fn residual_chroma<S: MbSyntax, K: LevelSink>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    sink: &mut K,
    coeffs: &mut Coeffs,
) -> Result<()> {
    if cur.cbp_chroma() == 0 {
        return Ok(());
    }
    let intra = cur.is_intra();
    let qp = i32::from(cur.qp);
    let scan = Scan::of(cur.field);
    for comp in 0..2 {
        syntax.chroma_dc(nb, cur, comp, coeffs)?;
        sink.chroma_dc(coeffs, intra, qp, comp);
    }
    if cur.cbp_chroma() == 2 {
        for comp in 0..2 {
            for b in 0..params.chroma.chroma_blocks() {
                syntax.block_4x4::<0>(nb, cur, Block4x4::ChromaAc(comp, b), coeffs)?;
                sink.chroma_ac(coeffs, intra, (qp, scan), (comp, b));
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
/// neighbours without block modes (not `I_NxN` or `SI`) counting as DC.
/// Inline: the partitioned slice syntax's instance of the macroblock layer
/// is a second caller, which took it out of line (+0.1 % on CAVLC High).
#[inline]
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
        Some(if m.has_block_modes() {
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

/// The identity of the picture of entry `ref_idx` of list `list`. An entry
/// without a picture is "no reference picture" (§8.2.4.2) or a
/// "non-existing" frame of a `frame_num` gap (§8.2.5.2): a conforming
/// stream never predicts from either, so the picture it would have held was
/// lost.
fn reference_id(params: &SliceParams<'_>, list: usize, ref_idx: usize) -> Result<u32> {
    match params.ref_ids[list].get(ref_idx) {
        Some(Some(id)) => Ok(*id),
        Some(None) => Err(DecodeError::MissingReference),
        None => Err(DecodeError::Malformed("reference index outside the list")),
    }
}

/// Which lists a partition predicts from: bit 0 list 0, bit 1 list 1.
const PRED_L0: u8 = 1;
const PRED_L1: u8 = 2;
const PRED_BI: u8 = PRED_L0 | PRED_L1;

/// One partition of an inter macroblock, or one 8x8 block of a `P_8x8` /
/// `B_8x8` one with its sub-macroblock partitioning.
#[derive(Debug, Clone, Copy)]
struct Part {
    /// (`x4`, `y4`, `w4`, `h4`) in 4x4 blocks.
    rect: (usize, usize, usize, usize),
    /// Sub-macroblock partitioning (8x8 blocks; `P8x8`, one block, else).
    sub: SubPartition,
    preds: u8,
    /// `B_Direct_8x8`.
    direct: bool,
}

impl Part {
    const fn whole(rect: (usize, usize, usize, usize), preds: u8) -> Self {
        Self {
            rect,
            sub: SubPartition::P8x8,
            preds,
            direct: false,
        }
    }

    /// The (sub-)partitions of the part, as rectangles in decoding order:
    /// the part itself, or an 8x8 block's sub-partitions (raster order in
    /// the 2x2 grid of 4x4 blocks).
    fn rects(self) -> impl Iterator<Item = (usize, usize, usize, usize)> {
        let (x0, y0, w, h) = self.rect;
        let ((sw, sh), count) = if (w, h) == (2, 2) {
            (self.sub.size4(), self.sub.count())
        } else {
            ((w, h), 1)
        };
        (0..count).map(move |index| {
            let first = index * sw;
            (x0 + (first & 1), y0 + (first >> 1) * sh, sw, sh)
        })
    }
}

/// The partition rectangles of each partitioning (8x8 blocks for
/// `P8x8`).
const fn partition_rects(partition: Partition) -> &'static [(usize, usize, usize, usize)] {
    match partition {
        Partition::P16x16 => &[(0, 0, 4, 4)],
        Partition::P16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
        Partition::P8x16 => &[(0, 0, 2, 4), (2, 0, 2, 4)],
        Partition::P8x8 => &[(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)],
    }
}

/// B `mb_type` 1..=21 (Table 7-14): partitioning and the lists of its
/// partitions.
const B_MB_TYPES: [(Partition, [u8; 2]); 21] = [
    (Partition::P16x16, [PRED_L0, 0]),
    (Partition::P16x16, [PRED_L1, 0]),
    (Partition::P16x16, [PRED_BI, 0]),
    (Partition::P16x8, [PRED_L0, PRED_L0]),
    (Partition::P8x16, [PRED_L0, PRED_L0]),
    (Partition::P16x8, [PRED_L1, PRED_L1]),
    (Partition::P8x16, [PRED_L1, PRED_L1]),
    (Partition::P16x8, [PRED_L0, PRED_L1]),
    (Partition::P8x16, [PRED_L0, PRED_L1]),
    (Partition::P16x8, [PRED_L1, PRED_L0]),
    (Partition::P8x16, [PRED_L1, PRED_L0]),
    (Partition::P16x8, [PRED_L0, PRED_BI]),
    (Partition::P8x16, [PRED_L0, PRED_BI]),
    (Partition::P16x8, [PRED_L1, PRED_BI]),
    (Partition::P8x16, [PRED_L1, PRED_BI]),
    (Partition::P16x8, [PRED_BI, PRED_L0]),
    (Partition::P8x16, [PRED_BI, PRED_L0]),
    (Partition::P16x8, [PRED_BI, PRED_L1]),
    (Partition::P8x16, [PRED_BI, PRED_L1]),
    (Partition::P16x8, [PRED_BI, PRED_BI]),
    (Partition::P8x16, [PRED_BI, PRED_BI]),
];
/// B `mb_type` 0 is `B_Direct_16x16`, 22 `B_8x8`, 23.. intra.
const MB_TYPE_B_DIRECT: u32 = 0;
const MB_TYPE_B_8X8: u32 = 22;
const B_INTRA_BASE: u32 = 23;

/// B `sub_mb_type` 1..=12 (Table 7-18): partitioning and lists; 0 is
/// `B_Direct_8x8`.
const B_SUB_MB_TYPES: [(SubPartition, u8); 12] = [
    (SubPartition::P8x8, PRED_L0),
    (SubPartition::P8x8, PRED_L1),
    (SubPartition::P8x8, PRED_BI),
    (SubPartition::P8x4, PRED_L0),
    (SubPartition::P4x8, PRED_L0),
    (SubPartition::P8x4, PRED_L1),
    (SubPartition::P4x8, PRED_L1),
    (SubPartition::P8x4, PRED_BI),
    (SubPartition::P4x8, PRED_BI),
    (SubPartition::P4x4, PRED_L0),
    (SubPartition::P4x4, PRED_L1),
    (SubPartition::P4x4, PRED_BI),
];

/// The partitions of P `mb_type`s 0..=2 (Table 7-13), from list 0.
const P_PARTS: [&[Part]; 3] = [
    &[Part::whole((0, 0, 4, 4), PRED_L0)],
    &[
        Part::whole((0, 0, 4, 2), PRED_L0),
        Part::whole((0, 2, 4, 2), PRED_L0),
    ],
    &[
        Part::whole((0, 0, 2, 4), PRED_L0),
        Part::whole((2, 0, 2, 4), PRED_L0),
    ],
];
/// `mb_type` 3 of P slices.
const MB_TYPE_P_8X8: u32 = 3;

/// The partitions of B `mb_type`s 1..=21 (from [`B_MB_TYPES`]; the second
/// unused for 16x16).
const B_PARTS: [[Part; 2]; 21] = b_parts_table();

const fn b_parts_table() -> [[Part; 2]; 21] {
    let mut out = [[Part::whole((0, 0, 0, 0), 0); 2]; 21];
    let mut t = 0;
    while t < out.len() {
        let (partition, preds) = B_MB_TYPES[t];
        let rects = partition_rects(partition);
        let mut i = 0;
        while i < rects.len() {
            out[t][i] = Part::whole(rects[i], preds[i]);
            i += 1;
        }
        t += 1;
    }
    out
}

/// Read the four `sub_mb_type`s of a `P_8x8` or (`B`) `B_8x8` macroblock
/// into `parts` (Tables 7-17, 7-18).
fn sub_parts<S: MbSyntax, const B: bool>(syntax: &mut S, parts: &mut [Part; 4]) -> Result<()> {
    for (part, &rect) in parts.iter_mut().zip(partition_rects(Partition::P8x8)) {
        *part = Part::whole(rect, PRED_L0);
        if !B {
            part.sub = SubPartition::from_code(syntax.sub_mb_type(SliceType::P)?)
                .ok_or(DecodeError::Malformed("sub_mb_type"))?;
            continue;
        }
        match syntax.sub_mb_type(SliceType::B)? {
            0 => part.direct = true,
            code => {
                let (sub, preds) = B_SUB_MB_TYPES
                    .get(usize::from(code) - 1)
                    .ok_or(DecodeError::Malformed("sub_mb_type"))?;
                part.sub = *sub;
                part.preds = *preds;
            }
        }
    }
    Ok(())
}

/// Set the direct prediction of 8x8 block `b8` of macroblock `addr`
/// (§8.4.1.2) in `cur`.
fn direct_8x8(
    params: &SliceParams<'_>,
    spatial: Option<&Spatial>,
    addr: usize,
    cur: &mut MbState,
    b8: usize,
) -> Result<()> {
    let direct = params.direct.ok_or(DecodeError::Malformed(
        "direct prediction outside a B slice",
    ))?;
    let (x0, y0) = ((b8 % 2) * 2, (b8 / 2) * 2);
    let mut motion = direct.block(spatial, addr, x0, y0)?;
    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let (x, y) = (x0 + dx, y0 + dy);
        // Under direct_8x8_inference_flag the 8x8 block shares one motion.
        if (dx, dy) != (0, 0) && !direct.inference_8x8 {
            motion = direct.block(spatial, addr, x, y)?;
        }
        for list in 0..LISTS {
            cur.mvs[list][y * 4 + x] = motion.mv[list];
        }
    }
    for list in 0..LISTS {
        let ref_idx = motion.ref_idx[list];
        cur.ref_idx[list][b8] = ref_idx;
        cur.ref_pic[list][b8] = if ref_idx >= 0 {
            reference_id(params, list, ref_idx as usize)?
        } else {
            0
        };
    }
    cur.direct |= 1 << b8;
    Ok(())
}

/// The two fresh motion caches of a B macroblock.
fn b_caches(nb: &Neighbours<'_>) -> [MotionCache; LISTS] {
    [MotionCache::new(nb, 0), MotionCache::new(nb, 1)]
}

/// The spatial direct part of a B macroblock, in spatial mode.
fn spatial_part(params: &SliceParams<'_>, caches: &[MotionCache; LISTS]) -> Option<Spatial> {
    params
        .direct
        .filter(|d| d.spatial)
        .map(|_| Spatial::new(caches))
}

/// A `B_Skip` (`skip`) or `B_Direct_16x16` macroblock's prediction
/// (§8.4.1.2). Out of line, as [`b_inter_prediction`].
#[inline(never)]
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "one copy per sample type keeps the 8-bit instance inlined (high-bit-depth handoff)"
)]
pub(super) fn direct_16x16<T>(
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    addr: usize,
    skip: bool,
) -> Result<()> {
    cur.kind = if skip {
        MbKind::BSkip
    } else {
        MbKind::BDirect16x16
    };
    let spatial = spatial_part(params, &b_caches(nb));
    for b8 in 0..4 {
        direct_8x8(params, spatial.as_ref(), addr, cur, b8)?;
    }
    cur.single_motion = has_single_motion(cur);
    Ok(())
}

/// [`inter_prediction`] of a B macroblock, out of line: the B instance
/// kept inline enlarged the macroblock loop of P slices.
#[inline(never)]
fn b_inter_prediction<S: MbSyntax>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    addr: usize,
    mb_type: u32,
) -> Result<bool> {
    inter_prediction::<S, true>(syntax, params, nb, cur, addr, mb_type)
}

/// The prediction part of a P (`B` false) or B inter macroblock (`mb_pred`
/// / `sub_mb_pred`, §7.3.5.1, §7.3.5.2) and its motion vectors; returns
/// whether the 8x8 transform is not allowed (sub-partitions below 8x8, or
/// direct blocks without `direct_8x8_inference_flag`). One instance per
/// slice type keeps list 1 and direct prediction out of the P one.
fn inter_prediction<S: MbSyntax, const B: bool>(
    syntax: &mut S,
    params: &SliceParams<'_>,
    nb: &Neighbours<'_>,
    cur: &mut MbState,
    addr: usize,
    mb_type: u32,
) -> Result<bool> {
    let mut sub_buf = [Part::whole((0, 0, 0, 0), 0); 4];
    let (partition, parts): (Partition, &[Part]) = if B && mb_type == MB_TYPE_B_8X8
        || !B && mb_type >= MB_TYPE_P_8X8
    {
        sub_parts::<S, B>(syntax, &mut sub_buf)?;
        (Partition::P8x8, &sub_buf)
    } else if B {
        let index = mb_type as usize - 1;
        let partition = B_MB_TYPES[index].0;
        let count = partition_rects(partition).len();
        (partition, &B_PARTS[index][..count])
    } else {
        let partition = [Partition::P16x16, Partition::P16x8, Partition::P8x16][mb_type as usize];
        (partition, P_PARTS[mb_type as usize])
    };
    cur.kind = MbKind::Inter(partition);
    let lists = if B { LISTS } else { 1 };
    let ref0 = !B && mb_type == MB_TYPE_P_8X8_REF0;
    if B && parts.iter().any(|p| p.direct) {
        let spatial = spatial_part(params, &b_caches(nb));
        for (b8, part) in parts.iter().enumerate() {
            if part.direct {
                direct_8x8(params, spatial.as_ref(), addr, cur, b8)?;
            }
        }
    }
    for list in 0..lists {
        let coded = params.num_ref_idx_active[list] > 1 && !ref0;
        for part in parts {
            if (B && part.direct) || part.preds & (1 << list) == 0 {
                continue;
            }
            let (x, y, _, _) = part.rect;
            let ref_idx = if coded {
                syntax.ref_idx(nb, cur, list, x, y, params.num_ref_idx_active[list])?
            } else {
                0
            };
            set_ref(params, cur, list, part.rect, ref_idx)?;
        }
    }
    for list in 0..lists {
        // Fresh: the macroblock's partitions become available as they are
        // recorded, in decoding order.
        let mut cache = MotionCache::new(nb, list);
        for (index, part) in parts.iter().enumerate() {
            let b8 = (part.rect.1 / 2) * 2 + part.rect.0 / 2;
            if B && part.direct {
                for block in part
                    .rects()
                    .flat_map(|(x, y, _, _)| [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)])
                {
                    let mv = cur.mvs[list][block.1 * 4 + block.0];
                    cache.record((block.0, block.1, 1, 1), mv, cur.ref_idx[list][b8]);
                }
            } else if B && part.preds & (1 << list) == 0 {
                cache.record(part.rect, Mv::ZERO, -1);
            } else {
                for rect in part.rects() {
                    let mvd = syntax.mvd(nb, cur, list, rect.0, rect.1)?;
                    let ref_idx = cur.ref_idx[list][(rect.1 / 2) * 2 + rect.0 / 2];
                    let pred = cache.predict(rect, partition, index, ref_idx);
                    let mv = set_motion(cur, list, rect, pred, mvd);
                    cache.record(rect, mv, ref_idx);
                }
            }
        }
    }
    cur.single_motion = partition == Partition::P16x16 && cur.direct == 0 || has_single_motion(cur);
    #[cfg(feature = "analysis")]
    if partition == Partition::P8x8 {
        cur.sub_partitions = core::array::from_fn(|b8| parts[b8].sub);
    }
    let inference = params.direct.is_some_and(|d| d.inference_8x8);
    Ok(parts
        .iter()
        .any(|p| p.sub != SubPartition::P8x8 || (B && p.direct && !inference)))
}

/// Whether every 4x4 block of `cur` has the same vectors and reference
/// indices in both lists (the same indices also mean the same weights). An
/// unused list's vectors are not compared (stale outside B slices).
fn has_single_motion(cur: &MbState) -> bool {
    (0..LISTS).all(|list| {
        cur.ref_idx[list] == [-1; 4]
            || (cur.mvs[list].iter().all(|&mv| mv == cur.mvs[list][0])
                && cur.ref_idx[list].iter().all(|&r| r == cur.ref_idx[list][0]))
    })
}

/// Record list `list`'s `ref_idx` (and its picture) for the 8x8 blocks of
/// the partition `(x, y, w, h)`.
fn set_ref(
    params: &SliceParams<'_>,
    cur: &mut MbState,
    list: usize,
    (x, y, w, h): (usize, usize, usize, usize),
    ref_idx: u8,
) -> Result<()> {
    let id = reference_id(params, list, usize::from(ref_idx))?;
    // The 8x8 blocks the partition covers, as a 2x2 grid of bits.
    let covered = partition_blocks(x / 2, y / 2, w.div_ceil(2), h.div_ceil(2), 2);
    for (b8, (slot_idx, slot_pic)) in cur.ref_idx[list]
        .iter_mut()
        .zip(&mut cur.ref_pic[list])
        .enumerate()
    {
        let inside = (covered >> b8) & 1 != 0;
        *slot_idx = core::hint::select_unpredictable(inside, ref_idx as i8, *slot_idx);
        *slot_pic = core::hint::select_unpredictable(inside, id, *slot_pic);
    }
    Ok(())
}

/// `mvLX = mvpLX + mvdLX` (eq. 8-174) and the CABAC `absMvdComp` record of
/// list `list`; returns the vector.
fn set_motion(
    cur: &mut MbState,
    list: usize,
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
    for (block, (slot_mv, slot_abs)) in cur.mvs[list]
        .iter_mut()
        .zip(&mut cur.abs_mvd[list])
        .enumerate()
    {
        let inside = (covered >> block) & 1 != 0;
        *slot_mv = core::hint::select_unpredictable(inside, mv, *slot_mv);
        *slot_abs = core::hint::select_unpredictable(inside, abs, *slot_abs);
    }
    mv
}
