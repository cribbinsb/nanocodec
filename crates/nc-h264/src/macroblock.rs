//! Macroblock signal path: prediction, transform, quantisation and
//! reconstruction for `I_16x16` (DC) and `P_L0_16x16` / `P_Skip`
//! macroblocks.
//!
//! Intra: 16 4x4 core transforms plus the 4x4 Hadamard of their DC terms
//! (H.264 §8.5.2). Inter: 16 plain 4x4 transforms (§8.5.12) or four 8x8
//! transforms (§8.5.13, `transform_size_8x8_flag`), `CodedBlockPatternLuma`
//! per 8x8 either way. Chroma (both): 4 transforms per component plus the
//! 2x2 DC transform (§8.5.11). Reconstruction uses the normative scaling and
//! inverse transforms, so it is exactly what a decoder sees.

use nc_common::{Picture, Plane, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, H264Quantizer, copy_square};
use nc_encoder::Mv;

use crate::cabac::BitCounter;
use crate::mc::McKernels;
use crate::predict::{Availability, ChromaMode, Edges, Intra16x16Mode, chroma_8x8, luma_16x16};
use crate::residual::BlockCat;
use crate::residual::trellis::{BlockScale, RateModel, TrellisBlock, TrellisQuantizer};
use crate::tables::{ZIGZAG_4X4, ZIGZAG_8X8, chroma_qp};

/// Luma macroblock size and its 4:2:0 chroma block size.
const MB: usize = 16;
const MB_CHROMA: usize = 8;

/// `CodedBlockPatternLuma` with all four 8x8 blocks coded.
pub const CBP_LUMA_ALL: u8 = 15;
/// `CodedBlockPatternChroma` values (§7.4.5).
pub const CBP_CHROMA_NONE: u8 = 0;
pub const CBP_CHROMA_DC: u8 = 1;
pub const CBP_CHROMA_AC: u8 = 2;

/// The prediction of one macroblock.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// 16x16 luma, raster.
    pub luma: [u8; 256],
    /// 8x8 Cb and Cr, raster.
    pub chroma: [[u8; 64]; 2],
}

impl Prediction {
    /// An all-zero prediction (a buffer to predict into).
    pub const ZERO: Self = Self {
        luma: [0; 256],
        chroma: [[0; 64]; 2],
    };
}

/// The intra modes chosen for an `I_16x16` macroblock.
#[derive(Debug, Clone, Copy)]
pub struct IntraModes {
    pub luma: Intra16x16Mode,
    pub chroma: ChromaMode,
    /// SATD of the chosen `Intra_16x16` luma prediction.
    pub luma_satd: u32,
}

/// Intra prediction of the macroblock at `(x, y)` from the (unfiltered)
/// reconstruction of the current picture into `pred`: the `Intra_16x16`
/// mode and the chroma mode with the lowest SATD against the source, among
/// those whose neighbours exist (§8.3.3, §8.3.4). Ties keep the earlier
/// mode, DC first. Each mode is predicted on the stack and copied into
/// `pred` only when it wins so far (moving every candidate through an
/// iterator minimum cost about five 256-byte copies per macroblock).
/// With `dc_only` only the DC modes are predicted (no chroma SATD at all).
pub fn predict_intra(
    dsp: &Dsp,
    source: &Picture,
    recon: &Picture,
    (x, y): (usize, usize),
    pred: &mut Prediction,
    dc_only: bool,
) -> IntraModes {
    let avail = Availability::at(x, y);
    let luma_edges = Edges::gather(&recon.plane(PlaneIndex::Y).view(), x, y);
    let src_y = source
        .plane(PlaneIndex::Y)
        .region(x as isize, y as isize, MB, MB)
        .expect("macroblock inside the picture");
    let size_16 = BlockSize::new(MB, MB).expect("16x16 is a kernel size");
    let mut best_luma: Option<(Intra16x16Mode, u32)> = None;
    for mode in Intra16x16Mode::usable(avail) {
        if dc_only && mode != Intra16x16Mode::Dc {
            continue;
        }
        let candidate = luma_16x16(&luma_edges, mode);
        let view =
            PlaneView::new(&candidate, MB, MB, MB).expect("prediction covers the macroblock");
        let satd = dsp.satd(size_16, &src_y, &view);
        if best_luma.is_none_or(|(_, best)| satd < best) {
            best_luma = Some((mode, satd));
            pred.luma = candidate;
        }
    }
    let (luma_mode, luma_satd) = best_luma.expect("DC is always usable");

    let size_8 = BlockSize::new(MB_CHROMA, MB_CHROMA).expect("8x8 is a kernel size");
    let (cx, cy) = (x / 2, y / 2);
    let planes = [PlaneIndex::U, PlaneIndex::V];
    let chroma_edges = planes.map(|p| Edges::gather(&recon.plane(p).view(), cx, cy));
    let src_chroma = planes.map(|p| {
        source
            .plane(p)
            .region(cx as isize, cy as isize, MB_CHROMA, MB_CHROMA)
            .expect("macroblock inside the picture")
    });
    if dc_only {
        pred.chroma = chroma_edges.map(|edges| chroma_8x8(&edges, ChromaMode::Dc));
        return IntraModes {
            luma: luma_mode,
            chroma: ChromaMode::Dc,
            luma_satd,
        };
    }
    let mut best_chroma: Option<(ChromaMode, u32)> = None;
    for mode in ChromaMode::usable(avail) {
        let candidate = chroma_edges.map(|edges| chroma_8x8(&edges, mode));
        let satd = src_chroma
            .iter()
            .zip(&candidate)
            .map(|(src, plane)| {
                let view = PlaneView::new(plane, MB_CHROMA, MB_CHROMA, MB_CHROMA)
                    .expect("prediction covers the block");
                dsp.satd(size_8, src, &view)
            })
            .sum::<u32>();
        if best_chroma.is_none_or(|(_, best)| satd < best) {
            best_chroma = Some((mode, satd));
            pred.chroma = candidate;
        }
    }
    let (chroma_mode, _) = best_chroma.expect("DC is always usable");

    IntraModes {
        luma: luma_mode,
        chroma: chroma_mode,
        luma_satd,
    }
}

/// Motion-compensated prediction from `reference` (§8.4.2.2), one vector
/// per 8x8 quadrant (raster). Interpolation is per sample, so predicting by
/// quadrant gives exactly the samples of any partitioning they encode.
pub fn predict_inter(
    mc: &McKernels,
    reference: &Picture,
    x: usize,
    y: usize,
    mvs: &[Mv; 4],
    out: &mut Prediction,
) {
    if mvs.iter().all(|&mv| mv == mvs[0]) {
        mc.luma_16x16(reference.plane(PlaneIndex::Y), x, y, mvs[0], &mut out.luma);
        predict_inter_chroma(mc, reference, x, y, mvs[0], out);
        return;
    }
    let Prediction { luma, chroma } = out;
    let planes = [PlaneIndex::U, PlaneIndex::V];
    for (q, &mv) in mvs.iter().enumerate() {
        let (qx, qy) = (8 * (q % 2), 8 * (q / 2));
        let mut block = [0u8; 64];
        mc.luma_8x8(
            reference.plane(PlaneIndex::Y),
            (x + qx, y + qy),
            mv,
            &mut block,
        );
        for (r, row) in block.as_chunks::<8>().0.iter().enumerate() {
            luma[(qy + r) * MB + qx..][..8].copy_from_slice(row);
        }
        for (out, plane) in chroma.iter_mut().zip(planes) {
            let mut block = [0u8; 16];
            let (cx, cy) = (x / 2 + qx / 2, y / 2 + qy / 2);
            mc.chroma_4x4(reference.plane(plane), (cx, cy), mv, &mut block);
            for (r, row) in block.as_chunks::<4>().0.iter().enumerate() {
                out[(qy / 2 + r) * MB_CHROMA + qx / 2..][..4].copy_from_slice(row);
            }
        }
    }
}

/// The chroma prediction of the unsplit macroblock at `(x, y)` with the luma
/// vector `mv`; the luma prediction is left as it is.
pub fn predict_inter_chroma(
    mc: &McKernels,
    reference: &Picture,
    x: usize,
    y: usize,
    mv: Mv,
    out: &mut Prediction,
) {
    for (out, plane) in out.chroma.iter_mut().zip([PlaneIndex::U, PlaneIndex::V]) {
        mc.chroma_8x8(reference.plane(plane), x / 2, y / 2, mv, out);
    }
}

/// Residual components selected for omission by inter SAD thresholds.
/// Unit scales retain the original quantizer-derived zero bounds; larger
/// scales deliberately discard small residuals before transform and partition
/// search. Decoder reconstruction still uses the signalled zero residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualSkip {
    /// Every luma 4x4 block.
    pub luma: bool,
    /// Per luma 8x8 quadrant (bit `b8`, raster): its four 4x4 blocks.
    /// Selected quadrants omit their residual for either transform size;
    /// see [`inter_residual_skip`].
    pub luma_quadrants: u8,
    /// Every 4x4 block and the DC block of each chroma component.
    pub chroma: [bool; 2],
}

impl ResidualSkip {
    /// Nothing proven zero: every block is transformed.
    pub const NONE: Self = Self {
        luma: false,
        luma_quadrants: 0,
        chroma: [false; 2],
    };

    /// Nothing at all needs coding.
    pub fn all(self) -> bool {
        self.luma && self.chroma[0] && self.chroma[1]
    }
}

/// Select residual components to omit using cached quantizer-dependent SAD
/// bounds. Unit scales preserve the original 4x4 zero test; larger configured
/// scales make an approximate early-skip decision. Luma 8x8 transforms retain
/// the existing policy of omitting quadrants that pass the 4x4 test.
pub fn inter_residual_skip(
    dsp: &Dsp,
    source: &Picture,
    pred: &Prediction,
    x: usize,
    y: usize,
    quantizers: &MbQuantizers,
) -> ResidualSkip {
    residual_skip_within(dsp, source, pred, x, y, quantizers.inter_zero_limits)
}

/// [`inter_residual_skip`] against explicit `[luma 4x4, chroma 4x4,
/// chroma DC]` SAD bounds.
pub fn residual_skip_within(
    dsp: &Dsp,
    source: &Picture,
    pred: &Prediction,
    x: usize,
    y: usize,
    limits: [u32; 3],
) -> ResidualSkip {
    let luma = PlaneView::new(&pred.luma, MB, MB, MB).expect("prediction covers the block");
    let chroma = [0, 1].map(|c| {
        PlaneView::new(&pred.chroma[c], MB_CHROMA, MB_CHROMA, MB_CHROMA)
            .expect("prediction covers the block")
    });
    residual_skip_views(dsp, source, (x, y), &luma, &chroma, limits)
}

/// [`residual_skip_within`] for a prediction given as views (16x16 luma,
/// 8x8 chroma), e.g. blocks of the reference picture itself.
pub fn residual_skip_views(
    dsp: &Dsp,
    source: &Picture,
    (x, y): (usize, usize),
    pred_luma: &PlaneView<'_>,
    pred_chroma: &[PlaneView<'_>; 2],
    [luma_limit, ac_limit, dc_limit]: [u32; 3],
) -> ResidualSkip {
    let src_luma = source
        .plane(PlaneIndex::Y)
        .region(x as isize, y as isize, MB, MB)
        .expect("block inside the picture");
    let src_chroma = [PlaneIndex::U, PlaneIndex::V].map(|plane| {
        source
            .plane(plane)
            .region((x / 2) as isize, (y / 2) as isize, MB_CHROMA, MB_CHROMA)
            .expect("block inside the picture")
    });
    let (luma_sads, chroma_sads) = dsp.sad_quads_macroblock(
        (&src_luma, pred_luma),
        [
            (&src_chroma[0], &pred_chroma[0]),
            (&src_chroma[1], &pred_chroma[1]),
        ],
    );
    // Luma: a quadrant is selected when its four 4x4 SADs are within the
    // configured bound.
    let mut luma_quadrants = 0;
    for (b8, sads) in luma_sads.iter().enumerate() {
        if sads.iter().all(|&sad| sad <= luma_limit) {
            luma_quadrants |= 1 << b8;
        }
    }
    let luma = luma_quadrants == ALL_QUADRANTS;
    // Chroma: each 4x4 SAD within the AC bound and the 8x8 SAD within the
    // DC bound.
    let chroma = chroma_sads.map(|sads| {
        sads.iter().all(|&sad| sad <= ac_limit) && sads.iter().sum::<u32>() <= dc_limit
    });
    ResidualSkip {
        luma,
        luma_quadrants,
        chroma,
    }
}

/// Every luma 8x8 quadrant of a [`ResidualSkip`].
pub const ALL_QUADRANTS: u8 = 0b1111;

/// How the residual is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualKind {
    /// `I_16x16`: separate luma DC, AC in the 4x4 blocks.
    Intra16x16,
    /// Inter: plain 4x4 luma blocks.
    Inter,
    /// Inter with `transform_size_8x8_flag`: four 8x8 luma blocks.
    Inter8x8,
    /// `I_NxN` with 4x4 transforms: 16 luma 4x4 blocks, each predicted and
    /// reconstructed in turn (`intra4x4`).
    Intra4x4,
    /// `I_NxN` with 8x8 transforms: four luma 8x8 blocks, each predicted
    /// and reconstructed in turn (`intra8x8`).
    Intra8x8,
}

impl ResidualKind {
    /// True for the inter kinds (inter rounding, coefficient elimination).
    pub fn is_inter(self) -> bool {
        matches!(self, Self::Inter | Self::Inter8x8)
    }

    /// True for the kinds whose luma is in 8x8 blocks
    /// (`transform_size_8x8_flag`).
    pub fn uses_8x8_transform(self) -> bool {
        matches!(self, Self::Inter8x8 | Self::Intra8x8)
    }

    /// True for `I_NxN`, whose luma is quantised and reconstructed block by
    /// block while it is predicted, not with the macroblock.
    pub fn is_intra_nxn(self) -> bool {
        matches!(self, Self::Intra4x4 | Self::Intra8x8)
    }
}

/// Quantised levels of a macroblock, raster order within each block.
#[derive(Debug, Clone)]
pub struct Residual {
    /// What the levels mean.
    pub kind: ResidualKind,
    /// `Intra16x16`: luma DC levels, raster over the 4x4 grid of blocks.
    pub luma_dc: [i32; 16],
    /// Luma levels per 4x4 block (index `by * 4 + bx`); position 0 unused
    /// for `Intra16x16`, all zero for `Inter8x8`.
    pub luma: [[i32; 16]; 16],
    /// `Inter8x8` and `Intra8x8`: luma levels per 8x8 block (raster
    /// quadrants), raster within the block; all zero otherwise.
    pub luma_8x8: [[i32; 64]; 4],
    /// Chroma DC levels per component, raster over its 2x2 blocks.
    pub chroma_dc: [[i32; 4]; 2],
    /// Chroma AC levels per component and 4x4 block (index `by * 2 + bx`).
    pub chroma_ac: [[[i32; 16]; 4]; 2],
    /// `CodedBlockPatternLuma`.
    pub cbp_luma: u8,
    /// `CodedBlockPatternChroma`.
    pub cbp_chroma: u8,
    /// Scratch for the sample residual `source - pred` of the block being
    /// transformed. Kept with the reused levels so it is neither zeroed nor
    /// returned by value per block: those memset/memcpy calls cost about
    /// 0.42M instructions per 720p crowd P frame. Only the first
    /// `size * size` entries of the current block are meaningful.
    samples: [i16; MB * MB],
}

impl Residual {
    /// Prepare to hold a macroblock of `kind`, with nothing coded yet.
    ///
    /// Only the fields read regardless of the coded block patterns are
    /// cleared (the DC levels and the patterns); the block arrays are
    /// overwritten by whatever codes them, and readers (syntax,
    /// reconstruction) only look at blocks the patterns mark as coded. This
    /// keeps a reused `Residual` from being zeroed (2.6 KB) per macroblock.
    pub fn reset(&mut self, kind: ResidualKind) {
        self.kind = kind;
        self.luma_dc = [0; 16];
        self.chroma_dc = [[0; 4]; 2];
        self.cbp_luma = 0;
        self.cbp_chroma = CBP_CHROMA_NONE;
    }

    /// A residual of `kind` with nothing coded.
    pub fn empty(kind: ResidualKind) -> Self {
        Self {
            kind,
            luma_dc: [0; 16],
            luma: [[0; 16]; 16],
            luma_8x8: [[0; 64]; 4],
            chroma_dc: [[0; 4]; 2],
            chroma_ac: [[[0; 16]; 4]; 2],
            cbp_luma: 0,
            cbp_chroma: CBP_CHROMA_NONE,
            samples: [0; MB * MB],
        }
    }

    /// Take the chroma levels and pattern of `other`.
    pub fn copy_chroma_from(&mut self, other: &Self) {
        self.chroma_dc = other.chroma_dc;
        self.chroma_ac = other.chroma_ac;
        self.cbp_chroma = other.cbp_chroma;
    }

    /// Keep only the inter luma quadrants in `quadrants` (bit per raster
    /// quadrant); the others become zero.
    pub fn retain_luma_quadrants(&mut self, quadrants: u8) {
        debug_assert!(self.kind.is_inter(), "only inter residuals are pruned");
        for b8 in 0..4 {
            if quadrants & (1 << b8) == 0 && self.cbp_luma & (1 << b8) != 0 {
                // 4x4-transform reconstruction reads every block's levels.
                for blk in blocks_of_8x8(b8) {
                    self.luma[blk] = [0; 16];
                }
            }
        }
        self.cbp_luma &= quadrants;
    }

    /// True if nothing at all is coded (a `P_Skip` candidate).
    pub fn is_empty(&self) -> bool {
        self.cbp_luma == 0
            && self.cbp_chroma == CBP_CHROMA_NONE
            && self.luma_dc.iter().all(|&c| c == 0)
    }
}

/// Intra quantisation rounds at 1/3 of a step (`H264Quantizer::new`).
const INTRA_ROUNDING_DIVISOR: u8 = 3;

/// The quantisers of one QP: luma and chroma (`QPc`), each with the intra
/// (1/3) and configurable inter (default 1/6) rounding offsets.
#[derive(Debug, Clone, Copy)]
pub struct MbQuantizers {
    luma: [H264Quantizer; 2],
    chroma: [H264Quantizer; 2],
    elimination: Elimination,
    inter_zero_limits: [u32; 3],
    /// Smallest level kept, intra and inter: 1 normally, more past QP 51
    /// ([`MbQuantizers::with_planned_step`]).
    planned_min_level: [i32; 2],
}

impl MbQuantizers {
    /// The inter zero-residual SAD bounds (`[luma 4x4, chroma 4x4, chroma
    /// DC]`, see [`inter_residual_skip`]) with the luma and chroma bounds
    /// scaled by `luma` and `chroma`.
    pub fn inter_zero_limits_scaled(&self, luma: f32, chroma: f32) -> [u32; 3] {
        let [l, ac, dc] = self.inter_zero_limits;
        let scale = |v: u32, by: f32| (v as f32 * by) as u32;
        [scale(l, luma), scale(ac, chroma), scale(dc, chroma)]
    }

    /// Quantisers for luma QP `qp`, with sparse-residual elimination
    /// thresholds `elimination` and inter rounding `1 / inter_rounding_divisor`.
    pub fn new(
        qp: u8,
        elimination: Elimination,
        inter_rounding_divisor: u8,
        luma_zero_scale: f32,
        chroma_zero_scale: f32,
        chroma_offset: i8,
    ) -> Self {
        let qpc = chroma_qp(qp, chroma_offset);
        let luma_inter = H264Quantizer::with_rounding_divisor(qp, inter_rounding_divisor);
        let chroma_inter = H264Quantizer::with_rounding_divisor(qpc, inter_rounding_divisor);
        let inter_zero_limits = [
            (luma_inter.zero_sad_threshold_4x4() as f32 * luma_zero_scale) as u32,
            (chroma_inter.zero_sad_threshold_4x4() as f32 * chroma_zero_scale) as u32,
            (chroma_inter.zero_sad_threshold_chroma_dc() as f32 * chroma_zero_scale) as u32,
        ];
        Self {
            luma: [H264Quantizer::new(qp, true), luma_inter],
            chroma: [H264Quantizer::new(qpc, true), chroma_inter],
            elimination,
            inter_zero_limits,
            planned_min_level: [NO_DEAD_ZONE; 2],
        }
    }

    /// These quantisers when the frame was planned past QP 51 at a step
    /// `step_ratio` times the coded one (`FramePlan::lambda_scale`): levels
    /// a quantiser at the planned step would zero are dropped, i.e. those
    /// whose coefficient is below `(1 − rounding) · step_ratio` coded steps
    /// (rounding 1/3 intra, 1/`inter_rounding_divisor` inter).
    #[must_use]
    pub fn with_planned_step(mut self, step_ratio: f32, inter_rounding_divisor: u8) -> Self {
        let min_level = |divisor: u8| ((1.0 - 1.0 / f32::from(divisor)) * step_ratio).ceil() as i32;
        self.planned_min_level = [
            min_level(INTRA_ROUNDING_DIVISOR),
            min_level(inter_rounding_divisor),
        ]
        .map(|level| level.max(NO_DEAD_ZONE));
        self
    }

    fn planned_min_level(&self, kind: ResidualKind) -> i32 {
        self.planned_min_level[usize::from(kind.is_inter())]
    }

    /// The luma quantiser with intra rounding.
    pub fn luma_intra(&self) -> &H264Quantizer {
        &self.luma[0]
    }

    /// Chroma `Qstep` over luma `Qstep` of a macroblock coded as `kind`.
    pub fn chroma_step_ratio(&self, kind: ResidualKind) -> f32 {
        let (luma, chroma) = self.for_kind(kind);
        (chroma.step() / luma.step()) as f32
    }

    fn for_kind(&self, kind: ResidualKind) -> (&H264Quantizer, &H264Quantizer) {
        let i = usize::from(kind.is_inter());
        (&self.luma[i], &self.chroma[i])
    }
}

/// Forward transform and quantise `source - pred` for the macroblock at
/// luma position `(x, y)` into `out`.
#[allow(clippy::too_many_arguments)]
pub fn transform_macroblock(
    dsp: &Dsp,
    source: &Picture,
    pred: &Prediction,
    x: usize,
    y: usize,
    kind: ResidualKind,
    quantizers: &MbQuantizers,
    zero: ResidualSkip,
    out: &mut Residual,
) {
    debug_assert!(!kind.is_intra_nxn(), "see intra_nxn_residual");
    out.reset(kind);
    let (luma_q, chroma_q) = quantizers.for_kind(kind);
    let min_level = quantizers.planned_min_level(kind);
    out.cbp_chroma = transform_chroma(
        dsp,
        source,
        pred,
        (x, y),
        kind,
        (chroma_q, &quantizers.elimination, min_level),
        zero.chroma,
        out,
    );
    if zero.luma {
        return;
    }
    residual_into(
        dsp,
        source.plane(PlaneIndex::Y),
        (x, y),
        &pred.luma,
        MB,
        &mut out.samples,
    );
    let luma_residual = &out.samples;
    let nonzero_8x8 = if kind == ResidualKind::Inter8x8 {
        // Keep the omitted quadrants while dispatching the remaining
        // transforms and quantisation together, once per macroblock.
        let mut nonzero = dsp.h264_forward_quantize_mb(
            luma_residual,
            MB,
            zero.luma_quadrants,
            luma_q,
            &mut out.luma_8x8,
        );
        if min_level > NO_DEAD_ZONE {
            for (block, count) in out.luma_8x8.iter_mut().zip(&mut nonzero) {
                if *count > 0 {
                    apply_planned_dead_zone(block, min_level);
                    *count = block.iter().map(|&level| u32::from(level != 0)).sum();
                }
            }
        }
        nonzero
    } else {
        dsp.h264_forward_4x4(luma_residual, MB, MB / 4, &mut out.luma);
        [0; 4]
    };

    out.cbp_luma = match kind {
        ResidualKind::Intra16x16 => {
            // DC: 4x4 Hadamard over the blocks' DC terms, halved (reference
            // encoder convention), quantised with one more bit of precision.
            let luma = &mut out.luma;
            let dc_in: [i32; 16] = core::array::from_fn(|blk| luma[blk][0]);
            let mut dc = dsp.h264_hadamard_4x4(&dc_in).map(|v| v >> 1);
            dsp.h264_quantize_dc(&mut dc, luma_q);
            apply_planned_dead_zone(&mut dc, min_level);
            out.luma_dc = dc;
            dsp.h264_quantize_4x4(luma, luma_q);
            for block in luma.iter_mut() {
                block[0] = 0; // the DC is coded in Intra16x16DCLevel
            }
            apply_planned_dead_zone(luma.as_flattened_mut(), min_level);
            let any_ac = !all_zero(luma.as_flattened());
            if any_ac { CBP_LUMA_ALL } else { 0 }
        }
        ResidualKind::Inter => {
            dsp.h264_quantize_4x4(&mut out.luma, luma_q);
            apply_planned_dead_zone(out.luma.as_flattened_mut(), min_level);
            eliminate_sparse_luma(&mut out.luma, &quantizers.elimination)
        }
        ResidualKind::Inter8x8 => {
            eliminate_sparse_luma_8x8(&mut out.luma_8x8, nonzero_8x8, &quantizers.elimination)
        }
        ResidualKind::Intra4x4 | ResidualKind::Intra8x8 => {
            unreachable!("I_NxN luma is coded block by block")
        }
    };
}

/// What trellis quantisation (`quant.trellis`) of a macroblock's residual
/// needs besides the levels: the rate model's bin prices, λ (sample SSE per
/// bit) and the survivors per scan position.
#[derive(Clone, Copy)]
pub struct TrellisSettings<'a> {
    pub bins: &'a BitCounter<'a>,
    pub lambda: f32,
    pub max_states: usize,
}

/// [`TrellisSettings`] for a macroblock's luma, with the `coded_block_flag`
/// neighbours outside the macroblock.
pub struct LumaTrellis<'a> {
    pub settings: TrellisSettings<'a>,
    /// Whether the 4x4 block left of row `by` (in the left macroblock) is
    /// coded, as `condTermFlagA` (§9.3.3.1.1.9).
    pub left_coded: [bool; 4],
    /// Whether the 4x4 block above column `bx` is coded (`condTermFlagB`).
    pub above_coded: [bool; 4],
}

/// [`TrellisSettings`] for a macroblock's chroma AC, with the
/// `coded_block_flag` neighbours outside the macroblock per component.
pub struct ChromaTrellis<'a> {
    pub settings: TrellisSettings<'a>,
    /// `condTermFlagA` of row `by` of component `c` (`[c][by]`).
    pub left_coded: [[bool; 2]; 2],
    /// `condTermFlagB` of column `bx` of component `c` (`[c][bx]`).
    pub above_coded: [[bool; 2]; 2],
}

/// `ctxIdxInc` of a `coded_block_flag` (§9.3.3.1.1.9) from `condTermFlagA`
/// and `condTermFlagB`.
fn cbf_ctx_inc(left: bool, above: bool) -> usize {
    usize::from(left) + 2 * usize::from(above)
}

/// The forward quantiser of `q`'s 4x4 blocks as the trellis needs it.
pub(crate) fn scale_4x4(q: &H264Quantizer) -> BlockScale<'static> {
    let step = q.step();
    let (mf, qbits) = q.forward_scale_4x4();
    BlockScale {
        mf,
        qbits,
        step_sse: (step * step) as f32,
    }
}

/// Trellis-quantise the coded luma blocks of an inter or `I_16x16`
/// macroblock in `out` (its 4x4, 8x8 or AC levels, after quantisation and
/// elimination), re-deriving their coefficients from the luma residual `out`
/// still holds, and update `cbp_luma`. Blocks the plain quantiser left
/// empty stay empty; `I_16x16` DC levels keep the plain quantiser.
pub fn trellis_luma(
    dsp: &Dsp,
    out: &mut Residual,
    quantizers: &MbQuantizers,
    trellis: &LumaTrellis<'_>,
    search: &mut TrellisQuantizer,
) {
    debug_assert!(!out.kind.is_intra_nxn(), "see intra4x4 / intra8x8");
    let (luma_q, _) = quantizers.for_kind(out.kind);
    let TrellisSettings {
        bins,
        lambda,
        max_states,
    } = trellis.settings;
    if out.kind == ResidualKind::Inter8x8 {
        let step = luma_q.step();
        let (mf, qbits) = luma_q.forward_scale_8x8();
        let scale = BlockScale {
            mf,
            qbits,
            step_sse: (step * step) as f32,
        };
        for b8 in 0..4 {
            if (out.cbp_luma >> b8) & 1 == 0 {
                continue;
            }
            let (qx, qy) = (MB / 2 * (b8 % 2), MB / 2 * (b8 / 2));
            let mut coeffs = [0; 64];
            dsp.h264_forward_8x8(&out.samples[qy * MB + qx..], MB, &mut coeffs);
            let mut model = RateModel::new(bins, TrellisBlock::Luma8x8);
            let levels = &mut out.luma_8x8[b8];
            if !search.quantize(
                &mut model,
                &coeffs,
                &ZIGZAG_8X8,
                (scale, lambda, max_states),
                levels,
            ) {
                out.cbp_luma &= !(1 << b8);
            }
        }
        return;
    }
    // I_16x16 codes the AC of every block (positions 1..16) and its DC
    // apart; inter 4x4 blocks code all sixteen positions.
    let (cat, scan) = if out.kind == ResidualKind::Intra16x16 {
        (BlockCat::LumaAc, &ZIGZAG_4X4[1..])
    } else {
        (BlockCat::Luma4x4, &ZIGZAG_4X4[..])
    };
    let scale = scale_4x4(luma_q);
    let mut coeffs = [[0; 16]; 16];
    dsp.h264_forward_4x4(&out.samples, MB, MB / 4, &mut coeffs);
    let coded = |out: &Residual, blk: usize| out.luma[blk].iter().any(|&l| l != 0);
    for b8 in 0..4 {
        if (out.cbp_luma >> b8) & 1 == 0 {
            continue;
        }
        for blk in blocks_of_8x8(b8) {
            if !coded(out, blk) {
                continue;
            }
            let (bx, by) = (blk % 4, blk / 4);
            let a = if bx > 0 {
                coded(out, blk - 1)
            } else {
                trellis.left_coded[by]
            };
            let b = if by > 0 {
                coded(out, blk - 4)
            } else {
                trellis.above_coded[bx]
            };
            let block = TrellisBlock::Small {
                cat,
                cbf_ctx_inc: cbf_ctx_inc(a, b),
            };
            let mut model = RateModel::new(bins, block);
            search.quantize(
                &mut model,
                &coeffs[blk],
                scan,
                (scale, lambda, max_states),
                &mut out.luma[blk],
            );
        }
        if out.kind != ResidualKind::Intra16x16
            && blocks_of_8x8(b8).iter().all(|&blk| !coded(out, blk))
        {
            out.cbp_luma &= !(1 << b8);
        }
    }
    // I_16x16 signals its AC for the whole macroblock or not at all.
    if out.kind == ResidualKind::Intra16x16 && all_zero(out.luma.as_flattened()) {
        out.cbp_luma = 0;
    }
}

/// Trellis-quantise the coded chroma AC blocks of the macroblock at luma
/// `(x, y)` in `out`, re-deriving their coefficients from `source` and
/// `pred`, and update `cbp_chroma`. Blocks the plain quantiser left empty
/// stay empty; the DC levels keep the plain quantiser.
#[allow(clippy::too_many_arguments)]
pub fn trellis_chroma_ac(
    dsp: &Dsp,
    source: &Picture,
    pred: &Prediction,
    (x, y): (usize, usize),
    quantizers: &MbQuantizers,
    trellis: &ChromaTrellis<'_>,
    search: &mut TrellisQuantizer,
    out: &mut Residual,
) {
    if out.cbp_chroma != CBP_CHROMA_AC {
        return;
    }
    let (_, chroma_q) = quantizers.for_kind(out.kind);
    let scale = scale_4x4(chroma_q);
    let TrellisSettings {
        bins,
        lambda,
        max_states,
    } = trellis.settings;
    for (c, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
        let coded = |out: &Residual, blk: usize| !all_zero(&out.chroma_ac[c][blk]);
        if (0..4).all(|blk| !coded(out, blk)) {
            continue;
        }
        residual_into(
            dsp,
            source.plane(plane),
            (x / 2, y / 2),
            &pred.chroma[c],
            MB_CHROMA,
            &mut out.samples,
        );
        let mut coeffs = [[0; 16]; 4];
        dsp.h264_forward_4x4(
            &out.samples[..MB_CHROMA * MB_CHROMA],
            MB_CHROMA,
            MB_CHROMA / 4,
            &mut coeffs,
        );
        for (blk, coeffs) in coeffs.iter().enumerate() {
            if !coded(out, blk) {
                continue;
            }
            let (bx, by) = (blk % 2, blk / 2);
            let a = if bx > 0 {
                coded(out, blk - 1)
            } else {
                trellis.left_coded[c][by]
            };
            let b = if by > 0 {
                coded(out, blk - 2)
            } else {
                trellis.above_coded[c][bx]
            };
            let block = TrellisBlock::Small {
                cat: BlockCat::ChromaAc,
                cbf_ctx_inc: cbf_ctx_inc(a, b),
            };
            let mut model = RateModel::new(bins, block);
            search.quantize(
                &mut model,
                coeffs,
                &ZIGZAG_4X4[1..],
                (scale, lambda, max_states),
                &mut out.chroma_ac[c][blk],
            );
        }
    }
    out.cbp_chroma = chroma_pattern(out);
}

/// `CodedBlockPatternChroma` of the chroma levels in `out`.
fn chroma_pattern(out: &Residual) -> u8 {
    if !all_zero(out.chroma_ac.as_flattened().as_flattened()) {
        CBP_CHROMA_AC
    } else if !all_zero(out.chroma_dc.as_flattened()) {
        CBP_CHROMA_DC
    } else {
        CBP_CHROMA_NONE
    }
}

/// The luma levels of an `I_NxN` macroblock, coded block by block while it
/// was predicted (`intra4x4`, `intra8x8`).
#[derive(Debug, Clone, Copy)]
pub enum IntraNxnLevels<'a> {
    /// Per 4x4 block, raster.
    Blocks4x4(&'a [[i32; 16]; 16]),
    /// Per 8x8 block, raster.
    Blocks8x8(&'a [[i32; 64]; 4]),
}

/// The residual of an `I_NxN` macroblock whose luma was already coded block
/// by block (`levels`), into `out`; its chroma is transformed here like any
/// other intra macroblock's.
#[allow(clippy::too_many_arguments)]
pub fn intra_nxn_residual(
    dsp: &Dsp,
    source: &Picture,
    pred: &Prediction,
    x: usize,
    y: usize,
    quantizers: &MbQuantizers,
    levels: IntraNxnLevels<'_>,
    out: &mut Residual,
) {
    let kind = match levels {
        IntraNxnLevels::Blocks4x4(_) => ResidualKind::Intra4x4,
        IntraNxnLevels::Blocks8x8(_) => ResidualKind::Intra8x8,
    };
    out.reset(kind);
    let (_, chroma_q) = quantizers.for_kind(kind);
    out.cbp_chroma = transform_chroma(
        dsp,
        source,
        pred,
        (x, y),
        kind,
        (
            chroma_q,
            &quantizers.elimination,
            quantizers.planned_min_level(kind),
        ),
        [false; 2],
        out,
    );
    out.cbp_luma = (0..4).fold(0u8, |cbp, b8| {
        let nonzero = match levels {
            IntraNxnLevels::Blocks4x4(levels) => {
                blocks_of_8x8(b8).iter().any(|&blk| !all_zero(&levels[blk]))
            }
            IntraNxnLevels::Blocks8x8(levels) => !all_zero(&levels[b8]),
        };
        cbp | (u8::from(nonzero) << b8)
    });
    match levels {
        IntraNxnLevels::Blocks4x4(levels) => out.luma = *levels,
        IntraNxnLevels::Blocks8x8(levels) => out.luma_8x8 = *levels,
    }
}

/// Chroma DC and AC levels per component of the macroblock at luma `(x, y)`
/// (§8.5.11), written into `out` (whose chroma DC must be cleared); returns
/// `CodedBlockPatternChroma`. Components selected in `skip_chroma`
/// omit their residual and are not transformed.
#[allow(clippy::too_many_arguments)]
fn transform_chroma(
    dsp: &Dsp,
    source: &Picture,
    pred: &Prediction,
    (x, y): (usize, usize),
    kind: ResidualKind,
    (chroma_q, elimination, min_level): (&H264Quantizer, &Elimination, i32),
    skip_chroma: [bool; 2],
    out: &mut Residual,
) -> u8 {
    let mut any_ac = false;
    for (c, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
        let ac = &mut out.chroma_ac[c];
        if skip_chroma[c] {
            // Read (as zeros) if the other component codes AC.
            *ac = [[0; 16]; 4];
            continue;
        }
        residual_into(
            dsp,
            source.plane(plane),
            (x / 2, y / 2),
            &pred.chroma[c],
            MB_CHROMA,
            &mut out.samples,
        );
        let (mut dc, ac_levels) = dsp.h264_forward_quantize_chroma(
            &out.samples[..MB_CHROMA * MB_CHROMA],
            MB_CHROMA,
            chroma_q,
            ac,
        );
        apply_planned_dead_zone(&mut dc, min_level);
        out.chroma_dc[c] = dc;
        if ac_levels > 0 {
            apply_planned_dead_zone(ac.as_flattened_mut(), min_level);
            any_ac = true;
        }
    }
    // Elimination only ever clears AC levels: nothing to do without any.
    if kind.is_inter() && any_ac {
        eliminate_sparse_chroma_ac(&mut out.chroma_ac, elimination);
    }
    chroma_pattern(out)
}

/// Write `pred + residual` (normative scaling and inverse transforms,
/// §8.5.10–8.5.13) into `recon` for the macroblock at `(x, y)`: the
/// prediction is copied in, then each plane's coded residual is added in
/// place, block by block, skipping blocks without levels.
pub fn reconstruct_macroblock(
    dsp: &Dsp,
    recon: &mut Picture,
    pred: &Prediction,
    x: usize,
    y: usize,
    res: &Residual,
    quantizers: &MbQuantizers,
) {
    let (luma_q, chroma_q) = quantizers.for_kind(res.kind);
    // I_NxN luma was already reconstructed block by block while it was coded.
    if !res.kind.is_intra_nxn() {
        let mut dst = block_mut(recon.plane_mut(PlaneIndex::Y), (x, y), MB);
        copy_square::<MB>(&mut dst, &pred.luma, MB);
        add_luma_residual(dsp, &mut dst, res, luma_q);
    }
    for (c, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
        let mut dst = block_mut(recon.plane_mut(plane), (x / 2, y / 2), MB_CHROMA);
        copy_square::<MB_CHROMA>(&mut dst, &pred.chroma[c], MB_CHROMA);
        if res.cbp_chroma == CBP_CHROMA_NONE {
            continue;
        }
        // AC levels are only meaningful when the pattern says AC is coded.
        let ac = if res.cbp_chroma == CBP_CHROMA_AC {
            &res.chroma_ac[c]
        } else {
            &NO_CHROMA_AC
        };
        let dc = dsp.h264_dequantize_chroma_dc(&res.chroma_dc[c], chroma_q);
        dsp.h264_add_inverse_4x4(ac, Some(&dc), chroma_q, MB_CHROMA / 4, &mut dst);
    }
}

/// The AC levels of a chroma component whose AC is not coded.
const NO_CHROMA_AC: [[i32; 16]; 4] = [[0; 16]; 4];

/// Add the coded luma residual of `res` (not `I_NxN`) to the prediction in
/// the macroblock `dst`.
fn add_luma_residual(dsp: &Dsp, dst: &mut PlaneViewMut<'_>, res: &Residual, q: &H264Quantizer) {
    if res.cbp_luma == 0 && res.kind != ResidualKind::Intra16x16 {
        return; // no luma levels
    }
    match res.kind {
        ResidualKind::Inter8x8 => {
            for (b8, levels) in res.luma_8x8.iter().enumerate() {
                if res.cbp_luma & (1 << b8) != 0 {
                    let mut quadrant = dst
                        .sub_view_mut(8 * (b8 % 2), 8 * (b8 / 2), 8, 8)
                        .expect("quadrant inside the macroblock");
                    dsp.h264_add_inverse_8x8(levels, q, &mut quadrant);
                }
            }
        }
        ResidualKind::Intra16x16 => {
            let dc = dsp.h264_dequantize_luma_dc(&res.luma_dc, q);
            dsp.h264_add_inverse_4x4(&res.luma, Some(&dc), q, MB / 4, dst);
        }
        _ => dsp.h264_add_inverse_4x4(&res.luma, None, q, MB / 4, dst),
    }
}

/// The `n` x `n` block of `plane` at `pos`, writable.
#[inline]
fn block_mut(plane: &mut Plane, pos: (usize, usize), n: usize) -> PlaneViewMut<'_> {
    plane
        .region_mut(pos.0 as isize, pos.1 as isize, n, n)
        .expect("block inside the picture")
}

/// Residual `source - pred` of the `size` x `size` block at `pos` into the
/// first `size * size` entries of `out`, raster with stride `size`.
fn residual_into(
    dsp: &Dsp,
    source: &Plane,
    pos: (usize, usize),
    pred: &[u8],
    size: usize,
    out: &mut [i16; MB * MB],
) {
    let block = BlockSize::new(size, size).expect("macroblock sizes are kernel sizes");
    let src = source
        .region(pos.0 as isize, pos.1 as isize, size, size)
        .expect("block inside the picture");
    let pred = PlaneView::new(pred, size, size, size).expect("prediction covers the block");
    dsp.residual(block, &src, &pred, out, size);
}

/// Copy the `N` x `N` prediction (raster, stride `N`) into the block of
/// `plane` at `pos`.
fn copy_prediction<const N: usize>(plane: &mut Plane, pos: (usize, usize), pred: &[u8]) {
    copy_rows::<N>(plane, pos, pred, N);
}

/// Copy the 16x16 luma and 8x8 chroma blocks `src` (e.g. reference picture
/// blocks: a whole-sample skip's prediction) into `recon` at macroblock
/// `(x, y)`.
pub fn write_prediction_views(
    recon: &mut Picture,
    (x, y): (usize, usize),
    luma: &PlaneView<'_>,
    chroma: &[PlaneView<'_>; 2],
) {
    copy_rows::<MB>(
        recon.plane_mut(PlaneIndex::Y),
        (x, y),
        luma.data(),
        luma.stride(),
    );
    for (src, plane) in chroma.iter().zip([PlaneIndex::U, PlaneIndex::V]) {
        copy_rows::<MB_CHROMA>(
            recon.plane_mut(plane),
            (x / 2, y / 2),
            src.data(),
            src.stride(),
        );
    }
}

/// Copy `N` rows of `N` samples (`src` rows `src_stride` apart) into the
/// block of `plane` at `pos`.
fn copy_rows<const N: usize>(
    plane: &mut Plane,
    pos: (usize, usize),
    src: &[u8],
    src_stride: usize,
) {
    copy_square::<N>(&mut block_mut(plane, pos, N), src, src_stride);
}

/// Copy the prediction into `recon` unchanged (`P_Skip`).
pub fn write_prediction(recon: &mut Picture, pred: &Prediction, x: usize, y: usize) {
    copy_prediction::<MB>(recon.plane_mut(PlaneIndex::Y), (x, y), &pred.luma);
    for (c, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
        copy_prediction::<MB_CHROMA>(recon.plane_mut(plane), (x / 2, y / 2), &pred.chroma[c]);
    }
}

/// Coefficient elimination thresholds of the reference encoder (JM
/// `_LUMA_COEFF_COST_`, `_LUMA_MB_COEFF_COST_`, `_CHROMA_COEFF_COST_`):
/// an inter 8x8 luma block whose levels cost less than `LUMA_8X8` is dropped,
/// all luma is dropped if the macroblock total is below `LUMA_MB`, and chroma
/// AC below `CHROMA_AC`. Sparse ±1 levels cost more bits (cbp, significance
/// map, contexts) than the distortion they remove, especially at low rates.
/// The thresholds are tunable (`h264.eliminate_*`, see [`crate::H264Params`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Elimination {
    /// Drop an inter 8x8 luma block costing less than this.
    pub luma_8x8: u32,
    /// Drop all inter luma when the macroblock total is below this.
    pub luma_mb: u32,
    /// Drop inter chroma AC costing less than this.
    pub chroma_ac: u32,
}

/// Cost of a lone ±1 level by the number of zeros before it in scan order
/// (JM `COEFF_COST4x4`); any level beyond ±1 makes the block "expensive".
const COEFF_COST_BY_RUN: [u32; 16] = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const EXPENSIVE_COEFF_COST: u32 = 9;

/// The elimination cost of one 4x4 block's levels (raster), skipping the
/// first `first` scan positions.
fn coefficient_cost(levels: &[i32; 16], first: usize) -> u32 {
    if all_zero(levels) {
        return 0;
    }
    let mut cost = 0;
    let mut run = 0usize;
    for &pos in &ZIGZAG_4X4[first..] {
        match levels[pos].unsigned_abs() {
            0 => run += 1,
            1 => {
                cost += COEFF_COST_BY_RUN[run];
                run = 0;
            }
            _ => return EXPENSIVE_COEFF_COST,
        }
    }
    cost
}

/// Drop cheap, sparse inter luma residual (per 8x8, then the whole
/// macroblock) and return `CodedBlockPatternLuma`.
fn eliminate_sparse_luma(levels: &mut [[i32; 16]; 16], limits: &Elimination) -> u8 {
    let mut total = 0;
    for b8 in 0..4 {
        let blocks = blocks_of_8x8(b8);
        let cost: u32 = blocks
            .iter()
            .map(|&blk| coefficient_cost(&levels[blk], 0))
            .sum();
        if cost < limits.luma_8x8 {
            for blk in blocks {
                levels[blk] = [0; 16];
            }
        } else {
            total += cost;
        }
    }
    if total < limits.luma_mb {
        *levels = [[0; 16]; 16];
    }
    (0..4).fold(0u8, |cbp, b8| {
        let nonzero = blocks_of_8x8(b8).iter().any(|&blk| !all_zero(&levels[blk]));
        cbp | (u8::from(nonzero) << b8)
    })
}

/// Cost of a lone ±1 level in an 8x8 block by the number of zeros before it
/// in scan order (JM `COEFF_COST8x8`, frame scan).
const COEFF_COST_8X8_BY_RUN: [u32; 24] = [
    3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// The elimination cost of one 8x8 block's levels (raster).
fn coefficient_cost_8x8(levels: &[i32; 64], nonzero: u32) -> u32 {
    // The quantiser counted the non-zero levels: the scan stops after the
    // last one instead of walking all 64 positions (most coded blocks hold
    // a handful of low-frequency levels).
    let mut remaining = nonzero;
    let mut cost = 0;
    let mut run = 0usize;
    for &pos in &ZIGZAG_8X8 {
        if remaining == 0 {
            break;
        }
        match levels[pos].unsigned_abs() {
            0 => run += 1,
            1 => {
                cost += COEFF_COST_8X8_BY_RUN.get(run).copied().unwrap_or(0);
                run = 0;
                remaining -= 1;
            }
            _ => return EXPENSIVE_COEFF_COST,
        }
    }
    cost
}

/// [`eliminate_sparse_luma`] for 8x8 transform blocks, with the same
/// per-8x8 and per-macroblock thresholds. `nonzero` holds each block's
/// count of non-zero levels (from the quantiser).
/// Zero the levels below `min_level` in magnitude (the planned step's dead
/// zone, [`MbQuantizers::with_planned_step`]). Below the λ-headroom cap
/// `min_level` is 1, which zeroes nothing, so the pass is skipped (it was
/// ~1% of P-frame instructions, run over every coded block).
fn apply_planned_dead_zone(levels: &mut [i32], min_level: i32) {
    if min_level <= NO_DEAD_ZONE {
        return;
    }
    for level in levels.iter_mut() {
        if level.abs() < min_level {
            *level = 0;
        }
    }
}

/// The smallest `min_level` of [`apply_planned_dead_zone`]: every non-zero
/// level is kept.
const NO_DEAD_ZONE: i32 = 1;

fn eliminate_sparse_luma_8x8(
    levels: &mut [[i32; 64]; 4],
    mut nonzero: [u32; 4],
    limits: &Elimination,
) -> u8 {
    let mut total = 0;
    for (block, count) in levels.iter_mut().zip(nonzero.iter_mut()) {
        if *count == 0 {
            continue;
        }
        let cost = coefficient_cost_8x8(block, *count);
        if cost < limits.luma_8x8 {
            *block = [0; 64];
            *count = 0;
        } else {
            total += cost;
        }
    }
    if total < limits.luma_mb {
        if nonzero.iter().any(|&c| c != 0) {
            *levels = [[0; 64]; 4];
        }
        return 0;
    }
    (0..4).fold(0u8, |cbp, b8| cbp | (u8::from(nonzero[b8] != 0) << b8))
}

/// Drop cheap, sparse inter chroma AC residual (both components together).
fn eliminate_sparse_chroma_ac(ac: &mut [[[i32; 16]; 4]; 2], limits: &Elimination) {
    let cost: u32 = ac
        .iter()
        .flatten()
        .map(|block| coefficient_cost(block, 1))
        .sum();
    if cost < limits.chroma_ac {
        *ac = [[[0; 16]; 4]; 2];
    }
}

/// True if every level is zero. Written as an OR-reduction rather than
/// `iter().any(..)` so it vectorises (no early exit per element); most
/// blocks at video bitrates are all zero, so this is the common test.
pub fn all_zero(levels: &[i32]) -> bool {
    levels.iter().fold(0, |acc, &c| acc | c) == 0
}

/// The four 4x4 blocks (`by * 4 + bx`) of 8x8 block `b8` (raster quadrants).
pub fn blocks_of_8x8(b8: usize) -> [usize; 4] {
    let (x, y) = (2 * (b8 % 2), 2 * (b8 / 2));
    [
        y * 4 + x,
        y * 4 + x + 1,
        (y + 1) * 4 + x,
        (y + 1) * 4 + x + 1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_skip_scales_preserve_every_quantizer_bound() {
        let elimination = Elimination {
            luma_8x8: 4,
            luma_mb: 6,
            chroma_ac: 4,
        };
        for qp in 0..=51 {
            for divisor in 2..=16 {
                let cached = MbQuantizers::new(qp, elimination, divisor, 1.0, 1.0, 0);
                let (luma, chroma) = cached.for_kind(ResidualKind::Inter);
                let legacy = [
                    luma.zero_sad_threshold_4x4(),
                    chroma.zero_sad_threshold_4x4(),
                    chroma.zero_sad_threshold_chroma_dc(),
                ];
                assert_eq!(
                    cached.inter_zero_limits, legacy,
                    "QP {qp}, divisor {divisor}"
                );
            }
        }
    }
}
