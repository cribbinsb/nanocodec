//! Intra blocks: their mode syntax (AV1 §5.11.7 `intra_frame_mode_info`,
//! §5.11.22 `intra_block_mode_info`, §5.11.42–46 angle, `CfL` and filter
//! intra info) and the prediction of each transform block (§7.11.2,
//! §7.11.5) with nc-dsp's kernels for every block shape.
use super::super::block_info::Neighbours;
use super::super::transform::chroma_block;
use super::{BlockPos, FrameDecoder};
use crate::backend::{ANGLE_DELTA_SYMBOLS, INTRA_MODES, Picture, UV_CFL_PRED};
use crate::decoder_range::SymbolReader;
use crate::intra::{FILTER_INTRA_MODES, LumaIntra, MAX_ANGLE_DELTA};
use crate::predict::{FilterIntraMode, IntraMode};
use crate::tables::{
    CFL_ALLOWED, CFL_NOT_ALLOWED, UV_INTRA_MODES_CFL_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED,
};
use nc_common::PlaneIndex;
use nc_common::decode::DecodeError;
use nc_dsp::intra_prediction::av1_block::cfl::CflAc;
use nc_dsp::intra_prediction::av1_block::{BlockEdges, EdgeAvail, EdgeFilterCtx, IntraKind};

/// Blocks of at most this side (log2) may use `CfL` and filter intra.
const SMALL_BLOCK_MAX_LOG2: u8 = 5;
/// log2 of the 4-sample side of the chroma blocks lossless `CfL` allows.
const MIN_TX_SIDE_LOG2: u8 = 2;
/// `CfL` joint-sign and magnitude alphabets.
const CFL_SIGN_SYMBOLS: usize = 8;
const CFL_ALPHA_SYMBOLS: usize = 16;
/// Sign classes of a joint sign: zero, negative, positive.
const CFL_SIGN_CLASSES: usize = 3;
const CFL_SIGN_NEGATIVE: usize = 1;
/// Degrees per angle-delta step (`ANGLE_STEP`).
const ANGLE_STEP: i16 = 3;
/// Largest side of a palette block (`MAX_PALETTE_BLOCK_WIDTH`).
const PALETTE_MAX_BLOCK: usize = 64;

/// An intra block's modes.
#[derive(Debug, Clone, Copy)]
pub(super) struct IntraModes {
    /// `YMode` with its angle delta and filter intra.
    pub luma: LumaIntra,
    /// `UVMode` (13 is `UV_CFL_PRED`).
    pub uv: usize,
    /// `AngleDeltaUV`.
    pub uv_angle_delta: i8,
    /// `CflAlphaU`, `CflAlphaV` of a `UV_CFL_PRED` block.
    pub alpha: [i8; 2],
}

impl IntraModes {
    /// Whether `plane`'s prediction of this block is a smooth mode (the
    /// intra edge filter's `get_filter_type`).
    pub fn smooth(&self, plane: PlaneIndex) -> bool {
        let mode = if plane == PlaneIndex::Y {
            self.luma.mode
        } else {
            IntraMode::from_symbol(self.uv).unwrap_or_default()
        };
        is_smooth(mode)
    }
}

/// `SMOOTH_PRED`, `SMOOTH_V_PRED` and `SMOOTH_H_PRED`.
pub(super) fn is_smooth(mode: IntraMode) -> bool {
    matches!(
        mode,
        IntraMode::Smooth | IntraMode::SmoothVertical | IntraMode::SmoothHorizontal
    )
}

/// A transform block to predict: its plane, top-left sample there, size,
/// availability and edge-filter context.
#[derive(Debug, Clone, Copy)]
pub(super) struct IntraTransform {
    pub plane: PlaneIndex,
    pub origin: (usize, usize),
    pub size: (usize, usize),
    pub avail: EdgeAvail,
    /// The intra edge filter's context; `None` when the sequence disables
    /// the filter.
    pub edge_filter: Option<EdgeFilterCtx>,
}

impl FrameDecoder<'_> {
    /// The intra modes of `block` (§5.11.7, §5.11.22): `YMode` (from the
    /// neighbours' modes in intra frames, by size group otherwise), its
    /// angle, then, where the block carries chroma, `UVMode` with `CfL`
    /// alphas or its angle, then filter intra.
    pub(super) fn intra_modes(
        &mut self,
        reader: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
        intra_frame: bool,
    ) -> Result<IntraModes, DecodeError> {
        let y = if intra_frame {
            let (above, left) = near.y_mode_contexts();
            reader.symbol(&mut self.cdfs.kf_y_mode[above][left], INTRA_MODES)
        } else {
            reader.symbol(&mut self.cdfs.y_mode[block.size.size_group()], INTRA_MODES)
        };
        let mode = IntraMode::from_symbol(y).ok_or(DecodeError::Malformed("AV1 intra mode"))?;
        let mut luma = LumaIntra::plain(mode);
        let angles = block.size.index() >= block_size_8x8_index();
        if angles && let Some(ctx) = mode.angle_context() {
            luma.angle_delta = self.angle_delta(reader, ctx);
        }
        let mut modes = IntraModes {
            luma,
            uv: 0,
            uv_angle_delta: 0,
            alpha: [0; 2],
        };
        if block.has_chroma() {
            // `is_cfl_allowed`: up to 32x32, or 4x4 chroma when lossless.
            let cfl_allowed = if self.header.lossless {
                chroma_block(block.size) == (MIN_TX_SIDE_LOG2, MIN_TX_SIDE_LOG2)
            } else {
                block.size.width_log2().max(block.size.height_log2()) <= SMALL_BLOCK_MAX_LOG2
            };
            let (allowed, symbols) = if cfl_allowed {
                (CFL_ALLOWED, UV_INTRA_MODES_CFL_ALLOWED)
            } else {
                (CFL_NOT_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED)
            };
            modes.uv = reader.symbol(&mut self.cdfs.uv_mode[allowed][y], symbols);
            if modes.uv == UV_CFL_PRED {
                modes.alpha = self.cfl_alphas(reader);
            } else if angles
                && let Some(ctx) =
                    IntraMode::from_symbol(modes.uv).and_then(IntraMode::angle_context)
            {
                modes.uv_angle_delta = self.angle_delta(reader, ctx);
            }
        }
        // palette_mode_info() (§5.11.46): blocks 8x8 (in `MiSize` order) to
        // 64x64.
        let mut palette_y = false;
        if let Some(palette) = &mut self.palette
            && block.size.index() >= block_size_8x8_index()
            && block.size.width().max(block.size.height()) <= PALETTE_MAX_BLOCK
        {
            let uv_dc = modes.uv == IntraMode::Dc.symbol();
            palette.read_mode_info(reader, &mut self.cdfs, block, mode == IntraMode::Dc, uv_dc);
            palette_y = palette.block.y.size > 0;
        }
        // filter_intra_mode_info() (§5.11.24).
        let small = block.size.width_log2().max(block.size.height_log2()) <= SMALL_BLOCK_MAX_LOG2;
        if self.filter_intra && mode == IntraMode::Dc && !palette_y && small {
            let cdf = &mut self.cdfs.filter_intra[block.size.index()];
            if reader.boolean(cdf) {
                let filter = reader.symbol(&mut self.cdfs.filter_intra_mode, FILTER_INTRA_MODES);
                modes.luma.filter = Some(FilterIntraMode::ALL[filter]);
            }
        }
        Ok(modes)
    }

    /// `angle_delta_y` / `angle_delta_uv` of the directional mode `ctx`.
    fn angle_delta(&mut self, reader: &mut SymbolReader<'_>, ctx: usize) -> i8 {
        let symbol = reader.symbol(&mut self.cdfs.angle_delta[ctx], ANGLE_DELTA_SYMBOLS);
        symbol as i8 - MAX_ANGLE_DELTA
    }

    /// `read_cfl_alphas()` (§5.11.45): `CflAlphaU`, `CflAlphaV`.
    pub(super) fn cfl_alphas(&mut self, r: &mut SymbolReader<'_>) -> [i8; 2] {
        let joint = r.symbol(&mut self.cdfs.cfl_sign, CFL_SIGN_SYMBOLS) + 1;
        let signs = [joint / CFL_SIGN_CLASSES, joint % CFL_SIGN_CLASSES];
        let mut alpha = [0; 2];
        for p in 0..2 {
            if signs[p] != 0 {
                let ctx = signs[p] * CFL_SIGN_CLASSES + signs[1 - p] - CFL_SIGN_CLASSES;
                let magnitude =
                    r.symbol(&mut self.cdfs.cfl_alpha[ctx], CFL_ALPHA_SYMBOLS) as i8 + 1;
                alpha[p] = if signs[p] == CFL_SIGN_NEGATIVE {
                    -magnitude
                } else {
                    magnitude
                };
            }
        }
        alpha
    }

    /// Predict one intra transform block (§7.11.2) into the
    /// reconstruction. `luma_extent` is the decoded luma (width, height)
    /// a `CfL` chroma block may read, from its luma origin.
    pub(super) fn predict_intra(
        &mut self,
        tx: IntraTransform,
        modes: &IntraModes,
        luma_extent: (usize, usize),
    ) -> Result<(), DecodeError> {
        let (x, y) = tx.origin;
        let (w, h) = tx.size;
        if let Some(palette) = &self.palette
            && palette.predict(tx.plane, tx.origin, &mut region(&mut self.recon, tx)?)
        {
            return Ok(());
        }
        let cfl = tx.plane != PlaneIndex::Y && modes.uv == UV_CFL_PRED;
        let ac = cfl.then(|| {
            let luma = self.recon.plane(PlaneIndex::Y);
            let view = luma
                .region(2 * x as isize, 2 * y as isize, 2 * w, 2 * h)
                .expect("chroma blocks lie inside the coded frame");
            CflAc::from_luma(
                &view,
                luma_extent.0.min(2 * w),
                luma_extent.1.min(2 * h),
                w,
                h,
            )
        });
        let filter = if tx.plane == PlaneIndex::Y {
            modes.luma.filter
        } else {
            None
        };
        if let Some(filter) = filter {
            let edges = BlockEdges::read(self.recon.plane(tx.plane), x, y, w, h, tx.avail);
            let dsp = self.dsp;
            let mut dst = self.destination(tx)?;
            dsp.av1_intra_predict_filter_block(&edges, filter, &mut dst);
            return Ok(());
        }
        // Read after the filter branch: built here, in place, the edges
        // are not moved (a memcpy per transform block when read earlier).
        let edges = BlockEdges::read(self.recon.plane(tx.plane), x, y, w, h, tx.avail);
        let kind = if cfl {
            IntraKind::Dc
        } else if tx.plane == PlaneIndex::Y {
            intra_kind(modes.luma.mode, modes.luma.angle_delta, tx.edge_filter)
        } else {
            let mode =
                IntraMode::from_symbol(modes.uv).ok_or(DecodeError::Malformed("AV1 UV mode"))?;
            intra_kind(mode, modes.uv_angle_delta, tx.edge_filter)
        };
        let dsp = self.dsp;
        let mut dst = self.destination(tx)?;
        dsp.av1_intra_predict_block(&edges, kind, &mut dst);
        if let Some(ac) = ac {
            let alpha = modes.alpha[tx.plane as usize - 1];
            ac.predict(i32::from(alpha), &mut dst);
        }
        Ok(())
    }

    fn destination(
        &mut self,
        tx: IntraTransform,
    ) -> Result<nc_common::PlaneViewMut<'_>, DecodeError> {
        region(&mut self.recon, tx)
    }
}

/// The samples of `tx` in `recon`.
fn region(
    recon: &mut Picture,
    tx: IntraTransform,
) -> Result<nc_common::PlaneViewMut<'_>, DecodeError> {
    recon
        .plane_mut(tx.plane)
        .region_mut(
            tx.origin.0 as isize,
            tx.origin.1 as isize,
            tx.size.0,
            tx.size.1,
        )
        .ok_or(DecodeError::Malformed("AV1 intra rectangle"))
}

/// `MiSize` of `BLOCK_8X8`: smaller blocks code no angle deltas.
fn block_size_8x8_index() -> usize {
    super::super::block_size::BlockSize::BLOCK_8X8.index()
}

/// The prediction of `mode` with angle delta `delta` (§7.11.2.4: `pAngle`
/// is the nominal angle plus `3 * delta`).
fn intra_kind(mode: IntraMode, delta: i8, edge_filter: Option<EdgeFilterCtx>) -> IntraKind {
    match mode {
        IntraMode::Dc => IntraKind::Dc,
        IntraMode::Smooth => IntraKind::Smooth,
        IntraMode::SmoothVertical => IntraKind::SmoothVertical,
        IntraMode::SmoothHorizontal => IntraKind::SmoothHorizontal,
        IntraMode::Paeth => IntraKind::Paeth,
        directional => {
            let nominal = directional.nominal_angle().expect("a directional mode");
            IntraKind::Directional {
                angle: (nominal as i16 + ANGLE_STEP * i16::from(delta)) as u16,
                edge_filter,
            }
        }
    }
}
