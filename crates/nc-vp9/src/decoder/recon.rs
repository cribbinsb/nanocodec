//! Reconstruction of one block: intra prediction per transform block with
//! VP9's frame-edge rules, inter prediction (sub-8x8, compound, scaled
//! references), coefficient tokens and the inverse transform (VP9 §8.5,
//! §8.6; libvpx `vp9_decodeframe.c` `dec_build_inter_predictors`,
//! `predict_and_reconstruct_intra_block`, `reconstruct_inter_block`,
//! `vp9_reconintra.c` `build_intra_predictors`, `vp9_detokenize.c`
//! `vp9_decode_block_tokens`, `vp9_scale.c`; BSD-3-Clause, see
//! `THIRD_PARTY.md`).

use nc_common::decode::{DecodeError, Result};
use nc_common::{Picture, Plane, PlaneIndex, PlaneView};
use nc_dsp::{
    VP9_MAX_SCALED_STEP, VP9_TAPS, VP9_TAPS_BEFORE, Vp9Filter, Vp9IntraEdges, Vp9IntraMode,
};

use super::block::{BlockInfo, REFS_PER_FRAME};
use super::frame::{BlockPosition, FrameDecoder, Tile};
use super::header::valid_reference_size;
use crate::bool_coder::BoolReader;
use crate::coeff::{self, Coder, Dequantizer, Extent, Model, PlaneType};
use crate::recon::{ResidualBuffers, TxBlock, tx_type};
use crate::transform::{Steps, TxSize, TxType};
use crate::types::{IntraMode, MI_SIZE, Mv};

/// `REF_SCALE_SHIFT`: fixed-point bits of a reference scale factor.
const REF_SCALE_SHIFT: u32 = 14;
const REF_NO_SCALE: i64 = 1 << REF_SCALE_SHIFT;
/// `SUBPEL_BITS`: 1/16-sample positions.
const SUBPEL_BITS: u32 = 4;
const SUBPEL_MASK: i32 = (1 << SUBPEL_BITS) - 1;
/// `VP9_INTERP_EXTEND`: filter support kept beyond a clamped vector.
const INTERP_EXTEND: i32 = 4;
/// Largest block side.
const MAX_SIDE: usize = 64;
/// Largest reference window: a 64-sample block at the largest step.
const MAX_WINDOW: usize = (((MAX_SIDE - 1) * VP9_MAX_SCALED_STEP + 15) >> 4) + VP9_TAPS;
/// Unavailable above samples (127) and left samples (129).
const BASE_ABOVE: u8 = 127;
const BASE_LEFT: u8 = 129;
/// Largest transform side.
const MAX_TX_SIDE: usize = 32;

/// libvpx `INTERP_FILTER` numbers to kernels.
fn kernel(filter: u8) -> Vp9Filter {
    [
        Vp9Filter::Regular,
        Vp9Filter::Smooth,
        Vp9Filter::Sharp,
        Vp9Filter::Bilinear,
    ][usize::from(filter)]
}

/// Scale factors of a reference of another size (`struct scale_factors`).
#[derive(Debug, Clone, Copy)]
struct Scale {
    x_fp: i64,
    y_fp: i64,
    x_step: i32,
    y_step: i32,
}

impl Scale {
    fn x(&self, value: i32) -> i32 {
        ((i64::from(value) * self.x_fp) >> REF_SCALE_SHIFT) as i32
    }

    fn y(&self, value: i32) -> i32 {
        ((i64::from(value) * self.y_fp) >> REF_SCALE_SHIFT) as i32
    }
}

/// A reference picture as one frame sees it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Reference<'a> {
    /// Each plane, resolved once per frame.
    planes: [ReferencePlane<'a>; 3],
    scale: Scaling,
}

/// One plane of a reference, resolved once per frame: windows inside its
/// padded area are read in place (one range check per prediction).
#[derive(Debug, Clone, Copy)]
struct ReferencePlane<'a> {
    /// The whole padded area.
    padded: PlaneView<'a>,
    /// The visible area, for windows that leave the padded one.
    visible: PlaneView<'a>,
    /// Padding columns and rows before the visible area.
    border: (usize, usize),
    /// Visible size of the reference plane (`y_crop_width`,
    /// `y_crop_height`, halved and rounded up for chroma).
    crop: (usize, usize),
}

impl ReferencePlane<'_> {
    /// The `span` window at `corner` (visible coordinates), as samples
    /// `stride` apart, if it lies inside the padded area.
    fn padded_window(
        &self,
        (left, top): (i32, i32),
        (w, h): (usize, usize),
    ) -> Option<(&[u8], usize)> {
        let col = usize::try_from(left as isize + self.border.0 as isize).ok()?;
        let row = usize::try_from(top as isize + self.border.1 as isize).ok()?;
        let padded = &self.padded;
        (col + w <= padded.width() && row + h <= padded.height()).then(|| {
            (
                &padded.data()[row * padded.stride() + col..],
                padded.stride(),
            )
        })
    }

    /// The `span` window at `corner`: in place inside the padded area, else
    /// rebuilt in `window` with clamped edges (rows `span.0` apart).
    fn window<'w>(
        &'w self,
        corner: (i32, i32),
        span: (usize, usize),
        window: &'w mut [u8],
    ) -> (&'w [u8], usize) {
        if let Some(padded) = self.padded_window(corner, span) {
            return padded;
        }
        nc_dsp::emulate_edges(
            (self.visible.data(), self.visible.stride()),
            self.crop,
            (corner.0 as isize, corner.1 as isize),
            span,
            (window, span.0),
        );
        (window, span.0)
    }
}

impl<'a> Reference<'a> {
    /// `vp9_setup_scale_factors_for_frame` for a frame of `frame` size.
    pub(super) fn new(
        picture: &'a Picture,
        (width, height): (usize, usize),
        frame: (usize, usize),
    ) -> Self {
        let scale = if valid_reference_size((width, height), frame) {
            let mut scale = Scale {
                x_fp: ((width as i64) << REF_SCALE_SHIFT) / frame.0 as i64,
                y_fp: ((height as i64) << REF_SCALE_SHIFT) / frame.1 as i64,
                x_step: 0,
                y_step: 0,
            };
            scale.x_step = scale.x(16);
            scale.y_step = scale.y(16);
            if scale.x_fp == REF_NO_SCALE && scale.y_fp == REF_NO_SCALE {
                Scaling::None
            } else {
                Scaling::Scaled(scale)
            }
        } else {
            Scaling::Invalid
        };
        let planes = PlaneIndex::ALL.map(|index| {
            let plane = picture.plane(index);
            let ss = usize::from(index != PlaneIndex::Y);
            ReferencePlane {
                padded: plane.padded_view(),
                visible: plane.view(),
                border: (plane.border_x(), plane.border_y()),
                crop: ((width + ss) >> ss, (height + ss) >> ss),
            }
        });
        Self { planes, scale }
    }
}

/// How a reference maps onto the frame.
#[derive(Debug, Clone, Copy)]
enum Scaling {
    /// Same scale: read in place.
    None,
    Scaled(Scale),
    /// Beyond 2x larger or 16x smaller (`REF_INVALID_SCALE`).
    Invalid,
}

/// Scratch of inter prediction ([`predict_from`]), reused across blocks.
#[derive(Debug)]
pub(super) struct InterScratch {
    /// A reference window rebuilt with clamped edges.
    window: Vec<u8>,
    /// The scaled prediction's working memory.
    scaled: nc_dsp::Vp9ScaledScratch,
}

impl InterScratch {
    pub(super) fn new() -> Self {
        Self {
            window: vec![0; MAX_WINDOW * (MAX_WINDOW + VP9_TAPS - 1)],
            scaled: nc_dsp::Vp9ScaledScratch::default(),
        }
    }
}

/// One prediction request inside a block.
struct PredictionBlock {
    plane: PlaneIndex,
    /// The block's top-left in plane samples, the sub-block's offset and
    /// size, and the whole block's plane size.
    origin: (i32, i32),
    offset: (i32, i32),
    size: (usize, usize),
    block_size: (i32, i32),
    mv: Mv,
}

/// Where a prediction goes: picture samples `stride` apart, written or
/// averaged with what is there.
struct Target<'a> {
    block: &'a mut [u8],
    stride: usize,
    average: bool,
}

impl Target<'_> {
    /// Interpolate into the target (see [`nc_dsp::Dsp::vp9_convolve_into`]).
    fn convolve(
        self,
        dsp: &nc_dsp::Dsp,
        kind: Vp9Filter,
        (window, window_stride): (&[u8], usize),
        phase: (usize, usize),
        size: (usize, usize),
    ) {
        if self.average {
            dsp.vp9_convolve_avg_into(
                kind,
                window,
                window_stride,
                phase,
                size,
                self.block,
                self.stride,
            );
        } else {
            dsp.vp9_convolve_into(
                kind,
                window,
                window_stride,
                phase,
                size,
                self.block,
                self.stride,
            );
        }
    }
}

/// `dec_build_inter_predictors` from a reference of the frame's size: the
/// `size` block at 1/16-sample position `(x16, y16)` of `plane` into
/// `target`. References are extended ([`extend_reference`]), so a window
/// inside the padded plane reads as the clamped visible one.
#[allow(
    clippy::inline_always,
    reason = "one call per prediction; keeps the block loop's invariants in registers"
)]
#[inline(always)]
fn predict_unscaled(
    dsp: &nc_dsp::Dsp,
    plane: &ReferencePlane<'_>,
    kind: Vp9Filter,
    (x16, y16): (i32, i32),
    (w, h): (usize, usize),
    window: &mut [u8],
    target: Target<'_>,
) {
    let span = (w + VP9_TAPS - 1, h + VP9_TAPS - 1);
    let before = VP9_TAPS_BEFORE as i32;
    let corner = ((x16 >> SUBPEL_BITS) - before, (y16 >> SUBPEL_BITS) - before);
    let phase = ((x16 & SUBPEL_MASK) as usize, (y16 & SUBPEL_MASK) as usize);
    target.convolve(dsp, kind, plane.window(corner, span, window), phase, (w, h));
}

/// The prediction of a block of 8x8 or more from references of the frame's
/// size (almost all blocks): one prediction per plane and reference, without
/// the sub-block grid or the scaled path's state in the loop.
#[allow(
    clippy::inline_always,
    reason = "one call per block; out of line, the plane loop measured slower (x86 +0.9 %)"
)]
#[inline(always)]
fn predict_whole_block(
    dsp: &nc_dsp::Dsp,
    planes: &mut [Plane; 3],
    (references, kind): (&[Option<&Reference<'_>>; 2], Vp9Filter),
    (info, position): (&BlockInfo, &BlockPosition),
    window: &mut [u8],
) {
    for (plane, picture_plane) in PlaneIndex::ALL.into_iter().zip(planes) {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (n4w, n4h) = position.plane_size_4x4(ss);
        let size = (n4w * 4, n4h * 4);
        let (x, y) = (
            ((position.mi_col * MI_SIZE) >> ss) as i32,
            ((position.mi_row * MI_SIZE) >> ss) as i32,
        );
        let stride = picture_plane.stride();
        let unit = 1 << (1 - ss);
        for (r, reference) in references.iter().flatten().enumerate() {
            let mv = info.mv[r];
            let block = picture_plane
                .samples_at_mut(x as isize, y as isize, size.0, size.1)
                .expect("prediction blocks lie inside the plane");
            let target = Target {
                block,
                stride,
                average: r == 1,
            };
            // Positions in 1/16 plane samples.
            let at = (x * 16 + mv.col * unit, y * 16 + mv.row * unit);
            let plane_ref = &reference.planes[plane as usize];
            predict_unscaled(dsp, plane_ref, kind, at, size, window, target);
        }
    }
}

/// A block's references (the second for compound blocks), each checked once
/// per block to be present and of a valid size.
fn block_references<'r>(
    frame_references: &'r [Option<Reference<'r>>; REFS_PER_FRAME],
    info: &BlockInfo,
) -> Result<[Option<&'r Reference<'r>>; 2]> {
    let refs = 1 + usize::from(info.is_compound());
    let mut references = [None; 2];
    for (slot, &frame) in references.iter_mut().zip(&info.ref_frame).take(refs) {
        let reference = frame_references[(frame - 1) as usize]
            .as_ref()
            .ok_or(DecodeError::Malformed("VP9 missing reference"))?;
        if matches!(reference.scale, Scaling::Invalid) {
            return Err(DecodeError::Malformed("VP9 reference of invalid size"));
        }
        *slot = Some(reference);
    }
    Ok(references)
}

/// `dec_build_inter_predictors` from a reference of another size into
/// `target`. Out of line: only spatial layers and size changes use it.
#[inline(never)]
fn predict_scaled(
    dsp: &nc_dsp::Dsp,
    (scale, plane, kind): (&Scale, &ReferencePlane<'_>, Vp9Filter),
    position: &BlockPosition,
    request: &PredictionBlock,
    scratch: &mut InterScratch,
    Target {
        block,
        stride,
        average,
    }: Target<'_>,
) {
    let ss = u32::from(request.plane != PlaneIndex::Y);
    let (w, h) = request.size;
    let (x, y) = scaled_position(scale, position, request, ss);
    let (x_phase, y_phase) = ((x & SUBPEL_MASK) as usize, (y & SUBPEL_MASK) as usize);
    let (xs, ys) = (scale.x_step as usize, scale.y_step as usize);
    // The footprint plus the rows above and below that the scaled
    // driver's horizontal pass reads (see `vp9_convolve_scaled_into`).
    let span = (
        ((x_phase + (w - 1) * xs) >> SUBPEL_BITS) + VP9_TAPS,
        ((y_phase + (h - 1) * ys) >> SUBPEL_BITS) + 2 * VP9_TAPS - 1,
    );
    let before = VP9_TAPS_BEFORE as i32;
    let corner = ((x >> SUBPEL_BITS) - before, (y >> SUBPEL_BITS) - 2 * before);
    dsp.vp9_convolve_scaled_into(
        kind,
        plane.window(corner, span, &mut scratch.window),
        (x_phase, xs),
        (y_phase, ys),
        (w, h),
        (block, stride),
        average,
        &mut scratch.scaled,
    );
}

/// Luma border of decoded pictures (chroma: half): inter predictions whose
/// filter window lies within it read the reference in place, others are
/// rebuilt by [`nc_dsp::emulate_edges`]. 32, as libvpx's decoder: against
/// 64 and 16 (2026-11-12, pinned CPU time and callgrind) it was as fast as
/// 64 at 720p and 3 % faster at 1080p (less border extension), and 16 was
/// no faster; the 1080p peak RSS was 25.9 MB against 28.0 MB with 64.
pub(super) const REFERENCE_BORDER: usize = 32;

/// Make a decoded `picture` of visible size `(width, height)` readable as
/// a reference: every sample beyond the visible area, in the block-aligned
/// area and in the border, becomes a copy of the nearest visible sample,
/// as the reference reads clamped coordinates (VP9 §8.5.2.4).
pub(super) fn extend_reference(picture: &mut Picture, (width, height): (usize, usize)) {
    for (index, plane) in picture.planes_mut().iter_mut().enumerate() {
        let ss = usize::from(index > 0);
        let (keep_w, keep_h) = ((width + ss) >> ss, (height + ss) >> ss);
        let mut view = plane.view_mut();
        let (plane_w, plane_h) = (view.width(), view.height());
        for y in 0..keep_h {
            let row = &mut view.row_mut(y)[..plane_w];
            let edge = row[keep_w - 1];
            row[keep_w..].fill(edge);
        }
        let stride = view.stride();
        for y in keep_h..plane_h {
            let (above, below) = view.data_mut().split_at_mut(y * stride);
            below[..plane_w].copy_from_slice(&above[(keep_h - 1) * stride..][..plane_w]);
        }
        plane.extend_borders();
    }
}

impl FrameDecoder<'_> {
    /// `dec_build_inter_predictors_sb`: every plane of an inter block.
    /// References, the filter and each plane's destination are resolved
    /// once per block; a block of 8x8 or more makes one prediction per
    /// plane and reference.
    pub(super) fn predict_inter_block(
        &mut self,
        info: &BlockInfo,
        position: &BlockPosition,
    ) -> Result<()> {
        let references = block_references(&self.inputs.references, info)?;
        let kind = kernel(info.filter);
        let sub8x8 = info.size.is_sub8x8();
        let dsp = self.inputs.dsp;
        let scratch = &mut self.scratch.inter;
        let unscaled = references
            .iter()
            .flatten()
            .all(|reference| matches!(reference.scale, Scaling::None));
        if !sub8x8 && unscaled {
            predict_whole_block(
                dsp,
                self.recon.planes_mut(),
                (&references, kind),
                (info, position),
                &mut scratch.window,
            );
            return Ok(());
        }
        for (plane, picture_plane) in PlaneIndex::ALL.into_iter().zip(self.recon.planes_mut()) {
            let ss = usize::from(plane != PlaneIndex::Y);
            let (n4w, n4h) = position.plane_size_4x4(ss);
            let origin = (
                ((position.mi_col * MI_SIZE) >> ss) as i32,
                ((position.mi_row * MI_SIZE) >> ss) as i32,
            );
            // Sub-8x8 blocks predict each 4x4 block with its own vector
            // (chroma: one 4x4 block with their mean).
            let (grid, size) = if sub8x8 {
                ((n4w, n4h), (4, 4))
            } else {
                ((1, 1), (n4w * 4, n4h * 4))
            };
            let stride = picture_plane.stride();
            for (r, reference) in references.iter().flatten().enumerate() {
                let plane_ref = &reference.planes[plane as usize];
                for row in 0..grid.1 {
                    for col in 0..grid.0 {
                        let offset = ((4 * col) as i32, (4 * row) as i32);
                        let mv = if !sub8x8 {
                            info.mv[r]
                        } else if ss == 0 {
                            info.sub_mvs[row * grid.0 + col][r]
                        } else {
                            average_sub_mvs(info, r)
                        };
                        let (x, y) = (origin.0 + offset.0, origin.1 + offset.1);
                        // The first prediction goes straight into the
                        // picture; a compound block's second averages into
                        // it (`(first + second + 1) >> 1`, `convolve_avg`).
                        let block = picture_plane
                            .samples_at_mut(x as isize, y as isize, size.0, size.1)
                            .expect("prediction blocks lie inside the plane");
                        let target = Target {
                            block,
                            stride,
                            average: r == 1,
                        };
                        match &reference.scale {
                            Scaling::None => {
                                // Positions in 1/16 plane samples.
                                let unit = 1 << (1 - ss);
                                let at = (x * 16 + mv.col * unit, y * 16 + mv.row * unit);
                                predict_unscaled(
                                    dsp,
                                    plane_ref,
                                    kind,
                                    at,
                                    size,
                                    &mut scratch.window,
                                    target,
                                );
                            }
                            Scaling::Scaled(scale) => {
                                let request = PredictionBlock {
                                    plane,
                                    origin,
                                    offset,
                                    size,
                                    block_size: ((n4w * 4) as i32, (n4h * 4) as i32),
                                    mv,
                                };
                                predict_scaled(
                                    dsp,
                                    (scale, plane_ref, kind),
                                    position,
                                    &request,
                                    scratch,
                                    target,
                                );
                            }
                            Scaling::Invalid => unreachable!("checked above"),
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Plane geometry of `plane` for transform-block loops.
    fn plane_tx(info: &BlockInfo, plane: usize) -> TxSize {
        if plane == 0 {
            info.tx_size
        } else {
            info.size.uv_tx(info.tx_size)
        }
    }

    /// Quantizer steps of `plane` in the block's segment.
    fn steps(&self, segment: u8, plane: usize) -> Steps {
        self.steps[usize::from(segment)][usize::from(plane > 0)]
    }

    /// `predict_and_reconstruct_intra_block` over every transform block.
    pub(super) fn reconstruct_intra(
        &mut self,
        reader: &mut BoolReader<'_>,
        tile: &mut Tile,
        info: &BlockInfo,
        position: &BlockPosition,
    ) {
        for (plane, &index) in PLANES.iter().enumerate() {
            let ss = usize::from(plane > 0);
            let tx = Self::plane_tx(info, plane);
            let step = tx.units();
            let (max_w, max_h) = position.plane_limits_4x4(ss);
            let (n4w, _) = position.plane_size_4x4(ss);
            for row in (0..max_h).step_by(step) {
                for col in (0..max_w).step_by(step) {
                    let mode = if plane > 0 {
                        info.uv_mode
                    } else {
                        info.y_mode((row << 1) + col).intra()
                    };
                    let edges = IntraAvailability {
                        top: row > 0 || position.above.is_some(),
                        left: col > 0 || position.left.is_some(),
                        right: col + step < n4w,
                    };
                    self.predict_intra(plane, position, (col, row), tx, mode, edges);
                    if !info.skip {
                        let ty = tx_type(
                            index,
                            Some(mode),
                            tx,
                            self.inputs.header.quantizer.lossless(),
                        );
                        self.residual(reader, tile, info, position, (plane, col, row), tx, ty);
                    }
                }
            }
        }
    }

    /// Residuals of an inter block; returns whether any coefficient was coded.
    pub(super) fn reconstruct_inter(
        &mut self,
        reader: &mut BoolReader<'_>,
        tile: &mut Tile,
        info: &BlockInfo,
        position: &BlockPosition,
    ) -> bool {
        let mut coded = false;
        for plane in 0..PLANES.len() {
            coded |= self.reconstruct_inter_plane(reader, tile, info, position, plane);
        }
        coded
    }

    /// Inter transforms have one type and model throughout a block-plane.
    /// Resolve their token setup once, as in libvpx's `vp9_decode_block_tokens`.
    // Keep the transform loop isolated instead of expanding the block decoder.
    #[inline(never)]
    fn reconstruct_inter_plane(
        &mut self,
        reader: &mut BoolReader<'_>,
        tile: &mut Tile,
        info: &BlockInfo,
        position: &BlockPosition,
        plane: usize,
    ) -> bool {
        let ss = usize::from(plane > 0);
        let tx = Self::plane_tx(info, plane);
        let step = tx.units();
        let lossless = self.inputs.header.quantizer.lossless();
        let ty = tx_type(PLANES[plane], None, tx, lossless);
        let (bx, by) = position.plane_4x4(ss);
        let (max_w, max_h) = position.plane_limits_4x4(ss);
        let left_mask = (16 >> ss) - 1;
        let steps = self.steps(info.segment_id, plane);
        let model = Model::of(PLANES[plane], tx, ty, true);
        let setup = coeff::ReadSetup::new(
            model,
            Dequantizer {
                dc: steps.dc,
                ac: steps.ac,
                shift: tx.dequant_shift(),
            },
        );
        let mut coder = Coder::new(model, self.inputs.context, &mut self.counts);
        let above_plane = &mut self.above_nonzero[plane];
        let left_plane = &mut tile.left_nonzero[plane];
        let scratch = &mut *self.scratch;
        let mut coded = false;
        for row in (0..max_h).step_by(step) {
            let ly = (by + row) & left_mask;
            for col in (0..max_w).step_by(step) {
                let above = &mut above_plane[bx + col..bx + col + step];
                let left = &mut left_plane[ly..ly + step];
                let ctx = usize::from(any_set(above)) + usize::from(any_set(left));
                let extent = coeff::read_prepared(
                    reader,
                    &mut scratch.coefficients,
                    &mut scratch.token_cache,
                    &setup,
                    &mut coder,
                    ctx,
                );
                let nonzero = extent.eob > 0;
                // Units outside the coded frame must remain zero, even for a
                // partially visible transform with nonzero coefficients.
                set_contexts(above, u8::from(nonzero), max_w.saturating_sub(col));
                set_contexts(left, u8::from(nonzero), max_h.saturating_sub(row));
                if nonzero {
                    let block = TxBlock {
                        plane: PLANES[plane],
                        x: (bx + col) * 4,
                        y: (by + row) * 4,
                        tx,
                        offset_x: 0,
                        offset_y: 0,
                        block_side: 0,
                    };
                    add_residual(
                        self.inputs.dsp,
                        &mut scratch.buffers,
                        &mut self.recon,
                        &block,
                        (ty, lossless),
                        &scratch.coefficients,
                        extent,
                    );
                    setup.clear(&mut scratch.coefficients, extent.eob);
                    coded = true;
                }
            }
        }
        coded
    }

    /// `vp9_decode_block_tokens` and the inverse transform of the
    /// transform block at 4x4 offset `(col, row)` of the block in `plane`.
    #[allow(
        clippy::too_many_arguments,
        reason = "one transform block's full context"
    )]
    fn residual(
        &mut self,
        reader: &mut BoolReader<'_>,
        tile: &mut Tile,
        info: &BlockInfo,
        position: &BlockPosition,
        (plane, col, row): (usize, usize, usize),
        tx: TxSize,
        ty: TxType,
    ) -> bool {
        let steps = self.steps(info.segment_id, plane);
        let dequantizer = Dequantizer {
            dc: steps.dc,
            ac: steps.ac,
            shift: tx.dequant_shift(),
        };
        let ss = usize::from(plane > 0);
        let (bx, by) = position.plane_4x4(ss);
        let (max_w, max_h) = position.plane_limits_4x4(ss);
        let n = tx.units();
        let (ax, ly) = (bx + col, (by + row) & ((16 >> ss) - 1));
        let above = &mut self.above_nonzero[plane][ax..ax + n];
        let left = &mut tile.left_nonzero[plane][ly..ly + n];
        let ctx = usize::from(any_set(above)) + usize::from(any_set(left));
        let model = Model {
            tx,
            ty,
            plane: if plane == 0 {
                PlaneType::Y
            } else {
                PlaneType::Uv
            },
            inter: info.is_inter(),
        };
        let mut coder = Coder::new(model, self.inputs.context, &mut self.counts);
        let extent = coeff::read(
            reader,
            &mut self.scratch.coefficients[..],
            &mut self.scratch.token_cache,
            model,
            &mut coder,
            (ctx, dequantizer),
        );
        let eob = extent.eob;
        // `get_ctx_shift`: units beyond the frame edge stay zero.
        let nonzero = u8::from(eob > 0);
        set_contexts(above, nonzero, max_w.saturating_sub(col));
        set_contexts(left, nonzero, max_h.saturating_sub(row));
        if eob > 0 {
            let block = TxBlock {
                plane: PLANES[plane],
                x: (bx + col) * 4,
                y: (by + row) * 4,
                tx,
                offset_x: 0,
                offset_y: 0,
                block_side: 0,
            };
            let lossless = self.inputs.header.quantizer.lossless();
            add_residual(
                self.inputs.dsp,
                &mut self.scratch.buffers,
                &mut self.recon,
                &block,
                (ty, lossless),
                &self.scratch.coefficients[..],
                extent,
            );
            coeff::clear(&mut self.scratch.coefficients[..], model, eob);
        }
        eob > 0
    }

    /// `build_intra_predictors` of the transform block at 4x4 offset
    /// `(col, row)`: edges beyond the 8-aligned frame repeat its last
    /// sample (`frame_width` is the aligned width here).
    fn predict_intra(
        &mut self,
        plane: usize,
        position: &BlockPosition,
        (col, row): (usize, usize),
        tx: TxSize,
        mode: IntraMode,
        available: IntraAvailability,
    ) {
        let ss = usize::from(plane > 0);
        let (bx, by) = position.plane_4x4(ss);
        let (x, y) = ((bx + col) * 4, (by + row) * 4);
        let side = tx.side();
        let frame_w = (self.mi_cols * MI_SIZE) >> ss;
        let frame_h = (self.mi_rows * MI_SIZE) >> ss;
        // The block with its above row and left column (one offset check
        // instead of a view per transform block); `base` is sample (x, y).
        let picture_plane = self.recon.plane(PLANES[plane]);
        let stride = picture_plane.stride();
        // Only 4x4 blocks read above-right samples from the row.
        let above_width = if side == 4 { 2 * side } else { side };
        let region = picture_plane
            .samples_at(x as isize - 1, y as isize - 1, above_width + 1, side + 1)
            .expect("intra blocks lie inside the plane");
        let base = stride + 1;
        let mut edges = Vp9IntraEdges {
            above: [BASE_ABOVE; 2 * MAX_TX_SIDE + 1],
            left: [BASE_LEFT; MAX_TX_SIDE],
            has_above: available.top,
            has_left: available.left,
        };
        if available.top {
            let above = &region[1..];
            // Above-right samples come from the row only for 4x4 blocks
            // with `have_right`; otherwise they repeat the last above one.
            let count = if side == 4 && available.right {
                2 * side
            } else {
                side
            };
            // Samples past the aligned frame width repeat its last one.
            let end = (x + count).min(frame_w);
            let inside = end - x;
            if inside == count {
                // Inside the frame (almost always): fixed-size copies.
                match side {
                    4 => above_edge::<4>(&mut edges.above, above, count),
                    8 => above_edge::<8>(&mut edges.above, above, count),
                    16 => above_edge::<16>(&mut edges.above, above, count),
                    _ => above_edge::<32>(&mut edges.above, above, count),
                }
            } else {
                edges.above[1..=inside].copy_from_slice(&above[..inside]);
                edges.above[1 + inside..=count].fill(above[inside - 1]);
                let last_above = edges.above[side];
                edges.above[1 + count..=2 * side].fill(last_above);
            }
            edges.above[0] = if available.left { region[0] } else { BASE_LEFT };
        }
        if available.left {
            // Rows past the aligned frame height repeat its last one.
            let last = frame_h - 1 - y;
            for (i, sample) in edges.left[..side].iter_mut().enumerate() {
                *sample = region[base - 1 + i.min(last) * stride];
            }
        }
        let prediction = &mut self.scratch.intra[..side * side];
        self.inputs.dsp.vp9_intra_predict(
            Vp9IntraMode::ALL[mode as usize],
            side,
            &edges,
            prediction,
        );
        let (block, stride) = block_samples(&mut self.recon, PLANES[plane], (x, y), (side, side));
        match side {
            4 => store_rows::<4>(prediction, block, stride),
            8 => store_rows::<8>(prediction, block, stride),
            16 => store_rows::<16>(prediction, block, stride),
            _ => store_rows::<32>(prediction, block, stride),
        }
    }
}

/// The above edge of an `S`-wide block whose `count` above samples (`S`, or
/// `2 * S` with the above-right ones) lie in `row`: copied, and the rest of
/// the `2 * S` repeating the last above sample. Constant sizes keep these
/// inline moves (runtime ones were a `memcpy` and a `memset` call).
fn above_edge<const S: usize>(above: &mut [u8; 2 * MAX_TX_SIDE + 1], row: &[u8], count: usize) {
    let (near, far) = above[1..=2 * S].split_at_mut(S);
    near.copy_from_slice(&row[..S]);
    if count == 2 * S {
        far.copy_from_slice(&row[S..2 * S]);
    } else {
        far.fill(near[S - 1]);
    }
}

/// Copy packed `S`-wide rows into the block at the start of `block` (rows
/// `stride` apart). A constant width keeps each row inline vector moves (a
/// runtime width became a `memcpy` call per row).
fn store_rows<const S: usize>(packed: &[u8], block: &mut [u8], stride: usize) {
    for (line, row) in packed
        .as_chunks::<S>()
        .0
        .iter()
        .zip(block.chunks_mut(stride))
    {
        row[..S].copy_from_slice(line);
    }
}

const PLANES: [PlaneIndex; 3] = PlaneIndex::ALL;

/// Inverse-transform `coefficients` (dequantized, raster) into the picture
/// under `block` with the fused kernels (4x4: libvpx's DC-only and zero-row
/// shortcuts, `vpx_idct4x4_1_add`, `vp9_iht4x4_16_add`; the shared residual
/// path's 4x4 inverse wrote a residual block and added it in a second loop).
/// Lossless (WHT) blocks keep the shared path.
fn add_residual(
    dsp: &nc_dsp::Dsp,
    buffers: &mut ResidualBuffers,
    recon: &mut Picture,
    block: &TxBlock,
    (ty, lossless): (TxType, bool),
    coefficients: &[i32],
    extent: Extent,
) {
    if lossless {
        buffers.add_coefficients(recon, block, (ty, lossless), coefficients, extent);
        return;
    }
    let side = block.tx.side();
    dsp.vp9_inverse_add(
        block.tx.log2(),
        crate::transform::passes(ty),
        &coefficients[..block.tx.area()],
        (extent.rows, extent.columns),
        block_samples(recon, block.plane, (block.x, block.y), (side, side)),
    );
}

/// The samples of `plane` from `(x, y)` on, rows `stride` apart, for a
/// `w` x `h` block: one offset check per block instead of a plane view.
fn block_samples(
    picture: &mut Picture,
    plane: PlaneIndex,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
) -> (&mut [u8], usize) {
    let plane = picture.plane_mut(plane);
    let stride = plane.stride();
    let samples = plane
        .samples_at_mut(x as isize, y as isize, w, h)
        .expect("blocks lie inside the plane");
    (samples, stride)
}

/// `get_entropy_context`: whether any of a transform block's 1, 2, 4 or 8
/// coefficient contexts is set, read as one word as libvpx does.
fn any_set(contexts: &[u8]) -> bool {
    match contexts.len() {
        2 => u16::from_ne_bytes(contexts.try_into().expect("two contexts")) != 0,
        4 => u32::from_ne_bytes(contexts.try_into().expect("four contexts")) != 0,
        8 => u64::from_ne_bytes(contexts.try_into().expect("eight contexts")) != 0,
        _ => contexts.iter().any(|&v| v != 0),
    }
}

/// `dec_set_contexts`: set a transform block's contexts to `value`, those
/// from `inside` on (beyond the frame edge) to zero.
fn set_contexts(contexts: &mut [u8], value: u8, inside: usize) {
    if inside >= contexts.len() {
        // Fixed sizes: whole-word stores instead of a `memset` call.
        match contexts.len() {
            2 => set_all::<2>(contexts, value),
            4 => set_all::<4>(contexts, value),
            8 => set_all::<8>(contexts, value),
            _ => contexts.fill(value),
        }
    } else {
        let (set, outside) = contexts.split_at_mut(inside);
        set.fill(value);
        outside.fill(0);
    }
}

fn set_all<const N: usize>(contexts: &mut [u8], value: u8) {
    let contexts: &mut [u8; N] = contexts.try_into().expect("N contexts");
    *contexts = [value; N];
}

/// Which neighbours of a transform block intra prediction may read.
#[derive(Debug, Clone, Copy)]
struct IntraAvailability {
    top: bool,
    left: bool,
    /// The above-right samples lie inside the block (`have_right`).
    right: bool,
}

/// `mi_mv_pred_q4`: the chroma vector of a sub-8x8 block, the rounded
/// mean of its four 4x4 vectors.
fn average_sub_mvs(info: &BlockInfo, r: usize) -> Mv {
    let round = |sum: i32| (if sum < 0 { sum - 2 } else { sum + 2 }) / 4;
    let sum = |f: fn(&Mv) -> i32| info.sub_mvs.iter().map(|m| f(&m[r])).sum::<i32>();
    Mv {
        row: round(sum(|m| m.row)),
        col: round(sum(|m| m.col)),
    }
}

/// The scaled start of a sub-block in 1/16 reference samples
/// (`dec_build_inter_predictors` with `is_scaled`): the vector is clamped
/// near the frame (`clamp_mv_to_umv_border_sb`), the block position and
/// vector scale separately, and the vector takes the 1/16 phase of the
/// scaled luma-based position (`vp9_scale_mv`).
fn scaled_position(
    scale: &Scale,
    position: &BlockPosition,
    request: &PredictionBlock,
    ss: u32,
) -> (i32, i32) {
    let factor = 1 << (1 - ss);
    let (bw, bh) = request.block_size;
    let spel_left = (INTERP_EXTEND + bw) << SUBPEL_BITS;
    let spel_right = spel_left - (1 << SUBPEL_BITS);
    let spel_top = (INTERP_EXTEND + bh) << SUBPEL_BITS;
    let spel_bottom = spel_top - (1 << SUBPEL_BITS);
    let mv_row = (request.mv.row * factor).clamp(
        position.to_top * factor - spel_top,
        position.to_bottom * factor + spel_bottom,
    );
    let mv_col = (request.mv.col * factor).clamp(
        position.to_left * factor - spel_left,
        position.to_right * factor + spel_right,
    );
    let (x_start, y_start) = request.origin;
    let (ox, oy) = request.offset;
    let x0 = scale.x(x_start + ox);
    let y0 = scale.y(y_start + oy);
    let (mi_x, mi_y) = (
        (position.mi_col * MI_SIZE) as i32,
        (position.mi_row * MI_SIZE) as i32,
    );
    let col = scale.x(mv_col) + (scale.x((mi_x + ox) << SUBPEL_BITS) & SUBPEL_MASK);
    let row = scale.y(mv_row) + (scale.y((mi_y + oy) << SUBPEL_BITS) & SUBPEL_MASK);
    (
        ((x0 + (col >> SUBPEL_BITS)) << SUBPEL_BITS) + (col & SUBPEL_MASK),
        ((y0 + (row >> SUBPEL_BITS)) << SUBPEL_BITS) + (row & SUBPEL_MASK),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fused 4x4 kernel against the shared residual path (inverse, then
    /// add) for every type, DC-only and zero-row extents and extreme values.
    #[test]
    fn fused_4x4_matches_the_residual_path() {
        let dsp = nc_dsp::Dsp::new();
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for case in 0..4000 {
            let ty = [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
            ][case % 4];
            let rows = 1 + (next() as usize % 4);
            let columns = if case % 7 == 0 {
                1
            } else {
                1 + (next() as usize % 4)
            };
            let mut coefficients = [0i32; 16];
            for (i, c) in coefficients.iter_mut().enumerate() {
                if i / 4 < rows && i % 4 < columns {
                    *c = match next() % 5 {
                        0 => 0,
                        1 => i32::from(next() as i16),
                        2 => [i32::MAX, i32::MIN, 32767, -32768][next() as usize % 4],
                        _ => (next() % 2001) as i32 - 1000,
                    };
                }
            }
            let prediction: Vec<u8> = (0..4 * 6).map(|_| next() as u8).collect();
            let mut residual = [0i32; 16];
            crate::transform::inverse(&dsp, TxSize::Tx4, ty, false, &coefficients, &mut residual);
            let mut want = prediction.clone();
            for (y, row) in residual.as_chunks::<4>().0.iter().enumerate() {
                for (x, &r) in row.iter().enumerate() {
                    let sample = &mut want[y * 6 + x];
                    *sample = (i32::from(*sample) + r).clamp(0, 255) as u8;
                }
            }
            let mut got = prediction;
            dsp.vp9_inverse_add(
                2,
                crate::transform::passes(ty),
                &coefficients,
                (rows, columns),
                (&mut got, 6),
            );
            assert_eq!(got, want, "case {case} {ty:?} extent {rows}x{columns}");
        }
    }
}
