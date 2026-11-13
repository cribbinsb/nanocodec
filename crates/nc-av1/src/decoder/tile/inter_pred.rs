//! Prediction of inter blocks (AV1 §7.11.3): translation with the block's
//! interpolation filters (§7.11.3.3–4), local and global warps
//! (§7.11.3.5), chroma of blocks 4 samples wide or high from the vectors
//! of every block its 4x4 chroma covers (§7.11.3.1), overlapped block
//! motion compensation (§7.11.3.10), inter-intra (§7.11.3.13–14), and the
//! blends of compound blocks: average, distance weights (§7.11.3.15),
//! wedge and difference-weighted masks (§7.11.3.11–12, §7.11.3.14).
//!
//! Reference samples outside the frame repeat its edge (the spec's `Clip3`
//! of reference coordinates): reconstructions are edge-replicated beyond
//! the frame and carry extended borders; windows that reach past those
//! are gathered with clamped coordinates. References of another size than
//! the frame are read at scaled positions and steps (§7.11.3.3) and are
//! never warped (libaom `av1_allow_warp`).
//!
//! Block geometry follows libaom `reconinter_template.inc` and the decoder's
//! OBMC set-up (`dec_build_prediction_by_above_pred`; BSD-2-Clause, see
//! `THIRD_PARTY.md`).
use super::super::frame_header::WarpType;
use super::super::motion::MotionCell;
use super::inter::{
    CompoundType, InterIntra, InterModes, MAX_NEIGHBOR_OBMC, MotionMode, WarpModel, wedge_used,
};
use super::{BlockPos, FrameDecoder, MI_SIZE};
use crate::backend::{Av1Mv, Picture};
use crate::scaled_reference::scaled_window;
use nc_common::decode::DecodeError;
use nc_common::{Plane, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::Dsp;
use nc_dsp::av1_mc::blend::{
    Av1CompoundPred, Av1Mask, Av1ObmcEdge, av1_blend_average, av1_blend_distance,
    av1_blend_interintra, av1_blend_mask, av1_distance_weights, av1_obmc_blend,
};
use nc_dsp::av1_mc::convolve::Av1McAxis;
use nc_dsp::av1_mc::filter::{AV1_FILTER_BEFORE, AV1_FILTER_TAPS, Av1Filters};
use nc_dsp::av1_mc::mask::{
    Av1InterIntraMode, Av1WedgeMasks, av1_difference_weight_mask, av1_interintra_smooth_mask,
};
use nc_dsp::av1_mc::scale::Av1ScaleSizes;
use nc_dsp::av1_mc::warp::{Av1Shear, av1_setup_shear, av1_warp, av1_warp_compound};
use nc_dsp::intra_prediction::av1_block::{BlockEdges, EdgeAvail, IntraKind};

/// 1/16-sample positions per sample (`1 << SUBPEL_BITS`).
const SUBPEL_SHIFTS: i32 = 16;
const SUBPEL_BITS: u32 = 4;
/// Smallest side of a plane block that may be warped (8 samples).
const WARP_MIN_SIDE: usize = 8;
/// Smallest OBMC prediction across the shared edge, and the largest
/// overlap (half of a 64-sample side).
const OBMC_MIN_PREDICTION: usize = 4;
const OBMC_MAX_OVERLAP: usize = 32;
/// Nominal angles of the vertical and horizontal inter-intra modes.
const VERTICAL_ANGLE: u16 = 90;
const HORIZONTAL_ANGLE: u16 = 180;

/// Scratch the inter predictions reuse across blocks.
#[derive(Debug, Default)]
pub(in super::super) struct PredictionScratch {
    /// Reference windows gathered with clamped coordinates.
    pub(super) window: Vec<u8>,
    /// Neighbour predictions of OBMC and intra predictions of inter-intra.
    block: Vec<u8>,
    /// Smooth inter-intra masks and difference-weighted compound masks.
    mask: Vec<u8>,
    /// The two predictions of a compound block, at compound precision.
    compound: [Vec<i16>; 2],
    /// The wedge masks, built on first use.
    wedges: Option<Box<Av1WedgeMasks>>,
}

/// One plane's reference: its samples, its size in them, and the sizes
/// positions scale between when it is scaled.
#[derive(Clone, Copy)]
pub(super) struct PlaneReference<'p> {
    pub plane: &'p Plane,
    pub size: (usize, usize),
    pub scale: Option<Av1ScaleSizes>,
}

impl PlaneReference<'_> {
    /// The filter window of a `w`x`h` block at plane sample (`x`, `y`) of
    /// a plane subsampled by `sub`, displaced by luma vector `mv`: its
    /// top-left, its size, and each direction's phase and step.
    #[expect(
        clippy::inline_always,
        reason = "part of `translate`, which is always inlined"
    )]
    #[inline(always)]
    fn window(
        self,
        (x, y): (usize, usize),
        (w, h): (usize, usize),
        mv: Av1Mv,
        sub: (bool, bool),
    ) -> ((isize, isize), (usize, usize), (Av1McAxis, Av1McAxis)) {
        if let Some(sizes) = self.scale {
            let (origin, axes) = scaled_window(sizes, (x, y), mv, sub);
            return (origin, (axes.0.span(w), axes.1.span(h)), axes);
        }
        let (origin, axes) = window_position((x, y), mv, sub);
        let span = (w + AV1_FILTER_TAPS - 1, h + AV1_FILTER_TAPS - 1);
        (origin, span, axes)
    }
}

/// Where the block at plane sample (`x`, `y`) of a plane subsampled by
/// `sub`, displaced by luma vector `mv`, reads its reference (§7.11.3.3
/// unscaled): the top-left of its filter window and its subsample phases.
#[expect(
    clippy::inline_always,
    reason = "part of `translate`, which is always inlined"
)]
#[inline(always)]
pub(super) fn window_position(
    (x, y): (usize, usize),
    mv: Av1Mv,
    sub: (bool, bool),
) -> ((isize, isize), (Av1McAxis, Av1McAxis)) {
    let (origin, (frac_x, frac_y)) = unscaled_window((x, y), mv, sub);
    (
        origin,
        (Av1McAxis::unscaled(frac_x), Av1McAxis::unscaled(frac_y)),
    )
}

/// [`window_position`] as the top-left of the filter window and the
/// sub-sample offsets (1/16) of each direction.
#[inline]
fn unscaled_window(
    (x, y): (usize, usize),
    mv: Av1Mv,
    (sub_x, sub_y): (bool, bool),
) -> ((isize, isize), (usize, usize)) {
    // Positions in 1/16 samples of the plane (`mv * 2 >> sub`).
    let pos_x = ((x as i32) << SUBPEL_BITS) + i32::from(mv.col) * (2 >> u32::from(sub_x));
    let pos_y = ((y as i32) << SUBPEL_BITS) + i32::from(mv.row) * (2 >> u32::from(sub_y));
    let frac = |pos: i32| (pos & (SUBPEL_SHIFTS - 1)) as usize;
    let x0 = (pos_x >> SUBPEL_BITS) as isize - AV1_FILTER_BEFORE as isize;
    let y0 = (pos_y >> SUBPEL_BITS) as isize - AV1_FILTER_BEFORE as isize;
    ((x0, y0), (frac(pos_x), frac(pos_y)))
}

/// [`translate`] from an unscaled reference into `dst` (the block's samples
/// at its stride): its window read in place when it lies inside the
/// reference's borders, and the prediction through the kernels' direct
/// sample-buffer entry ([`Dsp::av1_put_samples`], as dav1d's `mc`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn translate_unscaled(
    dsp: &Dsp,
    reference: PlaneReference<'_>,
    window: &mut Vec<u8>,
    origin: (usize, usize),
    (w, h): (usize, usize),
    mv: Av1Mv,
    sub: (bool, bool),
    filters: Av1Filters,
    dst: (&mut [u8], usize),
) {
    let ((x0, y0), fracs) = unscaled_window(origin, mv, sub);
    let span = (w + AV1_FILTER_TAPS - 1, h + AV1_FILTER_TAPS - 1);
    let plane = reference.plane;
    if let Some(src) = plane.samples_at(x0, y0, span.0, span.1) {
        dsp.av1_put_samples(filters, (src, plane.stride()), (w, h), fracs, dst);
    } else {
        let view = gather(reference, window, (x0, y0), span);
        dsp.av1_put_samples(filters, (view.data(), view.stride()), (w, h), fracs, dst);
    }
}

/// Predict the `w`x`h` block at plane sample (`x`, `y`) of a plane
/// subsampled by `sub`, displaced by luma vector `mv`, with `filters`
/// (§7.11.3.3 unscaled, §7.11.3.4).
#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::inline_always,
    reason = "three times per inter block; out of line its set-up cost ~40M instructions on static 720p"
)]
#[inline(always)]
fn translate(
    dsp: &Dsp,
    reference: PlaneReference<'_>,
    window: &mut Vec<u8>,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Av1Mv,
    sub: (bool, bool),
    filters: Av1Filters,
    dst: &mut PlaneViewMut<'_>,
) {
    let ((x0, y0), (span_w, span_h), axes) = reference.window((x, y), (w, h), mv, sub);
    if let Some(view) = reference.plane.region(x0, y0, span_w, span_h) {
        dsp.av1_predict(filters, &view, (w, h), axes, dst);
        return;
    }
    let view = gather(reference, window, (x0, y0), (span_w, span_h));
    dsp.av1_predict(filters, &view, (w, h), axes, dst);
}

/// [`translate`] at compound precision into `dst` (`h` rows of `w`): one
/// of a compound block's two predictions.
#[allow(clippy::too_many_arguments)]
fn translate_compound(
    dsp: &Dsp,
    reference: PlaneReference<'_>,
    window: &mut Vec<u8>,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Av1Mv,
    sub: (bool, bool),
    filters: Av1Filters,
    dst: &mut [i16],
) {
    let ((x0, y0), span, axes) = reference.window((x, y), (w, h), mv, sub);
    let view = match reference.plane.region(x0, y0, span.0, span.1) {
        Some(view) => view,
        None => gather(reference, window, (x0, y0), span),
    };
    dsp.av1_predict_compound(filters, &view, (w, h), axes, dst, w);
}

/// The `w`x`h` window at (`x0`, `y0`) of a reference plane with clamped
/// coordinates (the shared `nc_dsp::emulate_edges`), in `buffer`: only for
/// vectors reaching past the border.
#[cold]
#[inline(never)]
pub(super) fn gather<'b>(
    reference: PlaneReference<'_>,
    buffer: &'b mut Vec<u8>,
    (x0, y0): (isize, isize),
    (w, h): (usize, usize),
) -> PlaneView<'b> {
    let view = reference.plane.view();
    buffer.clear();
    buffer.resize(w * h, 0);
    nc_dsp::emulate_edges(
        (view.data(), view.stride()),
        reference.size,
        (x0, y0),
        (w, h),
        (buffer, w),
    );
    PlaneView::new(buffer, w, w, h).expect("window covers its samples")
}

/// The warp of a plane block: the local model, or the global one of a
/// large `GLOBALMV` block (libaom `av1_allow_warp`), if valid.
#[derive(Clone, Copy)]
struct Warp {
    params: [i32; 6],
    shear: Av1Shear,
}

impl<'a> FrameDecoder<'a> {
    /// The reference picture plane of reference name `name`.
    fn plane_reference(
        &self,
        name: i8,
        plane: PlaneIndex,
    ) -> Result<PlaneReference<'a>, DecodeError> {
        let picture = self.references[name as usize]
            .ok_or(DecodeError::Malformed("AV1 missing reference"))?;
        let scale = self.inter_tools.scales[name as usize];
        // A scaled reference has its own size (`RefUpscaledWidth`,
        // `RefFrameHeight`), any other the frame's.
        let (w, h) = scale.map_or((self.header.size.width, self.header.size.height), |s| {
            (s.reference.0 as usize, s.reference.1 as usize)
        });
        let ss = usize::from(plane != PlaneIndex::Y);
        Ok(PlaneReference {
            plane: picture.plane(plane),
            size: ((w + ss) >> ss, (h + ss) >> ss),
            scale,
        })
    }

    /// `is_scaled` of reference name `name`.
    pub(super) fn scaled(&self, name: i8) -> bool {
        self.inter_tools.scales[name as usize].is_some()
    }

    /// The warp a block of `modes` with local model `local` uses for a
    /// `w`x`h` plane block of its first reference.
    fn warp(
        &self,
        block: BlockPos,
        modes: &InterModes,
        local: Option<WarpModel>,
        (w, h): (usize, usize),
    ) -> Option<Warp> {
        if w < WARP_MIN_SIDE
            || h < WARP_MIN_SIDE
            || self.header.force_integer_mv
            || self.scaled(modes.refs[0])
        {
            return None;
        }
        if let MotionMode::Warped = modes.motion {
            return local.map(|model| Warp {
                params: model.params,
                shear: model.shear,
            });
        }
        if !self.global_warp_block(block.size, modes) {
            return None;
        }
        self.global_warp(modes.refs[0])
    }

    /// The global warp of reference `name`, if its model is valid and the
    /// reference unscaled.
    fn global_warp(&self, name: i8) -> Option<Warp> {
        if self.scaled(name) {
            return None;
        }
        let gm = &self.header.global_motion[name as usize];
        debug_assert!(gm.kind > WarpType::Translation);
        av1_setup_shear(&gm.params).map(|shear| Warp {
            params: gm.params,
            shear,
        })
    }

    /// Predict every plane of an inter block, then blend OBMC predictions
    /// into it.
    #[inline(never)]
    pub(super) fn predict_inter(
        &mut self,
        block: BlockPos,
        modes: &InterModes,
    ) -> Result<(), DecodeError> {
        if let Some(kind) = modes.compound {
            return self.predict_compound(block, modes, kind);
        }
        let translated = matches!(modes.motion, MotionMode::Simple)
            && modes.interintra.is_none()
            && !self.global_warp_block(block.size, modes);
        if translated && !(block.has_chroma() && self.sub8x8_chroma(block)) {
            return self.predict_translation(block, modes);
        }
        let local = match modes.motion {
            MotionMode::Warped => self.local_warp(block, modes),
            _ => None,
        };
        self.predict_plane(block, modes, local, PlaneIndex::Y)?;
        if block.has_chroma() {
            for plane in [PlaneIndex::U, PlaneIndex::V] {
                if self.sub8x8_chroma(block) {
                    self.predict_sub8x8_chroma(block, plane)?;
                } else {
                    self.predict_plane(block, modes, local, plane)?;
                }
            }
        }
        if let MotionMode::Obmc = modes.motion {
            self.obmc(block)?;
        }
        Ok(())
    }

    /// The common case: every plane of the block translated from its one
    /// reference, with none of the warp, inter-intra or 4xN chroma checks,
    /// and the reference unscaled (a scaled one takes the general path).
    fn predict_translation(
        &mut self,
        block: BlockPos,
        modes: &InterModes,
    ) -> Result<(), DecodeError> {
        if self.scaled(modes.refs[0]) {
            return self.predict_scaled(block, modes);
        }
        let picture = self.references[modes.refs[0] as usize]
            .ok_or(DecodeError::Malformed("AV1 missing reference"))?;
        let (width, height) = (self.header.size.width, self.header.size.height);
        let planes: &[PlaneIndex] = if block.has_chroma() {
            &PlaneIndex::ALL
        } else {
            &[PlaneIndex::Y]
        };
        for &plane in planes {
            let ss = usize::from(plane != PlaneIndex::Y);
            let (origin, size) = Self::plane_block(block, plane);
            let target = self.recon.plane_mut(plane);
            let stride = target.stride();
            let dst = target
                .samples_at_mut(origin.0 as isize, origin.1 as isize, size.0, size.1)
                .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
            let reference = PlaneReference {
                plane: picture.plane(plane),
                size: ((width + ss) >> ss, (height + ss) >> ss),
                scale: None,
            };
            translate_unscaled(
                self.dsp,
                reference,
                &mut self.maps.prediction.window,
                origin,
                size,
                modes.mv[0],
                (ss == 1, ss == 1),
                modes.filters,
                (dst, stride),
            );
        }
        Ok(())
    }

    /// [`Self::predict_translation`] from a scaled reference, through the
    /// general per-plane path.
    #[cold]
    #[inline(never)]
    fn predict_scaled(&mut self, block: BlockPos, modes: &InterModes) -> Result<(), DecodeError> {
        self.predict_plane(block, modes, None, PlaneIndex::Y)?;
        if block.has_chroma() {
            for plane in [PlaneIndex::U, PlaneIndex::V] {
                self.predict_plane(block, modes, None, plane)?;
            }
        }
        Ok(())
    }

    /// The plane block of `block`: its plane sample position and size (4x4
    /// chroma at least, starting a unit up or left for 4-sample sides).
    pub(super) fn plane_block(
        block: BlockPos,
        plane: PlaneIndex,
    ) -> ((usize, usize), (usize, usize)) {
        if plane == PlaneIndex::Y {
            return (
                (block.mi_col * MI_SIZE, block.mi_row * MI_SIZE),
                (block.size.width(), block.size.height()),
            );
        }
        let col = block.mi_col - usize::from(block.size.width_mi() == 1);
        let row = block.mi_row - usize::from(block.size.height_mi() == 1);
        (
            (col * MI_SIZE / 2, row * MI_SIZE / 2),
            (
                (block.size.width() / 2).max(MI_SIZE),
                (block.size.height() / 2).max(MI_SIZE),
            ),
        )
    }

    /// Predict one plane of the block from its own motion (and local warp
    /// model `local` of a `WARPED_CAUSAL` block).
    fn predict_plane(
        &mut self,
        block: BlockPos,
        modes: &InterModes,
        local: Option<WarpModel>,
        plane: PlaneIndex,
    ) -> Result<(), DecodeError> {
        let (origin, size) = Self::plane_block(block, plane);
        let sub = plane != PlaneIndex::Y;
        let warp = self.warp(block, modes, local, size);
        let reference = self.plane_reference(modes.refs[0], plane)?;
        let mut dst = self
            .recon
            .plane_mut(plane)
            .region_mut(origin.0 as isize, origin.1 as isize, size.0, size.1)
            .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
        match warp {
            Some(warp) => {
                let view = reference
                    .plane
                    .region(0, 0, reference.size.0, reference.size.1)
                    .ok_or(DecodeError::Malformed(
                        "AV1 reference smaller than its frame",
                    ))?;
                av1_warp(
                    self.dsp,
                    &view,
                    &warp.params,
                    warp.shear,
                    origin,
                    size,
                    (sub, sub),
                    &mut dst,
                );
            }
            None => translate(
                self.dsp,
                reference,
                &mut self.maps.prediction.window,
                origin,
                size,
                modes.mv[0],
                (sub, sub),
                modes.filters,
                &mut dst,
            ),
        }
        if let Some(interintra) = modes.interintra {
            self.interintra(block, plane, interintra, (origin, size))?;
        }
        Ok(())
    }

    /// Predict every plane of a compound block (8x8 or larger, so each
    /// carries its chroma) from both references and blend them as `kind`
    /// says (§7.11.3.1).
    // Out of line: our own streams have no compound blocks, and inlined
    // this grows the single-reference path the block walk inlines.
    #[cold]
    #[inline(never)]
    fn predict_compound(
        &mut self,
        block: BlockPos,
        modes: &InterModes,
        kind: CompoundType,
    ) -> Result<(), DecodeError> {
        let luma = (block.size.width(), block.size.height());
        // Warped or scaled references take the general per-list path; the
        // common case resolves both references once.
        let general = !self.header.force_integer_mv
            && (self.global_warp_list(block.size, modes, 0)
                || self.global_warp_list(block.size, modes, 1))
            || self.scaled(modes.refs[0])
            || self.scaled(modes.refs[1]);
        let pictures = [0, 1].map(|list| self.references[modes.refs[list] as usize]);
        for plane in PlaneIndex::ALL {
            let (origin, size) = Self::plane_block(block, plane);
            let sub = plane != PlaneIndex::Y;
            if general {
                for list in 0..2 {
                    self.predict_one_of_two(block, modes, list, plane, (origin, size))?;
                }
            } else {
                self.translate_both(pictures, modes, plane, (origin, size))?;
            }
            let scratch = &mut self.maps.prediction;
            let [first, second] = &scratch.compound;
            let p0 = Av1CompoundPred {
                data: first,
                stride: size.0,
            };
            let p1 = Av1CompoundPred {
                data: second,
                stride: size.0,
            };
            let mut dst = self
                .recon
                .plane_mut(plane)
                .region_mut(origin.0 as isize, origin.1 as isize, size.0, size.1)
                .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
            match kind {
                CompoundType::Average => match nc_dsp::BlockSize::new(size.0, size.1) {
                    Some(kernel_size) => {
                        self.dsp
                            .average_compound(kernel_size, (first, second), size.0, &mut dst);
                    }
                    None => av1_blend_average(p0, p1, size, &mut dst),
                },
                CompoundType::Distance => {
                    let offsets = &self.inter_tools.ref_offsets;
                    let weights = av1_distance_weights(
                        offsets[modes.refs[0] as usize],
                        offsets[modes.refs[1] as usize],
                    );
                    av1_blend_distance(p0, p1, weights, size, &mut dst);
                }
                CompoundType::Wedge { index, sign } => {
                    let wedges = scratch.wedges.get_or_insert_with(Default::default);
                    let mask = Av1Mask {
                        data: wedges
                            .mask(luma, sign, usize::from(index))
                            .ok_or(DecodeError::Malformed("AV1 wedge size"))?,
                        stride: luma.0,
                        sub_x: sub,
                        sub_y: sub,
                    };
                    av1_blend_mask(self.dsp, p0, p1, mask, size, &mut dst);
                }
                CompoundType::DifferenceWeighted { inverse } => {
                    // The luma predictions set the mask of every plane.
                    if !sub {
                        scratch.mask.clear();
                        scratch.mask.resize(luma.0 * luma.1, 0);
                        av1_difference_weight_mask(p0, p1, inverse, luma, &mut scratch.mask);
                    }
                    let mask = Av1Mask {
                        data: &scratch.mask,
                        stride: luma.0,
                        sub_x: sub,
                        sub_y: sub,
                    };
                    av1_blend_mask(self.dsp, p0, p1, mask, size, &mut dst);
                }
            }
        }
        Ok(())
    }

    /// Predict both lists of a compound block's plane block at `origin` into
    /// the compound scratch, by translation from the unscaled reference
    /// `pictures`.
    fn translate_both(
        &mut self,
        pictures: [Option<&'a Picture>; 2],
        modes: &InterModes,
        plane: PlaneIndex,
        (origin, size): ((usize, usize), (usize, usize)),
    ) -> Result<(), DecodeError> {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (width, height) = (self.header.size.width, self.header.size.height);
        let scratch = &mut self.maps.prediction;
        let samples = size.0 * size.1;
        for (list, picture) in pictures.into_iter().enumerate() {
            let picture = picture.ok_or(DecodeError::Malformed("AV1 missing reference"))?;
            let reference = PlaneReference {
                plane: picture.plane(plane),
                size: ((width + ss) >> ss, (height + ss) >> ss),
                scale: None,
            };
            // Grown, never cleared: the prediction writes every sample it uses.
            if scratch.compound[list].len() < samples {
                scratch.compound[list].resize(samples, 0);
            }
            let ((x0, y0), fracs) = unscaled_window(origin, modes.mv[list], (ss == 1, ss == 1));
            let span = (size.0 + AV1_FILTER_TAPS - 1, size.1 + AV1_FILTER_TAPS - 1);
            let dst = (&mut scratch.compound[list][..samples], size.0);
            let source = reference.plane;
            if let Some(src) = source.samples_at(x0, y0, span.0, span.1) {
                let src = (src, source.stride());
                self.dsp
                    .av1_prep_samples(modes.filters, src, size, fracs, dst);
            } else {
                let view = gather(reference, &mut scratch.window, (x0, y0), span);
                let src = (view.data(), view.stride());
                self.dsp
                    .av1_prep_samples(modes.filters, src, size, fracs, dst);
            }
        }
        Ok(())
    }

    /// Predict list `list` of a compound block's plane block at `origin`
    /// into the compound scratch: a global warp of a `GLOBAL_GLOBALMV`
    /// block, or translation.
    fn predict_one_of_two(
        &mut self,
        block: BlockPos,
        modes: &InterModes,
        list: usize,
        plane: PlaneIndex,
        (origin, size): ((usize, usize), (usize, usize)),
    ) -> Result<(), DecodeError> {
        let sub = plane != PlaneIndex::Y;
        let reference = self.plane_reference(modes.refs[list], plane)?;
        let warp = (size.0 >= WARP_MIN_SIDE
            && size.1 >= WARP_MIN_SIDE
            && !self.header.force_integer_mv
            && self.global_warp_list(block.size, modes, list))
        .then(|| self.global_warp(modes.refs[list]))
        .flatten();
        let scratch = &mut self.maps.prediction;
        // Grown, never cleared: the prediction writes every sample it uses.
        let samples = size.0 * size.1;
        if scratch.compound[list].len() < samples {
            scratch.compound[list].resize(samples, 0);
        }
        let dst = &mut scratch.compound[list][..samples];
        match warp {
            Some(warp) => {
                let view = reference
                    .plane
                    .region(0, 0, reference.size.0, reference.size.1)
                    .ok_or(DecodeError::Malformed(
                        "AV1 reference smaller than its frame",
                    ))?;
                av1_warp_compound(
                    self.dsp,
                    &view,
                    &warp.params,
                    warp.shear,
                    origin,
                    size,
                    (sub, sub),
                    dst,
                    size.0,
                );
            }
            None => translate_compound(
                self.dsp,
                reference,
                &mut scratch.window,
                origin,
                size,
                modes.mv[list],
                (sub, sub),
                modes.filters,
                dst,
            ),
        }
        Ok(())
    }

    /// Whether the chroma of a block 4 samples wide or high is predicted
    /// piecewise from every block its chroma covers: all must be inter
    /// (libaom `is_sub8x8_inter`).
    #[inline]
    fn sub8x8_chroma(&self, block: BlockPos) -> bool {
        let (narrow, short) = (block.size.width_mi() == 1, block.size.height_mi() == 1);
        (narrow || short) && self.covered_blocks_inter(block, (narrow, short))
    }

    /// Whether every block the chroma of a `narrow` or `short` block
    /// covers is an inter block.
    fn covered_blocks_inter(&self, block: BlockPos, (narrow, short): (bool, bool)) -> bool {
        let rows = block.mi_row - usize::from(short)..=block.mi_row;
        rows.flat_map(|row| {
            (block.mi_col - usize::from(narrow)..=block.mi_col).map(move |col| (row, col))
        })
        .all(|(row, col)| self.maps.motion.at(row, col).is_inter())
    }

    /// Predict the chroma of a block 4 samples wide or high in pieces, each
    /// from the vector, reference and filters of the block it belongs to.
    fn predict_sub8x8_chroma(
        &mut self,
        block: BlockPos,
        plane: PlaneIndex,
    ) -> Result<(), DecodeError> {
        let (origin, size) = Self::plane_block(block, plane);
        let piece = (block.size.width() / 2, block.size.height() / 2);
        let first = (
            block.mi_row + 1 - size.1 / piece.1,
            block.mi_col + 1 - size.0 / piece.0,
        );
        for (i, y) in (0..size.1).step_by(piece.1).enumerate() {
            for (j, x) in (0..size.0).step_by(piece.0).enumerate() {
                let cell = *self.maps.motion.at(first.0 + i, first.1 + j);
                let reference = self.plane_reference(cell.refs[0], plane)?;
                let at = (origin.0 + x, origin.1 + y);
                let mut dst = self
                    .recon
                    .plane_mut(plane)
                    .region_mut(at.0 as isize, at.1 as isize, piece.0, piece.1)
                    .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
                translate(
                    self.dsp,
                    reference,
                    &mut self.maps.prediction.window,
                    at,
                    piece,
                    cell.mv[0],
                    (true, true),
                    cell.filters(),
                    &mut dst,
                );
            }
        }
        Ok(())
    }

    /// Blend the intra prediction of an inter-intra block into one plane's
    /// inter prediction (§7.11.3.13–14).
    fn interintra(
        &mut self,
        block: BlockPos,
        plane: PlaneIndex,
        interintra: InterIntra,
        ((x, y), (w, h)): ((usize, usize), (usize, usize)),
    ) -> Result<(), DecodeError> {
        let (above, left) = block.available(plane);
        let edges = BlockEdges::read(
            self.recon.plane(plane),
            x,
            y,
            w,
            h,
            EdgeAvail {
                have_above: above,
                have_left: left,
                have_above_right: false,
                have_below_left: false,
            },
        );
        let kind = match interintra.mode {
            Av1InterIntraMode::Dc => IntraKind::Dc,
            Av1InterIntraMode::Smooth => IntraKind::Smooth,
            Av1InterIntraMode::Vertical => IntraKind::Directional {
                angle: VERTICAL_ANGLE,
                edge_filter: None,
            },
            Av1InterIntraMode::Horizontal => IntraKind::Directional {
                angle: HORIZONTAL_ANGLE,
                edge_filter: None,
            },
        };
        let scratch = &mut self.maps.prediction;
        scratch.block.clear();
        scratch.block.resize(w * h, 0);
        let mut intra = PlaneViewMut::new(&mut scratch.block, w, w, h).expect("block buffer");
        edges.predict(kind, &mut intra);
        let sub = plane != PlaneIndex::Y;
        let mask = if let Some(index) = interintra.wedge {
            debug_assert!(wedge_used(block.size));
            let wedges = scratch.wedges.get_or_insert_with(Default::default);
            let luma = (block.size.width(), block.size.height());
            Av1Mask {
                data: wedges
                    .mask(luma, false, usize::from(index))
                    .ok_or(DecodeError::Malformed("AV1 wedge size"))?,
                stride: luma.0,
                sub_x: sub,
                sub_y: sub,
            }
        } else {
            scratch.mask.clear();
            scratch.mask.resize(w * h, 0);
            av1_interintra_smooth_mask(interintra.mode, (w, h), &mut scratch.mask);
            Av1Mask {
                data: &scratch.mask,
                stride: w,
                sub_x: false,
                sub_y: false,
            }
        };
        let intra = PlaneView::new(&scratch.block, w, w, h).expect("block buffer");
        let mut dst = self
            .recon
            .plane_mut(plane)
            .region_mut(x as isize, y as isize, w, h)
            .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
        av1_blend_interintra(&intra, mask, (w, h), &mut dst);
        Ok(())
    }

    /// Overlapped block motion compensation (§7.11.3.10): blend the
    /// predictions of the inter blocks above, then of those to the left,
    /// into the block's edges.
    fn obmc(&mut self, block: BlockPos) -> Result<(), DecodeError> {
        let mut above = Vec::new();
        let limit = MAX_NEIGHBOR_OBMC[usize::from(block.size.width_mi_log2())];
        self.overlappable_above(block, limit, |offset, units, cell| {
            above.push((offset, units, *cell));
        });
        let mut left = Vec::new();
        let limit = MAX_NEIGHBOR_OBMC[usize::from(block.size.height_mi_log2())];
        self.overlappable_left(block, limit, |offset, units, cell| {
            left.push((offset, units, *cell));
        });
        for &(offset, units, cell) in &above {
            for plane in PlaneIndex::ALL {
                self.obmc_neighbour(block, plane, Av1ObmcEdge::Above, (offset, units), &cell)?;
            }
        }
        for &(offset, units, cell) in &left {
            for plane in PlaneIndex::ALL {
                self.obmc_neighbour(block, plane, Av1ObmcEdge::Left, (offset, units), &cell)?;
            }
        }
        Ok(())
    }

    /// Predict one plane of the overlap with `neighbour` (at `offset` 4x4
    /// units along the edge, `units` long) from its motion and blend it
    /// in.
    fn obmc_neighbour(
        &mut self,
        block: BlockPos,
        plane: PlaneIndex,
        edge: Av1ObmcEdge,
        (offset, units): (usize, usize),
        neighbour: &MotionCell,
    ) -> Result<(), DecodeError> {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (bw, bh) = (block.size.width(), block.size.height());
        // Chroma 4x4, 8x4 and 4x8 blends only from the left (libaom
        // `av1_skip_u4x4_pred_in_obmc`).
        let small_chroma = matches!((bw >> 1, bh >> 1), (4 | 8, 4) | (4, 8));
        if ss == 1 && edge == Av1ObmcEdge::Above && small_chroma {
            return Ok(());
        }
        // (predicted, blended) samples across the edge.
        let across = |side: usize| {
            (
                (side >> (ss + 1)).clamp(OBMC_MIN_PREDICTION, OBMC_MAX_OVERLAP >> ss),
                side.min(2 * OBMC_MAX_OVERLAP) >> (1 + ss),
            )
        };
        let along = (units * MI_SIZE) >> ss;
        let along_start = (offset * MI_SIZE) >> ss;
        let (x, y) = (
            (block.mi_col * MI_SIZE) >> ss,
            (block.mi_row * MI_SIZE) >> ss,
        );
        let ((px, py), (pw, ph), (ow, oh)) = match edge {
            Av1ObmcEdge::Above => {
                let (predicted, overlap) = across(bh);
                ((x + along_start, y), (along, predicted), (along, overlap))
            }
            Av1ObmcEdge::Left => {
                let (predicted, overlap) = across(bw);
                ((x, y + along_start), (predicted, along), (overlap, along))
            }
        };
        let reference = self.plane_reference(neighbour.refs[0], plane)?;
        let scratch = &mut self.maps.prediction;
        scratch.block.clear();
        scratch.block.resize(pw * ph, 0);
        let mut prediction =
            PlaneViewMut::new(&mut scratch.block, pw, pw, ph).expect("block buffer");
        translate(
            self.dsp,
            reference,
            &mut scratch.window,
            (px, py),
            (pw, ph),
            neighbour.mv[0],
            (ss == 1, ss == 1),
            neighbour.filters(),
            &mut prediction,
        );
        let prediction = PlaneView::new(&scratch.block, pw, pw, ph).expect("block buffer");
        let mut dst = self
            .recon
            .plane_mut(plane)
            .region_mut(px as isize, py as isize, ow, oh)
            .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
        av1_obmc_blend(&prediction, edge, (ow, oh), &mut dst);
        Ok(())
    }
}
