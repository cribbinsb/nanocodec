//! General intra and inter reconstruction into shared pictures with shared DSP prediction.
use super::{
    CoeffArena, Inter, Sequence, SliceHeader, Unit, cclm,
    inter_prediction::InterScratch,
    references::{Availability, RefScratch, References},
    transform::{decode_plane_residual, primary_kinds},
};
use nc_common::{ChromaFormat, Picture, PlaneIndex};
use nc_dsp::{DepthKernels, Pixel, VvcIntraMode};

fn channel_range(unit: &Unit) -> std::ops::Range<usize> {
    if unit.combined {
        0..3
    } else if unit.channel == 0 {
        0..1
    } else {
        1..3
    }
}

#[derive(Debug)]
pub(crate) struct ReconScratch<T: Pixel> {
    inter: Box<InterScratch<T>>,
    refs: Box<RefScratch>,
    cclm: cclm::CclmScratch<T>,
    levels: Box<[i32; 64 * 64]>,
    residual: Box<[i16; 64 * 64]>,
    joint_cb: Box<[i16; 64 * 64]>,
    joint_cr: Box<[i16; 64 * 64]>,
    /// An ACT transform unit's three converted residuals.
    act: [Box<[i16; ACT_MAX_AREA]>; 3],
    factors: Vec<u8>,
    prediction_buf: Vec<T>,
    available: Availability,
}

impl<T: Pixel> Default for ReconScratch<T> {
    fn default() -> Self {
        Self {
            inter: Box::default(),
            refs: Box::default(),
            cclm: cclm::CclmScratch::default(),
            levels: Box::new([0; 64 * 64]),
            residual: Box::new([0; 64 * 64]),
            joint_cb: Box::new([0; 64 * 64]),
            joint_cr: Box::new([0; 64 * 64]),
            act: std::array::from_fn(|_| Box::new([0; ACT_MAX_AREA])),
            factors: Vec::new(),
            prediction_buf: vec![T::default(); 64 * 64],
            available: Availability::default(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ResidualContext<'a> {
    pub arena: &'a CoeffArena,
    pub header: &'a SliceHeader,
    pub lists: Option<&'a crate::decoder_headers::ScalingLists>,
}

#[derive(Clone, Copy)]
struct SliceReconEnv<'a, T: Pixel> {
    available: &'a Availability,
    sequence: &'a Sequence,
    res_ctx: ResidualContext<'a>,
    references: super::inter_prediction::Sources<'a, T>,
    dsp: DepthKernels<'a>,
    reshape: Option<(&'a nc_dsp::VvcLmcs<T>, bool)>,
}

struct UnitReconEnv<'a, 'b, T: Pixel> {
    isp: &'a mut Option<super::isp::Prediction<T>>,
    ciip: Option<&'a [Vec<T>; 3]>,
    chroma_scale: Option<&'a mut super::reshape::ChromaScale<'b, T>>,
}

/// Reconstruct the CUs `units` of one slice into `picture` (before the
/// loop filters). Neighbours in other slices are unavailable, so each slice
/// reconstructs on its own.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct<T: Pixel>(
    picture: &mut Picture<T>,
    units: &[Unit],
    sequence: &Sequence,
    dimensions: [usize; 2],
    res_ctx: ResidualContext<'_>,
    references: super::inter_prediction::Sources<'_, T>,
    dsp: DepthKernels<'_>,
    reshape: Option<(&nc_dsp::VvcLmcs<T>, bool)>,
    scratch: &mut ReconScratch<T>,
) {
    let max_tile = units.iter().map(|u| u.tile).max().unwrap_or(0);
    scratch.available.reset(
        dimensions,
        sequence.entropy_sync.then_some(sequence.ctu_log2),
        (max_tile, sequence.subsampling()),
    );
    let mut chroma_scale = reshape.filter(|(_, enabled)| *enabled).map(|(model, _)| {
        super::reshape::ChromaScale::new(model, units, dimensions, sequence.ctu_log2)
    });
    let mut prediction_buf = std::mem::take(&mut scratch.prediction_buf);
    if prediction_buf.len() < 64 * 64 {
        prediction_buf.resize(64 * 64, T::default());
    }
    let mut available = std::mem::take(&mut scratch.available);
    for unit in units {
        available.enter(unit);
        if let Some(palette) = &unit.palette {
            reconstruct_palette(picture, unit, palette, sequence, res_ctx.header);
            for transform in &unit.transforms {
                available.mark_transform(unit, transform);
            }
            continue;
        }
        let mut isp = super::isp::Prediction::prepare(unit, picture, &available, dsp.depth());
        let ciip = matches!(unit.inter, Some(Inter::Ciip(_)))
            .then(|| super::ciip::prepare(unit, picture, &available, dsp));
        let any_coeffs = unit
            .transforms
            .iter()
            .any(|t| t.spans.iter().any(|&(_, count)| count != 0));
        let kinds = if any_coeffs {
            primary_kinds(unit, sequence)
        } else {
            [nc_dsp::VvcTransformKind::Dct2; 2]
        };
        for (transform_index, transform) in unit.transforms.iter().enumerate() {
            scratch.cclm.invalidate();
            let has_joint =
                prepare_joint_chroma(dsp, unit, transform, kinds, sequence, res_ctx, scratch);
            if unit.act {
                prepare_act(
                    dsp,
                    unit,
                    transform,
                    (has_joint, kinds),
                    (sequence, res_ctx),
                    scratch,
                );
            }
            for index in channel_range(unit) {
                let slice_env = SliceReconEnv {
                    available: &available,
                    sequence,
                    res_ctx,
                    references,
                    dsp,
                    reshape,
                };
                let mut unit_env = UnitReconEnv {
                    isp: &mut isp,
                    ciip: ciip.as_ref(),
                    chroma_scale: chroma_scale.as_mut(),
                };
                reconstruct_plane_block(
                    picture,
                    unit,
                    (transform_index, transform, has_joint, kinds),
                    index,
                    slice_env,
                    &mut unit_env,
                    &mut prediction_buf,
                    scratch,
                );
            }
            available.mark_transform(unit, transform);
        }
    }
    available.finish_slice(max_tile);
    scratch.available = available;
    scratch.prediction_buf = prediction_buf;
}

/// Write a palette CU (§8.4.5.3); its escapes dequantise at the plane's QP,
/// at least `QpPrimeTsMin` (§7.4.3.4).
#[inline(never)]
fn reconstruct_palette<T: Pixel>(
    picture: &mut Picture<T>,
    unit: &Unit,
    palette: &super::palette::Palette,
    sequence: &Sequence,
    header: &SliceHeader,
) {
    let minimum = MIN_TS_QP + TS_QP_STEP * sequence.tools.min_qp_ts;
    let mut qps = [0; 3];
    let planes = palette.planes();
    for (plane, qp) in qps
        .iter_mut()
        .enumerate()
        .take(planes.end)
        .skip(planes.start)
    {
        *qp = super::transform::qp(unit.quant(), 0, plane, sequence, header).max(minimum);
    }
    super::palette::reconstruct(picture, unit, palette, qps, sequence.depth.bits());
}

/// `QpPrimeTsMin = 4 + 6 · sps_min_qp_prime_ts` (H.266 §7.4.3.4).
const MIN_TS_QP: u8 = 4;
const TS_QP_STEP: u8 = 6;

fn prepare_joint_chroma<T: Pixel>(
    dsp: DepthKernels<'_>,
    unit: &Unit,
    transform: &super::Transform,
    kinds: [nc_dsp::VvcTransformKind; 2],
    sequence: &Sequence,
    res_ctx: ResidualContext<'_>,
    scratch: &mut ReconScratch<T>,
) -> bool {
    if transform.joint == 0 {
        return false;
    }
    let src_plane = if transform.joint == 1 { 2 } else { 1 };
    let coeffs = res_ctx.arena.slice(transform.spans[src_plane]);
    let Some(cb_blk) = transform.chroma_block.filter(|_| !coeffs.is_empty()) else {
        return false;
    };
    let c_size = sequence.scaled(true, (cb_blk.width(), cb_blk.height()));
    let c_area = c_size.0 * c_size.1;
    decode_plane_residual(
        dsp,
        unit,
        transform,
        (src_plane, c_size, kinds),
        (sequence, res_ctx.header, res_ctx.lists),
        coeffs,
        &mut scratch.factors,
        &mut scratch.levels,
        &mut scratch.residual[..c_area],
    );
    nc_dsp::vvc_joint_chroma(
        &scratch.residual[..c_area],
        transform.joint,
        res_ctx.header.picture.joint_sign,
        &mut scratch.joint_cb[..c_area],
        &mut scratch.joint_cr[..c_area],
    );
    true
}

/// The largest transform block of an ACT CU: ACT excludes 64-sample
/// transforms (§7.3.2.4).
const ACT_MAX_AREA: usize = 32 * 32;

/// The residuals of a transform unit of an ACT CU (§8.7.4.6): each plane's
/// (zero where none is coded; joint `CbCr` already mixed), converted from
/// YCgCo-R into `scratch.act`. ACT is 4:4:4 only, so the planes share one
/// size.
#[inline(never)]
fn prepare_act<T: Pixel>(
    dsp: DepthKernels<'_>,
    unit: &Unit,
    transform: &super::Transform,
    (has_joint, kinds): (bool, [nc_dsp::VvcTransformKind; 2]),
    (sequence, res_ctx): (&Sequence, ResidualContext<'_>),
    scratch: &mut ReconScratch<T>,
) {
    let size = (transform.block.width(), transform.block.height());
    let area = size.0 * size.1;
    for index in 0..3 {
        let out = &mut scratch.act[index][..area];
        if transform.joint != 0 && index != 0 {
            let mixed = if index == 1 {
                &scratch.joint_cb
            } else {
                &scratch.joint_cr
            };
            if has_joint {
                out.copy_from_slice(&mixed[..area]);
            } else {
                out.fill(0);
            }
        } else if transform.spans[index].1 != 0 {
            decode_plane_residual(
                dsp,
                unit,
                transform,
                (index, size, kinds),
                (sequence, res_ctx.header, res_ctx.lists),
                res_ctx.arena.slice(transform.spans[index]),
                &mut scratch.factors,
                &mut scratch.levels,
                out,
            );
        } else {
            out.fill(0);
        }
    }
    let [y, cb, cr] = &mut scratch.act;
    nc_dsp::vvc_act_inverse(
        (&mut y[..area], &mut cb[..area], &mut cr[..area]),
        dsp.depth(),
    );
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_plane_block<T: Pixel>(
    picture: &mut Picture<T>,
    unit: &Unit,
    (transform_index, transform, has_joint, kinds): (
        usize,
        &super::Transform,
        bool,
        [nc_dsp::VvcTransformKind; 2],
    ),
    index: usize,
    env: SliceReconEnv<'_, T>,
    unit_env: &mut UnitReconEnv<'_, '_, T>,
    prediction_buf: &mut Vec<T>,
    scratch: &mut ReconScratch<T>,
) {
    let plane = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V][index];
    let Some(n) = (if index == 0 {
        Some(transform.block)
    } else {
        transform.chroma_block
    }) else {
        return;
    };
    let (x, y) = env.sequence.scaled(index != 0, (n.x, n.y));
    let (width, height) = env.sequence.scaled(index != 0, (n.width(), n.height()));
    let area = width * height;
    let overlay = unit.sbt.is_some() && transform_index != 0;
    // An ACT CU's planes all take a converted residual.
    let has_residual = if unit.act {
        true
    } else if transform.joint != 0 && index != 0 {
        has_joint
    } else {
        transform.spans[index].1 != 0
    };
    let chroma_factor = if has_residual && index != 0 && area > 4 {
        unit_env
            .chroma_scale
            .as_deref_mut()
            .map(|scale| scale.get((n.x, n.y), picture, env.available))
    } else {
        None
    };
    if !overlay
        && unit_env.ciip.is_none()
        && unit.motion.is_some()
        && try_reconstruct_inter_direct(
            picture,
            unit,
            (plane, index, n, (x, y, width, height)),
            (transform, has_residual, kinds, chroma_factor),
            env,
            scratch,
        )
    {
        return;
    }
    if area > prediction_buf.len() {
        prediction_buf.resize(area, T::default());
    }
    if overlay {
        let view = picture.plane(plane).view();
        for (dy, dst) in prediction_buf[..area].chunks_exact_mut(width).enumerate() {
            nc_common::copy_row(dst, &view.row(y + dy)[x..x + width]);
        }
    } else if let Some(isp) = unit_env.isp.as_mut().filter(|_| index == 0) {
        let isp_pred = isp.predict(n, unit.luma_mode, picture, env.available, env.dsp);
        prediction_buf[..area].copy_from_slice(&isp_pred[..area]);
    } else {
        predict_into(
            unit,
            n,
            index,
            picture,
            env.available,
            env.sequence,
            env.references,
            env.dsp,
            scratch,
            &mut prediction_buf[..area],
        );
    }
    let prediction = &mut prediction_buf[..area];
    if !overlay
        && index == 0
        && unit.motion.is_some()
        && let Some((model, _)) = env.reshape
    {
        env.dsp.vvc_forward_lmcs(model, prediction);
    }
    if let Some(ciip_planes) = unit_env.ciip {
        let subsampling = picture.chroma_format().plane_subsampling(plane);
        super::ciip::blend(
            unit,
            n,
            subsampling,
            &ciip_planes[index],
            prediction,
            env.dsp,
        );
    }
    if has_residual {
        let res = prepare_residual_slice(
            env.dsp,
            unit,
            (transform, index, (width, height), kinds),
            (env.sequence, &env.res_ctx),
            chroma_factor,
            scratch,
        );
        env.dsp.vvc_add_residual((width, height), prediction, res);
    }
    let mut view = picture.plane_mut(plane).view_mut();
    for (dy, src) in prediction.chunks_exact(width).enumerate() {
        nc_common::copy_row(&mut view.row_mut(y + dy)[x..x + width], src);
    }
}

fn try_reconstruct_inter_direct<T: Pixel>(
    picture: &mut Picture<T>,
    unit: &Unit,
    (plane, index, n, (x, y, width, height)): (
        PlaneIndex,
        usize,
        nc_encoder::Block,
        (usize, usize, usize, usize),
    ),
    (transform, has_residual, kinds, chroma_factor): (
        &super::Transform,
        bool,
        [nc_dsp::VvcTransformKind; 2],
        Option<i32>,
    ),
    env: SliceReconEnv<'_, T>,
    scratch: &mut ReconScratch<T>,
) -> bool {
    let Some(mut dst_view) = picture
        .plane_mut(plane)
        .region_mut(x as isize, y as isize, width, height)
    else {
        return false;
    };
    if !super::inter_prediction::try_predict_direct(
        env.dsp,
        env.references,
        unit,
        plane,
        n,
        &mut scratch.inter,
        &mut dst_view,
    ) {
        return false;
    }
    if index == 0
        && let Some((model, _)) = env.reshape
    {
        env.dsp.vvc_forward_lmcs_into(model, &mut dst_view);
    }
    if has_residual {
        let res = prepare_residual_slice(
            env.dsp,
            unit,
            (transform, index, (width, height), kinds),
            (env.sequence, &env.res_ctx),
            chroma_factor,
            scratch,
        );
        env.dsp.vvc_add_residual_into(&mut dst_view, res);
    }
    true
}

#[inline]
fn prepare_residual_slice<'a, T: Pixel>(
    dsp: DepthKernels<'_>,
    unit: &Unit,
    (transform, index, (width, height), kinds): (
        &super::Transform,
        usize,
        (usize, usize),
        [nc_dsp::VvcTransformKind; 2],
    ),
    (sequence, res_ctx): (&Sequence, &ResidualContext<'_>),
    chroma_factor: Option<i32>,
    scratch: &'a mut ReconScratch<T>,
) -> &'a [i16] {
    let area = width * height;
    let res_slice: &mut [i16] = if unit.act {
        &mut scratch.act[index][..area]
    } else if transform.joint != 0 && index != 0 {
        if index == 1 {
            &mut scratch.joint_cb[..area]
        } else {
            &mut scratch.joint_cr[..area]
        }
    } else {
        let coeffs = res_ctx.arena.slice(transform.spans[index]);
        decode_plane_residual(
            dsp,
            unit,
            transform,
            (index, (width, height), kinds),
            (sequence, res_ctx.header, res_ctx.lists),
            coeffs,
            &mut scratch.factors,
            &mut scratch.levels,
            &mut scratch.residual[..area],
        );
        &mut scratch.residual[..area]
    };
    if let Some(factor) = chroma_factor {
        dsp.vvc_scale_chroma_residual(factor, res_slice);
    }
    res_slice
}

#[allow(clippy::too_many_arguments)]
fn predict_into<T: Pixel>(
    unit: &Unit,
    block: nc_encoder::Block,
    index: usize,
    picture: &Picture<T>,
    available: &Availability,
    sequence: &Sequence,
    references: super::inter_prediction::Sources<'_, T>,
    dsp: DepthKernels<'_>,
    scratch: &mut ReconScratch<T>,
    prediction: &mut [T],
) {
    let plane = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V][index];
    let channel = usize::from(index != 0);
    let (x, y) = sequence.scaled(index != 0, (block.x, block.y));
    let (width, height) = sequence.scaled(index != 0, (block.width(), block.height()));
    let direction = if index == 0 {
        unit.luma_mode
    } else {
        chroma_direction(unit, sequence)
    };
    if unit.motion.is_some() {
        super::inter_prediction::predict_unit(
            dsp,
            references,
            unit,
            plane,
            block,
            &mut scratch.inter,
            prediction,
        );
    } else if let Some(vector) = unit.block_vector {
        super::ibc::predict(picture, plane, vector, (x, y, width, height), prediction);
    } else if index == 0 && unit.reference_line != 0 {
        let refs = References::gather_mrl_into(
            &mut scratch.refs,
            picture,
            available,
            (x, y, width, height),
            (unit.reference_line, dsp.depth()),
        );
        dsp.vvc_mrl_predict(
            (width, height),
            direction,
            unit.reference_line,
            refs.top,
            refs.left,
            refs.corner,
            prediction,
        );
    } else if unit.bdpcm[channel] != 0
        || (index == 0 && unit.mip.is_some())
        || (index != 0 && unit.chroma_mip)
    {
        predict_from_edges(
            unit,
            (picture, available),
            (plane, channel),
            (x, y, width, height),
            dsp,
            &mut scratch.refs,
            prediction,
        );
    } else if direction >= 67 {
        cclm::predict_cached(
            &mut scratch.cclm,
            picture,
            available,
            sequence,
            unit,
            block,
            (plane, dsp.depth()),
            prediction,
        );
    } else {
        let smooth = index == 0 && nc_dsp::vvc_intra_filter_references((width, height), direction);
        let refs = References::gather_into(
            &mut scratch.refs,
            picture,
            available,
            (plane, dsp.depth()),
            (x, y, width, height),
            smooth,
        );
        let mode = match direction {
            0 => VvcIntraMode::Planar,
            1 => VvcIntraMode::Dc,
            _ => VvcIntraMode::Angular {
                direction,
                luma: index == 0,
            },
        };
        dsp.vvc_intra_predict_rect(
            (width, height),
            mode,
            refs.top,
            refs.left,
            refs.corner,
            prediction,
        );
    }
}

/// The intra prediction mode of `unit`'s chroma (`IntraPredModeC`, H.266
/// §8.4.3): its coded or derived direction, mapped through Table 20 in
/// 4:2:2 (CCLM modes, 67 to 69, are not mapped).
pub(super) fn chroma_direction(unit: &Unit, sequence: &Sequence) -> u8 {
    let mode = unit.chroma_mode;
    if sequence.format == ChromaFormat::Yuv422 && mode < CCLM_FIRST_MODE {
        crate::intra::CHROMA_422_MODES[usize::from(mode)]
    } else {
        mode
    }
}

/// The first CCLM mode (`INTRA_LT_CCLM`, H.266 Table 19).
const CCLM_FIRST_MODE: u8 = 67;

/// BDPCM (H.266 §8.4.5.2.12) or MIP (§8.4.5.2.2; of 4:4:4 chroma too)
/// prediction of `plane` (channel `channel`) of `unit`'s block at `rect`
/// from its unfiltered edges.
fn predict_from_edges<T: Pixel>(
    unit: &Unit,
    (picture, available): (&Picture<T>, &Availability),
    (plane, channel): (PlaneIndex, usize),
    (x, y, width, height): (usize, usize, usize, usize),
    dsp: DepthKernels<'_>,
    ref_scratch: &mut RefScratch,
    prediction: &mut [T],
) {
    let refs = References::gather_into(
        ref_scratch,
        picture,
        available,
        (plane, dsp.depth()),
        (x, y, width, height),
        false,
    );
    if unit.bdpcm[channel] != 0 {
        dsp.vvc_bdpcm_predict(
            (width, height),
            unit.bdpcm[channel] == 2,
            refs.top,
            refs.left,
            prediction,
        );
    } else if let Some((mode, transpose)) = unit.mip {
        dsp.vvc_mip_predict(
            (width, height),
            mode,
            transpose,
            refs.top,
            refs.left,
            prediction,
        );
    }
}

/// [`reconstruct`] of a one-slice picture into a new picture (for unit tests
/// that pre-populate `Transform::residuals` via `inverse_residuals`).
#[cfg(test)]
#[allow(
    clippy::too_many_lines,
    reason = "test-only pre-decoded residual adapter"
)]
pub(crate) fn reconstructed(
    units: &[Unit],
    sequence: &Sequence,
    dimensions: [usize; 2],
    joint_sign: bool,
    references: super::inter_prediction::Sources<'_>,
    dsp: &nc_dsp::Dsp,
    reshape: Option<(&nc_dsp::VvcLmcs, bool)>,
) -> nc_common::decode::Result<Picture> {
    let mut picture = Picture::new(dimensions[0], dimensions[1], 0)
        .map_err(|_| nc_common::decode::DecodeError::Malformed("VVC picture dimensions"))?;
    let mut available = Availability::new(
        dimensions,
        sequence.entropy_sync.then_some(sequence.ctu_log2),
    );
    let mut chroma_scale = reshape.filter(|(_, enabled)| *enabled).map(|(model, _)| {
        super::reshape::ChromaScale::new(model, units, dimensions, sequence.ctu_log2)
    });
    let kernels = dsp.at(nc_common::BitDepth::EIGHT);
    let mut prediction_buf = vec![0u8; 64 * 64];
    let mut scaled_buf: Vec<i16> = Vec::new();
    let mut scratch = ReconScratch::default();
    for unit in units {
        available.enter(unit);
        let mut isp = super::isp::Prediction::prepare(unit, &picture, &available, kernels.depth());
        let ciip = super::ciip::prepare(unit, &picture, &available, kernels);
        for (transform_index, transform) in unit.transforms.iter().enumerate() {
            scratch.cclm.invalidate();
            let mixed = if transform.joint == 0 {
                [Vec::new(), Vec::new()]
            } else {
                let coded = &transform.residuals[if transform.joint == 1 { 2 } else { 1 }];
                let mut cb = vec![0; coded.len()];
                let mut cr = cb.clone();
                nc_dsp::vvc_joint_chroma(coded, transform.joint, joint_sign, &mut cb, &mut cr);
                [cb, cr]
            };
            for index in channel_range(unit) {
                let plane = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V][index];
                let Some(n) = (if index == 0 {
                    Some(transform.block)
                } else {
                    transform.chroma_block
                }) else {
                    continue;
                };
                let (x, y) = sequence.scaled(index != 0, (n.x, n.y));
                let (width, height) = sequence.scaled(index != 0, (n.width(), n.height()));
                let area = width * height;
                let overlay = unit.sbt.is_some() && transform_index != 0;
                if area > prediction_buf.len() {
                    prediction_buf.resize(area, 0);
                }
                if overlay {
                    let view = picture.plane(plane).view();
                    for (dy, dst) in prediction_buf[..area].chunks_exact_mut(width).enumerate() {
                        nc_common::copy_row(dst, &view.row(y + dy)[x..x + width]);
                    }
                } else if let Some(isp) = isp.as_mut().filter(|_| index == 0) {
                    let isp_pred = isp.predict(n, unit.luma_mode, &picture, &available, kernels);
                    prediction_buf[..area].copy_from_slice(&isp_pred[..area]);
                } else {
                    predict_into(
                        unit,
                        n,
                        index,
                        &picture,
                        &available,
                        sequence,
                        references,
                        kernels,
                        &mut scratch,
                        &mut prediction_buf[..area],
                    );
                }
                let prediction = &mut prediction_buf[..area];
                if !overlay
                    && index == 0
                    && unit.motion.is_some()
                    && let Some((model, _)) = reshape
                {
                    kernels.vvc_forward_lmcs(model, prediction);
                }
                let subsampling = picture.chroma_format().plane_subsampling(plane);
                super::ciip::blend(unit, n, subsampling, &ciip[index], prediction, kernels);
                let residual = if transform.joint != 0 && index != 0 {
                    &mixed[index - 1]
                } else {
                    &transform.residuals[index]
                };
                let residual = if index != 0 && residual.len() > 4 {
                    if let Some(scale) = &mut chroma_scale {
                        scaled_buf.clear();
                        scaled_buf.extend_from_slice(residual);
                        kernels.vvc_scale_chroma_residual(
                            scale.get((n.x, n.y), &picture, &available),
                            &mut scaled_buf,
                        );
                        scaled_buf.as_slice()
                    } else {
                        residual
                    }
                } else {
                    residual
                };
                if !residual.is_empty() {
                    kernels.vvc_add_residual((width, height), prediction, residual);
                }
                let mut view = picture.plane_mut(plane).view_mut();
                for (dy, src) in prediction.chunks_exact(width).enumerate() {
                    nc_common::copy_row(&mut view.row_mut(y + dy)[x..x + width], src);
                }
            }
            available.mark_transform(unit, transform);
        }
    }
    Ok(picture)
}
