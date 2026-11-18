//! General intra reconstruction into shared pictures with shared DSP prediction.
use super::{
    Sequence, Unit, cclm,
    inter_prediction::InterScratch,
    references::{Availability, RefScratch, References},
};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::{DepthKernels, Pixel, VvcIntraMode};

fn joint_residuals(transform: &super::Transform, joint_sign: bool) -> [Vec<i16>; 2] {
    if transform.joint == 0 {
        return [Vec::new(), Vec::new()];
    }
    let coded = &transform.residuals[if transform.joint == 1 { 2 } else { 1 }];
    let mut cb = vec![0; coded.len()];
    let mut cr = cb.clone();
    nc_dsp::vvc_joint_chroma(coded, transform.joint, joint_sign, &mut cb, &mut cr);
    [cb, cr]
}

fn channel_range(unit: &Unit) -> std::ops::Range<usize> {
    if unit.combined {
        0..3
    } else if unit.channel == 0 {
        0..1
    } else {
        1..3
    }
}

#[derive(Default)]
struct ReconScratch<T: Pixel> {
    inter: Box<InterScratch<T>>,
    refs: Box<RefScratch>,
}

/// Reconstruct the CUs `units` of one slice into `picture` (before the
/// loop filters). Neighbours in other slices are unavailable, so each slice
/// reconstructs on its own.
pub(crate) fn reconstruct<T: Pixel>(
    picture: &mut Picture<T>,
    units: &[Unit],
    sequence: &Sequence,
    dimensions: [usize; 2],
    joint_sign: bool,
    references: super::inter_prediction::Sources<'_, T>,
    dsp: DepthKernels<'_>,
    reshape: Option<(&nc_dsp::VvcLmcs<T>, bool)>,
) {
    let mut available = Availability::new(
        dimensions,
        sequence.entropy_sync.then_some(sequence.ctu_log2),
    );
    let mut chroma_scale = reshape.filter(|(_, enabled)| *enabled).map(|(model, _)| {
        super::reshape::ChromaScale::new(model, units, dimensions, sequence.ctu_log2)
    });
    let mut prediction_buf = vec![T::default(); 64 * 64];
    let mut scaled_buf: Vec<i16> = Vec::new();
    let mut scratch = ReconScratch::default();
    for unit in units {
        available.enter(unit);
        let mut isp = super::isp::Prediction::prepare(unit, picture, &available, dsp.depth());
        let ciip = super::ciip::prepare(unit, picture, &available, dsp);
        for (transform_index, transform) in unit.transforms.iter().enumerate() {
            let mixed = joint_residuals(transform, joint_sign);
            for index in channel_range(unit) {
                let plane = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V][index];
                let Some(n) = (if index == 0 {
                    Some(transform.block)
                } else {
                    transform.chroma_block
                }) else {
                    continue;
                };
                let shift = usize::from(index != 0);
                let (x, y, width, height) = (
                    n.x >> shift,
                    n.y >> shift,
                    n.width() >> shift,
                    n.height() >> shift,
                );
                let area = width * height;
                let overlay = unit.sbt.is_some() && transform_index != 0;
                if area > prediction_buf.len() {
                    prediction_buf.resize(area, T::default());
                }
                if overlay {
                    let view = picture.plane(plane).view();
                    for (dy, dst) in prediction_buf[..area].chunks_exact_mut(width).enumerate() {
                        dst.copy_from_slice(&view.row(y + dy)[x..x + width]);
                    }
                } else if let Some(isp) = isp.as_mut().filter(|_| index == 0) {
                    let isp_pred = isp.predict(n, unit.luma_mode, picture, &available, dsp);
                    prediction_buf[..area].copy_from_slice(&isp_pred[..area]);
                } else {
                    predict_into(
                        unit,
                        n,
                        index,
                        picture,
                        &available,
                        sequence,
                        references,
                        dsp,
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
                    dsp.vvc_forward_lmcs(model, prediction);
                }
                super::ciip::blend(unit, n, index, &ciip[index], prediction, dsp);
                let residual = if transform.joint != 0 && index != 0 {
                    &mixed[index - 1]
                } else {
                    &transform.residuals[index]
                };
                let residual = if index != 0 && residual.len() > 4 {
                    if let Some(scale) = &mut chroma_scale {
                        scaled_buf.clear();
                        scaled_buf.extend_from_slice(residual);
                        dsp.vvc_scale_chroma_residual(
                            scale.get((n.x, n.y), picture, &available),
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
                    dsp.vvc_add_residual((width, height), prediction, residual);
                }
                let mut view = picture.plane_mut(plane).view_mut();
                for (dy, src) in prediction.chunks_exact(width).enumerate() {
                    view.row_mut(y + dy)[x..x + width].copy_from_slice(src);
                }
            }
            available.mark_transform(unit, transform);
        }
    }
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
    let shift = usize::from(index != 0);
    let (x, y, width, height) = (
        block.x >> shift,
        block.y >> shift,
        block.width() >> shift,
        block.height() >> shift,
    );
    let direction = if index == 0 {
        unit.luma_mode
    } else {
        unit.chroma_mode
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
    } else if unit.bdpcm[shift] != 0 || (index == 0 && unit.mip.is_some()) {
        predict_from_edges(
            unit,
            (picture, available),
            (plane, shift),
            (x, y, width, height),
            dsp,
            &mut scratch.refs,
            prediction,
        );
    } else if direction >= 67 {
        cclm::predict(
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

/// BDPCM (H.266 §8.4.5.2.12) or MIP (§8.4.5.2.2) prediction of `plane`
/// (channel `shift`) of `unit`'s block at `rect` from its unfiltered edges.
fn predict_from_edges<T: Pixel>(
    unit: &Unit,
    (picture, available): (&Picture<T>, &Availability),
    (plane, shift): (PlaneIndex, usize),
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
    if unit.bdpcm[shift] != 0 {
        dsp.vvc_bdpcm_predict(
            (width, height),
            unit.bdpcm[shift] == 2,
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

/// [`reconstruct`] of a one-slice picture into a new picture.
#[cfg(test)]
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
    reconstruct(
        &mut picture,
        units,
        sequence,
        dimensions,
        joint_sign,
        references,
        dsp.at(nc_common::BitDepth::EIGHT),
        reshape,
    );
    Ok(picture)
}
