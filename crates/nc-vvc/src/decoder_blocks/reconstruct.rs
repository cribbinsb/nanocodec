//! General intra reconstruction into shared pictures with shared DSP prediction.
use super::{
    Sequence, Unit, cclm,
    references::{Availability, References},
};
use nc_common::{
    Picture, PlaneIndex,
    decode::{DecodeError, Result},
};
use nc_dsp::{Dsp, VvcIntraMode};

pub(crate) fn reconstruct(
    units: &[Unit],
    sequence: &Sequence,
    dimensions: [usize; 2],
    joint_sign: bool,
    references: [&[&Picture]; 2],
    dsp: &Dsp,
    reshape: Option<(&nc_dsp::VvcLmcs, bool)>,
) -> Result<Picture> {
    let mut picture = Picture::new(dimensions[0], dimensions[1], 0)
        .map_err(|_| DecodeError::Malformed("VVC picture dimensions"))?;
    let mut available = Availability::new(dimensions);
    let mut chroma_scale = reshape.filter(|(_, enabled)| *enabled).map(|(model, _)| {
        super::reshape::ChromaScale::new(model, units, dimensions, sequence.ctu_log2)
    });
    for unit in units {
        let mut isp = super::isp::Prediction::prepare(unit, &picture, &available);
        let ciip = super::ciip::prepare(unit, &picture, &available, dsp);
        for (transform_index, transform) in unit.transforms.iter().enumerate() {
            let mut mixed = [Vec::new(), Vec::new()];
            if transform.joint != 0 {
                let coded = &transform.levels[if transform.joint == 1 { 2 } else { 1 }];
                let mut cb = vec![0; coded.len()];
                let mut cr = cb.clone();
                nc_dsp::vvc_joint_chroma(coded, transform.joint, joint_sign, &mut cb, &mut cr);
                mixed = [cb, cr];
            }
            for index in if unit.combined {
                0..3
            } else if unit.channel == 0 {
                0..1
            } else {
                1..3
            } {
                let plane = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V][index];
                let n = if index == 0 {
                    transform.block
                } else {
                    let Some(block) = transform.chroma_block else {
                        continue;
                    };
                    block
                };
                let shift = usize::from(index != 0);
                let (x, y, width, height) = (
                    n.x >> shift,
                    n.y >> shift,
                    n.width() >> shift,
                    n.height() >> shift,
                );
                // SBT's base entry already predicts the whole CU. Reuse it so
                // narrow chroma residuals retain the CU's affine subblock grid.
                let overlay = unit.sbt.is_some() && transform_index != 0;
                let mut prediction = if overlay {
                    let view = picture.plane(plane).view();
                    (y..y + height)
                        .flat_map(|row| view.row(row)[x..x + width].iter().copied())
                        .collect()
                } else if let Some(isp) = isp.as_mut().filter(|_| index == 0) {
                    isp.predict(n, unit.luma_mode, &picture, &available, dsp)
                } else {
                    predict(
                        unit, n, index, &picture, &available, sequence, references, dsp,
                    )
                };
                if !overlay
                    && index == 0
                    && unit.motion.is_some()
                    && let Some((model, _)) = reshape
                {
                    dsp.vvc_forward_lmcs(model, &mut prediction);
                }
                super::ciip::blend(unit, n, index, &ciip[index], &mut prediction, dsp);
                let residual = if transform.joint != 0 && index != 0 {
                    &mixed[index - 1]
                } else {
                    &transform.levels[index]
                };
                let mut scaled;
                let residual = if index != 0 && residual.len() > 4 {
                    if let Some(scale) = &mut chroma_scale {
                        scaled = residual.clone();
                        dsp.vvc_scale_chroma_residual(
                            scale.get((n.x, n.y), &picture, &available),
                            &mut scaled,
                        );
                        &scaled
                    } else {
                        residual
                    }
                } else {
                    residual
                };
                if !residual.is_empty() {
                    dsp.vvc_add_residual((width, height), &mut prediction, residual);
                }
                let mut view = picture.plane_mut(plane).view_mut();
                for dy in 0..height {
                    view.row_mut(y + dy)[x..x + width]
                        .copy_from_slice(&prediction[dy * width..(dy + 1) * width]);
                }
            }
            available.mark_transform(unit, transform);
        }
    }
    Ok(picture)
}

fn predict(
    unit: &Unit,
    block: nc_encoder::Block,
    index: usize,
    picture: &Picture,
    available: &Availability,
    sequence: &Sequence,
    references: [&[&Picture]; 2],
    dsp: &Dsp,
) -> Vec<u8> {
    let plane = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V][index];
    let shift = usize::from(index != 0);
    let node = block;
    let (x, y, width, height) = (
        node.x >> shift,
        node.y >> shift,
        node.width() >> shift,
        node.height() >> shift,
    );
    let direction = if index == 0 {
        unit.luma_mode
    } else {
        unit.chroma_mode
    };
    let mut prediction = vec![0; width * height];
    if unit.motion.is_some() {
        super::inter_prediction::predict_unit(dsp, references, unit, plane, block, &mut prediction);
    } else if index == 0 && unit.reference_line != 0 {
        let refs = References::gather_mrl(
            picture,
            available,
            (x, y, width, height),
            unit.reference_line,
        );
        dsp.vvc_mrl_predict(
            (width, height),
            direction,
            unit.reference_line,
            &refs.top,
            &refs.left,
            refs.corner,
            &mut prediction,
        );
    } else if unit.bdpcm[shift] != 0 || (index == 0 && unit.mip.is_some()) {
        let refs = References::gather(picture, available, plane, (x, y, width, height), false);
        if unit.bdpcm[shift] != 0 {
            dsp.vvc_bdpcm_predict(
                (width, height),
                unit.bdpcm[shift] == 2,
                &refs.top,
                &refs.left,
                &mut prediction,
            );
        } else if let Some((mode, transpose)) = unit.mip {
            dsp.vvc_mip_predict(
                (width, height),
                mode,
                transpose,
                &refs.top,
                &refs.left,
                &mut prediction,
            );
        }
    } else if direction >= 67 {
        cclm::predict(
            picture,
            available,
            sequence,
            unit,
            block,
            plane,
            &mut prediction,
        );
    } else {
        let smooth = index == 0 && nc_dsp::vvc_intra_filter_references((width, height), direction);
        let refs = References::gather(picture, available, plane, (x, y, width, height), smooth);
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
            &refs.top,
            &refs.left,
            refs.corner,
            &mut prediction,
        );
    }
    prediction
}
