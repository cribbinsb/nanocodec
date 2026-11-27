//! H.266 decoder QP derivation and rectangular inverse residuals via shared DSP:
//! each coded plane's levels to its 16-bit residual (`Transform::residuals`).
use super::{Sequence, SliceHeader, Unit};
use nc_dsp::{DepthKernels, VvcExtendedBlock, VvcLfnst, VvcTransformKind};

/// Inverse-scale and inverse-transform one non-zero plane of `transform` from
/// its sparse `(pos, level)` entries into `residual_out[..size.0 * size.1]`,
/// restoring `level_buf` to all zeros upon return.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_plane_residual(
    dsp: DepthKernels<'_>,
    unit: &Unit,
    transform: &super::Transform,
    (plane, size, kinds): (usize, (usize, usize), [VvcTransformKind; 2]),
    (sequence, header, lists): (
        &Sequence,
        &SliceHeader,
        Option<&crate::decoder_headers::ScalingLists>,
    ),
    coeffs: super::Levels<'_>,
    factors: &mut Vec<u8>,
    level_buf: &mut [i32; 64 * 64],
    residual_out: &mut [i16],
) {
    let area = size.0 * size.1;
    coeffs.scatter(&mut level_buf[..]);
    let levels = &level_buf[..area];
    let residual = &mut residual_out[..area];
    let channel = usize::from(plane != 0);
    let qp_val = qp(unit.quant(), transform.joint, plane, sequence, header);
    let qp_val = if unit.act {
        colour_transform_qp(qp_val, transform.joint, plane, sequence)
    } else {
        qp_val
    };
    if transform.skip[plane] {
        let qp = qp_val.max(4 + 6 * sequence.tools.min_qp_ts);
        if sequence.range.extended_precision {
            let block = extended(size, (qp, false), None, sequence);
            dsp.vvc_inverse_skip_extended(block, unit.bdpcm[channel], levels, residual);
        } else {
            dsp.vvc_inverse_skip(size, levels, qp, unit.bdpcm[channel], residual);
        }
    } else {
        let scaling = Scaling {
            qp: qp_val,
            dependent: header.dependent_quant,
            lists,
            factors,
        };
        inverse_transform(
            dsp,
            unit,
            sequence,
            (plane, size, kinds),
            scaling,
            levels,
            residual,
        );
    }
    coeffs.clear(&mut level_buf[..]);
}

/// Each coded plane's levels to its residual, at the kernels' depth (the
/// transforms' QP is `Qp′`, H.266 §8.7.3).
#[cfg(test)]
pub(crate) fn inverse_residuals(
    units: &mut [Unit],
    sequence: &Sequence,
    header: &SliceHeader,
    lists: Option<&crate::decoder_headers::ScalingLists>,
    dsp: DepthKernels<'_>,
) {
    let mut factors = Vec::new();
    for unit in units {
        if unit
            .transforms
            .iter()
            .all(|t| t.levels.iter().all(Vec::is_empty))
        {
            continue;
        }
        let quant = unit.quant();
        // Before the transforms are taken: the kinds depend on the first.
        let kinds = primary_kinds(unit, sequence);
        let mut transforms = std::mem::take(&mut unit.transforms);
        for transform in &mut transforms {
            for plane in 0..3 {
                if transform.levels[plane].is_empty() {
                    continue;
                }
                let channel = usize::from(plane != 0);
                let block = if plane == 0 {
                    transform.block
                } else {
                    transform.chroma_block.unwrap()
                };
                let size = sequence.scaled(plane != 0, (block.width(), block.height()));
                let qp = qp(quant, transform.joint, plane, sequence, header);
                let levels = &transform.levels[plane];
                let residual = &mut transform.residuals[plane];
                residual.clear();
                residual.resize(levels.len(), 0);
                if transform.skip[plane] {
                    dsp.vvc_inverse_skip(
                        size,
                        levels,
                        qp.max(4 + 6 * sequence.tools.min_qp_ts),
                        unit.bdpcm[channel],
                        residual,
                    );
                } else {
                    let scaling = Scaling {
                        qp,
                        dependent: header.dependent_quant,
                        lists,
                        factors: &mut factors,
                    };
                    inverse_transform(
                        dsp,
                        unit,
                        sequence,
                        (plane, size, kinds),
                        scaling,
                        levels,
                        residual,
                    );
                }
            }
        }
        unit.transforms = transforms;
    }
}

/// Inverse scaling parameters of one transform block.
struct Scaling<'a> {
    qp: u8,
    dependent: bool,
    /// The slice's explicit scaling lists, and scratch for a block's factors.
    lists: Option<&'a crate::decoder_headers::ScalingLists>,
    factors: &'a mut Vec<u8>,
}

/// Scale and inverse-transform (LFNST, then the primary transform) one
/// plane of a transform block that does not skip the transform;
/// `luma_kinds` are the CU's luma primary transforms.
fn inverse_transform(
    dsp: DepthKernels<'_>,
    unit: &Unit,
    sequence: &Sequence,
    (plane, size, luma_kinds): (usize, (usize, usize), [VvcTransformKind; 2]),
    scaling: Scaling<'_>,
    levels: &[i32],
    residual: &mut [i16],
) {
    // CCLM chroma takes the collocated luma mode, unmapped (VTM
    // `xInvLfnst`); other chroma its `IntraPredModeC`.
    let direction = if unit.mip.is_some() {
        0
    } else if unit.channel == 0 || unit.chroma_mode >= 67 {
        unit.luma_mode
    } else {
        super::reconstruct::chroma_direction(unit, sequence)
    };
    let lfnst = (unit.lfnst != 0 && (plane == 0 || !unit.combined)).then(|| VvcLfnst {
        index: unit.lfnst,
        direction,
        prediction_size: if unit.isp.is_some() {
            (unit.node.width, unit.node.height)
        } else {
            size
        },
    });
    let kinds = if plane == 0 && lfnst.is_none() {
        luma_kinds
    } else {
        [VvcTransformKind::Dct2; 2]
    };
    let Scaling {
        qp,
        dependent,
        lists,
        factors,
    } = scaling;
    // With ACT, the SPS may keep the matrices to one colour space.
    let colour_space = sequence.tools.act_scaling != Some(unit.act);
    let lists =
        lists.filter(|_| colour_space && super::scaling_lists::applies(sequence, lfnst.is_some()));
    if sequence.range.extended_precision {
        let matrix = lists.map(|lists| {
            super::scaling_lists::factors(lists, unit.inter.is_none(), plane, size, factors);
            &factors[..]
        });
        let block = extended(size, (qp, dependent), matrix, sequence);
        dsp.vvc_inverse_extended(block, (kinds, lfnst), levels, residual);
    } else if let Some(lists) = lists {
        let intra = unit.inter.is_none();
        super::scaling_lists::factors(lists, intra, plane, size, factors);
        dsp.vvc_inverse_with_matrix(
            size,
            kinds,
            lfnst,
            levels,
            (qp, dependent),
            factors,
            residual,
        );
    } else if let Some(lfnst) = lfnst {
        dsp.vvc_inverse_lfnst(size, lfnst, levels, qp, dependent, residual);
    } else {
        dsp.vvc_inverse_primary(size, kinds, levels, qp, dependent, residual);
    }
}

/// One block of extended-precision inverse scaling and transform
/// (`sps_extended_precision_flag`, §7.4.3.22).
fn extended<'a>(
    size: (usize, usize),
    quantizer: (u8, bool),
    matrix: Option<&'a [u8]>,
    sequence: &Sequence,
) -> VvcExtendedBlock<'a> {
    VvcExtendedBlock {
        size,
        quantizer,
        matrix,
        log2_range: sequence.log2_transform_range(),
    }
}

/// The quantisation parameter of one plane of a transform unit of `unit`
/// (H.266 §8.7.1: `Qp′Y`, `Qp′Cb`, `Qp′Cr` or `Qp′CbCr`, each the QP plus
/// `QpBdOffset`, so `0 ..= 63 + QpBdOffset`).
pub(super) fn qp(
    (luma, offsets): (i8, [i8; 3]),
    joint: u8,
    plane: usize,
    sequence: &Sequence,
    header: &SliceHeader,
) -> u8 {
    let qp_bd_offset = sequence.qp_bd_offset();
    let qp = if plane == 0 {
        i32::from(luma)
    } else {
        // Joint mode 3 codes one Cb residual at the joint CbCr QP.
        // The other joint modes retain the coded component's QP.
        let index = if joint == 3 { 2 } else { plane - 1 };
        let tables = &sequence.transforms.qp_tables;
        let mapped = tables[if tables.len() == 1 { 0 } else { index }].map(i32::from(luma));
        // SliceHeader stores the sum of PPS and slice offsets.
        (mapped + i32::from(header.chroma_offsets[index]) + i32::from(offsets[index]))
            .clamp(-qp_bd_offset, MAX_QP)
    };
    (qp + qp_bd_offset) as u8
}

/// The largest QP (`QpY`, chroma QPs) at every depth (H.266 §7.4.3.4).
const MAX_QP: i32 = 63;

/// The QP offsets of ACT CUs' Y, Cb, Cr and joint `CbCr` residuals (H.266
/// §8.7.1: `ActQpOffsetY` and the others; −5, 1, 3, 1).
const ACT_QP_OFFSETS: [i32; 4] = [-5, 1, 3, 1];

/// `qp` (a `Qp′` of [`qp`]) of an ACT CU's `plane`, moved by its ACT
/// offset and clipped to `0 ..= 63 + QpBdOffset` (§8.7.1).
fn colour_transform_qp(qp: u8, joint: u8, plane: usize, sequence: &Sequence) -> u8 {
    let component = if plane != 0 && joint == 3 { 3 } else { plane };
    (i32::from(qp) + ACT_QP_OFFSETS[component]).clamp(0, MAX_QP + sequence.qp_bd_offset()) as u8
}

pub(super) fn primary_kinds(unit: &Unit, sequence: &Sequence) -> [VvcTransformKind; 2] {
    use VvcTransformKind::{Dct2, Dct8, Dst7};
    let mts = sequence.transforms.mts;
    if !mts[0] || (unit.isp.is_some() && unit.lfnst != 0) {
        return [Dct2; 2];
    }
    if let Some(selection) = unit.sbt {
        return selection.primary(unit.node);
    }
    let implicit = unit.inter.is_none() && !mts[1] && unit.lfnst == 0 && unit.mip.is_none();
    if implicit || unit.isp.is_some() {
        let block = unit.transforms[0].block;
        return [block.width(), block.height()]
            .map(|n| if (4..=16).contains(&n) { Dst7 } else { Dct2 });
    }
    if unit.mts >= 2 {
        let index = unit.mts - 2;
        [
            if index & 1 == 0 { Dst7 } else { Dct8 },
            if index & 2 == 0 { Dst7 } else { Dct8 },
        ]
    } else {
        [Dct2; 2]
    }
}
