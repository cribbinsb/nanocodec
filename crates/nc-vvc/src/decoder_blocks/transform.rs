//! H.266 decoder QP derivation and rectangular inverse residuals via shared DSP.
use super::{Sequence, SliceHeader, Unit};
use nc_dsp::{Dsp, VvcLfnst, VvcTransformKind};

pub(crate) fn inverse_residuals(
    units: &mut [Unit],
    sequence: &Sequence,
    header: &SliceHeader,
    dsp: &Dsp,
) {
    for unit in units {
        let kinds = primary_kinds(unit, sequence);
        let direction = if unit.channel == 0 || unit.chroma_mode >= 67 {
            unit.luma_mode
        } else {
            unit.chroma_mode
        };
        let direction = if unit.mip.is_some() { 0 } else { direction };
        for transform in &mut unit.transforms {
            for (plane, levels) in transform
                .levels
                .iter_mut()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
            {
                let channel = usize::from(plane != 0);
                let block = if plane == 0 {
                    transform.block
                } else {
                    transform.chroma_block.unwrap()
                };
                let qp = qp(transform.joint, plane, sequence, header);
                let mut residual = vec![0; levels.len()];
                if transform.skip[plane] {
                    dsp.vvc_inverse_skip(
                        (block.width() >> channel, block.height() >> channel),
                        levels,
                        qp.max(4 + 6 * sequence.tools.min_qp_ts),
                        unit.bdpcm[channel],
                        &mut residual,
                    );
                } else {
                    let size = (block.width() >> channel, block.height() >> channel);
                    if unit.lfnst != 0 && (plane == 0 || !unit.combined) {
                        let prediction_size = if unit.isp.is_some() {
                            (unit.node.width, unit.node.height)
                        } else {
                            size
                        };
                        dsp.vvc_inverse_lfnst(
                            size,
                            VvcLfnst {
                                index: unit.lfnst,
                                direction,
                                prediction_size,
                            },
                            levels,
                            qp,
                            header.dependent_quant,
                            &mut residual,
                        );
                    } else {
                        let kinds = if plane == 0 {
                            kinds
                        } else {
                            [VvcTransformKind::Dct2; 2]
                        };
                        dsp.vvc_inverse_primary(
                            size,
                            kinds,
                            levels,
                            qp,
                            header.dependent_quant,
                            &mut residual,
                        );
                    }
                }
                *levels = residual;
            }
        }
    }
}

pub(super) fn qp(joint: u8, plane: usize, sequence: &Sequence, header: &SliceHeader) -> u8 {
    (if plane == 0 {
        header.qp
    } else {
        // Joint mode 3 codes one Cb residual at the joint CbCr QP.
        // The other joint modes retain the coded component's QP.
        let index = if joint == 3 { 2 } else { plane - 1 };
        let tables = &sequence.transforms.qp_tables;
        let mapped = tables[if tables.len() == 1 { 0 } else { index }][header.qp as usize];
        // SliceHeader stores the sum of PPS and slice offsets.
        (i32::from(mapped) + i32::from(header.chroma_offsets[index])).clamp(0, 63)
    }) as u8
}

fn primary_kinds(unit: &Unit, sequence: &Sequence) -> [VvcTransformKind; 2] {
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
