//! H.266 §7.3.2.7 picture headers and picture-level coding constraints.
use super::{DecodeError, PictureParameters, Reader, Result, Sequence, rpl, sps::Partition};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct AlfSelection {
    pub enabled: [bool; 3],
    pub luma: Vec<u8>,
    pub chroma: Option<u8>,
    pub cross: [Option<u8>; 2],
}
impl AlfSelection {
    pub(super) fn parse(r: &mut Reader<'_>, s: &Sequence) -> Result<Self> {
        let mut value = Self::default();
        value.enabled[0] = r.flag()?;
        if value.enabled[0] {
            let count = r.bits(3)?;
            for _ in 0..count {
                value.luma.push(r.bits(3)? as u8);
            }
            if s.chroma {
                value.enabled[1] = r.flag()?;
                value.enabled[2] = r.flag()?;
            }
            if value.enabled[1] || value.enabled[2] {
                value.chroma = Some(r.bits(3)? as u8);
            }
            if s.tools.ccalf {
                for selected in &mut value.cross {
                    if r.flag()? {
                        *selected = Some(r.bits(3)? as u8);
                    }
                }
            }
        }
        Ok(value)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Independent PH syntax flags.
pub(crate) struct PictureHeader {
    pub pps_id: u8,
    pub irap_or_gdr: bool,
    pub gdr: bool,
    pub non_reference: bool,
    pub inter_allowed: bool,
    pub intra_allowed: bool,
    pub order: PictureOrder,
    pub alf: AlfSelection,
    pub lmcs: Option<u8>,
    pub chroma_scale: bool,
    pub scaling: Option<u8>,
    pub boundaries: [Vec<u32>; 2],
    pub output: bool,
    pub references: [rpl::Selected; 2],
    pub partitions: [Partition; 3],
    pub qp_subdiv: [u8; 2],
    pub chroma_subdiv: [u8; 2],
    pub motion: Motion,
    pub weights: super::weights::Weights,
    pub qp: Option<i32>,
    pub joint_sign: bool,
    pub sao: [bool; 2],
    pub deblock: Deblocking,
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // Each inter prediction tool has its own PH flag.
pub(crate) struct Motion {
    pub temporal: bool,
    pub collocated: [u8; 2],
    pub fullpel_mmvd: bool,
    pub mvd_l1_zero: bool,
    pub bdof_disabled: bool,
    pub dmvr_disabled: bool,
    pub prof_disabled: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Deblocking {
    pub disabled: bool,
    pub offsets: [[i8; 2]; 3],
}
impl Deblocking {
    pub(super) fn parse(r: &mut Reader<'_>, p: &PictureParameters, present: bool) -> Result<Self> {
        let override_parameters = present && r.flag()?;
        let disabled = if override_parameters {
            !p.deblock.disabled && r.flag()?
        } else {
            p.deblock.disabled
        };
        let mut offsets = p.deblock.offsets;
        if override_parameters && !disabled {
            offsets[0] = filter_offsets(r)?;
            if p.chroma_offsets_present {
                offsets[1] = filter_offsets(r)?;
                offsets[2] = filter_offsets(r)?;
            }
        }
        if !p.chroma_offsets_present {
            offsets[1] = offsets[0];
            offsets[2] = offsets[0];
        }
        Ok(Self { disabled, offsets })
    }
}
fn filter_offsets(r: &mut Reader<'_>) -> Result<[i8; 2]> {
    Ok([
        r.se(-12, 12, "VVC beta offset")? as i8,
        r.se(-12, 12, "VVC tc offset")? as i8,
    ])
}
impl PictureHeader {
    pub(crate) fn parse(r: &mut Reader<'_>, s: &Sequence, p: &PictureParameters) -> Result<Self> {
        let irap_or_gdr = r.flag()?;
        let non_reference = r.flag()?;
        let gdr = irap_or_gdr && r.flag()?;
        if gdr && !s.gdr {
            return Err(DecodeError::Malformed("VVC GDR disabled by SPS"));
        }
        let inter_allowed = r.flag()?;
        let intra_allowed = !inter_allowed || r.flag()?;
        let pps_id = r.ue(63, "VVC PH parameter set")? as u8;
        if pps_id != p.id {
            return Err(DecodeError::Malformed("VVC PH missing PPS"));
        }
        let order = PictureOrder::parse(r, s, gdr)?;
        let alf = if s.tools.alf && p.location.alf_in_ph {
            AlfSelection::parse(r, s)?
        } else {
            AlfSelection::default()
        };
        let lmcs = if s.tools.lmcs && r.flag()? {
            Some(r.bits(2)? as u8)
        } else {
            None
        };
        let chroma_scale = lmcs.is_some() && s.chroma && r.flag()?;
        let scaling = if s.tools.scaling_list && r.flag()? {
            Some(r.bits(3)? as u8)
        } else {
            None
        };
        let boundaries = boundaries(r, s, p)?;
        let output = !p.output_flag || non_reference || r.flag()?;
        let references = if p.location.rpl_in_ph {
            rpl::select(r, s, p)?
        } else {
            Default::default()
        };
        if p.location.rpl_in_ph && inter_allowed && references[0].entries.is_empty() {
            return Err(DecodeError::Malformed("VVC empty PH L0"));
        }
        let (partitions, qp_subdiv, chroma_subdiv) =
            partitions(r, s, p, [intra_allowed, inter_allowed])?;
        let motion = if inter_allowed {
            Motion::parse(r, s, p, &references)?
        } else {
            Motion::default()
        };
        let weights = if inter_allowed && p.location.weights_in_ph {
            super::weights::Weights::parse(
                r,
                s,
                p,
                std::array::from_fn(|i| references[i].entries.len() as u8),
            )?
        } else {
            super::weights::Weights::default()
        };
        let qp = if p.location.qp_in_ph {
            Some(p.initial_qp + r.se(-p.initial_qp, 63 - p.initial_qp, "VVC PH QP")?)
        } else {
            None
        };
        let joint_sign = s.transforms.joint_cbcr && r.flag()?;
        let sao = if s.tools.sao && p.location.sao_in_ph {
            [r.flag()?, s.chroma && r.flag()?]
        } else {
            [false; 2]
        };
        let deblock = Deblocking::parse(r, p, p.deblock.in_ph)?;
        extension(r, p.ph_extension)?;
        Ok(Self {
            pps_id,
            irap_or_gdr,
            gdr,
            non_reference,
            inter_allowed,
            intra_allowed,
            order,
            alf,
            lmcs,
            chroma_scale,
            scaling,
            boundaries,
            output,
            references,
            partitions,
            qp_subdiv,
            chroma_subdiv,
            motion,
            weights,
            qp,
            joint_sign,
            sao,
            deblock,
        })
    }
}
pub(super) fn extension(r: &mut Reader<'_>, present: bool) -> Result<()> {
    if present {
        for _ in 0..r.ue(256, "VVC header extension length")? {
            r.bits(8)?;
        }
    }
    Ok(())
}
fn boundaries(r: &mut Reader<'_>, s: &Sequence, p: &PictureParameters) -> Result<[Vec<u32>; 2]> {
    let mut boundaries = s.tools.boundaries.clone();
    if s.tools.virtual_boundaries && !s.tools.virtual_boundaries_present && r.flag()? {
        for (axis, positions) in boundaries.iter_mut().enumerate() {
            let size = p.dimensions[axis];
            let count = r.ue(if size <= 8 { 0 } else { 3 }, "VVC PH boundary count")?;
            for _ in 0..count {
                let position = (r.ue(size.div_ceil(8) - 2, "VVC PH boundary")? + 1) * 8;
                if positions
                    .last()
                    .is_some_and(|&prev| position < prev + (1 << s.ctu_log2))
                {
                    return Err(DecodeError::Malformed("VVC PH boundary spacing"));
                }
                positions.push(position);
            }
        }
        if boundaries.iter().all(Vec::is_empty) {
            return Err(DecodeError::Malformed("VVC empty virtual boundaries"));
        }
    }
    Ok(boundaries)
}
type PartitionOverrides = ([Partition; 3], [u8; 2], [u8; 2]);
fn partitions(
    r: &mut Reader<'_>,
    s: &Sequence,
    p: &PictureParameters,
    allowed: [bool; 2],
) -> Result<PartitionOverrides> {
    let override_constraints = s.partition_override && r.flag()?;
    let mut partitions = s.partitions.clone();
    let mut qp = [0; 2];
    let mut chroma = [0; 2];
    for index in 0..2 {
        if !allowed[index] {
            continue;
        }
        if override_constraints {
            partitions[index] = Partition::parse(
                r,
                u32::from(s.min_cb_log2),
                u32::from(s.ctu_log2),
                index == 0 && s.dual_tree,
            )?;
            if index == 0 && s.dual_tree {
                partitions[2] =
                    Partition::parse(r, u32::from(s.min_cb_log2), u32::from(s.ctu_log2), true)?;
            }
        }
        let qt = if index == 1 && !override_constraints {
            s.min_cb_log2
        } else {
            partitions[index].min_qt_log2
        };
        let max = 2 * u32::from(s.ctu_log2 - qt + partitions[index].mtt_depth);
        if p.cu_qp_delta {
            qp[index] = r.ue(max, "VVC QP subdivision")? as u8;
        }
        if !p.cu_chroma_offsets.is_empty() {
            chroma[index] = r.ue(max, "VVC chroma QP subdivision")? as u8;
        }
    }
    Ok((partitions, qp, chroma))
}
impl Motion {
    fn parse(
        r: &mut Reader<'_>,
        s: &Sequence,
        p: &PictureParameters,
        lists: &[rpl::Selected; 2],
    ) -> Result<Self> {
        let temporal = s.tools.temporal_mvp && r.flag()?;
        let mut collocated = [0; 2];
        if temporal && p.location.rpl_in_ph {
            if !lists[1].entries.is_empty() {
                collocated[0] = u8::from(!r.flag()?);
            }
            let count = lists[usize::from(collocated[0])].entries.len();
            if count == 0 {
                return Err(DecodeError::Malformed(
                    "VVC empty collocated reference list",
                ));
            }
            if count > 1 {
                collocated[1] = r.ue(count as u32 - 1, "VVC collocated reference")? as u8;
            }
        }
        let fullpel_mmvd = s.tools.mmvd_fullpel && r.flag()?;
        let presence = !p.location.rpl_in_ph || !lists[1].entries.is_empty();
        let mvd_l1_zero = presence && r.flag()?;
        let bdof_disabled = if s.tools.bdof_control {
            !presence || r.flag()?
        } else {
            !s.tools.bdof
        };
        let dmvr_disabled = if s.tools.dmvr_control {
            !presence || r.flag()?
        } else {
            !s.tools.dmvr
        };
        let prof_disabled = if s.tools.prof_control {
            r.flag()?
        } else {
            !s.tools.prof
        };
        Ok(Self {
            temporal,
            collocated,
            fullpel_mmvd,
            mvd_l1_zero,
            bdof_disabled,
            dmvr_disabled,
            prof_disabled,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PictureOrder {
    pub lsb: u32,
    pub msb_cycle: Option<u32>,
    pub recovery: Option<u32>,
}
impl PictureOrder {
    fn parse(r: &mut Reader<'_>, s: &Sequence, gdr: bool) -> Result<Self> {
        let poc_lsb = r.bits(u32::from(s.poc_bits))?;
        let recovery = if gdr {
            Some(r.ue(1 << s.poc_bits, "VVC recovery POC")?)
        } else {
            None
        };
        for &present in &s.extra_ph {
            if present {
                r.flag()?;
            }
        }
        let poc_msb_cycle = if let Some(bits) = s.poc_msb_bits {
            if r.flag()? {
                Some(r.bits(u32::from(bits))?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            lsb: poc_lsb,
            msb_cycle: poc_msb_cycle,
            recovery,
        })
    }
}
