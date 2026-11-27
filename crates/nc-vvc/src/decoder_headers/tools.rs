//! H.266 SPS coding-tool flags and their conditionally present parameters.
use super::{DecodeError, Reader, Result, rpl};
use nc_common::BitDepth;

/// The largest QP (`QpY`, chroma QPs) at every depth (H.266 §7.4.3.4).
const MAX_QP: i32 = 63;
/// The deepest `QpBdOffset` of the Main 10 profiles (10 bits): the chroma
/// QP tables hold `−QP_TABLE_LOW ..= 63`.
pub(crate) const QP_TABLE_LOW: i32 = 12;
/// Entries of a chroma QP table.
const QP_TABLE_SIZE: usize = (QP_TABLE_LOW + MAX_QP + 1) as usize;

/// One `ChromaQpTable[i]` of H.266 §7.4.3.4, indexed by luma QP from
/// `−QpBdOffset` to 63.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChromaQpMapping([i8; QP_TABLE_SIZE]);

impl ChromaQpMapping {
    /// The chroma QP of luma QP `qp` (`−QpBdOffset ..= 63`).
    pub fn map(&self, qp: i32) -> i32 {
        i32::from(self.0[(qp + QP_TABLE_LOW) as usize])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Transforms {
    pub skip_log2: Option<u8>,
    pub bdpcm: bool,
    pub mts: [bool; 3],
    pub lfnst: bool,
    pub joint_cbcr: bool,
    pub qp_tables: Vec<ChromaQpMapping>,
}
impl Transforms {
    /// `qp_bd_offset`: `QpBdOffset` of the sequence (the chroma QP tables'
    /// lower end).
    pub fn parse(r: &mut Reader<'_>, chroma: bool, qp_bd_offset: i32) -> Result<Self> {
        let skip_log2 = if r.flag()? {
            Some(r.ue(3, "VVC transform skip size")? as u8 + 2)
        } else {
            None
        };
        let bdpcm = skip_log2.is_some() && r.flag()?;
        let mts_enabled = r.flag()?;
        let mts = [
            mts_enabled,
            mts_enabled && r.flag()?,
            mts_enabled && r.flag()?,
        ];
        let lfnst = r.flag()?;
        let joint_cbcr = chroma && r.flag()?;
        let count = if !chroma {
            0
        } else if r.flag()? {
            1
        } else if joint_cbcr {
            3
        } else {
            2
        };
        let qp_tables = (0..count)
            .map(|_| qp_table(r, qp_bd_offset))
            .collect::<Result<_>>()?;
        Ok(Self {
            skip_log2,
            bdpcm,
            mts,
            lfnst,
            joint_cbcr,
            qp_tables,
        })
    }
}

/// One chroma QP mapping table (H.266 §7.3.2.4, §7.4.3.4): pivot points
/// from `qpInVal[0] = sps_qp_table_start_minus26 + 26`, linear between them,
/// slope one beyond them, every value clipped to `−QpBdOffset ..= 63`.
fn qp_table(r: &mut Reader<'_>, qp_bd_offset: i32) -> Result<ChromaQpMapping> {
    let start = r.se(-26 - qp_bd_offset, 36, "VVC chroma QP start")? + 26;
    let count = r.ue((MAX_QP - 1 - start) as u32, "VVC chroma QP points")? + 1;
    let clip = |qp: i32| qp.clamp(-qp_bd_offset, MAX_QP);
    let mut table = [0i32; QP_TABLE_SIZE];
    let index = |qp: i32| (qp + QP_TABLE_LOW) as usize;
    let (mut input, mut output) = (start, start);
    table[index(start)] = start;
    for _ in 0..count {
        let step = r.ue(MAX_QP as u32, "VVC chroma QP input delta")? as i32;
        let delta = (r.ue(127, "VVC chroma QP output delta")? ^ step as u32) as i32;
        let next = input + step + 1;
        if next > MAX_QP || output + delta > MAX_QP {
            return Err(DecodeError::Malformed("VVC chroma QP endpoints"));
        }
        // Truncating division from the table's value at the pivot, as the
        // specification writes it.
        let base = table[index(input)];
        for offset in 1..=step + 1 {
            table[index(input + offset)] = base + (delta * offset + (step + 1) / 2) / (step + 1);
        }
        input = next;
        output += delta;
    }
    for qp in (-qp_bd_offset..start).rev() {
        table[index(qp)] = clip(table[index(qp + 1)] - 1);
    }
    for qp in input + 1..=MAX_QP {
        table[index(qp)] = clip(table[index(qp - 1)] + 1);
    }
    Ok(ChromaQpMapping(table.map(|qp| clip(qp) as i8)))
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // Independent coding-tool enable flags.
pub(crate) struct Tools {
    pub sao: bool,
    pub alf: bool,
    pub ccalf: bool,
    pub lmcs: bool,
    pub weighted: [bool; 2],
    pub long_term: bool,
    pub idr_rpl: bool,
    pub rpl1_same: bool,
    pub rpls: [Vec<rpl::List>; 2],
    pub wraparound: bool,
    pub temporal_mvp: bool,
    pub subblock_mvp: bool,
    pub amvr: bool,
    pub bdof: bool,
    pub bdof_control: bool,
    pub smvd: bool,
    pub dmvr: bool,
    pub dmvr_control: bool,
    pub mmvd: bool,
    pub mmvd_fullpel: bool,
    pub max_merge: u8,
    pub sbt: bool,
    pub affine: bool,
    pub max_subblock_merge: u8,
    pub affine_six: bool,
    pub affine_amvr: bool,
    pub prof: bool,
    pub prof_control: bool,
    pub bcw: bool,
    pub ciip: bool,
    pub gpm: bool,
    pub max_gpm: u8,
    pub parallel_merge_log2: u8,
    pub isp: bool,
    pub mrl: bool,
    pub mip: bool,
    pub cclm: bool,
    pub chroma_collocated: [bool; 2],
    pub palette: bool,
    pub min_qp_ts: u8,
    pub ibc: bool,
    pub max_ibc_merge: u8,
    pub ladf: Vec<(u16, i8)>,
    pub scaling_list: bool,
    pub scaling_lfnst_disabled: bool,
    pub dependent_quant: bool,
    pub sign_hiding: bool,
    pub virtual_boundaries: bool,
    pub virtual_boundaries_present: bool,
    pub boundaries: [Vec<u32>; 2],
}
impl Tools {
    pub fn parse(
        r: &mut Reader<'_>,
        (chroma, depth): (bool, BitDepth),
        ctu_log2: u32,
        poc_bits: u32,
        dimensions: [u32; 2],
        transforms: &Transforms,
        refers_to_vps: bool,
    ) -> Result<Self> {
        let mut t = Self::default();
        t.sao = r.flag()?;
        t.alf = r.flag()?;
        t.ccalf = t.alf && chroma && r.flag()?;
        t.lmcs = r.flag()?;
        t.weighted = [r.flag()?, r.flag()?];
        t.long_term = r.flag()?;
        // sps_inter_layer_prediction_enabled_flag (§7.3.2.4) is 0 in an
        // independent layer, as the base layer is (§7.4.3.4); with it set,
        // reference lists carry inter-layer entries we do not decode.
        if refers_to_vps && r.flag()? {
            return Err(DecodeError::Unsupported("VVC inter-layer prediction"));
        }
        t.idr_rpl = r.flag()?;
        t.rpl1_same = r.flag()?;
        for list in 0..if t.rpl1_same { 1 } else { 2 } {
            let count = r.ue(64, "VVC SPS reference lists")?;
            t.rpls[list] = (0..count)
                .map(|_| {
                    rpl::List::parse(r, t.long_term, t.weighted.contains(&true), poc_bits, false)
                })
                .collect::<Result<_>>()?;
        }
        if t.rpl1_same {
            let [l0, l1] = &mut t.rpls;
            l1.clone_from(l0);
        }
        t.wraparound = r.flag()?;
        t.temporal_mvp = r.flag()?;
        t.subblock_mvp = t.temporal_mvp && r.flag()?;
        t.amvr = r.flag()?;
        t.bdof = r.flag()?;
        t.bdof_control = t.bdof && r.flag()?;
        t.smvd = r.flag()?;
        t.dmvr = r.flag()?;
        t.dmvr_control = t.dmvr && r.flag()?;
        t.mmvd = r.flag()?;
        t.mmvd_fullpel = t.mmvd && r.flag()?;
        t.max_merge = 6 - r.ue(5, "VVC merge candidates")? as u8;
        t.sbt = r.flag()?;
        t.affine = r.flag()?;
        if t.affine {
            t.max_subblock_merge =
                5 - r.ue(5 - u32::from(t.subblock_mvp), "VVC affine merge candidates")? as u8;
            t.affine_six = r.flag()?;
            t.affine_amvr = t.amvr && r.flag()?;
            t.prof = r.flag()?;
            t.prof_control = t.prof && r.flag()?;
        } else {
            t.max_subblock_merge = u8::from(t.subblock_mvp);
        }
        t.bcw = r.flag()?;
        t.ciip = r.flag()?;
        t.gpm = t.max_merge >= 2 && r.flag()?;
        if t.gpm {
            t.max_gpm = if t.max_merge >= 3 {
                t.max_merge - r.ue(u32::from(t.max_merge - 2), "VVC geometric candidates")? as u8
            } else {
                2
            };
        }
        t.parallel_merge_log2 = 2 + r.ue(ctu_log2 - 2, "VVC parallel merge level")? as u8;
        t.isp = r.flag()?;
        t.mrl = r.flag()?;
        t.mip = r.flag()?;
        t.cclm = chroma && r.flag()?;
        if chroma {
            t.chroma_collocated = [r.flag()?, r.flag()?];
        }
        t.palette = r.flag()?;
        if transforms.skip_log2.is_some() || t.palette {
            t.min_qp_ts = r.ue(8, "VVC minimum transform skip QP")? as u8;
        }
        t.ibc = r.flag()?;
        if t.ibc {
            t.max_ibc_merge = 6 - r.ue(5, "VVC IBC merge candidates")? as u8;
        }
        t.parse_ladf(r, depth)?;
        t.scaling_list = r.flag()?;
        t.scaling_lfnst_disabled = t.scaling_list && transforms.lfnst && r.flag()?;
        t.dependent_quant = r.flag()?;
        t.sign_hiding = r.flag()?;
        t.virtual_boundaries = r.flag()?;
        t.virtual_boundaries_present = t.virtual_boundaries && r.flag()?;
        if t.virtual_boundaries_present {
            for (axis, size) in dimensions.into_iter().enumerate() {
                let count = r.ue(if size <= 8 { 0 } else { 3 }, "VVC virtual boundary count")?;
                for _ in 0..count {
                    let position =
                        (r.ue(size.div_ceil(8) - 2, "VVC virtual boundary position")? + 1) * 8;
                    if t.boundaries[axis]
                        .last()
                        .is_some_and(|&last| last >= position)
                    {
                        return Err(DecodeError::Malformed("VVC virtual boundary order"));
                    }
                    t.boundaries[axis].push(position);
                }
            }
        }
        Ok(t)
    }
    /// The LADF intervals (H.266 §7.3.2.4): lower bounds are sample levels
    /// of `depth` (`sps_ladf_delta_threshold_minus1` up to `2^BitDepth − 3`).
    fn parse_ladf(&mut self, r: &mut Reader<'_>, depth: BitDepth) -> Result<()> {
        if r.flag()? {
            let count = r.bits(2)? + 2;
            self.ladf.push((0, r.se(-63, 63, "VVC LADF offset")? as i8));
            let mut lower = 0;
            let largest = depth.max() as u16;
            for _ in 1..count {
                let offset = r.se(-63, 63, "VVC LADF offset")? as i8;
                lower += r.ue(u32::from(largest) - 2, "VVC LADF threshold")? as u16 + 1;
                if lower > largest {
                    return Err(DecodeError::Malformed("VVC LADF interval"));
                }
                self.ladf.push((lower, offset));
            }
        }
        Ok(())
    }
}
