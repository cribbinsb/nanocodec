//! Quantiser, segmentation and block-level delta syntax of the frame header
//! (AV1 §5.9.12–14, §5.9.17–18).
use super::PRIMARY_REF_NONE;
use crate::backend::decoder::sequence::Sequence;
use nc_common::decode::{BitReader, DecodeError};

/// `MAX_SEGMENTS`, `SEG_LVL_MAX` (AV1 §3).
pub(crate) const MAX_SEGMENTS: usize = 8;
pub(crate) const SEG_LVL_MAX: usize = 8;
/// Segment features (AV1 §6.8.13).
pub(crate) const SEG_LVL_ALT_Q: usize = 0;
pub(crate) const SEG_LVL_REF_FRAME: usize = 5;
/// `MAXQ`: the largest qindex.
const MAXQ: i32 = 255;
/// `Segmentation_Feature_Bits`, `_Signed`, `_Max` (AV1 §5.9.14).
const FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
const FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];
/// Bits of a coded `delta_q` (`su(1+6)`).
const DELTA_Q_BITS: u32 = 7;

/// `quantization_params()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Quantization {
    pub base_q_idx: u32,
    pub delta_y_dc: i32,
    pub delta_u_dc: i32,
    pub delta_u_ac: i32,
    pub delta_v_dc: i32,
    pub delta_v_ac: i32,
    /// `qm_y`, `qm_u`, `qm_v` when `using_qmatrix`.
    pub qm: Option<[u8; 3]>,
}
impl Quantization {
    /// Every DC/AC delta is zero (part of the lossless test, §5.9.2).
    pub(crate) fn deltas_zero(&self) -> bool {
        [
            self.delta_y_dc,
            self.delta_u_dc,
            self.delta_u_ac,
            self.delta_v_dc,
            self.delta_v_ac,
        ] == [0; 5]
    }
}

/// `FeatureEnabled[][]` and `FeatureData[][]`: what slots save.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct SegmentationFeatures {
    pub enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// `segmentation_params()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(crate) struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub features: SegmentationFeatures,
    /// `SegIdPreSkip`: segment ids are read before the skip flag.
    pub pre_skip: bool,
    /// `LastActiveSegId`.
    pub last_active: u8,
}
impl Segmentation {
    /// `get_qindex(1, segmentId)` (§7.12.2) ignoring block deltas.
    pub(crate) fn qindex(&self, base_q_idx: u32, segment: usize) -> u32 {
        if self.enabled && self.features.enabled[segment][SEG_LVL_ALT_Q] {
            let q = base_q_idx as i32 + i32::from(self.features.data[segment][SEG_LVL_ALT_Q]);
            q.clamp(0, MAXQ) as u32
        } else {
            base_q_idx
        }
    }
}

/// `delta_q_params()` and `delta_lf_params()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(crate) struct DeltaParams {
    pub q_present: bool,
    /// `delta_q_res` as a shift.
    pub q_res: u8,
    pub lf_present: bool,
    pub lf_res: u8,
    pub lf_multi: bool,
}

fn read_su(r: &mut BitReader<'_>, bits: u32) -> Result<i32, DecodeError> {
    let value = r.read_bits(bits)? as i32;
    let shift = 32 - bits;
    Ok((value << shift) >> shift)
}

fn read_delta_q(r: &mut BitReader<'_>) -> Result<i32, DecodeError> {
    if r.read_bit()? {
        read_su(r, DELTA_Q_BITS)
    } else {
        Ok(0)
    }
}

pub(crate) fn quantization(
    r: &mut BitReader<'_>,
    seq: &Sequence,
) -> Result<Quantization, DecodeError> {
    let mut q = Quantization {
        base_q_idx: r.read_bits(8)?,
        delta_y_dc: read_delta_q(r)?,
        ..Quantization::default()
    };
    if !seq.mono_chrome {
        let diff_uv_delta = seq.separate_uv_delta_q && r.read_bit()?;
        q.delta_u_dc = read_delta_q(r)?;
        q.delta_u_ac = read_delta_q(r)?;
        (q.delta_v_dc, q.delta_v_ac) = if diff_uv_delta {
            (read_delta_q(r)?, read_delta_q(r)?)
        } else {
            (q.delta_u_dc, q.delta_u_ac)
        };
    }
    if r.read_bit()? {
        let qm_y = r.read_bits(4)? as u8;
        let qm_u = r.read_bits(4)? as u8;
        let qm_v = if seq.separate_uv_delta_q {
            r.read_bits(4)? as u8
        } else {
            qm_u
        };
        q.qm = Some([qm_y, qm_u, qm_v]);
    }
    Ok(q)
}

/// `segmentation_params()`; `previous` holds the primary reference's
/// features (`load_segmentation_params`), `None` for past independence.
pub(crate) fn segmentation(
    r: &mut BitReader<'_>,
    primary_ref_frame: u8,
    previous: Option<&SegmentationFeatures>,
) -> Result<Segmentation, DecodeError> {
    let mut s = Segmentation {
        enabled: r.read_bit()?,
        ..Segmentation::default()
    };
    if !s.enabled {
        return Ok(s);
    }
    let update_data = if primary_ref_frame == PRIMARY_REF_NONE {
        s.update_map = true;
        true
    } else {
        s.update_map = r.read_bit()?;
        s.temporal_update = s.update_map && r.read_bit()?;
        r.read_bit()?
    };
    if update_data {
        for segment in 0..MAX_SEGMENTS {
            for feature in 0..SEG_LVL_MAX {
                let enabled = r.read_bit()?;
                s.features.enabled[segment][feature] = enabled;
                let limit = FEATURE_MAX[feature];
                let value = if !enabled {
                    0
                } else if FEATURE_SIGNED[feature] {
                    read_su(r, 1 + FEATURE_BITS[feature])?.clamp(-limit, limit)
                } else {
                    (r.read_bits(FEATURE_BITS[feature])? as i32).clamp(0, limit)
                };
                s.features.data[segment][feature] = value as i16;
            }
        }
    } else if let Some(previous) = previous {
        s.features = *previous;
    }
    for segment in 0..MAX_SEGMENTS {
        for feature in 0..SEG_LVL_MAX {
            if s.features.enabled[segment][feature] {
                s.last_active = segment as u8;
                s.pre_skip |= feature >= SEG_LVL_REF_FRAME;
            }
        }
    }
    Ok(s)
}

pub(crate) fn delta_params(
    r: &mut BitReader<'_>,
    base_q_idx: u32,
    allow_intrabc: bool,
) -> Result<DeltaParams, DecodeError> {
    let mut d = DeltaParams {
        q_present: base_q_idx > 0 && r.read_bit()?,
        ..DeltaParams::default()
    };
    if d.q_present {
        d.q_res = r.read_bits(2)? as u8;
        d.lf_present = !allow_intrabc && r.read_bit()?;
        if d.lf_present {
            d.lf_res = r.read_bits(2)? as u8;
            d.lf_multi = r.read_bit()?;
        }
    }
    Ok(d)
}
