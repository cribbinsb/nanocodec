//! The chroma QP offsets of the engine chroma policy
//! (`nc_encoder::chroma_policy`) in H.264 syntax.
//!
//! H.264 carries `chroma_qp_index_offset` only in the PPS (§7.4.2.2), and
//! its Table 8-15 map is fixed, so the policy's QP-dependent target needs a
//! different offset at different picture QPs. Each key frame therefore
//! sends one PPS per offset the policy can use, and every slice names the
//! PPS of its picture's QP (`pic_parameter_set_id`, §7.4.3). A lost frame
//! never leaves a later one with the wrong PPS: they all arrive with the key.

use nc_encoder::chroma_policy::{ChromaPolicy, closest_index};

use crate::tables::chroma_qp;

/// Luma QPs the policy is evaluated at (8-bit H.264, §7.4.2.2).
const QP_RANGE: std::ops::RangeInclusive<u8> = 0..=51;
/// `chroma_qp_index_offset` range (§7.4.2.2).
const OFFSET_LIMIT: i32 = 12;
/// QP steps per doubling of the quantiser step.
const QP_PER_OCTAVE: f64 = 6.0;

/// The distinct offsets the policy uses over every luma QP, one PPS each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChromaOffsets {
    /// Offset of each luma QP.
    by_qp: Vec<i8>,
    /// The distinct offsets, ascending: PPS `k` carries `offsets[k]`.
    offsets: Vec<i8>,
}

impl ChromaOffsets {
    pub(crate) fn new(policy: ChromaPolicy) -> Self {
        let by_qp: Vec<i8> = QP_RANGE.map(|qp| policy_offset(policy, qp)).collect();
        let mut offsets = by_qp.clone();
        offsets.sort_unstable();
        offsets.dedup();
        Self { by_qp, offsets }
    }

    /// Every PPS's offset, in PPS order.
    pub(crate) fn offsets(&self) -> &[i8] {
        &self.offsets
    }

    /// The PPS index and offset of a picture at luma QP `qp`.
    pub(crate) fn at(&self, qp: u8) -> (usize, i8) {
        let offset = self.by_qp[usize::from(qp.min(*QP_RANGE.end()))];
        let index = self
            .offsets
            .binary_search(&offset)
            .expect("every QP's offset has a PPS");
        (index, offset)
    }
}

/// `pic_parameter_set_id` of PPS `index` of the parameter-set generation
/// `parameter_set_id` (0 or 1, alternating across resolution changes):
/// the generations interleave so neither overwrites the other's PPSs.
pub(crate) fn pps_id(parameter_set_id: u32, index: usize) -> u32 {
    parameter_set_id + 2 * index as u32
}

/// The offset whose Table 8-15 chroma QP at luma `qp` is nearest the
/// policy's chroma QP (ties to the smallest offset).
fn policy_offset(policy: ChromaPolicy, qp: u8) -> i8 {
    let step = |offset: i32| 2f64.powf(f64::from(chroma_qp(qp, offset as i8)) / QP_PER_OCTAVE);
    let target = 2f64.powf(policy.chroma_q(f64::from(qp)) / QP_PER_OCTAVE);
    closest_index(-OFFSET_LIMIT..=OFFSET_LIMIT, step, target) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_reach_the_policy_where_table_8_15_allows() {
        let offsets = ChromaOffsets::new(ChromaPolicy { offset: 0.0 });
        // Chroma follows luma to QP 36, then gets finer by half a QP per QP.
        for (qp, chroma) in [(20, 20), (36, 36), (38, 37), (42, 39)] {
            let (_, offset) = offsets.at(qp);
            assert_eq!(chroma_qp(qp, offset), chroma, "QP {qp}");
        }
        // Table 8-15 tops out at 39: high QPs take its coarsest chroma.
        let (_, offset) = offsets.at(51);
        assert_eq!(chroma_qp(51, offset), 39);
        assert!(offsets.offsets().len() <= 13);
        let (index, offset) = offsets.at(40);
        assert_eq!(offsets.offsets()[index], offset);
    }

    #[test]
    fn generations_never_share_a_pps_id() {
        assert_eq!((pps_id(0, 0), pps_id(1, 0), pps_id(0, 3)), (0, 1, 6));
    }
}
