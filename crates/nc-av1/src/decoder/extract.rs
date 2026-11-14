//! Per-receiver AV1 stream extraction for spatial SVC and K-SVC: what a
//! selective forwarding relay sends one receiver. Given an operating point,
//! it drops the OBUs outside `operating_point_idc` (AV1 §7.18) and, within
//! each temporal unit, keeps only the frames the unit's highest kept frame
//! reaches through reference slots (`ref_frame_idx` against earlier frames'
//! `refresh_frame_flags`, §5.9.2, §7.20). In full SVC that is `{S0..=Sk}` on
//! every unit; in K-SVC (`L2T*_KEY`, `L3T*_KEY`) it is `{S0..=Sk}` on key
//! units and `{Sk}` alone otherwise, which is the stream `nanocodec extract
//! --operating-point` writes and the conformance check decodes.

use super::frame_header::{self, FrameHeader, NUM_REF_FRAMES, SlotHeader};
use super::sequence::{self, Sequence, idc_highest_layers};
use super::{MAX_OPERATING_POINT, obu, slot_header};
use nc_common::decode::{BitReader, DecodeError};

/// `refresh_frame_flags` of a shown key frame: every slot (§7.20).
const ALL_SLOTS: u8 = 0xFF;

/// Find the sequence header's lowest-spatial-layer full-temporal-rate operating
/// point index (`AV1 §5.5.1`, `§6.4.1`) from the first temporal unit.
///
/// # Errors
///
/// Malformed OBU framing or missing sequence header.
pub fn lowest_spatial_operating_point(temporal_unit: &[u8]) -> Result<u8, DecodeError> {
    for unit in obu::split(temporal_unit)? {
        if unit.kind == obu::OBU_SEQUENCE_HEADER {
            let seq = sequence::parse(unit.payload)?;
            if seq.operating_point_count <= 1 {
                return Ok(0);
            }
            let mut best: Option<(u8, u8, u8)> = None;
            for (idx, op) in seq.operating_points[..seq.operating_point_count]
                .iter()
                .enumerate()
            {
                let idc = op.idc;
                if idc == 0 {
                    return Ok(idx as u8);
                }
                let (max_s, max_t) = idc_highest_layers(idc);
                let replace = match best {
                    None => true,
                    Some((_, best_s, best_t)) => {
                        max_s < best_s || (max_s == best_s && max_t > best_t)
                    }
                };
                if replace {
                    best = Some((idx as u8, max_s, max_t));
                }
            }
            return Ok(best.map_or(0, |(idx, _, _)| idx));
        }
    }
    Err(DecodeError::MissingKey)
}

fn frame_needed_slots(header: &FrameHeader) -> u8 {
    if let Some(slot) = header.show_existing {
        1u8 << slot
    } else if header.frame_type.is_intra() {
        0
    } else {
        header
            .ref_frame_idx
            .iter()
            .fold(0u8, |mask, &slot| mask | (1u8 << slot))
    }
}

fn frame_refreshed_slots(header: &FrameHeader) -> u8 {
    if header.show_existing.is_some() {
        if header.frame_type == frame_header::FrameType::Key {
            ALL_SLOTS
        } else {
            0
        }
    } else if header.frame_type == frame_header::FrameType::Key && header.show_frame {
        ALL_SLOTS
    } else {
        header.refresh_frame_flags
    }
}

fn apply_slot_refresh(
    slots: &mut [Option<(u64, SlotHeader)>; NUM_REF_FRAMES],
    header: &FrameHeader,
    refreshed: u8,
    uid: u64,
) -> Option<u64> {
    let saved = if let Some(existing) = header.show_existing {
        slots[existing].clone()
    } else {
        Some((uid, slot_header(header)))
    };
    let saved_uid = saved.as_ref().map(|(u, _)| *u);
    if refreshed != 0 {
        for (slot, dst) in slots.iter_mut().enumerate() {
            if refreshed & (1u8 << slot) != 0 {
                dst.clone_from(&saved);
            }
        }
    }
    saved_uid
}

/// Extracted OBU bytes for one temporal unit plus its output `(upscaled_width, height)`.
pub type ExtractedTemporalUnit = (Vec<u8>, (usize, usize));

/// Stateful per-receiver AV1 stream extractor (`AV1 §5.9.2`, `§7.18`, `§7.20`).
///
/// For a target `operating_point`, drops OBUs outside `operating_point_idc[op]`
/// and, within each included temporal unit, walks DPB reference-slot reachability
/// backward from the unit's highest spatial frame (`ref_frame_idx` vs lower-layer
/// `refresh_frame_flags`). In full SVC this retains `{S0..=Sk}` on every unit;
/// in K-SVC (`L2T*_KEY`, `L3T*_KEY`) it retains `{S0..=Sk}` on key units and
/// `{Sk}` alone on non-key units and standalone `INTRA_ONLY_FRAME` repairs
/// (skipping non-key units whose `Sk` was dropped or lost).
#[derive(Debug, Clone)]
pub struct Av1StreamExtractor {
    operating_point: u8,
    sequence: Option<Sequence>,
    max_spatial_id: u8,
    next_uid: u64,
    stream_slots: [Option<(u64, SlotHeader)>; NUM_REF_FRAMES],
    kept_uids: [Option<u64>; NUM_REF_FRAMES],
    stripped_lower: bool,
}

struct ExtractedCandidate {
    obu_indices: Vec<usize>,
    spatial_id: u8,
    header: FrameHeader,
    refreshed: u8,
    saved_uid: Option<u64>,
}

impl Av1StreamExtractor {
    /// Create an extractor for `operating_point` (`0..=31`, `AV1 §6.4.1`).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Unsupported`] if `operating_point > MAX_OPERATING_POINT`.
    pub fn new(operating_point: u8) -> Result<Self, DecodeError> {
        if operating_point > MAX_OPERATING_POINT {
            return Err(DecodeError::Unsupported("AV1 operating point > 31"));
        }
        Ok(Self {
            operating_point,
            sequence: None,
            max_spatial_id: 0,
            next_uid: 1,
            stream_slots: std::array::from_fn(|_| None),
            kept_uids: [None; NUM_REF_FRAMES],
            stripped_lower: false,
        })
    }

    fn update_sequence(&mut self, obus: &[obu::Obu<'_>]) -> Result<Sequence, DecodeError> {
        for unit in obus {
            if unit.kind == obu::OBU_SEQUENCE_HEADER {
                let mut seq = sequence::parse(unit.payload)?;
                let op = usize::from(self.operating_point);
                if op >= seq.operating_point_count {
                    return Err(DecodeError::Unsupported("AV1 operating point out of range"));
                }
                seq.operating_point_idc = seq.operating_points[op].idc;
                if self.sequence != Some(seq) {
                    self.max_spatial_id = idc_highest_layers(seq.operating_point_idc).0;
                    self.stream_slots = std::array::from_fn(|_| None);
                    self.kept_uids = [None; NUM_REF_FRAMES];
                    self.stripped_lower = false;
                    self.sequence = Some(seq);
                }
            }
        }
        self.sequence.ok_or(DecodeError::MissingKey)
    }

    fn scan_candidates(
        &mut self,
        obus: &[obu::Obu<'_>],
        seq: &Sequence,
    ) -> Result<Vec<ExtractedCandidate>, DecodeError> {
        let mut scratch_slots = self.stream_slots.clone();
        let mut candidates: Vec<ExtractedCandidate> = Vec::new();
        let mut open_candidate: Option<usize> = None;

        for (idx, unit) in obus.iter().enumerate() {
            if unit.has_extension
                && !matches!(
                    unit.kind,
                    obu::OBU_SEQUENCE_HEADER | obu::OBU_TEMPORAL_DELIMITER
                )
                && !seq.in_operating_point(unit.layer.temporal, unit.layer.spatial)
            {
                open_candidate = None;
                continue;
            }
            match unit.kind {
                obu::OBU_FRAME | obu::OBU_FRAME_HEADER => {
                    let refs: [Option<&SlotHeader>; NUM_REF_FRAMES] =
                        scratch_slots.each_ref().map(|s| s.as_ref().map(|(_, h)| h));
                    let mut r = BitReader::new(unit.payload);
                    let header = frame_header::parse(&mut r, seq, &refs, unit.layer)?;
                    let refreshed = frame_refreshed_slots(&header);
                    let uid = self.next_uid;
                    self.next_uid += 1;
                    let saved_uid = apply_slot_refresh(&mut scratch_slots, &header, refreshed, uid);
                    let cid = candidates.len();
                    candidates.push(ExtractedCandidate {
                        obu_indices: vec![idx],
                        spatial_id: unit.layer.spatial,
                        header,
                        refreshed,
                        saved_uid,
                    });
                    open_candidate = (unit.kind == obu::OBU_FRAME_HEADER).then_some(cid);
                }
                obu::OBU_TILE_GROUP | obu::OBU_REDUNDANT_FRAME_HEADER => {
                    if let Some(cid) = open_candidate {
                        candidates[cid].obu_indices.push(idx);
                    }
                }
                _ => {
                    open_candidate = None;
                }
            }
        }
        self.stream_slots = scratch_slots;
        Ok(candidates)
    }

    /// Filter `temporal_unit` down to the exact OBUs needed by a receiver of
    /// `operating_point`, returning `Ok(None)` when the entire temporal unit is
    /// skipped (e.g. a higher temporal layer outside `operating_point`, or a
    /// non-key K-SVC unit whose target spatial layer is absent), or
    /// `Ok(Some((unit_bytes, (upscaled_width, height))))` for the output frame.
    ///
    /// # Errors
    ///
    /// Malformed OBU or frame-header syntax, missing sequence header, or
    /// reference to an unpopulated DPB slot.
    pub fn extract_temporal_unit(
        &mut self,
        temporal_unit: &[u8],
    ) -> Result<Option<ExtractedTemporalUnit>, DecodeError> {
        let obus = obu::split(temporal_unit)?;
        let seq = self.update_sequence(&obus)?;
        let stream_uids_before_tu: [Option<u64>; NUM_REF_FRAMES] = self
            .stream_slots
            .each_ref()
            .map(|s| s.as_ref().map(|(uid, _)| *uid));
        let candidates = self.scan_candidates(&obus, &seq)?;
        if candidates.is_empty() {
            return Ok(None);
        }

        let top = candidates.len() - 1;
        if candidates[0].spatial_id == 0
            && candidates[0].header.frame_type == frame_header::FrameType::Key
            && candidates[0].header.show_frame
        {
            self.stripped_lower = false;
            if seq.operating_point_idc == 0 {
                self.max_spatial_id = self.max_spatial_id.max(candidates[top].spatial_id);
            }
        }

        if self.stripped_lower && candidates[top].spatial_id < self.max_spatial_id {
            return Ok(None);
        }

        let (keep_candidate, needed) = reachable_from_top(&candidates);
        let top_size = (
            candidates[top].header.size.upscaled_width,
            candidates[top].header.size.height,
        );

        let missing_ref = (0..NUM_REF_FRAMES).any(|slot| {
            (needed & (1u8 << slot)) != 0 && self.kept_uids[slot] != stream_uids_before_tu[slot]
        });
        if missing_ref {
            return Ok(None);
        }

        if keep_candidate.iter().any(|&kept| !kept) {
            self.stripped_lower = true;
        }

        let mut keep_obu = vec![false; obus.len()];
        for (idx, unit) in obus.iter().enumerate() {
            if !unit.has_extension
                || matches!(
                    unit.kind,
                    obu::OBU_SEQUENCE_HEADER | obu::OBU_TEMPORAL_DELIMITER
                )
            {
                keep_obu[idx] = true;
            }
        }
        for (cand, &kept) in candidates.iter().zip(&keep_candidate) {
            if kept {
                for (slot, dst) in self.kept_uids.iter_mut().enumerate() {
                    if cand.refreshed & (1u8 << slot) != 0 {
                        *dst = cand.saved_uid;
                    }
                }
                for &obu_idx in &cand.obu_indices {
                    keep_obu[obu_idx] = true;
                }
            }
        }

        let mut out = Vec::with_capacity(temporal_unit.len());
        for (unit, kept) in obus.iter().zip(keep_obu) {
            if kept {
                out.extend_from_slice(unit.raw);
            }
        }
        Ok(Some((out, top_size)))
    }
}

/// Which of a unit's frames its last (highest) frame reaches through
/// reference slots refreshed earlier in the unit, and the slots still needed
/// from before the unit.
fn reachable_from_top(candidates: &[ExtractedCandidate]) -> (Vec<bool>, u8) {
    let top = candidates.len() - 1;
    let mut keep = vec![false; candidates.len()];
    keep[top] = true;
    let mut needed = frame_needed_slots(&candidates[top].header);
    for (i, cand) in candidates[..top].iter().enumerate().rev() {
        if cand.refreshed & needed != 0 {
            keep[i] = true;
            needed = (needed & !cand.refreshed) | frame_needed_slots(&cand.header);
        }
    }
    (keep, needed)
}
