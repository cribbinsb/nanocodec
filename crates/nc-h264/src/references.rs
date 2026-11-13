//! H.264 signalling of the engine's temporal structure (`docs/ARCHITECTURE.md`
//! §6.1): `nal_ref_idc`, `frame_num` and the one reference list
//! modification a layered structure needs.
//!
//! - **Reference marking** is the sliding window (§8.2.5.3) with
//!   `max_num_ref_frames` equal to the engine's slot count. The last
//!   `max_num_ref_frames` reference pictures are then exactly the pictures
//!   the slots hold, so no memory management operations are needed.
//! - **Top-layer frames** are non-reference (`nal_ref_idc` 0, §7.4.1) and
//!   carry no `dec_ref_pic_marking()`. T1 frames of three layers use
//!   `nal_ref_idc` 1 and T0 frames 2 (IDR pictures and parameter sets 3), so
//!   an extractor keeps T0 by dropping NAL units with `nal_ref_idc` < 2 and
//!   T0+T1 by dropping those with 0.
//! - **`frame_num`** is the previous reference picture's plus one (§7.4.3).
//!   Dropping T1 pictures leaves one `frame_num` gap per mini-GOP, so
//!   three-layer streams set `gaps_in_frame_num_value_allowed_flag`; the
//!   inferred "non-existing" frame (§8.2.5.2) only fills the DPB slot the
//!   dropped T1 frame held, and the old T0 survives until the next T0 is
//!   marked, exactly as in the full stream.
//! - **The reference list** (§8.2.4.2.1) starts with the most recent
//!   reference picture. When the slot to predict from holds an older one (a
//!   T0 frame referencing the previous T0 over a T1), a
//!   `ref_pic_list_modification()` moves it to index 0 by picture-number
//!   difference (§8.2.4.3.1). The difference is the same in the full and in
//!   the extracted streams, so one syntax serves both.
//! - **Long-term reference** (`ltr.interval`, the engine's `ltr.rs`): the
//!   slot after the structure's holds one long-term picture,
//!   `max_num_ref_frames` is the structure's slots plus two (the long-term
//!   picture, and room for `MaxLongTermFrameIdx` 2, which must stay below
//!   `max_num_ref_frames`) and `frame_num` gaps are allowed, so a stream with lost frames stays conforming (the
//!   receiver infers "non-existing" frames for the gap, §8.2.5.2, and its
//!   DPB matches the encoder's). The IDR picture sets
//!   `long_term_reference_flag` (§8.2.5.1: it becomes long-term frame 0
//!   and `MaxLongTermFrameIdx` 0). A picture marked for the long-term slot
//!   (a T0 picture) carries `adaptive_ref_pic_marking_mode_flag` 1 with
//!   MMCO 1 for each short-term picture no slot holds any more, MMCO 2
//!   releasing the old long-term picture, MMCO 4 allowing two indices and
//!   MMCO 6 making the current picture long-term (§8.2.5.4). Marks
//!   alternate between indices 1 and 2, so a receiver that lost a mark
//!   holds no picture under the new index (the mark before released it): a
//!   picture predicting from the new mark finds no reference and is
//!   refused, where with one index it would predict from the old long-term
//!   picture unnoticed (`docs/handoff/loss-resilience.md`). Index 0 is the
//!   IDR picture's alone and is released at the first mark, which comes
//!   long before `frame_num` wraps (`MAX_REFERENCE_AGE`): a receiver that
//!   sees a `frame_num` gap through 0 lost an IDR picture, drops index 0
//!   (`decoder/dpb.rs`) and refuses the pictures predicting from it;
//!   every other reference picture uses the sliding window, which never
//!   removes the long-term picture and, with one more frame than the
//!   structure's slots, only removes pictures no slot holds. The tracker
//!   models that DPB: the initial list (§8.2.4.2.1) puts short-term
//!   pictures first, so a picture predicting from the long-term one moves
//!   it to index 0 with `modification_of_pic_nums_idc` 2
//!   (`long_term_pic_num` 0, §8.2.4.3.2), and one predicting from an
//!   older short-term picture subtracts as above.
//! - **Second reference** (`refs.second`, the engine's
//!   `second_reference.rs`): the last high-quality picture is the long-term
//!   one, marked as above (every high-quality picture refreshes the
//!   long-term slot, and nothing waits for an acknowledgement). A picture
//!   that may predict from it sets `num_ref_idx_active_override_flag` with
//!   two active references, list index 0 the primary reference and index 1
//!   the long-term picture. A flat chain's initial list is already that
//!   (its one short-term picture, then the long-term one); otherwise both
//!   are placed with explicit commands, since commands fill the list from
//!   index 0 (§8.2.4.3).

use nc_common::decode::{DecodeError, annex_b_units};
use nc_encoder::{FramePlan, FrameType, MAX_REFERENCE_SLOTS, SlotMask};

use crate::params::MAX_FRAME_NUM;

/// `nal_ref_idc` of IDR pictures and parameter sets.
pub const NAL_REF_IDC_IDR: u8 = 3;
/// `nal_ref_idc` of T0 P pictures (and every P picture of a flat chain).
pub const NAL_REF_IDC_BASE: u8 = 2;
/// `nal_ref_idc` of the T1 reference pictures of three-layer structures.
pub const NAL_REF_IDC_MIDDLE: u8 = 1;
/// `nal_ref_idc` of non-reference (top-layer) pictures.
pub const NAL_REF_IDC_NONE: u8 = 0;

/// `LongTermFrameIdx` (and so `LongTermPicNum`, §8.2.4.1) of the IDR
/// picture's long-term marking.
pub const IDR_LONG_TERM_FRAME_IDX: u32 = 0;
/// The two `LongTermFrameIdx` values marks alternate between.
const MARK_LONG_TERM_FRAME_IDX: [u32; 2] = [1, 2];
/// `MaxLongTermFrameIdx + 1` once a picture was marked (MMCO 4).
pub const LONG_TERM_FRAME_INDICES: u32 = 3;
/// DPB frames beside the structure's with a long-term reference: the
/// long-term picture and one more, as `max_long_term_frame_idx_plus1`
/// ([`LONG_TERM_FRAME_INDICES`]) may not exceed `max_num_ref_frames`.
pub const LONG_TERM_EXTRA_FRAMES: usize = 2;

/// The one `ref_pic_list_modification()` command (§7.3.3.1) that moves the
/// reference to index 0 when it is not there initially.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListModification {
    /// `modification_of_pic_nums_idc` 0 with this `abs_diff_pic_num_minus1`:
    /// an older short-term picture.
    Subtract(u32),
    /// `modification_of_pic_nums_idc` 2 with this `long_term_pic_num`: the
    /// long-term picture.
    LongTerm(u32),
}

/// How a reference picture is marked (`dec_ref_pic_marking()`, §7.3.3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Marking {
    /// IDR: `long_term_reference_flag` 0. P: the sliding window.
    ShortTerm,
    /// IDR: `long_term_reference_flag` 1 (`LongTermFrameIdx` 0). P:
    /// MMCO 1 releasing each short-term picture no slot holds any more,
    /// MMCO 2 releasing the previous long-term picture, MMCO 4 and MMCO 6
    /// making the current picture long-term frame `idx`.
    LongTerm {
        /// `difference_of_pic_nums_minus1` of each short-term picture to
        /// release.
        release: Vec<u32>,
        /// `long_term_pic_num` of the long-term picture this one replaces
        /// (`None` for the IDR picture).
        replaced: Option<u32>,
        /// `LongTermFrameIdx` of the current picture.
        idx: u32,
    },
}

/// How one picture's slice header and NAL header refer to the DPB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictureReferences {
    /// `nal_ref_idc` of the slice NAL unit.
    pub nal_ref_idc: u8,
    /// `frame_num` (0 for IDR pictures).
    pub frame_num: u32,
    /// The list modification, when the reference is not at index 0 of the
    /// initial list (or the second reference needs explicit placement).
    pub list_modification: Option<ListModification>,
    /// `num_ref_idx_l0_active_minus1 + 1`: 2 with the second reference.
    pub active_references: u8,
    /// The command placing the long-term picture at index 1 after
    /// `list_modification`, when the initial list does not hold it there.
    pub second_modification: Option<ListModification>,
    /// Marking of a reference picture (ignored for non-reference ones).
    pub marking: Marking,
}

impl PictureReferences {
    /// Whether the picture is a reference picture (it then carries
    /// `dec_ref_pic_marking()`).
    #[must_use]
    pub fn is_reference(&self) -> bool {
        self.nal_ref_idc != NAL_REF_IDC_NONE
    }
}

/// Tracks `frame_num` and which `frame_num` each slot holds.
#[derive(Debug, Clone)]
pub struct ReferenceTracker {
    /// `frame_num` of the most recent reference picture (`PrevRefFrameNum`).
    prev_ref_frame_num: u32,
    /// `frame_num` of the picture each engine slot holds.
    slot_frame_num: [u32; MAX_REFERENCE_SLOTS],
    /// The slot holding the long-term picture (`ltr.interval`, see the
    /// module notes); without it every picture is short-term.
    long_term: Option<LongTermDpb>,
}

/// The DPB of a stream with a long-term reference, as the decoder holds it.
#[derive(Debug, Clone)]
struct LongTermDpb {
    /// The engine slot of the long-term reference.
    slot: usize,
    /// `max_num_ref_frames`: the structure's slots plus
    /// [`LONG_TERM_EXTRA_FRAMES`].
    max_frames: usize,
    /// Whether each slot holds the long-term picture.
    slot_long_term: [bool; MAX_REFERENCE_SLOTS],
    /// `frame_num` of every short-term picture in the DPB, oldest first.
    short_terms: Vec<u32>,
    /// `LongTermFrameIdx` of the long-term picture.
    idx: u32,
}

impl Default for ReferenceTracker {
    fn default() -> Self {
        Self::new(None, 1)
    }
}

impl ReferenceTracker {
    /// A tracker before the first IDR picture for a structure of
    /// `structure_slots` slots; `long_term_slot` keeps that slot as a
    /// long-term picture.
    #[must_use]
    pub fn new(long_term_slot: Option<usize>, structure_slots: usize) -> Self {
        Self {
            prev_ref_frame_num: 0,
            slot_frame_num: [0; MAX_REFERENCE_SLOTS],
            long_term: long_term_slot.map(|slot| LongTermDpb {
                slot,
                max_frames: structure_slots + LONG_TERM_EXTRA_FRAMES,
                slot_long_term: [false; MAX_REFERENCE_SLOTS],
                short_terms: Vec::with_capacity(structure_slots + 1),
                idx: IDR_LONG_TERM_FRAME_IDX,
            }),
        }
    }

    /// The references of the picture `plan` describes; advances the state.
    pub fn next(&mut self, plan: &FramePlan) -> PictureReferences {
        let refs = &plan.references;
        if plan.frame_type == FrameType::Key {
            self.prev_ref_frame_num = 0;
            self.slot_frame_num = [0; MAX_REFERENCE_SLOTS];
            let marking = match self.long_term.as_mut() {
                Some(dpb) => {
                    dpb.slot_long_term = [true; MAX_REFERENCE_SLOTS];
                    dpb.short_terms.clear();
                    dpb.idx = IDR_LONG_TERM_FRAME_IDX;
                    Marking::LongTerm {
                        release: Vec::new(),
                        replaced: None,
                        idx: IDR_LONG_TERM_FRAME_IDX,
                    }
                }
                None => Marking::ShortTerm,
            };
            return PictureReferences {
                nal_ref_idc: NAL_REF_IDC_IDR,
                frame_num: 0,
                list_modification: None,
                active_references: 1,
                second_modification: None,
                marking,
            };
        }
        let frame_num = (self.prev_ref_frame_num + 1) % MAX_FRAME_NUM;
        let nal_ref_idc = if refs.refresh.is_empty() {
            NAL_REF_IDC_NONE
        } else if refs.position.layer == 0 {
            NAL_REF_IDC_BASE
        } else {
            NAL_REF_IDC_MIDDLE
        };
        let target = self.slot_frame_num[refs.reference.slot];
        let mut second_modification = None;
        let (list_modification, marking) = if let Some(dpb) = self.long_term.as_mut() {
            let mut modification = dpb.list_modification(refs.reference.slot, target, frame_num);
            if let Some(secondary) = refs.secondary {
                debug_assert!(dpb.slot_long_term[secondary.slot] && dpb.slot == secondary.slot);
                debug_assert!(!dpb.slot_long_term[refs.reference.slot]);
                let initial_fits = modification.is_none() && dpb.short_terms == [target];
                if !initial_fits {
                    modification = Some(modification.unwrap_or(ListModification::Subtract(
                        pic_num_difference(frame_num, target) - 1,
                    )));
                    second_modification = Some(ListModification::LongTerm(dpb.idx));
                }
            }
            let marking = if nal_ref_idc == NAL_REF_IDC_NONE {
                Marking::ShortTerm
            } else {
                dpb.mark(refs.refresh, &self.slot_frame_num, frame_num)
            };
            (modification, marking)
        } else {
            let modification = (target != self.prev_ref_frame_num)
                .then(|| ListModification::Subtract(pic_num_difference(frame_num, target) - 1));
            (modification, Marking::ShortTerm)
        };
        if nal_ref_idc != NAL_REF_IDC_NONE {
            self.prev_ref_frame_num = frame_num;
            for (slot, held) in self.slot_frame_num.iter_mut().enumerate() {
                if refs.refresh.contains(slot) {
                    *held = frame_num;
                }
            }
        }
        debug_assert!(refs.secondary.is_none() || self.long_term.is_some());
        PictureReferences {
            nal_ref_idc,
            frame_num,
            list_modification,
            active_references: if refs.secondary.is_some() { 2 } else { 1 },
            second_modification,
            marking,
        }
    }
}

impl LongTermDpb {
    /// The modification moving the picture in `slot` (`frame_num`
    /// `target` if short-term) to index 0 of the initial list: short-term
    /// pictures by descending `PicNum`, then the long-term one
    /// (§8.2.4.2.1).
    fn list_modification(
        &self,
        slot: usize,
        target: u32,
        frame_num: u32,
    ) -> Option<ListModification> {
        if self.slot_long_term[slot] {
            return (!self.short_terms.is_empty()).then_some(ListModification::LongTerm(self.idx));
        }
        let newest = self.short_terms.last().copied();
        (newest != Some(target))
            .then(|| ListModification::Subtract(pic_num_difference(frame_num, target) - 1))
    }

    /// Mark the reference picture with `frame_num` refreshing `refresh`
    /// (slots hold `slot_frame_num` before it): long-term when it refreshes
    /// the long-term slot, releasing the short-term pictures no slot keeps
    /// (MMCO 1, 6), otherwise the sliding window (§8.2.5.3), which then
    /// removes only pictures no slot keeps.
    fn mark(&mut self, refresh: SlotMask, slot_frame_num: &[u32], frame_num: u32) -> Marking {
        let kept = |held: u32, slot_long_term: &[bool]| {
            (0..MAX_REFERENCE_SLOTS).any(|slot| {
                !refresh.contains(slot) && !slot_long_term[slot] && slot_frame_num[slot] == held
            })
        };
        if refresh.contains(self.slot) {
            let (keep, release): (Vec<u32>, Vec<u32>) = self
                .short_terms
                .iter()
                .partition(|&&held| kept(held, &self.slot_long_term));
            self.short_terms = keep;
            for slot in 0..MAX_REFERENCE_SLOTS {
                if refresh.contains(slot) {
                    self.slot_long_term[slot] = true;
                }
            }
            let replaced = self.idx;
            self.idx = if replaced == MARK_LONG_TERM_FRAME_IDX[0] {
                MARK_LONG_TERM_FRAME_IDX[1]
            } else {
                MARK_LONG_TERM_FRAME_IDX[0]
            };
            return Marking::LongTerm {
                release: release
                    .into_iter()
                    .map(|held| pic_num_difference(frame_num, held) - 1)
                    .collect(),
                replaced: Some(replaced),
                idx: self.idx,
            };
        }
        // One long-term picture is always present.
        if self.short_terms.len() + 1 >= self.max_frames {
            let oldest = self.short_terms.remove(0);
            debug_assert!(
                !kept(oldest, &self.slot_long_term),
                "the sliding window would drop a picture a slot keeps"
            );
        }
        self.short_terms.push(frame_num);
        for slot in 0..MAX_REFERENCE_SLOTS {
            if refresh.contains(slot) {
                self.slot_long_term[slot] = false;
            }
        }
        Marking::ShortTerm
    }
}

/// `CurrPicNum − PicNum` of a short-term picture with `frame_num` `held`
/// seen from the picture with `frame_num` `current` (eq. 8-28, §8.2.4.1).
fn pic_num_difference(current: u32, held: u32) -> u32 {
    (current + MAX_FRAME_NUM - held) % MAX_FRAME_NUM
}

/// `nal_unit_type` of coded slices (Table 7-1).
const NAL_TYPE_SLICES: [u8; 2] = [1, 5];

/// The temporal layer of an emitted access unit in a structure of
/// `temporal_layers` layers, from its slice's `nal_ref_idc` (see the module
/// notes): what an extractor compares against the highest layer it keeps.
///
/// # Errors
///
/// Malformed Annex B framing, or an access unit without a slice.
pub fn access_unit_layer(access_unit: &[u8], temporal_layers: u8) -> Result<u8, DecodeError> {
    let slice = annex_b_units(access_unit)?
        .into_iter()
        .filter_map(|unit| unit.first().copied())
        .find(|header| NAL_TYPE_SLICES.contains(&(header & 0x1f)))
        .ok_or(DecodeError::Malformed("access unit without a slice"))?;
    let top = temporal_layers.saturating_sub(1);
    Ok(match slice >> 5 {
        idc if idc == NAL_REF_IDC_NONE => top,
        idc if idc == NAL_REF_IDC_MIDDLE => 1.min(top),
        _ => 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_encoder::{FrameReferences, LayerPosition, ReferenceLink};

    /// The long-term slot beside a flat chain.
    const LONG_TERM_SLOT: usize = 1;

    fn plan(index: u64, key: bool, layer: u8, slot: usize, refresh: SlotMask) -> FramePlan {
        FramePlan {
            index,
            frame_type: if key {
                FrameType::Key
            } else {
                FrameType::Inter
            },
            quantizer: 30,
            references: FrameReferences {
                position: LayerPosition {
                    layer,
                    layers: 3,
                    position: 0,
                    period: 4,
                },
                reference: ReferenceLink { slot, distance: 1 },
                refresh,
                secondary: None,
            },
            lambda_scale: 1.0,
            block_qp: false,
        }
    }

    #[test]
    fn three_layers_number_frames_and_modify_only_t0() {
        let mut t = ReferenceTracker::new(None, 1);
        let gop = [
            plan(0, true, 0, 0, SlotMask::ALL),
            plan(1, false, 2, 0, SlotMask::NONE),
            plan(2, false, 1, 0, SlotMask::only(1)),
            plan(3, false, 2, 1, SlotMask::NONE),
            plan(4, false, 0, 0, SlotMask::only(0)),
            plan(5, false, 2, 0, SlotMask::NONE),
        ];
        let refs: Vec<PictureReferences> = gop.iter().map(|p| t.next(p)).collect();
        let frame_nums: Vec<u32> = refs.iter().map(|r| r.frame_num).collect();
        assert_eq!(frame_nums, [0, 1, 1, 2, 2, 3]);
        let idc: Vec<u8> = refs.iter().map(|r| r.nal_ref_idc).collect();
        assert_eq!(idc, [3, 0, 1, 0, 2, 0]);
        let modified: Vec<_> = refs.iter().map(|r| r.list_modification).collect();
        let subtract_one = Some(ListModification::Subtract(1));
        assert_eq!(modified, [None, None, None, None, subtract_one, None]);
    }

    #[test]
    fn flat_chain_counts_every_frame() {
        let mut t = ReferenceTracker::new(None, 1);
        let mut flat = |index, key| {
            t.next(&FramePlan {
                index,
                frame_type: if key {
                    FrameType::Key
                } else {
                    FrameType::Inter
                },
                quantizer: 30,
                references: FrameReferences::FLAT,
                lambda_scale: 1.0,
                block_qp: false,
            })
        };
        assert_eq!(flat(0, true).frame_num, 0);
        for i in 1..20u32 {
            let r = flat(u64::from(i), false);
            assert_eq!(r.frame_num, i % MAX_FRAME_NUM);
            assert_eq!(r.nal_ref_idc, NAL_REF_IDC_BASE);
            assert_eq!(r.list_modification, None);
        }
    }

    #[test]
    fn long_term_marks_recovers_and_releases_the_short_term_picture() {
        let mut t = ReferenceTracker::new(Some(LONG_TERM_SLOT), 1);
        let mut frame = |key: bool, slot: usize, refresh: SlotMask| {
            let mut p = plan(0, key, 0, slot, refresh);
            p.references.position.layers = 1;
            t.next(&p)
        };
        let key = frame(true, 0, SlotMask::ALL);
        let idr_marking = Marking::LongTerm {
            release: vec![],
            replaced: None,
            idx: 0,
        };
        assert_eq!(key.marking, idr_marking);
        // The first P picture predicts from the long-term IDR, alone in the DPB.
        let p1 = frame(false, 0, SlotMask::only(0));
        assert_eq!(
            (p1.list_modification, p1.marking),
            (None, Marking::ShortTerm)
        );
        let p2 = frame(false, 0, SlotMask::only(0));
        assert_eq!(
            (p2.list_modification, p2.marking),
            (None, Marking::ShortTerm)
        );
        // Mark frame_num 3: release the IDR picture's long-term index 0
        // and go long-term under index 1.
        let mark = frame(false, 0, SlotMask::ALL);
        assert!(matches!(
            mark.marking,
            Marking::LongTerm {
                replaced: Some(0),
                idx: 1,
                ..
            }
        ));
        let p4 = frame(false, 0, SlotMask::only(0));
        assert_eq!(p4.list_modification, None);
        // Recovery: predict from slot 1 while a short-term picture exists.
        let recovery = frame(false, LONG_TERM_SLOT, SlotMask::only(0));
        let long_term = Some(ListModification::LongTerm(1));
        assert_eq!(recovery.list_modification, long_term);
        assert_eq!(recovery.marking, Marking::ShortTerm);
        assert_eq!(recovery.frame_num, 5);
        // Marks alternate between indices 1 and 2.
        let mark = frame(false, 0, SlotMask::ALL);
        assert!(matches!(
            mark.marking,
            Marking::LongTerm {
                replaced: Some(1),
                idx: 2,
                ..
            }
        ));
    }
}
