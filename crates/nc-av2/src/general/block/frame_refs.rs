//! What an inter frame's block syntax knows of its references, derived
//! once per frame (dav2d `dav2d_submit_frame`'s `refdist`, `absrefdist`,
//! `refdir`, `furthest_future_refidx`, `skip_mode_refs`,
//! `no_switch_opfl_refine_type`, `no_switch_refine_mv`, `svc`; dav2d
//! commit 237229e, BSD-2-Clause, see `docs/licenses/dav2d`).
use super::inter_record::{NO_REFERENCE, Reference, TIP_FRAME};
use crate::general::frame_header::{FrameHeader, FrameType};
use crate::general::references::MAX_REFERENCES;
use crate::general::sequence::SequenceHeader;
use nc_common::decode::{DecodeError, Result};

/// Display-order differences are clamped to this (dav2d `get_poc_diff`).
const MAX_DISTANCE: i64 = 127;
/// Skip mode pairs the first two references when their distances differ
/// by at most this.
const SKIP_MODE_PAIR_DISTANCE: u32 = 1;

/// Scale factors are in 1/16384 (dav2d `scale_fac`'s `<< 14`).
const SCALE_BITS: u32 = 14;
/// Steps are scale factors in 1/1024 (dav2d `(scale + 8) >> 4`).
const STEP_SHIFT: u32 = 4;
/// A reference may be at most twice the frame's size and at least a
/// sixteenth of it (dav2d `dav2d_submit_frame`).
const MAX_DOWNSCALE: u32 = 2;
const MAX_UPSCALE: u32 = 16;

/// How a reference of another size than the frame maps the frame's
/// positions to its own (dav2d `svc`): per direction (horizontal,
/// vertical) the scale factor in 1/16384 and the step between adjacent
/// samples in 1/1024.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceScale {
    /// Reference size over frame size, 1/16384 (dav2d `svc[][].scale`).
    pub scale: [i32; 2],
    /// The same in 1/1024 (dav2d `svc[][].step`).
    pub step: [i32; 2],
}

impl ReferenceScale {
    /// The scale of a reference of `reference` size for a frame of `frame`
    /// size, in luma samples; `None` when the sizes are equal.
    ///
    /// # Errors
    /// A reference more than twice the frame's size or less than a
    /// sixteenth of it.
    pub fn new(reference: (u32, u32), frame: (u32, u32)) -> Result<Option<Self>> {
        let fits = |reference: u32, frame: u32| {
            frame * MAX_DOWNSCALE >= reference && frame <= reference * MAX_UPSCALE
        };
        if !fits(reference.0, frame.0) || !fits(reference.1, frame.1) {
            return Err(DecodeError::Malformed("AV2 reference size"));
        }
        if reference == frame {
            return Ok(None);
        }
        let factor = |reference: u32, frame: u32| {
            ((u64::from(reference) << SCALE_BITS) + u64::from(frame >> 1)) / u64::from(frame)
        };
        let scale = [
            factor(reference.0, frame.0) as i32,
            factor(reference.1, frame.1) as i32,
        ];
        Ok(Some(Self {
            scale,
            step: scale.map(|scale| (scale + (1 << (STEP_SHIFT - 1))) >> STEP_SHIFT),
        }))
    }
}

/// The reference ranks of an inter frame as its block syntax sees them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameReferences {
    /// Distinct references (`num_total_refs`).
    pub count: usize,
    /// Each rank's display order minus the frame's, clamped to +-127
    /// (dav2d `refdist`): positive for a future reference.
    pub distance: [i8; MAX_REFERENCES],
    /// Its magnitude (dav2d `absrefdist`).
    pub abs_distance: [u8; MAX_REFERENCES],
    /// Whether each rank, and the TIP frame (index [`TIP_FRAME`]), lies in
    /// the future (dav2d `refdir`).
    pub future: [bool; MAX_REFERENCES + 1],
    /// The rank furthest in the future, if any (dav2d
    /// `furthest_future_refidx`).
    pub furthest_future: Option<usize>,
    /// The references of skip-mode blocks (dav2d `skip_mode_refs`).
    pub skip_mode: [Reference; 2],
    /// Optical-flow refinement as the frame allows it, 0 in switch
    /// frames (dav2d `no_switch_opfl_refine_type`).
    pub optical_flow: u8,
    /// MV refinement allowed, not in switch frames (dav2d
    /// `no_switch_refine_mv`).
    pub refine_mv: bool,
    /// Each rank's scale, for references of another size than the frame.
    pub scales: [Option<ReferenceScale>; MAX_REFERENCES],
}

impl FrameReferences {
    /// The values of `frame` (none for key and intra-only frames).
    #[must_use]
    pub fn new(sequence: &SequenceHeader, frame: &FrameHeader) -> Self {
        let mut references = Self::default();
        if frame.is_intra() {
            return references;
        }
        references.count = usize::from(frame.reference_count);
        let ordered = sequence.inter.order_hint_bits != 0;
        // Only the active references (dav2d d1bde236): the slots past
        // them are stale or copies of the first.
        for rank in 0..references.count {
            let distance = if ordered {
                (i64::from(frame.reference_orders[rank]) - i64::from(frame.order))
                    .clamp(-MAX_DISTANCE, MAX_DISTANCE) as i8
            } else {
                0
            };
            references.distance[rank] = distance;
            references.abs_distance[rank] = distance.unsigned_abs();
            references.future[rank] = distance > 0;
            let further = references
                .furthest_future
                .is_none_or(|furthest| references.distance[furthest] < distance);
            if distance > 0 && further {
                references.furthest_future = Some(rank);
            }
        }
        references.future[TIP_FRAME as usize] = true;
        let pair = frame.skip_mode
            && references.count > 1
            && u32::from(references.abs_distance[0].abs_diff(references.abs_distance[1]))
                <= SKIP_MODE_PAIR_DISTANCE;
        references.skip_mode = [0, i8::from(pair)];
        let switch = frame.frame_type == FrameType::Switch;
        references.optical_flow = if switch { 0 } else { frame.optical_flow };
        references.refine_mv = !switch && sequence.inter.refine_mv;
        references
    }

    /// Whether rank `rank` is a reference of another size (dav2d
    /// `svc[rank][0].scale` set).
    #[must_use]
    pub fn scaled(&self, rank: usize) -> bool {
        self.scales.get(rank).is_some_and(Option::is_some)
    }

    /// Whether `reference` lies in the future, as dav2d's `refdir` with
    /// intra (`-1`) reads it: intra counts as -1, so only `== 1` tests
    /// tell it apart.
    #[must_use]
    pub fn direction(&self, reference: Reference) -> i8 {
        if reference == NO_REFERENCE {
            -1
        } else {
            i8::from(self.future[reference as usize])
        }
    }
}
