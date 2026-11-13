//! The reference/GOP manager: which earlier frame each frame predicts from,
//! which reference slots it replaces, and its temporal layer.
//!
//! Structures (`docs/ARCHITECTURE.md` §6.1; SVT-AV1 real-time `--hierarchical-levels`,
//! `docs/reference/svt-av1-m12-rtc.md` §1 "Reference structure"):
//!
//! ```text
//! flat (1 layer)   T0 <- T0 <- T0 <- T0 ...           every frame refreshes slot 0
//! L1T2 (2 layers)  T0 <- T1   T0 <- T1 ...            period 2
//!                  pos 0: T0, refs slot 0 (previous T0), refreshes slot 0
//!                  pos 1: T1, refs slot 0, non-reference
//! L1T3 (3 layers)  period 4
//!                  pos 0: T0, refs slot 0 (previous T0), refreshes slot 0
//!                  pos 1: T2, refs slot 0, non-reference
//!                  pos 2: T1, refs slot 0, refreshes slot 1
//!                  pos 3: T2, refs slot 1, non-reference
//! ```
//!
//! An adaptive structure (`TemporalStructure::adaptive`) chooses flat or
//! the full layered pattern per mini-GOP from the camera and scene motion
//! the analysis measures, switching at T0 frames only.
//!
//! Every frame references exactly one earlier frame, so coding order is
//! display order (no reorder delay). T0 frames reference only T0 frames,
//! and top-layer frames are never referenced, so a decoder can drop the top
//! layer (or keep T0 alone) and still decode what is left. A key frame
//! refreshes every slot and restarts the pattern at position 0.

use nc_analysis::link::{FrameLink, MAX_REFERENCE_SLOTS, ReferenceLink, SlotMask};

use crate::layer::LayerPosition;

/// Most temporal layers supported.
pub const MAX_TEMPORAL_LAYERS: u8 = 3;

/// Everything the structure decides for one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameReferences {
    /// Temporal layer and mini-GOP position.
    pub position: LayerPosition,
    /// The slot inter prediction reads, and how many frames back it is.
    /// Key frames carry the reference their position would use, for
    /// source analysis only.
    pub reference: ReferenceLink,
    /// Slots this frame's reconstruction replaces (empty: non-reference).
    pub refresh: SlotMask,
    /// A second slot inter blocks may predict from, holding an older frame
    /// than `reference` (`second_reference.rs`); `None` in a
    /// single-reference stream.
    pub secondary: Option<ReferenceLink>,
}

impl FrameReferences {
    /// An inter frame of a flat chain: references the previous frame in
    /// slot 0 and replaces it.
    pub const FLAT: FrameReferences = FrameReferences {
        position: LayerPosition::FLAT,
        reference: FrameLink::PREVIOUS.reference,
        refresh: FrameLink::PREVIOUS.refresh,
        secondary: None,
    };

    /// The analysis link of this frame.
    #[must_use]
    pub fn link(&self) -> FrameLink {
        FrameLink {
            reference: self.reference,
            refresh: self.refresh,
        }
    }
}

/// One position of a mini-GOP pattern.
#[derive(Debug, Clone, Copy)]
struct PatternEntry {
    layer: u8,
    reference_slot: usize,
    refresh: SlotMask,
}

const fn entry(layer: u8, reference_slot: usize, refresh: SlotMask) -> PatternEntry {
    PatternEntry {
        layer,
        reference_slot,
        refresh,
    }
}

/// The patterns, indexed by `layers - 1`.
const PATTERNS: [&[PatternEntry]; MAX_TEMPORAL_LAYERS as usize] = [
    &[entry(0, 0, SlotMask::only(0))],
    &[entry(0, 0, SlotMask::only(0)), entry(1, 0, SlotMask::NONE)],
    &[
        entry(0, 0, SlotMask::only(0)),
        entry(2, 0, SlotMask::NONE),
        entry(1, 0, SlotMask::only(1)),
        entry(2, 1, SlotMask::NONE),
    ],
];

/// Entry `position` of the `pattern_layers`-layer pattern in a stream of
/// `stream_layers` layers, its top layer labelled as the stream's top layer.
fn pattern_entry(pattern_layers: u8, stream_layers: u8, position: usize) -> PatternEntry {
    let mut entry = PATTERNS[usize::from(pattern_layers - 1)][position];
    if pattern_layers > 1 && entry.layer + 1 == pattern_layers {
        entry.layer = stream_layers - 1;
    }
    entry
}

/// Reference slots a structure of `temporal_layers` layers keeps (codecs
/// size their reference buffers from it).
#[must_use]
pub fn reference_slots(temporal_layers: u8) -> usize {
    TemporalStructure::new(temporal_layers).slot_count()
}

/// Farthest back, in frames, any frame of a structure of `temporal_layers`
/// layers references: its period, the distance from one T0 frame to the
/// next (an adaptive structure switches patterns at T0 frames only, and a
/// restart refreshes every slot).
#[must_use]
pub fn longest_reference_distance(temporal_layers: u8) -> u64 {
    PATTERNS[usize::from(temporal_layers - 1)].len() as u64
}

/// When an adaptive structure layers a mini-GOP (`LayerParams::adaptive*`).
///
/// Layering pays where the longer reference distance costs little: a still
/// camera. It loses on camera motion and on scenes whose difference is
/// mostly compensable motion, which grows with distance (measured: static
/// TRAIN clips gain 7-11% BD-rate with L1T3, moving ones lose 2-12%;
/// `docs/handoff/hierarchical-p.md`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveRule {
    /// Largest camera motion (luma samples per frame: global translation
    /// plus zoom at the picture edge) of a layered mini-GOP.
    pub max_camera_motion: f32,
    /// Largest compensable share of the frame difference
    /// (`MotionAnalysis::compensable_share`) of a layered mini-GOP.
    pub max_compensable: f32,
    /// Smallest still-block fraction (`nc_analysis::still`) of a layered
    /// mini-GOP; 0 disables the gate.
    pub min_still: f32,
    /// Largest share of repeated input frames (`FrameAnalysis::repeat`) of
    /// a layered mini-GOP; 1 disables the gate. A source that duplicates
    /// frames codes its repeats almost free in a flat chain, while a
    /// layer pattern out of step with the repeats predicts real changes
    /// over longer distances: on the layer-gate fitting clips the two
    /// sources repeating every other frame lost 18% and 46% `J_real` with
    /// L1T3, every other clip had none (`docs/handoff/hierarchical-p.md`
    /// "Layer gate").
    pub max_repeat_share: f32,
    /// Largest static-block fraction (`PreAnalysis::static_fraction`,
    /// distance-1 frames) of a layered mini-GOP; 1 disables the gate.
    /// Mostly unchanged scenes already code cheaply flat (static skip, the
    /// flat QP cadence's static scaling); on the layer-gate fitting clips
    /// layering gained on handheld, pan and busy scenes and lost on the
    /// most static ones (fixed cameras over quiet scenes, conference).
    pub max_static: f32,
    /// Largest camera jitter of a layered mini-GOP: the change in camera
    /// motion (luma samples per frame) between distance-1 frames two frames
    /// apart, so that flat and layered streams (whose distance-1 frames
    /// are two apart) measure it alike; at or above
    /// [`CAMERA_JITTER_OFF`] the gate is off. A shaking camera (bodycam)
    /// defeats the longer-distance prediction of the base layer: on the
    /// layer-gate fitting clips layered bodycam footage gained on average
    /// but its worst 1% of windows lost 11%; up to a jitter of 2 the gain
    /// stayed (-6.8%) and the worst windows held (+0.6%).
    pub max_camera_jitter: f32,
    /// Largest motion-compensated residual (`MotionSignal::residual`, SAD
    /// per luma sample, running average) of a layered mini-GOP whose
    /// compensable share is below [`AdaptiveRule::noise_compensable`]; at
    /// or above [`NOISE_RESIDUAL_OFF`] the gate is off. Such a scene is
    /// still but its frame difference is large temporal noise that no
    /// reference predicts, so the better-funded base layer passes nothing
    /// on to the frames above it, which only lose: `static_otw_homes390`
    /// (residual 2.3-2.5, compensable 0.08) lost 1-10% `J_real` with L1T3
    /// in every gated preset, while the quieter `fit_static_otw399` (1.3-1.5,
    /// same compensable share) gained 5-15% (`docs/handoff/hierarchical-p.md`
    /// "Layer gate re-fit").
    pub max_noise_residual: f32,
    /// Compensable share below which the residual gate applies: moving
    /// content (compensable share 0.35 and above on the fitting clips)
    /// has a large residual too, and gains from layering.
    pub noise_compensable: f32,
    /// A flat stream switches to layering only below this fraction of both
    /// motion limits, and above the still minimum divided by it (hysteresis): the averages of borderline scenes hover around
    /// a single threshold and would otherwise flip structure every few
    /// mini-GOPs (`crowd_mot17_09`, compensable share 0.51–0.80).
    pub enter_ratio: f32,
}

/// One frame's motion, as the adaptive rule reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSignal {
    /// Camera motion in luma samples per frame.
    pub camera_motion: f32,
    /// `MotionAnalysis::compensable_share` of a frame analysed against its
    /// immediate predecessor (`None` over longer distances, where it is not
    /// comparable).
    pub compensable: Option<f32>,
    /// Fraction of the frame's blocks that are still.
    pub still_fraction: f32,
    /// Whether the input repeats the previous input frame, when detected.
    pub repeat: Option<bool>,
    /// Static-block fraction of a frame analysed against its immediate
    /// predecessor (`None` over longer distances, where fewer blocks pass
    /// as static).
    pub static_fraction: Option<f32>,
    /// Mean best-match SAD per luma sample of a frame analysed against its
    /// immediate predecessor (`MotionAnalysis::residual_per_sample`; `None`
    /// over longer distances, where it is not comparable).
    pub residual: Option<f32>,
}

/// A camera-jitter limit at or above this switches the jitter gate off.
pub const CAMERA_JITTER_OFF: f32 = 64.0;

/// A noise-residual limit at or above this switches the residual gate off.
pub const NOISE_RESIDUAL_OFF: f32 = 64.0;

/// Running averages of the motion signal.
#[derive(Debug, Clone, Copy, Default)]
struct MotionTrend {
    camera_motion: f32,
    compensable: f32,
    still_fraction: f32,
    repeat_share: f32,
    static_fraction: f32,
    residual: f32,
    /// `None` until two distance-1 frames two apart have been seen (the
    /// jitter gate keeps the stream flat until then).
    camera_jitter: Option<f32>,
    /// Camera motion of the last two distance-1 frames, newest first.
    adjacent_camera: [Option<f32>; 2],
    seen: bool,
}

/// Weight of the newest frame in the motion trend. A slow average (about a
/// 10-frame time constant) separates scene classes better than the
/// per-frame values: the global-motion fit of noisy still scenes jumps to
/// 3 px/frame on single frames (`static_meva_hospital`).
const TREND_WEIGHT: f32 = 0.1;

/// The reference/GOP manager for one stream.
#[derive(Debug, Clone)]
pub struct TemporalStructure {
    /// Layers of the stream (its signalling and layer ids).
    layers: u8,
    /// Layers of the current mini-GOP's pattern (`layers` unless adaptive).
    pattern_layers: u8,
    /// Position of the next frame in the current mini-GOP.
    position: usize,
    /// Per-mini-GOP structure choice; `None` keeps `layers` throughout.
    adaptive: Option<AdaptiveRule>,
    /// An adaptive structure held flat from its next mini-GOP on (a feature
    /// that needs every frame to reference the previous one, intra refresh).
    hold_flat: bool,
    trend: MotionTrend,
    /// Index of the frame each slot holds (valid once a key frame has been
    /// coded, which refreshes all of them).
    slot_frame: [u64; MAX_REFERENCE_SLOTS],
}

impl TemporalStructure {
    /// A structure with `layers` temporal layers (1..=[`MAX_TEMPORAL_LAYERS`]).
    ///
    /// # Panics
    ///
    /// If `layers` is out of range (the configuration is validated first).
    #[must_use]
    pub fn new(layers: u8) -> Self {
        assert!((1..=MAX_TEMPORAL_LAYERS).contains(&layers));
        Self {
            layers,
            pattern_layers: layers,
            position: 0,
            adaptive: None,
            hold_flat: false,
            trend: MotionTrend::default(),
            slot_frame: [0; MAX_REFERENCE_SLOTS],
        }
    }

    /// A stream signalled with `layers` layers whose mini-GOPs are flat or
    /// fully layered as `rule` decides from the motion trend. Switching
    /// happens only at T0 frames, which always reference the previous T0,
    /// so every codec's signalling stays the `layers`-layer one.
    #[must_use]
    pub fn adaptive(layers: u8, rule: AdaptiveRule) -> Self {
        Self {
            adaptive: Some(rule),
            pattern_layers: 1,
            ..Self::new(layers)
        }
    }

    /// Reference slots this structure uses (the encoder keeps one
    /// reconstruction per slot).
    #[must_use]
    pub fn slot_count(&self) -> usize {
        PATTERNS[usize::from(self.layers - 1)]
            .iter()
            .flat_map(|e| (0..MAX_REFERENCE_SLOTS).filter(move |&s| e.refresh.contains(s)))
            .max()
            .map_or(1, |s| s + 1)
    }

    /// Whether mini-GOPs are chosen adaptively ([`TemporalStructure::adaptive`]).
    #[must_use]
    pub fn is_adaptive(&self) -> bool {
        self.adaptive.is_some()
    }

    /// Hold an adaptive structure flat from its next mini-GOP on, or release
    /// it; the mini-GOP in progress finishes as chosen. No effect on a fixed
    /// structure.
    pub fn set_hold_flat(&mut self, hold: bool) {
        self.hold_flat = hold;
    }

    /// Whether the adaptive rule reads `MotionSignal::repeat` (the analysis
    /// detects repeats only then).
    #[must_use]
    pub fn reads_repeats(&self) -> bool {
        self.adaptive
            .is_some_and(|rule| rule.max_repeat_share < 1.0)
    }

    /// Feed the motion of the frame just analysed (adaptive structures).
    pub fn observe(&mut self, signal: MotionSignal) {
        let trend = &mut self.trend;
        let blend = |old: f32, new: f32| {
            if trend.seen {
                old + TREND_WEIGHT * (new - old)
            } else {
                new
            }
        };
        trend.camera_motion = blend(trend.camera_motion, signal.camera_motion);
        trend.still_fraction = blend(trend.still_fraction, signal.still_fraction);
        if let Some(share) = signal.compensable {
            trend.compensable = blend(trend.compensable, share);
            // Only distance-1 frames carry a compensable share.
            if let Some(older) = trend.adjacent_camera[1] {
                let jitter = (signal.camera_motion - older).abs();
                trend.camera_jitter = Some(
                    trend
                        .camera_jitter
                        .map_or(jitter, |old| old + TREND_WEIGHT * (jitter - old)),
                );
            }
            trend.adjacent_camera = [Some(signal.camera_motion), trend.adjacent_camera[0]];
        }
        if let Some(fraction) = signal.static_fraction {
            trend.static_fraction = blend(trend.static_fraction, fraction);
        }
        if let Some(residual) = signal.residual {
            trend.residual = blend(trend.residual, residual);
        }
        if let Some(repeat) = signal.repeat {
            trend.repeat_share = blend(trend.repeat_share, if repeat { 1.0 } else { 0.0 });
        }
        trend.seen = true;
    }

    /// The pattern of the next mini-GOP.
    fn choose_pattern(&self) -> u8 {
        match self.adaptive {
            None => self.layers,
            Some(_) if self.hold_flat => 1,
            Some(rule) => {
                // Staying layered uses the limits; switching to it needs
                // margin (`enter_ratio`).
                let scale = if self.pattern_layers > 1 {
                    1.0
                } else {
                    rule.enter_ratio
                };
                // A limit of 1 on a fraction is off (no margin either).
                let below = |value: f32, limit: f32| limit >= 1.0 || value <= limit * scale;
                let noisy_still = rule.max_noise_residual < NOISE_RESIDUAL_OFF
                    && self.trend.compensable < rule.noise_compensable
                    && self.trend.residual > rule.max_noise_residual * scale;
                let layered = self.trend.seen
                    && self.trend.camera_motion <= rule.max_camera_motion * scale
                    && self.trend.compensable <= rule.max_compensable * scale
                    && self.trend.still_fraction * scale >= rule.min_still
                    && below(self.trend.repeat_share, rule.max_repeat_share)
                    && below(self.trend.static_fraction, rule.max_static)
                    && (rule.max_camera_jitter >= CAMERA_JITTER_OFF
                        || self
                            .trend
                            .camera_jitter
                            .is_some_and(|jitter| jitter <= rule.max_camera_jitter * scale))
                    && !noisy_still;
                if layered { self.layers } else { 1 }
            }
        }
    }

    /// Entry `position` of the current pattern, with its top layer labelled
    /// as the stream's top layer (an L1T2 mini-GOP of a three-layer stream
    /// codes its leaf as T2, keeping every top-layer frame droppable).
    fn entry(&self, position: usize) -> PatternEntry {
        pattern_entry(self.pattern_layers, self.layers, position)
    }

    fn period(&self) -> usize {
        PATTERNS[usize::from(self.pattern_layers - 1)].len()
    }

    /// Whether the next inter frame may be skipped without disturbing the
    /// structure: any frame of a flat mini-GOP (the chain simply continues
    /// from the last coded frame), or a top-layer frame of a layered one,
    /// which no frame references (skipping it is partial sub-stream
    /// extraction).
    #[must_use]
    pub fn next_is_droppable(&self) -> bool {
        self.droppable_at(self.position)
    }

    /// Frames after the next one that cannot be dropped before the next
    /// droppable one: 0 in a flat chain, the lower-layer frames between
    /// two top-layer frames otherwise.
    #[must_use]
    pub fn protected_after_next(&self) -> u32 {
        (1..self.period())
            .take_while(|&ahead| !self.droppable_at((self.position + ahead) % self.period()))
            .count() as u32
    }

    fn droppable_at(&self, position: usize) -> bool {
        let entry = PATTERNS[usize::from(self.pattern_layers - 1)][position % self.period()];
        self.pattern_layers == 1 || entry.layer + 1 == self.pattern_layers
    }

    /// Skip the next frame without coding it. A layered mini-GOP keeps the
    /// skipped top-layer frame's place and returns its layer; a flat one
    /// returns `None`, and its next frame takes the skipped frame's index.
    pub fn drop_next(&mut self) -> Option<u8> {
        if self.pattern_layers == 1 {
            return None;
        }
        let layer = self.entry(self.position).layer;
        self.advance();
        Some(layer)
    }

    fn advance(&mut self) {
        self.position += 1;
        if self.position == self.period() {
            self.position = 0;
        }
    }

    /// The references frame `index` (coding order) would have when it
    /// `restart`s the pattern (a key frame, or a long-term-reference
    /// recovery frame, `ltr.rs`) or continues it, without committing to
    /// either: the encoder may analyse a frame as inter and then code it as
    /// a key (scene cut). A restart refreshes every slot.
    #[must_use]
    pub fn peek(&self, index: u64, restart: bool) -> FrameReferences {
        let position = if restart { 0 } else { self.position };
        let pattern_layers = if position == 0 {
            self.choose_pattern()
        } else {
            self.pattern_layers
        };
        let entry = pattern_entry(pattern_layers, self.layers, position);
        let reference = ReferenceLink {
            slot: entry.reference_slot,
            distance: u32::try_from(index - self.slot_frame[entry.reference_slot])
                .unwrap_or(u32::MAX)
                .max(1),
        };
        FrameReferences {
            position: LayerPosition {
                layer: entry.layer,
                // A flat mini-GOP of an adaptive stream is coded as a flat
                // stream (QP cadence, rate model, decision base cadence);
                // only its signalling keeps the stream's layers. Reporting
                // the stream's layers made every such frame a T0 "base":
                // HEVC Fast on crowd, flat throughout, lost 4-8% J_real
                // against a flat stream (static-detect, 2026-11-09).
                layers: if pattern_layers == 1 { 1 } else { self.layers },
                position: position as u8,
                period: PATTERNS[usize::from(pattern_layers - 1)].len() as u8,
            },
            reference,
            refresh: if restart {
                SlotMask::ALL
            } else {
                entry.refresh
            },
            secondary: None,
        }
    }

    /// The references of frame `index` (coding order), which restarts the
    /// pattern if `restart` ([`TemporalStructure::peek`]). Must be called
    /// once per coded frame, in order.
    pub fn next(&mut self, index: u64, restart: bool) -> FrameReferences {
        let references = self.peek(index, restart);
        if restart {
            self.position = 0;
        }
        if self.position == 0 {
            self.pattern_layers = self.choose_pattern();
        }
        for (slot, frame) in self.slot_frame.iter_mut().enumerate() {
            if references.refresh.contains(slot) {
                *frame = index;
            }
        }
        self.advance();
        references
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_top_layer_frames_are_droppable() {
        for (layers, droppable, protected) in [
            (1u8, [true; 4], [0u32; 4]),
            // A top-layer frame is followed by one lower-layer frame.
            (2, [false, true, false, true], [0, 1, 0, 1]),
            (3, [false, true, false, true], [0, 1, 0, 1]),
        ] {
            let mut s = TemporalStructure::new(layers);
            for frame in 0..4u64 {
                assert_eq!(
                    s.next_is_droppable(),
                    droppable[frame as usize],
                    "{layers} {frame}"
                );
                assert_eq!(s.protected_after_next(), protected[frame as usize]);
                s.next(frame, frame == 0);
            }
        }
    }

    fn run(layers: u8, keys: &[u64], frames: u64) -> Vec<FrameReferences> {
        let mut s = TemporalStructure::new(layers);
        (0..frames).map(|i| s.next(i, keys.contains(&i))).collect()
    }

    #[test]
    fn flat_references_the_previous_frame() {
        let refs = run(1, &[0], 5);
        assert!(refs[1..].iter().all(|r| r.link() == FrameLink::PREVIOUS));
        assert_eq!(TemporalStructure::new(1).slot_count(), 1);
    }

    #[test]
    fn three_layers_follow_the_dyadic_pattern() {
        let refs = run(3, &[0], 9);
        let layers: Vec<u8> = refs.iter().map(|r| r.position.layer).collect();
        assert_eq!(layers, [0, 2, 1, 2, 0, 2, 1, 2, 0]);
        let distances: Vec<u32> = refs[1..].iter().map(|r| r.reference.distance).collect();
        assert_eq!(distances, [1, 2, 1, 4, 1, 2, 1, 4]);
        assert_eq!(refs[3].reference.slot, 1);
        assert!(refs[1].refresh.is_empty() && refs[3].refresh.is_empty());
        assert!(refs[3].position.is_top_layer() && !refs[2].position.is_top_layer());
        assert_eq!(TemporalStructure::new(3).slot_count(), 2);
    }

    #[test]
    fn no_reference_reaches_past_the_longest_distance() {
        for layers in 1..=MAX_TEMPORAL_LAYERS {
            let longest = run(layers, &[0, 7], 20)[1..]
                .iter()
                .map(|r| u64::from(r.reference.distance))
                .max();
            assert_eq!(longest, Some(longest_reference_distance(layers)));
        }
    }

    #[test]
    fn base_layer_references_only_base_layer() {
        for layers in 1..=MAX_TEMPORAL_LAYERS {
            let refs = run(layers, &[0, 7], 20);
            let mut slot_layer = [0u8; MAX_REFERENCE_SLOTS];
            for r in &refs {
                if r.position.layer == 0 && r.refresh != SlotMask::ALL {
                    assert_eq!(slot_layer[r.reference.slot], 0);
                }
                assert!(
                    slot_layer[r.reference.slot] <= r.position.layer || r.refresh == SlotMask::ALL
                );
                for (slot, layer) in slot_layer.iter_mut().enumerate() {
                    if r.refresh.contains(slot) {
                        *layer = r.position.layer;
                    }
                }
            }
        }
    }

    #[test]
    fn patterns_match_the_shared_layer_cycle() {
        for layers in 1..=MAX_TEMPORAL_LAYERS {
            let refs = run(layers, &[0], 8);
            let cycle = refs[0].position.cycle();
            for r in &refs {
                assert_eq!(r.position, cycle[usize::from(r.position.position)]);
            }
        }
    }

    #[test]
    fn adaptive_structure_switches_only_at_t0() {
        let rule = AdaptiveRule {
            max_camera_motion: 1.0,
            max_compensable: 0.5,
            min_still: 0.0,
            max_repeat_share: 1.0,
            max_static: 1.0,
            max_camera_jitter: CAMERA_JITTER_OFF,
            max_noise_residual: NOISE_RESIDUAL_OFF,
            noise_compensable: 0.0,
            enter_ratio: 0.9,
        };
        let still = MotionSignal {
            camera_motion: 0.0,
            compensable: Some(0.1),
            still_fraction: 0.0,
            repeat: None,
            static_fraction: None,
            residual: None,
        };
        let panning = MotionSignal {
            camera_motion: 8.0,
            compensable: Some(0.9),
            still_fraction: 0.0,
            repeat: None,
            static_fraction: None,
            residual: None,
        };
        let mut s = TemporalStructure::adaptive(3, rule);
        let mut layers = Vec::new();
        for i in 0..24u64 {
            let r = s.next(i, i == 0);
            layers.push(r.position.layer);
            // A frame never references a later-refreshed slot's frame.
            assert!(r.reference.distance >= 1);
            s.observe(if i < 10 { still } else { panning });
        }
        // The key is flat (no trend yet); layered from the next T0 while
        // still, flat again once the pan has moved the trend; changes
        // happen at T0 frames only.
        assert_eq!(layers[..9], [0, 0, 2, 1, 2, 0, 2, 1, 2]);
        assert!(layers[20..].iter().all(|&l| l == 0));
    }

    #[test]
    fn noisy_still_scenes_stay_flat() {
        let rule = AdaptiveRule {
            max_camera_motion: 16.0,
            max_compensable: 1.0,
            min_still: 0.0,
            max_repeat_share: 1.0,
            max_static: 1.0,
            max_camera_jitter: CAMERA_JITTER_OFF,
            max_noise_residual: 1.8,
            noise_compensable: 0.2,
            enter_ratio: 0.9,
        };
        let layered_after = |compensable: f32, residual: f32| {
            let mut s = TemporalStructure::adaptive(3, rule);
            let mut layers = Vec::new();
            for i in 0..12u64 {
                layers.push(s.next(i, i == 0).position.layer);
                s.observe(MotionSignal {
                    camera_motion: 0.0,
                    compensable: Some(compensable),
                    still_fraction: 0.0,
                    repeat: None,
                    static_fraction: Some(0.3),
                    residual: Some(residual),
                });
            }
            layers[8..].iter().any(|&l| l > 0)
        };
        // Large uncompensable residual: temporal noise, kept flat.
        assert!(!layered_after(0.05, 2.5));
        // A quieter still scene, and moving content with the same residual.
        assert!(layered_after(0.05, 1.3));
        assert!(layered_after(0.6, 2.5));
    }

    #[test]
    fn held_adaptive_structure_turns_flat_after_its_mini_gop() {
        let rule = AdaptiveRule {
            max_camera_motion: 1.0,
            max_compensable: 0.5,
            min_still: 0.0,
            max_repeat_share: 1.0,
            max_static: 1.0,
            max_camera_jitter: CAMERA_JITTER_OFF,
            max_noise_residual: NOISE_RESIDUAL_OFF,
            noise_compensable: 0.0,
            enter_ratio: 0.9,
        };
        let still = MotionSignal {
            camera_motion: 0.0,
            compensable: Some(0.1),
            still_fraction: 0.0,
            repeat: None,
            static_fraction: None,
            residual: None,
        };
        let mut s = TemporalStructure::adaptive(3, rule);
        let mut flat = Vec::new();
        for i in 0..16u64 {
            // Held from the middle of the second mini-GOP (frames 5-8).
            s.set_hold_flat(i >= 6);
            let r = s.next(i, i == 0);
            flat.push(r.position.layers == 1);
            s.observe(still);
        }
        // The layered mini-GOP in progress finishes; flat from the next T0.
        assert_eq!(
            flat[..9],
            [true, false, false, false, false, false, false, false, false]
        );
        assert!(flat[9..].iter().all(|&f| f));
    }

    #[test]
    fn adaptive_hysteresis_holds_borderline_scenes() {
        let rule = AdaptiveRule {
            max_camera_motion: 2.0,
            max_compensable: 0.68,
            min_still: 0.0,
            max_repeat_share: 1.0,
            max_static: 1.0,
            max_camera_jitter: CAMERA_JITTER_OFF,
            max_noise_residual: NOISE_RESIDUAL_OFF,
            noise_compensable: 0.0,
            enter_ratio: 0.9,
        };
        let signal = |compensable| MotionSignal {
            camera_motion: 0.0,
            compensable: Some(compensable),
            still_fraction: 0.0,
            repeat: None,
            static_fraction: None,
            residual: None,
        };
        // Between the enter (0.612) and stay (0.68) limits: a flat stream
        // stays flat, a layered one stays layered.
        let mut flat = TemporalStructure::adaptive(3, rule);
        flat.observe(signal(0.65));
        assert_eq!(flat.choose_pattern(), 1);
        let mut layered = TemporalStructure::adaptive(3, rule);
        layered.observe(signal(0.3));
        layered.next(0, true);
        layered.next(1, false);
        assert_eq!(layered.pattern_layers, 3);
        for _ in 0..40 {
            layered.observe(signal(0.65));
        }
        assert_eq!(layered.choose_pattern(), 3);
    }

    #[test]
    fn adaptive_layers_only_still_scenes_when_gated() {
        let rule = AdaptiveRule {
            max_camera_motion: 2.0,
            max_compensable: 0.68,
            min_still: 0.5,
            max_repeat_share: 1.0,
            max_static: 1.0,
            max_camera_jitter: CAMERA_JITTER_OFF,
            max_noise_residual: NOISE_RESIDUAL_OFF,
            noise_compensable: 0.0,
            enter_ratio: 0.9,
        };
        // A fixed camera over a crowd: little compensable motion on
        // average, but few still blocks.
        let crowd = MotionSignal {
            camera_motion: 0.0,
            compensable: Some(0.3),
            still_fraction: 0.3,
            repeat: None,
            static_fraction: None,
            residual: None,
        };
        let mut s = TemporalStructure::adaptive(2, rule);
        s.observe(crowd);
        assert_eq!(s.choose_pattern(), 1);
        s.observe(MotionSignal {
            still_fraction: 0.9,
            ..crowd
        });
        // The running average (0.36) is still below the minimum.
        assert_eq!(s.choose_pattern(), 1);
        for _ in 0..40 {
            s.observe(MotionSignal {
                still_fraction: 0.9,
                ..crowd
            });
        }
        assert_eq!(s.choose_pattern(), 2);
    }

    #[test]
    fn adaptive_keeps_repeating_static_and_shaking_sources_flat() {
        let rule = AdaptiveRule {
            max_camera_motion: f32::INFINITY,
            max_compensable: 1.0,
            min_still: 0.0,
            max_repeat_share: 0.25,
            max_static: 0.7,
            max_camera_jitter: 2.0,
            max_noise_residual: NOISE_RESIDUAL_OFF,
            noise_compensable: 0.0,
            enter_ratio: 0.9,
        };
        // A handheld camera: much compensable motion, few still blocks.
        let handheld = MotionSignal {
            camera_motion: 10.0,
            compensable: Some(0.85),
            still_fraction: 0.05,
            repeat: Some(false),
            static_fraction: Some(0.2),
            residual: None,
        };
        let mut s = TemporalStructure::adaptive(3, rule);
        s.observe(handheld);
        // No jitter measured yet: flat until two distance-1 frames two
        // apart have been seen.
        assert_eq!(s.choose_pattern(), 1);
        s.observe(handheld);
        s.observe(handheld);
        assert_eq!(s.choose_pattern(), 3);
        // Every other input repeated: the trend settles near one half.
        for frame in 0..40 {
            s.observe(MotionSignal {
                repeat: Some(frame % 2 == 0),
                ..handheld
            });
        }
        assert_eq!(s.choose_pattern(), 1);
        // A mostly static scene stays flat too.
        let mut s = TemporalStructure::adaptive(3, rule);
        for _ in 0..3 {
            s.observe(MotionSignal {
                static_fraction: Some(0.9),
                ..handheld
            });
        }
        assert_eq!(s.choose_pattern(), 1);
        // A shaking camera stays flat: camera motion alternating 4 and 10
        // samples per frame between distance-1 frames two apart.
        let mut s = TemporalStructure::adaptive(3, rule);
        for frame in 0..40 {
            s.observe(MotionSignal {
                camera_motion: if frame % 4 < 2 { 4.0 } else { 10.0 },
                ..handheld
            });
        }
        assert_eq!(s.choose_pattern(), 1);
        // Undetected (None) leaves the trend as it is.
        s.observe(MotionSignal {
            repeat: None,
            static_fraction: None,
            ..handheld
        });
        assert_eq!(s.choose_pattern(), 1);
    }

    #[test]
    fn key_frames_restart_the_pattern() {
        let refs = run(2, &[0, 3], 6);
        let positions: Vec<u8> = refs.iter().map(|r| r.position.position).collect();
        assert_eq!(positions, [0, 1, 0, 0, 1, 0]);
        assert_eq!(refs[3].refresh, SlotMask::ALL);
        assert_eq!(refs[5].reference.distance, 2);
    }
}
