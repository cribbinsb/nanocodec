//! The per-frame analysis pipeline: ingest, pyramid, pre-analysis, motion
//! search (zero-vector gates, quarter-resolution HME, full-pel window),
//! global motion, noise, stillness.
//!
//! [`Analyzer`] keeps everything carried between frames (the source frame
//! and pyramid of each reference slot, the previous motion fields, the
//! noise level), so callers just feed frames in order, naming each frame's
//! reference ([`FrameLink`]; the previous frame by default).

use nc_common::{Picture, PlaneIndex, Result};
use nc_dsp::Dsp;
use nc_dsp::resample::resample_plane;

use crate::effort::MotionEffort;
use crate::global::{self, GlobalMotion};
use crate::grid::Grid;
use crate::link::{self, DistanceRatio, FrameLink, MAX_REFERENCE_SLOTS, SlotMask};
use crate::motion::{
    self, AggregatedField, BLOCK, MotionBlock, MotionInputs, SB, SbStats, SearchConfig,
};
use crate::mv::Mv;
use crate::noise;
use crate::preanalysis::{self, PreAnalysis};
use crate::pyramid::{Downsample, Pyramid};
use crate::still::{StillConfig, StillMap, StillTracker};

/// Luma border of the internal frame copies: one SB, which an SB overhanging
/// the picture's right or bottom edge reads. Search candidates reaching
/// further out (up to `FINE_LIMIT`) are gathered (`motion::Level::gathered`).
const LUMA_BORDER: usize = SB;

/// Chroma border of the analysed pictures: only pre-analysis reads chroma,
/// in 8x8 chroma blocks covering the plane, whose last row and column reach
/// at most 7 samples past it. (Half the luma border, the [`Picture::new`]
/// default, made the border extension write ~0.5 MB more per 720p frame.)
const CHROMA_BORDER: usize = 8;

/// A stage of the pipeline, for optional timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Copying the input into padded internal buffers (unless it was
    /// written there directly) and filling their borders.
    Ingest,
    /// Building the half/quarter resolution levels.
    Pyramid,
    /// Per-block statistics and static detection.
    PreAnalysis,
    /// Zero-vector gates, quarter-resolution HME and full-pel search.
    Motion,
    /// Fitting the camera motion model.
    GlobalMotion,
}

impl Stage {
    /// Every stage, in pipeline order.
    pub const ALL: [Stage; 5] = [
        Stage::Ingest,
        Stage::Pyramid,
        Stage::PreAnalysis,
        Stage::Motion,
        Stage::GlobalMotion,
    ];
}

/// Receives stage boundaries. The library itself never reads a clock (wasm32
/// has none), so timing is the caller's business.
pub trait StageTimer {
    /// `stage` is about to run.
    fn begin(&mut self, stage: Stage);
    /// `stage` has finished.
    fn end(&mut self, stage: Stage);
}

/// A timer that records nothing.
#[derive(Debug, Default)]
pub struct NoTimer;

impl StageTimer for NoTimer {
    fn begin(&mut self, _: Stage) {}
    fn end(&mut self, _: Stage) {}
}

/// Motion results for a frame that has a predecessor.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionAnalysis {
    /// Per-superblock statistics (zz SAD, search path, `me_dist`, 8x8 cost
    /// variance).
    pub sb: Grid<SbStats>,
    /// Per-superblock HME vector (luma samples).
    pub hme: Grid<Mv>,
    /// 8x8 field (vectors in luma samples).
    pub fine: Grid<MotionBlock>,
    /// 16x16, 32x32 and 64x64 fields.
    pub aggregated: [AggregatedField; 3],
    /// Camera motion model.
    pub global: GlobalMotion,
    /// How much searching this frame took.
    pub effort: MotionEffort,
}

impl MotionAnalysis {
    /// The share of the zero-vector difference that motion compensation
    /// removes: `Σ (zz SAD − best 8x8 SAD) / Σ zz SAD` over superblocks.
    /// Near 0 for a still scene (even a noisy one: noise is not
    /// compensable), towards 1 for moving content. Hierarchical-P structure
    /// choice reads it: compensable motion grows with reference distance,
    /// noise does not.
    #[must_use]
    pub fn compensable_share(&self) -> f32 {
        let (mut zero, mut best) = (0u64, 0u64);
        for stats in self.sb.cells() {
            zero += u64::from(stats.zz_sad);
            best += u64::from(stats.me_dist[0].min(stats.zz_sad));
        }
        if zero == 0 {
            0.0
        } else {
            (zero - best) as f32 / zero as f32
        }
    }
}

impl MotionAnalysis {
    /// Mean best-match SAD per luma sample over the 8x8 field: the frame
    /// difference that motion compensation leaves (temporal noise in a
    /// still scene). Hierarchical-P structure choice reads it.
    #[must_use]
    pub fn residual_per_sample(&self) -> f32 {
        let cells = self.fine.cells();
        let total: u64 = cells.iter().map(|block| u64::from(block.sad)).sum();
        total as f32 / (cells.len().max(1) * BLOCK * BLOCK) as f32
    }

    /// Mean over superblocks of the summed best 8x8 SADs (SVT's
    /// `norm_me_dist`, `initial_rc_process.c`): how well motion search
    /// predicts the frame.
    #[must_use]
    pub fn mean_me_dist(&self) -> f32 {
        let cells = self.sb.cells();
        let total: u64 = cells.iter().map(|stats| u64::from(stats.me_dist[0])).sum();
        total as f32 / cells.len().max(1) as f32
    }
}

/// Everything known about one source frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameAnalysis {
    /// Per-block statistics.
    pub pre: PreAnalysis,
    /// Motion against the reference frame (`None` when there is none yet).
    pub motion: Option<MotionAnalysis>,
    /// Running estimate of the sensor-noise sigma after this frame, in luma
    /// sample units (see `noise`). Already-compressed sources read close to
    /// zero.
    pub noise_sigma: f32,
    /// Blocks that are reliably static (see `still`): static for several
    /// frames, with a static neighbourhood.
    pub still: StillMap,
    /// Whether this input repeats the previous input frame (whatever the
    /// reference), when detected ([`Analyzer::set_repeat_detection`]).
    pub repeat: Option<bool>,
}

/// One retained source frame: full-resolution picture and its pyramid.
#[derive(Debug)]
struct SourceFrame {
    picture: Picture,
    /// `picture`'s levels, once built: a frame nothing references whose
    /// search reuses the previous HME vectors (`SearchConfig::hme_from_seed`)
    /// skips them.
    pyramid: Pyramid,
}

impl SourceFrame {
    fn new(width: usize, height: usize) -> Result<Self> {
        Ok(Self {
            picture: Picture::with_borders(width, height, LUMA_BORDER, CHROMA_BORDER)?,
            pyramid: Pyramid::new(width, height)?,
        })
    }

    fn build_pyramid(&mut self, dsp: &Dsp) {
        self.pyramid.build(dsp, self.picture.plane(PlaneIndex::Y));
    }
}

/// Runs the analysis pipeline over a sequence of frames.
#[derive(Debug)]
pub struct Analyzer {
    dsp: Dsp,
    /// One frame per reference slot, plus the one being analysed.
    frames: Vec<SourceFrame>,
    /// The frame each reference slot holds (`None` until one refreshes it).
    slots: [Option<usize>; MAX_REFERENCE_SLOTS],
    /// The frame the next input is written into; no slot holds it.
    next: usize,
    /// The most recently analysed frame.
    current: usize,
    last: Option<FrameAnalysis>,
    /// Reference distance the last frame's motion was measured over.
    last_distance: u32,
    search: SearchConfig,
    still: StillTracker,
    still_config: StillConfig,
    /// Which rows pre-analysis' statistics read.
    stat_rows: preanalysis::StatRows,
    /// Detect [`FrameAnalysis::repeat`].
    detect_repeats: bool,
    /// Pre-analysis of the written next frame, computed ahead of its
    /// analysis ([`Analyzer::pre_analyze_next`]), which reuses it.
    pending: Option<PendingPre>,
}

/// A next frame's pre-analysis, against the reference in `reference_slot`.
#[derive(Debug)]
struct PendingPre {
    reference_slot: usize,
    pre: PreAnalysis,
}

impl Analyzer {
    /// An analyzer for `width` x `height` frames, each searched against the
    /// previous one ([`FrameLink::PREVIOUS`]).
    ///
    /// # Errors
    ///
    /// If the dimensions are invalid for a picture or too small for the
    /// pyramid.
    pub fn new(width: usize, height: usize, dsp: Dsp) -> Result<Self> {
        Self::with_reference_slots(width, height, dsp, 1)
    }

    /// An analyzer keeping the source of `slots` reference frames (1 to
    /// [`MAX_REFERENCE_SLOTS`]), for [`Analyzer::analyze_linked`].
    ///
    /// # Errors
    ///
    /// As [`Analyzer::new`].
    ///
    /// # Panics
    ///
    /// If `slots` is outside 1..=[`MAX_REFERENCE_SLOTS`].
    pub fn with_reference_slots(
        width: usize,
        height: usize,
        dsp: Dsp,
        slots: usize,
    ) -> Result<Self> {
        assert!(
            (1..=MAX_REFERENCE_SLOTS).contains(&slots),
            "unsupported reference slot count {slots}"
        );
        let frames = (0..=slots)
            .map(|_| SourceFrame::new(width, height))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dsp,
            frames,
            slots: [None; MAX_REFERENCE_SLOTS],
            next: 0,
            current: 0,
            last: None,
            last_distance: 1,
            search: SearchConfig::default(),
            still: StillTracker::new(
                width.div_ceil(preanalysis::BLOCK),
                height.div_ceil(preanalysis::BLOCK),
            ),
            still_config: StillConfig::default(),
            stat_rows: preanalysis::StatRows::All,
            detect_repeats: false,
            pending: None,
        })
    }

    /// Continue at `width` x `height`, for an encoder that changes its
    /// coded picture size without a key frame: the source of every
    /// reference slot (and the most recent frame) is resampled to the new
    /// size (`nc_dsp::resample`, once per switch) and its levels rebuilt, so
    /// the next frame searches against it as usual. The previous frame's
    /// vectors no longer seed the search and the stillness history
    /// restarts; the noise estimate carries over.
    ///
    /// # Errors
    ///
    /// As [`Analyzer::new`].
    pub fn resize(&mut self, width: usize, height: usize) -> Result<()> {
        let mut frames = (0..self.frames.len())
            .map(|_| SourceFrame::new(width, height))
            .collect::<Result<Vec<_>>>()?;
        let downsample = self.frames[0].pyramid.downsample();
        for (index, (old, new)) in self.frames.iter().zip(&mut frames).enumerate() {
            new.pyramid.set_downsample(downsample);
            if index != self.current && !self.slots.contains(&Some(index)) {
                continue;
            }
            for plane in PlaneIndex::ALL {
                resample_plane(
                    &old.picture.plane(plane).view(),
                    &mut new.picture.plane_mut(plane).view_mut(),
                );
            }
            new.picture.extend_borders();
            new.build_pyramid(&self.dsp);
        }
        self.frames = frames;
        self.pending = None;
        if let Some(last) = self.last.as_mut() {
            last.motion = None;
        }
        self.still = StillTracker::new(
            width.div_ceil(preanalysis::BLOCK),
            height.div_ceil(preanalysis::BLOCK),
        );
        Ok(())
    }

    /// Set how much the motion search does (default: SVT M12's search).
    /// Out-of-range window sizes are clamped.
    pub fn set_search_config(&mut self, config: SearchConfig) {
        self.search = config.clamped();
    }

    /// Which rows pre-analysis' statistics read and how the pyramid's half
    /// level is made (default: every row, filtered).
    pub fn set_sampling(&mut self, stat_rows: preanalysis::StatRows, downsample: Downsample) {
        self.stat_rows = stat_rows;
        for frame in &mut self.frames {
            frame.pyramid.set_downsample(downsample);
        }
    }

    /// Set how strict the stillness test is (default
    /// [`StillConfig::default`]).
    pub fn set_still_config(&mut self, config: StillConfig) {
        self.still_config = config;
    }

    /// Whether to detect repeated input frames ([`FrameAnalysis::repeat`];
    /// off by default: only the temporal-layer gate reads it).
    pub fn set_repeat_detection(&mut self, detect: bool) {
        self.detect_repeats = detect;
    }

    /// [`FrameAnalysis::repeat`] of a frame with pre-analysis `pre`, the
    /// previous input's analysis being `last`.
    fn repeat_of(&self, pre: &PreAnalysis, last: Option<&FrameAnalysis>) -> Option<bool> {
        self.detect_repeats
            .then(|| last.is_some_and(|l| preanalysis::repeats(pre, &l.pre)))
    }

    /// Analyse the next frame against the previous one.
    pub fn analyze(&mut self, input: &Picture) -> &FrameAnalysis {
        self.analyze_linked(input, FrameLink::PREVIOUS)
    }

    /// Analyse the next frame against the reference `link` names.
    pub fn analyze_linked(&mut self, input: &Picture, link: FrameLink) -> &FrameAnalysis {
        self.analyze_timed(input, link, &mut NoTimer)
    }

    /// [`Analyzer::analyze_linked`], reporting stage boundaries to `timer`.
    ///
    /// # Panics
    ///
    /// If `input` is not the size this analyzer was created for.
    pub fn analyze_timed(
        &mut self,
        input: &Picture,
        link: FrameLink,
        timer: &mut impl StageTimer,
    ) -> &FrameAnalysis {
        timed(timer, Stage::Ingest, || self.write_next(input));
        self.analyze_written_timed(link, timer)
    }

    /// Copy `input` into the next frame's buffer ([`Analyzer::next_frame_mut`]).
    ///
    /// # Panics
    ///
    /// If `input` is not the size this analyzer was created for.
    pub fn write_next(&mut self, input: &Picture) {
        let dsp = self.dsp;
        let next = self.next_frame_mut();
        assert_eq!(
            (input.width(), input.height()),
            (next.width(), next.height()),
            "frame size changed"
        );
        for plane in PlaneIndex::ALL {
            dsp.copy_plane(
                &input.plane(plane).view(),
                &mut next.plane_mut(plane).view_mut(),
            );
        }
    }

    /// The buffer the next frame can be written into directly (its visible
    /// area), saving [`Analyzer::analyze`]'s copy; then call
    /// [`Analyzer::analyze_written`]. Its border is the analyzer's business.
    pub fn next_frame_mut(&mut self) -> &mut Picture {
        // A new frame: any pre-analysis or levels of the one before are
        // stale.
        self.pending = None;
        let frame = &mut self.frames[self.next];
        frame.pyramid.invalidate();
        &mut frame.picture
    }

    /// Pre-analyse the written next frame against the reference `link`
    /// names, before the rest of its analysis (frame hold decides from it
    /// whether to analyse and code the frame at all; [`Analyzer::next_pre`]).
    /// The frame's analysis against the same reference reuses it, so this
    /// costs nothing extra. It also fills the frame's half-resolution level
    /// unless the current search settings reuse HME seeds for `link`, so
    /// set the frame's search settings first.
    pub fn pre_analyze_next(&mut self, link: FrameLink) {
        self.pending = None;
        let pre = self.next_frame_pre(link, &mut NoTimer);
        self.pending = Some(PendingPre {
            reference_slot: link.reference.slot,
            pre,
        });
    }

    /// The written next frame's pre-analysis against the reference `link`
    /// names, its border extended: the one [`Analyzer::pre_analyze_next`]
    /// left, or measured now.
    fn next_frame_pre(&mut self, link: FrameLink, timer: &mut impl StageTimer) -> PreAnalysis {
        if let Some(pending) = self
            .pending
            .take()
            .filter(|p| p.reference_slot == link.reference.slot)
        {
            return pending.pre;
        }
        let builds_pyramid = !self.reuses_hme_seeds(link);
        let stat_rows = self.stat_rows;
        let noise_hint = self.last.as_ref().map_or(0.0, |l| l.noise_sigma);
        let dsp = &self.dsp;
        let reference = self.slots[link.reference.slot];
        // The reference is never the frame being written (`next` is free).
        let (next, prev) = match reference {
            Some(i) => {
                let [next, prev] = self
                    .frames
                    .get_disjoint_mut([self.next, i])
                    .expect("the next frame is not a reference");
                (next, Some(&prev.picture))
            }
            None => (&mut self.frames[self.next], None),
        };
        timed(timer, Stage::Ingest, || {
            next.picture.extend_borders_with(dsp);
        });
        let downsample = next.pyramid.downsample();
        let half = builds_pyramid.then(|| (next.pyramid.half_to_fill(), downsample));
        timed(timer, Stage::PreAnalysis, || {
            preanalysis::analyze(dsp, &next.picture, prev, noise_hint, half, stat_rows)
        })
    }

    /// Whether a frame analysed against `link` centres its search on the
    /// previous frame's HME vectors instead of searching its own levels
    /// (`SearchConfig::hme_from_seed`), so builds no pyramid.
    fn reuses_hme_seeds(&self, link: FrameLink) -> bool {
        self.search.hme_from_seed
            && link.refresh == SlotMask::NONE
            && self.last.as_ref().is_some_and(|l| l.motion.is_some())
    }

    /// The written next frame's pre-analysis, after
    /// [`Analyzer::pre_analyze_next`].
    #[must_use]
    pub fn next_pre(&self) -> Option<&PreAnalysis> {
        self.pending.as_ref().map(|p| &p.pre)
    }

    /// Whether the reference `link` names holds the most recently analysed
    /// frame (so a next frame's `sad_prev` is its change from that frame).
    #[must_use]
    pub fn references_last(&self, link: FrameLink) -> bool {
        self.last.is_some() && self.slots[link.reference.slot] == Some(self.current)
    }

    /// Make the most recently analysed frame the reference source of every
    /// slot in `refresh` as well: a frame analysed as an inter frame and then
    /// coded as a key frame (a scene cut) refreshes every slot.
    pub fn refresh_current(&mut self, refresh: SlotMask) {
        // A reference needs its levels: later frames search against them.
        self.frames[self.current].build_pyramid(&self.dsp);
        let slot_count = self.frames.len() - 1;
        for (slot, held) in self.slots.iter_mut().enumerate().take(slot_count) {
            if refresh.contains(slot) {
                *held = Some(self.current);
            }
        }
        self.next = (0..self.frames.len())
            .find(|i| !self.slots.contains(&Some(*i)))
            .expect("one frame more than slots, so one is always free");
    }

    /// The most recently analysed frame (its border extended).
    #[must_use]
    pub fn frame(&self) -> &Picture {
        &self.frames[self.current].picture
    }

    /// The analysis of the most recently analysed frame, if any.
    #[must_use]
    pub fn analysis(&self) -> Option<&FrameAnalysis> {
        self.last.as_ref()
    }

    /// Analyse the frame written into [`Analyzer::next_frame_mut`] against
    /// the previous one.
    pub fn analyze_written(&mut self) -> &FrameAnalysis {
        self.analyze_written_timed(FrameLink::PREVIOUS, &mut NoTimer)
    }

    /// Analyse the frame written into [`Analyzer::next_frame_mut`] against
    /// the reference `link` names, reporting stage boundaries to `timer`.
    ///
    /// # Panics
    ///
    /// If `link` names a slot this analyzer does not keep.
    pub fn analyze_written_timed(
        &mut self,
        link: FrameLink,
        timer: &mut impl StageTimer,
    ) -> &FrameAnalysis {
        let slot_count = self.frames.len() - 1;
        assert!(
            link.reference.slot < slot_count,
            "reference slot {} not kept",
            link.reference.slot
        );
        let pre = self.next_frame_pre(link, timer);
        let index = self.next;
        let dsp = &self.dsp;
        // Only the HME search reads a frame's own levels, and later frames
        // read only references' levels.
        let seeded = self.reuses_hme_seeds(link);
        let frame = &mut self.frames[index];
        if !seeded {
            timed(timer, Stage::Pyramid, || frame.build_pyramid(dsp));
        }

        // HME level 1 is the only reader of the half levels' borders.
        if self.search.hme_level1 {
            let prev = self.slots[link.reference.slot];
            for i in [Some(index), prev].into_iter().flatten() {
                if self.frames[i].pyramid.is_built() {
                    self.frames[i].pyramid.ensure_half_border(&self.dsp);
                }
            }
        }
        let cur = &self.frames[index];
        let prev = self.slots[link.reference.slot].map(|i| &self.frames[i]);
        let last = self.last.take();
        let noise_hint = last.as_ref().map_or(0.0, |l| l.noise_sigma);
        let repeat = self.repeat_of(&pre, last.as_ref());

        let ratio = DistanceRatio::new(link.reference.distance, self.last_distance);
        let seeds = last
            .as_ref()
            .and_then(|l| l.motion.as_ref())
            .map(|m| TemporalSeeds::rescaled(m, ratio));
        let motion = prev.map(|prev| {
            let seeds = seeds.as_ref().map(TemporalSeeds::view);
            let inputs = MotionInputs {
                cur: cur.picture.plane(PlaneIndex::Y),
                prev: prev.picture.plane(PlaneIndex::Y),
                cur_quarter: cur.pyramid.quarter(),
                prev_quarter: prev.pyramid.quarter(),
                cur_half: cur.pyramid.half(),
                prev_half: prev.pyramid.half(),
                pre: &pre,
                last_hme: seeds.map(|s| s.hme),
                last_32x32: seeds.map(|s| s.field_32x32),
                last_16x16: seeds.map(|s| s.field_16x16),
                last_global: seeds.map_or(GlobalMotion::NONE, |s| s.global),
                config: self.search,
            };
            let mut effort = MotionEffort::default();
            let fields = timed(timer, Stage::Motion, || {
                motion::search(dsp, &inputs, &mut effort)
            });
            let global = timed(timer, Stage::GlobalMotion, || {
                global::fit(
                    &fields.aggregated[2],
                    &pre.blocks,
                    cur.picture.width(),
                    cur.picture.height(),
                )
            });
            MotionAnalysis {
                sb: fields.sb,
                hme: fields.hme,
                fine: fields.field_8x8,
                aggregated: fields.aggregated,
                global,
                effort,
            }
        });
        // The noise model reads flat blocks' best-match SADs as sensor noise;
        // over a longer distance real change adds to them, so only frames
        // matched against their immediate predecessor update it.
        let noise_sigma = match &motion {
            Some(m) if link.reference.distance == 1 => {
                noise::update(noise_hint, &pre.blocks, &m.fine)
            }
            Some(_) => noise_hint,
            None => noise::first_frame(&pre.blocks),
        };
        let still = self.still.update(&pre, self.still_config);
        for (slot, held) in self.slots.iter_mut().enumerate().take(slot_count) {
            if link.refresh.contains(slot) {
                *held = Some(index);
            }
        }
        self.current = index;
        self.next = (0..self.frames.len())
            .find(|i| !self.slots.contains(&Some(*i)))
            .expect("one frame more than slots, so one is always free");
        self.last_distance = link.reference.distance;
        self.last.insert(FrameAnalysis {
            pre,
            motion,
            noise_sigma,
            still,
            repeat,
        })
    }
}

/// Last frame's motion used to seed this frame's search: borrowed as is
/// when the reference distance is unchanged, rescaled otherwise.
enum TemporalSeeds<'a> {
    Borrowed(&'a MotionAnalysis),
    Rescaled {
        hme: Grid<Mv>,
        field_32x32: AggregatedField,
        field_16x16: AggregatedField,
        global: GlobalMotion,
    },
}

/// Borrowed views of [`TemporalSeeds`].
#[derive(Clone, Copy)]
struct SeedView<'a> {
    hme: &'a Grid<Mv>,
    field_32x32: &'a AggregatedField,
    field_16x16: &'a AggregatedField,
    global: GlobalMotion,
}

impl<'a> TemporalSeeds<'a> {
    fn rescaled(motion: &'a MotionAnalysis, ratio: DistanceRatio) -> Self {
        if ratio.is_identity() {
            return Self::Borrowed(motion);
        }
        Self::Rescaled {
            hme: link::scale_mv_grid(&motion.hme, ratio),
            field_32x32: link::scale_field(&motion.aggregated[1], ratio),
            field_16x16: link::scale_field(&motion.aggregated[0], ratio),
            global: link::scale_global(motion.global, ratio),
        }
    }

    fn view(&self) -> SeedView<'_> {
        match self {
            Self::Borrowed(m) => SeedView {
                hme: &m.hme,
                field_32x32: &m.aggregated[1],
                field_16x16: &m.aggregated[0],
                global: m.global,
            },
            Self::Rescaled {
                hme,
                field_32x32,
                field_16x16,
                global,
            } => SeedView {
                hme,
                field_32x32,
                field_16x16,
                global: *global,
            },
        }
    }
}

fn timed<T>(timer: &mut impl StageTimer, stage: Stage, f: impl FnOnce() -> T) -> T {
    timer.begin(stage);
    let result = f();
    timer.end(stage);
    result
}
