//! Motion estimation on source frames, following SVT-AV1 real-time M12
//! (`docs/reference/svt-av1-m12-rtc.md` §1; `docs/ARCHITECTURE.md` §2).
//!
//! Per 64x64 superblock (SB), against the previous source frame:
//!
//! 1. **Zero-vector gates.** The SB's zero-vector SAD, measured on every
//!    other row and doubled ("zz SAD"), decides how much searching it gets:
//!    - below [`STATIC_SAD_PER_PIXEL`] (or every 8x8 block static in
//!      pre-analysis): static bypass, zero vector, no search at all;
//!    - below [`SKIP_HME_SAD_PER_PIXEL`]: no hierarchical search, the
//!      full-pel search centres on zero;
//!    - below [`ONE_POSITION_SAD_PER_PIXEL`]: the full-pel search collapses
//!      to the centre position alone.
//!
//!    Pre-analysis's static map is kept alongside SVT's 1-per-pixel test:
//!    sensor noise alone often exceeds 1 per pixel on camera content, where
//!    the noise-aware map still recognises still blocks.
//! 2. **Hierarchical search (HME)**: at quarter resolution the SB (16x16
//!    there) is searched exhaustively over [`HME_RANGE_X`] x [`HME_RANGE_Y`]
//!    (by default; [`SearchConfig`] can narrow it) around the centre, on
//!    every other row, one SAD-scan call per row of positions; the centre
//!    is the best of zero, last frame's vector for this SB and the camera
//!    motion (see [`SEED_CANDIDATES`]). An 8x3 refinement at half
//!    resolution follows (HME level 1).
//! 3. **Full-pel refinement** at full resolution over a tiny 8x3 window
//!    around the HME result, plus a few predictor vectors (neighbouring SBs,
//!    last frame; more for SBs with mixed motion). One kernel call per
//!    position measures every square block from 8x8 to 64x64 (85 per SB, on
//!    every other row) and keeps each block's own best vector.
//!
//! Per SB it also records the best SAD at each block size (`me_dist` in
//! SVT) and the variance of its 8x8 SADs (`me_8x8_cost_variance`), which
//! later decisions (partitioning, intra gating, rate control) use.
//!
//! Vectors are whole luma samples; sub-pel refinement is a codec concern
//! and happens later, against reconstructed references. Live camera input
//! has no scene cuts; the zz gates cost one SAD per SB on still areas.

use nc_common::{Plane, PlaneView};
use nc_dsp::{BlockSize, Dsp, GRID_BLOCKS, GridBest, GridRows, prefetch_block};

use crate::effort::MotionEffort;
use crate::global::GlobalMotion;
use crate::grid::Grid;
use crate::mv::Mv;
use crate::preanalysis::PreAnalysis;

/// Superblock edge, luma samples.
pub const SB: usize = 64;
/// Smallest block the search reports, luma samples.
pub const BLOCK: usize = 8;
/// 8x8 blocks per SB edge.
const BLOCKS_PER_SB: usize = SB / BLOCK;
/// Pyramid scale of the HME level (quarter resolution).
pub const HME_SCALE: i16 = 4;
/// SB edge at the HME level.
pub(crate) const HME_BLOCK: usize = SB / HME_SCALE as usize;

/// Largest HME window half-width, in quarter-resolution samples (±64
/// luma); the default.
pub const HME_RANGE_X: i16 = 16;
/// Largest HME window half-height, in quarter-resolution samples (±16
/// luma); the default: the window is 33x9 positions, as SVT's distance-1
/// window at M12 (32x8). Larger vertical motion is reached through the
/// window centre (last frame's vector or the camera motion) once it has
/// been seen.
pub const HME_RANGE_Y: i16 = 4;
/// Positions per row of the largest HME window.
const HME_ROW_POSITIONS: usize = 2 * HME_RANGE_X as usize + 1;
/// Half-resolution refinement window: x in -4..=3, y in -1..=1 half-res
/// samples (8x3, as SVT's level 1).
const L1_RANGE_X: i16 = 4;
const L1_RANGE_Y: i16 = 1;
const L1_ROW_POSITIONS: usize = 2 * L1_RANGE_X as usize;
const L1_ROWS: usize = 2 * L1_RANGE_Y as usize + 1;
/// The half-resolution region every level-1 window position of an SB reads.
const L1_REGION_WIDTH: usize = HALF_BLOCK + L1_ROW_POSITIONS - 1;
const L1_REGION_HEIGHT: usize = HALF_BLOCK + L1_ROWS - 1;
/// SB edge at half resolution.
const HALF_BLOCK: usize = SB / 2;
/// An SB has mixed motion when its best single-vector SAD exceeds this
/// multiple of the sum of its 8x8 blocks' own bests. Measured (60 frames):
/// at 2, about 50 of 240 SBs per frame on crowd and 35 on bodycam qualify,
/// for -2% and -0.5% final 8x8 SAD; pan and static CCTV, under 5 SBs.
/// Provisional.
const MIXED_MOTION_FACTOR: u32 = 2;
/// Rows the full-pel grid reads (see `GridRows`): every other row, as SVT
/// does for its subsampled searches. Measured on the Orin (5 clips): motion
/// 26-29% faster; exact final 8x8 SAD of the chosen vectors 1-4% higher.
const GRID_ROWS: GridRows = GridRows::Even;
/// Largest full-pel window half-width (SVT M12's 8x3 window: x in -4..=3).
pub const MAX_WINDOW_HALF_WIDTH: i16 = 4;
/// Largest full-pel window half-height (y in -1..=1).
pub const MAX_WINDOW_HALF_HEIGHT: i16 = 1;
/// Largest full-resolution vector component: the HME reach plus the
/// full-pel window. Full-resolution candidates that reach past the frame
/// copies' border are measured on a gathered copy ([`Level::gathered`]).
pub const FINE_LIMIT: i16 = HME_LIMIT * HME_SCALE + MAX_WINDOW_HALF_WIDTH;

/// How much the search does per SB. The default is SVT M12's search;
/// lighter settings trade vector quality for speed, for codecs that need
/// less (registered as `analysis.*` params in the encoder).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchConfig {
    /// Full-pel window half-width: x in `-w..=w-1` around the HME result
    /// (1..=[`MAX_WINDOW_HALF_WIDTH`]).
    pub window_half_width: i16,
    /// Full-pel window half-height: y in `-h..=h` (0..=[`MAX_WINDOW_HALF_HEIGHT`]).
    pub window_half_height: i16,
    /// The full-pel window stops after its centre when the SB's 64x64 SAD
    /// per visible pixel there is below this (0: never stops early).
    pub window_exit_sad_per_pixel: f32,
    /// HME window half-width, quarter-resolution samples (1..=[`HME_RANGE_X`]).
    pub hme_range_x: i16,
    /// HME window half-height, quarter-resolution samples (0..=[`HME_RANGE_Y`]).
    pub hme_range_y: i16,
    /// Half-resolution refinement of the HME vector (SVT's level 1).
    pub hme_level1: bool,
    /// Level 1 runs only on frames whose mean 8x8 luma variance is below
    /// this (SVT M12 flat IPP: 1000); infinite: on every frame.
    pub hme_level1_max_variance: f32,
    /// Extra full-pel positions for SBs with mixed motion.
    pub mixed_motion: bool,
    /// The full-pel window of a hierarchical SB centres on the previous
    /// frame's HME vector, rescaled to this frame's distance, instead of a
    /// new HME search (when that vector exists). A disposable-frame lever:
    /// the frame then needs no pyramid of its own.
    pub hme_from_seed: bool,
    /// Where the zero-vector SADs (each SB's zz SAD, and the zero-vector
    /// grid of static and one-position SBs) come from.
    pub zero_sads: ZeroSads,
    /// zz SAD gates choosing each SB's search path.
    pub gates: ZeroVectorGates,
}

/// Where the motion search's zero-vector SADs come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroSads {
    /// Measured on even rows of both frames.
    #[default]
    Measured,
    /// Pre-analysis' 8x8 SADs against the same reference: exact SADs
    /// instead of even-row estimates, without a second read of both frames.
    PreAnalysis,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            window_half_width: MAX_WINDOW_HALF_WIDTH,
            window_half_height: MAX_WINDOW_HALF_HEIGHT,
            window_exit_sad_per_pixel: 0.0,
            hme_range_x: HME_RANGE_X,
            hme_range_y: HME_RANGE_Y,
            hme_level1: true,
            hme_level1_max_variance: f32::INFINITY,
            mixed_motion: true,
            hme_from_seed: false,
            zero_sads: ZeroSads::Measured,
            gates: ZeroVectorGates::default(),
        }
    }
}

impl SearchConfig {
    /// This config with the windows clamped to the supported range.
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            window_half_width: self.window_half_width.clamp(1, MAX_WINDOW_HALF_WIDTH),
            window_half_height: self.window_half_height.clamp(0, MAX_WINDOW_HALF_HEIGHT),
            hme_range_x: self.hme_range_x.clamp(1, HME_RANGE_X),
            hme_range_y: self.hme_range_y.clamp(0, HME_RANGE_Y),
            ..self
        }
    }

    fn window_x(self) -> std::ops::RangeInclusive<i16> {
        -self.window_half_width..=self.window_half_width - 1
    }

    fn window_y(self) -> std::ops::RangeInclusive<i16> {
        -self.window_half_height..=self.window_half_height
    }

    fn in_window(self, offset: Mv) -> bool {
        self.window_x().contains(&offset.x) && self.window_y().contains(&offset.y)
    }
}

/// zz SAD per pixel thresholds choosing an SB's search path (step 1 of the
/// module notes). Raising them sends more SBs down the cheaper paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroVectorGates {
    /// Below this the SB is static: no search.
    pub static_per_pixel: f32,
    /// Below this the full-pel window is one position at zero.
    pub one_position_per_pixel: f32,
    /// Below this HME is skipped and the window centres on zero.
    pub skip_hme_per_pixel: f32,
}

impl Default for ZeroVectorGates {
    fn default() -> Self {
        Self {
            static_per_pixel: STATIC_SAD_PER_PIXEL,
            one_position_per_pixel: ONE_POSITION_SAD_PER_PIXEL,
            skip_hme_per_pixel: SKIP_HME_SAD_PER_PIXEL,
        }
    }
}

/// zz SAD per pixel below which an SB is static by default: no search
/// (SVT's `me_static_b64_th`, 4096 for 64x64).
pub const STATIC_SAD_PER_PIXEL: f32 = 1.0;
/// zz SAD per pixel below which HME is skipped by default (SVT: 8192 for
/// 64x64).
pub const SKIP_HME_SAD_PER_PIXEL: f32 = 2.0;
/// zz SAD per pixel below which the full-pel window is one position by
/// default (SVT: 5461 for 64x64).
pub const ONE_POSITION_SAD_PER_PIXEL: f32 = 4.0 / 3.0;

/// Which extra HME centres are tried (one quarter-resolution SAD each); the
/// window centres on the best of zero and these. Measured (60 frames, final
/// 8x8 SAD): both together -7% on bodycam (either alone about -6%), no
/// change on pan, crowd or handheld; they also let steadily accelerating
/// motion be followed beyond the window.
pub const SEED_CANDIDATES: SeedCandidates = SeedCandidates {
    temporal: true,
    global: true,
};

/// Extra HME centre candidates.
#[derive(Debug, Clone, Copy)]
pub struct SeedCandidates {
    /// Last frame's HME vector for this SB.
    pub temporal: bool,
    /// Last frame's camera motion at this SB's centre.
    pub global: bool,
}

/// Best vector and SAD of one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionBlock {
    /// Vector, in luma samples.
    pub mv: Mv,
    /// SAD at `mv`.
    pub sad: u32,
}

/// One block of a 16x16, 32x32 or 64x64 field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AggregatedBlock {
    /// Best single vector for the whole block.
    pub mv: Mv,
    /// SAD of the whole block at `mv`.
    pub whole_sad: u32,
    /// Sum of the four children's best SADs: the cost if each child keeps
    /// its own vector. `whole_sad` close to it means the area moves as one.
    pub split_sad: u32,
}

/// A 16x16, 32x32 or 64x64 field.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregatedField {
    /// Block edge in luma samples (16, 32 or 64).
    pub size: usize,
    /// Per-block results.
    pub blocks: Grid<AggregatedBlock>,
}

/// How an SB was searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SbSearch {
    /// Static bypass: zero vector, no search.
    #[default]
    Static,
    /// Full-pel search at the zero centre only (zz SAD tiny).
    OnePosition,
    /// Full-pel window around zero, no HME (zz SAD small).
    WindowAtZero,
    /// HME, then the full-pel window around its result.
    Hierarchical,
}

/// Per-SB motion statistics.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SbStats {
    /// Zero-vector SAD of the SB (every other row, doubled).
    pub zz_sad: u32,
    /// Which search path the SB took.
    pub search: SbSearch,
    /// Best total SAD at each block size [8x8, 16x16, 32x32, 64x64]: the
    /// sum over the SB of each block's own best SAD (SVT's `me_dist`).
    pub me_dist: [u32; 4],
    /// Variance of the SB's 64 best 8x8 SADs (SVT's `me_8x8_cost_variance`):
    /// high means the SB mixes well- and badly-predicted areas.
    pub cost_variance_8x8: f32,
}

/// The motion fields of one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionFields {
    /// Per-SB statistics and HME/zz results.
    pub sb: Grid<SbStats>,
    /// Per-SB HME vector (luma samples), the next frame's temporal seed.
    pub hme: Grid<Mv>,
    /// 8x8 field.
    pub field_8x8: Grid<MotionBlock>,
    /// 16x16, 32x32 and 64x64 fields.
    pub aggregated: [AggregatedField; 3],
}

/// Everything the search reads.
#[derive(Debug)]
pub struct MotionInputs<'a> {
    /// Full-resolution luma of the current frame (borders of at least
    /// `SB - 1`, which an SB overhanging the picture's edge reads).
    pub cur: &'a Plane,
    /// Full-resolution luma of the previous frame (same borders).
    pub prev: &'a Plane,
    /// Quarter-resolution luma of the current frame.
    pub cur_quarter: &'a Plane,
    /// Quarter-resolution luma of the previous frame.
    pub prev_quarter: &'a Plane,
    /// Half-resolution luma of the current frame.
    pub cur_half: &'a Plane,
    /// Half-resolution luma of the previous frame.
    pub prev_half: &'a Plane,
    /// This frame's pre-analysis (static map).
    pub pre: &'a PreAnalysis,
    /// Last frame's HME vectors, if any (temporal seeds).
    pub last_hme: Option<&'a Grid<Mv>>,
    /// Last frame's 32x32 field, if any (temporal full-pel candidates).
    pub last_32x32: Option<&'a AggregatedField>,
    /// Last frame's 16x16 field, if any (candidates for mixed motion).
    pub last_16x16: Option<&'a AggregatedField>,
    /// Last frame's camera motion (global seed; `GlobalMotion::NONE` if
    /// none).
    pub last_global: GlobalMotion,
    /// How much to search.
    pub config: SearchConfig,
}

/// Search every SB of the frame.
#[must_use]
pub fn search(dsp: &Dsp, inputs: &MotionInputs<'_>, effort: &mut MotionEffort) -> MotionFields {
    let (width, height) = (inputs.cur.width(), inputs.cur.height());
    let sb_cols = width.div_ceil(SB);
    let sb_rows = height.div_ceil(SB);
    let mut fields = Fields::new(width, height);
    let mut sb_stats = Grid::filled(sb_cols, sb_rows, SbStats::default());
    let mut hme = Grid::filled(sb_cols, sb_rows, Mv::ZERO);
    let full = Level::new(inputs.cur, inputs.prev);
    let quarter = Level::new(inputs.cur_quarter, inputs.prev_quarter);
    let half = Level::new(inputs.cur_half, inputs.prev_half);
    let level1 = level1_enabled(inputs);
    for (sb_col, sb_row) in sb_stats.positions() {
        let sb = SbSearchContext {
            dsp,
            inputs,
            level1,
            full: &full,
            half: &half,
            quarter: &quarter,
            sb_col,
            sb_row,
            neighbours: fields.neighbour_vectors(sb_col, sb_row),
            fields: &fields,
            temporal: inputs.last_32x32.map_or([None; 4], |f| {
                [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dc, dr)| {
                    f.blocks
                        .try_get((2 * sb_col + dc) as isize, (2 * sb_row + dr) as isize)
                        .map(|b| b.mv)
                })
            }),
        };
        let (stats, centre, grid_best) = sb.run(effort);
        if sb_col + 1 < sb_cols {
            full.prefetch_window(sb_col + 1, sb_row, centre, inputs.config);
        }
        *sb_stats.get_mut(sb_col, sb_row) = stats;
        *hme.get_mut(sb_col, sb_row) = centre;
        fields.store(sb_col, sb_row, &grid_best);
    }
    let (field_8x8, aggregated) = fields.finish();
    MotionFields {
        sb: sb_stats,
        hme,
        field_8x8,
        aggregated,
    }
}

/// Whether this frame runs HME level 1: always when configured without a
/// variance limit; otherwise only on frames whose mean 8x8 luma variance is
/// below it (SVT M12 runs level 1 on low-variance frames only).
fn level1_enabled(inputs: &MotionInputs<'_>) -> bool {
    let config = inputs.config;
    config.hme_level1
        && (config.hme_level1_max_variance.is_infinite()
            || inputs.pre.mean_variance() < config.hme_level1_max_variance)
}

/// A source plane, the reference plane it is matched against, and the
/// reference's padded view (blocks are addressed by position + vector).
struct Level<'a> {
    cur: &'a Plane,
    reference: PlaneView<'a>,
    border: (isize, isize),
}

impl<'a> Level<'a> {
    fn new(cur: &'a Plane, prev: &'a Plane) -> Self {
        Self {
            cur,
            reference: prev.padded_view(),
            border: (prev.border_x() as isize, prev.border_y() as isize),
        }
    }

    fn source(&self, x: isize, y: isize, size: usize) -> PlaneView<'a> {
        self.cur
            .region(x, y, size, size)
            .expect("block inside padded plane")
    }

    /// Hint the reference rows of the full-pel window of SB (`sb_col`,
    /// `sb_row`) centred on `centre` (neighbouring SBs mostly move alike, so
    /// the left SB's centre is the guess). The hint is issued before the
    /// SB's zero SAD and HME, which hide the fetch; without it the grid
    /// kernel's first positions wait on reference cache misses. Exact (a
    /// hint only); Jetson h264-fast cycles about -0.5% crowd / -0.35% bodycam
    /// (aarch64 only, see `nc_dsp::prefetch_block`).
    fn prefetch_window(&self, sb_col: usize, sb_row: usize, centre: Mv, config: SearchConfig) {
        let corner = Mv::new(
            centre.x - config.window_half_width,
            centre.y - config.window_half_height,
        )
        .clamped(FINE_LIMIT);
        // A hint only: a window reaching past the border is hinted from the
        // border's edge.
        let (x, y) = (
            ((sb_col * SB) as isize + isize::from(corner.x) + self.border.0).max(0) as usize,
            ((sb_row * SB) as isize + isize::from(corner.y) + self.border.1).max(0) as usize,
        );
        let width = SB + 2 * config.window_half_width as usize;
        let height = SB + 2 * config.window_half_height as usize;
        prefetch_block(&self.reference, x, y, width, height);
    }

    /// Position in the padded reference of the block at visible (`x`, `y`)
    /// displaced by `mv`.
    fn position(&self, x: isize, y: isize, mv: Mv) -> (usize, usize) {
        let px = x + isize::from(mv.x) + self.border.0;
        let py = y + isize::from(mv.y) + self.border.1;
        (
            usize::try_from(px).expect("search limits keep blocks inside the padded reference"),
            usize::try_from(py).expect("search limits keep blocks inside the padded reference"),
        )
    }

    /// Position in the padded reference of the `width` x `height` region
    /// whose top-left sample is at visible (`x`, `y`), if the region lies
    /// inside the padded reference.
    fn region_position(
        &self,
        (x, y): (isize, isize),
        width: usize,
        height: usize,
    ) -> Option<(usize, usize)> {
        let px = usize::try_from(x + self.border.0).ok()?;
        let py = usize::try_from(y + self.border.1).ok()?;
        let inside = px + width <= self.reference.width() && py + height <= self.reference.height();
        inside.then_some((px, py))
    }

    /// [`Self::position`] of the SB-square block at visible (`x`, `y`)
    /// displaced by `mv`, if that block lies inside the padded reference.
    fn sb_position(&self, x: isize, y: isize, mv: Mv) -> Option<(usize, usize)> {
        self.region_position((x + isize::from(mv.x), y + isize::from(mv.y)), SB, SB)
    }

    /// The reference region whose top-left sample is at visible (`x`, `y`),
    /// `width` samples wide (`out`'s row stride), gathered as the border
    /// extension would hold it: coordinates clamped to the picture. For
    /// regions past the border: the levels keep a border of one block only,
    /// since borders as wide as the search reach (`FINE_LIMIT` plus an SB at
    /// full resolution) were ~4% of HEVC Faster's x86 time (non-temporal
    /// stores of ~1 MB per 720p frame) for vectors that rarely reach that
    /// far.
    fn gathered(&self, (x, y): (isize, isize), width: usize, out: &mut [u8]) {
        let (bx, by) = self.border;
        let visible_width = self.reference.width() as isize - 2 * bx;
        let visible_height = self.reference.height() as isize - 2 * by;
        for (r, row) in out.chunks_exact_mut(width).enumerate() {
            let sy = ((y + r as isize).clamp(0, visible_height - 1) + by) as usize;
            let line = self.reference.row(sy);
            for (c, sample) in row.iter_mut().enumerate() {
                *sample = line[((x + c as isize).clamp(0, visible_width - 1) + bx) as usize];
            }
        }
    }
}

/// Best vector and SAD of every square block of one SB, in the kernel's
/// layout (`GridBest`: 64 8x8, 16 16x16, 4 32x32 and the 64x64 block,
/// row-major within each size), vectors packed into the tag.
#[derive(Debug, Clone, Copy, Default)]
struct SbBest(GridBest);

/// Where each block size starts in `GridBest`.
const AT_16: usize = 64;
const AT_32: usize = 80;
const AT_64: usize = 84;

impl SbBest {
    fn block(&self, i: usize) -> MotionBlock {
        MotionBlock {
            mv: unpack(self.0.tag[i]),
            sad: self.0.sad[i],
        }
    }

    /// [8x8, 16x16, 32x32, 64x64] sums of the blocks' best SADs.
    fn me_dist(&self) -> [u32; 4] {
        let total = |range: std::ops::Range<usize>| self.0.sad[range].iter().sum();
        [
            total(0..AT_16),
            total(AT_16..AT_32),
            total(AT_32..AT_64),
            self.0.sad[AT_64],
        ]
    }

    /// `(n Σs² − (Σs)²) / n²` in integers. Each 8x8 SAD is below 2^14, so
    /// the numerator stays below 2^40 and converts to `f64` exactly: the
    /// result is the same `f64` as the two-pass mean-removed sum (whose
    /// terms are also exact), without its chain of 128 dependent `f64` adds.
    fn cost_variance_8x8(&self) -> f32 {
        let b8 = &self.0.sad[..AT_16];
        let n = b8.len() as u64;
        let sum: u64 = b8.iter().map(|&s| u64::from(s)).sum();
        let sum_sq: u64 = b8.iter().map(|&s| u64::from(s) * u64::from(s)).sum();
        let numerator = n * sum_sq - sum * sum;
        (numerator as f64 / (n * n) as f64) as f32
    }
}

/// Every 8x8 block of an SB counts (the SB is inside the picture).
const ALL_VISIBLE: [u32; 64] = [u32::MAX; 64];

fn pack(mv: Mv) -> u32 {
    u32::from(mv.x as u16) | (u32::from(mv.y as u16) << 16)
}

fn unpack(packed: u32) -> Mv {
    Mv::new(packed as u16 as i16, (packed >> 16) as u16 as i16)
}

struct SbSearchContext<'a, 'b> {
    dsp: &'a Dsp,
    inputs: &'a MotionInputs<'b>,
    /// Whether HME results are refined at half resolution this frame.
    level1: bool,
    full: &'a Level<'b>,
    half: &'a Level<'b>,
    quarter: &'a Level<'b>,
    sb_col: usize,
    sb_row: usize,
    /// 64x64 vectors of the left, top and top-right SBs (already searched).
    neighbours: [Option<Mv>; 3],
    /// Last frame's 32x32 vectors of this SB.
    temporal: [Option<Mv>; 4],
    /// This frame's fields so far, for the mixed-motion candidates. These
    /// are gathered only for SBs that take that step (~1K instructions
    /// each otherwise spent on every SB).
    fields: &'a Fields,
}

impl SbSearchContext<'_, '_> {
    /// Search the SB: its statistics, HME centre and per-block bests.
    fn run(&self, effort: &mut MotionEffort) -> (SbStats, Mv, SbBest) {
        let (x, y) = ((self.sb_col * SB) as isize, (self.sb_row * SB) as isize);
        let src = self.full.source(x, y, SB);
        let size = BlockSize::new(SB, SB).expect("64x64 is valid");
        let from_pre = self.inputs.config.zero_sads == ZeroSads::PreAnalysis;
        let zz_sad = if from_pre {
            self.zero_sad_from_pre()
        } else {
            let zero = self.full.position(x, y, Mv::ZERO);
            2 * self
                .dsp
                .sad_even_rows_at(size, &src, &self.full.reference, zero)
        };
        let per_pixel = zz_sad as f32 / (SB * SB) as f32;

        let gates = self.inputs.config.gates;
        let (search, centre) = if per_pixel < gates.static_per_pixel || self.all_static() {
            (SbSearch::Static, Mv::ZERO)
        } else if per_pixel < gates.one_position_per_pixel {
            (SbSearch::OnePosition, Mv::ZERO)
        } else if per_pixel < gates.skip_hme_per_pixel {
            (SbSearch::WindowAtZero, Mv::ZERO)
        } else {
            (SbSearch::Hierarchical, self.hme_or_seed(effort))
        };
        effort.record_sb(search);

        let best = match search {
            SbSearch::Static | SbSearch::OnePosition => {
                // One grid at zero: the SADs of the still blocks.
                if from_pre {
                    self.zero_grid_from_pre()
                } else {
                    let mut best = SbBest::default();
                    self.evaluate(&src, Mv::ZERO, &self.visible(), &mut best);
                    best
                }
            }
            SbSearch::WindowAtZero | SbSearch::Hierarchical => self.full_pel(&src, centre, effort),
        };
        let stats = SbStats {
            zz_sad,
            search,
            me_dist: best.me_dist(),
            cost_variance_8x8: best.cost_variance_8x8(),
        };
        (stats, centre, best)
    }

    /// Mask of the SB's 64 8x8 blocks inside the picture (`u32::MAX`) or
    /// outside it (0): blocks outside count as SAD zero, so larger blocks at
    /// the picture edge are judged on their visible part only.
    fn visible(&self) -> [u32; 64] {
        let blocks = &self.inputs.pre.blocks;
        let (c0, r0) = (self.sb_col * BLOCKS_PER_SB, self.sb_row * BLOCKS_PER_SB);
        if c0 + BLOCKS_PER_SB <= blocks.cols() && r0 + BLOCKS_PER_SB <= blocks.rows() {
            return ALL_VISIBLE;
        }
        std::array::from_fn(|i| {
            if c0 + i % 8 < blocks.cols() && r0 + i / 8 < blocks.rows() {
                u32::MAX
            } else {
                0
            }
        })
    }

    /// Fold the SB's SADs at `mv` into `best`.
    fn evaluate(&self, src: &PlaneView<'_>, mv: Mv, visible: &[u32; 64], best: &mut SbBest) {
        let (x, y) = ((self.sb_col * SB) as isize, (self.sb_row * SB) as isize);
        match self.full.sb_position(x, y, mv) {
            Some(position) => self.dsp.sad_grid_best_64x64_at(
                src,
                &self.full.reference,
                position,
                visible,
                GRID_ROWS,
                &mut best.0,
                pack(mv),
            ),
            None => self.evaluate_gathered(src, mv, visible, best),
        }
    }

    /// [`Self::evaluate`] of four vectors in order (the batched kernel keeps
    /// input-order ties, so this equals four calls).
    fn evaluate_x4(
        &self,
        src: &PlaneView<'_>,
        mvs: [Mv; 4],
        visible: &[u32; 64],
        best: &mut SbBest,
    ) {
        let (x, y) = ((self.sb_col * SB) as isize, (self.sb_row * SB) as isize);
        let positions = mvs.map(|mv| self.full.sb_position(x, y, mv));
        if let [Some(a), Some(b), Some(c), Some(d)] = positions {
            self.dsp.sad_grid_best_x4_at(
                src,
                &self.full.reference,
                [a, b, c, d],
                visible,
                GRID_ROWS,
                &mut best.0,
                mvs.map(pack),
            );
        } else {
            for mv in mvs {
                self.evaluate(src, mv, visible, best);
            }
        }
    }

    /// [`Self::evaluate`] for a vector reaching past the reference's border.
    #[cold]
    fn evaluate_gathered(
        &self,
        src: &PlaneView<'_>,
        mv: Mv,
        visible: &[u32; 64],
        best: &mut SbBest,
    ) {
        let (x, y) = ((self.sb_col * SB) as isize, (self.sb_row * SB) as isize);
        let mut block = [0u8; SB * SB];
        let corner = (x + isize::from(mv.x), y + isize::from(mv.y));
        self.full.gathered(corner, SB, &mut block);
        let reference = PlaneView::new(&block, SB, SB, SB).expect("an SB-square block");
        self.dsp.sad_grid_best_64x64_at(
            src,
            &reference,
            (0, 0),
            visible,
            GRID_ROWS,
            &mut best.0,
            pack(mv),
        );
    }

    /// The SB's zero-vector SAD from pre-analysis' 8x8 SADs (blocks
    /// outside the grid count as zero, as in [`Self::evaluate`]).
    fn zero_sad_from_pre(&self) -> u32 {
        let blocks = &self.inputs.pre.blocks;
        let (c0, r0) = (self.sb_col * BLOCKS_PER_SB, self.sb_row * BLOCKS_PER_SB);
        (r0..r0 + BLOCKS_PER_SB)
            .flat_map(|r| blocks.row_run(c0, r, BLOCKS_PER_SB))
            .map(|block| block.sad_prev.unwrap_or(0))
            .sum()
    }

    /// The SB's zero-vector grid from pre-analysis' 8x8 SADs (blocks outside
    /// the grid count as zero, as in [`Self::evaluate`]); larger blocks sum
    /// their 8x8 blocks.
    fn zero_grid_from_pre(&self) -> SbBest {
        let blocks = &self.inputs.pre.blocks;
        let (c0, r0) = (self.sb_col * BLOCKS_PER_SB, self.sb_row * BLOCKS_PER_SB);
        let mut grid = GridBest {
            sad: [0; GRID_BLOCKS],
            tag: [pack(Mv::ZERO); GRID_BLOCKS],
        };
        for (r, out) in grid.sad[..AT_16]
            .as_chunks_mut::<BLOCKS_PER_SB>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (sad, block) in out
                .iter_mut()
                .zip(blocks.row_run(c0, r0 + r, BLOCKS_PER_SB))
            {
                *sad = block.sad_prev.unwrap_or(0);
            }
        }
        // 16x16, 32x32 and 64x64 blocks, row-major within each size.
        for i in 0..AT_32 - AT_16 {
            let (c, r) = (2 * (i % 4), 2 * (i / 4));
            grid.sad[AT_16 + i] = [(c, r), (c + 1, r), (c, r + 1), (c + 1, r + 1)]
                .iter()
                .map(|&(c, r)| grid.sad[r * BLOCKS_PER_SB + c])
                .sum();
        }
        for i in 0..AT_64 - AT_32 {
            let (c, r) = (2 * (i % 2), 2 * (i / 2));
            grid.sad[AT_32 + i] = [(c, r), (c + 1, r), (c, r + 1), (c + 1, r + 1)]
                .iter()
                .map(|&(c, r)| grid.sad[AT_16 + r * 4 + c])
                .sum();
        }
        grid.sad[AT_64] = grid.sad[AT_32..AT_64].iter().sum();
        SbBest(grid)
    }

    /// Whether pre-analysis found every 8x8 block of the SB (inside the
    /// picture) static.
    fn all_static(&self) -> bool {
        let blocks = &self.inputs.pre.blocks;
        let (c0, r0) = (self.sb_col * BLOCKS_PER_SB, self.sb_row * BLOCKS_PER_SB);
        (r0..(r0 + BLOCKS_PER_SB).min(blocks.rows())).all(|r| {
            (c0..(c0 + BLOCKS_PER_SB).min(blocks.cols())).all(|c| blocks.get(c, r).is_static)
        })
    }

    /// Quarter-resolution search: the best vector (in luma samples) over
    /// the window around the best of zero and the seed candidates.
    /// The window centre of a hierarchical SB: the previous frame's
    /// (rescaled) HME vector with `hme_from_seed`, else a new HME search.
    fn hme_or_seed(&self, effort: &mut MotionEffort) -> Mv {
        let seed = self
            .inputs
            .last_hme
            .filter(|_| self.inputs.config.hme_from_seed)
            .and_then(|grid| grid.try_get(self.sb_col as isize, self.sb_row as isize));
        match seed {
            Some(&mv) => mv,
            None => self.hme(effort),
        }
    }

    fn hme(&self, effort: &mut MotionEffort) -> Mv {
        let q = self.quarter;
        // A superblock overhanging the picture edge would match its padding
        // too; the HME block slides back inside the picture instead (it
        // then also covers part of the neighbouring SB, which moves alike).
        let inside =
            |sb: usize, extent: usize| (sb * HME_BLOCK).min(extent.saturating_sub(HME_BLOCK));
        let (x, y) = (
            inside(self.sb_col, q.cur.width()) as isize,
            inside(self.sb_row, q.cur.height()) as isize,
        );
        let src = q.source(x, y, HME_BLOCK);
        let centre = self.hme_centre(&src, x, y, effort);
        let config = self.inputs.config;
        let (range_x, range_y) = (config.hme_range_x, config.hme_range_y);
        let mut positions = [0u32; HME_ROW_POSITIONS];
        let row = &mut positions[..=2 * range_x as usize];
        let rows = 2 * range_y as usize + 1;
        let mut best = (u32::MAX, 0, 0);
        for iy in 0..rows {
            // `hme_centre` keeps the centre far enough inside the border.
            let left = Mv::new(centre.x - range_x, centre.y - range_y + iy as i16);
            self.dsp
                .sad_scan_16x16_even_rows_at(&src, &q.reference, q.position(x, y, left), row);
            // The row minimum is a vectorisable reduction; the position of
            // it is looked up only when it beats the best so far.
            let row_min = row.iter().copied().min().unwrap_or(u32::MAX);
            if row_min < best.0 {
                let ix = row.iter().position(|&sad| sad == row_min).unwrap_or(0);
                best = (row_min, ix, iy);
            }
        }
        effort.hme.add_block((row.len() * rows) as u32);
        let (_, ix, iy) = best;
        let quarter = Mv::new(
            centre.x - range_x + ix as i16,
            centre.y - range_y + iy as i16,
        );
        if self.level1 {
            self.hme_level1(quarter, effort)
        } else {
            Mv::new(quarter.x * HME_SCALE, quarter.y * HME_SCALE)
        }
    }

    /// Half-resolution refinement of a quarter-resolution vector over an
    /// 8x3 window (SVT's HME level 1). One quarter-resolution sample is four
    /// luma samples, more than the full-pel window's ±1 rows can bridge;
    /// after this step the full-pel window starts within a luma sample of
    /// the best vector. Returns the vector in luma samples.
    ///
    /// SVT M12 runs level 1 only on low-variance frames; it is always on
    /// here because without it about half of the 8x8 blocks of a synthetic
    /// (-60, 30) translation missed their exact vector.
    fn hme_level1(&self, quarter: Mv, effort: &mut MotionEffort) -> Mv {
        let h = self.half;
        let (x, y) = (
            (self.sb_col * HALF_BLOCK) as isize,
            (self.sb_row * HALF_BLOCK) as isize,
        );
        let centre = quarter.scaled(2);
        let mut sums = [[0u32; L1_ROW_POSITIONS]; L1_ROWS];
        let mut row = [0u32; L1_ROW_POSITIONS];
        // Every window position's block lies in one region; past the half
        // level's border it is gathered (`Level::gathered`).
        let corner = (
            x + isize::from(centre.x - L1_RANGE_X),
            y + isize::from(centre.y - L1_RANGE_Y),
        );
        let mut gathered = [0u8; L1_REGION_WIDTH * L1_REGION_HEIGHT];
        let mut reference = (h.reference, (0, 0));
        if let Some(position) = h.region_position(corner, L1_REGION_WIDTH, L1_REGION_HEIGHT) {
            reference.1 = position;
        } else {
            h.gathered(corner, L1_REGION_WIDTH, &mut gathered);
            reference.0 = PlaneView::new(
                &gathered,
                L1_REGION_WIDTH,
                L1_REGION_WIDTH,
                L1_REGION_HEIGHT,
            )
            .expect("the region's samples");
        }
        let (reference, (px, py)) = reference;
        // The 32x32 half-resolution SB as four 16x16 scans per window row.
        for (dqx, dqy) in [(0, 0), (16, 0), (0, 16), (16, 16)] {
            let src = h.source(x + dqx, y + dqy, 16);
            for (iy, total) in sums.iter_mut().enumerate() {
                self.dsp.sad_scan_16x16_even_rows_at(
                    &src,
                    &reference,
                    (px + dqx as usize, py + dqy as usize + iy),
                    &mut row,
                );
                for (t, &sad) in total.iter_mut().zip(&row) {
                    *t += sad;
                }
            }
        }
        effort
            .hme_level1
            .add_block((L1_ROW_POSITIONS * L1_ROWS) as u32);
        let mut best = (u32::MAX, centre);
        for (iy, total) in sums.iter().enumerate() {
            for (ix, &sad) in total.iter().enumerate() {
                if sad < best.0 {
                    let mv = Mv::new(
                        centre.x - L1_RANGE_X + ix as i16,
                        centre.y - L1_RANGE_Y + iy as i16,
                    );
                    best = (sad, mv);
                }
            }
        }
        best.1.scaled(2)
    }

    /// The HME window centre: zero, or a seed candidate if it matches better
    /// (quarter-resolution vector).
    fn hme_centre(&self, src: &PlaneView<'_>, x: isize, y: isize, effort: &mut MotionEffort) -> Mv {
        let q = self.quarter;
        let mut seeds = [None; 2];
        if SEED_CANDIDATES.temporal {
            seeds[0] = self
                .inputs
                .last_hme
                .map(|f| *f.get(self.sb_col, self.sb_row));
        }
        if SEED_CANDIDATES.global {
            let centre = (
                (self.sb_col * SB + SB / 2) as f32,
                (self.sb_row * SB + SB / 2) as f32,
            );
            seeds[1] = Some(self.inputs.last_global.mv_at(centre.0, centre.1));
        }
        let mut best = (u32::MAX, Mv::ZERO);
        let mut tried = 0;
        for mv in std::iter::once(Mv::ZERO).chain(seeds.into_iter().flatten().map(to_quarter)) {
            let mv = mv.clamped(HME_LIMIT - HME_RANGE_X.max(HME_RANGE_Y));
            if tried > 0 && mv == Mv::ZERO {
                continue;
            }
            let mut sad = [0u32; 1];
            self.dsp
                .sad_scan_16x16_even_rows_at(src, &q.reference, q.position(x, y, mv), &mut sad);
            tried += 1;
            if sad[0] < best.0 {
                best = (sad[0], mv);
            }
        }
        effort.hme_seeds += tried - 1;
        best.1
    }

    /// The 8x3 full-pel window around `centre`, every block keeping its own
    /// best.
    fn full_pel(&self, src: &PlaneView<'_>, centre: Mv, effort: &mut MotionEffort) -> SbBest {
        let mut best = SbBest::default();
        let visible = self.visible();
        let config = self.inputs.config;
        let mut positions = 0;
        // Early exit (off by default): the centre first, and the rest of
        // the window only when the centre leaves a large residual. What it
        // skips: up to 23 grid positions per SB whose motion the HME result
        // already matches; measured effect in `AnalysisParams`.
        let centre_only = config.window_exit_sad_per_pixel > 0.0 && {
            let start = centre.clamped(FINE_LIMIT);
            self.evaluate(src, start, &visible, &mut best);
            positions += 1;
            let pixels = visible.iter().filter(|&&v| v != 0).count() * BLOCK * BLOCK;
            (best.0.sad[AT_64] as f32) < config.window_exit_sad_per_pixel * pixels as f32
        };
        if !centre_only {
            let skip_centre = positions > 0;
            let mut batch = [Mv::ZERO; 4];
            let mut pending = 0;
            for dy in config.window_y() {
                for dx in config.window_x() {
                    if skip_centre && (dx, dy) == (0, 0) {
                        continue;
                    }
                    let mv = Mv::new(centre.x + dx, centre.y + dy).clamped(FINE_LIMIT);
                    batch[pending] = mv;
                    pending += 1;
                    if pending == 4 {
                        self.evaluate_x4(src, batch, &visible, &mut best);
                        pending = 0;
                    }
                    positions += 1;
                }
            }
            for mv in &batch[..pending] {
                self.evaluate(src, *mv, &visible, &mut best);
            }
        }
        // Extra positions where the window misses them: the 64x64 vectors of
        // the neighbouring SBs (blocks on a motion boundary that move with
        // the neighbour) and last frame's 32x32 vectors of this SB (objects
        // that keep moving). Measured (60 frames, final 8x8 SAD): -12% to
        // -24% on bodycam and crowd, -2% to -6% on pan, handheld and static
        // CCTV, for 4-15% more grid positions.
        let mut tried = [Mv::ZERO; 7];
        let mut count = 0;
        for mv in self.neighbours.into_iter().chain(self.temporal).flatten() {
            let mv = mv.clamped(FINE_LIMIT);
            let in_window = config.in_window(Mv::new(mv.x - centre.x, mv.y - centre.y));
            if in_window || tried[..count].contains(&mv) {
                continue;
            }
            tried[count] = mv;
            count += 1;
            self.evaluate(src, mv, &visible, &mut best);
            positions += 1;
        }
        // Mixed motion (the SB's blocks match much better on their own than
        // with one vector, e.g. a crowd): one position per further vector
        // that blocks of this area had recently. Static and uniformly moving
        // SBs never get here.
        if config.mixed_motion && best.0.sad[AT_64] > MIXED_MOTION_FACTOR * best.me_dist()[0].max(1)
        {
            effort.mixed_sbs += 1;
            let mut more = [Mv::ZERO; 20];
            let mut extra = 0;
            let mixed = mixed_candidates(
                self.inputs.last_16x16,
                self.fields,
                self.sb_col,
                self.sb_row,
            );
            for mv in mixed.into_iter().flatten() {
                let mv = mv.clamped(FINE_LIMIT);
                let in_window = config.in_window(Mv::new(mv.x - centre.x, mv.y - centre.y));
                if in_window || tried[..count].contains(&mv) || more[..extra].contains(&mv) {
                    continue;
                }
                more[extra] = mv;
                extra += 1;
                self.evaluate(src, mv, &visible, &mut best);
                positions += 1;
            }
        }
        effort.full_pel.add_block(positions);
        best
    }
}

/// Candidate vectors for an SB with mixed motion: last frame's 16x16 vectors
/// of this SB and this frame's 32x32 vectors next to it (left SB's right
/// column, top SB's bottom row).
fn mixed_candidates(
    last_16x16: Option<&AggregatedField>,
    fields: &Fields,
    sb_col: usize,
    sb_row: usize,
) -> [Option<Mv>; 20] {
    let mut out = [None; 20];
    if let Some(f) = last_16x16 {
        for (i, slot) in out[..16].iter_mut().enumerate() {
            let (c, r) = (4 * sb_col + i % 4, 4 * sb_row + i / 4);
            *slot = f.blocks.try_get(c as isize, r as isize).map(|b| b.mv);
        }
    }
    let (c, r) = (2 * sb_col as isize, 2 * sb_row as isize);
    let neighbours = [(c - 1, r), (c - 1, r + 1), (c, r - 1), (c + 1, r - 1)];
    for (slot, (nc, nr)) in out[16..].iter_mut().zip(neighbours) {
        *slot = fields.agg[1].try_get(nc, nr).map(|b| b.mv);
    }
    out
}

/// Largest quarter-resolution vector component the HME may reach (128 luma
/// samples, 10% of the picture width per frame at 720p). A seeded window
/// centre can be at most `HME_LIMIT - HME_RANGE_X` from zero, so this also
/// bounds how far steady motion can be followed. It sets the frame and
/// pyramid borders, which every frame pays to fill, so it is no larger than
/// live camera motion needs.
pub const HME_LIMIT: i16 = 32;

fn to_quarter(mv: Mv) -> Mv {
    Mv::new(mv.x / HME_SCALE, mv.y / HME_SCALE)
}

/// The frame's 8x8 field and aggregated fields, filled SB by SB.
struct Fields {
    b8: Grid<MotionBlock>,
    agg: [Grid<AggregatedBlock>; 3],
}

impl Fields {
    fn new(width: usize, height: usize) -> Self {
        let grid = |size: usize| {
            Grid::filled(
                width.div_ceil(size),
                height.div_ceil(size),
                AggregatedBlock::default(),
            )
        };
        Self {
            b8: Grid::filled(
                width.div_ceil(BLOCK),
                height.div_ceil(BLOCK),
                MotionBlock::default(),
            ),
            agg: [grid(16), grid(32), grid(64)],
        }
    }

    /// Copy one SB's bests into the frame fields (blocks outside the
    /// picture are dropped).
    fn store(&mut self, sb_col: usize, sb_row: usize, best: &SbBest) {
        // Whole rows of each block size at a time: one bounds check per row
        // rather than per block (~20K blocks per 720p frame).
        for r in 0..BLOCKS_PER_SB {
            store_row(&mut self.b8, (sb_col * 8, sb_row * 8 + r), 8, |c| {
                best.block(r * 8 + c)
            });
        }
        let split = |kids: [usize; 4]| kids.iter().map(|&k| best.0.sad[k]).sum();
        for r in 0..4 {
            store_row(&mut self.agg[0], (sb_col * 4, sb_row * 4 + r), 4, |c| {
                let kids = [0, 1, 8, 9].map(|k| (2 * r) * 8 + 2 * c + k);
                aggregated(best.block(AT_16 + r * 4 + c), split(kids))
            });
        }
        for r in 0..2 {
            store_row(&mut self.agg[1], (sb_col * 2, sb_row * 2 + r), 2, |c| {
                let kids = [0, 1, 4, 5].map(|k| AT_16 + (2 * r) * 4 + 2 * c + k);
                aggregated(best.block(AT_32 + r * 2 + c), split(kids))
            });
        }
        let kids = [0, 1, 2, 3].map(|k| AT_32 + k);
        set(
            &mut self.agg[2],
            sb_col,
            sb_row,
            aggregated(best.block(AT_64), split(kids)),
        );
    }

    /// The 64x64 vectors of the left, top and top-right SBs, where stored.
    fn neighbour_vectors(&self, sb_col: usize, sb_row: usize) -> [Option<Mv>; 3] {
        let (c, r) = (sb_col as isize, sb_row as isize);
        [(c - 1, r), (c, r - 1), (c + 1, r - 1)]
            .map(|(nc, nr)| self.agg[2].try_get(nc, nr).map(|b| b.mv))
    }

    fn finish(self) -> (Grid<MotionBlock>, [AggregatedField; 3]) {
        let [a16, a32, a64] = self.agg;
        (
            self.b8,
            [
                AggregatedField {
                    size: 16,
                    blocks: a16,
                },
                AggregatedField {
                    size: 32,
                    blocks: a32,
                },
                AggregatedField {
                    size: 64,
                    blocks: a64,
                },
            ],
        )
    }
}

fn aggregated(block: MotionBlock, split_sad: u32) -> AggregatedBlock {
    AggregatedBlock {
        mv: block.mv,
        whole_sad: block.sad,
        split_sad,
    }
}

/// Set `count` cells of `row` from `col` on to `value(i)` for the `i`-th,
/// keeping only those inside the grid.
fn store_row<T>(
    grid: &mut Grid<T>,
    (col, row): (usize, usize),
    count: usize,
    value: impl Fn(usize) -> T,
) {
    if row >= grid.rows() || col >= grid.cols() {
        return;
    }
    for (i, cell) in grid.row_mut(row)[col..].iter_mut().take(count).enumerate() {
        *cell = value(i);
    }
}

fn set<T>(grid: &mut Grid<T>, col: usize, row: usize, value: T) {
    if col < grid.cols() && row < grid.rows() {
        *grid.get_mut(col, row) = value;
    }
}

#[cfg(test)]
mod tests {
    use super::{AT_16, Level, Mv, SB, SbBest};
    use nc_common::Plane;

    /// A gathered block equals what a border wide enough to hold it reads.
    #[test]
    fn gathered_blocks_match_the_extended_border() {
        let (width, height, wide) = (160, 72, 3 * SB);
        let mut plane = Plane::new(width, height, wide).unwrap();
        let mut view = plane.view_mut();
        for y in 0..height {
            for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                *sample = (x * 7 + y * 13) as u8;
            }
        }
        plane.extend_borders();
        let level = Level::new(&plane, &plane);
        let mut block = [0u8; SB * SB];
        for (x, y) in [(0, 0), (64, 0), (128, 64), (96, 32)] {
            for mv in [(-130, -3), (40, 70), (-70, 100), (0, 0), (127, -128)] {
                let mv = Mv::new(mv.0, mv.1);
                level.gathered(
                    (x + isize::from(mv.x), y + isize::from(mv.y)),
                    SB,
                    &mut block,
                );
                let (px, py) = level.position(x, y, mv);
                for (r, row) in block.as_chunks::<SB>().0.iter().enumerate() {
                    let wide_row = &level.reference.row(py + r)[px..px + SB];
                    assert_eq!(row, wide_row, "block at ({x}, {y}) mv {mv:?} row {r}");
                }
            }
        }
    }

    /// The two-pass mean-removed variance the integer form replaced.
    fn two_pass(sads: &[u32]) -> f32 {
        let n = sads.len() as f64;
        let mean = sads.iter().map(|&s| f64::from(s)).sum::<f64>() / n;
        let var = sads
            .iter()
            .map(|&s| (f64::from(s) - mean).powi(2))
            .sum::<f64>()
            / n;
        var as f32
    }

    #[test]
    fn integer_cost_variance_matches_two_pass_bit_for_bit() {
        // Largest 8x8 SAD (64 * 255) and a pseudo-random spread below it.
        const MAX_SAD: u64 = 64 * 255;
        let mut state = 0x9e37_79b9_u64;
        for case in 0..2000 {
            let mut best = SbBest::default();
            for sad in &mut best.0.sad[..AT_16] {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *sad = match case % 4 {
                    0 => ((state >> 33) % (MAX_SAD + 1)) as u32,
                    1 => ((state >> 33) % 64) as u32,
                    2 => {
                        if state >> 63 == 0 {
                            0
                        } else {
                            MAX_SAD as u32
                        }
                    }
                    _ => (MAX_SAD - (state >> 33) % 16) as u32,
                };
            }
            assert_eq!(
                best.cost_variance_8x8().to_bits(),
                two_pass(&best.0.sad[..AT_16]).to_bits(),
                "case {case}"
            );
        }
    }
}
