//! Source-frame analysis for the shared encoder engine (`docs/ARCHITECTURE.md`
//! §5 "Source analysis"): everything computed from the input pictures alone,
//! before any codec-specific decision.
//!
//! - [`pyramid`]: half/quarter-resolution luma for hierarchical search.
//! - [`preanalysis`]: per-8x8 statistics, static detection.
//! - [`motion`]: SVT-AV1-style motion search per 64x64 superblock
//!   (zero-vector gates, quarter-resolution HME, full-pel window), giving
//!   the best vector of every square block from 8x8 to 64x64.
//! - [`global`]: camera motion (translation + zoom) fitted to the 16x16 field.
//! - [`noise`]: sensor-noise level from flat blocks' best matches.
//! - [`grain`]: film grain noise model (AR filter, strength per intensity)
//!   of the texture coding removed.
//! - [`still`]: persistent stillness (static runs plus a calm
//!   neighbourhood) for the static-content levers.
//! - [`link`]: which earlier source frame a frame is searched against.
//! - [`Analyzer`]: runs all of the above frame by frame.
//!
//! All vectors here are whole-sample and measured against the source of the
//! frame's reference (the previous frame in a flat chain). Codecs refine
//! against their own reconstructions.

#![forbid(unsafe_code)]

mod analyzer;
pub mod effort;
pub mod global;
pub mod grain;
pub mod grid;
pub mod link;
pub mod motion;
pub mod mv;
pub mod noise;
pub mod preanalysis;
pub mod pyramid;
pub mod still;

pub use analyzer::{Analyzer, FrameAnalysis, MotionAnalysis, NoTimer, Stage, StageTimer};
pub use grid::Grid;
pub use link::{FrameLink, ReferenceLink, SlotMask};
pub use mv::Mv;
