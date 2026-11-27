//! Every AV2 intra predictor of the general decoder, for 4:2:0: DC
//! (with its top-only, left-only and flat forms), vertical, horizontal,
//! Paeth, the three smooth predictors, directional prediction in its three
//! zones with edge filtering, multiple reference lines and intra
//! bi-prediction, data-driven intra prediction (DIP), chroma from luma
//! (explicit, implicit and multi-hypothesis MHCCP), palette, and the edge
//! preparation that picks the predictor and fills its edge buffer.
//!
//! Every function is generic over the sample type `T` (`u8` for 8-bit
//! streams, `u16` deeper) and takes the stream's [`BitDepth`] where dav2d's
//! 16bpc template reads `bitdepth_max` (clips, mid-grey fills, the `CfL` and
//! MHCCP scales); see `docs/handoff/high-bit-depth.md`. The `u8` instance
//! folds the depth to the constant 8 ([`BitDepth::of`]).
//!
//! Ported from dav2d (`src/ipred_tmpl.c`, `src/ipred_prepare_tmpl.c`,
//! `src/ipred.h`, `src/ipred_prepare.h`, `src/intra_edge.h`, `src/ibp.c`,
//! `src/derivation.h` and the tables of `src/tables.c`, commit 237229e;
//! Copyright © 2018-2026 `VideoLAN` and dav2d authors and Two Orioles, LLC,
//! BSD-2-Clause, see `docs/licenses/dav2d`), which is bit-exact with AVM
//! v1.0.0. The functions keep dav2d's contracts, numbering and flag bits so
//! the decoder can pass dav2d's values unchanged; a C `pixel *` with a stride
//! becomes a slice, an offset and a stride.
//!
//! # The edge buffer
//!
//! [`prepare_intra_edges`] fills, and [`predict`] reads, an edge buffer
//! `edge` with the top-left sample at index `topleft` (dav2d's `topleft`
//! pointer). Offsets below are relative to `topleft`; `w` and `h` are the
//! transform's width and height in samples, `m` the reference line
//! (`mrl_idx`, 0..=3; nonzero only for directional modes):
//!
//! - the buffer holds reference line `m` (the row `m + 1` above the block and
//!   the column `m + 1` to its left) unrolled around its corner `[0]`, the
//!   sample `m + 1` above and left of the block's first sample;
//! - `[1..=m]` continue along the row towards the block's first column and
//!   `[-1..=-m]` down the column towards its first row;
//! - `m + 1 ..`: the top row of line `m` above the block, left to right, `w`
//!   samples then the top-right extension (`h + 2m` for directional modes,
//!   `w / 4` for DIP, one for smooth);
//! - `-(m + 1)` downwards: the left column of line `m`, top to bottom, `h`
//!   samples then the bottom-left extension (`w + 2m`, `h / 4` or one);
//! - with [`ANGLE_MULTI_MRL_FLAG`], line 0 follows at `e = 2 (w + h) + 3m + 1`:
//!   its corner at `e`, its top row from `e + 1` and its left column from
//!   `e - 1` downwards (`w + h` samples each).
//!
//! The buffer needs [`EDGE_TOPLEFT`] samples before the corner and
//! [`EDGE_LEN`] in all, as dav2d's `scratch.edge`.

use nc_common::{BitDepth, Sample};

mod cfl;
mod dc;
mod dip;
mod directional;
mod ibp;
mod mhccp;
mod palette;
mod plain;
mod prepare;
mod smooth;
mod tables;
#[cfg(test)]
mod tests;

pub use cfl::{CflChroma, cfl_pred_420, derive_alpha};
pub use mhccp::{
    CFL_MH_MAX_EDGE_SAMPLES, CFL_MH_MAX_LUMA_SIZE, CFL_MH_TOP_STRIDE, cfl_mh_calc_alphas,
    cfl_mh_gen_mat, cfl_mh_gen_y_420, cfl_mh_pred,
};
pub use palette::pal_pred;
pub use prepare::{EdgeRequest, prepare_intra_edges};

/// Angle bits of the `angle` argument; the flags below sit above them.
pub const ANGLE_MASK: i32 = 511;
/// The sequence header enables intra bi-prediction (gates the corner filter).
pub const ANGLE_SEQHDR_IBP_FLAG: i32 = 1 << 20;
/// Luma: directional prediction uses the 4-tap filter, chroma the 2-tap one.
pub const ANGLE_IS_LUMA: i32 = 1 << 19;
/// Data-driven intra prediction replaces DC (the angle bits carry its mode).
pub const ANGLE_DIP_FLAG: i32 = 1 << 18;
/// The top edge exists (inside the tile).
pub const ANGLE_HAS_TOP_FLAG: i32 = 1 << 17;
/// The left edge exists (inside the tile).
pub const ANGLE_HAS_LEFT_FLAG: i32 = 1 << 16;
/// Average the prediction from line `mrl_idx` with that from line 0.
pub const ANGLE_MULTI_MRL_FLAG: i32 = 1 << 15;
/// Shift of the reference line index (`mrl_idx`, 0..=3).
pub const ANGLE_MRL_IDX_SHIFT: u32 = 13;
/// Bits of the reference line index.
pub const ANGLE_MRL_IDX_MASK: i32 = 3 << ANGLE_MRL_IDX_SHIFT;
/// Intra bi-prediction for this block.
pub const ANGLE_IBP_FLAG: i32 = 1 << 12;
/// The sequence header enables the intra edge filter.
pub const ANGLE_USE_EDGE_FILTER_FLAG: i32 = 1 << 11;
/// The top neighbour is smooth-predicted: stronger edge filtering.
pub const ANGLE_SMOOTH_TOP_EDGE_FLAG: i32 = 1 << 10;
/// The left neighbour is smooth-predicted: stronger edge filtering.
pub const ANGLE_SMOOTH_LEFT_EDGE_FLAG: i32 = 1 << 9;

/// `CfL` luma downsampling filter: 2x2 mean.
pub const CFL_FLT_TYPE_UNIFORM: u32 = 0;
/// `CfL` luma downsampling filter: 3x2 with a heavier centre column.
pub const CFL_FLT_TYPE_VSTRIP: u32 = 1;
/// `CfL` luma downsampling filter: five-tap cross.
pub const CFL_FLT_TYPE_GAUSS: u32 = 2;
/// Bits of the downsampling filter in the `CfL` flags.
pub const CFL_FLT_TYPE: u32 = 0x3;
/// `CfL`: the top edge exists.
pub const CFL_HAS_TOP: u32 = 1 << 2;
/// `CfL`: the left edge exists.
pub const CFL_HAS_LEFT: u32 = 1 << 3;
/// `CfL`: the block is at a superblock's top, so the row above comes from the
/// pre-filter line buffer (one row only).
pub const CFL_IS_TOP_SB_EDGE: u32 = 1 << 4;
/// Bits of each explicit `CfL` alpha.
pub const CFL_ALPHA_LOG2: u32 = 5;
/// Shift of the U alpha in the explicit `CfL` flags.
pub const CFL_ALPHA_U_SHIFT: u32 = 16 - CFL_ALPHA_LOG2;
/// Shift of the V alpha in the explicit `CfL` flags.
pub const CFL_ALPHA_V_SHIFT: u32 = 32 - CFL_ALPHA_LOG2;
/// Bits of the U alpha.
pub const CFL_ALPHA_U_MASK: u32 = ((1 << CFL_ALPHA_LOG2) - 1) << CFL_ALPHA_U_SHIFT;
/// Bits of the V alpha.
pub const CFL_ALPHA_V_MASK: u32 = ((1 << CFL_ALPHA_LOG2) - 1) << CFL_ALPHA_V_SHIFT;
/// Bits of the MHCCP direction ([`CflMhDir`]) in the MHCCP flags.
pub const CFL_DIR_MASK: u32 = 3;

/// Samples in front of the corner in the edge buffer (dav2d's
/// `scratch.edge + 128 + 9`): the longest left column with its bottom-left
/// extension on reference line 3.
pub const EDGE_TOPLEFT: usize = 128 + 9;
/// Length of the edge buffer (dav2d's `scratch.edge`).
pub const EDGE_LEN: usize = 64 * 8 + 2 + 2 * 9;
/// Stride of the scratch predictions that multi-line and bi-prediction blend.
const TMP_STRIDE: usize = 64;
/// Size of one scratch prediction.
const TMP_LEN: usize = TMP_STRIDE * TMP_STRIDE;

/// Intra modes as coded (dav2d `enum IntraPredMode`, the bitstream order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(missing_docs, reason = "the AV2 mode names")]
pub enum IntraPredMode {
    Dc = 0,
    Vert = 1,
    Hor = 2,
    DiagDownLeft = 3,
    DiagDownRight = 4,
    VertRight = 5,
    HorDown = 6,
    HorUp = 7,
    VertLeft = 8,
    Smooth = 9,
    SmoothV = 10,
    SmoothH = 11,
    Paeth = 12,
    /// Chroma from luma (`CFL_PRED`, chroma only).
    Cfl = 13,
}

/// Coded luma modes (`N_INTRA_PRED_MODES`).
pub const N_INTRA_PRED_MODES: usize = 13;
/// Predictor implementations (`N_IMPL_INTRA_PRED_MODES`).
pub const N_IMPL_INTRA_PRED_MODES: usize = 14;

impl IntraPredMode {
    /// The mode with dav2d's number `index`.
    #[must_use]
    pub fn from_index(index: u8) -> Option<Self> {
        use IntraPredMode::{
            Cfl, Dc, DiagDownLeft, DiagDownRight, Hor, HorDown, HorUp, Paeth, Smooth, SmoothH,
            SmoothV, Vert, VertLeft, VertRight,
        };
        const ALL: [IntraPredMode; 14] = [
            Dc,
            Vert,
            Hor,
            DiagDownLeft,
            DiagDownRight,
            VertRight,
            HorDown,
            HorUp,
            VertLeft,
            Smooth,
            SmoothV,
            SmoothH,
            Paeth,
            Cfl,
        ];
        ALL.get(usize::from(index)).copied()
    }
}

/// Predictor implementations, the index of dav2d's `intra_pred[]` table
/// (`enum IntraPredMode` values `LEFT_DC_PRED`, `Z1_PRED`, `DIP_PRED`... reuse
/// the coded modes' numbers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ImplMode {
    /// Mean of the top and left edges.
    Dc = 0,
    /// Copy of the top edge.
    Vert = 1,
    /// Copy of the left edge.
    Hor = 2,
    /// Mean of the left edge.
    LeftDc = 3,
    /// Mean of the top edge.
    TopDc = 4,
    /// Mid-grey (no edges).
    Dc128 = 5,
    /// Directional, angle below 90 degrees (top and top-right).
    Z1 = 6,
    /// Directional, angle between 90 and 180 degrees (top and left).
    Z2 = 7,
    /// Directional, angle above 180 degrees (left and bottom-left).
    Z3 = 8,
    /// Smooth in both directions.
    Smooth = 9,
    /// Smooth vertically.
    SmoothV = 10,
    /// Smooth horizontally.
    SmoothH = 11,
    /// Paeth.
    Paeth = 12,
    /// Data-driven intra prediction.
    Dip = 13,
}

impl ImplMode {
    /// The predictor with dav2d's number `index`.
    #[must_use]
    pub fn from_index(index: u8) -> Option<Self> {
        use ImplMode::{
            Dc, Dc128, Dip, Hor, LeftDc, Paeth, Smooth, SmoothH, SmoothV, TopDc, Vert, Z1, Z2, Z3,
        };
        const ALL: [ImplMode; N_IMPL_INTRA_PRED_MODES] = [
            Dc, Vert, Hor, LeftDc, TopDc, Dc128, Z1, Z2, Z3, Smooth, SmoothV, SmoothH, Paeth, Dip,
        ];
        ALL.get(usize::from(index)).copied()
    }
}

/// Chroma-from-luma kinds (dav2d `enum CflType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CflType {
    /// Coded alphas ([`cfl_pred_420`] with `implicit` false).
    Explicit = 0,
    /// Alphas fitted to the edges ([`cfl_pred_420`] with `implicit` true).
    Implicit = 1,
    /// Multi-hypothesis cross-component prediction ([`cfl_mh_pred`]).
    Mhccp = 2,
}

/// MHCCP neighbour direction (dav2d `enum CflMhDir`), the low bits of the
/// MHCCP flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CflMhDir {
    /// Each luma sample is paired with itself.
    Center = 0,
    /// Each luma sample is paired with the one above.
    Top = 1,
    /// Each luma sample is paired with the one to its left.
    Left = 2,
}

impl CflMhDir {
    /// The direction in `flags`' [`CFL_DIR_MASK`] bits.
    fn from_flags(flags: u32) -> Self {
        match flags & CFL_DIR_MASK {
            1 => Self::Top,
            2 => Self::Left,
            _ => Self::Center,
        }
    }
}

/// A read position in a sample buffer: dav2d's `const pixel *`, which edge
/// and neighbour reads index at negative offsets too.
#[derive(Debug, Clone, Copy)]
pub struct SampleRef<'a, T = u8> {
    /// The buffer.
    pub data: &'a [T],
    /// Index of the position in `data`.
    pub origin: usize,
}

impl<'a, T: Sample> SampleRef<'a, T> {
    /// The position `origin` in `data`.
    #[must_use]
    pub fn new(data: &'a [T], origin: usize) -> Self {
        Self { data, origin }
    }

    /// The sample `offset` away.
    fn at(self, offset: isize) -> T {
        self.data[index(self.origin, offset)]
    }

    /// The sample `offset` away, widened for arithmetic.
    fn wide(self, offset: isize) -> i32 {
        self.at(offset).to_i32()
    }

    /// The position `offset` away.
    fn shifted(self, offset: isize) -> Self {
        Self::new(self.data, index(self.origin, offset))
    }

    /// The `len` samples from `offset` on.
    fn slice(self, offset: isize, len: usize) -> &'a [T] {
        let start = index(self.origin, offset);
        &self.data[start..start + len]
    }
}

/// `origin + offset`, which must not fall before the buffer.
fn index(origin: usize, offset: isize) -> usize {
    origin
        .checked_add_signed(offset)
        .expect("sample offset before the start of its buffer")
}

/// A destination block: dav2d's `pixel *dst` with its stride.
#[derive(Debug)]
struct Block<'a, T> {
    buf: &'a mut [T],
    offset: usize,
    stride: usize,
}

impl<'a, T: Sample> Block<'a, T> {
    fn new(buf: &'a mut [T], offset: usize, stride: usize) -> Self {
        Self {
            buf,
            offset,
            stride,
        }
    }

    /// The first `width` samples of row `y`.
    fn row(&mut self, y: usize, width: usize) -> &mut [T] {
        let start = self.offset + y * self.stride;
        &mut self.buf[start..start + width]
    }

    /// The block `rows` rows down and `cols` columns right.
    fn sub(&mut self, rows: usize, cols: usize) -> Block<'_, T> {
        Block::new(
            self.buf,
            self.offset + rows * self.stride + cols,
            self.stride,
        )
    }
}

/// Fills `width` x `height` of `dst` with `value` (`splat_dc`).
fn fill<T: Sample>(dst: &mut Block<'_, T>, width: usize, height: usize, value: T) {
    for y in 0..height {
        dst.row(y, width).fill(value);
    }
}

/// Clamps to the sample range of `depth` (`iclip_pixel`). The depth is
/// folded here, not only at the entry points: the predictors that are not
/// inlined into them would otherwise clip to a runtime bound at `u8`.
fn clip_pixel<T: Sample>(v: i32, depth: BitDepth) -> T {
    depth.of::<T>().clip_to(v)
}

/// The reference line index in an angle argument.
fn mrl_idx(angle: i32) -> usize {
    ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize
}

/// Predicts `width` x `height` samples into `dst` (at `offset`, rows `stride`
/// apart) with predictor `mode` from the edge buffer `edge` prepared by
/// [`prepare_intra_edges`] (corner at `topleft`), as dav2d's
/// `intra_pred[mode]`.
///
/// `angle` is the directional angle in degrees (or DIP's mode for
/// [`ImplMode::Dip`]) OR'ed with the `ANGLE_*` flags; `max_width` and
/// `max_height` are the samples from the block to the picture's right and
/// bottom (they limit the edge filter). `depth` is the samples' bit depth.
pub fn predict<T: Sample>(
    mode: ImplMode,
    dst: &mut [T],
    offset: usize,
    stride: usize,
    edge: &[T],
    topleft: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    depth: BitDepth,
) {
    let depth = depth.of::<T>();
    let mut dst = Block::new(dst, offset, stride);
    let edge = SampleRef::new(edge, topleft);
    let limits = directional::PictureLimits {
        max_width,
        max_height,
    };
    match mode {
        ImplMode::Dc => dc::dc(&mut dst, edge, width, height, angle, depth),
        ImplMode::TopDc => dc::dc_top(&mut dst, edge, width, height, angle),
        ImplMode::LeftDc => dc::dc_left(&mut dst, edge, width, height, angle),
        ImplMode::Dc128 => dc::dc_128(&mut dst, width, height, depth),
        ImplMode::Vert => plain::vertical(&mut dst, edge, width, height, angle),
        ImplMode::Hor => plain::horizontal(&mut dst, edge, width, height, angle),
        ImplMode::Paeth => plain::paeth(&mut dst, edge, width, height),
        ImplMode::Smooth => smooth::smooth(&mut dst, edge, width, height),
        ImplMode::SmoothV => smooth::smooth_v(&mut dst, edge, width, height),
        ImplMode::SmoothH => smooth::smooth_h(&mut dst, edge, width, height),
        ImplMode::Z1 => directional::z1(&mut dst, edge, width, height, angle, limits, depth),
        ImplMode::Z2 => directional::z2(&mut dst, edge, width, height, angle, limits, depth),
        ImplMode::Z3 => directional::z3(&mut dst, edge, width, height, angle, limits, depth),
        ImplMode::Dip => dip::dip(&mut dst, edge, width, height, angle, depth),
    }
}
