//! AV1 intra prediction of any transform-block shape (AV1 §7.11.2), for the
//! decoder: rectangular DC, smooth and Paeth, directional prediction with
//! the intra edge filter and edge upsampling (§7.11.2.4, .9–.12), and
//! chroma from luma at every size ([`cfl`], §7.11.5), and filter intra of
//! every shape up to 32x32 (§7.11.2.3; [`BlockEdges::predict_filter`],
//! with SIMD patches through `Dsp::av1_intra_predict_filter_block`).
//!
//! [`BlockEdges::read`] forms `AboveRow` and `LeftCol` from the picture and
//! the availability the block layer derives (§7.11.2); [`BlockEdges::predict`]
//! fills the block. The edges are stored in the encoder's square
//! [`IntraEdges`], so a square block whose prediction the square kernels
//! cover (every non-directional mode, and directional modes when
//! `enable_intra_edge_filter` is 0, as in our own streams) runs exactly those
//! kernels on it, without a copy. Every other case runs the scalar code here.
//!
//! Edge processing, zones and the rectangular predictors follow libaom
//! `av1/common/reconintra.c` and `aom_dsp/intrapred.c` (BSD-2-Clause,
//! Copyright (c) 2016, Alliance for Open Media; see `THIRD_PARTY.md`).

pub mod cfl;
mod directional;
mod edge;

use nc_common::{Plane, PlaneViewMut};

use super::filter::{FILTER_INTRA_MAX_SIZE, FilterGrid, FilterIntraMode, predict_patches};
use super::{
    DC_NO_NEIGHBOURS, EDGE_LEN, IntraEdges, IntraMode, MAX_INTRA_SIZE, SMOOTH_SCALE, SMOOTH_SHIFT,
    SMOOTH_WEIGHTS, paeth,
};

/// Which neighbours of the block are decoded (§7.11.2 `haveAbove`,
/// `haveLeft`, `haveAboveRt`, `haveBelowLft`), as the block layer derives
/// them from tile edges and decoding order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // The spec's four independent flags.
pub struct EdgeAvail {
    /// The row above the block is available.
    pub have_above: bool,
    /// The column left of the block is available.
    pub have_left: bool,
    /// The row above continues right of the block for another `width`.
    pub have_above_right: bool,
    /// The column left continues below the block for another `height`.
    pub have_below_left: bool,
}

/// The intra edge filter's context (§7.11.2.4): whether the block above or
/// to the left uses a smooth mode (`get_filter_type`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeFilterCtx {
    /// The above neighbour block is smooth-predicted.
    pub smooth_above: bool,
    /// The left neighbour block is smooth-predicted.
    pub smooth_left: bool,
}

impl EdgeFilterCtx {
    /// `filterType` of §7.11.2.9 / .10.
    const fn smooth(self) -> bool {
        self.smooth_above || self.smooth_left
    }
}

/// An intra prediction of one transform block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraKind {
    /// Mean of the available edges (§7.11.2.5).
    Dc,
    /// Smooth in both directions (§7.11.2.6).
    Smooth,
    /// Smooth down each column.
    SmoothVertical,
    /// Smooth across each row.
    SmoothHorizontal,
    /// Paeth (§7.11.2.2).
    Paeth,
    /// Directional at `angle` degrees (`pAngle`, §7.11.2.4), including
    /// `V_PRED` (90) and `H_PRED` (180) with their angle deltas.
    /// `edge_filter` is `None` when `enable_intra_edge_filter` is 0.
    Directional {
        /// `pAngle`: the nominal angle plus `3 * AngleDelta`.
        angle: u16,
        /// The edge filter context, if the sequence enables the filter.
        edge_filter: Option<EdgeFilterCtx>,
    },
}

/// Largest coded side of an intra transform block.
const MAX_SIDE: usize = MAX_INTRA_SIZE;
/// Longest edge that is upsampled (§7.11.2.10 allows `w + h <= 16`).
const MAX_UPSAMPLE: usize = 16;
/// `(1 << (BitDepth - 1)) - 1`: `AboveRow` when nothing is available.
const ABOVE_DEFAULT: u8 = DC_NO_NEIGHBOURS - 1;
/// `(1 << (BitDepth - 1)) + 1`: `LeftCol` when nothing is available.
const LEFT_DEFAULT: u8 = DC_NO_NEIGHBOURS + 1;

/// `AboveRow[-1..w + h]` and `LeftCol[-1..w + h]` of one block.
#[derive(Debug)]
pub struct BlockEdges {
    /// The samples, in the square kernels' layout: `above` / `left` hold
    /// `w + h` samples (at most [`EDGE_LEN`]); `size` is the width and
    /// `dc` the DC value, both meaningful to the square kernels only when
    /// the block is square.
    square: IntraEdges,
    width: usize,
    height: usize,
    avail: EdgeAvail,
    /// Samples of the row above inside the picture, at most `width`
    /// (`Min(w, maxX - x + 1)` of §7.11.2.4).
    above_inside: usize,
    /// Samples of the left column inside the picture, at most `height`.
    left_inside: usize,
}

impl BlockEdges {
    /// Read the edges of the `width`x`height` block at `(x, y)` of `plane`
    /// as §7.11.2 forms them: reads beyond the block's own sides only where
    /// `avail` says they are decoded, and every read clamped to the picture
    /// (the plane is the `MiCols` / `MiRows` area, so `maxX` / `maxY` are
    /// its last column and row).
    ///
    /// # Panics
    ///
    /// If a side is not 4, 8, 16, 32 or 64, the aspect ratio exceeds 4:1,
    /// or the block's origin lies outside `plane`.
    #[must_use]
    pub fn read(
        plane: &Plane,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        avail: EdgeAvail,
    ) -> Self {
        assert!(
            is_side(width) && is_side(height),
            "intra block {width}x{height}"
        );
        assert!(
            width <= 4 * height && height <= 4 * width,
            "intra block {width}x{height}"
        );
        let view = plane.view();
        assert!(
            x < view.width() && y < view.height(),
            "intra block origin outside the plane"
        );
        let (max_x, max_y) = (view.width() - 1, view.height() - 1);
        let span = width + height;
        let mut square = IntraEdges {
            above: [ABOVE_DEFAULT; EDGE_LEN],
            left: [LEFT_DEFAULT; EDGE_LEN],
            corner: DC_NO_NEIGHBOURS,
            dc: DC_NO_NEIGHBOURS,
            size: width,
        };
        if avail.have_above {
            let row = view.row(y - 1);
            let reach = if avail.have_above_right {
                2 * width
            } else {
                width
            };
            let limit = (x + reach - 1).min(max_x);
            for (i, sample) in square.above[..span].iter_mut().enumerate() {
                *sample = row[(x + i).min(limit)];
            }
        } else if avail.have_left {
            square.above[..span].fill(view.row(y)[x - 1]);
        }
        if avail.have_left {
            let reach = if avail.have_below_left {
                2 * height
            } else {
                height
            };
            let limit = (y + reach - 1).min(max_y);
            for (i, sample) in square.left[..span].iter_mut().enumerate() {
                *sample = view.row((y + i).min(limit))[x - 1];
            }
        } else if avail.have_above {
            square.left[..span].fill(view.row(y - 1)[x]);
        }
        square.corner = match (avail.have_above, avail.have_left) {
            (true, true) => view.row(y - 1)[x - 1],
            (true, false) => view.row(y - 1)[x],
            (false, true) => view.row(y)[x - 1],
            (false, false) => DC_NO_NEIGHBOURS,
        };
        let mut edges = Self {
            square,
            width,
            height,
            avail,
            above_inside: width.min(max_x + 1 - x),
            left_inside: height.min(max_y + 1 - y),
        };
        if width == height {
            edges.square.dc = edges.dc();
        }
        edges
    }

    /// Fill the block with `kind`'s prediction.
    ///
    /// # Panics
    ///
    /// If `dst` is not `width`x`height`, or a directional angle is not
    /// strictly between 0 and 270 degrees.
    pub fn predict(&self, kind: IntraKind, dst: &mut PlaneViewMut<'_>) {
        self.predict_with(
            |square, prediction, dst| match prediction {
                Square::Angle(angle) => square.predict_angle(angle, dst),
                Square::Mode(mode) => square.predict(mode, dst),
            },
            kind,
            dst,
        );
    }

    /// [`Self::predict`] with `square` predicting square blocks that need
    /// no edge filter (`IntraEdges::predict_angle`'s and
    /// `IntraEdges::predict`'s contracts).
    pub(crate) fn predict_with(
        &self,
        square: impl FnOnce(&IntraEdges, Square, &mut PlaneViewMut<'_>),
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let (w, h) = (self.width, self.height);
        assert!(
            dst.width() == w && dst.height() == h,
            "intra destination size"
        );
        if w == h {
            match kind {
                IntraKind::Directional {
                    angle,
                    edge_filter: None,
                } => return square(&self.square, Square::Angle(angle), dst),
                IntraKind::Directional { .. } => {}
                _ => return square(&self.square, Square::Mode(square_mode(kind)), dst),
            }
        }
        match kind {
            IntraKind::Dc => dst.fill(self.dc()),
            IntraKind::Paeth => {
                let (above, left) = (&self.square.above, &self.square.left);
                for (r, &l) in left[..h].iter().enumerate() {
                    for (pixel, &a) in dst.row_mut(r)[..w].iter_mut().zip(&above[..w]) {
                        *pixel = paeth(l, a, self.square.corner);
                    }
                }
            }
            IntraKind::Smooth | IntraKind::SmoothVertical | IntraKind::SmoothHorizontal => {
                self.smooth(kind, dst);
            }
            IntraKind::Directional { angle, edge_filter } => {
                directional::predict(self, angle, edge_filter, dst);
            }
        }
    }

    /// Fill the block with filter intra `mode` (§7.11.2.3); the scalar
    /// reference.
    ///
    /// # Panics
    ///
    /// If a side exceeds 32 or `dst` is not the block's size.
    pub fn predict_filter(&self, mode: FilterIntraMode, dst: &mut PlaneViewMut<'_>) {
        self.predict_filter_with(predict_patches, mode, dst);
    }

    /// [`Self::predict_filter`] with `patches` computing the patches.
    pub(crate) fn predict_filter_with(
        &self,
        patches: impl FnOnce(&mut FilterGrid, usize, usize, FilterIntraMode),
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(self.width <= FILTER_INTRA_MAX_SIZE && self.height <= FILTER_INTRA_MAX_SIZE);
        self.square
            .predict_filter_block(self.width, self.height, patches, mode, dst);
    }

    /// The DC value (§7.11.2.5): the rounded mean of the available edges.
    fn dc(&self) -> u8 {
        let (w, h) = (self.width, self.height);
        let sum = |edge: &[u8]| -> usize { edge.iter().map(|&v| usize::from(v)).sum() };
        let mean = match (self.avail.have_above, self.avail.have_left) {
            (true, true) => {
                let span = w + h;
                let total = sum(&self.square.above[..w]) + sum(&self.square.left[..h]);
                (total + span / 2) / span
            }
            (true, false) => (sum(&self.square.above[..w]) + w / 2) >> w.trailing_zeros(),
            (false, true) => (sum(&self.square.left[..h]) + h / 2) >> h.trailing_zeros(),
            (false, false) => usize::from(DC_NO_NEIGHBOURS),
        };
        mean as u8
    }

    /// The three smooth predictors (§7.11.2.6) with the width's weights
    /// across and the height's down.
    fn smooth(&self, kind: IntraKind, dst: &mut PlaneViewMut<'_>) {
        let (w, h) = (self.width, self.height);
        let (above, left) = (&self.square.above, &self.square.left);
        let weights_x = smooth_weights(w);
        let weights_y = smooth_weights(h);
        let (bottom, right) = (u32::from(left[h - 1]), u32::from(above[w - 1]));
        for (r, &wy) in weights_y.iter().enumerate() {
            let wy = u32::from(wy);
            let l = u32::from(left[r]);
            for (c, pixel) in dst.row_mut(r)[..w].iter_mut().enumerate() {
                let wx = u32::from(weights_x[c]);
                let vertical = wy * u32::from(above[c]) + (SMOOTH_SCALE - wy) * bottom;
                let horizontal = wx * l + (SMOOTH_SCALE - wx) * right;
                *pixel = match kind {
                    IntraKind::Smooth => {
                        (vertical + horizontal + SMOOTH_SCALE) >> (SMOOTH_SHIFT + 1)
                    }
                    IntraKind::SmoothVertical => (vertical + SMOOTH_SCALE / 2) >> SMOOTH_SHIFT,
                    _ => (horizontal + SMOOTH_SCALE / 2) >> SMOOTH_SHIFT,
                } as u8;
            }
        }
    }
}

/// The square kernels' mode for a non-directional kind.
/// A square prediction [`BlockEdges::predict_with`] hands out.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Square {
    /// Directional at this angle, no edge filter.
    Angle(u16),
    /// A non-directional mode (or vertical / horizontal).
    Mode(IntraMode),
}

fn square_mode(kind: IntraKind) -> IntraMode {
    match kind {
        IntraKind::Dc => IntraMode::Dc,
        IntraKind::Smooth => IntraMode::Smooth,
        IntraKind::SmoothVertical => IntraMode::SmoothVertical,
        IntraKind::SmoothHorizontal => IntraMode::SmoothHorizontal,
        IntraKind::Paeth => IntraMode::Paeth,
        IntraKind::Directional { .. } => unreachable!("directional kinds take an angle"),
    }
}

/// The smooth weights of a side of `n` samples (§7.11.2.6
/// `Sm_Weights_Tx_*`).
fn smooth_weights(n: usize) -> &'static [u8] {
    &SMOOTH_WEIGHTS[n - 4..2 * n - 4]
}

const fn is_side(n: usize) -> bool {
    n.is_power_of_two() && n >= 4 && n <= MAX_SIDE
}
