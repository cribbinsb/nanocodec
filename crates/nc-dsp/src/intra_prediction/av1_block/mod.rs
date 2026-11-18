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

use nc_common::{BitDepth, Plane, PlaneViewMut, Sample};

use super::filter::{FILTER_INTRA_MAX_SIZE, FilterGrid, FilterIntraMode, predict_patches_at};
use super::{
    EDGE_LEN, IntraEdges, IntraMode, MAX_INTRA_SIZE, SMOOTH_SCALE, SMOOTH_SHIFT, SMOOTH_WEIGHTS,
    no_neighbour_values, paeth,
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

/// `AboveRow[-1..w + h]` and `LeftCol[-1..w + h]` of one block, samples of
/// any depth.
#[derive(Debug)]
pub struct BlockEdges<T: Sample = u8> {
    /// The samples, in the square kernels' layout: `above` / `left` hold
    /// `w + h` samples (at most [`EDGE_LEN`]); `size` is the width and
    /// `dc` the DC value, both meaningful to the square kernels only when
    /// the block is square.
    square: IntraEdges<T>,
    /// The samples' depth: the no-neighbour base value and the clips of
    /// upsampling and filter intra.
    depth: BitDepth,
    width: usize,
    height: usize,
    avail: EdgeAvail,
    /// Samples of the row above inside the picture, at most `width`
    /// (`Min(w, maxX - x + 1)` of §7.11.2.4).
    above_inside: usize,
    /// Samples of the left column inside the picture, at most `height`.
    left_inside: usize,
}

impl<T: Sample> BlockEdges<T> {
    /// Read the edges of the `width`x`height` block at `(x, y)` of `plane`
    /// as §7.11.2 forms them: reads beyond the block's own sides only where
    /// `avail` says they are decoded, and every read clamped to the picture
    /// (the plane is the `MiCols` / `MiRows` area, so `maxX` / `maxY` are
    /// its last column and row). `depth` is the plane's bit depth.
    ///
    /// # Panics
    ///
    /// If a side is not 4, 8, 16, 32 or 64, the aspect ratio exceeds 4:1,
    /// or the block's origin lies outside `plane`.
    #[inline]
    #[must_use]
    pub fn read(
        plane: &Plane<T>,
        (x, y): (usize, usize),
        (width, height): (usize, usize),
        avail: EdgeAvail,
        depth: BitDepth,
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
        let (base, above_default, left_default) = no_neighbour_values::<T>(depth);
        let mut square = IntraEdges {
            above: [T::default(); EDGE_LEN],
            left: [T::default(); EDGE_LEN],
            corner: base,
            dc: base,
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
            // Samples x..=limit, then the last of them repeated (§7.11.2:
            // reads clamp to `limit`); copied whole rather than per sample.
            let inside = (limit + 1 - x).min(span);
            square.above[..inside].copy_from_slice(&row[x..x + inside]);
            square.above[inside..span].fill(row[limit]);
        } else if avail.have_left {
            square.above[..span].fill(view.row(y)[x - 1]);
        } else {
            square.above[..span].fill(above_default);
        }
        if avail.have_left {
            let reach = if avail.have_below_left {
                2 * height
            } else {
                height
            };
            let limit = (y + reach - 1).min(max_y);
            // Rows y..=limit of column x - 1, then the last repeated; the
            // column is walked by stride, not row by row.
            let inside = (limit + 1 - y).min(span);
            let stride = view.stride();
            let first = y * stride + (x - 1);
            let col_slice = &view.data()[first..=(first + (inside - 1) * stride)];
            for (r, sample) in square.left[..inside].iter_mut().enumerate() {
                *sample = col_slice[r * stride];
            }
            square.left[inside..span].fill(view.row(limit)[x - 1]);
        } else if avail.have_above {
            square.left[..span].fill(view.row(y - 1)[x]);
        } else {
            square.left[..span].fill(left_default);
        }
        square.corner = match (avail.have_above, avail.have_left) {
            (true, true) => view.row(y - 1)[x - 1],
            (true, false) => view.row(y - 1)[x],
            (false, true) => view.row(y)[x - 1],
            (false, false) => base,
        };
        Self {
            square,
            depth,
            width,
            height,
            avail,
            above_inside: width.min(max_x + 1 - x),
            left_inside: height.min(max_y + 1 - y),
        }
    }

    /// The block's DC prediction value (§7.11.2.5).
    #[must_use]
    pub fn dc_value(&self) -> T {
        self.dc()
    }

    pub(crate) const fn shape(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub(crate) const fn edges(&self) -> (&[T; EDGE_LEN], &[T; EDGE_LEN], T) {
        (&self.square.above, &self.square.left, self.square.corner)
    }

    /// Fill the block with `kind`'s prediction.
    ///
    /// # Panics
    ///
    /// If `dst` is not `width`x`height`, or a directional angle is not
    /// strictly between 0 and 270 degrees.
    pub fn predict(&self, kind: IntraKind, dst: &mut PlaneViewMut<'_, T>) {
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
        square: impl FnOnce(&IntraEdges<T>, Square, &mut PlaneViewMut<'_, T>),
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        self.predict_full(
            square,
            Self::paeth_scalar,
            Self::smooth,
            crate::intra_prediction::directional::interpolate_run,
            kind,
            dst,
        );
    }

    pub(crate) fn paeth_scalar(&self, dst: &mut PlaneViewMut<'_, T>) {
        let (w, h) = (self.width, self.height);
        let (above, left) = (&self.square.above, &self.square.left);
        for (r, &l) in left[..h].iter().enumerate() {
            for (pixel, &a) in dst.row_mut(r)[..w].iter_mut().zip(&above[..w]) {
                *pixel = paeth(l, a, self.square.corner);
            }
        }
    }

    /// [`Self::predict_with`] with custom rectangular Paeth, Smooth, and
    /// directional run kernels.
    pub(crate) fn predict_full(
        &self,
        square: impl FnOnce(&IntraEdges<T>, Square, &mut PlaneViewMut<'_, T>),
        rect_paeth: impl FnOnce(&Self, &mut PlaneViewMut<'_, T>),
        rect_smooth: impl FnOnce(&Self, IntraKind, &mut PlaneViewMut<'_, T>),
        rect_run: impl Fn(&[T], usize, &mut [T]),
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        let (w, h) = (self.width, self.height);
        assert!(
            dst.width() == w && dst.height() == h,
            "intra destination size"
        );
        if w == h {
            match kind {
                IntraKind::Dc => return dst.fill(self.dc()),
                IntraKind::Directional {
                    angle,
                    edge_filter: None,
                } => return square(&self.square, Square::Angle(angle), dst),
                IntraKind::Directional {
                    angle,
                    edge_filter: Some(ctx),
                } => match directional::prepared_square(self, angle, ctx) {
                    directional::PreparedSquare::Unchanged => {
                        return square(&self.square, Square::Angle(angle), dst);
                    }
                    directional::PreparedSquare::Filtered(prepared) => {
                        return square(&prepared, Square::Angle(angle), dst);
                    }
                    directional::PreparedSquare::Upsampled => {}
                },
                _ => return square(&self.square, Square::Mode(square_mode(kind)), dst),
            }
        }
        match kind {
            IntraKind::Dc => dst.fill(self.dc()),
            IntraKind::Paeth => rect_paeth(self, dst),
            IntraKind::Smooth | IntraKind::SmoothVertical | IntraKind::SmoothHorizontal => {
                rect_smooth(self, kind, dst);
            }
            IntraKind::Directional { angle, edge_filter } => {
                directional::predict_with_run(self, angle, edge_filter, dst, rect_run);
            }
        }
    }

    /// Fill the block with filter intra `mode` (§7.11.2.3); the scalar
    /// reference.
    ///
    /// # Panics
    ///
    /// If a side exceeds 32 or `dst` is not the block's size.
    pub fn predict_filter(&self, mode: FilterIntraMode, dst: &mut PlaneViewMut<'_, T>) {
        let depth = self.depth;
        self.predict_filter_with(
            |grid, width, height, mode| predict_patches_at(grid, width, height, mode, depth),
            mode,
            dst,
        );
    }

    /// [`Self::predict_filter`] with `patches` computing the patches.
    pub(crate) fn predict_filter_with(
        &self,
        patches: impl FnOnce(&mut FilterGrid<T>, usize, usize, FilterIntraMode),
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        assert!(self.width <= FILTER_INTRA_MAX_SIZE && self.height <= FILTER_INTRA_MAX_SIZE);
        self.square
            .predict_filter_block(self.width, self.height, patches, mode, dst);
    }

    /// The DC value (§7.11.2.5): the rounded mean of the available edges.
    fn dc(&self) -> T {
        let (w, h) = (self.width, self.height);
        let sum = |edge: &[T]| -> usize { edge.iter().map(|&v| v.to_i32() as usize).sum() };
        let mean = match (self.avail.have_above, self.avail.have_left) {
            (true, true) => {
                let span = w + h;
                let total = sum(&self.square.above[..w]) + sum(&self.square.left[..h]);
                if span.is_power_of_two() {
                    (total + span / 2) >> span.trailing_zeros()
                } else {
                    (total + span / 2) / span
                }
            }
            (true, false) => (sum(&self.square.above[..w]) + w / 2) >> w.trailing_zeros(),
            (false, true) => (sum(&self.square.left[..h]) + h / 2) >> h.trailing_zeros(),
            (false, false) => self.depth.mid() as usize,
        };
        T::from_i32(mean as i32)
    }

    /// The three smooth predictors (§7.11.2.6) with the width's weights
    /// across and the height's down.
    pub(crate) fn smooth(&self, kind: IntraKind, dst: &mut PlaneViewMut<'_, T>) {
        let (w, h) = (self.width, self.height);
        let (above, left) = (&self.square.above, &self.square.left);
        let weights_x = smooth_weights(w);
        let weights_y = smooth_weights(h);
        let (bottom, right): (u32, u32) = (left[h - 1].into(), above[w - 1].into());
        for (r, &wy) in weights_y.iter().enumerate() {
            let wy = u32::from(wy);
            let l: u32 = left[r].into();
            for (c, pixel) in dst.row_mut(r)[..w].iter_mut().enumerate() {
                let wx = u32::from(weights_x[c]);
                let top: u32 = above[c].into();
                let vertical = wy * top + (SMOOTH_SCALE - wy) * bottom;
                let horizontal = wx * l + (SMOOTH_SCALE - wx) * right;
                *pixel = T::from_i32(match kind {
                    IntraKind::Smooth => {
                        (vertical + horizontal + SMOOTH_SCALE) >> (SMOOTH_SHIFT + 1)
                    }
                    IntraKind::SmoothVertical => (vertical + SMOOTH_SCALE / 2) >> SMOOTH_SHIFT,
                    _ => (horizontal + SMOOTH_SCALE / 2) >> SMOOTH_SHIFT,
                } as i32);
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
