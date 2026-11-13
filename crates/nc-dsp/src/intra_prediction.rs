//! Shared DC, axis-aligned, AV1 smooth and Paeth intra prediction, and the
//! edges every AV1 predictor reads. AV1's directional modes at any angle are
//! in [`directional`], its recursive filter intra in [`filter`]; the
//! decoder's predictions of every block shape, with the edge filter,
//! upsampling and chroma from luma, in [`av1_block`]. The AV2
//! kernels (all modes, AVM edge rules) are in [`av2`].
//!
//! Prediction formulas, edge defaults and smooth weights follow libaom
//! `aom_dsp/intrapred.c`, `intrapred_common.h` and `av1/common/reconintra.c`
//! (BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media;
//! see `THIRD_PARTY.md`).

pub mod av1_block;
pub mod av2;
pub mod directional;
pub mod filter;

pub use filter::FilterIntraMode;

use nc_common::{Plane, PlaneViewMut};

/// Base value when no neighbours are available: `1 << (BitDepth - 1)`.
const DC_NO_NEIGHBOURS: u8 = 128;
/// Largest square AV1 intra block in our subset.
const MAX_INTRA_SIZE: usize = 64;
/// Edge samples kept per side: the block's own and as many beyond it
/// (above-right, below-left), which directional prediction reads.
const EDGE_LEN: usize = 2 * MAX_INTRA_SIZE;
/// Smooth weights have eight fractional bits (AV1 §7.11.2).
pub(crate) const SMOOTH_SHIFT: u32 = 8;
const SMOOTH_SCALE: u32 = 1 << SMOOTH_SHIFT;

/// The DC prediction value of the `size`×`size` block at `(x, y)` of
/// `plane`, from the already-reconstructed row above and column to the
/// left. `have_above` / `have_left` say whether those neighbours exist
/// (they do not at the top or left edge of the tile).
#[must_use]
pub fn dc_value(
    plane: &Plane,
    x: usize,
    y: usize,
    size: usize,
    have_above: bool,
    have_left: bool,
) -> u8 {
    let view = plane.view();
    let log2 = size.trailing_zeros();
    let above_sum = || -> u32 {
        view.row(y - 1)[x..x + size]
            .iter()
            .map(|&v| u32::from(v))
            .sum()
    };
    // One slice of the left column's rows and a strided walk: indexing a
    // row per sample cost 1.5% of AV1 key-frame instructions (callgrind,
    // 720p crowd, three key frames: 1.567G -> 1.544G; streams identical).
    let left_sum = || -> u32 {
        let stride = view.stride();
        let column = &view.data()[y * stride + x - 1..(y + size - 1) * stride + x];
        column.iter().step_by(stride).map(|&v| u32::from(v)).sum()
    };
    let half = (size / 2) as u32;
    match (have_above, have_left) {
        // (w + h) = 2 * size, so the spec's division is an exact shift here.
        (true, true) => ((above_sum() + left_sum() + size as u32) >> (log2 + 1)) as u8,
        (true, false) => ((above_sum() + half) >> log2) as u8,
        (false, true) => ((left_sum() + half) >> log2) as u8,
        (false, false) => DC_NO_NEIGHBOURS,
    }
}

/// The supported AV1 luma intra modes, with their standard symbol values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum IntraMode {
    /// Mean of the available above/left edges.
    #[default]
    Dc = 0,
    /// Copy the above edge down each column (90 degrees).
    Vertical = 1,
    /// Copy the left edge across each row (180 degrees).
    Horizontal = 2,
    /// Directional, 45 degrees (from the above-right).
    D45 = 3,
    /// Directional, 135 degrees (from the above-left).
    D135 = 4,
    /// Directional, 113 degrees.
    D113 = 5,
    /// Directional, 157 degrees.
    D157 = 6,
    /// Directional, 203 degrees (from the below-left).
    D203 = 7,
    /// Directional, 67 degrees.
    D67 = 8,
    /// Blend both edges toward the opposite corner estimates.
    Smooth = 9,
    /// Smooth interpolation down each column.
    SmoothVertical = 10,
    /// Smooth interpolation across each row.
    SmoothHorizontal = 11,
    /// Choose the edge sample nearest a planar extrapolation.
    Paeth = 12,
}

impl IntraMode {
    /// Modes requiring no directional-angle syntax.
    pub const NON_DIRECTIONAL: [Self; 5] = [
        Self::Dc,
        Self::Smooth,
        Self::SmoothVertical,
        Self::SmoothHorizontal,
        Self::Paeth,
    ];
    /// The non-directional set plus vertical and horizontal prediction.
    pub const EXTENDED: [Self; 7] = [
        Self::Dc,
        Self::Vertical,
        Self::Horizontal,
        Self::Smooth,
        Self::SmoothVertical,
        Self::SmoothHorizontal,
        Self::Paeth,
    ];

    /// The six directional modes that are neither vertical nor horizontal.
    pub const OBLIQUE: [Self; 6] = [
        Self::D45,
        Self::D135,
        Self::D113,
        Self::D157,
        Self::D203,
        Self::D67,
    ];
    /// Every mode, in symbol order.
    pub const ALL: [Self; 13] = [
        Self::Dc,
        Self::Vertical,
        Self::Horizontal,
        Self::D45,
        Self::D135,
        Self::D113,
        Self::D157,
        Self::D203,
        Self::D67,
        Self::Smooth,
        Self::SmoothVertical,
        Self::SmoothHorizontal,
        Self::Paeth,
    ];

    /// The mode's `YMode` symbol.
    #[must_use]
    pub const fn symbol(self) -> usize {
        self as usize
    }

    /// The mode with `YMode` symbol `symbol`, if it is one.
    #[must_use]
    pub fn from_symbol(symbol: usize) -> Option<Self> {
        Self::ALL.get(symbol).copied()
    }

    /// The nominal angle in degrees of a directional mode (AV1
    /// `Mode_To_Angle`).
    #[must_use]
    pub const fn nominal_angle(self) -> Option<u16> {
        match self {
            Self::Vertical => Some(90),
            Self::Horizontal => Some(180),
            Self::D45 => Some(45),
            Self::D135 => Some(135),
            Self::D113 => Some(113),
            Self::D157 => Some(157),
            Self::D203 => Some(203),
            Self::D67 => Some(67),
            _ => None,
        }
    }

    /// The `angle_delta_y` CDF index of a directional mode
    /// (`YMode - V_PRED`), or `None` for modes without an angle.
    #[must_use]
    pub const fn angle_context(self) -> Option<usize> {
        match self.nominal_angle() {
            Some(_) => Some(self.symbol() - Self::Vertical.symbol()),
            None => None,
        }
    }
}

/// Which edge samples beyond the block's own side are available (AV1
/// §5.11.35 `haveAboveRt` / `haveBelowLft`). Unavailable ones repeat the
/// last available sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeExtension {
    /// The row above continues to the right of the block.
    pub above_right: bool,
    /// The column to the left continues below the block.
    pub below_left: bool,
}

/// Reconstructed neighbours for a square prediction, extended exactly as
/// AV1 specifies when the above or left tile edge is missing.
#[derive(Debug)]
pub struct IntraEdges {
    /// `AboveRow[0..2n]`, clamped to the picture as AV1 §7.11.2 reads it.
    above: [u8; EDGE_LEN],
    /// `LeftCol[0..2n]`.
    left: [u8; EDGE_LEN],
    corner: u8,
    dc: u8,
    size: usize,
}

impl IntraEdges {
    /// Read the available neighbours of the square at `(x, y)`, with no
    /// samples beyond its own side (enough for every non-directional mode
    /// and for vertical and horizontal).
    ///
    /// # Panics
    ///
    /// If `size` is not 4/8/16/32/64 or the block lies outside `plane`.
    #[must_use]
    pub fn new(plane: &Plane, x: usize, y: usize, size: usize) -> Self {
        Self::extended(plane, x, y, size, EdgeExtension::default())
    }

    /// Read the neighbours of the square at `(x, y)` as AV1 §7.11.2 forms
    /// `AboveRow` and `LeftCol`: `2 * size` samples each, the part beyond
    /// the block read only where `extension` says it is decoded, and every
    /// read clamped to the picture (whose size is the coded `MiCols` /
    /// `MiRows` area).
    ///
    /// # Panics
    ///
    /// If `size` is not 4/8/16/32/64 or the block lies outside `plane`.
    #[must_use]
    pub fn extended(
        plane: &Plane,
        x: usize,
        y: usize,
        size: usize,
        extension: EdgeExtension,
    ) -> Self {
        assert!([4, 8, 16, 32, 64].contains(&size));
        let view = plane.view();
        assert!(x + size <= view.width() && y + size <= view.height());
        let (have_above, have_left) = (y > 0, x > 0);
        let mut edges = Self {
            above: [DC_NO_NEIGHBOURS - 1; EDGE_LEN],
            left: [DC_NO_NEIGHBOURS + 1; EDGE_LEN],
            corner: DC_NO_NEIGHBOURS,
            dc: dc_value(plane, x, y, size, have_above, have_left),
            size,
        };
        let reach = |extended: bool| if extended { 2 * size } else { size };
        if have_above {
            let row = view.row(y - 1);
            let limit = (x + reach(extension.above_right) - 1).min(view.width() - 1);
            for (i, sample) in edges.above[..2 * size].iter_mut().enumerate() {
                *sample = row[(x + i).min(limit)];
            }
        }
        if have_left {
            let limit = (y + reach(extension.below_left) - 1).min(view.height() - 1);
            for (i, sample) in edges.left[..2 * size].iter_mut().enumerate() {
                *sample = view.row((y + i).min(limit))[x - 1];
            }
        }
        if !have_above && have_left {
            edges.above.fill(view.row(y)[x - 1]);
        }
        if !have_left && have_above {
            edges.left.fill(view.row(y - 1)[x]);
        }
        edges.corner = match (have_above, have_left) {
            (true, true) => view.row(y - 1)[x - 1],
            (true, false) => view.row(y - 1)[x],
            (false, true) => view.row(y)[x - 1],
            (false, false) => DC_NO_NEIGHBOURS,
        };
        edges
    }

    /// Fill a square destination using the selected mode.
    ///
    /// # Panics
    ///
    /// If the destination does not match the square edge size.
    pub fn predict(&self, mode: IntraMode, dst: &mut PlaneViewMut<'_>) {
        let n = self.size;
        assert!(dst.width() == n && dst.height() == n);
        if mode == IntraMode::Dc {
            dst.fill(self.dc);
            return;
        }
        if let Some(angle) = mode.nominal_angle()
            && !matches!(mode, IntraMode::Vertical | IntraMode::Horizontal)
        {
            self.predict_angle(angle, dst);
            return;
        }
        match mode {
            IntraMode::Vertical => {
                (0..n).for_each(|y| dst.row_mut(y)[..n].copy_from_slice(&self.above[..n]));
            }
            IntraMode::Horizontal => (0..n).for_each(|y| dst.row_mut(y)[..n].fill(self.left[y])),
            IntraMode::Paeth => self.predict_paeth(dst),
            IntraMode::Smooth | IntraMode::SmoothVertical | IntraMode::SmoothHorizontal => {
                self.predict_smooth(mode, dst);
            }
            _ => unreachable!("DC and oblique directions handled above"),
        }
    }

    /// Paeth prediction (AV1 §7.11.2.2), the scalar reference of the
    /// `crate::av1_intra` kernels.
    fn predict_paeth(&self, dst: &mut PlaneViewMut<'_>) {
        let n = self.size;
        for (y, &left) in self.left[..n].iter().enumerate() {
            for (pixel, &top) in dst.row_mut(y)[..n].iter_mut().zip(&self.above[..n]) {
                *pixel = paeth(left, top, self.corner);
            }
        }
    }

    /// Smooth prediction (AV1 §7.11.2.6), `mode` one of the three smooth
    /// modes; the scalar reference of the `crate::av1_intra` kernels.
    fn predict_smooth(&self, mode: IntraMode, dst: &mut PlaneViewMut<'_>) {
        let n = self.size;
        let weights = smooth_weights(n);
        let (bottom, right) = (u32::from(self.left[n - 1]), u32::from(self.above[n - 1]));
        for (y, (&wy, &left)) in weights.iter().zip(&self.left[..n]).enumerate() {
            let wy = u32::from(wy);
            let row = dst.row_mut(y)[..n]
                .iter_mut()
                .zip(&self.above[..n])
                .zip(weights);
            for ((pixel, &top), &wx) in row {
                let wx = u32::from(wx);
                let vertical = wy * u32::from(top) + (SMOOTH_SCALE - wy) * bottom;
                let horizontal = wx * u32::from(left) + (SMOOTH_SCALE - wx) * right;
                *pixel = match mode {
                    IntraMode::Smooth => {
                        ((vertical + horizontal + SMOOTH_SCALE) >> (SMOOTH_SHIFT + 1)) as u8
                    }
                    IntraMode::SmoothVertical => {
                        ((vertical + SMOOTH_SCALE / 2) >> SMOOTH_SHIFT) as u8
                    }
                    _ => ((horizontal + SMOOTH_SCALE / 2) >> SMOOTH_SHIFT) as u8,
                };
            }
        }
    }

    /// The block side.
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// `AboveRow[0..size]`.
    pub(crate) fn above_row(&self) -> &[u8] {
        &self.above[..self.size]
    }

    /// `LeftCol[0..size]`.
    pub(crate) fn left_column(&self) -> &[u8] {
        &self.left[..self.size]
    }

    /// `AboveRow[-1]`.
    pub(crate) fn corner(&self) -> u8 {
        self.corner
    }
}

/// The smooth weights of a block side (4, 8, 16, 32 or 64). Every weight
/// is in `4..=255`, so it and `256 − weight` fit a byte.
pub(crate) fn smooth_weights(side: usize) -> &'static [u8] {
    &SMOOTH_WEIGHTS[side - 4..2 * side - 4]
}

fn paeth(left: u8, top: u8, corner: u8) -> u8 {
    let base = i32::from(left) + i32::from(top) - i32::from(corner);
    let dl = (base - i32::from(left)).abs();
    let dt = (base - i32::from(top)).abs();
    let dc = (base - i32::from(corner)).abs();
    if dl <= dt && dl <= dc {
        left
    } else if dt <= dc {
        top
    } else {
        corner
    }
}

/// AV1 smooth weights for sizes 4, 8, 16, 32 and 64, concatenated.
const SMOOTH_WEIGHTS: [u8; 124] = [
    255, 149, 85, 64, 255, 197, 146, 105, 73, 50, 37, 32, 255, 225, 196, 170, 145, 123, 102, 84,
    68, 54, 43, 33, 26, 20, 17, 16, 255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111,
    101, 92, 83, 74, 66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8, 255, 248, 240,
    233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133, 127, 121, 116,
    111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27,
    25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_edges_stay_flat_with_either_neighbour_missing() {
        for n in [4, 8, 16, 32, 64] {
            let mut plane = Plane::new(2 * n, 2 * n, 0).unwrap();
            plane.view_mut().fill(73);
            for (x, y) in [(n, n), (0, n), (n, 0)] {
                let edges = IntraEdges::new(&plane, x, y, n);
                for mode in IntraMode::EXTENDED {
                    let mut dst = Plane::new(n, n, 0).unwrap();
                    edges.predict(mode, &mut dst.view_mut());
                    for row in 0..n {
                        assert!(
                            dst.view().row(row).iter().all(|&pixel| pixel == 73),
                            "{mode:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tile_corner_defaults_and_smooth_rounding() {
        let plane = Plane::new(8, 8, 0).unwrap();
        let edges = IntraEdges::new(&plane, 0, 0, 8);
        let mut dst = Plane::new(8, 8, 0).unwrap();
        for (mode, value) in [
            (IntraMode::Dc, 128),
            (IntraMode::Vertical, 127),
            (IntraMode::Horizontal, 129),
            (IntraMode::Paeth, 128),
        ] {
            edges.predict(mode, &mut dst.view_mut());
            assert!((0..8).all(|y| dst.view().row(y).iter().all(|&v| v == value)));
        }
        edges.predict(IntraMode::SmoothVertical, &mut dst.view_mut());
        assert_eq!(dst.view().row(0)[0], 127);
        assert_eq!(dst.view().row(7)[0], 129);
        edges.predict(IntraMode::SmoothHorizontal, &mut dst.view_mut());
        assert_eq!(dst.view().row(0)[0], 129);
        assert_eq!(dst.view().row(0)[7], 127);
    }

    #[test]
    fn smooth_four_by_four_has_known_corner_values() {
        let mut plane = Plane::new(8, 8, 0).unwrap();
        plane.view_mut().row_mut(3)[4..8].copy_from_slice(&[10, 20, 30, 40]);
        for (row, value) in [50, 60, 70, 80].into_iter().enumerate() {
            plane.view_mut().row_mut(4 + row)[3] = value;
        }
        let edges = IntraEdges::new(&plane, 4, 4, 4);
        let mut dst = Plane::new(4, 4, 0).unwrap();
        edges.predict(IntraMode::Smooth, &mut dst.view_mut());
        assert_eq!(
            [
                dst.view().row(0)[0],
                dst.view().row(0)[3],
                dst.view().row(3)[0],
                dst.view().row(3)[3]
            ],
            [30, 41, 71, 60]
        );
    }

    #[test]
    fn paeth_ties_prefer_left_then_above() {
        assert_eq!(paeth(0, 30, 20), 0);
        assert_eq!(paeth(30, 0, 20), 0);
        assert_eq!(paeth(50, 10, 30), 30);
    }
}
