//! AV1 recursive filter intra prediction (AV1 §7.11.2.3): the block is
//! predicted in 4x2 patches, each a 7-tap filter of the corner, four
//! samples above and two to the left, which come from the edges or from
//! patches already predicted.
//!
//! Scalar reference; the SIMD patch kernels are in `crate::av1_intra`. The
//! taps are libaom `av1_filter_intra_taps`
//! (`av1/common/reconintra.c`, BSD-2-Clause, Copyright (c) 2016, Alliance
//! for Open Media; see `THIRD_PARTY.md`).

use nc_common::PlaneViewMut;

use super::IntraEdges;

/// Largest block side filter intra is allowed on.
pub const FILTER_INTRA_MAX_SIZE: usize = 32;
/// Patch width and height.
const PATCH_W: usize = 4;
const PATCH_H: usize = 2;
/// Outputs per patch.
pub(crate) const PATCH_SAMPLES: usize = PATCH_W * PATCH_H;
/// Inputs per patch: corner, four above, two left.
pub(crate) const TAPS: usize = 7;
/// Filter taps have four fractional bits (`FILTER_INTRA_SCALE_BITS`).
pub(crate) const SCALE_BITS: u32 = 4;

/// The prediction's working grid: row 0 is the corner and the above edge,
/// column 0 the corner and the left edge; the patches fill the rest.
pub(crate) type FilterGrid = [[u8; FILTER_INTRA_MAX_SIZE + 1]; FILTER_INTRA_MAX_SIZE + 1];

/// The five filter intra modes, with their `filter_intra_mode` symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FilterIntraMode {
    /// `FILTER_DC_PRED`.
    Dc = 0,
    /// `FILTER_V_PRED`.
    Vertical = 1,
    /// `FILTER_H_PRED`.
    Horizontal = 2,
    /// `FILTER_D157_PRED`.
    D157 = 3,
    /// `FILTER_PAETH_PRED`.
    Paeth = 4,
}

impl FilterIntraMode {
    /// Every mode, in symbol order.
    pub const ALL: [Self; 5] = [
        Self::Dc,
        Self::Vertical,
        Self::Horizontal,
        Self::D157,
        Self::Paeth,
    ];

    /// The mode's `filter_intra_mode` symbol.
    #[must_use]
    pub const fn symbol(self) -> usize {
        self as usize
    }

    /// The `YMode` symbol whose transform-type CDFs this mode uses
    /// (`Filter_Intra_Mode_To_Intra_Dir`: DC, V, H, D157, DC).
    #[must_use]
    pub const fn intra_dir(self) -> usize {
        const DC_PRED: usize = 0;
        const V_PRED: usize = 1;
        const H_PRED: usize = 2;
        const D157_PRED: usize = 6;
        match self {
            Self::Dc | Self::Paeth => DC_PRED,
            Self::Vertical => V_PRED,
            Self::Horizontal => H_PRED,
            Self::D157 => D157_PRED,
        }
    }
}

/// Taps per mode, per output sample of the patch (row-major), per input
/// (corner, above 0..3, left 0..1).
const FILTER_TAPS: [[[i32; TAPS]; PATCH_W * PATCH_H]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

/// [`FILTER_TAPS`] per mode and input, across the patch's outputs: one
/// vector of weights per input, for kernels that compute a patch at once.
pub(crate) const FILTER_TAP_COLUMNS: [[[i16; PATCH_SAMPLES]; TAPS]; 5] = {
    let mut columns = [[[0; PATCH_SAMPLES]; TAPS]; 5];
    let mut mode = 0;
    while mode < 5 {
        let mut input = 0;
        while input < TAPS {
            let mut output = 0;
            while output < PATCH_SAMPLES {
                columns[mode][input][output] = FILTER_TAPS[mode][output][input] as i16;
                output += 1;
            }
            input += 1;
        }
        mode += 1;
    }
    columns
};

/// The inputs of the patch whose top-left output is `grid[r][c]`: the
/// corner, four above and two to the left.
pub(crate) fn patch_inputs(grid: &FilterGrid, r: usize, c: usize) -> [u8; TAPS] {
    [
        grid[r - 1][c - 1],
        grid[r - 1][c],
        grid[r - 1][c + 1],
        grid[r - 1][c + 2],
        grid[r - 1][c + 3],
        grid[r][c - 1],
        grid[r + 1][c - 1],
    ]
}

/// Write a patch's outputs (row-major) at `grid[r][c]`.
pub(crate) fn store_patch(grid: &mut FilterGrid, r: usize, c: usize, values: [u8; PATCH_SAMPLES]) {
    grid[r][c..c + PATCH_W].copy_from_slice(&values[..PATCH_W]);
    grid[r + 1][c..c + PATCH_W].copy_from_slice(&values[PATCH_W..]);
}

/// The scalar patches of a `width`x`height` block (see [`FilterGrid`]).
pub(crate) fn predict_patches(
    grid: &mut FilterGrid,
    width: usize,
    height: usize,
    mode: FilterIntraMode,
) {
    let taps = &FILTER_TAPS[mode.symbol()];
    for r in (1..=height).step_by(PATCH_H) {
        for c in (1..=width).step_by(PATCH_W) {
            let inputs = patch_inputs(grid, r, c);
            let values = taps.map(|tap| {
                let sum: i32 = tap.iter().zip(inputs).map(|(&t, p)| t * i32::from(p)).sum();
                // Clip1(Round2Signed(sum, 4)): negative sums clip to 0
                // either way, so the unsigned rounding is exact.
                ((sum + (1 << (SCALE_BITS - 1))) >> SCALE_BITS).clamp(0, 255) as u8
            });
            store_patch(grid, r, c, values);
        }
    }
}

impl IntraEdges {
    /// Predict the square with filter intra `mode`.
    ///
    /// # Panics
    ///
    /// If the block is larger than 32x32 or the destination does not match
    /// the edge size.
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
        self.predict_filter_block(self.size, self.size, patches, mode, dst);
    }

    /// Filter intra of a `width`x`height` block from these edges
    /// (`AboveRow[0..width]`, `LeftCol[0..height]` and the corner):
    /// §7.11.2.3 for any shape up to 32x32.
    ///
    /// # Panics
    ///
    /// If a side exceeds [`FILTER_INTRA_MAX_SIZE`] or `dst` is not
    /// `width`x`height`.
    pub(crate) fn predict_filter_block(
        &self,
        width: usize,
        height: usize,
        patches: impl FnOnce(&mut FilterGrid, usize, usize, FilterIntraMode),
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(width <= FILTER_INTRA_MAX_SIZE && height <= FILTER_INTRA_MAX_SIZE);
        assert!(dst.width() == width && dst.height() == height);
        let mut grid: FilterGrid = [[0; FILTER_INTRA_MAX_SIZE + 1]; FILTER_INTRA_MAX_SIZE + 1];
        grid[0][0] = self.corner;
        grid[0][1..=width].copy_from_slice(&self.above[..width]);
        for (r, &sample) in self.left[..height].iter().enumerate() {
            grid[r + 1][0] = sample;
        }
        patches(&mut grid, width, height, mode);
        for (r, row) in grid[1..=height].iter().enumerate() {
            dst.row_mut(r)[..width].copy_from_slice(&row[1..=width]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;

    #[test]
    fn every_mode_keeps_flat_edges_flat() {
        // Each tap row sums to 16, so a flat neighbourhood stays flat.
        let mut plane = Plane::new(64, 64, 0).unwrap();
        plane.view_mut().fill(200);
        for n in [8, 16, 32] {
            let edges = IntraEdges::new(&plane, n, n, n);
            for mode in FilterIntraMode::ALL {
                let mut dst = Plane::new(n, n, 0).unwrap();
                edges.predict_filter(mode, &mut dst.view_mut());
                for y in 0..n {
                    assert!(dst.view().row(y)[..n].iter().all(|&v| v == 200), "{mode:?}");
                }
            }
        }
    }

    #[test]
    fn first_patch_matches_hand_computation() {
        // Corner 10, above 20 30 40 50, left 60 70; FILTER_DC_PRED sample 0:
        // (-6*10 + 10*20 + 12*60 + 8) >> 4 = 54.
        let mut plane = Plane::new(16, 16, 0).unwrap();
        let mut view = plane.view_mut();
        view.row_mut(3)[3..8].copy_from_slice(&[10, 20, 30, 40, 50]);
        view.row_mut(4)[3] = 60;
        view.row_mut(5)[3] = 70;
        let edges = IntraEdges::new(&plane, 4, 4, 8);
        let mut dst = Plane::new(8, 8, 0).unwrap();
        edges.predict_filter(FilterIntraMode::Dc, &mut dst.view_mut());
        assert_eq!(dst.view().row(0)[0], 54);
    }
}
