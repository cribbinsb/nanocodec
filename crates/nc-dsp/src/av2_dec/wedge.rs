//! The masks of AV2 wedge compound and inter-intra prediction in the
//! general decoder: the 68 wedge shapes of every block size from 8x8 to
//! 64x64 for each chroma layout, their per-8x8 summary for the temporal
//! motion field, and the smooth inter-intra masks ([`InterMasks`]).
//!
//! Ported from dav2d `src/wedge.c` (`wedge_codebook_16`, `gen_master`,
//! `copy2d`, `subsample_420`, `subsample_422`, `fill_tmvp`,
//! `init_wedge_masks`, `init_ii_masks`; and the `WEDGE_MASK`,
//! `WEDGE_TMVP`, `II_MASK` lookups of `wedge.h`; commit 237229e, Copyright
//! © 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see
//! `docs/licenses/dav2d`). AV2's wedges (20 directions, 68 shapes for
//! every size, a sharp edge on the small sizes) are not AV1's, so they
//! are generated here; the inter-intra masks are AV1's
//! ([`av1_interintra_smooth_mask`]: dav2d's 64-entry weight table is
//! AV1's 128-entry one at even positions, with the same per-size step).
//! The decoder builds one [`InterMasks`] and keeps it; nothing is global.

use crate::av1_mc::mask::{Av1InterIntraMode, av1_interintra_smooth_mask};

use super::compound::MaskLayout;

/// Wedge shapes of every block size.
pub const WEDGE_TYPES: usize = 68;
/// Smallest wedge block side.
const WEDGE_MIN_SIDE: usize = 8;
/// Smallest inter-intra block side.
const II_MIN_SIDE: usize = 4;
/// Block sides from the smallest: 8, 16, 32, 64 (wedges) or 4 to 64.
const WEDGE_SIDES: usize = 4;
const II_SIDES: usize = 5;
/// Inter-intra modes (DC, vertical, horizontal, smooth).
const II_MODES: usize = 4;
/// Side of a master mask.
const MASTER: usize = 128;
/// A summary cell of the temporal motion field covers 8x8 samples.
const TMVP_CELL: usize = 8;
/// A cell is one prediction's when this many of its 64 weights are
/// nearly all of one side.
const TMVP_MAJORITY: u32 = 60;
/// Weights below this count as the second prediction's.
const TMVP_LOW: u8 = 4;
/// Weights above this count as the first prediction's.
const TMVP_HIGH: u8 = 60;
/// The summary values: mostly the second prediction, mostly the first,
/// mixed.
const TMVP_SECOND: u8 = 0;
const TMVP_FIRST: u8 = 1;
const TMVP_MIXED: u8 = 2;
/// Wedges of blocks up to this side in both directions have a sharp
/// edge (master slope 2); larger ones a soft edge (slope 1).
const SHARP_MAX_SIDE: usize = 16;
const SHARP_SLOPE: i32 = 2;
const SOFT_SLOPE: i32 = 1;

/// A wedge direction (dav2d `enum WedgeDirectionType`), as the index of
/// its cosine and sine.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Direction {
    D0,
    D14,
    D27,
    D45,
    D63,
    D90,
    D117,
    D135,
    D153,
    D166,
    D180,
    D194,
    D207,
    D225,
    D243,
    D270,
    D297,
    D315,
    D333,
    D346,
}

/// Each direction's cosine and sine, scaled by 4 (`cos_lut`, `sin_lut`).
const COS: [i8; 20] = [
    4, 4, 4, 2, 2, 0, -2, -2, -4, -4, -4, -4, -4, -2, -2, 0, 2, 2, 4, 4,
];
const SIN: [i8; 20] = [
    0, -1, -2, -2, -4, -4, -4, -2, -2, -1, 0, 1, 2, 2, 4, 4, 4, 2, 2, 1,
];
/// The weight of the far side by distance from the edge (`weight`);
/// distances are clamped to its last index.
const EDGE_WEIGHT: [u8; 29] = [
    8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
];
/// Half the weight scale of [`EDGE_WEIGHT`] (the mask is 4 times it).
const EDGE_HALF: u8 = 16;
const EDGE_SCALE: u8 = 4;

/// The codebook (`wedge_codebook_16`): each shape's direction and the
/// position of its edge in eighths of the block, horizontally and
/// vertically.
#[rustfmt::skip]
const CODEBOOK: [(Direction, u8, u8); WEDGE_TYPES] = {
    use Direction::{
        D0, D117, D135, D14, D153, D166, D180, D194, D207, D225, D243, D27, D270, D297, D315, D333,
        D346, D45, D63, D90,
    };
    [
        (D0, 5, 4), (D0, 6, 4), (D0, 7, 4),
        (D14, 4, 4), (D14, 5, 4), (D14, 6, 4), (D14, 7, 4),
        (D27, 4, 4), (D27, 5, 4), (D27, 6, 4), (D27, 7, 4),
        (D45, 4, 4), (D45, 5, 4), (D45, 6, 4), (D45, 7, 4),
        (D63, 4, 4), (D63, 4, 3), (D63, 4, 2), (D63, 4, 1),
        (D90, 4, 3), (D90, 4, 2), (D90, 4, 1),
        (D117, 4, 4), (D117, 4, 3), (D117, 4, 2), (D117, 4, 1),
        (D135, 4, 4), (D135, 3, 4), (D135, 2, 4), (D135, 1, 4),
        (D153, 4, 4), (D153, 3, 4), (D153, 2, 4), (D153, 1, 4),
        (D166, 4, 4), (D166, 3, 4), (D166, 2, 4), (D166, 1, 4),
        (D180, 3, 4), (D180, 2, 4), (D180, 1, 4),
        (D194, 3, 4), (D194, 2, 4), (D194, 1, 4),
        (D207, 3, 4), (D207, 2, 4), (D207, 1, 4),
        (D225, 3, 4), (D225, 2, 4), (D225, 1, 4),
        (D243, 4, 5), (D243, 4, 6), (D243, 4, 7),
        (D270, 4, 5), (D270, 4, 6), (D270, 4, 7),
        (D297, 4, 5), (D297, 4, 6), (D297, 4, 7),
        (D315, 5, 4), (D315, 6, 4), (D315, 7, 4),
        (D333, 5, 4), (D333, 6, 4), (D333, 7, 4),
        (D346, 5, 4), (D346, 6, 4), (D346, 7, 4),
    ]
};

/// Wedge directions.
const DIRECTIONS: usize = COS.len();

/// The 128x128 master mask of direction `direction` (its index) with edge
/// slope `slope` (dav2d `gen_master`).
fn master(direction: usize, slope: i32) -> Vec<u8> {
    let max_distance = (EDGE_WEIGHT.len() - 1) as i32;
    let s = i32::from(SIN[direction]) * slope;
    let c = i32::from(COS[direction]) * slope;
    let centre = MASTER as i32 - 1;
    let mut out = vec![0_u8; MASTER * MASTER];
    for (y, row) in out.as_chunks_mut::<MASTER>().0.iter_mut().enumerate() {
        let dy = (2 * y as i32 - centre) * s;
        for (x, weight) in row.iter_mut().enumerate() {
            let d = ((2 * x as i32 - centre) * c + dy).clamp(-max_distance, max_distance);
            let far = EDGE_WEIGHT[d.unsigned_abs() as usize];
            *weight = EDGE_SCALE * if d >= 0 { EDGE_HALF - far } else { far };
        }
    }
    out
}

/// Index of a side among `count` sides from `min` (doubling).
fn side_index(side: usize, min: usize, count: usize) -> usize {
    assert!(side.is_power_of_two() && side >= min && side <= min << (count - 1));
    (side / min).trailing_zeros() as usize
}

/// The wedge, temporal-summary and inter-intra masks of every block size
/// (dav2d `Dav2dMasks`, built by `dav2d_init_ii_wedge_masks`).
#[derive(Debug, Clone)]
pub struct InterMasks {
    /// Wedge masks per layout (4:4:4, 4:2:2, 4:2:0): for each size in
    /// [`Self::wedge_size_index`] order, the 68 shapes of that size.
    wedge: [Vec<u8>; 3],
    /// Start of each size's shapes in the 4:4:4 masks (4:2:2 and 4:2:0
    /// start at this divided by 2 and 4).
    wedge_offsets: [usize; WEDGE_SIDES * WEDGE_SIDES],
    /// Per-8x8 summaries of the 4:4:4 shapes, size by size.
    tmvp: Vec<u8>,
    tmvp_offsets: [usize; WEDGE_SIDES * WEDGE_SIDES],
    /// Inter-intra masks: for each size from 4x4 to 64x64, the four modes.
    interintra: Vec<u8>,
    interintra_offsets: [usize; II_SIDES * II_SIDES],
}

impl Default for InterMasks {
    fn default() -> Self {
        Self::new()
    }
}

impl InterMasks {
    /// Generate every mask.
    #[must_use]
    pub fn new() -> Self {
        let mut masks = Self {
            wedge: [Vec::new(), Vec::new(), Vec::new()],
            wedge_offsets: [0; WEDGE_SIDES * WEDGE_SIDES],
            tmvp: Vec::new(),
            tmvp_offsets: [0; WEDGE_SIDES * WEDGE_SIDES],
            interintra: Vec::new(),
            interintra_offsets: [0; II_SIDES * II_SIDES],
        };
        masks.build_wedges();
        masks.build_interintra();
        masks
    }

    /// Each wedge block size, by index.
    fn wedge_sizes() -> impl Iterator<Item = (usize, usize)> {
        (0..WEDGE_SIDES * WEDGE_SIDES).map(|i| {
            (
                WEDGE_MIN_SIDE << (i / WEDGE_SIDES),
                WEDGE_MIN_SIDE << (i % WEDGE_SIDES),
            )
        })
    }

    /// The index of wedge block size `w` x `h`.
    fn wedge_size_index((w, h): (usize, usize)) -> usize {
        side_index(w, WEDGE_MIN_SIDE, WEDGE_SIDES) * WEDGE_SIDES
            + side_index(h, WEDGE_MIN_SIDE, WEDGE_SIDES)
    }

    fn build_wedges(&mut self) {
        let sharp: Vec<Vec<u8>> = (0..DIRECTIONS).map(|d| master(d, SHARP_SLOPE)).collect();
        let soft: Vec<Vec<u8>> = (0..DIRECTIONS).map(|d| master(d, SOFT_SLOPE)).collect();
        for (index, (w, h)) in Self::wedge_sizes().enumerate() {
            self.wedge_offsets[index] = self.wedge[0].len();
            self.tmvp_offsets[index] = self.tmvp.len();
            let masters = if w <= SHARP_MAX_SIDE && h <= SHARP_MAX_SIDE {
                &sharp
            } else {
                &soft
            };
            for &(direction, x_offset, y_offset) in &CODEBOOK {
                let full = crop(
                    &masters[direction as usize],
                    (w, h),
                    (x_offset.into(), y_offset.into()),
                );
                self.wedge[1].extend(subsample(&full, (w, h), (true, false)));
                self.wedge[2].extend(subsample(&full, (w, h), (true, true)));
                self.tmvp.extend(tmvp_summary(&full, (w, h)));
                self.wedge[0].extend(full);
            }
        }
    }

    fn build_interintra(&mut self) {
        const MODES: [Av1InterIntraMode; II_MODES] = [
            Av1InterIntraMode::Dc,
            Av1InterIntraMode::Vertical,
            Av1InterIntraMode::Horizontal,
            Av1InterIntraMode::Smooth,
        ];
        for index in 0..II_SIDES * II_SIDES {
            let (w, h) = (
                II_MIN_SIDE << (index / II_SIDES),
                II_MIN_SIDE << (index % II_SIDES),
            );
            self.interintra_offsets[index] = self.interintra.len();
            for mode in MODES {
                let start = self.interintra.len();
                self.interintra.resize(start + w * h, 0);
                av1_interintra_smooth_mask(mode, (w, h), &mut self.interintra[start..]);
            }
        }
    }

    /// The wedge mask `index` (0 to 67) of a `w` x `h` luma block (8 to
    /// 64 each) for `layout`'s plane: `w >> sx` x `h >> sy` weights of the
    /// first prediction, stride `w >> sx` (dav2d `WEDGE_MASK(bs, bw4,
    /// bh4, index, layout)`; the caller swaps the predictions for the
    /// other sign).
    ///
    /// # Panics
    ///
    /// If a size or the index is out of range.
    #[must_use]
    pub fn wedge(&self, (w, h): (usize, usize), index: usize, layout: MaskLayout) -> &[u8] {
        assert!(index < WEDGE_TYPES);
        let (plane, scale) = match layout {
            MaskLayout::I444 => (0, 1),
            MaskLayout::I422 => (1, 2),
            MaskLayout::I420 => (2, 4),
        };
        let area = w * h / scale;
        let start = self.wedge_offsets[Self::wedge_size_index((w, h))] / scale + index * area;
        &self.wedge[plane][start..start + area]
    }

    /// The temporal-field summary of wedge `index` of a `w` x `h` luma
    /// block: per 8x8 cell in raster order, 1 where the first prediction
    /// holds nearly the whole cell, 0 where the second does, 2 if mixed
    /// (dav2d `WEDGE_TMVP`, `fill_tmvp`).
    ///
    /// # Panics
    ///
    /// If a size or the index is out of range.
    #[must_use]
    pub fn wedge_tmvp(&self, (w, h): (usize, usize), index: usize) -> &[u8] {
        assert!(index < WEDGE_TYPES);
        let area = (w / TMVP_CELL) * (h / TMVP_CELL);
        let start = self.tmvp_offsets[Self::wedge_size_index((w, h))] + index * area;
        &self.tmvp[start..start + area]
    }

    /// The inter-intra mask of `mode` for a `w` x `h` plane block (4 to 64
    /// each): weights of the intra prediction, stride `w` (dav2d
    /// `II_MASK`).
    ///
    /// # Panics
    ///
    /// If a size is out of range.
    #[must_use]
    pub fn interintra(&self, (w, h): (usize, usize), mode: Av1InterIntraMode) -> &[u8] {
        let index =
            side_index(w, II_MIN_SIDE, II_SIDES) * II_SIDES + side_index(h, II_MIN_SIDE, II_SIDES);
        let start = self.interintra_offsets[index] + mode as usize * w * h;
        &self.interintra[start..start + w * h]
    }
}

/// The `w` x `h` window of a master mask whose edge sits at
/// (`x_offset`, `y_offset`) eighths of the block (dav2d `copy2d`).
fn crop(master: &[u8], (w, h): (usize, usize), (x_offset, y_offset): (usize, usize)) -> Vec<u8> {
    let left = MASTER / 2 - x_offset * w / WEDGE_MIN_SIDE;
    let top = MASTER / 2 - y_offset * h / WEDGE_MIN_SIDE;
    (0..h)
        .flat_map(|y| &master[(top + y) * MASTER + left..][..w])
        .copied()
        .collect()
}

/// A 4:4:4 mask averaged over horizontal pairs, and over vertical pairs
/// too with `sub_y` (dav2d `subsample_422`, `subsample_420`).
fn subsample(
    full: &[u8],
    (w, h): (usize, usize),
    (sub_x, sub_y): (bool, bool),
) -> impl Iterator<Item = u8> + '_ {
    debug_assert!(sub_x, "every subsampled layout halves the width");
    let rows = if sub_y { h / 2 } else { h };
    (0..rows).flat_map(move |y| {
        let row = if sub_y { 2 * y } else { y };
        (0..w / 2).map(move |x| {
            let at = |dy: usize, dx: usize| u16::from(full[(row + dy) * w + 2 * x + dx]);
            if sub_y {
                ((at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1) + 2) >> 2) as u8
            } else {
                ((at(0, 0) + at(0, 1) + 1) >> 1) as u8
            }
        })
    })
}

/// The per-8x8 summary of a 4:4:4 mask (dav2d `fill_tmvp`).
fn tmvp_summary(full: &[u8], (w, h): (usize, usize)) -> Vec<u8> {
    let mut out = Vec::with_capacity((w / TMVP_CELL) * (h / TMVP_CELL));
    for cy in 0..h / TMVP_CELL {
        for cx in 0..w / TMVP_CELL {
            let (mut low, mut high) = (0_u32, 0_u32);
            for y in cy * TMVP_CELL..(cy + 1) * TMVP_CELL {
                for &m in &full[y * w + cx * TMVP_CELL..][..TMVP_CELL] {
                    low += u32::from(m < TMVP_LOW);
                    high += u32::from(m > TMVP_HIGH);
                }
            }
            out.push(if low >= TMVP_MAJORITY {
                TMVP_SECOND
            } else if high >= TMVP_MAJORITY {
                TMVP_FIRST
            } else {
                TMVP_MIXED
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values captured from dav2d's `dav2d_masks` after
    /// `dav2d_init_ii_wedge_masks`.
    #[test]
    fn matches_dav2d() {
        let masks = InterMasks::new();
        assert_eq!(
            masks.wedge((16, 8), 5, MaskLayout::I444)[..16],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 16, 44, 60, 64, 64, 64]
        );
        assert_eq!(
            masks.wedge((32, 16), 40, MaskLayout::I420)[32..48],
            [62, 46, 18, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            masks.wedge_tmvp((32, 32), 20),
            [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(masks.interintra((4, 4), Av1InterIntraMode::Dc), [32; 16]);
        assert_eq!(
            masks.interintra((4, 4), Av1InterIntraMode::Vertical)[..5],
            [60, 60, 60, 60, 19]
        );
    }
}
