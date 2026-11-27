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
//! Decoders share immutable mask data generated on the first request.

use std::sync::OnceLock;

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

/// The 128x128 master masks of all 20 directions with edge slope `slope`
/// (dav2d `gen_master`).
fn master_set(slope: i32) -> Vec<u8> {
    let max_distance = (EDGE_WEIGHT.len() - 1) as i32;
    let centre = MASTER as i32 - 1;
    let plane = MASTER * MASTER;
    let mut out = vec![0_u8; DIRECTIONS * plane];
    for (direction, master) in out.chunks_exact_mut(plane).enumerate() {
        let s = i32::from(SIN[direction]) * slope;
        let c = i32::from(COS[direction]) * slope;
        for (y, row) in master.as_chunks_mut::<MASTER>().0.iter_mut().enumerate() {
            let dy = (2 * y as i32 - centre) * s;
            for (x, weight) in row.iter_mut().enumerate() {
                let d = ((2 * x as i32 - centre) * c + dy).clamp(-max_distance, max_distance);
                let far = EDGE_WEIGHT[d.unsigned_abs() as usize];
                *weight = EDGE_SCALE * if d >= 0 { EDGE_HALF - far } else { far };
            }
        }
    }
    out
}

/// Index of a side among `count` sides from `min` (doubling).
fn side_index(side: usize, min: usize, count: usize) -> usize {
    assert!(side.is_power_of_two() && side >= min && side <= min << (count - 1));
    (side / min).trailing_zeros() as usize
}

/// The full-resolution wedges of a single block size and the derived
/// chroma and temporal masks, built independently when requested.
#[derive(Debug, Clone)]
struct WedgeSize {
    full: Vec<u8>,
    half: OnceLock<Vec<u8>>,
    quarter: OnceLock<Vec<u8>>,
    tmvp: OnceLock<Vec<u8>>,
}

/// AV2 wedge, temporal-summary and inter-intra masks by block size.
#[derive(Debug, Clone)]
pub struct InterMasks {
    masters: [OnceLock<Vec<u8>>; 2],
    wedge: [OnceLock<WedgeSize>; WEDGE_SIDES * WEDGE_SIDES],
    interintra: [OnceLock<Vec<u8>>; II_SIDES * II_SIDES],
}

impl Default for InterMasks {
    fn default() -> Self {
        Self::new()
    }
}

impl InterMasks {
    /// Start with an empty mask cache. Each entry is built on its first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            masters: [const { OnceLock::new() }; 2],
            wedge: [const { OnceLock::new() }; WEDGE_SIDES * WEDGE_SIDES],
            interintra: [const { OnceLock::new() }; II_SIDES * II_SIDES],
        }
    }

    /// The process-wide immutable masks. Instances decoding different
    /// streams can reuse them because they depend only on block geometry.
    #[must_use]
    pub fn shared() -> &'static Self {
        static MASKS: InterMasks = InterMasks::new();
        &MASKS
    }

    fn wedge_size(&self, (w, h): (usize, usize)) -> &WedgeSize {
        let index = side_index(w, WEDGE_MIN_SIDE, WEDGE_SIDES) * WEDGE_SIDES
            + side_index(h, WEDGE_MIN_SIDE, WEDGE_SIDES);
        self.wedge[index].get_or_init(|| {
            let soft = w > SHARP_MAX_SIDE || h > SHARP_MAX_SIDE;
            let masters = self.masters[usize::from(soft)]
                .get_or_init(|| master_set(if soft { SOFT_SLOPE } else { SHARP_SLOPE }));
            let plane = MASTER * MASTER;
            let mut full = Vec::with_capacity(WEDGE_TYPES * w * h);
            for &(direction, x_offset, y_offset) in &CODEBOOK {
                let master = &masters[direction as usize * plane..][..plane];
                append_crop(
                    master,
                    (w, h),
                    (x_offset.into(), y_offset.into()),
                    &mut full,
                );
            }
            WedgeSize {
                full,
                half: OnceLock::new(),
                quarter: OnceLock::new(),
                tmvp: OnceLock::new(),
            }
        })
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
        let size = self.wedge_size((w, h));
        let full_area = w * h;
        let (masks, area) = match layout {
            MaskLayout::I444 => (&size.full, full_area),
            MaskLayout::I422 => (
                size.half.get_or_init(|| {
                    let mut out = Vec::with_capacity(size.full.len() / 2);
                    for mask in size.full.chunks_exact(full_area) {
                        append_subsample_422(mask, w, &mut out);
                    }
                    out
                }),
                full_area / 2,
            ),
            MaskLayout::I420 => (
                size.quarter.get_or_init(|| {
                    let mut out = Vec::with_capacity(size.full.len() / 4);
                    for mask in size.full.chunks_exact(full_area) {
                        append_subsample_420(mask, w, &mut out);
                    }
                    out
                }),
                full_area / 4,
            ),
        };
        &masks[index * area..][..area]
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
        let size = self.wedge_size((w, h));
        let area = (w / TMVP_CELL) * (h / TMVP_CELL);
        let masks = size.tmvp.get_or_init(|| {
            let mut out = Vec::with_capacity(WEDGE_TYPES * area);
            for mask in size.full.chunks_exact(w * h) {
                append_tmvp_summary(mask, (w, h), &mut out);
            }
            out
        });
        &masks[index * area..][..area]
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
        const MODES: [Av1InterIntraMode; II_MODES] = [
            Av1InterIntraMode::Dc,
            Av1InterIntraMode::Vertical,
            Av1InterIntraMode::Horizontal,
            Av1InterIntraMode::Smooth,
        ];
        let index =
            side_index(w, II_MIN_SIDE, II_SIDES) * II_SIDES + side_index(h, II_MIN_SIDE, II_SIDES);
        let area = w * h;
        let masks = self.interintra[index].get_or_init(|| {
            let mut out = vec![0; II_MODES * area];
            for (output, mode) in out.chunks_exact_mut(area).zip(MODES) {
                av1_interintra_smooth_mask(mode, (w, h), output);
            }
            out
        });
        &masks[mode as usize * area..][..area]
    }
}

/// Append the `w` x `h` window of a master mask whose edge sits at
/// (`x_offset`, `y_offset`) eighths of the block (dav2d `copy2d`).
fn append_crop(
    master: &[u8],
    (w, h): (usize, usize),
    (x_offset, y_offset): (usize, usize),
    out: &mut Vec<u8>,
) {
    let left = MASTER / 2 - x_offset * w / WEDGE_MIN_SIDE;
    let top = MASTER / 2 - y_offset * h / WEDGE_MIN_SIDE;
    for y in 0..h {
        out.extend_from_slice(&master[(top + y) * MASTER + left..][..w]);
    }
}

/// Append a 4:4:4 mask averaged over horizontal pairs (dav2d `subsample_422`).
fn append_subsample_422(full: &[u8], w: usize, out: &mut Vec<u8>) {
    for row in full.chunks_exact(w) {
        for &[a, b] in row.as_chunks::<2>().0 {
            out.push(((u16::from(a) + u16::from(b) + 1) >> 1) as u8);
        }
    }
}

/// Append a 4:4:4 mask averaged over 2x2 pairs (dav2d `subsample_420`).
fn append_subsample_420(full: &[u8], w: usize, out: &mut Vec<u8>) {
    for pair in full.chunks_exact(2 * w) {
        let (r0, r1) = pair.split_at(w);
        for (&[a0, b0], &[a1, b1]) in r0.as_chunks::<2>().0.iter().zip(r1.as_chunks::<2>().0) {
            let sum = u16::from(a0) + u16::from(b0) + u16::from(a1) + u16::from(b1);
            out.push(((sum + 2) >> 2) as u8);
        }
    }
}

/// Append the per-8x8 summary of a 4:4:4 mask (dav2d `fill_tmvp`).
fn append_tmvp_summary(full: &[u8], (w, h): (usize, usize), out: &mut Vec<u8>) {
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
