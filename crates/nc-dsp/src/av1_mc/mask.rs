//! Blend masks (AV1 §7.11.3.11–13): the wedge masks of compound and
//! inter-intra wedge prediction, the difference-weighted mask of compound
//! prediction, and the smooth inter-intra masks. Every weight is for the
//! first input of the blend ([`super::blend`]), 0 to 64.
//!
//! Master masks, codebooks and `Ii_Weights_1d`: libaom v3.15.3
//! `av1/common/reconinter.c` and `tools/gen_wedge_masks_data.py`
//! (BSD-2-Clause, see `THIRD_PARTY.md`).

use super::INTER_POST_ROUND;
use super::blend::{AV1_MASK_MAX, Av1CompoundPred};
use super::round2;

/// Wedge shapes per block size (`WEDGE_TYPES`).
pub const AV1_WEDGE_TYPES: usize = 16;
/// Side of the master masks (`MASK_MASTER_SIZE`).
const MASTER_SIZE: usize = 64;

/// `Wedge_Master_Oblique_Even` (AV1 §7.11.3.11).
const MASTER_OBLIQUE_EVEN: [u8; MASTER_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 11,
    27, 46, 58, 62, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];
/// `Wedge_Master_Oblique_Odd` (AV1 §7.11.3.11).
const MASTER_OBLIQUE_ODD: [u8; MASTER_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 6,
    18, 37, 53, 60, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];
/// `Wedge_Master_Vertical` (AV1 §7.11.3.11).
const MASTER_VERTICAL: [u8; MASTER_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7,
    21, 43, 57, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];

/// Wedge edge directions (`WEDGE_HORIZONTAL` ... `WEDGE_OBLIQUE153`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Horizontal,
    Vertical,
    Oblique27,
    Oblique63,
    Oblique117,
    Oblique153,
}
const DIRECTIONS: usize = 6;

/// One codebook entry: the edge direction and its offsets in eighths of the
/// block width and height.
type Code = (Direction, u8, u8);

use Direction::{
    Horizontal as H, Oblique27 as O27, Oblique63 as O63, Oblique117 as O117, Oblique153 as O153,
    Vertical as V,
};

/// `Wedge_Codebook` (AV1 §7.11.3.11) for blocks taller than wide.
const CODEBOOK_TALL: [Code; AV1_WEDGE_TYPES] = [
    (O27, 4, 4),
    (O63, 4, 4),
    (O117, 4, 4),
    (O153, 4, 4),
    (H, 4, 2),
    (H, 4, 4),
    (H, 4, 6),
    (V, 4, 4),
    (O27, 4, 2),
    (O27, 4, 6),
    (O153, 4, 2),
    (O153, 4, 6),
    (O63, 2, 4),
    (O63, 6, 4),
    (O117, 2, 4),
    (O117, 6, 4),
];
/// `Wedge_Codebook` for blocks wider than tall.
const CODEBOOK_WIDE: [Code; AV1_WEDGE_TYPES] = [
    (O27, 4, 4),
    (O63, 4, 4),
    (O117, 4, 4),
    (O153, 4, 4),
    (V, 2, 4),
    (V, 4, 4),
    (V, 6, 4),
    (H, 4, 4),
    (O27, 4, 2),
    (O27, 4, 6),
    (O153, 4, 2),
    (O153, 4, 6),
    (O63, 2, 4),
    (O63, 6, 4),
    (O117, 2, 4),
    (O117, 6, 4),
];
/// `Wedge_Codebook` for square blocks.
const CODEBOOK_SQUARE: [Code; AV1_WEDGE_TYPES] = [
    (O27, 4, 4),
    (O63, 4, 4),
    (O117, 4, 4),
    (O153, 4, 4),
    (H, 4, 2),
    (H, 4, 6),
    (V, 2, 4),
    (V, 6, 4),
    (O27, 4, 2),
    (O27, 4, 6),
    (O153, 4, 2),
    (O153, 4, 6),
    (O63, 2, 4),
    (O63, 6, 4),
    (O117, 2, 4),
    (O117, 6, 4),
];

/// The block sizes (width, height) that may use wedges (`Wedge_Bits` 4).
pub const AV1_WEDGE_SIZES: [(usize, usize); 9] = [
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (8, 32),
    (32, 8),
];

/// The 64x64 master mask of each direction (`MasterMask`, AV1 §7.11.3.11).
fn master_masks() -> Vec<[[u8; MASTER_SIZE]; MASTER_SIZE]> {
    let mut masters = vec![[[0u8; MASTER_SIZE]; MASTER_SIZE]; DIRECTIONS];
    let shifted = |master: &[u8; MASTER_SIZE], shift: isize, j: usize| {
        master[(j as isize - shift).clamp(0, MASTER_SIZE as isize - 1) as usize]
    };
    // Odd rows of the 63-degree master sit one sample further left.
    let mut shift = (MASTER_SIZE / 4) as isize;
    for i in (0..MASTER_SIZE).step_by(2) {
        for j in 0..MASTER_SIZE {
            masters[Direction::Oblique63 as usize][i][j] = shifted(&MASTER_OBLIQUE_EVEN, shift, j);
            masters[Direction::Oblique63 as usize][i + 1][j] =
                shifted(&MASTER_OBLIQUE_ODD, shift - 1, j);
            masters[Direction::Vertical as usize][i][j] = MASTER_VERTICAL[j];
            masters[Direction::Vertical as usize][i + 1][j] = MASTER_VERTICAL[j];
        }
        shift -= 1;
    }
    let max = AV1_MASK_MAX;
    let (oblique63, vertical) = (
        masters[Direction::Oblique63 as usize],
        masters[Direction::Vertical as usize],
    );
    for (i, (oblique_row, vertical_row)) in oblique63.iter().zip(&vertical).enumerate() {
        for (j, (&oblique, &edge)) in oblique_row.iter().zip(vertical_row).enumerate() {
            masters[Direction::Oblique27 as usize][j][i] = oblique;
            masters[Direction::Oblique117 as usize][i][MASTER_SIZE - 1 - j] = max - oblique;
            masters[Direction::Oblique153 as usize][MASTER_SIZE - 1 - j][i] = max - oblique;
            masters[Direction::Horizontal as usize][j][i] = edge;
        }
    }
    masters
}

/// Every wedge mask (`WedgeMasks`, AV1 §7.11.3.11), built once by the
/// decoder that owns it: for each wedge block size, both signs of each of
/// the 16 shapes, `width` weights per row.
#[derive(Debug, Clone)]
pub struct Av1WedgeMasks {
    weights: Vec<u8>,
}

impl Default for Av1WedgeMasks {
    fn default() -> Self {
        Self::new()
    }
}

impl Av1WedgeMasks {
    /// Generate the masks (AV1 §7.11.3.11).
    #[must_use]
    pub fn new() -> Self {
        let masters = master_masks();
        let mut weights = Vec::new();
        for &(w, h) in &AV1_WEDGE_SIZES {
            let codebook = codebook(w, h);
            let mut block = vec![0u8; w * h];
            for &(direction, x_offset, y_offset) in codebook {
                let master = &masters[direction as usize];
                let x0 = MASTER_SIZE / 2 - ((usize::from(x_offset) * w) >> 3);
                let y0 = MASTER_SIZE / 2 - ((usize::from(y_offset) * h) >> 3);
                for (i, row) in block.chunks_exact_mut(w).enumerate() {
                    row.copy_from_slice(&master[y0 + i][x0..x0 + w]);
                }
                // The sign whose mask mostly weighs the first input is
                // decided by the average weight along the top and left
                // edges.
                let edge: u32 = block[..w].iter().map(|&m| u32::from(m)).sum::<u32>()
                    + (1..h).map(|i| u32::from(block[i * w])).sum::<u32>();
                let count = (w + h - 1) as u32;
                let flip_sign = (edge + count / 2) / count < u32::from(AV1_MASK_MAX / 2);
                let complement: Vec<u8> = block.iter().map(|&m| AV1_MASK_MAX - m).collect();
                let (sign0, sign1) = if flip_sign {
                    (&complement, &block)
                } else {
                    (&block, &complement)
                };
                weights.extend_from_slice(sign0);
                weights.extend_from_slice(sign1);
            }
        }
        Self { weights }
    }

    /// The mask of wedge `index` with sign `sign` (`wedge_sign`) for a
    /// `w`x`h` block, `w` weights per row, or `None` if blocks of that size
    /// cannot use wedges.
    #[must_use]
    pub fn mask(&self, (w, h): (usize, usize), sign: bool, index: usize) -> Option<&[u8]> {
        assert!(index < AV1_WEDGE_TYPES, "wedge index {index}");
        let mut offset = 0;
        for &size in &AV1_WEDGE_SIZES {
            let area = size.0 * size.1;
            if size == (w, h) {
                let start = offset + (2 * index + usize::from(sign)) * area;
                return Some(&self.weights[start..start + area]);
            }
            offset += 2 * AV1_WEDGE_TYPES * area;
        }
        None
    }
}

/// `get_wedge_codebook`: the codebook by block shape.
fn codebook(w: usize, h: usize) -> &'static [Code; AV1_WEDGE_TYPES] {
    match w.cmp(&h) {
        std::cmp::Ordering::Less => &CODEBOOK_TALL,
        std::cmp::Ordering::Greater => &CODEBOOK_WIDE,
        std::cmp::Ordering::Equal => &CODEBOOK_SQUARE,
    }
}

/// Base weight of the difference-weighted mask.
const DIFFERENCE_BASE: i32 = 38;
/// The prediction difference is divided by this (`DIFF_FACTOR`).
const DIFFERENCE_FACTOR: i32 = 16;

/// The difference-weighted mask (AV1 §7.11.3.12) of two compound-precision
/// predictions: `Clip3(0, 64, 38 + Round2(|p0 - p1|, InterPostRound) / 16)`,
/// or 64 minus that when `inverse` (`mask_type`). Writes `h` rows of `w`
/// weights to `mask`.
pub fn av1_difference_weight_mask(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    inverse: bool,
    (w, h): (usize, usize),
    mask: &mut [u8],
) {
    let max = i32::from(AV1_MASK_MAX);
    for (r, row) in mask[..w * h].chunks_exact_mut(w).enumerate() {
        let first = &p0.data[r * p0.stride..][..w];
        let second = &p1.data[r * p1.stride..][..w];
        for ((out, &x), &y) in row.iter_mut().zip(first).zip(second) {
            let difference = round2((i32::from(x) - i32::from(y)).abs(), INTER_POST_ROUND);
            let m = (DIFFERENCE_BASE + difference / DIFFERENCE_FACTOR).clamp(0, max);
            *out = if inverse { max - m } else { m } as u8;
        }
    }
}

/// `Ii_Weights_1d` (AV1 §7.11.3.13): smooth inter-intra weights of the
/// intra prediction by distance from its edge, for a 128-sample block.
const II_WEIGHTS_1D: [u8; 128] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 27,
    26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12,
    11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];
/// `MAX_SB_SIZE`: the block size `Ii_Weights_1d` is laid out for.
const II_WEIGHTS_SIZE: usize = II_WEIGHTS_1D.len();

/// The inter-intra prediction mode (`interintra_mode`; the discriminants
/// are the syntax values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1InterIntraMode {
    /// `II_DC_PRED`: equal weights.
    Dc = 0,
    /// `II_V_PRED`: weights fade down the rows.
    Vertical = 1,
    /// `II_H_PRED`: weights fade along the columns.
    Horizontal = 2,
    /// `II_SMOOTH_PRED`: weights fade with distance from the nearer edge.
    Smooth = 3,
}

/// The smooth inter-intra mask (AV1 §7.11.3.13) of a `w`x`h` plane block:
/// `h` rows of `w` weights of the intra prediction.
pub fn av1_interintra_smooth_mask(
    mode: Av1InterIntraMode,
    (w, h): (usize, usize),
    mask: &mut [u8],
) {
    let scale = II_WEIGHTS_SIZE / w.max(h);
    for (i, row) in mask[..w * h].chunks_exact_mut(w).enumerate() {
        for (j, out) in row.iter_mut().enumerate() {
            *out = match mode {
                Av1InterIntraMode::Dc => AV1_MASK_MAX / 2,
                Av1InterIntraMode::Vertical => II_WEIGHTS_1D[i * scale],
                Av1InterIntraMode::Horizontal => II_WEIGHTS_1D[j * scale],
                Av1InterIntraMode::Smooth => II_WEIGHTS_1D[i.min(j) * scale],
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// libaom's `wedge_signflip_lookup` for the sizes of
    /// [`AV1_WEDGE_SIZES`]: the flip sign the edge average must reproduce.
    const SIGN_FLIP: [[u8; AV1_WEDGE_TYPES]; 9] = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    ];

    /// The flip sign is where the master mask lands: the sign-`flip` mask
    /// is the unflipped master region.
    #[test]
    fn computed_flip_signs_match_the_libaom_lookup() {
        let masters = master_masks();
        let masks = Av1WedgeMasks::new();
        for (s, &(w, h)) in AV1_WEDGE_SIZES.iter().enumerate() {
            for (index, &(direction, x_offset, y_offset)) in codebook(w, h).iter().enumerate() {
                let x0 = MASTER_SIZE / 2 - ((usize::from(x_offset) * w) >> 3);
                let y0 = MASTER_SIZE / 2 - ((usize::from(y_offset) * h) >> 3);
                let flip = SIGN_FLIP[s][index] == 1;
                let mask = masks.mask((w, h), flip, index).unwrap();
                assert_eq!(
                    mask[0], masters[direction as usize][y0][x0],
                    "{w}x{h} wedge {index}"
                );
                assert_eq!(
                    mask[w * h - 1],
                    masters[direction as usize][y0 + h - 1][x0 + w - 1]
                );
            }
        }
    }
}
