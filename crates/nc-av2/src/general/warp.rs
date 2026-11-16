//! Warp model parameters: the model type, the shear parameters the warp
//! filter steps by, the affine fit of a block's neighbour samples, and the
//! vector a model gives a block's centre (dav2d `warpmv.c`:
//! `dav2d_resolve_divisor_32`, `dav2d_get_shear_params`,
//! `dav2d_set_affine_mv2d`, `dav2d_find_affine_int`; `env.h`: `warp_type`,
//! `get_warpmv_2d`, `get_gmv_2d`; dav2d commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`).
//!
//! Used by the frame header (global motion rescaling) and the block layer
//! (warped blocks, warp candidates). The warp *pixel* kernels are nc-dsp's.
use super::block::motion::vector::Mv;

/// Warp model types (dav2d `Dav2dWarpedMotionType`, in its order).
pub mod kind {
    /// No motion.
    pub const IDENTITY: u8 = 0;
    /// Translation only.
    pub const TRANSLATION: u8 = 1;
    /// Rotation and zoom.
    pub const ROT_ZOOM: u8 = 2;
    /// Affine.
    pub const AFFINE: u8 = 3;
    /// Not usable (dav2d `DAV2D_WM_TYPE_INVALID`: shear out of range).
    pub const INVALID: u8 = 4;
}

/// The unit of the diagonal matrix entries (1.0 in Q16).
pub const UNITY: i32 = 1 << 16;
/// The identity matrix (translation, then the 2×2 part row by row).
pub const IDENTITY: [i32; 6] = [0, 0, UNITY, 0, 0, UNITY];

/// Reciprocal table of divisions (dav2d `dav2d_div_recip`):
/// `2^9 * 2^7 / (2^7 + f)` rounded.
const DIVISOR_RECIPROCALS: [u16; 129] = [
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462, 458, 455, 452, 449,
    446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417, 415, 412, 410, 407, 405, 402, 400, 397,
    395, 392, 390, 388, 386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356,
    354, 352, 350, 349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324, 323,
    321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295,
    294, 293, 291, 290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 277, 275, 274, 273, 272,
    271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256,
];
/// Bits of the reciprocal's mantissa index, and the precision added.
const DIVISOR_INDEX_BITS: u32 = 7;
const DIVISOR_PRECISION: u32 = 9;

/// Shear and matrix entries are kept to multiples of 64 (Q16 / 1024).
const STEP_MASK: i32 = 0x3f;
const STEP_HALF: i32 = 0x20;
/// The shear parameters' range.
const SHEAR_MIN: i32 = -0x8000;
const SHEAR_MAX: i32 = 0x7fc0;
/// The range of an off-diagonal matrix entry, and of a diagonal one
/// around [`UNITY`].
pub const OFF_DIAGONAL_MAX: i32 = 0x7fc0;
const DIAGONAL_MIN: i32 = UNITY - OFF_DIAGONAL_MAX;
const DIAGONAL_MAX: i32 = UNITY + OFF_DIAGONAL_MAX;
/// The translation's range (Q16 samples at eighth-sample vectors).
const TRANSLATION_MIN: i64 = -0x800_0000;
const TRANSLATION_MAX: i64 = 0x7ff_ffc0;
/// Eighth-sample vector to the matrix translation (Q16 / 8).
const VECTOR_TO_TRANSLATION_LOG2: u32 = 13;
/// A model is affine (the warp filter may use it) when the weighted
/// shears stay below this bound.
const AFFINE_BOUND: i32 = 0x30000;
/// Samples whose source and destination offsets differ by this much or
/// more are outliers of the least-squares fit.
const FIT_OUTLIER: i32 = 256;
/// The fit's determinant reciprocal keeps 16 more bits.
const FIT_SHIFT: i32 = 16;
/// Vector precisions (dav2d `mv_prec`: 0..6 from eight samples to eighth
/// samples); below eighth, model vectors are rounded to quarters.
const EIGHTH_PRECISION: u8 = 6;
/// Vector components are clipped to this magnitude.
pub const MAX_VECTOR: i32 = 0xffff;
/// Vectors may point this many 4x4 units past the frame.
const CLIP_MARGIN_UNITS: i32 = 4;
/// Eighth samples per 4x4 unit.
const VECTOR_UNITS_PER_UNIT: i32 = 32;

/// The shear parameters of an affine model (dav2d `u.p`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Shear {
    /// Horizontal filter step per column.
    pub alpha: i16,
    /// Horizontal filter step per row.
    pub beta: i16,
    /// Vertical filter step per column.
    pub gamma: i16,
    /// Vertical filter step per row.
    pub delta: i16,
}

/// A warp model with its derived parameters (dav2d
/// `Dav2dWarpedMotionParams`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpModel {
    /// [`kind`].
    pub kind: u8,
    /// Translation (Q16) then the 2x2 matrix (Q16), row by row.
    pub matrix: [i32; 6],
    /// The shear parameters ([`WarpModel::shear`] sets them).
    pub shear: Shear,
    /// The shears are small enough for the warp filter (dav2d `affine`).
    pub affine: bool,
}

impl Default for WarpModel {
    fn default() -> Self {
        Self::of_matrix(kind::IDENTITY, IDENTITY)
    }
}

/// `v` negated when `sign` is negative (dav2d `apply_sign`).
fn apply_sign(v: i32, sign: i32) -> i32 {
    if sign < 0 { v.wrapping_neg() } else { v }
}

/// `v` truncated to 32 bits and negated when `sign` is negative (dav2d
/// `apply_sign64`).
fn apply_sign64(v: i64, sign: i64) -> i32 {
    let v = v as i32;
    if sign < 0 { v.wrapping_neg() } else { v }
}

/// Clipped to 32 bits (dav2d `iclip64to32`).
fn clip64(v: i64, min: i64, max: i64) -> i32 {
    v.clamp(min, max) as i32
}

/// `1 / d` (d > 0) as a multiplier of [`DIVISOR_PRECISION`] bits and the
/// shift that goes with it (dav2d `dav2d_resolve_divisor_32` and its
/// 64-bit twin).
#[must_use]
pub fn resolve_divisor(d: u64) -> (i32, u32) {
    let shift = d.ilog2();
    let excess = d - (1 << shift);
    let index = if shift > DIVISOR_INDEX_BITS {
        (excess + (1 << (shift - DIVISOR_INDEX_BITS - 1))) >> (shift - DIVISOR_INDEX_BITS)
    } else {
        excess << (DIVISOR_INDEX_BITS - shift)
    };
    (
        i32::from(DIVISOR_RECIPROCALS[index as usize]),
        shift + DIVISOR_PRECISION,
    )
}

/// Rounded to a multiple of 64 (toward the nearest, halves away from
/// zero below `pivot`) and clipped (dav2d's `(v + 0x20 - (v < pivot)) &
/// ~0x3f` with `iclip`).
fn round_step(v: i32, pivot: i32, min: i32, max: i32) -> i32 {
    (v.wrapping_add(STEP_HALF - i32::from(v < pivot)) & !STEP_MASK).clamp(min, max)
}

/// The model type of a matrix (dav2d `warp_type`).
#[must_use]
pub fn matrix_kind(matrix: &[i32; 6]) -> u8 {
    if matrix[2] != matrix[5] || matrix[3] != matrix[4].wrapping_neg() {
        kind::AFFINE
    } else if matrix[2] != UNITY || matrix[3] != 0 {
        kind::ROT_ZOOM
    } else if matrix[0] | matrix[1] != 0 {
        kind::TRANSLATION
    } else {
        kind::IDENTITY
    }
}

impl WarpModel {
    /// A model of a known type, its shear not derived yet.
    #[must_use]
    pub fn of_matrix(kind: u8, matrix: [i32; 6]) -> Self {
        Self {
            kind,
            matrix,
            shear: Shear::default(),
            affine: false,
        }
    }

    /// Whether the model is usable (every type but [`kind::INVALID`]).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.kind != kind::INVALID
    }

    /// Derive the shear parameters (dav2d `dav2d_get_shear_params`);
    /// false when the matrix cannot be sheared (its first diagonal entry
    /// is not positive), the shear then left as it was.
    pub fn shear(&mut self) -> bool {
        let m = &self.matrix;
        if m[2] <= 0 {
            return false;
        }
        let clip = |v: i32| round_step(v, 0, SHEAR_MIN, SHEAR_MAX) as i16;
        let alpha = clip(m[2] - UNITY);
        let beta = clip(m[3]);
        let (multiplier, shift) = resolve_divisor(u64::from(m[2].unsigned_abs()));
        let y = i64::from(apply_sign(multiplier, m[2]));
        let round = (1_i64 << shift) >> 1;
        let divide = |v: i64| apply_sign64((v.unsigned_abs() as i64 + round) >> shift, v);
        let v1 = (i64::from(m[4]) * i64::from(UNITY)).wrapping_mul(y);
        let gamma = clip(divide(v1));
        let v2 = (i64::from(m[3]) * i64::from(m[4])).wrapping_mul(y);
        let delta = clip(m[5].wrapping_sub(divide(v2)).wrapping_sub(UNITY));
        self.shear = Shear {
            alpha,
            beta,
            gamma,
            delta,
        };
        let weigh = |a: i16, ka: i32, b: i16, kb: i32| {
            ka * i32::from(a).abs() + kb * i32::from(b).abs() < AFFINE_BOUND
        };
        self.affine = weigh(alpha, 4, beta, 7) && weigh(gamma, 4, delta, 4);
        true
    }

    /// Derive the shear and set the type from the matrix, or
    /// [`kind::INVALID`] when it cannot be sheared (dav2d's `type =
    /// get_shear_params() ? INVALID : warp_type(matrix)`).
    pub fn finish(&mut self) {
        self.kind = if self.shear() {
            matrix_kind(&self.matrix)
        } else {
            kind::INVALID
        };
    }

    /// Set the translation so that the block's centre moves by `mv`
    /// (dav2d `dav2d_set_affine_mv2d`); (`row`, `column`) and the size
    /// (`bw4`, `bh4`) in 4x4 units.
    pub fn set_translation(&mut self, (bw4, bh4): (u32, u32), mv: Mv, (row, column): (u32, u32)) {
        let m = &mut self.matrix;
        let (centre_x, centre_y) = block_centre((row, column), (bw4, bh4));
        let to_translation = |v: i32| i64::from(v) << VECTOR_TO_TRANSLATION_LOG2;
        m[0] = clip64(
            to_translation(mv.x) - centre_x * i64::from(m[2] - UNITY) - centre_y * i64::from(m[3]),
            TRANSLATION_MIN,
            TRANSLATION_MAX,
        );
        m[1] = clip64(
            to_translation(mv.y) - centre_x * i64::from(m[4]) - centre_y * i64::from(m[5] - UNITY),
            TRANSLATION_MIN,
            TRANSLATION_MAX,
        );
    }

    /// Fit the 2x2 matrix to neighbour samples by least squares, then set
    /// the translation from `mv` (dav2d `dav2d_find_affine_int`). Each
    /// sample is a source position and where it moved, in eighth samples
    /// relative to the block's top-left (`[[x, y], [x', y']]`).
    pub fn fit(
        &mut self,
        samples: &[[[i32; 2]; 2]],
        (bw4, bh4): (u32, u32),
        mv: Mv,
        position: (u32, u32),
    ) {
        let centre_y = (2 * bh4 as i32 - 1) * 8;
        let centre_x = (2 * bw4 as i32 - 1) * 8;
        let moved_y = centre_y + mv.y;
        let moved_x = centre_x + mv.x;
        let (mut a00, mut a01, mut a11) = (0_i32, 0_i32, 0_i32);
        let (mut bx0, mut bx1, mut by0, mut by1) = (0_i32, 0_i32, 0_i32, 0_i32);
        for sample in samples {
            let dx = sample[1][0].wrapping_sub(moved_x);
            let dy = sample[1][1].wrapping_sub(moved_y);
            let sx = sample[0][0] - centre_x;
            let sy = sample[0][1] - centre_y;
            if (sx - dx).abs() < FIT_OUTLIER && (sy - dy).abs() < FIT_OUTLIER {
                let term = |p: i32, q: i32, r: i32| {
                    (p.wrapping_mul(q) >> 2)
                        .wrapping_add(p)
                        .wrapping_add(q)
                        .wrapping_add(r)
                };
                a00 = a00.wrapping_add(term(sx, sx, 8));
                a01 = a01.wrapping_add(term(sx, sy, 4));
                a11 = a11.wrapping_add(term(sy, sy, 8));
                bx0 = bx0.wrapping_add(term(sx, dx, 8));
                bx1 = bx1.wrapping_add(term(sy, dx, 4));
                by0 = by0.wrapping_add(term(sx, dy, 4));
                by1 = by1.wrapping_add(term(sy, dy, 8));
            }
        }
        let wide = i64::from;
        let det = wide(a00) * wide(a11) - wide(a01) * wide(a01);
        let m = &mut self.matrix;
        if det == 0 {
            (m[2], m[3], m[4], m[5]) = (UNITY, 0, 0, UNITY);
        } else {
            let (multiplier, shift) = resolve_divisor(det.unsigned_abs());
            let mut inverse = apply_sign64(i64::from(multiplier), det);
            let mut shift = shift as i32 - FIT_SHIFT;
            if shift < 0 {
                inverse = inverse.wrapping_shl((-shift) as u32);
                shift = 0;
            }
            let round = (1_i64 << shift) >> 1;
            let solve = |p: i64, pivot: i32, min: i32, max: i32| {
                let v1 = p.wrapping_mul(i64::from(inverse));
                let v2 = (v1.wrapping_add(round - i64::from(v1 < 0)) >> shift) as i32;
                round_step(v2, pivot, min, max)
            };
            let cross = |p: i32, q: i32, r: i32, s: i32| {
                wide(p).wrapping_mul(wide(q)) - wide(r).wrapping_mul(wide(s))
            };
            m[2] = solve(cross(a11, bx0, a01, bx1), UNITY, DIAGONAL_MIN, DIAGONAL_MAX);
            m[3] = solve(
                cross(a00, bx1, a01, bx0),
                0,
                -OFF_DIAGONAL_MAX,
                OFF_DIAGONAL_MAX,
            );
            m[4] = solve(
                cross(a11, by0, a01, by1),
                0,
                -OFF_DIAGONAL_MAX,
                OFF_DIAGONAL_MAX,
            );
            m[5] = solve(cross(a00, by1, a01, by0), UNITY, DIAGONAL_MIN, DIAGONAL_MAX);
        }
        self.set_translation((bw4, bh4), mv, position);
    }
}

/// A block's centre in luma samples, minus a half (the warp's reference
/// point: dav2d `isux`, `isuy`).
fn block_centre((row, column): (u32, u32), (bw4, bh4): (u32, u32)) -> (i64, i64) {
    (
        i64::from(column * 4 + 2 * bw4 - 1),
        i64::from(row * 4 + 2 * bh4 - 1),
    )
}

/// Clip a vector of a block at (`row`, `column`) of `bw4` x `bh4` units to
/// the frame of `columns` x `rows` units, with the 4-unit margin (dav2d's
/// `-(bx4 + bw4 + 4) * 32 .. (iw4 - bx4 + 4) * 32` bounds).
#[must_use]
pub fn clip_to_frame(mv: Mv, block: BlockArea, frame: (u32, u32)) -> Mv {
    let bound = |position: u32, extent: u32, frame: u32| {
        let position = position as i32;
        (
            -(position + extent as i32 + CLIP_MARGIN_UNITS) * VECTOR_UNITS_PER_UNIT,
            (frame as i32 - position + CLIP_MARGIN_UNITS) * VECTOR_UNITS_PER_UNIT,
        )
    };
    let (min_y, max_y) = bound(block.row, block.bh4, frame.1);
    let (min_x, max_x) = bound(block.column, block.bw4, frame.0);
    Mv {
        y: mv.y.clamp(min_y, max_y),
        x: mv.x.clamp(min_x, max_x),
    }
}

/// A block's position and size in 4x4 units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockArea {
    /// Top row.
    pub row: u32,
    /// Left column.
    pub column: u32,
    /// Width.
    pub bw4: u32,
    /// Height.
    pub bh4: u32,
}

/// The vector a model gives the block's centre, at `precision` (dav2d
/// `get_warpmv_2d`; below eighth samples it is rounded to quarters),
/// clipped to the frame of `frame` (columns, rows) units.
#[must_use]
pub fn model_vector(matrix: &[i32; 6], block: BlockArea, frame: (u32, u32), precision: u8) -> Mv {
    let (x, y) = block_centre((block.row, block.column), (block.bw4, block.bh4));
    let xc = i64::from(matrix[2].wrapping_sub(UNITY)) * x
        + i64::from(matrix[3]) * y
        + i64::from(matrix[0]);
    let yc = i64::from(matrix[5].wrapping_sub(UNITY)) * y
        + i64::from(matrix[4]) * x
        + i64::from(matrix[1]);
    let not_eighth = u32::from(precision < EIGHTH_PRECISION);
    let shift = VECTOR_TO_TRANSLATION_LOG2 + not_eighth;
    let round = (1_i64 << shift) >> 1;
    let max = MAX_VECTOR - not_eighth as i32;
    let scale = |v: i64| {
        apply_sign64(
            ((v.unsigned_abs() as i64 + round) >> shift) << not_eighth,
            v,
        )
        .clamp(-max, max)
    };
    clip_to_frame(
        Mv {
            y: scale(yc),
            x: scale(xc),
        },
        block,
        frame,
    )
}

/// The vector a global motion model gives a block (dav2d `get_gmv_2d`):
/// none for the identity, the clipped translation, else the model's
/// vector at the frame's precision (`mv_precision` 0..3, eighth at 3);
/// whole samples when the frame forces them.
#[must_use]
pub fn global_vector(
    model: &WarpModel,
    block: BlockArea,
    frame: (u32, u32),
    mv_precision: u8,
    force_integer: bool,
) -> Mv {
    /// dav2d's precision numbering starts three steps coarser.
    const PRECISION_OFFSET: u8 = 3;
    let mv = match model.kind {
        kind::IDENTITY => return Mv::default(),
        // dav2d reads the translation's rows from `matrix[0]` (AV2 frame
        // headers never code a translation-only model).
        kind::TRANSLATION => clip_to_frame(
            Mv {
                y: model.matrix[0] >> VECTOR_TO_TRANSLATION_LOG2,
                x: model.matrix[1] >> VECTOR_TO_TRANSLATION_LOG2,
            },
            block,
            frame,
        ),
        _ => model_vector(&model.matrix, block, frame, mv_precision + PRECISION_OFFSET),
    };
    if force_integer { mv.to_integer() } else { mv }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divisor_reciprocals() {
        assert_eq!(resolve_divisor(1), (512, 9));
        assert_eq!(resolve_divisor(128), (512, 16));
        assert_eq!(resolve_divisor(3 << 20), (341, 30));
    }

    #[test]
    fn identity_shear() {
        let mut model = WarpModel::of_matrix(kind::AFFINE, IDENTITY);
        model.finish();
        assert_eq!(model.kind, kind::IDENTITY);
        assert_eq!(model.shear, Shear::default());
        assert!(model.affine);
    }

    #[test]
    fn not_shearable() {
        let mut model = WarpModel::of_matrix(kind::AFFINE, [0, 0, 0, 0, 0, UNITY]);
        model.finish();
        assert_eq!(model.kind, kind::INVALID);
    }

    /// Samples moving with the block fit a matrix within one step of the
    /// identity (the fit's rounding terms bias it), through the vector.
    #[test]
    fn fit_translation() {
        let mv = Mv { y: 8, x: -16 };
        let samples: Vec<[[i32; 2]; 2]> = [[-8, 40], [40, -8], [-8, -8]]
            .iter()
            .map(|&[x, y]| [[x, y], [x + mv.x, y + mv.y]])
            .collect();
        let mut model = WarpModel::default();
        model.fit(&samples, (4, 4), mv, (8, 8));
        let m = model.matrix;
        assert!((m[2] - UNITY).abs() <= 64 && (m[5] - UNITY).abs() <= 64);
        assert_eq!((m[3], m[4]), (0, 0));
        let area = BlockArea {
            row: 8,
            column: 8,
            bw4: 4,
            bh4: 4,
        };
        assert_eq!(model_vector(&model.matrix, area, (64, 64), 6), mv);
    }

    #[test]
    fn kinds() {
        assert_eq!(matrix_kind(&[8, 0, UNITY, 0, 0, UNITY]), kind::TRANSLATION);
        assert_eq!(
            matrix_kind(&[0, 0, UNITY + 64, 64, -64, UNITY + 64]),
            kind::ROT_ZOOM
        );
        assert_eq!(
            matrix_kind(&[0, 0, UNITY + 64, 64, 64, UNITY + 64]),
            kind::AFFINE
        );
    }
}
