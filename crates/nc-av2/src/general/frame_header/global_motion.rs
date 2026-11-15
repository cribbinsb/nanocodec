//! Global motion parameters of a frame header: each reference's warp,
//! coded relative to a base model taken from a reference frame's own
//! global motion and rescaled by distance (dav2d `parse_frame_hdr`
//! "gmv" and `rescale_matrix`, AVM `read_global_motion`).
use super::{FrameHeader, FrameType, Parser};
use crate::general::bits::{signed_subexp, uniform};
use crate::general::references::MAX_REFERENCES;
use nc_common::decode::Result;

/// Warp model types (dav2d `Dav2dWarpedMotionType`).
pub mod kind {
    /// No motion.
    pub const IDENTITY: u8 = 0;
    /// Translation only.
    pub const TRANSLATION: u8 = 1;
    /// Rotation and zoom.
    pub const ROT_ZOOM: u8 = 2;
    /// Affine.
    pub const AFFINE: u8 = 3;
}

/// The unit of the diagonal matrix entries (1.0 in Q16).
const UNITY: i32 = 1 << 16;
/// The identity matrix (translation, then the 2×2 part row by row).
pub const IDENTITY: [i32; 6] = [0, 0, UNITY, 0, 0, UNITY];
/// Coded steps of the non-translation entries (Q16 / 64) and of the
/// translation (Q16 / 8192), with their code ranges.
const ALPHA_STEP: i32 = 64;
const ALPHA_STEP_LOG2: u32 = 6;
const ALPHA_RANGE: u32 = 512;
const TRANSLATION_STEP: i32 = 8192;
const TRANSLATION_STEP_LOG2: u32 = 13;
const TRANSLATION_RANGE: u32 = 0x4000;
/// Clamps of the rescaled base model.
const TRANSLATION_CLAMP: i64 = 0x07ff_e000;
const TRANSLATION_SOURCE_CLAMP: i64 = 0x0040_0000;
const ALPHA_CLAMP: i64 = 0x7fc0;
/// Reciprocal table of the distance division (dav2d `dav2d_div_recip`):
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
/// A reciprocal at or above this is halved (keeps products in range).
const DIVISOR_HALVING: i32 = 512;

/// One reference's warp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Warp {
    /// [`kind`].
    pub kind: u8,
    /// Translation (Q16) then the 2x2 matrix (Q16), row by row.
    pub matrix: [i32; 6],
}

impl Default for Warp {
    fn default() -> Self {
        Self {
            kind: kind::IDENTITY,
            matrix: IDENTITY,
        }
    }
}

/// The frame's global motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GlobalMotion {
    /// The rank whose global motion is the base (the reference count when
    /// the identity is), and that frame's rank of the base model.
    pub base: u8,
    /// That frame's rank of the base model.
    pub base_reference: u8,
    /// Each reference rank's warp.
    pub warps: [Warp; MAX_REFERENCES],
}

/// Parse the global motion of an inter or switch frame.
pub(super) fn parse(parser: &mut Parser<'_, '_, '_, '_>, header: &mut FrameHeader) -> Result<()> {
    if parser.context.sequence.inter.global_motion && parser.bits.read_bit()? {
        let count = header.reference_count;
        let base = if header.frame_type == FrameType::Switch {
            count
        } else {
            uniform(parser.bits, u32::from(count) + 1)? as u8
        };
        header.global_motion.base = base;
        let (base_matrix, base_distance) = if base == count {
            (IDENTITY, 1)
        } else {
            let reference = parser.reference(header, base)?;
            if reference.reference_count == 0 {
                (IDENTITY, 1)
            } else {
                let rank = if reference.reference_count == 1 {
                    0
                } else {
                    uniform(parser.bits, u32::from(reference.reference_count))? as u8
                };
                header.global_motion.base_reference = rank;
                let reference = parser.reference(header, base)?;
                (
                    reference.global_motion.warps[usize::from(rank)].matrix,
                    parser.order_difference(
                        reference.order,
                        reference.reference_orders[usize::from(rank)],
                    ),
                )
            }
        };
        for rank in 0..count {
            let order = parser.reference(header, rank)?.order;
            let distance = parser.order_difference(header.order, order);
            if distance == 0 {
                continue;
            }
            let bits = &mut *parser.bits;
            let kind = if !bits.read_bit()? {
                kind::IDENTITY
            } else if bits.read_bit()? {
                kind::ROT_ZOOM
            } else {
                kind::AFFINE
            };
            if kind == kind::IDENTITY {
                continue;
            }
            let predicted = rescale(&base_matrix, base_distance, distance);
            let mut matrix = IDENTITY;
            let alpha = |bits: &mut nc_common::decode::BitReader<'_>, entry: usize| {
                let bias = if entry == 2 || entry == 5 { UNITY } else { 0 };
                signed_subexp(
                    bits,
                    (predicted[entry] - bias) >> ALPHA_STEP_LOG2,
                    ALPHA_RANGE,
                )
                .map(|value| bias + ALPHA_STEP * value)
            };
            matrix[2] = alpha(bits, 2)?;
            matrix[3] = alpha(bits, 3)?;
            if kind == kind::AFFINE {
                matrix[4] = alpha(bits, 4)?;
                matrix[5] = alpha(bits, 5)?;
            } else {
                matrix[4] = -matrix[3];
                matrix[5] = matrix[2];
            }
            for entry in 0..2 {
                matrix[entry] = TRANSLATION_STEP
                    * signed_subexp(
                        bits,
                        predicted[entry] >> TRANSLATION_STEP_LOG2,
                        TRANSLATION_RANGE,
                    )?;
            }
            header.global_motion.warps[usize::from(rank)] = Warp { kind, matrix };
            parser.checkpoint("post-gmv");
        }
    }
    parser.checkpoint("post-gmv");
    Ok(())
}

/// `1 / divisor` as a multiplier and shift (dav2d
/// `dav2d_resolve_divisor_32`).
fn reciprocal(divisor: u32) -> (i64, u32) {
    let shift = divisor.ilog2();
    let excess = divisor - (1 << shift);
    let index = if shift > DIVISOR_INDEX_BITS {
        (excess + (1 << (shift - DIVISOR_INDEX_BITS - 1))) >> (shift - DIVISOR_INDEX_BITS)
    } else {
        excess << (DIVISOR_INDEX_BITS - shift)
    };
    (
        i64::from(DIVISOR_RECIPROCALS[index as usize]),
        shift + DIVISOR_PRECISION,
    )
}

/// The base model scaled from its own distance to the current one, the
/// translation rounded to its coded step (dav2d `rescale_matrix`).
fn rescale(base: &[i32; 6], from: i32, to: i32) -> [i32; 6] {
    if from == 0 {
        return IDENTITY;
    }
    let (mut inverse, mut shift) = reciprocal(from.unsigned_abs());
    if inverse >= i64::from(DIVISOR_HALVING) {
        inverse >>= 1;
        shift -= 1;
    }
    if from < 0 {
        inverse = -inverse;
    }
    let round = (1_i64 << shift) >> 1;
    let scale = |value: i64| -> i64 {
        let product = value * inverse;
        ((product + round - i64::from(product < 0)) >> shift) * i64::from(to)
    };
    let mut matrix = [0; 6];
    for entry in 0..2 {
        let scaled = scale(
            i64::from(base[entry]).clamp(-TRANSLATION_SOURCE_CLAMP, TRANSLATION_SOURCE_CLAMP),
        );
        let half = i64::from(TRANSLATION_STEP) >> 1;
        let rounded = (scaled + half - i64::from(scaled < 0)) & !(i64::from(TRANSLATION_STEP) - 1);
        matrix[entry] = rounded.clamp(-TRANSLATION_CLAMP, TRANSLATION_CLAMP) as i32;
    }
    for entry in 2..6 {
        let bias = if entry == 2 || entry == 5 { UNITY } else { 0 };
        let scaled = scale(i64::from(base[entry] - bias));
        let half = i64::from(ALPHA_STEP) >> 1;
        let rounded = (scaled + half - i64::from(scaled < 0)) & !(i64::from(ALPHA_STEP) - 1);
        matrix[entry] = bias + rounded.clamp(-ALPHA_CLAMP, ALPHA_CLAMP) as i32;
    }
    matrix
}
