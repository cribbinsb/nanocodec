//! Intra bi-prediction weights: for each of the seven angle classes, the
//! weight (out of 128) of the primary prediction at each point of a 16x16
//! grid, from its distances to the two edges along the angle.
//!
//! Ported from dav2d `src/ibp.c` (`dav2d_init_ibp_weights`, `fast_div32`;
//! commit 237229e, BSD-2-Clause, see `docs/licenses/dav2d`), evaluated at
//! compile time instead of at library init.

use super::tables::DIV_RECIP;

/// Angle classes with their own weights.
const IBP_MODES: usize = 7;
/// Side of the weight grid; larger blocks step through it.
const GRID: usize = 16;
/// Each class's step along the edge per unit distance, 1/64 units (AVM
/// `dr_dy_q6`-style derivatives of the classes' representative angles).
const MODE_DY: [u32; IBP_MODES] = [682, 256, 170, 128, 81, 64, 50];
/// Fraction bits of a divisor's index into [`DIV_RECIP`].
const RECIP_INDEX_BITS: u32 = 7;
/// Fraction bits of distances.
const DISTANCE_BITS: u32 = 6;
/// Shift of the weight result beyond the divisor's power of two (the
/// reciprocal's 512 is 1.0, the weight's 128 is 1.0).
const WEIGHT_SHIFT: u32 = 2;

/// `num / den` scaled to 128 by the reciprocal table (`fast_div32`).
const fn fast_div32(num: u32, den: u32) -> u8 {
    let shift = den.ilog2();
    let rem = den - (1 << shift);
    let index = ((rem << RECIP_INDEX_BITS) + (1 << (shift - 1))) >> shift;
    let shift = shift + WEIGHT_SHIFT;
    ((num * DIV_RECIP[index as usize] as u32 + ((1 << shift) >> 1)) >> shift) as u8
}

/// The weight grids.
const fn build() -> [[[u8; GRID]; GRID]; IBP_MODES] {
    let mut out = [[[0; GRID]; GRID]; IBP_MODES];
    let mut m = 0;
    while m < IBP_MODES {
        let dy = MODE_DY[m];
        let mut y = 0;
        while y < GRID {
            let yy = (y as u32 + 1) << DISTANCE_BITS;
            let mut x = 0;
            while x < GRID {
                let y_pos = dy * (x as u32 + 1);
                out[m][y][x] = fast_div32(y_pos, yy + y_pos);
                x += 1;
            }
            y += 1;
        }
        m += 1;
    }
    out
}

/// Bi-prediction weights by angle class, row and column
/// (`dav2d_ibp_weights`).
pub(super) const IBP_WEIGHTS: [[[u8; GRID]; GRID]; IBP_MODES] = build();
