//! Warped motion (AV1 §7.11.3.5–8): the affine block warp filter, the shear
//! setup that validates a warp model and derives its filter steps, and the
//! least-squares warp estimation of local warped motion from neighbouring
//! motion vectors.
//!
//! A warp model `[i32; 6]` maps luma sample `(x, y)` to
//! `(m[2] x + m[3] y + m[0], m[4] x + m[5] y + m[1])` in 1/65536 samples
//! (`WARPEDMODEL_PREC_BITS`).
//!
//! `Warped_Filters` and `Div_Lut`: libaom v3.15.3
//! `av1/common/warped_motion.c` (BSD-2-Clause, see `THIRD_PARTY.md`).

use nc_common::{PlaneView, PlaneViewMut};

use super::warp_kernels::{
    WARP_BLOCK, WINDOW_LEN, WINDOW_SIDE, WINDOW_STRIDE, WarpOut, WarpPhases, window_span,
};
use super::{INTER_ROUND1_COMPOUND, INTER_ROUND1_FINAL, clip_pixel, round2, round2_signed};
use crate::Dsp;

/// `WARPEDMODEL_PREC_BITS`: precision of the warp model.
const MODEL_BITS: u32 = 16;
/// One sample at model precision.
const MODEL_ONE: i64 = 1 << MODEL_BITS;
/// `WARP_PARAM_REDUCE_BITS`: the shear parameters keep only multiples of 64.
const REDUCE_BITS: u32 = 6;
/// `WARPEDDIFF_PREC_BITS`: model bits below the filter phase.
const DIFF_BITS: u32 = MODEL_BITS - WARPEDPIXEL_BITS;
/// `WARPEDPIXEL_PREC_BITS`: the warp filter has 64 phases per sample.
const WARPEDPIXEL_BITS: u32 = 6;
/// `WARPEDPIXEL_PREC_SHIFTS`: offset of phase 0 in [`WARPED_FILTERS`],
/// whose phases run over `[-1, 2)` samples.
const PHASE_OFFSET: i32 = 1 << WARPEDPIXEL_BITS;
/// Warp filter taps.
const TAPS: usize = 8;
/// `DIV_LUT_BITS`: bits of a divisor's mantissa that index [`DIV_LUT`].
const DIV_LUT_BITS: u32 = 8;
/// `DIV_LUT_PREC_BITS`: precision of the reciprocals in [`DIV_LUT`].
const DIV_LUT_PREC_BITS: u32 = 14;
/// `WARPEDMODEL_NONDIAGAFFINE_CLAMP`: bound of the estimated shear terms.
const NONDIAGONAL_CLAMP: i64 = 1 << (MODEL_BITS - 3);
/// `WARPEDMODEL_TRANS_CLAMP`: bound of the estimated translation.
const TRANSLATION_CLAMP: i64 = 128 << MODEL_BITS;
/// `LS_MV_MAX`: samples whose motion differs by this much (1/8 samples)
/// from the block's are left out of the estimation.
const LS_MV_MAX: i32 = 256;
/// `LS_STEP`: the sample positions are multiples of 8 (1/8 samples of
/// whole luma samples), which the least-squares sums divide out.
const LS_STEP: i64 = 8;
/// `LS_MAT_DOWN_BITS`: extra down-shift of the least-squares sums.
const LS_MAT_DOWN_BITS: u32 = 2;

/// `Warped_Filters` (AV1 §7.11.3.5): 8 taps at 1/64-sample phases over
/// `[-1, 2)` samples.
#[rustfmt::skip]
const WARPED_FILTERS: [[i8; 8]; 193] = [
    [0, 0, 127, 1, 0, 0, 0, 0],
    [0, -1, 127, 2, 0, 0, 0, 0],
    [1, -3, 127, 4, -1, 0, 0, 0],
    [1, -4, 126, 6, -2, 1, 0, 0],
    [1, -5, 126, 8, -3, 1, 0, 0],
    [1, -6, 125, 11, -4, 1, 0, 0],
    [1, -7, 124, 13, -4, 1, 0, 0],
    [2, -8, 123, 15, -5, 1, 0, 0],
    [2, -9, 122, 18, -6, 1, 0, 0],
    [2, -10, 121, 20, -6, 1, 0, 0],
    [2, -11, 120, 22, -7, 2, 0, 0],
    [2, -12, 119, 25, -8, 2, 0, 0],
    [3, -13, 117, 27, -8, 2, 0, 0],
    [3, -13, 116, 29, -9, 2, 0, 0],
    [3, -14, 114, 32, -10, 3, 0, 0],
    [3, -15, 113, 35, -10, 2, 0, 0],
    [3, -15, 111, 37, -11, 3, 0, 0],
    [3, -16, 109, 40, -11, 3, 0, 0],
    [3, -16, 108, 42, -12, 3, 0, 0],
    [4, -17, 106, 45, -13, 3, 0, 0],
    [4, -17, 104, 47, -13, 3, 0, 0],
    [4, -17, 102, 50, -14, 3, 0, 0],
    [4, -17, 100, 52, -14, 3, 0, 0],
    [4, -18, 98, 55, -15, 4, 0, 0],
    [4, -18, 96, 58, -15, 3, 0, 0],
    [4, -18, 94, 60, -16, 4, 0, 0],
    [4, -18, 91, 63, -16, 4, 0, 0],
    [4, -18, 89, 65, -16, 4, 0, 0],
    [4, -18, 87, 68, -17, 4, 0, 0],
    [4, -18, 85, 70, -17, 4, 0, 0],
    [4, -18, 82, 73, -17, 4, 0, 0],
    [4, -18, 80, 75, -17, 4, 0, 0],
    [4, -18, 78, 78, -18, 4, 0, 0],
    [4, -17, 75, 80, -18, 4, 0, 0],
    [4, -17, 73, 82, -18, 4, 0, 0],
    [4, -17, 70, 85, -18, 4, 0, 0],
    [4, -17, 68, 87, -18, 4, 0, 0],
    [4, -16, 65, 89, -18, 4, 0, 0],
    [4, -16, 63, 91, -18, 4, 0, 0],
    [4, -16, 60, 94, -18, 4, 0, 0],
    [3, -15, 58, 96, -18, 4, 0, 0],
    [4, -15, 55, 98, -18, 4, 0, 0],
    [3, -14, 52, 100, -17, 4, 0, 0],
    [3, -14, 50, 102, -17, 4, 0, 0],
    [3, -13, 47, 104, -17, 4, 0, 0],
    [3, -13, 45, 106, -17, 4, 0, 0],
    [3, -12, 42, 108, -16, 3, 0, 0],
    [3, -11, 40, 109, -16, 3, 0, 0],
    [3, -11, 37, 111, -15, 3, 0, 0],
    [2, -10, 35, 113, -15, 3, 0, 0],
    [3, -10, 32, 114, -14, 3, 0, 0],
    [2, -9, 29, 116, -13, 3, 0, 0],
    [2, -8, 27, 117, -13, 3, 0, 0],
    [2, -8, 25, 119, -12, 2, 0, 0],
    [2, -7, 22, 120, -11, 2, 0, 0],
    [1, -6, 20, 121, -10, 2, 0, 0],
    [1, -6, 18, 122, -9, 2, 0, 0],
    [1, -5, 15, 123, -8, 2, 0, 0],
    [1, -4, 13, 124, -7, 1, 0, 0],
    [1, -4, 11, 125, -6, 1, 0, 0],
    [1, -3, 8, 126, -5, 1, 0, 0],
    [1, -2, 6, 126, -4, 1, 0, 0],
    [0, -1, 4, 127, -3, 1, 0, 0],
    [0, 0, 2, 127, -1, 0, 0, 0],
    [0, 0, 0, 127, 1, 0, 0, 0],
    [0, 0, -1, 127, 2, 0, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -5, 127, 6, -2, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [-1, 2, -7, 126, 11, -4, 2, -1],
    [-1, 3, -8, 125, 13, -5, 2, -1],
    [-1, 3, -10, 124, 16, -6, 3, -1],
    [-1, 4, -11, 123, 18, -7, 3, -1],
    [-1, 4, -12, 122, 20, -7, 3, -1],
    [-1, 4, -13, 121, 23, -8, 3, -1],
    [-2, 5, -14, 120, 25, -9, 4, -1],
    [-1, 5, -15, 119, 27, -10, 4, -1],
    [-1, 5, -16, 118, 30, -11, 4, -1],
    [-2, 6, -17, 116, 33, -12, 5, -1],
    [-2, 6, -17, 114, 35, -12, 5, -1],
    [-2, 6, -18, 113, 38, -13, 5, -1],
    [-2, 7, -19, 111, 41, -14, 6, -2],
    [-2, 7, -19, 110, 43, -15, 6, -2],
    [-2, 7, -20, 108, 46, -15, 6, -2],
    [-2, 7, -20, 106, 49, -16, 6, -2],
    [-2, 7, -21, 104, 51, -16, 7, -2],
    [-2, 7, -21, 102, 54, -17, 7, -2],
    [-2, 8, -21, 100, 56, -18, 7, -2],
    [-2, 8, -22, 98, 59, -18, 7, -2],
    [-2, 8, -22, 96, 62, -19, 7, -2],
    [-2, 8, -22, 94, 64, -19, 7, -2],
    [-2, 8, -22, 91, 67, -20, 8, -2],
    [-2, 8, -22, 89, 69, -20, 8, -2],
    [-2, 8, -22, 87, 72, -21, 8, -2],
    [-2, 8, -21, 84, 74, -21, 8, -2],
    [-2, 8, -22, 82, 77, -21, 8, -2],
    [-2, 8, -21, 79, 79, -21, 8, -2],
    [-2, 8, -21, 77, 82, -22, 8, -2],
    [-2, 8, -21, 74, 84, -21, 8, -2],
    [-2, 8, -21, 72, 87, -22, 8, -2],
    [-2, 8, -20, 69, 89, -22, 8, -2],
    [-2, 8, -20, 67, 91, -22, 8, -2],
    [-2, 7, -19, 64, 94, -22, 8, -2],
    [-2, 7, -19, 62, 96, -22, 8, -2],
    [-2, 7, -18, 59, 98, -22, 8, -2],
    [-2, 7, -18, 56, 100, -21, 8, -2],
    [-2, 7, -17, 54, 102, -21, 7, -2],
    [-2, 7, -16, 51, 104, -21, 7, -2],
    [-2, 6, -16, 49, 106, -20, 7, -2],
    [-2, 6, -15, 46, 108, -20, 7, -2],
    [-2, 6, -15, 43, 110, -19, 7, -2],
    [-2, 6, -14, 41, 111, -19, 7, -2],
    [-1, 5, -13, 38, 113, -18, 6, -2],
    [-1, 5, -12, 35, 114, -17, 6, -2],
    [-1, 5, -12, 33, 116, -17, 6, -2],
    [-1, 4, -11, 30, 118, -16, 5, -1],
    [-1, 4, -10, 27, 119, -15, 5, -1],
    [-1, 4, -9, 25, 120, -14, 5, -2],
    [-1, 3, -8, 23, 121, -13, 4, -1],
    [-1, 3, -7, 20, 122, -12, 4, -1],
    [-1, 3, -7, 18, 123, -11, 4, -1],
    [-1, 3, -6, 16, 124, -10, 3, -1],
    [-1, 2, -5, 13, 125, -8, 3, -1],
    [-1, 2, -4, 11, 126, -7, 2, -1],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -2, 6, 127, -5, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, 0, 2, 127, -1, 0, 0],
    [0, 0, 0, 1, 127, 0, 0, 0],
    [0, 0, 0, -1, 127, 2, 0, 0],
    [0, 0, 1, -3, 127, 4, -1, 0],
    [0, 0, 1, -4, 126, 6, -2, 1],
    [0, 0, 1, -5, 126, 8, -3, 1],
    [0, 0, 1, -6, 125, 11, -4, 1],
    [0, 0, 1, -7, 124, 13, -4, 1],
    [0, 0, 2, -8, 123, 15, -5, 1],
    [0, 0, 2, -9, 122, 18, -6, 1],
    [0, 0, 2, -10, 121, 20, -6, 1],
    [0, 0, 2, -11, 120, 22, -7, 2],
    [0, 0, 2, -12, 119, 25, -8, 2],
    [0, 0, 3, -13, 117, 27, -8, 2],
    [0, 0, 3, -13, 116, 29, -9, 2],
    [0, 0, 3, -14, 114, 32, -10, 3],
    [0, 0, 3, -15, 113, 35, -10, 2],
    [0, 0, 3, -15, 111, 37, -11, 3],
    [0, 0, 3, -16, 109, 40, -11, 3],
    [0, 0, 3, -16, 108, 42, -12, 3],
    [0, 0, 4, -17, 106, 45, -13, 3],
    [0, 0, 4, -17, 104, 47, -13, 3],
    [0, 0, 4, -17, 102, 50, -14, 3],
    [0, 0, 4, -17, 100, 52, -14, 3],
    [0, 0, 4, -18, 98, 55, -15, 4],
    [0, 0, 4, -18, 96, 58, -15, 3],
    [0, 0, 4, -18, 94, 60, -16, 4],
    [0, 0, 4, -18, 91, 63, -16, 4],
    [0, 0, 4, -18, 89, 65, -16, 4],
    [0, 0, 4, -18, 87, 68, -17, 4],
    [0, 0, 4, -18, 85, 70, -17, 4],
    [0, 0, 4, -18, 82, 73, -17, 4],
    [0, 0, 4, -18, 80, 75, -17, 4],
    [0, 0, 4, -18, 78, 78, -18, 4],
    [0, 0, 4, -17, 75, 80, -18, 4],
    [0, 0, 4, -17, 73, 82, -18, 4],
    [0, 0, 4, -17, 70, 85, -18, 4],
    [0, 0, 4, -17, 68, 87, -18, 4],
    [0, 0, 4, -16, 65, 89, -18, 4],
    [0, 0, 4, -16, 63, 91, -18, 4],
    [0, 0, 4, -16, 60, 94, -18, 4],
    [0, 0, 3, -15, 58, 96, -18, 4],
    [0, 0, 4, -15, 55, 98, -18, 4],
    [0, 0, 3, -14, 52, 100, -17, 4],
    [0, 0, 3, -14, 50, 102, -17, 4],
    [0, 0, 3, -13, 47, 104, -17, 4],
    [0, 0, 3, -13, 45, 106, -17, 4],
    [0, 0, 3, -12, 42, 108, -16, 3],
    [0, 0, 3, -11, 40, 109, -16, 3],
    [0, 0, 3, -11, 37, 111, -15, 3],
    [0, 0, 2, -10, 35, 113, -15, 3],
    [0, 0, 3, -10, 32, 114, -14, 3],
    [0, 0, 2, -9, 29, 116, -13, 3],
    [0, 0, 2, -8, 27, 117, -13, 3],
    [0, 0, 2, -8, 25, 119, -12, 2],
    [0, 0, 2, -7, 22, 120, -11, 2],
    [0, 0, 1, -6, 20, 121, -10, 2],
    [0, 0, 1, -6, 18, 122, -9, 2],
    [0, 0, 1, -5, 15, 123, -8, 2],
    [0, 0, 1, -4, 13, 124, -7, 1],
    [0, 0, 1, -4, 11, 125, -6, 1],
    [0, 0, 1, -3, 8, 126, -5, 1],
    [0, 0, 1, -2, 6, 126, -4, 1],
    [0, 0, 0, -1, 4, 127, -3, 1],
    [0, 0, 0, 0, 2, 127, -1, 0],
    [0, 0, 0, 0, 2, 127, -1, 0],
];
/// `Div_Lut` (AV1 §7.11.3.7): `2^14 / (1 + i / 256)`, rounded.
#[rustfmt::skip]
const DIV_LUT: [u16; 257] = [
    16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768, 15709,
    15650, 15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142, 15087, 15033,
    14980, 14926, 14873, 14821, 14769, 14717, 14665, 14614, 14564, 14513, 14463, 14413,
    14364, 14315, 14266, 14218, 14170, 14122, 14075, 14028, 13981, 13935, 13888, 13843,
    13797, 13752, 13707, 13662, 13618, 13574, 13530, 13487, 13443, 13400, 13358, 13315,
    13273, 13231, 13190, 13148, 13107, 13066, 13026, 12985, 12945, 12906, 12866, 12827,
    12788, 12749, 12710, 12672, 12633, 12596, 12558, 12520, 12483, 12446, 12409, 12373,
    12336, 12300, 12264, 12228, 12193, 12157, 12122, 12087, 12053, 12018, 11984, 11950,
    11916, 11882, 11848, 11815, 11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555,
    11523, 11491, 11460, 11429, 11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185,
    11155, 11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894, 10866, 10838,
    10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538, 10512,
    10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255, 10230, 10205,
    10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986, 9963, 9939, 9916,
    9892, 9869, 9846, 9823, 9800, 9777, 9754, 9732, 9709, 9687, 9664, 9642,
    9620, 9598, 9576, 9554, 9533, 9511, 9489, 9468, 9447, 9425, 9404, 9383,
    9362, 9341, 9321, 9300, 9279, 9259, 9239, 9218, 9198, 9178, 9158, 9138,
    9118, 9098, 9079, 9059, 9039, 9020, 9001, 8981, 8962, 8943, 8924, 8905,
    8886, 8867, 8849, 8830, 8812, 8793, 8775, 8756, 8738, 8720, 8702, 8684,
    8666, 8648, 8630, 8613, 8595, 8577, 8560, 8542, 8525, 8508, 8490, 8473,
    8456, 8439, 8422, 8405, 8389, 8372, 8355, 8339, 8322, 8306, 8289, 8273,
    8257, 8240, 8224, 8208, 8192,
];

/// The shear parameters of a valid warp model (AV1 §7.11.3.6): the steps
/// of the horizontal (`alpha`, `beta`) and vertical (`gamma`, `delta`)
/// filter phases per column and row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1Shear {
    /// Horizontal phase step per column.
    pub alpha: i32,
    /// Horizontal phase step per row.
    pub beta: i32,
    /// Vertical phase step per column.
    pub gamma: i32,
    /// Vertical phase step per row.
    pub delta: i32,
}

/// `FloorLog2` of a nonzero value.
fn floor_log2(value: u64) -> u32 {
    value.ilog2()
}

/// Resolve divisor process (AV1 §7.11.3.7): `1 / d` as
/// `factor / 2^shift`, from the top 8 bits of `d`'s mantissa. Returns
/// `(factor, shift)`.
fn resolve_divisor(d: u64) -> (i64, u32) {
    let n = floor_log2(d);
    let e = d - (1 << n);
    let f = if n > DIV_LUT_BITS {
        (e + (1 << (n - DIV_LUT_BITS - 1))) >> (n - DIV_LUT_BITS)
    } else {
        e << (DIV_LUT_BITS - n)
    };
    (i64::from(DIV_LUT[f as usize]), n + DIV_LUT_PREC_BITS)
}

/// Clamp to the `i16` range, as the shear parameters are stored.
fn clamp_i16(value: i64) -> i64 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX))
}

/// Keep only the top bits of a shear parameter (`WARP_PARAM_REDUCE_BITS`).
fn reduce(value: i64) -> i32 {
    (round2_signed(value, REDUCE_BITS) << REDUCE_BITS) as i32
}

/// Setup shear process (AV1 §7.11.3.6): the shear parameters of warp model
/// `m`, or `None` when the model is not valid for the warp filter
/// (`warpValid` 0: `m[2]` not positive, or a shear too large for 8-tap
/// filtering of an 8x8 block).
#[must_use]
pub fn av1_setup_shear(m: &[i32; 6]) -> Option<Av1Shear> {
    if m[2] <= 0 {
        return None;
    }
    let m = m.map(i64::from);
    let alpha0 = clamp_i16(m[2] - MODEL_ONE);
    let beta0 = clamp_i16(m[3]);
    let (factor, shift) = resolve_divisor(m[2] as u64);
    let v = (m[4] << MODEL_BITS) * factor;
    let gamma0 = clamp_i16(round2_signed(v, shift));
    let w = m[3] * m[4] * factor;
    let delta0 = clamp_i16(m[5] - round2_signed(w, shift) - MODEL_ONE);
    let shear = Av1Shear {
        alpha: reduce(alpha0),
        beta: reduce(beta0),
        gamma: reduce(gamma0),
        delta: reduce(delta0),
    };
    let limit = 1 << MODEL_BITS;
    let valid = 4 * shear.alpha.abs() + 7 * shear.beta.abs() < limit
        && 4 * shear.gamma.abs() + 4 * shear.delta.abs() < limit;
    valid.then_some(shear)
}

/// One neighbour sample for warp estimation: the centre of a neighbouring
/// block relative to the current block's top-left sample, before
/// (`source`) and after (`destination`) the neighbour's motion vector, as
/// (x, y) in 1/8 luma samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Av1WarpSample {
    /// Neighbour centre (x, y).
    pub source: (i32, i32),
    /// Neighbour centre moved by its motion vector (x, y).
    pub destination: (i32, i32),
}

/// `LS_SQUARE(a)`: one least-squares square term, rounded for the
/// positions' multiples of [`LS_STEP`].
fn ls_square(a: i64) -> i64 {
    (a * a * 4 + a * 4 * LS_STEP + LS_STEP * LS_STEP * 2) >> (2 + LS_MAT_DOWN_BITS)
}

/// `LS_PRODUCT1(a, b)`.
fn ls_product1(a: i64, b: i64) -> i64 {
    (a * b * 4 + (a + b) * 2 * LS_STEP + LS_STEP * LS_STEP) >> (2 + LS_MAT_DOWN_BITS)
}

/// `LS_PRODUCT2(a, b)`.
fn ls_product2(a: i64, b: i64) -> i64 {
    (a * b * 4 + (a + b) * 2 * LS_STEP + LS_STEP * LS_STEP * 2) >> (2 + LS_MAT_DOWN_BITS)
}

/// `p / det` at model precision, as `p * inverse >> shift`, for a diagonal
/// (`diagonal`) or off-diagonal model term, clamped to the warp model's
/// range around 1 or 0.
fn divide_term(p: i64, inverse: i64, shift: u32, diagonal: bool) -> i32 {
    let product = p * inverse;
    let value = if shift == 0 {
        product
    } else {
        round2_signed(product, shift)
    };
    let centre = if diagonal { MODEL_ONE } else { 0 };
    value.clamp(
        centre - NONDIAGONAL_CLAMP + 1,
        centre + NONDIAGONAL_CLAMP - 1,
    ) as i32
}

/// Warp estimation process (AV1 §7.11.3.8): the local warp model of a
/// `width`x`height` (luma) block at luma sample `(block_x, block_y)` with
/// motion vector
/// `mv` = (row, col) in 1/8 samples, fitted by least squares to the
/// neighbour `samples`. `None` when the fit is singular (`LocalValid` 0);
/// the model must still pass [`av1_setup_shear`] to be used.
#[must_use]
pub fn av1_warp_estimation(
    samples: &[Av1WarpSample],
    (mv_row, mv_col): (i32, i32),
    (block_x, block_y): (u32, u32),
    (width, height): (u32, u32),
) -> Option<[i32; 6]> {
    // Origins: the block centre, and the block centre moved by its motion
    // vector.
    let mid_y = (height / 2) as i32 - 1;
    let mid_x = (width / 2) as i32 - 1;
    let (suy, sux) = (mid_y * 8, mid_x * 8);
    let (duy, dux) = (suy + mv_row, sux + mv_col);
    let (mut a00, mut a01, mut a11) = (0i64, 0i64, 0i64);
    let (mut bx0, mut bx1, mut by0, mut by1) = (0i64, 0i64, 0i64, 0i64);
    for sample in samples {
        let sx = sample.source.0 - sux;
        let sy = sample.source.1 - suy;
        let dx = sample.destination.0 - dux;
        let dy = sample.destination.1 - duy;
        if (sx - dx).abs() < LS_MV_MAX && (sy - dy).abs() < LS_MV_MAX {
            let (sx, sy, dx, dy) = (i64::from(sx), i64::from(sy), i64::from(dx), i64::from(dy));
            a00 += ls_square(sx);
            a01 += ls_product1(sx, sy);
            a11 += ls_square(sy);
            bx0 += ls_product2(sx, dx);
            bx1 += ls_product1(sy, dx);
            by0 += ls_product1(sx, dy);
            by1 += ls_product2(sy, dy);
        }
    }
    let det = a00 * a11 - a01 * a01;
    if det == 0 {
        return None;
    }
    let (factor, shift) = resolve_divisor(det.unsigned_abs());
    let mut inverse = factor * det.signum();
    // A tiny determinant (|det| < 4) needs a left shift. The spec (and
    // dav1d) keep the full value; libaom stores it in 16 bits and would
    // wrap, a case no realistic sample set reaches.
    let shift = if shift < MODEL_BITS {
        inverse <<= MODEL_BITS - shift;
        0
    } else {
        shift - MODEL_BITS
    };
    let divide = |p: i64, diagonal: bool| divide_term(p, inverse, shift, diagonal);
    let mut model = [0i32; 6];
    model[2] = divide(a11 * bx0 - a01 * bx1, true);
    model[3] = divide(-a01 * bx0 + a00 * bx1, false);
    model[4] = divide(a11 * by0 - a01 * by1, false);
    model[5] = divide(-a01 * by0 + a00 * by1, true);
    // The block centre in the frame keeps the block's own motion vector.
    let isuy = i64::from(block_y) + i64::from(mid_y);
    let isux = i64::from(block_x) + i64::from(mid_x);
    let (m2, m3, m4, m5) = (
        i64::from(model[2]),
        i64::from(model[3]),
        i64::from(model[4]),
        i64::from(model[5]),
    );
    let vx = i64::from(mv_col) * (1 << (MODEL_BITS - 3)) - (isux * (m2 - MODEL_ONE) + isuy * m3);
    let vy = i64::from(mv_row) * (1 << (MODEL_BITS - 3)) - (isux * m4 + isuy * (m5 - MODEL_ONE));
    model[0] = vx.clamp(-TRANSLATION_CLAMP, TRANSLATION_CLAMP - 1) as i32;
    model[1] = vy.clamp(-TRANSLATION_CLAMP, TRANSLATION_CLAMP - 1) as i32;
    Some(model)
}

/// [`WARPED_FILTERS`] with each filter's taps in the order 0, 2, 4, 6, 1,
/// 3, 5, 7: the byte multiply-add kernels pair every large centre tap
/// (3, 4) with a small outer one (1, 6), so no pair or group of four
/// exceeds 16 bits (`warp_kernels` tests).
#[cfg(any(target_arch = "x86_64", test))]
pub(crate) const WARPED_FILTERS_PAIRED: [[i8; TAPS]; WARPED_FILTERS.len()] = {
    const ORDER: [usize; TAPS] = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut paired = [[0i8; TAPS]; WARPED_FILTERS.len()];
    let mut phase = 0;
    while phase < WARPED_FILTERS.len() {
        let mut tap = 0;
        while tap < TAPS {
            paired[phase][tap] = WARPED_FILTERS[phase][ORDER[tap]];
            tap += 1;
        }
        phase += 1;
    }
    paired
};

/// [`WARPED_FILTERS`] widened to 16 bits, as the kernels multiply them.
pub(crate) const WARPED_FILTERS_I16: [[i16; TAPS]; WARPED_FILTERS.len()] = {
    let mut wide = [[0i16; TAPS]; WARPED_FILTERS.len()];
    let mut phase = 0;
    while phase < WARPED_FILTERS.len() {
        let mut tap = 0;
        while tap < TAPS {
            wide[phase][tap] = WARPED_FILTERS[phase][tap] as i16;
            tap += 1;
        }
        phase += 1;
    }
    wide
};

/// Byte coefficients for the baseline NEON warp kernel. Checked indexing
/// preserves the scalar table contract for invalid internal phases.
#[cfg(target_arch = "aarch64")]
pub(super) fn warp_taps_i8(phase: i32) -> &'static [i8; TAPS] {
    &WARPED_FILTERS[warp_filter_index(phase)]
}

/// The warp filter taps for model-precision phase `phase` (relative to the
/// sample, within `[-1, 2)` samples).
pub(crate) fn warp_taps_i16(phase: i32) -> &'static [i16; TAPS] {
    &WARPED_FILTERS_I16[warp_filter_index(phase)]
}

/// The [`WARPED_FILTERS`] row of model-precision phase `phase`.
pub(crate) fn warp_filter_index(phase: i32) -> usize {
    (round2(phase, DIFF_BITS) + PHASE_OFFSET) as usize
}

/// [`warp_filter_index`] is `(phase + PHASE_ROUND) >> PHASE_SHIFT` (for
/// kernels that compute it in vector lanes).
#[cfg(target_arch = "x86_64")]
pub(crate) const PHASE_SHIFT: u32 = DIFF_BITS;
/// See [`PHASE_SHIFT`]: the rounding and the offset of phase 0.
#[cfg(target_arch = "x86_64")]
pub(crate) const PHASE_ROUND: i32 = (1 << (DIFF_BITS - 1)) + (PHASE_OFFSET << DIFF_BITS);

/// Block warp process (AV1 §7.11.3.5) for the 8x8 block at plane sample
/// `(x, y)`: the source window around the block's warped centre (samples
/// clamped to the plane) and the filter phases, warped by `dsp`'s kernel
/// into `Round2(sum, round1)` for each of its samples, row by row.
fn block_warp(
    dsp: &Dsp,
    reference: &PlaneView<'_>,
    (model, shear): (&[i32; 6], Av1Shear),
    (x, y): (usize, usize),
    (sub_x, sub_y): (bool, bool),
    round1: u32,
) -> WarpOut {
    let m64 = model.map(i64::from);
    let src_x = ((x + WARP_BLOCK / 2) << u32::from(sub_x)) as i64;
    let src_y = ((y + WARP_BLOCK / 2) << u32::from(sub_y)) as i64;
    let dst_x = m64[2] * src_x + m64[3] * src_y + m64[0];
    let dst_y = m64[4] * src_x + m64[5] * src_y + m64[1];
    let x4 = dst_x >> u32::from(sub_x);
    let y4 = dst_y >> u32::from(sub_y);
    let ix4 = x4 >> MODEL_BITS;
    let iy4 = y4 >> MODEL_BITS;
    let half = (WARP_BLOCK / 2) as i32;
    let reduce_mask = !((1 << REDUCE_BITS) - 1);
    let sx4 = ((x4 & (MODEL_ONE - 1)) as i32 - half * (shear.alpha + shear.beta)) & reduce_mask;
    let sy4 = ((y4 & (MODEL_ONE - 1)) as i32 - half * (shear.gamma + shear.delta)) & reduce_mask;
    let phases = WarpPhases {
        x: (sx4, shear.alpha, shear.beta),
        y: (sy4, shear.gamma, shear.delta),
    };
    // Window row 0 is source row iy4 - 7, column 0 source column ix4 - 7.
    let reach = (WINDOW_SIDE / 2) as i64;
    let corner = (ix4 - reach, iy4 - reach);
    let mut out = [0; WARP_BLOCK * WARP_BLOCK];
    if let Some(start) = interior_window(reference, corner) {
        // The window lies inside the plane: read it in place.
        dsp.av1_warp8x8(
            (&reference.data()[start..], reference.stride()),
            &phases,
            round1,
            &mut out,
        );
    } else {
        let window = clamped_window(reference, corner);
        dsp.av1_warp8x8((&window, WINDOW_STRIDE), &phases, round1, &mut out);
    }
    out
}

/// Where the window whose top-left sample is `(left, top)` starts in
/// `reference`'s data, when all of its samples lie inside the plane and
/// every row can be read as a whole vector of [`WINDOW_STRIDE`] samples.
fn interior_window(reference: &PlaneView<'_>, (left, top): (i64, i64)) -> Option<usize> {
    let side = WINDOW_SIDE as i64;
    let inside = left >= 0
        && top >= 0
        && left + side <= reference.width() as i64
        && top + side <= reference.height() as i64;
    let stride = reference.stride();
    if !inside || stride < WINDOW_STRIDE {
        return None;
    }
    let start = top as usize * stride + left as usize;
    (start + window_span(stride) <= reference.data().len()).then_some(start)
}

/// The [`WINDOW_SIDE`]-square source window whose top-left sample is
/// `(left, top)` (partly outside the plane), every sample clamped to the
/// plane as §7.11.3.5 reads them, rows [`WINDOW_STRIDE`] apart.
fn clamped_window(reference: &PlaneView<'_>, (left, top): (i64, i64)) -> [u8; WINDOW_LEN] {
    let (width, height) = (reference.width() as i64, reference.height() as i64);
    let mut window = [0u8; WINDOW_LEN];
    for (i, row) in window
        .as_chunks_mut::<WINDOW_STRIDE>()
        .0
        .iter_mut()
        .enumerate()
    {
        let source = reference.row((top + i as i64).clamp(0, height - 1) as usize);
        for (c, sample) in row[..WINDOW_SIDE].iter_mut().enumerate() {
            *sample = source[(left + c as i64).clamp(0, width - 1) as usize];
        }
    }
    window
}

/// Warp the `w`x`h` plane block at plane sample `(x, y)` by `model`
/// (8x8 blocks of §7.11.3.5; the last ones are cropped to the block),
/// `store_row(row, column, values)` receiving each row of a block's
/// outputs that lies in the block.
#[allow(clippy::too_many_arguments, reason = "the warp process's inputs")]
fn warp(
    dsp: &Dsp,
    reference: &PlaneView<'_>,
    model: &[i32; 6],
    shear: Av1Shear,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    sub: (bool, bool),
    round1: u32,
    mut store_row: impl FnMut(usize, usize, &[i32]),
) {
    assert!(
        reference.width() > 0 && reference.height() > 0,
        "empty warp reference"
    );
    for i8 in (0..h).step_by(WARP_BLOCK) {
        for j8 in (0..w).step_by(WARP_BLOCK) {
            let out = block_warp(
                dsp,
                reference,
                (model, shear),
                (x + j8, y + i8),
                sub,
                round1,
            );
            let columns = WARP_BLOCK.min(w - j8);
            for (r, row) in out
                .as_chunks::<WARP_BLOCK>()
                .0
                .iter()
                .take(h - i8)
                .enumerate()
            {
                store_row(i8 + r, j8, &row[..columns]);
            }
        }
    }
}

/// Warped prediction (AV1 §7.11.3.5) of the `w`x`h` block at plane sample
/// `(x, y)` of a plane subsampled by `sub` = (`sub_x`, `sub_y`), from the
/// whole `reference` plane (its samples clamped at the edges), at final
/// 8-bit precision into `dst`.
///
/// # Panics
///
/// If `dst` does not cover the block or the reference is empty.
pub fn av1_warp(
    dsp: &Dsp,
    reference: &PlaneView<'_>,
    model: &[i32; 6],
    shear: Av1Shear,
    position: (usize, usize),
    (w, h): (usize, usize),
    sub: (bool, bool),
    dst: &mut PlaneViewMut<'_>,
) {
    assert!(
        dst.width() >= w && dst.height() >= h,
        "warp destination smaller than {w}x{h}"
    );
    warp(
        dsp,
        reference,
        model,
        shear,
        position,
        (w, h),
        sub,
        INTER_ROUND1_FINAL,
        |r, c, values| {
            let line = &mut dst.row_mut(r)[c..c + values.len()];
            for (out, &v) in line.iter_mut().zip(values) {
                *out = clip_pixel(v);
            }
        },
    );
}

/// [`av1_warp`] at compound precision into `dst` (`h` rows of `w` at
/// `dst_stride`).
///
/// # Panics
///
/// If `dst` is too short or the reference is empty.
#[allow(clippy::too_many_arguments, reason = "the warp process's inputs")]
pub fn av1_warp_compound(
    dsp: &Dsp,
    reference: &PlaneView<'_>,
    model: &[i32; 6],
    shear: Av1Shear,
    position: (usize, usize),
    (w, h): (usize, usize),
    sub: (bool, bool),
    dst: &mut [i16],
    dst_stride: usize,
) {
    assert!(
        dst_stride >= w && dst.len() >= (h - 1) * dst_stride + w,
        "warp compound buffer too small for {w}x{h}"
    );
    warp(
        dsp,
        reference,
        model,
        shear,
        position,
        (w, h),
        sub,
        INTER_ROUND1_COMPOUND,
        |r, c, values| {
            let line = &mut dst[r * dst_stride + c..][..values.len()];
            for (out, &v) in line.iter_mut().zip(values) {
                *out = v as i16;
            }
        },
    );
}
