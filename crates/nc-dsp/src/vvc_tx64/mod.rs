//! H.266 64-point DCT-II transform blocks (`MaxTbSizeY` 64): a 64-point
//! side codes only its 32 lowest frequencies (§7.3.11.11 zero-out), so a
//! 64x64 transform block has a 32x32 coded region, stored here compactly
//! (row-major, 32 per row).
//!
//! - **Inverse** (normative, exact): dequantised low-frequency coefficients
//!   to the 64x64 residual, columns rounded by 7 bits and clipped to 16,
//!   then rows rounded by 12 and clipped (§8.7.4.1, with §8.7.4.5's
//!   `transMatrix`, `codec_txfm::vvc_primary::tables::DCT64`). Bit-exact
//!   with the general `vvc_primary::inverse` (tests); plain sums over the
//!   nonzero coefficient rows, written so that the compiler vectorises them.
//! - **Forward** (encoder only, approximate): the 32 low frequencies of a
//!   64-point DCT-II are close to the 32-point DCT-II of the block averaged
//!   over 2x2 samples, times `cos(πk/128)` per dimension. Two neighbours
//!   `x[2m]`, `x[2m+1]` weigh frequency `k` by `cos(π(4m+1)k/128) +
//!   cos(π(4m+3)k/128) = 2 cos(π(2m+1)k/64) cos(πk/128)`, exactly so when
//!   they are equal; their difference aliases in, most toward frequency 31,
//!   where the levels of the large smooth blocks that choose 64x64
//!   transforms are mostly zero anyway. The factor 2 per dimension is the
//!   64-point forward's one more bit of scaling per pass (rows shifted by 5,
//!   columns by 12, against 4 and 11), so the averaged block goes through
//!   the existing (SIMD) 32-point forward unchanged, then the per-frequency
//!   `cos` repair ([`REPAIR`]). An encoder's forward transform may be
//!   approximate: it only steers decisions, never the decoded picture.
//!   Accuracy against the exact low half is measured in the tests.
//!
//! The only new SIMD kernel is the 2x2-mean residual ([`HalveFn`], generic
//! over [`crate::simd::I16Pairs`]: the vertical pair sums in 16-bit lanes,
//! the horizontal ones by a pairwise multiply-add with ones); the forward's
//! 32-point transform is `codec_txfm`'s.

#[allow(unsafe_code)]
mod generic;

use crate::codec_txfm::vvc_primary::tables::DCT64;
use crate::{Dsp, HevcForwardMode, HevcTransform, Isa};
use nc_common::PlaneView;

/// Samples on a side of the transform block.
pub(crate) const BLOCK_SIDE: usize = 64;
/// Frequencies a 64-point side codes (§7.3.11.11 `log2ZoTbWidth` 5).
pub(crate) const CODED_SIDE: usize = 32;
/// Values of the coded region.
const CODED_AREA: usize = CODED_SIDE * CODED_SIDE;
/// Values of the transform block.
const BLOCK_AREA: usize = BLOCK_SIDE * BLOCK_SIDE;
/// The 2x2 mean of four residual samples: `(sum + 2) >> 2`.
const HALVE_SHIFT: u32 = 2;
/// §8.7.4.1: the first inverse stage rounds by 7 bits and clips to 16.
const INVERSE_FIRST_SHIFT: u32 = 7;
/// §8.7.4.1 `bdShift` at 8-bit: 20 − `BitDepth`.
const INVERSE_SECOND_SHIFT: u32 = 12;
/// Fraction bits of [`REPAIR`].
const REPAIR_BITS: u32 = 14;
/// `round(2^14 · cos(πk/128))`, `k` 0..32: the 2x2 mean's response at
/// frequency `k` of the 64-point transform, which the approximate forward
/// restores per dimension.
const REPAIR: [i32; CODED_SIDE] = [
    16384, 16379, 16364, 16340, 16305, 16261, 16207, 16143, 16069, 15986, 15893, 15791, 15679,
    15557, 15426, 15286, 15137, 14978, 14811, 14635, 14449, 14256, 14053, 13842, 13623, 13395,
    13160, 12916, 12665, 12406, 12140, 11866,
];

/// The 32x32 2x2-mean residual `(Σ src − Σ pred + 2) >> 2` of the 64x64
/// blocks `src` and `pred` (at their strides) into `out` (32 per row).
pub(crate) type HalveFn =
    unsafe fn(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize, out: &mut [i16]);

/// Rows `2y` and `2y + 1` of a 64-wide block at `stride`.
fn row_pair(plane: &[u8], stride: usize, y: usize) -> (&[u8], &[u8]) {
    (
        &plane[2 * y * stride..][..BLOCK_SIDE],
        &plane[(2 * y + 1) * stride..][..BLOCK_SIDE],
    )
}

/// The scalar reference of [`HalveFn`] (`unsafe` only to fit the table).
#[allow(unsafe_code)]
unsafe fn halve(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize, out: &mut [i16]) {
    for (y, line) in out[..CODED_AREA]
        .as_chunks_mut::<CODED_SIDE>()
        .0
        .iter_mut()
        .enumerate()
    {
        let (s0, s1) = row_pair(src, src_stride, y);
        let (p0, p1) = row_pair(pred, pred_stride, y);
        for (x, value) in line.iter_mut().enumerate() {
            let sum = |a: &[u8], b: &[u8]| {
                let pair = 2 * x..2 * x + 2;
                a[pair.clone()]
                    .iter()
                    .chain(&b[pair])
                    .map(|&v| i32::from(v))
                    .sum::<i32>()
            };
            *value = ((sum(s0, s1) - sum(p0, p1) + (1 << (HALVE_SHIFT - 1))) >> HALVE_SHIFT) as i16;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2 {
    use crate::simd::avx2::I16x16;

    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn halve(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        out: &mut [i16],
    ) {
        // SAFETY: AVX2 is enabled in this function; the kernel indexes its slices.
        unsafe { super::generic::halve::<I16x16>(src, src_stride, pred, pred_stride, out) }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use crate::simd::neon::I16x8;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn halve(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        out: &mut [i16],
    ) {
        // SAFETY: NEON is enabled in this function; the kernel indexes its slices.
        unsafe { super::generic::halve::<I16x8>(src, src_stride, pred, pred_stride, out) }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128 {
    use crate::simd::simd128::I16x8;

    pub(super) unsafe fn halve(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        out: &mut [i16],
    ) {
        // SAFETY: SIMD128 is enabled for the whole build; the kernel indexes
        // its slices.
        unsafe { super::generic::halve::<I16x8>(src, src_stride, pred, pred_stride, out) }
    }
}

/// The [`HalveFn`] of `isa`.
pub(crate) fn kernel(isa: Isa) -> HalveFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::halve,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::halve,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::halve,
        _ => halve,
    }
}

/// Multiply each coefficient of the 32x32 block by [`REPAIR`] of its
/// vertical and of its horizontal frequency.
fn repair(coefficients: &mut [i32]) {
    let round = 1 << (REPAIR_BITS - 1);
    let lines = coefficients.as_chunks_mut::<CODED_SIDE>().0;
    for (line, &vertical) in lines.iter_mut().zip(&REPAIR) {
        for (c, &horizontal) in line.iter_mut().zip(&REPAIR) {
            let once = (*c * horizontal + round) >> REPAIR_BITS;
            *c = (once * vertical + round) >> REPAIR_BITS;
        }
    }
}

fn round_clip(sum: i32, shift: u32) -> i32 {
    ((sum + (1 << (shift - 1))) >> shift).clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

/// The normative inverse of the dequantised coded region `coefficients`
/// (each within 16 bits) into the 64x64 residual `out`. Only the rows and
/// columns up to the last nonzero coefficient enter the sums: the rest
/// adds nothing, and inter blocks at our rates code a few levels near DC.
fn inverse(coefficients: &[i32], out: &mut [i16]) {
    let lines = coefficients[..CODED_AREA].as_chunks::<CODED_SIDE>().0;
    let (mut rows, mut cols) = (0, 0);
    for (r, line) in lines.iter().enumerate() {
        if let Some(c) = line.iter().rposition(|&v| v != 0) {
            rows = r + 1;
            cols = cols.max(c + 1);
        }
    }
    // Columns: `columns[y][x]` for each coded horizontal frequency `x`.
    let mut columns = [[0i32; CODED_SIDE]; BLOCK_SIDE];
    for (line, basis) in lines[..rows].iter().zip(&DCT64) {
        for (sums, &weight) in columns.iter_mut().zip(basis) {
            let weight = i32::from(weight);
            for (sum, &c) in sums[..cols].iter_mut().zip(&line[..cols]) {
                *sum += weight * c;
            }
        }
    }
    for (sums, residual) in columns.iter().zip(out.as_chunks_mut::<BLOCK_SIDE>().0) {
        let mut row = [0i32; BLOCK_SIDE];
        for (&sum, basis) in sums[..cols].iter().zip(&DCT64) {
            let value = round_clip(sum, INVERSE_FIRST_SHIFT);
            if value == 0 {
                continue;
            }
            for (r, &weight) in row.iter_mut().zip(basis) {
                *r += value * i32::from(weight);
            }
        }
        for (r, &sum) in residual.iter_mut().zip(&row) {
            // Within 16 bits: `round_clip` clips.
            *r = round_clip(sum, INVERSE_SECOND_SHIFT) as i16;
        }
    }
}

#[allow(unsafe_code)]
impl Dsp {
    /// The encoder's approximate forward 64-point DCT-II of the residual
    /// `source − prediction` (64x64 views): the coded region's 32x32
    /// coefficients (row-major, scaled for [`crate::VvcQuantizer::rect`] of
    /// log2 6 x 6) into `out`, through the 32-point forward of `mode` on the
    /// 2x2-mean residual (module docs). False, leaving `out` unspecified,
    /// when that forward proves every coefficient within `zero_limit` (they
    /// would all quantise to zero).
    ///
    /// # Panics
    ///
    /// If a view is smaller than 64x64 or `out` shorter than 32x32.
    pub fn vvc_forward64(
        &self,
        source: &PlaneView<'_>,
        prediction: &PlaneView<'_>,
        zero_limit: u32,
        mode: HevcForwardMode,
        out: &mut [i32],
    ) -> bool {
        for view in [source, prediction] {
            assert!(view.width() >= BLOCK_SIDE && view.height() >= BLOCK_SIDE);
        }
        assert!(out.len() >= CODED_AREA);
        let mut halved = [0i16; CODED_AREA];
        // SAFETY: the table matches this CPU; the kernel indexes its slices.
        unsafe {
            (self.vvc_tx64)(
                source.data(),
                source.stride(),
                prediction.data(),
                prediction.stride(),
                &mut halved,
            );
        }
        let transform = HevcTransform::dct(CODED_SIDE.ilog2() as u8).expect("32-point DCT");
        let out = &mut out[..CODED_AREA];
        let coded = if mode == HevcForwardMode::EXACT {
            self.hevc_forward_unless_zero(transform, &halved, CODED_SIDE, zero_limit, out)
        } else {
            self.hevc_forward_staged_unless_zero(
                transform, &halved, CODED_SIDE, zero_limit, mode, out,
            )
        };
        if coded {
            repair(out);
        }
        coded
    }

    /// The normative dequantisation and inverse of a 64x64 DCT-II block
    /// at `qp` from the levels (or, under dependent quantisation, the
    /// reconstruction indices) of its 32x32 coded region (row-major) into
    /// the 64x64 residual `out` (16 bits, clipped as §8.7.4.1 clips).
    ///
    /// # Panics
    ///
    /// If `qp > 63`, `levels` is shorter than 32x32 or `out` than 64x64.
    pub fn vvc_inverse_dct64(&self, levels: &[i32], qp: u8, dependent: bool, out: &mut [i16]) {
        assert!(qp <= 63 && levels.len() >= CODED_AREA && out.len() >= BLOCK_AREA);
        let scaling =
            crate::codec_txfm::vvc_inverse::scaling((BLOCK_SIDE, BLOCK_SIDE), qp, dependent);
        let mut scaled = [0; CODED_AREA];
        self.hevc_dequantize(&levels[..CODED_AREA], &scaling, &mut scaled);
        inverse(&scaled, &mut out[..BLOCK_AREA]);
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests;
