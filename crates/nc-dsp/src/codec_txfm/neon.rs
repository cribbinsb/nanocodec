//! NEON transforms; staged HEVC rows reuse eight narrow AV1 butterfly lanes,
//! HEVC dequantisation runs four independent vectors per step, the H.264
//! 8x8 forward runs its row pass in eight 16-bit lanes, and the H.264 4x4
//! forward runs entirely in 16-bit lanes, two blocks per vector.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint32x4_t, vaddl_s16, vaddq_s16, vaddq_s32, vaddv_u8, vandq_u8,
    vcombine_u8, vcombine_u16, vdupq_n_s32, vget_high_s16, vget_high_u8, vget_low_s16, vget_low_u8,
    vld1q_s16, vld1q_s32, vld1q_u8, vmaxq_s32, vminq_s32, vmlal_n_s16, vmlaq_n_s16, vmovl_high_s16,
    vmovl_s16, vmovn_u16, vmovn_u32, vmull_n_s16, vmulq_n_s16, vmulq_s32, vqmovn_s32, vqshlq_s32,
    vreinterpretq_s16_s32, vreinterpretq_s32_s16, vrshlq_s32, vrshrq_n_s32, vshlq_n_s16,
    vshlq_n_s32, vst1q_s32, vsubl_s16, vsubq_s16, vsubq_s32, vtrn1q_s16, vtrn1q_s32, vtrn2q_s16,
    vtrn2q_s32, vtstq_s32,
};

use super::generic::{H264Forward8x8, HEVC_DEQUANT_LEVEL_LIMIT};
use super::{H264Quantizer, HevcForwardMode, HevcForwardStaging, HevcQuantizer};
use crate::simd::neon::I32x4;
use crate::txfm::neon_forward::transpose;

mod wide {
    use crate::simd::neon::I32x4;
    super::super::generic::codec_txfm_kernels!(#[target_feature(enable = "neon")]; I32x4, I32x4, I32x4);
    super::super::generic::hevc_bounded_inverse_kernels!(#[target_feature(enable = "neon")]; I32x4);
}

pub(crate) fn codec_txfm_fns() -> super::CodecTxfmFns {
    let mut table = wide::codec_txfm_fns();
    table.vvc_forward_dct = super::vvc_forward_simd::neon;
    wide::install_bounded_inverse(&mut table);
    table.hevc_inverse_add[super::HevcTransform::Dct16 as usize] =
        inverse_add_buffered::<super::generic::HevcIdct16, 16>;
    table.hevc_inverse_add[super::HevcTransform::Dct32 as usize] =
        inverse_add_buffered::<super::generic::HevcIdct32, 32>;
    table.hevc_forward_bounded[2] = staged16_bounded;
    table.hevc_forward_bounded[3] = staged32_bounded;
    table.hevc_dequantize = hevc_dequantize;
    table.h264_forward_8x8 = h264_forward_8x8;
    table.h264_forward_4x4 = h264_forward_4x4;
    table.h264_forward_quantize_mb = h264_forward_quantize_mb;
    table.nonzero_mask = nonzero_mask;
    table
}

/// Lane bits of 16 byte flags (two bytes of mask).
const BYTE_LANE_BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// `generic::nonzero_mask`: sixteen levels at a time are compared with
/// zero, narrowed to byte flags, masked with their lane bits and summed per
/// half (scalar: about four instructions per level).
#[target_feature(enable = "neon")]
unsafe fn nonzero_mask(levels: &[i32]) -> u64 {
    let (rows, rest) = levels.as_chunks::<16>();
    // SAFETY: NEON is enabled here; the table holds 16 bytes.
    let bits = unsafe { vld1q_u8(BYTE_LANE_BITS.as_ptr()) };
    let mut mask = 0;
    for (r, row) in rows.iter().enumerate() {
        // SAFETY: NEON is enabled here; each quarter of `row` holds 4 levels.
        let flags: [uint32x4_t; 4] = core::array::from_fn(|q| unsafe {
            let v = vld1q_s32(row[4 * q..].as_ptr());
            vtstq_s32(v, v)
        });
        let halves = [
            vcombine_u16(vmovn_u32(flags[0]), vmovn_u32(flags[1])),
            vcombine_u16(vmovn_u32(flags[2]), vmovn_u32(flags[3])),
        ];
        let bytes = vandq_u8(
            vcombine_u8(vmovn_u16(halves[0]), vmovn_u16(halves[1])),
            bits,
        );
        let low = u64::from(vaddv_u8(vget_low_u8(bytes)));
        let high = u64::from(vaddv_u8(vget_high_u8(bytes)));
        mask |= (low | high << 8) << (16 * r);
    }
    if !rest.is_empty() {
        mask |= super::generic::nonzero_mask(rest) << (16 * rows.len());
    }
    mask
}

/// H.264 forward 8x8, as `generic::h264_forward_8x8`. With `|r| <= 255`
/// (module docs) every row-pass output stays within ±16320 (DC: 64 · 255),
/// so the row pass runs on eight 16-bit lanes between two in-register
/// transposes; the column pass (outputs up to 64 times larger) widens to 32
/// bits after its first butterfly ([`h264_fdct8_columns`]).
#[target_feature(enable = "neon")]
unsafe fn h264_forward_8x8(residual: &[i16], stride: usize, out: &mut [i32; 64]) {
    let rows: [int16x8_t; 8] = core::array::from_fn(|r| {
        let row = &residual[r * stride..][..8];
        // SAFETY: NEON is enabled here; `row` holds 8 values.
        unsafe { vld1q_s16(row.as_ptr()) }
    });
    // SAFETY: NEON is enabled here.
    unsafe {
        // Lane r of columns[c] is residual (r, c); lane j of coefficients[r]
        // is row r's coefficient j.
        let columns = transpose(rows);
        let coefficients = transpose(h264_fdct8_rows(columns));
        for (u, lanes) in h264_fdct8_columns(coefficients).iter().enumerate() {
            let [low, high] = lanes;
            vst1q_s32(out[u * 8..][..4].as_mut_ptr(), *low);
            vst1q_s32(out[u * 8 + 4..][..4].as_mut_ptr(), *high);
        }
    }
}

/// The column pass of the H.264 forward 8x8 (`generic::H264Fdct8`, then
/// the `(x + 32) >> 6` rounding): `x[r]` holds row `r`'s eight row-pass
/// coefficients, output `u` coefficient row `u` as two halves. The first
/// butterfly stays in 16 bits (`|x| <= 16320`, so `x[i] ± x[7 − i]` fits);
/// everything after it widens: the even half with widening adds, the odd
/// half with widening multiply-adds straight from the 16-bit differences.
#[inline]
#[target_feature(enable = "neon")]
fn h264_fdct8_columns(x: [int16x8_t; 8]) -> [[int32x4_t; 2]; 8] {
    let s = [
        vaddq_s16(x[0], x[7]),
        vaddq_s16(x[1], x[6]),
        vaddq_s16(x[2], x[5]),
        vaddq_s16(x[3], x[4]),
    ];
    let d = [
        vsubq_s16(x[0], x[7]),
        vsubq_s16(x[1], x[6]),
        vsubq_s16(x[2], x[5]),
        vsubq_s16(x[3], x[4]),
    ];
    let mut y = [[vdupq_n_s32(0); 2]; 8];
    for half in 0..2 {
        let pick = |v: int16x8_t| {
            if half == 0 {
                vget_low_s16(v)
            } else {
                vget_high_s16(v)
            }
        };
        let (s, d) = (
            [pick(s[0]), pick(s[1]), pick(s[2]), pick(s[3])],
            [pick(d[0]), pick(d[1]), pick(d[2]), pick(d[3])],
        );
        let (e0, e1) = (vaddl_s16(s[0], s[3]), vaddl_s16(s[1], s[2]));
        let (e2, e3) = (vsubl_s16(s[0], s[3]), vsubl_s16(s[1], s[2]));
        let dot4 = |w: [i16; 4]| {
            let sum = vmlal_n_s16(vmull_n_s16(d[0], w[0]), d[1], w[1]);
            vmlal_n_s16(vmlal_n_s16(sum, d[2], w[2]), d[3], w[3])
        };
        let coefficients = [
            vshlq_n_s32::<3>(vaddq_s32(e0, e1)),
            dot4([12, 10, 6, 3]),
            vaddq_s32(vshlq_n_s32::<3>(e2), vshlq_n_s32::<2>(e3)),
            dot4([10, -3, -12, -6]),
            vshlq_n_s32::<3>(vsubq_s32(e0, e1)),
            dot4([6, -12, 3, 10]),
            vsubq_s32(vshlq_n_s32::<2>(e2), vshlq_n_s32::<3>(e3)),
            dot4([3, -6, 10, -12]),
        ];
        for (row, value) in y.iter_mut().zip(coefficients) {
            row[half] = vrshrq_n_s32::<6>(value);
        }
    }
    y
}

/// `generic::H264Fdct8` on eight 16-bit lanes: `x[c]` holds sample `c` of
/// eight lines, the result holds their coefficients `0..8`.
#[inline]
#[target_feature(enable = "neon")]
fn h264_fdct8_rows(x: [int16x8_t; 8]) -> [int16x8_t; 8] {
    let s = [0, 1, 2, 3].map(|i| vaddq_s16(x[i], x[7 - i]));
    let d = [0, 1, 2, 3].map(|i| vsubq_s16(x[i], x[7 - i]));
    let (e0, e1) = (vaddq_s16(s[0], s[3]), vaddq_s16(s[1], s[2]));
    let (e2, e3) = (vsubq_s16(s[0], s[3]), vsubq_s16(s[1], s[2]));
    let dot4 = |w: [i16; 4]| {
        let mut sum = vmulq_n_s16(d[0], w[0]);
        for k in 1..4 {
            sum = vmlaq_n_s16(sum, d[k], w[k]);
        }
        sum
    };
    [
        vshlq_n_s16::<3>(vaddq_s16(e0, e1)),
        dot4([12, 10, 6, 3]),
        vaddq_s16(vshlq_n_s16::<3>(e2), vshlq_n_s16::<2>(e3)),
        dot4([10, -3, -12, -6]),
        vshlq_n_s16::<3>(vsubq_s16(e0, e1)),
        dot4([6, -12, 3, 10]),
        vsubq_s16(vshlq_n_s16::<2>(e2), vshlq_n_s16::<3>(e3)),
        dot4([3, -6, 10, -12]),
    ]
}

/// Samples per block row of the H.264 4x4 transform.
const BLOCK4: usize = 4;

/// H.264 forward 4x4 of every block, as `generic::h264_forward_4x4`: two
/// horizontally adjacent blocks per eight 16-bit lanes. With `|r| <= 255`
/// (module docs) the first pass stays within ±1530 and the second within
/// ±9180, so neither overflows; the transform has no rounding, so
/// columns-then-rows equals the reference's rows-then-columns exactly.
/// An odd number of blocks per row takes the generic kernel.
#[target_feature(enable = "neon")]
unsafe fn h264_forward_4x4(
    residual: &[i16],
    stride: usize,
    blocks_wide: usize,
    out: &mut [[i32; 16]],
) {
    if !blocks_wide.is_multiple_of(2) {
        // SAFETY: NEON is enabled here; the caller's contract is forwarded.
        unsafe { super::generic::h264_forward_4x4::<I32x4>(residual, stride, blocks_wide, out) };
        return;
    }
    for (pair, blocks) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let b = 2 * pair;
        let origin = (b / blocks_wide) * BLOCK4 * stride + (b % blocks_wide) * BLOCK4;
        let rows: [int16x8_t; 4] = core::array::from_fn(|r| {
            let row = &residual[origin + r * stride..][..2 * BLOCK4];
            // SAFETY: NEON is enabled here; `row` holds 8 values.
            unsafe { vld1q_s16(row.as_ptr()) }
        });
        // Lanes are (block, column): the vertical pass, then (block,
        // vertical frequency) after the transpose: the horizontal pass;
        // the second transpose puts each coefficient row in its lanes.
        let coefficients = transpose4(h264_fdct4(transpose4(h264_fdct4(rows))));
        let [left, right] = blocks;
        for (u, &row) in coefficients.iter().enumerate() {
            // SAFETY: NEON is enabled here; each destination holds 4 values.
            unsafe {
                vst1q_s32(
                    left[BLOCK4 * u..][..BLOCK4].as_mut_ptr(),
                    vmovl_s16(vget_low_s16(row)),
                );
                vst1q_s32(
                    right[BLOCK4 * u..][..BLOCK4].as_mut_ptr(),
                    vmovl_high_s16(row),
                );
            }
        }
    }
}

/// `generic::H264Fdct4` across four vectors.
#[inline]
#[target_feature(enable = "neon")]
fn h264_fdct4(x: [int16x8_t; 4]) -> [int16x8_t; 4] {
    let (s03, d03) = (vaddq_s16(x[0], x[3]), vsubq_s16(x[0], x[3]));
    let (s12, d12) = (vaddq_s16(x[1], x[2]), vsubq_s16(x[1], x[2]));
    [
        vaddq_s16(s03, s12),
        vaddq_s16(vshlq_n_s16::<1>(d03), d12),
        vsubq_s16(s03, s12),
        vsubq_s16(d03, vshlq_n_s16::<1>(d12)),
    ]
}

/// Transpose the 4x4 block in each 64-bit half of four rows.
#[inline]
#[target_feature(enable = "neon")]
fn transpose4(rows: [int16x8_t; 4]) -> [int16x8_t; 4] {
    let even01 = vreinterpretq_s32_s16(vtrn1q_s16(rows[0], rows[1]));
    let odd01 = vreinterpretq_s32_s16(vtrn2q_s16(rows[0], rows[1]));
    let even23 = vreinterpretq_s32_s16(vtrn1q_s16(rows[2], rows[3]));
    let odd23 = vreinterpretq_s32_s16(vtrn2q_s16(rows[2], rows[3]));
    [
        vreinterpretq_s16_s32(vtrn1q_s32(even01, even23)),
        vreinterpretq_s16_s32(vtrn1q_s32(odd01, odd23)),
        vreinterpretq_s16_s32(vtrn2q_s32(even01, even23)),
        vreinterpretq_s16_s32(vtrn2q_s32(odd01, odd23)),
    ]
}

/// [`h264_forward_8x8`] for the macroblock driver.
struct NarrowForward8x8;

impl H264Forward8x8 for NarrowForward8x8 {
    #[inline(always)]
    unsafe fn forward(residual: &[i16], stride: usize, out: &mut [i32; 64]) {
        // SAFETY: forwarded contract (NEON is always present on AArch64).
        unsafe { h264_forward_8x8(residual, stride, out) }
    }
}

/// `generic::h264_forward_quantize_mb` with [`h264_forward_8x8`].
#[target_feature(enable = "neon")]
unsafe fn h264_forward_quantize_mb(
    residual: &[i16],
    stride: usize,
    zero_mask: u8,
    q: &H264Quantizer,
    out: &mut [[i32; 64]; 4],
) -> [u32; 4] {
    // SAFETY: NEON is enabled here and the caller's contract is forwarded.
    unsafe {
        super::generic::h264_forward_quantize_mb::<NarrowForward8x8, I32x4>(
            residual, stride, zero_mask, q, out,
        )
    }
}

/// Levels per step: four independent vectors.
const DEQUANT_STEP: usize = 16;

/// HEVC scaling (eq. 8-309, flat lists) with the arithmetic of
/// `generic::hevc_dequantize`, 16 levels per step. The generic 4-lane loop
/// was latency-bound: each step was one serial 9-instruction chain behind
/// four slice bounds checks. The ragged tail uses the generic kernel.
#[target_feature(enable = "neon")]
unsafe fn hevc_dequantize(levels: &[i32], q: &HevcQuantizer, out: &mut [i32]) {
    let scaler = HevcScaler::new(q);
    let n = levels.len();
    let (src, _) = levels.as_chunks::<DEQUANT_STEP>();
    let (dst, _) = out[..n].as_chunks_mut::<DEQUANT_STEP>();
    for (s, d) in src.iter().zip(dst) {
        for k in (0..DEQUANT_STEP).step_by(4) {
            // SAFETY: `k + 4 <= 16` values of each chunk.
            unsafe {
                let value = scaler.value(vld1q_s32(s.as_ptr().add(k)));
                vst1q_s32(d.as_mut_ptr().add(k), value);
            }
        }
    }
    let done = src.len() * DEQUANT_STEP;
    // SAFETY: NEON is enabled for this function.
    unsafe {
        super::generic::hevc_dequantize::<crate::simd::neon::I32x4>(
            &levels[done..],
            q,
            &mut out[done..n],
        );
    }
}

/// The HEVC scaling constants as lanes.
struct HevcScaler {
    level_min: int32x4_t,
    level_max: int32x4_t,
    level_scale: int32x4_t,
    per: int32x4_t,
    /// `-bdShift`: a negative count makes `vrshl` a rounding right shift.
    bd_shift: int32x4_t,
}

impl HevcScaler {
    #[inline(always)]
    fn new(q: &HevcQuantizer) -> Self {
        // SAFETY: NEON kernels only.
        unsafe {
            Self {
                level_min: vdupq_n_s32(-HEVC_DEQUANT_LEVEL_LIMIT),
                level_max: vdupq_n_s32(HEVC_DEQUANT_LEVEL_LIMIT),
                level_scale: vdupq_n_s32(q.level_scale),
                per: vdupq_n_s32(q.per as i32),
                bd_shift: vdupq_n_s32(-(q.bd_shift as i32)),
            }
        }
    }

    /// `clamp16(round_shift(clamp(l) * levelScale << per, bdShift))`.
    ///
    /// The generic kernel clamps the product to `±2^(16 + bdShift) >> per`
    /// before the shift. Here the left shift saturates instead: any product
    /// beyond that limit (saturated or not) still shifts to at least `2^16`
    /// in magnitude, which the final clamp maps to the same ±32767 / -32768.
    /// The level clamp keeps the product itself within 32 bits
    /// (`2^16 * 1152 < 2^27`).
    #[inline(always)]
    fn value(&self, level: int32x4_t) -> int32x4_t {
        // SAFETY: NEON kernels only.
        unsafe {
            let level = vminq_s32(vmaxq_s32(level, self.level_min), self.level_max);
            let product = vmulq_s32(level, self.level_scale);
            let scaled = vrshlq_s32(vqshlq_s32(product, self.per), self.bd_shift);
            // Saturating narrowing is the clamp to 16 bits.
            vmovl_s16(vqmovn_s32(scaled))
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn staged16_bounded(
    residual: &[i16],
    stride: usize,
    limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    use super::generic::{self as g, HevcFdct16};
    // SAFETY: checked caller slices and this entry enables NEON.
    unsafe {
        if mode.staging == HevcForwardStaging::Exact {
            g::hevc_forward_bounded::<crate::simd::neon::I32x4, HevcFdct16, 16, 4>(
                residual, stride, limit, out, mode,
            )
        } else {
            g::hevc_forward_neon_rows::<HevcFdct16, 16, 4>(residual, stride, limit, out, mode)
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn staged32_bounded(
    residual: &[i16],
    stride: usize,
    limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    use super::generic::{self as g, HevcFdct32};
    // SAFETY: checked caller slices and this entry enables NEON.
    unsafe {
        if mode.staging == HevcForwardStaging::Exact {
            g::hevc_forward_bounded::<crate::simd::neon::I32x4, HevcFdct32, 32, 5>(
                residual, stride, limit, out, mode,
            )
        } else {
            g::hevc_forward_neon_rows::<HevcFdct32, 32, 5>(residual, stride, limit, out, mode)
        }
    }
}

/// On NEON the bounded inverse plus contiguous eight-pixel reconstruction
/// beats fused four-lane transposed stores for 16/32-point blocks. Keep that
/// architecture choice behind the same shared inverse/add entry point.
#[target_feature(enable = "neon")]
unsafe fn inverse_add_buffered<T: super::generic::BoundedLine, const N: usize>(
    coeffs: &[i32],
    extent: (usize, usize),
    dst: &mut [u8],
    stride: usize,
) {
    use super::generic as g;
    use core::arch::aarch64::{
        vcombine_s16, vld1_u8, vmovl_u8, vqaddq_s16, vqmovun_s16, vreinterpretq_s16_u16, vst1_u8,
    };
    if extent.0 == 0 || extent.1 == 0 {
        return;
    }
    if extent == (1, 1) {
        super::inverse_add::dc::<I32x4, N>(coeffs[0], dst, stride);
        return;
    }
    let mut residual = [0; g::MAX_AREA];
    // SAFETY: NEON is enabled; the dispatcher checked the coefficient block.
    unsafe { g::hevc_inverse_extent::<I32x4, T, N>(coeffs, &mut residual, extent) };
    for row in 0..N {
        let output = &mut dst[row * stride..][..N];
        let residual = &residual[row * N..][..N];
        for (output, residual) in output
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(residual.as_chunks::<8>().0)
        {
            // SAFETY: each chunk contains eight pixels/residuals, and NEON
            // is enabled. Saturating to i16 before adding preserves u8 clip.
            unsafe {
                let lo = vqmovn_s32(vld1q_s32(residual.as_ptr()));
                let hi = vqmovn_s32(vld1q_s32(residual.as_ptr().add(4)));
                let pred = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(output.as_ptr())));
                let sum = vqaddq_s16(vcombine_s16(lo, hi), pred);
                vst1_u8(output.as_mut_ptr(), vqmovun_s16(sum));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hevc_dequantize_matches_scalar_on_ragged_lengths_and_extremes() {
        let reference = super::super::scalar::codec_txfm_fns();
        let mut state = 0x2545_f491_u32;
        for qp in [0_u8, 5, 22, 37, 51] {
            for log2 in 2..=5_u8 {
                let q = HevcQuantizer::new(qp, log2, false);
                for len in [0, 1, 3, 4, 15, 16, 17, 31, 33, 100, 1024] {
                    let levels: Vec<i32> = (0..len)
                        .map(|i| {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            match i % 5 {
                                0 => state as i32,
                                1 => (state % 2001) as i32 - 1000,
                                2 => [i32::MIN, i32::MAX, 1 << 16, -(1 << 16) - 1][(i / 5) % 4],
                                3 => (state % 65_537) as i32 - 32_768,
                                _ => (state % (1 << 18)) as i32 - (1 << 17),
                            }
                        })
                        .collect();
                    let mut expected = vec![0x1234; len + 2];
                    let mut actual = expected.clone();
                    // SAFETY: NEON is mandatory on AArch64; outputs cover the levels.
                    unsafe {
                        (reference.hevc_dequantize)(&levels, &q, &mut expected);
                        hevc_dequantize(&levels, &q, &mut actual);
                    }
                    assert_eq!(actual, expected, "qp {qp} log2 {log2} len {len}");
                }
            }
        }
    }

    /// Every level magnitude near the saturation threshold, at every QP:
    /// the saturating shift must match the generic product clamp.
    #[test]
    fn hevc_dequantize_saturation_threshold_matches_scalar() {
        let reference = super::super::scalar::codec_txfm_fns();
        for qp in 0..=51_u8 {
            for log2 in 2..=5_u8 {
                let q = HevcQuantizer::new(qp, log2, true);
                let threshold =
                    super::super::generic::hevc_dequant_product_limit(&q) / q.level_scale;
                let levels: Vec<i32> = (-40..=40)
                    .flat_map(|d| [threshold + d, -threshold + d])
                    .collect();
                let mut expected = vec![0; levels.len()];
                let mut actual = expected.clone();
                // SAFETY: NEON is mandatory on AArch64; outputs cover the levels.
                unsafe {
                    (reference.hevc_dequantize)(&levels, &q, &mut expected);
                    hevc_dequantize(&levels, &q, &mut actual);
                }
                assert_eq!(actual, expected, "qp {qp} log2 {log2}");
            }
        }
    }
}
