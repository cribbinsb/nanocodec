//! HEVC 4×4 inverse DCT/DST with fused byte and word reconstruction.
//! Follows libhevc's `ihevc_itrans_recon_4x4_av8` and its DST companion
//! (Apache-2.0, Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore).

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x4_t, vaddl_s16, vaddq_s32, vcombine_s16, vcreate_u8, vdup_n_s16, vget_lane_u32, vld1_s16,
    vld1_u16, vmax_s16, vmin_u16, vmlal_lane_s16, vmlsl_lane_s16, vmovl_u8, vmull_lane_s16,
    vqadd_s16, vqaddq_s16, vqmovun_s16, vqrshrn_n_s32, vreinterpret_s16_s32, vreinterpret_s16_u16,
    vreinterpret_s32_s16, vreinterpret_u16_s16, vreinterpret_u32_u8, vreinterpretq_s16_u16,
    vshlq_n_s32, vst1_s16, vst1_u16, vsubl_s16, vsubq_s32, vtrn1_s16, vtrn1_s32, vtrn2_s16,
    vtrn2_s32,
};
use nc_common::BitDepth;

use super::{
    DCT32, DST4,
    generic::{HevcIdct4, HevcIdst4},
    inverse_add,
};
use crate::simd::neon::I32x4;

const DCT_COEFFS: [i16; 4] = [DCT32[8][0] as i16, DCT32[8][1] as i16, 0, 0];
const DST_COEFFS: [i16; 4] = [
    DST4[0][0] as i16,
    DST4[0][1] as i16,
    DST4[0][2] as i16,
    DST4[0][3] as i16,
];

#[inline(always)]
pub(super) unsafe fn transpose4([a, b, c, d]: [int16x4_t; 4]) -> [int16x4_t; 4] {
    // SAFETY: used inside the NEON entry points.
    unsafe {
        let ab0 = vreinterpret_s32_s16(vtrn1_s16(a, b));
        let ab1 = vreinterpret_s32_s16(vtrn2_s16(a, b));
        let cd0 = vreinterpret_s32_s16(vtrn1_s16(c, d));
        let cd1 = vreinterpret_s32_s16(vtrn2_s16(c, d));
        [
            vreinterpret_s16_s32(vtrn1_s32(ab0, cd0)),
            vreinterpret_s16_s32(vtrn1_s32(ab1, cd1)),
            vreinterpret_s16_s32(vtrn2_s32(ab0, cd0)),
            vreinterpret_s16_s32(vtrn2_s32(ab1, cd1)),
        ]
    }
}

#[inline(always)]
#[allow(
    clippy::many_single_char_names,
    reason = "the four input rows (a-d) and weights (w) of the 4-point transform"
)]
unsafe fn line<const DST: bool, const SHIFT: i32>(
    [a, b, c, d]: [int16x4_t; 4],
    w: int16x4_t,
) -> [int16x4_t; 4] {
    // SAFETY: NEON is enabled. All four-term sums fit signed dwords.
    unsafe {
        let values = if DST {
            let b74 = vmull_lane_s16::<2>(b, w);
            let first = vmlal_lane_s16::<3>(
                vmlal_lane_s16::<1>(vmlal_lane_s16::<0>(b74, a, w), d, w),
                c,
                w,
            );
            let second = vmlsl_lane_s16::<3>(
                vmlsl_lane_s16::<0>(vmlal_lane_s16::<1>(b74, a, w), c, w),
                d,
                w,
            );
            let third =
                vmlal_lane_s16::<2>(vmlsl_lane_s16::<2>(vmull_lane_s16::<2>(a, w), c, w), d, w);
            let fourth = vmlal_lane_s16::<3>(
                vmlsl_lane_s16::<0>(vsubq_s32(vmull_lane_s16::<1>(c, w), b74), d, w),
                a,
                w,
            );
            [first, second, third, fourth]
        } else {
            let e0 = vshlq_n_s32::<6>(vaddl_s16(a, c));
            let e1 = vshlq_n_s32::<6>(vsubl_s16(a, c));
            let o0 = vmlal_lane_s16::<1>(vmull_lane_s16::<0>(b, w), d, w);
            let o1 = vmlsl_lane_s16::<0>(vmull_lane_s16::<1>(b, w), d, w);
            [
                vaddq_s32(e0, o0),
                vaddq_s32(e1, o1),
                vsubq_s32(e1, o1),
                vsubq_s32(e0, o0),
            ]
        };
        // Keep the clamped first pass in 16-bit lanes.
        [
            vqrshrn_n_s32::<SHIFT>(values[0]),
            vqrshrn_n_s32::<SHIFT>(values[1]),
            vqrshrn_n_s32::<SHIFT>(values[2]),
            vqrshrn_n_s32::<SHIFT>(values[3]),
        ]
    }
}

#[inline(always)]
unsafe fn residual<const DST: bool, const SHIFT: i32>(coeffs: &[i16]) -> [int16x4_t; 4] {
    let c = &coeffs[..16];
    // SAFETY: each load lies within the checked block.
    unsafe {
        let input = [
            vld1_s16(c.as_ptr()),
            vld1_s16(c.as_ptr().add(4)),
            vld1_s16(c.as_ptr().add(8)),
            vld1_s16(c.as_ptr().add(12)),
        ];
        let w = vld1_s16(if DST { &DST_COEFFS } else { &DCT_COEFFS }.as_ptr());
        transpose4(line::<DST, SHIFT>(transpose4(line::<DST, 7>(input, w)), w))
    }
}

/// The same inverse writes signed residuals for the encoders.
///
/// # Safety
/// NEON is enabled and both slices cover 4×4.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse<const DST: bool>(coeffs: &[i16], out: &mut [i16]) {
    let out = &mut out[..16];
    // SAFETY: all input and output rows lie in the checked blocks.
    unsafe {
        for (y, row) in residual::<DST, 12>(coeffs).into_iter().enumerate() {
            vst1_s16(out.as_mut_ptr().add(y * 4), row);
        }
    }
}

/// # Safety
/// NEON is enabled and the checked blocks cover 4×4.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_u8<const DST: bool>(
    coeffs: &[i16],
    extent: (usize, usize),
    dst: &mut [u8],
    stride: usize,
) {
    if extent.0 == 0 || extent.1 == 0 {
        return;
    }
    if !DST && extent == (1, 1) {
        inverse_add::dc::<I32x4, 4>(coeffs[0], dst, stride);
        return;
    }
    let dst = &mut dst[..3 * stride + 4];
    // SAFETY: every prediction load and store is four bytes in a checked row.
    unsafe {
        let [a, b, c, d] = residual::<DST, 12>(coeffs);
        let pair = |sample0, sample1, y| {
            let ptr = dst.as_mut_ptr().add(y * stride);
            let prev = ptr.cast::<u32>().read_unaligned();
            let next = ptr.add(stride).cast::<u32>().read_unaligned();
            let pred = vcreate_u8(u64::from(prev) | (u64::from(next) << 32));
            let sum = vqaddq_s16(
                vcombine_s16(sample0, sample1),
                vreinterpretq_s16_u16(vmovl_u8(pred)),
            );
            let out = vreinterpret_u32_u8(vqmovun_s16(sum));
            ptr.cast::<u32>().write_unaligned(vget_lane_u32::<0>(out));
            ptr.add(stride)
                .cast::<u32>()
                .write_unaligned(vget_lane_u32::<1>(out));
        };
        let mut pair = pair;
        pair(a, b, 0);
        pair(c, d, 2);
    }
}

/// # Safety
/// As `add_u8`, at the supplied sample depth.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_u16<const DST: bool>(
    coeffs: &[i16],
    extent: (usize, usize),
    dst: &mut [u16],
    stride: usize,
    depth: BitDepth,
) {
    if extent.0 == 0 || extent.1 == 0 {
        return;
    }
    if !DST && extent == (1, 1) {
        inverse_add::dc_u16::<I32x4, 4>(coeffs[0], dst, stride, depth);
        return;
    }
    if !matches!(depth.bits(), 10 | 12) {
        // SAFETY: the fallback uses the same checked coefficient and sample spans.
        unsafe {
            if DST {
                inverse_add::inverse_add_u16::<I32x4, HevcIdst4, 4>(
                    coeffs, extent, dst, stride, false, depth,
                );
            } else {
                inverse_add::inverse_add_u16::<I32x4, HevcIdct4, 4>(
                    coeffs, extent, dst, stride, true, depth,
                );
            }
        }
        return;
    }
    let dst = &mut dst[..3 * stride + 4];
    // SAFETY: every load and store fits the checked destination.
    unsafe {
        let rows = if depth.bits() == 10 {
            residual::<DST, 10>(coeffs)
        } else {
            residual::<DST, 8>(coeffs)
        };
        let zero = vdup_n_s16(0);
        let max = vreinterpret_u16_s16(vdup_n_s16(depth.max() as i16));
        for (y, row) in rows.into_iter().enumerate() {
            let ptr = dst.as_mut_ptr().add(y * stride);
            let pred = vreinterpret_s16_u16(vld1_u16(ptr));
            let sum = vmax_s16(vqadd_s16(row, pred), zero);
            vst1_u16(ptr, vmin_u16(vreinterpret_u16_s16(sum), max));
        }
    }
}
