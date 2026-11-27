//! The shared HEVC DCT-II 8×8 inverse with fused reconstruction at byte
//! and word depth. It keeps the clamped first pass in sixteen-bit lanes.
//! The lane-MAC schedule follows libhevc's `ihevc_itrans_recon_8x8_av8`
//! (Apache-2.0, Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore).

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int32x4_t, vaddl_s16, vaddq_s32, vcombine_s16, vdupq_n_s16,
    vget_high_s16, vget_low_s16, vld1_s16, vld1_u8, vld1q_s16, vld1q_u16, vmaxq_s16, vminq_u16,
    vmlal_laneq_s16, vmlsl_laneq_s16, vmovl_u8, vmull_laneq_s16, vqaddq_s16, vqmovun_s16,
    vqrshrn_n_s32, vreinterpretq_s16_u16, vreinterpretq_u16_s16, vshlq_n_s32, vst1_u8, vst1q_s16,
    vst1q_u16, vsubl_s16, vsubq_s32,
};
use nc_common::BitDepth;

use super::{DCT32, generic::HevcIdct8, inverse_add};
use crate::simd::neon::I32x4;
use crate::txfm::neon_forward::transpose;

const SIDE: usize = 8;
pub(super) const WEIGHTS: [i16; 8] = [
    DCT32[8][0] as i16, // 83
    DCT32[8][1] as i16, // 36
    DCT32[4][0] as i16, // 89
    DCT32[4][1] as i16, // 75
    DCT32[4][2] as i16, // 50
    DCT32[4][3] as i16, // 18
    0,
    0,
];

#[inline(always)]
#[allow(
    clippy::many_single_char_names,
    reason = "the butterfly's coefficient rows (c), weights (w) and even/odd terms (e, o)"
)]
pub(super) unsafe fn butterfly<const LOW: bool>(
    c: [int16x4_t; SIDE],
    w: int16x8_t,
) -> [int32x4_t; SIDE] {
    // SAFETY: only instantiated in the NEON entry points below.
    unsafe {
        let (e0, e1) = if LOW {
            let dc = vshlq_n_s32::<6>(vaddl_s16(c[0], c[4]));
            (dc, dc)
        } else {
            (
                vshlq_n_s32::<6>(vaddl_s16(c[0], c[4])),
                vshlq_n_s32::<6>(vsubl_s16(c[0], c[4])),
            )
        };
        let mut e2 = vmull_laneq_s16::<0>(c[2], w);
        let mut e3 = vmull_laneq_s16::<1>(c[2], w);
        let mut o0 = vmlal_laneq_s16::<3>(vmull_laneq_s16::<2>(c[1], w), c[3], w);
        let mut o1 = vmlsl_laneq_s16::<5>(vmull_laneq_s16::<3>(c[1], w), c[3], w);
        let mut o2 = vmlsl_laneq_s16::<2>(vmull_laneq_s16::<4>(c[1], w), c[3], w);
        let mut o3 = vmlsl_laneq_s16::<4>(vmull_laneq_s16::<5>(c[1], w), c[3], w);
        if !LOW {
            e2 = vmlal_laneq_s16::<1>(e2, c[6], w);
            e3 = vmlsl_laneq_s16::<0>(e3, c[6], w);
            o0 = vmlal_laneq_s16::<5>(vmlal_laneq_s16::<4>(o0, c[5], w), c[7], w);
            o1 = vmlsl_laneq_s16::<4>(vmlsl_laneq_s16::<2>(o1, c[5], w), c[7], w);
            o2 = vmlal_laneq_s16::<3>(vmlal_laneq_s16::<5>(o2, c[5], w), c[7], w);
            o3 = vmlsl_laneq_s16::<2>(vmlal_laneq_s16::<3>(o3, c[5], w), c[7], w);
        }
        let a = vaddq_s32(e0, e2);
        let b = vaddq_s32(e1, e3);
        let c = vsubq_s32(e1, e3);
        let d = vsubq_s32(e0, e2);
        [
            vaddq_s32(a, o0),
            vaddq_s32(b, o1),
            vaddq_s32(c, o2),
            vaddq_s32(d, o3),
            vsubq_s32(d, o3),
            vsubq_s32(c, o2),
            vsubq_s32(b, o1),
            vsubq_s32(a, o0),
        ]
    }
}

#[inline(always)]
unsafe fn line<const SHIFT: i32, const LOW: bool>(
    c: [int16x4_t; SIDE],
    w: int16x8_t,
) -> [int16x4_t; SIDE] {
    // SAFETY: only entered by the NEON inverse.
    unsafe { butterfly::<LOW>(c, w).map(|v| vqrshrn_n_s32::<SHIFT>(v)) }
}

/// Both coefficient halves use the same four-line partial butterfly.
#[inline(always)]
unsafe fn pass<const SHIFT: i32, const LOW: bool>(
    c: [int16x8_t; SIDE],
    w: int16x8_t,
) -> [int16x8_t; SIDE] {
    // SAFETY: NEON is enabled. Constant loop bounds keep vectors in registers.
    unsafe {
        let mut low = [vget_low_s16(c[0]); SIDE];
        let mut high = [vget_high_s16(c[0]); SIDE];
        for i in 1..SIDE {
            low[i] = vget_low_s16(c[i]);
            high[i] = vget_high_s16(c[i]);
        }
        let low = line::<SHIFT, LOW>(low, w);
        let high = line::<SHIFT, LOW>(high, w);
        let mut out = [c[0]; SIDE];
        for i in 0..SIDE {
            out[i] = vcombine_s16(low[i], high[i]);
        }
        out
    }
}

#[inline(always)]
unsafe fn residual<const SHIFT: i32, const LOW: bool>(coeffs: &[i16]) -> [int16x8_t; SIDE] {
    let coeffs = &coeffs[..SIDE * SIDE];
    // SAFETY: full block and basis slices are checked and NEON is enabled.
    unsafe {
        let w = vld1q_s16(WEIGHTS.as_ptr());
        let zero = vget_low_s16(vdupq_n_s16(0));
        let col = if LOW {
            let mut top_left = [zero; SIDE];
            for (row, out) in top_left[..4].iter_mut().enumerate() {
                *out = vld1_s16(coeffs.as_ptr().add(row * SIDE));
            }
            let mut columns = [vcombine_s16(zero, zero); SIDE];
            for (output, left) in columns.iter_mut().zip(line::<7, true>(top_left, w)) {
                *output = vcombine_s16(left, zero);
            }
            columns
        } else {
            let mut rows = [vdupq_n_s16(0); SIDE];
            for (row, out) in rows.iter_mut().enumerate() {
                *out = vld1q_s16(coeffs.as_ptr().add(row * SIDE));
            }
            pass::<7, false>(rows, w)
        };
        transpose(pass::<SHIFT, LOW>(transpose(col), w))
    }
}

/// The same inverse writes signed residuals for HEVC and VVC encoders.
///
/// # Safety
/// NEON is enabled and both slices cover 8×8.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse(coeffs: &[i16], out: &mut [i16]) {
    let extent = super::generic::nonzero_extent::<SIDE>(coeffs);
    if extent == (0, 0) || extent == (1, 1) {
        // SAFETY: the nonzero span and both checked blocks are unchanged.
        unsafe {
            super::generic::hevc_inverse_extent::<I32x4, HevcIdct8, SIDE>(coeffs, out, extent);
        }
        return;
    }
    let out = &mut out[..SIDE * SIDE];
    // SAFETY: complete validated output rows and NEON are available.
    unsafe {
        let rows = if extent.0.max(extent.1) <= 4 {
            residual::<12, true>(coeffs)
        } else {
            residual::<12, false>(coeffs)
        };
        for (y, row) in rows.into_iter().enumerate() {
            vst1q_s16(out.as_mut_ptr().add(y * SIDE), row);
        }
    }
}

/// # Safety
/// NEON is enabled, the checked blocks cover 8×8 and the extent contains
/// every nonzero coefficient.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_u8(coeffs: &[i16], extent: (usize, usize), dst: &mut [u8], stride: usize) {
    if extent.0 == 0 || extent.1 == 0 || extent == (1, 1) {
        // SAFETY: same extent and buffer contract as the original dispatcher.
        unsafe {
            inverse_add::inverse_add::<I32x4, HevcIdct8, SIDE>(coeffs, extent, dst, stride, true);
        }
        return;
    }
    let dst = &mut dst[..(SIDE - 1) * stride + SIDE];
    // SAFETY: full blocks protected by the checked spans.
    unsafe {
        let residual = if extent.0.max(extent.1) <= 4 {
            residual::<12, true>(coeffs)
        } else {
            residual::<12, false>(coeffs)
        };
        for (y, row) in residual.into_iter().enumerate() {
            let ptr = dst.as_mut_ptr().add(y * stride);
            let prediction = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr)));
            vst1_u8(ptr, vqmovun_s16(vqaddq_s16(row, prediction)));
        }
    }
}

/// # Safety
/// As `add_u8` for u16 samples. The depth's clip applies after addition.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_u16(
    coeffs: &[i16],
    extent: (usize, usize),
    dst: &mut [u16],
    stride: usize,
    depth: BitDepth,
) {
    if !matches!(depth.bits(), 10 | 12) || extent.0 == 0 || extent.1 == 0 || extent == (1, 1) {
        // SAFETY: same extent and buffer contract as the original dispatcher.
        unsafe {
            inverse_add::inverse_add_u16::<I32x4, HevcIdct8, SIDE>(
                coeffs, extent, dst, stride, true, depth,
            );
        }
        return;
    }
    let dst = &mut dst[..(SIDE - 1) * stride + SIDE];
    // SAFETY: full blocks protected by the checked spans.
    unsafe {
        let residual = match (depth.bits(), extent.0.max(extent.1) <= 4) {
            (10, true) => residual::<10, true>(coeffs),
            (10, false) => residual::<10, false>(coeffs),
            (_, true) => residual::<8, true>(coeffs),
            _ => residual::<8, false>(coeffs),
        };
        let max = vreinterpretq_u16_s16(vdupq_n_s16(depth.max() as i16));
        let zero = vdupq_n_s16(0);
        for (y, row) in residual.into_iter().enumerate() {
            let ptr = dst.as_mut_ptr().add(y * stride);
            let prediction = vreinterpretq_s16_u16(vld1q_u16(ptr));
            let sum = vmaxq_s16(vqaddq_s16(row, prediction), zero);
            vst1q_u16(ptr, vminq_u16(vreinterpretq_u16_s16(sum), max));
        }
    }
}
