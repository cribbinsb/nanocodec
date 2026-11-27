//! HEVC DCT-II 32×32 using the four-column lane-MAC and transposed
//! scratch schedule of libhevc's `ihevc_itrans_recon_32x32_av8`
//! (Apache-2.0, Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore).

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, vaddq_s32, vcombine_s16, vdup_n_s16, vdupq_n_s16, vdupq_n_u16, vld1_s16,
    vld1q_s16, vld1q_u16, vmaxq_s16, vminq_u16, vmlal_laneq_s16, vmull_laneq_s16, vqaddq_s16,
    vqrshrn_n_s32, vreinterpretq_s16_u16, vreinterpretq_u16_s16, vst1_s16, vst1q_u16, vsubq_s32,
};
use core::mem::MaybeUninit;
use nc_common::BitDepth;

use super::{
    DCT32, generic::HevcIdct32, inverse_add, neon_inverse4::transpose4, neon_inverse8,
    neon_inverse16,
};
use crate::simd::neon::I32x4;

const SIDE: usize = 32;
const AREA: usize = SIDE * SIDE;
const ODD: [[[i16; 8]; 2]; 16] = {
    let mut values = [[[0; 8]; 2]; 16];
    let mut x = 0;
    while x < 16 {
        let mut k = 0;
        while k < 16 {
            values[x][k / 8][k % 8] = DCT32[2 * k + 1][x] as i16;
            k += 1;
        }
        x += 1;
    }
    values
};

#[inline(always)]
unsafe fn line<const SHIFT: i32, const LIMIT: usize>(
    c: [int16x4_t; SIDE],
    even_weights: int16x8_t,
) -> [int16x4_t; SIDE] {
    // SAFETY: NEON is enabled. The widest dword sum has 32 terms
    // of magnitude at most 32768 × 90.
    unsafe {
        let even =
            neon_inverse16::butterfly::<LIMIT>(core::array::from_fn(|k| c[2 * k]), even_weights);
        let mut out = [vdup_n_s16(0); SIDE];
        for x in 0..16 {
            let w = vld1q_s16(ODD[x][0].as_ptr());
            let mut odd = vmlal_laneq_s16::<1>(vmull_laneq_s16::<0>(c[1], w), c[3], w);
            if LIMIT > 4 {
                odd = vmlal_laneq_s16::<2>(odd, c[5], w);
                odd = vmlal_laneq_s16::<3>(odd, c[7], w);
            }
            if LIMIT > 8 {
                odd = vmlal_laneq_s16::<4>(odd, c[9], w);
                odd = vmlal_laneq_s16::<5>(odd, c[11], w);
                odd = vmlal_laneq_s16::<6>(odd, c[13], w);
                odd = vmlal_laneq_s16::<7>(odd, c[15], w);
            }
            if LIMIT > 16 {
                let w = vld1q_s16(ODD[x][1].as_ptr());
                odd = vmlal_laneq_s16::<0>(odd, c[17], w);
                odd = vmlal_laneq_s16::<1>(odd, c[19], w);
                odd = vmlal_laneq_s16::<2>(odd, c[21], w);
                odd = vmlal_laneq_s16::<3>(odd, c[23], w);
                odd = vmlal_laneq_s16::<4>(odd, c[25], w);
                odd = vmlal_laneq_s16::<5>(odd, c[27], w);
                odd = vmlal_laneq_s16::<6>(odd, c[29], w);
                odd = vmlal_laneq_s16::<7>(odd, c[31], w);
            }
            out[x] = vqrshrn_n_s32::<SHIFT>(vaddq_s32(even[x], odd));
            out[SIDE - 1 - x] = vqrshrn_n_s32::<SHIFT>(vsubq_s32(even[x], odd));
        }
        out
    }
}

#[inline(always)]
unsafe fn pass<const SHIFT: i32, const LIMIT: usize>(
    input: &[i16; AREA],
    output: *mut i16,
    columns: usize,
    even_weights: int16x8_t,
) {
    // SAFETY: every column group fits the 32×32 raster. Partial inputs
    // have zero coefficients after LIMIT rows in this pass.
    unsafe {
        for x in (0..columns).step_by(4) {
            let mut c = [vdup_n_s16(0); SIDE];
            for (y, row) in c[..LIMIT].iter_mut().enumerate() {
                *row = vld1_s16(input.as_ptr().add(y * SIDE + x));
            }
            let transformed = line::<SHIFT, LIMIT>(c, even_weights);
            for (y, &group) in transformed.as_chunks::<4>().0.iter().enumerate() {
                for (i, row) in transpose4(group).into_iter().enumerate() {
                    vst1_s16(output.add((x + i) * SIDE + 4 * y), row);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn reconstruct<const SHIFT: i32, const LIMIT: usize, T>(
    coeffs: &[i16],
    dst: *mut T,
    stride: usize,
    mut add8: impl FnMut(*mut T, int16x8_t),
) {
    let coeffs = coeffs.first_chunk::<AREA>().expect("complete DCT block");
    let mut scratch = MaybeUninit::<[i16; AREA]>::uninit();
    let scratch = scratch.as_mut_ptr().cast::<i16>();
    // SAFETY: the first pass writes all LIMIT rows of the transposed scratch.
    // Each second-pass group then reconstructs four rows at all 32 columns.
    unsafe {
        let even_weights = vld1q_s16(neon_inverse8::WEIGHTS.as_ptr());
        pass::<7, LIMIT>(coeffs, scratch, LIMIT, even_weights);
        for y in (0..SIDE).step_by(4) {
            let mut c = [vdup_n_s16(0); SIDE];
            for (x, lane) in c[..LIMIT].iter_mut().enumerate() {
                *lane = vld1_s16(scratch.add(x * SIDE + y));
            }
            let sums = line::<SHIFT, LIMIT>(c, even_weights);
            for (x, group) in sums.as_chunks::<8>().0.iter().enumerate() {
                let a = transpose4([group[0], group[1], group[2], group[3]]);
                let b = transpose4([group[4], group[5], group[6], group[7]]);
                for row in 0..4 {
                    add8(
                        dst.add((y + row) * stride + x * 8),
                        vcombine_s16(a[row], b[row]),
                    );
                }
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn reconstruct_extent<const SHIFT: i32, T>(
    coeffs: &[i16],
    extent: (usize, usize),
    dst: &mut [T],
    stride: usize,
    add8: impl FnMut(*mut T, int16x8_t),
) {
    let dst = dst[..(SIDE - 1) * stride + SIDE].as_mut_ptr();
    // SAFETY: NEON is enabled and coefficients outside the extent are zero.
    unsafe {
        match extent.0.max(extent.1) {
            0..=4 => reconstruct::<SHIFT, 4, T>(coeffs, dst, stride, add8),
            5..=8 => reconstruct::<SHIFT, 8, T>(coeffs, dst, stride, add8),
            9..=16 => reconstruct::<SHIFT, 16, T>(coeffs, dst, stride, add8),
            _ => reconstruct::<SHIFT, SIDE, T>(coeffs, dst, stride, add8),
        }
    }
}

/// # Safety
/// NEON is enabled and the buffers cover a full DCT32 block at `depth`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_u16(
    coeffs: &[i16],
    extent: (usize, usize),
    dst: &mut [u16],
    stride: usize,
    depth: BitDepth,
) {
    if extent.0 == 0 || extent.1 == 0 {
        return;
    }
    if extent == (1, 1) {
        inverse_add::dc_u16::<I32x4, SIDE>(coeffs[0], dst, stride, depth);
        return;
    }
    // SAFETY: the dispatcher validated the destination and enabled NEON.
    unsafe {
        let zero = vdupq_n_s16(0);
        let max = vdupq_n_u16(depth.max() as u16);
        let add8 = |out: *mut u16, raw| {
            let pred = vreinterpretq_s16_u16(vld1q_u16(out));
            let sum = vmaxq_s16(vqaddq_s16(raw, pred), zero);
            vst1q_u16(out, vminq_u16(vreinterpretq_u16_s16(sum), max));
        };
        match depth.bits() {
            10 => reconstruct_extent::<10, _>(coeffs, extent, dst, stride, add8),
            12 => reconstruct_extent::<8, _>(coeffs, extent, dst, stride, add8),
            _ => inverse_add::inverse_add_u16::<I32x4, HevcIdct32, SIDE>(
                coeffs, extent, dst, stride, true, depth,
            ),
        }
    }
}
