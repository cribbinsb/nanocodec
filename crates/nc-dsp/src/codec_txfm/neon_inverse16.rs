//! HEVC DCT-II 16×16 using the four-column lane-MAC and transposed
//! scratch schedule of libhevc's `ihevc_itrans_recon_16x16_av8`
//! (Apache-2.0, Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore).

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int32x4_t, vaddq_s32, vdup_n_s16, vdupq_n_s16, vdupq_n_s32, vdupq_n_u16,
    vld1_s16, vld1_u8, vld1q_s16, vld1q_u16, vmaxq_s16, vminq_u16, vmlal_laneq_s16, vmovl_u8,
    vmull_laneq_s16, vqaddq_s16, vqmovun_s16, vqrshrn_n_s32, vreinterpretq_s16_u16,
    vreinterpretq_u16_s16, vst1_s16, vst1_u8, vst1q_u16, vsubq_s32,
};
use nc_common::BitDepth;

use super::{DCT32, generic::HevcIdct16, inverse_add, neon_inverse4::transpose4, neon_inverse8};
use crate::simd::neon::I32x4;

const SIDE: usize = 16;
const AREA: usize = SIDE * SIDE;
const ODD: [[i16; 8]; 8] = {
    let mut values = [[0; 8]; 8];
    let mut x = 0;
    while x < 8 {
        let mut k = 0;
        while k < 8 {
            values[x][k] = DCT32[4 * k + 2][x] as i16;
            k += 1;
        }
        x += 1;
    }
    values
};

#[inline(always)]
pub(super) unsafe fn butterfly<const LIMIT: usize>(
    c: [int16x4_t; 16],
    even_weights: int16x8_t,
) -> [int32x4_t; 16] {
    // SAFETY: NEON is enabled. The coefficient bank is the same HEVC
    // DCT-II even-frequency basis used by the 16-point transform.
    unsafe {
        let input = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
        let even = if LIMIT <= 8 {
            neon_inverse8::butterfly::<true>(input, even_weights)
        } else {
            neon_inverse8::butterfly::<false>(input, even_weights)
        };
        let mut out = [vdupq_n_s32(0); 16];
        for x in 0..8 {
            let odd_inputs = [c[1], c[3], c[5], c[7], c[9], c[11], c[13], c[15]];
            let odd = odd_dot::<LIMIT>(x, odd_inputs);
            out[x] = vaddq_s32(even[x], odd);
            out[15 - x] = vsubq_s32(even[x], odd);
        }
        out
    }
}

#[inline(always)]
unsafe fn odd_dot<const LIMIT: usize>(x: usize, c: [int16x4_t; 8]) -> int32x4_t {
    // SAFETY: NEON is enabled and x selects an eight-word coefficient row.
    unsafe {
        let w = vld1q_s16(ODD[x].as_ptr());
        let mut odd = vmull_laneq_s16::<0>(c[0], w);
        if LIMIT > 4 {
            odd = vmlal_laneq_s16::<1>(odd, c[1], w);
        }
        if LIMIT > 8 {
            odd = vmlal_laneq_s16::<2>(odd, c[2], w);
            odd = vmlal_laneq_s16::<3>(odd, c[3], w);
        }
        if LIMIT > 16 {
            odd = vmlal_laneq_s16::<4>(odd, c[4], w);
            odd = vmlal_laneq_s16::<5>(odd, c[5], w);
            odd = vmlal_laneq_s16::<6>(odd, c[6], w);
            odd = vmlal_laneq_s16::<7>(odd, c[7], w);
        }
        odd
    }
}

#[inline(always)]
unsafe fn line<const SHIFT: i32, const ROWS: usize>(
    c: [int16x4_t; SIDE],
    even_weights: int16x8_t,
) -> [int16x4_t; SIDE] {
    // SAFETY: only used by the NEON entries, with fixed-size input.
    unsafe {
        let coefficients = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
        let even = if ROWS <= 8 {
            neon_inverse8::butterfly::<true>(coefficients, even_weights)
        } else {
            neon_inverse8::butterfly::<false>(coefficients, even_weights)
        };
        let odd_inputs = [c[1], c[3], c[5], c[7], c[9], c[11], c[13], c[15]];
        let mut out = [vdup_n_s16(0); SIDE];
        for x in 0..8 {
            let odd = match ROWS {
                4 => odd_dot::<8>(x, odd_inputs),
                8 => odd_dot::<16>(x, odd_inputs),
                _ => odd_dot::<32>(x, odd_inputs),
            };
            out[x] = vqrshrn_n_s32::<SHIFT>(vaddq_s32(even[x], odd));
            out[SIDE - 1 - x] = vqrshrn_n_s32::<SHIFT>(vsubq_s32(even[x], odd));
        }
        out
    }
}

#[inline(always)]
unsafe fn pass<const SHIFT: i32, const ROWS: usize>(
    input: &[i16; AREA],
    output: &mut [i16; AREA],
    columns: usize,
    even_weights: int16x8_t,
) {
    // SAFETY: every column group fits the 16×16 raster. Partial inputs
    // have zero coefficients after the selected row bound in this pass.
    unsafe {
        for x in (0..columns).step_by(4) {
            let mut c = [vdup_n_s16(0); SIDE];
            for (y, row) in c[..ROWS].iter_mut().enumerate() {
                *row = vld1_s16(input.as_ptr().add(y * SIDE + x));
            }
            let transformed = line::<SHIFT, ROWS>(c, even_weights);
            for (y, &group) in transformed.as_chunks::<4>().0.iter().enumerate() {
                for (i, row) in transpose4(group).into_iter().enumerate() {
                    vst1_s16(output.as_mut_ptr().add((x + i) * SIDE + 4 * y), row);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn residual<const SHIFT: i32>(coeffs: &[i16], sparse: bool, output: &mut [i16; AREA]) {
    let coeffs = coeffs.first_chunk::<AREA>().expect("complete DCT block");
    let mut scratch = [0; AREA];
    // SAFETY: both passes cover full block arrays and NEON is enabled.
    unsafe {
        let even_weights = vld1q_s16(neon_inverse8::WEIGHTS.as_ptr());
        if sparse {
            pass::<7, 4>(coeffs, &mut scratch, 4, even_weights);
            pass::<SHIFT, 4>(&scratch, output, SIDE, even_weights);
        } else {
            pass::<7, SIDE>(coeffs, &mut scratch, SIDE, even_weights);
            pass::<SHIFT, SIDE>(&scratch, output, SIDE, even_weights);
        }
    }
}

/// The encoder uses independent row and column extents from the sparse
/// coefficient block. Each pass applies the smallest covered butterfly.
#[inline(always)]
unsafe fn residual_with_extent(
    coeffs: &[i16],
    (rows, columns): (usize, usize),
    out: &mut [i16; AREA],
) {
    let coeffs = coeffs.first_chunk::<AREA>().expect("complete DCT block");
    let mut scratch = [0; AREA];
    let groups = columns.next_multiple_of(4);
    // SAFETY: the selected bounds cover nonzero coefficients. The first
    // pass initializes the transposed rows the second pass can reach.
    unsafe {
        let weights = vld1q_s16(neon_inverse8::WEIGHTS.as_ptr());
        match rows {
            0..=4 => pass::<7, 4>(coeffs, &mut scratch, groups, weights),
            5..=8 => pass::<7, 8>(coeffs, &mut scratch, groups, weights),
            _ => pass::<7, SIDE>(coeffs, &mut scratch, groups, weights),
        }
        match columns {
            0..=4 => pass::<12, 4>(&scratch, out, SIDE, weights),
            5..=8 => pass::<12, 8>(&scratch, out, SIDE, weights),
            _ => pass::<12, SIDE>(&scratch, out, SIDE, weights),
        }
    }
}

/// Signed DCT16 residuals for HEVC and VVC, with the same packed passes
/// as the decoder's reconstruction.
/// # Safety
/// NEON is enabled and both slices cover the complete transform.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse(coeffs: &[i16], out: &mut [i16]) {
    let extent = super::generic::nonzero_extent::<SIDE>(coeffs);
    if extent == (0, 0) || extent == (1, 1) {
        // SAFETY: identical validated block and coefficient extent.
        unsafe {
            super::generic::hevc_inverse_extent::<I32x4, HevcIdct16, SIDE>(coeffs, out, extent);
        }
        return;
    }
    let out = out
        .first_chunk_mut::<AREA>()
        .expect("complete residual block");
    // SAFETY: the checked arrays and extent cover both NEON passes.
    unsafe { residual_with_extent(coeffs, extent, out) };
}

/// # Safety
/// NEON available, coefficient and destination spans cover the block.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_u8(coeffs: &[i16], extent: (usize, usize), dst: &mut [u8], stride: usize) {
    if extent.0 == 0 || extent.1 == 0 {
        return;
    }
    if extent == (1, 1) {
        inverse_add::dc::<I32x4, SIDE>(coeffs[0], dst, stride);
        return;
    }
    let dst = &mut dst[..(SIDE - 1) * stride + SIDE];
    let mut block = [0; AREA];
    // SAFETY: the checked destination and residual cover all 16 rows.
    unsafe {
        residual::<12>(coeffs, extent.0.max(extent.1) <= 4, &mut block);
        for row in 0..SIDE {
            for col in [0, 8] {
                let out = dst.as_mut_ptr().add(row * stride + col);
                let raw = vld1q_s16(block.as_ptr().add(row * SIDE + col));
                let pred = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(out)));
                vst1_u8(out, vqmovun_s16(vqaddq_s16(raw, pred)));
            }
        }
    }
}

/// # Safety
/// As `add_u8`, with the supplied sample depth.
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
    let dst = &mut dst[..(SIDE - 1) * stride + SIDE];
    let mut block = [0; AREA];
    let sparse = extent.0.max(extent.1) <= 4;
    // SAFETY: the full block is covered by the validated buffers.
    unsafe {
        match depth.bits() {
            10 => residual::<10>(coeffs, sparse, &mut block),
            12 => residual::<8>(coeffs, sparse, &mut block),
            _ => {
                inverse_add::inverse_add_u16::<I32x4, HevcIdct16, SIDE>(
                    coeffs, extent, dst, stride, true, depth,
                );
                return;
            }
        }
        let zero = vdupq_n_s16(0);
        let max = vdupq_n_u16(depth.max() as u16);
        for row in 0..SIDE {
            for col in [0, 8] {
                let out = dst.as_mut_ptr().add(row * stride + col);
                let raw = vld1q_s16(block.as_ptr().add(row * SIDE + col));
                let pred = vreinterpretq_s16_u16(vld1q_u16(out));
                let sum = vmaxq_s16(vqaddq_s16(raw, pred), zero);
                vst1q_u16(out, vminq_u16(vreinterpretq_u16_s16(sum), max));
            }
        }
    }
}
