//! Four H.264 4x4 blocks per vector through the shared one-dimensional
//! transform lines. Row-before-column order and the final inverse rounding
//! are preserved. Short batches retain the existing four-lane driver.
use super::generic::{self as g, H264Fdct4, H264Fdct8Paired, H264Idct4, Line};
use super::{CodecTxfmFns, H264Quantizer, QUANT_MF_8X8_POS};
use crate::simd::{
    I32Lanes,
    avx2::{I32x4, I32x8},
    avx512::I32x16,
};

pub(crate) fn install(table: &mut CodecTxfmFns) {
    table.h264_forward_4x4 = forward;
    table.h264_forward_quantize_mb = forward_quantize_mb;
    table.h264_inverse_4x4 = inverse;
}

/// Shared H.264 lines, with four blocks carried in independent sublanes.
#[inline]
fn transform<T: Line<I32x16>>(rows: [I32x16; 4], shift: u32) -> [I32x16; 4] {
    let columns = I32x16::transpose4(rows);
    let mut temp = [I32x16::splat(0); 4];
    T::apply(&columns, &mut temp);
    let rows = I32x16::transpose4(temp);
    T::apply(&rows, &mut temp);
    for row in &mut temp {
        *row = row.round_shift(shift);
    }
    I32x16::transpose_chunks(temp)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward(src: &[i16], stride: usize, blocks_wide: usize, out: &mut [[i16; 16]]) {
    let full = out.len() / 4 * 4;
    for first in (0..full).step_by(4) {
        let mut rows = [I32x16::splat(0); 4];
        for (r, row) in rows.iter_mut().enumerate() {
            if first % blocks_wide + 4 <= blocks_wide {
                let origin = (first / blocks_wide * 4 + r) * stride + first % blocks_wide * 4;
                *row = I32x16::load_i16(&src[origin..]);
            } else {
                let mut values = [0i16; 16];
                for (i, block) in values.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let b = first + i;
                    let origin = (b / blocks_wide * 4 + r) * stride + b % blocks_wide * 4;
                    block.copy_from_slice(&src[origin..origin + 4]);
                }
                *row = I32x16::load_i16(&values);
            }
        }
        for (block, value) in out[first..first + 4]
            .iter_mut()
            .zip(transform::<H264Fdct4>(rows, 0))
        {
            value.store_sat16(block);
        }
    }
    for (index, block) in out[full..].iter_mut().enumerate() {
        let b = full + index;
        let origin = b / blocks_wide * 4 * stride + b % blocks_wide * 4;
        // SAFETY: exactly one validated 4x4 block; AVX2 is available.
        unsafe {
            g::h264_forward_4x4::<I32x4>(&src[origin..], stride, 1, core::slice::from_mut(block));
        }
    }
}

macro_rules! from_coefficients {
    ($name:ident,$line:ty,$shift:literal,$fallback:ident) => {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn $name(src: &[[i32; 16]], out: &mut [[i32; 16]]) {
            let full = src.len() / 4 * 4;
            for first in (0..full).step_by(4) {
                let mut blocks = [I32x16::splat(0); 4];
                for (i, b) in blocks.iter_mut().enumerate() {
                    *b = I32x16::load(&src[first + i]);
                }
                let rows = I32x16::transpose_chunks(blocks);
                for (block, value) in out[first..first + 4]
                    .iter_mut()
                    .zip(transform::<$line>(rows, $shift))
                {
                    value.store(block);
                }
            }
            // SAFETY: validated remaining blocks; AVX2 is available.
            unsafe {
                g::$fallback::<I32x4>(&src[full..], &mut out[full..]);
            }
        }
    };
}
from_coefficients!(inverse, H264Idct4, 6, h264_inverse_4x4);

/// Run two horizontally adjacent transforms in independent eight-lane
/// halves, and quantise the resulting rows while they are in registers.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_quantize_pair(
    residual: *const i16,
    stride: usize,
    mf: &[[i32; 8]; 8],
    qbits: u32,
    offset: i32,
    out: &mut [[i16; 64]; 2],
) -> [u32; 2] {
    let rows = core::array::from_fn(|r| {
        // SAFETY: the caller validated the eight rows of sixteen samples.
        unsafe { I32x16::load_i16_row(residual.add(r * stride)) }
    });
    let columns = I32x16::transpose8_pair(&rows);
    let mut scratch = [I32x16::splat(0); 8];
    H264Fdct8Paired::apply(&columns, &mut scratch);
    let rows = I32x16::transpose8_pair(&scratch);
    H264Fdct8Paired::apply(&rows, &mut scratch);

    let offset = I32x16::splat(offset);
    let (mut left_flags, mut right_flags) = (0u64, 0u64);
    let [left, right] = out;
    for (i, ((raw, multiplier), (l, r))) in scratch
        .into_iter()
        .zip(mf)
        .zip(
            left.as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(right.as_chunks_mut::<8>().0),
        )
        .enumerate()
    {
        // The transform matrix's row L1 norm is at most 64. With the
        // final shift by six and |residual| <= 255, the coefficients fit
        // the i16 values the separate forward pass would store. The
        // multipliers are smaller than 2^15 and qbits >= 16, so the
        // resulting levels also fit signed words without clipping.
        let raw = raw.round_shift(6);
        let level = raw
            .abs()
            .mul(I32x16::repeat_eight(multiplier))
            .add(offset)
            .shr_logical(qbits)
            .with_sign_of(raw);
        let flags = level.store_two_rows(l, r);
        left_flags |= u64::from(flags as u8) << (8 * i);
        right_flags |= u64::from(flags >> 8) << (8 * i);
    }
    [left_flags.count_ones(), right_flags.count_ones()]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_quantize_mb(
    residual: &[i16],
    stride: usize,
    zero_mask: u8,
    q: &H264Quantizer,
    out: &mut [[i16; 64]; 4],
) -> [u32; 4] {
    if zero_mask != 0 {
        // SAFETY: the tier includes AVX2; the caller covers the macroblock.
        return unsafe {
            g::h264_forward_quantize_mb::<g::LanesForward8x8<I32x8>, I32x8>(
                residual, stride, zero_mask, q, out,
            )
        };
    }
    let (qbits, offset) = q.forward_terms(1);
    let (mf, _) = QUANT_MF_8X8_POS[q.rem].as_chunks::<8>();
    let mf: &[[i32; 8]; 8] = mf
        .try_into()
        .expect("one quantiser row per coefficient row");
    let ([top, bottom], _) = out.as_chunks_mut::<2>() else {
        unreachable!("a macroblock holds four transforms")
    };
    // SAFETY: the caller covers all four transforms and this tier supports
    // AVX-512F/BW/VL.
    let ([top_left, top_right], [bottom_left, bottom_right]) = unsafe {
        (
            forward_quantize_pair(residual.as_ptr(), stride, mf, qbits, offset, top),
            forward_quantize_pair(
                residual.as_ptr().add(8 * stride),
                stride,
                mf,
                qbits,
                offset,
                bottom,
            ),
        )
    };
    [top_left, top_right, bottom_left, bottom_right]
}
