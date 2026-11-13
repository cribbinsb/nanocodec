//! Four H.264 4x4 blocks per vector through the shared one-dimensional
//! transform lines. Row-before-column order and the final inverse rounding
//! are preserved. Short batches retain the existing four-lane driver.
use super::CodecTxfmFns;
use super::generic::{self as g, H264Fdct4, H264Idct4, Line};
use crate::simd::{I32Lanes, avx2::I32x4, avx512::I32x16};

pub(crate) fn install(table: &mut CodecTxfmFns) {
    table.h264_forward_4x4 = forward;
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
unsafe fn forward(src: &[i16], stride: usize, blocks_wide: usize, out: &mut [[i32; 16]]) {
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
            value.store(block);
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
