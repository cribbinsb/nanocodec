//! AVX2 conversion kernels (x86-64).
//!
//! Deinterleave: each 16-bit lane holds one (Cb, Cr) pair; masking the low
//! byte gives Cb and shifting right by 8 gives Cr, both as 16-bit lanes that
//! `vpackuswb` narrows without saturating. 32 pairs per iteration, with the
//! scalar row function for the tail.
//!
//! Kernel contract: see the `convert` module docs.

use core::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64,
    _mm256_set1_epi16, _mm256_srli_epi16, _mm256_storeu_si256,
};

use super::{ConvertFns, scalar};

/// Pairs split per iteration of the deinterleave loop.
const DEINTERLEAVE_STEP: usize = 32;

/// The AVX2 kernel table.
///
/// # Safety
///
/// The CPU must support AVX2.
pub(crate) unsafe fn convert_fns() -> ConvertFns {
    ConvertFns {
        deinterleave_uv,
        i420_to_rgba_rows: super::to_rgba_avx2::i420_to_rgba_rows,
        rgba_to_i420: super::rgba_avx2::rgba_to_i420,
        sse_row: scalar::sse_row,
    }
}

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[allow(clippy::too_many_arguments, reason = "matches DeinterleaveUvFn")]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_uv(
    src: &[u8],
    src_stride: usize,
    dst_u: &mut [u8],
    u_stride: usize,
    dst_v: &mut [u8],
    v_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DEINTERLEAVE_STEP;
    let low_bytes = _mm256_set1_epi16(0x00ff);
    for y in 0..h {
        let pairs = &src[y * src_stride..y * src_stride + 2 * w];
        let cb_row = &mut dst_u[y * u_stride..y * u_stride + w];
        let cr_row = &mut dst_v[y * v_stride..y * v_stride + w];
        for x in (0..simd_width).step_by(DEINTERLEAVE_STEP) {
            // SAFETY: source bytes 2x .. 2x + 64 lie inside the 2w-byte row
            // and outputs x .. x + 32 inside the w-sample rows, since
            // x + 32 <= w (kernel contract).
            unsafe {
                let start = pairs.as_ptr().add(2 * x);
                let low: __m256i = _mm256_loadu_si256(start.cast());
                let high: __m256i = _mm256_loadu_si256(start.add(32).cast());
                // packus interleaves 128-bit lanes: [low0-7, high0-7,
                // low8-15, high8-15]; the permute restores the order.
                let cb = _mm256_packus_epi16(
                    _mm256_and_si256(low, low_bytes),
                    _mm256_and_si256(high, low_bytes),
                );
                let cr =
                    _mm256_packus_epi16(_mm256_srli_epi16::<8>(low), _mm256_srli_epi16::<8>(high));
                _mm256_storeu_si256(
                    cb_row.as_mut_ptr().add(x).cast(),
                    _mm256_permute4x64_epi64::<0b11_01_10_00>(cb),
                );
                _mm256_storeu_si256(
                    cr_row.as_mut_ptr().add(x).cast(),
                    _mm256_permute4x64_epi64::<0b11_01_10_00>(cr),
                );
            }
        }
        scalar::deinterleave_uv_row(
            &pairs[2 * simd_width..],
            &mut cb_row[simd_width..],
            &mut cr_row[simd_width..],
        );
    }
}
