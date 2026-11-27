//! VP9's byte-clipped intermediate uses the common 32-output AVX-512
//! byte-dot rows. The unsigned offset from the VP9 AVX2 path accommodates
//! full-scale taps whose completed sum can exceed i16.

use super::avx2::{OFFSET, OFFSET_UNITS};
use super::driver::{self, ROUND, RowKernel, SHIFT, layout};
use crate::Vp9Filter;
use crate::mc::av1_avx512::{
    byte_pairs, column_32, gather_vectors, row_32, spread_32, vertical_32,
};
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_avg_epu8, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi16,
    _mm512_cvtusepi16_epi8, _mm512_set1_epi16, _mm512_srli_epi16, _mm512_subs_epu16,
};

struct Avx512;

struct Filters {
    weights: [__m512i; 4],
    gathers: [__m512i; 4],
    centre_weight: [__m512i; 1],
    centre_gather: [__m512i; 1],
}

/// Undo the offset after the rounding shift. The unsigned subtract
/// clamps below zero, and the unsigned narrow clips to 255 in lane order.
#[inline(always)]
unsafe fn finish(sum: __m512i) -> __m256i {
    // SAFETY: the caller enables the AVX-512 tier.
    unsafe {
        let biased = _mm512_add_epi16(sum, _mm512_set1_epi16(ROUND + OFFSET));
        _mm512_cvtusepi16_epi8(_mm512_subs_epu16(
            _mm512_srli_epi16::<SHIFT>(biased),
            _mm512_set1_epi16(OFFSET_UNITS),
        ))
    }
}

impl RowKernel for Avx512 {
    type Filters = Filters;
    type Row = __m256i;
    const LANES: usize = 32;
    const COPY_ISA: crate::Isa = crate::Isa::Avx512;

    #[inline(always)]
    unsafe fn prepare<const LAYOUT: u8>(taps: &[i16; 8]) -> Filters {
        // SAFETY: the entry enables AVX-512. Nonidentity taps fit i8.
        unsafe {
            Filters {
                weights: byte_pairs(core::array::from_fn(|p| (taps[2 * p], taps[2 * p + 1]))),
                gathers: gather_vectors(),
                centre_weight: byte_pairs([(taps[3], taps[4])]),
                centre_gather: gather_vectors(),
            }
        }
    }

    #[inline(always)]
    unsafe fn load<const N: usize>(src: *const u8) -> __m256i {
        const { assert!(N == 32) };
        // SAFETY: src provides a row of 32 readable bytes.
        unsafe { _mm256_loadu_si256(src.cast()) }
    }

    #[inline(always)]
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: __m256i) {
        const { assert!(N == 32) };
        // SAFETY: dst provides 32 writable (and for AVG readable) bytes.
        unsafe {
            let row = if AVG {
                _mm256_avg_epu8(row, Self::load::<N>(dst))
            } else {
                row
            };
            _mm256_storeu_si256(dst.cast(), row);
        }
    }

    #[inline(always)]
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(src: *const u8, f: &Filters) -> __m256i {
        const { assert!(N == 32) };
        // SAFETY: the caller provides N + 7 readable bytes. The bilinear
        // centre pair reads only N + 1 bytes starting at tap 3.
        unsafe {
            if LAYOUT == layout::BILINEAR {
                finish(row_32(src.add(3), &f.centre_gather, &f.centre_weight))
            } else {
                finish(row_32(src, &f.gathers, &f.weights))
            }
        }
    }

    #[inline(always)]
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(
        rows: &[__m256i; 8],
        f: &Filters,
    ) -> __m256i {
        const { assert!(N == 32) };
        // SAFETY: the entry enables AVX-512.
        unsafe {
            let mut spread = rows.map(|row| spread_32(row));
            if LAYOUT == layout::BILINEAR {
                spread[0] = spread[3];
                spread[1] = spread[4];
                finish(vertical_32(&spread, &f.centre_weight))
            } else {
                finish(vertical_32(&spread, &f.weights))
            }
        }
    }

    #[inline(always)]
    unsafe fn column<const N: usize, const LAYOUT: u8, const AVG: bool>(
        src: *const u8,
        stride: usize,
        (dst, dst_stride): (*mut u8, usize),
        h: usize,
        f: &Filters,
    ) {
        // SAFETY: the driver validates h + 7 source rows (or the centre pair)
        // and h destination rows, each at least N (32) bytes.
        unsafe {
            let store = |y, sum| Self::store::<N, AVG>(dst.add(y * dst_stride), finish(sum));
            if LAYOUT == layout::BILINEAR {
                column_32(
                    h,
                    &f.centre_weight,
                    |y| Self::load::<N>(src.add((y + 3) * stride)),
                    store,
                );
            } else {
                column_32(
                    h,
                    &f.weights,
                    |y| Self::load::<N>(src.add(y * stride)),
                    store,
                );
            }
        }
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi")]
pub(super) unsafe fn convolve(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
    average: bool,
) {
    // SAFETY: Dsp validated the buffers and this entry enables the tier.
    unsafe {
        if size.0 == 0 || size.1 == 0 || !size.0.is_multiple_of(32) || phase == (0, 0) {
            super::avx2::convolve(kind, src, stride, phase, size, dst, dst_stride, average);
            return;
        }
        macro_rules! filter {
            ($layout:expr, $avg:expr) => {
                driver::convolve::<Avx512, 32, $layout, $avg>(
                    kind, src, stride, phase, size, dst, dst_stride,
                )
            };
        }
        match (kind, average) {
            (Vp9Filter::Bilinear, false) => filter!({ layout::BILINEAR }, false),
            (Vp9Filter::Bilinear, true) => filter!({ layout::BILINEAR }, true),
            (_, false) => filter!({ layout::ALTERNATING }, false),
            (_, true) => filter!({ layout::ALTERNATING }, true),
        }
    }
}
