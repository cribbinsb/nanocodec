//! AVX-512 sample rows. Full chunks avoid mask construction; only the last
//! partial chunk uses masks. Moments reduce four blocks together in registers.
//! The pair-sum layout follows our AVX2 kernels; SVT-AV1's Intel 2021 BSD-2/AOM
//! `variance_avx512.c` also keeps full-width loads separate from edge handling.
use super::{Moments, SampleFns};
use core::arch::x86_64::{
    _mm_storeu_si128, _mm256_add_epi64, _mm256_cvtepi64_epi32, _mm256_loadu_si256,
    _mm256_mask_storeu_epi8, _mm256_sad_epu8, _mm256_setzero_si256, _mm256_storeu_si256,
    _mm512_add_epi16, _mm512_add_epi32, _mm512_and_si512, _mm512_castsi512_si128,
    _mm512_cvtepu8_epi16, _mm512_cvtusepi16_epi8, _mm512_loadu_si512, _mm512_madd_epi16,
    _mm512_maddubs_epi16, _mm512_maskz_compress_epi32, _mm512_maskz_loadu_epi8, _mm512_set1_epi8,
    _mm512_set1_epi16, _mm512_setzero_si512, _mm512_shuffle_epi32, _mm512_srli_epi16,
};

pub(crate) fn install(table: &mut SampleFns) {
    table.moments_8x8_row = moments::<8>;
    table.moments_8x4_row = moments::<4>;
    table.downscale_2x = resize::<true, false>;
    table.pair_mean_2x = resize::<false, false>;
    table.pair_pick_2x = resize::<false, true>;
}
const OUTPUTS: usize = 32;
const MOMENTS_GROUP: usize = 4;

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn resize<const BOX: bool, const PICK: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    let full = w / OUTPUTS * OUTPUTS;
    let tail = w - full;
    for y in 0..h {
        let row = y * if BOX { 2 } else { 1 };
        for x in (0..full).step_by(OUTPUTS) {
            // SAFETY: complete 64-byte input and 32-byte output spans.
            unsafe {
                chunk::<BOX, PICK, false>(
                    src.as_ptr().add(row * ss + 2 * x),
                    ss,
                    dst.as_mut_ptr().add(y * ds + x),
                    OUTPUTS,
                );
            }
        }
        if tail != 0 {
            // SAFETY: masks cover exactly the remaining input/output pixels.
            unsafe {
                chunk::<BOX, PICK, true>(
                    src.as_ptr().add(row * ss + 2 * full),
                    ss,
                    dst.as_mut_ptr().add(y * ds + full),
                    tail,
                );
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn chunk<const BOX: bool, const PICK: bool, const TAIL: bool>(
    src: *const u8,
    ss: usize,
    dst: *mut u8,
    count: usize,
) {
    let mask = u64::MAX >> (64 - 2 * count);
    // SAFETY: caller supplies a full span or a mask bounded by the visible row.
    let a = unsafe {
        if TAIL {
            _mm512_maskz_loadu_epi8(mask, src.cast())
        } else {
            _mm512_loadu_si512(src.cast())
        }
    };
    let value = if PICK {
        _mm512_and_si512(a, _mm512_set1_epi16(255))
    } else {
        let mut sum = _mm512_maddubs_epi16(a, _mm512_set1_epi8(1));
        if BOX {
            // SAFETY: the second input row has the same validated span.
            let b = unsafe {
                if TAIL {
                    _mm512_maskz_loadu_epi8(mask, src.add(ss).cast())
                } else {
                    _mm512_loadu_si512(src.add(ss).cast())
                }
            };
            sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(b, _mm512_set1_epi8(1)));
            _mm512_srli_epi16::<2>(_mm512_add_epi16(sum, _mm512_set1_epi16(2)))
        } else {
            _mm512_srli_epi16::<1>(_mm512_add_epi16(sum, _mm512_set1_epi16(1)))
        }
    };
    let bytes = _mm512_cvtusepi16_epi8(value);
    // SAFETY: store only the full output span or the masked visible tail.
    unsafe {
        if TAIL {
            _mm256_mask_storeu_epi8(dst.cast(), u32::MAX >> (32 - count), bytes);
        } else {
            _mm256_storeu_si256(dst.cast(), bytes);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn moments<const ROWS: usize>(src: &[u8], stride: usize, out: &mut [Moments]) {
    let (groups, tail) = out.as_chunks_mut::<MOMENTS_GROUP>();
    for (batch, blocks) in groups.iter_mut().enumerate() {
        let mut sums = _mm256_setzero_si256();
        let mut squares = _mm512_setzero_si512();
        for y in 0..ROWS {
            // SAFETY: these four blocks supply 32 complete source bytes per row.
            let bytes =
                unsafe { _mm256_loadu_si256(src.as_ptr().add(y * stride + batch * 32).cast()) };
            sums = _mm256_add_epi64(sums, _mm256_sad_epu8(bytes, _mm256_setzero_si256()));
            let words = _mm512_cvtepu8_epi16(bytes);
            squares = _mm512_add_epi32(squares, _mm512_madd_epi16(words, words));
        }
        // Each 128-bit lane holds one block. Reduce all four blocks together,
        // then gather one total per lane rather than extracting/scalar-summing.
        let pairs = _mm512_add_epi32(squares, _mm512_shuffle_epi32::<0x4e>(squares));
        let totals = _mm512_add_epi32(pairs, _mm512_shuffle_epi32::<0xb1>(pairs));
        let square = _mm512_castsi512_si128(_mm512_maskz_compress_epi32(0x1111, totals));
        let sum = _mm256_cvtepi64_epi32(sums);
        let mut sum_out = [0u32; MOMENTS_GROUP];
        let mut square_out = [0u32; MOMENTS_GROUP];
        // SAFETY: each local array holds four u32 lanes. No Moments layout assumption.
        unsafe {
            _mm_storeu_si128(sum_out.as_mut_ptr().cast(), sum);
            _mm_storeu_si128(square_out.as_mut_ptr().cast(), square);
        }
        for i in 0..MOMENTS_GROUP {
            blocks[i] = Moments {
                sum: sum_out[i],
                sum_sq: square_out[i],
            };
        }
    }
    let first = groups.len() * MOMENTS_GROUP * 8;
    // SAFETY: same validated contract; AVX2 is required. Short rows retain its
    // efficient single-block path, and tail handling stays outside the full loop.
    unsafe {
        super::avx2::moments_row::<ROWS>(&src[first..], stride, tail);
    }
}
