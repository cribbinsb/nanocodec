//! Four independent 8x8 Hadamard tiles in AVX-512's 128-bit sublanes.
//! Uses the existing AVX2 vertical-butterfly design; horizontal butterflies
//! use lane shuffles and masked subtraction instead of unavailable 512-bit
//! horizontal-add instructions. Every coefficient fits i16 (64*255=16320).

use core::arch::x86_64::{
    __m512i, _mm_loadu_si128, _mm256_loadu_si256, _mm256_set_m128i, _mm256_setr_epi64x,
    _mm512_abs_epi16, _mm512_add_epi16, _mm512_add_epi32, _mm512_cvtepu8_epi16, _mm512_madd_epi16,
    _mm512_mask_sub_epi16, _mm512_set1_epi16, _mm512_setzero_si512, _mm512_shuffle_epi32,
    _mm512_shufflehi_epi16, _mm512_shufflelo_epi16, _mm512_storeu_si512, _mm512_sub_epi16,
};

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn tiles(src: &[u8], ss: usize, rf: &[u8], rs: usize, w: usize, h: usize) -> u32 {
    // Narrow tile gathering loses on Zen 5 / SPR (0.46–0.75x at 16x16/8x32).
    if w < 32 || w * h < 256 {
        // SAFETY: the same block contract, and AVX2 is part of this tier.
        return unsafe { super::avx2::satd_8x8_tiles(src, ss, rf, rs, w, h) };
    }
    // SAFETY: the same block contract, with four complete tiles per batch.
    unsafe { wide_tiles(src, ss, rf, rs, w, h) }
}

// Keep the wide body's register saves off the narrow AVX2 fallback path.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn wide_tiles(src: &[u8], ss: usize, rf: &[u8], rs: usize, w: usize, h: usize) -> u32 {
    let count = w * h / 64;
    let mut total = 0;
    for first in (0..count).step_by(4) {
        let mut rows = [_mm512_setzero_si512(); 8];
        for (y, row) in rows.iter_mut().enumerate() {
            // SAFETY: four full tiles remain (power-of-two block area >=256).
            let (a, b) = unsafe {
                (
                    load_tiles(src, ss, w, first, y),
                    load_tiles(rf, rs, w, first, y),
                )
            };
            *row = _mm512_sub_epi16(a, b);
        }
        // SAFETY: this function enables the same AVX-512 features.
        total += unsafe { transform_quads(rows) }.into_iter().sum::<u32>();
    }
    total
}

/// One row of four successive tiles, widened into four independent sublanes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn load_tiles(
    data: &[u8],
    stride: usize,
    width: usize,
    first: usize,
    row: usize,
) -> __m512i {
    let tiles_per_row = width / 8;
    if width >= 32 {
        let offset = (first / tiles_per_row * 8 + row) * stride + first % tiles_per_row * 8;
        // SAFETY: a four-tile group is contiguous and cannot cross a row
        // for power-of-two widths >=32; all 32 bytes belong to this group.
        return unsafe {
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(data.as_ptr().add(offset).cast()))
        };
    }
    let mut words = [0i64; 4];
    for (i, word) in words.iter_mut().enumerate() {
        let tile = first + i;
        let offset = (tile / tiles_per_row * 8 + row) * stride + tile % tiles_per_row * 8;
        // SAFETY: tile is one of the four complete 8x8 tiles in this call.
        *word = unsafe { data.as_ptr().add(offset).cast::<i64>().read_unaligned() };
    }
    _mm512_cvtepu8_epi16(_mm256_setr_epi64x(words[0], words[1], words[2], words[3]))
}

/// Pack two 16-byte rows into four 128-bit transform lanes, avoiding the
/// general narrow-tile gather. Each input view covers exactly 16x16 or more.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn quads(src: &[u8], ss: usize, rf: &[u8], rs: usize) -> [u32; 4] {
    let mut rows = [_mm512_setzero_si512(); 8];
    for (y, row) in rows.iter_mut().enumerate() {
        // SAFETY: both complete 16-byte rows lie in each checked view.
        let (a, b) = unsafe {
            let a = _mm256_set_m128i(
                _mm_loadu_si128(src.as_ptr().add((y + 8) * ss).cast()),
                _mm_loadu_si128(src.as_ptr().add(y * ss).cast()),
            );
            let b = _mm256_set_m128i(
                _mm_loadu_si128(rf.as_ptr().add((y + 8) * rs).cast()),
                _mm_loadu_si128(rf.as_ptr().add(y * rs).cast()),
            );
            (_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(b))
        };
        *row = _mm512_sub_epi16(a, b);
    }
    // SAFETY: this function enables the same AVX-512 features.
    unsafe { transform_quads(rows) }
}

/// Shared Hadamard graph; normalization remains independent per tile.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn transform_quads(mut rows: [__m512i; 8]) -> [u32; 4] {
    for half in [1, 2, 4] {
        for start in (0..8).step_by(2 * half) {
            for i in start..start + half {
                let (a, b) = (rows[i], rows[i + half]);
                rows[i] = _mm512_add_epi16(a, b);
                rows[i + half] = _mm512_sub_epi16(a, b);
            }
        }
    }
    let mut sum = _mm512_setzero_si512();
    for row in rows {
        let adjacent = _mm512_shufflehi_epi16::<0xb1>(_mm512_shufflelo_epi16::<0xb1>(row));
        let row =
            _mm512_mask_sub_epi16(_mm512_add_epi16(row, adjacent), 0xaaaa_aaaa, adjacent, row);
        let adjacent = _mm512_shufflehi_epi16::<0x4e>(_mm512_shufflelo_epi16::<0x4e>(row));
        let row =
            _mm512_mask_sub_epi16(_mm512_add_epi16(row, adjacent), 0xcccc_cccc, adjacent, row);
        let adjacent = _mm512_shuffle_epi32::<0x4e>(row);
        let row =
            _mm512_mask_sub_epi16(_mm512_add_epi16(row, adjacent), 0xf0f0_f0f0, adjacent, row);
        sum = _mm512_add_epi32(
            sum,
            _mm512_madd_epi16(_mm512_abs_epi16(row), _mm512_set1_epi16(1)),
        );
    }
    let mut partial = [0u32; 16];
    // SAFETY: partial holds 64 writable bytes.
    unsafe { _mm512_storeu_si512(partial.as_mut_ptr().cast(), sum) };
    let mut costs = [0; 4];
    for (cost, tile) in costs.iter_mut().zip(partial.as_chunks::<4>().0) {
        *cost = super::normalise_satd_8x8(tile.iter().sum());
    }
    costs
}
