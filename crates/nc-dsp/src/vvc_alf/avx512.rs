//! ICL linear ALF: thirty-two independent pixels in signed-word lanes.
//! This is the AVX2 family's paired-tap layout extended to four 128-bit
//! groups, each carrying one 4-pixel block's coefficients. Differences
//! stay in [-510,510]; products and twelve-tap accumulation widen to i32.
//! The scalar reference and AVX2 tail preserve the same H.266 arithmetic.

use super::{VVC_ALF_BLOCK, VVC_ALF_REACH, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS};
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128,
    _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi16, _mm512_add_epi32, _mm512_adds_epi16,
    _mm512_castsi128_si512, _mm512_cvtepu8_epi16, _mm512_cvtusepi16_epi8, _mm512_inserti32x4,
    _mm512_madd_epi16, _mm512_max_epi16, _mm512_packs_epi32, _mm512_set1_epi32,
    _mm512_setzero_si512, _mm512_shuffle_epi32, _mm512_sra_epi32, _mm512_sub_epi16,
    _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
};

const LANES: usize = 32;
const WINDOW: usize = LANES + 2 * VVC_ALF_REACH;
const BLOCKS: usize = LANES / VVC_ALF_BLOCK;
const PAIRS: usize = VVC_ALF_TAPS / 2;

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load(row: &[u8; WINDOW], at: usize) -> __m512i {
    let bytes: &[u8; LANES] = row[at..at + LANES].try_into().expect("sample window");
    // SAFETY: enabled ICL; exactly one checked thirty-two-byte window.
    _mm512_cvtepu8_epi16(unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) })
}

/// Four blocks' coefficients, one block in each 128-bit group. The last
/// four coefficients need only an eight-byte load, avoiding overread.
#[inline]
#[target_feature(enable = "avx512f")]
fn coefficient_groups<const LAST: bool>(
    blocks: &[[i16; VVC_ALF_TAPS]; BLOCKS],
    first: usize,
) -> __m512i {
    let at = if LAST { 8 } else { 0 };
    let load = |block: usize| -> __m128i {
        // SAFETY: enabled ISA; at=0 reads eight words, at=8 four words,
        // both entirely inside the twelve coefficients of a valid block.
        unsafe {
            if LAST {
                _mm_loadl_epi64(blocks[block][at..].as_ptr().cast())
            } else {
                _mm_loadu_si128(blocks[block][at..].as_ptr().cast())
            }
        }
    };
    let mut groups = _mm512_castsi128_si512(load(first));
    groups = _mm512_inserti32x4::<1>(groups, load(first + 2));
    groups = _mm512_inserti32x4::<2>(groups, load(first + 4));
    _mm512_inserti32x4::<3>(groups, load(first + 6))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn coefficient_pairs(blocks: &[[i16; VVC_ALF_TAPS]; BLOCKS], first: usize) -> [__m512i; PAIRS] {
    let head = coefficient_groups::<false>(blocks, first);
    let tail = coefficient_groups::<true>(blocks, first);
    [
        _mm512_shuffle_epi32::<0x00>(head),
        _mm512_shuffle_epi32::<0x55>(head),
        _mm512_shuffle_epi32::<0xaa>(head),
        _mm512_shuffle_epi32::<0xff>(head),
        _mm512_shuffle_epi32::<0x00>(tail),
        _mm512_shuffle_epi32::<0x55>(tail),
    ]
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn filter_vector(
    windows: [&[u8; WINDOW]; 7],
    low: &[__m512i; PAIRS],
    high: &[__m512i; PAIRS],
    shift: u32,
) -> __m256i {
    let centre = load(windows[3], VVC_ALF_REACH);
    let twice = _mm512_add_epi16(centre, centre);
    let mut sum_low = _mm512_set1_epi32(1 << (shift - 1));
    let mut sum_high = sum_low;
    for p in 0..PAIRS {
        let mut sums = [_mm512_setzero_si512(); 2];
        for (k, sum) in sums.iter_mut().enumerate() {
            let (dx, dy) = VVC_ALF_TAP_OFFSETS[2 * p + k];
            let a = load(windows[(3 + dy) as usize], (3 + dx) as usize);
            let b = load(windows[(3 - dy) as usize], (3 - dx) as usize);
            *sum = _mm512_sub_epi16(_mm512_add_epi16(a, b), twice);
        }
        sum_low = _mm512_add_epi32(
            sum_low,
            _mm512_madd_epi16(_mm512_unpacklo_epi16(sums[0], sums[1]), low[p]),
        );
        sum_high = _mm512_add_epi32(
            sum_high,
            _mm512_madd_epi16(_mm512_unpackhi_epi16(sums[0], sums[1]), high[p]),
        );
    }
    let count = _mm_cvtsi32_si128(shift as i32);
    let offsets = _mm512_packs_epi32(
        _mm512_sra_epi32(sum_low, count),
        _mm512_sra_epi32(sum_high, count),
    );
    // As in AVX2, signed saturation before adding the centre cannot turn
    // an out-of-range result into an in-range byte. Zero negatives before
    // the unsigned narrowing, whose upper clamp is exactly 255.
    let filtered = _mm512_adds_epi16(centre, offsets);
    _mm512_cvtusepi16_epi8(_mm512_max_epi16(filtered, _mm512_setzero_si512()))
}

/// Four rows, thirty-two columns per step; coefficient broadcasts are
/// reused across rows. Remaining columns retain the existing AVX2 kernel.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn filter_block_row(
    rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u8]; VVC_ALF_BLOCK],
) {
    let whole = out[0].len() / LANES * LANES;
    let (groups, _) = coefficients.as_chunks::<BLOCKS>();
    for (i, blocks) in groups.iter().take(whole / LANES).enumerate() {
        let low = coefficient_pairs(blocks, 0);
        let high = coefficient_pairs(blocks, 1);
        for ((rows, &shift), out) in rows.iter().zip(shifts).zip(out.iter_mut()) {
            let windows = rows.map(|row| {
                row[i * LANES..i * LANES + WINDOW]
                    .try_into()
                    .expect("window")
            });
            let filtered = filter_vector(windows, &low, &high, shift);
            let dest: &mut [u8; LANES] = (&mut out[i * LANES..(i + 1) * LANES])
                .try_into()
                .expect("output vector");
            // SAFETY: enabled ICL; exactly thirty-two checked output bytes.
            unsafe { _mm256_storeu_si256(dest.as_mut_ptr().cast(), filtered) };
        }
    }
    if whole < out[0].len() {
        let rows = rows.map(|rows| rows.map(|row| &row[whole..]));
        let mut tail = out.each_mut().map(|row| &mut row[whole..]);
        // SAFETY: AVX2 is part of ICL; all tail slices retain the checked
        // reach and matching four-pixel coefficient blocks.
        unsafe {
            super::avx2::filter_block_row(
                &rows,
                &coefficients[whole / VVC_ALF_BLOCK..],
                shifts,
                &mut tail,
            );
        };
    }
}
