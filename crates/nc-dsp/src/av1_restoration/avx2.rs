//! AVX2 loop restoration kernels. The Wiener rows port dav1d 1.5.4
//! `src/x86/looprestoration_avx2.asm` (`wiener_filter7_8bpc`, BSD-2-Clause,
//! see `THIRD_PARTY.md`): the horizontal pass pairs bytes with `pshufb`
//! and multiplies them by the taps with `pmaddubsw` in 16-bit lanes, adds
//! `128 · centre` with a saturating add whose bounds are exactly the
//! intermediate clamp, and the vertical pass multiplies pair sums of rows
//! with `pmaddwd`. The self-guided rows are [`super::sgr_lanes`] on eight
//! 32-bit lanes.
//!
//! Why the 16-bit horizontal sum is exact: with the centre tap less 128,
//! the sum is `Σ c_k · (p_k + p_{6-k} − 2 · p_3)` over the three outer
//! taps, at most `510 · 64 = 32640` in magnitude for every legal tap set
//! (§5.11.57 ranges: the positive taps sum to at most 64, the negative
//! ones to at most 45), so wrapping adds are exact; each `pmaddubsw` pair
//! is at most `255 · 69`.

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32, _mm_packus_epi16,
    _mm_storel_epi64, _mm256_add_epi16, _mm256_add_epi32, _mm256_adds_epi16,
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi32, _mm256_cvtepi32_ps,
    _mm256_cvtepu8_epi32, _mm256_cvttps_epi32, _mm256_div_ps, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_loadu2_m128i, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_max_epi32, _mm256_min_epi32, _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_slli_epi32, _mm256_srai_epi16,
    _mm256_srai_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_sub_epi32,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

use super::kernels::{Kernels, WIENER_ROWS};
use super::sgr_lanes::{SgrLanes, sgr_kernels};
use super::wiener::{ROUND1, SATURATION_BIAS, SATURATION_OFFSET, WienerTaps};
use crate::simd::packed_pair;

/// This ISA's table.
pub(super) const KERNELS: Kernels = Kernels {
    wiener_h,
    wiener_v,
    box_sums3,
    box_sums5,
    box_ab3,
    box_ab5,
    between,
    on,
    around,
    blend,
};

/// Outputs per Wiener step.
const STEP: usize = 16;
/// Source bytes the step's two 128-bit loads cover.
const STEP_SOURCE: usize = STEP + 8;

/// Byte pairs, per output `o` of a lane's eight: (`o`, `o + 6`) for the
/// outer tap, (`o + 1`, `o + 2`) and (`o + 5`, `o + 4`) for the second and
/// third taps, and the centre `o + 3` zero-extended.
const OUTER: [u8; 16] = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13];
const INNER_LEFT: [u8; 16] = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9];
const INNER_RIGHT: [u8; 16] = [5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11];
/// `pshufb` index that zeroes its byte.
const ZERO: u8 = 0x80;
const CENTRE: [u8; 16] = [
    3, ZERO, 4, ZERO, 5, ZERO, 6, ZERO, 7, ZERO, 8, ZERO, 9, ZERO, 10, ZERO,
];

/// A byte shuffle pattern in both 128-bit lanes.
#[inline(always)]
fn pattern(bytes: &[u8; 16]) -> __m256i {
    // SAFETY: AVX2 kernels only (module docs); the load reads the array.
    unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(bytes.as_ptr().cast())) }
}

/// Two taps as the byte pairs of `pmaddubsw`.
#[inline(always)]
fn byte_pair(first: i16, second: i16) -> __m256i {
    let pair = i16::from_le_bytes([first as i8 as u8, second as i8 as u8]);
    // SAFETY: AVX2 kernels only (module docs).
    unsafe { _mm256_set1_epi16(pair) }
}

#[target_feature(enable = "avx2")]
unsafe fn wiener_h(src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize) {
    let blocks = n.div_ceil(STEP);
    // Every load and store below stays within these bounds.
    assert!(src.len() >= blocks * STEP + STEP_SOURCE - STEP && out.len() >= blocks * STEP);
    let (outer, left, right, centre) = (
        pattern(&OUTER),
        pattern(&INNER_LEFT),
        pattern(&INNER_RIGHT),
        pattern(&CENTRE),
    );
    let [c0, c1, c2, c3] = taps;
    let (outer_taps, inner_taps) = (byte_pair(c0, c0), byte_pair(c1, c2));
    let centre_tap = _mm256_set1_epi16(c3);
    let offset = _mm256_set1_epi16(SATURATION_OFFSET);
    let bias = _mm256_set1_epi16(SATURATION_BIAS);
    for block in 0..blocks {
        let j = block * STEP;
        // SAFETY: AVX2 is enabled; bytes j .. j + 24 lie in `src` (checked
        // above).
        let x = unsafe {
            let at = src.as_ptr().add(j);
            _mm256_loadu2_m128i(at.add(8).cast(), at.cast())
        };
        let p3 = _mm256_shuffle_epi8(x, centre);
        let sum = _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_shuffle_epi8(x, outer), outer_taps),
                _mm256_maddubs_epi16(_mm256_shuffle_epi8(x, left), inner_taps),
            ),
            _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_shuffle_epi8(x, right), inner_taps),
                _mm256_mullo_epi16(p3, centre_tap),
            ),
        );
        let saturated =
            _mm256_adds_epi16(sum, _mm256_add_epi16(_mm256_slli_epi16::<7>(p3), offset));
        let result = _mm256_add_epi16(_mm256_srai_epi16::<3>(saturated), bias);
        // SAFETY: AVX2 is enabled; values j .. j + 16 lie in `out`.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().add(j).cast(), result) };
    }
}

/// Outputs per vertical Wiener step: two 16-lane halves.
const V_STEP: usize = 2 * STEP;

/// Sixteen vertical Wiener outputs at `rows[k] + j`, as 16-bit lanes
/// (`clamp(Σ >> 11)` before the final byte saturation).
///
/// # Safety
///
/// AVX2 must be available and every `rows[k]` readable for 16 values at
/// `j`.
#[inline(always)]
unsafe fn vertical16(
    rows: &[*const i16; WIENER_ROWS],
    j: usize,
    (outer, inner): (__m256i, __m256i),
) -> __m256i {
    // SAFETY: the caller's contract.
    unsafe {
        let row = |k: usize| _mm256_loadu_si256(rows[k].add(j).cast());
        let a = _mm256_add_epi16(row(0), row(6));
        let b = _mm256_add_epi16(row(1), row(5));
        let c = _mm256_add_epi16(row(2), row(4));
        let d = row(3);
        let low = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), outer),
            _mm256_madd_epi16(_mm256_unpacklo_epi16(c, d), inner),
        );
        let high = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), outer),
            _mm256_madd_epi16(_mm256_unpackhi_epi16(c, d), inner),
        );
        _mm256_packs_epi32(
            _mm256_srai_epi32::<{ ROUND1 as i32 }>(low),
            _mm256_srai_epi32::<{ ROUND1 as i32 }>(high),
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn wiener_v(rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u8]) {
    let n = dst.len();
    let blocks = n.div_ceil(V_STEP);
    // Every load below stays within these bounds.
    assert!(rows.iter().all(|row| row.len() >= blocks * V_STEP));
    let pointers = rows.map(<[i16]>::as_ptr);
    let [c0, c1, c2, c3] = taps;
    let taps = (
        _mm256_set1_epi32(packed_pair(c0, c1)),
        _mm256_set1_epi32(packed_pair(c2, c3)),
    );
    for block in 0..blocks {
        let j = block * V_STEP;
        // SAFETY: AVX2 is enabled; values j .. j + 32 of every row lie in
        // it (checked above).
        let (first, second) = unsafe {
            (
                vertical16(&pointers, j, taps),
                vertical16(&pointers, j + STEP, taps),
            )
        };
        // packus interleaves the halves by 128-bit lane; the permute puts
        // the 32 bytes back in order.
        let bytes = _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_packus_epi16(first, second));
        if j + V_STEP <= n {
            // SAFETY: AVX2 is enabled; bytes j .. j + 32 lie in `dst`.
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(j).cast(), bytes) };
        } else {
            let mut tail = [0u8; V_STEP];
            // SAFETY: AVX2 is enabled; the store covers `tail`.
            unsafe { _mm256_storeu_si256(tail.as_mut_ptr().cast(), bytes) };
            dst[j..].copy_from_slice(&tail[..n - j]);
        }
    }
}

/// Eight `i32` lanes for the self-guided kernels.
#[derive(Clone, Copy)]
struct Lanes(__m256i);

// SAFETY (every method): these values exist only inside the AVX2 kernels
// of this module (module docs); loads and stores cover checked subslices.
impl SgrLanes for Lanes {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_set1_epi32(value) })
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let src = &src[..8];
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_loadu_si256(src.as_ptr().cast()) })
    }
    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        let src = &src[..8];
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(src.as_ptr().cast())) })
    }
    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        let dst = &mut dst[..8];
        // SAFETY: see the impl comment.
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), self.0) }
    }
    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        let dst = &mut dst[..8];
        // SAFETY: see the impl comment.
        unsafe {
            let words: __m128i = _mm_packs_epi32(
                _mm256_castsi256_si128(self.0),
                _mm256_extracti128_si256::<1>(self.0),
            );
            _mm_storel_epi64(dst.as_mut_ptr().cast(), _mm_packus_epi16(words, words));
        }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_add_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_sub_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_mullo_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn shl<const BITS: i32>(self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_slli_epi32::<BITS>(self.0) })
    }
    #[inline(always)]
    fn shr<const BITS: i32>(self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_srli_epi32::<BITS>(self.0) })
    }
    #[inline(always)]
    fn sra<const BITS: i32>(self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_srai_epi32::<BITS>(self.0) })
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_min_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_max_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn div_floor(self, den: Self) -> Self {
        // SAFETY: see the impl comment.
        Self(unsafe {
            _mm256_cvttps_epi32(_mm256_div_ps(
                _mm256_cvtepi32_ps(self.0),
                _mm256_cvtepi32_ps(den.0),
            ))
        })
    }
}

sgr_kernels!(#[target_feature(enable = "avx2")]; Lanes);
