//! AVX2 loop restoration kernels. The Wiener rows port dav1d 1.5.4
//! `src/x86/looprestoration_avx2.asm` (`wiener_filter7_8bpc`, BSD-2-Clause,
//! see `THIRD_PARTY.md`): the horizontal pass pairs bytes with `pshufb`
//! and multiplies them by the taps with `pmaddubsw` in 16-bit lanes, adds
//! `128 · centre` with a saturating add whose bounds are exactly the
//! intermediate clamp, and the vertical pass multiplies pair sums of rows
//! with `pmaddwd`. A complete-stripe entry prepares taps once and rotates
//! the seven row pointers, following the reference stripe schedule. The
//! horizontal and vertical arithmetic is unchanged. The self-guided rows
//! are [`super::sgr_lanes`] on eight 32-bit lanes; their factor uses
//! dav1d's reciprocal form with exact rounding.
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
use super::self_guided::{SGR_BITS, Z_MAX};
use super::sgr_lanes::{SgrLanes, sgr_kernels};
use super::wiener::{ROUND1, SATURATION_BIAS, SATURATION_OFFSET, WienerTaps};
use crate::simd::packed_pair;

/// This ISA's table.
pub(super) const KERNELS: Kernels = Kernels {
    wiener_stripe: Some(wiener_stripe),
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

/// Tap and shuffle setup shared by every row in one stripe.
#[derive(Clone, Copy)]
struct Horizontal {
    outer: __m256i,
    left: __m256i,
    right: __m256i,
    centre: __m256i,
    outer_taps: __m256i,
    inner_taps: __m256i,
    centre_tap: __m256i,
    offset: __m256i,
    bias: __m256i,
}
impl Horizontal {
    #[inline(always)]
    unsafe fn new([c0, c1, c2, c3]: WienerTaps) -> Self {
        // SAFETY: callers enable AVX2.
        unsafe {
            Self {
                outer: pattern(&OUTER),
                left: pattern(&INNER_LEFT),
                right: pattern(&INNER_RIGHT),
                centre: pattern(&CENTRE),
                outer_taps: byte_pair(c0, c0),
                inner_taps: byte_pair(c1, c2),
                centre_tap: _mm256_set1_epi16(c3),
                offset: _mm256_set1_epi16(SATURATION_OFFSET),
                bias: _mm256_set1_epi16(SATURATION_BIAS),
            }
        }
    }
    /// Write sixteen samples per block with the original arithmetic.
    #[inline(always)]
    unsafe fn row(self, src: *const u8, out: *mut i16, blocks: usize) {
        // SAFETY: caller enables AVX2 and checks blocks*16+8 readable bytes
        // and blocks*16 writable i16 values before calling.
        unsafe {
            for block in 0..blocks {
                let j = block * STEP;
                let at = src.add(j);
                let x = _mm256_loadu2_m128i(at.add(8).cast(), at.cast());
                let p3 = _mm256_shuffle_epi8(x, self.centre);
                let sum = _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_maddubs_epi16(_mm256_shuffle_epi8(x, self.outer), self.outer_taps),
                        _mm256_maddubs_epi16(_mm256_shuffle_epi8(x, self.left), self.inner_taps),
                    ),
                    _mm256_add_epi16(
                        _mm256_maddubs_epi16(_mm256_shuffle_epi8(x, self.right), self.inner_taps),
                        _mm256_mullo_epi16(p3, self.centre_tap),
                    ),
                );
                let saturated = _mm256_adds_epi16(
                    sum,
                    _mm256_add_epi16(_mm256_slli_epi16::<7>(p3), self.offset),
                );
                let result = _mm256_add_epi16(_mm256_srai_epi16::<3>(saturated), self.bias);
                _mm256_storeu_si256(out.add(j).cast(), result);
            }
        }
    }
}
#[target_feature(enable = "avx2")]
unsafe fn wiener_h(src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize) {
    let blocks = n.div_ceil(STEP);
    assert!(src.len() >= blocks * STEP + STEP_SOURCE - STEP && out.len() >= blocks * STEP);
    // SAFETY: AVX2 enabled and both row extents checked above.
    unsafe { Horizontal::new(taps).row(src.as_ptr(), out.as_mut_ptr(), blocks) };
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

#[inline(always)]
unsafe fn vertical_taps([c0, c1, c2, c3]: WienerTaps) -> (__m256i, __m256i) {
    // SAFETY: callers enable AVX2.
    unsafe {
        (
            _mm256_set1_epi32(packed_pair(c0, c1)),
            _mm256_set1_epi32(packed_pair(c2, c3)),
        )
    }
}
/// Original vertical arithmetic after whole-stripe extent validation.
#[inline(always)]
unsafe fn vertical_row(
    rows: &[*const i16; WIENER_ROWS],
    taps: (__m256i, __m256i),
    dst: *mut u8,
    n: usize,
) {
    // SAFETY: caller enables AVX2, each row holds ceil(n/32)*32 readable i16s,
    // and dst holds n writable bytes. Partial output uses an exact copy.
    unsafe {
        for block in 0..n.div_ceil(V_STEP) {
            let j = block * V_STEP;
            let first = vertical16(rows, j, taps);
            let second = vertical16(rows, j + STEP, taps);
            let bytes = _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_packus_epi16(first, second));
            if j + V_STEP <= n {
                _mm256_storeu_si256(dst.add(j).cast(), bytes);
            } else {
                let mut tail = [0u8; V_STEP];
                _mm256_storeu_si256(tail.as_mut_ptr().cast(), bytes);
                core::ptr::copy_nonoverlapping(tail.as_ptr(), dst.add(j), n - j);
            }
        }
    }
}
#[target_feature(enable = "avx2")]
unsafe fn wiener_v(rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u8]) {
    let blocks = dst.len().div_ceil(V_STEP);
    assert!(rows.iter().all(|row| row.len() >= blocks * V_STEP));
    let pointers = rows.map(<[i16]>::as_ptr);
    // SAFETY: AVX2 enabled, input extents checked, dst is its exact slice.
    unsafe { vertical_row(&pointers, vertical_taps(taps), dst.as_mut_ptr(), dst.len()) };
}
/// Prepare once and rotate pointers as dav1d's Wiener stripe does. The
/// original arithmetic remains; per-row indirect calls and modulo go away.
#[target_feature(enable = "avx2")]
unsafe fn wiener_stripe(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    (window, stride): (&[u8], usize),
    (w, h): (usize, usize),
    ring: &mut [i16],
    (dst, x, y): (&mut nc_common::PlaneViewMut<'_>, usize, usize),
) {
    use super::{AV1_LR_BORDER, AV1_LR_OVERREAD};
    assert!(x <= dst.width() && w <= dst.width() - x);
    assert!(y <= dst.height() && h <= dst.height() - y);
    if w == 0 || h == 0 {
        return;
    }
    let row_len = w.checked_add(AV1_LR_OVERREAD).expect("Wiener row extent");
    let source_len = row_len
        .checked_add(2 * AV1_LR_BORDER)
        .expect("Wiener source row");
    let source_rows = h
        .checked_add(2 * AV1_LR_BORDER)
        .expect("Wiener source rows");
    let source_end = (source_rows - 1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(source_len))
        .expect("Wiener source extent");
    assert!(window.len() >= source_end);
    assert!(
        ring.len()
            >= WIENER_ROWS
                .checked_mul(row_len)
                .expect("Wiener ring extent")
    );
    let dst_stride = dst.stride();
    // SAFETY: checked spans cover every window/ring row and output rectangle.
    // Horizontal writes ceil(w/16)*16 and vertical reads ceil(w/32)*32,
    // both within the overread. The seven scratch rows are distinct and
    // horizontal completes before vertical reads; pointers rotate among them.
    unsafe {
        let horizontal = Horizontal::new(horizontal);
        let vertical = vertical_taps(vertical);
        let blocks = w.div_ceil(STEP);
        let start = ring.as_mut_ptr();
        let mut rows: [*const i16; WIENER_ROWS] =
            core::array::from_fn(|k| start.add(k * row_len).cast_const());
        for (k, &row) in rows[..WIENER_ROWS - 1].iter().enumerate() {
            horizontal.row(window.as_ptr().add(k * stride), row.cast_mut(), blocks);
        }
        let output = dst.data_mut().as_mut_ptr().add(y * dst_stride + x);
        for r in 0..h {
            horizontal.row(
                window.as_ptr().add((r + WIENER_ROWS - 1) * stride),
                rows[WIENER_ROWS - 1].cast_mut(),
                blocks,
            );
            vertical_row(&rows, vertical, output.add(r * dst_stride), w);
            let [a, b, c, d, e, f, g] = rows;
            rows = [b, c, d, e, f, g, a];
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
    /// For n=z+1 in 2..=255, 256/n is at least 1/(2n) away from a
    /// half integer: 512 cannot equal an odd multiple of n in that range.
    /// Intel's RCPPS relative-error bound (1.5 * 2^-12) gives absolute error
    /// at most (3/32)/n, well below that distance. Multiplication by 256 is
    /// exact; adding 0.5 has error at most 2^-15, still below the margin.
    /// Truncation therefore rounds to the exact integer independently of
    /// MXCSR's conversion rounding mode. The endpoints are explicit.
    #[inline(always)]
    fn sgr_factor(self) -> Self {
        use core::arch::x86_64::{
            _mm256_add_ps, _mm256_andnot_si256, _mm256_mul_ps, _mm256_rcp_ps, _mm256_set1_ps,
        };
        // SAFETY: see the impl comment; z is clipped to 0..=255 by box_ab.
        unsafe {
            let unit = _mm256_set1_epi32(1 << SGR_BITS);
            let den = _mm256_cvtepi32_ps(_mm256_add_epi32(self.0, _mm256_set1_epi32(1)));
            let rounded = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_rcp_ps(den),
                    _mm256_set1_ps(f32::from(1_u16 << SGR_BITS)),
                ),
                _mm256_set1_ps(0.5),
            ));
            let max = _mm256_set1_epi32(Z_MAX);
            let complement = _mm256_andnot_si256(
                _mm256_cmpeq_epi32(self.0, max),
                _mm256_min_epi32(rounded, max),
            );
            Self(_mm256_sub_epi32(unit, complement))
        }
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

#[cfg(test)]
mod tests {
    use super::{Lanes, SgrLanes, Z_MAX};
    use crate::av1_restoration::self_guided::x_by_xplus1;

    #[test]
    fn reciprocal_factor_matches_entire_legal_domain() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // SAFETY: AVX2 was detected above.
        unsafe {
            check_reciprocal_factors();
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn check_reciprocal_factors() {
        for first in (0..=Z_MAX).step_by(8) {
            let z: [i32; 8] = core::array::from_fn(|i| first + i as i32);
            let mut got = [0; 8];
            Lanes::load(&z).sgr_factor().store(&mut got);
            for (&z, &factor) in z.iter().zip(&got) {
                assert_eq!(factor as u32, x_by_xplus1(z as u32), "z={z}");
            }
        }
    }
}
