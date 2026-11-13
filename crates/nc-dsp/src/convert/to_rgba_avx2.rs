//! AVX2 I420 to RGBA (x86-64).
//!
//! The same exact 16-bit split form of the BT.601 inverse as the SIMD128
//! kernel ([`bt601_split`]), 32 pixels of each row per iteration as two
//! 16-lane halves. The 16 chroma samples of a row pair are widened once and
//! reordered by one 64-bit permute, so that the in-lane `vpunpck{l,h}wd`
//! that repeat each term for its pixel pair yield pixels 0-15 and 16-31 in
//! order. Each half packs `[R, B]` and `[G, alpha]` with saturation (the
//! scalar clamp), interleaves bytes then 16-bit pairs into RGBA, and a
//! 128-bit permute puts the four-pixel groups back in order. The scalar row
//! function takes the tail.
//!
//! The row-function structure follows libyuv's `I422ToARGBRow_AVX2` (BSD
//! 3-clause, Copyright 2011 The `LibYuv` Project Authors; see
//! `THIRD_PARTY.md`); the arithmetic is our exact 8.8 form, not libyuv's
//! 6-bit one.
//!
//! Kernel contract: see the `convert` module docs.

use core::arch::x86_64::{
    __m256i, _mm_loadu_si128, _mm256_add_epi16, _mm256_cvtepu8_epi16, _mm256_mullo_epi16,
    _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi16,
    _mm256_slli_epi16, _mm256_srai_epi16, _mm256_storeu_si256, _mm256_sub_epi16,
    _mm256_unpackhi_epi8, _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
};

use super::{ALPHA_OPAQUE, bt601, bt601_split, scalar};

/// Pixels of each row converted per iteration.
const STEP: usize = 32;

/// Chroma terms for 32 pixels of a row (16 chroma samples, each repeated
/// for its pixel pair), as pixels 0-15 and 16-31.
struct ChromaTerms {
    /// Remainders of R, G and B without the luma term, `BIAS` included
    /// (`bt601_split`).
    remainder: [[__m256i; 2]; 3],
    /// `Cr'` (added to R, subtracted from G).
    cr: [__m256i; 2],
    /// `2 Cb'` (added to B).
    cb2: [__m256i; 2],
}

/// # Safety
///
/// Kernel contract (`convert` module docs); the CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn i420_to_rgba_rows(
    luma: [&[u8]; 2],
    cb: &[u8],
    cr: &[u8],
    rgba: [&mut [u8]; 2],
) {
    let [top, bottom] = rgba;
    // SAFETY: forwarded kernel contract; a lone row has an empty partner.
    unsafe {
        if luma[1].is_empty() {
            rows([luma[0]], cb, cr, [top]);
        } else {
            rows(luma, cb, cr, [top, bottom]);
        }
    }
}

/// `N` rows sharing one chroma row.
///
/// # Safety
///
/// Kernel contract (`convert` module docs); the CPU must support AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn rows<const N: usize>(luma: [&[u8]; N], cb: &[u8], cr: &[u8], mut rgba: [&mut [u8]; N]) {
    let w = luma[0].len();
    let simd_width = w - w % STEP;
    for x in (0..simd_width).step_by(STEP) {
        // SAFETY: x + 32 <= w, so chroma bytes x / 2 .. x / 2 + 16 lie inside
        // w.div_ceil(2), luma bytes x .. x + 32 inside each w-sample row and
        // RGBA bytes 4x .. 4x + 128 inside each 4w-byte row (kernel contract).
        unsafe {
            let terms = chroma_terms(
                _mm_loadu_si128(cb.as_ptr().add(x / 2).cast()),
                _mm_loadu_si128(cr.as_ptr().add(x / 2).cast()),
            );
            for (row, out) in luma.iter().zip(rgba.iter_mut()) {
                for half in 0..2 {
                    let samples = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                        row.as_ptr().add(x + 16 * half).cast(),
                    ));
                    store_rgba(
                        out.as_mut_ptr().add(4 * (x + 16 * half)),
                        rgb16(samples, &terms, half),
                    );
                }
            }
        }
    }
    let chroma = simd_width / 2;
    for (row, out) in luma.into_iter().zip(rgba) {
        scalar::i420_to_rgba_row(
            &row[simd_width..],
            &cb[chroma..],
            &cr[chroma..],
            &mut out[4 * simd_width..],
        );
    }
}

/// The chroma terms of 16 Cb and Cr samples for their 32 pixels. `BIAS` is
/// folded into the remainders here, once per 2x2 group.
#[inline]
#[target_feature(enable = "avx2")]
fn chroma_terms(cb: core::arch::x86_64::__m128i, cr: core::arch::x86_64::__m128i) -> ChromaTerms {
    let offset = _mm256_set1_epi16(bt601::CHROMA_OFFSET as i16);
    // Quadwords [c0-3, c8-11, c4-7, c12-15]: the in-lane pair-ups below then
    // give c0-7 (pixels 0-15) and c8-15 (pixels 16-31).
    let widen = |c| {
        _mm256_sub_epi16(
            _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_cvtepu8_epi16(c)),
            offset,
        )
    };
    let (u, v) = (widen(cb), widen(cr));
    let bias = _mm256_set1_epi16(bt601_split::BIAS);
    let red = _mm256_add_epi16(
        _mm256_mullo_epi16(v, _mm256_set1_epi16(bt601_split::R_CR)),
        bias,
    );
    let green = _mm256_add_epi16(
        _mm256_add_epi16(
            _mm256_mullo_epi16(u, _mm256_set1_epi16(bt601_split::G_CB)),
            _mm256_mullo_epi16(v, _mm256_set1_epi16(bt601_split::G_CR)),
        ),
        bias,
    );
    let blue = _mm256_add_epi16(
        _mm256_slli_epi16::<{ bt601_split::B_CB_SHIFT as i32 }>(u),
        bias,
    );
    let cb2 = _mm256_add_epi16(u, u);
    ChromaTerms {
        remainder: [pair_up(red), pair_up(green), pair_up(blue)],
        cr: pair_up(v),
        cb2: pair_up(cb2),
    }
}

/// Each chroma lane repeated for its two pixels (input in the quadword
/// order of `chroma_terms`): pixels 0-15, then 16-31.
#[inline]
#[target_feature(enable = "avx2")]
fn pair_up(c: __m256i) -> [__m256i; 2] {
    [_mm256_unpacklo_epi16(c, c), _mm256_unpackhi_epi16(c, c)]
}

/// R, G and B of 16 pixels in 16-bit lanes (unclamped) from their luma and
/// half `half` (0: pixels 0-15, 1: 16-31) of the chroma terms.
#[inline]
#[target_feature(enable = "avx2")]
fn rgb16(luma: __m256i, terms: &ChromaTerms, half: usize) -> [__m256i; 3] {
    let scaled = _mm256_mullo_epi16(luma, _mm256_set1_epi16(bt601_split::Y));
    let linear = [
        _mm256_add_epi16(luma, terms.cr[half]),
        _mm256_sub_epi16(luma, terms.cr[half]),
        _mm256_add_epi16(luma, terms.cb2[half]),
    ];
    [0, 1, 2].map(|c| {
        let remainder = _mm256_add_epi16(scaled, terms.remainder[c][half]);
        _mm256_add_epi16(
            linear[c],
            _mm256_srai_epi16::<{ bt601::SHIFT as i32 }>(remainder),
        )
    })
}

/// Clamp 16 pixels' channels to bytes and store them as 64 RGBA bytes.
///
/// # Safety
///
/// `out` must be valid for 64 bytes of writes; the CPU must support AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_rgba(out: *mut u8, [red, green, blue]: [__m256i; 3]) {
    let alpha = _mm256_set1_epi16(i16::from(ALPHA_OPAQUE));
    // Per 128-bit lane (pixels 0-7, then 8-15): [R0-7, B0-7] and [G0-7, A].
    let red_blue = _mm256_packus_epi16(red, blue);
    let green_alpha = _mm256_packus_epi16(green, alpha);
    let red_green = _mm256_unpacklo_epi8(red_blue, green_alpha);
    let blue_alpha = _mm256_unpackhi_epi8(red_blue, green_alpha);
    // Pixels [0-3 | 8-11] and [4-7 | 12-15].
    let first = _mm256_unpacklo_epi16(red_green, blue_alpha);
    let second = _mm256_unpackhi_epi16(red_green, blue_alpha);
    // SAFETY: two 32-byte stores inside the 64 bytes (caller contract).
    unsafe {
        _mm256_storeu_si256(out.cast(), _mm256_permute2x128_si256::<0x20>(first, second));
        _mm256_storeu_si256(
            out.add(32).cast(),
            _mm256_permute2x128_si256::<0x31>(first, second),
        );
    }
}
