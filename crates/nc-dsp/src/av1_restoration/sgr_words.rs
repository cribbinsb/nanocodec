//! High-depth horizontal SGR sums over the same word pairs as dav1d's
//! looprestoration16 kernels. Five samples sum to at most 20475, so
//! only the square accumulation needs 32-bit lanes.

#![allow(unsafe_code, clippy::inline_always)]

use crate::simd::I16Lanes;

pub(super) trait SquareWords: I16Lanes {
    #[inline(always)]
    fn add_pair_squares(acc: Self::Acc, a: Self, b: Self) -> Self::Acc {
        Self::acc_vec_fma2(acc, a, a, b, b)
    }

    fn store_squares(acc: Self::Acc, dst: &mut [i32]);
}

#[cfg(target_arch = "x86_64")]
impl SquareWords for crate::simd::avx2::I16x16 {
    #[inline(always)]
    fn store_squares([a, b]: Self::Acc, dst: &mut [i32]) {
        use core::arch::x86_64::{_mm256_permute2x128_si256, _mm256_storeu_si256};
        let dst = &mut dst[..16];
        // SAFETY: AVX2 is enabled and both checked eight-dword stores fit.
        unsafe {
            _mm256_storeu_si256(
                dst.as_mut_ptr().cast(),
                _mm256_permute2x128_si256::<0x20>(a, b),
            );
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(8).cast(),
                _mm256_permute2x128_si256::<0x31>(a, b),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl SquareWords for crate::simd::avx512_i16::I16x32 {
    #[inline(always)]
    fn add_pair_squares(acc: Self::Acc, a: Self, b: Self) -> Self::Acc {
        use core::arch::x86_64::{
            _mm512_dpwssd_epi32, _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
        };
        // SAFETY: the Ice Lake caller enables AVX-512 VNNI. The five
        // 12-bit squares are well within i32.
        unsafe {
            let lo = _mm512_unpacklo_epi16(a.0, b.0);
            let hi = _mm512_unpackhi_epi16(a.0, b.0);
            [
                _mm512_dpwssd_epi32(acc[0], lo, lo),
                _mm512_dpwssd_epi32(acc[1], hi, hi),
            ]
        }
    }

    #[inline(always)]
    fn store_squares([a, b]: Self::Acc, dst: &mut [i32]) {
        use core::arch::x86_64::{
            _mm512_permutex2var_epi64, _mm512_setr_epi64, _mm512_storeu_si512,
        };
        let dst = &mut dst[..32];
        // SAFETY: the Ice Lake caller enables AVX-512F. The two checked
        // sixteen-dword stores are in pixel order.
        unsafe {
            let lo = _mm512_permutex2var_epi64(a, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), b);
            let hi = _mm512_permutex2var_epi64(a, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15), b);
            _mm512_storeu_si512(dst.as_mut_ptr().cast(), lo);
            _mm512_storeu_si512(dst.as_mut_ptr().add(16).cast(), hi);
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl SquareWords for crate::simd::neon::I16x8 {
    #[inline(always)]
    fn store_squares([a, b]: Self::Acc, dst: &mut [i32]) {
        use core::arch::aarch64::vst1q_s32;
        let dst = &mut dst[..8];
        // SAFETY: the NEON caller checked both four-lane stores.
        unsafe {
            vst1q_s32(dst.as_mut_ptr(), a);
            vst1q_s32(dst.as_mut_ptr().add(4), b);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl SquareWords for crate::simd::simd128::I16x8 {
    #[inline(always)]
    fn store_squares([a, b]: Self::Acc, dst: &mut [i32]) {
        use core::arch::wasm32::v128_store;
        let dst = &mut dst[..8];
        // SAFETY: SIMD128 is enabled and each store writes four checked lanes.
        unsafe {
            v128_store(dst.as_mut_ptr().cast(), a);
            v128_store(dst.as_mut_ptr().add(4).cast(), b);
        }
    }
}

#[inline(always)]
unsafe fn load<V: SquareWords>(ptr: *const u16) -> V {
    // SAFETY: the caller checked V::LANES readable samples, all at most 4095.
    V::load(unsafe { core::slice::from_raw_parts(ptr.cast::<i16>(), V::LANES) })
}

/// A complete row of horizontal sums plus vector padding.
///
/// # Safety
/// The caller has selected the ISA of V.
#[inline(always)]
pub(super) unsafe fn box_sums<V: SquareWords, const R: usize>(
    src: &[u16],
    sum: &mut [i16],
    squares: &mut [i32],
    n: usize,
) {
    let padded = n.div_ceil(V::LANES) * V::LANES;
    assert!(src.len() >= padded + 2 * R);
    assert!(sum.len() >= padded && squares.len() >= padded);
    let zero = V::splat(0);
    // SAFETY: input padding covers every full vector of taps, and the
    // output rows include padding for each final vector store.
    unsafe {
        for col in (0..padded).step_by(V::LANES) {
            let start = src.as_ptr().add(col);
            let first = load::<V>(start);
            let second = load::<V>(start.add(1));
            let third = load::<V>(start.add(2));
            let sum3 = first.add(second).add(third);
            let acc = V::acc_vec_mul_add2(first, first, second, second);
            let (samples, squared) = if R == 2 {
                let fourth = load::<V>(start.add(3));
                let fifth = load::<V>(start.add(4));
                let square4 = V::add_pair_squares(acc, third, fourth);
                (
                    sum3.add(fourth).add(fifth),
                    V::add_pair_squares(square4, fifth, zero),
                )
            } else {
                (sum3, V::add_pair_squares(acc, third, zero))
            };
            samples.store(&mut sum[col..col + V::LANES]);
            V::store_squares(squared, &mut squares[col..col + V::LANES]);
        }
    }
}
