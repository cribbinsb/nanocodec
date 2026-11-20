//! Byte-pair inputs to the shared relaxed-SIMD sub-pel filters.
//! The byte operand is signed. Tap magnitudes are below 128, so either
//! interpretation of the second operand produces the same value. The pair
//! magnitude is at most 2 × 128 × 127, within signed 16 bits.

use core::arch::wasm32::{
    i8x16_relaxed_swizzle, i8x16_shuffle, i16x8_relaxed_dot_i8x16_i7x16, i16x8_splat, u8x16, v128,
    v128_load32_lane, v128_load64_lane, v128_load64_zero, v128_xor,
};

/// Both magnitudes in each two-byte lane.
#[inline]
pub(crate) fn magnitudes(lo: i16, hi: i16) -> v128 {
    debug_assert!(lo.unsigned_abs() < 128 && hi.unsigned_abs() < 128);
    i16x8_splat((lo.unsigned_abs() | hi.unsigned_abs() << 8) as i16)
}

/// SIMD128's two overlapping loads place source bytes 0..8 in the low lane
/// and bytes T-1..T+7 in the high lane. Build the mask in that layout.
pub(crate) const fn indices<const T: usize>(lo: usize, hi: usize) -> [u8; 16] {
    let mut out = [0; 16];
    let mut x = 0;
    while x < 8 {
        out[2 * x] = row_lane::<T>(x + lo);
        out[2 * x + 1] = row_lane::<T>(x + hi);
        x += 1;
    }
    out
}

const fn row_lane<const T: usize>(source: usize) -> u8 {
    (if source < 8 { source } else { source + 9 - T }) as u8
}

#[inline]
pub(crate) fn mask(b: [u8; 16]) -> v128 {
    u8x16(
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
        b[14], b[15],
    )
}

/// Load the T-tap window for eight outputs, or for four outputs with
/// eight taps. The latter only uses the first four pairs after its shuffle.
///
/// # Safety
/// The caller provides N + T - 1 readable bytes at src.
#[inline]
pub(crate) unsafe fn load_row<const N: usize, const T: usize>(src: *const u8) -> v128 {
    debug_assert!((N >= 8 && (T == 4 || T == 6 || T == 8)) || (N == 4 && T == 8));
    // SAFETY: the second load ends at byte N + T - 1, inside the window.
    unsafe {
        let low = v128_load64_zero(src.cast());
        if N == 4 {
            v128_load32_lane::<2>(low, src.add(T - 1).cast())
        } else {
            v128_load64_lane::<1>(low, src.add(T - 1).cast())
        }
    }
}

#[inline]
pub(crate) fn interleave(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
}

#[inline]
pub(crate) fn dot(pairs: v128, mags: v128) -> v128 {
    i16x8_relaxed_dot_i8x16_i7x16(pairs, mags)
}

/// Gather both samples of each output and apply the pixel sign bias. A
/// filter with one bias for the entire row XORs at load time and uses `flip = 0`.
#[inline]
pub(crate) fn horizontal(pixels: v128, indices: v128, flip: v128, mags: v128) -> v128 {
    dot(v128_xor(i8x16_relaxed_swizzle(pixels, indices), flip), mags)
}
