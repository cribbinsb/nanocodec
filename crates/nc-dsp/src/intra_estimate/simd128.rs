//! SIMD128 intra estimate statistics: 16 bytes of a row per vector, the
//! even samples as the low bytes of its 16-bit lanes (`v128.and`), every
//! square by `i32x4.dot_i16x8_s` of the 16-bit differences. This module
//! only exists in builds with SIMD128 enabled.

use core::arch::wasm32::{
    i16x8_splat, i16x8_sub, i32x4_add, i32x4_dot_i16x8, i32x4_extract_lane, i32x4_splat, v128,
    v128_and, v128_load, v128_load32_zero, v128_load64_zero,
};

use super::{EstimateStats, SAMPLE_STEP};

/// Bytes of a row per vector.
const CHUNK: usize = 16;

/// The even samples of `bytes` (`n` of them, 4 to 16; the rest zero) as
/// 16-bit lanes.
#[inline]
fn even(bytes: &[u8], n: usize) -> v128 {
    // SAFETY: every load reads exactly `n` checked bytes.
    let raw = unsafe {
        match n {
            4 => v128_load32_zero(bytes[..4].as_ptr().cast()),
            8 => v128_load64_zero(bytes[..8].as_ptr().cast()),
            _ => v128_load(bytes[..CHUNK].as_ptr().cast()),
        }
    };
    v128_and(raw, i16x8_splat(0x00ff))
}

/// The sum of the four 32-bit lanes.
#[inline]
fn total(lanes: v128) -> u32 {
    (i32x4_extract_lane::<0>(lanes)
        + i32x4_extract_lane::<1>(lanes)
        + i32x4_extract_lane::<2>(lanes)
        + i32x4_extract_lane::<3>(lanes)) as u32
}

/// The SIMD128 [`super::StatsFn`].
///
/// # Safety
///
/// None beyond the build's SIMD128.
pub(super) unsafe fn stats(src: &[u8], stride: usize, n: usize, edge: &[u8]) -> EstimateStats {
    assert!(super::ESTIMATE_SIDES.contains(&n) && edge.len() > 3 * n);
    let width = n.min(CHUNK);
    let ones = i16x8_splat(1);
    let valid = even(&[u8::MAX; CHUNK], width);
    let (mut sum, mut squares) = (i32x4_splat(0), i32x4_splat(0));
    let mut sse = [i32x4_splat(0); 4];
    let square = |a: v128, b: v128| {
        let d = i16x8_sub(a, b);
        i32x4_dot_i16x8(d, d)
    };
    for r in (0..n).step_by(SAMPLE_STEP) {
        let left = v128_and(i16x8_splat(i16::from(edge[n - 1 - r])), valid);
        for c0 in (0..n).step_by(CHUNK) {
            let v = even(&src[r * stride + c0..], width);
            sum = i32x4_add(sum, i32x4_dot_i16x8(v, ones));
            squares = i32x4_add(squares, i32x4_dot_i16x8(v, v));
            let predictions = [
                even(&edge[n + 1 + c0..], width),
                left,
                even(&edge[n + 2 + r + c0..], width),
                even(&edge[n - r + c0..], width),
            ];
            for (acc, p) in sse.iter_mut().zip(predictions) {
                *acc = i32x4_add(*acc, square(v, p));
            }
        }
    }
    EstimateStats {
        sum: total(sum),
        squares: total(squares),
        sse: sse.map(total),
    }
}
