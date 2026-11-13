//! WebAssembly SIMD128 kernels: the generic code on 4 lanes; the HEVC
//! 8/16/32-point inverses multiply 16-bit coefficient pairs and skip
//! trailing zero coefficients (`simd128_inverse`); the staged HEVC forward
//! rows run on 16-bit lanes (`txfm::simd128_forward`); `nonzero_mask` narrows
//! sixteen zero tests to bytes for one `bitmask`. This module only exists
//! in builds with SIMD128 enabled, so its kernels need no run-time check.

use super::{HevcForwardMode, HevcForwardStaging};
use crate::simd::simd128::I32x4;
use core::arch::wasm32::{
    i8x16_bitmask, i8x16_narrow_i16x8, i16x8_narrow_i32x4, i32x4_eq, i32x4_splat, v128_load,
};

mod wide {
    use crate::simd::simd128::I32x4;
    super::super::generic::codec_txfm_kernels!(; I32x4, I32x4, I32x4);
}

/// The SIMD128 table.
pub(crate) fn codec_txfm_fns() -> super::CodecTxfmFns {
    let mut table = wide::codec_txfm_fns();
    table.vvc_forward_dct = super::vvc_forward_simd::simd128;
    super::simd128_inverse::install(&mut table);
    table.nonzero_mask = nonzero_mask;
    table.hevc_forward_bounded[2] = staged16_bounded;
    table.hevc_forward_bounded[3] = staged32_bounded;
    table
}

/// The 16-point bounded forward: staged modes take the 16-bit packed rows
/// (`txfm::simd128_forward`), the exact one the generic 4-lane driver.
///
/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
unsafe fn staged16_bounded(
    residual: &[i16],
    stride: usize,
    limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    use super::generic::{self as g, HevcFdct16};
    // SAFETY: the caller's contract is forwarded.
    unsafe {
        if mode.staging == HevcForwardStaging::Exact {
            g::hevc_forward_bounded::<I32x4, HevcFdct16, 16, 4>(residual, stride, limit, out, mode)
        } else {
            g::hevc_forward_simd128_rows::<HevcFdct16, 16, 4>(residual, stride, limit, out, mode)
        }
    }
}

/// The 32-point bounded forward, as [`staged16_bounded`].
///
/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
unsafe fn staged32_bounded(
    residual: &[i16],
    stride: usize,
    limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    use super::generic::{self as g, HevcFdct32};
    // SAFETY: the caller's contract is forwarded.
    unsafe {
        if mode.staging == HevcForwardStaging::Exact {
            g::hevc_forward_bounded::<I32x4, HevcFdct32, 32, 5>(residual, stride, limit, out, mode)
        } else {
            g::hevc_forward_simd128_rows::<HevcFdct32, 32, 5>(residual, stride, limit, out, mode)
        }
    }
}

/// Levels tested per `bitmask`.
const MASK_GROUP: usize = 16;

/// `generic::nonzero_mask`: sixteen levels at a time are compared with
/// zero; the all-ones/all-zeros lane results survive the two saturating
/// narrows to bytes unchanged, and `i8x16.bitmask` gathers their signs
/// (the scalar fold costs about four instructions per level).
///
/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
#[target_feature(enable = "simd128")]
unsafe fn nonzero_mask(levels: &[i32]) -> u64 {
    let (groups, rest) = levels.as_chunks::<MASK_GROUP>();
    let zero = i32x4_splat(0);
    let mut zeros = 0;
    for (g, group) in groups.iter().enumerate() {
        // SAFETY: each quarter of `group` holds 4 levels.
        let [a, b, c, d] = unsafe {
            [
                v128_load(group[0..].as_ptr().cast()),
                v128_load(group[4..].as_ptr().cast()),
                v128_load(group[8..].as_ptr().cast()),
                v128_load(group[12..].as_ptr().cast()),
            ]
        }
        .map(|v| i32x4_eq(v, zero));
        let bytes = i8x16_narrow_i16x8(i16x8_narrow_i32x4(a, b), i16x8_narrow_i32x4(c, d));
        zeros |= u64::from(i8x16_bitmask(bytes)) << (MASK_GROUP * g);
    }
    let whole = MASK_GROUP * groups.len();
    let mut mask = !zeros;
    if whole < u64::BITS as usize {
        mask &= (1 << whole) - 1;
        mask |= super::generic::nonzero_mask(rest) << whole;
    }
    mask
}
