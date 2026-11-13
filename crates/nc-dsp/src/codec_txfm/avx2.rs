//! AVX2 kernels: the generic code on 4 lanes for 4-point transforms and 8
//! lanes otherwise; the HEVC 16- and 32-point forwards run their staged
//! rows on 16-bit lanes (`txfm::avx2_forward`) and their exact lines on
//! 16-bit pairs (`avx2_paired`); HEVC inverses skip trailing zero
//! coefficients.

mod wide {
    use super::super::avx2_paired::{PairedFdct16, PairedFdct32};
    use crate::simd::avx2::{I32x4, I32x8};
    super::super::generic::codec_txfm_kernels!(#[target_feature(enable = "avx2")]; I32x4, I32x8, I32x8;
        PairedFdct16, PairedFdct32);
    super::super::generic::hevc_bounded_inverse_kernels!(#[target_feature(enable = "avx2")]; I32x8);
}

use core::mem::MaybeUninit;

use super::avx2_paired::{PairedFdct16, PairedFdct32};
use super::generic::{self as g, Line, MAX_AREA};
use super::{HevcForwardMode, HevcForwardStaging};
use crate::simd::avx2::I32x8;

use core::arch::x86_64::{
    _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_movemask_ps,
    _mm256_setzero_si256,
};

/// The AVX2 table.
pub(crate) fn codec_txfm_fns() -> super::CodecTxfmFns {
    let mut table = wide::codec_txfm_fns();
    table.vvc_forward_dct = super::vvc_forward_simd::avx2;
    wide::install_bounded_inverse(&mut table);
    table.hevc_forward_bounded[2] = staged16_bounded;
    table.hevc_forward_bounded[3] = staged32_bounded;
    table.nonzero_mask = nonzero_mask;
    table
}

/// The 16-point bounded forward: staged modes take the sixteen-lane 16-bit
/// rows (`txfm::avx2_forward`), the exact one the generic driver; both run
/// the paired column pass.
///
/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn staged16_bounded(
    residual: &[i16],
    stride: usize,
    limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    // SAFETY: the caller's contract is forwarded.
    unsafe { forward_bounded::<PairedFdct16, 16, 4>(residual, stride, limit, out, mode) }
}

/// The 32-point bounded forward, as [`staged16_bounded`].
///
/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn staged32_bounded(
    residual: &[i16],
    stride: usize,
    limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    // SAFETY: the caller's contract is forwarded.
    unsafe { forward_bounded::<PairedFdct32, 32, 5>(residual, stride, limit, out, mode) }
}

/// [`staged16_bounded`] of `N` points with the column line `T`.
///
/// # Safety
///
/// As [`staged16_bounded`].
#[allow(clippy::inline_always)] // a kernel body, inlined into its AVX2 wrappers
#[inline(always)]
unsafe fn forward_bounded<T: Line<I32x8>, const N: usize, const LOG2: u32>(
    residual: &[i16],
    stride: usize,
    limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    if mode.staging == HevcForwardStaging::Exact {
        // SAFETY: the caller's contract is forwarded.
        return unsafe {
            g::hevc_forward_bounded::<I32x8, T, N, LOG2>(residual, stride, limit, out, mode)
        };
    }
    let mut m = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    // SAFETY: AVX2 is available (kernel contract); the caller's slices cover
    // the block.
    unsafe { crate::txfm::avx2_forward::hevc_rows::<N>(residual, stride, &mut m[..N * N]) };
    // SAFETY: the rows initialised the complete N-square intermediate.
    let m = unsafe { g::written(&m, N * N) };
    // SAFETY: the caller's contract is forwarded.
    unsafe { g::hevc_forward_columns::<I32x8, T, N, LOG2>(m, limit, out, mode) }
}

/// `generic::nonzero_mask`: eight levels at a time compared with zero, their
/// lane signs gathered by one `movemask`.
#[target_feature(enable = "avx2")]
unsafe fn nonzero_mask(levels: &[i32]) -> u64 {
    const LANES: usize = 8;
    const LANE_MASK: u64 = (1 << LANES) - 1;
    let (groups, rest) = levels.as_chunks::<LANES>();
    let zero = _mm256_setzero_si256();
    let mut mask = 0;
    for (g, group) in groups.iter().enumerate() {
        // SAFETY: AVX2 is enabled here; `group` holds 8 levels.
        let v = unsafe { _mm256_loadu_si256(group.as_ptr().cast()) };
        let zeros = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero)));
        mask |= (!(zeros as u64) & LANE_MASK) << (LANES * g);
    }
    if !rest.is_empty() {
        mask |= super::generic::nonzero_mask(rest) << (LANES * groups.len());
    }
    mask
}
