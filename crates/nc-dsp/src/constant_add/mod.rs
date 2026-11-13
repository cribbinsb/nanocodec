//! Add one constant residual to a block of 8-bit samples, clipping to
//! 0..=255: the reconstruction of a DC-only inverse transform, whose residual
//! is the same at every position (AV1 `DCT_DCT` with end of block 1; VP9's
//! `_1_add` class and the H.264/HEVC DC shortcuts have the same shape).
//!
//! The SIMD kernels work on bytes, as dav1d's DC-only inverse transforms do
//! (`inv_txfm_add_dct_dct_*` DC paths, BSD-2-Clause; see `THIRD_PARTY.md`):
//! the residual splits into a saturating unsigned add of `max(r, 0)` and a
//! saturating unsigned subtract of `max(-r, 0)`. At most one of the two is
//! nonzero, so the result is `clamp(p + r, 0, 255)` for every `r`; residuals
//! beyond `±255` saturate to the same clip, so they are first clamped to a
//! byte. Sixteen samples per instruction instead of widening to 16 bits.
//!
//! Widths are 4, 8 or a multiple of 16 (every transform side of AV1, VP9,
//! AV2, H.264 and HEVC).

#![allow(clippy::inline_always)]

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128;

/// The scalar reference: `dst = clamp(dst + residual, 0, 255)` over the
/// `width` x `height` block at the start of `dst`, rows `stride` apart.
pub(crate) fn scalar(residual: i32, width: usize, height: usize, dst: &mut [u8], stride: usize) {
    let residual = residual.clamp(-255, 255);
    for row in dst.chunks_mut(stride).take(height) {
        for sample in &mut row[..width] {
            *sample = (i32::from(*sample) + residual).clamp(0, 255) as u8;
        }
    }
}

/// Sixteen (or the low eight or four) byte lanes of one ISA.
pub(crate) trait ByteLanes: Copy {
    fn splat(value: u8) -> Self;
    fn add_sat(self, other: Self) -> Self;
    fn sub_sat(self, other: Self) -> Self;
    /// # Safety
    /// `src` must have `bytes` (16, 8 or 4) readable bytes; the ISA must be
    /// enabled.
    unsafe fn load(src: *const u8, bytes: usize) -> Self;
    /// # Safety
    /// `dst` must have `bytes` (16, 8 or 4) writable bytes; the ISA must be
    /// enabled.
    unsafe fn store(self, dst: *mut u8, bytes: usize);
}

/// The byte-saturating driver every ISA instantiates.
///
/// # Safety
///
/// `B`'s ISA must be enabled.
///
/// # Panics
///
/// If `dst` does not cover the block, or `width` is not 4, 8 or a multiple
/// of 16.
#[inline(always)]
pub(crate) unsafe fn add<B: ByteLanes>(
    residual: i32,
    width: usize,
    height: usize,
    dst: &mut [u8],
    stride: usize,
) {
    const VECTOR: usize = 16;
    assert!(width == 4 || width == 8 || width.is_multiple_of(VECTOR));
    assert!(height == 0 || (stride >= width && dst.len() >= (height - 1) * stride + width));
    let residual = residual.clamp(-255, 255);
    let up = B::splat(residual.max(0) as u8);
    let down = B::splat((-residual).max(0) as u8);
    let bytes = width.min(VECTOR);
    let base = dst.as_mut_ptr();
    for y in 0..height {
        for x in (0..width).step_by(VECTOR) {
            // SAFETY: the block check above covers `bytes` samples from
            // (x, y); the caller enables the ISA.
            unsafe {
                let p = base.add(y * stride + x);
                B::load(p, bytes).add_sat(up).sub_sat(down).store(p, bytes);
            }
        }
    }
}
