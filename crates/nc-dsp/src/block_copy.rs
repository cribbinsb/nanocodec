//! Shared plain, rounded-average and compound-precision block copies for
//! encoder predictions/reconstructions and decoder motion compensation.
//! One validated rectangle feeds the same four-row schedule on scalar,
//! AVX2, Ice Lake AVX-512, NEON and SIMD128. Codec-specific prep precision
//! is `(sample << shift) - bias`; byte inputs and outputs stay bytes.
//!
//! Kernels receive nonoverlapping complete rows and a supported ISA.
//! Partial vectors access exactly the remaining payload, including an
//! unpadded final row. The independent checkasm oracle verifies every
//! destination sample and guard, not another instance of this schedule.

#![allow(unsafe_code)]

use crate::Isa;
use nc_common::{PlaneView, PlaneViewMut, Sample};

mod rows;
pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;

/// Validated strided rectangle: strides cover the width; slices do not overlap.
pub(crate) type CopyFn = unsafe fn(&[u8], usize, &mut [u8], usize, usize, usize);
type PrepUnbiasedFn = unsafe fn(&[u8], usize, &mut [i16], usize, (usize, usize));
type WholePrepFn = unsafe fn(&[u8], usize, &mut [i16], usize, usize, usize);
type PrepFn = unsafe fn(&[u8], usize, &mut [i16], usize, (usize, usize), (u32, i16));

/// Per-instance block-copy dispatch. Plain copies, rounded byte averages
/// and scaled byte-to-word copies share one row schedule across codecs.
#[derive(Clone, Copy, Debug)]
pub struct BlockCopy {
    pub(crate) copy: CopyFn,
    pub(crate) average: CopyFn,
    pub(crate) prep: PrepFn,
    pub(crate) prep_unbiased: PrepUnbiasedFn,
    /// Prediction already resolves a width class. Reuse it rather than
    /// entering the arbitrary-width dispatcher a second time.
    pub(crate) whole_put: [CopyFn; 5],
    pub(crate) whole_prep: [WholePrepFn; 5],
}

impl BlockCopy {
    /// Build the copy family for a supported ISA, retaining the scalar
    /// reference for explicit scalar dispatch.
    #[must_use]
    pub fn with_isa(isa: Isa) -> Option<Self> {
        if !isa.is_supported() {
            return None;
        }
        Some(match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => Self::table::<avx512::Avx512>(
                avx512::copy::<false>,
                avx512::copy::<true>,
                avx512::prep,
                avx512::prep_unbiased,
            ),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => Self::table::<avx2::Avx2>(
                avx2::copy::<false>,
                avx2::copy::<true>,
                avx2::prep,
                avx2::prep_unbiased,
            ),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => Self::table::<neon::Neon>(
                neon::copy::<false>,
                neon::copy::<true>,
                neon::prep,
                neon::prep_unbiased,
            ),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => Self::table::<simd128::Simd128>(
                simd128::copy::<false>,
                simd128::copy::<true>,
                simd128::prep,
                simd128::prep_unbiased,
            ),
            _ => Self::table::<scalar::Scalar>(
                scalar::copy::<false>,
                scalar::copy::<true>,
                scalar::prep,
                scalar::prep_unbiased,
            ),
        })
    }

    /// One family table: whole-pel prediction enters the width-specific
    /// loop directly; arbitrary rectangles keep the same width dispatcher.
    fn table<K: rows::Ops>(
        copy: CopyFn,
        average: CopyFn,
        prep: PrepFn,
        prep_unbiased: PrepUnbiasedFn,
    ) -> Self {
        Self {
            copy,
            average,
            prep,
            prep_unbiased,
            whole_put: [
                K::copy_width::<4, false>,
                K::copy_width::<8, false>,
                K::copy_width::<16, false>,
                K::copy_width::<32, false>,
                K::copy_width::<64, false>,
            ],
            whole_prep: [
                K::prep_unbiased_width::<4>,
                K::prep_unbiased_width::<8>,
                K::prep_unbiased_width::<16>,
                K::prep_unbiased_width::<32>,
                K::prep_unbiased_width::<64>,
            ],
        }
    }

    /// Copy a rectangle, optionally replacing each destination with
    /// `(source + destination + 1) >> 1` (VP9 compound's byte average).
    ///
    /// # Panics
    /// If either strided rectangle is not covered by its slice.
    #[inline]
    pub fn copy_rect(
        &self,
        src: &[u8],
        ss: usize,
        dst: &mut [u8],
        ds: usize,
        size: (usize, usize),
        average: bool,
    ) {
        let (w, h) = size;
        check_rect(src.len(), ss, w, h);
        check_rect(dst.len(), ds, w, h);
        let kernel = if average { self.average } else { self.copy };
        // SAFETY: both footprints are checked, references cannot overlap,
        // and construction checked the entry's ISA.
        unsafe { kernel(src, ss, dst, ds, w, h) }
    }

    /// Widen bytes to compound words: `(source << shift) - bias`.
    /// Strides are in samples, including the i16 destination stride.
    ///
    /// # Panics
    /// If a rectangle is too short, shift exceeds seven, or the scaled
    /// byte range does not fit i16.
    #[inline]
    pub fn prep_rect(
        &self,
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        size: (usize, usize),
        scale: (u32, i16),
    ) {
        let (w, h) = size;
        let (shift, bias) = scale;
        check_rect(src.len(), ss, w, h);
        check_rect(dst.len(), ds, w, h);
        assert!(
            shift <= 7 && (255_i32 << shift) - i32::from(bias) <= i32::from(i16::MAX),
            "block prep precision out of range"
        );
        // SAFETY: footprints/precision and this table's ISA are checked.
        unsafe {
            if scale == (4, 0) {
                (self.prep_unbiased)(src, ss, dst, ds, size);
            } else {
                (self.prep)(src, ss, dst, ds, size, scale);
            }
        }
    }
}

// A full usize product plus one row width fits twice the pointer width.
// wasm32 has native i64 arithmetic; using u128 there emits software calls.
#[cfg(target_pointer_width = "32")]
type Extent = u64;
#[cfg(target_pointer_width = "64")]
type Extent = u128;

/// A nonempty rectangle's exclusive tail must fit its sample slice. Widen
/// before the multiply and addition so neither can wrap. With N-bit inputs,
/// `(h - 1) * stride + w <= (2^N - 1)^2`, which fits the 2N-bit extent.
#[inline]
pub(crate) fn covers_rect(len: usize, stride: usize, w: usize, h: usize) -> bool {
    stride >= w && (h - 1) as Extent * stride as Extent + w as Extent <= len as Extent
}

/// Empty rectangles access no memory; all others validate their full tail
/// once before entering the raw row schedule.
fn check_rect(len: usize, stride: usize, w: usize, h: usize) {
    if w != 0 && h != 0 {
        assert!(
            covers_rect(len, stride, w, h),
            "block copy rectangle too small"
        );
    }
}

/// Typed scalar fallback for decoder samples wider than a byte.
pub(crate) fn copy_samples<T: Sample>(
    src: &[T],
    ss: usize,
    dst: &mut [T],
    ds: usize,
    size: (usize, usize),
) {
    let (w, h) = size;
    check_rect(src.len(), ss, w, h);
    check_rect(dst.len(), ds, w, h);
    if w == 0 || h == 0 {
        return;
    }
    for y in 0..h {
        dst[y * ds..][..w].copy_from_slice(&src[y * ss..][..w]);
    }
}

pub(crate) use scalar::prep_samples;

/// Copy the `N` x `N` block at the top left of `src` (rows `src_stride`
/// apart) into the top left of `dst`.
///
/// # Panics
///
/// If `src_stride < N`, `src` does not cover `N` rows of `N` samples, or
/// `dst` is smaller than `N` x `N`.
#[inline]
pub fn copy_square<const N: usize>(dst: &mut PlaneViewMut<'_>, src: &[u8], src_stride: usize) {
    assert!(
        N != 0 && covers_rect(src.len(), src_stride, N, N),
        "source does not cover {N}x{N}"
    );
    assert!(
        dst.width() >= N && dst.height() >= N,
        "destination smaller than {N}x{N}"
    );
    let dst_stride = dst.stride();
    let dst = dst.data_mut();
    assert!(covers_rect(dst.len(), dst_stride, N, N), "destination rows");
    // SAFETY: both N-square footprints were checked above. This static
    // convenience API shares the portable row driver without ISA detection.
    unsafe { rows::copy_width::<scalar::Scalar, N, false>(src, src_stride, dst, dst_stride, N, N) }
}

/// [`copy_square`] from a view.
///
/// # Panics
///
/// If `src` is smaller than `N` x `N`, or `dst` is.
#[inline]
pub fn copy_square_view<const N: usize>(dst: &mut PlaneViewMut<'_>, src: &PlaneView<'_>) {
    assert!(
        src.width() >= N && src.height() >= N,
        "source smaller than {N}x{N}"
    );
    copy_square::<N>(dst, src.data(), src.stride());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_exactly_the_block() {
        let src: Vec<u8> = (0..40 * 20).map(|i| (i * 7 % 251) as u8).collect();
        for (n, stride) in [(8usize, 13usize), (16, 40)] {
            let mut out = vec![0xa5u8; 24 * 18];
            let mut dst = PlaneViewMut::new(&mut out, 24, 18, 18).unwrap();
            match n {
                8 => copy_square::<8>(&mut dst, &src, stride),
                _ => copy_square::<16>(&mut dst, &src, stride),
            }
            for y in 0..18 {
                for x in 0..24 {
                    let want = if x < n && y < n {
                        src[y * stride + x]
                    } else {
                        0xa5
                    };
                    assert_eq!(out[y * 24 + x], want, "{n} at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "source does not cover")]
    fn rejects_a_short_source() {
        let mut out = vec![0u8; 64];
        let mut dst = PlaneViewMut::new(&mut out, 8, 8, 8).unwrap();
        copy_square::<8>(&mut dst, &[0u8; 63], 8);
    }

    #[test]
    fn rejects_overflowing_rectangles_before_pointer_arithmetic() {
        let copies = BlockCopy::with_isa(Isa::Scalar).unwrap();
        for (w, h, ss) in [
            (8, 4, usize::MAX),
            (usize::MAX, 2, usize::MAX),
            (8, usize::MAX, 8),
        ] {
            assert!(
                std::panic::catch_unwind(|| {
                    let mut dst = [0; 64];
                    copies.copy_rect(&[0; 64], ss, &mut dst, 8, (w, h), false);
                })
                .is_err()
            );
        }
    }

    #[test]
    fn rejects_an_unrepresentable_prep_scale() {
        let copies = BlockCopy::with_isa(Isa::Scalar).unwrap();
        for scale in [(8, 0), (7, -128), (0, i16::MIN)] {
            assert!(
                std::panic::catch_unwind(|| {
                    let mut dst = [0; 4];
                    copies.prep_rect(&[255; 4], 4, &mut dst, 4, (4, 1), scale);
                })
                .is_err()
            );
        }
    }
}
