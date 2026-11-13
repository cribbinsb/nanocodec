//! Square block copies between planes and prediction buffers (motion
//! compensation results, reconstructions): one bounds check per block, then
//! one fixed-size row move per row.
//!
//! Safe slice code checks every row's range (about a dozen instructions per
//! row, ~570 per 16x16 + two 8x8 macroblock copy in callgrind); after the one
//! up-front check here each row is a single vector load and store. A copy
//! uses compiler-generated fixed-size moves; unit tests compare it with
//! row-by-row `copy_from_slice`. Dynamic plane copies also have a per-instance
//! AVX-512 entry, checked against the scalar row copy in checkasm.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;

/// Validated strided rectangle; source and destination do not overlap.
pub(crate) type CopyFn = unsafe fn(&[u8], usize, &mut [u8], usize, usize, usize);
pub(crate) fn copy(src: &[u8], ss: usize, dst: &mut [u8], ds: usize, w: usize, h: usize) {
    for y in 0..h {
        dst[y * ds..y * ds + w].copy_from_slice(&src[y * ss..y * ss + w]);
    }
}

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
        src_stride >= N && src.len() >= (N - 1) * src_stride + N,
        "source does not cover {N}x{N}"
    );
    assert!(
        dst.width() >= N && dst.height() >= N,
        "destination smaller than {N}x{N}"
    );
    let dst_stride = dst.stride();
    let dst = dst.data_mut();
    assert!(dst.len() >= (N - 1) * dst_stride + N, "destination rows");
    let (from, to) = (src.as_ptr(), dst.as_mut_ptr());
    for r in 0..N {
        // SAFETY: rows 0..N of N samples lie inside both slices (checked
        // above), which cannot overlap (one is borrowed mutably).
        unsafe {
            to.add(r * dst_stride)
                .cast::<[u8; N]>()
                .write_unaligned(from.add(r * src_stride).cast::<[u8; N]>().read_unaligned());
        }
    }
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
}
