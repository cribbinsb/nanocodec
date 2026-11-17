//! Compact CDEF scratch windows in dav1d's NEON layout: stride 16 for
//! luma 8x8, stride 8 for chroma 4x4, with two rows/columns of padding.
//! Based on dav1d 1.5.4 `src/arm/64/cdef.S` (BSD-2-Clause, Copyright 2018
//! VideoLAN/dav1d authors, 2019 Martin Storsjo; see `THIRD_PARTY.md`).

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use super::CDEF_PADDING;

pub(super) const LUMA_STRIDE: usize = 16;
pub(super) const CHROMA_STRIDE: usize = 8;
pub(super) const LUMA_SAMPLES: usize = LUMA_STRIDE * (8 + 2 * CDEF_PADDING);
pub(super) const CHROMA_SAMPLES: usize = CHROMA_STRIDE * (4 + 2 * CDEF_PADDING);

/// Only each row's N+4 sample payload is initialized. Trailing scratch
/// columns stay uninitialized, as in dav1d; no filter load reaches them.
#[repr(C, align(16))]
pub(super) struct Padded<T: Copy, const A: usize> {
    data: [MaybeUninit<T>; A],
}

impl<T: Copy, const A: usize> Padded<T, A> {
    #[inline(always)]
    pub(super) fn as_ptr(&self) -> *const T {
        self.data.as_ptr().cast()
    }
}

/// Validate the N-square block plus two samples on every side and return
/// its window's top-left index. Units are samples for byte and word inputs.
#[inline(always)]
pub(super) fn window_start<T, const N: usize>(src: &[T], stride: usize, origin: usize) -> usize {
    let side = N + 2 * CDEF_PADDING;
    // The caller can supply arbitrary usize indices. The exclusive tail
    // can need more than usize, so validate it once in u128. A successful
    // check also proves the smaller prefix offset fits usize.
    let end = origin as u128
        + (N + CDEF_PADDING - 1) as u128 * stride as u128
        + (N + CDEF_PADDING) as u128;
    assert!(
        stride >= side && end <= src.len() as u128,
        "CDEF source window too small"
    );
    origin
        .checked_sub(CDEF_PADDING * stride + CDEF_PADDING)
        .expect("CDEF padding inside source")
}

/// Compact a word window whose out-of-frame values already carry
/// `CDEF_UNAVAILABLE`, when its pitch cannot be borrowed directly. This
/// helper inlines into the NEON entry and creates no intermediate copy.
#[inline(always)]
pub(super) fn compact<T: Copy, const N: usize, const S: usize, const A: usize>(
    src: &[T],
    stride: usize,
    origin: usize,
) -> Padded<T, A> {
    let side = N + 2 * CDEF_PADDING;
    assert!(S >= side && A == S * side, "CDEF compact layout");
    let top = window_start::<T, N>(src, stride, origin);
    let mut out = Padded {
        data: [MaybeUninit::uninit(); A],
    };
    for row in 0..side {
        // SAFETY: the source payload is within the checked window. Each
        // destination payload lies in its distinct S-sample row; the local
        // scratch cannot alias src. No reference covers uninitialized holes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.as_ptr().add(top + row * stride),
                out.data.as_mut_ptr().cast::<T>().add(row * S),
                side,
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_windows<const N: usize>() {
        let side = N + 2 * CDEF_PADDING;
        for stride in [side, side + 1, 1280] {
            for prefix in [0, 1, 3] {
                let len = prefix + (side - 1) * stride + side;
                let source = vec![0u8; len];
                let origin = prefix + CDEF_PADDING * stride + CDEF_PADDING;
                assert_eq!(window_start::<_, N>(&source, stride, origin), prefix);
                assert!(
                    std::panic::catch_unwind(|| {
                        window_start::<_, N>(&source[..len - 1], stride, origin)
                    })
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn minimal_windows_include_the_last_sample() {
        minimal_windows::<4>();
        minimal_windows::<8>();
        assert_eq!(window_start::<i16, 8>(&[0; 144], 12, 26), 0);
    }

    #[test]
    fn oversized_indices_are_rejected_before_pointer_arithmetic() {
        let source = [0u8; 256];
        for (stride, origin) in [(usize::MAX, usize::MAX), (16, usize::MAX), (4, 10)] {
            assert!(
                std::panic::catch_unwind(|| { window_start::<_, 4>(&source, stride, origin) })
                    .is_err()
            );
        }
    }
}
