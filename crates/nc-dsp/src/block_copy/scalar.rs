//! Portable copy/average/scale reference, also used for exact SIMD tails.

use nc_common::Sample;

pub(super) struct Scalar;
impl super::rows::Ops for Scalar {
    const COPY_BYTES: usize = 16;
    const PREP_BYTES: usize = 8;
    #[inline(never)]
    unsafe fn copy_width<const W: usize, const AVG: bool>(
        src: &[u8],
        ss: usize,
        dst: &mut [u8],
        ds: usize,
        w: usize,
        h: usize,
    ) {
        // SAFETY: forwarded rectangle and ISA contract. Only this width's
        // loop can contribute register saves to its frame.
        unsafe { super::rows::copy_width::<Self, W, AVG>(src, ss, dst, ds, w, h) }
    }
    #[inline(never)]
    unsafe fn prep_width<const W: usize, const SHIFT: i32>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
        bias: i16,
    ) {
        // SAFETY: validated footprint and bounded word precision.
        unsafe { super::rows::prep_width::<Self, W, SHIFT>(src, ss, dst, ds, w, h, bias) }
    }
    #[inline(never)]
    unsafe fn prep_unbiased_width<const W: usize>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
    ) {
        // SAFETY: validated footprint; byte samples <<4 fit signed words.
        unsafe { super::rows::prep_width::<Self, W, 4>(src, ss, dst, ds, w, h, 0) }
    }

    #[inline]
    unsafe fn copy<const N: usize, const AVG: bool>(src: *const u8, dst: *mut u8) {
        // SAFETY: the caller provides N initialized, nonoverlapping samples.
        unsafe {
            if AVG {
                for x in 0..N {
                    *dst.add(x) =
                        ((u16::from(*src.add(x)) + u16::from(*dst.add(x)) + 1) >> 1) as u8;
                }
            } else {
                dst.cast::<[u8; N]>()
                    .write_unaligned(src.cast::<[u8; N]>().read_unaligned());
            }
        }
    }

    #[inline]
    unsafe fn prep<const N: usize, const SHIFT: i32>(src: *const u8, dst: *mut i16, bias: i16) {
        // SAFETY: exactly N inputs/outputs; the caller bounds the result to i16.
        unsafe {
            for x in 0..N {
                *dst.add(x) = ((i32::from(*src.add(x)) << SHIFT) - i32::from(bias)) as i16;
            }
        }
    }
}

#[inline]
pub(crate) unsafe fn copy<const AVG: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: validated nonoverlapping rectangle, with readable output for AVG.
    unsafe { super::rows::copy::<Scalar, AVG>(src, ss, dst, ds, w, h) }
}

pub(crate) unsafe fn prep(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    size: (usize, usize),
    scale: (u32, i16),
) {
    let (w, h) = size;
    let (shift, bias) = scale;
    // SAFETY: validated rectangle, scaled values fit i16 and shift <= 7.
    unsafe { super::rows::prep_entry!(Scalar, src, ss, dst, ds, w, h, shift, bias) }
}

/// Generic high-depth callers share the same scalar scaling equation;
/// eight-bit callers with a DSP table use its vector entry instead.
pub(crate) fn prep_samples<T: Sample>(
    src: &[T],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    (w, h): (usize, usize),
    (shift, bias): (u32, i32),
) {
    if w == 0 || h == 0 {
        return;
    }
    super::check_rect(src.len(), ss, w, h);
    super::check_rect(dst.len(), ds, w, h);
    for y in 0..h {
        for (out, &sample) in dst[y * ds..][..w].iter_mut().zip(&src[y * ss..][..w]) {
            *out = ((sample.to_i32() << shift) - bias) as i16;
        }
    }
}

/// Common eight-bit AV1/AV2 compound precision: unbiased sample << 4.
#[inline]
pub(crate) unsafe fn prep_unbiased(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    (w, h): (usize, usize),
) {
    // SAFETY: validated rectangle, byte << 4 fits i16, and this entry's ISA is enabled.
    unsafe { super::rows::prep_unbiased::<Scalar>(src, ss, dst, ds, w, h) }
}
