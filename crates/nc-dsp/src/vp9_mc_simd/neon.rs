//! NEON VP9 interpolation with widening byte products of tap magnitudes on
//! 16-byte rows (`uint8x16_t`), plus 16-wide `sdot` horizontal rows when
//! `dotprod` is available.
//!
//! Positive and negative products accumulate separately in `u16` (each sum
//! fits; `positive_and_negative_sums_fit_u16`). Their saturating difference is
//! the filter sum clipped below at zero, so a rounding saturating narrow gives
//! the reference's round-shift-clip exactly, for every table.

use super::driver::{self, RowKernel, SHIFT, layout};
use crate::Vp9Filter;
use crate::mc::neon_dotprod::DotFilter;
use core::arch::aarch64::{
    uint8x8_t, uint8x16_t, vcombine_u8, vdup_n_u8, vdup_n_u32, vdupq_n_u8, vdupq_n_u16,
    vget_low_u8, vld1_lane_u32, vld1_u8, vld1q_u8, vmlal_high_u8, vmlal_u8, vmull_high_u8,
    vmull_u8, vqrshrn_high_n_u16, vqrshrn_n_u16, vqsubq_u16, vreinterpret_u8_u32,
    vreinterpret_u32_u8, vrhadd_u8, vrhaddq_u8, vst1_lane_u32, vst1_u8, vst1q_u8,
};

struct Neon;

/// 8 outputs from the low halves of eight tap inputs `pixels(k)`.
#[inline(always)]
unsafe fn filter8<const LAYOUT: u8>(
    pixels: impl Fn(usize) -> uint8x8_t,
    f: &[uint8x16_t; 8],
) -> uint8x8_t {
    // SAFETY: the ISA entry enables NEON.
    unsafe {
        if LAYOUT == layout::BILINEAR {
            vqrshrn_n_u16::<{ SHIFT as i32 }>(vmlal_u8(
                vmull_u8(pixels(3), vget_low_u8(f[3])),
                pixels(4),
                vget_low_u8(f[4]),
            ))
        } else {
            let (mut positive, mut negative) = (vdupq_n_u16(0), vdupq_n_u16(0));
            for (k, &magnitude) in f.iter().enumerate() {
                let mag = vget_low_u8(magnitude);
                if layout::negative(LAYOUT, k) {
                    negative = vmlal_u8(negative, pixels(k), mag);
                } else {
                    positive = vmlal_u8(positive, pixels(k), mag);
                }
            }
            vqrshrn_n_u16::<{ SHIFT as i32 }>(vqsubq_u16(positive, negative))
        }
    }
}

/// 16 outputs from eight 16-byte tap inputs `pixels(k)`.
#[inline(always)]
unsafe fn filter16<const LAYOUT: u8>(
    pixels: impl Fn(usize) -> uint8x16_t,
    f: &[uint8x16_t; 8],
) -> uint8x16_t {
    // SAFETY: the ISA entry enables NEON.
    unsafe {
        if LAYOUT == layout::BILINEAR {
            let p3 = pixels(3);
            let p4 = pixels(4);
            let lo = vmlal_u8(
                vmull_u8(vget_low_u8(p3), vget_low_u8(f[3])),
                vget_low_u8(p4),
                vget_low_u8(f[4]),
            );
            let hi = vmlal_high_u8(vmull_high_u8(p3, f[3]), p4, f[4]);
            vqrshrn_high_n_u16::<{ SHIFT as i32 }>(vqrshrn_n_u16::<{ SHIFT as i32 }>(lo), hi)
        } else {
            let (mut pos_lo, mut neg_lo) = (vdupq_n_u16(0), vdupq_n_u16(0));
            let (mut pos_hi, mut neg_hi) = (vdupq_n_u16(0), vdupq_n_u16(0));
            for (k, &magnitude) in f.iter().enumerate() {
                let p = pixels(k);
                let p_lo = vget_low_u8(p);
                let m_lo = vget_low_u8(magnitude);
                if layout::negative(LAYOUT, k) {
                    neg_lo = vmlal_u8(neg_lo, p_lo, m_lo);
                    neg_hi = vmlal_high_u8(neg_hi, p, magnitude);
                } else {
                    pos_lo = vmlal_u8(pos_lo, p_lo, m_lo);
                    pos_hi = vmlal_high_u8(pos_hi, p, magnitude);
                }
            }
            let lo = vqsubq_u16(pos_lo, neg_lo);
            let hi = vqsubq_u16(pos_hi, neg_hi);
            vqrshrn_high_n_u16::<{ SHIFT as i32 }>(vqrshrn_n_u16::<{ SHIFT as i32 }>(lo), hi)
        }
    }
}

impl RowKernel for Neon {
    type Filters = [uint8x16_t; 8];
    type Row = uint8x16_t;
    const LANES: usize = 16;
    const COPY_ISA: crate::Isa = crate::Isa::Neon;

    #[inline(always)]
    unsafe fn prepare<const LAYOUT: u8>(taps: &[i16; 8]) -> Self::Filters {
        // SAFETY: NEON is enabled by the entry point; magnitudes are <= 127.
        // Avoid `taps.map`, which outlined `try_map` on aarch64.
        unsafe {
            [
                vdupq_n_u8(taps[0].unsigned_abs() as u8),
                vdupq_n_u8(taps[1].unsigned_abs() as u8),
                vdupq_n_u8(taps[2].unsigned_abs() as u8),
                vdupq_n_u8(taps[3].unsigned_abs() as u8),
                vdupq_n_u8(taps[4].unsigned_abs() as u8),
                vdupq_n_u8(taps[5].unsigned_abs() as u8),
                vdupq_n_u8(taps[6].unsigned_abs() as u8),
                vdupq_n_u8(taps[7].unsigned_abs() as u8),
            ]
        }
    }

    #[inline(always)]
    unsafe fn load<const N: usize>(src: *const u8) -> uint8x16_t {
        // SAFETY: the caller provides N readable bytes.
        unsafe {
            match N {
                16 => vld1q_u8(src),
                8 => vcombine_u8(vld1_u8(src), vdup_n_u8(0)),
                _ => vcombine_u8(
                    vreinterpret_u8_u32(vld1_lane_u32::<0>(src.cast(), vdup_n_u32(0))),
                    vdup_n_u8(0),
                ),
            }
        }
    }

    #[inline(always)]
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: uint8x16_t) {
        // SAFETY: the caller provides N writable bytes, also readable when AVG.
        unsafe {
            match N {
                16 => {
                    let row = if AVG {
                        vrhaddq_u8(row, vld1q_u8(dst))
                    } else {
                        row
                    };
                    vst1q_u8(dst, row);
                }
                8 => {
                    let lo = vget_low_u8(row);
                    let lo = if AVG { vrhadd_u8(lo, vld1_u8(dst)) } else { lo };
                    vst1_u8(dst, lo);
                }
                _ => {
                    let lo = vget_low_u8(row);
                    let lo = if AVG {
                        vrhadd_u8(
                            lo,
                            vreinterpret_u8_u32(vld1_lane_u32::<0>(dst.cast(), vdup_n_u32(0))),
                        )
                    } else {
                        lo
                    };
                    vst1_lane_u32::<0>(dst.cast(), vreinterpret_u32_u8(lo));
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(
        src: *const u8,
        f: &Self::Filters,
    ) -> uint8x16_t {
        // SAFETY: the caller provides N + 7 readable bytes: tap k reads k..k + N.
        unsafe {
            if N == 16 {
                filter16::<LAYOUT>(|k| vld1q_u8(src.add(k)), f)
            } else {
                vcombine_u8(
                    filter8::<LAYOUT>(|k| vget_low_u8(Self::load::<N>(src.add(k))), f),
                    vdup_n_u8(0),
                )
            }
        }
    }

    #[inline(always)]
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(
        rows: &[uint8x16_t; 8],
        f: &Self::Filters,
    ) -> uint8x16_t {
        // SAFETY: the ISA entry enables NEON.
        unsafe {
            if N == 16 {
                filter16::<LAYOUT>(|k| rows[k], f)
            } else {
                vcombine_u8(filter8::<LAYOUT>(|k| vget_low_u8(rows[k]), f), vdup_n_u8(0))
            }
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
    average: bool,
) {
    // SAFETY: dispatch validated the input and this entry enables NEON.
    unsafe {
        if average {
            super::driver::entry::<Neon, true>(kind, src, stride, phase, size, dst, dst_stride);
        } else {
            super::driver::entry::<Neon, false>(kind, src, stride, phase, size, dst, dst_stride);
        }
    }
}

/// Horizontal signed dot rows share the 32-bit accumulators used by
/// AV1/HEVC/VVC, widened to 16 outputs per row for 16/32/64-wide blocks.
/// The VP9 vertical pass reads byte-clipped horizontal rows and runs 16-wide
/// widening byte products (`filter16`).
struct Dotprod;

impl RowKernel for Dotprod {
    type Filters = (DotFilter, [uint8x16_t; 8]);
    type Row = uint8x16_t;
    const LANES: usize = 16;
    const COPY_ISA: crate::Isa = crate::Isa::Neon;

    #[inline(always)]
    unsafe fn prepare<const LAYOUT: u8>(taps: &[i16; 8]) -> Self::Filters {
        // SAFETY: the entry enables dotprod and all nonidentity taps fit i8.
        unsafe { (DotFilter::new(taps), Neon::prepare::<LAYOUT>(taps)) }
    }

    #[inline(always)]
    unsafe fn load<const N: usize>(src: *const u8) -> uint8x16_t {
        // SAFETY: forwarded from the driver.
        unsafe { Neon::load::<N>(src) }
    }

    #[inline(always)]
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: uint8x16_t) {
        // SAFETY: forwarded from the driver.
        unsafe { Neon::store::<N, AVG>(dst, row) }
    }

    #[inline(always)]
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(
        src: *const u8,
        filters: &Self::Filters,
    ) -> uint8x16_t {
        // SAFETY: the caller validates the N+7 byte source window.
        unsafe {
            if LAYOUT == layout::BILINEAR {
                Neon::horizontal::<N, LAYOUT>(src, &filters.1)
            } else if N == 16 {
                filters.0.vp9_row16(src)
            } else {
                vcombine_u8(filters.0.vp9_row::<N>(src), vdup_n_u8(0))
            }
        }
    }

    #[inline(always)]
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(
        rows: &[uint8x16_t; 8],
        filters: &Self::Filters,
    ) -> uint8x16_t {
        // SAFETY: forwarded from the driver.
        unsafe { Neon::vertical::<N, LAYOUT>(rows, &filters.1) }
    }
}

#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn convolve_dotprod(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
    average: bool,
) {
    // SAFETY: Dsp validated the buffers. The dot path handles 4-, 8- and
    // 16-output strips; the plain entry covers copies, bilinear and vertical-only.
    unsafe {
        if size.0 == 0
            || size.1 == 0
            || !size.0.is_multiple_of(4)
            || phase.0 == 0
            || kind == Vp9Filter::Bilinear
        {
            convolve(kind, src, stride, phase, size, dst, dst_stride, average);
            return;
        }
        macro_rules! filter {
            ($n:expr, $layout:expr, $avg:expr) => {
                driver::convolve::<Dotprod, $n, $layout, $avg>(
                    kind, src, stride, phase, size, dst, dst_stride,
                )
            };
        }
        macro_rules! at_width {
            ($n:expr) => {
                match (driver::layout_of(kind), average) {
                    (layout::OUTER, false) => filter!($n, { layout::OUTER }, false),
                    (layout::OUTER, true) => filter!($n, { layout::OUTER }, true),
                    (_, false) => filter!($n, { layout::ALTERNATING }, false),
                    (_, true) => filter!($n, { layout::ALTERNATING }, true),
                }
            };
        }
        if size.0.is_multiple_of(16) {
            at_width!(16);
        } else if size.0.is_multiple_of(8) {
            at_width!(8);
        } else {
            at_width!(4);
        }
    }
}
