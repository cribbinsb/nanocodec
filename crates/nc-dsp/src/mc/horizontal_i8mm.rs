//! Six-tap horizontal rows using dav1d 1.5.4's USMMLA layout in
//! `src/arm/64/mc_dotprod.S` (BSD-2-Clause, see `THIRD_PARTY.md`).

use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, uint8x16_t, vcombine_s8, vext_s8, vld1q_s16, vld1q_u8,
    vmovn_s16, vqtbl1q_u8, vreinterpretq_s16_s32, vuzp1q_s16,
};

pub(super) const SAMPLES: [[u8; 16]; 2] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13],
];

pub(super) struct SixTap {
    coefficients: int8x16_t,
    samples: [uint8x16_t; 2],
}

impl SixTap {
    /// The outer taps are zero. Stagger the two columns by one sample so
    /// each matrix multiply yields four consecutive outputs.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn new(taps: &[i16; 8]) -> Self {
        debug_assert_eq!((taps[0], taps[7]), (0, 0));
        // SAFETY: Coefficients and masks each cover the complete vector load.
        unsafe {
            let taps = vmovn_s16(vld1q_s16(taps.as_ptr()));
            Self {
                coefficients: vcombine_s8(vext_s8::<1>(taps, taps), taps),
                samples: [vld1q_u8(SAMPLES[0].as_ptr()), vld1q_u8(SAMPLES[1].as_ptr())],
            }
        }
    }

    /// Eight sums already fit i16. `row` starts at the first nonzero tap
    /// and must have 16 readable bytes. The CPU must support I8MM.
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "keep matrix operations in the row loop"
    )]
    pub(super) unsafe fn sum(&self, row: *const u8, bias: int32x4_t) -> int16x8_t {
        // SAFETY: the caller supplies 16 readable bytes and I8MM.
        unsafe { matrix_sum(vld1q_u8(row), self.samples, self.coefficients, bias) }
    }
}

/// The first eight outputs from two four-output byte matrix products.
///
/// # Safety
///
/// Requires I8MM. The input and masks must place all samples used by the
/// filter in the eight-byte matrix rows, and each sum must fit i16.
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "keep matrix operations in the row loop"
)]
pub(super) unsafe fn matrix_sum(
    pixels: uint8x16_t,
    samples: [uint8x16_t; 2],
    coefficients: int8x16_t,
    bias: int32x4_t,
) -> int16x8_t {
    // SAFETY: the caller enables I8MM and supplies the filter operands. The
    // two accumulators only touch registers.
    unsafe {
        let first = vqtbl1q_u8(pixels, samples[0]);
        let second = vqtbl1q_u8(pixels, samples[1]);
        let (mut low, mut high) = (bias, bias);
        core::arch::asm!(
            ".arch_extension i8mm",
            "usmmla {low:v}.4s, {first:v}.16b, {taps:v}.16b",
            "usmmla {high:v}.4s, {second:v}.16b, {taps:v}.16b",
            low = inout(vreg) low, high = inout(vreg) high,
            first = in(vreg) first, second = in(vreg) second,
            taps = in(vreg) coefficients,
            options(pure, nomem, nostack, preserves_flags),
        );
        vuzp1q_s16(vreinterpretq_s16_s32(low), vreinterpretq_s16_s32(high))
    }
}
