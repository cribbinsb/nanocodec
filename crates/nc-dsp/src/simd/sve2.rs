//! Fixed-size SVE2 operations on the low 128 bits of each register.
//! These lane-local operations are valid at any supported SVE vector length.
//!
//! Rust 1.90 exposes ASIMD registers. The SVE2 operations use assembly. A
//! scoped assembler alias maps each allocator-selected v register to its
//! corresponding z register. The wrapper ABI uses fixed-size Rust SIMD types.

#![allow(
    clippy::inline_always,
    reason = "These operations stay in their caller's ISA context."
)]

use core::arch::aarch64::{
    int16x8_t, int32x4_t, int64x2_t, uint8x16_t, uint16x8_t, uint32x4_t, vdupq_n_s64, vld1q_s16,
    vld1q_u8, vqtbl1q_u8, vreinterpretq_s16_u8, vreinterpretq_s32_s64, vreinterpretq_u8_s16,
    vuzp1q_s32,
};
use core::arch::asm;

macro_rules! aliases {
    ($($reg:ident),+ $(,)?) => {
        concat!(
            ".arch_extension sve2\n",
            ".irp i,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n",
            $(concat!(".ifc v\\i,{", stringify!($reg), ":v}\n",
                      "nc_sve_", stringify!($reg), " .req z\\i\n.endif\n"),)+
            ".endr",
        )
    };
}

macro_rules! unalias {
    ($($reg:ident),+ $(,)?) => {
        concat!($(concat!(".unreq nc_sve_", stringify!($reg), "\n"),)+)
    };
}

/// High halves of four unsigned 32×32 products.
///
/// # Safety
/// The CPU and operating system must support SVE2.
#[inline(always)]
pub(crate) unsafe fn mul_high(left: uint32x4_t, right: uint32x4_t) -> uint32x4_t {
    let out;
    // SAFETY: one register-only SVE2 instruction, with no cross-lane effects.
    unsafe {
        asm!(
            aliases!(out, left, right),
            "umulh nc_sve_out.s, nc_sve_left.s, nc_sve_right.s",
            unalias!(out, left, right),
            out = lateout(vreg) out, left = in(vreg) left, right = in(vreg) right,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// High halves of eight unsigned 16×16 products.
///
/// # Safety
/// The CPU and operating system must support SVE2.
#[inline(always)]
pub(crate) unsafe fn mul_high_words(left: uint16x8_t, right: uint16x8_t) -> uint16x8_t {
    let out;
    // SAFETY: one register-only SVE2 instruction, with no cross-lane effects.
    unsafe {
        asm!(
            aliases!(out, left, right),
            "umulh nc_sve_out.h, nc_sve_left.h, nc_sve_right.h",
            unalias!(out, left, right),
            out = lateout(vreg) out, left = in(vreg) left, right = in(vreg) right,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// Add each group of four squared 16-bit differences to a 64-bit sum.
///
/// # Safety
/// The CPU and operating system must support SVE2. The accumulators and
/// their updated values fit in 32 bits each, with the high words zero.
#[inline(always)]
pub(crate) unsafe fn add_squares(mut sum: int32x4_t, diff: int16x8_t) -> int32x4_t {
    // SAFETY: SDOT groups four adjacent lanes without crossing 128 bits.
    unsafe {
        asm!(
            aliases!(sum, diff),
            "sdot nc_sve_sum.d, nc_sve_diff.h, nc_sve_diff.h",
            unalias!(sum, diff),
            sum = inout(vreg) sum, diff = in(vreg) diff,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}

/// Byte indices of two neighbouring four-tap windows. The last load
/// starts a sample earlier, so its indices shift by two bytes.
const HORIZONTAL_PAIRS: [[u8; 16]; 4] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13],
    [2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 10, 11],
    [6, 7, 8, 9, 10, 11, 12, 13, 8, 9, 10, 11, 12, 13, 14, 15],
];

/// Two groups of four sample/weight products, accumulated in signed words.
///
/// # Safety
/// SVE2 is available and INDEX selects one group of four coefficients.
#[inline(always)]
pub(crate) unsafe fn dot_halfwords<const INDEX: u32>(
    mut acc: int64x2_t,
    pixels: int16x8_t,
    taps: int16x8_t,
) -> int64x2_t {
    // SAFETY: the table selected SVE2 and all operands are initialized.
    unsafe {
        asm!(
            aliases!(acc, pixels, taps),
            "sdot nc_sve_acc.d, nc_sve_pixels.h, nc_sve_taps.h[{index}]",
            unalias!(acc, pixels, taps),
            acc = inout(vreg) acc, pixels = in(vreg) pixels, taps = in(vreg_low16) taps,
            index = const INDEX,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    acc
}

#[inline(always)]
unsafe fn indexed_dot<const INDEX: u32>(
    acc: int64x2_t,
    source: uint8x16_t,
    indices: uint8x16_t,
    taps: int16x8_t,
) -> int64x2_t {
    // SAFETY: the caller supplies the transpose and enables SVE2.
    unsafe { dot_halfwords::<INDEX>(acc, vreinterpretq_s16_u8(vqtbl1q_u8(source, indices)), taps) }
}

/// Eight adjacent outputs with dav1d's halfword SDOT schedule
/// (`src/arm/64/mc16_sve.S`, BSD-2-Clause). The three loads cover exactly
/// 15 samples. Arrange the two four-tap sums in 64-bit lanes, then extract
/// the low 32 bits of each sum in output order.
///
/// # Safety
/// SVE2 is available, `src` covers 15 signed samples, and sums fit i32.
#[inline(always)]
pub(crate) unsafe fn horizontal_8tap(src: *const i16, taps: &[i16; 8]) -> (int32x4_t, int32x4_t) {
    // SAFETY: the loads and narrowing honor the caller's row and range contract.
    unsafe {
        let a = vreinterpretq_u8_s16(vld1q_s16(src));
        let b = vreinterpretq_u8_s16(vld1q_s16(src.add(4)));
        let c = vreinterpretq_u8_s16(vld1q_s16(src.add(7)));
        let taps = vld1q_s16(taps.as_ptr());
        let [p01, p23, p01_shift, p23_shift] = HORIZONTAL_PAIRS.map(|m| vld1q_u8(m.as_ptr()));
        let zero = vdupq_n_s64(0);
        let r01 = indexed_dot::<1>(indexed_dot::<0>(zero, a, p01, taps), b, p01, taps);
        let r23 = indexed_dot::<1>(indexed_dot::<0>(zero, a, p23, taps), b, p23, taps);
        let r45 = indexed_dot::<1>(indexed_dot::<0>(zero, b, p01, taps), c, p01_shift, taps);
        let r67 = indexed_dot::<1>(indexed_dot::<0>(zero, b, p23, taps), c, p23_shift, taps);
        (
            vuzp1q_s32(vreinterpretq_s32_s64(r01), vreinterpretq_s32_s64(r23)),
            vuzp1q_s32(vreinterpretq_s32_s64(r45), vreinterpretq_s32_s64(r67)),
        )
    }
}

/// Widen, multiply-accumulate and round 8 signed halfword rotations.
/// SVE2's adjacent-lane narrowing mirrors its even/odd widening.
///
/// # Safety
/// The CPU and operating system must support SVE2. The rounded results
/// must fit in an i16.
#[inline(always)]
pub(crate) unsafe fn rotate<const BITS: u32>(
    a: int16x8_t,
    b: int16x8_t,
    w0: int16x8_t,
    w1: int16x8_t,
) -> int16x8_t {
    let out;
    // SAFETY: four widening products, then rounding and narrowing each pair.
    unsafe {
        asm!(
            aliases!(out, even, odd, a, b, w0, w1),
            "smullb nc_sve_even.s, nc_sve_a.h, nc_sve_w0.h",
            "smullt nc_sve_odd.s, nc_sve_a.h, nc_sve_w0.h",
            "smlalb nc_sve_even.s, nc_sve_b.h, nc_sve_w1.h",
            "smlalt nc_sve_odd.s, nc_sve_b.h, nc_sve_w1.h",
            "rshrnb nc_sve_out.h, nc_sve_even.s, #{bits}",
            "rshrnt nc_sve_out.h, nc_sve_odd.s, #{bits}",
            unalias!(out, even, odd, a, b, w0, w1),
            bits = const BITS,
            out = lateout(vreg) out, even = out(vreg) _, odd = out(vreg) _,
            a = in(vreg) a, b = in(vreg) b, w0 = in(vreg) w0, w1 = in(vreg) w1,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}
