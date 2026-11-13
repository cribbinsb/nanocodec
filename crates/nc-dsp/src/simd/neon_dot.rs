//! The byte dot products (`udot`, `sdot`, Armv8.2 dot-product extension),
//! shared by the kernels installed after runtime dot-product detection.

use core::arch::aarch64::{int8x16_t, int32x4_t, uint8x16_t, uint32x4_t};

/// `sum` plus, in each u32 lane `k`, the dot product of bytes `4k..4k + 4`
/// of `a` and `b`.
///
/// Rust's dot-product intrinsic postdates the project's Rust 1.90 minimum,
/// so this is the single instruction as inline assembly.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub(crate) fn udot(mut sum: uint32x4_t, a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
    // SAFETY: dotprod is enabled; this instruction only accesses its
    // registers (no memory, stack or flags).
    unsafe {
        core::arch::asm!(
            "udot {sum:v}.4s, {a:v}.16b, {b:v}.16b",
            sum = inout(vreg) sum, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}

/// `sum` plus, in each i32 lane `k`, the dot product of the signed bytes
/// `4k..4k + 4` of `a` and `b` (inline assembly, as [`udot`]).
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub(crate) fn sdot(mut sum: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: dotprod is enabled; this instruction only accesses its
    // registers (no memory, stack or flags).
    unsafe {
        core::arch::asm!(
            "sdot {sum:v}.4s, {a:v}.16b, {b:v}.16b",
            sum = inout(vreg) sum, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}
