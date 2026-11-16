//! NEON end of block: 8 levels per step, the table masked by `vtst`
//! (`level != 0`), a running `umax`, then `umaxv`.

use core::arch::aarch64::{
    vabsq_s32, vandq_u16, vcombine_u16, vdupq_n_u16, vdupq_n_u32, vld1q_s16, vld1q_s32, vld1q_u16,
    vmaxq_u16, vmaxvq_u16, vmovn_u32, vorrq_u32, vreinterpretq_u32_s32, vtstq_s16, vtstq_s32,
};

/// Levels per step.
const LANES: usize = 8;

/// The NEON [`super::EobFn`].
///
/// # Safety
///
/// None beyond NEON, which `AArch64` always has.
#[target_feature(enable = "neon")]
pub(super) unsafe fn end_of_block(levels: &[i16], table: &[u16]) -> usize {
    let n = levels.len().min(table.len());
    let (whole, _) = levels[..n].as_chunks::<LANES>();
    let (ends, _) = table[..n].as_chunks::<LANES>();
    let mut largest = vdupq_n_u16(0);
    for (chunk, end) in whole.iter().zip(ends) {
        // SAFETY: each load is one whole 8-element chunk.
        let (level, end) = unsafe { (vld1q_s16(chunk.as_ptr()), vld1q_u16(end.as_ptr())) };
        let coded = vtstq_s16(level, level);
        largest = vmaxq_u16(largest, vandq_u16(coded, end));
    }
    let done = whole.len() * LANES;
    usize::from(vmaxvq_u16(largest)).max(super::scalar(&levels[done..n], &table[done..n]))
}

/// The NEON [`super::WideEobFn`]: 8 levels per step as two 4-lane loads,
/// whose nonzero masks narrow (`xtn`) onto the 8 table entries.
///
/// # Safety
///
/// None beyond NEON, which `AArch64` always has.
#[target_feature(enable = "neon")]
pub(super) unsafe fn end_of_block_wide(levels: &[i32], table: &[u16]) -> (usize, u32) {
    let n = levels.len().min(table.len());
    let (whole, _) = levels[..n].as_chunks::<LANES>();
    let (ends, _) = table[..n].as_chunks::<LANES>();
    let mut largest = vdupq_n_u16(0);
    let mut bits = vdupq_n_u32(0);
    for (chunk, end) in whole.iter().zip(ends) {
        // SAFETY: each load is one whole chunk or one half of it.
        let (low, high, end) = unsafe {
            (
                vld1q_s32(chunk.as_ptr()),
                vld1q_s32(chunk[LANES / 2..].as_ptr()),
                vld1q_u16(end.as_ptr()),
            )
        };
        let coded = vcombine_u16(
            vmovn_u32(vtstq_s32(low, low)),
            vmovn_u32(vtstq_s32(high, high)),
        );
        largest = vmaxq_u16(largest, vandq_u16(coded, end));
        bits = vorrq_u32(
            bits,
            vorrq_u32(
                vreinterpretq_u32_s32(vabsq_s32(low)),
                vreinterpretq_u32_s32(vabsq_s32(high)),
            ),
        );
    }
    // The OR of the four lanes.
    let mut lanes = [0u32; 4];
    // SAFETY: `lanes` holds four `u32`.
    unsafe { core::arch::aarch64::vst1q_u32(lanes.as_mut_ptr(), bits) };
    let done = whole.len() * LANES;
    let (end, tail_bits) = super::scalar_wide(&levels[done..n], &table[done..n]);
    (
        usize::from(vmaxvq_u16(largest)).max(end),
        lanes.iter().fold(tail_bits, |any, &lane| any | lane),
    )
}
