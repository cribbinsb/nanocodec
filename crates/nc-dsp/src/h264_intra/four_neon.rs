use core::arch::aarch64::{
    uint8x16x2_t, vdupq_n_u8, vextq_u8, vhaddq_u8, vld1q_u8, vqtbl2q_u8, vrhaddq_u8, vst1q_u8,
};

use super::{DIRECTIONAL, Kernels};

pub(super) const KERNELS: Kernels = Kernels {
    prepare,
    select,
    single,
};

#[target_feature(enable = "neon")]
#[inline]
unsafe fn prepare(line: &[u8; 16]) -> [u8; 32] {
    let mut output = [0; 32];
    // SAFETY: the input is 16 valid bytes and each store writes one half of output.
    unsafe {
        let a = vld1q_u8(line.as_ptr());
        let zero = vdupq_n_u8(0);
        let b = vextq_u8::<1>(a, zero);
        let c = vextq_u8::<2>(a, zero);
        let pairs = vrhaddq_u8(a, b);
        let triples = vrhaddq_u8(vhaddq_u8(a, c), b);
        vst1q_u8(output.as_mut_ptr(), pairs);
        vst1q_u8(output.as_mut_ptr().add(16), triples);
    }
    output
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn select(table: &[u8; 32], mode: usize) -> [u8; 16] {
    // SAFETY: the dispatcher checks the directional mode.
    let masks = unsafe { DIRECTIONAL.get_unchecked(mode) };
    let mut output = [0; 16];
    // SAFETY: both tables, the permutation mask and destination are 16 bytes.
    unsafe {
        let pairs = vld1q_u8(table.as_ptr());
        let triples = vld1q_u8(table.as_ptr().add(16));
        let row = vqtbl2q_u8(uint8x16x2_t(pairs, triples), vld1q_u8(masks.as_ptr()));
        vst1q_u8(output.as_mut_ptr(), row);
    }
    output
}

#[target_feature(enable = "neon")]
unsafe fn single(line: &[u8; 16], mode: usize) -> [u8; 16] {
    // SAFETY: both operations share the checked ISA, line and mode.
    unsafe { select(&prepare(line), mode) }
}
