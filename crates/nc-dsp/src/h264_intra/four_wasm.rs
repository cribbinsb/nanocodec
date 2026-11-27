use core::arch::wasm32::{
    i8x16_shuffle, i8x16_swizzle, u8x16_avgr, u8x16_splat, u8x16_sub, v128_and, v128_load, v128_or,
    v128_store, v128_xor,
};

use super::{Kernels, MASKS};

pub(super) const KERNELS: Kernels = Kernels {
    prepare,
    select,
    single,
};

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn prepare(line: &[u8; 16]) -> [u8; 32] {
    let mut output = [0; 32];
    // SAFETY: the input is 16 valid bytes and each store writes one half of output.
    unsafe {
        let a = v128_load(line.as_ptr().cast());
        let zero = u8x16_splat(0);
        let b = i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(a, zero);
        let c = i8x16_shuffle::<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16>(a, zero);
        let pairs = u8x16_avgr(a, b);
        let parity = v128_and(v128_xor(a, c), u8x16_splat(1));
        let triples = u8x16_avgr(u8x16_sub(u8x16_avgr(a, c), parity), b);
        v128_store(output.as_mut_ptr().cast(), pairs);
        v128_store(output.as_mut_ptr().add(16).cast(), triples);
    }
    output
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn select(table: &[u8; 32], mode: usize) -> [u8; 16] {
    // SAFETY: the dispatcher checks the directional mode.
    let masks = unsafe { MASKS.get_unchecked(mode) };
    let mut output = [0; 16];
    // SAFETY: each table, permutation mask and destination is 16 valid bytes.
    unsafe {
        let pairs = v128_load(table.as_ptr().cast());
        let triples = v128_load(table.as_ptr().add(16).cast());
        let pair_mask = v128_load(masks[0].as_ptr().cast());
        let triple_mask = v128_load(masks[1].as_ptr().cast());
        let row = v128_or(
            i8x16_swizzle(pairs, pair_mask),
            i8x16_swizzle(triples, triple_mask),
        );
        v128_store(output.as_mut_ptr().cast(), row);
    }
    output
}

#[target_feature(enable = "simd128")]
unsafe fn single(line: &[u8; 16], mode: usize) -> [u8; 16] {
    // SAFETY: both operations share the checked ISA, line and mode.
    unsafe { select(&prepare(line), mode) }
}
