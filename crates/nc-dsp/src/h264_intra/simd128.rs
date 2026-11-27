//! Packed libavc-style H.264 8x8 averages for SIMD128.

use core::arch::wasm32::{
    i8x16_swizzle, u8x16_avgr, u8x16_splat, u8x16_sub, v128_and, v128_load, v128_or, v128_store,
    v128_xor,
};

use super::{H264Intra8x8Refs, h264_intra_8x8_mode as mode, rows};

#[target_feature(enable = "simd128")]
pub(super) unsafe fn predict(refs: &H264Intra8x8Refs, mode: u8, out: &mut [u8; 64]) {
    if mode < mode::DIAGONAL_DOWN_LEFT {
        return rows::plain(refs, mode, out);
    }
    let index = usize::from(mode - mode::DIAGONAL_DOWN_LEFT);
    // SAFETY: the dispatcher checks the mode, and the padded border
    // contains all three vector loads.
    unsafe {
        let &first = rows::FIRST.get_unchecked(index);
        let [pair_mask, triple_mask] = rows::MASKS.get_unchecked(index);
        let start = refs.border.as_ptr().add(usize::from(first));
        let a = v128_load(start.cast());
        let b = v128_load(start.add(1).cast());
        let c = v128_load(start.add(2).cast());
        let parity = v128_and(v128_xor(a, c), u8x16_splat(1));
        let pair = u8x16_avgr(a, b);
        let triple = u8x16_avgr(u8x16_sub(u8x16_avgr(a, c), parity), b);
        for offset in [0, 16, 32, 48] {
            let p = v128_load(pair_mask.as_ptr().add(offset).cast());
            let q = v128_load(triple_mask.as_ptr().add(offset).cast());
            let block = v128_or(i8x16_swizzle(pair, p), i8x16_swizzle(triple, q));
            v128_store(out.as_mut_ptr().add(offset).cast(), block);
        }
    }
}
