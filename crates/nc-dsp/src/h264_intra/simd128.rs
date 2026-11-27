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
        let left = v128_load(start.cast());
        let centre = v128_load(start.add(1).cast());
        let right = v128_load(start.add(2).cast());
        let parity = v128_and(v128_xor(left, right), u8x16_splat(1));
        let pair = u8x16_avgr(left, centre);
        let triple = u8x16_avgr(u8x16_sub(u8x16_avgr(left, right), parity), centre);
        for offset in [0, 16, 32, 48] {
            let pair_lanes = v128_load(pair_mask.as_ptr().add(offset).cast());
            let triple_lanes = v128_load(triple_mask.as_ptr().add(offset).cast());
            let block = v128_or(
                i8x16_swizzle(pair, pair_lanes),
                i8x16_swizzle(triple, triple_lanes),
            );
            v128_store(out.as_mut_ptr().add(offset).cast(), block);
        }
    }
}
