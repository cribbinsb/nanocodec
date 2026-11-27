//! Packed libavc-style H.264 8x8 averages and NEON table selection.

use core::arch::aarch64::{uint8x16x2_t, vhaddq_u8, vld1q_u8, vqtbl2q_u8, vrhaddq_u8, vst1q_u8};

use super::{H264Intra8x8Refs, h264_intra_8x8_mode as mode, rows};

#[target_feature(enable = "neon")]
pub(super) unsafe fn predict(refs: &H264Intra8x8Refs, mode: u8, out: &mut [u8; 64]) {
    if mode < mode::DIAGONAL_DOWN_LEFT {
        return rows::plain(refs, mode, out);
    }
    let index = usize::from(mode - mode::DIAGONAL_DOWN_LEFT);
    // SAFETY: the dispatcher checks the mode. All three vector loads
    // fit the padded border. Each result is two rows of the block.
    unsafe {
        let &first = rows::FIRST.get_unchecked(index);
        let permutation = rows::INDICES.get_unchecked(index);
        let start = refs.border.as_ptr().add(usize::from(first));
        let a = vld1q_u8(start);
        let b = vld1q_u8(start.add(1));
        let c = vld1q_u8(start.add(2));
        let table = uint8x16x2_t(vrhaddq_u8(a, b), vrhaddq_u8(vhaddq_u8(a, c), b));
        for offset in [0, 16, 32, 48] {
            let indices = vld1q_u8(permutation.as_ptr().add(offset));
            vst1q_u8(out.as_mut_ptr().add(offset), vqtbl2q_u8(table, indices));
        }
    }
}
