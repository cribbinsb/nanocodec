//! NEON reconstruction: eight i16 residuals per saturating add.
use super::Residual16;
use core::arch::aarch64::{
    vcombine_s16, vcreate_u8, vdupq_n_s16, vget_lane_u32, vld1_s16, vld1_u8, vld1q_s16, vmovl_u8,
    vqaddq_s16, vqmovun_s16, vreinterpret_u32_u8, vreinterpretq_s16_u16, vst1_u8,
};

#[target_feature(enable = "neon")]
pub(super) unsafe fn reconstruct(
    pred: &[u8],
    ps: usize,
    residual: Residual16<'_>,
    dst: &mut [u8],
    ds: usize,
    width: usize,
    height: usize,
) {
    let dc = vdupq_n_s16(residual.constant_or_zero());
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            // SAFETY: this eight-pixel group lies in the validated row spans;
            // Samples is packed width*height and the load reads eight values.
            unsafe {
                let r = match residual {
                    Residual16::Constant(_) => dc,
                    Residual16::Samples(r) => vld1q_s16(r.as_ptr().add(y * width + x)),
                };
                let p = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pred.as_ptr().add(y * ps + x))));
                vst1_u8(
                    dst.as_mut_ptr().add(y * ds + x),
                    vqmovun_s16(vqaddq_s16(p, r)),
                );
            }
            x += 8;
        }
        if x < width {
            // SAFETY: block widths are four or multiples of eight, so the tail
            // contains exactly four pixels and four packed i16 residuals.
            unsafe {
                let r = match residual {
                    Residual16::Constant(_) => dc,
                    Residual16::Samples(r) => {
                        let four = vld1_s16(r.as_ptr().add(y * width + x));
                        vcombine_s16(four, four)
                    }
                };
                let packed = pred.as_ptr().add(y * ps + x).cast::<u32>().read_unaligned();
                let p = vreinterpretq_s16_u16(vmovl_u8(vcreate_u8(u64::from(packed))));
                let bytes = vqmovun_s16(vqaddq_s16(p, r));
                dst.as_mut_ptr()
                    .add(y * ds + x)
                    .cast::<u32>()
                    .write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(bytes)));
            }
        }
    }
}
