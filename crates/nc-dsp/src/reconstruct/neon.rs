//! NEON reconstruction: eight residuals narrowed before a saturating add.
use super::Residual32;
use core::arch::aarch64::{
    vcombine_s16, vcreate_u8, vdupq_n_s16, vget_lane_u32, vld1_u8, vld1q_s32, vmovl_u8, vqaddq_s16,
    vqmovn_s32, vqmovun_s16, vreinterpret_u32_u8, vreinterpretq_s16_u16, vst1_u8,
};

#[target_feature(enable = "neon")]
pub(super) unsafe fn reconstruct(
    pred: &[u8],
    ps: usize,
    residual: Residual32<'_>,
    dst: &mut [u8],
    ds: usize,
    width: usize,
    height: usize,
) {
    let dc = vdupq_n_s16(
        (match residual {
            Residual32::Constant(dc) => dc,
            Residual32::Samples(_) => 0,
        })
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
    );
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            // SAFETY: this eight-pixel group lies in the validated row spans;
            // Samples is packed width*height and each load reads four values.
            unsafe {
                let r = match residual {
                    Residual32::Constant(_) => dc,
                    Residual32::Samples(r) => vcombine_s16(
                        vqmovn_s32(vld1q_s32(r.as_ptr().add(y * width + x))),
                        vqmovn_s32(vld1q_s32(r.as_ptr().add(y * width + x + 4))),
                    ),
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
            // contains exactly four pixels and four packed i32 residuals.
            unsafe {
                let r = match residual {
                    Residual32::Constant(_) => dc,
                    Residual32::Samples(r) => {
                        let four = vqmovn_s32(vld1q_s32(r.as_ptr().add(y * width + x)));
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
