//! SIMD128 reconstruction: eight signed residuals packed before clipping pixels.
use super::Residual32;
use core::arch::wasm32::{
    i16x8_add_sat, i16x8_narrow_i32x4, i16x8_splat, u8x16_narrow_i16x8, u16x8_extend_low_u8x16,
    v128_load, v128_load32_zero, v128_load64_zero, v128_store32_lane, v128_store64_lane,
};

#[target_feature(enable = "simd128")]
pub(super) unsafe fn reconstruct(
    pred: &[u8],
    ps: usize,
    residual: Residual32<'_>,
    dst: &mut [u8],
    ds: usize,
    width: usize,
    height: usize,
) {
    let dc = i16x8_splat(
        (match residual {
            Residual32::Constant(dc) => dc,
            Residual32::Samples(_) => 0,
        })
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
    );
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            // SAFETY: the validated row spans contain this group of eight;
            // Samples covers the corresponding eight i32 residuals.
            unsafe {
                let r = match residual {
                    Residual32::Constant(_) => dc,
                    Residual32::Samples(r) => i16x8_narrow_i32x4(
                        v128_load(r.as_ptr().add(y * width + x).cast()),
                        v128_load(r.as_ptr().add(y * width + x + 4).cast()),
                    ),
                };
                let p =
                    u16x8_extend_low_u8x16(v128_load64_zero(pred.as_ptr().add(y * ps + x).cast()));
                let sum = i16x8_add_sat(p, r);
                v128_store64_lane::<0>(
                    u8x16_narrow_i16x8(sum, sum),
                    dst.as_mut_ptr().add(y * ds + x).cast(),
                );
            }
            x += 8;
        }
        if x < width {
            // SAFETY: the only tail is a complete four-pixel block row.
            unsafe {
                let r = match residual {
                    Residual32::Constant(_) => dc,
                    Residual32::Samples(r) => {
                        let four = v128_load(r.as_ptr().add(y * width + x).cast());
                        i16x8_narrow_i32x4(four, four)
                    }
                };
                let p =
                    u16x8_extend_low_u8x16(v128_load32_zero(pred.as_ptr().add(y * ps + x).cast()));
                let sum = i16x8_add_sat(p, r);
                v128_store32_lane::<0>(
                    u8x16_narrow_i16x8(sum, sum),
                    dst.as_mut_ptr().add(y * ds + x).cast(),
                );
            }
        }
    }
}
