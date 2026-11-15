//! SIMD128 reconstruction: eight i16 residuals per saturating add.
use super::Residual16;
use core::arch::wasm32::{
    i16x8_add_sat, i16x8_splat, u8x16_narrow_i16x8, u16x8_extend_low_u8x16, v128_load,
    v128_load32_zero, v128_load64_zero, v128_store32_lane, v128_store64_lane,
};

#[target_feature(enable = "simd128")]
pub(super) unsafe fn reconstruct(
    pred: &[u8],
    ps: usize,
    residual: Residual16<'_>,
    dst: &mut [u8],
    ds: usize,
    width: usize,
    height: usize,
) {
    let dc = i16x8_splat(residual.constant_or_zero());
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            // SAFETY: the validated row spans contain this group of eight;
            // Samples covers the corresponding eight i16 residuals.
            unsafe {
                let r = match residual {
                    Residual16::Constant(_) => dc,
                    Residual16::Samples(r) => v128_load(r.as_ptr().add(y * width + x).cast()),
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
                    Residual16::Constant(_) => dc,
                    Residual16::Samples(r) => {
                        v128_load64_zero(r.as_ptr().add(y * width + x).cast())
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
