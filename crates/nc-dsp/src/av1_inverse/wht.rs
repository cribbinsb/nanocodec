//! The AV1 lossless inverse transform (§7.13.2.10, used when `Lossless`):
//! the reversible 4x4 Walsh–Hadamard lifting shared with VP9 and AV2
//! ([`crate::reversible_wht4`]), on coefficients first shifted right by
//! [`UNIT_QUANT_SHIFT`]. Matches libaom `av1_highbd_iwht4x4_16_add_c`
//! (BSD-2-Clause, see `THIRD_PARTY.md`): rows first, then columns, the
//! result added to the prediction with 8-bit clipping.
//!
//! Scalar only: one libaom test vector is lossless, our encoder never is.

use nc_common::PlaneViewMut;

use crate::reversible_wht4;

/// Lossless coefficients carry two fractional bits (§7.13.2.10 `shift`,
/// libaom `UNIT_QUANT_SHIFT`).
const UNIT_QUANT_SHIFT: u32 = 2;
/// The only lossless transform size.
const SIDE: usize = 4;

/// Inverse WHT of the column-major 4x4 `coeffs` (coefficient (row `r`,
/// column `c`) at `c * 4 + r`), added to the top-left 4x4 of `dst`.
pub(crate) fn inverse_wht_add(coeffs: &[i32; SIDE * SIDE], dst: &mut PlaneViewMut<'_>) {
    let mut raster = [0; SIDE * SIDE];
    for (i, value) in raster.iter_mut().enumerate() {
        let (r, c) = (i / SIDE, i % SIDE);
        *value = coeffs[c * SIDE + r] >> UNIT_QUANT_SHIFT;
    }
    let residual = reversible_wht4::inverse(&raster);
    for (r, residual_row) in residual.as_chunks::<SIDE>().0.iter().enumerate() {
        for (pixel, &delta) in dst.row_mut(r)[..SIDE].iter_mut().zip(residual_row) {
            *pixel = (i32::from(*pixel) + delta).clamp(0, i32::from(u8::MAX)) as u8;
        }
    }
}
