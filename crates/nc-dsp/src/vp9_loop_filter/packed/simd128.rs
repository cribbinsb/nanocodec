//! SIMD128 entry of the VP9 edge-filter schedule.

use crate::lpf_bytes::simd128::B;
use crate::{Vp9FilterLimits, Vp9FilterWidth};

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn filter(
    data: &mut [u8],
    edge: usize,
    step: usize,
    along: usize,
    lines: usize,
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
) {
    // SAFETY: dispatch validated the full footprint and this function enables the ISA.
    unsafe {
        match width {
            Vp9FilterWidth::Four => {
                super::run::<B, 4>(data, edge, step, along, lines, width, limits);
            }
            Vp9FilterWidth::Eight => {
                super::run::<B, 8>(data, edge, step, along, lines, width, limits);
            }
            Vp9FilterWidth::Sixteen => {
                super::run::<B, 16>(data, edge, step, along, lines, width, limits);
            }
        }
    }
}
