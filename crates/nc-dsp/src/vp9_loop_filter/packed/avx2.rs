//! AVX2 entry of the VP9 edge-filter schedule (paired 16-line groups on
//! `avx2_wide::B`, falling back to 8-line `avx2::B` for short tails).

use crate::{Vp9FilterLimits, Vp9FilterWidth};

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn filter(
    data: &mut [u8],
    edge: usize,
    step: usize,
    along: usize,
    lines: usize,
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
) {
    // SAFETY: dispatch validated the full footprint and this function enables AVX2.
    unsafe {
        match width {
            Vp9FilterWidth::Four => {
                super::run_paired::<4>(data, edge, step, along, lines, width, limits);
            }
            Vp9FilterWidth::Eight => {
                super::run_paired::<8>(data, edge, step, along, lines, width, limits);
            }
            Vp9FilterWidth::Sixteen => {
                super::run_paired::<16>(data, edge, step, along, lines, width, limits);
            }
        }
    }
}
