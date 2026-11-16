//! AVX-512 four-group entry of the VP9 edge-filter schedule.

use crate::{Vp9FilterLimits, Vp9FilterWidth};

#[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
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
                super::run_wide::<4>(data, edge, step, along, lines, width, limits);
            }
            Vp9FilterWidth::Eight => {
                super::run_wide::<8>(data, edge, step, along, lines, width, limits);
            }
            Vp9FilterWidth::Sixteen => {
                super::run_wide::<16>(data, edge, step, along, lines, width, limits);
            }
        }
    }
}
