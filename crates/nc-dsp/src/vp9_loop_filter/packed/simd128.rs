//! SIMD128 VP9 edge filtering with the shared sixteen-line dual schedule.
//! Reuses the paired-byte adapter already used by AV1 filtering.

use crate::lpf_bytes::{LANES, simd128::B};
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
                if lines >= 2 * LANES {
                    super::run_paired::<4>(data, edge, step, along, lines, width, limits);
                } else {
                    super::run::<B, 4>(data, edge, step, along, lines, width, limits);
                }
            }
            Vp9FilterWidth::Eight => {
                if lines >= 2 * LANES {
                    super::run_paired::<8>(data, edge, step, along, lines, width, limits);
                } else {
                    super::run::<B, 8>(data, edge, step, along, lines, width, limits);
                }
            }
            Vp9FilterWidth::Sixteen => {
                if lines >= 2 * LANES {
                    super::run_paired::<16>(data, edge, step, along, lines, width, limits);
                } else {
                    super::run::<B, 16>(data, edge, step, along, lines, width, limits);
                }
            }
        }
    }
}
