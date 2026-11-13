//! NEON entry of the VP9 edge-filter schedule: sixteen-line dual groups.
//! Layout reference: libvpx 1.16.0 `vpx_dsp/arm/loopfilter_neon.c`,
//! `vpx_lpf_*_dual_neon` (BSD-3-Clause; see `THIRD_PARTY.md`).

use crate::lpf_bytes::{LANES, neon::B};
use crate::{Vp9FilterLimits, Vp9FilterWidth};

#[target_feature(enable = "neon")]
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
