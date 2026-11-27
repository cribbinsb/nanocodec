//! High-depth luma grain: two four-lane halfword gathers replace the
//! individual word lookups in dav1d's NEON application schedule.

use super::{HighGrainScaling, scalar_luma};

core::arch::global_asm!(include_str!("high_sve2.S"), options(raw));

unsafe extern "C" {
    fn nc_av1_grain_high_luma_sve2(
        source: *const u16,
        grain: *const i16,
        scales: *const i16,
        output: *mut u16,
        samples: usize,
        lower: i32,
        upper: i32,
    );
}

/// # Safety
/// SVE2 must be available on this CPU and OS.
pub(super) unsafe fn luma(
    source: &[u16],
    grain: &[i16],
    scaling: &HighGrainScaling,
    bounds: [i32; 2],
    out: &mut [u16],
) {
    let len = source.len().min(grain.len()).min(out.len());
    let end = if scaling.depth.bits() <= 12 {
        len & !7
    } else {
        0
    };
    if end > 0 {
        // SAFETY: SVE2 is selected by the DSP table. The three row slices
        // cover end samples, a nonzero multiple of eight, and the assembly
        // clamps every gather index to the 4096-entry scaling domain.
        unsafe {
            nc_av1_grain_high_luma_sve2(
                source.as_ptr(),
                grain.as_ptr(),
                scaling.q15.as_ptr(),
                out.as_mut_ptr(),
                end,
                bounds[0],
                bounds[1],
            );
        }
    }
    if end < len {
        scalar_luma(
            &source[end..len],
            &grain[end..len],
            scaling,
            bounds,
            &mut out[end..len],
        );
    }
}
