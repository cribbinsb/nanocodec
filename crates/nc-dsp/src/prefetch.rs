//! Cache prefetch hints for sample blocks that control code will read soon
//! (the motion search's next reference window).
//!
//! A hint never changes a result, only when the cache lines arrive, so there
//! is nothing for a checkasm test to compare: the "scalar reference" is no
//! hint at all. Hints are issued on aarch64 only, where they were measured
//! (Cortex-A78, h264-fast: cycles about -0.5% crowd / -0.35% bodycam, mean
//! of four runs, each -0.13..-0.77%, for +0.6% instructions). On x86-64 the same hints cost +0.5% instructions
//! (callgrind) and their cycle effect could not be timed on the busy host,
//! so they stay off there until it is.

use nc_common::PlaneView;

/// Bytes per cache line on the Cortex-A78 (and x86-64).
const CACHE_LINE: usize = 64;

/// Hint that the `width` x `height` block of `view` at (`x`, `y`) will be
/// read soon: one prefetch per cache line of each row. The part of the
/// block outside `view` is ignored.
pub fn prefetch_block(view: &PlaneView<'_>, x: usize, y: usize, width: usize, height: usize) {
    if !cfg!(target_arch = "aarch64") {
        return;
    }
    let width = width.min(view.width().saturating_sub(x));
    let rows = height.min(view.height().saturating_sub(y));
    if width == 0 {
        return;
    }
    let (base, stride) = (view.data().as_ptr(), view.stride());
    for row in y..y + rows {
        // Inside the view (clipped above): a view covers `width` samples
        // from column `x` of each of its rows. Each line holding one of
        // them is hinted once, by its first byte.
        let first = base.wrapping_add(row * stride + x);
        let last = first.wrapping_add(width - 1);
        let mut line = first.wrapping_sub(first as usize % CACHE_LINE);
        while line <= last {
            hint(line);
            line = line.wrapping_add(CACHE_LINE);
        }
    }
}

/// Prefetch the cache line holding `address` into L1 for reading. A
/// prefetch never faults, and callers pass lines holding view samples.
#[inline]
fn hint(address: *const u8) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm` only hints the memory system: it reads nothing the
    // program sees and never faults.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{address}]",
            address = in(reg) address,
            options(nostack, preserves_flags, readonly)
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = address;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_past_the_view_are_clipped() {
        let samples = vec![0u8; 100 * 50];
        let view = PlaneView::new(&samples, 100, 100, 50).unwrap();
        prefetch_block(&view, 0, 0, 100, 50);
        prefetch_block(&view, 90, 45, 64, 64);
        prefetch_block(&view, 100, 0, 8, 8);
        prefetch_block(&view, 0, 50, 8, 8);
        prefetch_block(&view, 200, 200, 8, 8);
    }
}
