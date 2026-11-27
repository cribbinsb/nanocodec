//! Shared row traversal for a half-pel cross around a whole-pel centre.

#![allow(clippy::inline_always)]

/// The row load and four distortion accumulators of one native tier.
/// Implementations run inside a wrapper that enables their ISA.
pub(super) trait Cross {
    type Pixels: Copy;
    const LANES: usize;

    /// Read exactly `LANES` bytes at `ptr`.
    unsafe fn load(ptr: *const u8) -> Self::Pixels;

    fn update(&mut self, source: Self::Pixels, middle: Self::Pixels, taps: [Self::Pixels; 4]);
}

/// Share the source loads and roll the vertical taps from row to row.
///
/// # Safety
///
/// The caller enables the ISA and validates the whole source and cross
/// window. `W` is a multiple of `C::LANES`.
#[inline(always)]
pub(super) unsafe fn measure<C: Cross, const W: usize>(
    acc: &mut C,
    src: &[u8],
    ss: usize,
    rf: &[u8],
    rs: usize,
    h: usize,
) {
    // SAFETY: all loads are inside the source block or its checked
    // (W + 2) by (h + 2) reference window.
    unsafe {
        for x in (0..W).step_by(C::LANES) {
            let mut above = C::load(rf.as_ptr().add(x + 1));
            let mut middle = C::load(rf.as_ptr().add(rs + x + 1));
            for y in 0..h {
                let source = C::load(src.as_ptr().add(y * ss + x));
                let row = rf.as_ptr().add((y + 1) * rs + x);
                let left = C::load(row);
                let right = C::load(row.add(2));
                let below = C::load(row.add(rs + 1));
                acc.update(source, middle, [left, right, above, below]);
                above = middle;
                middle = below;
            }
        }
    }
}
