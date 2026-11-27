//! One- and two-segment HEVC luma edges. Complete and partial edges
//! share the same decisions and arithmetic as the ordered-run kernel.

use super::{HevcEdge, batch, samples};
use crate::simd::{I16x8Shuffles, I16x8Tiles};

pub(crate) use samples::transpose;

/// HEVC luma decisions and filtering, with each tC covering four lines.
///
/// # Safety
/// V's ISA is enabled and the caller validates the edge footprint.
/// V holds eight words.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn luma<V: I16x8Shuffles + I16x8Tiles>(plane: &mut [u8], e: &HevcEdge) {
    match e.lines {
        4 => luma_lines::<V, 4>(plane, e),
        8 => luma_lines::<V, 8>(plane, e),
        _ => unreachable!("HEVC luma edge needs four or eight lines"),
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn luma_lines<V: I16x8Shuffles + I16x8Tiles, const LINES: usize>(plane: &mut [u8], e: &HevcEdge) {
    let edge = samples::Edge {
        base: e.base,
        across: e.across,
        along: e.along,
        lines: LINES,
    };
    let values = samples::load::<V>(plane, &edge);
    let second = if LINES == 8 { e.tc[1] } else { 0 };
    let tc = V::splat_segments(&[e.tc[0], second, 0, 0, 0, 0, 0, 0]);
    if let Some(filtered) = batch::filter::<V, false>(values, V::splat(e.beta), tc) {
        samples::store(plane, &edge, filtered, (1, samples::POSITIONS - 1));
    }
}
