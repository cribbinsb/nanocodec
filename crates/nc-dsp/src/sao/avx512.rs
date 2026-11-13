//! Encoder SAO statistics on thirty-two word lanes, reusing the shared
//! classification and accumulation graph. Application and band kernels keep
//! their inherited AVX2 pointers; decoder source-buffer handling is unchanged.

use super::{Rect, SaoFns, SaoStats, generic};
use crate::simd::avx512_i16::I16x32;

pub(crate) fn install(fns: &mut SaoFns) {
    fns.stats = stats;
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
unsafe fn stats(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    // A cropped 31/63-wide CTB would leave 31 scalar samples per row with
    // 32 lanes, versus 15 with AVX2. Keep that existing path until measured.
    let width = rect.x1 - rect.x0;
    // A single 32-wide sampled row cannot amortise the wider reduction:
    // SDE measured 484 vs 415 kernel instructions before routing (API +9.6%).
    // The router reduces that gap to 28 instructions. Two rows were neutral
    // before routing and now cost about 3.5% more; full blocks still win. No
    // samples or offset decisions are skipped, so streams remain identical.
    let short = width == 32 && rect.y1 - rect.y0 <= rect.row_step;
    // SAFETY: dispatch checked the rectangle and neighbours; this wrapper
    // enables both lane implementations' CPU features.
    unsafe {
        if width.is_multiple_of(32) && !short {
            wide_stats(src, src_stride, rec, rec_stride, offs, rect)
        } else {
            super::avx2::stats(src, src_stride, rec, rec_stride, offs, rect)
        }
    }
}

// Keep vector accumulators and their frame out of the fallback router. In
// particular, sampled final rows pay the AVX2 kernel plus a small router cost.
#[inline(never)]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
unsafe fn wide_stats(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    // SAFETY: the router retains dispatch's validated rectangle and CPU contract.
    unsafe { generic::stats::<I16x32>(src, src_stride, rec, rec_stride, offs, rect) }
}
