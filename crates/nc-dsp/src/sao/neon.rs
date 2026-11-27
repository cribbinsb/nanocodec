//! NEON SAO: byte-wide edge classification for statistics, generic application.
//!
//! Statistics run on 16 byte lanes for full CTBs and cropped rows of at
//! least 31 samples, accumulating
//! masked source and reconstruction bytes pairwise into 16-bit lanes. Other
//! shapes use the generic 8-lane kernel. See docs/handoff/dsp.md.

#![allow(clippy::inline_always)]

use super::{Rect, SaoFns, SaoStats, generic};
use crate::simd::{neon::I16x8, neon_dot::sdot};
use core::arch::aarch64::{
    int32x4_t, uint8x16_t, uint16x8_t, vaddlvq_u8, vaddlvq_u16, vaddq_u8, vaddvq_s32, vandq_u8,
    vceqq_u8, vcgtq_u8, vcltq_u8, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, vdupq_n_u16, veorq_u8,
    vld1q_u8, vpadalq_u8, vreinterpretq_s8_u8, vsubq_u8,
};

/// The NEON table.
///
/// NEON is always present on `AArch64`.
pub(crate) fn sao_fns() -> SaoFns {
    let stats = if std::arch::is_aarch64_feature_detected!("dotprod") {
        dot_stats
    } else {
        stats
    };
    SaoFns { stats, apply, band }
}

#[target_feature(enable = "neon")]
unsafe fn stats(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    // Orin core 3, `compare_byte_stats_with_generic_kernel`: the pairwise
    // byte path takes 0.61-0.67x the 8-lane kernel's time on 32- and 64-wide
    // rows at row steps 1/2/4. Include the 31-wide chroma CTB cropped by
    // one sample at the picture edge.
    const MIN_BYTE_STATS_WIDTH: usize = 31;
    if rect.x1 - rect.x0 < MIN_BYTE_STATS_WIDTH {
        // SAFETY: same checked rectangle and NEON dispatch contract.
        return unsafe { generic::stats::<I16x8>(src, src_stride, rec, rec_stride, offs, rect) };
    }
    // SAFETY: as above.
    unsafe { byte_stats::<ByteAccumulators>(src, src_stride, rec, rec_stride, offs, rect) }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_stats(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    // SAFETY: dispatch validated the rectangle and neighbours.
    unsafe {
        if rect.x1 - rect.x0 < 31 {
            generic::stats::<I16x8>(src, src_stride, rec, rec_stride, offs, rect)
        } else {
            byte_stats::<DotAccumulators>(src, src_stride, rec, rec_stride, offs, rect)
        }
    }
}

/// Samples per byte vector.
const BYTES: usize = 16;
/// Edge categories with statistics, as the wrapped byte value of the edge
/// sum (`-2, -1, 1, 2`); category 0 (flat) needs none.
const EDGE_CATEGORIES: [u8; 4] = [254, 255, 1, 2];
trait EdgeAccumulators {
    const FLUSH: usize;
    fn new() -> Self;
    fn add(&mut self, edge: uint8x16_t, source: uint8x16_t, center: uint8x16_t);
    fn flush(&mut self, out: &mut SaoStats);
}

/// Per-category sums and counts, reduced every 128 steps.
struct ByteAccumulators {
    source: [uint16x8_t; 4],
    reconstruction: [uint16x8_t; 4],
    count: [uint8x16_t; 4],
}

impl EdgeAccumulators for ByteAccumulators {
    // Each step adds at most 510 to a 16-bit lane and one to a byte count.
    const FLUSH: usize = 128;

    #[inline(always)]
    fn new() -> Self {
        // SAFETY: NEON kernels only.
        unsafe {
            Self {
                source: [vdupq_n_u16(0); 4],
                reconstruction: [vdupq_n_u16(0); 4],
                count: [vdupq_n_u8(0); 4],
            }
        }
    }

    /// `sum(source - reconstruction)` per category is accumulated as two
    /// unsigned pairwise byte sums, which costs two masks and two pairwise
    /// adds instead of widening, masking and adding signed differences.
    #[inline(always)]
    fn add(&mut self, edge: uint8x16_t, source: uint8x16_t, center: uint8x16_t) {
        // SAFETY: NEON kernels only.
        unsafe {
            for (k, value) in EDGE_CATEGORIES.into_iter().enumerate() {
                let mask = vceqq_u8(edge, vdupq_n_u8(value));
                self.source[k] = vpadalq_u8(self.source[k], vandq_u8(source, mask));
                self.reconstruction[k] = vpadalq_u8(self.reconstruction[k], vandq_u8(center, mask));
                // All-ones masks are -1 modulo 256: subtracting counts.
                self.count[k] = vsubq_u8(self.count[k], mask);
            }
        }
    }

    /// Reduce into `out` and restart.
    #[inline(always)]
    fn flush(&mut self, out: &mut SaoStats) {
        // SAFETY: NEON kernels only.
        unsafe {
            for k in 0..4 {
                out.sum[k] += i64::from(vaddlvq_u16(self.source[k]))
                    - i64::from(vaddlvq_u16(self.reconstruction[k]));
                out.count[k] += i64::from(vaddlvq_u8(self.count[k]));
            }
        }
        *self = Self::new();
    }
}

/// Each comparison mask is either -1 or 0 as signed bytes. Dot it with
/// centred source and reconstruction lanes separately. The bias cancels
/// when subtracting the sums, and dotting two -1s counts a matching lane.
struct DotAccumulators {
    source: [int32x4_t; 4],
    reconstruction: [int32x4_t; 4],
    count: [int32x4_t; 4],
}

impl EdgeAccumulators for DotAccumulators {
    // Even the horizontal sum of all 16 signed byte products fits i32.
    const FLUSH: usize = 65_536;

    #[inline(always)]
    fn new() -> Self {
        // SAFETY: NEON is present.
        let zero = unsafe { vdupq_n_s32(0) };
        Self {
            source: [zero; 4],
            reconstruction: [zero; 4],
            count: [zero; 4],
        }
    }

    #[inline(always)]
    fn add(&mut self, edge: uint8x16_t, source: uint8x16_t, center: uint8x16_t) {
        // SAFETY: only the dot-product kernel instantiates these accumulators.
        unsafe {
            let bias = vdupq_n_u8(128);
            let source = vreinterpretq_s8_u8(veorq_u8(source, bias));
            let center = vreinterpretq_s8_u8(veorq_u8(center, bias));
            let minus_one = vdupq_n_s8(-1);
            for (k, value) in EDGE_CATEGORIES.into_iter().enumerate() {
                let mask = vreinterpretq_s8_u8(vceqq_u8(edge, vdupq_n_u8(value)));
                self.source[k] = sdot(self.source[k], source, mask);
                self.reconstruction[k] = sdot(self.reconstruction[k], center, mask);
                self.count[k] = sdot(self.count[k], minus_one, mask);
            }
        }
    }

    #[inline(always)]
    fn flush(&mut self, out: &mut SaoStats) {
        // SAFETY: NEON is present.
        unsafe {
            for k in 0..4 {
                out.sum[k] += i64::from(vaddvq_s32(self.reconstruction[k]))
                    - i64::from(vaddvq_s32(self.source[k]));
                out.count[k] += i64::from(vaddvq_s32(self.count[k]));
            }
        }
        *self = Self::new();
    }
}

/// Edge statistics on 16 byte lanes. For the first partial vector, load
/// inside the rectangle and set the unused lanes to flat (category zero).
#[inline(always)]
unsafe fn byte_stats<A: EdgeAccumulators>(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    const LANES: [u8; BYTES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let width = rect.x1 - rect.x0;
    let tail = width % BYTES;
    // SAFETY: the array holds a full vector.
    let tail_mask = unsafe { vcltq_u8(vld1q_u8(LANES.as_ptr()), vdupq_n_u8(tail as u8)) };
    let mut acc = A::new();
    let mut steps = 0;
    let mut out = SaoStats::default();
    for y in (rect.y0..rect.y1).step_by(rect.row_step) {
        let base = y * rec_stride + rect.x0;
        let a = base
            .checked_add_signed(offs.0)
            .expect("SAO neighbour inside plane");
        let b = base
            .checked_add_signed(offs.1)
            .expect("SAO neighbour inside plane");
        let center = &rec[base..base + width];
        let left = &rec[a..a + width];
        let right = &rec[b..b + width];
        let source_base = y * src_stride + rect.x0;
        let source = &src[source_base..source_base + width];
        if tail != 0 {
            // SAFETY: each checked row has at least one complete vector.
            let (edge, s, c) = unsafe { edge_samples(center, left, right, source, 0) };
            // SAFETY: the caller guarantees NEON.
            acc.add(unsafe { vandq_u8(edge, tail_mask) }, s, c);
            steps += 1;
            if steps == A::FLUSH {
                acc.flush(&mut out);
                steps = 0;
            }
        }
        for x in (tail..width).step_by(BYTES) {
            // SAFETY: the suffix beginning at tail has a whole number of vectors.
            let (edge, s, c) = unsafe { edge_samples(center, left, right, source, x) };
            acc.add(edge, s, c);
            steps += 1;
            if steps == A::FLUSH {
                acc.flush(&mut out);
                steps = 0;
            }
        }
    }
    acc.flush(&mut out);
    out
}

#[inline(always)]
unsafe fn edge_samples(
    center: &[u8],
    left: &[u8],
    right: &[u8],
    source: &[u8],
    x: usize,
) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // SAFETY: NEON is enabled and every load takes a checked 16-byte slice.
    unsafe {
        let middle = vld1q_u8(center[x..x + BYTES].as_ptr());
        let before = vld1q_u8(left[x..x + BYTES].as_ptr());
        let after = vld1q_u8(right[x..x + BYTES].as_ptr());
        let original = vld1q_u8(source[x..x + BYTES].as_ptr());
        // Comparison masks are 0/255. Subtracting modulo 256 represents
        // sign(middle - neighbour) as 0, 1, 255; adding gives edges -2..2.
        let edge = vaddq_u8(
            vsubq_u8(vcltq_u8(middle, before), vcgtq_u8(middle, before)),
            vsubq_u8(vcltq_u8(middle, after), vcgtq_u8(middle, after)),
        );
        (edge, original, middle)
    }
}

#[target_feature(enable = "neon")]
unsafe fn apply(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i8; 4],
    rect: Rect,
) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe {
        super::bytes::edge::<super::neon_bytes::Lanes, I16x8>(
            input, in_stride, out, out_stride, offs, offsets, rect,
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn band(
    plane: &mut [u8],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i8; 4],
    exempt: &[bool],
) {
    // SAFETY: dispatch checked the rectangle and exemption mask; this enables the ISA.
    unsafe {
        super::bytes::band::<super::neon_bytes::Lanes>(
            plane, stride, rect, position, offsets, exempt,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_stats_match_scalar_across_tails_flushes_and_sampling() {
        let stride = 137;
        let mut src = vec![0; stride * 132];
        let mut rec = vec![0; stride * 132];
        for pattern in 0..3 {
            for i in 0..src.len() {
                rec[i] = match pattern {
                    0 => ((i * 73 + i / 7) % 256) as u8,
                    _ => {
                        if i % 2 == 0 {
                            0
                        } else {
                            255
                        }
                    }
                };
                src[i] = match pattern {
                    0 => ((i * 31 + i / 13) % 256) as u8,
                    1 => 255,
                    _ => 0,
                };
            }
            for width in [1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129] {
                for height in [1, 17, 64, 129] {
                    for step in [1, 2, 4] {
                        let rect = Rect {
                            x0: 1,
                            x1: 1 + width,
                            y0: 1,
                            y1: 1 + height,
                            row_step: step,
                        };
                        for off in [1, stride, stride + 1, stride - 1] {
                            let offs = (-(off as isize), off as isize);
                            // SAFETY: all samples and neighbours fit in both planes;
                            // AArch64 always has NEON.
                            unsafe {
                                let want = (super::super::scalar::sao_fns().stats)(
                                    &src, stride, &rec, stride, offs, rect,
                                );
                                assert_eq!(
                                    stats(&src, stride, &rec, stride, offs, rect),
                                    want,
                                    "pattern{pattern} width{width} height{height} step{step} off{off}"
                                );
                                if std::arch::is_aarch64_feature_detected!("dotprod") {
                                    assert_eq!(
                                        dot_stats(&src, stride, &rec, stride, offs, rect),
                                        want,
                                        "dot pattern{pattern} width{width} height{height} step{step} off{off}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "native microbenchmark; run on an idle fixed-frequency core"]
    fn compare_byte_stats_with_generic_kernel() {
        use std::{hint::black_box, time::Instant};
        let stride = 96;
        let src: Vec<u8> = (0..stride * 68)
            .map(|i| ((i * 31 + i / 13) % 256) as u8)
            .collect();
        let rec: Vec<u8> = (0..stride * 68)
            .map(|i| ((i * 73 + i / 7) % 256) as u8)
            .collect();
        for width in [32, 64] {
            for step in [1, 2, 4] {
                for off in [1, stride, stride + 1, stride - 1] {
                    let rect = Rect {
                        x0: 1,
                        x1: 1 + width,
                        y0: 1,
                        y1: 1 + width,
                        row_step: step,
                    };
                    let offs = (-(off as isize), off as isize);
                    let mut timings = [0.0; 2];
                    for (variant, time) in timings.iter_mut().enumerate() {
                        let start = Instant::now();
                        for _ in 0..5000 {
                            // SAFETY: valid neighbour rectangle and NEON is present.
                            let result = unsafe {
                                if variant == 0 {
                                    generic::stats::<I16x8>(
                                        black_box(&src),
                                        stride,
                                        black_box(&rec),
                                        stride,
                                        offs,
                                        rect,
                                    )
                                } else {
                                    byte_stats::<ByteAccumulators>(
                                        black_box(&src),
                                        stride,
                                        black_box(&rec),
                                        stride,
                                        offs,
                                        rect,
                                    )
                                }
                            };
                            black_box(result);
                        }
                        *time = start.elapsed().as_secs_f64();
                    }
                    println!(
                        "SAO width={width} step={step} off={off} generic={} byte={} ratio={}",
                        timings[0],
                        timings[1],
                        timings[1] / timings[0]
                    );
                }
            }
        }
    }
}
