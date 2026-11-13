//! Scalar SAO edge-offset kernels: the per-sample reference (and the
//! kernels of builds without SIMD).

use super::{Rect, SaoFns, SaoStats, category};

/// The scalar table.
pub(crate) fn sao_fns() -> SaoFns {
    SaoFns {
        stats: stats_kernel,
        apply: apply_kernel,
        band,
    }
}

/// `edgeIdx − 2` of the sample at index `i`.
pub(crate) fn edge(plane: &[u8], i: usize, (off_a, off_b): (isize, isize)) -> i32 {
    let c = i32::from(plane[i]);
    let a = i32::from(plane[i.checked_add_signed(off_a).expect("neighbour inside")]);
    let b = i32::from(plane[i.checked_add_signed(off_b).expect("neighbour inside")]);
    (c - a).signum() + (c - b).signum()
}

/// Statistics of the samples `x0..x1` of row `y` (inlined into the SIMD
/// kernels' row tails, as [`apply_row`]).
#[inline(always)]
#[allow(clippy::inline_always)]
pub(crate) fn stats_row(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    y: usize,
    (x0, x1): (usize, usize),
    out: &mut SaoStats,
) {
    for x in x0..x1 {
        let i = y * rec_stride + x;
        if let Some(cat) = category(edge(rec, i, offs)) {
            out.sum[cat] += i64::from(src[y * src_stride + x]) - i64::from(rec[i]);
            out.count[cat] += 1;
        }
    }
}

/// Apply the offsets to the samples `x0..x1` of row `y`. Always inlined:
/// the SIMD kernels finish each row's tail with it, and compiled on its own
/// for the x86-64 baseline it was 0.4 % of x86 HEVC Faster instructions
/// (inlined, it runs under the kernel's target features).
#[inline(always)]
#[allow(clippy::inline_always)]
pub(crate) fn apply_row(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i8; 4],
    y: usize,
    (x0, x1): (usize, usize),
) {
    for x in x0..x1 {
        let i = y * in_stride + x;
        let offset = category(edge(input, i, offs)).map_or(0, |cat| i32::from(offsets[cat]));
        out[y * out_stride + x] = (i32::from(input[i]) + offset).clamp(0, 255) as u8;
    }
}

fn stats_kernel(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    let mut out = SaoStats::default();
    for y in (rect.y0..rect.y1).step_by(rect.row_step) {
        stats_row(
            src,
            src_stride,
            rec,
            rec_stride,
            offs,
            y,
            (rect.x0, rect.x1),
            &mut out,
        );
    }
    out
}

fn apply_kernel(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i8; 4],
    rect: Rect,
) {
    for y in rect.y0..rect.y1 {
        apply_row(
            input,
            in_stride,
            out,
            out_stride,
            offs,
            offsets,
            y,
            (rect.x0, rect.x1),
        );
    }
}

/// Independent scalar reference: select the wrapped band then add and clip.
#[inline(always)]
#[allow(clippy::inline_always)]
pub(super) fn band_sample(sample: u8, position: u8, offsets: [i8; 4]) -> u8 {
    let band = (sample >> 3).wrapping_sub(position) & 31;
    let offset = offsets.get(usize::from(band)).copied().unwrap_or(0);
    (i16::from(sample) + i16::from(offset)).clamp(0, 255) as u8
}
pub(super) fn band(
    plane: &mut [u8],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i8; 4],
    exempt: &[bool],
) {
    let width = rect.x1 - rect.x0;
    for y in rect.y0..rect.y1 {
        for (x, sample) in plane[y * stride + rect.x0..y * stride + rect.x1]
            .iter_mut()
            .enumerate()
        {
            if exempt.is_empty() || !exempt[(y - rect.y0) * width + x] {
                *sample = band_sample(*sample, position, offsets);
            }
        }
    }
}
