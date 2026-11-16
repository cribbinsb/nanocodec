//! Scalar SAO kernels: the per-sample reference (and the kernels of builds
//! without SIMD). The apply and band kernels are written once over the
//! sample type and depth (`*_at`), for the 8-bit table and the
//! high-bit-depth one (`crate::high_depth`).

use nc_common::{BitDepth, Sample};

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
pub(crate) fn edge<T: Sample>(plane: &[T], i: usize, (off_a, off_b): (isize, isize)) -> i32 {
    let c = plane[i].to_i32();
    let a = plane[i.checked_add_signed(off_a).expect("neighbour inside")].to_i32();
    let b = plane[i.checked_add_signed(off_b).expect("neighbour inside")].to_i32();
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
    apply_row_at(
        input,
        in_stride,
        out,
        out_stride,
        offs,
        offsets.map(i16::from),
        y,
        (x0, x1),
        BitDepth::EIGHT,
    );
}

/// [`apply_row`] at any sample type and depth: `SaoOffsetVal` in
/// `offsets` (already scaled to the depth), `Clip1` to `depth`.
#[inline(always)]
#[allow(clippy::inline_always, clippy::too_many_arguments)]
pub(crate) fn apply_row_at<T: Sample>(
    input: &[T],
    in_stride: usize,
    out: &mut [T],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i16; 4],
    y: usize,
    (x0, x1): (usize, usize),
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    for x in x0..x1 {
        let i = y * in_stride + x;
        let offset = category(edge(input, i, offs)).map_or(0, |cat| i32::from(offsets[cat]));
        out[y * out_stride + x] = depth.clip_to(input[i].to_i32() + offset);
    }
}

/// The edge-offset apply kernel at any sample type and depth (the scalar
/// [`super::ApplyFn`] with offsets scaled to the depth).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_at<T: Sample>(
    input: &[T],
    in_stride: usize,
    out: &mut [T],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i16; 4],
    rect: Rect,
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    for y in rect.y0..rect.y1 {
        apply_row_at(
            input,
            in_stride,
            out,
            out_stride,
            offs,
            offsets,
            y,
            (rect.x0, rect.x1),
            depth,
        );
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
    apply_at(
        input,
        in_stride,
        out,
        out_stride,
        offs,
        offsets.map(i16::from),
        rect,
        BitDepth::EIGHT,
    );
}

/// Bands of the band offset: the 32 equal ranges of the sample values
/// (§8.7.3.2: `bandShift = BitDepth − 5`).
const BAND_BITS: u32 = 5;

/// Independent scalar reference: select the wrapped band then add and clip.
#[inline(always)]
#[allow(clippy::inline_always)]
pub(super) fn band_sample(sample: u8, position: u8, offsets: [i8; 4]) -> u8 {
    band_sample_at(sample, position, offsets.map(i16::from), BitDepth::EIGHT)
}

/// [`band_sample`] at any sample type and depth.
#[inline(always)]
#[allow(clippy::inline_always)]
pub(crate) fn band_sample_at<T: Sample>(
    sample: T,
    position: u8,
    offsets: [i16; 4],
    depth: BitDepth,
) -> T {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let value = sample.to_i32();
    let band = ((value >> (depth.bits() - BAND_BITS)) - i32::from(position)) & 31;
    let offset = offsets.get(band as usize).copied().unwrap_or(0);
    depth.clip_to(value + i32::from(offset))
}

pub(super) fn band(
    plane: &mut [u8],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i8; 4],
    exempt: &[bool],
) {
    band_at(
        plane,
        stride,
        rect,
        position,
        offsets.map(i16::from),
        exempt,
        BitDepth::EIGHT,
    );
}

/// The band-offset kernel at any sample type and depth.
pub(crate) fn band_at<T: Sample>(
    plane: &mut [T],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i16; 4],
    exempt: &[bool],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let width = rect.x1 - rect.x0;
    for y in rect.y0..rect.y1 {
        for (x, sample) in plane[y * stride + rect.x0..y * stride + rect.x1]
            .iter_mut()
            .enumerate()
        {
            if exempt.is_empty() || !exempt[(y - rect.y0) * width + x] {
                *sample = band_sample_at(*sample, position, offsets, depth);
            }
        }
    }
}
