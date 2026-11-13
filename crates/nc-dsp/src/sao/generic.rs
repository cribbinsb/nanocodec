//! SAO edge-offset kernels written once over [`I16FilterLanes`]: `V::LANES`
//! samples per step, the rest of each row by the scalar code.
//!
//! The category of a lane is found without branches: `edge = clamp(c − a,
//! −1, 1) + clamp(c − b, −1, 1)` (−2..2), and category `k`'s mask is
//! `edge == value_k` (all bits set, or zero). Statistics add the masked
//! difference and subtract the mask (−1) per category in 16-bit lanes,
//! widened into 32-bit totals every [`FLUSH_STEPS`] steps, before they could
//! overflow.
//!
//! Each row's centre, neighbour and source samples are sliced once, so the
//! steps index only those row slices.

use super::scalar::{apply_row, stats_row};
use super::{Rect, SaoStats};
use crate::simd::I16FilterLanes;

/// `edgeIdx − 2` of each category index 0..3.
const EDGE_VALUES: [i16; 4] = [-2, -1, 1, 2];
/// Steps a 16-bit lane may accumulate: `|source − reconstruction| <= 255`,
/// and `128 * 255 < 2^15`.
const FLUSH_STEPS: usize = 128;

/// Lane-wise `edgeIdx − 2` of `c` against `a`, `b`.
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
fn edge<V: I16FilterLanes>(c: V, a: V, b: V) -> V {
    let (lo, hi) = (V::splat(-1), V::splat(1));
    c.sub(a).max(lo).min(hi).add(c.sub(b).max(lo).min(hi))
}

/// Row `y`'s samples `x0..x1` of `plane`, displaced by `offset` (index
/// units): one bounds-checked slice per row.
#[allow(clippy::inline_always)]
#[inline(always)]
fn row(plane: &[u8], stride: usize, y: usize, (x0, x1): (usize, usize), offset: isize) -> &[u8] {
    let start = (y * stride + x0)
        .checked_add_signed(offset)
        .expect("SAO neighbour inside the plane");
    &plane[start..start + (x1 - x0)]
}

/// The 16-bit running sums of one flush period.
struct Partial<V> {
    sums: [V; 4],
    counts: [V; 4],
    steps: usize,
}

#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn stats<V: I16FilterLanes>(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    (off_a, off_b): (isize, isize),
    rect: Rect,
) -> SaoStats {
    let zero = V::splat(0);
    let values = [
        V::splat(EDGE_VALUES[0]),
        V::splat(EDGE_VALUES[1]),
        V::splat(EDGE_VALUES[2]),
        V::splat(EDGE_VALUES[3]),
    ];
    let mut sums = [V::acc_zero(); 4];
    let mut counts = [V::acc_zero(); 4];
    let mut partial = Partial {
        sums: [zero; 4],
        counts: [zero; 4],
        steps: 0,
    };
    let mut out = SaoStats::default();
    let span = (rect.x0, rect.x1);
    for y in (rect.y0..rect.y1).step_by(rect.row_step) {
        let cur = row(rec, rec_stride, y, span, 0);
        let nb_a = row(rec, rec_stride, y, span, off_a);
        let nb_b = row(rec, rec_stride, y, span, off_b);
        let source = row(src, src_stride, y, span, 0);
        let mut x = 0;
        while x + V::LANES <= cur.len() {
            let c = V::load_u8(&cur[x..]);
            let class_edge = edge(c, V::load_u8(&nb_a[x..]), V::load_u8(&nb_b[x..]));
            let diff = V::load_u8(&source[x..]).sub(c);
            for (k, &value) in values.iter().enumerate() {
                let mask = class_edge.eq_mask(value);
                partial.sums[k] = partial.sums[k].add(diff.and(mask));
                partial.counts[k] = partial.counts[k].sub(mask);
            }
            partial.steps += 1;
            if partial.steps == FLUSH_STEPS {
                flush(&mut partial, &mut sums, &mut counts);
            }
            x += V::LANES;
        }
        stats_row(
            src,
            src_stride,
            rec,
            rec_stride,
            (off_a, off_b),
            y,
            (rect.x0 + x, rect.x1),
            &mut out,
        );
    }
    flush(&mut partial, &mut sums, &mut counts);
    for k in 0..4 {
        out.sum[k] += V::acc_sum(sums[k]);
        out.count[k] += V::acc_sum(counts[k]);
    }
    out
}

/// Widen `partial` into the 32-bit totals and restart it.
#[allow(clippy::inline_always)]
#[inline(always)]
fn flush<V: I16FilterLanes>(
    partial: &mut Partial<V>,
    sums: &mut [V::Acc; 4],
    counts: &mut [V::Acc; 4],
) {
    let zero = V::splat(0);
    for k in 0..4 {
        sums[k] = V::acc_add(sums[k], partial.sums[k]);
        counts[k] = V::acc_add(counts[k], partial.counts[k]);
        partial.sums[k] = zero;
        partial.counts[k] = zero;
    }
    partial.steps = 0;
}

#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn apply<V: I16FilterLanes>(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    (off_a, off_b): (isize, isize),
    offsets: [i8; 4],
    rect: Rect,
) {
    let values = [
        V::splat(EDGE_VALUES[0]),
        V::splat(EDGE_VALUES[1]),
        V::splat(EDGE_VALUES[2]),
        V::splat(EDGE_VALUES[3]),
    ];
    // No closure (`array::map`): closures do not inherit target features.
    let offset_vectors = [
        V::splat(i16::from(offsets[0])),
        V::splat(i16::from(offsets[1])),
        V::splat(i16::from(offsets[2])),
        V::splat(i16::from(offsets[3])),
    ];
    let span = (rect.x0, rect.x1);
    for y in rect.y0..rect.y1 {
        let cur = row(input, in_stride, y, span, 0);
        let nb_a = row(input, in_stride, y, span, off_a);
        let nb_b = row(input, in_stride, y, span, off_b);
        let start = y * out_stride + rect.x0;
        let dst = &mut out[start..start + cur.len()];
        let mut x = 0;
        while x + V::LANES <= cur.len() {
            let c = V::load_u8(&cur[x..]);
            let class_edge = edge(c, V::load_u8(&nb_a[x..]), V::load_u8(&nb_b[x..]));
            let mut value = c;
            for (&offset, &category) in offset_vectors.iter().zip(&values) {
                value = value.add(offset.and(class_edge.eq_mask(category)));
            }
            value.store_u8_sat(&mut dst[x..]);
            x += V::LANES;
        }
        apply_row(
            input,
            in_stride,
            out,
            out_stride,
            (off_a, off_b),
            offsets,
            y,
            (rect.x0 + x, rect.x1),
        );
    }
}
