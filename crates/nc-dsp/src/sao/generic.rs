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

use super::scalar::stats_row;
use super::{Rect, SaoStats};
use crate::simd::I16FilterLanes;
use nc_common::{BitDepth, Sample};

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
fn row<T>(plane: &[T], stride: usize, y: usize, (x0, x1): (usize, usize), offset: isize) -> &[T] {
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

/// Sample adapters keep the edge-classification graph shared at both depths.
pub(crate) trait SaoSample: Sample {
    unsafe fn load<V: I16FilterLanes>(row: &[Self]) -> V;
    fn offset_add<V: I16FilterLanes>(value: V, offset: V) -> V;
    unsafe fn store<V: I16FilterLanes>(value: V, zero: V, maximum: V, row: &mut [Self]);
}

#[allow(clippy::inline_always)]
impl SaoSample for u8 {
    #[inline(always)]
    unsafe fn load<V: I16FilterLanes>(row: &[Self]) -> V {
        V::load_u8(row)
    }
    #[inline(always)]
    fn offset_add<V: I16FilterLanes>(value: V, offset: V) -> V {
        value.add(offset)
    }
    #[inline(always)]
    unsafe fn store<V: I16FilterLanes>(value: V, _zero: V, _maximum: V, row: &mut [Self]) {
        value.store_u8_sat(row);
    }
}

#[allow(clippy::inline_always)]
impl SaoSample for u16 {
    #[inline(always)]
    unsafe fn load<V: I16FilterLanes>(row: &[Self]) -> V {
        // SAFETY: the caller supplies a full vector of words, bounded to
        // 15-bit samples. Reinterpreting u16 words as i16 preserves their bits.
        V::load(unsafe { core::slice::from_raw_parts(row.as_ptr().cast(), V::LANES) })
    }
    #[inline(always)]
    fn offset_add<V: I16FilterLanes>(value: V, offset: V) -> V {
        value.add_sat(offset)
    }
    #[inline(always)]
    unsafe fn store<V: I16FilterLanes>(value: V, zero: V, maximum: V, row: &mut [Self]) {
        // SAFETY: a full vector fits the output; clipping bounds every stored sample.
        value
            .max(zero)
            .min(maximum)
            .store(unsafe { core::slice::from_raw_parts_mut(row.as_mut_ptr().cast(), V::LANES) });
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn apply<V: I16FilterLanes>(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i8; 4],
    rect: Rect,
) {
    // SAFETY: inherited complete-vector, extent and ISA contracts.
    unsafe {
        apply_at::<V, u8>(
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
}

/// The same edge-offset graph at 8..15 bits. Larger depths use the scalar
/// reference: subtracting two 15-bit samples is bounded by +/-32767. The
/// saturating offset add preserves Clip1 even for an extreme i16 offset.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn apply_at<V: I16FilterLanes, T: SaoSample>(
    input: &[T],
    in_stride: usize,
    out: &mut [T],
    out_stride: usize,
    (off_a, off_b): (isize, isize),
    offsets: [i16; 4],
    rect: Rect,
    depth: BitDepth,
) {
    if depth.bits() >= i16::BITS {
        super::scalar::apply_at(
            input,
            in_stride,
            out,
            out_stride,
            (off_a, off_b),
            offsets,
            rect,
            depth,
        );
        return;
    }
    let values = [
        V::splat(EDGE_VALUES[0]),
        V::splat(EDGE_VALUES[1]),
        V::splat(EDGE_VALUES[2]),
        V::splat(EDGE_VALUES[3]),
    ];
    let offset_vectors = [
        V::splat(offsets[0]),
        V::splat(offsets[1]),
        V::splat(offsets[2]),
        V::splat(offsets[3]),
    ];
    let zero = V::splat(0);
    let maximum = V::splat(depth.max() as i16);
    for y in rect.y0..rect.y1 {
        let span = (rect.x0, rect.x1);
        let current = row(input, in_stride, y, span, 0);
        let a = row(input, in_stride, y, span, off_a);
        let b = row(input, in_stride, y, span, off_b);
        let start = y * out_stride + rect.x0;
        let dst = &mut out[start..start + current.len()];
        let mut x = 0;
        while x + V::LANES <= current.len() {
            // SAFETY: each row contains this complete vector, and the wrapper enables its ISA.
            let centre = unsafe { T::load::<V>(&current[x..]) };
            // SAFETY: neighbours have exactly the same validated row extent.
            let (left, right) = unsafe { (T::load::<V>(&a[x..]), T::load::<V>(&b[x..])) };
            let class = edge(centre, left, right);
            let mut value = centre;
            for (&offset, &category) in offset_vectors.iter().zip(&values) {
                value = T::offset_add(value, offset.and(class.eq_mask(category)));
            }
            // SAFETY: output covers the vector; clipping is to 0..depth.max.
            unsafe { T::store::<V>(value, zero, maximum, &mut dst[x..]) };
            x += V::LANES;
        }
        super::scalar::apply_row_at(
            input,
            in_stride,
            out,
            out_stride,
            (off_a, off_b),
            offsets,
            y,
            (rect.x0 + x, rect.x1),
            depth,
        );
    }
}
