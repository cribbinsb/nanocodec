use crate::simd::I16FilterLanes;

/// Four final butterflies reduced to half the absolute sum, one value per
/// lane. The packed graph follows rav1e's `HADAMARD_8x8` in `satd.asm`
/// (BSD-2-Clause, see `THIRD_PARTY.md`).
#[allow(
    clippy::inline_always,
    reason = "keep the transform and its transpose in SIMD registers"
)]
#[inline(always)]
pub(super) fn satd_half<V: I16FilterLanes>(mut rows: [V; 8]) -> V {
    stage::<1, V>(&mut rows);
    stage::<2, V>(&mut rows);
    stage::<4, V>(&mut rows);
    let mut columns = V::transpose_groups8(rows);
    stage::<1, V>(&mut columns);
    stage::<2, V>(&mut columns);
    let half = |a: V, b: V| a.abs().max(b.abs());
    let a = half(columns[0], columns[4]).add(half(columns[1], columns[5]));
    let b = half(columns[2], columns[6]).add(half(columns[3], columns[7]));
    // Four sums of at most 8 * 4 * 255 fit signed words.
    a.add(b)
}

#[allow(
    clippy::inline_always,
    reason = "eight rows fold to fixed SIMD butterflies"
)]
#[inline(always)]
fn stage<const HALF: usize, V: I16FilterLanes>(rows: &mut [V; 8]) {
    for start in (0..8).step_by(2 * HALF) {
        for i in start..start + HALF {
            let (a, b) = (rows[i], rows[i + HALF]);
            rows[i] = a.add(b);
            rows[i + HALF] = a.sub(b);
        }
    }
}
