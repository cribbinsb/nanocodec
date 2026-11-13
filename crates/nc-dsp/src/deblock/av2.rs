//! AV2 deblocking of one 4-line edge segment (AVM v1.0.0
//! `avm_highbd_lpf_{vertical,horizontal}_generic_c`, `filt_choice_highbd`
//! and `filt_generic_asym_highbd` in `avm_dsp/loopfilter.c`).
//!
//! AV2's filter is not AV1's: instead of fixed 4/6/8/14-tap filters with
//! `limit`/`blimit`/`thresh` masks, it chooses one modification width for
//! the whole segment from second derivatives of its first and last lines,
//! and then moves up to eight samples on each side by a linear ramp of one
//! clamped edge step. Both sides may get different widths (asymmetric
//! filters at superblock rows). The codec (nc-av2 `deblock`) decides which
//! edges to filter, their maximum lengths and thresholds.
//!
//! Scalar reference, 8-bit samples. Reimplemented from AVM (BSD-3-Clause-
//! Clear, see `THIRD_PARTY.md`).

/// Lines one call filters (one 4×4 unit's edge).
pub const AV2_EDGE_LINES: usize = 4;
/// Most samples the filter modifies on one side (`MAX_DBL_FLT_LEN`).
pub(super) const MAX_WIDTH: usize = 8;
/// `DF_SHIFT`: fixed-point bits of the ramp weights.
const WEIGHT_SHIFT: u32 = 8;
/// The ramp's extra three bits: the edge step is scaled by four and by the
/// ramp position, normalised together.
const RAMP_SHIFT: u32 = 3 + WEIGHT_SHIFT;
/// `w_mult`: ramp weight per modification width (index width − 1).
const RAMP_WEIGHT: [i32; MAX_WIDTH] = [85, 51, 37, 28, 23, 20, 17, 15];
/// A ramp offset `(step · weight · f + 2^10) >> RAMP_SHIFT` equals the
/// rounding Q15 multiply `(step · (weight · f · 2^4) + 2^14) >> 15`: with
/// |step| <= 4080 (`4 · (3 · 255 + 255)`) and `weight(w) · w <= 120`, both
/// factors fit `i16` (the scaled one at most 1920), so 16-bit lanes compute
/// every offset exactly.
pub(crate) const AV2_Q15_RAMP_SCALE: i32 = 1 << (15 - RAMP_SHIFT);

/// The Q15 factor of a ramp of width `width` (0..=8, 0: no ramp) at
/// `distance` samples from the edge (0 at p0/q0): `weight(width) · (width
/// − distance) ·` [`AV2_Q15_RAMP_SCALE`], 0 beyond the ramp.
const fn q15_ramp_factor(width: usize, distance: usize) -> i16 {
    if distance >= width {
        return 0;
    }
    (RAMP_WEIGHT[width - 1] * (width - distance) as i32 * AV2_Q15_RAMP_SCALE) as i16
}

/// [`q15_ramp_factor`] by width (0..=8) and distance from the edge: a
/// segment's factors for one side are one table row (one 8-lane vector),
/// looked up rather than computed position by position.
pub(crate) const Q15_RAMP_FACTORS: [[i16; MAX_WIDTH]; MAX_WIDTH + 1] = {
    let mut table = [[0; MAX_WIDTH]; MAX_WIDTH + 1];
    let mut width = 0;
    while width <= MAX_WIDTH {
        let mut distance = 0;
        while distance < MAX_WIDTH {
            table[width][distance] = q15_ramp_factor(width, distance);
            distance += 1;
        }
        width += 1;
    }
    table
};

/// [`Q15_RAMP_FACTORS`] with each row reversed: position `k` holds the
/// factor at distance `7 − k`, for a p-side window that ends at p0.
pub(crate) const Q15_RAMP_FACTORS_REVERSED: [[i16; MAX_WIDTH]; MAX_WIDTH + 1] = {
    let mut table = [[0; MAX_WIDTH]; MAX_WIDTH + 1];
    let mut width = 0;
    while width <= MAX_WIDTH {
        let mut k = 0;
        while k < MAX_WIDTH {
            table[width][k] = Q15_RAMP_FACTORS[width][MAX_WIDTH - 1 - k];
            k += 1;
        }
        width += 1;
    }
    table
};

/// [`q15_ramp_factor`] of one width per lane is `weight · max(width − i, 0)`
/// scaled: the per-width multiplier `weight(width) · AV2_Q15_RAMP_SCALE`
/// by width (0..=8; 0 for no ramp). Equal to the factor at distance
/// `width − 1`.
pub(crate) const Q15_RAMP_WEIGHTS: [i16; MAX_WIDTH + 1] = {
    let mut table = [0; MAX_WIDTH + 1];
    let mut width = 1;
    while width <= MAX_WIDTH {
        table[width] = q15_ramp_factor(width, width - 1);
        width += 1;
    }
    table
};
/// `q_thresh_mults`: clamp of the edge step per width, in units of the
/// quantizer threshold.
pub(super) const STEP_CLAMP: [i32; MAX_WIDTH] = [32, 25, 19, 19, 18, 18, 17, 17];
/// `DF_6_THRESH` / `DF_8_THRESH`: largest transition (the two second
/// derivatives at the edge) for two- and three-sample modification, in
/// quantizer thresholds.
pub(super) const TWO_SAMPLE_TRANSITION: u32 = 4;
pub(super) const THREE_SAMPLE_TRANSITION: u32 = 3;
/// Side-smoothness shifts of the two- and three-sample tests (`>> 2`,
/// `FILT_8_THRESH_SHIFT`).
pub(super) const TWO_SAMPLE_SIDE_SHIFT: u32 = 2;
pub(super) const THREE_SAMPLE_SIDE_SHIFT: u32 = 3;
/// `DF_Q_THRESH_SHIFT`: fixed-point bits of the wide tests' transition.
pub(super) const WIDE_TRANSITION_SHIFT: u32 = 4;
/// End-of-ramp straightness thresholds are `side · distance >> 4`.
pub(super) const END_SHIFT: u32 = 4;
/// Distance of the three-sample end test.
pub(super) const THREE_SAMPLE_DISTANCE: usize = 3;

/// One of the 4-, 6- and 8-sample tests of [`choose_width`].
pub(super) struct WideTest {
    /// Width chosen when the test passes and the q side allows no more.
    pub distance: usize,
    /// Distance of the straightness test (the 8-sample test checks 7).
    pub tested: usize,
    /// Transition limit (`q_first`), in quantizer thresholds scaled by
    /// [`WIDE_TRANSITION_SHIFT`].
    pub limit: u32,
    /// Width chosen when the test fails (the widest test passed so far).
    pub fallback: usize,
}

/// The wide tests in [`choose_width`]'s order.
pub(super) const WIDE_TESTS: [WideTest; 3] = [
    WideTest {
        distance: 4,
        tested: 4,
        limit: 45,
        fallback: 3,
    },
    WideTest {
        distance: 6,
        tested: 6,
        limit: 40,
        fallback: 4,
    },
    WideTest {
        distance: 8,
        tested: 7,
        limit: 32,
        fallback: 6,
    },
];

/// One edge segment's filter, as the codec derived it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av2EdgeFilter {
    /// `filter_length_neg`: maximum filter length on the p side (4, 6, 8,
    /// 10, 14 or 18); the side modifies at most `length / 2 − 1` samples.
    pub length_neg: u8,
    /// `filter_length_pos`: the same on the q side, never 6 (only p sides
    /// are shortened; [`Av2EdgeFilter::reach`] assumes it).
    pub length_pos: u8,
    /// Quantizer threshold (`df_quant_from_qindex`).
    pub q_threshold: u16,
    /// Side threshold (`df_side_from_qindex`).
    pub side_threshold: u16,
}

impl Av2EdgeFilter {
    /// Samples the filter may read or write on each side: `(p side, q
    /// side)`. The curvature tests always read three; the straightness
    /// tests read one beyond the modified samples, up to the 8-sample cap.
    #[must_use]
    pub fn reach(self) -> (usize, usize) {
        let reach = |length: u8| (max_samples(length) + 1).clamp(MIN_REACH, MAX_WIDTH);
        (reach(self.length_neg), reach(self.length_pos))
    }
}

/// Samples every decision reads on each side (the outer curvature tests).
const MIN_REACH: usize = 3;

/// Most samples a side of maximum filter length `length` may modify.
pub(super) fn max_samples(length: u8) -> usize {
    usize::from(length / 2).saturating_sub(1)
}

/// What the decision and the step clamp read of an [`Av2EdgeFilter`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentLimits {
    /// Quantizer threshold.
    pub q: u32,
    /// Side threshold.
    pub side: u32,
    /// Most samples the p side may modify ([`max_samples`]).
    pub max_neg: usize,
    /// Most samples the q side may modify.
    pub max_pos: usize,
}

impl From<Av2EdgeFilter> for SegmentLimits {
    fn from(filter: Av2EdgeFilter) -> Self {
        Self {
            q: u32::from(filter.q_threshold),
            side: u32::from(filter.side_threshold),
            max_neg: max_samples(filter.length_neg),
            max_pos: max_samples(filter.length_pos),
        }
    }
}

/// Samples across one edge of one line: `at(i)` is `q[i]` for `i ≥ 0` and
/// `p[−i−1]` for `i < 0`.
struct Line<'a> {
    plane: &'a [u8],
    q0: usize,
    across: usize,
    /// Samples p2..q2 (`at(-3)..=at(2)`), read once: the curvature tests
    /// and the edge step use only these, several times each.
    core: [i32; CORE_SAMPLES],
}

/// First and last offsets of [`Line::core`], and its length. Every
/// decision reads at least [`MIN_REACH`] samples on each side, so they are
/// always inside the plane.
const CORE_FIRST: isize = -3;
const CORE_LAST: isize = 2;
const CORE_SAMPLES: usize = (CORE_LAST - CORE_FIRST + 1) as usize;

impl<'a> Line<'a> {
    fn new(plane: &'a [u8], q0: usize, across: usize) -> Self {
        let read = |i: isize| i32::from(plane[q0.wrapping_add_signed(i * across as isize)]);
        Self {
            plane,
            q0,
            across,
            core: core::array::from_fn(|k| read(CORE_FIRST + k as isize)),
        }
    }

    fn at(&self, i: isize) -> i32 {
        if (CORE_FIRST..=CORE_LAST).contains(&i) {
            self.core[(i - CORE_FIRST) as usize]
        } else {
            i32::from(self.plane[self.q0.wrapping_add_signed(i * self.across as isize)])
        }
    }

    /// `|s[a] − 2·s[b] + s[c]|` for consecutive positions.
    fn second_derivative(&self, first: isize) -> u32 {
        (self.at(first) - 2 * self.at(first + 1) + self.at(first + 2)).unsigned_abs()
    }

    /// Deviation from a straight line out to `distance` samples on the p
    /// side (`negative`) or the q side.
    fn bend(&self, distance: usize, negative: bool) -> u32 {
        let d = distance as isize;
        let gap = d as i32;
        if negative {
            ((self.at(-1) - self.at(-d - 1)) - gap * (self.at(-1) - self.at(-2))).unsigned_abs()
        } else {
            ((self.at(0) - self.at(d)) - gap * (self.at(0) - self.at(1))).unsigned_abs()
        }
    }
}

/// Mean of a measurement on the segment's first and last lines, rounded up.
fn both(first: &Line<'_>, last: &Line<'_>, measure: impl Fn(&Line<'_>) -> u32) -> u32 {
    (measure(first) + measure(last) + 1) >> 1
}

/// `filt_choice_highbd`: how many samples to modify on the q side, from the
/// segment's first and last lines (0 = leave the segment alone).
fn choose_width(first: &Line<'_>, last: &Line<'_>, limits: SegmentLimits) -> usize {
    let SegmentLimits {
        q,
        side,
        max_neg,
        max_pos,
    } = limits;
    if q == 0 || side == 0 {
        return 0;
    }
    if max_pos < 1 || max_pos < max_neg {
        return 0;
    }
    // Curvature just outside the edge on each side, and at the edge.
    let outer_p = both(first, last, |l| l.second_derivative(-3));
    let outer_q = both(first, last, |l| l.second_derivative(0));
    if outer_p > side || outer_q > side {
        return 0;
    }
    if max_pos == 1 {
        return 1;
    }
    let side2 = side >> TWO_SAMPLE_SIDE_SHIFT;
    let transition = both(first, last, |l| l.second_derivative(-2))
        + both(first, last, |l| l.second_derivative(-1));
    if outer_p > side2 || outer_q > side2 || transition > q * TWO_SAMPLE_TRANSITION {
        return 1;
    }
    let side3 = side >> THREE_SAMPLE_SIDE_SHIFT;
    let straight = |distance: usize| {
        let end = (side * distance as u32) >> END_SHIFT;
        (max_neg < distance || both(first, last, |l| l.bend(distance, true)) <= end)
            && both(first, last, |l| l.bend(distance, false)) <= end
    };
    if outer_p > side3
        || outer_q > side3
        || transition > q * THREE_SAMPLE_TRANSITION
        || !straight(THREE_SAMPLE_DISTANCE)
    {
        return 2;
    }
    if max_pos == THREE_SAMPLE_DISTANCE {
        return 3;
    }
    let transition = transition << WIDE_TRANSITION_SHIFT;
    for test in &WIDE_TESTS {
        let end = (side * test.distance as u32) >> END_SHIFT;
        let bent = (max_neg >= test.tested
            && both(first, last, |l| l.bend(test.tested, true)) > end)
            || both(first, last, |l| l.bend(test.tested, false)) > end;
        if transition > q * test.limit || bent {
            return test.fallback;
        }
        if max_pos <= test.distance {
            return test.distance;
        }
    }
    MAX_WIDTH
}

/// The ramps of one edge segment, ready to apply
/// (`filt_generic_asym_highbd` once the width is chosen): every line moves
/// `widths.0` p samples by `(scaled.0 · (widths.0 − i) + round) >> 11` and
/// `widths.1` q samples by minus `(scaled.1 · (widths.1 − i) + round) >> 11`,
/// `i` counting away from the edge, each clipped to 8 bits.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Av2Ramps {
    /// Index of the first line's q0 sample.
    pub q0: usize,
    /// Index step from p0 to q0 (1 for a vertical edge, the stride for a
    /// horizontal one).
    pub across: usize,
    /// Index step from one line to the next.
    pub along: usize,
    /// Samples moved on the p side and on the q side (both at least one).
    pub widths: (usize, usize),
    /// Per line, the clamped edge step times each side's ramp weight.
    pub scaled: [(i32, i32); AV2_EDGE_LINES],
}

/// Applies [`Av2Ramps`] to a plane.
pub(crate) type Av2RampFn = unsafe fn(plane: &mut [u8], ramps: &Av2Ramps);

/// Fixed-point bits of a ramp offset (see [`Av2Ramps`]).
pub(crate) const AV2_RAMP_SHIFT: u32 = RAMP_SHIFT;
/// Most samples a ramp moves on one side.
pub(crate) const AV2_MAX_RAMP: usize = MAX_WIDTH;

/// Scale of the edge step (the `· 4` of `step = (3 · Δ0 − Δ1) · 4`).
pub(crate) const EDGE_STEP_SCALE: i32 = 4;
/// Segments one [`Av2ChunkFn`] call filters.
pub(crate) const AV2_DECIDE_SEGMENTS: usize = 8;
/// Samples a chunk's decisions read on each side of the edge, whatever the
/// segments' filter lengths (p7..q7 at most, see [`Av2EdgeFilter::reach`]).
pub(crate) const AV2_DECIDE_REACH: usize = MAX_WIDTH;
/// Largest quantizer threshold a chunk kernel takes: its 16-bit lanes hold
/// `q · 45` (the 4-sample test's transition limit). AV2's thresholds stay
/// far below (at most 120); larger ones go segment by segment.
pub(crate) const AV2_CHUNK_MAX_Q: u16 = 727;
/// Largest side threshold a chunk kernel takes (16-bit lanes hold twice the
/// sum of two line measures compared with it; AV2's are at most 52).
pub(crate) const AV2_CHUNK_MAX_SIDE: u16 = 4095;

/// [`AV2_DECIDE_SEGMENTS`] consecutive segments of one edge line (one
/// vertical edge column or horizontal edge row): segment `s` has its first
/// line's q0 at `q0 + s * AV2_EDGE_LINES * along`. Segments of one edge
/// line never share a sample, so they can be filtered together.
///
/// The filters are stored by field (one value per segment and lane, as
/// [`SegmentLimits`]; all zero for a segment left alone), so the kernels
/// load each as one vector.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Av2Segments {
    /// Index of the first segment's first q0 sample.
    pub q0: usize,
    /// Index step from p0 to q0 (1 for a vertical edge, the stride for a
    /// horizontal one).
    pub across: usize,
    /// Index step from one line to the next.
    pub along: usize,
    /// Quantizer thresholds (at most [`AV2_CHUNK_MAX_Q`]).
    pub q: [i16; AV2_DECIDE_SEGMENTS],
    /// Side thresholds (at most [`AV2_CHUNK_MAX_SIDE`]).
    pub side: [i16; AV2_DECIDE_SEGMENTS],
    /// Most samples each p side may modify ([`max_samples`]).
    pub max_neg: [i16; AV2_DECIDE_SEGMENTS],
    /// Most samples each q side may modify.
    pub max_pos: [i16; AV2_DECIDE_SEGMENTS],
}

impl Av2Segments {
    /// The chunk of `filters` (`None`: left alone) at `q0`, or `None` if a
    /// threshold exceeds the kernels' range.
    pub(crate) fn new(
        q0: usize,
        across: usize,
        along: usize,
        filters: &[Option<Av2EdgeFilter>; AV2_DECIDE_SEGMENTS],
    ) -> Option<Self> {
        let mut segments = Self {
            q0,
            across,
            along,
            q: [0; AV2_DECIDE_SEGMENTS],
            side: [0; AV2_DECIDE_SEGMENTS],
            max_neg: [0; AV2_DECIDE_SEGMENTS],
            max_pos: [0; AV2_DECIDE_SEGMENTS],
        };
        for (s, filter) in filters.iter().enumerate() {
            let Some(filter) = filter else { continue };
            if filter.q_threshold > AV2_CHUNK_MAX_Q || filter.side_threshold > AV2_CHUNK_MAX_SIDE {
                return None;
            }
            // In range: thresholds checked above, lengths are bytes.
            segments.q[s] = filter.q_threshold as i16;
            segments.side[s] = filter.side_threshold as i16;
            segments.max_neg[s] = max_samples(filter.length_neg) as i16;
            segments.max_pos[s] = max_samples(filter.length_pos) as i16;
        }
        Some(segments)
    }

    /// The limits of segment `s`.
    fn limits(&self, s: usize) -> SegmentLimits {
        // Every field is non-negative (see `new`).
        SegmentLimits {
            q: self.q[s] as u32,
            side: self.side[s] as u32,
            max_neg: self.max_neg[s] as usize,
            max_pos: self.max_pos[s] as usize,
        }
    }
}

/// Decides and filters every segment of an [`Av2Segments`], equal to
/// [`segment_ramps`] and [`apply_ramps`] per segment. The caller guarantees
/// that [`AV2_DECIDE_REACH`] samples on each side of every line lie inside
/// the plane.
pub(crate) type Av2ChunkFn = unsafe fn(plane: &mut [u8], segments: &Av2Segments);

/// Scalar reference of [`Av2ChunkFn`].
pub(crate) fn filter_chunk(plane: &mut [u8], segments: &Av2Segments) {
    for s in 0..AV2_DECIDE_SEGMENTS {
        let q0 = segments.q0 + s * AV2_EDGE_LINES * segments.along;
        if let Some(ramps) = limited_ramps(
            plane,
            q0,
            segments.across,
            segments.along,
            segments.limits(s),
        ) {
            apply_ramps(plane, &ramps);
        }
    }
}

/// Decide the [`AV2_EDGE_LINES`] lines across one edge of `plane` (`q0`:
/// the first line's q0 index, `across`: the step from p0 to q0, `along`:
/// from one line to the next) and return their ramps, or `None` when the
/// segment is left alone. The caller guarantees that
/// [`Av2EdgeFilter::reach`] samples on each side lie inside the plane.
pub(crate) fn segment_ramps(
    plane: &[u8],
    q0: usize,
    across: usize,
    along: usize,
    filter: Av2EdgeFilter,
) -> Option<Av2Ramps> {
    limited_ramps(plane, q0, across, along, filter.into())
}

/// [`segment_ramps`] of a filter given by its limits.
fn limited_ramps(
    plane: &[u8],
    q0: usize,
    across: usize,
    along: usize,
    limits: SegmentLimits,
) -> Option<Av2Ramps> {
    let line_at = |line: usize| Line::new(plane, q0 + line * along, across);
    let width_pos = choose_width(&line_at(0), &line_at(AV2_EDGE_LINES - 1), limits);
    let (widths, steps) = clamped_steps(limits, width_pos, edge_steps(plane, q0, across, along))?;
    Some(Av2Ramps {
        q0,
        across,
        along,
        widths,
        scaled: scaled_steps(widths, steps),
    })
}

/// Per line of the segment at `q0`, the edge step `4 · (3 · (q0 − p0) −
/// (q1 − p1))` before its clamp (it reads only p1..q1).
fn edge_steps(plane: &[u8], q0: usize, across: usize, along: usize) -> [i32; AV2_EDGE_LINES] {
    core::array::from_fn(|line| {
        let line_q0 = q0 + line * along;
        let at = |i: isize| i32::from(plane[line_q0.wrapping_add_signed(i * across as isize)]);
        (3 * (at(0) - at(-1)) - (at(1) - at(-2))) * EDGE_STEP_SCALE
    })
}

/// A segment's ramp widths `(p side, q side)` and clamped edge steps, or
/// `None` when it is left alone: the p side moves at most `max_neg`
/// samples, and a segment with no sample to move on either side is not
/// filtered.
fn clamped_steps(
    limits: SegmentLimits,
    width_pos: usize,
    steps: [i32; AV2_EDGE_LINES],
) -> Option<((usize, usize), [i16; AV2_EDGE_LINES])> {
    let width_neg = width_pos.min(limits.max_neg);
    if width_neg < 1 || width_pos < 1 {
        return None;
    }
    // width_neg <= width_pos, so the larger width is the q side's.
    let clamp = limits.q as i32 * STEP_CLAMP[width_pos - 1];
    // The unclamped step is within ±4080, so the clamped one fits i16.
    let steps = steps.map(|step| step.clamp(-clamp, clamp) as i16);
    Some(((width_neg, width_pos), steps))
}

/// Per line, a clamped step times each side's ramp weight.
fn scaled_steps(
    (width_neg, width_pos): (usize, usize),
    steps: [i16; AV2_EDGE_LINES],
) -> [(i32, i32); AV2_EDGE_LINES] {
    steps.map(|step| {
        let step = i32::from(step);
        (
            step * RAMP_WEIGHT[width_neg - 1],
            step * RAMP_WEIGHT[width_pos - 1],
        )
    })
}

/// Scalar reference of [`Av2RampFn`].
pub(crate) fn apply_ramps(plane: &mut [u8], ramps: &Av2Ramps) {
    for (line, &(scaled_neg, scaled_pos)) in ramps.scaled.iter().enumerate() {
        ramp_line(
            plane,
            ramps.q0 + line * ramps.along,
            ramps.across,
            ramps.widths,
            (scaled_neg, scaled_pos),
        );
    }
}

/// One line of [`apply_ramps`].
pub(crate) fn ramp_line(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    (width_neg, width_pos): (usize, usize),
    (scaled_neg, scaled_pos): (i32, i32),
) {
    let round = 1 << (RAMP_SHIFT - 1);
    for i in 0..width_neg {
        let index = q0 - (i + 1) * across;
        let offset = (scaled_neg * (width_neg - i) as i32 + round) >> RAMP_SHIFT;
        plane[index] = (i32::from(plane[index]) + offset).clamp(0, 255) as u8;
    }
    for i in 0..width_pos {
        let index = q0 + i * across;
        let offset = (scaled_pos * (width_pos - i) as i32 + round) >> RAMP_SHIFT;
        plane[index] = (i32::from(plane[index]) - offset).clamp(0, 255) as u8;
    }
}

/// Filter the [`AV2_EDGE_LINES`] lines across one edge of `plane` with the
/// scalar ramps (arguments as [`segment_ramps`]).
#[cfg(test)]
fn filter_edge(plane: &mut [u8], q0: usize, across: usize, along: usize, filter: Av2EdgeFilter) {
    if let Some(ramps) = segment_ramps(plane, q0, across, along, filter) {
        apply_ramps(plane, &ramps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(length: u8, q: u16, side: u16) -> Av2EdgeFilter {
        Av2EdgeFilter {
            length_neg: length,
            length_pos: length,
            q_threshold: q,
            side_threshold: side,
        }
    }

    /// Four identical lines of `row` across a vertical edge at index 9.
    fn lines(row: [u8; 18]) -> Vec<u8> {
        row.iter()
            .copied()
            .cycle()
            .take(18 * AV2_EDGE_LINES)
            .collect()
    }

    #[test]
    fn small_step_on_flat_sides_is_ramped_symmetrically() {
        let mut row = [100; 18];
        row[9..].fill(108);
        let mut plane = lines(row);
        filter_edge(&mut plane, 9, 1, 18, filter(18, 8, 400));
        let out = &plane[..18];
        assert!(out[8] > 100 && out[9] < 108, "{out:?}");
        // The ramp is antisymmetric about the edge and monotone.
        for i in 0..8 {
            assert_eq!(i32::from(out[8 - i]) - 100, 108 - i32::from(out[9 + i]));
            assert!(out[i] <= out[i + 1]);
        }
        // Every line got the same treatment.
        for line in 1..AV2_EDGE_LINES {
            assert_eq!(&plane[line * 18..line * 18 + 18], out);
        }
    }

    #[test]
    fn zero_thresholds_or_textured_sides_leave_the_edge() {
        let mut row = [100; 18];
        row[9..].fill(108);
        for f in [filter(8, 0, 400), filter(8, 8, 0)] {
            let mut plane = lines(row);
            filter_edge(&mut plane, 9, 1, 18, f);
            assert_eq!(plane, lines(row));
        }
        // Strong curvature next to the edge exceeds the side threshold.
        row[6] = 160;
        let mut plane = lines(row);
        filter_edge(&mut plane, 9, 1, 18, filter(8, 8, 20));
        assert_eq!(plane, lines(row));
    }

    #[test]
    fn widths_follow_the_maximum_lengths() {
        let mut row = [100; 18];
        row[9..].fill(106);
        for (length, width) in [(4, 1), (8, 3), (10, 4), (14, 6), (18, 8)] {
            let mut plane = lines(row);
            filter_edge(&mut plane, 9, 1, 18, filter(length, 8, 400));
            let changed = (0..18).filter(|&i| plane[i] != row[i]).count();
            assert!(changed <= 2 * width, "length {length}: {:?}", &plane[..18]);
            assert_eq!(plane[9 + width], 106, "length {length}");
        }
        // Asymmetric chroma lengths at superblock rows: at most two p samples.
        let mut plane = lines(row);
        let f = Av2EdgeFilter {
            length_neg: 6,
            length_pos: 10,
            ..filter(10, 8, 400)
        };
        filter_edge(&mut plane, 9, 1, 18, f);
        assert_eq!(&plane[..7], &row[..7]);
    }

    #[test]
    fn horizontal_edges_match_vertical_edges() {
        let mut state = 0x2468_ace1_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 28) as u8
        };
        for length in [4, 8, 10, 14, 18] {
            let rows: Vec<[u8; 18]> = (0..AV2_EDGE_LINES)
                .map(|_| std::array::from_fn(|i| 90 + next() + if i >= 9 { 12 } else { 0 }))
                .collect();
            let mut vertical: Vec<u8> = rows.iter().flatten().copied().collect();
            filter_edge(&mut vertical, 9, 1, 18, filter(length, 6, 300));
            let mut horizontal = vec![0; 18 * AV2_EDGE_LINES];
            for (l, row) in rows.iter().enumerate() {
                for (i, &v) in row.iter().enumerate() {
                    horizontal[i * AV2_EDGE_LINES + l] = v;
                }
            }
            filter_edge(
                &mut horizontal,
                9 * AV2_EDGE_LINES,
                AV2_EDGE_LINES,
                1,
                filter(length, 6, 300),
            );
            for (l, row) in vertical.chunks(18).enumerate() {
                for (i, &v) in row.iter().enumerate() {
                    assert_eq!(horizontal[i * AV2_EDGE_LINES + l], v);
                }
            }
        }
    }
}
