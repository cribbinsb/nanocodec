//! AV2 deblocking of one chunk of edge segments ([`Av2ChunkFn`]) on packed
//! lanes (`lpf_bytes`'s [`QuadBytes`] and [`MaskWords`]): one lane per
//! segment, all [`AV2_DECIDE_SEGMENTS`] segments of the chunk in one
//! vector. The scalar `choose_width` cascade, the clamped edge steps and
//! the ramps of `deblock::av2` run for every segment at once.
//!
//! Gathering: the decision reads only each segment's first and last
//! lines, the steps and the ramps all four. Samples are loaded straight
//! into position vectors:
//!
//! - Horizontal edge: a row's 32 bytes are the chunk's 32 lines, four per
//!   segment, so one de-interleaving load ([`QuadBytes::load_lines4`],
//!   NEON `ld4`) gives every line's vector at that position, and the ramps
//!   store a moved row back the same way.
//! - Vertical edge: a line is a row. The rows of one line of every segment
//!   transpose into position vectors: p3..q3 from one 8-byte load per row,
//!   p7..p4 with q4..q7 (read only by the 4- to 8-sample tests) from two
//!   4-byte loads. The ramps move each line's eight samples per side as
//!   one vector, with the factors of its segment's widths.
//!
//! The cascade: each early return of the scalar code is a "settle" step,
//! in which the lanes whose condition holds and that are still open take
//! that step's width and close. When every lane is closed the kernel stops
//! deciding, as the scalar code does for one segment.
//!
//! Lane ranges (16-bit): a line measure is at most 2295 (a bend at
//! distance 8), the transition at most 2040 and 32640 shifted; `q · 45` is
//! at most 32715 with `q <=` [`AV2_CHUNK_MAX_Q`]; steps lie within ±4080
//! (`deblock::av2`) and their clamps at `32 q`.
//!
//! The functions are `unsafe` only because `B`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]

#[cfg(doc)]
use super::av2::{AV2_CHUNK_MAX_Q, Av2ChunkFn};
use super::av2::{
    AV2_DECIDE_REACH, AV2_DECIDE_SEGMENTS, AV2_EDGE_LINES, AV2_MAX_RAMP, Av2Segments,
    EDGE_STEP_SCALE, END_SHIFT, MAX_WIDTH, Q15_RAMP_FACTORS, Q15_RAMP_FACTORS_REVERSED,
    Q15_RAMP_WEIGHTS, STEP_CLAMP, THREE_SAMPLE_DISTANCE, THREE_SAMPLE_SIDE_SHIFT,
    THREE_SAMPLE_TRANSITION, TWO_SAMPLE_SIDE_SHIFT, TWO_SAMPLE_TRANSITION, WIDE_TESTS,
    WIDE_TRANSITION_SHIFT,
};
use crate::lpf_bytes::{Bytes, LANES, MaskWords, QuadBytes, WordTable, Words, transpose};

const _: () = assert!(LANES == AV2_DECIDE_SEGMENTS && AV2_MAX_RAMP == LANES);

/// Positions one gather returns (one byte per row on a vertical edge).
const OCTET: usize = LANES;
/// First positions (relative to q0) of the inner octet, p3..q3: every test
/// except the 4- to 8-sample ones reads only these (curvatures p2..q2,
/// straightness out to p3 and q3). As `[first four, last four]`.
const INNER: [isize; 2] = [-4, 0];
/// The outer octet: p7..p4, then q4..q7.
const OUTER: [isize; 2] = [-(AV2_DECIDE_REACH as isize), 4];
/// Position of p1, the first sample of the edge step (p1, p0, q0, q1).
const STEP_POSITION: isize = -2;
/// Index of p1 in the inner octet.
const STEP_FIRST: usize = (STEP_POSITION - INNER[0]) as usize;
/// Weight of `q0 − p0` in the edge step.
const STEP_CENTRE_WEIGHT: i16 = 3;
/// Mask lanes: every bit set.
const ALL_LANES: i16 = -1;
/// The segments' first and last lines.
const END_LINES: [usize; 2] = [0, AV2_EDGE_LINES - 1];

type W<B> = <B as Bytes>::W;

/// Index in an octet gathered from `first` of position `i` (relative to
/// q0), if it holds it.
const fn octet_index(first: [isize; 2], i: isize) -> Option<usize> {
    let half = (OCTET / 2) as isize;
    if i >= first[0] && i < first[0] + half {
        Some((i - first[0]) as usize)
    } else if i >= first[1] && i < first[1] + half {
        Some((i - first[1]) as usize + OCTET / 2)
    } else {
        None
    }
}

/// Where a chunk's samples are, and how its moved samples go back.
trait Chunk<B: QuadBytes> {
    /// The octet of positions `first` (see [`INNER`]) of lines `END_LINES`
    /// of every segment.
    fn ends(&self, first: [isize; 2]) -> Ends<W<B>>;
    /// p1, p0, q0, q1 of every line of every segment (`inner`: the first
    /// and last lines' inner octet, already gathered).
    fn step_samples(&self, inner: &Ends<W<B>>) -> [[W<B>; 4]; AV2_EDGE_LINES];
    /// Apply the ramps of every segment: per lane the widths (zero: no
    /// ramp on that side), per line the clamped steps.
    fn ramp(&mut self, widths: Widths<W<B>>, steps: [W<B>; AV2_EDGE_LINES]);
}

/// One octet of positions of the segments' first and last lines.
#[derive(Clone, Copy)]
struct Ends<V> {
    first: [V; OCTET],
    last: [V; OCTET],
}

/// The ramp widths of every lane: samples moved on the p and q sides.
#[derive(Clone, Copy)]
struct Widths<V> {
    neg: V,
    pos: V,
}

/// A horizontal edge's chunk (`along == 1`).
///
/// Invariant (established by [`horizontal`], the only constructor): `q0`
/// points to the chunk's first q0 sample in a plane that holds 32 samples
/// from each row [`AV2_DECIDE_REACH`] rows above and below it.
struct Rows {
    q0: *mut u8,
    across: isize,
}

impl Rows {
    /// The first sample of the row `position` across the edge.
    #[inline(always)]
    fn row(&self, position: isize) -> *mut u8 {
        debug_assert!(
            (-(AV2_DECIDE_REACH as isize)..AV2_DECIDE_REACH as isize).contains(&position)
        );
        // SAFETY: rows within the reach lie inside the plane (invariant).
        unsafe { self.q0.offset(position * self.across) }
    }

    /// The four lines of every segment at `position` across the edge.
    #[inline(always)]
    fn lines<B: QuadBytes>(&self, position: isize) -> [B; AV2_EDGE_LINES] {
        // SAFETY: the row holds the chunk's 32 lines (invariant); the
        // kernel's ISA is enabled.
        unsafe { B::load_lines4(self.row(position)) }
    }

    /// Ramp the row `position`: line `l` of lane `k` moves by
    /// `sign · steps[l] · factor` (Q15).
    #[inline(always)]
    fn ramp_row<B: QuadBytes>(
        &mut self,
        position: isize,
        factor: W<B>,
        steps: &[W<B>; AV2_EDGE_LINES],
        negative: bool,
    ) {
        let mut lines = self.lines::<B>(position);
        for (line, step) in lines.iter_mut().zip(steps) {
            let offset = step.mul_q15_round(factor);
            let wide = line.widen();
            *line = B::narrow(if negative {
                wide.add(offset)
            } else {
                wide.sub(offset)
            });
        }
        // SAFETY: as `lines`.
        unsafe { B::store_lines4(lines, self.row(position)) };
    }
}

impl<B: QuadBytes> Chunk<B> for Rows {
    #[inline(always)]
    fn ends(&self, first: [isize; 2]) -> Ends<W<B>> {
        let mut ends = Ends {
            first: [W::<B>::splat(0); OCTET],
            last: [W::<B>::splat(0); OCTET],
        };
        for (k, position) in (first[0]..first[0] + 4)
            .chain(first[1]..first[1] + 4)
            .enumerate()
        {
            let lines = self.lines::<B>(position);
            ends.first[k] = lines[END_LINES[0]].widen();
            ends.last[k] = lines[END_LINES[1]].widen();
        }
        ends
    }

    #[inline(always)]
    fn step_samples(&self, _inner: &Ends<W<B>>) -> [[W<B>; 4]; AV2_EDGE_LINES] {
        let mut samples = [[W::<B>::splat(0); 4]; AV2_EDGE_LINES];
        for j in 0..4 {
            let lines = self.lines::<B>(STEP_POSITION + j as isize);
            for (line, samples) in lines.iter().zip(&mut samples) {
                samples[j] = line.widen();
            }
        }
        samples
    }

    #[inline(always)]
    fn ramp(&mut self, widths: Widths<W<B>>, steps: [W<B>; AV2_EDGE_LINES]) {
        let zero = W::<B>::splat(0);
        let weights = (ramp_weights(widths.neg), ramp_weights(widths.pos));
        // Row i away from the edge moves by weight · (width − i) where the
        // width exceeds i (the scalar `ramp_line`); a zero factor leaves
        // a sample unchanged.
        for i in 0..AV2_MAX_RAMP {
            let distance = W::<B>::splat(i as i16);
            let (reach_neg, reach_pos) = (widths.neg.gt(distance), widths.pos.gt(distance));
            if !reach_neg.or(reach_pos).any() {
                break;
            }
            if reach_neg.any() {
                let factor = widths.neg.sub(distance).max(zero).mul(weights.0);
                self.ramp_row::<B>(-(i as isize) - 1, factor, &steps, true);
            }
            if reach_pos.any() {
                let factor = widths.pos.sub(distance).max(zero).mul(weights.1);
                self.ramp_row::<B>(i as isize, factor, &steps, false);
            }
        }
    }
}

/// A vertical edge's chunk (`across == 1`).
///
/// Invariant (established by [`vertical`], the only constructor): `q0`
/// points to the chunk's first q0 sample in a plane that holds
/// [`AV2_DECIDE_REACH`] samples on each side of q0 on each of the chunk's
/// lines, `along` apart.
struct Columns {
    q0: *mut u8,
    along: usize,
}

impl Columns {
    /// The sample `position` across the edge on line `line` of segment `s`.
    #[inline(always)]
    fn at(&self, s: usize, line: usize, position: isize) -> *mut u8 {
        debug_assert!(
            s < AV2_DECIDE_SEGMENTS
                && line < AV2_EDGE_LINES
                && (-(AV2_DECIDE_REACH as isize)..AV2_DECIDE_REACH as isize).contains(&position)
        );
        // SAFETY: the chunk's lines hold the reach on both sides
        // (invariant).
        unsafe {
            self.q0
                .add((s * AV2_EDGE_LINES + line) * self.along)
                .offset(position)
        }
    }

    /// The octet of positions `first` of line `line` of every segment.
    #[inline(always)]
    fn octet<B: QuadBytes>(&self, first: [isize; 2], line: usize) -> [W<B>; OCTET] {
        let mut rows = [B::splat(0); OCTET];
        for (s, row) in rows.iter_mut().enumerate() {
            let (low, high) = (self.at(s, line, first[0]), self.at(s, line, first[1]));
            // SAFETY: both halves lie within the reach (`at`); the kernel's
            // ISA is enabled.
            *row = unsafe {
                if first[1] == first[0] + (OCTET / 2) as isize {
                    B::load(low)
                } else {
                    B::load_halves(low, high)
                }
            };
        }
        let mut positions = [W::<B>::splat(0); OCTET];
        for (position, row) in positions.iter_mut().zip(transpose(rows)) {
            *position = row.widen();
        }
        positions
    }
}

impl<B: QuadBytes> Chunk<B> for Columns {
    #[inline(always)]
    fn ends(&self, first: [isize; 2]) -> Ends<W<B>> {
        Ends {
            first: self.octet::<B>(first, END_LINES[0]),
            last: self.octet::<B>(first, END_LINES[1]),
        }
    }

    #[inline(always)]
    fn step_samples(&self, inner: &Ends<W<B>>) -> [[W<B>; 4]; AV2_EDGE_LINES] {
        let octets = [
            inner.first,
            self.octet::<B>(INNER, 1),
            self.octet::<B>(INNER, 2),
            inner.last,
        ];
        let mut samples = [[W::<B>::splat(0); 4]; AV2_EDGE_LINES];
        for (samples, octet) in samples.iter_mut().zip(&octets) {
            samples.copy_from_slice(&octet[STEP_FIRST..STEP_FIRST + 4]);
        }
        samples
    }

    #[inline(always)]
    fn ramp(&mut self, widths: Widths<W<B>>, steps: [W<B>; AV2_EDGE_LINES]) {
        let mut width_neg = [0; LANES];
        let mut width_pos = [0; LANES];
        widths.neg.store(&mut width_neg);
        widths.pos.store(&mut width_pos);
        let mut steps_by_line = [[0; LANES]; AV2_EDGE_LINES];
        for (line, steps) in steps.iter().enumerate() {
            steps.store(&mut steps_by_line[line]);
        }
        let ramp = AV2_MAX_RAMP as isize;
        for s in 0..AV2_DECIDE_SEGMENTS {
            // Widths are 0..=8 (`choose_width`, min'd with the p side's).
            let (neg, pos) = (width_neg[s] as usize, width_pos[s] as usize);
            if pos == 0 {
                continue;
            }
            // Window position k on the p side is sample p(7 − k); on the q
            // side, sample q(k).
            let factors_neg = W::<B>::load(&Q15_RAMP_FACTORS_REVERSED[neg]);
            let factors_pos = W::<B>::load(&Q15_RAMP_FACTORS[pos]);
            for (line, steps) in steps_by_line.iter().enumerate() {
                let step = W::<B>::splat(steps[s]);
                let (p_side, q_side) = (self.at(s, line, -ramp), self.at(s, line, 0));
                // SAFETY: both windows lie within the reach (`at`); the
                // kernel's ISA is enabled.
                unsafe {
                    let moved = B::load(p_side).widen().add(step.mul_q15_round(factors_neg));
                    B::narrow(moved).store(p_side);
                    let moved = B::load(q_side).widen().sub(step.mul_q15_round(factors_pos));
                    B::narrow(moved).store(q_side);
                }
            }
        }
    }
}

/// Per lane, the Q15 ramp weight of its width ([`Q15_RAMP_WEIGHTS`]).
#[inline(always)]
fn ramp_weights<V: MaskWords>(widths: V) -> V {
    widths.lookup(&RAMP_WEIGHTS)
}

// Widths are bounded to 0..=8 by the decision cascade. Register lookups
// replace nine compare/select pairs per table, on every ISA.
const RAMP_WEIGHTS: WordTable = WordTable::new(&Q15_RAMP_WEIGHTS);
const CLAMP_SCALES: WordTable = WordTable::new(&STEP_CLAMP_BY_WIDTH);

/// The first and last lines of the segments: the inner octet, and the
/// outer one once a lane reaches the wide tests.
struct Lines<V> {
    inner: Ends<V>,
    outer: Option<Ends<V>>,
}

impl<V: MaskWords> Lines<V> {
    /// Sample `i` across the edge (`q[i]` for `i ≥ 0`, `p[−i−1]` below) of
    /// the first and last lines.
    #[inline(always)]
    fn at(&self, i: isize) -> (V, V) {
        let (ends, index) = match octet_index(INNER, i) {
            Some(index) => (&self.inner, index),
            None => (
                self.outer.as_ref().expect("outer samples gathered"),
                octet_index(OUTER, i).expect("position within the reach"),
            ),
        };
        (ends.first[index], ends.last[index])
    }

    /// The scalar `both(first, last, |l| l.second_derivative(first))`.
    #[inline(always)]
    fn curvature(&self, first: isize) -> V {
        let (a0, a1) = self.at(first);
        let (b0, b1) = self.at(first + 1);
        let (c0, c1) = self.at(first + 2);
        mean(second_derivative(a0, b0, c0), second_derivative(a1, b1, c1))
    }

    /// The scalar `both(first, last, |l| l.bend(distance, negative))`.
    #[inline(always)]
    fn bend(&self, distance: usize, negative: bool) -> V {
        let d = distance as isize;
        let ((edge0, edge1), (next0, next1), (far0, far1)) = if negative {
            (self.at(-1), self.at(-2), self.at(-d - 1))
        } else {
            (self.at(0), self.at(1), self.at(d))
        };
        mean(
            deviation(edge0, next0, far0, distance),
            deviation(edge1, next1, far1, distance),
        )
    }
}

/// `|a − 2b + c|` per lane (the vector bodies stay out of closures, which
/// would not inherit the kernel's target features).
#[inline(always)]
fn second_derivative<V: MaskWords>(a: V, b: V, c: V) -> V {
    a.sub(b.add(b)).add(c).abs()
}

/// Deviation of `far` from the straight line through `edge` and `next`,
/// `distance` samples out from `edge`.
#[inline(always)]
fn deviation<V: MaskWords>(edge: V, next: V, far: V, distance: usize) -> V {
    edge.sub(far)
        .sub(edge.sub(next).mul(V::splat(distance as i16)))
        .abs()
}

/// `(a + b + 1) >> 1` of non-negative lanes.
#[inline(always)]
fn mean<V: MaskWords>(a: V, b: V) -> V {
    a.add(b).add(V::splat(1)).shr::<1>()
}

/// The cascade's state: each lane's width and whether it is still open.
struct Cascade<V> {
    width: V,
    open: V,
}

impl<V: MaskWords> Cascade<V> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            width: V::splat(0),
            open: V::splat(ALL_LANES),
        }
    }

    /// The open lanes where `condition` holds take `width` and close.
    #[inline(always)]
    fn settle(&mut self, condition: V, width: usize) {
        let hit = condition.and(self.open);
        self.width = hit.select(V::splat(width as i16), self.width);
        self.open = self.open.and_not(hit);
    }

    #[inline(always)]
    fn done(&self) -> bool {
        !self.open.any()
    }

    /// The widths, with `width` in the lanes still open.
    #[inline(always)]
    fn finish(self, width: usize) -> V {
        self.open.select(V::splat(width as i16), self.width)
    }
}

/// The lanes' filter parameters.
struct LaneFilters<V> {
    q: V,
    side: V,
    max_neg: V,
    max_pos: V,
}

impl<V: MaskWords> LaneFilters<V> {
    #[inline(always)]
    fn new(segments: &Av2Segments) -> Self {
        Self {
            q: V::load(&segments.q),
            side: V::load(&segments.side),
            max_neg: V::load(&segments.max_neg),
            max_pos: V::load(&segments.max_pos),
        }
    }
}

/// `choose_width` of every lane (the q side's width; all zero: `None`).
#[inline(always)]
fn choose_widths<B: QuadBytes, C: Chunk<B>>(
    chunk: &C,
    lines: &mut Lines<W<B>>,
    f: &LaneFilters<W<B>>,
) -> Option<W<B>> {
    let (zero, one) = (W::<B>::splat(0), W::<B>::splat(1));
    let mut cascade = Cascade::new();
    // Curvature just outside the edge on each side.
    let outer_p = lines.curvature(-3);
    let outer_q = lines.curvature(0);
    let off = f.q.eq(zero).or(f.side.eq(zero));
    let short = one.gt(f.max_pos).or(f.max_neg.gt(f.max_pos));
    let curved = outer_p.gt(f.side).or(outer_q.gt(f.side));
    cascade.settle(off.or(short).or(curved), 0);
    if cascade.done() {
        return None;
    }
    cascade.settle(f.max_pos.eq(one), 1);
    let side2 = f.side.shr::<{ TWO_SAMPLE_SIDE_SHIFT as i32 }>();
    let transition = lines.curvature(-2).add(lines.curvature(-1));
    let two_limit = f.q.mul(W::<B>::splat(TWO_SAMPLE_TRANSITION as i16));
    cascade.settle(
        outer_p
            .gt(side2)
            .or(outer_q.gt(side2))
            .or(transition.gt(two_limit)),
        1,
    );
    if cascade.done() {
        return Some(cascade.width);
    }
    let side3 = f.side.shr::<{ THREE_SAMPLE_SIDE_SHIFT as i32 }>();
    let three_limit = f.q.mul(W::<B>::splat(THREE_SAMPLE_TRANSITION as i16));
    let three_bent = bent(lines, f, THREE_SAMPLE_DISTANCE, THREE_SAMPLE_DISTANCE);
    cascade.settle(
        outer_p
            .gt(side3)
            .or(outer_q.gt(side3))
            .or(transition.gt(three_limit))
            .or(three_bent),
        2,
    );
    cascade.settle(
        f.max_pos.eq(W::<B>::splat(THREE_SAMPLE_DISTANCE as i16)),
        THREE_SAMPLE_DISTANCE,
    );
    if cascade.done() {
        return Some(cascade.width);
    }
    lines.outer = Some(chunk.ends(OUTER));
    // Within i16 (module docs).
    let transition = transition.mul(W::<B>::splat(1 << WIDE_TRANSITION_SHIFT));
    for test in &WIDE_TESTS {
        let limit = f.q.mul(W::<B>::splat(test.limit as i16));
        let wide_bent = bent(lines, f, test.distance, test.tested);
        cascade.settle(transition.gt(limit).or(wide_bent), test.fallback);
        let wider = f.max_pos.gt(W::<B>::splat(test.distance as i16));
        cascade.settle(W::<B>::splat(ALL_LANES).and_not(wider), test.distance);
    }
    Some(cascade.finish(MAX_WIDTH))
}

/// Lanes not straight out to `tested` samples (the p side only where it
/// may be modified that far), against the end threshold of `distance`.
#[inline(always)]
fn bent<V: MaskWords>(lines: &Lines<V>, f: &LaneFilters<V>, distance: usize, tested: usize) -> V {
    let end = f
        .side
        .mul(V::splat(distance as i16))
        .shr::<{ END_SHIFT as i32 }>();
    let p_tested = f.max_neg.gt(V::splat(tested as i16 - 1));
    p_tested
        .and(lines.bend(tested, true).gt(end))
        .or(lines.bend(tested, false).gt(end))
}

/// The clamped edge steps of every line: `4 · (3 · (q0 − p0) − (q1 −
/// p1))` within ±`clamp`.
#[inline(always)]
fn clamped_steps<V: MaskWords>(samples: [[V; 4]; AV2_EDGE_LINES], clamp: V) -> [V; AV2_EDGE_LINES] {
    let low = V::splat(0).sub(clamp);
    let mut steps = [low; AV2_EDGE_LINES];
    for (step, &[p1, p0, q0, q1]) in steps.iter_mut().zip(&samples) {
        *step = q0
            .sub(p0)
            .mul(V::splat(STEP_CENTRE_WEIGHT))
            .sub(q1.sub(p1))
            .mul(V::splat(EDGE_STEP_SCALE as i16))
            .min(clamp)
            .max(low);
    }
    steps
}

/// Decide and ramp every segment of `chunk`.
#[inline(always)]
fn filter<B: QuadBytes, C: Chunk<B>>(chunk: &mut C, segments: &Av2Segments) {
    let f = LaneFilters::new(segments);
    let mut lines = Lines {
        inner: chunk.ends(INNER),
        outer: None,
    };
    let Some(width_pos) = choose_widths(chunk, &mut lines, &f) else {
        return;
    };
    // The scalar `clamped_steps`: the p side moves at most max_neg samples,
    // and a segment with none to move there is left alone on both sides.
    let width_neg = width_pos.min(f.max_neg);
    let ramped = width_neg.gt(W::<B>::splat(0));
    if !ramped.any() {
        return;
    }
    let widths = Widths {
        neg: width_neg,
        pos: ramped.select(width_pos, W::<B>::splat(0)),
    };
    // The clamp of the larger width, the q side's.
    let clamp_scale = widths.pos.lookup(&CLAMP_SCALES);
    let steps = clamped_steps(chunk.step_samples(&lines.inner), f.q.mul(clamp_scale));
    chunk.ramp(widths, steps);
}

/// [`STEP_CLAMP`] by width (0..=8; 0 for no ramp).
const STEP_CLAMP_BY_WIDTH: [i16; MAX_WIDTH + 1] = {
    let mut table = [0; MAX_WIDTH + 1];
    let mut width = 1;
    while width <= MAX_WIDTH {
        table[width] = STEP_CLAMP[width - 1] as i16;
        width += 1;
    }
    table
};

/// Filter a chunk of a horizontal edge (`along == 1`).
///
/// # Safety
///
/// `B`'s ISA must be available.
///
/// # Panics
///
/// If a row within [`AV2_DECIDE_REACH`] of the edge does not hold the
/// chunk's lines.
#[inline(always)]
pub(crate) unsafe fn horizontal<B: QuadBytes>(plane: &mut [u8], segments: &Av2Segments) {
    let reach = AV2_DECIDE_REACH * segments.across;
    let lines = AV2_DECIDE_SEGMENTS * AV2_EDGE_LINES;
    assert!(
        segments.along == 1
            && segments.q0 >= reach
            && segments.q0 + reach - segments.across + lines <= plane.len(),
        "AV2 deblocking chunk outside the plane"
    );
    // SAFETY: q0 is inside the plane (asserted).
    let q0 = unsafe { plane.as_mut_ptr().add(segments.q0) };
    // The assertion establishes the invariant of `Rows`.
    filter::<B, _>(
        &mut Rows {
            q0,
            across: segments.across as isize,
        },
        segments,
    );
}

/// Filter a chunk of a vertical edge (`across == 1`).
///
/// # Safety
///
/// `B`'s ISA must be available.
///
/// # Panics
///
/// If a line lacks [`AV2_DECIDE_REACH`] samples on either side of the edge.
#[inline(always)]
pub(crate) unsafe fn vertical<B: QuadBytes>(plane: &mut [u8], segments: &Av2Segments) {
    let last_line = (AV2_DECIDE_SEGMENTS * AV2_EDGE_LINES - 1) * segments.along;
    assert!(
        segments.across == 1
            && segments.q0 >= AV2_DECIDE_REACH
            && segments.q0 + last_line + AV2_DECIDE_REACH <= plane.len(),
        "AV2 deblocking chunk outside the plane"
    );
    // SAFETY: q0 is inside the plane (asserted).
    let q0 = unsafe { plane.as_mut_ptr().add(segments.q0) };
    // The assertion establishes the invariant of `Columns`.
    filter::<B, _>(
        &mut Columns {
            q0,
            along: segments.along,
        },
        segments,
    );
}
