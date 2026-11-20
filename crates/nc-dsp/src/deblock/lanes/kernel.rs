//! The lane kernels of [`super`]: the filters of one group of lines, generic
//! over the ISA's byte vectors ([`LaneBytes`]) and the codec's filters
//! ([`LanePolicy`]).

#![allow(clippy::inline_always)]

use super::{
    GROUP_UNITS, LaneEdge, LoopFilterLanes, UNIT_LINES, WIDTH_MID, WIDTH_NARROW, WIDTH_WIDE,
};
use crate::lpf_bytes::{Bytes, LANES, Words, narrow};

/// A unit (or group) without an edge.
const WIDTH_NONE: u8 = 0;

/// `flat`/`flat2`: samples within this of p0 (q0).
const FLAT_THRESHOLD: u8 = 1;

/// The smoothing widths of one plane: what differs between AV1's luma
/// and chroma filters.
pub(crate) trait LanePolicy {
    /// Taps of the middle filter: 6 (chroma) or 8 (luma).
    const MID: usize;
    /// Taps of the wide filter: 14 (luma) or 0 (none).
    const WIDE: usize;
    /// The scalar reference for one unit (`lines` lines from `q0`), for
    /// groups whose tiles would leave the plane.
    fn scalar_unit(data: &mut [u8], q0: usize, across: usize, along: usize, edge: LaneEdge);
}

/// AV1 luma: 4, 8 and 14 taps.
pub(crate) struct Av1Luma;
/// AV1 chroma: 4 and 6 taps.
pub(crate) struct Av1Chroma;

fn av1_unit(
    data: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    edge: LaneEdge,
    mid: crate::deblock::av1::Av1FilterLength,
) {
    use crate::deblock::av1::{Av1EdgeFilter, Av1FilterLength};
    let length = match edge.width {
        WIDTH_NARROW => Av1FilterLength::Four,
        WIDTH_MID => mid,
        _ => Av1FilterLength::Fourteen,
    };
    let filter = Av1EdgeFilter {
        length,
        limit: edge.limit,
        blimit: edge.blimit,
        thresh: edge.thresh,
    };
    crate::deblock::av1::filter_edge(data, q0, across, along, UNIT_LINES, filter);
}

impl LanePolicy for Av1Luma {
    const MID: usize = 8;
    const WIDE: usize = 14;
    fn scalar_unit(data: &mut [u8], q0: usize, across: usize, along: usize, edge: LaneEdge) {
        av1_unit(
            data,
            q0,
            across,
            along,
            edge,
            crate::deblock::av1::Av1FilterLength::Eight,
        );
    }
}
impl LanePolicy for Av1Chroma {
    const MID: usize = 6;
    const WIDE: usize = 0;
    fn scalar_unit(data: &mut [u8], q0: usize, across: usize, along: usize, edge: LaneEdge) {
        av1_unit(
            data,
            q0,
            across,
            along,
            edge,
            crate::deblock::av1::Av1FilterLength::Six,
        );
    }
}
/// Byte vectors holding [`LaneBytes::LINES`] lines, with the group IO and
/// the operations of this layout. ISA must be enabled for every method.
pub(crate) trait LaneBytes: Bytes {
    /// Lines per vector (a multiple of [`UNIT_LINES`]).
    const LINES: usize;
    /// The entries of one group's units.
    unsafe fn load_units(p: *const LaneEdge) -> Self;
    /// Zero one group's entries.
    unsafe fn clear_units(p: *mut LaneEdge);
    /// The widest unit of a group ([`WIDTH_NONE`] if none).
    fn group_width(units: Self) -> u8;
    /// Byte `K` of each unit's entry in every lane of the unit.
    fn unit_byte<const K: i32>(units: Self) -> Self;
    /// All-one lanes whose unit width is at least `WIDTH`.
    fn width_at_least<const WIDTH: i32>(units: Self) -> Self;
    /// `new` where `mask` is set, else `self`.
    fn blend(self, new: Self, mask: Self) -> Self;
    /// `WA · a + WB · b` per lane, widened.
    fn madd<const WA: i8, const WB: i8>(a: Self, b: Self) -> Self::W;
    /// `(w + 2^(SHIFT−1)) >> SHIFT` per lane, for sums below 2^15 − 2^SHIFT.
    fn round_shr<const SHIFT: i32>(w: Self::W) -> Self::W;
    /// `N` rows of [`LaneBytes::LINES`] samples, `stride` apart.
    unsafe fn load_rows<const N: usize>(first: *const u8, stride: isize) -> [Self; N];
    unsafe fn store_row(self, p: *mut u8);
    /// Columns `0..N` (4, 8 or 16) of [`LaneBytes::LINES`] rows from `first`.
    unsafe fn load_columns<const N: usize>(first: *const u8, stride: isize) -> [Self; N];
    /// [`LaneBytes::load_columns`] inverted.
    unsafe fn store_columns<const N: usize>(columns: &[Self; N], first: *mut u8, stride: isize);
}

/// All-one lanes where `v <= limit` (unsigned).
#[inline(always)]
fn le<B: Bytes>(v: B, limit: B) -> B {
    v.sub_sat(limit).eq(B::splat(0))
}

/// The box-sum smoothing of the `av1_packed` module docs, outputs
/// `1..N−1` of the `N`-sample line `x` blended into `out` where `mask` is
/// set: output `k` is `(Σ_{|j−k|≤R} x[clamp(j)] + Σ_{|j−k|≤C} x[j] +
/// round) >> SHIFT`. One running total, moved one sample on by a
/// subtract/add pair per box (dav1d's `pmaddubsw` with `-1, 1`).
#[inline(always)]
fn smooth<B: LaneBytes, const N: usize, const R: usize, const C: usize, const SHIFT: i32>(
    x: &[B; N],
    out: &mut [B; N],
    mask: B,
) {
    let clamp = |j: isize| j.clamp(0, N as isize - 1) as usize;
    // Weights of output 1: the outer box with repeated ends, plus the centre.
    let mut weights = [0i8; N];
    for j in 1 - R as isize..=1 + R as isize {
        weights[clamp(j)] += 1;
    }
    for weight in &mut weights[1 - C..=1 + C] {
        *weight += 1;
    }
    let mut total = B::W::splat(0);
    for j in (0..N).step_by(2) {
        if weights[j] != 0 || weights[j + 1] != 0 {
            total = total.add(weighted_pair(x[j], x[j + 1], weights[j], weights[j + 1]));
        }
    }
    for k in 1..N - 1 {
        out[k] = out[k].blend(B::narrow(B::round_shr::<SHIFT>(total)), mask);
        if k + 1 < N - 1 {
            let (leave, enter) = (clamp(k as isize - R as isize), clamp((k + R + 1) as isize));
            if leave != enter {
                total = total.add(B::madd::<-1, 1>(x[leave], x[enter]));
            }
            total = total.add(B::madd::<-1, 1>(x[k - C], x[k + C + 1]));
        }
    }
}

/// `wa · a + wb · b` for the small constant weights of [`smooth`].
#[inline(always)]
fn weighted_pair<B: LaneBytes>(a: B, b: B, wa: i8, wb: i8) -> B::W {
    macro_rules! pick {
        ($($wa:literal, $wb:literal);*) => {
            match (wa, wb) {
                $(($wa, $wb) => B::madd::<$wa, $wb>(a, b),)*
                _ => unreachable!("smoothing weights"),
            }
        };
    }
    pick!(3, 2; 2, 1; 1, 1; 1, 0; 7, 2)
}

/// Samples p7..q7 of a group's lines, by distance from the edge (`p[0]`
/// is p0); only a filter's reach is loaded.
#[derive(Clone, Copy)]
struct Sides<B> {
    p: [B; LANES],
    q: [B; LANES],
}

impl<B: Bytes> Sides<B> {
    /// The `N`-sample line p(N/2−1)..q(N/2−1).
    #[inline(always)]
    fn line<const N: usize>(&self) -> [B; N] {
        join(&self.p, &self.q)
    }

    #[inline(always)]
    fn set_line<const N: usize>(&mut self, line: &[B; N]) {
        split(line, &mut self.p, &mut self.q);
    }
}

/// A group's per-lane thresholds and widths, unpacked from its entries.
struct Lanes<B> {
    limit: B,
    blimit: B,
    thresh: B,
    /// Lines whose unit has an edge, and whose edge allows the middle /
    /// wide filters.
    active: B,
    mid: B,
    wide: B,
}

impl<B: LaneBytes> Lanes<B> {
    #[inline(always)]
    fn of(units: B) -> Self {
        Self {
            limit: B::unit_byte::<0>(units),
            blimit: B::unit_byte::<1>(units),
            thresh: B::unit_byte::<2>(units),
            active: B::width_at_least::<{ WIDTH_NARROW as i32 }>(units),
            mid: B::width_at_least::<{ WIDTH_MID as i32 }>(units),
            wide: B::width_at_least::<{ WIDTH_WIDE as i32 }>(units),
        }
    }
}

/// Columns a vertical edge's tile spans each side, by group width.
const fn tile_columns<P: LanePolicy>(width: u8) -> usize {
    match width {
        WIDTH_NARROW => NARROW_REACH,
        WIDTH_MID => INNER_REACH,
        _ => LANES,
    }
}

/// Rows a horizontal edge's group reads each side before the wide stage.
const fn inner_rows<P: LanePolicy>(width: u8) -> usize {
    if width == WIDTH_NARROW {
        NARROW_REACH
    } else {
        P::MID / 2
    }
}

/// Samples each side of the edge the 4-tap filter reads.
const NARROW_REACH: usize = 2;
/// Samples each side of the edge the 8-tap filter reads.
const INNER_REACH: usize = 4;

/// A group's first line's q0 and the step across the edge's lines.
#[derive(Clone, Copy)]
struct Tile {
    at: *mut u8,
    stride: isize,
}

impl Tile {
    /// The address `i` samples across the edge from q0 (horizontal edges).
    #[inline(always)]
    fn row(self, i: isize) -> *mut u8 {
        self.at.wrapping_offset(i * self.stride)
    }

    /// Load a group's samples: a vertical edge's whole tile, a horizontal
    /// edge's rows p(n−1)..q(n−1) for the first `n`.
    ///
    /// # Safety
    ///
    /// As [`group`].
    #[inline(always)]
    unsafe fn load<B: LaneBytes, P: LanePolicy, const VERTICAL: bool, const WIDTH: u8>(
        self,
        sides: &mut Sides<B>,
    ) {
        // SAFETY: the caller checked this group's tile.
        unsafe {
            if VERTICAL {
                match tile_columns::<P>(WIDTH) {
                    NARROW_REACH => {
                        let columns = B::load_columns::<4>(self.at.sub(NARROW_REACH), self.stride);
                        sides.set_line(&columns);
                    }
                    INNER_REACH => {
                        sides
                            .set_line(&B::load_columns::<8>(self.at.sub(INNER_REACH), self.stride));
                    }
                    _ => sides.set_line(&B::load_columns::<16>(self.at.sub(LANES), self.stride)),
                }
            } else {
                self.load_rows(0..inner_rows::<P>(WIDTH), sides);
            }
        }
    }

    /// Rows p(i) and q(i) of a horizontal edge for `i` in `range`.
    ///
    /// # Safety
    ///
    /// The rows are readable; the ISA is enabled.
    #[inline(always)]
    unsafe fn load_rows<B: LaneBytes>(self, range: core::ops::Range<usize>, sides: &mut Sides<B>) {
        for i in range {
            // SAFETY: forwarded.
            unsafe {
                sides.p[i] = B::load_rows::<1>(self.row(-(i as isize) - 1), self.stride)[0];
                sides.q[i] = B::load_rows::<1>(self.row(i as isize), self.stride)[0];
            }
        }
    }

    /// Store a group: a vertical edge's whole tile, a horizontal edge's
    /// rows p(n−1)..q(n−1) for the first `written`.
    ///
    /// # Safety
    ///
    /// As [`group`].
    #[inline(always)]
    unsafe fn store<B: LaneBytes, P: LanePolicy, const VERTICAL: bool, const WIDTH: u8>(
        self,
        sides: &Sides<B>,
        written: usize,
    ) {
        // SAFETY: the same checked tile as the loads.
        unsafe {
            if VERTICAL {
                match tile_columns::<P>(WIDTH) {
                    NARROW_REACH => {
                        B::store_columns::<4>(
                            &sides.line(),
                            self.at.sub(NARROW_REACH),
                            self.stride,
                        );
                    }
                    INNER_REACH => {
                        B::store_columns::<8>(&sides.line(), self.at.sub(INNER_REACH), self.stride);
                    }
                    _ => B::store_columns::<16>(&sides.line(), self.at.sub(LANES), self.stride),
                }
            } else {
                for i in 0..written {
                    sides.p[i].store_row(self.row(-(i as isize) - 1));
                    sides.q[i].store_row(self.row(i as isize));
                }
            }
        }
    }
}

/// The filter mask (`filterMask` lanes that filter at all), `hevMask` and
/// the middle filter's `flat` lanes of a group of width `WIDTH`.
#[inline(always)]
fn decide<B: LaneBytes, P: LanePolicy, const WIDTH: u8>(
    s: &Sides<B>,
    lanes: &Lanes<B>,
) -> (B, B, B) {
    let (p, q) = (&s.p, &s.q);
    let mut adjacent = p[1].difference(p[0]).max(q[1].difference(q[0]));
    let hev = le(adjacent, lanes.thresh).xor(B::splat(u8::MAX));
    let mut flat = adjacent;
    // The middle and wide filters' masks also read the steps beyond p1..q1:
    // only those lanes' (dav1d's "only apply fm-wide to wd>4 blocks").
    if WIDTH >= WIDTH_MID {
        let mut steps = p[2].difference(p[1]).max(q[2].difference(q[1]));
        if P::MID == 8 {
            steps = steps.max(p[3].difference(p[2])).max(q[3].difference(q[2]));
        }
        adjacent = adjacent.max(steps.and(lanes.mid));
    }
    let across = p[0].difference(q[0]);
    let boundary = across
        .add_sat(across)
        .add_sat(p[1].difference(q[1]).shr::<1>());
    let mask = le(adjacent, lanes.limit)
        .and(le(boundary, lanes.blimit))
        .and(lanes.active);
    if WIDTH >= WIDTH_MID {
        flat = flat.max(p[2].difference(p[0])).max(q[2].difference(q[0]));
        if P::MID == 8 {
            flat = flat.max(p[3].difference(p[0])).max(q[3].difference(q[0]));
        }
    }
    let flat = le(flat, B::splat(FLAT_THRESHOLD)).and(mask).and(lanes.mid);
    (mask, hev, flat)
}

/// The middle filter (6 or 8 taps) of the `flat` lanes, from the original
/// samples, into `out`.
#[inline(always)]
fn filter_mid<B: LaneBytes, P: LanePolicy>(original: &Sides<B>, out: &mut Sides<B>, flat: B) {
    if P::MID == 8 {
        let mut line = out.line::<8>();
        smooth::<B, 8, 3, 0, 3>(&original.line(), &mut line, flat);
        out.set_line(&line);
    } else {
        let mut line = out.line::<6>();
        smooth::<B, 6, 2, 1, 3>(&original.line(), &mut line, flat);
        out.set_line(&line);
    }
}

/// The wide filter (14 taps) of the lanes flat out to its reach,
/// into `out`. `original` holds the unfiltered samples (a horizontal
/// edge's outer rows are loaded into it here). Returns whether any lane
/// took it.
///
/// # Safety
///
/// As [`group`], for a wide group.
#[inline(always)]
unsafe fn filter_wide<B: LaneBytes, P: LanePolicy, const VERTICAL: bool>(
    tile: Tile,
    original: &mut Sides<B>,
    out: &mut Sides<B>,
    (flat, lanes): (B, &Lanes<B>),
) -> bool {
    let reach = P::WIDE / 2;
    if !VERTICAL {
        // SAFETY: the caller checked the wide tile.
        unsafe { tile.load_rows(P::MID / 2..reach, original) };
    }
    let mut far = B::splat(0);
    for i in INNER_REACH..reach {
        far = far
            .max(original.p[i].difference(original.p[0]))
            .max(original.q[i].difference(original.q[0]));
    }
    let flat2 = le(far, B::splat(FLAT_THRESHOLD)).and(flat).and(lanes.wide);
    if !flat2.any() {
        return false;
    }
    out.p[INNER_REACH..reach].copy_from_slice(&original.p[INNER_REACH..reach]);
    out.q[INNER_REACH..reach].copy_from_slice(&original.q[INNER_REACH..reach]);
    let mut line = out.line::<14>();
    smooth::<B, 14, 6, 1, 4>(&original.line(), &mut line, flat2);
    out.set_line(&line);
    true
}

/// The decisions and filters of one group whose widest unit is `WIDTH`.
///
/// # Safety
///
/// `at` is the first line's q0; every sample of the `WIDTH` tile (the
/// filter's reach each side, [`LaneBytes::LINES`] lines) is readable and
/// writable; the ISA is enabled.
#[inline(always)]
unsafe fn group<B: LaneBytes, P: LanePolicy, const VERTICAL: bool, const WIDTH: u8>(
    at: *mut u8,
    stride: isize,
    units: B,
) {
    let tile = Tile { at, stride };
    let lanes = Lanes::of(units);
    let mut sides = Sides {
        p: [B::splat(0); LANES],
        q: [B::splat(0); LANES],
    };
    // SAFETY: forwarded.
    unsafe { tile.load::<B, P, VERTICAL, WIDTH>(&mut sides) };
    let (mask, hev, flat) = decide::<B, P, WIDTH>(&sides, &lanes);
    if !mask.any() {
        return;
    }
    let mut original = sides;
    narrow(&mut sides.p, &mut sides.q, mask, hev);
    // Rows written back (horizontal edges): p1..q1, or the smoothing's.
    let mut written = 2;
    if WIDTH >= WIDTH_MID && flat.any() {
        filter_mid::<B, P>(&original, &mut sides, flat);
        written = P::MID / 2 - 1;
        if WIDTH == WIDTH_WIDE && P::WIDE > 0 {
            // SAFETY: forwarded, for a wide group.
            let wide = unsafe {
                filter_wide::<B, P, VERTICAL>(tile, &mut original, &mut sides, (flat, &lanes))
            };
            if wide {
                written = P::WIDE / 2 - 1;
            }
        }
    }
    // SAFETY: forwarded.
    unsafe { tile.store::<B, P, VERTICAL, WIDTH>(&sides, written) };
}

/// Columns `0..N` (p(N/2−1)..q(N/2−1)) into `p`/`q` by distance.
#[inline(always)]
fn split<B: Bytes, const N: usize>(columns: &[B; N], p: &mut [B; LANES], q: &mut [B; LANES]) {
    for i in 0..N / 2 {
        p[i] = columns[N / 2 - 1 - i];
        q[i] = columns[N / 2 + i];
    }
}

/// [`split`] inverted.
#[inline(always)]
fn join<B: Bytes, const N: usize>(p: &[B; LANES], q: &[B; LANES]) -> [B; N] {
    let mut columns = [B::splat(0); N];
    for i in 0..N / 2 {
        columns[N / 2 - 1 - i] = p[i];
        columns[N / 2 + i] = q[i];
    }
    columns
}

/// Samples each side a group of `width` reads (the tile it loads).
fn reach<P: LanePolicy, const VERTICAL: bool>(width: u8) -> usize {
    if VERTICAL {
        tile_columns::<P>(width)
    } else if width == WIDTH_WIDE {
        P::WIDE / 2
    } else {
        inner_rows::<P>(width)
    }
}

impl LoopFilterLanes {
    /// Filter every entry with lane type `B` and policy `P`, clearing them.
    ///
    /// # Safety
    ///
    /// The ISA of `B` is enabled.
    #[inline(always)]
    pub(crate) unsafe fn filter<B: LaneBytes, P: LanePolicy, const VERTICAL: bool>(
        &mut self,
        data: &mut [u8],
        stride: usize,
    ) {
        let units = self.units;
        for line in self.touched.drain(..) {
            let line = line as usize;
            let (lo, hi) = core::mem::replace(&mut self.groups[line], (u32::MAX, 0));
            let (lo, hi) = (lo as usize * GROUP_UNITS, hi as usize * GROUP_UNITS);
            let first_line = lo * UNIT_LINES;
            let q0 = if VERTICAL {
                first_line * stride + line * UNIT_LINES
            } else {
                line * UNIT_LINES * stride + first_line
            };
            let entries = &mut self.entries[line * units + lo..line * units + hi];
            // SAFETY: forwarded ISA contract; groups are whole.
            unsafe { filter_line::<B, P, VERTICAL>(data, q0, stride, entries) };
        }
    }
}

/// Filter the units `units` of one edge line (`q0`: the first line's q0)
/// and clear their entries. Groups whose tile would leave `data` take the
/// scalar reference per unit.
///
/// # Safety
///
/// The ISA of `B` is enabled; `units.len()` is a multiple of a group.
#[inline(always)]
unsafe fn filter_line<B: LaneBytes, P: LanePolicy, const VERTICAL: bool>(
    data: &mut [u8],
    q0: usize,
    stride: usize,
    units: &mut [LaneEdge],
) {
    let unit_count = B::LINES / UNIT_LINES;
    debug_assert!(units.len().is_multiple_of(unit_count));
    let (across, along) = if VERTICAL { (1, stride) } else { (stride, 1) };
    for (g, entries) in units.chunks_exact_mut(unit_count).enumerate() {
        // SAFETY: a group's entries are `LINES` bytes; the ISA is enabled.
        let packed = unsafe { B::load_units(entries.as_ptr()) };
        let width = B::group_width(packed);
        if width == WIDTH_NONE {
            continue;
        }
        let start = q0 + g * B::LINES * along;
        let r = reach::<P, VERTICAL>(width);
        let (before, after) = if VERTICAL {
            (r, (B::LINES - 1) * stride + r)
        } else {
            (r * stride, (r - 1) * stride + B::LINES)
        };
        let fits = start >= before && start + after <= data.len();
        if fits {
            let at = data.as_mut_ptr().wrapping_add(start);
            // SAFETY: the tile of `width` lies in `data` (checked above).
            unsafe {
                let stride = stride as isize;
                match width {
                    WIDTH_NARROW => group::<B, P, VERTICAL, WIDTH_NARROW>(at, stride, packed),
                    WIDTH_MID => group::<B, P, VERTICAL, WIDTH_MID>(at, stride, packed),
                    _ => group::<B, P, VERTICAL, WIDTH_WIDE>(at, stride, packed),
                }
            }
        } else {
            for (u, edge) in entries.iter().enumerate() {
                if edge.width != WIDTH_NONE {
                    P::scalar_unit(data, start + u * UNIT_LINES * along, across, along, *edge);
                }
            }
        }
        // SAFETY: as the load.
        unsafe { B::clear_units(entries.as_mut_ptr()) };
    }
}
