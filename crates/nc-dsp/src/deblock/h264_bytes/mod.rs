//! H.264 deblocking (§8.7.2.3, §8.7.2.4) on sixteen byte lanes: one lane
//! per line across the edge, so a whole luma edge (16 lines), or one chroma
//! edge of both 4:2:0 planes (8 lines each), is one pass. The 16-bit
//! kernels (`generic`) need two passes of eight lanes, each widened,
//! transposed and narrowed again.
//!
//! The filters stay in bytes wherever an exact byte form exists:
//!
//! - the decisions are absolute differences against α / β;
//! - the bS < 4 weak `p1` / `q1` update, `p1 + Clip3(−tC0, tC0, (p2 +
//!   ((p0 + q0 + 1) >> 1) − 2·p1) >> 1)`, equals `(p2 + avg) >> 1` (a
//!   floor average) clamped to `p1 ± tC0`, since `2·p1` is even and the
//!   result always lies between `p1` and that average;
//! - the bS == 4 non-strong `p0' = (2·p1 + p0 + q1 + 2) >> 2` equals
//!   `(((p0 + q1) >> 1) + p1 + 1) >> 1`: with `s = p0 + q1` odd the two
//!   numerators are `N` and `N + 1` for an even `N`, and `N + 1` (odd) is
//!   never a multiple of 4, so both floors agree.
//!
//! Only `delta` (signed, widened to 16 bits and narrowed with saturation:
//! the clip to ±tC ≤ 27 makes the saturation invisible) and the strong
//! filter's sums are computed wider, inside [`H264Bytes`].
//!
//! Vertical edges: a line's samples are a row, so the 16 rows' p3..q3
//! (luma) or p1..q1 (chroma) are transposed into lanes, in whatever way
//! suits each ISA.
//!
//! Jetson (Cortex-A78AE): a fully filtered luma macroblock (bS 2, the
//! checkasm bench) takes 342 ns, against 613 ns for the 16-bit kernels;
//! h264-fast encodes −2.2% (crowd QP 39) and −3.4% (bodycam QP 45)
//! cycles (jperf, 5 rounds), streams identical.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use super::{CHROMA_LINES, Edge, LUMA_LINES};

/// Lanes per vector: the lines of one luma edge, or of one chroma edge in
/// both planes.
const LANES: usize = 16;
/// Lanes holding one chroma plane's lines (U in lanes 0..8, V in 8..16).
const PLANE_LANES: usize = CHROMA_LINES;
/// Luma lines per bS / tC0 entry.
const LUMA_LINES_PER_ENTRY: usize = 4;
/// Chroma lines per bS / tC0 entry.
const CHROMA_LINES_PER_ENTRY: usize = 2;
/// The bS value that selects the bS == 4 filters (§8.7.2.4).
const STRONG_BS: u8 = 4;
const _: () = assert!(LUMA_LINES == LANES && 2 * PLANE_LANES == LANES);

/// Sixteen unsigned byte lanes, with the few wider operations the filter
/// needs. Masks are all-ones (true) or zero bytes.
pub(crate) trait H264Bytes: Copy {
    /// Every lane `v`.
    fn splat(v: u8) -> Self;
    /// Bytes `0..8` from `lo`, `8..16` from `hi` (little-endian).
    fn from_halves(lo: u64, hi: u64) -> Self;
    /// Sixteen bytes at `p`.
    unsafe fn load(p: *const u8) -> Self;
    /// Store sixteen bytes at `p`.
    unsafe fn store(self, p: *mut u8);
    /// Bytes `0..8` from `lo`, `8..16` from `hi`.
    unsafe fn load_halves(lo: *const u8, hi: *const u8) -> Self;
    /// Inverse of [`Self::load_halves`].
    unsafe fn store_halves(self, lo: *mut u8, hi: *mut u8);
    /// Eight bytes of each of 16 rows, transposed: byte `j` of output `k`
    /// is byte `k` of row `j`.
    unsafe fn load_rows_8(rows: [*const u8; LANES]) -> [Self; 8];
    /// Inverse of [`Self::load_rows_8`].
    unsafe fn store_rows_8(columns: [Self; 8], rows: [*mut u8; LANES]);
    /// As [`Self::load_rows_8`] for four bytes of each row.
    unsafe fn load_rows_4(rows: [*const u8; LANES]) -> [Self; 4];
    /// Inverse of [`Self::load_rows_4`].
    unsafe fn store_rows_4(columns: [Self; 4], rows: [*mut u8; LANES]);
    /// `|a − b|`.
    fn abs_diff(self, b: Self) -> Self;
    /// Mask of `self < b`.
    fn less(self, b: Self) -> Self;
    fn and(self, b: Self) -> Self;
    /// `self & !b`.
    fn and_not(self, b: Self) -> Self;
    /// `mask ? a : b` per lane (`self` is the mask).
    fn select(self, a: Self, b: Self) -> Self;
    /// Wrapping `self − b`.
    fn sub(self, b: Self) -> Self;
    fn add_sat(self, b: Self) -> Self;
    fn sub_sat(self, b: Self) -> Self;
    fn min(self, b: Self) -> Self;
    fn max(self, b: Self) -> Self;
    /// `(a + b + 1) >> 1`.
    fn avg_round(self, b: Self) -> Self;
    /// `(a + b) >> 1`.
    fn avg_floor(self, b: Self) -> Self;
    /// Whether any lane is nonzero.
    fn any(self) -> bool;
    /// `((q0 − p0)·4 + (p1 − q1) + 4) >> 3` as signed bytes, saturated.
    fn normal_delta(p1: Self, p0: Self, q0: Self, q1: Self) -> Self;
    /// `Clip3(−t, t, self)`, `self` signed, `t` at most 127.
    fn clip_signed(self, t: Self) -> Self;
    /// `Clip1(self + delta)`, `delta` signed.
    fn add_signed(self, delta: Self) -> Self;
    /// `Clip1(self − delta)`, `delta` signed and above −128.
    fn sub_signed(self, delta: Self) -> Self;
    /// The bS == 4 strong outputs `[x2', x1', x0']` of one side (§8.7.2.4,
    /// 8-475..8-477 for `x = p, y = q`).
    fn strong_side(x3: Self, x2: Self, x1: Self, x0: Self, y0: Self, y1: Self) -> [Self; 3];
}

/// Multiplier copying a byte into the low 4 bytes of a word.
const REPEAT_4: u64 = 0x0101_0101;
/// Multiplier copying a byte into the low 2 bytes of a word.
const REPEAT_2: u64 = 0x0101;

/// Per-lane bS and tC0 of a luma edge: lane `i` takes entry `i / 4`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn luma_lanes<B: H264Bytes>(values: [i16; 4]) -> B {
    const _: () = assert!(LUMA_LINES_PER_ENTRY == 4);
    let v = values.map(|x| u64::from(x as u8) * REPEAT_4);
    B::from_halves(v[0] | (v[1] << 32), v[2] | (v[3] << 32))
}

/// Per-lane bS and tC0 of a chroma edge in both planes: lane `i` takes
/// entry `(i % 8) / 2`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn chroma_lanes<B: H264Bytes>(values: [i16; 4]) -> B {
    const _: () = assert!(CHROMA_LINES_PER_ENTRY == 2);
    let v = values.map(|x| u64::from(x as u8) * REPEAT_2);
    let half = v[0] | (v[1] << 16) | (v[2] << 32) | (v[3] << 48);
    B::from_halves(half, half)
}

/// Index of sample `pos` across the edge (p3 = −4 … q3 = 3) of line `line`.
/// Unchecked: [`crate::Dsp`] verified every sample the edge reads.
#[allow(clippy::inline_always)]
#[inline(always)]
fn index(e: &Edge, line: usize, pos: isize) -> usize {
    (e.base + line * e.along).wrapping_add_signed(pos * e.across as isize)
}

/// The filterSamplesFlag mask (§8.7.2.2).
#[allow(clippy::inline_always)]
#[inline(always)]
fn filter_flag<B: H264Bytes>(e: &Edge, bs: B, [p1, p0, q0, q1]: [B; 4]) -> B {
    let (alpha, beta) = (B::splat(e.alpha as u8), B::splat(e.beta as u8));
    let zero = B::splat(0);
    zero.less(bs)
        .and(p0.abs_diff(q0).less(alpha))
        .and(p1.abs_diff(p0).less(beta))
        .and(q1.abs_diff(q0).less(beta))
}

/// `(2·x1 + x0 + y1 + 2) >> 2`, the bS == 4 non-strong `x0'` (exact: see
/// the module docs).
#[allow(clippy::inline_always)]
#[inline(always)]
fn chroma_style<B: H264Bytes>(x1: B, x0: B, y1: B) -> B {
    x0.avg_floor(y1).avg_round(x1)
}

/// Filter one H.264 luma edge (16 lines).
///
/// # Safety
///
/// The ISA of `B` is enabled; every sample p3..q3 of every line lies in
/// `plane` (checked by [`crate::Dsp`]).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn luma<B: H264Bytes>(plane: &mut [u8], e: &Edge) {
    if e.bs.contains(&STRONG_BS) {
        // SAFETY: forwarded.
        unsafe { luma_edge::<B, true>(plane, e) }
    } else {
        // SAFETY: forwarded.
        unsafe { luma_edge::<B, false>(plane, e) }
    }
}

/// [`luma`] with the bS == 4 filters compiled in (`STRONG`) or out (no
/// entry of the edge is 4).
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn luma_edge<B: H264Bytes, const STRONG: bool>(plane: &mut [u8], e: &Edge) {
    // Every access below is to a sample p3..q3 of one of the 16 lines,
    // which the caller guarantees lie inside `plane`.
    let ptr = plane.as_mut_ptr();
    let positions: [B; 8] = if e.along == 1 {
        let mut rows = [B::splat(0); 8];
        // p3 and q3 are only read by the strong filter.
        let used = if STRONG { 0..8 } else { 1..7 };
        for k in used {
            // SAFETY: position k − 4 of the 16 lines (see above).
            rows[k] = unsafe { B::load(ptr.add(index(e, 0, k as isize - 4))) };
        }
        rows
    } else {
        let rows = core::array::from_fn(|line| ptr.wrapping_add(index(e, line, -4)).cast_const());
        // SAFETY: p3..q3 of each line (see above).
        unsafe { B::load_rows_8(rows) }
    };
    let Some(out) = luma_filter::<B, STRONG>(e, positions) else {
        return;
    };
    if e.along == 1 {
        // The normal filter changes p1..q1 only.
        let changed = if STRONG { 0..6 } else { 1..5 };
        for k in changed {
            // SAFETY: position k − 3 of the 16 lines (see above).
            unsafe { out[k].store(ptr.add(index(e, 0, k as isize - 3))) };
        }
    } else {
        let [p2, p1, p0, q0, q1, q2] = out;
        let rows = core::array::from_fn(|line| ptr.wrapping_add(index(e, line, -4)));
        // SAFETY: p3..q3 of each line (see above); p3 and q3 are written
        // back unchanged.
        unsafe { B::store_rows_8([positions[0], p2, p1, p0, q0, q1, q2, positions[7]], rows) }
    }
}

/// The filtered p2..q2 of lines whose samples p3..q3 are `positions`, or
/// `None` when no line is filtered.
#[allow(clippy::inline_always)]
#[inline(always)]
#[allow(clippy::similar_names)]
fn luma_filter<B: H264Bytes, const STRONG: bool>(e: &Edge, positions: [B; 8]) -> Option<[B; 6]> {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = positions;
    let bs = luma_lanes::<B>(e.bs.map(i16::from));
    let flag = filter_flag(e, bs, [p1, p0, q0, q1]);
    if !flag.any() {
        return None;
    }
    let tc0 = luma_lanes::<B>(e.tc0);
    let beta = B::splat(e.beta as u8);
    let ap = p2.abs_diff(p0).less(beta);
    let aq = q2.abs_diff(q0).less(beta);
    let four = if STRONG {
        B::splat(STRONG_BS - 1).less(bs).and(flag)
    } else {
        B::splat(0)
    };
    let normal = flag.and_not(four);

    // bS < 4 (§8.7.2.3): tC = tC0 + (ap < β) + (aq < β); the masks are −1.
    let tc = tc0.sub(ap).sub(aq);
    let delta = B::normal_delta(p1, p0, q0, q1).clip_signed(tc).and(normal);
    let mut new_p0 = p0.add_signed(delta);
    let mut new_q0 = q0.sub_signed(delta);
    let average = p0.avg_round(q0);
    let weak_p1 = p2
        .avg_floor(average)
        .max(p1.sub_sat(tc0))
        .min(p1.add_sat(tc0));
    let weak_q1 = q2
        .avg_floor(average)
        .max(q1.sub_sat(tc0))
        .min(q1.add_sat(tc0));
    let mut new_p1 = ap.and(normal).select(weak_p1, p1);
    let mut new_q1 = aq.and(normal).select(weak_q1, q1);
    let (mut new_p2, mut new_q2) = (p2, q2);

    if STRONG {
        // bS == 4 (§8.7.2.4).
        let alpha_quarter = B::splat(((e.alpha >> 2) + 2) as u8);
        let strong = p0.abs_diff(q0).less(alpha_quarter).and(four);
        let (strong_p, strong_q) = (ap.and(strong), aq.and(strong));
        let [sp2, sp1, sp0] = B::strong_side(p3, p2, p1, p0, q0, q1);
        let [sq2, sq1, sq0] = B::strong_side(q3, q2, q1, q0, p0, p1);
        new_p0 = strong_p.select(sp0, four.select(chroma_style(p1, p0, q1), new_p0));
        new_q0 = strong_q.select(sq0, four.select(chroma_style(q1, q0, p1), new_q0));
        new_p1 = strong_p.select(sp1, new_p1);
        new_q1 = strong_q.select(sq1, new_q1);
        new_p2 = strong_p.select(sp2, p2);
        new_q2 = strong_q.select(sq2, q2);
    }
    Some([new_p2, new_p1, new_p0, new_q0, new_q1, new_q2])
}

/// Filter one H.264 4:2:0 chroma edge in both planes (8 lines each): `u`
/// and `v` share the edge's geometry (same stride) and parameters.
///
/// # Safety
///
/// The ISA of `B` is enabled; in both planes every sample p1..q1 of every
/// line lies inside the plane (for a vertical edge, p3..q3: checked by
/// [`crate::Dsp`]).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn chroma_pair<B: H264Bytes>(u: &mut [u8], v: &mut [u8], e: &Edge) {
    // Every access below is to a sample p1..q1 of one of the 8 lines of
    // either plane, which the caller guarantees lie inside it.
    let (pu, pv) = (u.as_mut_ptr(), v.as_mut_ptr());
    let positions: [B; 4] = if e.along == 1 {
        let mut rows = [B::splat(0); 4];
        for (k, row) in rows.iter_mut().enumerate() {
            let at = index(e, 0, k as isize - 2);
            // SAFETY: position k − 2 of the 8 lines in both planes.
            *row = unsafe { B::load_halves(pu.add(at), pv.add(at)) };
        }
        rows
    } else {
        let rows = chroma_rows(pu, pv, e);
        // SAFETY: p1..q1 of each line of both planes (see above).
        unsafe { B::load_rows_4(rows.map(<*mut u8>::cast_const)) }
    };
    let Some([p0, q0]) = chroma_filter(e, positions) else {
        return;
    };
    if e.along == 1 {
        for (k, row) in [(-1, p0), (0, q0)] {
            let at = index(e, 0, k);
            // SAFETY: position k of the 8 lines in both planes.
            unsafe { row.store_halves(pu.add(at), pv.add(at)) };
        }
    } else {
        let rows = chroma_rows(pu, pv, e);
        // SAFETY: p1..q1 of each line of both planes (see above); p1 and q1
        // are written back unchanged.
        unsafe { B::store_rows_4([positions[0], p0, q0, positions[3]], rows) }
    }
}

/// The first sample (p1) of each line of a vertical chroma edge: U's lines
/// in lanes 0..8, V's in 8..16.
#[allow(clippy::inline_always)]
#[inline(always)]
fn chroma_rows(u: *mut u8, v: *mut u8, e: &Edge) -> [*mut u8; LANES] {
    core::array::from_fn(|lane| {
        let plane = if lane < PLANE_LANES { u } else { v };
        plane.wrapping_add(index(e, lane % PLANE_LANES, -2))
    })
}

/// The filtered p0 and q0 of chroma lines whose samples p1..q1 are
/// `positions` (§8.7.2.3, §8.7.2.4 with chromaStyleFilteringFlag), or
/// `None` when no line is filtered.
#[allow(clippy::inline_always)]
#[inline(always)]
fn chroma_filter<B: H264Bytes>(e: &Edge, [p1, p0, q0, q1]: [B; 4]) -> Option<[B; 2]> {
    let bs = chroma_lanes::<B>(e.bs.map(i16::from));
    let flag = filter_flag(e, bs, [p1, p0, q0, q1]);
    if !flag.any() {
        return None;
    }
    let four = B::splat(STRONG_BS - 1).less(bs).and(flag);
    let tc = chroma_lanes::<B>(e.tc0).add_sat(B::splat(1));
    let delta = B::normal_delta(p1, p0, q0, q1)
        .clip_signed(tc)
        .and(flag.and_not(four));
    Some([
        four.select(chroma_style(p1, p0, q1), p0.add_signed(delta)),
        four.select(chroma_style(q1, q0, p1), q0.sub_signed(delta)),
    ])
}
