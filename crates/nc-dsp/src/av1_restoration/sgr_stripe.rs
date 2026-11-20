//! The self-guided filter of one 8-bit stripe (AV1 §7.17.3) in dav1d's
//! layout: the ISA-neutral schedule of the complete-stripe kernels
//! ([`super::sgr_avx2`], [`super::sgr_neon`]), after dav1d 1.5.4
//! `src/x86/looprestoration_avx2.asm` (`sgr_filter_5x5_8bpc`,
//! `sgr_filter_3x3_8bpc`, `sgr_filter_mix_8bpc`; BSD-2-Clause, see
//! `THIRD_PARTY.md`), written over our padded stripe window (so none of its
//! edge cases). Each ISA supplies the per-step arithmetic through
//! [`StripeLanes`]; this module owns the scratch rows and the order in
//! which window rows, box rows and output rows are visited.
//!
//! What it does differently from the shared row kernels
//! ([`super::sgr_lanes`], eight 32-bit lanes per step, each statistic in its
//! own row):
//!
//! - **One pass per window row.** Sixteen columns a step: the bytes are
//!   widened to 16-bit taps, the box sums kept in 16 bits, the squares in
//!   32; the vertical sum, `A` and `B` follow in the same step. Mixed sets
//!   take both radii from one load.
//! - **Running vertical sums.** Radius 2 keeps sums of row pairs, so a box
//!   row adds two pairs and one row; radius 1 keeps the two rows before.
//! - **`A` and `B` in one 32-bit word.** dav1d keeps `x = 256 − A` (at most
//!   255) in the low 12 bits and `B` (at most 16 bits, plus its rounding)
//!   above them. A neighbourhood weighted by 16 then still fits each field
//!   (`16 · 255 < 2^12`), so the weighted sums of both are one add; radius 1
//!   splits its 32 weights into two such halves (222 + 343 of a box row each
//!   side).
//! - **No filtered rows.** With `A = 256 − x` the filtered sample is
//!   `16 · src + ((Σ B − Σ x · src + round) >> 9)`, so the projection works
//!   on the difference `d` alone: `src + ((w0 · d0 + w1 · d1 + 2^10) >> 11)`,
//!   for one pass `src + ((w · d + 2^10) >> 11)`; both are exactly the
//!   scalar reference's blend ([`super::self_guided::blend_at`]).

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use nc_common::PlaneViewMut;

use super::AV1_LR_BORDER;
use super::self_guided::{RECIP_BITS, SgrStripe, SgrStripeScratch};

/// Box columns per step: sixteen 16-bit sums.
pub(super) const STEP: usize = 16;
/// Bits of the packed word below `B`: the field of `x`.
pub(super) const X_BITS: i32 = 12;
/// `B`'s rounding, added once per box (`2^3 · 32 = 2^8`, the rounding of
/// the shift by 9 after a neighbourhood weighted by 32), above the `x` field.
pub(super) const B_ROUND: i32 = (1 << (RECIP_BITS - 1)) + (1 << (X_BITS + 3));
/// Shift of an output between two radius-2 box rows, and of every radius-1
/// output (neighbourhood weights summing to 32, as `TWO_ROW_SHIFT`).
pub(super) const TWO_ROW_SHIFT: i32 = 9;
/// Shift of an output on a radius-2 box row (weights summing to 16).
pub(super) const ONE_ROW_SHIFT: i32 = 8;
/// The projection's shift (`SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS`).
pub(super) const BLEND_SHIFT: i32 = 11;

/// One row of horizontal (or running vertical) sums of [`STEP`]-column
/// steps: the sums in 16 bits, the squares in 32 (in an order of the ISA's
/// choosing within a step).
#[derive(Clone, Copy)]
pub(super) struct SumRow {
    pub(super) sum: *mut i16,
    pub(super) squares: *mut i32,
}

/// The `x` and `B` sums kept between output rows: the radius-2 565 rows
/// (`x`, `B`), or the radius-1 222 row and one 343 row (packed).
pub(super) type Kept = (*mut i32, *mut i32);

/// Which neighbourhood a single-pass output row takes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Single {
    /// Radius 2, between the kept box row and the new one.
    FiveBetween,
    /// Radius 2, on the kept box row.
    FiveOn,
    /// Radius 1.
    Three,
}

/// One ISA's per-step arithmetic of the stripe kernel. Rows of packed
/// words, `x` and `B` sums are in column order (shared layout); a
/// [`SumRow`]'s squares are in the ISA's own order.
///
/// # Safety
///
/// Every method executes the ISA's instructions: callers run inside the
/// ISA's `target_feature` entry point. Pointer arguments are rows carved
/// by [`run`] for the stripe, read and written as its contract states.
pub(super) trait StripeLanes {
    /// One step's box sums.
    type Sums: Copy;
    /// One radius's constants.
    type Stats: Copy;

    /// The constants of a radius with `count` (9 or 25) samples a box and
    /// scale `s` (below 2^12).
    unsafe fn statistics(count: i32, scale: u32) -> Self::Stats;
    /// Step `k` of `row`.
    unsafe fn load(row: SumRow, k: usize) -> Self::Sums;
    /// Store step `k` of `row`.
    unsafe fn store(row: SumRow, k: usize, v: Self::Sums);
    unsafe fn add(a: Self::Sums, b: Self::Sums) -> Self::Sums;
    /// Radius-1 horizontal sums of one step: box column `c` of the step
    /// reads bytes `c + 1 ..= c + 3` from `src` (24 bytes are readable).
    unsafe fn sums3(src: *const u8) -> Self::Sums;
    /// Radius-2 sums of one step: bytes `c ..= c + 4`.
    unsafe fn sums5(src: *const u8) -> Self::Sums;
    /// Both radii's sums of one step, from one load.
    unsafe fn sums35(src: *const u8) -> (Self::Sums, Self::Sums);
    /// Store the packed `x | B << 12` words of one step's box sums at step
    /// `k` of `dst` (AV1 §7.17.3 `p`, `z`, `a2`, `b2`).
    unsafe fn store_packed(stats: Self::Stats, sums: Self::Sums, dst: *mut i32, k: usize);
    /// Keep a radius-1 box row's 222 and 343 rows (the first two box rows,
    /// which have no output row of their own).
    unsafe fn radius1_start(packed: *const i32, kept: Kept, w: usize);
    /// Keep the first radius-2 box row's 565 sums.
    unsafe fn radius2_start(packed: *const i32, kept: Kept, w: usize);
    /// One output row of a mixed set (radius 2 between box rows when
    /// `BETWEEN`, else on one): `src + ((w0 · d5 + w1 · d3 + 2^10) >> 11)`.
    unsafe fn mix_row<const BETWEEN: bool>(
        src: &[u8],
        dst: &mut [u8],
        five: (*const i32, Kept),
        three: (*const i32, Kept),
        weights: [i32; 2],
    );
    /// One output row of a single-pass set: `src + ((w · d + 2^10) >> 11)`.
    unsafe fn single_row(
        src: &[u8],
        dst: &mut [u8],
        weight: i32,
        neighbourhood: (Single, *const i32, Kept),
    );
}

// ------------------------------------------------------------- layout ----

/// The rows of the stripe's scratch, carved from the caller's buffers.
struct Layout {
    /// 16-column steps of box columns (`-1 ..= w`).
    steps: usize,
    /// Values per row: the steps and a margin for the neighbourhood's reads
    /// one and two columns on.
    row: usize,
}

/// 16-bit rows: radius-1 sums (2), radius-2 sums (3).
const WORD_ROWS: usize = 5;
/// 32-bit rows: the squares of those (5), packed radius-2 statistics (1),
/// packed radius-1 statistics (2), the radius-2 neighbourhood's `x` and
/// `B` sums (2), the radius-1 222 row (1) and 343 rows (2).
const DWORD_ROWS: usize = 13;

impl Layout {
    fn new(w: usize) -> Self {
        let steps = (w + 2).div_ceil(STEP);
        Self {
            steps,
            row: steps * STEP + STEP,
        }
    }
}

/// Pointers to the rows of one stripe.
struct Rows {
    sums3: [SumRow; 2],
    sums5: [SumRow; 3],
    packed5: *mut i32,
    packed3: [*mut i32; 2],
    x565: *mut i32,
    b565: *mut i32,
    ab222: *mut i32,
    ab343: [*mut i32; 2],
}

impl Rows {
    fn carve(layout: &Layout, scratch: &mut SgrStripeScratch) -> Self {
        let words = super::grow(&mut scratch.words, WORD_ROWS * layout.row).as_mut_ptr();
        let dwords = super::grow(&mut scratch.dwords, DWORD_ROWS * layout.row).as_mut_ptr();
        // `wrapping_add` keeps this safe; every offset is inside the
        // buffers just grown (row index < WORD_ROWS / DWORD_ROWS).
        let word = |i: usize| words.wrapping_add(i * layout.row);
        let dword = |i: usize| dwords.wrapping_add(i * layout.row);
        let sum_row = |i: usize, j: usize| SumRow {
            sum: word(i),
            squares: dword(j),
        };
        Self {
            sums3: [sum_row(0, 0), sum_row(1, 1)],
            sums5: [sum_row(2, 2), sum_row(3, 3), sum_row(4, 4)],
            packed5: dword(5),
            packed3: [dword(6), dword(7)],
            x565: dword(8),
            b565: dword(9),
            ab222: dword(10),
            ab343: [dword(11), dword(12)],
        }
    }
}

// ------------------------------------------------------ window rows ----

/// What a window row does to a radius's running sums.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Nothing (the radius does not run, or the row is not in its boxes).
    Skip,
    /// Start a running sum: `target = row`.
    Store,
    /// Complete a radius-2 row pair: `target += row`.
    Add,
    /// Complete a box (`row + running[0] + running[1]`) into its packed
    /// statistics, and keep the row in `target`.
    Box,
}

/// One radius's work on a window row.
#[derive(Clone, Copy)]
struct RowWork {
    step: Step,
    target: SumRow,
    running: [SumRow; 2],
    packed: *mut i32,
}

impl RowWork {
    const SKIP: Self = Self {
        step: Step::Skip,
        target: SumRow {
            sum: core::ptr::null_mut(),
            squares: core::ptr::null_mut(),
        },
        running: [SumRow {
            sum: core::ptr::null_mut(),
            squares: core::ptr::null_mut(),
        }; 2],
        packed: core::ptr::null_mut(),
    };

    #[inline(always)]
    unsafe fn apply<V: StripeLanes>(self, k: usize, row: V::Sums, statistics: V::Stats) {
        // SAFETY: the rows were carved for this stripe; `k` is a step.
        unsafe {
            match self.step {
                Step::Skip => {}
                Step::Store => V::store(self.target, k, row),
                Step::Add => V::store(self.target, k, V::add(row, V::load(self.target, k))),
                Step::Box => {
                    let total = V::add(
                        V::add(row, V::load(self.running[0], k)),
                        V::load(self.running[1], k),
                    );
                    V::store(self.target, k, row);
                    V::store_packed(statistics, total, self.packed, k);
                }
            }
        }
    }
}

/// One window row (`src` at box column −1's radius-2 tap 0) through both
/// radii's work: the horizontal sums, then each radius's running sums and
/// box statistics.
#[inline(always)]
unsafe fn window_row<V: StripeLanes>(
    src: *const u8,
    steps: usize,
    (three, five): (RowWork, RowWork),
    (stat3, stat5): (V::Stats, V::Stats),
) {
    // SAFETY: the ISA is enabled in the calling kernel; the row holds
    // `steps · 16 + 8` bytes and the overread.
    unsafe {
        for k in 0..steps {
            let bytes = src.add(k * STEP);
            match (three.step == Step::Skip, five.step == Step::Skip) {
                (false, false) => {
                    let (row3, row5) = V::sums35(bytes);
                    three.apply::<V>(k, row3, stat3);
                    five.apply::<V>(k, row5, stat5);
                }
                (false, true) => three.apply::<V>(k, V::sums3(bytes), stat3),
                (true, false) => five.apply::<V>(k, V::sums5(bytes), stat5),
                (true, true) => {}
            }
        }
    }
}

// ----------------------------------------------------------- stripe ----

/// Self-guided-filter a `w`x`h` region of a stripe window (the contract of
/// [`super::self_guided::self_guided`]) into `dst` at `(x, y)` with the
/// ISA's lanes `V`.
///
/// # Safety
///
/// `V`'s ISA must be enabled in the caller. Extents are checked here.
#[inline(always)]
pub(super) unsafe fn run<V: StripeLanes>(
    params: SgrStripe,
    (window, stride): (&[u8], usize),
    (w, h): (usize, usize),
    scratch: &mut SgrStripeScratch,
    (dst, x, y): (&mut PlaneViewMut<'_>, usize, usize),
) {
    let layout = Layout::new(w);
    // Bytes read from a row's start: the steps' 24-byte loads, and the
    // projection's source loads (16 bytes per step from column 3). Rows
    // may read on into the next; the last must end inside the window.
    let row_bytes = (layout.steps * STEP + 8).max(AV1_LR_BORDER + w.next_multiple_of(STEP));
    assert!(w > 0 && h > 0 && stride >= w + 2 * AV1_LR_BORDER);
    assert!(window.len() >= (h + 2 * AV1_LR_BORDER - 1) * stride + row_bytes);
    let rows = Rows::carve(&layout, scratch);
    let stripe = Stripe {
        window,
        stride,
        w,
        h,
        steps: layout.steps,
    };
    let [s5, s3] = params.scales;
    // SAFETY: the ISA is enabled (caller contract); every row pointer was
    // carved for `layout`, the window covers every row read (checked
    // above), and each output row is borrowed alone.
    unsafe {
        let (stat5, stat3) = (V::statistics(25, s5), V::statistics(9, s3));
        let out = |dst: &mut PlaneViewMut<'_>, i: usize| -> *mut [u8] {
            core::ptr::from_mut(&mut dst.row_mut(y + i)[x..x + w])
        };
        match (s5 > 0, s3 > 0) {
            (true, true) => {
                stripe.mix::<V>(&rows, (stat3, stat5), params.weights, |i| &mut *out(dst, i));
            }
            (true, false) => {
                stripe.five::<V>(&rows, stat5, params.weights[0], |i| &mut *out(dst, i));
            }
            (false, true) => {
                stripe.three::<V>(&rows, stat3, params.weights[1], |i| &mut *out(dst, i));
            }
            (false, false) => {}
        }
    }
}

/// The stripe window being filtered.
struct Stripe<'a> {
    window: &'a [u8],
    stride: usize,
    w: usize,
    h: usize,
    steps: usize,
}

impl Stripe<'_> {
    /// Window row of region row `r` (−3 ..= h + 2), from box column −1's
    /// radius-2 tap 0 (window column 0).
    #[inline(always)]
    fn row_ptr(&self, r: isize) -> *const u8 {
        let t = (r + AV1_LR_BORDER as isize) as usize;
        self.window[t * self.stride..].as_ptr()
    }

    /// Region row `i`'s own samples (the projection's source).
    #[inline(always)]
    fn source(&self, i: usize) -> &[u8] {
        let start = (i + AV1_LR_BORDER) * self.stride + AV1_LR_BORDER;
        &self.window[start..start + self.w.next_multiple_of(STEP)]
    }

    #[inline(always)]
    unsafe fn rows<V: StripeLanes>(
        &self,
        r: isize,
        work: (RowWork, RowWork),
        stats: (V::Stats, V::Stats),
    ) {
        // SAFETY: the caller's row is in the window.
        unsafe { window_row::<V>(self.row_ptr(r), self.steps, work, stats) }
    }

    /// Radius 2 only (sets 14 and 15): box rows −1, 1, 3, …; the output
    /// rows either side of each.
    #[inline(always)]
    unsafe fn five<'d, V: StripeLanes>(
        &self,
        rows: &Rows,
        stat: V::Stats,
        weight: i32,
        mut dst: impl FnMut(usize) -> &'d mut [u8],
    ) {
        let five = |step, target, running, packed| {
            (
                RowWork::SKIP,
                RowWork {
                    step,
                    target,
                    running,
                    packed,
                },
            )
        };
        let [mut pair_old, mut pair_new, mut single] = rows.sums5;
        let sums565 = (rows.x565, rows.b565);
        // SAFETY: rows carved for this stripe, window rows −3 ..= h + 2.
        unsafe {
            let none = V::statistics(9, 0);
            for (r, step, target) in [
                (-3, Step::Store, pair_old),
                (-2, Step::Add, pair_old),
                (-1, Step::Store, pair_new),
                (0, Step::Add, pair_new),
            ] {
                self.rows::<V>(
                    r,
                    five(step, target, [target; 2], rows.packed5),
                    (none, stat),
                );
            }
            self.rows::<V>(
                1,
                five(Step::Box, single, [pair_old, pair_new], rows.packed5),
                (none, stat),
            );
            V::radius2_start(rows.packed5, sums565, self.w);
            for m in 1..=self.h.div_ceil(2) {
                // single (row 2m − 1) + row 2m: the pair of box row 2m + 1.
                let r = 2 * m as isize;
                self.rows::<V>(
                    r,
                    five(Step::Add, single, [single; 2], rows.packed5),
                    (none, stat),
                );
                let next = pair_old;
                self.rows::<V>(
                    r + 1,
                    five(Step::Box, next, [pair_new, single], rows.packed5),
                    (none, stat),
                );
                (pair_old, pair_new, single) = (pair_new, single, next);
                let i = 2 * m - 2;
                let between = (Single::FiveBetween, rows.packed5.cast_const(), sums565);
                V::single_row(self.source(i), dst(i), weight, between);
                if i + 1 < self.h {
                    let on = (Single::FiveOn, rows.packed5.cast_const(), sums565);
                    V::single_row(self.source(i + 1), dst(i + 1), weight, on);
                }
            }
        }
    }

    /// Radius 1 only (sets 10 to 13): box rows −1 ..= h, each output row
    /// from the box rows above, on and below.
    #[inline(always)]
    unsafe fn three<'d, V: StripeLanes>(
        &self,
        rows: &Rows,
        stat: V::Stats,
        weight: i32,
        mut dst: impl FnMut(usize) -> &'d mut [u8],
    ) {
        let three = |step, target, running| {
            (
                RowWork {
                    step,
                    target,
                    running,
                    packed: rows.packed3[0],
                },
                RowWork::SKIP,
            )
        };
        let [mut older, mut newer] = rows.sums3;
        // SAFETY: rows carved for this stripe, window rows −2 ..= h + 1.
        unsafe {
            let none = V::statistics(25, 0);
            self.rows::<V>(-2, three(Step::Store, older, [older; 2]), (stat, none));
            self.rows::<V>(-1, three(Step::Store, newer, [newer; 2]), (stat, none));
            for r in -1..=self.h as isize {
                // Box row r: rows r − 1 ..= r + 1; row r + 1 replaces r − 1.
                self.rows::<V>(r + 1, three(Step::Box, older, [older, newer]), (stat, none));
                (older, newer) = (newer, older);
                // Box row r's 343 row goes to slot r & 1.
                let ab = (rows.ab222, rows.ab343[(r & 1) as usize]);
                if r < 1 {
                    V::radius1_start(rows.packed3[0], ab, self.w);
                    continue;
                }
                let i = (r - 1) as usize;
                let around = (Single::Three, rows.packed3[0].cast_const(), ab);
                V::single_row(self.source(i), dst(i), weight, around);
            }
        }
    }

    /// Both radii (sets 0 to 9): per pair of window rows, the radius-1 box
    /// rows they complete and one radius-2 box row, then two output rows.
    #[inline(always)]
    unsafe fn mix<'d, V: StripeLanes>(
        &self,
        rows: &Rows,
        stats: (V::Stats, V::Stats),
        weights: [i32; 2],
        mut dst: impl FnMut(usize) -> &'d mut [u8],
    ) {
        let work = |step, target, running, packed| RowWork {
            step,
            target,
            running,
            packed,
        };
        let [mut pair_old, mut pair_new, mut single] = rows.sums5;
        let sums565 = (rows.x565, rows.b565);
        let [packed_even, packed_odd] = rows.packed3;
        // Box row r (−1 ..= h) of radius 1 from window row r + 1, into
        // `packed`; the row replaces r − 1.
        let three = |sums: &mut [SumRow; 2], packed| {
            let [older, newer] = *sums;
            *sums = [newer, older];
            work(Step::Box, older, [older, newer], packed)
        };
        let mut sums3 = rows.sums3;
        // SAFETY: rows carved for this stripe, window rows −3 ..= h + 2.
        unsafe {
            self.rows::<V>(
                -3,
                (
                    RowWork::SKIP,
                    work(Step::Store, pair_old, [pair_old; 2], rows.packed5),
                ),
                stats,
            );
            let [older, newer] = sums3;
            self.rows::<V>(
                -2,
                (
                    work(Step::Store, older, [older; 2], packed_even),
                    work(Step::Add, pair_old, [pair_old; 2], rows.packed5),
                ),
                stats,
            );
            self.rows::<V>(
                -1,
                (
                    work(Step::Store, newer, [newer; 2], packed_even),
                    work(Step::Store, pair_new, [pair_new; 2], rows.packed5),
                ),
                stats,
            );
            self.rows::<V>(
                0,
                (
                    three(&mut sums3, packed_odd),
                    work(Step::Add, pair_new, [pair_new; 2], rows.packed5),
                ),
                stats,
            );
            V::radius1_start(packed_odd, (rows.ab222, rows.ab343[1]), self.w);
            self.rows::<V>(
                1,
                (
                    three(&mut sums3, packed_even),
                    work(Step::Box, single, [pair_old, pair_new], rows.packed5),
                ),
                stats,
            );
            V::radius1_start(packed_even, (rows.ab222, rows.ab343[0]), self.w);
            V::radius2_start(rows.packed5, sums565, self.w);
            for m in 1..=self.h.div_ceil(2) {
                let r = 2 * m as isize;
                // Box rows 2m − 1 (radius 1, odd: output row 2m − 2) and
                // 2m − 1 of radius 2's pair.
                self.rows::<V>(
                    r,
                    (
                        three(&mut sums3, packed_odd),
                        work(Step::Add, single, [single; 2], rows.packed5),
                    ),
                    stats,
                );
                let next = pair_old;
                self.rows::<V>(
                    r + 1,
                    (
                        three(&mut sums3, packed_even),
                        work(Step::Box, next, [pair_new, single], rows.packed5),
                    ),
                    stats,
                );
                (pair_old, pair_new, single) = (pair_new, single, next);
                let i = 2 * m - 2;
                V::mix_row::<true>(
                    self.source(i),
                    dst(i),
                    (rows.packed5, sums565),
                    (packed_odd, (rows.ab222, rows.ab343[1])),
                    weights,
                );
                if i + 1 < self.h {
                    V::mix_row::<false>(
                        self.source(i + 1),
                        dst(i + 1),
                        (rows.packed5, sums565),
                        (packed_even, (rows.ab222, rows.ab343[0])),
                        weights,
                    );
                }
            }
        }
    }
}
