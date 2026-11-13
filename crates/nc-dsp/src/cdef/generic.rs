//! The CDEF block filter written once over [`I16FilterLanes`]: `ROWS` block
//! rows per vector (one 8-sample or two 4- or 8-sample rows). All arithmetic
//! fits 16 bits: padded differences wrap harmlessly (module docs), and the
//! weighted sum of constrained differences is at most
//! `4 * 15 * 2 + 2 * 15 * 2 + 3 * 4 * 4`.
//!
//! The direction is a const parameter, so every tap is a constant row and
//! column of the block's window. Validate the complete source/destination
//! footprints once, then load exact-length spans with the shared lane API.

use super::{CDEF_PADDING, CdefArgs, DIRECTIONS, SEC_TAPS};
use crate::simd::I16FilterLanes;

/// A block's source window: `S` rows of `S` samples, the block plus
/// [`CDEF_PADDING`] on every side.
struct Window<'a, const S: usize> {
    samples: &'a [i16],
    stride: usize,
}

/// The `ROWS` block rows starting at window row `row`, from window column
/// `col`.
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
fn load<V: I16FilterLanes, const ROWS: usize, const S: usize>(
    window: &Window<'_, S>,
    row: usize,
    col: usize,
) -> V {
    let width = V::LANES / ROWS;
    debug_assert!(row + ROWS <= S && col + width <= S);
    // SAFETY: filter's loop and constant direction offsets keep these ROWS
    // rows and width columns within the complete window checked once below.
    // Exact-length slices let the ISA's existing loads omit redundant checks.
    unsafe {
        let first = window.samples.as_ptr().add(row * window.stride + col);
        let lo = core::slice::from_raw_parts(first, width);
        if ROWS == 1 {
            V::load(lo)
        } else {
            let hi = core::slice::from_raw_parts(first.add(window.stride), width);
            V::load_halves(lo, hi)
        }
    }
}

/// libaom `constrain` on a vector of differences, for a non-zero strength
/// `threshold` and its precomputed shift. The unsigned operations make
/// padded taps contribute zero (module docs).
#[allow(clippy::inline_always)]
#[inline(always)]
fn constrain<V: I16FilterLanes>(diff: V, threshold: V, shift: u32) -> V {
    let magnitude = diff.abs();
    magnitude
        .min_u16(threshold.sub_sat_u16(magnitude.sra(shift)))
        .with_sign_of(diff)
}

/// The clipping range of a row: the minimum and maximum of every tap that
/// is not padding (the centre sample included).
#[derive(Clone, Copy)]
struct Range<V> {
    lo: V,
    hi: V,
}

impl<V: I16FilterLanes> Range<V> {
    /// Widen by tap `p`; padding loses both comparisons (module docs).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn take(&mut self, p: V) {
        self.lo = self.lo.min_u16(p);
        self.hi = self.hi.max(p);
    }
}

/// The sum of the constrained differences of the taps at `±(dr, dc)` from
/// the rows at window row `row` (centre `x`), widening `range` when given.
#[allow(clippy::inline_always)]
#[inline(always)]
fn tap_pair<V: I16FilterLanes, const ROWS: usize, const S: usize>(
    window: &Window<'_, S>,
    row: usize,
    (dr, dc): (isize, isize),
    x: V,
    strength: V,
    shift: u32,
    range: Option<&mut Range<V>>,
) -> V {
    let at = |r: isize, c: isize| {
        (
            row.wrapping_add_signed(r),
            CDEF_PADDING.wrapping_add_signed(c),
        )
    };
    let (r0, c0) = at(dr, dc);
    let (r1, c1) = at(-dr, -dc);
    let p0 = load::<V, ROWS, S>(window, r0, c0);
    let p1 = load::<V, ROWS, S>(window, r1, c1);
    if let Some(range) = range {
        range.take(p0);
        range.take(p1);
    }
    constrain(p0.sub(x), strength, shift).add(constrain(p1.sub(x), strength, shift))
}

/// Filter one `S - 4` square block in direction `DIR` (module contract:
/// `cdef` docs). `origin` indexes the block's top-left sample in `src`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn filter<
    V: I16FilterLanes,
    const ROWS: usize,
    const S: usize,
    const DIR: usize,
    const USE_PRI: bool,
    const USE_SEC: bool,
>(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let n = S - 2 * CDEF_PADDING;
    let top = origin
        .checked_sub(CDEF_PADDING * stride + CDEF_PADDING)
        .expect("CDEF window inside the padded source");
    let window = Window::<S> {
        samples: &src[top..][..(S - 1) * stride + S],
        stride,
    };
    let dst = &mut dst[..(n - 1) * dst_stride + n];
    debug_assert_eq!(V::LANES, n * ROWS);
    let clip = USE_PRI && USE_SEC;
    let (pri, sec) = (V::splat(args.pri), V::splat(args.sec));
    let pri_taps = [V::splat(args.pri_taps[0]), V::splat(args.pri_taps[1])];
    let pri_dir = DIRECTIONS[DIR];
    let sec_dirs = [DIRECTIONS[(DIR + 2) & 7], DIRECTIONS[(DIR + 6) & 7]];
    for r in (0..n).step_by(ROWS) {
        let row = r + CDEF_PADDING;
        let x = load::<V, ROWS, S>(&window, row, CDEF_PADDING);
        let mut range = Range { lo: x, hi: x };
        let mut sum = V::splat(0);
        if USE_PRI {
            for k in 0..2 {
                let taps = tap_pair::<V, ROWS, S>(
                    &window,
                    row,
                    pri_dir[k],
                    x,
                    pri,
                    args.pri_shift,
                    clip.then_some(&mut range),
                );
                sum = sum.add(taps.mul(pri_taps[k]));
            }
        }
        if USE_SEC {
            // Weights SEC_TAPS = [2, 1]: the k = 0 sum is doubled.
            debug_assert_eq!(SEC_TAPS, [2, 1]);
            let mut sec_sum = [V::splat(0); 2];
            for (k, sec_k) in sec_sum.iter_mut().enumerate() {
                for dirs in &sec_dirs {
                    *sec_k = sec_k.add(tap_pair::<V, ROWS, S>(
                        &window,
                        row,
                        dirs[k],
                        x,
                        sec,
                        args.sec_shift,
                        clip.then_some(&mut range),
                    ));
                }
            }
            sum = sum.add(sec_sum[0].add(sec_sum[0])).add(sec_sum[1]);
        }
        // libaom: x + ((8 + sum - (sum < 0)) >> 4); `sum >> 15` is -1 when
        // negative.
        let mut y = x.add(sum.add(V::splat(8)).add(sum.sra(15)).sra(4));
        if clip {
            y = y.max(range.lo).min(range.hi);
        }
        // SAFETY: the destination footprint was checked once; each vector
        // writes ROWS disjoint rows of n pixels and never touches padding.
        unsafe {
            let first = dst.as_mut_ptr().add(r * dst_stride);
            if ROWS == 1 {
                y.store_u8_sat_at(first);
            } else {
                let lo = core::slice::from_raw_parts_mut(first, n);
                let hi = core::slice::from_raw_parts_mut(first.add(dst_stride), n);
                y.store_u8_sat_halves(lo, hi);
            }
        }
    }
}

/// [`filter`] for the block's direction `args.dir`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn filter_any_dir<V: I16FilterLanes, const ROWS: usize, const S: usize>(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // Select active taps once per block. Const cases eliminate per-tap
    // clipping masks and inner-loop branches visible in the ARM profile.
    // Five paired native joint-fast runs: crowd CPU -1.324%, bodycam +0.194%;
    // output is identical. The specialized NEON block8 grows10.5KiB to22.7KiB.
    // The arithmetic and clipping rules are unchanged; zero strengths are
    // still accepted by checkasm even though callers normally omit them.
    // SAFETY: each case forwards the caller's slice and SIMD contracts.
    unsafe {
        match (args.pri != 0, args.sec != 0) {
            (true, true) => {
                filter_dirs::<V, ROWS, S, true, true>(src, stride, origin, args, dst, dst_stride);
            }
            (true, false) => {
                filter_dirs::<V, ROWS, S, true, false>(src, stride, origin, args, dst, dst_stride);
            }
            (false, true) => {
                filter_dirs::<V, ROWS, S, false, true>(src, stride, origin, args, dst, dst_stride);
            }
            (false, false) => {
                filter_dirs::<V, ROWS, S, false, false>(src, stride, origin, args, dst, dst_stride);
            }
        }
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn filter_dirs<
    V: I16FilterLanes,
    const ROWS: usize,
    const S: usize,
    const USE_PRI: bool,
    const USE_SEC: bool,
>(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // Direct calls, not function pointers: each body must inline into the
    // caller's `#[target_feature]` function.
    // SAFETY: every arm forwards the caller's contract unchanged.
    unsafe {
        match args.dir {
            0 => filter::<V, ROWS, S, 0, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
            1 => filter::<V, ROWS, S, 1, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
            2 => filter::<V, ROWS, S, 2, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
            3 => filter::<V, ROWS, S, 3, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
            4 => filter::<V, ROWS, S, 4, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
            5 => filter::<V, ROWS, S, 5, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
            6 => filter::<V, ROWS, S, 6, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
            _ => filter::<V, ROWS, S, 7, USE_PRI, USE_SEC>(
                src, stride, origin, args, dst, dst_stride,
            ),
        }
    }
}
