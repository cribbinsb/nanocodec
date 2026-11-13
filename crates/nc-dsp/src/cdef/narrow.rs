//! The 8-bit CDEF block filter for interior blocks, written once over
//! [`U8FilterLanes`]: sixteen samples per vector, two rows of an 8x8 luma
//! block or all four rows of a 4x4 chroma block.
//!
//! Every tap lies inside the plane (the kernel contract), so there is no
//! unavailable sample and the clipping range takes every tap. The
//! arithmetic follows dav1d's 8-bit approach (ideas only): the constraint is
//! computed on unsigned bytes, `min(|d|, strength - (|d| >> shift))` with a
//! saturating subtraction, which is at most 15; its sign is applied in the
//! byte (as `i8`); and only the weighted accumulation widens to 16 bits.
//! That equals libaom's `constrain` for every 8-bit difference, so the
//! result is bit-exact with the padded 16-bit filter (checkasm).

use super::{CDEF_PADDING, CdefArgs, DIRECTIONS};
use crate::simd::U8FilterLanes;

/// Secondary tap weights as bytes (`SEC_TAPS`).
const SEC_WEIGHTS: [i8; 2] = [2, 1];

/// A block's source window: `S` rows of `S` samples, the block plus
/// [`CDEF_PADDING`] on every side, checked once against the plane so the
/// taps load without further checks.
#[derive(Clone, Copy)]
struct Window<const S: usize> {
    /// The window's top-left sample.
    top_left: *const u8,
    stride: usize,
}

impl<const S: usize> Window<S> {
    /// The window of the block at `origin` in `src`.
    ///
    /// # Panics
    ///
    /// If the window leaves `src`.
    #[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
    #[inline(always)]
    fn new(src: &[u8], stride: usize, origin: usize) -> Self {
        let top = origin
            .checked_sub(CDEF_PADDING * stride + CDEF_PADDING)
            .expect("CDEF window inside the plane");
        assert!(
            top + (S - 1) * stride + S <= src.len(),
            "CDEF window inside the plane"
        );
        Self {
            // SAFETY: `top` is inside `src` (asserted).
            top_left: unsafe { src.as_ptr().add(top) },
            stride,
        }
    }

    /// The vector of block rows starting at window row `row`, from window
    /// column `col`: two rows of 8 (`N` = 8) or four rows of 4 (`N` = 4).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn load<V: U8FilterLanes, const N: usize>(self, row: usize, col: usize) -> V {
        debug_assert!(row + 16 / N <= S && col + N <= S);
        let at = |r: usize| {
            // SAFETY: rows `row..row + 16 / N` and columns `col..col + N`
            // lie inside the window, which lies inside the plane (`new`).
            unsafe { self.top_left.add((row + r) * self.stride + col) }
        };
        // SAFETY: each pointer starts N readable samples (above); the
        // kernel's ISA is enabled.
        unsafe {
            if N == 8 {
                V::load_8x2(at(0), at(1))
            } else {
                V::load_4x4([at(0), at(1), at(2), at(3)])
            }
        }
    }
}

/// `constrain(p - x)` of libaom for a non-zero `strength`, signed, as `i8`
/// lanes.
#[allow(clippy::inline_always)]
#[inline(always)]
fn constrain<V: U8FilterLanes>(p: V, x: V, strength: V, shift: V) -> V {
    let magnitude = p.abs_diff(x);
    magnitude
        .min(strength.sub_sat(magnitude.shr(shift)))
        .negate_where(x.gt(p))
}

/// The two taps at `±(dr, dc)` from the rows at window row `row`: the sum
/// of their constrained differences (as `i8`), and the clipping range
/// widened.
#[allow(
    clippy::inline_always,
    clippy::too_many_arguments,
    clippy::needless_pass_by_value
)]
#[inline(always)]
fn tap_pair<V: U8FilterLanes, const N: usize, const S: usize>(
    window: Window<S>,
    row: usize,
    (dr, dc): (isize, isize),
    x: V,
    strength: V,
    shift: V,
    range: Option<&mut (V, V)>,
) -> V {
    let at = |r: isize, c: isize| {
        (
            row.wrapping_add_signed(r),
            CDEF_PADDING.wrapping_add_signed(c),
        )
    };
    let (r0, c0) = at(dr, dc);
    let (r1, c1) = at(-dr, -dc);
    let p0 = window.load::<V, N>(r0, c0);
    let p1 = window.load::<V, N>(r1, c1);
    if let Some((lo, hi)) = range {
        *lo = lo.min(p0).min(p1);
        *hi = hi.max(p0).max(p1);
    }
    constrain(p0, x, strength, shift).add_wrapping(constrain(p1, x, strength, shift))
}

/// Filter one `N`-square interior block in direction `DIR`. `origin`
/// indexes the block's top-left sample in `src`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn filter<
    V: U8FilterLanes,
    const N: usize,
    const S: usize,
    const DIR: usize,
    const USE_PRI: bool,
    const USE_SEC: bool,
>(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let rows_per_vector = 16 / N;
    let window = Window::<S>::new(src, stride, origin);
    let clip = USE_PRI && USE_SEC;
    let pri = V::splat(args.pri as u8);
    let sec = V::splat(args.sec as u8);
    let (pri_shift, sec_shift) = (
        V::shift_count(args.pri_shift),
        V::shift_count(args.sec_shift),
    );
    let pri_weights = [args.pri_taps[0] as i8, args.pri_taps[1] as i8];
    let pri_dir = DIRECTIONS[DIR];
    let sec_dirs = [DIRECTIONS[(DIR + 2) & 7], DIRECTIONS[(DIR + 6) & 7]];
    for r in (0..N).step_by(rows_per_vector) {
        let row = r + CDEF_PADDING;
        let x = window.load::<V, N>(row, CDEF_PADDING);
        let mut range = (x, x);
        let mut acc = V::acc_zero();
        // Taps of one weight are summed as bytes before their one multiply:
        // a constrained difference is at most its strength (15 primary, 4
        // secondary), so the two primary taps of a weight sum to at most 30
        // and the four secondary ones to 16, within `i8`.
        if USE_PRI {
            for k in 0..2 {
                let taps = tap_pair::<V, N, S>(
                    window,
                    row,
                    pri_dir[k],
                    x,
                    pri,
                    pri_shift,
                    clip.then_some(&mut range),
                );
                acc = V::acc_mla(acc, taps, pri_weights[k]);
            }
        }
        if USE_SEC {
            for (k, &weight) in SEC_WEIGHTS.iter().enumerate() {
                let [first, second] = sec_dirs.map(|dirs| dirs[k]);
                let taps = tap_pair::<V, N, S>(
                    window,
                    row,
                    first,
                    x,
                    sec,
                    sec_shift,
                    clip.then_some(&mut range),
                )
                .add_wrapping(tap_pair::<V, N, S>(
                    window,
                    row,
                    second,
                    x,
                    sec,
                    sec_shift,
                    clip.then_some(&mut range),
                ));
                acc = V::acc_mla(acc, taps, weight);
            }
        }
        let mut y = x.add_rounded(acc);
        if clip {
            y = y.max(range.0).min(range.1);
        }
        let out = &mut dst[r * dst_stride..];
        if N == 8 {
            let (top, bottom) = out.split_at_mut(dst_stride);
            y.store_8x2(top, bottom);
        } else {
            y.store_4x4(out, dst_stride);
        }
    }
}

/// [`filter`] for the block's direction and active strengths.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn filter_any<V: U8FilterLanes, const N: usize, const S: usize>(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: each case forwards the caller's slice and SIMD contracts.
    unsafe {
        match (args.pri != 0, args.sec != 0) {
            (true, true) => {
                filter_dirs::<V, N, S, true, true>(src, stride, origin, args, dst, dst_stride);
            }
            (true, false) => {
                filter_dirs::<V, N, S, true, false>(src, stride, origin, args, dst, dst_stride);
            }
            (false, true) => {
                filter_dirs::<V, N, S, false, true>(src, stride, origin, args, dst, dst_stride);
            }
            (false, false) => {
                filter_dirs::<V, N, S, false, false>(src, stride, origin, args, dst, dst_stride);
            }
        }
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn filter_dirs<
    V: U8FilterLanes,
    const N: usize,
    const S: usize,
    const USE_PRI: bool,
    const USE_SEC: bool,
>(
    src: &[u8],
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
            0 => filter::<V, N, S, 0, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
            1 => filter::<V, N, S, 1, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
            2 => filter::<V, N, S, 2, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
            3 => filter::<V, N, S, 3, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
            4 => filter::<V, N, S, 4, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
            5 => filter::<V, N, S, 5, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
            6 => filter::<V, N, S, 6, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
            _ => filter::<V, N, S, 7, USE_PRI, USE_SEC>(src, stride, origin, args, dst, dst_stride),
        }
    }
}
