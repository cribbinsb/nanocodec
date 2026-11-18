//! Shared NEON (dot-product) AV1 and HEVC/VVC eight-tap prediction, at final
//! 8-bit precision (`put`) and at compound precision (`prep`), in the
//! structure of dav1d's `put_8tap` / `prep_8tap` (`src/arm/64/mc.S` and
//! `mc_dotprod.S`, BSD-2-Clause, Copyright (c) 2018/2024, `VideoLAN`, dav1d
//! authors, Janne Grunau, Martin Storsjo and Arm Limited; see `THIRD_PARTY.md`),
//! reimplemented: the horizontal pass is
//! signed dot products of bias-flipped pixels ([`DotFilter`]); V-only uses
//! transposed bytes and SDOT, while HV accumulates horizontal i16 rows into
//! i32 vertical sums. Rounding biases are folded into H-only accumulators.
//!
//! Halved taps (module `mc` docs). With `t = (sum + 2) >> 2` the
//! horizontal-pass rounding, the outputs are:
//!
//! | Pass | put (8-bit, clipped) | prep (compound, 16-bit) |
//! |---|---|---|
//! | horizontal only | `(t + 8) >> 4` | `t` |
//! | vertical only | `(sum + 32) >> 6` | `(sum + 2) >> 2` |
//! | both | `(Σ t·tap + 512) >> 10` | `(Σ t·tap + 32) >> 6` |
//! | neither | the sample | the sample `<< 4` |
//!
//! Families whose outer taps are zero filter six vertical taps from one row
//! down; the horizontal dot products always span all eight taps (zeros
//! included), so they read the same window. Installed only after runtime
//! dot-product detection.
//!
//! HEVC/VVC use full signed-byte taps, unrounded horizontal intermediates,
//! and six/twelve-bit output shifts in these same loops.
//!
//! Kernel contract: see the `mc` module docs.

mod horizontal_dot;
mod vertical_dot;

use super::neon_dotprod::{DOT_PIXEL_BIAS, DotFilter};
use super::{
    AnyTaps, Av1Family, Av1MixedFns, ConvolveFn, OuterTaps, PrepFn, SeparableTaps, TAP_CENTRE,
    TAPS, TapPair, halved, slide,
};
use core::arch::aarch64::{
    int16x8_t, int32x4_t, vcombine_s16, vdupq_n_s16, vdupq_n_s32, vget_high_s16, vget_lane_u32,
    vget_low_s16, vmlal_n_s16, vqmovun_s16, vreinterpret_u32_u8, vrshrn_n_s32, vrshrq_n_s16,
    vshrn_n_s32, vshrq_n_s16, vst1_s16, vst1_u8, vst1q_s16, vsubq_s16,
};

/// Lanes of one 16-bit vector.
const LANES: usize = 8;

/// The `put` kernels of family `F` by width class (4 to 64).
pub(crate) fn put_fns<F: Av1Family>() -> [ConvolveFn; 5] {
    [
        put::<F, 4>,
        put::<F, 8>,
        put::<F, 16>,
        put::<F, 32>,
        put::<F, 64>,
    ]
}

/// The `prep` kernels of family `F` by width class (4 to 64).
pub(crate) fn prep_fns<F: Av1Family>() -> [PrepFn; 5] {
    [
        prep::<F, 4>,
        prep::<F, 8>,
        prep::<F, 16>,
        prep::<F, 32>,
        prep::<F, 64>,
    ]
}

/// Where one pass's vectors of up to 8 outputs go.
trait Output {
    /// Pixel sign bias and the complete H-only rounding, as in dav1d.
    const HORIZONTAL_BIAS: i32;
    /// Validate the complete output once before pointer-based row stores.
    fn check(&self, width: usize, height: usize);
    /// The horizontal dot sum including its pixel and rounding biases.
    unsafe fn horizontal(&mut self, row: usize, col: usize, t: int16x8_t);
    /// The unrounded sum of a vertical-only filter.
    unsafe fn vertical(&mut self, row: usize, col: usize, sum: int16x8_t);
    /// The vertical accumulators (outputs 0..4, 4..8) of a 2-D filter.
    unsafe fn both(&mut self, row: usize, col: usize, low: int32x4_t, high: int32x4_t);
}

/// Final 8-bit samples.
struct Put<'a, const W: usize, const AV1: bool = true> {
    dst: &'a mut [u8],
    stride: usize,
}

impl<const W: usize, const AV1: bool> Put<'_, W, AV1> {
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn store(&mut self, row: usize, col: usize, words: int16x8_t) {
        let bytes = vqmovun_s16(words);
        // SAFETY: Output::check validated every row and its W samples.
        unsafe {
            let out = self.dst.as_mut_ptr().add(row * self.stride + col);
            if W == 4 {
                out.cast::<u32>()
                    .write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(bytes)));
            } else {
                vst1_u8(out, bytes);
            }
        }
    }
}

impl<const W: usize, const AV1: bool> Output for Put<'_, W, AV1> {
    const HORIZONTAL_BIAS: i32 = DOT_PIXEL_BIAS + if AV1 { 34 } else { 32 };
    #[inline]
    fn check(&self, width: usize, height: usize) {
        assert!(
            crate::block_copy::covers_rect(self.dst.len(), self.stride, width, height),
            "prediction output too small"
        );
    }

    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn horizontal(&mut self, row: usize, col: usize, t: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, vshrq_n_s16::<6>(t)) };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn vertical(&mut self, row: usize, col: usize, sum: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, vrshrq_n_s16::<6>(sum)) };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn both(&mut self, row: usize, col: usize, low: int32x4_t, high: int32x4_t) {
        let words = if AV1 {
            vcombine_s16(vrshrn_n_s32::<10>(low), vrshrn_n_s32::<10>(high))
        } else {
            vcombine_s16(vrshrn_n_s32::<12>(low), vrshrn_n_s32::<12>(high))
        };
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, words) };
    }
}

/// Compound-precision samples.
struct Prep<'a, const W: usize, const AV1: bool = true> {
    dst: &'a mut [i16],
    stride: usize,
}

impl<const W: usize, const AV1: bool> Prep<'_, W, AV1> {
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn store(&mut self, row: usize, col: usize, words: int16x8_t) {
        // SAFETY: Output::check validated every row and its W words.
        unsafe {
            let out = self.dst.as_mut_ptr().add(row * self.stride + col);
            if W == 4 {
                vst1_s16(out, vget_low_s16(words));
            } else {
                vst1q_s16(out, words);
            }
        }
    }
}

impl<const W: usize, const AV1: bool> Output for Prep<'_, W, AV1> {
    const HORIZONTAL_BIAS: i32 = if AV1 { DOT_PIXEL_BIAS + 2 } else { 0 };
    #[inline]
    fn check(&self, width: usize, height: usize) {
        assert!(
            crate::block_copy::covers_rect(self.dst.len(), self.stride, width, height),
            "prediction output too small"
        );
    }

    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn horizontal(&mut self, row: usize, col: usize, t: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, if AV1 { vshrq_n_s16::<2>(t) } else { t }) };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn vertical(&mut self, row: usize, col: usize, sum: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe {
            self.store(
                row,
                col,
                if AV1 {
                    vrshrq_n_s16::<2>(sum)
                } else {
                    vsubq_s16(sum, vdupq_n_s16(super::HEVC_PREP_OFFSET))
                },
            );
        };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn both(&mut self, row: usize, col: usize, low: int32x4_t, high: int32x4_t) {
        let words = if AV1 {
            vcombine_s16(vrshrn_n_s32::<6>(low), vrshrn_n_s32::<6>(high))
        } else {
            vsubq_s16(
                vcombine_s16(vshrn_n_s32::<6>(low), vshrn_n_s32::<6>(high)),
                vdupq_n_s16(super::HEVC_PREP_OFFSET),
            )
        };
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, words) };
    }
}

/// The mixed-family kernels (`mc::Av1MixedFn`) by width class (4 to 64).
pub(crate) fn mixed_fns() -> Av1MixedFns {
    Av1MixedFns {
        put: [
            put_mixed::<4>,
            put_mixed::<8>,
            put_mixed::<16>,
            put_mixed::<32>,
            put_mixed::<64>,
        ],
        prep: [
            prep_mixed::<4>,
            prep_mixed::<8>,
            prep_mixed::<16>,
            prep_mixed::<32>,
            prep_mixed::<64>,
        ],
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn put<F: Av1Family, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    if (fx, fy) == (0, 0) {
        // SAFETY: Dsp validated the tap window and output; NEON is enabled.
        unsafe {
            crate::block_copy::neon::copy::<false>(
                &src[TAP_CENTRE * (ss + 1)..],
                ss,
                dst,
                ds,
                W,
                h,
            );
        };
        return;
    }
    let mut out = Put::<W> { dst, stride: ds };
    let taps = || (F::taps(W, fx), F::taps(h, fy));
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, F, true>(src, ss, h, (fx, fy), taps, &mut out) };
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn prep<F: Av1Family, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    if (fx, fy) == (0, 0) {
        // SAFETY: the caller checked complete source/output footprints; NEON is enabled.
        unsafe {
            crate::block_copy::neon::prep_unbiased(
                &src[TAP_CENTRE * (ss + 1)..],
                ss,
                dst,
                ds,
                (W, h),
            );
        };
        return;
    }
    let mut out = Prep::<W> { dst, stride: ds };
    let taps = || (F::taps(W, fx), F::taps(h, fy));
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, F, true>(src, ss, h, (fx, fy), taps, &mut out) };
}

/// [`put`] with the caller's taps per direction.
#[target_feature(enable = "neon,dotprod")]
unsafe fn put_mixed<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    if fracs == (0, 0) {
        // SAFETY: Dsp validated the tap window and output; NEON is enabled.
        unsafe {
            crate::block_copy::neon::copy::<false>(
                &src[TAP_CENTRE * (ss + 1)..],
                ss,
                dst,
                ds,
                W,
                h,
            );
        };
        return;
    }
    let mut out = Put::<W> { dst, stride: ds };
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, AnyTaps, true>(src, ss, h, fracs, || taps, &mut out) };
}

/// [`prep`] with the caller's taps per direction.
#[target_feature(enable = "neon,dotprod")]
unsafe fn prep_mixed<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    if fracs == (0, 0) {
        // SAFETY: the caller checked complete source/output footprints; NEON is enabled.
        unsafe {
            crate::block_copy::neon::prep_unbiased(
                &src[TAP_CENTRE * (ss + 1)..],
                ss,
                dst,
                ds,
                (W, h),
            );
        };
        return;
    }
    let mut out = Prep::<W> { dst, stride: ds };
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, AnyTaps, true>(src, ss, h, fracs, || taps, &mut out) };
}

/// Eight-tap HEVC/VVC luma on the same passes as AV1.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn separable<T: SeparableTaps<8>, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    if (fx, fy) == (0, 0) {
        // SAFETY: the caller validated both complete footprints; NEON is enabled.
        unsafe {
            crate::block_copy::neon::copy::<false>(
                &src[TAP_CENTRE * (ss + 1)..],
                ss,
                dst,
                ds,
                W,
                h,
            );
        }
        return;
    }
    let mut out = Put::<W, false> { dst, stride: ds };
    // SAFETY: dotprod is enabled and the caller supplies complete windows.
    unsafe {
        filter::<W, _, AnyTaps, false>(
            src,
            ss,
            h,
            (fx, fy),
            || (&T::TABLE[fx], &T::TABLE[fy]),
            &mut out,
        );
    };
}

/// HEVC luma compound preparation on the shared eight-tap passes.
pub(crate) fn separable_prep_fns<T: SeparableTaps<8>>() -> [PrepFn; 5] {
    [
        separable_prep::<T, 4>,
        separable_prep::<T, 8>,
        separable_prep::<T, 16>,
        separable_prep::<T, 32>,
        separable_prep::<T, 64>,
    ]
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn separable_prep<T: SeparableTaps<8>, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    if (fx, fy) == (0, 0) {
        // SAFETY: the caller validated both footprints; NEON is enabled.
        unsafe {
            crate::block_copy::neon::prep(
                &src[TAP_CENTRE * (ss + 1)..],
                ss,
                dst,
                ds,
                (W, h),
                (6, super::HEVC_PREP_OFFSET),
            );
        }
        return;
    }
    let mut out = Prep::<W, false> { dst, stride: ds };
    // SAFETY: dotprod is enabled and complete source/output windows are supplied.
    unsafe {
        filter::<W, _, AnyTaps, false>(
            src,
            ss,
            h,
            (fx, fy),
            || (&T::TABLE[fx], &T::TABLE[fy]),
            &mut out,
        );
    }
}

/// AV1 halves even coefficients; HEVC/VVC retain full signed-byte taps.
#[inline]
fn coefficients<const AV1: bool>(taps: &[i16; TAPS]) -> [i16; TAPS] {
    if AV1 { halved(taps) } else { *taps }
}

/// The four passes of a `W`x`h` block (module table) with the taps `taps`
/// gives (asked only when the block is filtered), into `out`; `Z` says whether the vertical taps'
/// outer ones are zero.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn filter<'t, const W: usize, O: Output, Z: OuterTaps, const AV1: bool>(
    src: &[u8],
    ss: usize,
    h: usize,
    (fx, fy): (usize, usize),
    taps: impl FnOnce() -> TapPair<'t>,
    out: &mut O,
) {
    if h == 0 {
        return;
    }
    assert!(
        crate::block_copy::covers_rect(src.len(), ss, W + TAPS - 1, h + TAPS - 1),
        "AV1 prediction window too small"
    );
    out.check(W, h);
    // SAFETY: every row and column read lies inside the
    // window checked above; the outputs' slices are bounds-checked.
    unsafe {
        match (fx, fy) {
            (0, 0) => unreachable!("whole-pel predictions use block_copy"),
            (_, 0) => {
                horizontal_dot::filter::<W, O>(
                    src.as_ptr(),
                    ss,
                    h,
                    &coefficients::<AV1>(taps().0),
                    out,
                );
            }
            (0, _) => {
                let taps = coefficients::<AV1>(taps().1);
                vertical_dot::filter::<W, O>(src.as_ptr(), ss, h, &taps, out);
            }
            _ => {
                let (h_taps, v_taps) = taps();
                let h_taps = DotFilter::new(&coefficients::<AV1>(h_taps));
                let v_taps = coefficients::<AV1>(v_taps);
                if Z::OUTER_ZERO {
                    both_rows::<W, 6, O, AV1>(
                        src.as_ptr().add(ss),
                        ss,
                        h,
                        h_taps,
                        six_taps(&v_taps),
                        out,
                    );
                } else if W >= 16 {
                    both::<W, TAPS, O, AV1, 2>(src.as_ptr(), ss, h, h_taps, &v_taps, out);
                } else {
                    both::<W, TAPS, O, AV1, 1>(src.as_ptr(), ss, h, h_taps, &v_taps, out);
                }
            }
        }
    }
}

/// Both directions: `N` vertical taps of rounded horizontal rows.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn both<const W: usize, const N: usize, O: Output, const AV1: bool, const G: usize>(
    src: *const u8,
    ss: usize,
    h: usize,
    h_taps: DotFilter,
    v_taps: &[i16; N],
    out: &mut O,
) {
    for c in (0..W).step_by(W.min(LANES * G)) {
        let row = |k: usize| {
            // SAFETY: row k of the window from column c, inside it.
            unsafe {
                if k + 1 < h + N - 1 {
                    h_taps.rows::<W, G, AV1, true>(src.add(k * ss + c))
                } else {
                    h_taps.rows::<W, G, AV1, false>(src.add(k * ss + c))
                }
            }
        };
        let mut rows = [row(0); N];
        for (k, slot) in rows.iter_mut().enumerate().take(N - 1).skip(1) {
            *slot = row(k);
        }
        for r in 0..h {
            rows[N - 1] = row(r + N - 1);
            for g in 0..G {
                let (mut low, mut high) = (vdupq_n_s32(0), vdupq_n_s32(0));
                for (row, &tap) in rows.iter().zip(v_taps) {
                    low = vmlal_n_s16(low, vget_low_s16(row[g]), tap);
                    if W != 4 {
                        high = vmlal_n_s16(high, vget_high_s16(row[g]), tap);
                    }
                }
                // SAFETY: this group's four/eight outputs fit the footprint.
                unsafe { out.both(r, c + g * LANES, low, high) };
            }
            slide(&mut rows, 1, N - 1);
        }
    }
}

/// Six-tap 2-D filtering, two output rows per iteration. Only N-1 row
/// vectors move after the pair, rather than after each output. Wider
/// eight-tap passes use the measured paired-column schedule.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn both_rows<const W: usize, const N: usize, O: Output, const AV1: bool>(
    src: *const u8,
    ss: usize,
    h: usize,
    h_taps: DotFilter,
    v_taps: &[i16; N],
    out: &mut O,
) {
    for c in (0..W).step_by(W.min(LANES)) {
        let row = |r: usize| {
            // SAFETY: the complete source footprint covers this filter row.
            unsafe { h_taps.row::<W, AV1, true>(src.add(r * ss + c)) }
        };
        let mut rows = [row(0); TAPS + 1];
        for (r, value) in rows.iter_mut().enumerate().take(N - 1).skip(1) {
            *value = row(r);
        }
        for r in (0..h.saturating_sub(1)).step_by(2) {
            rows[N - 1] = row(r + N - 1);
            rows[N] = row(r + N);
            // SAFETY: both accumulations use N initialized vectors, within
            // the nine-vector array; the output rows are inside the footprint.
            unsafe {
                let (lo0, hi0) = row_sum::<W, N, 0>(&rows, v_taps);
                let (lo1, hi1) = row_sum::<W, N, 1>(&rows, v_taps);
                out.both(r, c, lo0, hi0);
                out.both(r + 1, c, lo1, hi1);
            }
            slide(&mut rows, 2, N - 1);
        }
        if !h.is_multiple_of(2) {
            rows[N - 1] = row(h + N - 2);
            // SAFETY: the remaining row and its N taps fit the same footprint.
            unsafe {
                let (lo, hi) = row_sum::<W, N, 0>(&rows, v_taps);
                out.both(h - 1, c, lo, hi);
            }
        }
    }
}

/// One vertical accumulator of a paired-row window, without runtime slices.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn row_sum<const W: usize, const N: usize, const START: usize>(
    rows: &[int16x8_t; TAPS + 1],
    taps: &[i16; N],
) -> (int32x4_t, int32x4_t) {
    let (mut low, mut high) = (vdupq_n_s32(0), vdupq_n_s32(0));
    for (k, &tap) in taps.iter().enumerate() {
        let t = rows[k + START];
        low = vmlal_n_s16(low, vget_low_s16(t), tap);
        if W != 4 {
            high = vmlal_n_s16(high, vget_high_s16(t), tap);
        }
    }
    (low, high)
}

/// The six inner taps of a family whose outer taps are zero, as an array.
/// Taken as a slice, the vertical tap loops had a dynamic trip count (a
/// length check and loop per output row): -2.6% decode instructions and
/// -1.4% cycles on the Jetson, 1080p.
fn six_taps(taps: &[i16; TAPS]) -> &[i16; 6] {
    taps[1..7].try_into().expect("six inner taps")
}
