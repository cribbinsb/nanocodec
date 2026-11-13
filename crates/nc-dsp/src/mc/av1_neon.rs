//! NEON (dot-product) AV1 prediction of every filter family, at final
//! 8-bit precision (`put`) and at compound precision (`prep`), in the
//! structure of dav1d's `put_8tap` / `prep_8tap` (`src/arm/64/mc.S` and
//! `mc_dotprod.S`, BSD-2-Clause, Copyright (c) 2018, `VideoLAN` and dav1d
//! authors; see `THIRD_PARTY.md`), reimplemented: the horizontal pass is
//! signed dot products of bias-flipped pixels ([`DotFilter`]), the vertical
//! pass widening multiply-accumulates of 16-bit rows, and every rounding a
//! rounding shift.
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
//! Kernel contract: see the `mc` module docs.

use super::neon_dotprod::DotFilter;
use super::{
    AnyTaps, Av1Family, Av1MixedFns, ConvolveFn, OuterTaps, PrepFn, TAP_CENTRE, TAPS, TapPair,
    halved, slide,
};
use core::arch::aarch64::{
    int16x8_t, int32x4_t, vcombine_s16, vdup_n_u32, vdupq_n_s16, vdupq_n_s32, vget_high_s16,
    vget_lane_u32, vget_low_s16, vld1_u8, vmlal_n_s16, vmlaq_n_s16, vmovl_u8, vqmovun_s16,
    vreinterpret_u8_u32, vreinterpret_u32_u8, vreinterpretq_s16_u16, vrshrn_n_s32, vrshrq_n_s16,
    vshlq_n_s16, vst1_s16, vst1_u8, vst1q_s16,
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
    /// `t` of a horizontal-only filter.
    unsafe fn horizontal(&mut self, row: usize, col: usize, t: int16x8_t);
    /// The unrounded sum of a vertical-only filter.
    unsafe fn vertical(&mut self, row: usize, col: usize, sum: int16x8_t);
    /// The vertical accumulators (outputs 0..4, 4..8) of a 2-D filter.
    unsafe fn both(&mut self, row: usize, col: usize, low: int32x4_t, high: int32x4_t);
    /// The samples of a whole-sample vector.
    unsafe fn copy(&mut self, row: usize, col: usize, samples: int16x8_t);
}

/// Final 8-bit samples.
struct Put<'a, const W: usize> {
    dst: &'a mut [u8],
    stride: usize,
}

impl<const W: usize> Put<'_, W> {
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn store(&mut self, row: usize, col: usize, words: int16x8_t) {
        let bytes = vqmovun_s16(words);
        let at = row * self.stride + col;
        if W == 4 {
            let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(bytes));
            self.dst[at..at + 4].copy_from_slice(&packed.to_ne_bytes());
        } else {
            let out = &mut self.dst[at..at + LANES];
            // SAFETY: `out` is 8 writable bytes.
            unsafe { vst1_u8(out.as_mut_ptr(), bytes) };
        }
    }
}

impl<const W: usize> Output for Put<'_, W> {
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn horizontal(&mut self, row: usize, col: usize, t: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, vrshrq_n_s16::<4>(t)) };
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
        let words = vcombine_s16(vrshrn_n_s32::<10>(low), vrshrn_n_s32::<10>(high));
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, words) };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn copy(&mut self, row: usize, col: usize, samples: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, samples) };
    }
}

/// Compound-precision samples.
struct Prep<'a, const W: usize> {
    dst: &'a mut [i16],
    stride: usize,
}

impl<const W: usize> Prep<'_, W> {
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn store(&mut self, row: usize, col: usize, words: int16x8_t) {
        let at = row * self.stride + col;
        if W == 4 {
            let out = &mut self.dst[at..at + 4];
            // SAFETY: `out` is 4 writable `i16`s.
            unsafe { vst1_s16(out.as_mut_ptr(), vget_low_s16(words)) };
        } else {
            let out = &mut self.dst[at..at + LANES];
            // SAFETY: `out` is 8 writable `i16`s.
            unsafe { vst1q_s16(out.as_mut_ptr(), words) };
        }
    }
}

impl<const W: usize> Output for Prep<'_, W> {
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn horizontal(&mut self, row: usize, col: usize, t: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, t) };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn vertical(&mut self, row: usize, col: usize, sum: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, vrshrq_n_s16::<2>(sum)) };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn both(&mut self, row: usize, col: usize, low: int32x4_t, high: int32x4_t) {
        let words = vcombine_s16(vrshrn_n_s32::<6>(low), vrshrn_n_s32::<6>(high));
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, words) };
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn copy(&mut self, row: usize, col: usize, samples: int16x8_t) {
        // SAFETY: forwarded from the caller.
        unsafe { self.store(row, col, vshlq_n_s16::<4>(samples)) };
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

/// The `put` of a whole-sample block: rows of `W` samples copied from the
/// window's centre (no widening to the filters' 16-bit lanes and back).
#[inline]
fn copy_rows<const W: usize>(src: &[u8], ss: usize, dst: &mut [u8], ds: usize, h: usize) {
    for r in 0..h {
        let from = (r + TAP_CENTRE) * ss + TAP_CENTRE;
        dst[r * ds..r * ds + W].copy_from_slice(&src[from..from + W]);
    }
}

/// `N` (4 or 8) pixels of `row`, widened to 16 bits.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn pixels<const W: usize>(row: &[u8]) -> int16x8_t {
    let bytes = if W == 4 {
        let word = u32::from_le_bytes(row[..4].try_into().expect("4 samples"));
        vreinterpret_u8_u32(vdup_n_u32(word))
    } else {
        let row = &row[..LANES];
        // SAFETY: `row` is 8 readable bytes.
        unsafe { vld1_u8(row.as_ptr()) }
    };
    vreinterpretq_s16_u16(vmovl_u8(bytes))
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
        copy_rows::<W>(src, ss, dst, ds, h);
        return;
    }
    let mut out = Put::<W> { dst, stride: ds };
    let taps = || (F::taps(W, fx), F::taps(h, fy));
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, F>(src, ss, h, (fx, fy), taps, &mut out) };
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
    let mut out = Prep::<W> { dst, stride: ds };
    let taps = || (F::taps(W, fx), F::taps(h, fy));
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, F>(src, ss, h, (fx, fy), taps, &mut out) };
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
        copy_rows::<W>(src, ss, dst, ds, h);
        return;
    }
    let mut out = Put::<W> { dst, stride: ds };
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, AnyTaps>(src, ss, h, fracs, || taps, &mut out) };
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
    let mut out = Prep::<W> { dst, stride: ds };
    // SAFETY: dotprod is enabled; the window contract is the caller's.
    unsafe { filter::<W, _, AnyTaps>(src, ss, h, fracs, || taps, &mut out) };
}

/// The four passes of a `W`x`h` block (module table) with the taps `taps`
/// gives (asked only when the block is filtered), into `out`; `Z` says whether the vertical taps'
/// outer ones are zero.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn filter<'t, const W: usize, O: Output, Z: OuterTaps>(
    src: &[u8],
    ss: usize,
    h: usize,
    (fx, fy): (usize, usize),
    taps: impl FnOnce() -> TapPair<'t>,
    out: &mut O,
) {
    assert!(
        src.len() >= (h + TAPS - 2) * ss + W + TAPS - 1,
        "AV1 prediction window too small"
    );
    let step = W.min(LANES);
    // SAFETY: every row and column read lies inside the
    // window checked above; the outputs' slices are bounds-checked.
    unsafe {
        match (fx, fy) {
            (0, 0) => {
                for r in 0..h {
                    for c in (0..W).step_by(step) {
                        let row = &src[(r + TAP_CENTRE) * ss + TAP_CENTRE + c..];
                        out.copy(r, c, pixels::<W>(row));
                    }
                }
            }
            (_, 0) => {
                let taps = DotFilter::new::<W>(&halved(taps().0));
                for r in 0..h {
                    for c in (0..W).step_by(step) {
                        let t = taps.row::<W, true>(&src[(r + TAP_CENTRE) * ss + c..]);
                        out.horizontal(r, c, t);
                    }
                }
            }
            (0, _) => {
                let taps = halved(taps().1);
                if Z::OUTER_ZERO {
                    vertical::<W, 6, O>(&src[ss..], ss, h, six_taps(&taps), out);
                } else {
                    vertical::<W, TAPS, O>(src, ss, h, &taps, out);
                }
            }
            _ => {
                let (h_taps, v_taps) = taps();
                let h_taps = DotFilter::new::<W>(&halved(h_taps));
                let v_taps = halved(v_taps);
                if Z::OUTER_ZERO {
                    both::<W, 6, O>(&src[ss..], ss, h, h_taps, six_taps(&v_taps), out);
                } else {
                    both::<W, TAPS, O>(src, ss, h, h_taps, &v_taps, out);
                }
            }
        }
    }
}

/// Vertical only: `N` taps of widened pixel rows from window row 0 of `src`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn vertical<const W: usize, const N: usize, O: Output>(
    src: &[u8],
    ss: usize,
    h: usize,
    taps: &[i16; N],
    out: &mut O,
) {
    for c in (0..W).step_by(W.min(LANES)) {
        // SAFETY: row 0 of the window from column c + 3, inside it.
        let mut rows = [unsafe { pixels::<W>(&src[TAP_CENTRE + c..]) }; N];
        for (k, row) in rows.iter_mut().enumerate().take(N - 1) {
            // SAFETY: row k of the window, inside it.
            *row = unsafe { pixels::<W>(&src[k * ss + TAP_CENTRE + c..]) };
        }
        for r in 0..h {
            // SAFETY: the last tap's row, inside the window.
            rows[N - 1] = unsafe { pixels::<W>(&src[(r + N - 1) * ss + TAP_CENTRE + c..]) };
            let mut sum = vdupq_n_s16(0);
            for (row, &tap) in rows.iter().zip(taps) {
                sum = vmlaq_n_s16(sum, *row, tap);
            }
            // SAFETY: output row r, columns c..c + 8.
            unsafe { out.vertical(r, c, sum) };
            slide(&mut rows, 1, N - 1);
        }
    }
}

/// Both directions: `N` vertical taps of rounded horizontal rows.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn both<const W: usize, const N: usize, O: Output>(
    src: &[u8],
    ss: usize,
    h: usize,
    h_taps: DotFilter,
    v_taps: &[i16; N],
    out: &mut O,
) {
    for c in (0..W).step_by(W.min(LANES)) {
        let row = |k: usize| {
            // SAFETY: row k of the window from column c, inside it.
            unsafe { h_taps.row::<W, true>(&src[k * ss + c..]) }
        };
        let mut rows = [row(0); N];
        for (k, slot) in rows.iter_mut().enumerate().take(N - 1).skip(1) {
            *slot = row(k);
        }
        for r in 0..h {
            rows[N - 1] = row(r + N - 1);
            let (mut low, mut high) = (vdupq_n_s32(0), vdupq_n_s32(0));
            for (&t, &tap) in rows.iter().zip(v_taps) {
                low = vmlal_n_s16(low, vget_low_s16(t), tap);
                if W != 4 {
                    high = vmlal_n_s16(high, vget_high_s16(t), tap);
                }
            }
            // SAFETY: output row r, columns c..c + 8.
            unsafe { out.both(r, c, low, high) };
            slide(&mut rows, 1, N - 1);
        }
    }
}

/// The six inner taps of a family whose outer taps are zero, as an array.
/// Taken as a slice, the vertical tap loops had a dynamic trip count (a
/// length check and loop per output row): -2.6% decode instructions and
/// -1.4% cycles on the Jetson, 1080p.
fn six_taps(taps: &[i16; TAPS]) -> &[i16; 6] {
    taps[1..7].try_into().expect("six inner taps")
}
