//! Separable VP9 driver shared by ISA row filters; temporary samples are bytes.
//! Kernels are generic over the table's tap-sign layout ([`layout`]), so one
//! row kernel serves the regular, smooth, sharp and bilinear tables.
#![allow(clippy::inline_always)]
use super::MAX_SIDE;
use crate::vp9_mc::{VP9_TAPS, VP9_TAPS_BEFORE, Vp9Filter};
const MAX_ROWS: usize = MAX_SIDE + VP9_TAPS - 1;
pub(super) const ROUND: i16 = 1 << (crate::vp9_mc::FILTER_BITS - 1);
pub(super) const SHIFT: u32 = ROUND.ilog2() + 1;

/// Tap-sign layouts of the VP9 tables: bit `k` is set when tap `k` is never
/// positive in any phase. Kernels that multiply pixels by tap magnitudes need
/// the signs at compile time; `tables_match_their_layouts` checks every phase.
pub(super) mod layout {
    /// Regular and sharp: taps 0, 2, 5 and 7 are negative.
    pub(crate) const ALTERNATING: u8 = 0b1010_0101;
    /// Smooth: the outer taps 0, 1, 6 and 7 are negative.
    pub(crate) const OUTER: u8 = 0b1100_0011;
    /// Bilinear: no negative taps, and only the centre taps 3 and 4 are nonzero.
    pub(crate) const BILINEAR: u8 = 0;

    /// Whether tap `k` is negative (or zero) in tables with this layout.
    /// The AVX2 kernel multiplies by signed taps and needs only BILINEAR.
    #[cfg(any(test, not(target_arch = "x86_64")))]
    #[inline(always)]
    pub(crate) const fn negative(layout: u8, k: usize) -> bool {
        layout >> k & 1 == 1
    }
}

const fn layout_of(kind: Vp9Filter) -> u8 {
    match kind {
        Vp9Filter::Regular | Vp9Filter::Sharp => layout::ALTERNATING,
        Vp9Filter::Smooth => layout::OUTER,
        Vp9Filter::Bilinear => layout::BILINEAR,
    }
}

/// One ISA's row operations. A "row" is N (4, 8 or 16, at most `LANES`)
/// bytes in a vector register. Loading, filtering and storing are separate
/// steps so that the pass schedule (`convolve`, `column`) is shared by
/// every ISA and can serve other codecs' 8-tap filters.
pub(super) trait RowKernel {
    type Filters;
    type Row: Copy;
    const LANES: usize;
    /// Whether the pass schedule should filter two rows per step (the
    /// `_pair` methods); kernels whose vectors hold two narrow rows set it.
    const PAIR_ROWS: bool = false;
    /// `taps` is a nonidentity phase: every magnitude is at most 127.
    unsafe fn prepare(taps: &[i16; VP9_TAPS]) -> Self::Filters;
    /// Exactly N bytes from `src`.
    unsafe fn load<const N: usize>(src: *const u8) -> Self::Row;
    /// Exactly N bytes to `dst`, optionally averaged with its existing pixels.
    /// Intermediate rows must use AVG=false; averaging follows clipping.
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: Self::Row);
    /// N outputs filtering `src[0..N + 7]`. `LAYOUT` is the [`layout`] of
    /// the table `prepare` was given.
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(
        src: *const u8,
        filters: &Self::Filters,
    ) -> Self::Row;
    /// N outputs filtering eight rows (taps 0..8 top to bottom).
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(
        rows: &[Self::Row; VP9_TAPS],
        filters: &Self::Filters,
    ) -> Self::Row;
    /// [`Self::horizontal`] of the rows at `src` and `src + stride`.
    /// Kernels whose vectors hold two N-wide rows filter both at once.
    #[inline(always)]
    unsafe fn horizontal_pair<const N: usize, const LAYOUT: u8>(
        src: *const u8,
        stride: usize,
        filters: &Self::Filters,
    ) -> [Self::Row; 2] {
        // SAFETY: as for `horizontal`, for both rows.
        unsafe {
            [
                Self::horizontal::<N, LAYOUT>(src, filters),
                Self::horizontal::<N, LAYOUT>(src.add(stride), filters),
            ]
        }
    }
    /// [`Self::vertical`] of window rows 0..8 and 1..9: two output rows.
    #[inline(always)]
    unsafe fn vertical_pair<const N: usize, const LAYOUT: u8>(
        rows: &[Self::Row; VP9_TAPS + 1],
        filters: &Self::Filters,
    ) -> [Self::Row; 2] {
        let (upper, lower) = (
            rows[..VP9_TAPS].try_into().expect("eight rows"),
            rows[1..].try_into().expect("eight rows"),
        );
        // SAFETY: as for `vertical`.
        unsafe {
            [
                Self::vertical::<N, LAYOUT>(upper, filters),
                Self::vertical::<N, LAYOUT>(lower, filters),
            ]
        }
    }
    /// The vertical pass of one N-wide column: `h` outputs from `h + 7`
    /// source rows `stride` apart. The default loads each output's eight
    /// rows ([`column_rows`]); kernels with a sliding window override it.
    #[inline(always)]
    unsafe fn column<const N: usize, const LAYOUT: u8, const AVG: bool>(
        src: *const u8,
        stride: usize,
        dst: (*mut u8, usize),
        h: usize,
        filters: &Self::Filters,
    ) where
        Self: Sized,
    {
        // SAFETY: as for `column_rows`.
        unsafe { column_rows::<Self, N, LAYOUT, AVG>(src, stride, dst, h, filters) }
    }
    /// Store the second row of a pair from [`Self::horizontal_pair`] or
    /// [`Self::vertical_pair`] (overridden where the pair shares a vector).
    #[inline(always)]
    unsafe fn store_second<const N: usize, const AVG: bool>(dst: *mut u8, pair: [Self::Row; 2]) {
        // SAFETY: as for `store`.
        unsafe { Self::store::<N, AVG>(dst, pair[1]) }
    }
}

/// `0, step, 2 * step, ..` below `end`: a plain counted loop (`step_by`
/// over a range compiled to measurably more loop overhead here).
#[inline(always)]
fn strided(end: usize, step: usize) -> impl Iterator<Item = usize> {
    (0..end.div_ceil(step)).map(move |i| i * step)
}

/// `N` values from `f(0), .., f(N - 1)`, built in place (`array::from_fn`
/// left its intermediate array in memory in these kernels).
#[inline(always)]
pub(super) fn gather<T: Copy, const N: usize>(first: T, f: impl Fn(usize) -> T) -> [T; N] {
    let mut out = [first; N];
    for (i, value) in out.iter_mut().enumerate().skip(1) {
        *value = f(i);
    }
    out
}

#[inline(always)]
pub(super) unsafe fn entry<K: RowKernel, const AVG: bool>(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: Dsp validated dimensions, phases, source footprint and destination;
    // the ISA entry point enables K's instruction set.
    unsafe {
        match layout_of(kind) {
            layout::ALTERNATING => {
                by_width::<K, { layout::ALTERNATING }, AVG>(
                    kind, src, stride, phase, size, dst, dst_stride,
                );
            }
            layout::OUTER => {
                by_width::<K, { layout::OUTER }, AVG>(
                    kind, src, stride, phase, size, dst, dst_stride,
                );
            }
            _ => {
                by_width::<K, { layout::BILINEAR }, AVG>(
                    kind, src, stride, phase, size, dst, dst_stride,
                );
            }
        }
    }
}

// One out-of-line body per layout keeps each stack frame and register set to
// a single table's paths (measured: small blocks lost ~4 ns when all inlined).
#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn by_width<K: RowKernel, const LAYOUT: u8, const AVG: bool>(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: as for `entry`; N divides the width in every SIMD case.
    unsafe {
        if size.0 == 0 || size.1 == 0 || !size.0.is_multiple_of(4) {
            // Non-codec widths retain the scalar oracle; four-wide chroma is vectorised.
            crate::vp9_mc::vp9_convolve_into(kind, src, stride, phase, size, dst, dst_stride, AVG);
        } else if K::LANES == 16 && size.0.is_multiple_of(16) {
            convolve::<K, 16, LAYOUT, AVG>(kind, src, stride, phase, size, dst, dst_stride);
        } else if size.0.is_multiple_of(8) {
            convolve::<K, 8, LAYOUT, AVG>(kind, src, stride, phase, size, dst, dst_stride);
        } else {
            convolve::<K, 4, LAYOUT, AVG>(kind, src, stride, phase, size, dst, dst_stride);
        }
    }
}

/// The vertical pass of one N-wide column: `h` outputs from `h + 7` source
/// rows `stride` apart. Each output row loads its eight source rows: keeping
/// a sliding window of raw rows in registers measured slower (x86: +3-10 %
/// instructions; V8: up to 3x, the rotation spills), as did 16-lane SIMD128
/// rows. (AVX2 slides interleaved row pairs instead, as libvpx does.)
#[inline(always)]
pub(super) unsafe fn column_rows<
    K: RowKernel,
    const N: usize,
    const LAYOUT: u8,
    const AVG: bool,
>(
    src: *const u8,
    stride: usize,
    (dst, dst_stride): (*mut u8, usize),
    h: usize,
    filters: &K::Filters,
) {
    // SAFETY: the caller provides h + 7 readable rows of N bytes and h
    // writable rows of N bytes.
    unsafe {
        for y in strided(h, if K::PAIR_ROWS { 2 } else { 1 }) {
            let top = src.add(y * stride);
            let out = dst.add(y * dst_stride);
            if K::PAIR_ROWS && y + 1 < h {
                let rows: [K::Row; VP9_TAPS + 1] =
                    gather(K::load::<N>(top), |k| K::load::<N>(top.add(k * stride)));
                let pair = K::vertical_pair::<N, LAYOUT>(&rows, filters);
                K::store::<N, AVG>(out, pair[0]);
                K::store_second::<N, AVG>(out.add(dst_stride), pair);
            } else {
                let rows: [K::Row; VP9_TAPS] =
                    gather(K::load::<N>(top), |k| K::load::<N>(top.add(k * stride)));
                K::store::<N, AVG>(out, K::vertical::<N, LAYOUT>(&rows, filters));
            }
        }
    }
}

/// The horizontal pass over `rows` rows of `w` outputs (two rows at a time
/// for [`RowKernel::PAIR_ROWS`] kernels).
#[inline(always)]
unsafe fn horizontal_rows<K: RowKernel, const N: usize, const LAYOUT: u8, const AVG: bool>(
    (src, stride): (*const u8, usize),
    (dst, dst_stride): (*mut u8, usize),
    (w, rows): (usize, usize),
    filters: &K::Filters,
) {
    // SAFETY: the caller provides `rows` source rows of w + 7 bytes and
    // `rows` writable rows of w bytes; N divides w.
    unsafe {
        for y in strided(rows, if K::PAIR_ROWS { 2 } else { 1 }) {
            let (line, out) = (src.add(y * stride), dst.add(y * dst_stride));
            for x in strided(w, N) {
                if K::PAIR_ROWS && y + 1 < rows {
                    let pair = K::horizontal_pair::<N, LAYOUT>(line.add(x), stride, filters);
                    K::store::<N, AVG>(out.add(x), pair[0]);
                    K::store_second::<N, AVG>(out.add(dst_stride + x), pair);
                } else {
                    let row = K::horizontal::<N, LAYOUT>(line.add(x), filters);
                    K::store::<N, AVG>(out.add(x), row);
                }
            }
        }
    }
}

/// The full-sample copy (or average) of `h` rows of `COUNT` N-byte vectors.
/// A constant row width keeps each row as inline vector moves: with the
/// width a runtime value, the row loop became a `memcpy` call per row
/// (2.6 % of a 720p decode, 2026-11-12).
#[inline(always)]
unsafe fn copy<K: RowKernel, const N: usize, const COUNT: usize, const AVG: bool>(
    ((src, stride), (dst, dst_stride), h): ((*const u8, usize), (*mut u8, usize), usize),
) {
    // SAFETY: the caller provides h rows of N * COUNT readable source and
    // writable (for AVG also readable) destination bytes.
    unsafe {
        for y in 0..h {
            let (from, to) = (src.add(y * stride), dst.add(y * dst_stride));
            for x in 0..COUNT {
                K::store::<N, AVG>(to.add(x * N), K::load::<N>(from.add(x * N)));
            }
        }
    }
}

#[inline(always)]
unsafe fn convolve<K: RowKernel, const N: usize, const LAYOUT: u8, const AVG: bool>(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    (fx, fy): (usize, usize),
    (w, h): (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    let taps = kind.kernel();
    let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
    // Identity passes can be skipped because rounding 128*x by seven is exact.
    // SAFETY: all pointers below remain inside the window and output validated
    // by Dsp. Temporary rows include the full vertical support; N divides w.
    unsafe {
        let centre = src.add(VP9_TAPS_BEFORE * stride + VP9_TAPS_BEFORE);
        if fx == 0 && fy == 0 {
            let rows = ((centre, stride), (dst, dst_stride), h);
            match w / N {
                1 => copy::<K, N, 1, AVG>(rows),
                2 => copy::<K, N, 2, AVG>(rows),
                4 => copy::<K, N, 4, AVG>(rows),
                _ => {
                    for y in 0..h {
                        for x in strided(w, N) {
                            K::store::<N, AVG>(
                                dst.add(y * dst_stride + x),
                                K::load::<N>(centre.add(y * stride + x)),
                            );
                        }
                    }
                }
            }
        } else if fy == 0 {
            let filters = K::prepare(&taps[fx]);
            let line = src.add(VP9_TAPS_BEFORE * stride);
            horizontal_rows::<K, N, LAYOUT, AVG>(
                (line, stride),
                (dst, dst_stride),
                (w, h),
                &filters,
            );
        } else if fx == 0 {
            let filters = K::prepare(&taps[fy]);
            for x in strided(w, N) {
                let top = src.add(x + VP9_TAPS_BEFORE);
                K::column::<N, LAYOUT, AVG>(top, stride, (dst.add(x), dst_stride), h, &filters);
            }
        } else {
            // Every byte read below is first written by the horizontal pass.
            let mut temp = [core::mem::MaybeUninit::<u8>::uninit(); MAX_SIDE * MAX_ROWS];
            let temp = temp.as_mut_ptr().cast::<u8>();
            let horizontal = K::prepare(&taps[fx]);
            let vertical = K::prepare(&taps[fy]);
            horizontal_rows::<K, N, LAYOUT, false>(
                (src, stride),
                (temp, w),
                (w, h + VP9_TAPS - 1),
                &horizontal,
            );
            for x in strided(w, N) {
                K::column::<N, LAYOUT, AVG>(temp.add(x), w, (dst.add(x), dst_stride), h, &vertical);
            }
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(crate) const KINDS: [Vp9Filter; 4] = [
        Vp9Filter::Regular,
        Vp9Filter::Smooth,
        Vp9Filter::Sharp,
        Vp9Filter::Bilinear,
    ];

    /// Sums of the positive and of the negated negative taps of one phase.
    pub(crate) fn magnitudes(taps: &[i16]) -> (i32, i32) {
        let positive = taps.iter().map(|&t| i32::from(t.max(0))).sum();
        let negative = taps.iter().map(|&t| -i32::from(t.min(0))).sum();
        (positive, negative)
    }

    #[test]
    fn tables_match_their_layouts() {
        for kind in KINDS {
            let layout = layout_of(kind);
            for taps in &kind.kernel()[1..] {
                for (k, &tap) in taps.iter().enumerate() {
                    assert!(
                        tap.unsigned_abs() <= 127,
                        "{kind:?} tap exceeds a signed byte"
                    );
                    if layout::negative(layout, k) {
                        assert!(tap <= 0, "{kind:?} tap {k} should be negative");
                    } else {
                        assert!(tap >= 0, "{kind:?} tap {k} should be positive");
                    }
                    if layout == layout::BILINEAR && k != 3 && k != 4 {
                        assert_eq!(tap, 0, "bilinear uses only the centre taps");
                    }
                }
            }
        }
    }

    /// Magnitude kernels sum the positive and the negative products separately
    /// in u16 and take their saturating difference: each sum, plus rounding,
    /// must fit u16. (A negative difference saturates to 0, exactly the clip.)
    #[test]
    fn positive_and_negative_sums_fit_u16() {
        for kind in KINDS {
            for taps in &kind.kernel()[1..] {
                let (positive, negative) = magnitudes(taps);
                for sum in [positive, negative] {
                    assert!(sum * 255 + i32::from(ROUND) <= i32::from(u16::MAX));
                }
            }
        }
    }
}
