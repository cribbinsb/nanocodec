//! Eight-tap motion compensation of the general AV2 decoder, unscaled:
//! the three switchable interpolation filters at final precision
//! ([`put_8tap`]) and compound precision ([`prep_8tap`]), and the 4x4
//! blocks of the extended warp, whose 1/64-sample filter goes through the
//! same separable kernel ([`ext_warp4x4`], [`ext_warp4x4t`]).
//!
//! Ported from dav2d `src/mc_tmpl.c` (`put_8tap_c`, `prep_8tap_c`,
//! `ext_warp4x4_c`, `ext_warp4x4t_c`; commit 237229e, Copyright ©
//! 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see
//! `docs/licenses/dav2d`). The interpolation taps are AV1's
//! (`dav2d_mc_subpel_filters` is libaom's `sub_pel_filters_*` halved), so
//! they are derived from the shared table in [`crate::mc`]; the AV1
//! decoder's own convolution follows the spec's rounding stages
//! (`InterRound0` = 3 on 7-bit taps) and keeps its own window contract, so
//! it is not this kernel. Reference positions are the block's integer
//! position in `src`; the kernels read 3 samples before and 4 after it
//! along each filtered direction.

use nc_common::{BitDepth, Sample};

use super::mc::{
    MAX_SIDE, Precision, assert_compound_size, prep_bilinear, prep_copy, put_bilinear_at,
    round_shift, store_compound, store_pixels,
};
use super::mc_tables::EXT_WARP_FILTER;
use crate::mc::{REGULAR_4, REGULAR_8, SHARP_8, SMOOTH_4, SMOOTH_8, TAP_CENTRE, TAPS};

/// Sub-sample positions of the interpolation filters (1/16 sample).
const SUBPEL_POSITIONS: usize = 16;
/// Fraction bits of the (halved) interpolation taps.
const SUBPEL_FILTER_BITS: u32 = 6;
/// Fraction bits of the extended-warp taps.
const EXT_WARP_FILTER_BITS: u32 = 7;
/// Edges of at most this many samples take a family's 4-tap variant.
const SHORT_EDGE: usize = 4;
/// Side of an extended-warp block.
const EXT_WARP_SIDE: usize = 4;
/// Samples a filtered direction reads beyond the block: the taps around
/// each output sample.
const EXTRA: usize = TAPS - 1;

/// The taps of one filter at every 1/16-sample position, 6-bit.
type Family = [[i8; TAPS]; SUBPEL_POSITIONS];

/// An AV1 7-bit filter family halved to dav2d's 6-bit taps (exact: every
/// AV1 tap is even).
const fn halve(family: &[[i16; TAPS]; SUBPEL_POSITIONS]) -> Family {
    let mut out = [[0; TAPS]; SUBPEL_POSITIONS];
    let mut position = 0;
    while position < SUBPEL_POSITIONS {
        let mut tap = 0;
        while tap < TAPS {
            assert!(family[position][tap] % 2 == 0);
            out[position][tap] = (family[position][tap] / 2) as i8;
            tap += 1;
        }
        position += 1;
    }
    out
}

/// `dav2d_mc_subpel_filters[0..3]`: regular, smooth, sharp.
static FAMILIES: [Family; 3] = [halve(&REGULAR_8), halve(&SMOOTH_8), halve(&SHARP_8)];
/// `dav2d_mc_subpel_filters[3..5]`: the 4-tap regular and smooth filters
/// of edges of at most 4 samples.
static SHORT_FAMILIES: [Family; 2] = [halve(&REGULAR_4), halve(&SMOOTH_4)];

/// An interpolation filter (dav2d `Dav2dFilterMode`; the discriminants
/// are dav2d's values, the switchable ones the syntax values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpFilter {
    /// `DAV2D_FILTER_8TAP_REGULAR`.
    Regular = 0,
    /// `DAV2D_FILTER_8TAP_SMOOTH`.
    Smooth = 1,
    /// `DAV2D_FILTER_8TAP_SHARP`.
    Sharp = 2,
    /// `DAV2D_FILTER_BILINEAR`: [`put_bilinear_at`], [`prep_bilinear`].
    Bilinear = 3,
}

impl InterpFilter {
    /// The filter of dav2d value `index`, if any.
    #[must_use]
    pub fn from_index(index: u8) -> Option<Self> {
        [Self::Regular, Self::Smooth, Self::Sharp, Self::Bilinear]
            .get(usize::from(index))
            .copied()
    }

    /// The taps at 1/16-sample `phase` along an edge of `length` samples
    /// (`GET_H_FILTER`): short edges take the 4-tap regular filter (sharp
    /// too) or the 4-tap smooth one. `None` at phase 0.
    fn taps(self, phase: i32, length: usize) -> Option<&'static [i8; TAPS]> {
        let family = match (self, length > SHORT_EDGE) {
            (Self::Regular | Self::Sharp, false) => &SHORT_FAMILIES[0],
            (Self::Smooth, false) => &SHORT_FAMILIES[1],
            (Self::Bilinear, _) => unreachable!("bilinear has its own kernels"),
            (_, true) => &FAMILIES[self as usize],
        };
        (phase != 0).then(|| &family[phase as usize])
    }
}

/// The filters of one block and their precision.
#[derive(Clone, Copy)]
struct Filters {
    horizontal: Option<[i8; TAPS]>,
    vertical: Option<[i8; TAPS]>,
    bits: u32,
}

impl Filters {
    /// The switchable filter `filter` at (`mx`, `my`) in 1/16 samples.
    fn subpel(filter: InterpFilter, (w, h): (usize, usize), (mx, my): (i32, i32)) -> Self {
        Self {
            horizontal: filter.taps(mx, w).copied(),
            vertical: filter.taps(my, h).copied(),
            bits: SUBPEL_FILTER_BITS,
        }
    }

    /// The extended-warp filter at (`mx`, `my`) in 1/64 samples.
    fn ext_warp((mx, my): (i32, i32)) -> Self {
        let taps = |phase: i32| (phase != 0).then(|| EXT_WARP_FILTER[phase as usize - 1]);
        Self {
            horizontal: taps(mx),
            vertical: taps(my),
            bits: EXT_WARP_FILTER_BITS,
        }
    }
}

/// `sum(taps[k] * window[k * step])`: one 8-tap filter whose first tap
/// reads `window[0]`.
fn filter8<S: Copy + Into<i32>>(window: &[S], step: usize, taps: [i8; TAPS]) -> i32 {
    taps.iter()
        .enumerate()
        .map(|(k, &tap)| i32::from(tap) * window[k * step].into())
        .sum()
}

/// The horizontal pass of a two-pass filter: `h + 7` rows from three
/// above the block, rounded by `shift` (`mid` in dav2d).
fn horizontal_pass<T: Sample>(
    src: &[T],
    (src_offset, src_stride): (usize, usize),
    (w, h): (usize, usize),
    taps: [i8; TAPS],
    shift: u32,
) -> [[i16; MAX_SIDE]; MAX_SIDE + EXTRA] {
    let mut mid = [[0_i16; MAX_SIDE]; MAX_SIDE + EXTRA];
    let top = src_offset - TAP_CENTRE * src_stride - TAP_CENTRE;
    for (y, line) in mid.iter_mut().take(h + EXTRA).enumerate() {
        let row = &src[top + y * src_stride..][..w + EXTRA];
        for (x, value) in line[..w].iter_mut().enumerate() {
            *value = round_shift(filter8(&row[x..], 1, taps), shift) as i16;
        }
    }
    mid
}

/// The vertical pass over `mid` for output row `y`, column `x`.
fn vertical_of_mid(
    mid: &[[i16; MAX_SIDE]; MAX_SIDE + EXTRA],
    (x, y): (usize, usize),
    taps: [i8; TAPS],
) -> i32 {
    taps.iter()
        .enumerate()
        .map(|(k, &tap)| i32::from(tap) * i32::from(mid[y + k][x]))
        .sum()
}

/// One row of a single-direction filter over `src` at row `y`: horizontal
/// taps along the row, or vertical taps down the column.
fn one_pass_row<T: Sample>(
    src: &[T],
    (src_offset, src_stride): (usize, usize),
    (w, y): (usize, usize),
    taps: [i8; TAPS],
    vertical: bool,
) -> impl Iterator<Item = i32> + '_ {
    let row = src_offset + y * src_stride;
    let (start, step) = if vertical {
        (row - TAP_CENTRE * src_stride, src_stride)
    } else {
        (row - TAP_CENTRE, 1)
    };
    (0..w).map(move |x| filter8(&src[start + x..], step, taps))
}

/// The final-precision prediction of dav2d `put_8tap_c` with `filters`.
fn put<T: Sample>(
    (dst, dst_offset, dst_stride): (&mut [T], usize, usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    (w, h): (usize, usize),
    filters: Filters,
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let bits = filters.bits;
    let rows = dst[dst_offset..].chunks_mut(dst_stride).take(h);
    match (filters.horizontal, filters.vertical) {
        (Some(fh), Some(fv)) => {
            let mid = horizontal_pass(
                src,
                (src_offset, src_stride),
                (w, h),
                fh,
                bits - precision.bits,
            );
            for (y, line) in rows.enumerate() {
                let values = (0..w)
                    .map(|x| round_shift(vertical_of_mid(&mid, (x, y), fv), bits + precision.bits));
                store_pixels(&mut line[..w], values, depth);
            }
        }
        (Some(fh), None) => {
            // Both roundings in one: to intermediate precision, then to
            // samples (dav2d `intermediate_rnd`).
            let rounding = ((1 << bits) >> 1) + ((1 << (bits - precision.bits)) >> 1);
            for (y, line) in rows.enumerate() {
                let values = one_pass_row(src, (src_offset, src_stride), (w, y), fh, false)
                    .map(|sum| (sum + rounding) >> bits);
                store_pixels(&mut line[..w], values, depth);
            }
        }
        (None, Some(fv)) => {
            for (y, line) in rows.enumerate() {
                let values = one_pass_row(src, (src_offset, src_stride), (w, y), fv, true)
                    .map(|sum| round_shift(sum, bits));
                store_pixels(&mut line[..w], values, depth);
            }
        }
        (None, None) => {
            for (y, line) in rows.enumerate() {
                line[..w].copy_from_slice(&src[src_offset + y * src_stride..][..w]);
            }
        }
    }
}

/// The compound-precision prediction of dav2d `prep_8tap_c` with
/// `filters`.
fn prep<T: Sample>(
    (tmp, tmp_stride): (&mut [i16], usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    (w, h): (usize, usize),
    filters: Filters,
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let bits = filters.bits;
    let first_shift = bits - precision.bits;
    let rows = tmp.chunks_mut(tmp_stride).take(h);
    match (filters.horizontal, filters.vertical) {
        (Some(fh), Some(fv)) => {
            let mid = horizontal_pass(src, (src_offset, src_stride), (w, h), fh, first_shift);
            for (y, line) in rows.enumerate() {
                let values = (0..w).map(|x| round_shift(vertical_of_mid(&mid, (x, y), fv), bits));
                store_compound(&mut line[..w], values, precision);
            }
        }
        (Some(taps), None) | (None, Some(taps)) => {
            let vertical = filters.horizontal.is_none();
            for (y, line) in rows.enumerate() {
                let values = one_pass_row(src, (src_offset, src_stride), (w, y), taps, vertical)
                    .map(|sum| round_shift(sum, first_shift));
                store_compound(&mut line[..w], values, precision);
            }
        }
        (None, None) => prep_copy(
            tmp,
            tmp_stride,
            src,
            (src_offset, src_stride),
            (w, h),
            depth,
        ),
    }
}

/// Predict a `w` x `h` block (powers of two, 2 to 64) at
/// `dst[dst_offset..]` from the integer position `src[src_offset..]` with
/// `filter` at sub-sample position (`mx`, `my`) in 1/16 samples (dav2d
/// `put_8tap_{regular,smooth,sharp}_c`; [`InterpFilter::Bilinear`] goes to
/// [`put_bilinear_at`], dav2d `put_bilin_c`, whose source needs only one
/// sample after the block).
///
/// # Panics
///
/// If a size is out of range or a slice is too short for the block and
/// its filter taps.
#[allow(clippy::too_many_arguments, reason = "dav2d's kernel contract")]
pub fn put_8tap<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    (w, h): (usize, usize),
    phase: (i32, i32),
    filter: InterpFilter,
    depth: BitDepth,
) {
    if filter == InterpFilter::Bilinear {
        return put_bilinear_at(
            dst,
            dst_offset,
            dst_stride,
            src,
            src_offset,
            src_stride,
            (w, h),
            phase,
            depth,
        );
    }
    for side in [w, h] {
        assert!(side.is_power_of_two() && (2..=MAX_SIDE).contains(&side));
    }
    put(
        (dst, dst_offset, dst_stride),
        (src, src_offset, src_stride),
        (w, h),
        Filters::subpel(filter, (w, h), phase),
        depth,
    );
}

/// Predict a `w` x `h` block (powers of two, 4 to 64) at compound
/// precision into `tmp` (rows `tmp_stride` apart) from the integer
/// position `src[src_offset..]` with `filter` at (`mx`, `my`) in 1/16
/// samples (dav2d `prep_8tap_{regular,smooth,sharp}_c`; bilinear goes to
/// [`prep_bilinear`], dav2d `prep_bilin_c`).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
#[allow(clippy::too_many_arguments, reason = "dav2d's kernel contract")]
pub fn prep_8tap<T: Sample>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    (w, h): (usize, usize),
    phase: (i32, i32),
    filter: InterpFilter,
    depth: BitDepth,
) {
    if filter == InterpFilter::Bilinear {
        return prep_bilinear(
            tmp,
            tmp_stride,
            src,
            src_offset,
            src_stride,
            (w, h),
            phase,
            depth,
        );
    }
    assert_compound_size(w, h);
    prep(
        (tmp, tmp_stride),
        (src, src_offset, src_stride),
        (w, h),
        Filters::subpel(filter, (w, h), phase),
        depth,
    );
}

/// Predict one 4x4 block of the extended warp at `dst[dst_offset..]` from
/// the integer position `src[src_offset..]` at (`mx`, `my`) in 1/64
/// samples, 0 to 63 (dav2d `ext_warp4x4_c`).
///
/// # Panics
///
/// If a phase is out of range or a slice is too short.
#[allow(clippy::too_many_arguments, reason = "dav2d's kernel contract")]
pub fn ext_warp4x4<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    phase: (i32, i32),
    depth: BitDepth,
) {
    put(
        (dst, dst_offset, dst_stride),
        (src, src_offset, src_stride),
        (EXT_WARP_SIDE, EXT_WARP_SIDE),
        Filters::ext_warp(phase),
        depth,
    );
}

/// [`ext_warp4x4`] at compound precision into `tmp` (dav2d
/// `ext_warp4x4t_c`).
///
/// # Panics
///
/// If a phase is out of range or a slice is too short.
pub fn ext_warp4x4t<T: Sample>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    phase: (i32, i32),
    depth: BitDepth,
) {
    prep(
        (tmp, tmp_stride),
        (src, src_offset, src_stride),
        (EXT_WARP_SIDE, EXT_WARP_SIDE),
        Filters::ext_warp(phase),
        depth,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 15x15 test picture (dav2d capture inputs); blocks start at
    /// (3, 3) so the taps before them are inside.
    fn picture() -> Vec<u8> {
        (0..15 * 15_usize)
            .map(|i| {
                let (x, y) = (i % 15, i / 15);
                ((x * 13 + y * 29 + x * y * 7) % 256) as u8
            })
            .collect()
    }

    /// Offset of the block's integer position in [`picture`].
    const ORIGIN: usize = 3 * 15 + 3;

    /// Values captured from dav2d's `put_8tap_c` / `prep_8tap_c` /
    /// `ext_warp4x4{,t}_c` (8-bit unless named) on [`picture`].
    #[test]
    fn matches_dav2d() {
        let src = picture();
        let eight = BitDepth::EIGHT;
        let mut dst = [0_u8; 64];
        let mut tmp = [0_i16; 64];
        put_8tap(
            &mut dst,
            0,
            8,
            &src,
            ORIGIN,
            15,
            (8, 8),
            (3, 9),
            InterpFilter::Regular,
            eight,
        );
        assert_eq!(dst[..8], [228, 82, 17, 60, 101, 197, 193, 78]);
        put_8tap(
            &mut dst,
            0,
            4,
            &src,
            ORIGIN,
            15,
            (4, 4),
            (6, 0),
            InterpFilter::Sharp,
            eight,
        );
        assert_eq!(dst[..4], [223, 141, 0, 49]);
        prep_8tap(
            &mut tmp,
            4,
            &src,
            ORIGIN,
            15,
            (4, 4),
            (7, 0),
            InterpFilter::Smooth,
            eight,
        );
        assert_eq!(tmp[..4], [2887, 1895, 711, 807]);
        prep_8tap(
            &mut tmp,
            8,
            &src,
            ORIGIN,
            15,
            (8, 4),
            (0, 5),
            InterpFilter::Sharp,
            eight,
        );
        assert_eq!(tmp[..8], [3594, 2637, -48, 531, 1110, 2393, 2972, 2015]);
        prep_8tap(
            &mut tmp,
            8,
            &src,
            ORIGIN,
            15,
            (8, 8),
            (12, 2),
            InterpFilter::Regular,
            eight,
        );
        assert_eq!(tmp[56..64], [1211, 2994, 675, 1297, 2948, 631, 947, 2838]);
        ext_warp4x4(&mut dst, 0, 4, &src, ORIGIN, 15, (17, 40), eight);
        assert_eq!(dst[..4], [217, 63, 26, 67]);
        ext_warp4x4t(&mut tmp, 4, &src, ORIGIN, 15, (17, 40), eight);
        assert_eq!(tmp[12..16], [2299, 2295, 1284, 1845]);

        let deep: Vec<u16> = (0..15 * 15_usize)
            .map(|i| {
                let (x, y) = (i % 15, i / 15);
                ((x * 53 + y * 117 + x * y * 31) % 1024) as u16
            })
            .collect();
        let mut dst = [0_u16; 64];
        let ten = BitDepth::TEN;
        put_8tap(
            &mut dst,
            0,
            8,
            &deep,
            ORIGIN,
            15,
            (8, 8),
            (3, 9),
            InterpFilter::Regular,
            ten,
        );
        assert_eq!(dst[..8], [951, 381, 123, 314, 623, 732, 268, 504]);
    }

    /// The derived taps are dav2d's `dav2d_mc_subpel_filters` rows (a
    /// few, checked against `tables.c`).
    #[test]
    fn halved_taps_are_dav2ds() {
        assert_eq!(FAMILIES[0][1], [0, 1, -3, 63, 4, -1, 0, 0]);
        assert_eq!(FAMILIES[1][8], [0, -1, 7, 26, 26, 7, -1, 0]);
        assert_eq!(FAMILIES[2][15], [0, 1, -1, 4, 63, -3, 1, -1]);
        assert_eq!(SHORT_FAMILIES[0][1], [0, 0, -2, 63, 4, -1, 0, 0]);
        assert_eq!(SHORT_FAMILIES[1][15], [0, 0, 1, 17, 31, 15, 0, 0]);
        for family in FAMILIES.iter().chain(&SHORT_FAMILIES) {
            for taps in &family[1..] {
                assert_eq!(taps.iter().map(|&t| i32::from(t)).sum::<i32>(), 64);
            }
        }
    }
}
