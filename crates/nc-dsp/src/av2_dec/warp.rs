//! The affine warp of the general AV2 decoder: one 8x8 block at final
//! ([`warp_affine_8x8`]) or compound precision ([`warp_affine_8x8t`]),
//! each sample filtered at its own phase from the shear parameters.
//!
//! Ported from dav2d `src/mc_tmpl.c` (`warp_affine_8x8_c`,
//! `warp_affine_8x8t_c`; commit 237229e, Copyright © 2018-2026 `VideoLAN`
//! and dav2d authors, BSD-2-Clause, see `docs/licenses/dav2d`). The
//! structure is AV1's warp (`av1_mc::warp`), but AV2's filter table spans
//! [-3, 4) samples (449 phases, centre at 192) where AV1's spans [-1, 2)
//! (193 phases), so the kernel is ported beside it.

use nc_common::{BitDepth, Sample};

use crate::Dsp;
#[cfg(target_arch = "x86_64")]
use crate::av1_mc::warp_avx2::{Target, try_warp8x8};
#[cfg(not(target_arch = "x86_64"))]
use crate::av1_mc::warp_kernels::WarpOut;
use crate::av1_mc::warp_kernels::{WINDOW_STRIDE, WarpPhases};

use super::mc::{Precision, round_shift, store_compound, store_pixels};
use super::mc_tables::WARP_FILTER;
use crate::mc::{TAP_CENTRE, TAPS};

/// Side of a warp block.
const SIDE: usize = 8;
/// Rows of the horizontal pass: the block and the vertical taps around it.
const MID_ROWS: usize = SIDE + TAPS - 1;
/// Fraction bits of the warp taps.
const WARP_FILTER_BITS: u32 = 7;
/// Phase bits dropped before the table lookup (positions are 1/65536
/// sample; the table steps by 1/64).
const PHASE_SHIFT: u32 = 10;
/// Index of offset 0 in [`WARP_FILTER`]: three whole samples of 64 phases
/// precede it.
const FILTER_CENTRE: i32 = 3 * 64;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const PHASE_ROUND: i32 = (FILTER_CENTRE << PHASE_SHIFT) + (1 << (PHASE_SHIFT - 1));

/// The taps at position `position` (1/65536 sample, relative to the
/// sample being filtered).
#[inline]
fn taps(position: i32) -> &'static [i8; TAPS] {
    let index = FILTER_CENTRE + round_shift(position, PHASE_SHIFT);
    &WARP_FILTER[index as usize]
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) unsafe fn taps_unchecked(position: i32) -> &'static [i8; TAPS] {
    // SAFETY: the caller checked all phases against the AV2 table.
    unsafe { WARP_FILTER.get_unchecked(((position + PHASE_ROUND) >> PHASE_SHIFT) as usize) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) unsafe fn taps_i16_unchecked(position: i32) -> &'static [i16; TAPS] {
    static FILTERS: [[i16; TAPS]; WARP_FILTER.len()] = {
        let mut wide = [[0; TAPS]; WARP_FILTER.len()];
        let mut phase = 0;
        while phase < WARP_FILTER.len() {
            let mut tap = 0;
            while tap < TAPS {
                wide[phase][tap] = WARP_FILTER[phase][tap] as i16;
                tap += 1;
            }
            phase += 1;
        }
        wide
    };
    // SAFETY: the caller checked all phases against the AV2 table.
    unsafe { FILTERS.get_unchecked(((position + PHASE_ROUND) >> PHASE_SHIFT) as usize) }
}

/// The shear parameters `[alpha, beta, gamma, delta]` (dav2d
/// `Dav2dWarpedMotionParams::u.abcd`): the horizontal position step per
/// column and per row, the vertical step per column and per row.
pub type Shear = [i16; 4];

/// The horizontal pass: 15 rows from 3 above the block, each column at
/// its own phase, rounded to intermediate precision.
#[inline]
fn horizontal_pass<T: Sample>(
    (src, src_offset, src_stride): (&[T], usize, usize),
    shear: Shear,
    mx: i32,
    shift: u32,
) -> [[i16; SIDE]; MID_ROWS] {
    let [alpha, beta, ..] = shear.map(i32::from);
    let mut mid = [[0_i16; SIDE]; MID_ROWS];
    let top = src_offset - TAP_CENTRE * src_stride - TAP_CENTRE;
    for (y, line) in mid.iter_mut().enumerate() {
        let row = &src[top + y * src_stride..][..SIDE + TAPS - 1];
        let start = mx + y as i32 * beta;
        for (x, value) in line.iter_mut().enumerate() {
            let filter = taps(start + x as i32 * alpha);
            let sum: i32 = filter
                .iter()
                .zip(&row[x..])
                .map(|(&t, s)| i32::from(t) * s.to_i32())
                .sum();
            *value = round_shift(sum, shift) as i16;
        }
    }
    mid
}

/// The vertical pass of output row `y`: each column at its own phase.
#[inline]
fn vertical_row(
    mid: &[[i16; SIDE]; MID_ROWS],
    shear: Shear,
    my: i32,
    y: usize,
) -> impl Iterator<Item = i32> + '_ {
    let [_, _, gamma, delta] = shear.map(i32::from);
    let start = my + y as i32 * delta;
    (0..SIDE).map(move |x| {
        let filter = taps(start + x as i32 * gamma);
        filter
            .iter()
            .enumerate()
            .map(|(k, &t)| i32::from(t) * i32::from(mid[y + k][x]))
            .sum()
    })
}

/// Warp one 8x8 block to `dst[dst_offset..]` from the integer position
/// `src[src_offset..]` (the source holds 3 samples before and 4 after the
/// block in each direction) with `shear` and the block's starting phases
/// (`mx`, `my`) in 1/65536 sample (dav2d `warp_affine_8x8_c`).
///
/// # Panics
///
/// If a slice is too short or a phase leaves the filter table.
pub fn warp_affine_8x8<T: Sample>(
    dst: &mut [T],
    dst_offset: usize,
    dst_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    shear: Shear,
    (mx, my): (i32, i32),
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let mid = horizontal_pass(
        (src, src_offset, src_stride),
        shear,
        mx,
        WARP_FILTER_BITS - precision.bits,
    );
    let rows = dst[dst_offset..].chunks_mut(dst_stride).take(SIDE);
    for (y, line) in rows.enumerate() {
        let values = vertical_row(&mid, shear, my, y)
            .map(|sum| round_shift(sum, WARP_FILTER_BITS + precision.bits));
        store_pixels(&mut line[..SIDE], values, depth);
    }
}

const _: () = {
    let mut phase = 0;
    while phase < WARP_FILTER.len() {
        let mut tap = 0;
        let (mut sum, mut absolute) = (0_i32, 0_i32);
        while tap < TAPS {
            let weight = WARP_FILTER[phase][tap] as i32;
            sum += weight;
            absolute += weight.abs();
            tap += 1;
        }
        assert!(sum == 128 && absolute <= 222);
        phase += 1;
    }
};

/// AV2 phases use the same pairing as the AV1/dav1d AVX2 register
/// schedule. The extra row covers the final 16-byte coefficient load.
#[cfg(target_arch = "x86_64")]
static PAIRED_FILTERS: [[i8; TAPS]; WARP_FILTER.len() + 1] = {
    let mut paired = [[0; TAPS]; WARP_FILTER.len() + 1];
    let mut phase = 0;
    while phase < WARP_FILTER.len() {
        let row = WARP_FILTER[phase];
        paired[phase] = [
            row[0], row[2], row[4], row[6], row[1], row[3], row[5], row[7],
        ];
        let mut tap = 0;
        let (mut even, mut odd) = (0_i32, 0_i32);
        while tap < TAPS {
            let weight = row[tap] as i32;
            if weight > 0 {
                if tap % 2 == 0 {
                    even += weight;
                } else {
                    odd += weight;
                }
            }
            tap += 1;
        }
        assert!(even <= 128 && odd <= 128);
        phase += 1;
    }
    paired
};

fn warp_input(
    (src, offset, stride): (&[u8], usize, usize),
    shear: Shear,
    (mx, my): (i32, i32),
) -> Option<((&[u8], usize), WarpPhases)> {
    if stride < WINDOW_STRIDE {
        return None;
    }
    let before = stride.checked_mul(TAP_CENTRE)?.checked_add(TAP_CENTRE)?;
    let start = offset.checked_sub(before)?;
    let window = src.get(start..)?;
    let [alpha, beta, gamma, delta] = shear.map(i32::from);
    let x = mx.checked_add(3 * beta)?;
    let phases = WarpPhases {
        x: (x, alpha, beta),
        y: (my, gamma, delta),
    };
    Some(((window, stride), phases))
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn warp8_x86(
    dsp: &Dsp,
    src: (&[u8], usize, usize),
    shear: Shear,
    phase: (i32, i32),
    round: u32,
    out: Target<'_>,
) -> bool {
    if !matches!(dsp.isa(), crate::Isa::Avx2 | crate::Isa::Avx512) {
        return false;
    }
    let Some((source, phases)) = warp_input(src, shear, phase) else {
        return false;
    };
    // SAFETY: the tier enables AVX2, and the packed table satisfies
    // the arithmetic and padding invariants of the shared kernel.
    unsafe { try_warp8x8(source, &phases, round, out, &PAIRED_FILTERS, PHASE_ROUND) }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(unsafe_code)]
fn warp8_with_dsp(
    dsp: &Dsp,
    src: (&[u8], usize, usize),
    shear: Shear,
    phase: (i32, i32),
    round: u32,
    out: &mut WarpOut,
) -> bool {
    let Some(((window, stride), phases)) = warp_input(src, shear, phase) else {
        return false;
    };
    #[cfg(target_arch = "aarch64")]
    if matches!(dsp.isa(), crate::Isa::Neon | crate::Isa::Sve2)
        && stride
            .checked_mul(MID_ROWS - 1)
            .and_then(|n| n.checked_add(WINDOW_STRIDE))
            .is_some_and(|span| span <= window.len())
        && phases.in_table(PHASE_ROUND, PHASE_SHIFT, WARP_FILTER.len())
    {
        // SAFETY: both tiers enable NEON and the complete source window
        // is readable. The warp filter accesses the phase table safely.
        unsafe { crate::av1_mc::warp_neon::warp8x8_av2((window, stride), &phases, round, out) };
        return true;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dsp.isa() == crate::Isa::Simd128
        && stride
            .checked_mul(MID_ROWS - 1)
            .and_then(|n| n.checked_add(WINDOW_STRIDE))
            .is_some_and(|span| span <= window.len())
        && phases.in_table(PHASE_ROUND, PHASE_SHIFT, WARP_FILTER.len())
    {
        // SAFETY: SIMD128 is active and the window has the required span.
        unsafe {
            crate::av1_mc::warp_kernels::warp8x8_av2_simd128((window, stride), &phases, round, out);
        }
        return true;
    }
    let _ = (dsp, window, phases, round, out);
    false
}

/// Warp an 8-bit block to final pixels using the selected DSP tier.
///
/// # Panics
///
/// As [`warp_affine_8x8`].
pub fn warp_affine_8x8_with_dsp(
    dsp: &Dsp,
    (dst, offset, stride): (&mut [u8], usize, usize),
    src: (&[u8], usize, usize),
    shear: Shear,
    phase: (i32, i32),
) {
    let round = WARP_FILTER_BITS + Precision::of(BitDepth::EIGHT).bits;
    #[cfg(target_arch = "x86_64")]
    if warp8_x86(
        dsp,
        src,
        shear,
        phase,
        round,
        Target::Pixels(&mut dst[offset..], stride),
    ) {
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut out = [0_i32; SIDE * SIDE];
        if warp8_with_dsp(dsp, src, shear, phase, round, &mut out) {
            for (line, values) in dst[offset..]
                .chunks_mut(stride)
                .zip(out.as_chunks::<SIDE>().0)
            {
                store_pixels(&mut line[..SIDE], values.iter().copied(), BitDepth::EIGHT);
            }
            return;
        }
    }
    warp_affine_8x8(
        dst,
        offset,
        stride,
        src.0,
        src.1,
        src.2,
        shear,
        phase,
        BitDepth::EIGHT,
    );
}

/// Warp an 8-bit block to compound precision using the selected DSP tier.
///
/// # Panics
///
/// As [`warp_affine_8x8t`].
pub fn warp_affine_8x8t_with_dsp(
    dsp: &Dsp,
    (tmp, stride): (&mut [i16], usize),
    src: (&[u8], usize, usize),
    shear: Shear,
    phase: (i32, i32),
) {
    #[cfg(target_arch = "x86_64")]
    if warp8_x86(
        dsp,
        src,
        shear,
        phase,
        WARP_FILTER_BITS,
        Target::Compound(tmp, stride),
    ) {
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut out = [0_i32; SIDE * SIDE];
        if warp8_with_dsp(dsp, src, shear, phase, WARP_FILTER_BITS, &mut out) {
            let precision = Precision::of(BitDepth::EIGHT);
            for (line, values) in tmp.chunks_mut(stride).zip(out.as_chunks::<SIDE>().0) {
                store_compound(&mut line[..SIDE], values.iter().copied(), precision);
            }
            return;
        }
    }
    warp_affine_8x8t(
        tmp,
        stride,
        src.0,
        src.1,
        src.2,
        shear,
        phase,
        BitDepth::EIGHT,
    );
}

/// [`warp_affine_8x8`] at compound precision into `tmp` (rows
/// `tmp_stride` apart; dav2d `warp_affine_8x8t_c`).
///
/// # Panics
///
/// If a slice is too short or a phase leaves the filter table.
pub fn warp_affine_8x8t<T: Sample>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[T],
    src_offset: usize,
    src_stride: usize,
    shear: Shear,
    (mx, my): (i32, i32),
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let mid = horizontal_pass(
        (src, src_offset, src_stride),
        shear,
        mx,
        WARP_FILTER_BITS - precision.bits,
    );
    for (y, line) in tmp.chunks_mut(tmp_stride).take(SIDE).enumerate() {
        let values = vertical_row(&mid, shear, my, y).map(|sum| round_shift(sum, WARP_FILTER_BITS));
        store_compound(&mut line[..SIDE], values, precision);
    }
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

    /// Values captured from dav2d's `warp_affine_8x8{,t}_c` (8-bit) on
    /// [`picture`] with a sheared matrix.
    #[test]
    fn matches_dav2d() {
        let src = picture();
        let shear = [100, -200, 300, -50];
        let phase = (5632, 42944);
        let mut dst = [0_u8; 64];
        warp_affine_8x8(
            &mut dst,
            0,
            8,
            &src,
            ORIGIN,
            15,
            shear,
            phase,
            BitDepth::EIGHT,
        );
        assert_eq!(dst[..8], [248, 74, 26, 64, 93, 212, 229, 63]);
        assert_eq!(dst[56..], [32, 151, 54, 55, 196, 55, 45, 173]);
        let mut tmp = [0_i16; 64];
        warp_affine_8x8t(&mut tmp, 8, &src, ORIGIN, 15, shear, phase, BitDepth::EIGHT);
        assert_eq!(tmp[32..40], [2562, 4087, 180, 1475, 3405, 1034, 1848, 1597]);
    }
}
