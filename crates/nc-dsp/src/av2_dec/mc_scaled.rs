//! Motion compensation of the general AV2 decoder from a reference of
//! another size: the switchable 8-tap and bilinear filters stepping
//! through the reference by a fixed-point step per output sample
//! ([`put_scaled`], [`prep_scaled`]), and the 4x4 blocks of the extended
//! warp with a per-sample position ([`ext_warp4x4_scaled`]). Scalar only:
//! scaled references are rare (resolution changes), so no SIMD tier.
//!
//! Ported from dav2d `src/mc_tmpl.c` (`put_8tap_scaled_c`,
//! `prep_8tap_scaled_c`, `put_bilin_scaled_c`, `prep_bilin_scaled_c`,
//! `ext_warp4x4_scaled_c_internal`; commit 237229e, Copyright © 2018-2026
//! `VideoLAN` and dav2d authors, BSD-2-Clause, see `docs/licenses/dav2d`).
//! Same family as [`super::mc_8tap`] (taps and intermediate precision);
//! not shared with the AV1 decoder's scaled convolution for the reason
//! given there (AV1's spec rounding stages and window contract).
//!
//! Positions: the source offset is the reference sample at the block's
//! top-left integer position; `mx`, `my` are the fractions of that
//! position in 1/1024 sample and `dx`, `dy` the steps between output
//! samples (dav2d `mc_scaled`'s `pos & 0x3ff` and `svc[].step`). The
//! 8-tap kernels read 3 samples before and 4 after each filtered position,
//! the bilinear ones 1 after.

use nc_common::{BitDepth, Sample};

use super::mc::{MAX_SIDE, Precision, store_compound, store_pixels};
use super::mc_8tap::InterpFilter;
use super::mc_tables::EXT_WARP_FILTER;
use crate::mc::{TAP_CENTRE, TAPS};

/// Fraction bits of a scaled position (1/1024 sample).
const POSITION_BITS: u32 = 10;
const POSITION_MASK: i32 = (1 << POSITION_BITS) - 1;
/// A scaled position's fraction to a 1/16-sample filter phase.
const PHASE_SHIFT: u32 = 6;
/// Fraction bits of the (halved) interpolation taps.
const SUBPEL_FILTER_BITS: u32 = 6;
/// Fraction bits of the bilinear weights (1/16 sample).
const BILINEAR_BITS: u32 = 4;
/// Fraction bits of the extended-warp taps, and of its positions.
const EXT_WARP_FILTER_BITS: u32 = 7;
const EXT_WARP_POSITION_BITS: u32 = 16;
/// An extended-warp position's fraction to a 1/64-sample phase: the
/// rounding and shift (dav2d `((pos & 0xffff) + 512) >> 10`).
const EXT_WARP_PHASE_ROUND: i32 = 512;
const EXT_WARP_PHASE_SHIFT: u32 = 10;
/// Phases per extended-warp sample; phase 64 is phase 0 of the next sample.
const EXT_WARP_PHASES: i32 = 64;
/// Side of an extended-warp block.
const EXT_WARP_SIDE: usize = 4;

/// Where a scaled block reads: its fraction and step per direction.
#[derive(Debug, Clone, Copy)]
pub struct ScaledSteps {
    /// The start fractions (`mx`, `my`), 1/1024 sample.
    pub fraction: (i32, i32),
    /// The steps (`dx`, `dy`) between output samples, 1/1024 sample.
    pub step: (i32, i32),
}

/// Rounded right shift (`shift` may be 0).
#[inline]
fn round_shift(value: i32, shift: u32) -> i32 {
    (value + ((1 << shift) >> 1)) >> shift
}

/// The 8-tap sum of `taps` over `samples` (the tap centre at index 3).
#[inline]
fn filter8(taps: [i8; TAPS], sample: impl Fn(usize) -> i32) -> i32 {
    taps.iter()
        .enumerate()
        .map(|(k, &tap)| i32::from(tap) * sample(k))
        .sum()
}

/// The horizontally filtered rows a scaled block needs, from the row
/// 3 above its first source row: per row, each output column at
/// intermediate precision (dav2d `mid`).
#[allow(clippy::too_many_arguments, reason = "dav2d's scaled kernel contract")]
fn scaled_rows<T: Sample>(
    (src, src_offset, src_stride): (&[T], usize, usize),
    (w, h): (usize, usize),
    steps: ScaledSteps,
    filter: InterpFilter,
    precision: Precision,
    taps_before: usize,
    taps_after: usize,
) -> (Vec<i32>, usize) {
    let (mx, my) = steps.fraction;
    let (dx, dy) = steps.step;
    let last_row = ((my + (h as i32 - 1) * dy) >> POSITION_BITS) as usize;
    let rows = last_row + taps_before + taps_after + 1;
    let mut mid = vec![0; rows * w];
    // Each row from the first tap before the block's left column.
    let top = src_offset - taps_before * src_stride - taps_before;
    for (r, out) in mid.chunks_exact_mut(w).enumerate() {
        let row = &src[top + r * src_stride..];
        for (x, value) in out.iter_mut().enumerate() {
            let position = mx + x as i32 * dx;
            let at = (position >> POSITION_BITS) as usize;
            let phase = (position & POSITION_MASK) >> PHASE_SHIFT;
            *value = if filter == InterpFilter::Bilinear {
                let (a, b) = (row[at].to_i32(), row[at + 1].to_i32());
                round_shift(
                    (a << BILINEAR_BITS) + phase * (b - a),
                    BILINEAR_BITS - precision.bits,
                )
            } else {
                match filter.taps(phase, w) {
                    Some(taps) => round_shift(
                        filter8(*taps, |k| row[at + k].to_i32()),
                        SUBPEL_FILTER_BITS - precision.bits,
                    ),
                    None => row[at + TAP_CENTRE].to_i32() << precision.bits,
                }
            };
        }
    }
    (mid, rows)
}

/// The vertical stage of output row `row`: its value per column before
/// the final rounding, and whether it was filtered (unfiltered: the row
/// itself at intermediate precision).
fn vertical(
    mid: &[i32],
    width: usize,
    (row, steps): (usize, ScaledSteps),
    filter: InterpFilter,
    height: usize,
    taps_before: usize,
) -> impl Iterator<Item = (i32, bool)> + '_ {
    let position = steps.fraction.1 + row as i32 * steps.step.1;
    let first = (position >> POSITION_BITS) as usize + taps_before;
    let phase = (position & POSITION_MASK) >> PHASE_SHIFT;
    let taps = if filter == InterpFilter::Bilinear {
        None
    } else {
        filter.taps(phase, height)
    };
    (0..width).map(move |column| {
        if filter == InterpFilter::Bilinear {
            let upper = mid[first * width + column];
            let lower = mid[(first + 1) * width + column];
            ((upper << BILINEAR_BITS) + phase * (lower - upper), true)
        } else if let Some(taps) = taps {
            let sum = filter8(*taps, |k| mid[(first + k - TAP_CENTRE) * width + column]);
            (sum, true)
        } else {
            (mid[first * width + column], false)
        }
    })
}

/// The taps around a filtered position (the bilinear filter reads one
/// sample after it).
fn reach(filter: InterpFilter) -> (usize, usize) {
    if filter == InterpFilter::Bilinear {
        (0, 1)
    } else {
        (TAP_CENTRE, TAPS - 1 - TAP_CENTRE)
    }
}

/// Predict a `w` x `h` block (powers of two, 2 to 64) at `dst[dst_offset..]`
/// from a reference of another size (dav2d `put_8tap_scaled_c`,
/// `put_bilin_scaled_c`).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
#[allow(clippy::too_many_arguments, reason = "dav2d's mc_scaled contract")]
pub fn put_scaled<T: Sample>(
    (dst, dst_offset, dst_stride): (&mut [T], usize, usize),
    src: (&[T], usize, usize),
    (w, h): (usize, usize),
    steps: ScaledSteps,
    filter: InterpFilter,
    depth: BitDepth,
) {
    for side in [w, h] {
        assert!(side.is_power_of_two() && (2..=MAX_SIDE).contains(&side));
    }
    let precision = Precision::of(depth);
    let (before, after) = reach(filter);
    let (mid, _) = scaled_rows(src, (w, h), steps, filter, precision, before, after);
    let bits = if filter == InterpFilter::Bilinear {
        BILINEAR_BITS
    } else {
        SUBPEL_FILTER_BITS
    };
    for (y, line) in dst[dst_offset..].chunks_mut(dst_stride).take(h).enumerate() {
        let values = vertical(&mid, w, (y, steps), filter, h, before).map(|(value, filtered)| {
            if filtered {
                round_shift(value, bits + precision.bits)
            } else {
                round_shift(value, precision.bits)
            }
        });
        store_pixels(&mut line[..w], values, depth);
    }
}

/// [`put_scaled`] at compound precision into `tmp` (rows `tmp_stride`
/// apart; dav2d `prep_8tap_scaled_c`, `prep_bilin_scaled_c`).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
pub fn prep_scaled<T: Sample>(
    (tmp, tmp_stride): (&mut [i16], usize),
    src: (&[T], usize, usize),
    (w, h): (usize, usize),
    steps: ScaledSteps,
    filter: InterpFilter,
    depth: BitDepth,
) {
    for side in [w, h] {
        assert!(side.is_power_of_two() && (4..=MAX_SIDE).contains(&side));
    }
    let precision = Precision::of(depth);
    let (before, after) = reach(filter);
    let (mid, _) = scaled_rows(src, (w, h), steps, filter, precision, before, after);
    let bits = if filter == InterpFilter::Bilinear {
        BILINEAR_BITS
    } else {
        SUBPEL_FILTER_BITS
    };
    for (y, line) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
        let values = vertical(&mid, w, (y, steps), filter, h, before).map(|(value, filtered)| {
            if filtered {
                round_shift(value, bits)
            } else {
                value
            }
        });
        store_compound(&mut line[..w], values, precision);
    }
}

/// Where a 4x4 extended-warp block reads a reference of another size: the
/// 1/65536-sample fractions of its top-left sample relative to the source
/// offset and the steps between samples (dav2d `ext_warp_scaled`'s `mx`,
/// `my`, `dx`, `dy`).
#[derive(Debug, Clone, Copy)]
pub struct WarpSteps {
    /// The fractions of the top-left sample.
    pub fraction: (i32, i32),
    /// The steps between samples.
    pub step: (i32, i32),
}

/// The sample offset and 1/64 phase of an extended-warp position.
#[inline]
fn warp_phase(position: i32) -> (i32, i32) {
    let phase = ((position & ((1 << EXT_WARP_POSITION_BITS) - 1)) + EXT_WARP_PHASE_ROUND)
        >> EXT_WARP_PHASE_SHIFT;
    (
        (position >> EXT_WARP_POSITION_BITS) + phase / EXT_WARP_PHASES,
        phase % EXT_WARP_PHASES,
    )
}

/// The horizontal stage of a scaled extended-warp block: per source row
/// from 3 above, its 4 samples at intermediate precision.
fn ext_warp_rows<T: Sample>(
    (src, src_offset, src_stride): (&[T], usize, usize),
    steps: WarpSteps,
    precision: Precision,
) -> Vec<[i32; EXT_WARP_SIDE]> {
    let (mx, my) = steps.fraction;
    let (dx, dy) = steps.step;
    let rows = (((my + 3 * dy) >> EXT_WARP_POSITION_BITS) + TAPS as i32) as usize;
    let top = src_offset - TAP_CENTRE * src_stride - TAP_CENTRE;
    (0..rows)
        .map(|r| {
            let row = &src[top + r * src_stride..];
            std::array::from_fn(|x| {
                let (at, phase) = warp_phase(mx + x as i32 * dx);
                let at = at as usize;
                if phase == 0 {
                    row[at + TAP_CENTRE].to_i32() << precision.bits
                } else {
                    let taps = &EXT_WARP_FILTER[phase as usize - 1];
                    round_shift(
                        filter8(*taps, |k| row[at + k].to_i32()),
                        EXT_WARP_FILTER_BITS - precision.bits,
                    )
                }
            })
        })
        .collect()
}

/// The vertical stage of output row `y` of a scaled extended-warp block:
/// per sample, its unrounded value and whether it was filtered.
fn ext_warp_vertical(
    mid: &[[i32; EXT_WARP_SIDE]],
    steps: WarpSteps,
    y: usize,
) -> [(i32, bool); EXT_WARP_SIDE] {
    let (row, phase) = warp_phase(steps.fraction.1 + y as i32 * steps.step.1);
    let row = row as usize + TAP_CENTRE;
    std::array::from_fn(|x| {
        if phase == 0 {
            (mid[row][x], false)
        } else {
            let taps = &EXT_WARP_FILTER[phase as usize - 1];
            (filter8(*taps, |k| mid[row + k - TAP_CENTRE][x]), true)
        }
    })
}

/// A 4x4 extended-warp block into samples from a reference of another
/// size (dav2d `ext_warp4x4_scaled_c`).
pub fn ext_warp4x4_scaled<T: Sample>(
    (dst, dst_offset, dst_stride): (&mut [T], usize, usize),
    src: (&[T], usize, usize),
    steps: WarpSteps,
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let mid = ext_warp_rows(src, steps, precision);
    for (y, line) in dst[dst_offset..]
        .chunks_mut(dst_stride)
        .take(EXT_WARP_SIDE)
        .enumerate()
    {
        let values = ext_warp_vertical(&mid, steps, y).map(|(value, filtered)| {
            if filtered {
                round_shift(value, EXT_WARP_FILTER_BITS + precision.bits)
            } else {
                round_shift(value, precision.bits)
            }
        });
        store_pixels(&mut line[..EXT_WARP_SIDE], values.into_iter(), depth);
    }
}

/// A 4x4 extended-warp block at compound precision from a reference of
/// another size (dav2d `ext_warp4x4t_scaled_c`).
pub fn ext_warp4x4t_scaled<T: Sample>(
    (tmp, tmp_offset, tmp_stride): (&mut [i16], usize, usize),
    src: (&[T], usize, usize),
    steps: WarpSteps,
    depth: BitDepth,
) {
    let precision = Precision::of(depth);
    let mid = ext_warp_rows(src, steps, precision);
    for (y, line) in tmp[tmp_offset..]
        .chunks_mut(tmp_stride)
        .take(EXT_WARP_SIDE)
        .enumerate()
    {
        let values = ext_warp_vertical(&mid, steps, y).map(|(value, filtered)| {
            if filtered {
                round_shift(value, EXT_WARP_FILTER_BITS)
            } else {
                value
            }
        });
        store_compound(&mut line[..EXT_WARP_SIDE], values.into_iter(), precision);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scale of one (step 1024, no fraction) is the unscaled filter at
    /// phase 0: a copy.
    #[test]
    fn unit_step_copies() {
        let stride = 16;
        let src: Vec<u8> = (0..stride * 16).map(|i| (i * 7 % 251) as u8).collect();
        let mut dst = vec![0_u8; 4 * 4];
        let steps = ScaledSteps {
            fraction: (0, 0),
            step: (1 << POSITION_BITS, 1 << POSITION_BITS),
        };
        let offset = 4 * stride + 4;
        put_scaled(
            (&mut dst, 0, 4),
            (&src, offset, stride),
            (4, 4),
            steps,
            InterpFilter::Regular,
            BitDepth::EIGHT,
        );
        for y in 0..4 {
            assert_eq!(dst[y * 4..y * 4 + 4], src[offset + y * stride..][..4]);
        }
    }
}
