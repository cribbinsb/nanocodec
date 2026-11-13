//! The Wiener filter of loop restoration (AV1 §7.17.4): a separable,
//! symmetric 7-tap filter, horizontal pass first, whose centre tap makes
//! each direction sum to 128.
//!
//! Scheduled as dav1d 1.5.4 `src/looprestoration_tmpl.c` `wiener_c` does
//! (BSD-2-Clause, see `THIRD_PARTY.md`): each window row is filtered
//! horizontally into a ring of seven 16-bit rows, and every output row is
//! the vertical filter of the seven most recent ones, so the intermediate
//! stays in L1 whatever the unit size.
//!
//! The intermediate is dav1d's 8-bit form, which the SIMD kernels compute
//! with 16-bit wrapping and saturating arithmetic: the horizontal sum `S`
//! rounded by 3 bits and clamped to `-2048..=6143` (the spec's
//! `InterRound0` and intermediate clamp, as libaom's
//! `av1_wiener_convolve_add_src_c`), plus [`ROUND_BIAS`] so that the
//! vertical taps (summing to 128) add the vertical rounding: the output is
//! `clip(Σ v_k · h_k >> 11)`.

use nc_common::PlaneViewMut;

use super::kernels::Kernels;
use super::{AV1_LR_BORDER, AV1_LR_OVERREAD, Av1Wiener};

/// `FILTER_BITS`: the taps of each direction sum to `1 << 7`.
const FILTER_BITS: u32 = 7;
/// `InterRound0` of the horizontal pass.
pub(super) const ROUND0: u32 = 3;
/// `InterRound1` of the vertical pass.
pub(super) const ROUND1: u32 = 2 * FILTER_BITS - ROUND0;
/// Smallest horizontal output (`-(1 << (8 + 7 - 3 - 1))`).
pub(super) const H_MIN: i32 = -(1 << (8 + FILTER_BITS - ROUND0 - 1));
/// Largest horizontal output (`WIENER_CLAMP_LIMIT - 1` less the offset).
pub(super) const H_MAX: i32 = (1 << (8 + 1 + FILTER_BITS - ROUND0)) - 1 + H_MIN;
/// Added to each horizontal output: the vertical taps sum to
/// `1 << FILTER_BITS`, so the vertical sum gains `1 << (ROUND1 - 1)`, its
/// rounding term.
pub(super) const ROUND_BIAS: i32 = 1 << (ROUND1 - 1 - FILTER_BITS);
/// The 16-bit kernels' form of the clamp: they add `128 · centre` and
/// this offset with a saturating add, so that after the 3-bit shift the
/// 16-bit range `-2^15 ..= 2^15 - 1` lands on `H_MIN ..= H_MAX` less
/// `H_MIN` (`(S + 4 + (H_MIN << 3)) >> 3 = ((S + 4) >> 3) + H_MIN`).
pub(super) const SATURATION_OFFSET: i16 = ((1 << (ROUND0 - 1)) + (H_MIN << ROUND0)) as i16;
/// Added after the shift: undoes the offset's `H_MIN` and adds
/// [`ROUND_BIAS`].
pub(super) const SATURATION_BIAS: i16 = (ROUND_BIAS - H_MIN) as i16;
/// Filter taps.
const TAPS: usize = 7;

/// Taps of one direction as the kernels take them: the three outer taps
/// and the centre tap. Horizontal centres are stored less 128 (the
/// kernels add `128 · centre sample` separately, which keeps the
/// remaining 16-bit sum within range: `|Σ| ≤ 510 · 64`).
pub(crate) type WienerTaps = [i16; 4];

/// The horizontal and vertical kernel taps of `coefficients`.
pub(super) fn kernel_taps(coefficients: Av1Wiener) -> (WienerTaps, WienerTaps) {
    let taps = |c: [i8; 3]| {
        let [c0, c1, c2] = c.map(i16::from);
        [c0, c1, c2, (1 << FILTER_BITS) - 2 * (c0 + c1 + c2)]
    };
    let mut horizontal = taps(coefficients.horizontal);
    horizontal[3] -= 1 << FILTER_BITS;
    (horizontal, taps(coefficients.vertical))
}

/// Scalar reference of the horizontal row kernel: `out[x]` for
/// `x < n` from `src[x..x + 7]`.
pub(super) fn h_row(src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize) {
    let [c0, c1, c2, centre] = taps.map(i32::from);
    let centre = centre + (1 << FILTER_BITS);
    for (x, out) in out[..n].iter_mut().enumerate() {
        let p = |k: usize| i32::from(src[x + k]);
        let sum = c0 * (p(0) + p(6)) + c1 * (p(1) + p(5)) + c2 * (p(2) + p(4)) + centre * p(3);
        let rounded = (sum + (1 << (ROUND0 - 1))) >> ROUND0;
        *out = (rounded.clamp(H_MIN, H_MAX) + ROUND_BIAS) as i16;
    }
}

/// Scalar reference of the vertical row kernel: `dst[x]` from
/// `rows[0..7][x]`, top to bottom.
pub(super) fn v_row(rows: &[&[i16]; TAPS], taps: WienerTaps, dst: &mut [u8]) {
    let [c0, c1, c2, centre] = taps.map(i32::from);
    for (x, dst) in dst.iter_mut().enumerate() {
        let r = |k: usize| i32::from(rows[k][x]);
        let sum = c0 * (r(0) + r(6)) + c1 * (r(1) + r(5)) + c2 * (r(2) + r(4)) + centre * r(3);
        *dst = (sum >> ROUND1).clamp(0, 255) as u8;
    }
}

/// Wiener-filter a `w`x`h` region (AV1 §7.17.4). `window` holds the
/// region with [`AV1_LR_BORDER`] source samples on every side, rows
/// `stride` apart, each readable [`AV1_LR_OVERREAD`] samples past its end;
/// the result goes to `dst` at `(x, y)`.
pub(super) fn wiener(
    kernels: &Kernels,
    coefficients: Av1Wiener,
    (window, stride): (&[u8], usize),
    (w, h): (usize, usize),
    ring: &mut Vec<i16>,
    (dst, x, y): (&mut PlaneViewMut<'_>, usize, usize),
) {
    let (horizontal, vertical) = kernel_taps(coefficients);
    let row_len = w + AV1_LR_OVERREAD;
    let ring = super::grow(ring, TAPS * row_len);
    let source_len = w + 2 * AV1_LR_BORDER + AV1_LR_OVERREAD;
    for t in 0..h + 2 * AV1_LR_BORDER {
        let slot = t % TAPS;
        kernels.wiener_h(
            &window[t * stride..t * stride + source_len],
            horizontal,
            &mut ring[slot * row_len..(slot + 1) * row_len],
            w,
        );
        if let Some(r) = t.checked_sub(TAPS - 1) {
            let rows: [&[i16]; TAPS] = core::array::from_fn(|k| {
                let slot = (r + k) % TAPS;
                &ring[slot * row_len..(slot + 1) * row_len]
            });
            kernels.wiener_v(&rows, vertical, &mut dst.row_mut(y + r)[x..x + w]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel form equals the spec's: libaom's rounding and clamp of
    /// the horizontal sum, then the vertical sum rounded by 11 bits.
    #[test]
    fn kernel_form_matches_the_spec_arithmetic() {
        let coefficients = Av1Wiener {
            vertical: [3, -7, 15],
            horizontal: [-5, 8, 46],
        };
        let (horizontal, vertical) = kernel_taps(coefficients);
        let spec_taps = |c: [i8; 3]| {
            let [c0, c1, c2] = c.map(i32::from);
            [c0, c1, c2, 128 - 2 * (c0 + c1 + c2), c2, c1, c0]
        };
        let (h_spec, v_spec) = (
            spec_taps(coefficients.horizontal),
            spec_taps(coefficients.vertical),
        );
        let src: Vec<u8> = (0..64).map(|i| ((i * 97 + 13) % 256) as u8).collect();
        let n = src.len() - 6;
        let mut rows = vec![vec![0i16; n]; TAPS];
        let mut spec_rows = vec![vec![0i32; n]; TAPS];
        for (k, (row, spec)) in rows.iter_mut().zip(&mut spec_rows).enumerate() {
            let shifted: Vec<u8> = src.iter().map(|&s| s.wrapping_add(k as u8 * 31)).collect();
            h_row(&shifted, horizontal, row, n);
            for (x, spec) in spec.iter_mut().enumerate() {
                let sum: i32 = (0..TAPS)
                    .map(|t| h_spec[t] * i32::from(shifted[x + t]))
                    .sum();
                *spec = ((sum + 4) >> 3).clamp(H_MIN, H_MAX);
            }
        }
        let refs: [&[i16]; TAPS] = core::array::from_fn(|k| rows[k].as_slice());
        let mut out = vec![0u8; n];
        v_row(&refs, vertical, &mut out);
        for (x, &out) in out.iter().enumerate() {
            let sum: i32 = (0..TAPS).map(|t| v_spec[t] * spec_rows[t][x]).sum();
            assert_eq!(
                i32::from(out),
                ((sum + 1024) >> 11).clamp(0, 255),
                "x = {x}"
            );
        }
    }
}
