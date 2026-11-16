//! Nonlinear VVC ALF rows using the same tap geometry as the dispatched linear filter.
//! The references are generic over the sample type with a runtime depth
//! (the clipping values `AlfClip` and the output clip depend on it).
use nc_common::{BitDepth, Sample};

use super::{VVC_ALF_BLOCK, VVC_ALF_REACH, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, clip_values};

/// Taps of the chroma 5x5 diamond besides the centre.
const CHROMA_TAPS: usize = 6;
/// `(dx, dy)` of the first sample of each chroma tap's pair (§8.8.5.4).
const CHROMA_TAP_OFFSETS: [(isize, isize); CHROMA_TAPS] =
    [(0, 2), (1, 1), (0, 1), (-1, 1), (2, 0), (1, 0)];
/// `alfShiftY` / `alfShiftC` (§8.8.5.2, §8.8.5.4): 10 on the rows next to
/// a virtual boundary, 7 elsewhere.
const SHIFTS: [u32; 2] = [7, 10];
/// The range of a signalled ALF coefficient (§7.4.3.18 `AlfCoeffL`,
/// `AlfCoeffC`: −2^7..=2^7 − 1).
const COEFFICIENTS: std::ops::RangeInclusive<i16> = -128..=127;

impl crate::Dsp {
    /// Filter an 8-bit luma row with individually clipped tap differences.
    /// Rows are prepared by the shared ALF boundary/edge-extension controller;
    /// each block of four samples selects twelve coefficients and clip limits
    /// (`AlfClip` at 8 bits, [`crate::vvc_alf_clip_value`]).
    /// # Panics
    /// If rows/tables are short, limits are invalid, or shift is not seven/ten.
    pub fn vvc_alf_filter_clipped_row(
        &self,
        rows: &[&[u8]; 7],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shift: u32,
        out: &mut [u8],
    ) {
        check_clipped_row(rows, coefficients, limits, shift, out, BitDepth::EIGHT);
        clipped_row_at(rows, coefficients, limits, shift, out, BitDepth::EIGHT);
    }
    /// Filter a chroma row with the six symmetric 5×5 ALF taps.
    /// Rows use the luma kernel's three-sample padding and boundary preparation.
    /// # Panics
    /// If rows are short, coefficients/limits are invalid, or shift is not 7/10.
    pub fn vvc_alf_filter_chroma_row(
        &self,
        rows: &[&[u8]; 7],
        coefficients: [i16; 6],
        limits: [i16; 6],
        shift: u32,
        out: &mut [u8],
    ) {
        check_chroma_row(rows, coefficients, limits, shift, out, BitDepth::EIGHT);
        chroma_row_at(rows, coefficients, limits, shift, out, BitDepth::EIGHT);
    }
}

/// The checks of the clipped luma row at `depth` (any sample type).
pub(crate) fn check_clipped_row<T: Sample>(
    rows: &[&[T]; 7],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shift: u32,
    out: &[T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let blocks = out.len().div_ceil(VVC_ALF_BLOCK);
    assert!(
        rows.iter()
            .all(|r| r.len() >= out.len() + 2 * VVC_ALF_REACH)
    );
    assert!(coefficients.len() >= blocks && limits.len() >= blocks && SHIFTS.contains(&shift));
    assert!(
        coefficients[..blocks]
            .iter()
            .flatten()
            .all(|c| COEFFICIENTS.contains(c))
    );
    let clip_values = clip_values(depth);
    assert!(
        limits[..blocks]
            .iter()
            .flatten()
            .all(|v| clip_values.contains(v))
    );
}

/// The checks of the chroma row at `depth` (any sample type).
pub(crate) fn check_chroma_row<T: Sample>(
    rows: &[&[T]; 7],
    coefficients: [i16; CHROMA_TAPS],
    limits: [i16; CHROMA_TAPS],
    shift: u32,
    out: &[T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    assert!(
        rows.iter()
            .all(|r| r.len() >= out.len() + 2 * VVC_ALF_REACH)
    );
    assert!(SHIFTS.contains(&shift));
    assert!(coefficients.iter().all(|c| COEFFICIENTS.contains(c)));
    let clip_values = clip_values(depth);
    assert!(limits.iter().all(|v| clip_values.contains(v)));
}

/// The clipped luma row at any sample type and depth: `limits` are
/// `AlfClip` at `depth`.
pub(crate) fn clipped_row_at<T: Sample>(
    rows: &[&[T]; 7],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shift: u32,
    out: &mut [T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    filter_clipped(
        rows,
        |b| (coefficients[b], limits[b]),
        &VVC_ALF_TAP_OFFSETS,
        shift,
        out,
        depth,
    );
}

/// The chroma row at any sample type and depth: `limits` are `AlfClip` at
/// `depth`.
pub(crate) fn chroma_row_at<T: Sample>(
    rows: &[&[T]; 7],
    coefficients: [i16; CHROMA_TAPS],
    limits: [i16; CHROMA_TAPS],
    shift: u32,
    out: &mut [T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    filter_clipped(
        rows,
        |_| (coefficients, limits),
        &CHROMA_TAP_OFFSETS,
        shift,
        out,
        depth,
    );
}

/// `centre + round(Σ c_k (Clip3(−l_k, l_k, a_k − centre) + Clip3(−l_k,
/// l_k, b_k − centre)))`, clipped to `depth` (§8.8.5.2 eq. 1427,
/// §8.8.5.4).
fn filter_clipped<T: Sample, const N: usize>(
    rows: &[&[T]; 7],
    tables: impl Fn(usize) -> ([i16; N], [i16; N]),
    offsets: &[(isize, isize); N],
    shift: u32,
    out: &mut [T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let sample = |x: usize, dx: isize, dy: isize| {
        rows[(3 + dy) as usize][(x as isize + VVC_ALF_REACH as isize + dx) as usize].to_i32()
    };
    for (x, pixel) in out.iter_mut().enumerate() {
        let centre = sample(x, 0, 0);
        let (coefficients, limits) = tables(x / VVC_ALF_BLOCK);
        let sum: i32 = offsets
            .iter()
            .enumerate()
            .map(|(tap, &(dx, dy))| {
                let limit = i32::from(limits[tap]);
                let difference = (sample(x, dx, dy) - centre).clamp(-limit, limit)
                    + (sample(x, -dx, -dy) - centre).clamp(-limit, limit);
                i32::from(coefficients[tap]) * difference
            })
            .sum();
        *pixel = depth.clip_to(centre + ((sum + (1 << (shift - 1))) >> shift));
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn no_clipping_matches_the_shared_linear_filter() {
        let dsp = crate::Dsp::new();
        let samples: [[u8; 70]; 7] =
            std::array::from_fn(|y| std::array::from_fn(|x| ((x * 37 + y * 83) % 256) as u8));
        let rows = samples.each_ref().map(<[u8; 70]>::as_slice);
        let coefficients: [[i16; 12]; 16] =
            std::array::from_fn(|b| std::array::from_fn(|k| ((b * 17 + k * 11) % 31) as i16 - 15));
        for shift in [7, 10] {
            let mut expected = [0; 64];
            super::super::filter_row(&rows, &coefficients, shift, &mut expected);
            let mut actual = [0; 64];
            dsp.vvc_alf_filter_clipped_row(
                &rows,
                &coefficients,
                &[[256; 12]; 16],
                shift,
                &mut actual,
            );
            assert_eq!(actual, expected);
        }
    }
}
