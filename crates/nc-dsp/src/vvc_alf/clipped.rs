//! Nonlinear VVC ALF rows using the same tap geometry as the dispatched linear filter.
use super::{VVC_ALF_BLOCK, VVC_ALF_REACH, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS};
impl crate::Dsp {
    /// Filter an 8-bit luma row with individually clipped tap differences.
    /// Rows are prepared by the shared ALF boundary/edge-extension controller;
    /// each block of four samples selects twelve coefficients and clip limits.
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
        let blocks = out.len().div_ceil(VVC_ALF_BLOCK);
        assert!(
            rows.iter()
                .all(|r| r.len() >= out.len() + 2 * VVC_ALF_REACH)
        );
        assert!(coefficients.len() >= blocks && limits.len() >= blocks && [7, 10].contains(&shift));
        assert!(
            coefficients[..blocks]
                .iter()
                .flatten()
                .all(|&c| (-128..=127).contains(&c))
        );
        assert!(
            limits[..blocks]
                .iter()
                .flatten()
                .all(|&v| [2, 8, 32, 256].contains(&v))
        );
        filter_clipped(
            rows,
            |b| (coefficients[b], limits[b]),
            &VVC_ALF_TAP_OFFSETS,
            shift,
            out,
        );
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
        assert!(rows.iter().all(|r| r.len() >= out.len() + 6));
        assert!([7, 10].contains(&shift));
        assert!(coefficients.iter().all(|&c| (-128..=127).contains(&c)));
        assert!(limits.iter().all(|&v| [2, 8, 32, 256].contains(&v)));
        filter_clipped(
            rows,
            |_| (coefficients, limits),
            &[(0, 2), (1, 1), (0, 1), (-1, 1), (2, 0), (1, 0)],
            shift,
            out,
        );
    }
}
fn filter_clipped<const N: usize>(
    rows: &[&[u8]; 7],
    tables: impl Fn(usize) -> ([i16; N], [i16; N]),
    offsets: &[(isize, isize); N],
    shift: u32,
    out: &mut [u8],
) {
    let sample = |x: usize, dx: isize, dy: isize| {
        i32::from(rows[(3 + dy) as usize][(x as isize + 3 + dx) as usize])
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
        *pixel = (centre + ((sum + (1 << (shift - 1))) >> shift)).clamp(0, 255) as u8;
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
