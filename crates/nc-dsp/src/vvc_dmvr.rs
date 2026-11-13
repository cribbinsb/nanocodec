//! VVC decoder motion-vector refinement on prepared bilinear luma predictions.
//! Search and fractional refinement follow BSD-Clear `VVdeC` `InterPrediction`.
#![allow(unsafe_code)]
use crate::Dsp;

const SEARCH_RADIUS: usize = 2;
const SEARCH_WIDTH: usize = 2 * SEARCH_RADIUS + 1;

fn fractional(center: i32, left: i32, right: i32) -> i32 {
    let denominator = left + right - 2 * center;
    if denominator == 0 {
        0
    } else if left == center {
        -8
    } else if right == center {
        8
    } else {
        let numerator = 16 * (left - right);
        numerator.signum() * (numerator.abs() / (2 * denominator)).min(7)
    }
}

impl Dsp {
    /// Bilinear interpolation with VVC DMVR's 10-bit internal precision.
    /// The input includes one extra column and row for the two-tap filter.
    /// # Panics
    /// If the window/output is too short or a phase exceeds 15.
    pub fn vvc_dmvr_bilinear(
        &self,
        size: (usize, usize),
        input: &nc_common::PlaneView<'_>,
        phases: (usize, usize),
        out: &mut [i16],
    ) {
        let (width, height) = size;
        assert!(phases.0 < 16 && phases.1 < 16);
        assert!(input.width() > width && input.height() > height && out.len() >= width * height);
        // SAFETY: window, phase and output checked; instance ISA is supported.
        unsafe {
            (self.vvc_refinement_kernels().bilinear)(size, input, phases, out);
        }
    }

    /// Search a DMVR region in two 10-bit bilinear prediction windows.
    /// Each window has a two-sample border; the returned displacement is in
    /// sixteenth-sample units. The boolean enables BDOF for this region.
    /// # Panics
    /// If dimensions are not 8 or 16, or either padded window is too short.
    #[must_use]
    pub fn vvc_dmvr(&self, size: (usize, usize), inputs: [&[i16]; 2]) -> ([i32; 2], bool) {
        let (width, height) = size;
        assert!([8, 16].contains(&width) && [8, 16].contains(&height));
        let stride = width + 2 * SEARCH_RADIUS;
        assert!(
            inputs
                .iter()
                .all(|p| p.len() >= stride * (height + 2 * SEARCH_RADIUS))
        );
        // SAFETY: dimensions and padded windows checked; instance ISA is supported.
        unsafe { (self.vvc_refinement_kernels().dmvr)(size, inputs) }
    }
}

pub(crate) fn bilinear(
    size: (usize, usize),
    input: &nc_common::PlaneView<'_>,
    phases: (usize, usize),
    out: &mut [i16],
) {
    let (width, height) = size;
    let (fx, fy) = (phases.0 as i32, phases.1 as i32);
    for y in 0..height {
        let top = input.row(y);
        let bottom = input.row(y + 1);
        for x in 0..width {
            let horizontal =
                |row: &[u8]| (i32::from(row[x]) * (16 - fx) + i32::from(row[x + 1]) * fx + 2) >> 2;
            out[y * width + x] =
                ((horizontal(top) * (16 - fy) + horizontal(bottom) * fy + 8) >> 4) as i16;
        }
    }
}
pub(crate) fn scalar(size: (usize, usize), inputs: [&[i16]; 2]) -> ([i32; 2], bool) {
    let (width, height) = size;
    let stride = width + 2 * SEARCH_RADIUS;
    let sad = |dx: i32, dy: i32| {
        let mut sum = 0;
        // H.266 DMVR samples every second row; the search's SAD normalization
        // cancels the factor of two used by the full-height SAD primitive.
        for y in (0..height).step_by(2) {
            for x in 0..width {
                let at = (y + SEARCH_RADIUS) * stride + x + SEARCH_RADIUS;
                let offset = dy * stride as i32 + dx;
                sum += (i32::from(inputs[0][(at as i32 + offset) as usize])
                    - i32::from(inputs[1][(at as i32 - offset) as usize]))
                .abs();
            }
        }
        sum
    };

    search(size, sad)
}
#[inline]
pub(crate) fn search(
    size: (usize, usize),
    mut sad: impl FnMut(i32, i32) -> i32,
) -> ([i32; 2], bool) {
    let (width, height) = size;
    let center = sad(0, 0);
    let biased_center = center - (center >> 2);
    let mut best = biased_center;
    let area = (width * height) as i32;
    if best < area {
        return ([0; 2], false);
    }
    let mut costs = [0; SEARCH_WIDTH * SEARCH_WIDTH];
    let mut displacement = [0; 2];
    for y in -2..=2 {
        for x in -2..=2 {
            let cost = if x == 0 && y == 0 {
                biased_center
            } else {
                sad(x, y)
            };
            costs[((y + 2) * SEARCH_WIDTH as i32 + x + 2) as usize] = cost;
            if cost < best {
                best = cost;
                displacement = [x, y];
            }
        }
    }
    let [x, y] = displacement;
    let mut vector = displacement.map(|v| v * 16);
    if x.abs() < 2 && y.abs() < 2 {
        let at = ((y + 2) * SEARCH_WIDTH as i32 + x + 2) as usize;
        vector[0] += fractional(costs[at], costs[at - 1], costs[at + 1]);
        vector[1] += fractional(
            costs[at],
            costs[at - SEARCH_WIDTH],
            costs[at + SEARCH_WIDTH],
        );
    }
    (vector, best >= 2 * area)
}

#[cfg(test)]
mod tests {
    #[test]
    fn captured_dmvr_regions_match_reference() {
        for dsp in crate::Isa::ALL.into_iter().filter_map(crate::Dsp::with_isa) {
            for &(size, first, second, expected) in super::fixtures::CASES {
                assert_eq!(dsp.vvc_dmvr(size, [first, second]), expected);
            }
        }
    }

    #[test]
    #[ignore = "requires NC_VVC_OPTICAL_TRACE from independent VVdeC"]
    fn prepared_dmvr_matches_reference() {
        let trace =
            std::fs::read_to_string(std::env::var("NC_VVC_OPTICAL_TRACE").unwrap()).unwrap();
        let dsp = crate::Dsp::new();
        let mut count = 0;
        for line in trace
            .lines()
            .filter_map(|line| line.strip_prefix("NC_DMVR "))
        {
            let values: Vec<i32> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let size = (values[0] as usize, values[1] as usize);
            let length = (size.0 + 4) * (size.1 + 4);
            assert_eq!(values.len(), 2 * length + 5);
            let inputs: Vec<i16> = values[2..2 + 2 * length]
                .iter()
                .map(|&v| v as i16)
                .collect();
            let actual = dsp.vvc_dmvr(size, [&inputs[..length], &inputs[length..]]);
            assert_eq!(
                actual,
                (
                    [values[2 + 2 * length], values[3 + 2 * length]],
                    values[4 + 2 * length] != 0
                ),
                "DMVR region {count}"
            );
            count += 1;
        }
        assert!(count > 0);
        eprintln!("{count} prepared DMVR regions match VVdeC");
    }
}

#[cfg(test)]
#[path = "vvc_dmvr_fixtures.rs"]
mod fixtures;
