//! AV1 directional intra prediction at any angle (AV1 §7.11.2.4), without
//! the intra edge filter and edge upsampling (`enable_intra_edge_filter = 0`
//! in our sequence header): every sample interpolates two edge samples at
//! 1/32 precision.
//!
//! Scalar reference. The derivative table and the zone rules follow libaom
//! `av1/common/reconintra.{c,h}` (`dr_intra_derivative`,
//! `av1_dr_prediction_z{1,2,3}_c`; BSD-2-Clause, Copyright (c) 2016, Alliance
//! for Open Media; see `THIRD_PARTY.md`).
//!
//! The zones are written once over [`DirectionalKernels`]; each ISA
//! instantiates them under its target feature (`crate::av1_intra`), so the
//! whole call chain is `#[inline(always)]`.

#![allow(clippy::inline_always)]

use nc_common::PlaneViewMut;

use super::{EDGE_LEN, IntraEdges, MAX_INTRA_SIZE};

/// Angle of pure vertical prediction (degrees).
const VERTICAL: u16 = 90;
/// Angle of pure horizontal prediction (degrees).
const HORIZONTAL: u16 = 180;
/// Directional angles lie strictly between zero and this.
const MAX_ANGLE: u16 = 270;
/// Fractional bits of an edge position.
const POSITION_BITS: u32 = 6;
/// Interpolation weights sum to `1 << WEIGHT_BITS`.
const WEIGHT_BITS: u32 = 5;
const WEIGHT_SCALE: i32 = 1 << WEIGHT_BITS;
/// Mask of a position's interpolation weight after dropping one bit.
const WEIGHT_MASK: i32 = WEIGHT_SCALE - 1;
/// Largest zone-2 side predicted with per-sample left-column reads rather
/// than column runs and a transpose.
const ZONE2_DIRECT_MAX: usize = 4;
/// Zone-2 column runs start at a multiple of this row, so that their
/// lengths are whole vector steps.
const COLUMN_RUN_ALIGN: usize = 8;
/// Samples in front of a line's corner: a column run starting up to
/// `COLUMN_RUN_ALIGN − 1` rows early reads that far before the corner.
const LINE_LEAD: usize = COLUMN_RUN_ALIGN - 1;
/// An edge with its corner and lead in front, padded so that a run of up
/// to `MAX_INTRA_SIZE` samples starting anywhere on the edge stays inside.
const LINE_LEN: usize = LINE_LEAD + 1 + EDGE_LEN + MAX_INTRA_SIZE + 1;

/// Position step per unit distance, 64 × cot(angle), at the angles the
/// nominal directions and their deltas reach (`dr_intra_derivative`; the
/// zero entries are never read).
pub(super) const DERIVATIVE: [i32; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0,
    132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0,
    45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0,
    0, 7, 0, 0, 3, 0, 0,
];

/// The kernels directional prediction calls: the scalar reference
/// ([`ScalarDirectional`]) or an ISA's (`crate::av1_intra`), which
/// instantiates [`IntraEdges::predict_angle_with`] under its target
/// feature so that everything inlines.
pub(crate) trait DirectionalKernels {
    /// `out[k] = Round2(window[k] · (32 − weight) + window[k + 1] · weight,
    /// 5)`, with `window.len() == out.len() + 1` and `weight < 32`.
    fn run(window: &[u8], weight: usize, out: &mut [u8]);
    /// `dst` sample `(i, j)` = `columns[j * side + i]`, `side`x`side`.
    fn store_transposed(columns: &[u8], side: usize, dst: &mut PlaneViewMut<'_>);
}

/// The scalar reference kernels.
pub(crate) struct ScalarDirectional;

impl DirectionalKernels for ScalarDirectional {
    #[inline(always)]
    fn run(window: &[u8], weight: usize, out: &mut [u8]) {
        interpolate_run(window, weight, out);
    }

    #[inline(always)]
    fn store_transposed(columns: &[u8], side: usize, dst: &mut PlaneViewMut<'_>) {
        store_transposed(columns, side, dst);
    }
}

impl IntraEdges {
    /// Predict the square from the edges at `angle` degrees (a nominal
    /// direction plus `3 * angle_delta`), with the scalar kernels
    /// (`Dsp::av1_intra_predict_angle` runs this ISA's).
    ///
    /// # Panics
    ///
    /// If `angle` is not strictly between 0 and 270 degrees, or the
    /// destination does not match the edge size.
    pub fn predict_angle(&self, angle: u16, dst: &mut PlaneViewMut<'_>) {
        self.predict_angle_with::<ScalarDirectional>(angle, dst);
    }

    /// [`Self::predict_angle`] with the kernels `K`.
    #[inline(always)]
    pub(crate) fn predict_angle_with<K: DirectionalKernels>(
        &self,
        angle: u16,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let n = self.size;
        assert!(dst.width() == n && dst.height() == n);
        assert!(angle > 0 && angle < MAX_ANGLE, "directional angle {angle}");
        match angle {
            VERTICAL => (0..n).for_each(|i| dst.row_mut(i)[..n].copy_from_slice(&self.above[..n])),
            HORIZONTAL => (0..n).for_each(|i| dst.row_mut(i)[..n].fill(self.left[i])),
            _ => match n {
                4 => self.predict_zone::<K, 4>(angle, dst),
                8 => self.predict_zone::<K, 8>(angle, dst),
                16 => self.predict_zone::<K, 16>(angle, dst),
                32 => self.predict_zone::<K, 32>(angle, dst),
                _ => self.predict_zone::<K, MAX_INTRA_SIZE>(angle, dst),
            },
        }
    }

    /// The oblique angles on a side known at compile time (`N` = the edge
    /// size), so the runs have fixed lengths and the tiles only `N²` bytes.
    #[inline(always)]
    fn predict_zone<K: DirectionalKernels, const N: usize>(
        &self,
        angle: u16,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert_eq!(self.size, N);
        match angle {
            a if a < VERTICAL => self.zone1::<K, N>(DERIVATIVE[usize::from(a)], dst),
            a if a < HORIZONTAL => self.zone2::<K, N>(
                DERIVATIVE[usize::from(HORIZONTAL - a)],
                DERIVATIVE[usize::from(a - VERTICAL)],
                dst,
            ),
            a => self.zone3::<K, N>(DERIVATIVE[usize::from(MAX_ANGLE - a)], dst),
        }
    }

    /// `AboveRow[-1..2n]` (or `LeftCol`) as one array: the corner at index
    /// `LINE_LEAD`, edge sample `k` at `LINE_LEAD + k + 1`, so every read is
    /// a plain index. Past `2n` the line repeats `AboveRow[2n − 1]`: a run
    /// reading beyond the edge then interpolates that sample with itself,
    /// which is the sample the spec clamps to, so runs need no end test.
    /// The lead (outputs that are overwritten) holds the corner.
    #[inline(always)]
    fn line(&self, edge: &[u8; EDGE_LEN]) -> [u8; LINE_LEN] {
        let n = self.size;
        let mut line = [edge[2 * n - 1]; LINE_LEN];
        line[..=LINE_LEAD].fill(self.corner);
        line[LINE_LEAD + 1..][..2 * n].copy_from_slice(&edge[..2 * n]);
        line
    }

    /// Angles below 90: from the above row only, which ends at
    /// `AboveRow[2n − 1]`; samples projecting past it repeat it.
    #[inline(always)]
    fn zone1<K: DirectionalKernels, const N: usize>(&self, dx: i32, dst: &mut PlaneViewMut<'_>) {
        let line = self.line(&self.above);
        for i in 0..N {
            projected_run::<K, N>(&line, (i as i32 + 1) * dx, &mut dst.row_mut(i)[..N]);
        }
    }

    /// Angles between 90 and 180: from the above row where the projection
    /// lands on it (or its corner), else from the left column. In row `i`
    /// that is every column from `ceil(((i + 1) * dx − 64) / 64)` on.
    #[inline(always)]
    fn zone2<K: DirectionalKernels, const N: usize>(
        &self,
        dx: i32,
        dy: i32,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let (above, left) = (self.line(&self.above), self.line(&self.left));
        let unit = 1 << POSITION_BITS;
        let mut splits = [0; N];
        for (i, split) in splits.iter_mut().enumerate() {
            let reach = (i as i32 + 1) * dx - unit;
            *split = if reach <= 0 {
                0
            } else {
                ((reach + unit - 1) >> POSITION_BITS).min(N as i32) as usize
            };
        }
        if N <= ZONE2_DIRECT_MAX {
            // 4x4 blocks read the left column sample by sample: their
            // column runs are shorter than a vector step. From 8x8 the
            // column runs and the transpose win (Jetson NEON 8x8 at 129
            // degrees 210 → 170 ns, at 157 degrees 247 → 129 ns).
            for (i, &split) in splits.iter().enumerate() {
                for (j, sample) in dst.row_mut(i)[..split].iter_mut().enumerate() {
                    let position = ((i as i32) << POSITION_BITS) - (j as i32 + 1) * dy;
                    let at = line_index(position);
                    *sample = interpolate(left[at], left[at + 1], weight(position));
                }
            }
        } else {
            // Down a column the left projections step one whole sample per
            // row at one weight, so each column's left part is a single run
            // (into the transposed tile) instead of per-sample indexed
            // reads. Splits grow with the row, so column `j` takes the left
            // column from the first row whose split exceeds `j` down. The
            // run starts up to `COLUMN_RUN_ALIGN − 1` rows earlier, so that
            // it is whole vector steps: rows above the first are the above
            // part's (overwritten below), and their projections stay within
            // the line's lead because the first row's lands at or after
            // the corner (AV1 §7.11.2.4: `base_y ≥ −1`).
            let mut columns = [[0; N]; N];
            let mut first = 0;
            for (j, column) in columns.iter_mut().enumerate().take(splits[N - 1]) {
                while splits[first] <= j {
                    first += 1;
                }
                let start = first - first % COLUMN_RUN_ALIGN;
                let position = ((start as i32) << POSITION_BITS) - (j as i32 + 1) * dy;
                let at = line_index(position);
                let out = &mut column[start..];
                K::run(&left[at..=at + out.len()], weight(position), out);
            }
            // The whole tile goes down transposed (columns never reached
            // hold zeros); the above runs then overwrite each row from its
            // split.
            K::store_transposed(columns.as_flattened(), N, dst);
        }
        for (i, &split) in splits.iter().enumerate() {
            // Along the row the above projections step one whole sample per
            // column at one weight, so the above part is a single run.
            if split < N {
                let position = ((split as i32) << POSITION_BITS) - (i as i32 + 1) * dx;
                let at = line_index(position);
                let out = &mut dst.row_mut(i)[split..N];
                K::run(&above[at..=at + out.len()], weight(position), out);
            }
        }
    }

    /// Angles above 180: from the left column only, which ends at
    /// `LeftCol[2n − 1]`; zone 1 transposed. The columns are runs into a
    /// tile, transposed into the destination once.
    #[inline(always)]
    fn zone3<K: DirectionalKernels, const N: usize>(&self, dy: i32, dst: &mut PlaneViewMut<'_>) {
        let line = self.line(&self.left);
        let mut columns = [[0; N]; N];
        for (j, column) in columns.iter_mut().enumerate() {
            projected_run::<K, N>(&line, (j as i32 + 1) * dy, column);
        }
        K::store_transposed(columns.as_flattened(), N, dst);
    }
}

/// The line index of the first sample edge `position` (1/64 sample, from
/// the edge's first sample) interpolates.
#[inline(always)]
fn line_index(position: i32) -> usize {
    let index = (position >> POSITION_BITS) + 1 + LINE_LEAD as i32;
    usize::try_from(index).expect("projections stay within the line's lead")
}

/// The interpolation weight (1/32) of edge `position` (1/64 sample).
#[inline(always)]
fn weight(position: i32) -> usize {
    ((position >> 1) & WEIGHT_MASK) as usize
}

/// One row of zone 1 (or column of zone 3) at edge `position` (1/64
/// sample): `N` samples interpolated from `line` at one weight. A start at
/// or past `2N − 1` reads only the line's repeated last sample.
#[inline(always)]
fn projected_run<K: DirectionalKernels, const N: usize>(
    line: &[u8; LINE_LEN],
    position: i32,
    out: &mut [u8],
) {
    let start = ((position >> POSITION_BITS) as usize).min(2 * N - 1);
    K::run(
        &line[LINE_LEAD + start + 1..][..=N],
        weight(position),
        &mut out[..N],
    );
}

/// `Round2(a · (32 − weight) + b · weight, 5)` in 16 bits (the sum is at
/// most `255 · 32 + 16`), which lets runs vectorise on 16-bit lanes.
#[inline(always)]
fn interpolate(a: u8, b: u8, weight: usize) -> u8 {
    let far = weight as u16;
    let near = WEIGHT_SCALE as u16 - far;
    ((u16::from(a) * near + u16::from(b) * far + (1 << (WEIGHT_BITS - 1))) >> WEIGHT_BITS) as u8
}

/// Scalar [`DirectionalKernels::run`].
#[inline(always)]
pub(crate) fn interpolate_run(window: &[u8], weight: usize, out: &mut [u8]) {
    let len = out.len();
    let (first, second) = (&window[..len], &window[1..=len]);
    for ((sample, &a), &b) in out.iter_mut().zip(first).zip(second) {
        *sample = interpolate(a, b, weight);
    }
}

/// Scalar [`DirectionalKernels::store_transposed`].
#[inline(always)]
pub(crate) fn store_transposed(columns: &[u8], side: usize, dst: &mut PlaneViewMut<'_>) {
    let columns = &columns[..side * side];
    for i in 0..side {
        for (sample, column) in dst.row_mut(i)[..side]
            .iter_mut()
            .zip(columns.chunks_exact(side))
        {
            *sample = column[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_prediction::{EdgeExtension, IntraMode};
    use nc_common::Plane;

    /// A picture with distinct samples, so a wrong edge index shows.
    fn ramp(width: usize, height: usize) -> Plane {
        let mut plane = Plane::new(width, height, 0).unwrap();
        let mut view = plane.view_mut();
        for y in 0..height {
            for (x, sample) in view.row_mut(y)[..width].iter_mut().enumerate() {
                *sample = ((x * 7 + y * 13) % 251) as u8;
            }
        }
        plane
    }

    fn predict(edges: &IntraEdges, angle: u16, n: usize) -> Plane {
        let mut dst = Plane::new(n, n, 0).unwrap();
        edges.predict_angle(angle, &mut dst.view_mut());
        dst
    }

    #[test]
    fn exact_diagonals_copy_edge_samples() {
        // At 45 degrees dx = 64: sample (i, j) is AboveRow[i + j + 1].
        let plane = ramp(32, 32);
        let edges = IntraEdges::extended(
            &plane,
            8,
            8,
            8,
            EdgeExtension {
                above_right: true,
                below_left: true,
            },
        );
        let out = predict(&edges, 45, 8);
        for i in 0..8 {
            for j in 0..8 {
                let k = (i + j + 1).min(15);
                assert_eq!(out.view().row(i)[j], plane.view().row(7)[8 + k]);
            }
        }
        // At 135 degrees: (i, j) is AboveRow[j − i − 1], the corner on the
        // diagonal and LeftCol[i − j − 1] below it.
        let out = predict(&edges, 135, 8);
        for i in 0..8 {
            for j in 0..8 {
                let expected = match j.cmp(&i) {
                    std::cmp::Ordering::Greater => plane.view().row(7)[8 + j - i - 1],
                    std::cmp::Ordering::Equal => plane.view().row(7)[7],
                    std::cmp::Ordering::Less => plane.view().row(8 + i - j - 1)[7],
                };
                assert_eq!(out.view().row(i)[j], expected, "({i}, {j})");
            }
        }
    }

    /// AV1 §7.11.2.4 with `upsampleAbove = upsampleLeft = 0`, sample by
    /// sample, as the oracle for the row-wise kernels.
    fn spec_prediction(edges: &IntraEdges, angle: u16) -> Vec<u8> {
        let n = edges.size as i32;
        let at = |edge: &[u8; EDGE_LEN], k: i32| {
            usize::try_from(k).map_or(i32::from(edges.corner), |k| i32::from(edge[k]))
        };
        let blend = |a: i32, b: i32, position: i32| {
            let w = (position >> 1) & 31;
            ((a * (32 - w) + b * w + 16) >> 5) as u8
        };
        let (a, max_base) = (i32::from(angle), 2 * n - 1);
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                out.push(if a < 90 {
                    let p = (i + 1) * DERIVATIVE[a as usize];
                    let base = (p >> 6) + j;
                    if base < max_base {
                        blend(at(&edges.above, base), at(&edges.above, base + 1), p)
                    } else {
                        edges.above[max_base as usize]
                    }
                } else if a == 90 {
                    edges.above[j as usize]
                } else if a < 180 {
                    let p = (j << 6) - (i + 1) * DERIVATIVE[(180 - a) as usize];
                    if p >> 6 >= -1 {
                        blend(at(&edges.above, p >> 6), at(&edges.above, (p >> 6) + 1), p)
                    } else {
                        let p = (i << 6) - (j + 1) * DERIVATIVE[(a - 90) as usize];
                        blend(at(&edges.left, p >> 6), at(&edges.left, (p >> 6) + 1), p)
                    }
                } else if a == 180 {
                    edges.left[i as usize]
                } else {
                    let p = (j + 1) * DERIVATIVE[(270 - a) as usize];
                    let base = (p >> 6) + i;
                    if base < max_base {
                        blend(at(&edges.left, base), at(&edges.left, base + 1), p)
                    } else {
                        edges.left[max_base as usize]
                    }
                });
            }
        }
        out
    }

    #[test]
    fn row_kernels_match_the_spec_at_every_angle() {
        let plane = ramp(160, 160);
        let mut seed = 0x2545_f491_u32;
        let mut noisy = plane.clone();
        for y in 0..160 {
            for sample in &mut noisy.view_mut().row_mut(y)[..160] {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *sample = (seed >> 24) as u8;
            }
        }
        for source in [&plane, &noisy] {
            for n in [4, 8, 16, 32, 64] {
                for extension in [
                    EdgeExtension::default(),
                    EdgeExtension {
                        above_right: true,
                        below_left: true,
                    },
                ] {
                    let edges = IntraEdges::extended(source, n, n, n, extension);
                    for mode in IntraMode::ALL {
                        let Some(nominal) = mode.nominal_angle() else {
                            continue;
                        };
                        for delta in -3i16..=3 {
                            let angle = (i16::try_from(nominal).unwrap() + 3 * delta) as u16;
                            let out = predict(&edges, angle, n);
                            let got: Vec<u8> = (0..n)
                                .flat_map(|y| out.view().row(y)[..n].to_vec())
                                .collect();
                            assert_eq!(got, spec_prediction(&edges, angle), "n {n} angle {angle}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn missing_extensions_repeat_the_last_edge_sample() {
        let plane = ramp(32, 32);
        let edges = IntraEdges::new(&plane, 8, 8, 8);
        let last = plane.view().row(7)[15];
        // At 45 degrees the bottom-right sample reads past the block's edge.
        assert_eq!(predict(&edges, 45, 8).view().row(7)[7], last);
        let last = plane.view().row(15)[7];
        assert_eq!(predict(&edges, 203, 8).view().row(7)[7], last);
    }

    #[test]
    fn nominal_axis_angles_match_the_axis_modes() {
        let plane = ramp(32, 32);
        let edges = IntraEdges::new(&plane, 16, 8, 8);
        for (mode, angle) in [(IntraMode::Vertical, 90), (IntraMode::Horizontal, 180)] {
            let mut by_mode = Plane::new(8, 8, 0).unwrap();
            edges.predict(mode, &mut by_mode.view_mut());
            let by_angle = predict(&edges, angle, 8);
            for y in 0..8 {
                assert_eq!(by_mode.view().row(y)[..8], by_angle.view().row(y)[..8]);
            }
        }
    }

    #[test]
    fn flat_edges_stay_flat_at_every_reachable_angle() {
        let mut plane = Plane::new(128, 128, 0).unwrap();
        plane.view_mut().fill(91);
        for n in [4, 8, 16, 32, 64] {
            let edges = IntraEdges::extended(
                &plane,
                n,
                n,
                n,
                EdgeExtension {
                    above_right: true,
                    below_left: true,
                },
            );
            for mode in IntraMode::ALL {
                let Some(nominal) = mode.nominal_angle() else {
                    continue;
                };
                for delta in -3i16..=3 {
                    let angle = (i16::try_from(nominal).unwrap() + 3 * delta) as u16;
                    let out = predict(&edges, angle, n);
                    for y in 0..n {
                        assert!(out.view().row(y)[..n].iter().all(|&v| v == 91), "{angle}");
                    }
                }
            }
        }
    }
}
