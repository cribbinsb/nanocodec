//! AV2 intra prediction of one square transform from prepared edges.
//!
//! The codec (nc-av2 `intra_edges`) decides which neighbours exist and
//! fills [`Av2IntraEdges`]; these kernels then form DC, smooth, Paeth and
//! directional predictions exactly as AVM v1.0.0 does with the intra edge
//! filter, intra bi-prediction and multiple reference lines disabled:
//!
//! - DC and Paeth equal AV1's;
//! - AV2's smooth predictors are not AV1's weight tables: each sample blends
//!   a linear ramp towards the bottom-left (or top-right) sample with the
//!   nearest edge sample by a weight that halves with distance
//!   (`highbd_smooth*_predictor` in `avm_dsp/intrapred.c`);
//! - directional luma uses AV2's 4-tap interpolation (`*_idif`), chroma the
//!   2-tap one (`av2_highbd_dr_prediction_z{1,2,3}_c` and
//!   `highbd_dr_predictor{,_idif}` in `av2/common/reconintra.c`).
//!
//! Scalar reference, 8-bit samples. Reimplemented from AVM (BSD-3-Clause-
//! Clear, see `THIRD_PARTY.md`).

use super::{DC_NO_NEIGHBOURS, paeth};

/// Largest transform side.
pub const AV2_INTRA_MAX_SIDE: usize = 64;
/// Offset of edge sample zero inside the edge arrays (AVM keeps 32 samples
/// in front for the corner and the 4-tap filter's reach).
const ORIGIN: usize = 32;
/// Edge array length (AVM `NUM_INTRA_NEIGHBOUR_PIXELS`).
const EDGE_LEN: usize = 2 * AV2_INTRA_MAX_SIDE + 64;
/// Fractional bits of directional positions.
const POSITION_BITS: u32 = 6;
/// Mask of the fractional part of a directional position.
const POSITION_FRACTION: i32 = (1 << POSITION_BITS) - 1;
/// Rounding bits of the 2-tap directional interpolation (weights sum to 32).
pub(crate) const TWO_TAP_BITS: u32 = 5;
/// Rounding bits of the 4-tap directional interpolation
/// (`POWER_DR_INTERP_FILTER`).
pub(crate) const FOUR_TAP_BITS: u32 = 7;
/// Largest smooth blend weight (`BLEND_WEIGHT_MAX`); the blend divides by twice it.
pub(crate) const BLEND_WEIGHT_MAX: i32 = 32;
/// log2 of twice [`BLEND_WEIGHT_MAX`]: the blend's rounding shift and the
/// cap on the weight's halving count.
pub(crate) const BLEND_SHIFT: u32 = 6;
/// Angle of pure vertical prediction (degrees).
const VERTICAL_ANGLE: u16 = 90;
/// Angle of pure horizontal prediction (degrees).
const HORIZONTAL_ANGLE: u16 = 180;
/// Directional angles lie strictly between zero and this.
const MAX_ANGLE: u16 = 270;

/// Positional step per unit distance by angle, 256 × cotangent-like
/// (`dr_intra_derivative`); only angles used by the nominal directions and
/// their deltas are read.
const DERIVATIVE: [i32; 90] = [
    0, 4096, 2048, 1365, 1024, 819, 682, 585, 512, 455, 409, 409, 409, 372, 341, 292, 273, 256,
    227, 215, 204, 186, 178, 170, 157, 151, 146, 136, 132, 128, 117, 110, 107, 99, 97, 97, 93, 87,
    83, 81, 77, 74, 73, 69, 66, 64, 62, 59, 56, 55, 53, 50, 49, 47, 44, 42, 42, 41, 38, 37, 35, 32,
    31, 30, 28, 27, 26, 24, 23, 22, 20, 19, 18, 16, 15, 14, 12, 11, 10, 10, 10, 9, 8, 7, 6, 5, 4,
    3, 2, 1,
];

/// 4-tap interpolation filters by 1/32 position (`av2_dr_interp_filter`).
pub(crate) const FOUR_TAP: [[i32; 4]; 32] = [
    [0, 128, 0, 0],
    [-2, 127, 4, -1],
    [-3, 125, 8, -2],
    [-5, 123, 13, -3],
    [-6, 121, 17, -4],
    [-7, 118, 22, -5],
    [-9, 116, 27, -6],
    [-9, 112, 32, -7],
    [-10, 109, 37, -8],
    [-11, 106, 41, -8],
    [-11, 102, 46, -9],
    [-12, 98, 52, -10],
    [-12, 94, 56, -10],
    [-12, 90, 61, -11],
    [-12, 85, 66, -11],
    [-12, 81, 71, -12],
    [-12, 76, 76, -12],
    [-12, 71, 81, -12],
    [-11, 66, 85, -12],
    [-11, 61, 90, -12],
    [-10, 56, 94, -12],
    [-10, 52, 98, -12],
    [-9, 46, 102, -11],
    [-8, 41, 106, -11],
    [-8, 37, 109, -10],
    [-7, 32, 112, -9],
    [-6, 27, 116, -9],
    [-5, 22, 118, -7],
    [-4, 17, 121, -6],
    [-3, 13, 123, -5],
    [-2, 8, 125, -3],
    [-1, 4, 127, -2],
];

/// One AV2 intra predictor, after the codec has resolved the mode, angle
/// delta and edge availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av2Predictor {
    /// Mean of the available edges (`dc_pred_high[left][above]`).
    Dc {
        /// The above edge exists.
        above: bool,
        /// The left edge exists.
        left: bool,
    },
    /// AV2 smooth: both ramps blended.
    Smooth,
    /// AV2 smooth along columns only.
    SmoothVertical,
    /// AV2 smooth along rows only.
    SmoothHorizontal,
    /// Nearest of left, above and corner to their planar estimate.
    Paeth,
    /// Directional prediction at `angle` degrees (0 < angle < 270).
    Directional {
        /// Prediction angle in degrees, after the angle delta.
        angle: u16,
        /// Luma uses the 4-tap interpolation, chroma the 2-tap one.
        luma: bool,
    },
}

/// The edges of one square transform: the row above (from the corner at
/// index −1 to above-right) and the column to the left (corner, then down
/// to below-left), with AVM's defaults where the codec writes nothing.
#[derive(Debug, Clone)]
pub struct Av2IntraEdges {
    above: [u8; EDGE_LEN],
    left: [u8; EDGE_LEN],
}

impl Default for Av2IntraEdges {
    fn default() -> Self {
        Self::new()
    }
}

impl Av2IntraEdges {
    /// Edges holding AVM's initial values (`base − 1` above, `base + 1` left).
    #[must_use]
    pub fn new() -> Self {
        Self {
            above: [DC_NO_NEIGHBOURS - 1; EDGE_LEN],
            left: [DC_NO_NEIGHBOURS + 1; EDGE_LEN],
        }
    }

    /// The above row from its first sample (above-right follows the block).
    pub fn above_mut(&mut self) -> &mut [u8] {
        &mut self.above[ORIGIN..]
    }

    /// The left column from its first sample (below-left follows the block).
    pub fn left_mut(&mut self) -> &mut [u8] {
        &mut self.left[ORIGIN..]
    }

    /// Set the shared above-left corner sample (index −1 of both edges).
    pub fn set_corner(&mut self, value: u8) {
        self.above[ORIGIN - 1] = value;
        self.left[ORIGIN - 1] = value;
    }

    #[cfg(test)]
    fn above_at(&self, index: i32) -> i32 {
        i32::from(self.above[(ORIGIN as i32 + index) as usize])
    }

    #[cfg(test)]
    fn left_at(&self, index: i32) -> i32 {
        i32::from(self.left[(ORIGIN as i32 + index) as usize])
    }

    /// Predict a `side`×`side` raster `dst` (`side` 4..64). Directional
    /// luma first pads the edge ends it reads past, as AVM does in place.
    ///
    /// # Panics
    ///
    /// If `dst` is not `side * side` samples, `side` is not 4..64, or a
    /// directional angle is outside (0, 270).
    pub fn predict(&mut self, predictor: Av2Predictor, side: usize, dst: &mut [u8]) {
        self.predict_with(&ScalarKernels, predictor, side, dst);
    }

    /// [`Self::predict`] with the run and smooth kernels of `kernels`
    /// (`Dsp::av2_intra_predict` passes its ISA's table).
    pub(crate) fn predict_with(
        &mut self,
        kernels: &impl Av2IntraKernels,
        predictor: Av2Predictor,
        side: usize,
        dst: &mut [u8],
    ) {
        assert!(side.is_power_of_two() && (4..=AV2_INTRA_MAX_SIDE).contains(&side));
        assert_eq!(dst.len(), side * side);
        let smooth = |edges: &Self, kind: SmoothKind, dst: &mut [u8]| {
            kernels.smooth(&edges.smooth_edges(side, kind), dst);
        };
        match predictor {
            Av2Predictor::Dc { above, left } => dst.fill(self.dc(side, above, left)),
            Av2Predictor::Smooth => smooth(self, SmoothKind::Both, dst),
            Av2Predictor::SmoothVertical => smooth(self, SmoothKind::Vertical, dst),
            Av2Predictor::SmoothHorizontal => smooth(self, SmoothKind::Horizontal, dst),
            Av2Predictor::Paeth => self.paeth(side, dst),
            Av2Predictor::Directional { angle, luma } => {
                self.directional(kernels, angle, luma, side, dst);
            }
        }
    }

    /// The edges the smooth predictors read: `side + 1` samples each, the
    /// last being the above-right (`right`) and below-left (`below`) sample.
    fn smooth_edges(&self, side: usize, kind: SmoothKind) -> SmoothEdges<'_> {
        SmoothEdges {
            above: &self.above[ORIGIN..=ORIGIN + side],
            left: &self.left[ORIGIN..=ORIGIN + side],
            kind,
        }
    }

    fn dc(&self, side: usize, above: bool, left: bool) -> u8 {
        let n = side as u32;
        let sum = |edge: &[u8]| -> u32 {
            edge[ORIGIN..ORIGIN + side]
                .iter()
                .map(|&v| u32::from(v))
                .sum()
        };
        let value = match (above, left) {
            (true, true) => (sum(&self.above) + sum(&self.left) + n) / (2 * n),
            (true, false) => (sum(&self.above) + n / 2) / n,
            (false, true) => (sum(&self.left) + n / 2) / n,
            (false, false) => u32::from(DC_NO_NEIGHBOURS),
        };
        value as u8
    }

    fn paeth(&self, side: usize, dst: &mut [u8]) {
        let corner = self.above[ORIGIN - 1];
        let above = &self.above[ORIGIN..ORIGIN + side];
        for (row, &left) in dst.chunks_exact_mut(side).zip(&self.left[ORIGIN..]) {
            for (pixel, &top) in row.iter_mut().zip(above) {
                *pixel = paeth(left, top, corner);
            }
        }
    }

    fn directional(
        &mut self,
        kernels: &impl Av2IntraKernels,
        angle: u16,
        luma: bool,
        side: usize,
        dst: &mut [u8],
    ) {
        assert!(angle > 0 && angle < MAX_ANGLE);
        match angle {
            VERTICAL_ANGLE => {
                for row in dst.chunks_exact_mut(side) {
                    row.copy_from_slice(&self.above[ORIGIN..ORIGIN + side]);
                }
            }
            HORIZONTAL_ANGLE => {
                for (row, &left) in dst.chunks_exact_mut(side).zip(&self.left[ORIGIN..]) {
                    row.fill(left);
                }
            }
            _ if angle < VERTICAL_ANGLE => {
                let dx = DERIVATIVE[usize::from(angle)];
                if luma {
                    let last = ORIGIN + 2 * side - 1;
                    self.above[last + 1] = self.above[last];
                    self.above[last + 2] = self.above[last];
                }
                self.zone1(kernels, dx, luma, side, dst);
            }
            _ if angle < HORIZONTAL_ANGLE => {
                let dx = DERIVATIVE[usize::from(HORIZONTAL_ANGLE - angle)];
                let dy = DERIVATIVE[usize::from(angle - VERTICAL_ANGLE)];
                if luma {
                    self.above[ORIGIN - 2] = self.above[ORIGIN - 1];
                    self.left[ORIGIN - 2] = self.left[ORIGIN - 1];
                    self.above[ORIGIN + side] = self.above[ORIGIN + side - 1];
                    self.left[ORIGIN + side] = self.left[ORIGIN + side - 1];
                }
                self.zone2(kernels, dx, dy, luma, side, dst);
            }
            _ => {
                let dy = DERIVATIVE[usize::from(MAX_ANGLE - angle)];
                if luma {
                    let last = ORIGIN + 2 * side - 1;
                    self.left[last + 1] = self.left[last];
                    self.left[last + 2] = self.left[last];
                }
                self.zone3(kernels, dy, luma, side, dst);
            }
        }
    }

    /// Zone 1 (0 < angle < 90): from the above row only.
    fn zone1(
        &self,
        kernels: &impl Av2IntraKernels,
        dx: i32,
        luma: bool,
        side: usize,
        dst: &mut [u8],
    ) {
        let max_base = 2 * side as i32 - 1;
        let last = self.above[ORIGIN + max_base as usize];
        let mut x = dx;
        for row in dst.chunks_exact_mut(side) {
            let base = x >> POSITION_BITS;
            let shift = (x & POSITION_FRACTION) >> 1;
            // The 4-tap filter reads one position further than the 2-tap.
            let valid = (max_base - base + i32::from(luma)).clamp(0, side as i32) as usize;
            interpolate_run(
                kernels,
                &self.above,
                ORIGIN + base as usize,
                shift,
                luma,
                &mut row[..valid],
            );
            row[valid..].fill(last);
            x += dx;
        }
    }

    /// Zone 2 (90 < angle < 180): above row where the ray lands on it,
    /// otherwise the left column. Along a row the above positions advance
    /// one sample per column at a fixed fraction, so they form one run.
    #[allow(clippy::too_many_arguments)] // the ray steps of both edges
    fn zone2(
        &self,
        kernels: &impl Av2IntraKernels,
        dx: i32,
        dy: i32,
        luma: bool,
        side: usize,
        dst: &mut [u8],
    ) {
        const MIN_BASE: i32 = -1;
        // Row r takes the above row from column `first_above[r]` on; that
        // column only moves right going down, so each column's left part
        // is the rows from some r0 down, one run of the left column at the
        // column's fixed fraction.
        let mut first_above = [0; AV2_INTRA_MAX_SIDE];
        for (r, first) in first_above[..side].iter_mut().enumerate() {
            let base = (-(r as i32 + 1) * dx) >> POSITION_BITS;
            *first = (MIN_BASE - base).clamp(0, side as i32) as usize;
        }
        let mut column = [0; AV2_INTRA_MAX_SIDE];
        let mut top = 0;
        for c in 0..side {
            while top < side && first_above[top] <= c {
                top += 1;
            }
            if top == side {
                break;
            }
            let y = -(c as i32 + 1) * dy;
            let base = (top as i32) + (y >> POSITION_BITS);
            debug_assert!(base >= MIN_BASE);
            let shift = (y & POSITION_FRACTION) >> 1;
            let run = &mut column[..side - top];
            let start = (ORIGIN as i32 + base) as usize;
            interpolate_run(kernels, &self.left, start, shift, luma, run);
            for (row, &value) in dst.chunks_exact_mut(side).skip(top).zip(&*run) {
                row[c] = value;
            }
        }
        for (r, row) in dst.chunks_exact_mut(side).enumerate() {
            let first = first_above[r];
            if first < side {
                let x = -(r as i32 + 1) * dx;
                let start = (ORIGIN as i32 + (x >> POSITION_BITS) + first as i32) as usize;
                let shift = (x & POSITION_FRACTION) >> 1;
                interpolate_run(kernels, &self.above, start, shift, luma, &mut row[first..]);
            }
        }
    }

    /// Zone 3 (180 < angle < 270): from the left column only, one column
    /// at a time.
    fn zone3(
        &self,
        kernels: &impl Av2IntraKernels,
        dy: i32,
        luma: bool,
        side: usize,
        dst: &mut [u8],
    ) {
        let max_base = 2 * side as i32 - 1;
        let last = self.left[ORIGIN + max_base as usize];
        let mut column = [0; AV2_INTRA_MAX_SIDE];
        let mut y = dy;
        for c in 0..side {
            let base = y >> POSITION_BITS;
            let shift = (y & POSITION_FRACTION) >> 1;
            let valid = (max_base - base + i32::from(luma)).clamp(0, side as i32) as usize;
            interpolate_run(
                kernels,
                &self.left,
                ORIGIN + base as usize,
                shift,
                luma,
                &mut column[..valid],
            );
            column[valid..side].fill(last);
            for (row, &value) in dst.chunks_exact_mut(side).zip(&column[..side]) {
                row[c] = value;
            }
            y += dy;
        }
    }
}

/// Interpolate `out.len()` consecutive positions of `edge` starting at
/// index `start`, all at `shift`/32 past their sample: the 4-tap filter
/// over indices −1..=2 (luma), or the 2-tap over 0..=1 (chroma).
fn interpolate_run(
    kernels: &impl Av2IntraKernels,
    edge: &[u8],
    start: usize,
    shift: i32,
    luma: bool,
    out: &mut [u8],
) {
    if out.is_empty() {
        return;
    }
    let shift = shift as usize;
    if luma {
        kernels.run4(&edge[start - 1..start + out.len() + 2], shift, out);
    } else {
        kernels.run2(&edge[start..=start + out.len()], shift, out);
    }
}

/// Which ramps a smooth predictor blends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmoothKind {
    /// Both, averaged (`SMOOTH_PRED`).
    Both,
    /// Down the columns (`SMOOTH_V_PRED`).
    Vertical,
    /// Along the rows (`SMOOTH_H_PRED`).
    Horizontal,
}

/// The edges of one smooth prediction: `above` and `left` hold `side + 1`
/// samples, the last being the above-right and below-left sample.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SmoothEdges<'a> {
    pub above: &'a [u8],
    pub left: &'a [u8],
    pub kind: SmoothKind,
}

impl SmoothEdges<'_> {
    /// The block side.
    pub(crate) fn side(&self) -> usize {
        self.above.len() - 1
    }
}

/// The kernels [`Av2IntraEdges::predict_with`] calls: the scalar reference
/// ([`ScalarKernels`]) or an ISA's table (`av2_intra::Av2IntraFns`).
pub(crate) trait Av2IntraKernels {
    /// 4-tap run: `window` is `out.len() + 3` samples, from one before the
    /// first position; `shift` is the fraction in 1/32 (below 32).
    fn run4(&self, window: &[u8], shift: usize, out: &mut [u8]);
    /// 2-tap run: `window` is `out.len() + 1` samples from the first position.
    fn run2(&self, window: &[u8], shift: usize, out: &mut [u8]);
    /// Smooth prediction of the `side`×`side` raster `dst`.
    fn smooth(&self, edges: &SmoothEdges<'_>, dst: &mut [u8]);
}

/// The scalar reference kernels.
pub(crate) struct ScalarKernels;

impl Av2IntraKernels for ScalarKernels {
    fn run4(&self, window: &[u8], shift: usize, out: &mut [u8]) {
        run4(window, shift, out);
    }

    fn run2(&self, window: &[u8], shift: usize, out: &mut [u8]) {
        run2(window, shift, out);
    }

    fn smooth(&self, edges: &SmoothEdges<'_>, dst: &mut [u8]) {
        smooth(edges, dst);
    }
}

/// Scalar 4-tap run (see [`Av2IntraKernels::run4`]): each output is the
/// filter at `shift` over four consecutive window samples, clipped to 8 bits.
pub(crate) fn run4(window: &[u8], shift: usize, out: &mut [u8]) {
    let taps = &FOUR_TAP[shift];
    for (pixel, samples) in out.iter_mut().zip(window.windows(4)) {
        let value = taps[0] * i32::from(samples[0])
            + taps[1] * i32::from(samples[1])
            + taps[2] * i32::from(samples[2])
            + taps[3] * i32::from(samples[3]);
        *pixel = round_shift(value, FOUR_TAP_BITS).clamp(0, i32::from(u8::MAX)) as u8;
    }
}

/// Scalar 2-tap run (see [`Av2IntraKernels::run2`]).
pub(crate) fn run2(window: &[u8], shift: usize, out: &mut [u8]) {
    let shift = shift as i32;
    for (pixel, pair) in out.iter_mut().zip(window.windows(2)) {
        let value = i32::from(pair[0]) * (32 - shift) + i32::from(pair[1]) * shift;
        *pixel = round_shift(value, TWO_TAP_BITS) as u8;
    }
}

/// The smooth blend weight at `distance` from the edge, for a block of
/// side `2^log2`: it halves every `2^scale / 2` samples, slower for larger
/// blocks.
pub(crate) fn smooth_weight(distance: usize, log2: u32) -> i32 {
    let scale = (2 * (log2 - 2) + 2) >> 2;
    BLEND_WEIGHT_MAX >> ((distance << 1) >> scale).min(BLEND_SHIFT as usize)
}

/// Scalar AV2 smooth (`highbd_smooth*_predictor`): each ramp runs from the
/// far corner sample to the edge sample, then blends towards the edge sample
/// by a weight that halves with distance; `Both` averages the two.
pub(crate) fn smooth(edges: &SmoothEdges<'_>, dst: &mut [u8]) {
    let side = edges.side();
    let n = side as i32;
    let log2 = side.trailing_zeros();
    let below = i32::from(edges.left[side]);
    let right = i32::from(edges.above[side]);
    let mut weights = [0; AV2_INTRA_MAX_SIDE];
    for (distance, weight) in weights[..side].iter_mut().enumerate() {
        *weight = smooth_weight(distance, log2);
    }
    let weights = &weights[..side];
    // A ramp from `end` (distance n − 1) to the edge sample `edge`, blended
    // towards `edge` by the weight of `distance`.
    let blend = |edge: i32, end: i32, distance: usize| {
        let ramp = end + round_shift((edge - end) * (n - 1 - distance as i32), log2);
        ramp + round_shift((edge - ramp) * weights[distance], BLEND_SHIFT)
    };
    for (r, (row, &left)) in dst.chunks_exact_mut(side).zip(edges.left).enumerate() {
        let left = i32::from(left);
        for (c, (pixel, &top)) in row.iter_mut().zip(edges.above).enumerate() {
            let top = i32::from(top);
            *pixel = match edges.kind {
                SmoothKind::Both => round_shift(blend(top, below, r) + blend(left, right, c), 1),
                SmoothKind::Vertical => blend(top, below, r),
                SmoothKind::Horizontal => blend(left, right, c),
            } as u8;
        }
    }
}

/// `ROUND_POWER_OF_TWO` on signed values (arithmetic shift, as AVM's C).
const fn round_shift(value: i32, bits: u32) -> i32 {
    (value + (1 << (bits - 1))) >> bits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first, straightforward implementation, kept as the oracle the
    /// restructured kernels must match exactly.
    mod reference {
        use super::super::*;

        pub(super) struct Reference<'a>(pub &'a Av2IntraEdges);

        impl Reference<'_> {
            fn above_at(&self, index: i32) -> i32 {
                self.0.above_at(index)
            }

            fn left_at(&self, index: i32) -> i32 {
                self.0.left_at(index)
            }

            /// AV2 smooth (`vertical` and/or `horizontal` ramps; both average).
            pub(super) fn smooth(
                &self,
                side: usize,
                dst: &mut [u8],
                vertical: bool,
                horizontal: bool,
            ) {
                let n = side as i32;
                let log2 = side.trailing_zeros();
                let below = self.left_at(n);
                let right = self.above_at(n);
                // The weight halves every `2^scale / 2` samples: slower for larger blocks.
                let scale = (2 * (log2 - 2) + 2) >> 2;
                let weight = |distance: i32| {
                    BLEND_WEIGHT_MAX >> ((distance << 1) >> scale).min(BLEND_SHIFT as i32)
                };
                for (r, row) in dst.chunks_exact_mut(side).enumerate() {
                    let r = r as i32;
                    let left = self.left_at(r);
                    for (c, pixel) in row.iter_mut().enumerate() {
                        let c = c as i32;
                        let top = self.above_at(c);
                        let column = || {
                            let ramp = below + round_shift((top - below) * (n - 1 - r), log2);
                            ramp + round_shift((top - ramp) * weight(r), BLEND_SHIFT)
                        };
                        let line = || {
                            let ramp = right + round_shift((left - right) * (n - 1 - c), log2);
                            ramp + round_shift((left - ramp) * weight(c), BLEND_SHIFT)
                        };
                        *pixel = match (vertical, horizontal) {
                            (true, true) => round_shift(column() + line(), 1),
                            (true, false) => column(),
                            _ => line(),
                        } as u8;
                    }
                }
            }

            /// Zone 1 (0 < angle < 90): from the above row only.
            pub(super) fn zone1(&self, dx: i32, luma: bool, side: usize, dst: &mut [u8]) {
                let max_base = 2 * side as i32 - 1;
                let last = self.above_at(max_base) as u8;
                let mut x = dx;
                for (r, row) in dst.chunks_exact_mut(side).enumerate() {
                    let base = x >> POSITION_BITS;
                    let shift = (x & POSITION_FRACTION) >> 1;
                    // The 4-tap filter may read one position further than the 2-tap.
                    let beyond = if luma {
                        base > max_base
                    } else {
                        base >= max_base
                    };
                    if beyond {
                        dst[r * side..].fill(last);
                        return;
                    }
                    for (c, pixel) in row.iter_mut().enumerate() {
                        let position = base + c as i32;
                        *pixel = if luma && position <= max_base {
                            four_tap(|i| self.above_at(position + i), shift)
                        } else if !luma && position < max_base {
                            two_tap(self.above_at(position), self.above_at(position + 1), shift)
                        } else {
                            last
                        };
                    }
                    x += dx;
                }
            }

            /// Zone 2 (90 < angle < 180): above row where the ray lands on it,
            /// otherwise the left column.
            pub(super) fn zone2(&self, dx: i32, dy: i32, luma: bool, side: usize, dst: &mut [u8]) {
                const MIN_BASE: i32 = -1;
                for (r, row) in dst.chunks_exact_mut(side).enumerate() {
                    let r = r as i32;
                    for (c, pixel) in row.iter_mut().enumerate() {
                        let c = c as i32;
                        let x = (c << POSITION_BITS) - (r + 1) * dx;
                        let base_x = x >> POSITION_BITS;
                        *pixel = if base_x >= MIN_BASE {
                            let shift = (x & POSITION_FRACTION) >> 1;
                            interpolate(|i| self.above_at(base_x + i), shift, luma)
                        } else {
                            let y = (r << POSITION_BITS) - (c + 1) * dy;
                            let base_y = y >> POSITION_BITS;
                            debug_assert!(base_y >= MIN_BASE);
                            let shift = (y & POSITION_FRACTION) >> 1;
                            interpolate(|i| self.left_at(base_y + i), shift, luma)
                        };
                    }
                }
            }

            /// Zone 3 (180 < angle < 270): from the left column only.
            pub(super) fn zone3(&self, dy: i32, luma: bool, side: usize, dst: &mut [u8]) {
                let max_base = 2 * side as i32 - 1;
                let last = self.left_at(max_base) as u8;
                let mut y = dy;
                for c in 0..side {
                    let base = y >> POSITION_BITS;
                    let shift = (y & POSITION_FRACTION) >> 1;
                    for r in 0..side {
                        let position = base + r as i32;
                        dst[r * side + c] = if luma && position <= max_base {
                            four_tap(|i| self.left_at(position + i), shift)
                        } else if !luma && position < max_base {
                            two_tap(self.left_at(position), self.left_at(position + 1), shift)
                        } else {
                            last
                        };
                    }
                    y += dy;
                }
            }
        }

        /// Interpolate at `shift`/32 past sample 0 of `edge` (indices −1..=2).
        fn interpolate(edge: impl Fn(i32) -> i32, shift: i32, luma: bool) -> u8 {
            if luma {
                four_tap(edge, shift)
            } else {
                two_tap(edge(0), edge(1), shift)
            }
        }

        fn two_tap(a: i32, b: i32, shift: i32) -> u8 {
            round_shift(a * (32 - shift) + b * shift, TWO_TAP_BITS) as u8
        }

        /// The 4-tap filter at `shift`/32 over `edge(-1..=2)`, clipped to 8 bits.
        fn four_tap(edge: impl Fn(i32) -> i32, shift: i32) -> u8 {
            let taps = &FOUR_TAP[shift as usize];
            let value: i32 = (0..4).map(|i| taps[i as usize] * edge(i - 1)).sum();
            round_shift(value, FOUR_TAP_BITS).clamp(0, i32::from(u8::MAX)) as u8
        }
    }

    fn flat(value: u8) -> Av2IntraEdges {
        let mut edges = Av2IntraEdges::new();
        edges.above_mut().fill(value);
        edges.left_mut().fill(value);
        edges.set_corner(value);
        edges
    }

    #[test]
    fn flat_edges_predict_flat_for_every_predictor() {
        let mut predictors = vec![
            Av2Predictor::Dc {
                above: true,
                left: true,
            },
            Av2Predictor::Smooth,
            Av2Predictor::SmoothVertical,
            Av2Predictor::SmoothHorizontal,
            Av2Predictor::Paeth,
        ];
        for angle in (36..=212).step_by(3).filter(|a| a % 90 != 0) {
            for luma in [false, true] {
                predictors.push(Av2Predictor::Directional { angle, luma });
            }
        }
        for side in [4, 8, 16, 32, 64] {
            for &predictor in &predictors {
                let mut dst = vec![0; side * side];
                flat(91).predict(predictor, side, &mut dst);
                assert!(dst.iter().all(|&v| v == 91), "{predictor:?} {side}");
            }
        }
    }

    #[test]
    fn restructured_kernels_match_the_reference() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        };
        for _ in 0..8 {
            let mut edges = Av2IntraEdges::new();
            for i in 0..EDGE_LEN {
                edges.above[i] = next();
                edges.left[i] = next();
            }
            for side in [4, 8, 16, 32, 64] {
                for (vertical, horizontal, predictor) in [
                    (true, true, Av2Predictor::Smooth),
                    (true, false, Av2Predictor::SmoothVertical),
                    (false, true, Av2Predictor::SmoothHorizontal),
                ] {
                    let mut fast = vec![0; side * side];
                    let mut slow = vec![0; side * side];
                    edges.clone().predict(predictor, side, &mut fast);
                    reference::Reference(&edges).smooth(side, &mut slow, vertical, horizontal);
                    assert_eq!(fast, slow, "{predictor:?} {side}");
                }
                for angle in (36..=212).step_by(3).filter(|a| a % 90 != 0) {
                    for luma in [false, true] {
                        let mut fast = vec![0; side * side];
                        let mut slow = vec![0; side * side];
                        // `predict` pads the edges in place; the reference
                        // reads the padded edges.
                        let mut padded = edges.clone();
                        padded.predict(Av2Predictor::Directional { angle, luma }, side, &mut fast);
                        let oracle = reference::Reference(&padded);
                        if angle < VERTICAL_ANGLE {
                            oracle.zone1(DERIVATIVE[usize::from(angle)], luma, side, &mut slow);
                        } else if angle < HORIZONTAL_ANGLE {
                            let dx = DERIVATIVE[usize::from(HORIZONTAL_ANGLE - angle)];
                            let dy = DERIVATIVE[usize::from(angle - VERTICAL_ANGLE)];
                            oracle.zone2(dx, dy, luma, side, &mut slow);
                        } else {
                            let dy = DERIVATIVE[usize::from(MAX_ANGLE - angle)];
                            oracle.zone3(dy, luma, side, &mut slow);
                        }
                        assert_eq!(fast, slow, "angle {angle} luma {luma} side {side}");
                    }
                }
            }
        }
    }

    #[test]
    fn smooth_corners_follow_the_edges() {
        let mut edges = flat(0);
        edges.above_mut()[..9].copy_from_slice(&[100, 100, 100, 100, 100, 100, 100, 100, 20]);
        edges.left_mut()[..9].copy_from_slice(&[60, 60, 60, 60, 60, 60, 60, 60, 200]);
        let mut dst = [0; 64];
        edges.predict(Av2Predictor::SmoothVertical, 8, &mut dst);
        // Row 0 is half the above sample, half the ramp (113); the last row
        // is the below-left sample with no blend left.
        assert_eq!(dst[0], 107);
        assert_eq!(dst[56], 200);
    }
}
