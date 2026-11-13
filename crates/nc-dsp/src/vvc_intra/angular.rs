//! H.266 square-block angular reference extension, parameters and scalar pixels.
//! Angle/inverse-angle and cubic tables follow BSD-licensed `VVenC`
//! `IntraPrediction`/`InterpolationFilter`; see `THIRD_PARTY.md`.

pub(super) const MAX_SIDE: usize = 32;
/// Length of the extended main reference: a negative margin of one side,
/// the corner and two sides, plus the last interpolation taps.
pub(super) const MAIN_LEN: usize = 3 * MAX_SIDE + 3;
const MAIN_ORIGIN: usize = MAX_SIDE;
pub(super) const ANGLES: [i32; 17] = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32];
pub(super) const INVERSE: [i32; 17] = [
    0, 16384, 8192, 5461, 4096, 2731, 2048, 1638, 1365, 1170, 1024, 910, 819, 712, 630, 565, 512,
];
const FILTER_THRESHOLD: [u8; 4] = [24, 14, 2, 0];
pub(super) const CUBIC: [[i32; 4]; 32] = [
    [0, 64, 0, 0],
    [-1, 63, 2, 0],
    [-2, 62, 4, 0],
    [-2, 60, 7, -1],
    [-2, 58, 10, -2],
    [-3, 57, 12, -2],
    [-4, 56, 14, -2],
    [-4, 55, 15, -2],
    [-4, 54, 16, -2],
    [-5, 53, 18, -2],
    [-6, 52, 20, -2],
    [-6, 49, 24, -3],
    [-6, 46, 28, -4],
    [-5, 44, 29, -4],
    [-4, 42, 30, -4],
    [-4, 39, 33, -4],
    [-4, 36, 36, -4],
    [-4, 33, 39, -4],
    [-4, 30, 42, -4],
    [-4, 29, 44, -5],
    [-4, 28, 46, -6],
    [-3, 24, 49, -6],
    [-2, 20, 52, -6],
    [-2, 18, 53, -5],
    [-2, 16, 54, -4],
    [-2, 15, 55, -4],
    [-2, 14, 56, -4],
    [-2, 12, 57, -3],
    [-2, 10, 58, -2],
    [-1, 7, 60, -2],
    [0, 4, 62, -2],
    [0, 2, 63, -1],
];

/// Prepared rows include the corner at index zero and a negative main margin.
/// The codec has already substituted unavailable references and smoothed any
/// required integer-slope luma edges. Fractional filtering happens here.
pub(super) struct Angular {
    pub angle: i32,
    pub vertical: bool,
    pub pdpc_scale: Option<u32>,
    pub inverse: i32,
    gaussian: bool,
    main: [i32; MAIN_LEN],
    side: [i32; 2 * MAX_SIDE + 1],
}
impl Angular {
    pub fn new(
        n: usize,
        direction: u8,
        luma: bool,
        top: &[i32],
        left: &[i32],
        corner: i32,
    ) -> Self {
        let vertical = direction >= 34;
        let mode = if vertical {
            i32::from(direction) - 50
        } else {
            18 - i32::from(direction)
        };
        let magnitude = mode.unsigned_abs() as usize;
        let angle = ANGLES[magnitude] * mode.signum();
        let inverse = INVERSE[magnitude];
        let pdpc_scale = if angle > 0 {
            let scale = (n.ilog2() as i32 - ((3 * inverse - 2).ilog2() as i32 - 8)).min(2);
            (scale >= 0).then_some(scale as u32)
        } else {
            None
        };
        let diff = direction.abs_diff(18).min(direction.abs_diff(50));
        let gaussian = luma && diff > FILTER_THRESHOLD[n.ilog2() as usize - 2];
        let (main_edge, side_edge) = if vertical { (top, left) } else { (left, top) };
        let mut main = [main_edge[2 * n - 1]; MAIN_LEN];
        let mut side = [side_edge[2 * n - 1]; 2 * MAX_SIDE + 1];
        main[MAIN_ORIGIN] = corner;
        main[MAIN_ORIGIN + 1..MAIN_ORIGIN + 1 + 2 * n].copy_from_slice(&main_edge[..2 * n]);
        side[0] = corner;
        side[1..=2 * n].copy_from_slice(&side_edge[..2 * n]);
        if angle < 0 {
            for distance in 1..=n {
                let index = ((distance as i32 * inverse + 256) >> 9).min(n as i32) as usize;
                main[MAIN_ORIGIN - distance] = side[index];
            }
        }
        Self {
            angle,
            vertical,
            pdpc_scale,
            inverse,
            gaussian,
            main,
            side,
        }
    }
    pub fn row(&self, y: usize) -> &[i32] {
        &self.main[self.row_start(y)..]
    }
    /// Where row `y`'s interpolation window starts in [`Self::main`].
    pub fn row_start(&self, y: usize) -> usize {
        let offset = ((y as i32 + 1) * self.angle) >> 5;
        (MAIN_ORIGIN as i32 + offset) as usize
    }
    /// The extended main reference every row interpolates from.
    pub fn main(&self) -> &[i32] {
        &self.main
    }
    pub fn phase(&self, y: usize) -> i32 {
        ((y as i32 + 1) * self.angle) & 31
    }
    pub fn coefficients(&self, phase: i32) -> [i32; 4] {
        if self.gaussian {
            let half = phase >> 1;
            [16 - half, 32 - half, 16 + half, half]
        } else {
            CUBIC[phase as usize]
        }
    }
    pub fn side_value(&self, y: usize, x: usize) -> i32 {
        self.side_row(y)[self.side_offset(x)]
    }
    /// The side samples PDPC reads on row `y`, indexed by [`Self::side_offset`].
    pub fn side_row(&self, y: usize) -> &[i32] {
        &self.side[y + 1..]
    }
    /// Where column `x`'s PDPC sample lies along [`Self::side_row`]: the
    /// same on every row, so kernels compute it once per block.
    pub fn side_offset(&self, x: usize) -> usize {
        ((256 + (x as i32 + 1) * self.inverse) >> 9) as usize
    }
}

/// Scalar reference; returns whether `out` holds the transposed block
/// (horizontal directions with `transposed_ok`).
#[allow(clippy::too_many_arguments)]
pub(super) fn predict(
    n: usize,
    direction: u8,
    luma: bool,
    top: &[i32],
    left: &[i32],
    corner: i32,
    transposed_ok: bool,
    out: &mut [u8],
) -> bool {
    let angular = Angular::new(n, direction, luma, top, left, corner);
    for y in 0..n {
        let row = angular.row(y);
        let phase = angular.phase(y);
        let coeff = angular.coefficients(phase);
        for x in 0..n {
            let value = if angular.angle.abs() == 32 {
                row[x + 1]
            } else if luma {
                ((0..4).map(|tap| coeff[tap] * row[x + tap]).sum::<i32>() + 32) >> 6
            } else {
                ((32 - phase) * row[x + 1] + phase * row[x + 2] + 16) >> 5
            };
            let mut value = value.clamp(0, 255);
            if let Some(scale) = angular.pdpc_scale
                && x < (3 << scale).min(n)
            {
                let weight = 32 >> ((2 * x) >> scale);
                value += (weight * (angular.side_value(y, x) - value) + 32) >> 6;
            }
            let index = if angular.vertical || transposed_ok {
                y * n + x
            } else {
                x * n + y
            };
            out[index] = value as u8;
        }
    }
    !angular.vertical && transposed_ok
}
