//! Intra prediction: luma `Intra_16x16` (H.264 §8.3.3) and 4:2:0 chroma
//! (§8.3.4), all four modes of each.
//!
//! Neighbours are reconstructed samples of the current picture. With a single
//! slice and no constrained intra prediction, a neighbour is available
//! exactly when it lies inside the (coded) picture.

use nc_common::PlaneView;

/// Value used when no neighbour is available: `1 << (BitDepth - 1)`.
const NO_NEIGHBOUR_DC: u8 = 128;

/// The reconstructed neighbours of the `N` x `N` block at `(x, y)`: the row
/// above, the column to the left and the above-left corner, each where the
/// picture has it. Gathered once per block and plane, so trying every mode
/// reads the column (one row access per sample) only once.
#[derive(Debug, Clone, Copy)]
pub struct Edges<const N: usize> {
    /// `p[i, -1]`, i = 0..N (zero where unavailable).
    above: [u8; N],
    /// `p[-1, j]`, j = 0..N (zero where unavailable).
    left: [u8; N],
    /// `p[-1, -1]` (zero where unavailable).
    corner: u8,
    avail: Availability,
}

impl<const N: usize> Edges<N> {
    /// The neighbours of the block at `(x, y)` of `recon`.
    pub fn gather(recon: &PlaneView<'_>, x: usize, y: usize) -> Self {
        let avail = Availability::at(x, y);
        let above = if avail.above {
            recon.row(y - 1)[x..x + N]
                .try_into()
                .expect("N samples above")
        } else {
            [0; N]
        };
        let left = if avail.left {
            core::array::from_fn(|j| recon.row(y + j)[x - 1])
        } else {
            [0; N]
        };
        let corner = if avail.above_left {
            recon.row(y - 1)[x - 1]
        } else {
            0
        };
        Self {
            above,
            left,
            corner,
            avail,
        }
    }

    /// Sum of the `len` samples above from `start`, if that row exists.
    fn sum_above(&self, start: usize, len: usize) -> Option<u32> {
        self.avail.above.then(|| {
            self.above[start..start + len]
                .iter()
                .map(|&s| u32::from(s))
                .sum()
        })
    }

    /// Sum of the `len` samples left from `start`, if that column exists.
    fn sum_left(&self, start: usize, len: usize) -> Option<u32> {
        self.avail.left.then(|| {
            self.left[start..start + len]
                .iter()
                .map(|&s| u32::from(s))
                .sum()
        })
    }
}

/// `Intra_16x16` DC predictor of the macroblock with neighbours `edges`
/// (§8.3.3.3).
fn luma_dc_16x16(edges: &Edges<16>) -> u8 {
    match (edges.sum_above(0, 16), edges.sum_left(0, 16)) {
        (Some(top), Some(left)) => ((top + left + 16) >> 5) as u8,
        (Some(one), None) | (None, Some(one)) => ((one + 8) >> 4) as u8,
        (None, None) => NO_NEIGHBOUR_DC,
    }
}

/// Chroma DC predictors of the four 4x4 blocks of the 8x8 chroma block with
/// neighbours `edges`, in raster order (§8.3.4.1–8.3.4.3). The rule depends
/// on the block's position: the top-right block prefers the samples above,
/// the bottom-left block the samples to the left, the other two average
/// both.
fn chroma_dc_8x8(edges: &Edges<8>) -> [u8; 4] {
    let mut out = [NO_NEIGHBOUR_DC; 4];
    for (blk, dc) in out.iter_mut().enumerate() {
        // Neighbours of a 4x4 chroma block are only those of the whole 8x8
        // block: the samples above row y and left of column x.
        let top = edges.sum_above(4 * (blk % 2), 4);
        let left = edges.sum_left(4 * (blk / 2), 4);
        let avg4 = |s: u32| ((s + 2) >> 2) as u8;
        *dc = match (blk, top, left) {
            (0 | 3, Some(t), Some(l)) => ((t + l + 4) >> 3) as u8,
            // Top-right block: above first, then left.
            (1, Some(t), _) | (0 | 2 | 3, Some(t), None) => avg4(t),
            // Bottom-left block: left first, then above.
            (2, _, Some(l)) | (0 | 1 | 3, None, Some(l)) => avg4(l),
            _ => NO_NEIGHBOUR_DC,
        };
    }
    out
}

/// `Intra16x16PredMode` (H.264 Table 8-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra16x16Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    Plane = 3,
}

/// `intra_chroma_pred_mode` (H.264 Table 7-16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaMode {
    Dc = 0,
    Horizontal = 1,
    Vertical = 2,
    Plane = 3,
}

/// Which neighbours of a block exist: above, left, above-left.
#[derive(Debug, Clone, Copy)]
pub struct Availability {
    pub above: bool,
    pub left: bool,
    pub above_left: bool,
}

impl Availability {
    /// Neighbours of the block at `(x, y)` (inside one slice: the picture).
    pub fn at(x: usize, y: usize) -> Self {
        Self {
            above: y > 0,
            left: x > 0,
            above_left: x > 0 && y > 0,
        }
    }
}

impl Intra16x16Mode {
    /// The modes usable with `avail` (§8.3.3: each mode needs the
    /// neighbours it reads; DC works with none).
    pub fn usable(avail: Availability) -> impl Iterator<Item = Self> {
        [Self::Dc, Self::Vertical, Self::Horizontal, Self::Plane]
            .into_iter()
            .filter(move |m| match m {
                Self::Dc => true,
                Self::Vertical => avail.above,
                Self::Horizontal => avail.left,
                Self::Plane => avail.above && avail.left && avail.above_left,
            })
    }
}

impl ChromaMode {
    /// The modes usable with `avail` (§8.3.4).
    pub fn usable(avail: Availability) -> impl Iterator<Item = Self> {
        [Self::Dc, Self::Horizontal, Self::Vertical, Self::Plane]
            .into_iter()
            .filter(move |m| match m {
                Self::Dc => true,
                Self::Vertical => avail.above,
                Self::Horizontal => avail.left,
                Self::Plane => avail.above && avail.left && avail.above_left,
            })
    }
}

/// Plane prediction of an `N` x `N` block (`N` = 16 luma, 8 chroma) from its
/// neighbours: §8.3.3.4 (luma) and §8.3.4.4 (chroma, 4:2:0 `xCF = yCF = 0`).
/// The gradients use the `N / 2` samples either side of the centre of the
/// top row and left column; `scale` is 5 for luma and 34 for 8-wide chroma.
fn plane<const N: usize, const AREA: usize>(edges: &Edges<N>) -> [u8; AREA] {
    const { assert!(AREA == N * N) };
    let half = N / 2;
    // p[i, -1] and p[-1, j] for i, j in -1..N: the corner, then the edge.
    let top = |i: usize| {
        i32::from(if i == 0 {
            edges.corner
        } else {
            edges.above[i - 1]
        })
    };
    let left = |j: usize| {
        i32::from(if j == 0 {
            edges.corner
        } else {
            edges.left[j - 1]
        })
    };
    // H' and V' of eqs. 8-117/8-118 (luma) and 8-140/8-141 (chroma), with
    // indices shifted by one so that -1 is 0.
    let (mut grad_h, mut grad_v) = (0i32, 0i32);
    for k in 0..half {
        let (far, near) = (half + k + 1, half - 1 - k);
        grad_h += (k as i32 + 1) * (top(far) - top(near));
        grad_v += (k as i32 + 1) * (left(far) - left(near));
    }
    let scale = if N == 16 { 5 } else { 34 };
    let base = 16 * (left(N) + top(N)); // a
    let slope_x = (scale * grad_h + 32) >> 6; // b
    let slope_y = (scale * grad_v + 32) >> 6; // c
    let centre = half as i32 - 1;
    // The formula per sample (eq. 8-119 / 8-142) with the row's terms
    // hoisted and the column term a fixed ramp: independent samples, so the
    // row loop vectorises.
    let ramp: [i32; N] = core::array::from_fn(|i| slope_x * i as i32);
    let mut out = [0u8; AREA];
    for (row, line) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let start = base + slope_y * (row as i32 - centre) - slope_x * centre + 16;
        for (sample, step) in line.iter_mut().zip(&ramp) {
            *sample = ((start + step) >> 5).clamp(0, 255) as u8;
        }
    }
    out
}

/// The row above (vertical) or the column left (horizontal) of an `N` x
/// `N` block repeated over it (`AREA` = `N · N`). Built row by row as whole
/// arrays (row copies and splats); a per-sample `from_fn` over the area was
/// a scalar loop (~9 instructions a sample).
fn directional<const N: usize, const AREA: usize>(edges: &Edges<N>, vertical: bool) -> [u8; AREA] {
    const { assert!(AREA == N * N) };
    let rows: [[u8; N]; N] = if vertical {
        [edges.above; N]
    } else {
        core::array::from_fn(|r| [edges.left[r]; N])
    };
    rows.as_flattened().try_into().expect("N rows of N samples")
}

/// `Intra_16x16` prediction of the macroblock with neighbours `edges` in
/// `mode`, which must be usable there.
pub fn luma_16x16(edges: &Edges<16>, mode: Intra16x16Mode) -> [u8; 256] {
    match mode {
        Intra16x16Mode::Dc => [luma_dc_16x16(edges); 256],
        Intra16x16Mode::Vertical => directional::<16, 256>(edges, true),
        Intra16x16Mode::Horizontal => directional::<16, 256>(edges, false),
        Intra16x16Mode::Plane => plane::<16, 256>(edges),
    }
}

/// Chroma prediction of the 8x8 block with neighbours `edges` in `mode`,
/// which must be usable there.
pub fn chroma_8x8(edges: &Edges<8>, mode: ChromaMode) -> [u8; 64] {
    match mode {
        ChromaMode::Dc => {
            let dc = chroma_dc_8x8(edges);
            core::array::from_fn(|i| dc[(i / 8 / 4) * 2 + (i % 8) / 4])
        }
        ChromaMode::Vertical => directional::<8, 64>(edges, true),
        ChromaMode::Horizontal => directional::<8, 64>(edges, false),
        ChromaMode::Plane => plane::<8, 64>(edges),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect()
    }

    #[test]
    fn luma_dc_cases() {
        let data = plane(32, 32, |x, y| {
            if y < 16 {
                100
            } else if x < 16 {
                50
            } else {
                0
            }
        });
        let view = PlaneView::new(&data, 32, 32, 32).unwrap();
        let dc = |x, y| luma_dc_16x16(&Edges::gather(&view, x, y));
        assert_eq!(dc(0, 0), 128);
        // Bottom-right MB: above = 100s, left = 50s.
        assert_eq!(dc(16, 16), 75);
        // Top-right MB: only the left (100s) is available.
        assert_eq!(dc(16, 0), 100);
    }

    #[test]
    fn chroma_dc_position_rules() {
        // Above row = 40, left column = 80.
        let data = plane(16, 16, |x, y| {
            if y == 7 {
                40
            } else if x == 7 {
                80
            } else {
                0
            }
        });
        let view = PlaneView::new(&data, 16, 16, 16).unwrap();
        let dc = |x, y| chroma_dc_8x8(&Edges::gather(&view, x, y));
        assert_eq!(dc(8, 8), [60, 40, 80, 60]);
        // Top-left of the picture: no neighbours at all.
        assert_eq!(dc(0, 0), [128; 4]);
    }

    #[test]
    fn plane_reproduces_a_linear_ramp() {
        // A ramp of slope 2 across and 1 down is predicted exactly by the
        // plane mode (its gradients are integral after the >> 6 scaling).
        let ramp = |x: usize, y: usize| (20 + 2 * x + y) as u8;
        let data = plane(48, 48, ramp);
        let view = PlaneView::new(&data, 48, 48, 48).unwrap();
        let pred = luma_16x16(&Edges::gather(&view, 16, 16), Intra16x16Mode::Plane);
        for j in 0..16 {
            for i in 0..16 {
                let want = i32::from(ramp(16 + i, 16 + j));
                assert!((i32::from(pred[j * 16 + i]) - want).abs() <= 1, "({i},{j})");
            }
        }
    }

    #[test]
    fn directional_modes_copy_neighbours() {
        let data = plane(32, 32, |x, y| (x * 3 + y * 7) as u8);
        let view = PlaneView::new(&data, 32, 32, 32).unwrap();
        let edges = Edges::gather(&view, 8, 8);
        let v = chroma_8x8(&edges, ChromaMode::Vertical);
        let h = chroma_8x8(&edges, ChromaMode::Horizontal);
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(v[j * 8 + i], data[7 * 32 + 8 + i]);
                assert_eq!(h[j * 8 + i], data[(8 + j) * 32 + 7]);
            }
        }
    }
}
