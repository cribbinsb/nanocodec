//! A cheap directional estimate of what intra prediction could achieve on
//! an inter-frame block (`decision.intra_directional`).
//!
//! The inter-vs-intra decision (`decision::intra_or_inter`) used to price
//! intra as DC prediction alone: the source's AC energy. On texture with a
//! direction (rippling water, grass, foliage) intra with per-block
//! directions predicts far better than DC, so those blocks were never sent
//! to the codec's intra search; x264 codes ~70% of Riverbed's P macroblocks
//! `Intra_NxN` where H.264 here coded under half intra, mostly `Intra_16x16`.
//!
//! The estimate follows x264's lookahead intra cost (`x264_slicetype_mb_cost`,
//! an idea, no code): predict each sub-block from the *source* samples
//! above and to the left (open loop: no reconstruction exists yet) in a
//! handful of directions, and keep the best. Sub-blocks are the backend's
//! intra direction granularity ([`crate::CodecBackend::intra_estimate_log2`]):
//! 8x8 for H.264's `Intra_8x8`, the leaf itself where a block has one
//! direction. Distortion is the residual SSE, the same units as the inter
//! side's ranked distortion.
//!
//! Every direction tried copies its edge samples along whole rows, so each
//! row of a prediction is a slice of one edge array and the residual is
//! measured without building the prediction.

use nc_common::{Plane, PlaneView};

/// Largest sub-block side the estimate predicts.
const MAX_SIDE: usize = 64;
/// The edge every direction reads, for a side `n`: the left column bottom
/// to top, the top-left corner, then the row above and above-right
/// (`4n + 1` samples at most).
const EDGE_LEN: usize = 4 * MAX_SIDE + 1;

/// The estimate for one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntraEstimate {
    /// The summed residual SSE of each sub-block's best direction.
    pub distortion: u64,
    /// The summed per-sub-block AC-energy gap between source and that
    /// prediction (the psy term's input, [`crate::psy`]).
    pub energy_gap: u64,
}

/// The source samples around one `N`-square sub-block: `edge[N - 1 - r]` is
/// the left neighbour of row `r`, `edge[N]` the top-left corner and
/// `edge[N + 1 + c]` the sample above column `c` (`c < 2N`, the picture's
/// last column repeated past its edge).
struct Edges<const N: usize> {
    edge: [u8; EDGE_LEN],
}

impl<const N: usize> Edges<N> {
    /// Edges of the sub-block at `(x, y)` of `view`, which has both an
    /// above row and a left column (`x, y >= 1`).
    fn new(view: &PlaneView<'_>, (x, y): (usize, usize)) -> Self {
        let last = view.width() - 1;
        let above = view.row(y - 1);
        let mut edge = [0; EDGE_LEN];
        for (r, v) in edge[..N].iter_mut().rev().enumerate() {
            *v = view.row(y + r)[x - 1];
        }
        edge[N] = above[x - 1];
        for (c, v) in edge[(N + 1)..=(3 * N)].iter_mut().enumerate() {
            *v = above[(x + c).min(last)];
        }
        Self { edge }
    }

    /// The `N` edge samples from `start`.
    fn run(&self, start: usize) -> &[u8; N] {
        self.edge[start..start + N]
            .try_into()
            .expect("edge runs have N samples")
    }

    /// The row the vertical prediction repeats.
    fn above(&self) -> &[u8; N] {
        self.run(N + 1)
    }

    /// Row `r` of the prediction in `direction`.
    fn row(&self, direction: Direction, r: usize) -> [u8; N] {
        match direction {
            Direction::Vertical => *self.above(),
            Direction::Horizontal => [self.left(r); N],
            Direction::DownLeft => *self.run(N + 2 + r),
            Direction::DownRight => *self.run(N - r),
        }
    }

    /// The left neighbour of row `r` (the horizontal prediction).
    fn left(&self, r: usize) -> u8 {
        self.edge[N - 1 - r]
    }

    /// The DC prediction value: the rounded mean of the above row and left
    /// column.
    fn dc(&self) -> u32 {
        let sum: u32 = self.edge[..N]
            .iter()
            .chain(self.above())
            .map(|&v| u32::from(v))
            .sum();
        (sum + N as u32) / (2 * N as u32)
    }
}

/// The directions predicted per sub-block besides DC, whose residual
/// follows from the source's moments alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Vertical,
    Horizontal,
    /// 45 degrees from the above-right.
    DownLeft,
    /// 45 degrees from the top-left corner.
    DownRight,
}

/// Samples measured per sub-block: every other row and column, as x264's
/// lookahead measures intra at half resolution; sums are scaled back to
/// the whole sub-block. Every other row alone cost about twice as many
/// instructions (H.264 crowd 720p, `intra_direction_bits` 12: the estimate
/// was 2.4% of all instructions).
const SAMPLE_STEP: usize = 2;

/// Sum and sum of squares of samples.
#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    sum: u32,
    squares: u32,
}

impl Moments {
    fn add(&mut self, v: u8) {
        let v = u32::from(v);
        self.sum += v;
        self.squares += v * v;
    }

    /// `n · variance` over `count` samples: the energy about the mean.
    fn energy(self, count: u32) -> u32 {
        let sum = u64::from(self.sum);
        self.squares - (sum * sum / u64::from(count)) as u32
    }

    /// Σ(v - p)² over the samples, for a constant prediction `p`.
    fn sse_to(self, p: u32, count: u32) -> u32 {
        (self.squares + count * p * p).saturating_sub(2 * p * self.sum)
    }
}

/// Squared difference.
fn square_error(s: u8, p: u8) -> u32 {
    let d = i32::from(s) - i32::from(p);
    (d * d) as u32
}

/// The best direction's residual SSE and AC-energy gap of the `N`-square
/// sub-block at `(x, y)` of `view`, measured on every [`SAMPLE_STEP`]th
/// row and column. DC and horizontal follow from the sampled rows'
/// moments; the others are measured sample by sample.
fn sub_block<const N: usize>(view: &PlaneView<'_>, (x, y): (usize, usize)) -> (u64, u64) {
    let edges = Edges::<N>::new(view, (x, y));
    let per_row = N.div_ceil(SAMPLE_STEP) as u32;
    let mut source = Moments::default();
    let mut horizontal = 0;
    // Vertical, down-left, down-right.
    let mut sse = [0u32; 3];
    for r in (0..N).step_by(SAMPLE_STEP) {
        let row: &[u8; N] = view.row(y + r)[x..x + N]
            .try_into()
            .expect("sub-blocks lie inside the coded picture");
        let mut moments = Moments::default();
        for c in (0..N).step_by(SAMPLE_STEP) {
            let v = row[c];
            moments.add(v);
            sse[0] += square_error(v, edges.edge[N + 1 + c]);
            sse[1] += square_error(v, edges.edge[N + 2 + r + c]);
            sse[2] += square_error(v, edges.edge[N - r + c]);
        }
        horizontal += moments.sse_to(u32::from(edges.left(r)), per_row);
        source.sum += moments.sum;
        source.squares += moments.squares;
    }
    let count = per_row * per_row;
    let dc_sse = source.sse_to(edges.dc(), count);
    let (best_sse, best) = [
        (sse[0], Direction::Vertical),
        (horizontal, Direction::Horizontal),
        (sse[1], Direction::DownLeft),
        (sse[2], Direction::DownRight),
    ]
    .into_iter()
    .min_by_key(|&(sse, _)| sse)
    .expect("four directions");
    let scale = (SAMPLE_STEP * SAMPLE_STEP) as u64;
    let scaled = |v: u32| u64::from(v) * scale;
    if dc_sse <= best_sse {
        // A flat prediction has no AC energy.
        return (scaled(dc_sse), scaled(source.energy(count)));
    }
    let mut prediction = Moments::default();
    for r in (0..N).step_by(SAMPLE_STEP) {
        let row = edges.row(best, r);
        for c in (0..N).step_by(SAMPLE_STEP) {
            prediction.add(row[c]);
        }
    }
    let gap = source.energy(count).abs_diff(prediction.energy(count));
    (scaled(best_sse), scaled(gap))
}

/// The directional intra estimate of the `side`-square block at `(x, y)` of
/// source luma `plane`, in sub-blocks of `1 << sub_log2`. `None` on the
/// picture's top row or left column, which have no source neighbours on
/// one side (the DC estimate stands there).
pub(crate) fn estimate(
    plane: &Plane,
    (x, y): (usize, usize),
    side: usize,
    sub_log2: u8,
) -> Option<IntraEstimate> {
    if x == 0 || y == 0 {
        return None;
    }
    let n = (1usize << sub_log2).min(side).min(MAX_SIDE);
    let view = plane.view();
    let mut total = IntraEstimate {
        distortion: 0,
        energy_gap: 0,
    };
    for sy in (y..y + side).step_by(n) {
        for sx in (x..x + side).step_by(n) {
            let (sse, gap) = match n {
                4 => sub_block::<4>(&view, (sx, sy)),
                8 => sub_block::<8>(&view, (sx, sy)),
                16 => sub_block::<16>(&view, (sx, sy)),
                32 => sub_block::<32>(&view, (sx, sy)),
                _ => sub_block::<MAX_SIDE>(&view, (sx, sy)),
            };
            total.distortion += sse;
            total.energy_gap += gap;
        }
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_with(f: impl Fn(usize, usize) -> u8) -> Plane {
        let mut p = Plane::new(32, 32, 0).unwrap();
        let mut v = p.view_mut();
        for r in 0..32 {
            for (c, s) in v.row_mut(r).iter_mut().enumerate() {
                *s = f(r, c);
            }
        }
        p
    }

    fn distortion(plane: &Plane) -> u64 {
        estimate(plane, (8, 8), 16, 3).unwrap().distortion
    }

    #[test]
    fn each_direction_predicts_its_pattern_exactly() {
        let stripes = |v: usize| if v % 4 < 2 { 40 } else { 200 };
        // Vertical, horizontal, down-left (constant along anti-diagonals)
        // and down-right (constant along diagonals).
        for plane in [
            plane_with(|_, c| stripes(c)),
            plane_with(|r, _| stripes(r)),
            plane_with(|r, c| stripes(r + c)),
            plane_with(|r, c| stripes(r + 32 - c)),
        ] {
            let e = estimate(&plane, (8, 8), 16, 3).unwrap();
            assert_eq!((e.distortion, e.energy_gap), (0, 0));
        }
    }

    #[test]
    fn flat_blocks_cost_nothing_and_noise_costs_something() {
        assert_eq!(distortion(&plane_with(|_, _| 77)), 0);
        let noise = plane_with(|r, c| ((r * r * 7 + c * c * 3 + r * c) % 13 * 19) as u8);
        assert!(distortion(&noise) > 0);
    }

    #[test]
    fn edge_blocks_have_no_estimate() {
        let plane = plane_with(|_, _| 9);
        assert_eq!(estimate(&plane, (0, 8), 16, 3), None);
        assert_eq!(estimate(&plane, (8, 0), 16, 3), None);
        // The backend's unit may be the leaf itself.
        assert_eq!(estimate(&plane, (8, 8), 16, 6).unwrap().distortion, 0);
    }
}
