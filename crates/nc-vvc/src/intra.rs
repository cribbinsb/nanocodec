//! H.266 §8.4 planar/DC/angular prediction, shared candidate ranking and MPM syntax.
//! Algorithms checked against `VVenC` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
use crate::{
    coeff::{Bins, BitCost},
    contexts::Contexts,
    motion::MotionGrid,
    recon, transform_tree,
};
use nc_common::{
    Picture, PlaneIndex, PlaneViewMut,
    decode::{DecodeError, Result},
};
use nc_dsp::Dsp;
use nc_encoder::Block;

pub(crate) const PLANAR: u8 = 0;
pub(crate) const DC: u8 = 1;
pub(crate) const HORIZONTAL: u8 = 18;
pub(crate) const VERTICAL: u8 = 50;
const CORE_MODES: [u8; 4] = [PLANAR, DC, HORIZONTAL, VERTICAL];

pub(crate) fn mpm(left: u8, above: u8) -> [u8; 6] {
    if left <= DC && above <= DC {
        return [PLANAR, DC, VERTICAL, HORIZONTAL, 46, 54];
    }
    let (lo, hi) = (left.min(above), left.max(above));
    let prev = |a: u8| ((u32::from(a) + 61) % 64 + 2) as u8;
    let next = |a: u8| ((u32::from(a) - 1) % 64 + 2) as u8;
    if lo <= DC || left == above {
        return [
            PLANAR,
            hi,
            prev(hi),
            next(hi),
            prev(prev(hi)),
            next(next(hi)),
        ];
    }
    let tail = match hi - lo {
        1 => [prev(lo), next(hi), prev(prev(lo))],
        62.. => [next(lo), prev(hi), next(next(lo))],
        2 => [next(lo), prev(lo), next(hi)],
        _ => [prev(lo), next(lo), prev(hi)],
    };
    [PLANAR, left, above, tail[0], tail[1], tail[2]]
}

/// §7.3.11.5 luma direction syntax; `isp`: the CU uses intra
/// sub-partitions, which selects `intra_luma_not_planar_flag`'s context
/// (§9.3.4.2.2: ctxInc = !`intra_subpartitions_mode_flag`).
pub(crate) fn mode(
    io: &mut impl Bins,
    c: &mut Contexts,
    candidates: [u8; 6],
    selected: u8,
    isp: bool,
) -> Result<u8> {
    let index = candidates.iter().position(|&m| m == selected);
    if io.bin(&mut c.mpm, index.is_some())? {
        let mut idx = 0;
        if io.bin(
            &mut c.planar[usize::from(!isp)],
            index.is_some_and(|i| i > 0),
        )? {
            idx = 1;
            while idx < 5 && io.bypass(u32::from(index.is_some_and(|i| i > idx)), 1)? != 0 {
                idx += 1;
            }
        }
        Ok(candidates[idx])
    } else {
        let mut sorted = candidates;
        sorted.sort_unstable();
        let rem = u32::from(selected) - sorted.iter().filter(|&&m| m < selected).count() as u32;
        // 61 remaining modes: truncated binary, with three five-bit codewords.
        let prefix = if rem < 3 { rem } else { u32::midpoint(rem, 3) };
        let prefix = io.bypass(prefix, 5)?;
        let rem = if prefix < 3 {
            prefix
        } else {
            (prefix << 1) + io.bypass((rem + 3) & 1, 1)? - 3
        };
        let mut decoded = rem as u8;
        for m in sorted {
            if decoded >= m {
                decoded += 1;
            }
        }
        Ok(decoded)
    }
}

/// Exact signalling estimate of [`mode`] without sub-partitions; the two
/// context bins occur at most once.
pub(crate) fn mode_bits(c: &Contexts, candidates: [u8; 6], selected: u8) -> f32 {
    if let Some(index) = candidates.iter().position(|&m| m == selected) {
        c.mpm.bits(true)
            + c.planar[1].bits(index != 0)
            + if index == 0 { 0. } else { index.min(4) as f32 }
    } else {
        let remaining =
            usize::from(selected) - candidates.iter().filter(|&&m| m < selected).count();
        c.mpm.bits(false) + if remaining < 3 { 5. } else { 6. }
    }
}

/// The largest intra block side: a 64x64 CU of 64-point transforms
/// (`transform_tree`) predicts as one block.
const MAX_SIDE: usize = 64;
/// The longest reference edge is twice the largest block.
const MAX_EDGE: usize = 2 * MAX_SIDE;
/// The side from which the dispatched square predictor gives way to the
/// general one (`Dsp::vvc_intra_predict_rect`); directions are searched on
/// blocks up to this side ([`choose`]).
const SQUARE_KERNEL_MAX_SIDE: usize = 32;
/// The edge length of [`References`]: blocks up to 32x32. Edges sized for
/// 64x64 blocks everywhere, copied with every reference set, cost 0.5% of a
/// crowd encode's instructions without 64-point transforms (x86 callgrind).
const SMALL_EDGE: usize = 2 * SQUARE_KERNEL_MAX_SIDE;
/// Samples of the largest searched block: the direction search's buffers
/// (zeroing 64x64 ones for every candidate cost 3.4% of a crowd encode's
/// instructions, x86 callgrind).
const SEARCH_SAMPLES: usize = SQUARE_KERNEL_MAX_SIDE * SQUARE_KERNEL_MAX_SIDE;
/// The value of every reference sample when none is available (8-bit).
const SUBSTITUTE: i32 = 128;
/// The samples of a reference chain with edges of `edge` samples (both
/// edges and the corner).
const fn chain_len(edge: usize) -> usize {
    2 * edge + 1
}
/// Availability runs of a reference chain with edges of `edge` samples: a
/// cell's worth of samples on each edge (at most `edge` / chroma cell, a
/// chroma cell being half a luma one) and the corner.
const fn runs_len(edge: usize) -> usize {
    2 * (edge / (crate::motion::GRID_SIZE / 2)) + 1
}

/// Samples `start..start + len` of a reference chain, all available or not.
#[derive(Clone, Copy, Default)]
struct Run {
    start: usize,
    len: usize,
    available: bool,
}

#[derive(Clone)]
struct Edges<const EDGE: usize> {
    top: [i32; EDGE],
    left: [i32; EDGE],
    corner: i32,
}

impl<const EDGE: usize> Edges<EDGE> {
    /// The edges of an `n`-sample block from its reference `chain`
    /// (bottom-left to top-right); past 2n each edge repeats its last
    /// sample.
    fn of(chain: &[i32], n: usize) -> Self {
        let mut top = [0; EDGE];
        top[..2 * n].copy_from_slice(&chain[2 * n + 1..=4 * n]);
        top[2 * n..].fill(chain[4 * n]);
        let mut left = [0; EDGE];
        for (sample, &value) in left.iter_mut().zip(chain[..2 * n].iter().rev()) {
            *sample = value;
        }
        left[2 * n..].fill(chain[0]);
        Self {
            top,
            left,
            corner: chain[2 * n],
        }
    }
}

/// A block's intra reference samples, with edges of `EDGE` samples: blocks
/// up to `EDGE / 2` a side. `CHAIN` and `RUNS` size the working memory of
/// [`Self::new`] ([`chain_len`], [`runs_len`] of `EDGE`).
pub(crate) struct EdgeReferences<const EDGE: usize, const CHAIN: usize, const RUNS: usize> {
    raw: Edges<EDGE>,
    /// The [1, 2, 1]-filtered edges (§8.4.5.2.8), for luma from 8x8.
    smooth: Option<Edges<EDGE>>,
    side: usize,
    luma: bool,
}

/// The references of blocks up to 32x32 (every searched block).
pub(crate) type References =
    EdgeReferences<SMALL_EDGE, { chain_len(SMALL_EDGE) }, { runs_len(SMALL_EDGE) }>;
/// The references of a 64x64 block (`vvc.tx64`).
type LargeReferences = EdgeReferences<MAX_EDGE, { chain_len(MAX_EDGE) }, { runs_len(MAX_EDGE) }>;

impl<const EDGE: usize, const CHAIN: usize, const RUNS: usize> EdgeReferences<EDGE, CHAIN, RUNS> {
    /// The references of transform unit `tu` (luma coordinates) of coding
    /// unit `cu` in `plane`: samples of coded CUs (`grid`; `None`: every
    /// sample in the picture, for source-based estimates) and of the TUs
    /// of `cu` decoded before `tu`.
    pub(crate) fn new(
        picture: &Picture,
        grid: Option<&MotionGrid>,
        plane: PlaneIndex,
        tu: &Block,
        cu: &Block,
    ) -> Self {
        let shift = usize::from(plane != PlaneIndex::Y);
        let (x, y, n) = (tu.x >> shift, tu.y >> shift, tu.size() >> shift);
        debug_assert!(CHAIN == chain_len(EDGE) && RUNS == runs_len(EDGE));
        assert!(2 * n <= EDGE, "a {n}-sample block needs longer edges");
        let view = picture.plane(plane).view();
        let available = |px: isize, py: isize| {
            let (luma_x, luma_y) = ((px as usize) << shift, (py as usize) << shift);
            px >= 0
                && py >= 0
                && (px as usize) < view.width()
                && (py as usize) < view.height()
                && grid.is_none_or(|g| {
                    g.available(luma_x as isize, luma_y as isize)
                        || transform_tree::decoded_before(cu, tu, luma_x, luma_y)
                })
        };
        // Availability is decided per grid cell: coded blocks, transform
        // units and the coded picture size are multiples of a cell, so it
        // is the same for every sample of one. A run of `unit` samples on
        // either edge lies in one cell (the block is cell aligned).
        let unit = crate::motion::GRID_SIZE >> shift;
        debug_assert!(x.is_multiple_of(unit) && y.is_multiple_of(unit));
        // The reference chain from bottom-left (index 0) up the left
        // column, through the corner (2n) and along the top row (4n), as
        // runs of samples whose availability is decided together.
        let mut chain = [0; CHAIN];
        let mut runs = [Run::default(); RUNS];
        let mut run_count = 0;
        let mut push = |start: usize, len: usize, (px, py): (isize, isize)| {
            let available = available(px, py);
            runs[run_count] = Run {
                start,
                len,
                available,
            };
            run_count += 1;
            available
        };
        for start in (0..2 * n).step_by(unit) {
            // Bottom-up: chain index `start` is row y + 2n - 1 - start.
            let bottom = y + 2 * n - 1 - start;
            if push(start, unit, (x as isize - 1, bottom as isize)) {
                for (i, sample) in chain[start..start + unit].iter_mut().enumerate() {
                    *sample = i32::from(view.row(bottom - i)[x - 1]);
                }
            }
        }
        if push(2 * n, 1, (x as isize - 1, y as isize - 1)) {
            chain[2 * n] = i32::from(view.row(y - 1)[x - 1]);
        }
        for start in (2 * n + 1..=4 * n).step_by(unit) {
            let left = x + start - 2 * n - 1;
            if push(start, unit, (left as isize, y as isize - 1)) {
                let row = &view.row(y - 1)[left..left + unit];
                for (sample, &value) in chain[start..start + unit].iter_mut().zip(row) {
                    *sample = i32::from(value);
                }
            }
        }
        // Substitution (§8.4.5.2.8): an unavailable run takes the sample
        // before it; leading ones take the first available sample, 128 when
        // none is.
        let runs = &runs[..run_count];
        let mut last = runs
            .iter()
            .find(|run| run.available)
            .map_or(SUBSTITUTE, |run| chain[run.start]);
        for run in runs {
            let samples = &mut chain[run.start..run.start + run.len];
            if run.available {
                last = samples[run.len - 1];
            } else {
                samples.fill(last);
            }
        }
        let raw = Edges::of(&chain, n);
        let luma = plane == PlaneIndex::Y;
        let smooth = (luma && n >= 8).then(|| {
            let mut filtered = chain;
            for i in 1..4 * n {
                filtered[i] = (chain[i - 1] + 2 * chain[i] + chain[i + 1] + 2) >> 2;
            }
            Edges::of(&filtered, n)
        });
        Self {
            raw,
            smooth,
            side: n,
            luma,
        }
    }

    /// The unfiltered edges (top, left: twice the side each) and corner.
    pub(crate) fn raw_edges(&self) -> (&[i32], &[i32], i32) {
        (&self.raw.top, &self.raw.left, self.raw.corner)
    }

    pub(crate) fn predict(&self, dsp: &Dsp, selected: u8, out: &mut [u8]) {
        let (mode, edges) = self.mode(selected);
        // Only large references hold blocks past the square kernel's sides
        // (decided at compile time for the others).
        if EDGE > SMALL_EDGE && self.side > SQUARE_KERNEL_MAX_SIDE {
            let size = (self.side, self.side);
            dsp.vvc_intra_predict_rect(size, mode, &edges.top, &edges.left, edges.corner, out);
        } else {
            dsp.vvc_intra_predict(self.side, mode, &edges.top, &edges.left, edges.corner, out);
        }
    }

    /// [`Self::predict`] in the kernel's natural orientation: true when
    /// `out` holds the block transposed (horizontal directions), for
    /// transpose-invariant costs such as SATD.
    fn predict_oriented(&self, dsp: &Dsp, selected: u8, out: &mut [u8]) -> bool {
        if EDGE > SMALL_EDGE && self.side > SQUARE_KERNEL_MAX_SIDE {
            self.predict(dsp, selected, out);
            return false;
        }
        let (mode, edges) = self.mode(selected);
        dsp.vvc_intra_predict_oriented(self.side, mode, &edges.top, &edges.left, edges.corner, out)
    }

    /// The DSP mode of direction `selected` and the references it reads.
    fn mode(&self, selected: u8) -> (nc_dsp::VvcIntraMode, &Edges<EDGE>) {
        // Square-block integer diagonals exceed the normative filter threshold
        // from 8x8 upward. Fractional luma interpolation is handled by DSP.
        let filtered = self.luma && self.side >= 8 && matches!(selected, PLANAR | 2 | 34 | 66);
        let edges = match &self.smooth {
            Some(smooth) if filtered => smooth,
            _ => &self.raw,
        };
        let mode = match selected {
            PLANAR => nc_dsp::VvcIntraMode::Planar,
            DC => nc_dsp::VvcIntraMode::Dc,
            HORIZONTAL => nc_dsp::VvcIntraMode::Horizontal,
            VERTICAL => nc_dsp::VvcIntraMode::Vertical,
            _ => nc_dsp::VvcIntraMode::Angular {
                direction: selected,
                luma: self.luma,
            },
        };
        (mode, edges)
    }
}

/// The prediction of transform unit `tu` of `cu` (§8.4.5.1: intra
/// prediction runs per transform block).
pub(crate) fn prediction(
    dsp: &Dsp,
    picture: &Picture,
    grid: Option<&MotionGrid>,
    plane: PlaneIndex,
    (tu, cu): (&Block, &Block),
    mode: u8,
) -> Vec<u8> {
    let side = tu.size() >> usize::from(plane != PlaneIndex::Y);
    let mut out = vec![0; side * side];
    if side > SQUARE_KERNEL_MAX_SIDE {
        LargeReferences::new(picture, grid, plane, tu, cu).predict(dsp, mode, &mut out);
    } else {
        References::new(picture, grid, plane, tu, cu).predict(dsp, mode, &mut out);
    }
    out
}

pub(crate) fn supported(mode: u8) -> Result<u8> {
    if mode < 67 {
        Ok(mode)
    } else {
        Err(DecodeError::Unsupported("VVC intra direction"))
    }
}

/// Which luma directions get full RD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectionSearch {
    /// The four core modes and the cheapest `angular_candidates` others by
    /// the SATD proxy ([`shortlist_directions`]), whose first pass visits
    /// every `coarse_step`-th angular direction (a power of two); with a
    /// non-zero `rd_candidates`, only that many of them, cheapest first;
    /// with a non-zero `rd_margin_permille`, only those whose proxy is
    /// within that many ‰ of the cheapest.
    Ranked {
        angular_candidates: usize,
        coarse_step: u8,
        rd_candidates: usize,
        rd_margin_permille: u32,
    },
    /// The four core modes and the six MPMs, without a proxy sweep: HEVC's
    /// `hevc.coarse_inter_intra` for intra CUs in P frames; with a non-zero
    /// `rd_candidates`, only that many of them get full RD, the cheapest by
    /// the SATD proxy (HEVC Default codes its single cheapest).
    CoreAndMpm { rd_candidates: usize },
    /// DC alone, the mode the engine's intra test priced, without RD: HEVC's
    /// `hevc.inter_intra_dc_only` for intra CUs in P frames.
    DcOnly,
}

/// First angular direction (2) and the number of intra modes (67).
const FIRST_ANGULAR: u8 = 2;
const MODES: u8 = 67;

/// The directions given full RD: the four core modes plus the
/// `angular_candidates` cheapest others by `proxy` (SATD plus signalling),
/// and of those only the `rd_candidates` cheapest by `proxy` when it is
/// non-zero (core modes included: a smooth block's angular proxies, or an
/// edge's core ones, rarely win the full RD), and of those only the ones
/// within `rd_margin_permille` ‰ of the cheapest proxy when that is
/// non-zero (VTM's fast intra search likewise drops directions whose SATD
/// cost is far above the best before its RD stage).
fn shortlist_directions(
    angular_candidates: usize,
    coarse_step: u8,
    (rd_candidates, rd_margin_permille): (usize, u32),
    mut proxy: impl FnMut(u8) -> f32,
) -> Vec<u8> {
    let mut costs = [None; MODES as usize];
    let mut cost = |m: u8| *costs[usize::from(m)].get_or_insert_with(|| proxy(m));
    let exhaustive = usize::from(MODES) - CORE_MODES.len();
    let kept = if angular_candidates >= exhaustive {
        nc_encoder::shortlist_intra(CORE_MODES, 0..MODES, angular_candidates, &mut cost)
    } else {
        coarse_to_fine(angular_candidates, coarse_step, &mut cost)
    };
    let by_count = rd_candidates != 0 && kept.len() > rd_candidates;
    if !by_count && rd_margin_permille == 0 {
        return kept;
    }
    let mut ranked: Vec<(u8, f32)> = kept.into_iter().map(|m| (m, cost(m))).collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let limit = ranked[0].1 * (1. + rd_margin_permille as f32 / 1000.);
    let count = if by_count {
        rd_candidates
    } else {
        ranked.len()
    };
    ranked
        .into_iter()
        .take(count)
        .take_while(|&(_, cost)| rd_margin_permille == 0 || cost <= limit)
        .map(|(m, _)| m)
        .collect()
}

/// The core modes and the `angular_candidates` cheapest angular directions,
/// coarse to fine as VTM's fast intra search: the proxy visits every
/// `coarse_step`-th angular direction, then, halving the step down to one,
/// the two directions that far from each direction kept so far, and keeps
/// the cheapest `angular_candidates` of those seen. Neighbouring directions
/// predict nearly alike, so a direction whose neighbours both ranked poorly
/// rarely wins: a step of 2 skips about a quarter of the proxy predictions,
/// 4 about half.
fn coarse_to_fine(
    angular_candidates: usize,
    coarse_step: u8,
    cost: &mut impl FnMut(u8) -> f32,
) -> Vec<u8> {
    let mut kept = nc_encoder::shortlist_intra(
        CORE_MODES,
        (FIRST_ANGULAR..MODES).step_by(usize::from(coarse_step.max(1))),
        angular_candidates,
        &mut *cost,
    );
    let mut distance = coarse_step / 2;
    while distance > 0 {
        let neighbours: Vec<u8> = kept[CORE_MODES.len()..]
            .iter()
            .flat_map(|&m| [m.checked_sub(distance), m.checked_add(distance)])
            .flatten()
            .filter(|m| (FIRST_ANGULAR..MODES).contains(m))
            .collect();
        kept = nc_encoder::shortlist_intra(
            CORE_MODES,
            kept[CORE_MODES.len()..].iter().copied().chain(neighbours),
            angular_candidates,
            &mut *cost,
        );
        distance /= 2;
    }
    kept
}

/// The key-frame planner's leaf cost (`CodecBackend::intra_leaf_cost`):
/// luma only, DC predicted from the source neighbours (the reconstruction
/// does not exist yet), transformed, quantised and priced as coded, as
/// HEVC's `key_leaf_cost`. The direction search refines the chosen leaves.
/// A CU of several transform units sums them.
pub(crate) fn leaf_cost(
    dsp: &Dsp,
    source: &Picture,
    (block, max_tb_log2): (&Block, u8),
    contexts: &Contexts,
    quant: recon::Quant,
    lambda: f32,
    scratch: &mut recon::QuantScratch,
) -> f32 {
    let mut snapshot = contexts.clone();
    let mut bits = BitCost::default();
    mode(&mut bits, &mut snapshot, mpm(PLANAR, PLANAR), DC, false).expect("valid intra mode");
    let distortion: u64 = transform_tree::units(block, max_tb_log2)
        .map(|tu| {
            let pricing = (DC, quant, &mut bits, &mut snapshot, &mut *scratch);
            if tu.size() > SQUARE_KERNEL_MAX_SIDE {
                let references = LargeReferences::new(source, None, PlaneIndex::Y, &tu, block);
                luma_cost(dsp, source, &references, &tu, pricing).0
            } else {
                let references = References::new(source, None, PlaneIndex::Y, &tu, block);
                luma_cost(dsp, source, &references, &tu, pricing).0
            }
        })
        .sum();
    distortion as f32 + lambda * bits.0
}

/// Luma reconstruction SSE of `block` predicted in direction `selected`
/// at `quant`, and whether it codes a residual; adds its coefficient bits
/// to `bits` (adapting `contexts`).
fn luma_cost<const EDGE: usize, const CHAIN: usize, const RUNS: usize>(
    dsp: &Dsp,
    source: &Picture,
    references: &EdgeReferences<EDGE, CHAIN, RUNS>,
    block: &Block,
    (selected, quant, bits, contexts, scratch): (
        u8,
        recon::Quant,
        &mut BitCost,
        &mut Contexts,
        &mut recon::QuantScratch,
    ),
) -> (u64, bool) {
    let (x, y, n) = (block.x, block.y, block.size());
    // A 64x64 block (the key-frame planner's leaf cost with `vvc.tx64`)
    // takes a heap buffer; searched blocks a stack one.
    let mut buffer = [0u8; SEARCH_SAMPLES];
    let mut large = Vec::new();
    let pred = if n * n <= SEARCH_SAMPLES {
        &mut buffer[..n * n]
    } else {
        large.resize(n * n, 0);
        &mut large[..]
    };
    references.predict(dsp, selected, pred);
    let levels = recon::quantize(dsp, source, PlaneIndex::Y, x, y, pred, quant, None, scratch);
    let mut predicted = PlaneViewMut::new(pred, n, n, n).unwrap();
    recon::add(dsp, &mut predicted, levels, quant.scaling());
    let source = source
        .plane(PlaneIndex::Y)
        .region(x as isize, y as isize, n, n)
        .unwrap();
    let coded = levels.iter().any(|&v| v != 0);
    if coded {
        crate::coeff::emit(bits, &mut contexts.coeff, levels, n, 0);
    }
    let sse = dsp
        .variance(
            nc_dsp::BlockSize::new(n, n).unwrap(),
            &source,
            &predicted.as_view(),
        )
        .sse;
    (u64::from(sse), coded)
}

/// The directions' SATD proxy for full-RD shortlists: the prediction's
/// SATD against the source plus √λ times the direction's bits.
struct DirectionProxy<'a> {
    dsp: &'a Dsp,
    references: &'a References,
    source: &'a nc_common::PlaneView<'a>,
    /// The source transposed: SATD is transpose invariant, so horizontal
    /// directions are ranked in the kernel's transposed layout without
    /// transposing each prediction.
    transposed: [u8; SEARCH_SAMPLES],
    prediction: [u8; SEARCH_SAMPLES],
    sqrt_lambda: f32,
}

impl<'a> DirectionProxy<'a> {
    fn new(
        dsp: &'a Dsp,
        references: &'a References,
        source: &'a nc_common::PlaneView<'a>,
        lambda: f32,
    ) -> Self {
        let n = source.width();
        let mut transposed = [0u8; SEARCH_SAMPLES];
        for (row, out) in transposed[..n * n].chunks_exact_mut(n).enumerate() {
            for (column, sample) in out.iter_mut().enumerate() {
                *sample = source.row(column)[row];
            }
        }
        Self {
            dsp,
            references,
            source,
            transposed,
            prediction: [0; SEARCH_SAMPLES],
            sqrt_lambda: lambda.sqrt(),
        }
    }

    /// The proxy cost of direction `m` with MPM list `candidates`.
    fn cost(&mut self, m: u8, contexts: &Contexts, candidates: [u8; 6]) -> f32 {
        let n = self.source.width();
        let transposed =
            self.references
                .predict_oriented(self.dsp, m, &mut self.prediction[..n * n]);
        let predicted = nc_common::PlaneView::new(&self.prediction, n, n, n).unwrap();
        let source_transposed = nc_common::PlaneView::new(&self.transposed, n, n, n).unwrap();
        let source = if transposed {
            &source_transposed
        } else {
            self.source
        };
        self.dsp
            .satd(nc_dsp::BlockSize::new(n, n).unwrap(), source, &predicted) as f32
            + self.sqrt_lambda * mode_bits(contexts, candidates, m)
    }
}

/// Exact luma reconstruction SSE plus context-adaptive coefficient and mode
/// bit estimates. Chroma follows separately (`chroma::choose`).
///
/// A 64x64 CU is priced on its top-left 32x32 quadrant, whose references
/// all lie outside the CU. Without 64-point transforms that is its first
/// transform unit: pricing the others would need each candidate's earlier
/// units reconstructed first. (Not measured separately; screened with the
/// 64x64 CTUs, docs/handoff/vvc.md.) With them (`vvc.tx64`) the quadrant
/// stands in for the whole: a 64x64 prediction takes the general scalar
/// predictor (`Dsp::vvc_intra_predict_rect`), and pricing every candidate
/// direction with it cost +7% of a bodycam encode's instructions (CBR 300,
/// 30 frames, x86 callgrind).
/// The shared engine decides intra versus inter; this selects the codec's intra
/// direction, like the codec-local intra choices in the other backends.
/// Returns the best direction, its cost and whether it codes a luma
/// residual, and the runner-up direction (intra sub-partitions try both).
pub(crate) fn choose(
    dsp: &Dsp,
    source: &Picture,
    recon: &Picture,
    grid: Option<&MotionGrid>,
    block: &Block,
    contexts: &Contexts,
    quantizers: &recon::Quantizers,
    lambda: f32,
    search: DirectionSearch,
    scratch: &mut recon::QuantScratch,
) -> LumaChoice {
    let candidates = grid.map_or(mpm(PLANAR, PLANAR), |g| g.intra_mpm(block));
    let cu = block;
    let block = &transform_tree::units(cu, transform_tree::MAX_TB_LOG2)
        .next()
        .expect("a CU has a transform unit");
    let references = References::new(recon, grid, PlaneIndex::Y, block, cu);
    let n = block.size();
    let source_luma = source
        .plane(PlaneIndex::Y)
        .region(block.x as isize, block.y as isize, n, n)
        .unwrap();
    let selected = match search {
        DirectionSearch::DcOnly => vec![DC],
        DirectionSearch::CoreAndMpm { rd_candidates } => {
            let mut modes = CORE_MODES.to_vec();
            modes.extend(candidates.iter().filter(|m| !CORE_MODES.contains(m)));
            if rd_candidates == 0 || modes.len() <= rd_candidates {
                modes
            } else {
                let mut proxy = DirectionProxy::new(dsp, &references, &source_luma, lambda);
                let mut ranked: Vec<(u8, f32)> = modes
                    .into_iter()
                    .map(|m| (m, proxy.cost(m, contexts, candidates)))
                    .collect();
                ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
                ranked.truncate(rd_candidates);
                ranked.into_iter().map(|(m, _)| m).collect()
            }
        }
        DirectionSearch::Ranked {
            angular_candidates,
            coarse_step,
            rd_candidates,
            rd_margin_permille,
        } => {
            let mut proxy = DirectionProxy::new(dsp, &references, &source_luma, lambda);
            shortlist_directions(
                angular_candidates,
                coarse_step,
                (rd_candidates, rd_margin_permille),
                |m| proxy.cost(m, contexts, candidates),
            )
        }
    };
    let mut costs: Vec<(u8, f32, bool)> = selected
        .into_iter()
        .map(|selected| {
            let mut snapshot = contexts.clone();
            let mut bits = BitCost::default();
            mode(&mut bits, &mut snapshot, candidates, selected, false).expect("valid intra mode");
            let quant = quantizers.plane(PlaneIndex::Y, true);
            let pricing = (selected, quant, &mut bits, &mut snapshot, &mut *scratch);
            let (distortion, residual) = luma_cost(dsp, source, &references, block, pricing);
            (selected, distortion as f32 + lambda * bits.0, residual)
        })
        .collect();
    costs.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (mode, cost, residual) = costs[0];
    LumaChoice {
        mode,
        cost,
        residual,
        runner_up: costs.get(1).map(|&(m, ..)| m),
    }
}

/// A CU's chosen whole-CU luma direction ([`choose`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LumaChoice {
    pub(crate) mode: u8,
    /// Its rate-distortion cost (first transform unit, scaled as `choose`
    /// prices it).
    pub(crate) cost: f32,
    /// Whether it codes a luma residual.
    pub(crate) residual: bool,
    /// The second cheapest direction, if any was priced.
    pub(crate) runner_up: Option<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_proxy_margin_keeps_directions_near_the_cheapest() {
        // Planar costs 100, DC 105, every other direction 100 more per step
        // from DC: a 10% margin keeps planar and DC only.
        let proxy = |m: u8| {
            if m == PLANAR {
                100.
            } else {
                105. + 100. * f32::from(m.abs_diff(DC))
            }
        };
        let mut all = shortlist_directions(2, 4, (0, 0), proxy);
        let mut near = shortlist_directions(2, 4, (0, 100), proxy);
        all.sort_unstable();
        near.sort_unstable();
        assert!(all.len() > 2);
        assert_eq!(near, [PLANAR, DC]);
    }

    #[test]
    fn diagonal_content_selects_angular_and_preserves_core_rd_candidates() {
        let mut picture = Picture::new(64, 64, 0).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = picture.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, value) in view.row_mut(y).iter_mut().enumerate() {
                    *value = if plane == PlaneIndex::Y {
                        40 + (((x + y) / 4) % 2) as u8 * 160
                    } else {
                        128
                    };
                }
            }
        }
        let dsp = Dsp::new();
        let block = Block::square(16, 16, 4);
        let contexts = Contexts::new(22, false);
        let evaluate = |budget| {
            choose(
                &dsp,
                &picture,
                &picture,
                None,
                &block,
                &contexts,
                &recon::Quantizers::defaults(22),
                5.7,
                DirectionSearch::Ranked {
                    angular_candidates: budget,
                    coarse_step: 2,
                    rd_candidates: 0,
                    rd_margin_permille: 0,
                },
                &mut recon::QuantScratch::default(),
            )
        };
        let core = evaluate(0);
        let ranked = evaluate(8);
        let exhaustive = evaluate(63);
        assert!(
            !CORE_MODES.contains(&ranked.mode),
            "selected {}",
            ranked.mode
        );
        assert!(
            ranked.cost < core.cost,
            "ranked {} core {}",
            ranked.cost,
            core.cost
        );
        assert!(exhaustive.cost <= ranked.cost);
    }

    #[test]
    fn mpm_lists_are_valid_and_estimates_match_binarization() {
        for left in 0..67 {
            for above in 0..67 {
                let candidates = mpm(left, above);
                let mut sorted = candidates;
                sorted.sort_unstable();
                assert!(sorted.iter().all(|&m| m < 67));
                assert!(sorted.windows(2).all(|v| v[0] != v[1]));
            }
        }
        for left in [0, 1, 2, 18, 34, 50, 66] {
            for above in [0, 1, 2, 18, 34, 50, 66] {
                let candidates = mpm(left, above);
                let contexts = Contexts::new(37, false);
                for selected in 0..67 {
                    let mut snapshot = contexts.clone();
                    let mut bits = BitCost::default();
                    assert_eq!(
                        mode(&mut bits, &mut snapshot, candidates, selected, false).unwrap(),
                        selected
                    );
                    assert!((mode_bits(&contexts, candidates, selected) - bits.0).abs() < 1e-5);
                }
            }
        }
    }
}
