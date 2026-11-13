//! Macroblock motion partitioning: whether an inter macroblock keeps the
//! engine's single 16x16 vector or splits into 16x8, 8x16 or four 8x8
//! partitions (H.264 Table 7-13, `P_8x8` with `P_L0_8x8` only).
//!
//! The engine decides per macroblock; the motion search's 8x8 vectors
//! (`CodecBackend::sub_block_motion`) say where the motion inside it is not
//! uniform. Where it is (all four within `SplitSearch::uniform_qpel` of the
//! 16x16 vector) the macroblock may stay 16x16 at no cost. Otherwise each
//! 8x8 quadrant picks the better of the 16x16 vector and its own search vector, refined to
//! quarter samples; 16x8 and 8x16 halves pick among their quadrants'
//! vectors. The shapes are ranked by luma SATD plus `lambda` (SATD units
//! per bit) times estimated vector and mode bits.
//!
//! Per-quadrant caching, stack shape options and omitted unused deblock
//! strengths cut P instructions 0.93% on two legal TRAIN cost points;
//! GOP300 -0.92%, bit-identical output (zero BD-rate change).

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, Dsp};
use nc_encoder::Mv;

use crate::mbinfo::MbPartition;
use crate::mc::McKernels;

/// Estimated `mb_type` (+ `sub_mb_type`) bits per shape.
const SHAPE_BITS: [f32; 4] = [1.0, 10.0, 10.0, 20.0];
/// Quarter-sample refinement steps around a quadrant's best vector: a cross
/// at half then quarter sample.
const REFINE_STEPS: [i16; 2] = [2, 1];

/// How much partition search a macroblock gets (`h264.split_*` params).
#[derive(Clone, Copy, Debug)]
pub struct SplitSearch {
    /// Search vectors this close (quarter samples, per component) to the
    /// 16x16 vector mean the motion inside the macroblock is uniform, and it
    /// stays 16x16 without a search; `None` always searches. The search
    /// vectors are whole-sample, so this cannot see the sub-sample motion
    /// differences that fine texture is most sensitive to.
    pub uniform_qpel: Option<i16>,
    /// Splitting is only considered when the 16x16 prediction's SATD per
    /// sample exceeds this.
    pub gate_satd: f32,
    /// How many of [`REFINE_STEPS`] refine each quadrant's vector.
    pub refine_steps: usize,
    /// Refine a quadrant whose best candidate is the 16x16 vector. That
    /// vector was already refined by the engine for the whole macroblock,
    /// so skipping saves the search where the quadrant agrees with it.
    pub refine_mv16: bool,
}
/// Estimated bits of one `mvd` component (UEG3 binarization with sign):
/// `|d| + 1` prefix bins below `uCoff` 9, else 9 prefix bins and a 3rd-order
/// Exp-Golomb suffix of `2 · log2((|d| - 9) / 8 + 1) + 4` bins; plus the
/// sign. Small whole numbers, so counted in integers.
pub fn mvd_bits(d: i32) -> f32 {
    const PREFIX_MAX: u32 = 9;
    let a = d.unsigned_abs();
    let bins = if a == 0 {
        1
    } else if a < PREFIX_MAX {
        a + 2
    } else {
        PREFIX_MAX + 2 * ((a - PREFIX_MAX) / 8 + 1).ilog2() + 4 + 1
    };
    bins as f32
}

/// Bits of vector `mv` predicted by `pred`.
fn vector_bits(mv: Mv, pred: Mv) -> f32 {
    mvd_bits(i32::from(mv.x) - i32::from(pred.x)) + mvd_bits(i32::from(mv.y) - i32::from(pred.y))
}

/// Luma SATD of a quadrant at a given vector, each computed at most once.
struct QuadrantCosts<'a> {
    dsp: &'a Dsp,
    mc: &'a McKernels,
    source: &'a Picture,
    reference: &'a Picture,
    origin: (usize, usize),
    cache: SatdCache,
}

/// Each quadrant has a short independent vector search. Keeping its cached
/// SATDs together avoids scanning entries for the other three quadrants.
struct SatdCache {
    entries: [[(Mv, u32); SATD_CACHE_PER_QUADRANT]; 4],
    len: [usize; 4],
}

/// Search and refinement need at most a handful of distinct vectors per
/// quadrant. Extra evaluations remain correct when not cached.
const SATD_CACHE_PER_QUADRANT: usize = 16;

impl SatdCache {
    fn new() -> Self {
        Self {
            entries: [[(Mv::default(), 0); SATD_CACHE_PER_QUADRANT]; 4],
            len: [0; 4],
        }
    }

    fn find(&self, q: usize, mv: Mv) -> Option<u32> {
        self.entries[q][..self.len[q]]
            .iter()
            .find(|&&(v, _)| v == mv)
            .map(|&(_, cost)| cost)
    }

    fn push(&mut self, (q, mv, cost): (usize, Mv, u32)) {
        if self.len[q] < SATD_CACHE_PER_QUADRANT {
            self.entries[q][self.len[q]] = (mv, cost);
            self.len[q] += 1;
        }
    }
}

impl QuadrantCosts<'_> {
    /// SATD of quadrant `q` predicted with `mv`.
    fn satd(&mut self, q: usize, mv: Mv) -> u32 {
        if let Some(cost) = self.cache.find(q, mv) {
            return cost;
        }
        let block8 = BlockSize::new(8, 8).expect("8x8 is a kernel size");
        let (qx, qy) = (self.origin.0 + 8 * (q % 2), self.origin.1 + 8 * (q / 2));
        let mut pred = [0u8; 64];
        self.mc
            .luma_8x8(self.reference.plane(PlaneIndex::Y), (qx, qy), mv, &mut pred);
        let src = self
            .source
            .plane(PlaneIndex::Y)
            .region(qx as isize, qy as isize, 8, 8)
            .expect("quadrant inside the picture");
        let pred = PlaneView::new(&pred, 8, 8, 8).expect("prediction covers the quadrant");
        let cost = self.dsp.satd(block8, &src, &pred);
        self.cache.push((q, mv, cost));
        cost
    }

    /// SATD of a set of quadrants predicted with `mv`.
    fn satd_of(&mut self, quadrants: &[usize], mv: Mv) -> f32 {
        quadrants.iter().map(|&q| self.satd(q, mv) as f32).sum()
    }
}

/// Choose the partitioning of the inter macroblock at `origin` whose
/// engine-chosen vector is `mv16`, given the search vectors of its four
/// quadrants (`sub_vectors`, raster) and the 16x16 predicted vector `pred`.
/// Returns the shape and the vector of each quadrant.
pub fn choose_partition(
    dsp: &Dsp,
    mc: &McKernels,
    source: &Picture,
    reference: &Picture,
    origin: (usize, usize),
    (mv16, luma16): (Mv, &[u8; 256]),
    sub_vectors: &[Mv; 4],
    pred: Mv,
    lambda: f32,
    search: SplitSearch,
) -> (MbPartition, [Mv; 4]) {
    let unsplit = (MbPartition::P16x16, [mv16; 4]);
    let uniform = search.uniform_qpel.is_some_and(|limit| {
        sub_vectors
            .iter()
            .all(|v| (v.x - mv16.x).abs() <= limit && (v.y - mv16.y).abs() <= limit)
    });
    if uniform {
        return unsplit;
    }
    // The 16x16 prediction already exists: its quadrant SATDs seed the cache
    // and gate the search (a well-predicted macroblock gains nothing from
    // splitting).
    let (cache, total16) = seed_with_16x16(dsp, source, origin, mv16, luma16);
    if (total16 as f32) < search.gate_satd * 256.0 {
        return unsplit;
    }
    let mut costs = QuadrantCosts {
        dsp,
        mc,
        source,
        reference,
        origin,
        cache,
    };
    let cost_of = |costs: &mut QuadrantCosts<'_>, quadrants: &[usize], mv: Mv| {
        costs.satd_of(quadrants, mv) + lambda * vector_bits(mv, pred)
    };

    // Best vector of each quadrant: 16x16 or own search vector, refined.
    let best: [(Mv, f32); 4] = core::array::from_fn(|q| {
        let mut best = [mv16, sub_vectors[q]]
            .into_iter()
            .map(|mv| (mv, cost_of(&mut costs, &[q], mv)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("two candidates");
        let steps = if best.0 == mv16 && !search.refine_mv16 {
            0
        } else {
            search.refine_steps.min(REFINE_STEPS.len())
        };
        for &step in &REFINE_STEPS[..steps] {
            let centre = best.0;
            for (dx, dy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
                let mv = Mv {
                    x: centre.x + dx,
                    y: centre.y + dy,
                };
                let cost = cost_of(&mut costs, &[q], mv);
                if cost < best.1 {
                    best = (mv, cost);
                }
            }
        }
        best
    });

    let shape_cost = |shape: MbPartition| lambda * SHAPE_BITS[shape as usize];
    let mut options = [
        (
            MbPartition::P16x16,
            [mv16; 4],
            cost_of(&mut costs, &[0, 1, 2, 3], mv16) + shape_cost(MbPartition::P16x16),
        ),
        (
            MbPartition::P8x8,
            best.map(|(mv, _)| mv),
            best.iter().map(|&(_, c)| c).sum::<f32>() + shape_cost(MbPartition::P8x8),
        ),
        (MbPartition::P16x8, [mv16; 4], 0.0),
        (MbPartition::P8x16, [mv16; 4], 0.0),
    ];
    for option in &mut options[2..] {
        let shape = option.0;
        let halves: [[usize; 2]; 2] = if shape == MbPartition::P16x8 {
            [[0, 1], [2, 3]]
        } else {
            [[0, 2], [1, 3]]
        };
        let mut mvs = [mv16; 4];
        let mut total = shape_cost(shape);
        for half in halves {
            let (mv, cost) = [mv16, best[half[0]].0, best[half[1]].0]
                .into_iter()
                .map(|mv| (mv, cost_of(&mut costs, &half, mv)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("three candidates");
            for q in half {
                mvs[q] = mv;
            }
            total += cost;
        }
        *option = (shape, mvs, total);
    }
    let (shape, mvs, _) = options
        .into_iter()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .expect("four options");
    (shape, mvs)
}

/// The quadrant SATDs of the macroblock's existing 16x16 prediction
/// `luma16` (vector `mv16`): a cache seeded with them, and their total.
fn seed_with_16x16(
    dsp: &Dsp,
    source: &Picture,
    origin: (usize, usize),
    mv16: Mv,
    luma16: &[u8; 256],
) -> (SatdCache, u32) {
    let src = source
        .plane(PlaneIndex::Y)
        .region(origin.0 as isize, origin.1 as isize, 16, 16)
        .expect("macroblock inside the picture");
    let pred16 = PlaneView::new(luma16, 16, 16, 16).expect("prediction covers the macroblock");
    let mut cache = SatdCache::new();
    let mut total16 = 0;
    for (q, cost) in dsp.satd_8x8_quads(&src, &pred16).into_iter().enumerate() {
        total16 += cost;
        cache.push((q, mv16, cost));
    }
    (cache, total16)
}
