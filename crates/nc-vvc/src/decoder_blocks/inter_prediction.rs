//! General inter reconstruction through shared interpolation and weighted DSP kernels.
use super::motion::{Motion, Vector};
use super::refinement::Edges;
use super::resampling::{self, Region, Scaling};
use crate::decoder_headers::Weights;
use crate::motion::Mv16;
use nc_common::{BitDepth, Picture, PlaneIndex, PlaneView, PlaneViewMut, Sample};
use nc_dsp::{DepthKernels, Filter, HevcWeights, Pixel};
/// What inter prediction reads: both reference picture lists, and the
/// slice's explicit weights when it predicts with them (H.266 §8.5.6.6.3).
#[derive(Clone, Copy)]
pub(crate) struct Sources<'a, T: Sample = u8> {
    pub pictures: [&'a [&'a Picture<T>]; 2],
    pub weights: Option<&'a Weights>,
    /// Where luma reference samples may be read from.
    pub edges: Edges,
    /// Each reference's scaling when it is scaled, by list and index.
    scalings: [&'a [Option<Scaling>]; 2],
    /// Whether any reference of the slice is scaled.
    rescaled: bool,
}

impl<'a, T: Sample> Sources<'a, T> {
    /// The references of a slice, with its explicit weights when the PPS
    /// enables weighting for its type (the tables are empty otherwise).
    pub(crate) fn new(
        pictures: [&'a [&'a Picture<T>]; 2],
        header: &'a crate::decoder_headers::SliceHeader,
        edges: Edges,
        scalings: [&'a [Option<Scaling>]; 2],
    ) -> Self {
        Self {
            pictures,
            weights: Some(&header.weights).filter(|w| !w.lists[0].is_empty()),
            edges,
            scalings,
            rescaled: scalings
                .iter()
                .flat_map(|list| list.iter())
                .any(Option::is_some),
        }
    }

    /// The scaling of reference `index` of `list`, when it is scaled
    /// (`RprConstraintsActiveFlag`, §8.3.2).
    pub(super) fn scaling(&self, list: usize, index: u8) -> Option<Scaling> {
        if !self.rescaled {
            return None;
        }
        self.scalings[list]
            .get(usize::from(index))
            .copied()
            .flatten()
    }

    /// Whether `motion` predicts from a scaled reference.
    pub(super) fn scaled(&self, motion: Motion) -> bool {
        self.rescaled
            && motion.lists.iter().enumerate().any(|(list, vector)| {
                vector.is_some_and(|v| self.scaling(list, v.reference).is_some())
            })
    }

    /// Where reference samples of `plane` may be read from (the luma edges
    /// divided by the chroma subsampling, §8.5.6.3.4).
    pub(super) fn edges(&self, plane: PlaneIndex) -> Edges {
        let shift = u32::from(plane != PlaneIndex::Y);
        Edges {
            wrap: self.edges.wrap.map(|offset| offset >> shift),
            area: self.edges.area.map(|area| area.map(|value| value >> shift)),
        }
    }

    /// References predicted from with the default weights.
    #[cfg(test)]
    pub(crate) fn unweighted(pictures: [&'a [&'a Picture<T>]; 2]) -> Self {
        Self {
            pictures,
            weights: None,
            edges: Edges::default(),
            scalings: [&[], &[]],
            rescaled: false,
        }
    }
}

/// Bits of the high-precision prediction samples (14 at every depth): they
/// carry `shift1 = 14 − BitDepth` fractional bits (H.266 §8.5.6.6.2).
const PRECISION: u32 = 14;
/// BCW weights sum to `1 << BCW_BITS` (H.266 §8.5.6.6.2 `w0 + w1 = 8`).
const BCW_BITS: u32 = 3;
/// The BCW weight of list 1 by the block's BCW index (`Motion::weight`;
/// H.266 §8.5.6.6.2 `bcwWLut` in this decoder's index order).
const BCW_WEIGHTS: [i16; 5] = [4, -2, 3, 5, 10];

/// Fractional bits of the high-precision samples at `depth`
/// (`shift1 = 14 − BitDepth`).
fn precision_bits(depth: BitDepth) -> u32 {
    PRECISION - depth.bits()
}

/// Explicit weighted prediction of a block from the given lists (§8.5.6.6.3),
/// folded into the shared weighted kernel: the offsets, scaled by `1 <<
/// (BitDepth − 8)`, join its rounding term.
fn explicit(weights: &Weights, motion: Motion, component: usize, depth: BitDepth) -> HevcWeights {
    let shift = u32::from(weights.denominators[component]) + precision_bits(depth);
    let entry = |list: usize| {
        motion.lists[list].map(|vector| {
            // A reference past the signalled count keeps the default weight.
            let [weight, offset] = weights.lists[list]
                .get(usize::from(vector.reference))
                .map_or([1 << weights.denominators[component], 0], |entry| {
                    entry[component]
                });
            (weight, i32::from(offset) << depth.above_eight())
        })
    };
    match (entry(0), entry(1)) {
        (Some((w0, o0)), Some((w1, o1))) => HevcWeights {
            w0,
            w1,
            round: (o0 + o1 + 1) << shift,
            shift: shift + 1,
        },
        // One list (a block without any reference predicts nothing).
        (first, second) => {
            let (weight, offset) = first.or(second).unwrap_or_default();
            HevcWeights {
                w0: if first.is_some() { weight } else { 0 },
                w1: if first.is_some() { 0 } else { weight },
                round: (1 << (shift - 1)) + (offset << shift),
                shift,
            }
        }
    }
}

// Shared interpolation accepts blocks of at least four samples. Predicting a
// padded chroma block preserves the requested samples and its filter support.
fn inter_prediction<T: Pixel>(
    dsp: DepthKernels<'_>,
    reference: &nc_common::Plane<T>,
    position: (usize, usize),
    size: (usize, usize),
    vector: crate::motion::Mv16,
    luma: bool,
    out: &mut [T],
) {
    let predict = if luma {
        crate::inter::predict_luma
    } else {
        crate::inter::predict_chroma
    };
    if size.0 >= 4 && size.1 >= 4 {
        predict(
            dsp,
            reference,
            position,
            size,
            vector,
            &mut crate::inter::packed(out, size),
        );
    } else {
        let padded = (size.0.max(4), size.1.max(4));
        let mut small = [T::default(); 4 * 64];
        let mut heap;
        let samples: &mut [T] = if padded.0 * padded.1 <= small.len() {
            &mut small[..padded.0 * padded.1]
        } else {
            heap = vec![T::default(); padded.0 * padded.1];
            &mut heap
        };
        let mut view = crate::inter::packed(samples, padded);
        predict(dsp, reference, position, padded, vector, &mut view);
        for (dest, source) in out
            .chunks_exact_mut(size.0)
            .zip(samples.chunks_exact(padded.0))
        {
            dest.copy_from_slice(&source[..size.0]);
        }
    }
}

/// Reusable slice-level scratch for high-precision inter prediction, avoiding
/// per-block stack zeroing of prediction and bounded reference windows.
pub(super) struct InterScratch<T: Sample = u8> {
    pub s0: [i16; 64 * 64],
    pub s1: [i16; 64 * 64],
    pub window: [T; 71 * 71],
    pub pixels: [T; 16 * 16],
}

impl<T: Sample> Default for InterScratch<T> {
    fn default() -> Self {
        Self {
            s0: [0; 64 * 64],
            s1: [0; 64 * 64],
            window: [T::default(); 71 * 71],
            pixels: [T::default(); 16 * 16],
        }
    }
}

pub(super) fn predict<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: Sources<'_, T>,
    motion: Motion,
    (plane, affine): (PlaneIndex, bool),
    (anchor, position): ((usize, usize), (usize, usize)),
    size: (usize, usize),
    scratch: &mut InterScratch<T>,
    out: &mut [T],
) {
    let luma = plane == PlaneIndex::Y;
    let weighted = (references.weights.is_some() && motion.weight == 0)
        || references.edges != Edges::default()
        || references.scaled(motion);
    if motion.lists.iter().flatten().count() == 1 && !(motion.half && luma) && !weighted {
        let (list, vector) = motion
            .lists
            .iter()
            .enumerate()
            .find_map(|(i, v)| v.map(|v| (i, v)))
            .unwrap();
        let ref_plane = references.pictures[list][usize::from(vector.reference)].plane(plane);
        if affine && luma {
            crate::inter::predict_with(
                dsp,
                Filter::VvcAffineLuma,
                ref_plane,
                position,
                size,
                vector.vector,
                &mut crate::inter::packed(out, size),
            );
            return;
        }
        inter_prediction(dsp, ref_plane, position, size, vector.vector, luma, out);
        return;
    }
    let len = size.0 * size.1;
    let region = Region {
        anchor,
        position,
        size,
    };
    let flags = (motion.half, affine);
    assert!(len <= scratch.s0.len());
    let (s0, s1) = (&mut scratch.s0[..len], &mut scratch.s1[..len]);
    match motion.lists {
        [Some(v0), None] => {
            high_precision(dsp, references, (0, v0), plane, region, flags, s0);
            combine(dsp, (motion, references.weights, plane), s0, s0, size, out);
        }
        [None, Some(v1)] => {
            high_precision(dsp, references, (1, v1), plane, region, flags, s1);
            combine(dsp, (motion, references.weights, plane), s1, s1, size, out);
        }
        [Some(v0), Some(v1)] => {
            high_precision(dsp, references, (0, v0), plane, region, flags, s0);
            high_precision(dsp, references, (1, v1), plane, region, flags, s1);
            combine(dsp, (motion, references.weights, plane), s0, s1, size, out);
        }
        [None, None] => {
            s0.fill(0);
            combine(dsp, (motion, references.weights, plane), s0, s0, size, out);
        }
    }
}
/// Combine the high-precision predictions of a block's lists: explicit
/// weights when the slice has them and the block's BCW index is the default,
/// else the default or BCW average.
/// (H.266 §8.5.6.6.2: one list `shift1 = 14 − BitDepth`; BCW `shift3 =
/// 17 − BitDepth` with weights summing to 8, the default average as BCW's
/// equal weights.)
pub(super) fn combine<T: Pixel>(
    dsp: DepthKernels<'_>,
    (motion, explicit_weights, plane): (Motion, Option<&Weights>, PlaneIndex),
    first: &[i16],
    second: &[i16],
    size: (usize, usize),
    out: &mut [T],
) {
    let (w, h) = size;
    let depth = BitDepth::of::<T>(dsp.depth());
    let single = precision_bits(depth);
    let pair = single + BCW_BITS;
    let weights = match motion.lists {
        _ if explicit_weights.is_some() && motion.weight == 0 => {
            explicit(explicit_weights.unwrap(), motion, plane as usize, depth)
        }
        [Some(_), None] => HevcWeights {
            w0: 1,
            w1: 0,
            round: 1 << (single - 1),
            shift: single,
        },
        [None, Some(_)] => HevcWeights {
            w0: 0,
            w1: 1,
            round: 1 << (single - 1),
            shift: single,
        },
        _ => {
            let second = BCW_WEIGHTS[usize::from(motion.weight)];
            HevcWeights {
                w0: (1 << BCW_BITS) - second,
                w1: second,
                round: 1 << (pair - 1),
                shift: pair,
            }
        }
    };
    let mut destination = PlaneViewMut::new(out, w, w, h).unwrap();
    dsp.hevc_weighted_prediction(weights, first, second, &mut destination);
}
/// The high-precision prediction of `region` of `plane` from `vector` of
/// `list`: through the scaled positions when that reference is scaled
/// (§8.5.6.3.1), else the unscaled interpolation.
pub(super) fn high_precision<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: Sources<'_, T>,
    (list, vector): (usize, Vector),
    plane: PlaneIndex,
    region: Region,
    (half, affine): (bool, bool),
    out: &mut [i16],
) {
    let reference = references.pictures[list][usize::from(vector.reference)].plane(plane);
    let luma = plane == PlaneIndex::Y;
    let edges = references.edges(plane);
    match references.scaling(list, vector.reference) {
        // Scaled references do not wrap around (§8.5.6.3.2).
        Some(scaling) => resampling::predict(
            dsp,
            reference,
            (scaling, vector.vector),
            region,
            (luma, half, affine),
            edges.area,
            out,
        ),
        None => precision(
            dsp,
            reference,
            region.position,
            region.size,
            vector.vector,
            (luma, half, affine),
            edges,
            out,
        ),
    }
}

pub(super) fn precision<T: Pixel>(
    dsp: DepthKernels<'_>,
    reference: &nc_common::Plane<T>,
    (x, y): (usize, usize),
    size: (usize, usize),
    vector: Mv16,
    (luma, half, affine): (bool, bool, bool),
    edges: Edges,
    out: &mut [i16],
) {
    precision_window(
        dsp,
        reference,
        ((x, y), size),
        vector,
        (luma, half, affine),
        (None, edges),
        None,
        out,
    );
}

/// A DMVR window replicates the original merge prediction's filter support.
/// Ordinary inter prediction borrows the plane directly whenever possible.
#[allow(clippy::too_many_arguments)]
pub(super) fn precision_window<T: Pixel>(
    dsp: DepthKernels<'_>,
    reference: &nc_common::Plane<T>,
    ((x, y), size): ((usize, usize), (usize, usize)),
    vector: Mv16,
    (luma, half, affine): (bool, bool, bool),
    (bounds, edges): (Option<super::refinement::Bounds>, Edges),
    window_buf: Option<&mut [T]>,
    out: &mut [i16],
) {
    let filter = if luma && affine {
        Filter::VvcAffineLuma
    } else if luma {
        Filter::VvcLuma
    } else {
        Filter::VvcChroma
    };
    let bits = if luma { 4 } else { 5 };
    let phases = (
        (vector.x & ((1 << bits) - 1)) as usize,
        (vector.y & ((1 << bits) - 1)) as usize,
    );
    let origin = (
        x as i32 + (vector.x >> bits) - filter.before() as i32,
        y as i32 + (vector.y >> bits) - filter.before() as i32,
    );
    let (w, h) = (size.0 + filter.taps() - 1, size.1 + filter.taps() - 1);
    let direct = bounds.is_none() && edges.direct(reference, origin, (w, h));
    if let Some(window) = direct
        .then(|| reference.region(origin.0 as isize, origin.1 as isize, w, h))
        .flatten()
    {
        dsp.predict_high_precision(filter, size, &window, phases, luma && half, out);
    } else {
        let view = reference.view();
        let fetch = super::refinement::Fetch::new(reference, bounds, edges);
        let mut fallback;
        let samples: &mut [T] = if let Some(buf) = window_buf {
            &mut buf[..w * h]
        } else {
            fallback = [T::default(); 71 * 71];
            &mut fallback[..w * h]
        };
        fetch.fill_window(&view, origin, (w, h), samples);
        let window = PlaneView::new(samples, w, w, h).unwrap();
        dsp.predict_high_precision(filter, size, &window, phases, luma && half, out);
    }
}

/// Affine uses the existing model's sub-block vectors and chroma diagonal mean.
pub(super) fn predict_unit<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: Sources<'_, T>,
    unit: &super::Unit,
    plane: PlaneIndex,
    block: nc_encoder::Block,
    scratch: &mut InterScratch<T>,
    out: &mut [T],
) {
    let chroma = usize::from(plane != PlaneIndex::Y);
    let position = (block.x >> chroma, block.y >> chroma);
    let size = (block.width() >> chroma, block.height() >> chroma);
    let mut motion = unit.motion.unwrap();
    // CIIP always averages the two inter predictions before the planar blend;
    // an inherited BCW index remains motion metadata, not a blending weight.
    if matches!(unit.inter, Some(super::Inter::Ciip(_))) {
        motion.weight = 0;
    }
    if let (Some(pair), Some(super::Inter::Geometric { split, .. })) = (unit.geometric, unit.inter)
    {
        geometric(
            dsp,
            references,
            unit,
            (pair, split),
            (plane, block),
            scratch,
            out,
        );
        return;
    }
    if !unit.subblocks.is_empty() {
        let step = 8 >> chroma;
        for y in (0..size.1).step_by(step) {
            for x in (0..size.0).step_by(step) {
                let motion =
                    super::motion::at(unit, block.x + (x << chroma), block.y + (y << chroma))
                        .unwrap();
                let (w, h) = (step.min(size.0 - x), step.min(size.1 - y));
                let mut pixels = [T::default(); 64];
                let at = (position.0 + x, position.1 + y);
                predict(
                    dsp,
                    references,
                    motion,
                    (plane, false),
                    (at, at),
                    (w, h),
                    scratch,
                    &mut pixels[..w * h],
                );
                for dy in 0..h {
                    out[(y + dy) * size.0 + x..(y + dy) * size.0 + x + w]
                        .copy_from_slice(&pixels[dy * w..dy * w + w]);
                }
            }
        }
        return;
    }
    if !unit.dmvr.is_empty() || (unit.bdof && chroma == 0) {
        super::refinement::predict(dsp, references, unit, plane, block, scratch, out);
        return;
    }
    if unit.affine_models.iter().all(Option::is_none) {
        // The whole CU shares its scaled sample positions (§8.5.6.3.1).
        let anchor = (unit.node.x >> chroma, unit.node.y >> chroma);
        predict(
            dsp,
            references,
            motion,
            (plane, false),
            (anchor, position),
            size,
            scratch,
            out,
        );
        return;
    }
    affine(dsp, references, unit, motion, (plane, block), scratch, out);
}

fn affine<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: Sources<'_, T>,
    unit: &super::Unit,
    motion: Motion,
    (plane, block): (PlaneIndex, nc_encoder::Block),
    scratch: &mut InterScratch<T>,
    out: &mut [T],
) {
    let chroma = usize::from(plane != PlaneIndex::Y);
    let position = (block.x >> chroma, block.y >> chroma);
    let size = (block.width() >> chroma, block.height() >> chroma);
    let n = unit.node;
    let fields = unit.affine_models.map(|m| {
        m.map(|m| {
            m.field_for_prediction(n.width, n.height, motion.lists.iter().all(Option::is_some))
        })
    });
    for y in (0..size.1).step_by(4) {
        for x in (0..size.0).step_by(4) {
            let (col, row) = (
                (block.x - n.x + (x << chroma)) / 4,
                (block.y - n.y + (y << chroma)) / 4,
            );
            let mut local = motion;
            for (vector, field) in local.lists.iter_mut().zip(fields) {
                if let (Some(vector), Some(field)) = (vector, field) {
                    let first = field.at(col, row);
                    vector.vector = if chroma == 0 {
                        first
                    } else {
                        let last = field.at(col + 1, row + 1);
                        Mv16 {
                            x: crate::affine::round_shift(
                                i64::from(first.x) + i64::from(last.x),
                                1,
                            ),
                            y: crate::affine::round_shift(
                                i64::from(first.y) + i64::from(last.y),
                                1,
                            ),
                        }
                    };
                }
            }
            let mut pixels = [T::default(); 16];
            if unit.prof && chroma == 0 {
                super::prof::predict(
                    dsp,
                    references,
                    local,
                    fields,
                    (position.0 + x, position.1 + y),
                    &mut pixels,
                );
            } else {
                let at = (position.0 + x, position.1 + y);
                predict(
                    dsp,
                    references,
                    local,
                    (plane, true),
                    (at, at),
                    (4, 4),
                    scratch,
                    &mut pixels,
                );
            }
            for dy in 0..4 {
                out[(y + dy) * size.0 + x..(y + dy) * size.0 + x + 4]
                    .copy_from_slice(&pixels[dy * 4..dy * 4 + 4]);
            }
        }
    }
}

fn geometric<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: Sources<'_, T>,
    unit: &super::Unit,
    (pair, split): ([Motion; 2], u8),
    (plane, block): (PlaneIndex, nc_encoder::Block),
    scratch: &mut InterScratch<T>,
    out: &mut [T],
) {
    let chroma = usize::from(plane != PlaneIndex::Y);
    let position = (block.x >> chroma, block.y >> chroma);
    let size = (block.width() >> chroma, block.height() >> chroma);
    let n = unit.node;
    let region = Region {
        anchor: (n.x >> chroma, n.y >> chroma),
        position,
        size,
    };
    let len = size.0 * size.1;
    let (s0, s1) = (&mut scratch.s0[..len], &mut scratch.s1[..len]);
    for (motion, dst) in [pair[0], pair[1]].into_iter().zip([&mut *s0, &mut *s1]) {
        let (list, vector) = motion
            .lists
            .into_iter()
            .enumerate()
            .find_map(|(i, v)| v.map(|v| (i, v)))
            .unwrap();
        high_precision(
            dsp,
            references,
            (list, vector),
            plane,
            region,
            (false, false),
            dst,
        );
    }
    let mask = nc_dsp::VvcGeometric::new((n.width, n.height), split);
    dsp.vvc_geometric_blend(
        mask,
        (block.x - n.x, block.y - n.y),
        size,
        chroma != 0,
        (s0, s1),
        out,
    );
}

#[cfg(test)]
mod weight_tests {
    use super::*;
    use crate::decoder_blocks::motion::Vector;

    /// H.266 §8.5.6.6.3 written out, for one sample.
    fn reference(weights: &Weights, lists: [Option<(u8, i32)>; 2]) -> i32 {
        let shift = i32::from(weights.denominators[0]) + 6;
        let entry = |list: usize, index: u8| weights.lists[list][usize::from(index)][0];
        let value = match lists {
            [Some((a, sample)), None] | [None, Some((a, sample))] => {
                let list = usize::from(lists[0].is_none());
                let [weight, offset] = entry(list, a).map(i32::from);
                ((sample * weight + (1 << (shift - 1))) >> shift) + offset
            }
            [Some((a, first)), Some((b, second))] => {
                let ([w0, o0], [w1, o1]) = (entry(0, a).map(i32::from), entry(1, b).map(i32::from));
                (first * w0 + second * w1 + ((o0 + o1 + 1) << shift)) >> (shift + 1)
            }
            [None, None] => unreachable!(),
        };
        value.clamp(0, 255)
    }

    #[test]
    fn folded_offsets_match_the_specification_formula() {
        let weights = Weights {
            denominators: [5, 5, 5],
            lists: [
                vec![[[40, -7], [32, 0], [32, 0]], [[-20, 30], [32, 0], [32, 0]]],
                vec![[[25, 12], [32, 0], [32, 0]]],
            ],
            signalled: [vec![true, true], vec![true]],
        };
        let dsp = nc_dsp::Dsp::new();
        let vector = |reference| {
            Some(Vector {
                reference,
                vector: Mv16::ZERO,
            })
        };
        for lists in [
            [Some(0), None],
            [Some(1), None],
            [None, Some(0)],
            [Some(0), Some(0)],
            [Some(1), Some(0)],
        ] {
            let motion = Motion {
                lists: lists.map(|l| l.and_then(vector)),
                ..Motion::default()
            };
            // High-precision samples (pixel << 6, with filter over- and undershoot).
            for (first, second) in [
                (-8192, -8192),
                (0, 4000),
                (8128, -3000),
                (8128, 8128),
                (-100, 77),
            ] {
                let (a, b) = ([first as i16; 16], [second as i16; 16]);
                let mut out = [0u8; 16];
                combine(
                    dsp.at(BitDepth::EIGHT),
                    (motion, Some(&weights), PlaneIndex::Y),
                    &a,
                    &b,
                    (4, 4),
                    &mut out,
                );
                let expected = reference(
                    &weights,
                    [lists[0].map(|l| (l, first)), lists[1].map(|l| (l, second))],
                );
                assert_eq!(i32::from(out[0]), expected, "{lists:?} {first} {second}");
                assert!(out.iter().all(|&v| v == out[0]));
            }
        }
    }
}
