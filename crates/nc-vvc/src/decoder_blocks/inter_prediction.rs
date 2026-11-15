//! General inter reconstruction through shared interpolation and weighted DSP kernels.
use super::motion::Motion;
use super::refinement::Edges;
use crate::decoder_headers::Weights;
use crate::motion::Mv16;
use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{Dsp, Filter, HevcWeights};
/// What inter prediction reads: both reference picture lists, and the
/// slice's explicit weights when it predicts with them (H.266 §8.5.6.6.3).
#[derive(Clone, Copy)]
pub(crate) struct Sources<'a> {
    pub pictures: [&'a [&'a Picture]; 2],
    pub weights: Option<&'a Weights>,
    /// Where luma reference samples may be read from.
    pub edges: Edges,
}

impl<'a> Sources<'a> {
    /// The references of a slice, with its explicit weights when the PPS
    /// enables weighting for its type (the tables are empty otherwise).
    pub(crate) fn new(
        pictures: [&'a [&'a Picture]; 2],
        header: &'a crate::decoder_headers::SliceHeader,
        edges: Edges,
    ) -> Self {
        Self {
            pictures,
            weights: Some(&header.weights).filter(|w| !w.lists[0].is_empty()),
            edges,
        }
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
    pub(crate) fn unweighted(pictures: [&'a [&'a Picture]; 2]) -> Self {
        Self {
            pictures,
            weights: None,
            edges: Edges::default(),
        }
    }
}

/// Fractional bits of the high-precision prediction samples of 8-bit
/// pictures (`shift1 = 14 - BitDepth`).
const PRECISION_BITS: u32 = 6;

/// Explicit weighted prediction of a block from the given lists (§8.5.6.6.3),
/// folded into the shared weighted kernel: the offsets join its rounding term.
fn explicit(weights: &Weights, motion: Motion, component: usize) -> HevcWeights {
    let shift = u32::from(weights.denominators[component]) + PRECISION_BITS;
    let entry = |list: usize| {
        motion.lists[list].map(|vector| {
            // A reference past the signalled count keeps the default weight.
            let [weight, offset] = weights.lists[list]
                .get(usize::from(vector.reference))
                .map_or([1 << weights.denominators[component], 0], |entry| {
                    entry[component]
                });
            (weight, i32::from(offset))
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
fn inter_prediction(
    dsp: &Dsp,
    reference: &nc_common::Plane,
    position: (usize, usize),
    size: (usize, usize),
    vector: crate::motion::Mv16,
    luma: bool,
    out: &mut [u8],
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
        let mut samples = vec![0; padded.0 * padded.1];
        let mut view = crate::inter::packed(&mut samples, padded);
        predict(dsp, reference, position, padded, vector, &mut view);
        for (dest, source) in out
            .chunks_exact_mut(size.0)
            .zip(samples.chunks_exact(padded.0))
        {
            dest.copy_from_slice(&source[..size.0]);
        }
    }
}

pub(super) fn predict(
    dsp: &Dsp,
    references: Sources<'_>,
    motion: Motion,
    (plane, affine): (PlaneIndex, bool),
    position: (usize, usize),
    size: (usize, usize),
    out: &mut [u8],
) {
    let luma = plane == PlaneIndex::Y;
    // Explicit weights apply to the high-precision samples, also of one list;
    // so do the wrapped and subpicture-clamped fetches, which the direct path
    // does not have.
    let weighted = (references.weights.is_some() && motion.weight == 0)
        || references.edges != Edges::default();
    if motion.lists.iter().flatten().count() == 1 && !(motion.half && luma) && !weighted {
        let (list, vector) = motion
            .lists
            .iter()
            .enumerate()
            .find_map(|(i, v)| v.map(|v| (i, v)))
            .unwrap();
        if affine && luma {
            crate::inter::predict_with(
                dsp,
                Filter::VvcAffineLuma,
                references.pictures[list][usize::from(vector.reference)].plane(plane),
                position,
                size,
                vector.vector,
                &mut crate::inter::packed(out, size),
            );
            return;
        }
        inter_prediction(
            dsp,
            references.pictures[list][usize::from(vector.reference)].plane(plane),
            position,
            size,
            vector.vector,
            luma,
            out,
        );
        return;
    }
    let (w, h) = size;
    let mut samples = [vec![0i16; w * h], vec![0i16; w * h]];
    for (list, vector) in motion.lists.into_iter().enumerate() {
        if let Some(vector) = vector {
            precision(
                dsp,
                references.pictures[list][usize::from(vector.reference)].plane(plane),
                position,
                size,
                vector.vector,
                (luma, motion.half, affine),
                references.edges(plane),
                &mut samples[list],
            );
        }
    }
    combine(
        dsp,
        (motion, references.weights, plane),
        &samples[0],
        &samples[1],
        size,
        out,
    );
}
/// Combine the high-precision predictions of a block's lists: explicit
/// weights when the slice has them and the block's BCW index is the default,
/// else the default or BCW average.
pub(super) fn combine(
    dsp: &Dsp,
    (motion, explicit_weights, plane): (Motion, Option<&Weights>, PlaneIndex),
    first: &[i16],
    second: &[i16],
    size: (usize, usize),
    out: &mut [u8],
) {
    let (w, h) = size;
    let weights = match motion.lists {
        _ if explicit_weights.is_some() && motion.weight == 0 => {
            explicit(explicit_weights.unwrap(), motion, plane as usize)
        }
        [Some(_), None] => HevcWeights {
            w0: 1,
            w1: 0,
            round: 32,
            shift: 6,
        },
        [None, Some(_)] => HevcWeights {
            w0: 0,
            w1: 1,
            round: 32,
            shift: 6,
        },
        _ => {
            let second = [4, -2, 3, 5, 10][usize::from(motion.weight)];
            HevcWeights {
                w0: 8 - second,
                w1: second,
                round: 256,
                shift: 9,
            }
        }
    };
    let mut destination = PlaneViewMut::new(out, w, w, h).unwrap();
    dsp.hevc_weighted_prediction(weights, first, second, &mut destination);
}
pub(super) fn precision(
    dsp: &Dsp,
    reference: &nc_common::Plane,
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
        out,
    );
}

/// A DMVR window replicates the original merge prediction's filter support.
/// Ordinary inter prediction borrows the plane directly whenever possible.
pub(super) fn precision_window(
    dsp: &Dsp,
    reference: &nc_common::Plane,
    ((x, y), size): ((usize, usize), (usize, usize)),
    vector: Mv16,
    (luma, half, affine): (bool, bool, bool),
    (bounds, edges): (Option<super::refinement::Bounds>, Edges),
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
        let mut samples = vec![0; w * h];
        for (y, row) in samples.chunks_exact_mut(w).enumerate() {
            let source = view.row(fetch.row(origin.1 + y as i32));
            for (x, pixel) in row.iter_mut().enumerate() {
                *pixel = source[fetch.column(origin.0 + x as i32)];
            }
        }
        let window = PlaneView::new(&samples, w, w, h).unwrap();
        dsp.predict_high_precision(filter, size, &window, phases, luma && half, out);
    }
}

/// Affine uses the existing model's sub-block vectors and chroma diagonal mean.
pub(super) fn predict_unit(
    dsp: &Dsp,
    references: Sources<'_>,
    unit: &super::Unit,
    plane: PlaneIndex,
    block: nc_encoder::Block,
    out: &mut [u8],
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
        geometric(dsp, references, unit, (pair, split), (plane, block), out);
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
                let mut pixels = [0; 64];
                predict(
                    dsp,
                    references,
                    motion,
                    (plane, false),
                    (position.0 + x, position.1 + y),
                    (w, h),
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
        super::refinement::predict(dsp, references, unit, plane, block, out);
        return;
    }
    if unit.affine_models.iter().all(Option::is_none) {
        predict(dsp, references, motion, (plane, false), position, size, out);
        return;
    }
    affine(dsp, references, unit, motion, (plane, block), out);
}

fn affine(
    dsp: &Dsp,
    references: Sources<'_>,
    unit: &super::Unit,
    motion: Motion,
    (plane, block): (PlaneIndex, nc_encoder::Block),
    out: &mut [u8],
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
            let mut pixels = [0; 16];
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
                predict(
                    dsp,
                    references,
                    local,
                    (plane, true),
                    (position.0 + x, position.1 + y),
                    (4, 4),
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

fn geometric(
    dsp: &Dsp,
    references: Sources<'_>,
    unit: &super::Unit,
    (pair, split): ([Motion; 2], u8),
    (plane, block): (PlaneIndex, nc_encoder::Block),
    out: &mut [u8],
) {
    let chroma = usize::from(plane != PlaneIndex::Y);
    let position = (block.x >> chroma, block.y >> chroma);
    let size = (block.width() >> chroma, block.height() >> chroma);
    let mut samples = [vec![0i16; size.0 * size.1], vec![0i16; size.0 * size.1]];
    for (motion, samples) in pair.into_iter().zip(&mut samples) {
        let (list, vector) = motion
            .lists
            .into_iter()
            .enumerate()
            .find_map(|(i, v)| v.map(|v| (i, v)))
            .unwrap();
        precision(
            dsp,
            references.pictures[list][usize::from(vector.reference)].plane(plane),
            position,
            size,
            vector.vector,
            (chroma == 0, false, false),
            references.edges(plane),
            samples,
        );
    }
    let n = unit.node;
    let mask = nc_dsp::VvcGeometric::new((n.width, n.height), split);
    dsp.vvc_geometric_blend(
        mask,
        (block.x - n.x, block.y - n.y),
        size,
        chroma != 0,
        (&samples[0], &samples[1]),
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
        let dsp = Dsp::new();
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
                    &dsp,
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
