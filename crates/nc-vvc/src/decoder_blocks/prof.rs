//! Affine PROF reference padding; interpolation and refinement use shared DSP.
use super::{
    inter_prediction::{combine, precision},
    motion::Motion,
};
use nc_common::PlaneIndex;
use nc_dsp::{DepthKernels, Pixel};

pub(super) fn predict<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: super::inter_prediction::Sources<'_, T>,
    motion: Motion,
    fields: [Option<crate::affine::Field>; 2],
    position: (usize, usize),
    out: &mut [T; 16],
) {
    let mut samples = [[0i16; 16]; 2];
    for list in 0..2 {
        let Some(vector) = motion.lists[list] else {
            continue;
        };
        let reference =
            references.pictures[list][usize::from(vector.reference)].plane(PlaneIndex::Y);
        precision(
            dsp,
            reference,
            position,
            (4, 4),
            vector.vector,
            (true, false, true),
            references.edges(PlaneIndex::Y),
            &mut samples[list],
        );
        let Some(corrections) = fields[list].and_then(|f| f.refinement()) else {
            continue;
        };
        let mut padded = [0; 36];
        super::refinement::padding(
            (reference, dsp.depth()),
            position,
            (4, 4),
            vector.vector,
            (None, references.edges(PlaneIndex::Y)),
            &samples[list],
            &mut padded,
        );
        dsp.vvc_prof_refine(&padded, &corrections, &mut samples[list]);
    }
    combine(
        dsp,
        (motion, references.weights, PlaneIndex::Y),
        &samples[0],
        &samples[1],
        (4, 4),
        out,
    );
}
