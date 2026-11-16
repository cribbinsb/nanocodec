//! Affine PROF reference padding; interpolation and refinement use shared DSP.
use super::{
    inter_prediction::{combine, high_precision},
    motion::Motion,
    resampling::Region,
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
        high_precision(
            dsp,
            references,
            (list, vector),
            PlaneIndex::Y,
            Region {
                anchor: position,
                position,
                size: (4, 4),
            },
            (false, true),
            &mut samples[list],
        );
        // PROF is off for a list whose reference is scaled (§8.5.5.9
        // `cbProfFlagLX`).
        let Some(corrections) = fields[list]
            .and_then(|f| f.refinement())
            .filter(|_| references.scaling(list, vector.reference).is_none())
        else {
            continue;
        };
        let reference =
            references.pictures[list][usize::from(vector.reference)].plane(PlaneIndex::Y);
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
