//! Affine PROF reference padding; interpolation and refinement use shared DSP.
use super::{
    inter_prediction::{combine, precision},
    motion::Motion,
};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

pub(super) fn predict(
    dsp: &Dsp,
    references: [&[&Picture]; 2],
    motion: Motion,
    fields: [Option<crate::affine::Field>; 2],
    position: (usize, usize),
    out: &mut [u8; 16],
) {
    let mut samples = [[0i16; 16]; 2];
    for list in 0..2 {
        let Some(vector) = motion.lists[list] else {
            continue;
        };
        let reference = references[list][usize::from(vector.reference)].plane(PlaneIndex::Y);
        precision(
            dsp,
            reference,
            position,
            (4, 4),
            vector.vector,
            (true, false, true),
            &mut samples[list],
        );
        let Some(corrections) = fields[list].and_then(|f| f.refinement()) else {
            continue;
        };
        let mut padded = [0; 36];
        super::refinement::padding(
            reference,
            position,
            (4, 4),
            vector.vector,
            None,
            &samples[list],
            &mut padded,
        );
        dsp.vvc_prof_refine(&padded, &corrections, &mut samples[list]);
    }
    combine(dsp, motion, &samples[0], &samples[1], (4, 4), out);
}
