//! Checks of the batched sub-block prediction (`Dsp::predict_sub_blocks`,
//! H.266 affine motion): each ISA's batch against the scalar batch, and the
//! scalar batch against one scalar `Dsp::predict` per sub-block, over random
//! per-sub-block offsets and phases (zero phases often, as they take other
//! kernel paths). The destination is pre-filled and compared whole, so a
//! write outside a sub-block's place shows.
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, SubBlockMotion};

use crate::check::CheckReport;
use crate::check_codec::record;
use crate::rng::Rng;

/// The filters and sub-block sides of the codec's sub-block predictions:
/// affine luma and chroma 4x4, and the encoder's 8x8 search blocks.
const CASES: [(Filter, usize); 3] = [
    (Filter::VvcAffineLuma, 4),
    (Filter::VvcChroma, 4),
    (Filter::VvcLuma, 8),
];
/// Sub-blocks per row and rows of the batches tried (up to a 64x32 CU of
/// 4x4 sub-blocks).
const LAYOUTS: [(usize, usize); 6] = [(1, 1), (2, 1), (4, 4), (8, 2), (16, 8), (3, 5)];
/// Largest whole-sample spread of the sub-blocks' windows beyond their
/// places.
const SPREAD: usize = 12;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for (filter, side) in CASES {
        for (per_row, rows) in LAYOUTS {
            check_batch(
                candidate,
                reference,
                filter,
                side,
                (per_row, rows),
                rng,
                report,
            );
        }
    }
}

fn check_batch(
    candidate: &Dsp,
    reference: &Dsp,
    filter: Filter,
    side: usize,
    (per_row, rows): (usize, usize),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let size = BlockSize::new(side, side).expect("sub-blocks are kernel sizes");
    let extra = filter.taps() - 1;
    let (width, height) = (per_row * side, rows * side);
    let (window_w, window_h) = (width + SPREAD + extra, height + SPREAD + extra);
    let stride = window_w + rng.below(8);
    let mut samples = vec![0u8; (window_h - 1) * stride + window_w];
    rng.fill(&mut samples);
    let window = PlaneView::new(&samples, stride, window_w, window_h).expect("window fits");
    let phase = |rng: &mut Rng| {
        if rng.below(3) == 0 {
            0
        } else {
            rng.below(filter.positions()) as u8
        }
    };
    let motion: Vec<SubBlockMotion> = (0..per_row * rows)
        .map(|i| SubBlockMotion {
            x: ((i % per_row) * side + rng.below(SPREAD + 1)) as u16,
            y: ((i / per_row) * side + rng.below(SPREAD + 1)) as u16,
            frac_x: phase(rng),
            frac_y: phase(rng),
        })
        .collect();
    let dst_stride = width + rng.below(8);
    let mut start = vec![0u8; (height - 1) * dst_stride + width];
    rng.fill(&mut start);
    let batch = |dsp: &Dsp| {
        let mut out = start.clone();
        let mut dst = PlaneViewMut::new(&mut out, dst_stride, width, height).expect("fits");
        dsp.predict_sub_blocks(filter, size, &window, &motion, per_row, &mut dst);
        out
    };
    let expected = batch(reference);
    record(
        report,
        candidate.isa(),
        batch(candidate) == expected,
        || format!("predict_sub_blocks {filter:?} {side}x{side} {per_row}x{rows}"),
    );
    let mut one_by_one = start.clone();
    for (i, m) in motion.iter().enumerate() {
        let at = (i / per_row) * side * dst_stride + (i % per_row) * side;
        let sub = window
            .sub_view(
                usize::from(m.x),
                usize::from(m.y),
                side + extra,
                side + extra,
            )
            .expect("inside the window");
        let mut dst =
            PlaneViewMut::new(&mut one_by_one[at..], dst_stride, side, side).expect("fits");
        let fractions = (usize::from(m.frac_x), usize::from(m.frac_y));
        reference.predict(filter, size, &sub, fractions.0, fractions.1, &mut dst);
    }
    record(report, reference.isa(), one_by_one == expected, || {
        format!("predict_sub_blocks {filter:?} {side}x{side} {per_row}x{rows} vs predict")
    });
}
