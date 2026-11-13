//! Combined inter/intra prediction: whole-CU planar references, shared blending.
use super::{
    Inter, Unit,
    references::{Availability, References},
};
use nc_common::{Picture, PlaneIndex, PlaneViewMut};
use nc_dsp::{Dsp, HevcWeights, VvcIntraMode};
pub(super) fn prepare(
    unit: &Unit,
    picture: &Picture,
    available: &Availability,
    dsp: &Dsp,
) -> [Vec<u8>; 3] {
    if !matches!(unit.inter, Some(Inter::Ciip(_))) {
        return std::array::from_fn(|_| Vec::new());
    }
    std::array::from_fn(|plane| {
        let shift = usize::from(plane != 0);
        let node = unit.node;
        let (x, y, w, h) = (
            node.x >> shift,
            node.y >> shift,
            node.width >> shift,
            node.height >> shift,
        );
        if plane != 0 && w == 2 {
            return Vec::new();
        }
        let filtered = plane == 0 && nc_dsp::vvc_intra_filter_references((w, h), 0);
        let edges = References::gather(
            picture,
            available,
            PlaneIndex::ALL[plane],
            (x, y, w, h),
            filtered,
        );
        let mut prediction = vec![0; w * h];
        dsp.vvc_intra_predict_rect(
            (w, h),
            VvcIntraMode::Planar,
            &edges.top,
            &edges.left,
            edges.corner,
            &mut prediction,
        );
        prediction
    })
}
pub(super) fn blend(
    unit: &Unit,
    block: nc_encoder::Block,
    plane: usize,
    intra: &[u8],
    prediction: &mut [u8],
    dsp: &Dsp,
) {
    if intra.is_empty() {
        return;
    }
    let shift = usize::from(plane != 0);
    let (w, h) = (block.width() >> shift, block.height() >> shift);
    let (x, y) = (
        (block.x - unit.node.x) >> shift,
        (block.y - unit.node.y) >> shift,
    );
    let stride = unit.node.width >> shift;
    let intra: Vec<i16> = (y..y + h)
        .flat_map(|row| {
            intra[row * stride + x..row * stride + x + w]
                .iter()
                .copied()
                .map(i16::from)
        })
        .collect();
    let inter: Vec<_> = prediction.iter().copied().map(i16::from).collect();
    let weight = i16::from(unit.ciip_weight);
    let weights = HevcWeights {
        w0: 4 - weight,
        w1: weight,
        round: 2,
        shift: 2,
    };
    let mut out = PlaneViewMut::new(prediction, w, w, h).unwrap();
    dsp.hevc_weighted_prediction(weights, &inter, &intra, &mut out);
}
