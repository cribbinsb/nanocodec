//! Rate-distortion cost of one inter coding unit's luma, for partition
//! planning (`CodecBackend::inter_leaf_cost`, `planner.rect_rd`).
//!
//! The CU is predicted at one vector and every luma transform unit is
//! quantised (no trellis), reconstructed and priced with the coefficient
//! contexts as they stand, exactly as `encode_block` would code it, but
//! without touching the coded state: the engine compares a square leaf with
//! a rectangular shape on the costs the codec will actually pay.
use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp};
use nc_encoder::{Block, Mv};

use crate::coeff::{BitCost, emit_rect};
use crate::coeff_contexts::CoeffContexts;
use crate::{inter, recon, transform_tree};

/// Luma reconstruction squared error and coefficient bits of CU `cu`
/// predicted from `reference` at `mv` (quarter samples). `contexts` adapt
/// as the units are priced: pass a copy.
pub(crate) fn luma_cost(
    dsp: &Dsp,
    source: &Picture,
    reference: &Picture,
    (cu, mv): (&Block, Mv),
    contexts: &mut CoeffContexts,
    (quant, max_tb_log2): (recon::Quant, u8),
    scratch: &mut recon::QuantScratch,
) -> (u64, f32) {
    let (w, h) = (cu.width(), cu.height());
    // Sized to the CU: zeroing a 64x64 buffer for every 16x8 leaf cost
    // more than its prediction.
    let mut prediction = vec![0u8; w * h];
    let luma = reference.plane(PlaneIndex::Y);
    let mv = crate::motion::Mv16::from_quarter(mv);
    let mut out = inter::packed(&mut prediction, (w, h));
    inter::predict_luma(dsp, luma, (cu.x, cu.y), (w, h), mv, &mut out);
    residual_cost(
        dsp,
        source,
        cu,
        &prediction,
        contexts,
        (quant, max_tb_log2),
        scratch,
    )
}

/// Luma reconstruction squared error and coefficient bits of CU `cu` at
/// luma `prediction` (the CU's samples, row-major), as [`luma_cost`], with
/// `MaxTbSizeY` `1 << max_tb_log2`.
pub(crate) fn residual_cost(
    dsp: &Dsp,
    source: &Picture,
    cu: &Block,
    prediction: &[u8],
    contexts: &mut CoeffContexts,
    (quant, max_tb_log2): (recon::Quant, u8),
    scratch: &mut recon::QuantScratch,
) -> (u64, f32) {
    let mut bits = BitCost::default();
    let mut unit = Vec::new();
    let mut sse = 0;
    for tu in transform_tree::units(cu, max_tb_log2) {
        transform_tree::extract(prediction, cu, &tu, 0, &mut unit);
        let (tw, th) = (tu.width(), tu.height());
        let levels = recon::quantize_rect(
            dsp,
            source,
            PlaneIndex::Y,
            (tu.x, tu.y, tw),
            &unit,
            quant,
            None,
            scratch,
        );
        if !levels.is_empty() {
            emit_rect(&mut bits, contexts, levels, (tw, th), 0);
        }
        let mut reconstructed = PlaneViewMut::new(&mut unit, tw, tw, th).expect("unit buffer");
        recon::add(dsp, &mut reconstructed, levels, quant.scaling());
        let original = source
            .plane(PlaneIndex::Y)
            .region(tu.x as isize, tu.y as isize, tw, th)
            .expect("planned CUs lie inside the coded picture");
        let reconstructed = PlaneView::new(&unit, tw, tw, th).expect("unit buffer");
        let size = BlockSize::new(tw, th).expect("transform units are kernel sizes");
        sse += u64::from(dsp.variance(size, &original, &reconstructed).sse);
    }
    (sse, bits.0)
}
