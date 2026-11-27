//! Rate-distortion cost of one inter coding unit's luma, for partition
//! planning (`CodecBackend::inter_leaf_cost`, `planner.rect_rd`).
//!
//! The CU is predicted at one vector and every luma transform unit is
//! quantised (no trellis), reconstructed and priced with the coefficient
//! contexts as they stand, exactly as `encode_block` would code it, but
//! without touching the coded state: the engine compares a square leaf with
//! a rectangular shape on the costs the codec will actually pay.
use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, Dsp};
use nc_encoder::{Block, Mv};

use crate::coeff::{BitCost, emit_rect};
use crate::coeff_contexts::CoeffContexts;
use crate::{inter, recon, transform_tree};

/// Temporary samples and coefficients reused between pricing trials.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    prediction: Vec<u8>,
    reconstructed: Vec<u8>,
    quant: Box<recon::QuantScratch>,
}

/// Luma reconstruction squared error and coefficient bits of CU `cu`
/// predicted from `reference` at `mv` (quarter samples). `contexts` adapt
/// within the trial. Copy them only if a unit has coded coefficients.
/// A cached prediction may be supplied when the engine already made it
/// at this vector and reference.
pub(crate) fn luma_cost(
    dsp: &Dsp,
    source: &Picture,
    reference: &Picture,
    (cu, mv, cached): (&Block, Mv, Option<&[u8]>),
    contexts: &CoeffContexts,
    (quant, max_tb_log2): (recon::Quant, u8),
    scratch: &mut Scratch,
) -> (u64, f32) {
    let Scratch {
        prediction,
        reconstructed,
        quant: coefficients,
    } = scratch;
    let prediction = cached.unwrap_or_else(|| {
        let (w, h) = (cu.width(), cu.height());
        prediction.resize(w * h, 0);
        let luma = reference.plane(PlaneIndex::Y);
        let mv = crate::motion::Mv16::from_quarter(mv);
        let mut out = inter::packed(prediction, (w, h));
        inter::predict_luma(
            dsp.at(nc_common::BitDepth::EIGHT),
            luma,
            (cu.x, cu.y),
            (w, h),
            mv,
            &mut out,
        );
        prediction
    });
    cost(
        dsp,
        source,
        cu,
        prediction,
        contexts,
        (quant, max_tb_log2),
        (reconstructed, coefficients),
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
    contexts: &CoeffContexts,
    (quant, max_tb_log2): (recon::Quant, u8),
    scratch: &mut Scratch,
) -> (u64, f32) {
    cost(
        dsp,
        source,
        cu,
        prediction,
        contexts,
        (quant, max_tb_log2),
        (&mut scratch.reconstructed, &mut scratch.quant),
    )
}

fn cost(
    dsp: &Dsp,
    source: &Picture,
    cu: &Block,
    prediction: &[u8],
    contexts: &CoeffContexts,
    (quant, max_tb_log2): (recon::Quant, u8),
    (reconstructed, scratch): (&mut Vec<u8>, &mut recon::QuantScratch),
) -> (u64, f32) {
    let mut bits = BitCost::default();
    let mut trial_contexts = None;
    let mut sse = 0;
    let (w, h) = (cu.width(), cu.height());
    let prediction = PlaneView::new(prediction, w, w, h).expect("prediction covers the CU");
    let source = source.plane(PlaneIndex::Y);
    for tu in transform_tree::units(cu, max_tb_log2) {
        let (tw, th) = (tu.width(), tu.height());
        let prediction = prediction
            .sub_view(tu.x - cu.x, tu.y - cu.y, tw, th)
            .expect("TU inside prediction");
        let original = source
            .region(tu.x as isize, tu.y as isize, tw, th)
            .expect("planned CUs lie inside the coded picture");
        let size = BlockSize::new(tw, th).expect("transform units are kernel sizes");
        let levels = recon::quantize_view(dsp, &original, &prediction, quant, None, scratch);
        let recon = if levels.is_empty() {
            prediction
        } else {
            emit_rect(
                &mut bits,
                trial_contexts.get_or_insert_with(|| contexts.clone()),
                dsp,
                levels,
                (tw, th),
                0,
            );
            reconstructed.clear();
            for row in 0..th {
                reconstructed.extend_from_slice(prediction.row(row));
            }
            let mut dest = inter::packed(reconstructed, (tw, th));
            recon::add(dsp, &mut dest, levels, quant.scaling());
            PlaneView::new(reconstructed, tw, tw, th).expect("unit buffer")
        };
        sse += u64::from(dsp.variance(size, &original, &recon).sse);
    }
    (sse, bits.0)
}
