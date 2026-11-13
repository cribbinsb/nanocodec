//! Rate-distortion pruning of inter luma residuals.
//!
//! The zero-block tests and the JM-style elimination thresholds
//! (`macroblock`) decide from the levels alone, with fixed costs, at every
//! quantiser. At coarse quantisers that keeps sparse ±1 levels whose bits
//! (coded block pattern, significance map, `mb_qp_delta`, and the
//! `P_Skip` they prevent) are worth more than the distortion they remove:
//! `crowd_mot20_06` at QP 51 spent 30% of its P-frame bits on inter residual,
//! and coding none at QP 48 beat coding it at QP 51 by 0.4 dB at equal rate.
//!
//! Here each sparse (all ±1) coded luma 8x8 quadrant of an inter
//! macroblock is kept only when the squared error it removes exceeds λ
//! times its bits, priced with the current CABAC contexts
//! ([`crate::cabac::BitCounter`]); then, if no chroma is coded, the whole
//! residual is dropped when what is left does not pay for its shared syntax
//! and, where the vector is the skip vector, for not being `P_Skip`.
//!
//! Chroma is never pruned: its squared error is cheap per sample against
//! luma's, but PSNR-YUV averages the planes' PSNR in dB, where a chroma
//! plane at ~40 dB loses as much per unit of error as luma does at ~28 dB.
//! Pruning chroma too lost 1.1-1.7 dB of chroma PSNR (`static_cam22` CBR
//! 262/523 kbps) for nothing in luma, and failed the first `J_real` screen.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, Dsp};

use crate::cabac::{BitCounter, MbPart};
use crate::macroblock::{CBP_CHROMA_NONE, Prediction, Residual, ResidualKind, blocks_of_8x8};

const MB: usize = 16;
const QUADRANT: usize = 8;
/// Luma 8x8 quadrants of a macroblock.
const QUADRANTS: usize = 4;

/// The coded luma quadrants (bit per raster quadrant) of an inter
/// `residual` that pruning considers: those whose levels are all ±1. A
/// level of magnitude 2 or more needs a coefficient of at least `(2 - 1/N)`
/// steps (inter rounding `1/N`), and dropping it adds at least about 3.3
/// step² of squared error; λ is about 0.14 step² per bit
/// (`0.85·2^((QP-12)/3)` against a step of `0.625·2^(QP/6)`), so such a
/// coefficient pays for some 24 bits on its own, more than a sparse block
/// costs. Leaving those quadrants out keeps the pricing off dense
/// macroblocks. That bound holds for the λ of the coded quantizer: with
/// `any_level` (λ scaled past it, `FramePlan::lambda_scale`) every coded
/// quadrant is considered.
pub fn prunable_quadrants(residual: &Residual, any_level: bool) -> u8 {
    // A branch-free fold (no early exit) vectorises: -1, 0, 1 map to 0..=2.
    let sparse = |levels: &[i32]| {
        any_level
            || levels
                .iter()
                .fold(0, |m, &l| m.max(l.wrapping_add(1) as u32))
                <= 2
    };
    let mut quadrants = 0;
    for b8 in 0..QUADRANTS {
        if residual.cbp_luma & (1 << b8) == 0 {
            continue;
        }
        let levels_sparse = if residual.kind == ResidualKind::Inter8x8 {
            sparse(&residual.luma_8x8[b8])
        } else {
            blocks_of_8x8(b8)
                .iter()
                .all(|&blk| sparse(&residual.luma[blk]))
        };
        if levels_sparse {
            quadrants |= 1 << b8;
        }
    }
    quadrants
}

/// The squared error the residual removes from each of `quadrants` of the
/// macroblock at `(x, y)` (prediction SSE minus reconstruction SSE; `recon`
/// holds the reconstruction with the residual); 0 for other quadrants.
pub fn luma_gains<'a>(
    dsp: &Dsp,
    source: &'a Picture,
    recon: &'a Picture,
    pred: &Prediction,
    (x, y): (usize, usize),
    quadrants: u8,
) -> [f32; QUADRANTS] {
    let size_8 = BlockSize::new(QUADRANT, QUADRANT).expect("8x8 is a kernel size");
    core::array::from_fn(|b8| {
        if quadrants & (1 << b8) == 0 {
            return 0.0;
        }
        let (qx, qy) = ((b8 % 2) * QUADRANT, (b8 / 2) * QUADRANT);
        let region = |plane: &'a Picture| -> PlaneView<'a> {
            plane
                .plane(PlaneIndex::Y)
                .region((x + qx) as isize, (y + qy) as isize, QUADRANT, QUADRANT)
                .expect("block inside the picture")
        };
        let src = region(source);
        let pred = PlaneView::new(&pred.luma[qy * MB + qx..], MB, QUADRANT, QUADRANT)
            .expect("prediction covers the quadrant");
        let before = dsp.variance(size_8, &src, &pred).sse;
        let after = dsp.variance(size_8, &src, &region(recon)).sse;
        before as f32 - after as f32
    })
}

/// The luma squared error of `recon` against `source` over the macroblock
/// at `(x, y)`.
pub fn luma_sse(dsp: &Dsp, source: &Picture, recon: &Picture, (x, y): (usize, usize)) -> f32 {
    fn region(picture: &Picture, (x, y): (usize, usize)) -> PlaneView<'_> {
        picture
            .plane(PlaneIndex::Y)
            .region(x as isize, y as isize, MB, MB)
            .expect("macroblock inside the picture")
    }
    let size = BlockSize::new(MB, MB).expect("16x16 is a kernel size");
    let (src, rec) = (region(source, (x, y)), region(recon, (x, y)));
    dsp.variance(size, &src, &rec).sse as f32
}

/// The coded luma quadrants worth their bits at `lambda` (squared error
/// per bit); quadrants outside `considered` are kept. `priced` counted the
/// macroblock coded with all of them; `uncoded_saving_bits` is what
/// dropping the whole residual saves beyond the quadrants and their shared
/// syntax (header bits over `P_Skip` when the vector is the skip vector,
/// else 0).
pub fn choose_quadrants(
    residual: &Residual,
    (considered, gains): (u8, &[f32; QUADRANTS]),
    priced: &BitCounter<'_>,
    uncoded_saving_bits: f32,
    lambda: f32,
) -> u8 {
    let mut keep = residual.cbp_luma & !considered;
    let mut kept_gain = 0.0;
    let mut kept_bits = 0.0;
    for (b8, &gain) in gains.iter().enumerate() {
        if considered & (1 << b8) == 0 {
            continue;
        }
        let bits = priced.part_bits(MbPart::Luma8x8(b8 as u8));
        if gain > lambda * bits {
            keep |= 1 << b8;
            kept_gain += gain;
            kept_bits += bits;
        }
    }
    // The whole residual can only go when every coded part was considered.
    let all_considered = considered == residual.cbp_luma && residual.cbp_chroma == CBP_CHROMA_NONE;
    let shared_bits = priced.part_bits(MbPart::ResidualShared) + uncoded_saving_bits;
    if all_considered && kept_gain <= lambda * (kept_bits + shared_bits) {
        keep = 0;
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{BinCosts, BinSink, CabacEncoder, SliceKind};
    use crate::macroblock::CBP_CHROMA_DC;
    use nc_common::BitWriter;

    /// A counter that charged `bits` to luma quadrant 0 (one equiprobable
    /// bypass bin per bit).
    fn priced<'a>(cabac: &'a CabacEncoder, costs: &'a BinCosts, bits: u32) -> BitCounter<'a> {
        let mut counter = cabac.bit_counter(costs);
        counter.begin_part(MbPart::Luma8x8(0));
        (0..bits).for_each(|_| counter.encode_bypass(true));
        counter
    }

    fn residual() -> Residual {
        let mut residual = Residual::empty(ResidualKind::Inter8x8);
        residual.cbp_luma = 0b11;
        residual
    }

    #[test]
    fn quadrants_are_kept_when_their_gain_pays_for_their_bits() {
        let cabac = CabacEncoder::new(BitWriter::new(), SliceKind::P, 0, 30);
        let costs = BinCosts::new();
        let counter = priced(&cabac, &costs, 10);
        let gains = [1100.0, 0.0, 0.0, 0.0];
        // λ = 100: quadrant 0 (10 bits) pays; quadrant 1 (gain 0) does not.
        let keep = choose_quadrants(&residual(), (0b11, &gains), &counter, 0.0, 100.0);
        assert_eq!(keep, 0b01);
        // Saving 2 header bits by becoming P_Skip tips the whole residual.
        let keep = choose_quadrants(&residual(), (0b11, &gains), &counter, 2.0, 100.0);
        assert_eq!(keep, 0);
        // A dense quadrant is kept, and then nothing can make P_Skip.
        let keep = choose_quadrants(&residual(), (0b01, &gains), &counter, 2.0, 100.0);
        assert_eq!(keep, 0b11);
        // So does coded chroma.
        let mut with_chroma = residual();
        with_chroma.cbp_chroma = CBP_CHROMA_DC;
        let keep = choose_quadrants(&with_chroma, (0b11, &gains), &counter, 2.0, 100.0);
        assert_eq!(keep, 0b01);
    }

    #[test]
    fn only_unit_level_quadrants_are_prunable() {
        let mut residual = residual();
        residual.luma_8x8[0][3] = -1;
        residual.luma_8x8[1][5] = 2;
        assert_eq!(prunable_quadrants(&residual, false), 0b01);
        residual.luma_8x8[0][7] = -3;
        assert_eq!(prunable_quadrants(&residual, false), 0);
    }
}
