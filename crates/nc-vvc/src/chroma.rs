//! H.266 §7.3.10 independent chroma intra direction syntax and RD selection.
//! The four explicit modes follow `VVenC` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
//! Derived chroma remains a candidate; pixels, transforms and distortion use DSP.
use crate::{
    coeff::{Bins, BitCost},
    contexts::Contexts,
    intra,
    motion::MotionGrid,
    recon,
};
use nc_common::{Picture, PlaneIndex, PlaneViewMut, decode::Result};
use nc_dsp::Dsp;
use nc_encoder::Block;

pub(crate) fn candidates(luma: u8) -> [u8; 4] {
    [intra::PLANAR, intra::VERTICAL, intra::HORIZONTAL, intra::DC]
        .map(|direction| if direction == luma { 66 } else { direction })
}

/// The luma direction itself represents derived mode; explicit candidates
/// substitute direction 66 when a fixed candidate would duplicate luma.
pub(crate) fn code(
    io: &mut impl Bins,
    contexts: &mut Contexts,
    luma: u8,
    selected: u8,
) -> Result<u8> {
    if !io.bin(&mut contexts.chroma_mode, selected != luma)? {
        return Ok(luma);
    }
    let choices = candidates(luma);
    let index = choices.iter().position(|&m| m == selected).unwrap_or(0);
    Ok(choices[io.bypass(index as u32, 2)? as usize])
}

/// Refine chroma after luma selection, as in the other codec-local intra
/// searches: every legal chroma candidate, or with `rd_candidates` (more
/// than 0) the cheapest that many by a SATD proxy, receives full
/// reconstructed-distortion/bit costing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn choose(
    dsp: &Dsp,
    source: &Picture,
    picture: &Picture,
    grid: Option<&MotionGrid>,
    block: &Block,
    contexts: &Contexts,
    quantizers: &recon::Quantizers,
    lambda: f32,
    luma: u8,
    rd_candidates: usize,
    scratch: &mut [recon::QuantScratch; 2],
) -> (u8, f32) {
    // As `intra::choose`: a 64x64 CU is priced on its top-left quadrant.
    let cu = block;
    let block = &crate::transform_tree::units(cu, crate::transform_tree::MAX_TB_LOG2)
        .next()
        .expect("a CU has a transform unit");
    let references = PLANES.map(|plane| intra::References::new(picture, grid, plane, block, cu));
    let region = (block.x / 2, block.y / 2, block.size() / 2);
    let mut modes = nc_encoder::with_neighbours([luma], candidates(luma));
    if rd_candidates > 0 && modes.len() > rd_candidates {
        modes = shortlist(
            dsp,
            source,
            &references,
            region,
            (contexts, luma, lambda),
            modes,
        )
        .into_iter()
        .take(rd_candidates)
        .collect();
    }
    modes
        .into_iter()
        .map(|selected| {
            let pricing = (contexts, luma, quantizers, lambda);
            let cost = rd_cost(dsp, source, &references, region, selected, pricing, scratch);
            (selected, cost)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// The chroma planes, in coding order.
const PLANES: [PlaneIndex; 2] = [PlaneIndex::U, PlaneIndex::V];

/// The largest chroma block a search predicts (16x16: a 32x32 luma unit).
const SEARCH_SAMPLES: usize = 256;

/// Speed heuristic (`vvc.chroma_rd_candidates`, `VVenC`'s chroma SATD
/// pre-selection): `modes` ordered by the SATD of both planes' predictions
/// against the source plus √λ times the direction's bits, cheapest first, so
/// that only the first few get full RD. It skips the quantisation,
/// reconstruction and coefficient pricing of the later candidates; safe
/// where the prediction error ranks directions as RD would (the residual
/// left to code), which coarse chroma mostly does.
fn shortlist(
    dsp: &Dsp,
    source: &Picture,
    references: &[intra::References; 2],
    (x, y, n): (usize, usize, usize),
    (contexts, luma, lambda): (&Contexts, u8, f32),
    modes: Vec<u8>,
) -> Vec<u8> {
    let size = nc_dsp::BlockSize::new(n, n).expect("a square chroma block");
    let mut ranked: Vec<(u8, f32)> = modes
        .into_iter()
        .map(|selected| {
            let mut snapshot = contexts.clone();
            let mut bits = BitCost::default();
            code(&mut bits, &mut snapshot, luma, selected).expect("valid chroma mode");
            let satd: u32 = PLANES
                .iter()
                .zip(references)
                .map(|(&plane, references)| {
                    let mut buffer = [0u8; SEARCH_SAMPLES];
                    let pred = &mut buffer[..n * n];
                    references.predict(dsp, selected, pred);
                    let predicted = nc_common::PlaneView::new(pred, n, n, n).unwrap();
                    let original = source
                        .plane(plane)
                        .region(x as isize, y as isize, n, n)
                        .unwrap();
                    dsp.satd(size, &original, &predicted)
                })
                .sum();
            (selected, satd as f32 + lambda.sqrt() * bits.0)
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked.into_iter().map(|(mode, _)| mode).collect()
}

/// Full RD cost of chroma direction `selected`: both planes predicted,
/// quantised and reconstructed, their squared error plus λ times the
/// direction's, coded-block flags' and coefficients' bits.
fn rd_cost(
    dsp: &Dsp,
    source: &Picture,
    references: &[intra::References; 2],
    (x, y, n): (usize, usize, usize),
    selected: u8,
    (contexts, luma, quantizers, lambda): (&Contexts, u8, &recon::Quantizers, f32),
    scratch: &mut [recon::QuantScratch; 2],
) -> f32 {
    let mut snapshot = contexts.clone();
    let mut bits = BitCost::default();
    code(&mut bits, &mut snapshot, luma, selected).expect("valid chroma mode");
    let mut distortion = 0u64;
    let mut levels: [&[i32]; 2] = [&[]; 2];
    for ((index, plane), scratch) in PLANES.into_iter().enumerate().zip(scratch.iter_mut()) {
        let mut buffer = [0u8; SEARCH_SAMPLES];
        let pred = &mut buffer[..n * n];
        references[index].predict(dsp, selected, pred);
        let quant = quantizers.plane(plane, true);
        let quantized = recon::quantize(dsp, source, plane, x, y, pred, quant, None, scratch);
        let mut predicted = PlaneViewMut::new(pred, n, n, n).unwrap();
        recon::add(dsp, &mut predicted, quantized, quant.scaling());
        let original = source
            .plane(plane)
            .region(x as isize, y as isize, n, n)
            .unwrap();
        distortion += u64::from(
            dsp.variance(
                nc_dsp::BlockSize::new(n, n).unwrap(),
                &original,
                &predicted.as_view(),
            )
            .sse,
        );
        levels[index] = quantized;
    }
    let present = levels.each_ref().map(|c| c.iter().any(|&v| v != 0));
    bits.bin(&mut snapshot.cbf_cb, present[0])
        .expect("bit cost");
    bits.bin(&mut snapshot.cbf_cr[usize::from(present[0])], present[1])
        .expect("bit cost");
    for (coefficients, present) in levels.iter().zip(present) {
        if present {
            crate::coeff::emit(&mut bits, &mut snapshot.coeff, coefficients, n, 1);
        }
    }
    distortion as f32 + lambda * bits.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chroma_direction_can_differ_from_luma() {
        let dsp = Dsp::new();
        let mut source = Picture::new(64, 64, 0).unwrap();
        for plane in [PlaneIndex::U, PlaneIndex::V] {
            let mut view = source.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                view.row_mut(y).fill(40 + (y / 4 % 2) as u8 * 160);
            }
        }
        let block = Block::square(16, 16, 4);
        let choice = choose(
            &dsp,
            &source,
            &source,
            None,
            &block,
            &Contexts::new(22, false),
            &recon::Quantizers::defaults(22),
            5.7,
            intra::VERTICAL,
            0,
            &mut Default::default(),
        );
        assert_eq!(choice.0, intra::HORIZONTAL);
        // The SATD shortlist keeps the direction full RD picks.
        let shortlisted = choose(
            &dsp,
            &source,
            &source,
            None,
            &block,
            &Contexts::new(22, false),
            &recon::Quantizers::defaults(22),
            5.7,
            intra::VERTICAL,
            1,
            &mut Default::default(),
        );
        assert_eq!(shortlisted.0, intra::HORIZONTAL);
    }

    #[test]
    fn explicit_chroma_candidates_never_duplicate_derived() {
        for luma in 0..67 {
            let explicit = candidates(luma);
            assert!(!explicit.contains(&luma));
            let mut sorted = explicit;
            sorted.sort_unstable();
            assert!(sorted.windows(2).all(|pair| pair[0] != pair[1]));
        }
    }
}
