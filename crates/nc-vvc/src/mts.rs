//! H.266 multiple transform selection (MTS): the luma primary transform of
//! a coding unit (§8.7.4.2 `trTypeHor` / `trTypeVer`), its `mts_idx` syntax
//! (§7.3.11.5, §9.3.4.2) and the encoder's choice of it.
//!
//! With `sps_mts_enabled_flag` a luma transform block uses DST-VII or
//! DCT-VIII on a side:
//! - explicitly, by the CU's `mts_idx` (1 to 4: DST-VII or DCT-VIII on each
//!   side), when the SPS enables explicit MTS for its prediction type;
//! - implicitly (DST-VII on sides 4 to 16, DCT-II otherwise) for intra
//!   sub-partitions, and for every intra CU when explicit intra MTS is off.
//!
//! Chroma always uses the DCT-II; we code no SBT, LFNST, MIP or transform
//! skip, which would change these rules. Binarisation and contexts follow
//! `VVenC` `CABACWriter::mts_idx` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
use crate::{
    coeff::{Bins, BitCost},
    contexts::{Context, Contexts},
    recon,
};
use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut, decode::Result};
use nc_dsp::{
    BlockSize, Dsp, VvcQuantizer,
    VvcTransformKind::{self, Dct2, Dct8, Dst7},
};
use nc_encoder::Block;

/// The SPS's MTS (`sps_mts_enabled_flag` 1): which prediction types
/// signal `mts_idx` (`sps_explicit_mts_intra_enabled_flag`,
/// `sps_explicit_mts_inter_enabled_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Mts {
    pub intra: bool,
    pub inter: bool,
}

/// Bins of `mts_idx` (truncated unary, `cMax` 4), one context each.
pub(crate) const MTS_BINS: usize = 4;
/// The largest explicit `mts_idx`.
pub(crate) const MAX_INDEX: u8 = 4;
/// §7.3.11.5: `mts_idx` only for CUs no wider or taller than this.
const MAX_SIDE: usize = 32;
/// §7.3.11.11 `MtsZeroOutSigCoeffFlag`: luma coefficients beyond this
/// many columns or rows rule out `mts_idx`.
const ZERO_OUT_SIDE: usize = 16;
/// Implicit MTS uses DST-VII on sides from 4 to 16 (§8.7.4.2).
const IMPLICIT_SIDES: std::ops::RangeInclusive<usize> = 4..=16;

/// A luma transform block's primary transform rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Selection {
    /// DCT-II both ways (`mts_idx` 0 or absent).
    #[default]
    Dct2,
    /// DST-VII on sides from 4 to 16, DCT-II otherwise.
    Implicit,
    /// `mts_idx` 1 to 4 (Table 39, horizontal then vertical kind).
    Dst7Dst7,
    Dct8Dst7,
    Dst7Dct8,
    Dct8Dct8,
}

impl Selection {
    /// The (horizontal, vertical) kinds for a `width` x `height` block
    /// (§8.7.4.2, Table 39).
    pub(crate) fn kinds(self, (width, height): (usize, usize)) -> [VvcTransformKind; 2] {
        match self {
            Self::Dct2 => [Dct2; 2],
            Self::Implicit => [width, height].map(|n| {
                if IMPLICIT_SIDES.contains(&n) {
                    Dst7
                } else {
                    Dct2
                }
            }),
            Self::Dst7Dst7 => [Dst7, Dst7],
            Self::Dct8Dst7 => [Dct8, Dst7],
            Self::Dst7Dct8 => [Dst7, Dct8],
            Self::Dct8Dct8 => [Dct8, Dct8],
        }
    }

    /// The selection of explicit `mts_idx` `index` (1 to 4).
    fn explicit(index: u8) -> Self {
        match index {
            1 => Self::Dst7Dst7,
            2 => Self::Dct8Dst7,
            3 => Self::Dst7Dct8,
            _ => Self::Dct8Dct8,
        }
    }
}

/// The luma rule of a CU (§8.7.4.2 `implicitMtsEnabled`): `intra`, with
/// intra sub-partitions `isp`, coded `index` (its `mts_idx`, 0 if absent).
pub(crate) fn luma(tool: Option<Mts>, intra: bool, isp: bool, index: u8) -> Selection {
    let Some(mts) = tool else {
        return Selection::Dct2;
    };
    if isp || (intra && !mts.intra) {
        Selection::Implicit
    } else if index > 0 {
        Selection::explicit(index)
    } else {
        Selection::Dct2
    }
}

/// Whether CU `cu` (`intra` or inter, not sub-partitioned) codes
/// `mts_idx` after its transform units (§7.3.11.5), given its luma
/// `levels` (one transform unit, raster order; empty when none):
/// explicit MTS on for its prediction type, no side above 32, a coded
/// coefficient besides DC (`MtsDcOnly` 0) and none beyond the first 16
/// columns or rows (`MtsZeroOutSigCoeffFlag` 1; a coded coefficient
/// group holds a nonzero level, and the inferred ones are the last
/// group's and the DC's).
pub(crate) fn signalled(tool: Option<Mts>, intra: bool, cu: &Block, levels: &[i32]) -> bool {
    let width = cu.width();
    explicit(tool, intra, cu)
        && levels.iter().skip(1).any(|&v| v != 0)
        && levels
            .iter()
            .enumerate()
            .all(|(p, &v)| v == 0 || (p % width < ZERO_OUT_SIDE && p / width < ZERO_OUT_SIDE))
}

/// Whether CU `cu` (`intra` or inter, not sub-partitioned) may code
/// `mts_idx`: explicit MTS on for its prediction type, no side above 32.
pub(crate) fn explicit(tool: Option<Mts>, intra: bool, cu: &Block) -> bool {
    tool.is_some_and(|mts| if intra { mts.intra } else { mts.inter })
        && cu.width().max(cu.height()) <= MAX_SIDE
}

/// Read (or write or price, by `io`) `mts_idx` = `index`.
pub(crate) fn code(
    io: &mut impl Bins,
    contexts: &mut [Context; MTS_BINS],
    index: u8,
) -> Result<u8> {
    let mut coded = 0;
    for (bin, context) in contexts.iter_mut().enumerate() {
        if !io.bin(context, index > bin as u8)? {
            break;
        }
        coded += 1;
    }
    Ok(coded)
}

/// How explicit MTS is searched (`vvc.mts_*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Search {
    /// Whether intra CUs are searched (else they code the DCT-II).
    pub intra: bool,
    /// Nonzero levels a block's coded DCT-II needs before any other kind
    /// is tried.
    pub min_levels: usize,
    /// The widest and tallest CU searched.
    pub max_side: usize,
    /// The `mts_idx` values tried, 1 up to this.
    pub candidates: u8,
    /// Multiple of the mode-decision λ of the choice.
    pub lambda_scale: f32,
    /// Try only the index [`predicted_index`] picks (instead of the
    /// first `candidates`).
    pub predicted: bool,
}

impl Search {
    /// Whether CU `cu` (`intra` or inter) is searched: it may code
    /// `mts_idx` ([`explicit`]), and is within this search's sizes.
    pub(crate) fn tries(self, tool: Option<Mts>, intra: bool, cu: &Block) -> bool {
        explicit(tool, intra, cu)
            && (self.intra || !intra)
            && cu.width().max(cu.height()) <= self.max_side
    }
}

/// One explicit MTS decision's inputs: a luma transform block at `(x, y)`
/// of its CU's size, its prediction (row-major), quantiser and λ.
pub(crate) struct Candidate<'a> {
    pub source: &'a Picture,
    pub cu: &'a Block,
    pub intra: bool,
    pub prediction: &'a [u8],
    pub quant: recon::Quant,
    pub lambda: f32,
}

/// The transform-domain estimate's squared-error scale: a coefficient is
/// the residual's orthonormal transform times 128 / √(W·H) (the bases'
/// 64·√N gain over the forward's log2(W·H) + 5 bits of shifts), so the
/// sample-domain error is the coefficient-domain error times W·H / 2^14.
const TRANSFORM_DOMAIN_SHIFT: u32 = 14;
/// The estimate's bins per nonzero level (significance, greater-than-one,
/// sign and the parity / greater-than-three bins it usually pays), and per
/// doubling of a level's magnitude beyond one (its remainder).
const LEVEL_BITS: f32 = 4.;
const MAGNITUDE_BITS: f32 = 2.;
/// Coefficients dequantised at a time by the estimate.
const ESTIMATE_CHUNK: usize = 64;

/// The `mts_idx` of `block`: the DCT-II, or the explicit index (of the
/// first `search.candidates`) with the least transform-domain estimate when
/// that beats the DCT-II's estimate and then its exact RD cost (squared
/// error plus λ times the coefficient and `mts_idx` bins, plain quantiser).
/// The caller gates the blocks ([`Search::tries`], `min_levels`). The
/// contexts are copied, not adapted.
///
/// Speed: the estimates (forward transform and quantiser only) replace a
/// full RD check of every index; only an estimated winner is confirmed.
pub(crate) fn choose(
    dsp: &Dsp,
    block: &Candidate<'_>,
    tool: Option<Mts>,
    contexts: &Contexts,
    search: Search,
    scratch: &mut recon::QuantScratch,
) -> u8 {
    let Some(dct2) = estimate(dsp, block, tool, contexts, 0, scratch) else {
        return 0;
    };
    let mut best = (0, dct2);
    let tried = if search.predicted {
        let index = predicted_index(block);
        index..=index
    } else {
        1..=search.candidates.min(MAX_INDEX)
    };
    for index in tried {
        if let Some(cost) = estimate(dsp, block, tool, contexts, index, scratch)
            && cost < best.1
        {
            best = (index, cost);
        }
    }
    if best.0 == 0 {
        return 0;
    }
    match (
        cost(dsp, block, tool, contexts, best.0, scratch),
        cost(dsp, block, tool, contexts, 0, scratch),
    ) {
        (Some(chosen), Some(dct2)) if chosen < dct2 => best.0,
        (Some(_), None) => best.0,
        _ => 0,
    }
}

/// The `mts_idx` whose kinds suit where `block`'s residual energy lies:
/// the DST-VII's basis functions grow away from the block's left (top)
/// edge, the DCT-VIII's fall, so a residual stronger in the right (lower)
/// half takes the DST-VII on that side, else the DCT-VIII.
fn predicted_index(block: &Candidate<'_>) -> u8 {
    let cu = block.cu;
    let (width, height) = (cu.width(), cu.height());
    let source = block
        .source
        .plane(PlaneIndex::Y)
        .region(cu.x as isize, cu.y as isize, width, height)
        .unwrap();
    // Squared residual by half: [left, right], [top, bottom].
    let mut columns = [0u32; 2];
    let mut rows = [0u32; 2];
    for (y, predicted) in block.prediction.chunks_exact(width).enumerate() {
        for (x, (&p, &s)) in predicted.iter().zip(source.row(y)).enumerate() {
            let energy = u32::from(p.abs_diff(s)).pow(2);
            columns[usize::from(2 * x >= width)] += energy;
            rows[usize::from(2 * y >= height)] += energy;
        }
    }
    // Table 39: bit 0 of mts_idx − 1 selects the horizontal DCT-VIII, bit 1
    // the vertical.
    1 + u8::from(columns[0] > columns[1]) + 2 * u8::from(rows[0] > rows[1])
}

/// The transform-domain RD estimate of `block` at `mts_idx` `index` (up
/// to a constant of the block): the
/// dequantisation error of the plain levels (scaled to samples) plus λ
/// times approximate level bins and the `mts_idx` bins. `None` as for
/// [`cost`].
fn estimate(
    dsp: &Dsp,
    block: &Candidate<'_>,
    tool: Option<Mts>,
    contexts: &Contexts,
    index: u8,
    scratch: &mut recon::QuantScratch,
) -> Option<f32> {
    let cu = block.cu;
    let (width, height) = (cu.width(), cu.height());
    let quant = recon::Quant {
        transform: luma(tool, block.intra, false, index),
        ..block.quant
    };
    let (coefficients, levels) = recon::transform_levels(
        dsp,
        block.source,
        (cu.x, cu.y, width),
        block.prediction,
        quant,
        scratch,
    );
    let present = signalled(tool, block.intra, cu, levels);
    if (index > 0 && !present) || levels.iter().all(|&v| v == 0) {
        return None;
    }
    let quantizer = VvcQuantizer::rect(quant.qp as u8, width.ilog2() as u8, height.ilog2() as u8);
    // Only the coefficients with a nonzero level differ between the kinds:
    // a zero level's error is the coefficient itself, and every kind's
    // squared coefficients sum to about the same (the bases are close to
    // orthonormal, Parseval). So the estimate counts, for the nonzero
    // levels, their error less their coefficient's energy: a constant off
    // the true error, the same for every kind of the block. Dequantised a
    // chunk at a time, chunks without levels skipped.
    let mut error = 0i64;
    for (coefficients, levels) in coefficients
        .chunks(ESTIMATE_CHUNK)
        .zip(levels.chunks(ESTIMATE_CHUNK))
        .filter(|(_, levels)| levels.iter().any(|&v| v != 0))
    {
        let mut dequantized = [0; ESTIMATE_CHUNK];
        let dequantized = &mut dequantized[..levels.len()];
        dsp.vvc_dequantize(levels, &quantizer, dequantized);
        error += coefficients
            .iter()
            .zip(dequantized.iter())
            .zip(levels)
            .filter(|&(_, &level)| level != 0)
            .map(|((&c, &d), _)| i64::from(c - d).pow(2) - i64::from(c).pow(2))
            .sum::<i64>();
    }
    let distortion = (error * (width * height) as i64) >> TRANSFORM_DOMAIN_SHIFT;
    let mut bits = 0.;
    for &level in levels.iter().filter(|&&v| v != 0) {
        bits += LEVEL_BITS + MAGNITUDE_BITS * level.unsigned_abs().ilog2() as f32;
    }
    if present {
        let mut mts = contexts.mts_idx;
        let mut mts_bits = BitCost::default();
        code(&mut mts_bits, &mut mts, index).expect("valid mts_idx");
        bits += mts_bits.0;
    }
    Some(distortion as f32 + block.lambda * bits)
}

/// The exact RD cost of `block` at `mts_idx` `index`; `None` when its
/// levels could not signal it (an explicit index whose levels rule out
/// `mts_idx`, or DCT-II quantising to nothing).
fn cost(
    dsp: &Dsp,
    block: &Candidate<'_>,
    tool: Option<Mts>,
    contexts: &Contexts,
    index: u8,
    scratch: &mut recon::QuantScratch,
) -> Option<f32> {
    let cu = block.cu;
    let (width, height) = (cu.width(), cu.height());
    let quant = recon::Quant {
        transform: luma(tool, block.intra, false, index),
        ..block.quant
    };
    let levels = recon::quantize_rect(
        dsp,
        block.source,
        PlaneIndex::Y,
        (cu.x, cu.y, width),
        block.prediction,
        quant,
        None,
        scratch,
    );
    let present = signalled(tool, block.intra, cu, levels);
    if levels.is_empty() || (index > 0 && !present) {
        return None;
    }
    let mut coeff = contexts.coeff.clone();
    let mut bits = BitCost::default();
    crate::coeff::emit_rect(&mut bits, &mut coeff, levels, (width, height), 0);
    if present {
        let mut mts = contexts.mts_idx;
        code(&mut bits, &mut mts, index).expect("valid mts_idx");
    }
    let mut samples = [0u8; crate::transform_tree::MAX_TB_SAMPLES];
    let reconstructed = &mut samples[..width * height];
    reconstructed.copy_from_slice(block.prediction);
    let mut view = PlaneViewMut::new(reconstructed, width, width, height).unwrap();
    recon::add(dsp, &mut view, levels, quant.scaling());
    let original = block
        .source
        .plane(PlaneIndex::Y)
        .region(cu.x as isize, cu.y as isize, width, height)
        .unwrap();
    let reconstructed = PlaneView::new(&samples[..width * height], width, width, height).unwrap();
    let size = BlockSize::new(width, height).unwrap();
    let sse = dsp.variance(size, &original, &reconstructed).sse;
    Some(sse as f32 + block.lambda * bits.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_indices_pick_the_table_kinds() {
        let size = (8, 8);
        assert_eq!(Selection::explicit(1).kinds(size), [Dst7, Dst7]);
        assert_eq!(Selection::explicit(2).kinds(size), [Dct8, Dst7]);
        assert_eq!(Selection::explicit(3).kinds(size), [Dst7, Dct8]);
        assert_eq!(Selection::explicit(4).kinds(size), [Dct8, Dct8]);
        assert_eq!(Selection::Implicit.kinds((32, 4)), [Dct2, Dst7]);
        assert_eq!(Selection::Implicit.kinds((16, 16)), [Dst7, Dst7]);
    }

    #[test]
    fn the_rule_follows_the_sps_and_the_cu() {
        let explicit = Some(Mts {
            intra: true,
            inter: false,
        });
        assert_eq!(luma(None, true, true, 0), Selection::Dct2);
        assert_eq!(luma(explicit, true, true, 0), Selection::Implicit);
        assert_eq!(luma(explicit, true, false, 2), Selection::Dct8Dst7);
        assert_eq!(luma(explicit, false, false, 0), Selection::Dct2);
        assert_eq!(
            luma(Some(Mts::default()), true, false, 0),
            Selection::Implicit
        );
    }

    #[test]
    fn mts_idx_needs_an_ac_level_inside_the_zero_out() {
        let tool = Some(Mts {
            intra: true,
            inter: true,
        });
        let cu = Block::square(0, 0, 5);
        let mut levels = vec![0; 32 * 32];
        levels[0] = 3;
        assert!(!signalled(tool, true, &cu, &levels));
        levels[15 * 32 + 15] = 1;
        assert!(signalled(tool, false, &cu, &levels));
        levels[16] = 1;
        assert!(!signalled(tool, true, &cu, &levels));
        assert!(!signalled(tool, true, &Block::square(0, 0, 6), &[0, 1]));
    }

    #[test]
    fn mts_idx_is_truncated_unary() {
        let mut contexts = Contexts::new(32, true).mts_idx;
        for index in 0..=MAX_INDEX {
            let mut bits = BitCost::default();
            assert_eq!(code(&mut bits, &mut contexts, index).unwrap(), index);
            assert!(bits.0 > 0.);
        }
    }
}
