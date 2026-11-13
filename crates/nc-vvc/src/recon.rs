//! H.266 §8.7 DCT-II quantization and reconstruction through shared DSP.
//!
//! Normative arithmetic checked against `VVenC` Quant
//! (BSD-3-Clause-Clear; see `THIRD_PARTY.md`). Codec control is safe Rust; arithmetic uses shared DSP dispatch.
use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, HevcForwardMode, HevcForwardStaging, VvcQuantizer, VvcTransformKind};

/// Samples of the largest coded region (a 32x32 block's, a 64x64 one's).
const MAX_SAMPLES: usize = crate::transform_tree::MAX_TB_SAMPLES;
/// The side of a 64-point transform block (`nc_dsp::Dsp::vvc_forward64`).
const LARGE_SIDE: usize = 64;

/// HM's intra rounding offset, in 1/512 of a step (the default of
/// `vvc.intra_rounding`, as HEVC's `hevc.intra_rounding`).
pub(crate) const INTRA_ROUNDING_512THS: f32 = 171.;
/// Inter rounding offset (the default of `vvc.inter_rounding` and
/// `vvc.chroma_inter_rounding`): HEVC's 128, which keeps the small texture
/// coefficients a wider dead zone (HM 85) drops.
pub(crate) const INTER_ROUNDING_512THS: f32 = 128.;

/// One block's forward quantiser: the plane's QP and the encoder's rounding
/// offset (1/512 of a step; 256 rounds to nearest, smaller widens the dead
/// zone). Rounding is an encoder choice; the decoder only sees `qp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Quant {
    pub(crate) qp: u32,
    pub(crate) rounding_512ths: i32,
    /// The residual-SAD zero bound's multiple of the largest zero
    /// coefficient, in percent: 100 is exact (only blocks proven to quantise
    /// to zero skip the transform); more also drops near-zero residuals.
    pub(crate) zero_bound_percent: u32,
    /// [`shave_isolated_ones`]' minimum scan gap (0: never shave).
    pub(crate) shave_gap: usize,
    /// The 16- and 32-point forward DCT ([`EXACT_FORWARD`] or, with the
    /// trellis, [`HevcForwardMode::EXACT`]; or HEVC's staged rows:
    /// `vvc.fast_forward`).
    pub(crate) forward: HevcForwardMode,
    /// The block's primary transform (`mts.rs`): DCT-II unless MTS.
    pub(crate) transform: crate::mts::Selection,
}

impl Quant {
    /// The inverse of this quantiser's plain (not dependent) levels.
    pub(crate) fn scaling(self) -> Scaling {
        Scaling {
            qp: self.qp,
            dependent: false,
            transform: self.transform,
        }
    }
}

/// The default forward without the trellis: HEVC's exact passes, the
/// column pass skipping groups of eight horizontal frequencies whose
/// coefficients the bound proves quantise to zero (it writes zeros there).
/// The levels are those of the exact transform, and so is every
/// coefficient [`hide_signs`] reads: it only visits 4x4 groups holding a
/// nonzero level, which never lie in a skipped group. The trellis may raise
/// a zero level, so it keeps [`HevcForwardMode::EXACT`]. Crowd P frame
/// (x86 callgrind): 32-point forward 10.7M -> 9.1M, 16-point 5.2M -> 4.8M,
/// P -1.4%, key -1.6%; streams byte-identical.
pub(crate) const EXACT_FORWARD: HevcForwardMode = HevcForwardMode {
    staging: HevcForwardStaging::Exact,
    zero_columns: true,
    low_frequencies: false,
};

/// The staged forward of `vvc.fast_forward` 1: HEVC Default's
/// (`hevc.fast_forward16` / `hevc.fast_forward32` 1, exact columns that skip
/// proven-zero column groups).
pub(crate) const FAST_FORWARD: HevcForwardMode = HevcForwardMode {
    staging: HevcForwardStaging::Rows,
    zero_columns: true,
    low_frequencies: false,
};

/// [`Quant::zero_bound_percent`] of an exact bound.
pub(crate) const EXACT_ZERO_BOUND_PERCENT: u32 = 100;
/// The registered default of `vvc.pre_transform_zero_scale`: exact. VVC
/// Default sets HEVC Default's 2 (`DEFAULT_PARAMS`, screened in a bundle;
/// Jetson -2.5 / -2.1 / -1.9% cycles alone, x86 -1.2%).
pub(crate) const PRE_TRANSFORM_ZERO_SCALE: f32 = 1.;

/// The picture's forward quantisers by plane and prediction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Quantizers {
    /// Luma (slice) and chroma QP.
    pub(crate) qp: [u32; 2],
    /// Intra rounding (every plane).
    pub(crate) intra_rounding: i32,
    /// Inter rounding: luma, chroma.
    pub(crate) inter_rounding: [i32; 2],
    /// Inter blocks' [`Quant::zero_bound_percent`] (intra stays exact).
    pub(crate) inter_zero_bound_percent: u32,
    /// [`Quant::shave_gap`] of every block.
    pub(crate) shave_gap: usize,
    /// [`Quant::forward`] of every block.
    pub(crate) forward: HevcForwardMode,
    /// [`Quant::transform`] of intra luma blocks that are not sub-partitions:
    /// implicit MTS when the SPS enables MTS but not explicitly for intra.
    pub(crate) intra_luma: crate::mts::Selection,
}

impl Quantizers {
    /// λ for decisions that weigh chroma error against bits alone (chroma
    /// intra directions, chroma SAO): the luma `lambda` scaled by the chroma
    /// step's share, (`step_C` / `step_Y`)², as HM's chroma weight and the AV2
    /// backend's `chroma_rd_by_step`. λ follows the squared step, which
    /// doubles every 3 QP (§8.7.3); at the luma λ a finer chroma QP would buy
    /// detail that chroma's own decisions then trade away for bits.
    pub(crate) fn chroma_lambda(&self, lambda: f32) -> f32 {
        let [luma, chroma] = self.qp;
        lambda * 2f32.powf((chroma as f32 - luma as f32) / 3.)
    }

    pub(crate) fn plane(&self, plane: PlaneIndex, intra: bool) -> Quant {
        let chroma = usize::from(plane != PlaneIndex::Y);
        Quant {
            qp: self.qp[chroma],
            rounding_512ths: if intra {
                self.intra_rounding
            } else {
                self.inter_rounding[chroma]
            },
            zero_bound_percent: if intra {
                EXACT_ZERO_BOUND_PERCENT
            } else {
                self.inter_zero_bound_percent
            },
            shave_gap: self.shave_gap,
            forward: self.forward,
            transform: if intra && chroma == 0 {
                self.intra_luma
            } else {
                crate::mts::Selection::Dct2
            },
        }
    }
}

#[cfg(test)]
impl Quantizers {
    /// The default dead zones at `qp`, chroma at the luma QP.
    pub(crate) fn defaults(qp: u32) -> Self {
        Self {
            qp: [qp; 2],
            intra_rounding: INTRA_ROUNDING_512THS as i32,
            inter_rounding: [INTER_ROUNDING_512THS as i32; 2],
            inter_zero_bound_percent: EXACT_ZERO_BOUND_PERCENT,
            shave_gap: 0,
            forward: EXACT_FORWARD,
            intra_luma: crate::mts::Selection::Dct2,
        }
    }
}

/// The encoder forward modes exist for the 16- and 32-point DCTs only.
const STAGED_MIN_SIDE: usize = 16;

/// Largest DCT-II basis magnitude (the 8-bit `transMatrix` entries are <= 90).
const MAX_BASIS: u64 = 90;

/// An upper bound on every coefficient magnitude of the forward DCT of a
/// residual of `2^log2_area` samples whose absolute values sum to `abs_sum`:
/// each pass sums at most `MAX_BASIS` times its input magnitudes and
/// rounds, then shifts by `log2(width) - 1` and `log2(height) + 6` (8-bit);
/// the rounding adds at most 4.
fn forward_bound(log2_area: u32, abs_sum: u64) -> u64 {
    ((MAX_BASIS * MAX_BASIS * abs_sum) >> (log2_area + 5)) + 4
}

/// Working memory of [`quantize`] sized for the largest transform block,
/// owned by its caller and reused: the residual, coefficients and levels are
/// fully overwritten for each block, so no call zeroes or allocates them
/// (zeroing them per call was 1.4% of a crowd P frame, x86 callgrind).
pub(crate) struct QuantScratch {
    residual: [i16; MAX_SAMPLES],
    coefficients: [i32; MAX_SAMPLES],
    levels: [i32; MAX_SAMPLES],
}

impl Default for QuantScratch {
    fn default() -> Self {
        Self {
            residual: [0; MAX_SAMPLES],
            coefficients: [0; MAX_SAMPLES],
            levels: [0; MAX_SAMPLES],
        }
    }
}

/// Scratch contents carry no state between blocks.
impl std::fmt::Debug for QuantScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantScratch").finish_non_exhaustive()
    }
}

/// DCT-II uses the existing shared DSP implementation (same integer matrix and
/// pass scaling as HEVC for 8-bit square transforms). VVC QP extends to 63.
///
/// `search`, for the coded (winning) block only, re-chooses the levels by
/// rate and distortion: the trellis (before sign hiding), or dependent
/// quantisation's state search (instead of the plain quantiser).
///
/// Returns the levels in raster order (borrowed from `scratch`), or an empty
/// slice when every level is zero (most inter blocks at our rates).
#[allow(
    clippy::too_many_arguments,
    reason = "one block's complete quantiser inputs"
)]
pub(crate) fn quantize<'a>(
    dsp: &nc_dsp::Dsp,
    source: &Picture,
    plane: PlaneIndex,
    x: usize,
    y: usize,
    pred: &[u8],
    quant: Quant,
    search: Option<LevelSearch<'_>>,
    scratch: &'a mut QuantScratch,
) -> &'a [i32] {
    let side = pred.len().isqrt();
    quantize_rect(
        dsp,
        source,
        plane,
        (x, y, side),
        pred,
        quant,
        search,
        scratch,
    )
}

/// [`quantize`] of a block `width` samples wide (`pred.len() / width`
/// high) at `(x, y)`, its levels chosen by `search` when given.
#[allow(
    clippy::too_many_arguments,
    reason = "one block's complete quantiser inputs"
)]
pub(crate) fn quantize_rect<'a>(
    dsp: &nc_dsp::Dsp,
    source: &Picture,
    plane: PlaneIndex,
    (x, y, width): (usize, usize, usize),
    pred: &[u8],
    quant: Quant,
    search: Option<LevelSearch<'_>>,
    scratch: &'a mut QuantScratch,
) -> &'a [i32] {
    let height = pred.len() / width;
    let source = source
        .plane(plane)
        .region(x as isize, y as isize, width, height)
        .unwrap();
    let prediction = PlaneView::new(pred, width, width, height).unwrap();
    if quantize_into(dsp, &source, &prediction, quant, search, scratch) {
        &scratch.levels[..crate::transform_tree::coded_samples(width, height)]
    } else {
        &[]
    }
}

/// The forward coefficients of the block `width` samples wide
/// (`pred.len() / width` high) at `(x, y)` by `quant.transform`, and their
/// plain dead-zone levels (no sign hiding, no shaving), borrowed from
/// `scratch`: the inputs of transform-domain estimates (`mts::choose`).
pub(crate) fn transform_levels<'a>(
    dsp: &nc_dsp::Dsp,
    source: &Picture,
    (x, y, width): (usize, usize, usize),
    pred: &[u8],
    quant: Quant,
    scratch: &'a mut QuantScratch,
) -> (&'a [i32], &'a [i32]) {
    let height = pred.len() / width;
    let samples = width * height;
    let source = source
        .plane(PlaneIndex::Y)
        .region(x as isize, y as isize, width, height)
        .unwrap();
    let prediction = PlaneView::new(pred, width, width, height).unwrap();
    let residual = &mut scratch.residual[..samples];
    let size = BlockSize::new(width, height).unwrap();
    dsp.residual(size, &source, &prediction, residual, width);
    let coefficients = &mut scratch.coefficients[..samples];
    forward_selected(
        dsp,
        quant.transform,
        (width, height),
        residual,
        coefficients,
    );
    let quantizer = VvcQuantizer::rect_with_rounding(
        quant.qp as u8,
        width.ilog2() as u8,
        height.ilog2() as u8,
        quant.rounding_512ths,
    );
    let levels = &mut scratch.levels[..samples];
    dsp.vvc_quantize_from(coefficients, &quantizer, levels);
    (&scratch.coefficients[..samples], &scratch.levels[..samples])
}

/// Trellis quantisation of one block: the search's buffers and its inputs.
pub(crate) struct Trellis<'a> {
    pub(crate) search: &'a mut crate::trellis::TrellisQuantizer,
    pub(crate) block: crate::trellis::BlockTrellis<'a>,
}

/// How the coded (winning) block's levels are chosen beyond the plain
/// dead-zone quantiser.
pub(crate) enum LevelSearch<'a> {
    /// The trellis, then sign hiding.
    Trellis(Trellis<'a>),
    /// Dependent quantisation's four-state search (`dep_quant.rs`): no
    /// sign hiding, no shaving.
    Dependent(crate::dep_quant::DepQuant<'a>),
}

/// [`quantize_rect`] into `scratch.levels`; false when every level is zero.
fn quantize_into(
    dsp: &nc_dsp::Dsp,
    source: &PlaneView<'_>,
    prediction: &PlaneView<'_>,
    quant: Quant,
    search: Option<LevelSearch<'_>>,
    scratch: &mut QuantScratch,
) -> bool {
    let (width, height) = (prediction.width(), prediction.height());
    let quantizer = VvcQuantizer::rect_with_rounding(
        quant.qp as u8,
        width.ilog2() as u8,
        height.ilog2() as u8,
        quant.rounding_512ths,
    );
    // Most inter blocks are proven zero from the residual's SAD alone, before
    // any residual or transform work (HEVC's `pre_transform_zero_limit`);
    // narrow intra sub-partitions have no SAD kernel. A 64x64 block's
    // (approximate, `forward_large`) coefficients are bounded likewise: its
    // 2x2 means sum to a quarter of the SAD, give or take their rounding,
    // and its 32-point forward shifts two bits less.
    if let Some(size) = BlockSize::new(width, height) {
        let zero_limit = u64::from(quantizer.largest_zero_coefficient());
        let sad = u64::from(dsp.sad(size, source, prediction));
        if forward_bound(width.ilog2() + height.ilog2(), sad) * 100
            <= zero_limit * u64::from(quant.zero_bound_percent)
        {
            return false;
        }
    }
    let coded = if width == LARGE_SIDE {
        forward_large(dsp, (source, prediction), quant, &quantizer, scratch)
    } else {
        forward(dsp, (source, prediction), quant, &quantizer, scratch)
    };
    coded && choose_levels(dsp, quant, &quantizer, search, (width, height), scratch)
}

/// The encoder's forward 64-point DCT-II (`nc_dsp::Dsp::vvc_forward64`, an
/// approximation: it steers the levels, the reconstruction is exact) of the
/// 64x64 block into `scratch.coefficients` (its 32x32 coded region); false
/// when every coefficient provably quantises to zero. Only the DCT-II has
/// 64 points (§8.7.4.2: implicit MTS keeps it from 32; `mts_idx` needs
/// sides up to 32).
fn forward_large(
    dsp: &nc_dsp::Dsp,
    (source, prediction): (&PlaneView<'_>, &PlaneView<'_>),
    quant: Quant,
    quantizer: &VvcQuantizer,
    scratch: &mut QuantScratch,
) -> bool {
    debug_assert_eq!(
        quant.transform.kinds((LARGE_SIDE, LARGE_SIDE)),
        [VvcTransformKind::Dct2; 2]
    );
    dsp.vvc_forward64(
        source,
        prediction,
        quantizer.largest_zero_coefficient(),
        quant.forward,
        &mut scratch.coefficients,
    )
}

/// The forward transform of a block of up to 32x32 (or a narrow intra
/// sub-partition) into `scratch.coefficients`; false when every
/// coefficient provably quantises to zero.
fn forward(
    dsp: &nc_dsp::Dsp,
    (source, prediction): (&PlaneView<'_>, &PlaneView<'_>),
    quant: Quant,
    quantizer: &VvcQuantizer,
    scratch: &mut QuantScratch,
) -> bool {
    let (width, height) = (prediction.width(), prediction.height());
    let square = width == height;
    let samples = width * height;
    let residual = &mut scratch.residual[..samples];
    if let Some(size) = BlockSize::new(width, height) {
        dsp.residual(size, source, prediction, residual, width);
    } else {
        narrow_residual(source, prediction, residual);
    }
    let side = width;
    // Square blocks only (narrow intra sub-partitions have no HEVC size).
    let tx = || nc_dsp::HevcTransform::dct(side.ilog2() as u8).unwrap();
    // Exact: when every coefficient provably quantises to zero the forward
    // transform stops after its first pass and the levels stay zero.
    let coefficients = &mut scratch.coefficients[..samples];
    let limit = quantizer.largest_zero_coefficient();
    if quant.transform != crate::mts::Selection::Dct2 {
        forward_selected(
            dsp,
            quant.transform,
            (width, height),
            residual,
            coefficients,
        );
        true
    } else if !square {
        dsp.vvc_forward_dct(width, height, residual, coefficients);
        true
    } else if side >= STAGED_MIN_SIDE && quant.forward != HevcForwardMode::EXACT {
        dsp.hevc_forward_staged_unless_zero(
            tx(),
            residual,
            side,
            limit,
            quant.forward,
            coefficients,
        )
    } else {
        dsp.hevc_forward_unless_zero(tx(), residual, side, limit, coefficients)
    }
}

/// The levels of the coefficients in `scratch` of a transform block of
/// `sides` (their coded region, `transform_tree::coded_samples`) into
/// `scratch.levels`, chosen by `search` when given; false when every level
/// is zero.
fn choose_levels(
    dsp: &nc_dsp::Dsp,
    quant: Quant,
    quantizer: &VvcQuantizer,
    search: Option<LevelSearch<'_>>,
    sides: (usize, usize),
    scratch: &mut QuantScratch,
) -> bool {
    let size = (
        sides.0.min(crate::transform_tree::MAX_CODED_SIDE),
        sides.1.min(crate::transform_tree::MAX_CODED_SIDE),
    );
    let samples = size.0 * size.1;
    let coefficients = &scratch.coefficients[..samples];
    let levels = &mut scratch.levels[..samples];
    if let Some(LevelSearch::Dependent(search)) = &search {
        let nearest =
            VvcQuantizer::rect(quant.qp as u8, sides.0.ilog2() as u8, sides.1.ilog2() as u8);
        return crate::dep_quant::quantize(search, coefficients, &nearest, quant.qp, levels, sides);
    }
    if dsp.vvc_quantize_from(coefficients, quantizer, levels) == 0 {
        return false;
    }
    let mut coded = CodedGroups::of(levels, size);
    if quant.shave_gap > 0 && !shave_isolated_ones(levels, size, coded, quant.shave_gap) {
        return false;
    }
    if let Some(LevelSearch::Trellis(Trellis { search, block })) = search {
        let Some(last) = previous_nonzero(levels, size, coded, samples) else {
            return false;
        };
        let Some(chosen) = search.quantize(
            block,
            coefficients,
            quantizer,
            quant.qp,
            levels,
            sides,
            last,
        ) else {
            return false;
        };
        coded = chosen;
    }
    hide_signs(coefficients, levels, size, coded, quantizer);
    true
}

/// The residual of the one- or two-sample-wide luma blocks of intra
/// sub-partitions, which the dispatched kernels do not take (a few small
/// blocks per such CU).
#[inline(never)]
fn narrow_residual(source: &PlaneView<'_>, prediction: &PlaneView<'_>, residual: &mut [i16]) {
    let width = prediction.width();
    for (y, out) in residual.chunks_exact_mut(width).enumerate() {
        for ((out, &s), &p) in out.iter_mut().zip(source.row(y)).zip(prediction.row(y)) {
            *out = i16::from(s) - i16::from(p);
        }
    }
}

/// The forward transform of a block by `selection` (one resolving to a
/// square DCT-II, as implicit MTS on 32x32, takes the exact HEVC forward).
/// Out of line: the DCT-II blocks' path through [`quantize_into`] stays as
/// it was without MTS.
#[inline(never)]
fn forward_selected(
    dsp: &nc_dsp::Dsp,
    selection: crate::mts::Selection,
    size: (usize, usize),
    residual: &[i16],
    coefficients: &mut [i32],
) {
    let kinds = selection.kinds(size);
    if kinds == [VvcTransformKind::Dct2; 2] && size.0 == size.1 {
        let tx = nc_dsp::HevcTransform::dct(size.0.ilog2() as u8).unwrap();
        dsp.hevc_forward(tx, residual, size.0, coefficients);
    } else if size.0.min(size.1) < MIN_KERNEL_SIDE {
        // Intra sub-partitions 1 or 2 wide: DCT-II only (`isp::supported`).
        debug_assert_eq!(kinds, [VvcTransformKind::Dct2; 2]);
        dsp.vvc_forward_dct(size.0, size.1, residual, coefficients);
    } else {
        dsp.vvc_forward_primary(size, kinds, residual, coefficients);
    }
}

/// λ in squared quantiser steps: the HM λ, 0.57 · 2^((QP − 12) / 3), over
/// the squared step 2^((QP − 4) / 3) (§8.7.3), the same at every QP.
const LAMBDA_PER_SQUARED_STEP: f64 = 0.57 / 6.349_604_207_872_798; // 2^(8/3)
/// Approximate bins a one-step level change adds: a new ±1 pays its
/// significance, greater-than-one and sign bins; a larger level about one
/// more greater-than / remainder bin.
const NEW_LEVEL_BITS: f64 = 4.0;
const LARGER_LEVEL_BITS: f64 = 1.0;

/// Give every coefficient group that hides a sign (§7.3.11.11, more than
/// `SIGN_HIDING_THRESHOLD` scan positions between its first and last
/// nonzero level) the parity the decoder infers the sign from: the group's
/// level sum is odd exactly when its first (lowest scan position) level is
/// negative. Where it is not, the one-step change of a level inside the
/// span that adds the least squared error plus λ times its approximate
/// bins is made (HM's sign bit hiding adjustment, which also prices the
/// change's rate), never zeroing the
/// span's first or last level (that would move the span). `coefficients`
/// are the transform coefficients before quantisation, raster order.
pub(crate) fn hide_signs(
    coefficients: &[i32],
    levels: &mut [i32],
    (width, height): (usize, usize),
    coded: CodedGroups,
    quantizer: &VvcQuantizer,
) {
    use crate::{coeff::SIGN_HIDING_THRESHOLD, scan::GROUP_SIZE};
    let spans = crate::scan::rect_positions(width, height)
        .as_chunks::<GROUP_SIZE>()
        .0;
    for (index, span) in spans.iter().enumerate() {
        // Most groups of a coded block are empty.
        if !coded.contains(index) {
            continue;
        }
        let nonzero = |n: &usize| levels[span[*n]] != 0;
        let (Some(first), Some(last)) = (
            (0..GROUP_SIZE).find(nonzero),
            (0..GROUP_SIZE).rfind(nonzero),
        ) else {
            continue;
        };
        if last - first <= SIGN_HIDING_THRESHOLD {
            continue;
        }
        let sum: u32 = span.iter().map(|&p| levels[p].unsigned_abs()).sum();
        if (sum % 2 == 1) == (levels[span[first]] < 0) {
            continue;
        }
        // (added squared error in steps², position, change of magnitude)
        let mut best: Option<(f64, usize, i32)> = None;
        for (n, &p) in span.iter().enumerate().take(last + 1).skip(first) {
            let magnitude = levels[p].unsigned_abs();
            let level = f64::from(magnitude);
            let value = quantizer.magnitude_in_steps(coefficients[p]);
            let mut consider = |cost: f64, delta: i32| {
                if best.is_none_or(|(c, _, _)| cost < c) {
                    best = Some((cost, p, delta));
                }
            };
            let bits = if magnitude == 0 {
                NEW_LEVEL_BITS
            } else {
                LARGER_LEVEL_BITS
            };
            consider(
                2.0 * (level - value) + 1.0 + LAMBDA_PER_SQUARED_STEP * bits,
                1,
            );
            if magnitude > 1 || (magnitude == 1 && n != first && n != last) {
                let bits = if magnitude == 1 {
                    NEW_LEVEL_BITS
                } else {
                    LARGER_LEVEL_BITS
                };
                consider(
                    2.0 * (value - level) + 1.0 - LAMBDA_PER_SQUARED_STEP * bits,
                    -1,
                );
            }
        }
        let (_, p, delta) = best.expect("the span has at least one level");
        let magnitude = levels[p].unsigned_abs() as i32 + delta;
        // A new level takes its coefficient's sign (positive for zero).
        let negative = if levels[p] != 0 {
            levels[p] < 0
        } else {
            coefficients[p] < 0
        };
        levels[p] = if negative { -magnitude } else { magnitude };
    }
}

/// Drop trailing ±1 levels that stand at least `min_gap` scan positions
/// after the previous nonzero level, from the end of the block backwards
/// (SVT-AV1's `shave_coeff`, as HEVC's `hevc.shave_gap`: an isolated
/// high-frequency ±1 pays its last-position and significance bins for
/// almost no distortion; it is mostly noise). The block's first nonzero
/// level is always kept. `levels` are raster order, scanned in the §9.3
/// grouped diagonal order the syntax codes. Returns whether any level
/// remains.
pub(crate) fn shave_isolated_ones(
    levels: &mut [i32],
    size: (usize, usize),
    coded: CodedGroups,
    min_gap: usize,
) -> bool {
    let positions = crate::scan::rect_positions(size.0, size.1);
    let Some(mut last) = previous_nonzero(levels, size, coded, positions.len()) else {
        return false;
    };
    while let Some(previous) = previous_nonzero(levels, size, coded, last) {
        if levels[positions[last]].abs() != 1 || last - previous < min_gap {
            break;
        }
        levels[positions[last]] = 0;
        last = previous;
    }
    true
}

/// The coefficient groups (by scan order index, one bit each) that may hold
/// a nonzero level. Built once per block from each group's contiguous
/// raster rows, which is much cheaper than walking scan positions; a set bit
/// may go stale when a level is zeroed (walks then simply find nothing).
#[derive(Clone, Copy)]
pub(crate) struct CodedGroups(u64);

/// The side of a coefficient group of a block at least 4 a side, and the
/// widest coded block.
const GROUP_SIDE: usize = 4;
const MAX_SIDE: usize = 32;

impl CodedGroups {
    pub(crate) fn of(levels: &[i32], (width, height): (usize, usize)) -> Self {
        if width < GROUP_SIDE || height < GROUP_SIDE {
            return Self::of_narrow(levels, (width, height));
        }
        let columns = width / GROUP_SIDE;
        // Each band of four rows ORs its rows together (contiguous, so it
        // vectorises, with no early exit to defeat that), then each group of
        // the band tests its four columns; the raster group bits are then put
        // in coding order.
        let mut raster = 0u64;
        for (band, rows) in levels[..width * height]
            .chunks_exact(GROUP_SIDE * width)
            .enumerate()
        {
            let mut any = [0i32; MAX_SIDE];
            for row in rows.chunks_exact(width) {
                for (any, &level) in any.iter_mut().zip(row) {
                    *any |= level;
                }
            }
            for (column, group) in any[..width].as_chunks::<GROUP_SIDE>().0.iter().enumerate() {
                let coded = group.iter().fold(0, |a, &v| a | v) != 0;
                raster |= u64::from(coded) << (band * columns + column);
            }
        }
        let mask = crate::scan::rect_groups(width, height)
            .iter()
            .enumerate()
            .fold(0, |mask, (index, &group)| {
                mask | (raster >> group & 1) << index
            });
        Self(mask)
    }

    /// [`Self::of`] for the one- or two-sample-wide blocks of intra
    /// sub-partitions, from the scan (few and small).
    fn of_narrow(levels: &[i32], (width, height): (usize, usize)) -> Self {
        let positions = crate::scan::rect_positions(width, height);
        let mask = positions
            .as_chunks::<{ crate::scan::GROUP_SIZE }>()
            .0
            .iter()
            .enumerate()
            .fold(0, |mask, (index, group)| {
                mask | u64::from(group.iter().any(|&p| levels[p] != 0)) << index
            });
        Self(mask)
    }

    /// The groups whose bits are set in `mask`, one per group in coding
    /// (scan) order.
    pub(crate) fn from_scan_mask(mask: u64) -> Self {
        Self(mask)
    }

    fn contains(self, index: usize) -> bool {
        self.0 >> index & 1 != 0
    }

    /// The highest group index below `end` (a group index) that may be coded.
    fn last_below(self, end: usize) -> Option<usize> {
        let below = if end >= 64 {
            self.0
        } else {
            self.0 & ((1 << end) - 1)
        };
        (below != 0).then(|| 63 - below.leading_zeros() as usize)
    }
}

/// The highest scan index below `end` whose level is nonzero; groups not in
/// `coded` are skipped whole.
fn previous_nonzero(
    levels: &[i32],
    (width, height): (usize, usize),
    coded: CodedGroups,
    end: usize,
) -> Option<usize> {
    use crate::scan::GROUP_SIZE;
    let positions = crate::scan::rect_positions(width, height);
    let mut end = end;
    while let Some(index) = coded.last_below(end.div_ceil(GROUP_SIZE)) {
        let start = index * GROUP_SIZE;
        let stop = end.min(start + GROUP_SIZE);
        if let Some(i) = (start..stop).rev().find(|&i| levels[positions[i]] != 0) {
            return Some(i);
        }
        end = start;
    }
    None
}

/// The inverse quantiser of a picture's coded levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Scaling {
    pub qp: u32,
    /// `sh_dep_quant_used_flag`: levels pass through the quantiser state
    /// machine (§8.7.3), see [`add_dependent`].
    pub dependent: bool,
    /// The luma primary transform rule (`mts.rs`; chroma: DCT-II).
    pub transform: crate::mts::Selection,
}

/// Store the reconstruction of the block `width` samples wide
/// (`pred.len() / width` high) at `(x, y)`: `pred` plus the inverse
/// transform of `levels`.
pub(crate) fn store(
    dsp: &nc_dsp::Dsp,
    picture: &mut Picture,
    plane: PlaneIndex,
    (x, y, width): (usize, usize, usize),
    pred: &[u8],
    levels: &[i32],
    scaling: Scaling,
) {
    let height = pred.len() / width;
    let mut view = picture
        .plane_mut(plane)
        .region_mut(x as isize, y as isize, width, height)
        .unwrap();
    copy_prediction(&mut view, pred, width);
    add(dsp, &mut view, levels, scaling);
}

/// Copy `pred` (`width` samples per row) into `view`, by block width so
/// each row is a fixed-size move: a `memcpy` call per row was 1.3% of a
/// Jetson Default encode (gdb samples, CBR 600).
fn copy_prediction(view: &mut PlaneViewMut<'_>, pred: &[u8], width: usize) {
    match width {
        4 => copy_rows::<4>(view, pred),
        8 => copy_rows::<8>(view, pred),
        16 => copy_rows::<16>(view, pred),
        32 => copy_rows::<32>(view, pred),
        64 => copy_rows::<64>(view, pred),
        _ => {
            for (row, samples) in pred.chunks_exact(width).enumerate() {
                view.row_mut(row).copy_from_slice(samples);
            }
        }
    }
}

/// [`copy_prediction`] of rows `W` samples wide.
fn copy_rows<const W: usize>(view: &mut PlaneViewMut<'_>, pred: &[u8]) {
    for (row, samples) in pred.as_chunks::<W>().0.iter().enumerate() {
        let target: &mut [u8; W] = (&mut view.row_mut(row)[..W]).try_into().unwrap();
        *target = *samples;
    }
}

/// [`add`] of a block of other kinds (MTS): its levels, or their
/// dependent reconstruction indices. Out of line, as [`forward_selected`].
#[inline(never)]
fn add_primary(
    dsp: &nc_dsp::Dsp,
    prediction: &mut PlaneViewMut<'_>,
    levels: &[i32],
    scaling: Scaling,
    kinds: [VvcTransformKind; 2],
) {
    if levels.iter().all(|&level| level == 0) {
        return;
    }
    add_rect_scaled(
        dsp,
        prediction,
        levels,
        scaling.qp,
        scaling.dependent,
        kinds,
    );
}

/// [`add`] of dependently quantised DCT-II `levels` (§8.7.3): each level `k`
/// reconstructs at index 2k in quantiser states 0 and 1 and 2k − sgn(k)
/// in states 2 and 3 (states from 0 at the last level, along the reverse
/// scan, `coeff::next_state`), scaled at QP + 1 with one more shift.
fn add_dependent(dsp: &nc_dsp::Dsp, prediction: &mut PlaneViewMut<'_>, levels: &[i32], qp: u32) {
    if levels.iter().all(|&level| level == 0) {
        return;
    }
    let (width, height) = (prediction.width(), prediction.height());
    if width != height {
        add_rect_scaled(
            dsp,
            prediction,
            levels,
            qp,
            true,
            [VvcTransformKind::Dct2; 2],
        );
        return;
    }
    let log2 = width.ilog2() as u8;
    let quantizer = VvcQuantizer::rect_dependent(qp as u8, log2, log2);
    let dependent = Some((width, height));
    match levels.len() {
        16 => add_sized::<16>(dsp, prediction, levels, &quantizer, dependent),
        64 => add_sized::<64>(dsp, prediction, levels, &quantizer, dependent),
        256 => add_sized::<256>(dsp, prediction, levels, &quantizer, dependent),
        MAX_SAMPLES => add_sized::<MAX_SAMPLES>(dsp, prediction, levels, &quantizer, dependent),
        other => unreachable!("{other} levels: not a VVC transform block"),
    }
}

/// The reconstruction indices of dependently quantised `levels` of a
/// `width` x `height` block (raster order, at most `N` samples), as
/// [`add_dependent`] scales. Sized to the block: zeroing the largest
/// block's indices for every 4x4 block cost more than its transform.
fn dependent_indices<const N: usize>(levels: &[i32], (width, height): (usize, usize)) -> [i32; N] {
    let scan = crate::scan::rect_positions(width, height);
    let mut indices = [0; N];
    let Some(last) = scan.iter().rposition(|&p| levels[p] != 0) else {
        return indices;
    };
    let mut state = 0;
    for &p in scan[..=last].iter().rev() {
        let k = levels[p];
        indices[p] = 2 * k - k.signum() * i32::from(state > 1);
        state = crate::coeff::next_state(state, k.unsigned_abs());
    }
    indices
}

/// Add the inverse transform of `levels` (scaled by `scaling`) to the
/// predicted block `prediction`.
pub(crate) fn add(
    dsp: &nc_dsp::Dsp,
    prediction: &mut PlaneViewMut<'_>,
    levels: &[i32],
    scaling: Scaling,
) {
    if prediction.width() == LARGE_SIDE {
        add_large(dsp, prediction, levels, scaling);
    } else if scaling.transform != crate::mts::Selection::Dct2 {
        let kinds = scaling
            .transform
            .kinds((prediction.width(), prediction.height()));
        add_primary(dsp, prediction, levels, scaling, kinds);
    } else if scaling.dependent {
        add_dependent(dsp, prediction, levels, scaling.qp);
    } else {
        add_plain(dsp, prediction, levels, scaling.qp);
    }
}

/// [`add`] of a 64x64 block (a DCT-II: [`forward_large`]) from the levels
/// of its 32x32 coded region. Out of line, as [`forward_selected`].
#[inline(never)]
fn add_large(
    dsp: &nc_dsp::Dsp,
    prediction: &mut PlaneViewMut<'_>,
    levels: &[i32],
    scaling: Scaling,
) {
    const AREA: usize = LARGE_SIDE * LARGE_SIDE;
    if levels.iter().all(|&level| level == 0) {
        return;
    }
    let coded = crate::transform_tree::MAX_CODED_SIDE;
    let indices;
    let levels = if scaling.dependent {
        indices = dependent_indices::<MAX_SAMPLES>(levels, (coded, coded));
        &indices[..]
    } else {
        levels
    };
    let mut residual = [0; AREA];
    dsp.vvc_inverse_dct64(levels, scaling.qp as u8, scaling.dependent, &mut residual);
    dsp.add_residual(
        BlockSize::new(LARGE_SIDE, LARGE_SIDE).unwrap(),
        prediction,
        &residual,
        LARGE_SIDE,
    );
}

/// [`add`] of the DCT-II blocks' plain levels through the shared
/// residual-add kernel.
fn add_plain(dsp: &nc_dsp::Dsp, prediction: &mut PlaneViewMut<'_>, levels: &[i32], qp: u32) {
    if levels.iter().all(|&level| level == 0) {
        return; // no residual: the reconstruction is the prediction
    }
    if prediction.width() != prediction.height() {
        add_rect_scaled(
            dsp,
            prediction,
            levels,
            qp,
            false,
            [VvcTransformKind::Dct2; 2],
        );
        return;
    }
    // Scratch sized to the block: zeroing a 32x32 buffer for every 8x8
    // block cost more than the inverse transform itself.
    let side = prediction.width().ilog2() as u8;
    let quantizer = VvcQuantizer::new(qp as u8, side);
    match levels.len() {
        16 => add_sized::<16>(dsp, prediction, levels, &quantizer, None),
        64 => add_sized::<64>(dsp, prediction, levels, &quantizer, None),
        256 => add_sized::<256>(dsp, prediction, levels, &quantizer, None),
        MAX_SAMPLES => add_sized::<MAX_SAMPLES>(dsp, prediction, levels, &quantizer, None),
        other => unreachable!("{other} levels: not a VVC transform block"),
    }
}

/// [`add`] for square DCT-II blocks of `N` samples: inverse scaling by
/// `quantizer` and transform, then the residual added to the prediction.
/// `dependent` (the block's size) when `levels` are dependently quantised
/// levels, to be mapped to their reconstruction indices first.
fn add_sized<const N: usize>(
    dsp: &nc_dsp::Dsp,
    prediction: &mut PlaneViewMut<'_>,
    levels: &[i32],
    quantizer: &VvcQuantizer,
    dependent: Option<(usize, usize)>,
) {
    let side = N.isqrt();
    let indices;
    let levels = match dependent {
        Some(size) => {
            indices = dependent_indices::<N>(levels, size);
            &indices[..]
        }
        None => levels,
    };
    if levels[1..].iter().all(|&level| level == 0) {
        add_constant(prediction, dc_residual(dsp, levels[0], quantizer));
        return;
    }
    let mut scaled = [0; N];
    dsp.vvc_dequantize(levels, quantizer, &mut scaled);
    let tx = nc_dsp::HevcTransform::dct(side.ilog2() as u8).unwrap();
    let mut wide = [0; N];
    dsp.hevc_inverse(tx, &scaled, &mut wide);
    // The inverse 8-bit DCT-II stays within ±23040 even at 32x32: the
    // first pass clamps to 16 bits, then at most 32 taps of magnitude 90
    // accumulate before a 12-bit shift. This fits the shared add contract.
    let residual: [i16; N] = std::array::from_fn(|i| wide[i] as i16);
    dsp.add_residual(
        BlockSize::new(side, side).unwrap(),
        prediction,
        &residual,
        side,
    );
}

/// The DCT-II's DC basis value (every tap of its first row).
const DCT_DC_TAP: i32 = 64;
/// §8.7.4.1: the first (vertical) stage's shift, and the 16-bit range its
/// outputs are clipped to.
const FIRST_STAGE_SHIFT: u32 = 7;
const STAGE_RANGE: std::ops::RangeInclusive<i32> = -(1 << 15)..=(1 << 15) - 1;
/// §8.7.4.1: the second stage's shift at 8 bits (`bdShift` = 20 − 8).
const RESIDUAL_SHIFT: u32 = 12;

/// The residual of a DCT-II block whose only non-zero level is the DC
/// `level`: the same at every sample, since both one-dimensional stages see
/// a single DC input (§8.7.4.1, two roundings and the 16-bit clip between).
fn dc_residual(dsp: &nc_dsp::Dsp, level: i32, quantizer: &VvcQuantizer) -> i32 {
    // The scaling of one level does not depend on its neighbours: scale a
    // 4x4 block holding only the DC.
    let mut levels = [0; 16];
    levels[0] = level;
    let mut scaled = [0; 16];
    dsp.vvc_dequantize(&levels, quantizer, &mut scaled);
    let first = ((DCT_DC_TAP * scaled[0] + (1 << (FIRST_STAGE_SHIFT - 1))) >> FIRST_STAGE_SHIFT)
        .clamp(*STAGE_RANGE.start(), *STAGE_RANGE.end());
    (DCT_DC_TAP * first + (1 << (RESIDUAL_SHIFT - 1))) >> RESIDUAL_SHIFT
}

/// Add the constant residual `residual` to every sample of `prediction`.
fn add_constant(prediction: &mut PlaneViewMut<'_>, residual: i32) {
    if residual == 0 {
        return;
    }
    for y in 0..prediction.height() {
        for sample in prediction.row_mut(y) {
            *sample = (i32::from(*sample) + residual).clamp(0, 255) as u8;
        }
    }
}

/// [`add`] through the normative inverse scaling and primary transform of
/// `kinds` (§8.7.3, §8.7.4): rectangular DCT-II blocks and MTS, of plain
/// levels or of dependent reconstruction indices; then the residual added.
fn add_rect_scaled(
    dsp: &nc_dsp::Dsp,
    prediction: &mut PlaneViewMut<'_>,
    levels: &[i32],
    qp: u32,
    dependent: bool,
    kinds: [VvcTransformKind; 2],
) {
    // Buffers sized to the block, as `add` does for squares.
    let scaled = (qp, dependent, kinds);
    let samples = prediction.width() * prediction.height();
    if samples <= SMALL_SAMPLES {
        add_rect_sized::<SMALL_SAMPLES>(dsp, prediction, levels, scaled);
    } else if samples <= MEDIUM_SAMPLES {
        add_rect_sized::<MEDIUM_SAMPLES>(dsp, prediction, levels, scaled);
    } else {
        add_rect_sized::<MAX_SAMPLES>(dsp, prediction, levels, scaled);
    }
}

/// Buffer sizes of [`add_rect_sized`] below the largest block's.
const SMALL_SAMPLES: usize = 64;
const MEDIUM_SAMPLES: usize = 256;

/// [`add_rect_scaled`] with buffers of `N` samples, at least the block's.
/// The narrowest side the dispatched transform and residual kernels take.
const MIN_KERNEL_SIDE: usize = 4;

fn add_rect_sized<const N: usize>(
    dsp: &nc_dsp::Dsp,
    prediction: &mut PlaneViewMut<'_>,
    levels: &[i32],
    (qp, dependent, kinds): (u32, bool, [VvcTransformKind; 2]),
) {
    let (width, height) = (prediction.width(), prediction.height());
    let samples = width * height;
    let indices;
    let levels = if dependent {
        indices = dependent_indices::<N>(levels, (width, height));
        &indices[..samples]
    } else {
        levels
    };
    let mut wide = [0; N];
    dsp.vvc_inverse_primary(
        (width, height),
        kinds,
        levels,
        qp as u8,
        dependent,
        &mut wide[..samples],
    );
    let Some(size) = BlockSize::new(width, height) else {
        // Intra sub-partitions 1 or 2 wide: a few samples, in scalar.
        for (y, residual) in wide[..samples].chunks_exact(width).enumerate() {
            for (sample, &r) in prediction.row_mut(y).iter_mut().zip(residual) {
                *sample = (i32::from(*sample) + r).clamp(0, 255) as u8;
            }
        }
        return;
    };
    // Within ±23040 as for squares (`add_sized`).
    let mut residual = [0i16; N];
    for (out, &value) in residual.iter_mut().zip(&wide[..samples]) {
        *out = value as i16;
    }
    dsp.add_residual(size, prediction, &residual[..samples], width);
}

#[cfg(test)]
mod tests {
    use super::{CodedGroups, dc_residual, shave_isolated_ones};
    use nc_dsp::{HevcTransform, VvcQuantizer};

    /// The DC-only shortcut equals the full inverse scaling and transform
    /// at every sample, for every square DCT-II size, QP and both
    /// quantisers, over a sweep of DC levels.
    #[test]
    fn dc_residual_matches_the_inverse() {
        let dsp = nc_dsp::Dsp::new();
        for log2 in 2..=5u8 {
            let n = 1usize << log2;
            let transform = HevcTransform::dct(log2).unwrap();
            for qp in (0..=63u8).step_by(7) {
                for dependent in [false, true] {
                    let quantizer = if dependent {
                        VvcQuantizer::rect_dependent(qp, log2, log2)
                    } else {
                        VvcQuantizer::new(qp, log2)
                    };
                    for level in (-40_000..=40_000).step_by(373).chain([-1, 1, 2]) {
                        let mut levels = vec![0; n * n];
                        levels[0] = level;
                        let mut scaled = vec![0; n * n];
                        dsp.vvc_dequantize(&levels, &quantizer, &mut scaled);
                        let mut residual = vec![0; n * n];
                        dsp.hevc_inverse(transform, &scaled, &mut residual);
                        let constant = dc_residual(&dsp, level, &quantizer);
                        assert!(
                            residual.iter().all(|&r| r == constant),
                            "{n}x{n} qp {qp} dependent {dependent} level {level}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn coded_groups_mark_exactly_the_groups_with_levels() {
        let sizes = [
            (4, 4),
            (8, 4),
            (4, 32),
            (16, 8),
            (32, 32),
            (2, 8),
            (16, 2),
            (1, 32),
        ];
        for &(width, height) in &sizes {
            let groups = crate::scan::rect_groups(width, height);
            let (gw, gh) = crate::scan::group_shape(width, height);
            for seed in 0..40u32 {
                let mut state = seed.wrapping_mul(2_654_435_761) | 1;
                let levels: Vec<i32> = (0..width * height)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        if state % 23 == 0 {
                            (state >> 8) as i32 % 5 - 2
                        } else {
                            0
                        }
                    })
                    .collect();
                let coded = CodedGroups::of(&levels, (width, height));
                for (index, &group) in groups.iter().enumerate() {
                    let (x, y) = (group % (width / gw) * gw, group / (width / gw) * gh);
                    let nonzero = (y..y + gh)
                        .any(|row| levels[row * width + x..][..gw].iter().any(|&v| v != 0));
                    assert_eq!(coded.contains(index), nonzero, "{width}x{height} {seed}");
                }
            }
        }
    }

    #[test]
    fn isolated_trailing_ones_are_dropped_and_the_first_level_kept() {
        let positions = crate::scan::positions(8);
        let mut levels = [0i32; 64];
        levels[positions[0]] = 3;
        levels[positions[20]] = -1;
        levels[positions[40]] = 1;
        let coded = CodedGroups::of(&levels, (8, 8));
        assert!(shave_isolated_ones(&mut levels, (8, 8), coded, 8));
        assert_eq!(levels.iter().filter(|&&v| v != 0).count(), 1);
        assert_eq!(levels[positions[0]], 3);

        // A ±1 close to its predecessor, or a larger level, stops the shave.
        let mut levels = [0i32; 64];
        levels[positions[0]] = 1;
        levels[positions[5]] = 2;
        levels[positions[30]] = 1;
        levels[positions[33]] = -1;
        let coded = CodedGroups::of(&levels, (8, 8));
        assert!(shave_isolated_ones(&mut levels, (8, 8), coded, 8));
        assert_eq!(levels.iter().filter(|&&v| v != 0).count(), 4);

        let mut lone = [0i32; 64];
        lone[positions[63]] = 1;
        let coded = CodedGroups::of(&lone, (8, 8));
        assert!(shave_isolated_ones(&mut lone, (8, 8), coded, 8));
        assert_eq!(lone[positions[63]], 1);
    }
}
