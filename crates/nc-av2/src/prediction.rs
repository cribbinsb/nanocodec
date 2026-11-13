//! AV2 single-reference translational prediction of square blocks from 4×4
//! to 64×64 with the block's interpolation filter (the frame's, or the
//! block's own in a switchable frame), on nc-dsp's shared dispatched kernels
//! (no AVM filter loop is copied).
//!
//! - `EIGHTTAP_REGULAR`: AVM v1.0.0's `av2_sub_pel_filters_8` /
//!   `av2_sub_pel_filters_4` (`av2/common/filter.h`) equal AV1's regular
//!   8-tap and 4-tap filters, and `av2_get_interp_filter_params_with_block_size`
//!   selects the 4-tap variant for a direction whose block edge is at most
//!   four samples, exactly AV1's rule. Non-compound 8-bit rounding
//!   (`ROUND0_BITS` 3, then 11) matches too, since AVM's high-bit-depth
//!   offsets cancel exactly: this is `nc_dsp::Filter::Av1`.
//! - `EIGHTTAP_SMOOTH` and `MULTITAP_SHARP`: `av2_sub_pel_filters_8smooth`,
//!   `av2_sub_pel_filters_4smooth` and `av2_sub_pel_filters_8sharp` equal
//!   AV1's smooth and sharp filters, and `av2_interp_4tap` selects the 4-tap
//!   smooth filter for smooth and the 4-tap regular one for sharp on short
//!   edges, again AV1's rule: `nc_dsp::Filter::Av1Smooth` / `Av1Sharp`.
//! - `BILINEAR`: AVM's Q7 bilinear taps at the even 1/16 positions that
//!   quarter-pel luma and eighth-pel 4:2:0 chroma reach are multiples of
//!   sixteen, so the first shift of three loses nothing and the result is
//!   H.264's Q3×Q3 bilinear interpolation with one final rounding:
//!   `nc_dsp::Filter::H264Chroma` at eighth positions.
//!
//! Compound blocks (skip mode's two vectors on the one reference,
//! [`predict_compound`]) predict each vector at compound precision
//! (`ROUND0_BITS` 3, `COMPOUND_ROUND1_BITS` 7) and average the two with
//! one rounding, AVM's `av2_highbd_dist_wtd_comp_avg` at equal weights
//! with its offsets cancelled and dav2d's `prep` + `avg`: exactly AV1's
//! §7.11.3.4 / §7.11.3.15 average, so nc-dsp's AV1 compound kernels apply
//! (`Dsp::av1_predict_compound`, `Dsp::average_compound`).
use nc_common::decode::{DecodeError, Result};
use nc_common::{Plane, PlaneView, PlaneViewMut};
use nc_dsp::av1_mc::convolve::Av1McAxis;
use nc_dsp::av1_mc::filter::{AV1_FILTER_BEFORE, AV1_FILTER_TAPS, Av1Filter, Av1Filters};
use nc_dsp::{BlockSize, Dsp, Filter};

/// The interpolation filters of AV2 (AVM `InterpFilter`, without the
/// encoder-only `MULTITAP_SHARP2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpFilter {
    /// `EIGHTTAP_REGULAR`.
    Regular,
    /// `EIGHTTAP_SMOOTH`.
    Smooth,
    /// `MULTITAP_SHARP`.
    Sharp,
    /// `BILINEAR` (frame level only: not switchable).
    Bilinear,
}

impl InterpFilter {
    /// AVM `InterpFilter` value of `EIGHTTAP_REGULAR`.
    const REGULAR_CODE: u32 = 0;
    /// AVM `InterpFilter` value of `EIGHTTAP_SMOOTH`.
    const SMOOTH_CODE: u32 = 1;
    /// AVM `InterpFilter` value of `MULTITAP_SHARP`.
    const SHARP_CODE: u32 = 2;
    /// AVM `InterpFilter` value of `BILINEAR`.
    const BILINEAR_CODE: u32 = 3;
    /// The filters a block of a switchable frame may code, in symbol order
    /// (AVM `SWITCHABLE_FILTERS`).
    pub const SWITCHABLE: [Self; 3] = [Self::Regular, Self::Smooth, Self::Sharp];

    /// The value coded in the frame header, and the per-block symbol of a
    /// switchable filter.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Regular => Self::REGULAR_CODE,
            Self::Smooth => Self::SMOOTH_CODE,
            Self::Sharp => Self::SHARP_CODE,
            Self::Bilinear => Self::BILINEAR_CODE,
        }
    }

    /// The filter a frame header codes.
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            Self::REGULAR_CODE => Some(Self::Regular),
            Self::SMOOTH_CODE => Some(Self::Smooth),
            Self::SHARP_CODE => Some(Self::Sharp),
            Self::BILINEAR_CODE => Some(Self::Bilinear),
            _ => None,
        }
    }

    /// The same family as nc-dsp's AV1 kernels name it (equal taps and
    /// short-edge rule; module docs).
    const fn av1(self) -> Av1Filter {
        match self {
            Self::Regular => Av1Filter::Regular,
            Self::Smooth => Av1Filter::Smooth,
            Self::Sharp => Av1Filter::Sharp,
            Self::Bilinear => Av1Filter::Bilinear,
        }
    }

    /// The shared kernel computing this filter exactly (module docs).
    #[must_use]
    pub const fn kernel(self) -> Filter {
        match self {
            Self::Regular => Filter::Av1,
            Self::Smooth => Filter::Av1Smooth,
            Self::Sharp => Filter::Av1Sharp,
            Self::Bilinear => Filter::H264Chroma,
        }
    }
}

/// Filter of the native 8×8 picture coders (`inter`, `search`).
pub const NATIVE_FILTER: InterpFilter = InterpFilter::Regular;
/// Largest prediction side: 64×64 luma blocks.
const MAX_SIDE: usize = 64;
/// Largest reference window: the block plus the longest filter's extra taps.
const MAX_WINDOW: usize = MAX_SIDE + Filter::Av1.taps() - 1;
/// Largest vector component magnitude, in eighth samples.
const MAX_COMPONENT: i32 = 32766;
/// Motion vector units per luma sample (eighth-sample vectors).
const VECTOR_UNITS: i32 = 8;

/// Fill a raster `side`×`side` prediction (`side` is the square root of
/// `output.len()`: 4, 8, 16, 32 or 64) without allocation.
/// `x,y` are plane coordinates; the motion vector is in luma eighth-pel units.
/// `subsample` is zero for luma and one for 4:2:0 chroma. Reference edges extend
/// their nearest visible sample, independently of any allocated border padding
/// (AVM `extend_mc_border`).
/// # Errors
/// Rejects unsupported block sizes, precision, coordinates or vector ranges.
pub fn predict(
    dsp: &Dsp,
    filter: InterpFilter,
    reference: &Plane,
    x: usize,
    y: usize,
    motion: (i32, i32),
    subsample: u8,
    output: &mut [u8],
) -> Result<()> {
    let mut destination = square_raster(output)?;
    predict_into(
        dsp,
        filter,
        reference,
        (x, y),
        motion,
        subsample,
        &mut destination,
    )
}

/// The `side`×`side` raster `output` (`side` its length's square root) as
/// a destination view.
fn square_raster(output: &mut [u8]) -> Result<PlaneViewMut<'_>> {
    let side = output.len().isqrt();
    if side * side != output.len() {
        return Err(DecodeError::Unsupported("AV2 prediction block size"));
    }
    PlaneViewMut::new(output, side, side, side)
        .ok_or(DecodeError::Unsupported("AV2 prediction block size"))
}

/// [`predict`] into `destination`, a square of 4 to 64 samples a side: the
/// decoder predicts straight into the picture it reconstructs.
/// # Errors
/// As [`predict`].
pub fn predict_into(
    dsp: &Dsp,
    filter: InterpFilter,
    reference: &Plane,
    (x, y): (usize, usize),
    motion: (i32, i32),
    subsample: u8,
    destination: &mut PlaneViewMut<'_>,
) -> Result<()> {
    let side = destination.width();
    let Some(size) = BlockSize::new(side, side).filter(|_| destination.height() == side) else {
        return Err(DecodeError::Unsupported("AV2 prediction block size"));
    };
    if subsample > 1
        || x > reference.width().saturating_sub(side)
        || y > reference.height().saturating_sub(side)
        || reference.width() < side
        || reference.height() < side
        || [motion.0, motion.1]
            .iter()
            .any(|&v| !(-MAX_COMPONENT..=MAX_COMPONENT).contains(&v) || v & 1 != 0)
    {
        return Err(DecodeError::Malformed("AV2 inter prediction parameters"));
    }
    // AVM `dec_calc_subpel_params`: a plane position in 1/16 samples (luma
    // eighth-sample vectors double, 4:2:0 chroma ones are used as is), here
    // expressed in the kernel's own positions per sample.
    let kernel = filter.kernel();
    let denominator = VECTOR_UNITS << subsample;
    let positions = kernel.positions() as i32;
    let source_x = x as i32 + motion.1.div_euclid(denominator);
    let source_y = y as i32 + motion.0.div_euclid(denominator);
    let frac_x = (motion.1.rem_euclid(denominator) * positions / denominator) as usize;
    let frac_y = (motion.0.rem_euclid(denominator) * positions / denominator) as usize;
    let visible = |left: i32, top: i32, length: usize| {
        left >= 0
            && top >= 0
            && left as usize + length <= reference.width()
            && top as usize + length <= reference.height()
    };
    if frac_x == 0 && frac_y == 0 && visible(source_x, source_y, side) {
        let block = reference
            .region(source_x as isize, source_y as isize, side, side)
            .expect("visible reference block");
        destination.copy_from(&block);
        return Ok(());
    }
    let before = kernel.before() as i32;
    let span = side + kernel.taps() - 1;
    let (window_x, window_y) = (source_x - before, source_y - before);
    if visible(window_x, window_y, span) {
        let window = reference
            .region(window_x as isize, window_y as isize, span, span)
            .expect("visible reference window");
        dsp.predict(kernel, size, &window, frac_x, frac_y, destination);
        return Ok(());
    }
    // Only windows reaching outside the picture need a copied window;
    // interpolation itself still uses the shared dispatched kernel.
    predict_extended(
        dsp,
        (kernel, size),
        reference,
        (window_x, window_y),
        (frac_x, frac_y),
        destination,
    );
    Ok(())
}

/// [`predict`]'s prediction from the `size` block's window of `reference`
/// at (`x`, `y`), which reaches outside the picture: the window is copied
/// with every sample outside repeating the nearest visible one. Kept out
/// of [`predict`] so that its scratch (zero-filled on creation) is not set
/// up for the common windows inside the picture.
#[inline(never)]
fn predict_extended(
    dsp: &Dsp,
    (kernel, size): (Filter, BlockSize),
    reference: &Plane,
    origin: (i32, i32),
    phase: (usize, usize),
    destination: &mut PlaneViewMut<'_>,
) {
    // Scratch sized to the block: zeroing the 64x64 window for every 4x4
    // block cost more than its interpolation.
    let sized: ExtendedFn = match size.width() {
        0..=8 => extended_window::<{ window_area(8) }>,
        9..=16 => extended_window::<{ window_area(16) }>,
        17..=32 => extended_window::<{ window_area(32) }>,
        _ => extended_window::<{ window_area(MAX_SIDE) }>,
    };
    sized(dsp, (kernel, size), reference, origin, phase, destination);
}

/// [`predict_extended`]'s signature, at one scratch size.
type ExtendedFn =
    fn(&Dsp, (Filter, BlockSize), &Plane, (i32, i32), (usize, usize), &mut PlaneViewMut<'_>);

/// Samples of the largest reference window of a block `side` samples
/// square.
const fn window_area(side: usize) -> usize {
    let span = side + MAX_WINDOW - MAX_SIDE;
    span * span
}

/// [`predict_extended`] with a window scratch of `AREA` samples.
fn extended_window<const AREA: usize>(
    dsp: &Dsp,
    (kernel, size): (Filter, BlockSize),
    reference: &Plane,
    (x, y): (i32, i32),
    (frac_x, frac_y): (usize, usize),
    destination: &mut PlaneViewMut<'_>,
) {
    let span = size.width() + kernel.taps() - 1;
    let mut scratch = [0_u8; AREA];
    emulate_window(reference, (x, y), span, &mut scratch);
    let window = PlaneView::new(&scratch[..span * span], span, span, span)
        .expect("reference window fits the scratch");
    dsp.predict(kernel, size, &window, frac_x, frac_y, destination);
}

/// The `span`-square window of `reference` at (`x`, `y`), which may reach
/// outside the picture, into `window` (rows `span` apart): every sample
/// outside repeats the nearest visible one (AVM `extend_mc_border`),
/// through the shared [`nc_dsp::emulate_edges`].
fn emulate_window(reference: &Plane, (x, y): (i32, i32), span: usize, window: &mut [u8]) {
    let view = reference.view();
    nc_dsp::emulate_edges(
        (view.data(), view.stride()),
        (reference.width(), reference.height()),
        (x as isize, y as isize),
        (span, span),
        (window, span),
    );
}

/// AV1 compound phases per sample (1/16), which every plane position here
/// reaches exactly: luma eighth-sample vectors double, 4:2:0 chroma ones
/// are already sixteenths of a chroma sample.
const COMPOUND_PHASES: i32 = 16;

/// Buffers [`predict_compound`] reuses from call to call: the two
/// compound-precision predictions and one reference window.
#[derive(Debug, Default)]
pub struct CompoundScratch {
    predictions: [Vec<i16>; 2],
    window: Vec<u8>,
}

/// Fill a raster `side`×`side` compound prediction: the rounded average of
/// the block's predictions at `motions[0]` and `motions[1]` from the same
/// `reference` plane (module docs). Arguments as for [`predict`].
/// # Errors
/// Rejects unsupported block sizes, precision, coordinates or vector ranges.
#[allow(clippy::too_many_arguments)]
pub fn predict_compound(
    dsp: &Dsp,
    filter: InterpFilter,
    reference: &Plane,
    (x, y): (usize, usize),
    motions: [(i32, i32); 2],
    subsample: u8,
    scratch: &mut CompoundScratch,
    output: &mut [u8],
) -> Result<()> {
    let mut destination = square_raster(output)?;
    predict_compound_into(
        dsp,
        filter,
        reference,
        (x, y),
        motions,
        subsample,
        scratch,
        &mut destination,
    )
}

/// [`predict_compound`] into `destination`, as [`predict_into`].
/// # Errors
/// As [`predict_compound`].
#[allow(clippy::too_many_arguments)]
pub fn predict_compound_into(
    dsp: &Dsp,
    filter: InterpFilter,
    reference: &Plane,
    (x, y): (usize, usize),
    motions: [(i32, i32); 2],
    subsample: u8,
    scratch: &mut CompoundScratch,
    destination: &mut PlaneViewMut<'_>,
) -> Result<()> {
    let side = destination.width();
    if BlockSize::new(side, side).is_none()
        || destination.height() != side
        || subsample > 1
        || x > reference.width().saturating_sub(side)
        || y > reference.height().saturating_sub(side)
        || reference.width() < side
        || reference.height() < side
        || motions
            .iter()
            .flat_map(|&(row, col)| [row, col])
            .any(|v| !(-MAX_COMPONENT..=MAX_COMPONENT).contains(&v) || v & 1 != 0)
    {
        return Err(DecodeError::Malformed("AV2 compound prediction parameters"));
    }
    let filters = Av1Filters::both(filter.av1());
    let span = side + AV1_FILTER_TAPS - 1;
    let denominator = VECTOR_UNITS << subsample;
    let CompoundScratch {
        predictions,
        window,
    } = scratch;
    for (prediction, &(row, col)) in predictions.iter_mut().zip(&motions) {
        prediction.resize(side * side, 0);
        let start_x = x as i32 + col.div_euclid(denominator) - AV1_FILTER_BEFORE as i32;
        let start_y = y as i32 + row.div_euclid(denominator) - AV1_FILTER_BEFORE as i32;
        let phase = |v: i32| (v.rem_euclid(denominator) * COMPOUND_PHASES / denominator) as usize;
        let axes = (
            Av1McAxis::unscaled(phase(col)),
            Av1McAxis::unscaled(phase(row)),
        );
        let inside = start_x >= 0
            && start_y >= 0
            && start_x as usize + span <= reference.width()
            && start_y as usize + span <= reference.height();
        if inside {
            let view = reference
                .region(start_x as isize, start_y as isize, span, span)
                .expect("window inside the picture");
            dsp.av1_predict_compound(filters, &view, (side, side), axes, prediction, side);
            continue;
        }
        // AVM `extend_mc_border`: positions outside the picture repeat its
        // nearest sample.
        window.resize(span * span, 0);
        emulate_window(reference, (start_x, start_y), span, window);
        let view = PlaneView::new(window, span, span, span).expect("compound window");
        dsp.av1_predict_compound(filters, &view, (side, side), axes, prediction, side);
    }
    let [first, second] = &*predictions;
    let size = BlockSize::new(side, side).expect("checked above");
    dsp.average_compound(size, (first, second), side, destination);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_dsp::Isa;
    /// `BILINEAR` hashes, from independent AVM convolution goldens.
    const BILINEAR_HASHES: [u64; 12] = [
        6_396_820_888_593_953_854,
        16_810_956_350_860_985_665,
        2_171_647_985_108_975_452,
        13_633_421_599_655_151_228,
        873_093_400_959_350_533,
        15_868_998_846_104_277_322,
        7_178_182_686_141_433_432,
        5_463_879_013_670_196_617,
        13_797_924_594_454_066_921,
        13_117_056_657_580_404_337,
        6_686_517_549_857_571_365,
        16_412_718_813_187_182_358,
    ];
    /// `EIGHTTAP_REGULAR` hashes, from independent AVM convolution goldens
    /// (they also match streams that avmdec and dav2d decode to our
    /// reconstruction).
    const REGULAR_HASHES: [u64; 12] = [
        1_663_890_888_722_167_476,
        5_747_071_902_698_203_367,
        16_007_576_252_521_259_448,
        3_208_752_414_308_641_919,
        11_670_299_181_878_042_093,
        9_827_726_518_834_135_675,
        17_680_045_458_563_395_838,
        18_038_277_641_848_984_337,
        3_769_667_862_028_320_233,
        12_959_694_213_367_711_569,
        15_335_693_718_688_082_483,
        11_386_102_679_276_639_507,
    ];
    /// `EIGHTTAP_SMOOTH` hashes, from independent AVM convolution goldens.
    const SMOOTH_HASHES: [u64; 12] = [
        659_676_160_920_543_359,
        15_735_492_215_073_687_805,
        3_703_524_398_115_780_474,
        9_179_106_803_257_786_524,
        15_389_056_202_793_011_527,
        3_855_307_684_117_314_007,
        12_596_304_377_224_911_566,
        817_940_948_200_789_001,
        7_724_545_168_332_840_517,
        51_256_704_370_011_519,
        14_986_208_960_143_317_789,
        9_552_234_988_831_259_436,
    ];
    /// `MULTITAP_SHARP` hashes, from independent AVM convolution goldens
    /// (4×4 blocks take the 4-tap regular filter, so they equal regular's).
    const SHARP_HASHES: [u64; 12] = [
        1_663_890_888_722_167_476,
        5_747_071_902_698_203_367,
        16_007_576_252_521_259_448,
        3_208_752_414_308_641_919,
        11_670_299_181_878_042_093,
        9_827_726_518_834_135_675,
        16_490_791_385_235_925_282,
        15_359_677_189_094_153_091,
        3_039_017_019_891_085_517,
        6_194_614_667_191_081_200,
        788_602_930_147_314_233,
        17_188_125_265_953_426_573,
    ];

    fn phase_trace<const N: usize>(
        dsp: &Dsp,
        filter: InterpFilter,
        subsample: u8,
        pattern: usize,
    ) -> u64 {
        let side = N.isqrt();
        let mut reference = Plane::new(24, 24, 0).unwrap();
        let mut state = 17_u32;
        for y in 0..24 {
            for x in 0..24 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                reference.view_mut().row_mut(y)[x] = match pattern {
                    0 => ((x * 37 + y * 53 + x * y * 7 + 13) % 256) as u8,
                    1 => (((x + y) & 1) * 255) as u8,
                    _ => state as u8,
                };
            }
        }
        let mut hash = 14_695_981_039_346_656_037_u64;
        for position in [0, 8, 24 - side] {
            for row in (-18..=18).step_by(2) {
                for col in (-18..=18).step_by(2) {
                    let mut output = [0_u8; N];
                    predict(
                        dsp,
                        filter,
                        &reference,
                        position,
                        position,
                        (row, col),
                        subsample,
                        &mut output,
                    )
                    .unwrap();
                    for sample in output {
                        hash = (hash ^ u64::from(sample)).wrapping_mul(1_099_511_628_211);
                    }
                }
            }
        }
        hash
    }

    /// Scalar and SIMD kernels reproduce the AVM hashes of every filter
    /// (`tools/codegen/check_av2_prediction.py`).
    #[test]
    fn scalar_and_simd_match_pinned_hashes_all_phases() {
        for (filter, hashes) in [
            (InterpFilter::Bilinear, BILINEAR_HASHES),
            (InterpFilter::Regular, REGULAR_HASHES),
            (InterpFilter::Smooth, SMOOTH_HASHES),
            (InterpFilter::Sharp, SHARP_HASHES),
        ] {
            for dsp in [Dsp::with_isa(Isa::Scalar).unwrap(), Dsp::new()] {
                let mut index = 0;
                for side in [4, 8] {
                    for subsample in 0..2 {
                        for pattern in 0..3 {
                            let hash = if side == 4 {
                                phase_trace::<16>(&dsp, filter, subsample, pattern)
                            } else {
                                phase_trace::<64>(&dsp, filter, subsample, pattern)
                            };
                            assert_eq!(
                                hash, hashes[index],
                                "{filter:?} side{side} sub{subsample} pattern{pattern}"
                            );
                            index += 1;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn header_codes_round_trip() {
        for filter in [
            InterpFilter::Regular,
            InterpFilter::Smooth,
            InterpFilter::Sharp,
            InterpFilter::Bilinear,
        ] {
            assert_eq!(InterpFilter::from_code(filter.code()), Some(filter));
        }
        assert_eq!(InterpFilter::from_code(4), None);
    }

    /// The coded prediction equals what the shared engine ranks for the same
    /// quarter-sample vector (nc-encoder `Measurer::with_exact_prediction`:
    /// `Filter::Av1` at 1/16 positions over an edge-extended border), for
    /// every phase, both signs and blocks touching the picture edges.
    #[test]
    fn luma_matches_the_engine_ranking_prediction() {
        const BORDER: usize = 32;
        const QUARTERS: i32 = 4;
        let dsp = Dsp::new();
        let mut reference = Plane::new(40, 24, BORDER).unwrap();
        let mut state = 99_u32;
        for y in 0..24 {
            for x in 0..40 {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                reference.view_mut().row_mut(y)[x] = (state >> 16) as u8;
            }
        }
        reference.extend_borders();
        let kernel = InterpFilter::Regular.kernel();
        let positions = kernel.positions() as i32 / QUARTERS;
        for (x, y) in [(0, 0), (8, 8), (32, 16), (24, 0)] {
            for qy in -21..=21 {
                for qx in -21..=21 {
                    let mut coded = [0_u8; 64];
                    predict(
                        &dsp,
                        InterpFilter::Regular,
                        &reference,
                        x,
                        y,
                        (2 * qy, 2 * qx),
                        0,
                        &mut coded,
                    )
                    .unwrap();
                    let size = BlockSize::new(8, 8).unwrap();
                    let span = 8 + kernel.taps() - 1;
                    let before = kernel.before() as isize;
                    let window = reference
                        .region(
                            x as isize + qx.div_euclid(QUARTERS) as isize - before,
                            y as isize + qy.div_euclid(QUARTERS) as isize - before,
                            span,
                            span,
                        )
                        .unwrap();
                    let mut ranked = [0_u8; 64];
                    let mut view = PlaneViewMut::new(&mut ranked, 8, 8, 8).unwrap();
                    dsp.predict(
                        kernel,
                        size,
                        &window,
                        (qx.rem_euclid(QUARTERS) * positions) as usize,
                        (qy.rem_euclid(QUARTERS) * positions) as usize,
                        &mut view,
                    );
                    assert_eq!(coded, ranked, "block ({x}, {y}) vector ({qx}, {qy})");
                }
            }
        }
    }

    /// Windows reaching outside the picture read the nearest visible sample:
    /// the same block inside an explicitly edge-extended copy predicts alike.
    #[test]
    fn out_of_picture_windows_extend_the_edges() {
        const SIDE: usize = 8;
        const PAD: usize = 24;
        let dsp = Dsp::new();
        let mut reference = Plane::new(16, 16, 0).unwrap();
        let mut padded = Plane::new(16 + 2 * PAD, 16 + 2 * PAD, 0).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                reference.view_mut().row_mut(y)[x] = ((x * 41 + y * 23 + x * y * 5) % 256) as u8;
            }
        }
        for y in 0..padded.height() {
            for x in 0..padded.width() {
                let (sx, sy) = (x.saturating_sub(PAD).min(15), y.saturating_sub(PAD).min(15));
                padded.view_mut().row_mut(y)[x] = reference.view().row(sy)[sx];
            }
        }
        for filter in [InterpFilter::Regular, InterpFilter::Bilinear] {
            for motion in [(-70, -66), (-4, 90), (86, -2), (70, 74), (-150, 150)] {
                for subsample in 0..2 {
                    let mut edge = [0_u8; SIDE * SIDE];
                    let mut inside = [0_u8; SIDE * SIDE];
                    predict(&dsp, filter, &reference, 8, 0, motion, subsample, &mut edge).unwrap();
                    predict(
                        &dsp,
                        filter,
                        &padded,
                        8 + PAD,
                        PAD,
                        motion,
                        subsample,
                        &mut inside,
                    )
                    .unwrap();
                    assert_eq!(
                        edge, inside,
                        "{filter:?} motion {motion:?} subsample {subsample}"
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_parameters_preserve_output() {
        let dsp = Dsp::new();
        let reference = Plane::new(8, 8, 0).unwrap();
        let mut output = [17_u8; 16];
        for (x, y, motion, subsample) in [
            (8, 0, (0, 0), 0),
            (0, 8, (0, 0), 0),
            (0, 0, (1, 0), 0),
            (0, 0, (i32::MIN, 0), 0),
            (0, 0, (0, 0), 2),
        ] {
            let filter = InterpFilter::Regular;
            assert!(
                predict(
                    &dsp,
                    filter,
                    &reference,
                    x,
                    y,
                    motion,
                    subsample,
                    &mut output
                )
                .is_err()
            );
            assert_eq!(output, [17; 16]);
        }
    }
}
