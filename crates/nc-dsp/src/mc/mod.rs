//! Motion-compensated prediction kernels for the three codecs, all 8-bit,
//! single-reference ("uni-prediction", default weights), plus AV1's
//! equal-weight compound prediction ([`CompoundFns`]).
//!
//! **AV1** (AV1 §7.11.3.4; libaom `av1_convolve_2d_sr_c` for 8-bit,
//! non-compound): the regular `EIGHTTAP`, `EIGHTTAP_SMOOTH` or
//! `EIGHTTAP_SHARP` filter ([`Av1Family`]), or a 4-tap variant in a
//! direction whose block edge is at most 4 samples (the spec's `filterIdx`
//! 4 and 5 rule, applied to width and height separately: regular and sharp
//! take the 4-tap regular filter, smooth its own 4-tap smooth filter). The horizontal pass rounds by `InterRound0 = 3`, the
//! vertical pass by `InterRound1 = 11`, and the result is clipped to 8 bits.
//! When a sub-sample offset is zero its filter is the identity and the two
//! rounding stages fold into the other pass (libaom's `x_sr` / `y_sr`
//! paths), with identical results; both offsets zero is a copy.
//!
//! Every AV1 tap is even, so SIMD versions run with halved taps in 16 bits
//! (as libaom's SIMD does) and adjust the rounding shifts by one; the
//! results are identical. The halved sharp sums stay below `2^15` (at most
//! `120 * 255`).
//!
//! **HEVC** (HEVC §8.5.3.3.3): 8-tap luma at quarter-sample and 4-tap chroma
//! at eighth-sample positions (`fL`, `fC`). At 8-bit depth the horizontal
//! pass keeps full precision (`shift1 = 0`), the vertical pass of a 2-D
//! filter shifts by `shift2 = 6`, and uni-prediction rounds the 14-bit
//! result by `(x + 32) >> 6` (HEVC §8.5.3.3.4.2). A whole-sample position is
//! a copy.
//!
//! **H.266** (H.266 §8.5.6.3): the same arithmetic over its own tables:
//! 8-tap luma at sixteenth-sample positions, the 6-tap luma of 4x4 affine
//! sub-blocks, and 4-tap chroma at 1/32-sample positions ([`SeparableTaps`];
//! taps from `VVdeC` `InterpolationFilter`, BSD-3-Clause-Clear).
//!
//! **H.264** (H.264 §8.4.2.2): luma half-sample samples `b`, `h` from the
//! 6-tap filter `(1, -5, 20, 20, -5, 1)` with `(x + 16) >> 5`, the centre
//! `j` from the same filter over unrounded horizontal sums with
//! `(x + 512) >> 10`, and quarter-sample positions as rounded averages of
//! two neighbouring integer/half samples (Table 8-12). Chroma is bilinear at
//! eighth-sample positions, `(sum + 32) >> 6` (§8.4.2.2.2).
//!
//! # Kernel contract
//!
//! Every kernel is an `unsafe fn`, called only from [`crate::Dsp`], which
//! guarantees that the CPU supports the kernel's ISA and that `src` (the
//! *window*: the block's top-left reference sample offset up and left by the
//! filter's taps before the centre, [`Filter::before`]) covers `h + taps - 1`
//! rows of `w + taps - 1` samples at `src_stride`, and `dst` covers `h` rows
//! of `w` samples at `dst_stride`. `w` is fixed by the kernel and `h` is a
//! supported block edge; sub-sample offsets are below the filter's
//! positions. AV1 and shared two-pass kernels validate whole footprints
//! before pointer-based lane loads/stores. Other kernels index slices, so
//! broken contracts panic rather than reading out of bounds. H.264 kernels check
//! their whole window and destination once on entry and then run their row
//! loops on raw pointers (`generic::check_h264_window`); they are called
//! through [`crate::H264Predictor`] as well as [`crate::Dsp::predict`].
//!
//! Filter taps: libaom `av1/common/filter.h` (BSD-2-Clause, Alliance for Open
//! Media; see `THIRD_PARTY.md`).

#[allow(unsafe_code)]
pub(crate) mod generic;
mod precision;
pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod av1_avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_dotprod;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod av1_neon;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_h264;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod av1_simd128;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod separable_simd128;

/// Taps of the AV1 filter.
pub(crate) const TAPS: usize = 8;
/// Taps before the centre sample of the AV1 filter.
pub(crate) const TAP_CENTRE: usize = 3;
/// Sub-sample positions of the AV1 filter (1/16).
const AV1_POSITIONS: usize = 16;

/// `EIGHTTAP` (regular) filter by 1/16-sample position (libaom
/// `sub_pel_filters_8`).
pub(crate) const REGULAR_8: [[i16; TAPS]; AV1_POSITIONS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],
    [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],
    [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0],
    [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],
    [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0],
    [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],
    [0, 0, -2, 8, 126, -6, 2, 0],
];

/// The 4-tap regular filter for block edges of at most 4 samples (libaom
/// `sub_pel_filters_4`).
pub(crate) const REGULAR_4: [[i16; TAPS]; AV1_POSITIONS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -4, 126, 8, -2, 0, 0],
    [0, 0, -8, 122, 18, -4, 0, 0],
    [0, 0, -10, 116, 28, -6, 0, 0],
    [0, 0, -12, 110, 38, -8, 0, 0],
    [0, 0, -12, 102, 48, -10, 0, 0],
    [0, 0, -14, 94, 58, -10, 0, 0],
    [0, 0, -12, 84, 66, -10, 0, 0],
    [0, 0, -12, 76, 76, -12, 0, 0],
    [0, 0, -10, 66, 84, -12, 0, 0],
    [0, 0, -10, 58, 94, -14, 0, 0],
    [0, 0, -10, 48, 102, -12, 0, 0],
    [0, 0, -8, 38, 110, -12, 0, 0],
    [0, 0, -6, 28, 116, -10, 0, 0],
    [0, 0, -4, 18, 122, -8, 0, 0],
    [0, 0, -2, 8, 126, -4, 0, 0],
];

/// `EIGHTTAP_SMOOTH` filter by 1/16-sample position (libaom
/// `sub_pel_filters_8smooth`; AVM `av2_sub_pel_filters_8smooth` is equal).
pub(crate) const SMOOTH_8: [[i16; TAPS]; AV1_POSITIONS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, 28, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, -2, 16, 54, 48, 12, 0, 0],
    [0, -2, 14, 52, 52, 14, -2, 0],
    [0, 0, 12, 48, 54, 16, -2, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 28, 2, 0],
];

/// The 4-tap smooth filter for block edges of at most 4 samples (libaom
/// `sub_pel_filters_4smooth`; AVM `av2_sub_pel_filters_4smooth` is equal).
pub(crate) const SMOOTH_4: [[i16; TAPS]; AV1_POSITIONS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 30, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, 0, 14, 54, 48, 12, 0, 0],
    [0, 0, 12, 52, 52, 12, 0, 0],
    [0, 0, 12, 48, 54, 14, 0, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 30, 0, 0],
];

/// `EIGHTTAP_SHARP` filter by 1/16-sample position (libaom
/// `sub_pel_filters_8sharp`; AVM `av2_sub_pel_filters_8sharp`, its
/// `MULTITAP_SHARP`, is equal). Its outer taps are not zero.
pub(crate) const SHARP_8: [[i16; TAPS]; AV1_POSITIONS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-2, 2, -6, 126, 8, -2, 2, 0],
    [-2, 6, -12, 124, 16, -6, 4, -2],
    [-2, 8, -18, 120, 26, -10, 6, -2],
    [-4, 10, -22, 116, 38, -14, 6, -2],
    [-4, 10, -22, 108, 48, -18, 8, -2],
    [-4, 10, -24, 100, 60, -20, 8, -2],
    [-4, 10, -24, 90, 70, -22, 10, -2],
    [-4, 12, -24, 80, 80, -24, 12, -4],
    [-2, 10, -22, 70, 90, -24, 10, -4],
    [-2, 8, -20, 60, 100, -24, 10, -4],
    [-2, 8, -18, 48, 108, -22, 10, -4],
    [-2, 6, -14, 38, 116, -22, 10, -4],
    [-2, 6, -10, 26, 120, -18, 8, -2],
    [-2, 4, -6, 16, 124, -12, 6, -2],
    [0, 2, -2, 8, 126, -6, 2, -2],
];

/// One AV1 interpolation filter family: its taps for a block edge of
/// `length` samples at sub-sample position `frac` (the 4-tap variant for
/// edges of at most 4, AV1 `filterIdx` 4/5). Kernels are generic over the
/// family, so each family's table lookups are resolved at compile time.
pub(crate) trait Av1Family: OuterTaps {
    /// The taps for an edge of `length` samples at position `frac`.
    fn taps(length: usize, frac: usize) -> &'static [i16; TAPS];
}

/// Horizontal and vertical taps of one prediction.
pub(crate) type TapPair<'t> = (&'t [i16; TAPS], &'t [i16; TAPS]);

/// What kernels know at compile time of the taps they run with.
pub(crate) trait OuterTaps {
    /// The first and last taps are zero at every position and edge length:
    /// kernels then filter six taps and never load those samples.
    const OUTER_ZERO: bool;
}

/// Taps the caller chooses per block (mixed families, [`Av1MixedFn`]):
/// any of them, so all eight are filtered.
pub(crate) struct AnyTaps;

impl OuterTaps for AnyTaps {
    const OUTER_ZERO: bool = false;
}

/// Edges of at most this many samples take a family's 4-tap variant.
const SHORT_EDGE: usize = 4;

/// The regular `EIGHTTAP` family.
pub(crate) struct Regular;
/// The `EIGHTTAP_SMOOTH` family.
pub(crate) struct Smooth;
/// The `EIGHTTAP_SHARP` family.
pub(crate) struct Sharp;

impl OuterTaps for Regular {
    const OUTER_ZERO: bool = true;
}

impl Av1Family for Regular {
    #[inline]
    fn taps(length: usize, frac: usize) -> &'static [i16; TAPS] {
        if length <= SHORT_EDGE {
            &REGULAR_4[frac]
        } else {
            &REGULAR_8[frac]
        }
    }
}

impl OuterTaps for Smooth {
    const OUTER_ZERO: bool = true;
}

impl Av1Family for Smooth {
    #[inline]
    fn taps(length: usize, frac: usize) -> &'static [i16; TAPS] {
        if length <= SHORT_EDGE {
            &SMOOTH_4[frac]
        } else {
            &SMOOTH_8[frac]
        }
    }
}

impl OuterTaps for Sharp {
    const OUTER_ZERO: bool = false;
}

impl Av1Family for Sharp {
    /// Short edges take the 4-tap regular filter (AV1 `filterIdx` 4).
    #[inline]
    fn taps(length: usize, frac: usize) -> &'static [i16; TAPS] {
        if length <= SHORT_EDGE {
            &REGULAR_4[frac]
        } else {
            &SHARP_8[frac]
        }
    }
}

/// `filter` with every tap halved (all regular taps are even).
#[cfg(target_arch = "aarch64")]
pub(crate) fn halved(filter: &[i16; TAPS]) -> [i16; TAPS] {
    filter.map(|t| t / 2)
}

/// AV1 prediction of a `w`x`h` block (width fixed by the kernel).
pub(crate) type ConvolveFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
);

/// Center the six-fractional-bit HEVC predictor so its full interpolation
/// range fits i16: uncentered half-pel 2-D can reach 33150, centered bounds
/// are -25022..=24958. Weighted output restores this offset in its round term.
pub(crate) const HEVC_PREP_OFFSET: i16 = 1 << 13;

/// HEVC luma filter `fL` by quarter-sample position (HEVC Table 8-11).
pub(crate) const HEVC_LUMA: [[i16; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// HEVC chroma filter `fC` by eighth-sample position (HEVC Table 8-12).
pub(crate) const HEVC_CHROMA: [[i16; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// H.266 luma filter `fL` by sixteenth-sample position (H.266 Table 27,
/// `hpelIfIdx` 0): at quarter positions (4, 8, 12) it is HEVC's `fL`.
pub(crate) const VVC_LUMA: [[i16; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [0, 1, -3, 63, 4, -2, 1, 0],
    [-1, 2, -5, 62, 8, -3, 1, 0],
    [-1, 3, -8, 60, 13, -4, 1, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 52, 26, -8, 3, -1],
    [-1, 3, -9, 47, 31, -10, 4, -1],
    [-1, 4, -11, 45, 34, -10, 4, -1],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [-1, 4, -10, 34, 45, -11, 4, -1],
    [-1, 4, -10, 31, 47, -9, 3, -1],
    [-1, 3, -8, 26, 52, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
    [0, 1, -4, 13, 60, -8, 3, -1],
    [0, 1, -3, 8, 62, -5, 2, -1],
    [0, 1, -2, 4, 63, -3, 1, 0],
];

/// H.266 6-tap luma filter of 4x4 affine sub-blocks by sixteenth-sample
/// position (H.266 Table 28), in an 8-tap window with zero outer taps.
pub(crate) const VVC_AFFINE_LUMA: [[i16; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [0, 1, -3, 63, 4, -2, 1, 0],
    [0, 1, -5, 62, 8, -3, 1, 0],
    [0, 2, -8, 60, 13, -4, 1, 0],
    [0, 3, -10, 58, 17, -5, 1, 0],
    [0, 3, -11, 52, 26, -8, 2, 0],
    [0, 2, -9, 47, 31, -10, 3, 0],
    [0, 3, -11, 45, 34, -10, 3, 0],
    [0, 3, -11, 40, 40, -11, 3, 0],
    [0, 3, -10, 34, 45, -11, 3, 0],
    [0, 3, -10, 31, 47, -9, 2, 0],
    [0, 2, -8, 26, 52, -11, 3, 0],
    [0, 1, -5, 17, 58, -10, 3, 0],
    [0, 1, -4, 13, 60, -8, 2, 0],
    [0, 1, -3, 8, 62, -5, 1, 0],
    [0, 1, -2, 4, 63, -3, 1, 0],
];

/// H.266 chroma filter `fC` by 1/32-sample position (H.266 Table 33): at
/// eighth positions (multiples of 4) it is HEVC's `fC`.
pub(crate) const VVC_CHROMA: [[i16; 4]; 32] = [
    [0, 64, 0, 0],
    [-1, 63, 2, 0],
    [-2, 62, 4, 0],
    [-2, 60, 7, -1],
    [-2, 58, 10, -2],
    [-3, 57, 12, -2],
    [-4, 56, 14, -2],
    [-4, 55, 15, -2],
    [-4, 54, 16, -2],
    [-5, 53, 18, -2],
    [-6, 52, 20, -2],
    [-6, 49, 24, -3],
    [-6, 46, 28, -4],
    [-5, 44, 29, -4],
    [-4, 42, 30, -4],
    [-4, 39, 33, -4],
    [-4, 36, 36, -4],
    [-4, 33, 39, -4],
    [-4, 30, 42, -4],
    [-4, 29, 44, -5],
    [-4, 28, 46, -6],
    [-3, 24, 49, -6],
    [-2, 20, 52, -6],
    [-2, 18, 53, -5],
    [-2, 16, 54, -4],
    [-2, 15, 55, -4],
    [-2, 14, 56, -4],
    [-2, 12, 57, -3],
    [-2, 10, 58, -2],
    [-1, 7, 60, -2],
    [0, 4, 62, -2],
    [0, 2, 63, -1],
];

/// One separable HEVC-style filter table (HEVC §8.5.3.3.3, H.266
/// §8.5.6.3): `N` taps at each sub-sample position, an unrounded first
/// pass, `>> 6` after the second and uni-prediction rounding. The kernels
/// of every such table share `generic::hevc` (SIMD) and
/// `scalar::hevc_separable`.
pub(crate) trait SeparableTaps<const N: usize> {
    /// The taps by sub-sample position.
    const TABLE: &'static [[i16; N]];
}
/// HEVC `fL`.
pub(crate) struct HevcLumaTaps;
impl SeparableTaps<8> for HevcLumaTaps {
    const TABLE: &'static [[i16; 8]] = &HEVC_LUMA;
}
/// HEVC `fC`.
pub(crate) struct HevcChromaTaps;
impl SeparableTaps<4> for HevcChromaTaps {
    const TABLE: &'static [[i16; 4]] = &HEVC_CHROMA;
}
/// H.266 `fL`.
pub(crate) struct VvcLumaTaps;
impl SeparableTaps<8> for VvcLumaTaps {
    const TABLE: &'static [[i16; 8]] = &VVC_LUMA;
}
/// H.266 affine sub-block luma.
pub(crate) struct VvcAffineLumaTaps;
impl SeparableTaps<8> for VvcAffineLumaTaps {
    const TABLE: &'static [[i16; 8]] = &VVC_AFFINE_LUMA;
}
/// H.266 `fC`.
pub(crate) struct VvcChromaTaps;
impl SeparableTaps<4> for VvcChromaTaps {
    const TABLE: &'static [[i16; 4]] = &VVC_CHROMA;
}

/// H.264 luma half-sample filter (H.264 §8.4.2.2.1).
pub(crate) const H264_TAPS: [i16; 6] = [1, -5, 20, 20, -5, 1];

/// One sub-block of a [`crate::Dsp::predict_sub_blocks`] batch: where its
/// filter window starts in the batch's reference window (whole samples:
/// the sub-block's position plus its vector's whole-sample part, minus
/// [`Filter::before`]) and its vector's sub-sample parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubBlockMotion {
    /// Column of the filter window's first sample in the batch's window.
    pub x: u16,
    /// Row of the filter window's first sample in the batch's window.
    pub y: u16,
    /// Horizontal sub-sample position ([`Filter::positions`] units).
    pub frac_x: u8,
    /// Vertical sub-sample position.
    pub frac_y: u8,
}

/// Window geometry of one codec's prediction filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// AV1 regular `EIGHTTAP`, 1/16-sample positions.
    Av1,
    /// AV1 `EIGHTTAP_SMOOTH`, 1/16-sample positions.
    Av1Smooth,
    /// AV1 `EIGHTTAP_SHARP`, 1/16-sample positions.
    Av1Sharp,
    /// HEVC 8-tap luma, quarter-sample positions.
    HevcLuma,
    /// HEVC 4-tap chroma, eighth-sample positions.
    HevcChroma,
    /// H.264 6-tap luma, quarter-sample positions.
    H264Luma,
    /// H.264 bilinear chroma, eighth-sample positions.
    H264Chroma,
    /// H.266 8-tap luma, sixteenth-sample positions.
    VvcLuma,
    /// H.266 6-tap luma of 4x4 affine sub-blocks (an 8-tap window),
    /// sixteenth-sample positions.
    VvcAffineLuma,
    /// H.266 4-tap chroma, 1/32-sample positions.
    VvcChroma,
}

impl Filter {
    /// Taps of the filter: the window is `w + taps - 1` by `h + taps - 1`.
    #[must_use]
    pub const fn taps(self) -> usize {
        match self {
            Filter::Av1
            | Filter::Av1Smooth
            | Filter::Av1Sharp
            | Filter::HevcLuma
            | Filter::VvcLuma
            | Filter::VvcAffineLuma => 8,
            Filter::HevcChroma | Filter::VvcChroma => 4,
            Filter::H264Luma => 6,
            Filter::H264Chroma => 2,
        }
    }

    /// Taps before the centre sample: how far up and left the window starts.
    #[must_use]
    pub const fn before(self) -> usize {
        match self {
            Filter::Av1
            | Filter::Av1Smooth
            | Filter::Av1Sharp
            | Filter::HevcLuma
            | Filter::VvcLuma
            | Filter::VvcAffineLuma => 3,
            Filter::HevcChroma | Filter::VvcChroma => 1,
            Filter::H264Luma => 2,
            Filter::H264Chroma => 0,
        }
    }

    /// Sub-sample positions per sample.
    #[must_use]
    pub const fn positions(self) -> usize {
        match self {
            Filter::Av1
            | Filter::Av1Smooth
            | Filter::Av1Sharp
            | Filter::VvcLuma
            | Filter::VvcAffineLuma => 16,
            Filter::HevcLuma | Filter::H264Luma => 4,
            Filter::HevcChroma | Filter::H264Chroma => 8,
            Filter::VvcChroma => 32,
        }
    }
}

/// A compound-precision prediction kernel (the [`ConvolveFn`] contract
/// with `i16` outputs): AV1 (module docs of `generic::av1_prep`) or HEVC
/// 14-bit samples (`scalar::hevc_prep`).
pub(crate) type PrepFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
);

/// HEVC default-weight bi-prediction (HEVC eq. 8-251, 8-bit): the second
/// list's prediction from `src` ([`PrepFn`]'s window contract) averaged
/// with the first list's centred prediction `first` (`h` rows of the
/// kernel's width at `first_stride`) into 8-bit samples at `dst`.
pub(crate) type BiFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    first: &[i16],
    first_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
);

/// The rounded average of two compound predictions (`p0`, `p1` at
/// `src_stride`, `h` rows of the kernel's width) into `dst`.
pub(crate) type AverageFn = unsafe fn(
    p0: &[i16],
    p1: &[i16],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
);

/// One ISA's compound kernels by width class (4..64): AV1 compound-precision
/// prediction per filter family (regular, smooth, sharp) and the
/// equal-weight average (AV1 §7.11.3.4 with `InterRound1` = 7, §7.11.3.15);
/// HEVC 14-bit prediction samples `predSamplesLX` for luma and chroma
/// (HEVC §8.5.3.3.3 at 8-bit depth: `shift1 = 0`, `shift2 = 6`, a
/// whole-sample position is `(s << 6) - 8192`), which
/// `Dsp::hevc_weighted_prediction_centered` combines.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CompoundFns {
    pub prep: [[PrepFn; 5]; 3],
    pub average: [AverageFn; 5],
    /// HEVC luma (`fL`) and chroma (`fC`).
    pub hevc_prep: [[PrepFn; 5]; 2],
    /// HEVC default bi-prediction, luma and chroma: the second list's
    /// prediction fused with the average.
    pub hevc_bi: [[BiFn; 5]; 2],
}

impl CompoundFns {
    /// The compound-precision prediction kernels of `filter`; `None` for
    /// filters without them (H.264, H.266).
    pub(crate) fn prep(&self, filter: Filter) -> Option<&[PrepFn; 5]> {
        match filter {
            Filter::Av1 => Some(&self.prep[0]),
            Filter::Av1Smooth => Some(&self.prep[1]),
            Filter::Av1Sharp => Some(&self.prep[2]),
            Filter::HevcLuma => Some(&self.hevc_prep[0]),
            Filter::HevcChroma => Some(&self.hevc_prep[1]),
            _ => None,
        }
    }

    /// The HEVC bi-prediction kernels of `filter`; `None` for the other
    /// codecs' filters.
    pub(crate) fn hevc_bi(&self, filter: Filter) -> Option<&[BiFn; 5]> {
        match filter {
            Filter::HevcLuma => Some(&self.hevc_bi[0]),
            Filter::HevcChroma => Some(&self.hevc_bi[1]),
            _ => None,
        }
    }
}

/// One ISA's prediction kernels, indexed by width class (4..64), per
/// [`Filter`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct McFns {
    pub compound: CompoundFns,
    pub av1: [ConvolveFn; 5],
    pub av1_smooth: [ConvolveFn; 5],
    pub av1_sharp: [ConvolveFn; 5],
    pub hevc_luma: [ConvolveFn; 5],
    pub hevc_chroma: [ConvolveFn; 5],
    pub h264_luma: [ConvolveFn; 5],
    pub h264_chroma: [ConvolveFn; 5],
    pub vvc_luma: [ConvolveFn; 5],
    pub vvc_affine_luma: [ConvolveFn; 5],
    pub vvc_chroma: [ConvolveFn; 5],
    /// H.264 kernels resolved per block shape and sub-sample phase, where
    /// the ISA has them; otherwise every phase uses the width kernel.
    pub h264_phases: Option<H264PhaseFns>,
    /// AV1 kernels for blocks whose two directions use different filter
    /// families, where the ISA has them.
    pub av1_mixed: Option<Av1MixedFns>,
}

/// AV1 prediction of a `w`x`h` block (width fixed by the kernel, the
/// [`ConvolveFn`] contract) with the caller's taps per direction: the
/// horizontal filter's taps at `frac_x` and the vertical filter's at
/// `frac_y`, each already chosen for its block edge (the 4-tap variants of
/// edges up to 4). This serves blocks whose two directions use different
/// filter families (`dual_filter`), as dav1d's `put_8tap` / `prep_8tap`
/// take separate horizontal and vertical filters. The kernels filter all
/// eight taps, which is exact for every family.
pub(crate) type Av1MixedFn<T> = unsafe fn(
    src: &[u8],
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
);

/// One ISA's mixed-family AV1 kernels ([`Av1MixedFn`]) by width class
/// (4..64): final 8-bit samples and compound precision.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Av1MixedFns {
    pub put: [Av1MixedFn<u8>; 5],
    pub prep: [Av1MixedFn<i16>; 5],
}

/// H.264 luma block shapes (width, height) with a kernel per phase
/// ([`H264PhaseFns`]): the macroblock, its 16x8 / 8x16 partitions, 8x8 and
/// 4x4 blocks (decoder and encoder predict every partition through these).
pub(crate) const H264_LUMA_SHAPES: [(usize, usize); 5] =
    [(16, 16), (16, 8), (8, 16), (8, 8), (4, 4)];
/// The 4:2:0 chroma shapes of [`H264_LUMA_SHAPES`] (4x4 also stands for
/// the 2x2 blocks of 4x4 luma blocks).
pub(crate) const H264_CHROMA_SHAPES: [(usize, usize); 4] = [(8, 8), (8, 4), (4, 8), (4, 4)];
/// Kernel slots of one H.264 shape, [`h264_slot`] `0..16`.
pub(crate) const H264_PHASE_SLOTS: usize = 16;
/// [`h264_slot`]'s cap for luma: quarter-sample offsets are their own
/// coordinates.
pub(crate) const H264_LUMA_PHASE_CAP: usize = 3;
/// [`h264_slot`]'s cap for chroma: the slot is the offsets' class (each
/// zero or not); the eighth-sample weights stay arguments.
pub(crate) const H264_CHROMA_CLASS_CAP: usize = 1;
/// Chroma classes, as [`h264_slot`] numbers them: both offsets zero (a
/// copy), only horizontal, only vertical, both.
pub(crate) const H264_CHROMA_COPY: usize = 0;
pub(crate) const H264_CHROMA_HORIZONTAL: usize = 1;
pub(crate) const H264_CHROMA_VERTICAL: usize = 4;
pub(crate) const H264_CHROMA_BOTH: usize = 5;

/// The kernel slot of sub-sample offsets (`frac_x`, `frac_y`), each capped
/// at `cap` ([`H264_LUMA_PHASE_CAP`] or [`H264_CHROMA_CLASS_CAP`]):
/// `min(frac_x, cap) + 4 · min(frac_y, cap)`, without branches.
#[inline]
pub(crate) fn h264_slot(frac_x: usize, frac_y: usize, cap: usize) -> usize {
    frac_x.min(cap) | (frac_y.min(cap) << 2)
}

/// One ISA's H.264 kernels per block shape and phase: every size and
/// offset (luma) or offset class (chroma) is a constant of its kernel, so
/// a block resolves its phase once, by table, instead of the kernel
/// branching on it and running loops of varying trip counts (openh264's
/// per-phase `McHorVerXY` functions, BSD-2-Clause).
#[derive(Debug, Clone, Copy)]
pub(crate) struct H264PhaseFns {
    /// Per [`H264_LUMA_SHAPES`] entry, in [`h264_slot`] order.
    pub luma: [[ConvolveFn; H264_PHASE_SLOTS]; H264_LUMA_SHAPES.len()],
    /// Per [`H264_CHROMA_SHAPES`] entry: copy, horizontal, vertical, both.
    pub chroma: [[ConvolveFn; 4]; H264_CHROMA_SHAPES.len()],
}

impl H264PhaseFns {
    /// The 16 kernel slots of `filter` at (`w`, `h`), if specialised.
    pub(crate) fn slots(
        &self,
        filter: Filter,
        size: (usize, usize),
    ) -> Option<[ConvolveFn; H264_PHASE_SLOTS]> {
        match filter {
            Filter::H264Luma => {
                let shape = H264_LUMA_SHAPES.iter().position(|&s| s == size)?;
                Some(self.luma[shape])
            }
            Filter::H264Chroma => {
                let shape = H264_CHROMA_SHAPES.iter().position(|&s| s == size)?;
                let [copy, horizontal, vertical, both] = self.chroma[shape];
                let mut slots = [copy; H264_PHASE_SLOTS];
                slots[H264_CHROMA_HORIZONTAL] = horizontal;
                slots[H264_CHROMA_VERTICAL] = vertical;
                slots[H264_CHROMA_BOTH] = both;
                Some(slots)
            }
            _ => None,
        }
    }
}

impl McFns {
    pub(crate) fn table(&self, filter: Filter) -> &[ConvolveFn; 5] {
        match filter {
            Filter::Av1 => &self.av1,
            Filter::Av1Smooth => &self.av1_smooth,
            Filter::Av1Sharp => &self.av1_sharp,
            Filter::HevcLuma => &self.hevc_luma,
            Filter::HevcChroma => &self.hevc_chroma,
            Filter::H264Luma => &self.h264_luma,
            Filter::H264Chroma => &self.h264_chroma,
            Filter::VvcLuma => &self.vvc_luma,
            Filter::VvcAffineLuma => &self.vvc_affine_luma,
            Filter::VvcChroma => &self.vvc_chroma,
        }
    }
}

#[cfg(test)]
mod support_tests {
    use super::*;
    /// The six-tap kernel paths rely on `OUTER_ZERO`; the halved SIMD taps
    /// on every tap being even and each filter summing to 128.
    #[test]
    fn av1_family_properties_hold_for_every_phase() {
        fn check<F: Av1Family>() {
            for length in [SHORT_EDGE, SHORT_EDGE * 2] {
                for frac in 0..AV1_POSITIONS {
                    let taps = F::taps(length, frac);
                    if F::OUTER_ZERO {
                        assert_eq!((taps[0], taps[TAPS - 1]), (0, 0));
                    }
                    assert!(taps.iter().all(|t| t % 2 == 0));
                    assert_eq!(taps.iter().map(|&t| i32::from(t)).sum::<i32>(), 128);
                }
            }
        }
        check::<Regular>();
        check::<Smooth>();
        check::<Sharp>();
    }
}
