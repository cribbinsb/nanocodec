//! Reference picture resampling (RPR): prediction from a reference picture
//! whose size or scaling window differs from the current picture's (H.266
//! §8.3.2 `RprConstraintsActiveFlag`, §8.5.6.3.1 scaled reference positions).
//!
//! Each picture in the DPB keeps its [`Geometry`]; per slice and reference
//! the [`Scaling`] between the two pictures is worked out once
//! ([`Scaling::between`], `None` when the flag is 0). A scaled reference is
//! predicted through `nc_dsp::vvc_predict_scaled`, whose sample positions
//! this module derives as the BSD-licensed VTM
//! `InterPrediction::xPredInterBlkRPR` does (reimplemented; see
//! `THIRD_PARTY.md`). The callers switch DMVR, BDOF, PROF and wrap-around
//! off for such references (§8.5.1, §8.5.5.9, §8.5.6.3.2).
use crate::decoder_headers::{PictureParameters, Sequence};
use nc_common::{
    Plane,
    decode::{DecodeError, Result},
};
use nc_dsp::{DepthKernels, Pixel, VVC_RPR_POSITION_BITS, VvcRprFilter, VvcScaledBlock};

/// Fractional bits of a scaling ratio (`scalingRatio`, §8.5.6.3.1: 1.0 is
/// `1 << 14`).
const RATIO_BITS: u32 = 14;
/// The ratio of an unscaled axis.
const UNIT_RATIO: i32 = 1 << RATIO_BITS;
/// Above this ratio (1.25) an axis takes the 1.5x downsampling filters
/// (§8.5.6.3.2, `scalingRatio > 20480`).
const RATIO_1P5_THRESHOLD: i32 = UNIT_RATIO * 5 / 4;
/// Above this ratio (1.75) an axis takes the 2x downsampling filters.
const RATIO_2_THRESHOLD: i32 = UNIT_RATIO * 7 / 4;
/// Fractional bits of a luma motion vector (1/16 sample).
const MV_BITS: u32 = 4;
/// Reference positions further than this (in 1/1024 samples, 2^19
/// samples) outside any picture read only the edge sample, so origins are
/// clamped to it: the kernel's position arithmetic then stays in `i32` for
/// any coded motion vector and scaling window.
const ORIGIN_LIMIT: i64 = 1 << 29;

/// What resampling needs of a picture: its size and its scaling window
/// (§7.4.3.5 `pps_scaling_win_*`, in luma samples). The subpicture count
/// that `RprConstraintsActiveFlag` also compares cannot differ: it is the
/// SPS's, which is the same for every picture of a coded video sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Geometry {
    /// Width and height in luma samples.
    size: [i32; 2],
    /// Left, right, top and bottom scaling window offsets in luma samples.
    window: [i32; 4],
}

impl Geometry {
    /// The geometry of pictures coded with `parameters` of `sequence`.
    pub fn of(sequence: &Sequence, parameters: &PictureParameters) -> Self {
        // Offsets in units of SubWidthC and SubHeightC (Table 2).
        let units = sequence.crop_units().map(|v| v as i32);
        let w = parameters.scaling_window;
        Self {
            size: parameters.dimensions.map(|v| v as i32),
            window: [
                w[0] * units[0],
                w[1] * units[0],
                w[2] * units[1],
                w[3] * units[1],
            ],
        }
    }

    /// The scaling window's width or height (`fCurWidth`, `fCurHeight`).
    fn window_size(self, axis: usize) -> i64 {
        i64::from(self.size[axis]) - i64::from(self.window[2 * axis] + self.window[2 * axis + 1])
    }
}

/// How one reference picture maps onto the current picture, when
/// `RprConstraintsActiveFlag` is 1 for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Scaling {
    /// Horizontal and vertical `scalingRatio` (14 fractional bits).
    ratio: [i32; 2],
    /// The current picture's scaling window left and top offsets, luma
    /// samples.
    current: [i32; 2],
    /// The reference picture's scaling window left and top offsets.
    reference: [i32; 2],
    /// `sps_chroma_horizontal_collocated_flag` and the vertical one.
    collocated: [bool; 2],
}

impl Scaling {
    /// The scaling from a picture of geometry `reference` to one of
    /// `current`, with the SPS's chroma sample locations `collocated`;
    /// `None` when they agree in size and scaling window
    /// (`RprConstraintsActiveFlag` 0, §8.3.2).
    ///
    /// # Errors
    ///
    /// A reference window more than twice or less than an eighth of the
    /// current one (§7.4.3.5).
    pub fn between(
        current: Geometry,
        reference: Geometry,
        collocated: [bool; 2],
    ) -> Result<Option<Self>> {
        if current == reference {
            return Ok(None);
        }
        let mut ratio = [0; 2];
        for (axis, ratio) in ratio.iter_mut().enumerate() {
            let (to, from) = (current.window_size(axis), reference.window_size(axis));
            if to <= 0 || from <= 0 || 2 * to < from || to > 8 * from {
                return Err(DecodeError::Malformed("VVC scaling window ratio"));
            }
            *ratio = (((from << RATIO_BITS) + (to >> 1)) / to) as i32;
        }
        Ok(Some(Self {
            ratio,
            current: [current.window[0], current.window[2]],
            reference: [reference.window[0], reference.window[2]],
            collocated,
        }))
    }

    /// The filter set of `axis` (§8.5.6.3.2): by its ratio, with the
    /// alternative half-sample filter (`half`) only on an unscaled axis.
    fn filter(self, axis: usize, half: bool) -> VvcRprFilter {
        match self.ratio[axis] {
            ratio if ratio > RATIO_2_THRESHOLD => VvcRprFilter::Ratio2,
            ratio if ratio > RATIO_1P5_THRESHOLD => VvcRprFilter::Ratio1p5,
            UNIT_RATIO if half => VvcRprFilter::AlternativeHalf,
            _ => VvcRprFilter::Regular,
        }
    }

    /// The scaled reference position of sample `position` on `axis` of a
    /// plane with subsampling `shift` there (0 or 1), moved by `vector`
    /// (1/16 luma samples, or with `chroma` 1/32 chroma samples), in 1/1024
    /// samples of the reference plane (§8.5.6.3.1 `refxSbL`, `refxSbC`...,
    /// VTM `x0Int`).
    fn origin(
        self,
        axis: usize,
        position: usize,
        vector: i32,
        (shift, chroma): (u32, bool),
    ) -> i64 {
        let ratio = i64::from(self.ratio[axis]);
        let position = (((position as i64) << shift) - i64::from(self.current[axis])) >> shift;
        // Chroma sample positions not collocated with luma move by half a
        // luma sample's worth of the ratio's change.
        let offset = if chroma && !self.collocated[axis] {
            8 * (ratio - i64::from(UNIT_RATIO))
        } else {
            0
        };
        let vector_bits = MV_BITS + u32::from(chroma);
        let scaled = ((position << vector_bits) + i64::from(vector)) * ratio + offset;
        // From 1/16 (luma) or 1/32 (chroma) samples times the ratio to
        // 1/1024 samples, rounding half away from zero.
        let bits = RATIO_BITS + vector_bits - VVC_RPR_POSITION_BITS;
        let magnitude = (scaled.abs() + (1 << (bits - 1))) >> bits;
        let rounded = if scaled < 0 { -magnitude } else { magnitude };
        rounded + (i64::from(self.reference[axis]) << (VVC_RPR_POSITION_BITS - shift))
    }
}

/// One region of a block predicted from a scaled reference.
#[derive(Debug, Clone, Copy)]
pub(super) struct Region {
    /// The top-left sample of the block whose sample positions the region
    /// shares (`xSb`, `ySb`: the CU, or its affine or `SbTMVP` subblock), in
    /// samples of the plane.
    pub anchor: (usize, usize),
    /// The region's top-left sample, inside that block.
    pub position: (usize, usize),
    /// Its width and height.
    pub size: (usize, usize),
}

/// Predict `region` of a luma (`luma`) or chroma plane from `reference`,
/// scaled by `scaling`, moved by `vector`, as 14-bit high-precision samples
/// (§8.5.6.3.2 to §8.5.6.3.4). `half` asks for the alternative half-sample
/// luma filter, `affine` for the 6-tap affine luma filter; `area` is the
/// subpicture treated as a picture the samples are clamped to, in samples
/// of the plane, if any.
pub(super) fn predict<T: Pixel>(
    dsp: DepthKernels<'_>,
    reference: &Plane<T>,
    (scaling, vector): (Scaling, crate::motion::Mv16),
    (region, (sx, sy)): (Region, (usize, usize)),
    (luma, half, affine): (bool, bool, bool),
    area: Option<[i32; 4]>,
    out: &mut [i16],
) {
    let shifts = [(sx as u32, !luma), (sy as u32, !luma)];
    let step = scaling.ratio.map(|ratio| (ratio + 8) >> 4);
    // The anchor's position, then whole steps to the region (as one
    // prediction of the whole block would place them).
    let origin = |axis: usize, anchor: usize, position: usize, vector: i32| {
        let start = scaling.origin(axis, anchor, vector, shifts[axis])
            + (position - anchor) as i64 * i64::from(step[axis]);
        start.clamp(-ORIGIN_LIMIT, ORIGIN_LIMIT) as i32
    };
    let (width, height) = (reference.width() as i32, reference.height() as i32);
    let bounds = match area {
        Some([left, top, right, bottom]) => [
            left.min(width - 1),
            top.min(height - 1),
            right.min(width - 1),
            bottom.min(height - 1),
        ],
        None => [0, 0, width - 1, height - 1],
    };
    let block = VvcScaledBlock {
        size: region.size,
        origin: (
            origin(0, region.anchor.0, region.position.0, vector.x),
            origin(1, region.anchor.1, region.position.1, vector.y),
        ),
        step: (step[0], step[1]),
        bounds,
    };
    let filters = [
        scaling.filter(0, half && luma && !affine),
        scaling.filter(1, half && luma && !affine),
    ];
    dsp.vvc_predict_scaled(&reference.view(), &block, filters, (luma, affine), out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(width: i32, height: i32) -> Geometry {
        Geometry {
            size: [width, height],
            window: [0; 4],
        }
    }

    /// The ratios and filter sets of 2x and 1.5x references, and none for
    /// a reference of the same geometry.
    #[test]
    fn ratio_selects_the_filter_set() {
        let current = geometry(640, 360);
        assert_eq!(Scaling::between(current, current, [false; 2]), Ok(None));
        let double = Scaling::between(geometry(320, 180), geometry(640, 360), [false; 2])
            .unwrap()
            .unwrap();
        assert_eq!(double.ratio, [2 * UNIT_RATIO; 2]);
        assert_eq!(double.filter(0, true), VvcRprFilter::Ratio2);
        let wide = Scaling::between(geometry(640, 360), geometry(960, 360), [false; 2])
            .unwrap()
            .unwrap();
        assert_eq!(wide.ratio, [UNIT_RATIO * 3 / 2, UNIT_RATIO]);
        assert_eq!(wide.filter(0, true), VvcRprFilter::Ratio1p5);
        assert_eq!(wide.filter(1, true), VvcRprFilter::AlternativeHalf);
        assert_eq!(wide.filter(1, false), VvcRprFilter::Regular);
        assert!(Scaling::between(geometry(64, 64), geometry(192, 64), [false; 2]).is_err());
    }

    /// A 2x reference doubles luma positions; non-collocated chroma moves
    /// by a quarter chroma sample more (§8.5.6.3.1).
    #[test]
    fn positions_scale_with_the_ratio() {
        let unit = 1i64 << VVC_RPR_POSITION_BITS;
        let double = Scaling::between(geometry(320, 180), geometry(640, 360), [false, true])
            .unwrap()
            .unwrap();
        // Luma sample 8 moved by one sample: reference sample 18.
        assert_eq!(double.origin(0, 8, 16, (0, false)), 18 * unit);
        // Chroma sample 4 moved by half a chroma sample: 9, plus the offset
        // of the sited chroma (horizontal) but not the collocated (vertical).
        assert_eq!(double.origin(0, 4, 16, (1, true)), 9 * unit + unit / 4);
        assert_eq!(double.origin(1, 4, 16, (1, true)), 9 * unit);
        // 4:4:4 chroma (vector in 1/32 samples) lands where luma does.
        assert_eq!(double.origin(1, 8, 32, (0, true)), 18 * unit);
    }
}
