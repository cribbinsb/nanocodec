//! The sample types the general decoder is compiled for: `u8` for 8-bit
//! sequences and `u16` for 10- and 12-bit ones, one source compiled twice
//! as dav2d builds its `bitdepth` templates (docs/handoff/high-bit-depth.md).
//!
//! The decoding process ([`super::decoder`]'s `Core`) is generic over
//! [`Av2Sample`]; the stream's actual depth is a runtime
//! [`nc_common::BitDepth`] passed to every clip and shift. Our own
//! encoder's streams are 8-bit and decode through the subset tile decoder
//! ([`crate::tile_decoder`]), which only the `u8` instance reaches: the
//! conversions below are the identity there and refuse deeper samples.

use nc_common::{BitDepth, Picture, PlaneView, PlaneViewMut};
use nc_dsp::av1_film_grain::Av1FilmGrainParams;
use nc_dsp::av2_dec::coefficient::Av2Pixel;
use nc_dsp::av2_dec::gdf;
use nc_dsp::{AV2_EDGE_LINES, Av2EdgeFilter, Dsp, Pixel};

/// The coefficient storage of samples `T` (dav2d's `coef`: `i16` at 8
/// bits, `i32` deeper).
pub type Coefficient<T> = <T as Av2Pixel>::Coefficient;

/// A sample type of the general decoder, with the routes to the kernels
/// that have 8-bit SIMD tiers (deeper samples take the generic scalar
/// reference).
pub trait Av2Sample: Pixel + Av2Pixel {
    /// `samples` as 8-bit samples, for the kernels with 8-bit SIMD tiers:
    /// the identity for `u8` (so the other branch folds away there),
    /// `None` for deeper samples, which take the generic reference.
    fn eight(samples: &[Self]) -> Option<&[u8]>;

    /// [`Self::eight`] of a mutable slice.
    fn eight_mut(samples: &mut [Self]) -> Option<&mut [u8]>;

    /// [`gdf::gdf_add`]: a block's GDF corrections at `scale` added to
    /// `dst` (rows `stride` apart).
    fn gdf_add(
        dsp: &Dsp,
        dst: (&mut [Self], usize),
        error: &[i8],
        size: (usize, usize),
        scale_depth: (i32, BitDepth),
    );

    /// Lay film grain over a picture ([`Dsp::av2_add_film_grain`]).
    fn add_film_grain(
        dsp: &Dsp,
        grain: (&Av1FilmGrainParams, bool),
        input: ([&PlaneView<'_, Self>; 3], (usize, usize)),
        output: [&mut PlaneViewMut<'_, Self>; 3],
        depth: BitDepth,
    );

    /// Deblock every segment of one edge line ([`Dsp::av2_deblock_line`]).
    fn deblock_line(
        dsp: &Dsp,
        view: &mut PlaneViewMut<'_, Self>,
        line: (usize, bool),
        filters: &[Option<Av2EdgeFilter>],
        depth: BitDepth,
    );

    /// Deblock one segment, leaving the sides `keep` names unchanged
    /// ([`Dsp::av2_deblock_edge_keeping`]).
    fn deblock_edge_keeping(
        dsp: &Dsp,
        view: &mut PlaneViewMut<'_, Self>,
        edge: ((usize, usize), bool),
        filter: (Av2EdgeFilter, (bool, bool)),
        depth: BitDepth,
    );

    /// The picture as the subset tile decoder (8-bit) reads it; `None` for
    /// deeper samples.
    fn as_eight(picture: &Picture<Self>) -> Option<&Picture>;

    /// A picture the subset tile decoder made, as this type; `None` for
    /// deeper samples (whose sequences never reach that decoder).
    fn from_eight(picture: Picture) -> Option<Picture<Self>>;
}

impl Av2Sample for u8 {
    #[inline]
    fn eight(samples: &[Self]) -> Option<&[u8]> {
        Some(samples)
    }

    #[inline]
    fn eight_mut(samples: &mut [Self]) -> Option<&mut [u8]> {
        Some(samples)
    }

    #[inline]
    fn gdf_add(
        dsp: &Dsp,
        (dst, stride): (&mut [Self], usize),
        error: &[i8],
        size: (usize, usize),
        (scale, _): (i32, BitDepth),
    ) {
        gdf::gdf_add_with_dsp(dsp, dst, stride, error, size, scale);
    }

    fn add_film_grain(
        dsp: &Dsp,
        (params, large_blocks): (&Av1FilmGrainParams, bool),
        (input, size): ([&PlaneView<'_, Self>; 3], (usize, usize)),
        output: [&mut PlaneViewMut<'_, Self>; 3],
        _: BitDepth,
    ) {
        dsp.av2_add_film_grain(params, large_blocks, input, size, output);
    }

    #[inline]
    fn deblock_line(
        dsp: &Dsp,
        view: &mut PlaneViewMut<'_, Self>,
        (coordinate, vertical): (usize, bool),
        filters: &[Option<Av2EdgeFilter>],
        _: BitDepth,
    ) {
        dsp.av2_deblock_line(view, coordinate, vertical, filters);
    }

    #[inline]
    fn deblock_edge_keeping(
        dsp: &Dsp,
        view: &mut PlaneViewMut<'_, Self>,
        (at, vertical): ((usize, usize), bool),
        (filter, keep): (Av2EdgeFilter, (bool, bool)),
        _: BitDepth,
    ) {
        dsp.av2_deblock_edge_keeping(view, at, vertical, filter, keep);
    }
    #[inline]
    fn as_eight(picture: &Picture) -> Option<&Picture> {
        Some(picture)
    }

    #[inline]
    fn from_eight(picture: Picture) -> Option<Picture> {
        Some(picture)
    }
}

impl Av2Sample for u16 {
    #[inline]
    fn eight(_: &[Self]) -> Option<&[u8]> {
        None
    }

    #[inline]
    fn eight_mut(_: &mut [Self]) -> Option<&mut [u8]> {
        None
    }

    #[inline]
    fn gdf_add(
        _: &Dsp,
        (dst, stride): (&mut [Self], usize),
        error: &[i8],
        size: (usize, usize),
        (scale, depth): (i32, BitDepth),
    ) {
        gdf::gdf_add(dst, stride, error, size, scale, depth);
    }

    fn add_film_grain(
        dsp: &Dsp,
        grain: (&Av1FilmGrainParams, bool),
        (input, size): ([&PlaneView<'_, Self>; 3], (usize, usize)),
        output: [&mut PlaneViewMut<'_, Self>; 3],
        depth: BitDepth,
    ) {
        dsp.av2_add_film_grain_u16(grain, input, size, output, depth);
    }

    fn deblock_line(
        _: &Dsp,
        view: &mut PlaneViewMut<'_, Self>,
        (coordinate, vertical): (usize, bool),
        filters: &[Option<Av2EdgeFilter>],
        depth: BitDepth,
    ) {
        for (segment, filter) in filters.iter().enumerate() {
            if let Some(filter) = *filter {
                let along = segment * AV2_EDGE_LINES;
                let at = if vertical {
                    (coordinate, along)
                } else {
                    (along, coordinate)
                };
                Dsp::av2_deblock_edge_at(view, at, vertical, (filter, (false, false)), depth);
            }
        }
    }

    fn deblock_edge_keeping(
        _: &Dsp,
        view: &mut PlaneViewMut<'_, Self>,
        (at, vertical): ((usize, usize), bool),
        (filter, keep): (Av2EdgeFilter, (bool, bool)),
        depth: BitDepth,
    ) {
        Dsp::av2_deblock_edge_at(view, at, vertical, (filter, keep), depth);
    }
    #[inline]
    fn as_eight(_: &Picture<Self>) -> Option<&Picture> {
        None
    }

    #[inline]
    fn from_eight(_: Picture) -> Option<Picture<Self>> {
        None
    }
}
