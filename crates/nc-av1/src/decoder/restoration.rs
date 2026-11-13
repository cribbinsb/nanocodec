//! Super-resolution and loop restoration of a decoded frame (AV1 §7.16,
//! §7.17), on top of the nc-dsp kernels (`nc_dsp::av1_restoration`).
//!
//! Order within a frame: after deblocking, [`StripeBoundaries::save`]
//! keeps the deblocked rows around each internal stripe edge; after CDEF,
//! [`superres_upscale`] widens the frame when it is coded at a reduced
//! width; then [`apply`] restores every plane whose `FrameRestorationType`
//! is not `RESTORE_NONE`, unit by unit with the filters the block layer
//! read into [`RestorationUnits`] (§5.11.57–58).
//!
//! The reconstructed (downscaled) picture must keep its samples up to
//! `MiCols * 4` luma columns, in its border where that passes the frame
//! width: superres reads them (§7.16), as libaom's frame buffers do.

use nc_common::decode::DecodeError;
use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::Dsp;
use nc_dsp::av1_restoration::{
    Av1LoopRestoration, Av1LrFilter, Av1LrScratch, Av1StripeBoundaries, Av1Superres,
    av1_lr_unit_count,
};

use super::frame_header::{FrameSize, LrType, Restoration};

/// Width and height of plane `plane` of a `width`x`height` 4:2:0 frame.
fn plane_size(plane: PlaneIndex, (width, height): (usize, usize)) -> (usize, usize) {
    if plane == PlaneIndex::Y {
        (width, height)
    } else {
        (width.div_ceil(2), height.div_ceil(2))
    }
}

/// Whether `plane` is subsampled.
fn subsampled(plane: PlaneIndex) -> bool {
    plane != PlaneIndex::Y
}

/// The view of the top-left `width`x`height` samples of one plane, which
/// may extend into its border.
fn plane_view(
    picture: &Picture,
    plane: PlaneIndex,
    (width, height): (usize, usize),
) -> Result<PlaneView<'_>, DecodeError> {
    picture
        .plane(plane)
        .region(0, 0, width, height)
        .ok_or(DecodeError::Malformed("AV1 plane smaller than its frame"))
}

/// The superres resampling of each plane, when the frame uses it.
fn superres(size: &FrameSize) -> Option<[Av1Superres; 3]> {
    (size.width != size.upscaled_width).then(|| {
        PlaneIndex::ALL.map(|plane| {
            let (down, _) = plane_size(plane, (size.width, size.height));
            let (up, _) = plane_size(plane, (size.upscaled_width, size.height));
            Av1Superres::new(down, up)
        })
    })
}

/// Width of the samples superres reads in each plane: `MiCols * 4` luma.
fn readable_width(plane: PlaneIndex, size: &FrameSize) -> usize {
    (size.mi_cols() * 4) >> u32::from(subsampled(plane))
}

/// The deblocked rows around each internal stripe edge of each plane.
#[derive(Debug, Clone)]
pub(crate) struct StripeBoundaries {
    planes: [Av1StripeBoundaries; 3],
}

impl StripeBoundaries {
    /// Keep the boundary rows of the deblocked `picture` (before CDEF) of
    /// a frame of `size`, upscaled when the frame uses superres.
    pub(crate) fn save(picture: &Picture, size: &FrameSize) -> Result<Self, DecodeError> {
        let scalers = superres(size);
        let mut planes = Vec::with_capacity(PlaneIndex::ALL.len());
        for (index, plane) in PlaneIndex::ALL.into_iter().enumerate() {
            let (coded, rows) = plane_size(plane, (size.width, size.height));
            let width = if scalers.is_some() {
                readable_width(plane, size)
            } else {
                coded
            };
            let deblocked = plane_view(picture, plane, (width, rows))?;
            let scaler = scalers.as_ref().map(|s| &s[index]);
            planes.push(Av1StripeBoundaries::save(
                &deblocked,
                subsampled(plane),
                scaler,
            ));
        }
        let planes = planes.try_into().expect("three planes");
        Ok(Self { planes })
    }
}

/// Upscale every plane of `coded` (after CDEF) to the frame's upscaled
/// width into `upscaled` (AV1 §7.16). Only for frames that use superres.
pub(crate) fn superres_upscale(
    coded: &Picture,
    size: &FrameSize,
    upscaled: &mut Picture,
) -> Result<(), DecodeError> {
    let scalers = superres(size).ok_or(DecodeError::Malformed("AV1 superres without scaling"))?;
    for (plane, scaler) in PlaneIndex::ALL.into_iter().zip(&scalers) {
        let (_, rows) = plane_size(plane, (size.width, size.height));
        let source = plane_view(coded, plane, (readable_width(plane, size), rows))?;
        let mut dst = upscaled
            .plane_mut(plane)
            .region_mut(0, 0, scaler.upscaled_width(), rows)
            .ok_or(DecodeError::Malformed("AV1 upscaled plane too small"))?;
        scaler.upscale_plane(&source, &mut dst);
    }
    Ok(())
}

/// The filter of every restoration unit of each plane (`LrType`,
/// `LrWiener`, `LrSgrSet`, `LrSgrXqd`), all [`Av1LrFilter::None`] until the
/// block layer reads them.
#[derive(Debug, Clone)]
pub(crate) struct RestorationUnits {
    unit_size: [usize; 3],
    columns: [usize; 3],
    rows: [usize; 3],
    filters: [Vec<Av1LrFilter>; 3],
    enabled: [bool; 3],
}

impl RestorationUnits {
    /// The unit grid of each plane of a frame of `size` with `lr_params`.
    pub(crate) fn new(lr: &Restoration, size: &FrameSize) -> Self {
        let mut units = Self {
            unit_size: lr.unit_size,
            columns: [0; 3],
            rows: [0; 3],
            filters: Default::default(),
            enabled: lr.kind.map(|k| k != LrType::None),
        };
        for (index, plane) in PlaneIndex::ALL.into_iter().enumerate() {
            if !units.enabled[index] {
                continue;
            }
            let (width, height) = plane_size(plane, (size.upscaled_width, size.height));
            units.columns[index] = av1_lr_unit_count(lr.unit_size[index], width);
            units.rows[index] = av1_lr_unit_count(lr.unit_size[index], height);
            units.filters[index] =
                vec![Av1LrFilter::None; units.columns[index] * units.rows[index]];
        }
        units
    }

    /// Units per row and unit rows of `plane` (zero when it is not
    /// restored).
    pub(crate) fn grid(&self, plane: PlaneIndex) -> (usize, usize) {
        (self.columns[plane as usize], self.rows[plane as usize])
    }

    /// Set the filter of unit (`row`, `column`) of `plane`.
    pub(crate) fn set(
        &mut self,
        plane: PlaneIndex,
        (row, column): (usize, usize),
        filter: Av1LrFilter,
    ) {
        let index = plane as usize;
        self.filters[index][row * self.columns[index] + column] = filter;
    }
}

/// Restore every plane of `frame` (after CDEF and superres, at the
/// upscaled size) that has restoration, in place (AV1 §7.17).
pub(crate) fn apply(
    (dsp, scratch): (&Dsp, &mut Av1LrScratch),
    frame: &mut Picture,
    boundaries: &StripeBoundaries,
    units: &RestorationUnits,
    size: &FrameSize,
) -> Result<(), DecodeError> {
    for (index, plane) in PlaneIndex::ALL.into_iter().enumerate() {
        if !units.enabled[index] {
            continue;
        }
        let (width, height) = plane_size(plane, (size.upscaled_width, size.height));
        let mut view = frame
            .plane_mut(plane)
            .region_mut(0, 0, width, height)
            .ok_or(DecodeError::Malformed("AV1 plane smaller than its frame"))?;
        Av1LoopRestoration {
            unit_size: units.unit_size[index],
            columns: units.columns[index],
            rows: units.rows[index],
            filters: &units.filters[index],
        }
        .apply(
            dsp,
            &mut view,
            &boundaries.planes[index],
            subsampled(plane),
            scratch,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_dsp::av1_restoration::Av1Wiener;

    fn size(width: usize, upscaled_width: usize, height: usize) -> FrameSize {
        FrameSize {
            width,
            height,
            upscaled_width,
            render_width: upscaled_width,
            render_height: height,
            superres_denom: 8,
        }
    }

    fn textured(width: usize, height: usize) -> Picture {
        let mut picture = Picture::new(width, height, 64).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = picture.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                    *s = ((x * 7 + y * 13) % 251) as u8;
                }
            }
        }
        picture
    }

    /// Zero Wiener taps are the identity, and units without a filter keep
    /// the CDEF output.
    #[test]
    fn identity_filters_keep_the_frame() {
        let frame_size = size(96, 96, 80);
        let lr = Restoration {
            kind: [LrType::Wiener, LrType::Switchable, LrType::None],
            unit_size: [64, 32, 32],
        };
        let original = textured(96, 80);
        let mut frame = original.clone();
        let boundaries = StripeBoundaries::save(&frame, &frame_size).unwrap();
        let mut units = RestorationUnits::new(&lr, &frame_size);
        let (columns, rows) = units.grid(PlaneIndex::Y);
        for row in 0..rows {
            for column in 0..columns {
                units.set(
                    PlaneIndex::Y,
                    (row, column),
                    Av1LrFilter::Wiener(Av1Wiener::default()),
                );
            }
        }
        apply(
            (&Dsp::new(), &mut Av1LrScratch::default()),
            &mut frame,
            &boundaries,
            &units,
            &frame_size,
        )
        .unwrap();
        for plane in PlaneIndex::ALL {
            assert_eq!(
                frame.plane(plane).view().data(),
                original.plane(plane).view().data()
            );
        }
    }

    /// Upscaling a flat frame keeps it flat at the new width.
    #[test]
    fn superres_upscales_to_the_full_width() {
        let frame_size = size(64, 96, 32);
        let mut coded = Picture::new(64, 32, 64).unwrap();
        for plane in PlaneIndex::ALL {
            coded.plane_mut(plane).view_mut().fill(77);
        }
        coded.extend_borders();
        let mut upscaled = Picture::new(96, 32, 64).unwrap();
        superres_upscale(&coded, &frame_size, &mut upscaled).unwrap();
        for plane in PlaneIndex::ALL {
            assert!(
                upscaled
                    .plane(plane)
                    .view()
                    .rows()
                    .all(|r| r.iter().all(|&s| s == 77))
            );
        }
    }
}
