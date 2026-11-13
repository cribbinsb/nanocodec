//! A picture: the three planes of one 8-bit 4:2:0 (I420) frame.

use crate::error::{Error, Result};
use crate::plane::{BorderStores, Plane};

/// Chroma subsampling. Only 4:2:0 is supported (`docs/ARCHITECTURE.md` §1: 8-bit 4:2:0 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    /// Chroma at half resolution horizontally and vertically.
    Yuv420,
}

/// Which plane of a picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneIndex {
    /// Luma.
    Y = 0,
    /// Blue-difference chroma.
    U = 1,
    /// Red-difference chroma.
    V = 2,
}

impl PlaneIndex {
    /// All planes in storage order.
    pub const ALL: [PlaneIndex; 3] = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V];
}

/// One 8-bit 4:2:0 frame. Chroma planes get half the luma border.
#[derive(Debug, Clone)]
pub struct Picture {
    planes: [Plane; 3],
}

impl Picture {
    /// A zeroed `width` x `height` I420 picture with at least `luma_border`
    /// samples of luma padding (and half that for chroma).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDimensions`] if either dimension is odd, zero or too large.
    pub fn new(width: usize, height: usize, luma_border: usize) -> Result<Self> {
        Self::with_borders(width, height, luma_border, luma_border.div_ceil(2))
    }

    /// [`Picture::new`] with its own chroma border, for pictures whose
    /// chroma is read much closer to the edge than luma (the border is
    /// rewritten every frame, so its size is per-frame work).
    ///
    /// # Errors
    ///
    /// As for [`Picture::new`].
    pub fn with_borders(
        width: usize,
        height: usize,
        luma_border: usize,
        chroma_border: usize,
    ) -> Result<Self> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "4:2:0 pictures need even width and height",
            });
        }
        Ok(Self {
            planes: [
                Plane::new(width, height, luma_border)?,
                Plane::new(width / 2, height / 2, chroma_border)?,
                Plane::new(width / 2, height / 2, chroma_border)?,
            ],
        })
    }

    /// Luma width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.planes[0].width()
    }

    /// Luma height.
    #[must_use]
    pub fn height(&self) -> usize {
        self.planes[0].height()
    }

    /// The chroma format (always 4:2:0 for now).
    #[must_use]
    pub fn chroma_format(&self) -> ChromaFormat {
        ChromaFormat::Yuv420
    }

    /// One plane.
    #[must_use]
    pub fn plane(&self, index: PlaneIndex) -> &Plane {
        &self.planes[index as usize]
    }

    /// One plane, writable.
    pub fn plane_mut(&mut self, index: PlaneIndex) -> &mut Plane {
        &mut self.planes[index as usize]
    }

    /// All three planes in storage order (Y, U, V), writable at once.
    pub fn planes_mut(&mut self) -> &mut [Plane; 3] {
        &mut self.planes
    }

    /// Bytes of one tightly packed I420 frame of this size.
    #[must_use]
    pub fn packed_frame_len(&self) -> usize {
        self.planes.iter().map(|p| p.width() * p.height()).sum()
    }

    /// Replicate edge samples into the border of every plane.
    pub fn extend_borders(&mut self) {
        for plane in &mut self.planes {
            plane.extend_borders();
        }
    }

    /// [`Self::extend_borders`], writing the borders through `stores`.
    pub fn extend_borders_with(&mut self, stores: &impl BorderStores) {
        for plane in &mut self.planes {
            plane.extend_borders_with(stores);
        }
    }

    /// Overwrite everything right of and below the top-left `width` x
    /// `height` luma area (4:2:0: half that in chroma) with copies of its
    /// last column and row. Codecs whose motion compensation clamps to the
    /// frame size (AV1 §7.11.3.3) while coding a larger block-aligned area
    /// use this before the picture serves as a reference.
    ///
    /// # Panics
    ///
    /// If the area is empty or larger than the picture.
    pub fn replicate_beyond(&mut self, width: usize, height: usize) {
        for (index, plane) in self.planes.iter_mut().enumerate() {
            let shift = usize::from(index > 0);
            let (keep_w, keep_h) = (width >> shift, height >> shift);
            let mut view = plane.view_mut();
            assert!(keep_w > 0 && keep_h > 0 && keep_w <= view.width() && keep_h <= view.height());
            for y in 0..keep_h {
                let row = view.row_mut(y);
                let edge = row[keep_w - 1];
                row[keep_w..].fill(edge);
            }
            if keep_h < view.height() {
                let last = view.row_mut(keep_h - 1).to_vec();
                for y in keep_h..view.height() {
                    view.row_mut(y).copy_from_slice(&last);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Picture, PlaneIndex};

    #[test]
    fn replicate_beyond_copies_the_last_column_and_row() {
        let mut pic = Picture::new(8, 8, 0).unwrap();
        let mut y = pic.plane_mut(PlaneIndex::Y).view_mut();
        for row in 0..8 {
            for (col, s) in y.row_mut(row).iter_mut().enumerate() {
                *s = (row * 8 + col) as u8;
            }
        }
        pic.replicate_beyond(6, 4);
        let y = pic.plane(PlaneIndex::Y).view();
        assert_eq!(y.row(1), &[8, 9, 10, 11, 12, 13, 13, 13]);
        assert_eq!(y.row(7), &[24, 25, 26, 27, 28, 29, 29, 29]);
    }

    #[test]
    fn plane_sizes_follow_420() {
        let pic = Picture::new(1280, 720, 64).unwrap();
        let u = pic.plane(PlaneIndex::U);
        assert_eq!((u.width(), u.height()), (640, 360));
        assert!(u.border_y() >= 32);
        assert_eq!(pic.packed_frame_len(), 1280 * 720 * 3 / 2);
    }

    #[test]
    fn odd_dimensions_are_rejected() {
        assert!(Picture::new(1281, 720, 0).is_err());
        assert!(Picture::new(1280, 721, 0).is_err());
    }
}
