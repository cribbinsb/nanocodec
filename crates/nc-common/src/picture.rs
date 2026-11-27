//! A picture: the three planes of one frame, of 8-bit samples or
//! (`Picture<u16>`) deeper ones. Encoders and most decoding is 4:2:0
//! (I420, [`Picture::new`]); decoders of streams with other chroma formats
//! (4:2:2, 4:4:0, 4:4:4) allocate with [`Picture::with_format`], and every
//! consumer of a picture reads its planes' sizes from
//! [`Picture::chroma_format`] (`docs/handoff/chroma-formats.md`).

use crate::error::Result;
use crate::plane::Plane;
use crate::sample::Sample;

/// Chroma subsampling of a picture's two chroma planes. The encoders code
/// 4:2:0 only (`docs/ARCHITECTURE.md` §1); the decoders also take the
/// other formats their standards define. Monochrome (4:0:0) streams are
/// decoded into 4:2:0 pictures whose chroma planes the decoder fills or
/// ignores, as each codec's reference decoder outputs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaFormat {
    /// Chroma at half resolution horizontally and vertically.
    #[default]
    Yuv420,
    /// Chroma at half resolution horizontally, full vertically.
    Yuv422,
    /// Chroma at full resolution horizontally, half vertically (VP9 only).
    Yuv440,
    /// Chroma at full resolution.
    Yuv444,
}

impl ChromaFormat {
    /// Log2 of the horizontal and vertical chroma subsampling factors
    /// (`subsampling_x`, `subsampling_y`; `SubWidthC = 1 << x`).
    #[inline]
    #[must_use]
    pub const fn subsampling(self) -> (usize, usize) {
        match self {
            Self::Yuv420 => (1, 1),
            Self::Yuv422 => (1, 0),
            Self::Yuv440 => (0, 1),
            Self::Yuv444 => (0, 0),
        }
    }

    /// The format of a subsampling pair (each 0 or 1).
    #[inline]
    #[must_use]
    pub const fn from_subsampling(x: usize, y: usize) -> Option<Self> {
        match (x, y) {
            (1, 1) => Some(Self::Yuv420),
            (1, 0) => Some(Self::Yuv422),
            (0, 1) => Some(Self::Yuv440),
            (0, 0) => Some(Self::Yuv444),
            _ => None,
        }
    }

    /// Width and height of a chroma plane of a picture whose luma is
    /// `width` x `height` (odd luma sizes round chroma up).
    #[inline]
    #[must_use]
    pub const fn chroma_size(self, (width, height): (usize, usize)) -> (usize, usize) {
        let (x, y) = self.subsampling();
        ((width + x) >> x, (height + y) >> y)
    }

    /// Width and height of `plane` of a picture whose luma is `width` x
    /// `height`.
    #[inline]
    #[must_use]
    pub const fn plane_size(self, plane: PlaneIndex, size: (usize, usize)) -> (usize, usize) {
        if plane.is_subsampled() {
            self.chroma_size(size)
        } else {
            size
        }
    }

    /// The subsampling of `plane`: `(0, 0)` for luma.
    #[inline]
    #[must_use]
    pub const fn plane_subsampling(self, plane: PlaneIndex) -> (usize, usize) {
        if plane.is_subsampled() {
            self.subsampling()
        } else {
            (0, 0)
        }
    }
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

    /// Whether the plane is chroma (subsampled 2:1 both ways in 4:2:0;
    /// [`ChromaFormat::subsampling`] in general).
    #[must_use]
    pub const fn is_subsampled(self) -> bool {
        !matches!(self, Self::Y)
    }

    /// Width and height of this plane of a 4:2:0 picture whose luma is
    /// `width` x `height` (odd luma sizes round chroma up; other formats:
    /// [`ChromaFormat::plane_size`]).
    #[must_use]
    pub const fn size_in(self, size: (usize, usize)) -> (usize, usize) {
        ChromaFormat::Yuv420.plane_size(self, size)
    }
}

/// One frame of samples `T` (`u8`, or `u16` for more than 8 bits; the
/// stream's depth travels beside it, see [`crate::BitDepths`]) in a chroma
/// format, 4:2:0 unless made by [`Picture::with_format`]. 4:2:0 chroma
/// planes get half the luma border, the other formats' the whole border.
#[derive(Debug, Clone)]
pub struct Picture<T: Sample = u8> {
    planes: [Plane<T>; 3],
    format: ChromaFormat,
}

/// Read-only access to a picture with more of its replicated border visible.
/// Dropping the guard restores the dimensions without moving any samples.
#[derive(Debug)]
pub struct ExtendedPicture<'a, T: Sample> {
    picture: &'a mut Picture<T>,
    original: (usize, usize),
}

impl<T: Sample> core::ops::Deref for ExtendedPicture<'_, T> {
    type Target = Picture<T>;

    fn deref(&self) -> &Self::Target {
        self.picture
    }
}

impl<T: Sample> Drop for ExtendedPicture<'_, T> {
    fn drop(&mut self) {
        self.picture
            .set_size_in_storage(self.original.0, self.original.1);
    }
}

impl<T: Sample> Picture<T> {
    /// A zeroed `width` x `height` I420 picture with at least `luma_border`
    /// samples of luma padding (and half that for chroma).
    ///
    /// # Errors
    ///
    /// [`crate::Error::InvalidDimensions`] if either dimension is zero or too large.
    /// Odd sizes are valid (decoded AV1 and VP9 frames may have any size):
    /// their chroma rounds up.
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
        Self::allocate(
            (width, height),
            ChromaFormat::Yuv420,
            luma_border,
            chroma_border,
        )
    }

    /// A zeroed `width` x `height` picture in chroma `format` with at least
    /// `luma_border` samples of luma padding; chroma planes get half of it
    /// in 4:2:0 and all of it otherwise (a full-resolution axis reads as far
    /// beyond the edge as luma does).
    ///
    /// # Errors
    ///
    /// [`crate::Error::InvalidDimensions`] if a dimension is zero or too large; an
    /// odd size along a subsampled axis rounds its chroma up.
    pub fn with_format(
        width: usize,
        height: usize,
        format: ChromaFormat,
        luma_border: usize,
    ) -> Result<Self> {
        let chroma_border = if format == ChromaFormat::Yuv420 {
            luma_border.div_ceil(2)
        } else {
            luma_border
        };
        Self::allocate((width, height), format, luma_border, chroma_border)
    }

    fn allocate(
        (width, height): (usize, usize),
        format: ChromaFormat,
        luma_border: usize,
        chroma_border: usize,
    ) -> Result<Self> {
        let (chroma_w, chroma_h) = format.chroma_size((width, height));
        Ok(Self {
            planes: [
                Plane::new(width, height, luma_border)?,
                Plane::new(chroma_w, chroma_h, chroma_border)?,
                Plane::new(chroma_w, chroma_h, chroma_border)?,
            ],
            format,
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

    /// The chroma format.
    #[must_use]
    pub fn chroma_format(&self) -> ChromaFormat {
        self.format
    }

    /// One plane.
    #[must_use]
    pub fn plane(&self, index: PlaneIndex) -> &Plane<T> {
        &self.planes[index as usize]
    }

    /// One plane, writable.
    pub fn plane_mut(&mut self, index: PlaneIndex) -> &mut Plane<T> {
        &mut self.planes[index as usize]
    }

    /// All three planes in storage order (Y, U, V), writable at once.
    pub fn planes_mut(&mut self) -> &mut [Plane<T>; 3] {
        &mut self.planes
    }

    /// Samples of one tightly packed I420 frame of this size (bytes for
    /// `u8`; twice that in a 16-bit file).
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

    /// Whether all three planes can expose the requested area in their right
    /// and bottom borders.
    #[must_use]
    pub fn can_extend_to(&self, width: usize, height: usize) -> bool {
        let (x, y) = self.format.subsampling();
        width.is_multiple_of(1 << x)
            && height.is_multiple_of(1 << y)
            && self.planes[0].can_extend_to(width, height)
            && self.planes[1..]
                .iter()
                .all(|p| p.can_extend_to(width >> x, height >> y))
    }

    /// Expose the right and bottom edges already written by
    /// [`Self::extend_borders`] as coding samples. Use this only after extending
    /// the borders of the current visible picture.
    ///
    /// # Panics
    ///
    /// If the requested size does not fit ([`Self::can_extend_to`]).
    pub fn extended(&mut self, width: usize, height: usize) -> ExtendedPicture<'_, T> {
        assert!(
            self.can_extend_to(width, height),
            "the coding area fits in the border"
        );
        let original = (self.width(), self.height());
        self.set_size_in_storage(width, height);
        ExtendedPicture {
            picture: self,
            original,
        }
    }

    fn set_size_in_storage(&mut self, width: usize, height: usize) {
        let (x, y) = self.format.subsampling();
        self.planes[0].set_size_in_storage(width, height);
        for chroma in &mut self.planes[1..] {
            chroma.set_size_in_storage(width >> x, height >> y);
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
        let format = self.format;
        for (plane, index) in self.planes.iter_mut().zip(PlaneIndex::ALL) {
            // A subsampled plane of an odd luma size keeps its last,
            // half-covered sample (`(width + 1) >> 1`).
            let (x, y) = format.plane_subsampling(index);
            let (keep_w, keep_h) = ((width + x) >> x, (height + y) >> y);
            let mut view = plane.view_mut();
            assert!(keep_w > 0 && keep_h > 0 && keep_w <= view.width() && keep_h <= view.height());
            for y in 0..keep_h {
                let row = view.row_mut(y);
                let edge = row[keep_w - 1];
                row[keep_w..].fill(edge);
            }
            // Copied within the plane, without a copy of the last row.
            let (stride, row_width, rows) = (view.stride(), view.width(), view.height());
            let last = (keep_h - 1) * stride;
            let data = view.data_mut();
            for y in keep_h..rows {
                data.copy_within(last..last + row_width, y * stride);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChromaFormat, Picture, PlaneIndex};

    #[test]
    fn replicate_beyond_copies_the_last_column_and_row() {
        let mut pic = Picture::<u8>::new(8, 8, 0).unwrap();
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
        let pic = Picture::<u8>::new(1280, 720, 64).unwrap();
        let u = pic.plane(PlaneIndex::U);
        assert_eq!((u.width(), u.height()), (640, 360));
        assert!(u.border_y() >= 32);
        assert_eq!(pic.packed_frame_len(), 1280 * 720 * 3 / 2);
    }

    #[test]
    fn plane_sizes_follow_the_format() {
        for (format, chroma) in [
            (ChromaFormat::Yuv422, (32, 24)),
            (ChromaFormat::Yuv440, (64, 12)),
            (ChromaFormat::Yuv444, (64, 24)),
        ] {
            let pic = Picture::<u16>::with_format(64, 24, format, 16).unwrap();
            let v = pic.plane(PlaneIndex::V);
            assert_eq!((v.width(), v.height()), chroma);
            assert!(v.border_x() >= 16 && v.border_y() >= 16);
            assert_eq!(pic.chroma_format(), format);
        }
        assert!(Picture::<u8>::with_format(63, 24, ChromaFormat::Yuv444, 0).is_ok());
        assert!(Picture::<u8>::with_format(63, 24, ChromaFormat::Yuv422, 0).is_ok());
        assert!(Picture::<u8>::with_format(64, 23, ChromaFormat::Yuv422, 0).is_ok());
        assert_eq!(
            ChromaFormat::Yuv422.plane_size(PlaneIndex::U, (5, 3)),
            (3, 3)
        );
    }

    #[test]
    fn odd_dimensions_round_chroma_up() {
        let picture = Picture::<u8>::new(1281, 721, 0).unwrap();
        assert_eq!(picture.plane(PlaneIndex::U).view().width(), 641);
        assert_eq!(picture.plane(PlaneIndex::V).view().height(), 361);
        let picture = Picture::<u16>::with_format(5, 3, ChromaFormat::Yuv422, 0).unwrap();
        assert_eq!(picture.plane(PlaneIndex::U).view().width(), 3);
        assert_eq!(picture.plane(PlaneIndex::U).view().height(), 3);
    }
}
