//! Borrowed views of a decoder's shown picture.
//!
//! A decoder that outputs in decoding order can show the visible window of
//! its reconstruction directly instead of copying it into an output
//! picture (a full-frame copy per frame, ~6% of HEVC decoding in WASM).
//! The decoder keeps that reconstruction alive until its next decode, and
//! the borrow checker keeps callers from holding the views past it.

use crate::picture::{ChromaFormat, Picture, PlaneIndex};
use crate::sample::Sample;
use crate::view::PlaneView;

/// The visible samples of one decoded picture: three plane views (chroma
/// of the picture's chroma format, rounded up), borrowed from the decoder
/// (or picture) that owns them.
#[derive(Debug, Clone, Copy)]
pub struct ShownPicture<'a, T: Sample = u8> {
    planes: [PlaneView<'a, T>; 3],
    format: ChromaFormat,
}

impl<'a, T: Sample> ShownPicture<'a, T> {
    /// Every visible sample of `picture`.
    #[must_use]
    pub fn whole(picture: &'a Picture<T>) -> Self {
        Self {
            planes: PlaneIndex::ALL.map(|plane| picture.plane(plane).view()),
            format: picture.chroma_format(),
        }
    }

    /// The `width` x `height` luma window of `picture` at (`left`, `top`),
    /// with the matching chroma windows of its chroma format (an odd width
    /// or height along a subsampled axis takes the chroma column or row its
    /// last luma one shares, as VP9's odd frame sizes do; an origin between
    /// chroma samples starts at the chroma sample its first luma one shares:
    /// only 4:0:0 cropping makes one (H.264 `CropUnitX` 1, §7.4.2.1.1), whose
    /// chroma is flat, so the chroma window keeps its usual size); `None` if
    /// the window is empty or it does not fit.
    #[must_use]
    pub fn window(
        picture: &'a Picture<T>,
        (left, top): (usize, usize),
        (width, height): (usize, usize),
    ) -> Option<Self> {
        let format = picture.chroma_format();
        if width == 0 || height == 0 {
            return None;
        }
        let window = |plane: PlaneIndex| {
            let view = picture.plane(plane).view();
            let (sx, sy) = format.plane_subsampling(plane);
            let (w, h) = format.plane_size(plane, (width, height));
            view.sub_view(left >> sx, top >> sy, w, h)
        };
        Some(Self {
            planes: [
                window(PlaneIndex::Y)?,
                window(PlaneIndex::U)?,
                window(PlaneIndex::V)?,
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

    /// One plane's visible samples.
    #[must_use]
    pub fn plane(&self, index: PlaneIndex) -> PlaneView<'a, T> {
        self.planes[index as usize]
    }

    /// Samples of one tightly packed frame of this size (I420 for 4:2:0).
    #[must_use]
    pub fn packed_frame_len(&self) -> usize {
        self.planes.iter().map(|p| p.width() * p.height()).sum()
    }

    /// The samples copied into `target` (of this size and format), e.g. for
    /// a caller that keeps the picture past the decoder's next decode.
    ///
    /// # Panics
    /// If `target`'s size or format differs.
    pub fn copy_to(&self, target: &mut Picture<T>) {
        assert_eq!(
            (target.width(), target.height(), target.chroma_format()),
            (self.width(), self.height(), self.format),
            "copy_to: picture sizes differ"
        );
        for plane in PlaneIndex::ALL {
            target
                .plane_mut(plane)
                .view_mut()
                .copy_from(&self.plane(plane));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(width: usize, height: usize) -> Picture {
        let mut picture = Picture::<u8>::new(width, height, 8).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = picture.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                    *sample = (plane as usize * 64 + y * 8 + x) as u8;
                }
            }
        }
        picture
    }

    #[test]
    fn window_crops_every_plane() {
        let picture = numbered(8, 6);
        let shown = ShownPicture::window(&picture, (2, 2), (4, 2)).unwrap();
        assert_eq!((shown.width(), shown.height()), (4, 2));
        assert_eq!(shown.plane(PlaneIndex::Y).row(0), [18, 19, 20, 21]);
        assert_eq!(shown.plane(PlaneIndex::U).row(0), [64 + 9, 64 + 10]);
        assert_eq!(shown.packed_frame_len(), 8 + 2 + 2);
        let mut copy = Picture::<u8>::new(4, 2, 0).unwrap();
        shown.copy_to(&mut copy);
        assert_eq!(copy.plane(PlaneIndex::V).view().row(0), [128 + 9, 128 + 10]);
    }

    #[test]
    fn odd_window_takes_the_shared_chroma() {
        let picture = numbered(8, 6);
        let shown = ShownPicture::window(&picture, (2, 0), (5, 3)).unwrap();
        assert_eq!((shown.width(), shown.height()), (5, 3));
        assert_eq!(
            shown.plane(PlaneIndex::U).row(1),
            [64 + 8 + 1, 64 + 8 + 2, 64 + 8 + 3]
        );
        assert_eq!(shown.packed_frame_len(), 15 + 2 * 6);
    }

    #[test]
    fn window_rejects_empty_or_outside() {
        let picture = numbered(8, 6);
        // An odd origin (4:0:0 cropping) covers the chroma its luma shares.
        let odd = ShownPicture::window(&picture, (1, 1), (4, 3)).unwrap();
        assert_eq!((odd.width(), odd.height()), (4, 3));
        assert_eq!(odd.plane(PlaneIndex::U).width(), 2);
        assert_eq!(odd.plane(PlaneIndex::U).height(), 2);
        assert!(ShownPicture::window(&picture, (6, 0), (4, 2)).is_none());
        assert!(ShownPicture::window(&picture, (0, 0), (0, 2)).is_none());
        let whole = ShownPicture::whole(&picture);
        assert_eq!((whole.width(), whole.height()), (8, 6));
    }
}
