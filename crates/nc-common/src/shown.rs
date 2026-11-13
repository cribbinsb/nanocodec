//! Borrowed views of a decoder's shown picture.
//!
//! A decoder that outputs in decoding order can show the visible window of
//! its reconstruction directly instead of copying it into an output
//! picture (a full-frame copy per frame, ~6% of HEVC decoding in WASM).
//! The decoder keeps that reconstruction alive until its next decode, and
//! the borrow checker keeps callers from holding the views past it.

use crate::picture::{Picture, PlaneIndex};
use crate::view::PlaneView;

/// The visible samples of one decoded 4:2:0 picture: three plane views of
/// even luma size, borrowed from the decoder (or picture) that owns them.
#[derive(Debug, Clone, Copy)]
pub struct ShownPicture<'a> {
    planes: [PlaneView<'a>; 3],
}

impl<'a> ShownPicture<'a> {
    /// Every visible sample of `picture`.
    #[must_use]
    pub fn whole(picture: &'a Picture) -> Self {
        Self {
            planes: PlaneIndex::ALL.map(|plane| picture.plane(plane).view()),
        }
    }

    /// The `width` x `height` luma window of `picture` at (`left`, `top`),
    /// with the matching chroma windows; `None` if a value is odd or the
    /// window does not fit.
    #[must_use]
    pub fn window(
        picture: &'a Picture,
        (left, top): (usize, usize),
        (width, height): (usize, usize),
    ) -> Option<Self> {
        let even = [left, top, width, height]
            .iter()
            .all(|v| v.is_multiple_of(2));
        if !even || width == 0 || height == 0 {
            return None;
        }
        let window = |plane: PlaneIndex| {
            let shift = usize::from(plane != PlaneIndex::Y);
            picture.plane(plane).view().sub_view(
                left >> shift,
                top >> shift,
                width >> shift,
                height >> shift,
            )
        };
        Some(Self {
            planes: [
                window(PlaneIndex::Y)?,
                window(PlaneIndex::U)?,
                window(PlaneIndex::V)?,
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

    /// One plane's visible samples.
    #[must_use]
    pub fn plane(&self, index: PlaneIndex) -> PlaneView<'a> {
        self.planes[index as usize]
    }

    /// Bytes of one tightly packed I420 frame of this size.
    #[must_use]
    pub fn packed_frame_len(&self) -> usize {
        self.planes.iter().map(|p| p.width() * p.height()).sum()
    }

    /// The samples copied into `target` (of this size), e.g. for a caller
    /// that keeps the picture past the decoder's next decode.
    ///
    /// # Panics
    /// If `target`'s size differs.
    pub fn copy_to(&self, target: &mut Picture) {
        assert_eq!(
            (target.width(), target.height()),
            (self.width(), self.height()),
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
        let mut picture = Picture::new(width, height, 8).unwrap();
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
        let mut copy = Picture::new(4, 2, 0).unwrap();
        shown.copy_to(&mut copy);
        assert_eq!(copy.plane(PlaneIndex::V).view().row(0), [128 + 9, 128 + 10]);
    }

    #[test]
    fn window_rejects_odd_or_outside() {
        let picture = numbered(8, 6);
        assert!(ShownPicture::window(&picture, (1, 0), (4, 2)).is_none());
        assert!(ShownPicture::window(&picture, (6, 0), (4, 2)).is_none());
        assert!(ShownPicture::window(&picture, (0, 0), (0, 2)).is_none());
        let whole = ShownPicture::whole(&picture);
        assert_eq!((whole.width(), whole.height()), (8, 6));
    }
}
