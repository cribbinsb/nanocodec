//! Borrowed camera frames in the layouts capture APIs deliver.
//!
//! A [`RawFrame`] is the encoder's input type for frames it does not own:
//! 8-bit 4:2:0 in planar (I420, YV12) or semi-planar (NV12) layout, each
//! plane with its own stride, exactly as V4L2, Android `Image`, `AVFoundation`
//! `CVPixelBuffer` or `WebCodecs` `VideoFrame.copyTo` hand them out. The
//! encoder converts it straight into its own bordered source buffer (one
//! copy, the same as for a [`Picture`](crate::Picture)), so callers never
//! repack frames themselves.
//!
//! Dimensions are the *visible* frame size. Odd sizes round chroma up
//! (`ceil(width / 2)` x `ceil(height / 2)`), as in every 4:2:0 capture
//! format; padding to the codec's coded size happens inside the encoder.

use crate::error::{Error, Result};
use crate::view::PlaneView;
use crate::{Picture, PlaneIndex};

/// Chroma samples covering `luma` samples of a 4:2:0 frame (rounded up).
#[must_use]
pub const fn chroma_len(luma: usize) -> usize {
    luma.div_ceil(2)
}

/// Bytes per sample pair in an interleaved (NV12) chroma row: Cb then Cr.
pub const NV12_CHROMA_BYTES: usize = 2;

/// One borrowed 8-bit 4:2:0 frame.
#[derive(Debug, Clone, Copy)]
pub enum RawFrame<'a> {
    /// Three planes: Y, then Cb (U), then Cr (V). YV12 is the same with the
    /// chroma planes stored in the other order; see [`RawFrame::yv12`].
    I420 {
        /// Luma, `width` x `height`.
        y: PlaneView<'a>,
        /// Cb, `ceil(width / 2)` x `ceil(height / 2)`.
        u: PlaneView<'a>,
        /// Cr, `ceil(width / 2)` x `ceil(height / 2)`.
        v: PlaneView<'a>,
    },
    /// Luma plane plus one plane of interleaved Cb, Cr pairs.
    Nv12 {
        /// Luma, `width` x `height`.
        y: PlaneView<'a>,
        /// Interleaved chroma, `2 * ceil(width / 2)` bytes x
        /// `ceil(height / 2)` rows, Cb first in each pair.
        uv: PlaneView<'a>,
    },
}

impl<'a> RawFrame<'a> {
    /// An I420 frame from its three planes.
    #[must_use]
    pub fn i420(y: PlaneView<'a>, u: PlaneView<'a>, v: PlaneView<'a>) -> Self {
        Self::I420 { y, u, v }
    }

    /// The visible area of `picture` as an I420 frame (for callers that
    /// convert into a [`Picture`] of their own, e.g. from RGBA).
    #[must_use]
    pub fn from_picture(picture: &'a Picture) -> Self {
        Self::I420 {
            y: picture.plane(PlaneIndex::Y).view(),
            u: picture.plane(PlaneIndex::U).view(),
            v: picture.plane(PlaneIndex::V).view(),
        }
    }

    /// A YV12 frame: planar like I420 but with Cr stored before Cb. Only the
    /// plane order differs, so this is an I420 frame with the views swapped.
    #[must_use]
    pub fn yv12(y: PlaneView<'a>, v: PlaneView<'a>, u: PlaneView<'a>) -> Self {
        Self::I420 { y, u, v }
    }

    /// An NV12 frame from its luma and interleaved chroma planes.
    #[must_use]
    pub fn nv12(y: PlaneView<'a>, uv: PlaneView<'a>) -> Self {
        Self::Nv12 { y, uv }
    }

    /// A tightly packed I420 buffer (`y`, then `u`, then `v`, no row
    /// padding), the layout of raw `.yuv` files.
    ///
    /// # Errors
    ///
    /// [`Error::TruncatedFrame`] if `data` is shorter than one frame.
    pub fn packed_i420(data: &'a [u8], width: usize, height: usize) -> Result<Self> {
        let (cw, ch) = (chroma_len(width), chroma_len(height));
        let (luma, chroma) = (width * height, cw * ch);
        let [y, u, v] = split_planes(data, [luma, chroma, chroma])?;
        Ok(Self::I420 {
            y: packed_view(y, width, height),
            u: packed_view(u, cw, ch),
            v: packed_view(v, cw, ch),
        })
    }

    /// A tightly packed NV12 buffer (luma, then interleaved chroma).
    ///
    /// # Errors
    ///
    /// [`Error::TruncatedFrame`] if `data` is shorter than one frame.
    pub fn packed_nv12(data: &'a [u8], width: usize, height: usize) -> Result<Self> {
        let (uv_width, ch) = (NV12_CHROMA_BYTES * chroma_len(width), chroma_len(height));
        let [y, uv] = split_planes(data, [width * height, uv_width * ch])?;
        Ok(Self::Nv12 {
            y: packed_view(y, width, height),
            uv: packed_view(uv, uv_width, ch),
        })
    }

    /// The luma plane.
    #[must_use]
    pub fn luma(&self) -> PlaneView<'a> {
        match *self {
            Self::I420 { y, .. } | Self::Nv12 { y, .. } => y,
        }
    }

    /// Visible width (the luma plane's).
    #[must_use]
    pub fn width(&self) -> usize {
        self.luma().width()
    }

    /// Visible height (the luma plane's).
    #[must_use]
    pub fn height(&self) -> usize {
        self.luma().height()
    }

    /// Check that every plane has the size a `width` x `height` frame needs.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDimensions`] naming the first plane that does not match.
    pub fn check(&self, width: usize, height: usize) -> Result<()> {
        let (cw, ch) = (chroma_len(width), chroma_len(height));
        let fits = |view: &PlaneView<'_>, w: usize, h: usize, reason| {
            if (view.width(), view.height()) == (w, h) {
                Ok(())
            } else {
                Err(Error::InvalidDimensions {
                    width,
                    height,
                    reason,
                })
            }
        };
        match self {
            Self::I420 { y, u, v } => {
                fits(y, width, height, "luma plane size differs from the frame")?;
                fits(u, cw, ch, "Cb plane is not ceil(width/2) x ceil(height/2)")?;
                fits(v, cw, ch, "Cr plane is not ceil(width/2) x ceil(height/2)")
            }
            Self::Nv12 { y, uv } => {
                fits(y, width, height, "luma plane size differs from the frame")?;
                fits(
                    uv,
                    NV12_CHROMA_BYTES * cw,
                    ch,
                    "NV12 chroma plane is not 2*ceil(width/2) bytes x ceil(height/2) rows",
                )
            }
        }
    }
}

/// Consecutive planes of the given lengths at the start of `data`.
fn split_planes<const N: usize>(data: &[u8], lens: [usize; N]) -> Result<[&[u8]; N]> {
    let expected: usize = lens.iter().sum();
    if data.len() < expected {
        return Err(Error::TruncatedFrame {
            expected,
            got: data.len(),
        });
    }
    let mut rest = data;
    Ok(lens.map(|len| {
        let (plane, tail) = rest.split_at(len);
        rest = tail;
        plane
    }))
}

/// A view of a packed plane whose length was checked by [`split_planes`].
fn packed_view(data: &[u8], width: usize, height: usize) -> PlaneView<'_> {
    PlaneView::new(data, width, width, height).expect("plane length checked by split_planes")
}

#[cfg(test)]
mod tests {
    use super::{RawFrame, chroma_len};
    use crate::error::Error;
    use crate::view::PlaneView;

    #[test]
    fn chroma_rounds_up() {
        assert_eq!(chroma_len(4), 2);
        assert_eq!(chroma_len(5), 3);
    }

    #[test]
    fn packed_i420_splits_planes() {
        let data: Vec<u8> = (0..24).collect(); // 4x4 luma + 2x2 + 2x2
        let frame = RawFrame::packed_i420(&data, 4, 4).unwrap();
        frame.check(4, 4).unwrap();
        let RawFrame::I420 { y, u, v } = frame else {
            panic!("expected planar")
        };
        assert_eq!(y.row(3), &[12, 13, 14, 15]);
        assert_eq!(u.row(1), &[18, 19]);
        assert_eq!(v.row(0), &[20, 21]);
    }

    #[test]
    fn packed_nv12_odd_size_rounds_chroma_up() {
        // 3x3 luma (9) + interleaved 4 bytes x 2 rows (8).
        let data = [0u8; 17];
        let frame = RawFrame::packed_nv12(&data, 3, 3).unwrap();
        frame.check(3, 3).unwrap();
        assert!(matches!(
            RawFrame::packed_nv12(&data[..16], 3, 3),
            Err(Error::TruncatedFrame {
                expected: 17,
                got: 16
            })
        ));
    }

    #[test]
    fn yv12_swaps_chroma_planes() {
        let view = |d: &'static [u8], w| PlaneView::new(d, w, w, w).unwrap();
        let frame = RawFrame::yv12(view(&[0; 4], 2), view(&[2], 1), view(&[1], 1));
        let RawFrame::I420 { u, v, .. } = frame else {
            panic!("expected planar")
        };
        assert_eq!((u.row(0), v.row(0)), (&[1u8][..], &[2u8][..]));
    }

    #[test]
    fn check_rejects_wrong_chroma_size() {
        let (luma, chroma) = ([0u8; 16], [0u8; 4]);
        let y = PlaneView::new(&luma, 4, 4, 4).unwrap();
        let small = PlaneView::new(&chroma, 2, 2, 1).unwrap();
        let ok = PlaneView::new(&chroma, 2, 2, 2).unwrap();
        assert!(RawFrame::i420(y, small, ok).check(4, 4).is_err());
        assert!(RawFrame::i420(y, ok, ok).check(4, 4).is_ok());
        assert!(RawFrame::i420(y, ok, ok).check(4, 2).is_err());
    }
}
