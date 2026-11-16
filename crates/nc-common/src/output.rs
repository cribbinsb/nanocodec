//! Decoded pictures in output order, of either sample type.
//!
//! A decoder that reorders shares each output picture with its reference
//! buffers instead of copying it: [`OutputPicture`] holds the picture, its
//! visible (conformance) window and its bit depths. A decoder whose streams
//! may be 8-bit or deeper returns [`DecodedPicture`], one of the two
//! instances (`docs/handoff/high-bit-depth.md`); callers write it with
//! [`DecodedPicture::write_yuv`] (`yuv420p`, or 16-bit little-endian
//! samples such as `yuv420p10le`) or hash its rows.

use std::io::Write;
use std::sync::Arc;

use crate::error::Result;
use crate::picture::{Picture, PlaneIndex};
use crate::sample::{BitDepths, Sample};
use crate::shown::ShownPicture;
use crate::view::PlaneView;

/// A picture in output order: the decoded picture, shared with the
/// decoder's reference buffers (no copy), its visible window and its
/// depths. Its buffer returns to the decoder once both have dropped it.
#[derive(Debug, Clone)]
pub struct OutputPicture<T: Sample = u8> {
    /// The decoded picture, with its padding and border.
    pub picture: Arc<Picture<T>>,
    /// Top-left luma sample of the visible window.
    pub origin: (usize, usize),
    /// Luma size of the visible window.
    pub size: (usize, usize),
    /// The stream's luma and chroma bit depths.
    pub depths: BitDepths,
}

impl<T: Sample> OutputPicture<T> {
    /// The visible samples.
    ///
    /// # Panics
    /// If the window does not fit the picture (decoders check it before
    /// they output a picture).
    #[must_use]
    pub fn view(&self) -> ShownPicture<'_, T> {
        ShownPicture::window(&self.picture, self.origin, self.size)
            .expect("window checked when the picture was output")
    }
}

/// A decoded picture of either sample type.
#[derive(Debug, Clone)]
pub enum DecodedPicture {
    /// 8-bit samples.
    Eight(OutputPicture<u8>),
    /// Deeper samples (9 to 16 bits) in `u16`.
    High(OutputPicture<u16>),
}

impl DecodedPicture {
    /// Luma size of the visible window.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        match self {
            Self::Eight(p) => p.size,
            Self::High(p) => p.size,
        }
    }

    /// The stream's bit depths.
    #[must_use]
    pub fn depths(&self) -> BitDepths {
        match self {
            Self::Eight(p) => p.depths,
            Self::High(p) => p.depths,
        }
    }

    /// Bytes per sample in raw output: 1, or 2 for deep pictures.
    #[must_use]
    pub fn sample_bytes(&self) -> usize {
        match self {
            Self::Eight(_) => u8::BYTES,
            Self::High(_) => u16::BYTES,
        }
    }

    /// Write the visible samples of the first `planes` planes (3, or 1 for
    /// monochrome) as one raw frame: bytes, or 16-bit little-endian for
    /// deep pictures.
    ///
    /// # Errors
    /// [`crate::Error::Io`] if the writer fails.
    pub fn write_yuv<W: Write>(&self, writer: &mut W, planes: usize) -> Result<()> {
        fn visible<T: Sample>(
            picture: &OutputPicture<T>,
            planes: usize,
        ) -> impl Iterator<Item = PlaneView<'_, T>> {
            let view = picture.view();
            PlaneIndex::ALL
                .into_iter()
                .take(planes)
                .map(move |plane| view.plane(plane))
        }
        match self {
            Self::Eight(p) => crate::yuv::write_planes(writer, visible(p, planes)),
            Self::High(p) => crate::yuv::write_planes(writer, visible(p, planes)),
        }
    }
}

/// A decoder's shown picture of either sample type, borrowed until its
/// next decode (decoders that output in decoding order; see
/// [`ShownPicture`]).
#[derive(Debug, Clone, Copy)]
pub enum ShownFrame<'a> {
    /// 8-bit samples.
    Eight(ShownPicture<'a, u8>),
    /// Deeper samples in `u16`, with their depth.
    High(ShownPicture<'a, u16>, BitDepths),
}

impl ShownFrame<'_> {
    /// Luma size.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        match self {
            Self::Eight(p) => (p.width(), p.height()),
            Self::High(p, _) => (p.width(), p.height()),
        }
    }

    /// The bit depths.
    #[must_use]
    pub fn depths(&self) -> BitDepths {
        match self {
            Self::Eight(_) => BitDepths::EIGHT,
            Self::High(_, depths) => *depths,
        }
    }

    /// Write the first `planes` planes as one raw frame, as
    /// [`DecodedPicture::write_yuv`].
    ///
    /// # Errors
    /// [`crate::Error::Io`] if the writer fails.
    pub fn write_yuv<W: Write>(&self, writer: &mut W, planes: usize) -> Result<()> {
        fn visible<'p, T: Sample>(
            picture: &ShownPicture<'p, T>,
            planes: usize,
        ) -> impl Iterator<Item = PlaneView<'p, T>> {
            let picture = *picture;
            PlaneIndex::ALL
                .into_iter()
                .take(planes)
                .map(move |plane| picture.plane(plane))
        }
        match self {
            Self::Eight(p) => crate::yuv::write_planes(writer, visible(p, planes)),
            Self::High(p, _) => crate::yuv::write_planes(writer, visible(p, planes)),
        }
    }
}

impl From<OutputPicture<u8>> for DecodedPicture {
    fn from(picture: OutputPicture<u8>) -> Self {
        Self::Eight(picture)
    }
}

impl From<OutputPicture<u16>> for DecodedPicture {
    fn from(picture: OutputPicture<u16>) -> Self {
        Self::High(picture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::BitDepth;

    #[test]
    fn deep_pictures_write_two_bytes_per_sample() {
        let mut picture = Picture::<u16>::new(4, 2, 0).unwrap();
        picture.plane_mut(PlaneIndex::Y).view_mut().fill(1023);
        let ten = BitDepths {
            luma: BitDepth::TEN,
            chroma: BitDepth::TEN,
        };
        let decoded = DecodedPicture::from(OutputPicture {
            picture: Arc::new(picture),
            origin: (0, 0),
            size: (2, 2),
            depths: ten,
        });
        let mut out = Vec::new();
        decoded.write_yuv(&mut out, 3).unwrap();
        // 2x2 luma + 1 + 1 chroma samples, two bytes each.
        assert_eq!(out.len(), 12);
        assert_eq!(&out[..2], &[0xff, 0x03]);
        assert_eq!((decoded.size(), decoded.sample_bytes()), ((2, 2), 2));
    }
}
