//! Encoder-side AV1 super-resolution (§5.9.8 `superres_params`, §7.16).
//!
//! With `av1.superres_denom` above [`SUPERRES_NUM`], every frame is coded
//! `SUPERRES_NUM / denom` as wide as the stream (the engine resamples the
//! source to that width: `CodecBackend::frame_size`), filtered (deblocking,
//! CDEF) at that width, and then upscaled horizontally by the normative
//! filter. The upscaled picture is the decoder's output and what reference
//! slots hold, so every inter block predicts from a reference wider than
//! its frame: scaled prediction (`inter::predict_scaled`, §7.11.3.3).
//!
//! The engine keeps working on the coded-width reconstructions (motion
//! search, mode decision): the scaled prediction of a block differs from
//! the coded-width one only by the upscale and rescale filters, so the
//! decisions stay close while the coded residual and reconstruction use the
//! exact normative prediction from the upscaled references
//! (`reference_store.rs` holds them).

use std::rc::Rc;

use nc_common::{Picture, PlaneIndex};
use nc_dsp::av1_restoration::Av1Superres;

use crate::backend::coded_picture_size;
use crate::reference_store::HeldReference;

/// `SUPERRES_NUM` (§3): the denominator of an unscaled frame.
pub(crate) const SUPERRES_NUM: usize = 8;
/// `SUPERRES_DENOM_MIN` (§3): the smallest coded denominator.
pub(crate) const SUPERRES_DENOM_MIN: usize = 9;
/// `SUPERRES_DENOM_BITS` (§3).
pub(crate) const SUPERRES_DENOM_BITS: u32 = 3;
/// The largest denominator (`SUPERRES_DENOM_MIN + 2^SUPERRES_DENOM_BITS - 1`).
pub(crate) const SUPERRES_DENOM_MAX: usize = 16;
/// Border of the upscaled reference pictures (prediction clamps
/// positions itself; this only keeps `Picture` invariants).
const UPSCALED_BORDER: usize = 0;

/// `FrameWidth` for `upscaled_width` at denominator `denom` (§5.9.8).
pub(crate) fn downscaled_width(upscaled_width: usize, denom: usize) -> usize {
    (upscaled_width * SUPERRES_NUM + denom / 2) / denom
}

/// The superres state of a stream: its denominator and the upscaled
/// picture of the frame being coded and of the last committed one. The
/// upscaled reference pictures are held by the backend's
/// `ReferenceStore`.
#[derive(Debug)]
pub(crate) struct Superres {
    denom: usize,
    upscaled_width: usize,
    height: usize,
    /// The frame being coded, upscaled by [`Superres::upscale`].
    pending: Option<HeldReference>,
    /// The last committed frame's upscaled picture: the decoder's output.
    output: Option<Rc<Picture>>,
}

impl Superres {
    /// Superres at `denom` for a `upscaled_width` x `height` stream;
    /// `None` at [`SUPERRES_NUM`] (off).
    pub(crate) fn new(denom: usize, upscaled_width: usize, height: usize) -> Option<Self> {
        (denom != SUPERRES_NUM).then_some(Self {
            denom,
            upscaled_width,
            height,
            pending: None,
            output: None,
        })
    }

    /// The signalled denominator (`SuperresDenom`).
    pub(crate) fn denom(&self) -> usize {
        self.denom
    }

    /// Upscale the filtered reconstruction `recon` of a frame `coded_width`
    /// wide (§7.16): each row reads the `MiCols * 4` samples the decoder
    /// reconstructs, so this runs before the area past `coded_width` is
    /// replaced by edge replication.
    pub(crate) fn upscale(&mut self, recon: &Picture, coded_width: usize) {
        let (readable, _) = coded_picture_size(coded_width, self.height);
        let mut upscaled = Picture::new(self.upscaled_width, self.height, UPSCALED_BORDER)
            .expect("the stream size is a valid picture");
        for plane in PlaneIndex::ALL {
            let shift = usize::from(plane != PlaneIndex::Y);
            let (down, up) = (
                coded_width.div_ceil(1 << shift),
                self.upscaled_width.div_ceil(1 << shift),
            );
            let rows = self.height.div_ceil(1 << shift);
            let source = recon
                .plane(plane)
                .region(0, 0, readable >> shift, rows)
                .expect("the coded area covers MiCols * 4 samples");
            let mut dst = upscaled.plane_mut(plane).view_mut();
            Av1Superres::new(down, up).upscale_plane(&source, &mut dst);
        }
        self.pending = Some(HeldReference::new(
            Rc::new(upscaled),
            self.upscaled_width,
            self.height,
        ));
    }

    /// The frame just coded becomes the output; returns its upscaled
    /// picture, the reference of every slot it refreshes.
    pub(crate) fn commit(&mut self) -> HeldReference {
        let picture = self.pending.take().expect("a coded frame was upscaled");
        self.output = Some(Rc::clone(picture.picture()));
        picture
    }

    /// The decoder's output for the last committed frame.
    pub(crate) fn output(&self) -> Option<&Picture> {
        self.output.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscaled_widths_follow_superres_params() {
        assert_eq!(downscaled_width(1280, 16), 640);
        assert_eq!(downscaled_width(1280, 10), 1024);
        assert_eq!(downscaled_width(1280, 12), 853);
    }
}
