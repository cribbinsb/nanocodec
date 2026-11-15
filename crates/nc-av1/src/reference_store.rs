//! The AV1 backend's own copies of reference pictures, for the slots whose
//! normative picture is not the engine's reconstruction.
//!
//! Normally every reference slot's picture is exactly the engine's
//! reconstruction in its pool, and blocks predict from that (the fast path:
//! `inter::predict`). Two features break that equality, and then the
//! backend holds the slot's normative picture here and predicts from it by
//! §7.11.3.3 scaled prediction (`inter::predict_scaled`; at equal sizes the
//! scaled formulas reduce to the unscaled ones, so a held picture of the
//! frame's own size predicts exactly too):
//!
//! - superres (`superres.rs`): references are the upscaled pictures, wider
//!   than the coded frame;
//! - a coded-size switch without a key frame
//!   (`CodecBackend::set_frame_size`): the engine resamples its pool to the
//!   new size as proxies for its decisions, so every slot coded before the
//!   switch is held here at its old size until a later frame refreshes it.
//!
//! An empty store costs nothing: one `Option` test per predicted block.

use std::rc::Rc;

use nc_common::{Picture, Plane, PlaneIndex};
use nc_dsp::av1_mc::scale::Av1ScaleSizes;

/// A reference slot's normative picture and its size: `RefUpscaledWidth`
/// and `RefFrameHeight` (§7.20), which may be smaller than the picture
/// (an 8-aligned coded area past the frame edge is never read).
#[derive(Debug, Clone)]
pub(crate) struct HeldReference {
    picture: Rc<Picture>,
    width: usize,
    height: usize,
}

impl HeldReference {
    /// `picture`, whose frame is `width` x `height` luma samples.
    pub(crate) fn new(picture: Rc<Picture>, width: usize, height: usize) -> Self {
        Self {
            picture,
            width,
            height,
        }
    }

    /// `RefUpscaledWidth` and `RefFrameHeight`.
    pub(crate) fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// The picture.
    pub(crate) fn picture(&self) -> &Rc<Picture> {
        &self.picture
    }

    /// One plane and that plane's frame size (positions clamp to it).
    pub(crate) fn plane(&self, plane: PlaneIndex) -> (&Plane, (usize, usize)) {
        let shift = usize::from(plane != PlaneIndex::Y);
        let size = (
            self.width.div_ceil(1 << shift),
            self.height.div_ceil(1 << shift),
        );
        (self.picture.plane(plane), size)
    }

    /// The §7.11.3.3 scale from this reference to a frame of `frame` size.
    pub(crate) fn scale_sizes(&self, (width, height): (usize, usize)) -> Av1ScaleSizes {
        Av1ScaleSizes {
            reference: (self.width as u32, self.height as u32),
            frame: (width as u32, height as u32),
        }
    }
}

/// The held reference pictures by slot (`None`: the engine's is exact).
pub(crate) type ReferenceStore = nc_encoder::HeldSlots<HeldReference>;

/// Before a coded-size switch: keep the engine's current pictures
/// (`references[slot]`, each `size`) of every slot not already held, since
/// the engine's copies become proxies. Slots held already keep their
/// picture: the engine's copy of those was a proxy before.
pub(crate) fn hold_engine_pictures(
    store: &mut ReferenceStore,
    references: &[Option<&Picture>],
    (width, height): (usize, usize),
) {
    for (held, picture) in store.slots_mut().iter_mut().zip(references) {
        if let (None, Some(picture)) = (held.as_ref(), picture) {
            *held = Some(HeldReference::new(
                Rc::new((*picture).clone()),
                width,
                height,
            ));
        }
    }
}
