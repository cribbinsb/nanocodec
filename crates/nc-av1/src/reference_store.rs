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
use nc_encoder::{MAX_REFERENCE_SLOTS, SlotMask};

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
#[derive(Debug, Default)]
pub(crate) struct ReferenceStore {
    slots: [Option<HeldReference>; MAX_REFERENCE_SLOTS],
}

impl ReferenceStore {
    /// The picture slot `slot` predicts from, when it is held here.
    pub(crate) fn held(&self, slot: usize) -> Option<&HeldReference> {
        self.slots[slot].as_ref()
    }

    /// Before a coded-size switch: keep the engine's current pictures
    /// (`references[slot]`, each `size`) of every slot not already held,
    /// since the engine's copies become proxies. Slots held already keep
    /// their picture: the engine's copy of those was a proxy before.
    pub(crate) fn hold_engine_pictures(
        &mut self,
        references: &[Option<&Picture>],
        (width, height): (usize, usize),
    ) {
        for (held, picture) in self.slots.iter_mut().zip(references) {
            if let (None, Some(picture)) = (held.as_ref(), picture) {
                *held = Some(HeldReference::new(
                    Rc::new((*picture).clone()),
                    width,
                    height,
                ));
            }
        }
    }

    /// A frame was committed: every slot it refreshes (all of them for a
    /// key frame) now holds `picture`, or nothing when the engine's
    /// reconstruction is the frame's normative picture.
    pub(crate) fn commit(&mut self, key: bool, refresh: SlotMask, picture: Option<&HeldReference>) {
        for (slot, held) in self.slots.iter_mut().enumerate() {
            if key || refresh.contains(slot) {
                *held = picture.cloned();
            }
        }
    }

    /// Spatial-SVC commit: lower layers (`S0`, and `S1` in 3-layer SVC) store
    /// `picture` so the layer immediately above predicts normatively via
    /// §7.11.3.3 scaled prediction, while the top layer stores `None` (its
    /// engine pool holds exact pixels; intermediate layers bypass `store` on
    /// their own unscaled temporal slots via `BackendSpatial::slot_layer`).
    pub(crate) fn commit_spatial(
        &mut self,
        refresh: SlotMask,
        refresh_all: bool,
        picture: Option<&HeldReference>,
    ) {
        for (slot, held) in self.slots.iter_mut().enumerate() {
            if refresh_all || refresh.contains(slot) {
                match picture {
                    Some(pic) => {
                        if let Some(existing) = held {
                            existing.clone_from(pic);
                        } else {
                            *held = Some(pic.clone());
                        }
                    }
                    None => *held = None,
                }
            }
        }
    }
}

/// Whether a frame of `frame` size may predict from a reference of
/// `reference` size: at most 2x smaller and 16x larger per dimension
/// (§7.11.3.3 / §7.20 conformance requirements).
pub(crate) fn reference_scale_allowed(reference: (usize, usize), frame: (usize, usize)) -> bool {
    let fits = |r: usize, f: usize| 2 * f >= r && f <= 16 * r;
    fits(reference.0, frame.0) && fits(reference.1, frame.1)
}
