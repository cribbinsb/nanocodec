//! Picture buffers: decoded pictures are shared with the DPB and the shown
//! picture through `Arc`s that are also listed here; once both release one
//! (the list holds its only reference) it goes to the shared
//! [`PictureRecycler`], with the output pictures callers hand back. New reconstructions and output pictures come from the
//! recycler, skipping the allocation and zeroing of a frame: both are
//! written in full before they are read.

use std::sync::Arc;

use nc_common::{ChromaFormat, Picture, PictureRecycler, Sample};

use super::Result;

/// The decoder's picture buffers.
#[derive(Debug)]
pub(super) struct PicturePool<T: Sample = u8> {
    recycler: PictureRecycler<T>,
    /// Reference pictures handed to the DPB.
    shared: Vec<Arc<Picture<T>>>,
}

impl<T: Sample> Default for PicturePool<T> {
    fn default() -> Self {
        Self {
            recycler: PictureRecycler::default(),
            shared: Vec::new(),
        }
    }
}

impl<T: Sample> PicturePool<T> {
    /// A `width` x `height` picture of chroma format `format` with
    /// `border`; its samples are stale when it is reused.
    pub fn take_with_format(
        &mut self,
        (width, height): (usize, usize),
        format: ChromaFormat,
        border: usize,
    ) -> Result<Picture<T>> {
        let (released, kept): (Vec<_>, Vec<_>) = self
            .shared
            .drain(..)
            .partition(|p| Arc::strong_count(p) == 1);
        self.shared = kept;
        for picture in released {
            if let Ok(picture) = Arc::try_unwrap(picture) {
                self.recycler.give(picture);
            }
        }
        Ok(self
            .recycler
            .take_with_format(width, height, format, border)?)
    }

    /// Share the decoded `picture` with the DPB (if it is a reference) and
    /// the caller showing it; it is reused once neither holds it.
    pub fn share(&mut self, picture: Picture<T>) -> Arc<Picture<T>> {
        let shared = Arc::new(picture);
        self.shared.push(shared.clone());
        shared
    }

    /// Take back a shared picture for writing, if nothing else holds it.
    pub fn reclaim(
        &mut self,
        picture: Arc<Picture<T>>,
    ) -> std::result::Result<Picture<T>, Arc<Picture<T>>> {
        self.shared.retain(|p| !Arc::ptr_eq(p, &picture));
        Arc::try_unwrap(picture).inspect_err(|shared| self.shared.push(shared.clone()))
    }

    /// Keep a picture nothing else uses.
    pub fn give_back(&mut self, picture: Picture<T>) {
        self.recycler.give(picture);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address(picture: &Picture) -> *const u8 {
        picture
            .plane(nc_common::PlaneIndex::Y)
            .view()
            .data()
            .as_ptr()
    }

    #[test]
    fn released_references_are_reused() {
        let mut pool = PicturePool::<u8>::default();
        let picture = pool
            .take_with_format((16, 16), ChromaFormat::Yuv420, 0)
            .unwrap();
        let shared = pool.share(picture);
        let pointer = address(&shared);
        // Still referenced: a new buffer.
        assert_ne!(
            address(
                &pool
                    .take_with_format((16, 16), ChromaFormat::Yuv420, 0)
                    .unwrap()
            ),
            pointer
        );
        drop(shared);
        assert_eq!(
            address(
                &pool
                    .take_with_format((16, 16), ChromaFormat::Yuv420, 0)
                    .unwrap()
            ),
            pointer
        );
    }
}
