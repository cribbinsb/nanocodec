//! Picture buffers: decoded pictures are shared with the DPB and the shown
//! picture through `Arc`s that are also listed here; once both release one
//! (the list holds its only reference) it goes to the shared
//! [`PictureRecycler`], with the output pictures callers hand back. New reconstructions and output pictures come from the
//! recycler, skipping the allocation and zeroing of a frame: both are
//! written in full before they are read.

use std::sync::Arc;

use nc_common::{Picture, PictureRecycler};

use super::Result;

/// The decoder's picture buffers.
#[derive(Debug, Default)]
pub(super) struct PicturePool {
    recycler: PictureRecycler,
    /// Reference pictures handed to the DPB.
    shared: Vec<Arc<Picture>>,
}

impl PicturePool {
    /// A `width` x `height` picture with `border`; its samples are stale
    /// when it is reused.
    pub fn take(&mut self, width: usize, height: usize, border: usize) -> Result<Picture> {
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
        Ok(self.recycler.take(width, height, border)?)
    }

    /// Share the decoded `picture` with the DPB (if it is a reference) and
    /// the caller showing it; it is reused once neither holds it.
    pub fn share(&mut self, picture: Picture) -> Arc<Picture> {
        let shared = Arc::new(picture);
        self.shared.push(shared.clone());
        shared
    }

    /// Keep a picture nothing else uses.
    pub fn give_back(&mut self, picture: Picture) {
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
        let mut pool = PicturePool::default();
        let picture = pool.take(16, 16, 0).unwrap();
        let shared = pool.share(picture);
        let pointer = address(&shared);
        // Still referenced: a new buffer.
        assert_ne!(address(&pool.take(16, 16, 0).unwrap()), pointer);
        drop(shared);
        assert_eq!(address(&pool.take(16, 16, 0).unwrap()), pointer);
    }
}
