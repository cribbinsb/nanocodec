//! Reuse of picture buffers across frames.
//!
//! A decoder allocates a reconstruction and an output picture per frame and
//! drops pictures as references are replaced. [`Picture::new`] zeroes every
//! sample (borders included), which costs as much as a frame copy; keeping
//! released pictures for the next frame of the same geometry avoids both
//! the allocation and the zeroing. A reused picture holds stale samples, so
//! only code that writes every sample it later reads may take one.

use crate::error::Result;
use crate::picture::{Picture, PlaneIndex};
use crate::plane::ROW_ALIGNMENT;

/// Most pictures kept: a reconstruction, an output and two returned by the
/// caller cover one frame in flight; more would only hold memory.
const MAX_SPARE_PICTURES: usize = 4;

/// Released pictures kept for reuse by later frames.
#[derive(Debug, Default)]
pub struct PictureRecycler {
    spare: Vec<Picture>,
}

impl PictureRecycler {
    /// A `width` x `height` picture with `luma_border` (as [`Picture::new`]):
    /// a released one of exactly that geometry if available, its samples
    /// stale, else a new zeroed one.
    ///
    /// # Errors
    ///
    /// As [`Picture::new`].
    pub fn take(&mut self, width: usize, height: usize, luma_border: usize) -> Result<Picture> {
        match self
            .spare
            .iter()
            .position(|p| has_geometry(p, width, height, luma_border))
        {
            Some(index) => Ok(self.spare.swap_remove(index)),
            None => Picture::new(width, height, luma_border),
        }
    }

    /// Keep `picture` for a later [`PictureRecycler::take`] (dropped when
    /// enough are kept already).
    pub fn give(&mut self, picture: Picture) {
        if self.spare.len() < MAX_SPARE_PICTURES {
            self.spare.push(picture);
        }
    }
}

/// Whether `picture` is what `Picture::new(width, height, luma_border)`
/// would allocate (same size and padding, so the same strides).
fn has_geometry(picture: &Picture, width: usize, height: usize, luma_border: usize) -> bool {
    let luma = picture.plane(PlaneIndex::Y);
    (picture.width(), picture.height()) == (width, height)
        && luma.border_y() == luma_border
        && luma.border_x() == luma_border.next_multiple_of(ROW_ALIGNMENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_only_matching_geometry() {
        let mut recycler = PictureRecycler::default();
        let mut picture = recycler.take(16, 8, 4).unwrap();
        picture.plane_mut(PlaneIndex::Y).view_mut().row_mut(0)[0] = 7;
        recycler.give(picture);
        // Another border or size gets a fresh, zeroed picture.
        assert_eq!(
            recycler
                .take(16, 8, 0)
                .unwrap()
                .plane(PlaneIndex::Y)
                .view()
                .row(0)[0],
            0
        );
        assert_eq!(
            recycler
                .take(16, 10, 4)
                .unwrap()
                .plane(PlaneIndex::Y)
                .view()
                .row(0)[0],
            0
        );
        // The same geometry gets the released picture back, samples stale.
        let reused = recycler.take(16, 8, 4).unwrap();
        assert_eq!(reused.plane(PlaneIndex::Y).view().row(0)[0], 7);
        assert!(recycler.spare.is_empty());
    }

    #[test]
    fn keeps_a_bounded_number() {
        let mut recycler = PictureRecycler::default();
        for _ in 0..2 * MAX_SPARE_PICTURES {
            recycler.give(Picture::new(8, 8, 0).unwrap());
        }
        assert_eq!(recycler.spare.len(), MAX_SPARE_PICTURES);
    }
}
