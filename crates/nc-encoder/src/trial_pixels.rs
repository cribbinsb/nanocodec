//! Packed 4:2:0 reconstruction for reversible encoder trials.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::Dsp;
use std::ops::Range;

/// A block's packed reconstruction, with storage reused between trials.
#[derive(Debug, Default)]
pub struct TrialPixels(Vec<u8>);

/// A rectangle in luma samples.
type Rect = (usize, usize, usize, usize);

fn planes((x, y, width, height): Rect) -> impl Iterator<Item = (PlaneIndex, Rect, Range<usize>)> {
    let mut offset = 0;
    PlaneIndex::ALL.into_iter().map(move |plane| {
        let shift = usize::from(plane != PlaneIndex::Y);
        let (w, h) = (width >> shift, height >> shift);
        let start = offset;
        offset += w * h;
        (plane, (x >> shift, y >> shift, w, h), start..offset)
    })
}

impl TrialPixels {
    /// Save the reconstruction of `rect` in all three planes.
    pub fn save(&mut self, dsp: &Dsp, picture: &Picture, rect: Rect) {
        let (_, _, width, height) = rect;
        let samples = width * height + 2 * (width >> 1) * (height >> 1);
        self.0.resize(samples, 0);
        copy_out(dsp, picture, rect, &mut self.0);
    }

    /// Put the saved reconstruction of `rect` back.
    pub fn restore(&self, dsp: &Dsp, picture: &mut Picture, rect: Rect) {
        copy_in(dsp, picture, rect, &self.0);
    }
}

fn copy_out(dsp: &Dsp, picture: &Picture, rect: Rect, packed: &mut [u8]) {
    for (plane, (x, y, w, h), range) in planes(rect) {
        let src = picture
            .plane(plane)
            .view()
            .sub_view(x, y, w, h)
            .expect("trial lies inside the picture");
        let mut dst = PlaneViewMut::new(&mut packed[range], w, w, h).expect("packed trial plane");
        dsp.copy_plane(&src, &mut dst);
    }
}

fn copy_in(dsp: &Dsp, picture: &mut Picture, rect: Rect, packed: &[u8]) {
    for (plane, (x, y, w, h), range) in planes(rect) {
        let src = PlaneView::new(&packed[range], w, w, h).expect("packed trial plane");
        let mut view = picture.plane_mut(plane).view_mut();
        let mut dst = view
            .sub_view_mut(x, y, w, h)
            .expect("trial lies inside the picture");
        dsp.copy_plane(&src, &mut dst);
    }
}
