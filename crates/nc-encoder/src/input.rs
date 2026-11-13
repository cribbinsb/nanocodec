//! Camera-frame ingest: writes a borrowed [`RawFrame`] into the encoder's
//! planar source picture.
//!
//! This is the one copy every input makes (into the analyzer's bordered
//! buffer, see [`Encoder::source_mut`](crate::Encoder::source_mut)); planar
//! layouts are row copies and NV12 chroma is split by the
//! `deinterleave_uv` kernel on the way, so no format needs a second pass.

use nc_common::{Picture, PlaneIndex, RawFrame};
use nc_dsp::Dsp;

/// Write `frame` into the visible area of `dst`.
///
/// # Panics
///
/// If the frame's planes do not match `dst`'s (callers check the frame
/// with [`RawFrame::check`] first).
pub(crate) fn write_raw(dsp: &Dsp, frame: &RawFrame<'_>, dst: &mut Picture) {
    dsp.copy_plane(&frame.luma(), &mut dst.plane_mut(PlaneIndex::Y).view_mut());
    match frame {
        RawFrame::I420 { u, v, .. } => {
            dsp.copy_plane(u, &mut dst.plane_mut(PlaneIndex::U).view_mut());
            dsp.copy_plane(v, &mut dst.plane_mut(PlaneIndex::V).view_mut());
        }
        RawFrame::Nv12 { uv, .. } => {
            let [_, cb, cr] = dst.planes_mut();
            dsp.deinterleave_uv(uv, &mut cb.view_mut(), &mut cr.view_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use nc_common::{Picture, PlaneIndex, RawFrame};
    use nc_dsp::Dsp;

    use super::write_raw;

    /// A 4x2 frame: luma 0..8, Cb 100, 101, Cr 200, 201.
    const LUMA: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    fn planes(picture: &Picture) -> Vec<Vec<u8>> {
        PlaneIndex::ALL
            .iter()
            .map(|&p| picture.plane(p).view().rows().flatten().copied().collect())
            .collect()
    }

    #[test]
    fn nv12_and_i420_give_the_same_picture() {
        let nv12: Vec<u8> = LUMA.iter().copied().chain([100, 200, 101, 201]).collect();
        let i420: Vec<u8> = LUMA.iter().copied().chain([100, 101, 200, 201]).collect();
        let dsp = Dsp::new();
        let mut from_nv12 = Picture::new(4, 2, 8).unwrap();
        let mut from_i420 = Picture::new(4, 2, 8).unwrap();
        write_raw(
            &dsp,
            &RawFrame::packed_nv12(&nv12, 4, 2).unwrap(),
            &mut from_nv12,
        );
        write_raw(
            &dsp,
            &RawFrame::packed_i420(&i420, 4, 2).unwrap(),
            &mut from_i420,
        );
        assert_eq!(planes(&from_nv12), planes(&from_i420));
        assert_eq!(planes(&from_nv12)[2], [200, 201]);
    }
}
