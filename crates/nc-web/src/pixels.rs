//! Conversions between browser RGBA and I420 (limited-range BT.601 input,
//! display in the samples' own matrix, `nc_dsp::DisplayMatrix`), and the
//! packed I420 layout handed to a WebGL display (which applies the same
//! matrix in its shader, `web/renderer.js`) and between workers.

use nc_common::{BitDepths, Picture, PlaneIndex, ShownPicture};
use nc_dsp::{DisplayMatrix, Dsp};

/// Fill the visible samples of `picture` from packed RGBA (limited-range
/// BT.601, chroma from each 2x2 group's mean) with this instance's SIMD
/// kernels. The shared engine copies visible input and extends its own
/// analysis/reference borders; display PSNR also reads only visible samples.
pub(crate) fn from_rgba(dsp: &Dsp, rgba: &[u8], picture: &mut Picture) {
    let [luma, cb, cr] = picture.planes_mut();
    dsp.rgba_to_i420(
        rgba,
        &mut luma.view_mut(),
        &mut cb.view_mut(),
        &mut cr.view_mut(),
    );
}

/// Display conversion of the visible picture into `rgba` (exactly
/// `4 * width * height` bytes), with this instance's SIMD kernels.
pub(crate) fn to_rgba(
    dsp: &Dsp,
    matrix: DisplayMatrix,
    picture: ShownPicture<'_>,
    rgba: &mut [u8],
) {
    dsp.i420_to_rgba(
        matrix,
        &picture.plane(PlaneIndex::Y),
        &picture.plane(PlaneIndex::U),
        &picture.plane(PlaneIndex::V),
        rgba,
    );
}

/// Replace `bytes` with the visible planes of `picture`, Y then Cb then Cr,
/// each row without padding.
pub(crate) fn pack_i420(picture: ShownPicture<'_>, bytes: &mut Vec<u8>) {
    bytes.clear();
    bytes.reserve(picture.packed_frame_len());
    for plane in PlaneIndex::ALL {
        picture
            .plane(plane)
            .rows()
            .for_each(|row| bytes.extend_from_slice(row));
    }
}

/// Truncate deep samples to display I420 without changing decoder references.
/// Each plane uses its own depth (HEVC can signal 10-bit luma/9-bit chroma).
/// The caller reuses an eight-bit picture of exactly the visible size.
pub(crate) fn downshift(source: ShownPicture<'_, u16>, depths: BitDepths, target: &mut Picture) {
    for plane in PlaneIndex::ALL {
        let src = source.plane(plane);
        let shift = depths.plane(plane as usize).above_eight();
        let mut dst = target.plane_mut(plane).view_mut();
        for y in 0..src.height() {
            for (out, &sample) in dst.row_mut(y).iter_mut().zip(src.row(y)) {
                *out = (sample >> shift) as u8;
            }
        }
    }
}

/// Fill `picture` from packed I420 (`pack_i420`'s layout; exactly
/// `packed_frame_len` bytes, which the caller checks).
pub(crate) fn unpack_i420(bytes: &[u8], picture: &mut Picture) {
    let mut rest = bytes;
    for plane in picture.planes_mut() {
        let mut view = plane.view_mut();
        let width = view.width();
        for y in 0..view.height() {
            let (row, tail) = rest.split_at(width);
            view.row_mut(y).copy_from_slice(row);
            rest = tail;
        }
    }
}

/// Match the harness's average of per-plane PSNR, weighted 6:1:1, capped at
/// 100 dB. The squared errors come from this instance's SIMD kernel.
pub(crate) fn psnr(dsp: &Dsp, reference: &Picture, decoded: ShownPicture<'_>) -> f64 {
    let mut score = 0.0;
    for (index, weight) in [
        (PlaneIndex::Y, 6.0),
        (PlaneIndex::U, 1.0),
        (PlaneIndex::V, 1.0),
    ] {
        let a = reference.plane(index).view();
        let sum = dsp.sse(&a, &decoded.plane(index));
        let mse = sum as f64 / (a.width() * a.height()) as f64;
        let value = if sum == 0 {
            MAX_PSNR_DB
        } else {
            (10.0 * (PEAK_SQUARED / mse).log10()).min(MAX_PSNR_DB)
        };
        score += weight * value;
    }
    score / 8.0
}

/// PSNR of identical planes, and the cap of every plane's PSNR.
const MAX_PSNR_DB: f64 = 100.0;
/// The squared 8-bit peak, 255^2.
const PEAK_SQUARED: f64 = 65025.0;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neutral_extremes_and_psnr() {
        let mut p = Picture::new(4, 4, 0).unwrap();
        from_rgba(&Dsp::new(), &[0; 64], &mut p);
        assert_eq!(p.plane(PlaneIndex::Y).view().row(0), [16; 4]);
        assert_eq!(p.plane(PlaneIndex::U).view().row(0), [128; 2]);
        let mut rgba = [0; 64];
        let bt601 = DisplayMatrix::Bt601Limited;
        to_rgba(&Dsp::new(), bt601, ShownPicture::whole(&p), &mut rgba);
        assert!(rgba.as_chunks::<4>().0.iter().all(|c| *c == [0, 0, 0, 255]));
        from_rgba(&Dsp::new(), &[255; 64], &mut p);
        assert_eq!(p.plane(PlaneIndex::Y).view().row(0), [235; 4]);
        to_rgba(&Dsp::new(), bt601, ShownPicture::whole(&p), &mut rgba);
        assert!(rgba.as_chunks::<4>().0.iter().all(|c| *c == [255; 4]));
        assert_eq!(psnr(&Dsp::new(), &p, ShownPicture::whole(&p)), 100.0);
    }

    #[test]
    fn psnr_weights_planes_six_one_one() {
        let mut reference = Picture::new(4, 4, 0).unwrap();
        from_rgba(&Dsp::new(), &[0; 64], &mut reference);
        let mut decoded = reference.clone();
        // Every luma sample off by one: luma MSE 1, chroma exact.
        for y in 0..4 {
            decoded
                .plane_mut(PlaneIndex::Y)
                .view_mut()
                .row_mut(y)
                .fill(17);
        }
        let luma = 10.0 * PEAK_SQUARED.log10();
        let expected = (6.0 * luma + 2.0 * MAX_PSNR_DB) / 8.0;
        let measured = psnr(&Dsp::new(), &reference, ShownPicture::whole(&decoded));
        assert!(
            (measured - expected).abs() < 1e-9,
            "{measured} vs {expected}"
        );
    }

    #[test]
    fn packed_i420_has_no_row_padding() {
        // A border makes every plane's stride wider than its row.
        let mut p = Picture::new(6, 4, 8).unwrap();
        from_rgba(&Dsp::new(), &[255; 96], &mut p);
        let mut bytes = vec![1; 7];
        pack_i420(ShownPicture::whole(&p), &mut bytes);
        assert_eq!(bytes.len(), 6 * 4 + 2 * 3 * 2);
        assert!(bytes[..24].iter().all(|&y| y == 235));
        assert!(bytes[24..].iter().all(|&c| c == 128));
        let mut unpacked = Picture::new(6, 4, 3).unwrap();
        unpack_i420(&bytes, &mut unpacked);
        let mut again = Vec::new();
        pack_i420(ShownPicture::whole(&unpacked), &mut again);
        assert_eq!(again, bytes);
    }
    #[test]
    fn deep_display_uses_each_plane_depth_and_visible_window() {
        use nc_common::{BitDepth, BitDepths};
        let mut source = Picture::<u16>::new(8, 6, 8).unwrap();
        let depths = BitDepths {
            luma: BitDepth::TEN,
            chroma: BitDepth::new(9).unwrap(),
        };
        for plane in PlaneIndex::ALL {
            let mut view = source.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                    *sample =
                        ((x * 43 + y * 71) as i32 & depths.plane(plane as usize).max()) as u16;
                }
            }
        }
        let shown = ShownPicture::window(&source, (2, 2), (4, 2)).unwrap();
        let mut target = Picture::new(4, 2, 0).unwrap();
        downshift(shown, depths, &mut target);
        for plane in PlaneIndex::ALL {
            let from = shown.plane(plane);
            let to = target.plane(plane).view();
            for y in 0..from.height() {
                for (a, &b) in to.row(y).iter().zip(from.row(y)) {
                    assert_eq!(*a, (b >> depths.plane(plane as usize).above_eight()) as u8);
                }
            }
        }
    }

    #[test]
    fn deep_display_white_reaches_255_at_every_supported_depth() {
        use nc_common::{BitDepth, BitDepths};
        for bits in [8, 9, 10, 12, 16] {
            let depth = BitDepth::new(bits).unwrap();
            let mut source = Picture::<u16>::new(4, 2, 0).unwrap();
            for plane in source.planes_mut() {
                plane.view_mut().fill(depth.max() as u16);
            }
            let mut target = Picture::new(4, 2, 0).unwrap();
            downshift(
                ShownPicture::whole(&source),
                BitDepths {
                    luma: depth,
                    chroma: depth,
                },
                &mut target,
            );
            let mut packed = Vec::new();
            pack_i420(ShownPicture::whole(&target), &mut packed);
            assert!(packed.iter().all(|&v| v == 255));
        }
    }
}
