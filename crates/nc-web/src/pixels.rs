//! Explicit limited-range BT.601 conversion between browser RGBA and I420,
//! and the packed I420 layout handed to a WebGL display (which applies the
//! same BT.601 matrix in its shader, `web/renderer.js`).

use nc_common::{Picture, PlaneIndex, ShownPicture};
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
pub(crate) fn to_rgba(dsp: &Dsp, picture: ShownPicture<'_>, rgba: &mut [u8]) {
    dsp.i420_to_rgba(
        DisplayMatrix::Bt601Limited,
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
        to_rgba(&Dsp::new(), ShownPicture::whole(&p), &mut rgba);
        assert!(rgba.as_chunks::<4>().0.iter().all(|c| *c == [0, 0, 0, 255]));
        from_rgba(&Dsp::new(), &[255; 64], &mut p);
        assert_eq!(p.plane(PlaneIndex::Y).view().row(0), [235; 4]);
        to_rgba(&Dsp::new(), ShownPicture::whole(&p), &mut rgba);
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
    }
}
