//! Inter prediction sample interpolation (H.264 §8.4.2.2) through the
//! nc-dsp kernels (`Dsp::predict` with `Filter::H264Luma` / `H264Chroma`).
//!
//! Luma: the 6-tap filter (1, −5, 20, 20, −5, 1) for half samples and
//! averages for quarter samples (§8.4.2.2.1). Chroma (4:2:0): bilinear at
//! 1/8 sample (§8.4.2.2.2). The decoder clamps reference coordinates to the
//! picture (eq. 8-228..8-231). The reference picture's border is an edge
//! extension of the (deblocked) picture, so reading inside the border is the
//! same as clamping; windows reaching beyond it are copied with explicit
//! clamping instead.

#[cfg(test)]
mod reference;

use nc_common::{Picture, Plane, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, H264Predictor, copy_square_view};
use nc_encoder::Mv;

/// Luma macroblock size.
const MB: usize = 16;
/// Chroma block size of a macroblock (4:2:0).
const MB_CHROMA: usize = 8;
/// Fractional bits of a luma vector (quarter samples) and of the same vector
/// read as a 4:2:0 chroma vector (eighth samples).
const LUMA_FRAC_BITS: i32 = 2;
const CHROMA_FRAC_BITS: i32 = 3;

/// The largest window a macroblock's prediction reads: luma, 16 + 6 - 1.
const LUMA_WINDOW: usize = MB + 5;
const LUMA_WINDOW_AREA: usize = LUMA_WINDOW * LUMA_WINDOW;

/// The H.264 prediction kernels a macroblock uses, resolved once per
/// encoder (`Dsp::h264_predictor`) rather than per call: resolving them per
/// call (filter checks, block-size validation, table lookup) measured about
/// 25 instructions of the ~130 a prediction call spent outside its kernel.
#[derive(Debug, Clone, Copy)]
pub struct McKernels {
    luma_16: H264Predictor,
    luma_8: H264Predictor,
    chroma_8: H264Predictor,
    chroma_4: H264Predictor,
}

impl McKernels {
    /// The kernels of `dsp` for the macroblock and quadrant sizes.
    pub fn new(dsp: &Dsp) -> Self {
        let size = |n| BlockSize::new(n, n).expect("macroblock sizes are kernel sizes");
        Self {
            luma_16: dsp.h264_predictor(Filter::H264Luma, size(MB)),
            luma_8: dsp.h264_predictor(Filter::H264Luma, size(MB / 2)),
            chroma_8: dsp.h264_predictor(Filter::H264Chroma, size(MB_CHROMA)),
            chroma_4: dsp.h264_predictor(Filter::H264Chroma, size(MB_CHROMA / 2)),
        }
    }

    /// Predict the 16x16 luma block at `(x, y)` of `reference` displaced by
    /// `mv` (quarter samples), into `out` (raster).
    pub fn luma_16x16(&self, reference: &Plane, x: usize, y: usize, mv: Mv, out: &mut [u8; 256]) {
        predict::<MB>(
            &self.luma_16,
            Filter::H264Luma,
            LUMA_FRAC_BITS,
            reference,
            (x, y),
            mv,
            out,
        );
    }

    /// Predict the 8x8 chroma block at chroma position `(x, y)` with the
    /// luma vector `mv`, into `out` (raster).
    pub fn chroma_8x8(&self, reference: &Plane, x: usize, y: usize, mv: Mv, out: &mut [u8; 64]) {
        predict::<MB_CHROMA>(
            &self.chroma_8,
            Filter::H264Chroma,
            CHROMA_FRAC_BITS,
            reference,
            (x, y),
            mv,
            out,
        );
    }

    /// Predict the 8x8 luma quadrant at `(x, y)` displaced by `mv`, into
    /// `out` (raster, stride 8).
    pub fn luma_8x8(&self, reference: &Plane, (x, y): (usize, usize), mv: Mv, out: &mut [u8; 64]) {
        predict::<{ MB / 2 }>(
            &self.luma_8,
            Filter::H264Luma,
            LUMA_FRAC_BITS,
            reference,
            (x, y),
            mv,
            out,
        );
    }

    /// Predict the 4x4 chroma block at chroma position `(x, y)` with the
    /// luma vector `mv`, into `out` (raster, stride 4).
    pub fn chroma_4x4(
        &self,
        reference: &Plane,
        (x, y): (usize, usize),
        mv: Mv,
        out: &mut [u8; 16],
    ) {
        predict::<{ MB_CHROMA / 2 }>(
            &self.chroma_4,
            Filter::H264Chroma,
            CHROMA_FRAC_BITS,
            reference,
            (x, y),
            mv,
            out,
        );
    }
}

/// The prediction of macroblock `(x, y)` with `mv` when it needs no
/// interpolation: the reference picture's own 16x16 luma and 8x8 chroma
/// blocks. That holds when `mv` is a whole chroma sample (and so a whole luma
/// sample): both components multiples of 8 quarter-luma units. `None` for
/// other vectors, or blocks reaching beyond the reference's padded area (its
/// border is an edge extension, so reading inside it equals the decoder's
/// clamping).
pub fn whole_sample_macroblock(
    reference: &Picture,
    (x, y): (usize, usize),
    mv: Mv,
) -> Option<(PlaneView<'_>, [PlaneView<'_>; 2])> {
    let whole = 1 << CHROMA_FRAC_BITS;
    if mv.x % whole != 0 || mv.y % whole != 0 {
        return None;
    }
    let (dx, dy) = (
        isize::from(mv.x) >> LUMA_FRAC_BITS,
        isize::from(mv.y) >> LUMA_FRAC_BITS,
    );
    let luma = reference
        .plane(PlaneIndex::Y)
        .region(x as isize + dx, y as isize + dy, MB, MB)?;
    let (cx, cy) = ((x / 2) as isize + dx / 2, (y / 2) as isize + dy / 2);
    let u = reference
        .plane(PlaneIndex::U)
        .region(cx, cy, MB_CHROMA, MB_CHROMA)?;
    let v = reference
        .plane(PlaneIndex::V)
        .region(cx, cy, MB_CHROMA, MB_CHROMA)?;
    Some((luma, [u, v]))
}

/// One square `size` prediction with `predictor` (`filter`'s kernel for that
/// size); `frac_bits` fractional bits in each vector component.
#[allow(
    clippy::inline_always,
    reason = "a few calls per macroblock; inlined, the window arithmetic folds into the caller's constants"
)]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn predict<const N: usize>(
    predictor: &H264Predictor,
    filter: Filter,
    frac_bits: i32,
    reference: &Plane,
    (x, y): (usize, usize),
    mv: Mv,
    out: &mut [u8],
) {
    let size = N;
    let (mvx, mvy) = (i32::from(mv.x), i32::from(mv.y));
    let frac_mask = (1 << frac_bits) - 1;
    if (mvx | mvy) & frac_mask == 0
        && copy_whole_sample::<N>(
            reference,
            x as isize + (mvx >> frac_bits) as isize,
            y as isize + (mvy >> frac_bits) as isize,
            out,
        )
    {
        return;
    }
    let side = size + filter.taps() - 1;
    let left = x as isize + (mvx >> frac_bits) as isize - filter.before() as isize;
    let top = y as isize + (mvy >> frac_bits) as isize - filter.before() as isize;
    let (frac_x, frac_y) = ((mvx & frac_mask) as usize, (mvy & frac_mask) as usize);
    if let Some(window) = reference.region(left, top, side, side) {
        predictor.predict(window.data(), window.stride(), frac_x, frac_y, out, size);
    } else {
        let mut buf = [0u8; LUMA_WINDOW_AREA];
        let window = clamped_window(reference, left, top, side, &mut buf);
        predictor.predict(window.data(), window.stride(), frac_x, frac_y, out, size);
    }
}

/// A whole-sample vector predicts the reference block itself (every filter
/// phase 0 is the identity: §8.4.2.2.1 eq. 8-241, §8.4.2.2.2 eq. 8-266 with
/// weight 64). Copied directly when the block lies inside the reference's
/// padded area: the chroma kernels spent ~164 instructions per 8x8 block at
/// phase 0, about 0.7M per static 1080x720 P frame (mostly zero-vector
/// skips). Returns false, leaving `out` alone, when the block leaves the
/// padded area.
fn copy_whole_sample<const N: usize>(
    reference: &Plane,
    left: isize,
    top: isize,
    out: &mut [u8],
) -> bool {
    let Some(block) = reference.region(left, top, N, N) else {
        return false;
    };
    let mut dst = PlaneViewMut::new(out, N, N, N).expect("prediction covers the block");
    copy_square_view::<N>(&mut dst, &block);
    true
}

fn clamped_window<'a>(
    reference: &Plane,
    left: isize,
    top: isize,
    side: usize,
    buf: &'a mut [u8],
) -> PlaneView<'a> {
    let picture = reference.view();
    let (w, h) = (picture.width() as isize, picture.height() as isize);
    for (r, row) in buf.chunks_exact_mut(side).take(side).enumerate() {
        let src = picture.row((top + r as isize).clamp(0, h - 1) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = src[(left + c as isize).clamp(0, w - 1) as usize];
        }
    }
    PlaneView::new(&buf[..side * side], side, side, side).expect("window fits the buffer")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane with a padded border filled by edge extension, as the
    /// engine's reference pictures are.
    fn noisy_plane(w: usize, h: usize, border: usize, seed: u32) -> Plane {
        let mut plane = Plane::new(w, h, border).unwrap();
        let mut state = seed;
        let mut view = plane.view_mut();
        for y in 0..h {
            for v in view.row_mut(y) {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (state >> 24) as u8;
            }
        }
        plane.extend_borders();
        plane
    }

    /// The nc-dsp kernels match the per-sample reference at every fraction,
    /// inside the border and far outside the picture.
    #[test]
    fn kernels_match_reference() {
        let mc = McKernels::new(&Dsp::new());
        let luma = noisy_plane(64, 48, 32, 7);
        let chroma = noisy_plane(32, 24, 16, 11);
        let vectors = [(0, 0), (5, -3), (-70, 9), (13, 22), (-300, -260), (401, 77)];
        for (vx, vy) in vectors {
            for fx in 0..4 {
                for fy in 0..4 {
                    let mv = Mv {
                        x: vx * 4 + fx,
                        y: vy * 4 + fy,
                    };
                    let (mut got, mut want) = ([0u8; 256], [0u8; 256]);
                    mc.luma_16x16(&luma, 16, 16, mv, &mut got);
                    reference::predict_luma_16x16(&luma.view(), 16, 16, mv, &mut want);
                    assert_eq!(got, want, "luma mv {mv:?}");
                    let (mut got, mut want) = ([0u8; 64], [0u8; 64]);
                    let cmv = Mv {
                        x: mv.x * 2 + 1,
                        y: mv.y,
                    };
                    mc.chroma_8x8(&chroma, 8, 8, cmv, &mut got);
                    reference::predict_chroma_8x8(&chroma.view(), 8, 8, cmv, &mut want);
                    assert_eq!(got, want, "chroma mv {cmv:?}");
                }
            }
        }
    }
}
