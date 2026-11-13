//! The source of a stream whose pictures are coded smaller than the stream
//! (AV1 superres: [`crate::CodecBackend::frame_size`]; adaptive coded
//! resolution, `resolution.rs`; the base layer of spatial SVC,
//! `spatial_encode.rs`): the caller's frames arrive at the stream size and
//! are resampled to the coded picture size before analysis and coding. The
//! decoder (superres) or the player (render size) scales back.
//!
//! An exact halving, which adaptive resolution runs on every frame while
//! switched down, uses the 2:1 kernel of the analysis pyramid
//! (`Dsp::downscale_2x`, a 2x2 mean centred like the half-size grid; SIMD
//! on every target, checked by nc-checkasm). It codes as well as a smoother
//! [1 3 3 1] filter: 640x360 AV1 at 100 kbps scored at 720p, PSNR-Y /
//! MS-SSIM, box vs ffmpeg bilinear: Park Joy 20.03 / 0.623 vs 20.02 /
//! 0.622, Crowd Run 20.58 / 0.648 vs 20.59 / 0.640, Food Market 27.27 /
//! 0.790 vs 27.14 / 0.784 (docs/handoff/adaptive-resolution.md).
//!
//! Any other ratio uses centre-aligned Lanczos-3. Single-layer streams
//! (superres, adaptive resolution, reference proxies) keep nc-dsp's scalar
//! `resample_plane`, so their output cannot depend on the SIMD resampler.
//! Spatial SVC resamples every frame (`1280x720 -> 640x360` or `854x480` and
//! `2:1` upscaled reference proxies) with the cached-table SIMD Lanczos-3 path
//! ([`PictureResampler`], `Dsp::resample_plane_with`), which `nc-checkasm` and
//! `nc-dsp`'s resample tests pin bit-exact to the scalar reference; using
//! Lanczos-3 instead of the `2x2` box filter for the `2:1` base layer avoids
//! +11.7% BD-rate loss on `L2T1` (`docs/handoff/spatial-svc.md` §4).

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::resample::resample_plane;
use nc_dsp::{Dsp, PlaneResampler};

/// How one plane is resampled between two fixed sizes.
#[derive(Debug, Clone)]
enum PlaneScale {
    /// Same size: plain copy.
    Copy,
    /// Any non-identity ratio: cached Lanczos-3 polyphase tables and scratch.
    Lanczos(PlaneResampler),
    /// An exact 2:1 upscale with the centre-aligned [1 3]/4 bilinear filter in
    /// each direction (spatial SVC's search proxy, `spatial.proxy_bilinear`),
    /// with three rows of horizontally upscaled samples (x4) as scratch.
    Bilinear2x([Vec<u16>; 3]),
    /// An exact 2:1 downscale by the 2x2 mean (`Dsp::downscale_2x`, the analysis
    /// pyramid's SIMD kernel): spatial SVC's base source under
    /// `spatial.base_box_downscale`.
    Box2x,
}

impl PlaneScale {
    /// The method for a plane of `src` samples resampled to `dst` samples.
    fn between(src: (usize, usize), dst: (usize, usize)) -> Self {
        if src == dst {
            Self::Copy
        } else {
            Self::Lanczos(PlaneResampler::new(src, dst))
        }
    }
}

/// Per-plane resamplers between two fixed 4:2:0 picture sizes, for spatial
/// SVC's every-frame base-layer downscale and upscaled proxies.
#[derive(Debug, Clone)]
pub(crate) struct PictureResampler {
    dst_size: (usize, usize),
    planes: [PlaneScale; 3],
}

impl PictureResampler {
    /// Prepare resampling from `(src_w, src_h)` 4:2:0 to `(dst_w, dst_h)` 4:2:0.
    pub(crate) fn new((src_w, src_h): (usize, usize), (dst_w, dst_h): (usize, usize)) -> Self {
        let chroma = |w: usize, h: usize| (w.div_ceil(2), h.div_ceil(2));
        let luma = PlaneScale::between((src_w, src_h), (dst_w, dst_h));
        let chroma_scale = PlaneScale::between(chroma(src_w, src_h), chroma(dst_w, dst_h));
        Self {
            dst_size: (dst_w, dst_h),
            planes: [luma, chroma_scale.clone(), chroma_scale],
        }
    }

    /// Resample `src` into the top-left `dst_size` of `dst` (with zero heap
    /// allocations), edge-extending into `dst`'s right/bottom padding when
    /// `dst` is a padded coded-resolution buffer.
    pub(crate) fn scale(&mut self, dsp: &Dsp, src: &Picture, dst: &mut Picture) {
        let padded = (dst.width(), dst.height()) != self.dst_size;
        for (plane, method) in PlaneIndex::ALL.into_iter().zip(&mut self.planes) {
            let ss = usize::from(plane != PlaneIndex::Y);
            let (out_w, out_h) = ((self.dst_size.0 + ss) >> ss, (self.dst_size.1 + ss) >> ss);
            let (from, mut to) = (src.plane(plane).view(), dst.plane_mut(plane).view_mut());
            match method {
                PlaneScale::Copy => {
                    for y in 0..out_h {
                        to.row_mut(y)[..out_w].copy_from_slice(&from.row(y)[..out_w]);
                    }
                }
                PlaneScale::Lanczos(resampler) => {
                    dsp.resample_plane_with(resampler, &from, &mut to);
                }
                PlaneScale::Bilinear2x(rows) => upscale_2x_bilinear(&from, &mut to, rows),
                PlaneScale::Box2x => {
                    let mut out = to
                        .sub_view_mut(0, 0, out_w, out_h)
                        .expect("inside the plane");
                    dsp.downscale_2x(&from, &mut out);
                }
            }
        }
        if padded {
            dst.replicate_beyond(self.dst_size.0, self.dst_size.1);
        }
    }

    /// Exact 2:1 upscaling from `(src_w, src_h)` to `(dst_w, dst_h)` = twice it,
    /// bilinear (see [`PlaneScale::Bilinear2x`]); only [`Self::scale_luma_only`]
    /// is meant for it (spatial SVC's search proxy), so chroma keeps Lanczos-3.
    pub(crate) fn bilinear_upscale_2x(src: (usize, usize), dst: (usize, usize)) -> Self {
        debug_assert_eq!((src.0 * 2, src.1 * 2), dst);
        let mut resampler = Self::new(src, dst);
        resampler.planes[0] = PlaneScale::Bilinear2x(std::array::from_fn(|_| vec![0; dst.0]));
        resampler
    }

    /// Exact 2:1 downscaling of every plane by the 2x2 mean ([`PlaneScale::Box2x`]).
    pub(crate) fn box_downscale_2x(src: (usize, usize), dst: (usize, usize)) -> Self {
        debug_assert_eq!((dst.0 * 2, dst.1 * 2), src);
        let mut resampler = Self::new(src, dst);
        resampler.planes = [PlaneScale::Box2x, PlaneScale::Box2x, PlaneScale::Box2x];
        resampler
    }

    /// Resample only the luma (`Y`) plane of `src` into the top-left `dst_size`
    /// of `dst` (with zero heap allocations), edge-extending into `dst`'s
    /// right/bottom padding when `dst` is a padded coded-resolution buffer.
    ///
    /// Used when the resampled picture is an engine-only motion/mode search
    /// proxy (`partition_plan`, `decision::Measurer`), which reads only
    /// `PlaneIndex::Y` while the backend predicts coded blocks from the
    /// unscaled reference via `AV1 §7.11.3.3`.
    pub(crate) fn scale_luma_only(&mut self, dsp: &Dsp, src: &Picture, dst: &mut Picture) {
        let (out_w, out_h) = self.dst_size;
        let padded = (dst.width(), dst.height()) != (out_w, out_h);
        let (from, mut to) = (
            src.plane(PlaneIndex::Y).view(),
            dst.plane_mut(PlaneIndex::Y).view_mut(),
        );
        match &mut self.planes[0] {
            PlaneScale::Copy => {
                for y in 0..out_h {
                    to.row_mut(y)[..out_w].copy_from_slice(&from.row(y)[..out_w]);
                }
            }
            PlaneScale::Lanczos(resampler) => {
                dsp.resample_plane_with(resampler, &from, &mut to);
            }
            PlaneScale::Bilinear2x(rows) => upscale_2x_bilinear(&from, &mut to, rows),
            PlaneScale::Box2x => {
                let mut out = to
                    .sub_view_mut(0, 0, out_w, out_h)
                    .expect("inside the plane");
                dsp.downscale_2x(&from, &mut out);
            }
        }
        if padded {
            dst.replicate_beyond(out_w, out_h);
        }
    }
}

/// Weight of the nearer source sample in each direction of the 2:1 bilinear
/// upscale (the farther gets 1): output sample `2i` sits a quarter sample
/// before source sample `i`, `2i + 1` a quarter after it.
const BILINEAR_NEAR: u16 = 3;
/// log2 of the 2-D filter's total weight ((3 + 1)^2 = 16).
const BILINEAR_SHIFT: u32 = 4;

/// One source row upscaled 2:1 horizontally, x4 (edges clamped).
fn upscale_row_2x(src: &[u8], out: &mut [u16]) {
    let last = src.len() - 1;
    for (i, pair) in out[..2 * src.len()]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .enumerate()
    {
        let here = BILINEAR_NEAR * u16::from(src[i]);
        pair[0] = here + u16::from(src[i.saturating_sub(1)]);
        pair[1] = here + u16::from(src[(i + 1).min(last)]);
    }
}

/// `from` upscaled 2:1 into the top-left of `to` with the centre-aligned
/// [1 3]/4 filter in both directions (edges clamped); `rows` are three
/// scratch rows of at least `2 * from.width()` samples.
fn upscale_2x_bilinear(from: &PlaneView<'_>, to: &mut PlaneViewMut<'_>, rows: &mut [Vec<u16>; 3]) {
    let (w, h) = (from.width(), from.height());
    let round = 1u16 << (BILINEAR_SHIFT - 1);
    let blend = |near: &[u16], far: &[u16], out: &mut [u8]| {
        for ((o, &n), &f) in out.iter_mut().zip(near).zip(far) {
            *o = ((BILINEAR_NEAR * n + f + round) >> BILINEAR_SHIFT) as u8;
        }
    };
    // rows[0] / [1] / [2]: source rows y - 1, y and y + 1, horizontally upscaled.
    upscale_row_2x(from.row(0), &mut rows[1]);
    let (above, rest) = rows.split_at_mut(1);
    above[0].copy_from_slice(&rest[0]);
    for y in 0..h {
        upscale_row_2x(from.row((y + 1).min(h - 1)), &mut rows[2]);
        let [above, here, below] = &*rows;
        blend(
            &here[..2 * w],
            &above[..2 * w],
            &mut to.row_mut(2 * y)[..2 * w],
        );
        blend(
            &here[..2 * w],
            &below[..2 * w],
            &mut to.row_mut(2 * y + 1)[..2 * w],
        );
        rows.rotate_left(1);
    }
}

/// The stream-size picture the caller writes into, when the coded
/// pictures are smaller.
#[derive(Debug)]
pub(crate) struct SourceScaler {
    input: Picture,
}

impl SourceScaler {
    /// A scaler for frames of the `stream` size coded at the `picture`
    /// size; `None` when the two are equal.
    pub(crate) fn between(
        (width, height): (usize, usize),
        picture: (usize, usize),
    ) -> nc_common::Result<Option<Self>> {
        if (width, height) == picture {
            return Ok(None);
        }
        Ok(Some(Self {
            input: Picture::new(width, height, 0)?,
        }))
    }

    /// The stream-size buffer the next frame is written into.
    pub(crate) fn input_mut(&mut self) -> &mut Picture {
        &mut self.input
    }

    /// The frame written into [`Self::input_mut`].
    pub(crate) fn input(&self) -> &Picture {
        &self.input
    }
}

/// The source frame `src` at the size of `dst`: [`Dsp::downscale_2x`]
/// for planes exactly twice the destination, else scalar Lanczos-3.
pub(crate) fn scale_source_picture(dsp: &Dsp, src: &Picture, dst: &mut Picture) {
    for plane in PlaneIndex::ALL {
        let (from, mut to) = (src.plane(plane).view(), dst.plane_mut(plane).view_mut());
        if (from.width(), from.height()) == (2 * to.width(), 2 * to.height()) {
            dsp.downscale_2x(&from, &mut to);
        } else {
            resample_plane(&from, &mut to);
        }
    }
}

/// Resample the visible planes of `src` into the visible planes of `dst`.
pub(crate) fn scale_picture(src: &Picture, dst: &mut Picture) {
    for plane in PlaneIndex::ALL {
        resample_plane(
            &src.plane(plane).view(),
            &mut dst.plane_mut(plane).view_mut(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic textured 4:2:0 picture.
    fn pattern(width: usize, height: usize) -> Picture {
        let mut picture = Picture::new(width, height, 0).expect("valid size");
        for plane in PlaneIndex::ALL {
            let mut view = picture.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                    *sample = ((x * 37 + y * 101 + (x * y) % 29) % 256) as u8;
                }
            }
        }
        picture
    }

    /// [`PictureResampler`] matches [`scale_source_picture`] at every ratio
    /// spatial SVC uses (`Dsp::downscale_2x` for exact 2:1 plane halvings and
    /// cached-table SIMD Lanczos-3 matching scalar [`scale_picture`] for all
    /// other ratios, including 1.5:1, 2:1 upscaling, and odd chroma).
    #[test]
    fn picture_resampler_matches_scale_picture() {
        let dsp = Dsp::new();
        let cases = [
            ((1280, 720), (640, 360)),
            ((1280, 720), (854, 480)),
            ((1280, 720), (1024, 576)),
            ((1282, 722), (642, 362)),
            ((1284, 724), (642, 362)),
            ((854, 480), (1280, 720)),
            ((640, 360), (1280, 720)),
            ((1884, 64), (942, 32)),
            ((136, 104), (90, 70)),
        ];
        for (src_size, dst_size) in cases {
            let src = pattern(src_size.0, src_size.1);
            let mut expected = Picture::new(dst_size.0, dst_size.1, 0).expect("valid size");
            let mut actual = Picture::new(dst_size.0, dst_size.1, 0).expect("valid size");
            scale_picture(&src, &mut expected);
            let mut resampler = PictureResampler::new(src_size, dst_size);
            resampler.scale(&dsp, &src, &mut actual);
            for plane in PlaneIndex::ALL {
                let (e, a) = (expected.plane(plane).view(), actual.plane(plane).view());
                for y in 0..e.height() {
                    assert_eq!(
                        e.row(y),
                        a.row(y),
                        "{src_size:?} -> {dst_size:?} {plane:?} row {y}"
                    );
                }
            }
        }
    }

    /// The bilinear 2:1 proxy upscale equals the separable centre-aligned
    /// [1 3]/4 filter with clamped edges, computed directly.
    #[test]
    fn bilinear_upscale_matches_the_direct_filter() {
        let (width, height) = (24, 10);
        let source = pattern(width, height);
        let mut upscaled = Picture::new(2 * width, 2 * height, 0).expect("valid size");
        PictureResampler::bilinear_upscale_2x((width, height), (2 * width, 2 * height))
            .scale_luma_only(&Dsp::new(), &source, &mut upscaled);
        let luma = source.plane(PlaneIndex::Y).view();
        let at = |col: isize, row: isize| {
            let row = row.clamp(0, height as isize - 1) as usize;
            u32::from(luma.row(row)[col.clamp(0, width as isize - 1) as usize])
        };
        // An output sample's nearer source sample and the side of its farther one.
        let near = |out: usize| (out as isize / 2, if out.is_multiple_of(2) { -1 } else { 1 });
        let result = upscaled.plane(PlaneIndex::Y).view();
        for out_row in 0..2 * height {
            for out_col in 0..2 * width {
                let ((col, step_col), (row, step_row)) = (near(out_col), near(out_row));
                let sum = 9 * at(col, row)
                    + 3 * at(col + step_col, row)
                    + 3 * at(col, row + step_row)
                    + at(col + step_col, row + step_row);
                assert_eq!(
                    u32::from(result.row(out_row)[out_col]),
                    (sum + 8) >> 4,
                    "({out_col}, {out_row})"
                );
            }
        }
    }
}
