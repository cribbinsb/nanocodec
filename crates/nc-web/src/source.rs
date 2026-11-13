//! One captured source frame as an encoder's input picture at a view's size.
//!
//! The page captures each picture once and every view converts it to its
//! own size here, in its worker: a video or camera frame arrives as the
//! decoded YUV planes themselves (`VideoFrame.copyTo` in the frame's own
//! I420 or NV12 layout), so no YUV -> RGB -> YUV round trip touches the
//! samples, and the stream signals the frame's own colour matrix and range.
//! Planes of another size are scaled with the player model's bilinear filter
//! (`nc_dsp::display`, each plane on its own grid, the same filter the
//! offline reference streams use through its Python twin) into the view,
//! letterboxed with black bars when the aspect ratios differ. Pictures the
//! browser gives only as RGBA (the test scene, frames in other formats) come
//! already letterboxed by a canvas and are converted to limited-range BT.601
//! (`nc_dsp` `rgba_to_i420`).

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::DisplayMatrix;
use nc_dsp::display::DisplayScaler;
use wasm_bindgen::prelude::*;

use crate::js_error;
use crate::pixels;

/// Luma of a black bar in limited range; full range uses 0.
const LIMITED_BLACK: u8 = 16;
/// Chroma of a grey (or black) bar.
const NEUTRAL_CHROMA: u8 = 128;

/// The YUV layouts `from_planes` accepts (`WebCodecs` `VideoPixelFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Three planes: Y, then Cb, then Cr (`I420`, and `I420A` whose fourth,
    /// alpha plane is ignored).
    I420,
    /// Two planes: Y, then interleaved Cb, Cr pairs (`NV12`).
    Nv12,
}

impl Layout {
    fn parse(format: &str) -> Option<Self> {
        match format {
            "I420" | "I420A" => Some(Self::I420),
            "NV12" => Some(Self::Nv12),
            _ => None,
        }
    }
}

/// Parse a page colour name (`web/capture-worker.js`) into a matrix.
pub(crate) fn parse_matrix(name: &str) -> Option<DisplayMatrix> {
    match name {
        "bt601" => Some(DisplayMatrix::Bt601Limited),
        "bt709" => Some(DisplayMatrix::Bt709Limited),
        "bt601-full" => Some(DisplayMatrix::Bt601Full),
        "bt709-full" => Some(DisplayMatrix::Bt709Full),
        _ => None,
    }
}

/// Where the source lands in the view: the largest rectangle of the
/// source's aspect ratio that fits, centred, with even sizes and offsets so
/// that it covers whole chroma samples. A source within a sample of the
/// view's aspect ratio fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Placement {
    fn of(source: (usize, usize), view: (usize, usize)) -> Self {
        let ((sw, sh), (vw, vh)) = (source, view);
        let even = |numerator: usize, denominator: usize| {
            // round(numerator / denominator / 2) * 2
            2 * ((numerator + denominator) / (2 * denominator))
        };
        let (width, height) = if sw * vh >= sh * vw {
            (vw, even(sh * vw, sw).min(vh))
        } else {
            (even(sw * vh, sh).min(vw), vh)
        };
        let (width, height) = (width.max(1), height.max(1));
        Self {
            x: (vw - width) / 4 * 2,
            y: (vh - height) / 4 * 2,
            width,
            height,
        }
    }

    /// The same rectangle on a 4:2:0 chroma plane.
    fn chroma(self) -> Self {
        Self {
            x: self.x / 2,
            y: self.y / 2,
            width: self.width.div_ceil(2),
            height: self.height.div_ceil(2),
        }
    }

    fn fills(self, view: (usize, usize)) -> bool {
        (self.x, self.y, self.width, self.height) == (0, 0, view.0, view.1)
    }
}

/// A source frame converted for one view: the encoder's input picture
/// (`BrowserEncoder::encode`) and, packed, the reference the decode worker
/// scores and the page shows as the original.
#[wasm_bindgen]
#[derive(Debug)]
pub struct SourceFrame {
    picture: Picture,
    /// The matrix and range of the samples in `picture`.
    matrix: DisplayMatrix,
    dsp: nc_dsp::Dsp,
    /// Cb and Cr de-interleaved from an NV12 frame (the planes of a
    /// source-size picture whose luma is unused).
    chroma: Picture,
    scalers: [DisplayScaler; 3],
    /// `packed` output, reused.
    packed: Vec<u8>,
}

#[wasm_bindgen]
impl SourceFrame {
    /// An empty (black, BT.601) frame of the view size.
    /// # Errors
    /// Rejects sizes the encoders refuse (`BrowserEncoder::new`).
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> Result<Self, JsValue> {
        Ok(Self {
            picture: Picture::new(width, height, 0).map_err(js_error)?,
            matrix: DisplayMatrix::Bt601Limited,
            dsp: nc_dsp::Dsp::new(),
            chroma: Picture::new(2, 2, 0).map_err(js_error)?,
            scalers: Default::default(),
            packed: Vec::new(),
        })
    }

    /// Fill the frame from a `width` x `height` YUV picture in `data`, laid
    /// out as `WebCodecs` `VideoFrame.copyTo` reports it: `layout` holds each
    /// plane's byte offset and stride (`[offset, stride, ...]`); `format` is
    /// its `VideoPixelFormat` (`I420`, `I420A` or `NV12`) and `colour` its
    /// matrix and range (`bt601`, `bt709`, `bt601-full`, `bt709-full`).
    /// # Errors
    /// Rejects other formats or colours, and layouts that do not fit `data`.
    pub fn from_planes(
        &mut self,
        data: &[u8],
        format: &str,
        colour: &str,
        width: usize,
        height: usize,
        layout: &[u32],
    ) -> Result<(), JsValue> {
        let layout_kind =
            Layout::parse(format).ok_or_else(|| js_error("Unsupported pixel format"))?;
        let matrix = parse_matrix(colour).ok_or_else(|| js_error("Unsupported colour"))?;
        self.fill_planes(data, layout_kind, matrix, (width, height), layout)
            .map_err(js_error)
    }

    /// Fill the frame from packed RGBA of the view size (already letterboxed
    /// by the page's canvas), as limited-range BT.601.
    /// # Errors
    /// Rejects input whose byte count differs from the view size.
    pub fn from_rgba(&mut self, rgba: &[u8]) -> Result<(), JsValue> {
        let (width, height) = self.size();
        if rgba.len() != width * height * 4 {
            return Err(js_error(
                "RGBA frame length does not match the configuration",
            ));
        }
        pixels::from_rgba(&self.dsp, rgba, &mut self.picture);
        self.matrix = DisplayMatrix::Bt601Limited;
        Ok(())
    }

    /// Pack the frame as I420 (Y, then Cb, then Cr, rows without padding) in
    /// a buffer this frame reuses and return its byte offset in the module's
    /// WASM memory, read as for `BrowserDecoder::rgba_frame`
    /// (`packed_length` bytes).
    pub fn packed(&mut self) -> usize {
        pixels::pack_i420(
            nc_common::ShownPicture::whole(&self.picture),
            &mut self.packed,
        );
        self.packed.as_ptr() as usize
    }

    /// Byte count of `packed`.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn packed_length(&self) -> usize {
        nc_common::ShownPicture::whole(&self.picture).packed_frame_len()
    }

    /// The colour (`from_planes` names) of the samples.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn colour(&self) -> String {
        matrix_name(self.matrix).to_owned()
    }
}

impl SourceFrame {
    /// `from_planes` once its names are parsed.
    fn fill_planes(
        &mut self,
        data: &[u8],
        layout_kind: Layout,
        matrix: DisplayMatrix,
        (width, height): (usize, usize),
        layout: &[u32],
    ) -> Result<(), &'static str> {
        let planes = match layout_kind {
            Layout::I420 => 3,
            Layout::Nv12 => 2,
        };
        if width == 0 || height == 0 || layout.len() < 2 * planes {
            return Err("Invalid frame layout");
        }
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let view = |plane: usize, w: usize, h: usize| {
            let (offset, stride) = (layout[2 * plane] as usize, layout[2 * plane + 1] as usize);
            data.get(offset..)
                .and_then(|bytes| PlaneView::new(bytes, stride, w, h))
                .ok_or("Frame layout does not fit the data")
        };
        let luma = view(0, width, height)?;
        // Every plane is checked before any sample changes.
        let chroma_views = match layout_kind {
            Layout::I420 => [Some(view(1, cw, ch)?), Some(view(2, cw, ch)?)],
            Layout::Nv12 => [None, None],
        };
        let pairs = match layout_kind {
            Layout::I420 => None,
            Layout::Nv12 => Some(view(1, 2 * cw, ch)?),
        };
        self.matrix = matrix;
        let placement = Placement::of((width, height), self.size());
        self.fill_bars(placement);
        place(
            &mut self.scalers[0],
            &luma,
            &mut self.picture,
            PlaneIndex::Y,
            placement,
        );
        if let Some(pairs) = pairs {
            let chroma = &mut self.chroma;
            if (chroma.width(), chroma.height()) != (width, height) {
                *chroma = Picture::new(width, height, 0).map_err(|_| "Invalid frame size")?;
            }
            let [_, cb, cr] = chroma.planes_mut();
            self.dsp
                .deinterleave_uv(&pairs, &mut cb.view_mut(), &mut cr.view_mut());
        }
        for (i, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
            let source = chroma_views[i].unwrap_or_else(|| self.chroma.plane(plane).view());
            place(
                &mut self.scalers[1 + i],
                &source,
                &mut self.picture,
                plane,
                placement.chroma(),
            );
        }
        Ok(())
    }

    pub(crate) fn picture(&self) -> &Picture {
        &self.picture
    }

    pub(crate) fn matrix(&self) -> DisplayMatrix {
        self.matrix
    }

    fn size(&self) -> (usize, usize) {
        (self.picture.width(), self.picture.height())
    }

    /// Paint the bars around `placement` black in the frame's range.
    fn fill_bars(&mut self, placement: Placement) {
        let size = self.size();
        if placement.fills(size) {
            return;
        }
        let black = match self.matrix {
            DisplayMatrix::Bt601Full | DisplayMatrix::Bt709Full => 0,
            DisplayMatrix::Bt601Limited | DisplayMatrix::Bt709Limited => LIMITED_BLACK,
        };
        for (plane, value) in [
            (PlaneIndex::Y, black),
            (PlaneIndex::U, NEUTRAL_CHROMA),
            (PlaneIndex::V, NEUTRAL_CHROMA),
        ] {
            self.picture.plane_mut(plane).view_mut().fill(value);
        }
    }
}

/// `matrix` as the page names it.
pub(crate) fn matrix_name(matrix: DisplayMatrix) -> &'static str {
    match matrix {
        DisplayMatrix::Bt601Limited => "bt601",
        DisplayMatrix::Bt709Limited => "bt709",
        DisplayMatrix::Bt601Full => "bt601-full",
        DisplayMatrix::Bt709Full => "bt709-full",
    }
}

/// Copy or scale `source` into the `at` rectangle of `picture`'s `plane`.
fn place(
    scaler: &mut DisplayScaler,
    source: &PlaneView<'_>,
    picture: &mut Picture,
    plane: PlaneIndex,
    at: Placement,
) {
    let mut whole = picture.plane_mut(plane).view_mut();
    let mut target: PlaneViewMut<'_> = whole
        .sub_view_mut(at.x, at.y, at.width, at.height)
        .expect("the placement lies inside the plane");
    if (source.width(), source.height()) == (at.width, at.height) {
        target.copy_from(source);
    } else {
        scaler.scale(source, &mut target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_fills_near_aspect_and_letterboxes_others() {
        // 1280x720 into 854x480: 853.3 wide rounds to the even 854.
        assert_eq!(
            Placement::of((1280, 720), (854, 480)),
            Placement {
                x: 0,
                y: 0,
                width: 854,
                height: 480
            }
        );
        // 4:3 into 16:9: pillarbox, centred on an even column.
        assert_eq!(
            Placement::of((640, 480), (1280, 720)),
            Placement {
                x: 160,
                y: 0,
                width: 960,
                height: 720
            }
        );
        // Wider than the view: letterbox.
        assert_eq!(
            Placement::of((1920, 800), (1280, 720)),
            Placement {
                x: 0,
                y: 92,
                width: 1280,
                height: 534
            }
        );
    }

    /// A packed I420 picture of `width` x `height` whose samples follow
    /// `value(plane, x, y)`, with each plane's layout entry.
    fn i420(
        width: usize,
        height: usize,
        value: impl Fn(usize, usize, usize) -> u8,
    ) -> (Vec<u8>, Vec<u32>) {
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let mut data = Vec::new();
        let mut layout = Vec::new();
        for (plane, (w, h)) in [(width, height), (cw, ch), (cw, ch)]
            .into_iter()
            .enumerate()
        {
            layout.extend([data.len() as u32, w as u32]);
            for y in 0..h {
                data.extend((0..w).map(|x| value(plane, x, y)));
            }
        }
        (data, layout)
    }

    #[test]
    fn same_size_planes_pass_through_and_nv12_matches_i420() {
        let (w, h) = (66, 34);
        let value = |plane: usize, x: usize, y: usize| (plane * 80 + x * 3 + y * 5) as u8;
        let (data, layout) = i420(w, h, value);
        let mut frame = SourceFrame::new(w, h).unwrap();
        frame
            .fill_planes(
                &data,
                Layout::I420,
                DisplayMatrix::Bt709Limited,
                (w, h),
                &layout,
            )
            .unwrap();
        let offset = frame.packed();
        assert_eq!(offset, frame.packed.as_ptr() as usize);
        assert_eq!(frame.packed, data);
        assert_eq!(frame.colour(), "bt709");
        // The same picture as NV12, with padded rows.
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (stride, chroma_stride) = (w + 6, 2 * cw + 4);
        let mut nv12 = vec![0u8; stride * h + chroma_stride * ch];
        for y in 0..h {
            for x in 0..w {
                nv12[y * stride + x] = value(0, x, y);
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                nv12[stride * h + y * chroma_stride + 2 * x] = value(1, x, y);
                nv12[stride * h + y * chroma_stride + 2 * x + 1] = value(2, x, y);
            }
        }
        let layout = [0, stride as u32, (stride * h) as u32, chroma_stride as u32];
        let mut other = SourceFrame::new(w, h).unwrap();
        other
            .fill_planes(
                &nv12,
                Layout::Nv12,
                DisplayMatrix::Bt709Full,
                (w, h),
                &layout,
            )
            .unwrap();
        other.packed();
        assert_eq!(other.packed, data);
        assert_eq!(other.colour(), "bt709-full");
    }

    #[test]
    fn other_sizes_are_scaled_and_letterboxed() {
        // A flat 4:3 picture into a 16:9 view: flat inside, black bars.
        let (data, layout) = i420(64, 48, |plane, _, _| [200, 90, 160][plane]);
        let mut frame = SourceFrame::new(128, 72).unwrap();
        frame
            .fill_planes(
                &data,
                Layout::I420,
                DisplayMatrix::Bt601Limited,
                (64, 48),
                &layout,
            )
            .unwrap();
        let luma = frame.picture.plane(PlaneIndex::Y).view();
        // Placement: 96x72 at x = 16.
        assert!(luma.row(10)[..16].iter().all(|&s| s == LIMITED_BLACK));
        assert!(luma.row(10)[16..112].iter().all(|&s| s == 200));
        assert!(luma.row(10)[112..].iter().all(|&s| s == LIMITED_BLACK));
        let cb = frame.picture.plane(PlaneIndex::U).view();
        assert!(cb.row(3)[..8].iter().all(|&s| s == NEUTRAL_CHROMA));
        assert!(cb.row(3)[8..56].iter().all(|&s| s == 90));
    }

    #[test]
    fn rejects_bad_formats_colours_and_layouts() {
        assert_eq!(Layout::parse("RGBA"), None);
        assert_eq!(parse_matrix("bt2020"), None);
        for matrix in DisplayMatrix::ALL {
            assert_eq!(parse_matrix(matrix_name(matrix)), Some(matrix));
        }
        let (data, layout) = i420(64, 48, |_, _, _| 7);
        let mut frame = SourceFrame::new(64, 48).unwrap();
        let short = frame.fill_planes(
            &data[..100],
            Layout::I420,
            DisplayMatrix::Bt709Limited,
            (64, 48),
            &layout,
        );
        assert!(short.is_err());
        // A rejected frame leaves the picture as it was.
        assert!(
            frame
                .picture
                .plane(PlaneIndex::Y)
                .view()
                .row(0)
                .iter()
                .all(|&s| s == 0)
        );
    }
}
