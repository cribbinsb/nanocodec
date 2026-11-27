//! Native streaming decode of IVF (AV1, VP9, AV2) and Annex-B output (any
//! HEVC Main / Main 10 or H.264 Constrained High stream, pictures in output
//! order), to raw I420 (`--output`; pictures of more than 8 bits as 16-bit
//! little-endian samples, `yuv420p10le`) and/or one MD5 line per shown frame
//! (`--frame-md5`, the form of the libvpx test vectors' `.md5`), or neither
//! (`--discard 1`: decode every frame and write nothing, to time the decoder
//! alone as reference decoders do with `vpxdec --noblit` or `dav1d --muxer
//! null`).
//! `--bench N` times that decode in process instead: the stream is read into
//! memory once and decoded N times, each pass timed alone and written as one
//! JSON line, so no pass pays process start-up or file input (the same span
//! `tools/anchors/libavcodec_bench.c` times for the `FFmpeg` decoders).
//!
//! Container framing (`framing.rs`) is bounded independently of the codec
//! parsers.
use std::borrow::Cow;
use std::io::{BufRead, IoSlice, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use nc_common::{
    ChromaFormat, DecodedPicture, OutputPicture, Picture, PlaneIndex, PlaneView, Sample,
    ShownFrame, ShownPicture, decode::DecodeError,
};
use nc_encoder::Codec;

use crate::framing::{ivf_fourcc, read_ivf, read_nal_units};
use crate::md5::{self, Md5};
use crate::stdio;

#[derive(Debug)]
struct DecodeArgs {
    codec: Codec,
    input: PathBuf,
    output: Option<PathBuf>,
    frame_md5: Option<PathBuf>,
    film_grain: FilmGrain,
    operating_point: Option<u8>,
    all_layers: bool,
    bench: Option<u32>,
}

/// Parse an AV1 `--operating-point` argument in `0..=MAX_OPERATING_POINT`.
pub(crate) fn parse_operating_point(value: &str, prefix: &str) -> Result<u8, String> {
    let max = nc_av1::MAX_OPERATING_POINT;
    let op = value
        .parse::<u8>()
        .map_err(|_| format!("{prefix}--operating-point takes 0..={max}, not '{value}'"))?;
    if op > max {
        return Err(format!(
            "{prefix}--operating-point takes 0..={max}, not '{value}'"
        ));
    }
    Ok(op)
}

/// Whether AV1 frames are shown with the film grain they signal
/// (`--film-grain 1`, the default) or grain-free (`--film-grain 0`, as
/// `aomdec --skip-film-grain` and `dav1d --filmgrain 0`): the harness
/// scores and gates grain streams grain-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilmGrain {
    Apply,
    Skip,
}

fn parse(args: &[String]) -> Result<DecodeArgs, String> {
    stdio::check_streams(args, &[], &["--output", "--frame-md5"])?;
    let mut codec = None;
    let mut input = None;
    let mut output = None;
    let mut frame_md5 = None;
    let mut film_grain = None;
    let mut operating_point = None;
    let mut all_layers = None;
    let mut discard = None;
    let mut bench = None;
    let (pairs, remainder) = args.as_chunks::<2>();
    for pair in pairs {
        let value = pair[1].clone();
        match pair[0].as_str() {
            "--codec" if codec.is_none() => {
                codec = Some(match value.as_str() {
                    "av1" => Codec::Av1,
                    "vp9" => Codec::Vp9,
                    "av2" => Codec::Av2,
                    "h264" => Codec::H264,
                    "hevc" => Codec::Hevc,
                    "vvc" | "h266" => Codec::Vvc,
                    _ => return Err(format!("unsupported decode codec '{value}'")),
                });
            }
            "--input" if input.is_none() => input = Some(PathBuf::from(value)),
            "--output" if output.is_none() => output = Some(PathBuf::from(value)),
            "--frame-md5" if frame_md5.is_none() => frame_md5 = Some(PathBuf::from(value)),
            "--film-grain" if film_grain.is_none() => {
                film_grain = Some(match value.as_str() {
                    "1" => FilmGrain::Apply,
                    "0" => FilmGrain::Skip,
                    _ => return Err(format!("--film-grain takes 0 or 1, not '{value}'")),
                });
            }
            "--operating-point" if operating_point.is_none() => {
                operating_point = Some(parse_operating_point(&value, "")?);
            }
            "--all-layers" if all_layers.is_none() => {
                all_layers = Some(match value.as_str() {
                    "1" => true,
                    "0" => false,
                    _ => return Err(format!("--all-layers takes 0 or 1, not '{value}'")),
                });
            }
            "--discard" if discard.is_none() => {
                discard = Some(match value.as_str() {
                    "1" => true,
                    "0" => false,
                    _ => return Err(format!("--discard takes 0 or 1, not '{value}'")),
                });
            }
            "--bench" if bench.is_none() => {
                bench = Some(
                    value
                        .parse::<u32>()
                        .ok()
                        .filter(|&passes| passes > 0)
                        .ok_or(format!("--bench takes a pass count, not '{value}'"))?,
                );
            }
            flag => return Err(format!("unknown or repeated decode option '{flag}'")),
        }
    }
    if !remainder.is_empty() {
        return Err("decode options require a value".into());
    }
    let codec = codec.ok_or("missing --codec")?;
    if bench.is_some() && (output.is_some() || frame_md5.is_some()) {
        return Err("--bench writes no frames: drop --output and --frame-md5".into());
    }
    if film_grain.is_some() && codec != Codec::Av1 {
        return Err("--film-grain applies to AV1 only".into());
    }
    if operating_point.is_some() && !matches!(codec, Codec::Av1 | Codec::Vp9) {
        return Err("--operating-point applies to AV1 and VP9 only".into());
    }
    if all_layers.is_some() && !matches!(codec, Codec::Av1 | Codec::Vp9) {
        return Err("--all-layers applies to AV1 and VP9 only".into());
    }
    Ok(DecodeArgs {
        codec,
        input: input.ok_or("missing --input")?,
        output: if output.is_none()
            && frame_md5.is_none()
            && discard != Some(true)
            && bench.is_none()
        {
            return Err("missing --output, --frame-md5, --discard 1 or --bench N".into());
        } else {
            output
        },
        frame_md5,
        film_grain: film_grain.unwrap_or(FilmGrain::Apply),
        operating_point,
        all_layers: all_layers.unwrap_or(false),
        bench,
    })
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    let args = parse(args)?;
    if let Some(passes) = args.bench {
        return bench(&args, passes);
    }
    let create = |path: &PathBuf| -> Result<stdio::Output, String> {
        if let (Ok(input), Ok(output)) = (args.input.canonicalize(), path.canonicalize())
            && input == output
            && !stdio::is_stdio(path)
        {
            return Err("decode input and output must be different files".into());
        }
        stdio::create(path)
    };
    let mut yuv_output = args.output.as_ref().map(create).transpose()?;
    let mut md5_output = args.frame_md5.as_ref().map(create).transpose()?;
    let mut input = stdio::open(&args.input)?;
    decode_stream_with_op(
        args.codec,
        args.film_grain,
        args.operating_point,
        args.all_layers,
        &mut input,
        &mut |frame: &Shown| {
            if let Some(output) = yuv_output.as_mut() {
                frame
                    .write_rows(output)
                    .map_err(|error| error.to_string())?;
            }
            if let Some(output) = md5_output.as_mut() {
                writeln!(output, "{}", frame.md5()).map_err(|error| error.to_string())?;
            }
            Ok(())
        },
    )?;
    for output in yuv_output.iter_mut().chain(md5_output.iter_mut()) {
        output.flush().map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// `--bench N`: decode the in-memory stream `passes` times, each pass with a
/// fresh decoder, shown frames discarded; one JSON line per pass on stdout
/// (`repeat`, `frames`, `seconds`). A pass covers the decoder's construction
/// and the container framing over memory (IVF headers, Annex-B start codes)
/// along with the decode: it shares the one decode path rather than
/// pre-splitting packets on a second one.
fn bench(args: &DecodeArgs, passes: u32) -> Result<(), String> {
    let stream = std::fs::read(&args.input).map_err(|error| error.to_string())?;
    for repeat in 0..passes {
        let start = Instant::now();
        let frames = decode_stream_with_op(
            args.codec,
            args.film_grain,
            args.operating_point,
            args.all_layers,
            &mut std::io::Cursor::new(stream.as_slice()),
            &mut |_: &Shown| Ok(()),
        )?;
        let seconds = start.elapsed().as_secs_f64();
        println!("{{\"repeat\": {repeat}, \"frames\": {frames}, \"seconds\": {seconds:.9}}}");
    }
    Ok(())
}

/// A decoded picture and its visible size (odd sizes crop the even
/// picture; 4:2:0 chroma covers `(size + 1) / 2`).
pub(crate) struct Shown<'a> {
    planes: usize,
    samples: Samples<'a>,
    width: usize,
    height: usize,
}

/// Where a shown picture's samples are.
#[allow(
    clippy::large_enum_variant,
    reason = "one short-lived value per shown frame; boxing would allocate per frame"
)]
enum Samples<'a> {
    /// A picture of its own, visible from (0, 0).
    Owned(Picture),
    /// The decoder's own picture, visible from (0, 0), lent until the
    /// decoder's next call (no copy).
    Borrowed(&'a Picture),
    /// The decoder's picture, still shared with its reference buffers
    /// (no copy), visible from the luma sample `origin` (even).
    Shared {
        picture: Arc<Picture>,
        origin: (usize, usize),
    },
    /// The visible window the decoder shows until its next call (no copy).
    View(ShownPicture<'a>),
    /// Borrowed `u16` plane views of a deep picture shown in decoding order;
    /// converted row-by-row to 16-bit little-endian bytes only when written or
    /// hashed, so `--discard 1` performs no copy or allocation.
    HighViews([PlaneView<'a, u16>; 3]),
    /// Shared `u16` picture in output order; converted row-by-row only when
    /// written or hashed.
    HighShared(OutputPicture<u16>),
}

impl<'a> Shown<'a> {
    pub(crate) fn whole(picture: Picture) -> Self {
        Self {
            planes: 3,
            width: picture.width(),
            height: picture.height(),
            samples: Samples::Owned(picture),
        }
    }

    /// The picture a decoder shows (its first `planes` planes), written
    /// straight from the decoder's reconstruction.
    fn view(shown: ShownPicture<'a>, planes: usize) -> Self {
        Self {
            planes,
            width: shown.width(),
            height: shown.height(),
            samples: Samples::View(shown),
        }
    }

    /// A frame of either depth a decoder shows, borrowed without copying.
    fn frame(shown: ShownFrame<'a>) -> Self {
        let (width, height) = shown.size();
        let samples = match shown {
            ShownFrame::Eight(view) => Samples::View(view),
            ShownFrame::High(view, _) => {
                Samples::HighViews(PlaneIndex::ALL.map(|plane| view.plane(plane)))
            }
        };
        Self {
            planes: 3,
            width,
            height,
            samples,
        }
    }

    /// A VP9 frame of either depth, of any parity, borrowed without copying.
    fn vp9(shown: nc_vp9::ShownVp9Frame<'a>) -> Self {
        let (samples, (width, height)) = match shown {
            nc_vp9::ShownVp9Frame::Eight(picture, size) => (Samples::Borrowed(picture), size),
            nc_vp9::ShownVp9Frame::High(picture, (width, height), _) => {
                let format = picture.chroma_format();
                let planes = PlaneIndex::ALL.map(|plane| {
                    let (w, h) = format.plane_size(plane, (width, height));
                    picture
                        .plane(plane)
                        .view()
                        .sub_view(0, 0, w, h)
                        .expect("the visible size lies inside the picture")
                });
                (Samples::HighViews(planes), (width, height))
            }
        };
        Self {
            planes: 3,
            width,
            height,
            samples,
        }
    }

    /// An HEVC picture in output order (all three planes).
    fn hevc(frame: DecodedPicture) -> Result<Self, String> {
        Self::decoded(frame, 3)
    }

    /// A picture in output order, its first `planes` planes, shared with the
    /// decoder without copying.
    #[allow(clippy::unnecessary_wraps, reason = "uniform signature at call sites")]
    fn decoded(frame: DecodedPicture, planes: usize) -> Result<Self, String> {
        let (width, height) = frame.size();
        let samples = match frame {
            DecodedPicture::Eight(frame) => Samples::Shared {
                picture: frame.picture,
                origin: frame.origin,
            },
            DecodedPicture::High(frame) => Samples::HighShared(frame),
        };
        Ok(Self {
            planes,
            samples,
            width,
            height,
        })
    }

    /// The owned picture, to give back to its decoder for reuse; a shared
    /// or borrowed one stays with its decoder.
    fn into_owned(self) -> Option<Picture> {
        match self.samples {
            Samples::Owned(picture) => Some(picture),
            Samples::Borrowed(_)
            | Samples::Shared { .. }
            | Samples::View(_)
            | Samples::HighViews(_)
            | Samples::HighShared(_) => None,
        }
    }

    /// The shown samples as a picture of the shown size: the owned picture,
    /// or a copy of a decoder's (for callers that keep it past the
    /// decoder's next call).
    pub(crate) fn to_picture(&self) -> Result<Cow<'_, Picture>, String> {
        match &self.samples {
            Samples::Owned(picture) => return Ok(Cow::Borrowed(picture)),
            Samples::HighViews(_) | Samples::HighShared(_) => {
                return Err("8-bit copy of a picture of more than 8 bits".into());
            }
            _ => {}
        }
        let format = match &self.samples {
            Samples::Borrowed(picture) => picture.chroma_format(),
            Samples::Shared { picture, .. } => picture.chroma_format(),
            Samples::View(shown) => shown.chroma_format(),
            _ => ChromaFormat::Yuv420,
        };
        let mut copy =
            Picture::with_format(self.width, self.height, format, 0).map_err(|e| e.to_string())?;
        for plane in PlaneIndex::ALL.into_iter().take(self.planes) {
            let mut target = copy.plane_mut(plane).view_mut();
            self.plane_rows(plane, |y, row| {
                target.row_mut(y).copy_from_slice(row);
                Ok::<(), ()>(())
            })
            .expect("copying cannot fail");
        }
        Ok(Cow::Owned(copy))
    }

    /// Every visible row, planes in order (raw I420 or monochrome luma);
    /// deep pictures convert one row at a time into a reusable scratch buffer.
    fn visit_rows<E>(&self, mut visit: impl FnMut(&[u8]) -> Result<(), E>) -> Result<(), E> {
        match &self.samples {
            Samples::HighViews(planes) => {
                let mut scratch = Vec::new();
                for plane in &planes[..self.planes] {
                    for row in plane.rows() {
                        visit(u16::le_bytes(row, &mut scratch))?;
                    }
                }
                Ok(())
            }
            Samples::HighShared(frame) => {
                let view = frame.view();
                let mut scratch = Vec::new();
                for plane in PlaneIndex::ALL.into_iter().take(self.planes) {
                    for row in view.plane(plane).rows() {
                        visit(u16::le_bytes(row, &mut scratch))?;
                    }
                }
                Ok(())
            }
            _ => {
                for plane in PlaneIndex::ALL.into_iter().take(self.planes) {
                    self.plane_rows(plane, |_, row| visit(row))?;
                }
                Ok(())
            }
        }
    }

    /// Every visible row of an 8-bit `plane` with its index, top to bottom.
    fn plane_rows<'s, E>(
        &'s self,
        plane: PlaneIndex,
        mut visit: impl FnMut(usize, &'s [u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        let (picture, (x0, y0)) = match &self.samples {
            Samples::Owned(picture) => (picture, (0, 0)),
            Samples::Borrowed(picture) => (*picture, (0, 0)),
            Samples::Shared { picture, origin } => (&**picture, *origin),
            Samples::View(shown) => {
                let view = shown.plane(plane);
                return (0..view.height()).try_for_each(|y| visit(y, view.row(y)));
            }
            Samples::HighViews(_) | Samples::HighShared(_) => {
                unreachable!("deep pictures are visited in visit_rows / write_rows")
            }
        };
        let format = picture.chroma_format();
        let (sx, sy) = format.plane_subsampling(plane);
        let (w, h) = format.plane_size(plane, (self.width, self.height));
        let view = picture.plane(plane).view();
        for y in 0..h {
            let row = view.row(y + (y0 >> sy));
            visit(y, &row[x0 >> sx..(x0 >> sx) + w])?;
        }
        Ok(())
    }

    /// Write every visible row, planes in order: 8-bit pictures as one vectored
    /// write sequence (no copy into a buffered writer), deep pictures through
    /// [`nc_common::yuv::write_planes`] with a single reusable row buffer.
    fn write_rows(&self, out: &mut impl Write) -> std::io::Result<()> {
        match &self.samples {
            Samples::HighViews(planes) => {
                return nc_common::yuv::write_planes(out, planes[..self.planes].iter().copied())
                    .map_err(|err| match err {
                        nc_common::Error::Io(io) => io,
                        other => std::io::Error::other(other.to_string()),
                    });
            }
            Samples::HighShared(frame) => {
                let view = frame.view();
                let planes = PlaneIndex::ALL
                    .into_iter()
                    .take(self.planes)
                    .map(|p| view.plane(p));
                return nc_common::yuv::write_planes(out, planes).map_err(|err| match err {
                    nc_common::Error::Io(io) => io,
                    other => std::io::Error::other(other.to_string()),
                });
            }
            _ => {}
        }
        let mut rows = Vec::new();
        for plane in PlaneIndex::ALL.into_iter().take(self.planes) {
            self.plane_rows::<()>(plane, |_, row| {
                rows.push(IoSlice::new(row));
                Ok(())
            })
            .expect("collecting rows cannot fail");
        }
        nc_common::yuv::write_vectored_all(out, &mut rows)
    }

    pub(crate) fn md5(&self) -> String {
        let mut hasher = Md5::new();
        self.visit_rows::<()>(|row| {
            hasher.update(row);
            Ok(())
        })
        .expect("hashing cannot fail");
        md5::hex(&hasher.finish())
    }
}

pub(crate) enum Decoder {
    Av1(Box<nc_av1::Decoder>),
    Vp9(Box<nc_vp9::stream::Decoder>),
    Av2(Box<nc_av2::Decoder>),
    H264(Box<nc_h264::Decoder>),
    Hevc(Box<nc_hevc::Decoder>),
    Vvc(Box<nc_vvc::Decoder>),
}
impl Decoder {
    pub(crate) fn new(codec: Codec) -> Self {
        match codec {
            Codec::Av1 => Self::Av1(Box::default()),
            Codec::Vp9 => Self::Vp9(Box::default()),
            Codec::Av2 => Self::Av2(Box::default()),
            Codec::H264 => Self::H264(Box::default()),
            Codec::Hevc => Self::Hevc(Box::default()),
            Codec::Vvc => Self::Vvc(Box::default()),
        }
    }
    /// Hand a written picture back to decoders that reuse picture buffers
    /// (as the browser player does); others drop it.
    fn recycle(&mut self, picture: Picture) {
        match self {
            Self::Av1(decoder) => decoder.recycle(picture),
            Self::Vp9(decoder) => decoder.recycle(picture),
            Self::Hevc(decoder) => decoder.recycle(picture),
            _ => {}
        }
    }
    /// The picture a packet shows, if any (VP9 packets may hold only
    /// hidden frames).
    pub(crate) fn decode(&mut self, packet: &[u8]) -> Result<Option<Shown<'_>>, DecodeError> {
        match self {
            Self::Av1(decoder) => {
                // Written straight from the decoder's picture (no output copy;
                // deeper pictures packed as 16-bit samples). A unit whose
                // frames the operating point drops shows nothing.
                if !decoder.decode_any_depth(packet)? {
                    return Ok(None);
                }
                let shown = decoder.shown_frame().expect("a decoded picture is shown");
                Ok(Some(Shown::frame(shown)))
            }
            Self::Vp9(decoder) => {
                // 8-bit frames written straight from the decoder's picture,
                // deeper ones packed.
                if !decoder.decode_kept(packet)? {
                    return Ok(None);
                }
                let shown = decoder.shown_vp9_frame().expect("a kept frame is shown");
                Ok(Some(Shown::vp9(shown)))
            }
            Self::Av2(decoder) => {
                // Written straight from the decoder's picture. A unit may
                // show nothing (hidden frames) or more ([`Decoder::more`]).
                decoder.decode_unit(packet)?;
                Ok(Self::next_av2(decoder))
            }
            Self::H264(decoder) => decoder.decode(packet).map(Shown::whole).map(Some),
            Self::Hevc(decoder) => decoder.decode(packet).map(Shown::whole).map(Some),
            Self::Vvc(decoder) => decoder.decode(packet).map(Shown::whole).map(Some),
        }
    }

    /// The next picture of the AV2 output queue.
    fn next_av2(decoder: &mut nc_av2::Decoder) -> Option<Shown<'_>> {
        if !decoder.next_output() {
            return None;
        }
        let shown = decoder.shown().expect("a decoded picture is shown");
        Some(Shown::view(shown, 3))
    }

    /// A further picture the last packet shows (AV2 temporal units may
    /// release several), or after [`Decoder::finish`] one still due.
    pub(crate) fn more(&mut self) -> Option<Shown<'_>> {
        match self {
            Self::Av2(decoder) => Self::next_av2(decoder),
            _ => None,
        }
    }

    /// End of stream: AV2 queues the hidden frames still due for output.
    pub(crate) fn finish(&mut self) {
        if let Self::Av2(decoder) = self {
            decoder.flush();
        }
    }

    /// Every shown layer picture in `packet` in OBU/superframe order (AV1 and
    /// VP9 `--all-layers 1`, or the single shown frame on other codecs).
    pub(crate) fn decode_all(&mut self, packet: &[u8]) -> Result<Vec<Shown<'_>>, DecodeError> {
        match self {
            Self::Av1(decoder) => decoder.decode_all_layers(packet).map(|layers| {
                layers
                    .into_iter()
                    .map(|(pic, _)| Shown::whole(pic))
                    .collect()
            }),
            Self::Vp9(decoder) => decoder.decode_all_layers(packet).map(|layers| {
                layers
                    .into_iter()
                    .map(|frame| Shown {
                        planes: 3,
                        samples: Samples::Owned(frame.picture),
                        width: frame.width,
                        height: frame.height,
                    })
                    .collect()
            }),
            _ => self.decode(packet).map(|opt| opt.into_iter().collect()),
        }
    }

    /// Show AV1 frames grain-free, as the reconstruction is.
    pub(crate) fn skip_film_grain(&mut self) {
        if let Self::Av1(decoder) = self {
            decoder.set_film_grain(false);
        }
    }

    /// Select the operating point (`0..=31` on AV1, `0..=2` on VP9).
    pub(crate) fn set_operating_point(
        &mut self,
        operating_point: Option<u8>,
    ) -> Result<(), DecodeError> {
        match (self, operating_point) {
            (Self::Av1(decoder), Some(op)) => decoder.set_operating_point(op)?,
            (Self::Vp9(decoder), Some(op)) => decoder.set_operating_point(op)?,
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
fn decode_stream<R: BufRead>(
    codec: Codec,
    film_grain: FilmGrain,
    input: &mut R,
    output: &mut impl FnMut(&Shown) -> Result<(), String>,
) -> Result<u64, String> {
    decode_stream_with_op(codec, film_grain, None, false, input, output)
}

fn decode_stream_with_op<R: BufRead>(
    codec: Codec,
    film_grain: FilmGrain,
    operating_point: Option<u8>,
    all_layers: bool,
    input: &mut R,
    output: &mut impl FnMut(&Shown) -> Result<(), String>,
) -> Result<u64, String> {
    let mut decoder = Decoder::new(codec);
    if film_grain == FilmGrain::Skip {
        decoder.skip_film_grain();
    }
    decoder
        .set_operating_point(operating_point)
        .map_err(|e| e.to_string())?;
    let mut shown = 0;
    match codec {
        Codec::Av2 if !starts_with_ivf_signature(input)? => {
            decode_av2_obu_stream(&mut decoder, input, &mut shown, output)?;
            decoder.finish();
            write_more(&mut decoder, &mut shown, output)?;
        }
        Codec::Av1 | Codec::Vp9 | Codec::Av2 => {
            decode_ivf_stream(&mut decoder, codec, all_layers, input, &mut shown, output)?;
            decoder.finish();
            write_more(&mut decoder, &mut shown, output)?;
        }
        Codec::Hevc => {
            let Decoder::Hevc(hevc) = &mut decoder else {
                unreachable!("HEVC decoder for HEVC input");
            };
            let mut write = |hevc: &mut nc_hevc::Decoder| -> Result<(), String> {
                while let Some(picture) = hevc.next_output() {
                    shown += 1;
                    // Dropped after writing, the buffer returns to the decoder.
                    output(&Shown::hevc(picture)?)?;
                }
                Ok(())
            };
            read_nal_units(input, |nal| {
                hevc.push_nal(nal).map_err(|error| error.to_string())?;
                write(hevc)
            })?;
            hevc.flush().map_err(|error| error.to_string())?;
            write(hevc)?;
        }
        Codec::H264 => {
            let Decoder::H264(h264) = &mut decoder else {
                unreachable!("H.264 decoder for H.264 input");
            };
            // Written straight from the decoder's pictures, which it reuses
            // once they are dropped.
            let mut write = |h264: &mut nc_h264::Decoder| -> Result<(), String> {
                while let Some(picture) = h264.next_output() {
                    shown += 1;
                    output(&Shown::decoded(picture, 3)?)?;
                }
                Ok(())
            };
            read_nal_units(input, |nal| {
                h264.push_nal(nal).map_err(|error| error.to_string())?;
                write(h264)
            })?;
            h264.flush();
            write(h264)?;
        }
        Codec::Vvc => {
            let Decoder::Vvc(vvc) = &mut decoder else {
                unreachable!("VVC decoder for VVC input");
            };
            let mut write = |frame: Shown| {
                shown += 1;
                output(&frame)
            };
            let planes = |vvc: &nc_vvc::Decoder| if vvc.is_monochrome() { 1 } else { 3 };
            read_nal_units(input, |nal| {
                // Our own streams show in decoding order, written straight
                // from the decoder's picture; general streams are reordered
                // (deeper than 8 bits: 16-bit little-endian samples).
                if vvc.push_nal_shown(nal).map_err(|error| error.to_string())? {
                    let picture = vvc.shown().expect("a decoded picture is shown");
                    write(Shown::view(picture, planes(vvc)))?;
                }
                while let Some(picture) = vvc.next_output() {
                    write(Shown::decoded(picture, planes(vvc))?)?;
                }
                Ok(())
            })?;
            vvc.flush().map_err(|error| error.to_string())?;
            while let Some(picture) = vvc.next_output() {
                write(Shown::decoded(picture, planes(vvc))?)?;
            }
        }
    }
    if shown == 0 {
        return Err("input contains no decoded pictures".into());
    }
    Ok(shown)
}

/// The four bytes that open an IVF file.
const IVF_SIGNATURE: &[u8] = b"DKIF";

/// Whether `input` begins with the IVF signature (nothing is consumed).
fn starts_with_ivf_signature<R: BufRead>(input: &mut R) -> Result<bool, String> {
    let head = input.fill_buf().map_err(|e| e.to_string())?;
    Ok(head.starts_with(IVF_SIGNATURE))
}

/// Decode a raw AV2 OBU stream (size-prefixed OBUs back to back, AVM's
/// `--obu` output), one temporal unit at a time.
fn decode_av2_obu_stream<R: BufRead>(
    decoder: &mut Decoder,
    input: &mut R,
    shown: &mut u64,
    output: &mut impl FnMut(&Shown) -> Result<(), String>,
) -> Result<(), String> {
    let mut stream = Vec::new();
    input
        .by_ref()
        .take(MAX_OBU_STREAM_BYTES)
        .read_to_end(&mut stream)
        .map_err(|e| e.to_string())?;
    for unit in nc_av2::obu::temporal_units(&stream).map_err(|e| e.to_string())? {
        if let Some(frame) = decoder.decode(unit).map_err(|e| e.to_string())? {
            *shown += 1;
            output(&frame)?;
        }
        write_more(decoder, shown, output)?;
    }
    Ok(())
}

/// Write the further pictures the last packet released (AV2).
fn write_more(
    decoder: &mut Decoder,
    shown: &mut u64,
    output: &mut impl FnMut(&Shown) -> Result<(), String>,
) -> Result<(), String> {
    while let Some(frame) = decoder.more() {
        *shown += 1;
        output(&frame)?;
    }
    Ok(())
}

/// Most bytes of a raw OBU stream read into memory (IVF is streamed packet
/// by packet; a raw stream has no packet framing to stream by).
const MAX_OBU_STREAM_BYTES: u64 = 1 << 31;

fn decode_ivf_stream<R: BufRead>(
    decoder: &mut Decoder,
    codec: Codec,
    all_layers: bool,
    input: &mut R,
    shown: &mut u64,
    output: &mut impl FnMut(&Shown) -> Result<(), String>,
) -> Result<(), String> {
    read_ivf(input, |header, frame| {
        if ivf_fourcc(codec).is_none_or(|fourcc| header.fourcc() != fourcc) {
            return Err("IVF codec identifier disagrees with --codec".into());
        }
        // The IVF header carries the first picture size only: a stream
        // may change size at a key frame (`--resize`), each picture is
        // written at its own size.
        // Written buffers serve later pictures.
        let mut spare = Vec::new();
        if all_layers {
            for layer in decoder
                .decode_all(frame.packet)
                .map_err(|e| e.to_string())?
            {
                *shown += 1;
                output(&layer)?;
                spare.extend(layer.into_owned());
            }
        } else if let Some(frame) = decoder.decode(frame.packet).map_err(|e| e.to_string())? {
            *shown += 1;
            output(&frame)?;
            spare.extend(frame.into_owned());
        }
        write_more(decoder, shown, output)?;
        for picture in spare {
            decoder.recycle(picture);
        }
        Ok(())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::MAX_PACKET_BYTES;
    use nc_common::yuv;
    use std::io::Cursor;
    /// Collect the raw I420 output.
    fn yuv_sink(out: &mut Vec<u8>) -> impl FnMut(&Shown) -> Result<(), String> + '_ {
        move |frame| {
            frame.visit_rows::<String>(|row| {
                out.extend_from_slice(row);
                Ok(())
            })
        }
    }
    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).into()).collect()
    }
    #[test]
    fn decode_flags_are_explicit_and_validated() {
        let args = strings(&["--codec", "av1", "--input", "in.ivf", "--output", "out.yuv"]);
        assert_eq!(parse(&args).unwrap().codec, Codec::Av1);
        assert_eq!(parse(&args).unwrap().film_grain, FilmGrain::Apply);
        let skip = [&args[..], &strings(&["--film-grain", "0"])].concat();
        assert_eq!(parse(&skip).unwrap().film_grain, FilmGrain::Skip);
        let bench = strings(&["--codec", "h264", "--input", "a", "--bench", "3"]);
        assert_eq!(parse(&bench).unwrap().bench, Some(3));
        for bad in [
            vec![],
            strings(&["--codec", "vp9"]),
            strings(&["--input"]),
            strings(&["--input", "a", "--input", "b"]),
            strings(&["--unknown", "a"]),
            [&args[..], &strings(&["--film-grain", "2"])].concat(),
            [&args[..], &strings(&["--bench", "3"])].concat(),
            strings(&["--codec", "h264", "--input", "a", "--bench", "0"]),
            [
                &strings(&["--codec", "vp9"]),
                &args[2..],
                &strings(&["--film-grain", "0"]),
            ]
            .concat(),
        ] {
            assert!(parse(&bad).is_err());
        }
    }
    fn ivf_header() -> Vec<u8> {
        crate::framing::IvfWriter::new(Vec::new(), Codec::Av1, 64, 48, 30000, 1001, 0)
            .unwrap()
            .finish()
            .unwrap()
    }
    #[test]
    fn invalid_and_truncated_containers_return_errors_before_allocation() {
        let header = ivf_header();
        for n in 0..=header.len() {
            assert!(
                decode_stream(
                    Codec::Av1,
                    FilmGrain::Apply,
                    &mut Cursor::new(&header[..n]),
                    &mut yuv_sink(&mut Vec::new())
                )
                .is_err()
            );
        }
        let mut oversized = header.clone();
        oversized.extend_from_slice(&((MAX_PACKET_BYTES + 1) as u32).to_le_bytes());
        oversized.extend_from_slice(&[0; 8]);
        assert!(
            decode_stream(
                Codec::Av1,
                FilmGrain::Apply,
                &mut Cursor::new(oversized),
                &mut yuv_sink(&mut Vec::new())
            )
            .unwrap_err()
            .contains("64 MiB")
        );
        let mut partial = header.clone();
        partial.push(1);
        assert!(
            decode_stream(
                Codec::Av1,
                FilmGrain::Apply,
                &mut Cursor::new(partial),
                &mut yuv_sink(&mut Vec::new())
            )
            .unwrap_err()
            .contains("frame header")
        );
        let mut payload = header;
        payload.extend_from_slice(&100u32.to_le_bytes());
        payload.extend_from_slice(&[0; 8]);
        payload.push(0);
        assert!(
            decode_stream(
                Codec::Av1,
                FilmGrain::Apply,
                &mut Cursor::new(payload),
                &mut yuv_sink(&mut Vec::new())
            )
            .unwrap_err()
            .contains("payload")
        );
        for codec in [Codec::H264, Codec::Hevc] {
            for input in [
                vec![],
                vec![0, 0, 1],
                vec![0, 0, 1, 0x02],
                vec![0, 0, 1, 0x67, 0x80],
            ] {
                assert!(
                    decode_stream(
                        codec,
                        FilmGrain::Apply,
                        &mut Cursor::new(input),
                        &mut yuv_sink(&mut Vec::new())
                    )
                    .is_err()
                );
            }
        }
    }
    #[test]
    fn native_vp9_ivf_dispatch_roundtrip_and_fourcc_mismatch() {
        let config = nc_encoder::EncoderConfig {
            codec: Codec::Vp9,
            width: 16,
            height: 16,
            frame_rate: nc_encoder::FrameRate { num: 30, den: 1 },
            quantizer: 0,
            intra_period: 30,
            temporal_layers: 1,
            block_log2: 4,
            key_qp_offset: None,
            signal: nc_encoder::StreamSignal::default(),
            params: nc_encoder::ParamOverrides::default().coding_every_frame(),
        };
        let mut encoder = nc_encoder::Encoder::<nc_vp9::Vp9Backend>::new(config).unwrap();
        let picture = Picture::new(16, 16, 0).unwrap();
        let mut ivf =
            crate::framing::IvfWriter::new(Vec::new(), Codec::Vp9, 16, 16, 30, 1, 3).unwrap();
        let mut expected = Vec::new();
        for index in 0..3 {
            let frame = encoder.encode(&picture);
            ivf.write_frame(&frame.data, index).unwrap();
            yuv::write_frame(&mut expected, encoder.reconstruction()).unwrap();
        }
        let stream = ivf.finish().unwrap();
        let mut output = Vec::new();
        assert_eq!(
            decode_stream(
                Codec::Vp9,
                FilmGrain::Apply,
                &mut Cursor::new(&stream),
                &mut yuv_sink(&mut output)
            )
            .unwrap(),
            3
        );
        assert_eq!(output, expected);
        assert!(
            decode_stream(
                Codec::Av1,
                FilmGrain::Apply,
                &mut Cursor::new(&stream),
                &mut yuv_sink(&mut Vec::new())
            )
            .unwrap_err()
            .contains("identifier")
        );
    }
    #[test]
    fn native_vvc_annex_b_dispatch_roundtrip() {
        let config = nc_encoder::EncoderConfig {
            codec: Codec::Vvc,
            width: 34,
            height: 66,
            frame_rate: nc_encoder::FrameRate { num: 30, den: 1 },
            quantizer: 22,
            intra_period: 30,
            temporal_layers: 1,
            block_log2: 5,
            key_qp_offset: None,
            signal: nc_encoder::StreamSignal::default(),
            params: nc_encoder::ParamOverrides::default().coding_every_frame(),
        };
        let mut encoder = nc_encoder::Encoder::<nc_vvc::VvcBackend>::new(config).unwrap();
        let source = Picture::new(34, 66, 0).unwrap();
        let mut stream = Vec::new();
        let mut expected = Vec::new();
        for _ in 0..3 {
            stream.extend(encoder.encode(&source).data);
            yuv::write_frame(&mut expected, encoder.reconstruction()).unwrap();
        }
        let mut decoded = Vec::new();
        assert_eq!(
            decode_stream(
                Codec::Vvc,
                FilmGrain::Apply,
                &mut Cursor::new(stream),
                &mut yuv_sink(&mut decoded)
            )
            .unwrap(),
            3
        );
        assert_eq!(decoded, expected);
    }
}
