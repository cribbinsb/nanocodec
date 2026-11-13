//! Raw input file layouts (`--input-format`): planar I420 (the default),
//! YV12, or semi-planar NV12, tightly packed frame after frame.
//!
//! I420 is read straight into the encoder's source buffer. The other
//! layouts are read one packed frame at a time and handed to
//! [`Encoder::encode_raw`], the same path a camera capture loop uses.
//!
//! The input is only ever read in order, so it may be a pipe or FIFO (the
//! evaluation harness streams decoded clips through one): the look-ahead
//! frame is read ahead and replayed, never fetched by seeking.

use std::io::{Cursor, ErrorKind, Read};

use nc_common::{RawFrame, chroma_len, yuv};
use nc_encoder::{CodecBackend, EncodedFrame, Encoder};

/// Layout of the raw input file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// Y, Cb, Cr planes.
    I420,
    /// Y, Cr, Cb planes.
    Yv12,
    /// Y plane, then interleaved Cb, Cr pairs.
    Nv12,
}

impl InputFormat {
    /// Parse an `--input-format` value.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "i420" => Ok(Self::I420),
            "yv12" => Ok(Self::Yv12),
            "nv12" => Ok(Self::Nv12),
            other => Err(format!(
                "unsupported --input-format '{other}' (i420, yv12, nv12)"
            )),
        }
    }
}

/// Reads frames of one layout and encodes them.
pub struct FrameReader<R: Read> {
    input: Replayed<R>,
    format: InputFormat,
    /// One packed frame, for layouts that are not read in place.
    packed: Vec<u8>,
}

impl<R: Read> FrameReader<R> {
    /// A reader of `width` x `height` frames of `format` from `input`.
    pub fn new(input: R, format: InputFormat, width: usize, height: usize) -> Self {
        Self {
            input: Replayed {
                replay: Cursor::new(Vec::new()),
                input,
            },
            format,
            packed: packed_buffer(format, width, height),
        }
    }

    /// Frames from now on are `width` x `height` (`--resize`).
    pub fn set_size(&mut self, width: usize, height: usize) {
        self.packed = packed_buffer(self.format, width, height);
    }

    /// Read the next frame and encode it (captured at `timestamp_us`, if
    /// known), or `None` at the end of the input.
    pub fn encode_next<B: CodecBackend>(
        &mut self,
        encoder: &mut Encoder<B>,
        timestamp_us: Option<i64>,
    ) -> Result<Option<EncodedFrame>, String> {
        if self.format == InputFormat::I420 {
            // Read straight into the encoder's source buffer: no extra copy.
            let read = yuv::read_frame(&mut self.input, encoder.source_mut())
                .map_err(|e| e.to_string())?;
            return Ok(read.then(|| match timestamp_us {
                Some(t) => encoder.encode_written_at(t),
                None => encoder.encode_written(),
            }));
        }
        match self.input.read_exact(&mut self.packed) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.to_string()),
        }
        let (width, height) = (encoder.config().width, encoder.config().height);
        let frame = match self.format {
            InputFormat::Nv12 => RawFrame::packed_nv12(&self.packed, width, height),
            // YV12 is I420 with the chroma planes swapped.
            InputFormat::Yv12 => packed_yv12(&self.packed, width, height),
            InputFormat::I420 => unreachable!("read in place above"),
        }
        .map_err(|e| e.to_string())?;
        encoder
            .encode_raw(&frame)
            .map(Some)
            .map_err(|e| e.to_string())
    }

    /// Hand the second frame to `encoder` before the first is encoded
    /// ([`Encoder::look_ahead`]), if it wants one. Both frames are read
    /// ahead in order and replayed by [`FrameReader::encode_next`]; an input
    /// of one frame hands nothing over.
    pub fn look_ahead_second<B: CodecBackend>(
        &mut self,
        encoder: &mut Encoder<B>,
    ) -> Result<(), String> {
        if !encoder.wants_look_ahead() {
            return Ok(());
        }
        let (width, height) = (encoder.config().width, encoder.config().height);
        let frame_bytes = width * height + 2 * chroma_len(width) * chroma_len(height);
        let mut ahead = Vec::with_capacity(2 * frame_bytes);
        (&mut self.input)
            .take(2 * frame_bytes as u64)
            .read_to_end(&mut ahead)
            .map_err(|e| e.to_string())?;
        let handed = if ahead.len() == 2 * frame_bytes {
            let packed = &ahead[frame_bytes..];
            let frame = match self.format {
                InputFormat::I420 => RawFrame::packed_i420(packed, width, height),
                InputFormat::Nv12 => RawFrame::packed_nv12(packed, width, height),
                InputFormat::Yv12 => packed_yv12(packed, width, height),
            }
            .map_err(|e| e.to_string())?;
            encoder.look_ahead(&frame).map_err(|e| e.to_string())
        } else {
            Ok(())
        };
        self.input.replay = Cursor::new(ahead);
        handed
    }
}

/// The input, after any bytes read ahead and handed back for replay.
struct Replayed<R: Read> {
    replay: Cursor<Vec<u8>>,
    input: R,
}

impl<R: Read> Read for Replayed<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.replay.read(buf)? {
            0 => self.input.read(buf),
            read => Ok(read),
        }
    }
}

/// The buffer one packed `width` x `height` frame of `format` is read into
/// (I420 is read in place).
fn packed_buffer(format: InputFormat, width: usize, height: usize) -> Vec<u8> {
    let chroma = chroma_len(width) * chroma_len(height);
    match format {
        InputFormat::I420 => Vec::new(),
        // Both layouts carry two chroma planes' worth of samples.
        InputFormat::Yv12 | InputFormat::Nv12 => vec![0; width * height + 2 * chroma],
    }
}

/// A tightly packed YV12 frame (Y, then Cr, then Cb).
fn packed_yv12(data: &[u8], width: usize, height: usize) -> nc_common::Result<RawFrame<'_>> {
    // Read it as I420 and swap the chroma planes back.
    match RawFrame::packed_i420(data, width, height)? {
        RawFrame::I420 { y, u: cr, v: cb } => Ok(RawFrame::yv12(y, cr, cb)),
        RawFrame::Nv12 { .. } => unreachable!("packed_i420 builds a planar frame"),
    }
}

#[cfg(test)]
mod tests {
    use super::{InputFormat, packed_yv12};
    use nc_common::RawFrame;

    #[test]
    fn parses_formats() {
        assert_eq!(InputFormat::parse("nv12"), Ok(InputFormat::Nv12));
        assert!(InputFormat::parse("rgb").is_err());
    }

    #[test]
    fn yv12_reads_cr_before_cb() {
        let data = [0, 0, 0, 0, 9, 4]; // 2x2 luma, Cr 9, Cb 4
        let RawFrame::I420 { u, v, .. } = packed_yv12(&data, 2, 2).unwrap() else {
            panic!("planar")
        };
        assert_eq!((u.row(0)[0], v.row(0)[0]), (4, 9));
    }
}
