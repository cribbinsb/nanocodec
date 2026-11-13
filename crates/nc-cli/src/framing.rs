//! Container framing of our streams: IVF frames (AV1, VP9, AV2) and Annex-B
//! access units (H.264, HEVC), read with bounded memory per frame, and the
//! IVF writer the encoder uses.
//!
//! Annex-B parameter sets are attached to the following VCL NAL: the emitted
//! subset has exactly one slice per picture and needs no frame-number
//! heuristics.

use std::io::{self, BufRead, Write};

use nc_encoder::Codec;

/// Bounds memory per compressed frame, not the length of the input movie.
pub(crate) const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
const MAX_PARAMETER_NALS: usize = 16;
/// IVF file header size.
pub(crate) const IVF_HEADER_BYTES: usize = 32;
/// IVF per-frame header size (size and timestamp).
pub(crate) const IVF_FRAME_HEADER_BYTES: usize = 12;
/// Byte offset of the frame count in the IVF file header.
pub(crate) const IVF_FRAME_COUNT_OFFSET: usize = 24;
/// IVF file signature and the only header version we read or write.
const IVF_SIGNATURE: &[u8; 4] = b"DKIF";
const IVF_VERSION: u16 = 0;
/// Byte range of the codec `FourCC` in the IVF file header.
const IVF_FOURCC: std::ops::Range<usize> = 8..12;
const ANNEX_B_PREFIX: [u8; 4] = [0, 0, 0, 1];

/// A validated IVF file header.
pub(crate) struct IvfHeader {
    /// The header as read. Its picture size is the first picture's only (a
    /// stream may change size at a key frame), so it is validated, not used.
    pub bytes: [u8; IVF_HEADER_BYTES],
}

/// One IVF frame: its header (size, timestamp) and payload.
pub(crate) struct IvfFrame<'a> {
    pub header: &'a [u8; IVF_FRAME_HEADER_BYTES],
    pub packet: &'a [u8],
}

/// The IVF `FourCC` of `codec`, or `None` for the Annex-B codecs.
pub(crate) const fn ivf_fourcc(codec: Codec) -> Option<&'static [u8; 4]> {
    match codec {
        Codec::Av1 => Some(b"AV01"),
        Codec::Vp9 => Some(b"VP90"),
        Codec::Av2 => Some(b"AV02"),
        Codec::H264 | Codec::Hevc | Codec::Vvc => None,
    }
}

impl IvfHeader {
    /// The codec `FourCC` the container declares.
    pub fn fourcc(&self) -> &[u8] {
        &self.bytes[IVF_FOURCC]
    }
}

/// The `FourCC`s of the IVF files the decoders read directly.
const DECODER_FOURCCS: [&[u8; 4]; 3] = [b"AV01", b"VP90", b"AV02"];
/// Also accepted by [`read_ivf_annex_b`]: H.264 and HEVC access units in
/// IVF, as `web/streams-ref` stores them.
#[cfg(feature = "analysis")]
const ANNEX_B_FOURCCS: [&[u8; 4]; 2] = [b"H264", b"HEVC"];

/// Read an AV1, VP9 or AV2 IVF file, calling `visit` with the header and every
/// frame; returns the number of frames.
pub(crate) fn read_ivf<R: BufRead>(
    input: &mut R,
    visit: impl FnMut(&IvfHeader, IvfFrame<'_>) -> Result<(), String>,
) -> Result<u64, String> {
    read_ivf_of(input, &DECODER_FOURCCS, visit)
}

/// Read an IVF file of H.264 or HEVC Annex-B access units (one per frame).
#[cfg(feature = "analysis")]
pub(crate) fn read_ivf_annex_b<R: BufRead>(
    input: &mut R,
    visit: impl FnMut(&IvfHeader, IvfFrame<'_>) -> Result<(), String>,
) -> Result<u64, String> {
    read_ivf_of(input, &ANNEX_B_FOURCCS, visit)
}

/// Whether `data` starts like an IVF file.
#[cfg(feature = "analysis")]
pub(crate) fn is_ivf(data: &[u8]) -> bool {
    data.starts_with(IVF_SIGNATURE)
}

fn read_ivf_of<R: BufRead>(
    input: &mut R,
    fourccs: &[&[u8; 4]],
    mut visit: impl FnMut(&IvfHeader, IvfFrame<'_>) -> Result<(), String>,
) -> Result<u64, String> {
    let mut header = [0; IVF_HEADER_BYTES];
    input
        .read_exact(&mut header)
        .map_err(|error| format!("truncated IVF header: {error}"))?;
    if &header[..4] != IVF_SIGNATURE
        || u16::from_le_bytes([header[4], header[5]]) != IVF_VERSION
        || u16::from_le_bytes([header[6], header[7]]) as usize != IVF_HEADER_BYTES
        || !fourccs.iter().any(|f| header[IVF_FOURCC] == f[..])
    {
        return Err("unsupported IVF header (version or codec FourCC)".into());
    }
    // The header's picture size is informational: frames carry their own,
    // and libvpx writes 0 for spatial-layer streams (vp90-2-22-svc_*.ivf).
    if header[16..20] == [0; 4] || header[20..24] == [0; 4] {
        return Err("invalid IVF time base".into());
    }
    let header = IvfHeader { bytes: header };
    let mut count = 0;
    while !input
        .fill_buf()
        .map_err(|error| error.to_string())?
        .is_empty()
    {
        let mut frame_header = [0; IVF_FRAME_HEADER_BYTES];
        input
            .read_exact(&mut frame_header)
            .map_err(|error| format!("truncated IVF frame header: {error}"))?;
        let size = u32::from_le_bytes(
            frame_header[..4]
                .try_into()
                .map_err(|_| "invalid IVF frame header")?,
        ) as usize;
        if size == 0 || size > MAX_PACKET_BYTES {
            return Err("IVF frame exceeds 64 MiB limit or is empty".into());
        }
        let mut packet = vec![0; size];
        input
            .read_exact(&mut packet)
            .map_err(|error| format!("truncated IVF frame payload: {error}"))?;
        // Timestamps and the nonzero time base are deliberately independent
        // of decoding; neither a fixed frame rate nor timestamp step is assumed.
        let frame = IvfFrame {
            header: &frame_header,
            packet: &packet,
        };
        visit(&header, frame)?;
        count += 1;
    }
    Ok(count)
}

/// Read an H.264, HEVC or VVC Annex-B stream, calling `visit` with every access
/// unit (its parameter sets and one slice, start codes included); returns
/// the number of access units.
pub(crate) fn read_annex_b<R: BufRead>(
    codec: Codec,
    input: &mut R,
    mut visit: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<u64, String> {
    let mut reader = AnnexBReader::new(input, MAX_PACKET_BYTES);
    let mut packet = Vec::new();
    let mut parameter_count = 0;
    let mut count = 0;
    while let Some(nal) = reader.next_nal()? {
        let vcl = match codec {
            Codec::H264 => match nal[0] & 31 {
                1 | 5 => true,
                7 | 8 => false,
                _ => return Err("unsupported H264 NAL type outside emitted subset".into()),
            },
            Codec::Hevc => {
                if nal.len() < 2 {
                    return Err("truncated HEVC NAL header".into());
                }
                match (nal[0] >> 1) & 63 {
                    0 | 1 | 19 => true,
                    32..=34 => false,
                    _ => return Err("unsupported HEVC NAL type outside emitted subset".into()),
                }
            }
            Codec::Vvc => {
                if nal.len() < 2 {
                    return Err("truncated VVC NAL header".into());
                }
                // H.266 Table 5 NAL unit types: TRAIL_NUT 0 and IDR_N_LP 8
                // carry a picture; SPS/PPS and prefix APS precede it.
                match nal[1] >> 3 {
                    0 | 8 => true,
                    15..=17 => false,
                    _ => return Err("unsupported VVC NAL type outside emitted subset".into()),
                }
            }
            Codec::Av1 | Codec::Vp9 | Codec::Av2 => {
                return Err("AV1/AV2/VP9 input must be IVF".into());
            }
        };
        if packet.len() > MAX_PACKET_BYTES - ANNEX_B_PREFIX.len()
            || nal.len() > MAX_PACKET_BYTES - ANNEX_B_PREFIX.len() - packet.len()
        {
            return Err("Annex-B access unit exceeds 64 MiB limit".into());
        }
        packet.extend_from_slice(&ANNEX_B_PREFIX);
        packet.extend_from_slice(&nal);
        if vcl {
            visit(&packet)?;
            count += 1;
            packet.clear();
            parameter_count = 0;
        } else {
            parameter_count += 1;
            if parameter_count > MAX_PARAMETER_NALS {
                return Err("too many parameter NALs before picture".into());
            }
        }
    }
    if !packet.is_empty() {
        return Err("Annex-B stream ended without a picture after parameter sets".into());
    }
    Ok(count)
}

/// Visit every NAL unit of an Annex-B stream (without start codes), for
/// decoders that take NAL units one at a time. Returns the count.
pub(crate) fn read_nal_units<R: BufRead>(
    input: &mut R,
    mut visit: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<u64, String> {
    let mut reader = AnnexBReader::new(input, MAX_PACKET_BYTES);
    let mut count = 0;
    while let Some(nal) = reader.next_nal()? {
        visit(&nal)?;
        count += 1;
    }
    Ok(count)
}

/// Keeps only one bounded NAL. Pending zeros allow start codes to straddle
/// arbitrary reader chunks and exclude Annex-B `trailing_zero_8bits`.
struct AnnexBReader<R> {
    input: R,
    started: bool,
    ended: bool,
    limit: usize,
}
impl<R: BufRead> AnnexBReader<R> {
    fn new(input: R, limit: usize) -> Self {
        Self {
            input,
            started: false,
            ended: false,
            limit,
        }
    }
    fn byte(&mut self) -> Result<Option<u8>, String> {
        let byte = self
            .input
            .fill_buf()
            .map_err(|error| error.to_string())?
            .first()
            .copied();
        if byte.is_some() {
            self.input.consume(1);
        }
        Ok(byte)
    }
    fn start(&mut self) -> Result<bool, String> {
        let mut zeros = 0;
        while let Some(byte) = self.byte()? {
            if byte == 0 {
                zeros += 1;
                if zeros > self.limit {
                    return Err("Annex-B prefix exceeds allocation limit".into());
                }
            } else if byte == 1 && zeros >= 2 {
                self.started = true;
                return Ok(true);
            } else {
                return Err("missing Annex-B start code or leading garbage".into());
            }
        }
        if zeros != 0 {
            return Err("truncated Annex-B start code".into());
        }
        Ok(false)
    }
    /// The next NAL unit. Reads whole buffered chunks: each is searched for
    /// the next 01 byte, and only the bytes before it are inspected for
    /// zeros (a start code is 00 00 01; zeros before it, including
    /// `trailing_zero_8bits`, belong to no NAL).
    fn next_nal(&mut self) -> Result<Option<Vec<u8>>, String> {
        if self.ended || (!self.started && !self.start()?) {
            return Ok(None);
        }
        let mut nal = Vec::new();
        // Zeros read but not yet appended: they are data only if a byte
        // other than a start code's 01 follows.
        let mut zeros = 0;
        loop {
            let chunk = self.input.fill_buf().map_err(|error| error.to_string())?;
            if chunk.is_empty() {
                break;
            }
            let one = chunk.iter().position(|&byte| byte == 1);
            let before = &chunk[..one.unwrap_or(chunk.len())];
            let tail = before.iter().rev().take_while(|&&byte| byte == 0).count();
            let (data, run) = if tail == before.len() {
                (&before[..0], zeros + tail)
            } else {
                (&before[..before.len() - tail], tail)
            };
            if run > self.limit {
                return Err("Annex-B zero run exceeds allocation limit".into());
            }
            if !data.is_empty() {
                append(&mut nal, zeros, data, self.limit)?;
            }
            zeros = run;
            let Some(one) = one else {
                let consumed = chunk.len();
                self.input.consume(consumed);
                continue;
            };
            self.input.consume(one + 1);
            if zeros >= 2 {
                if nal.is_empty() {
                    return Err("empty Annex-B NAL".into());
                }
                return Ok(Some(nal));
            }
            append(&mut nal, zeros, &[1], self.limit)?;
            zeros = 0;
        }
        self.ended = true;
        if nal.is_empty() {
            return Err("empty Annex-B NAL".into());
        }
        Ok(Some(nal))
    }
}

/// Append `zeros` zero bytes and then `data` to `nal`, within `limit`.
fn append(nal: &mut Vec<u8>, zeros: usize, data: &[u8], limit: usize) -> Result<(), String> {
    if nal.len() + zeros + data.len() > limit {
        return Err("Annex-B NAL exceeds allocation limit".into());
    }
    nal.resize(nal.len() + zeros, 0);
    nal.extend_from_slice(data);
    Ok(())
}

/// Writes an IVF stream (the framing aomdec, vpxdec, avmdec, dav1d and ffmpeg
/// read) to any [`Write`]. Layout, all little-endian: the 32-byte file header,
/// then per frame a 12-byte header (payload size, 64-bit timestamp) and the
/// payload (one temporal unit or VP9 frame).
pub(crate) struct IvfWriter<W: Write> {
    out: W,
}

impl<W: Write> IvfWriter<W> {
    /// Write the file header. The time base is `fps_den / fps_num` seconds;
    /// `frame_count` is informational (decoders read until end of file).
    ///
    /// # Errors
    ///
    /// A codec without IVF framing, or any error of the underlying writer.
    pub fn new(
        mut out: W,
        codec: Codec,
        width: u16,
        height: u16,
        fps_num: u32,
        fps_den: u32,
        frame_count: u32,
    ) -> io::Result<Self> {
        let fourcc = ivf_fourcc(codec).ok_or_else(|| io::Error::other("codec does not use IVF"))?;
        out.write_all(IVF_SIGNATURE)?;
        out.write_all(&IVF_VERSION.to_le_bytes())?;
        out.write_all(&(IVF_HEADER_BYTES as u16).to_le_bytes())?;
        out.write_all(fourcc)?;
        out.write_all(&width.to_le_bytes())?;
        out.write_all(&height.to_le_bytes())?;
        out.write_all(&fps_num.to_le_bytes())?; // time base denominator
        out.write_all(&fps_den.to_le_bytes())?; // time base numerator
        out.write_all(&frame_count.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?; // unused
        Ok(Self { out })
    }

    /// Append one frame with presentation timestamp `pts` (time-base units).
    ///
    /// # Errors
    ///
    /// A payload over 4 GiB, or any error of the underlying writer.
    pub fn write_frame(&mut self, data: &[u8], pts: u64) -> io::Result<()> {
        let size = u32::try_from(data.len())
            .map_err(|_| io::Error::other("IVF frame larger than 4 GiB"))?;
        self.out.write_all(&size.to_le_bytes())?;
        self.out.write_all(&pts.to_le_bytes())?;
        self.out.write_all(data)
    }

    /// Flush and return the underlying writer.
    ///
    /// # Errors
    ///
    /// Any error flushing the underlying writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};
    #[test]
    fn ivf_header_is_32_bytes_and_frames_are_prefixed() {
        let mut ivf = IvfWriter::new(Vec::new(), Codec::Av1, 64, 48, 30, 1, 1).unwrap();
        ivf.write_frame(&[1, 2, 3], 0).unwrap();
        let bytes = ivf.finish().unwrap();
        assert_eq!(&bytes[..4], IVF_SIGNATURE);
        assert_eq!(&bytes[IVF_FOURCC], b"AV01");
        assert_eq!(bytes.len(), IVF_HEADER_BYTES + IVF_FRAME_HEADER_BYTES + 3);
        assert_eq!(
            &bytes[IVF_HEADER_BYTES..IVF_HEADER_BYTES + 4],
            &3u32.to_le_bytes()
        );
        assert!(IvfWriter::new(Vec::new(), Codec::Hevc, 64, 48, 30, 1, 1).is_err());
    }
    #[test]
    fn annex_b_start_codes_cross_every_reader_boundary() {
        let data = [
            0, 0, 0, 1, 0x67, 2, 0, 0, 3, 1, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4, 0, 0,
        ];
        for capacity in 1..9 {
            let input = BufReader::with_capacity(capacity, Cursor::new(data));
            let mut reader = AnnexBReader::new(input, 64);
            assert_eq!(reader.next_nal().unwrap().unwrap(), [0x67, 2, 0, 0, 3, 1]);
            assert_eq!(reader.next_nal().unwrap().unwrap(), [0x68, 3]);
            assert_eq!(reader.next_nal().unwrap().unwrap(), [0x65, 4]);
            assert!(reader.next_nal().unwrap().is_none());
        }
    }
    #[test]
    fn annex_b_rejects_garbage_empty_units_and_oversized_nals() {
        for input in [
            vec![1, 2, 3],
            vec![0, 0],
            vec![0, 0, 1],
            vec![0, 0, 1, 0, 0, 1],
            vec![0, 0, 1, 0x65, 2, 3, 4, 5, 6, 7],
        ] {
            assert!(AnnexBReader::new(Cursor::new(input), 5).next_nal().is_err());
        }
        let mut reader = AnnexBReader::new(
            Cursor::new([0, 0, 1, 0x65, 2, 3, 4, 5, 0, 0, 1, 0x61, 8]),
            5,
        );
        assert_eq!(reader.next_nal().unwrap().unwrap(), [0x65, 2, 3, 4, 5]);
        assert_eq!(reader.next_nal().unwrap().unwrap(), [0x61, 8]);
    }
}
