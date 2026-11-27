//! MP4 output (ISO/IEC 14496-12, with 14496-15 for H.264/HEVC/VVC, the
//! AV1-ISOBMFF and VP-codec-ISOBMFF bindings).
//!
//! Layout: `ftyp`, one `mdat` (64-bit size, patched at the end) holding the
//! samples as they come, then `moov` with one video track whose sample
//! table gives every sample's duration (`stts`: a held frame lengthens the
//! one before it), key samples (`stss`, for seeking), sizes (`stsz`) and
//! offsets (`co64`, one sample per chunk). The time scale is the stream's:
//! the nominal frame rate's numerator, or microseconds for timestamped
//! input. Sample entries: `avc3`, `hev1`, `vvi1` (parameter sets also
//! in-band), `av01`, `vp09`; with `colr` (`nclx`) when colour is signalled.

use std::io::{self, Seek, SeekFrom, Write};

use nc_encoder::Codec;

use super::codec_config::{self, CodecConfig};
use super::{Track, invalid};

/// 16.16 fixed-point one.
const FIXED_ONE: u32 = 0x0001_0000;
/// The identity matrix of `mvhd` and `tkhd`.
const IDENTITY_MATRIX: [u32; 9] = [FIXED_ONE, 0, 0, 0, FIXED_ONE, 0, 0, 0, 0x4000_0000];
/// `tkhd` flags: track enabled, in movie, in preview.
const TRACK_ENABLED_IN_MOVIE: u32 = 0x7;
/// ISO 639-2 "und" packed in `mdhd`.
const LANGUAGE_UNDETERMINED: u16 = 0x55C4;
/// 72 dpi in 16.16 (`VisualSampleEntry` resolution).
const DPI_72: u32 = 0x0048_0000;
/// `VisualSampleEntry` depth: colour without alpha.
const DEPTH_COLOUR: u16 = 0x0018;

/// One written sample.
struct Sample {
    offset: u64,
    size: u32,
    pts: u64,
    key: bool,
}

/// An MP4 file being written to `out`.
pub(crate) struct Mp4Writer<W: Write + Seek> {
    out: W,
    track: Track,
    config: Option<CodecConfig>,
    /// Where the `mdat` box starts.
    mdat_at: u64,
    samples: Vec<Sample>,
}

/// A box: size, `kind`, `body`.
fn boxed(kind: [u8; 4], body: &[u8]) -> Vec<u8> {
    let size = u32::try_from(body.len() + 8).expect("header boxes stay below 4 GiB");
    let mut out = size.to_be_bytes().to_vec();
    out.extend_from_slice(&kind);
    out.extend_from_slice(body);
    out
}

/// A full box: version, flags, then `body`.
fn full_box(kind: [u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut head = (u32::from(version) << 24 | flags).to_be_bytes().to_vec();
    head.extend_from_slice(body);
    boxed(kind, &head)
}

/// Big-endian fields appended in order.
#[derive(Default)]
struct Fields(Vec<u8>);

impl Fields {
    fn u8(mut self, v: u8) -> Self {
        self.0.push(v);
        self
    }
    fn u16(mut self, v: u16) -> Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }
    fn u32(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }
    fn u64(mut self, v: u64) -> Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }
    fn bytes(mut self, v: &[u8]) -> Self {
        self.0.extend_from_slice(v);
        self
    }
    fn matrix(self) -> Self {
        IDENTITY_MATRIX.iter().fold(self, |f, &v| f.u32(v))
    }
}

impl<W: Write + Seek> Mp4Writer<W> {
    pub(crate) fn new(out: W, track: Track) -> Self {
        Self {
            out,
            track,
            config: None,
            mdat_at: 0,
            samples: Vec::new(),
        }
    }

    /// `ftyp` and the `mdat` header, once the first key frame gives the
    /// configuration.
    fn start(&mut self) -> io::Result<()> {
        let mut brands = Fields::default()
            .bytes(b"isom")
            .u32(0x200)
            .bytes(b"isom")
            .bytes(b"iso6")
            .bytes(b"mp41");
        if self.track.codec == Codec::Av1 {
            brands = brands.bytes(b"av01");
        }
        self.out.write_all(&boxed(*b"ftyp", &brands.0))?;
        self.mdat_at = self.out.stream_position()?;
        self.out
            .write_all(&Fields::default().u32(1).bytes(b"mdat").u64(0).0)?;
        Ok(())
    }

    /// Append one coded frame presented at `pts`.
    pub(crate) fn write_frame(&mut self, unit: &[u8], pts: u64, key: bool) -> io::Result<()> {
        if self.config.is_none() {
            if !key {
                return Err(invalid("the stream must start with a key frame"));
            }
            self.config = Some(CodecConfig::from_key(&self.track, unit)?);
            self.start()?;
        }
        let sample = codec_config::sample(self.track.codec, unit)?;
        let offset = self.out.stream_position()?;
        self.out.write_all(&sample)?;
        self.samples.push(Sample {
            offset,
            size: u32::try_from(sample.len()).map_err(|_| invalid("sample too large"))?,
            pts,
            key,
        });
        Ok(())
    }

    /// Complete `mdat` and write `moov`; the last frame shows until
    /// `end_pts`.
    pub(crate) fn finish(mut self, end_pts: u64) -> io::Result<W> {
        let config = self
            .config
            .take()
            .ok_or_else(|| invalid("no frames were coded"))?;
        let end = self.out.stream_position()?;
        self.out.seek(SeekFrom::Start(self.mdat_at + 8))?;
        self.out.write_all(&(end - self.mdat_at).to_be_bytes())?;
        self.out.seek(SeekFrom::Start(end))?;
        let moov = self.moov(&config, end_pts)?;
        self.out.write_all(&moov)?;
        self.out.flush()?;
        Ok(self.out)
    }

    fn moov(&self, config: &CodecConfig, end_pts: u64) -> io::Result<Vec<u8>> {
        let base = self.track.time_base;
        let timescale = base.units_per_second;
        let duration = base.ticks(end_pts);
        let mvhd = Fields::default()
            .u64(0)
            .u64(0)
            .u32(timescale)
            .u64(duration)
            .u32(FIXED_ONE) // rate
            .u16(0x0100) // volume
            .u16(0)
            .u64(0)
            .matrix()
            .bytes(&[0; 24])
            .u32(2); // next_track_ID
        let mvhd = full_box(*b"mvhd", 1, 0, &mvhd.0);
        let tkhd = Fields::default()
            .u64(0)
            .u64(0)
            .u32(1) // track_ID
            .u32(0)
            .u64(duration)
            .u64(0)
            .u16(0) // layer
            .u16(0) // alternate_group
            .u16(0) // volume
            .u16(0)
            .matrix()
            .u32(self.track.width << 16)
            .u32(self.track.height << 16);
        let tkhd = full_box(*b"tkhd", 1, TRACK_ENABLED_IN_MOVIE, &tkhd.0);
        let mdhd = Fields::default()
            .u64(0)
            .u64(0)
            .u32(timescale)
            .u64(duration)
            .u16(LANGUAGE_UNDETERMINED)
            .u16(0);
        let mdhd = full_box(*b"mdhd", 1, 0, &mdhd.0);
        let hdlr = Fields::default()
            .u32(0)
            .bytes(b"vide")
            .bytes(&[0; 12])
            .bytes(b"VideoHandler\0");
        let hdlr = full_box(*b"hdlr", 0, 0, &hdlr.0);
        let vmhd = full_box(*b"vmhd", 0, 1, &[0; 8]);
        let dref = full_box(
            *b"dref",
            0,
            0,
            &[1u32.to_be_bytes().to_vec(), full_box(*b"url ", 0, 1, &[])].concat(),
        );
        let dinf = boxed(*b"dinf", &dref);
        let stbl = boxed(
            *b"stbl",
            &[
                self.stsd(config),
                self.stts(end_pts)?,
                self.stss(),
                full_box(
                    *b"stsc",
                    0,
                    0,
                    &Fields::default().u32(1).u32(1).u32(1).u32(1).0,
                ),
                self.stsz(),
                self.co64(),
            ]
            .concat(),
        );
        let minf = boxed(*b"minf", &[vmhd, dinf, stbl].concat());
        let mdia = boxed(*b"mdia", &[mdhd, hdlr, minf].concat());
        let trak = boxed(*b"trak", &[tkhd, mdia].concat());
        Ok(boxed(*b"moov", &[mvhd, trak].concat()))
    }

    /// `stsd` with the codec's sample entry.
    fn stsd(&self, config: &CodecConfig) -> Vec<u8> {
        let (kind, record) = match self.track.codec {
            Codec::H264 => (b"avc3", boxed(*b"avcC", &config.record)),
            Codec::Hevc => (b"hev1", boxed(*b"hvcC", &config.record)),
            Codec::Vvc => (b"vvi1", full_box(*b"vvcC", 0, 0, &config.record)),
            Codec::Av1 => (b"av01", boxed(*b"av1C", &config.record)),
            Codec::Vp9 | Codec::Av2 => (b"vp09", full_box(*b"vpcC", 1, 0, &config.record)),
        };
        let width = u16::try_from(self.track.width).unwrap_or(u16::MAX);
        let height = u16::try_from(self.track.height).unwrap_or(u16::MAX);
        let mut name = [0u8; 32];
        let label = b"nanocodec";
        name[0] = label.len() as u8;
        name[1..=label.len()].copy_from_slice(label);
        let mut entry = Fields::default()
            .bytes(&[0; 6])
            .u16(1) // data_reference_index
            .bytes(&[0; 16])
            .u16(width)
            .u16(height)
            .u32(DPI_72)
            .u32(DPI_72)
            .u32(0)
            .u16(1) // frame_count
            .bytes(&name)
            .u16(DEPTH_COLOUR)
            .u16(0xFFFF) // pre_defined -1
            .bytes(&record);
        let colour = self.track.colour;
        if colour.is_signalled() {
            let (primaries, transfer, matrix) = colour
                .description
                .map_or((2, 2, 2), |d| (d.primaries, d.transfer, d.matrix));
            let nclx = Fields::default()
                .bytes(b"nclx")
                .u16(u16::from(primaries))
                .u16(u16::from(transfer))
                .u16(u16::from(matrix))
                .u8(u8::from(colour.full_range) << 7);
            entry = entry.bytes(&boxed(*b"colr", &nclx.0));
        }
        full_box(
            *b"stsd",
            0,
            0,
            &[1u32.to_be_bytes().to_vec(), boxed(*kind, &entry.0)].concat(),
        )
    }

    /// `stts`: run-length coded sample durations, the last lasting until
    /// `end_pts`.
    fn stts(&self, end_pts: u64) -> io::Result<Vec<u8>> {
        let base = self.track.time_base;
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for (index, sample) in self.samples.iter().enumerate() {
            let next = self.samples.get(index + 1).map_or(end_pts, |s| s.pts);
            let delta = u32::try_from(base.ticks(next.saturating_sub(sample.pts)))
                .map_err(|_| invalid("sample duration"))?;
            match runs.last_mut() {
                Some((count, d)) if *d == delta => *count += 1,
                _ => runs.push((1, delta)),
            }
        }
        let mut body = Fields::default().u32(runs.len() as u32);
        for (count, delta) in runs {
            body = body.u32(count).u32(delta);
        }
        Ok(full_box(*b"stts", 0, 0, &body.0))
    }

    /// `stss`: the key samples (1-based).
    fn stss(&self) -> Vec<u8> {
        let keys: Vec<u32> = (1..)
            .zip(&self.samples)
            .filter(|(_, s)| s.key)
            .map(|(n, _)| n)
            .collect();
        let body = keys
            .iter()
            .fold(Fields::default().u32(keys.len() as u32), |f, &n| f.u32(n));
        full_box(*b"stss", 0, 0, &body.0)
    }

    fn stsz(&self) -> Vec<u8> {
        let body = self.samples.iter().fold(
            Fields::default().u32(0).u32(self.samples.len() as u32),
            |f, s| f.u32(s.size),
        );
        full_box(*b"stsz", 0, 0, &body.0)
    }

    fn co64(&self) -> Vec<u8> {
        let body = self
            .samples
            .iter()
            .fold(Fields::default().u32(self.samples.len() as u32), |f, s| {
                f.u64(s.offset)
            });
        full_box(*b"co64", 0, 0, &body.0)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use nc_common::ColourInfo;
    use nc_encoder::FrameRate;

    use super::super::TimeBase;
    use super::*;

    #[test]
    fn held_frames_lengthen_the_sample_before_them() {
        let rate = FrameRate { num: 30, den: 1 };
        let track = Track {
            codec: Codec::Vp9,
            width: 64,
            height: 48,
            frame_rate: rate,
            colour: ColourInfo::default(),
            time_base: TimeBase::frames(rate),
        };
        let mut writer = Mp4Writer::new(Cursor::new(Vec::new()), track);
        writer.write_frame(&[0x82, 0x49], 0, true).unwrap();
        writer.write_frame(&[0x86, 0x00], 3, false).unwrap();
        let bytes = writer.finish(4).unwrap().into_inner();
        // stts: two runs, a 3-frame sample then a 1-frame sample.
        let stts = bytes.windows(4).position(|w| w == b"stts").unwrap();
        assert_eq!(
            bytes[stts + 8..stts + 24],
            [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1]
        );
        // mdat holds both samples and its 64-bit size covers them.
        let mdat = bytes.windows(4).position(|w| w == b"mdat").unwrap() - 4;
        let size = u64::from_be_bytes(bytes[mdat + 8..mdat + 16].try_into().unwrap());
        // The 16-byte header (size 1, type, 64-bit size) and two samples.
        assert_eq!(size, 16 + 4);
    }
}
