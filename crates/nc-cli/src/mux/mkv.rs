//! Matroska and `WebM` output (RFC 9559; `WebM`: the VP9/AV1 subset).
//!
//! Layout: EBML header; `Segment` (its size patched at the end) holding a
//! `SeekHead` (written last into reserved space), `Info` (1 ms timestamps,
//! `Duration` patched at the end), `Tracks` (one video track, the codec's
//! configuration record as `CodecPrivate`, `DefaultDuration` the nominal
//! frame), one `Cluster` per key frame (or per 30 s, so block times fit
//! their 16 bits), and `Cues` listing every cluster's key frame for
//! seeking. Codec IDs: `V_MPEG4/ISO/AVC`, `V_MPEGH/ISO/HEVC`,
//! `V_MPEGI/ISO/VVC`, `V_AV1`, `V_VP9` (Matroska codec mappings).

use std::io::{self, Seek, SeekFrom, Write};

use nc_encoder::Codec;

use super::codec_config::{self, CodecConfig};
use super::ebml::{self, element, float, open, string, uint, uint8, void};
use super::{Track, invalid};

const EBML: u32 = 0x1A45_DFA3;
const EBML_VERSION: u32 = 0x4286;
const EBML_READ_VERSION: u32 = 0x42F7;
const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
const DOC_TYPE: u32 = 0x4282;
const DOC_TYPE_VERSION: u32 = 0x4287;
const DOC_TYPE_READ_VERSION: u32 = 0x4285;
const SEGMENT: u32 = 0x1853_8067;
const SEEK_HEAD: u32 = 0x114D_9B74;
const SEEK: u32 = 0x4DBB;
const SEEK_ID: u32 = 0x53AB;
const SEEK_POSITION: u32 = 0x53AC;
const INFO: u32 = 0x1549_A966;
const TIMESTAMP_SCALE: u32 = 0x2A_D7B1;
const DURATION: u32 = 0x4489;
const MUXING_APP: u32 = 0x4D80;
const WRITING_APP: u32 = 0x5741;
const TRACKS: u32 = 0x1654_AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const TRACK_UID: u32 = 0x73C5;
const TRACK_TYPE: u32 = 0x83;
const FLAG_LACING: u32 = 0x9C;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;
const DEFAULT_DURATION: u32 = 0x23_E383;
const VIDEO: u32 = 0xE0;
const PIXEL_WIDTH: u32 = 0xB0;
const PIXEL_HEIGHT: u32 = 0xBA;
const COLOUR: u32 = 0x55B0;
const MATRIX_COEFFICIENTS: u32 = 0x55B1;
const RANGE: u32 = 0x55B9;
const TRANSFER_CHARACTERISTICS: u32 = 0x55BA;
const PRIMARIES: u32 = 0x55BB;
const CLUSTER: u32 = 0x1F43_B675;
const TIMESTAMP: u32 = 0xE7;
const SIMPLE_BLOCK: u32 = 0xA3;
const CUES: u32 = 0x1C53_BB6B;
const CUE_POINT: u32 = 0xBB;
const CUE_TIME: u32 = 0xB3;
const CUE_TRACK_POSITIONS: u32 = 0xB7;
const CUE_TRACK: u32 = 0xF7;
const CUE_CLUSTER_POSITION: u32 = 0xF1;

/// Nanoseconds per timestamp tick: 1 ms, the Matroska default.
const NANOS_PER_TICK: u64 = 1_000_000;
/// `TrackType` video.
const TRACK_TYPE_VIDEO: u64 = 1;
/// `SimpleBlock` flag of a key frame.
const KEYFRAME_FLAG: u8 = 0x80;
/// The track number as a `SimpleBlock` writes it (variable-size integer 1).
const TRACK_ONE: u8 = 0x81;
/// A cluster ends before a block would sit this far (ms) past its start:
/// block times are signed 16-bit offsets.
const MAX_CLUSTER_SPAN_MS: u64 = 30_000;
/// Bytes reserved after the segment start for the `SeekHead` (three
/// entries take 68).
const SEEK_HEAD_SPACE: usize = 100;
/// `Range` values (RFC 9559 §5.1.4.1.28.40): broadcast, full.
const RANGE_LIMITED: u64 = 1;
const RANGE_FULL: u64 = 2;

/// A Matroska or `WebM` file being written to `out`.
pub(crate) struct MkvWriter<W: Write + Seek> {
    out: W,
    track: Track,
    webm: bool,
    /// File offset of the segment's payload (positions count from here).
    segment_start: u64,
    /// Where the `Duration` value's 8 bytes are.
    duration_at: u64,
    /// Segment positions of `Info` and `Tracks`.
    info_position: u64,
    tracks_position: u64,
    /// Started once the first key frame gave the codec configuration.
    started: bool,
    /// The cluster being gathered: its time (ms) and blocks.
    cluster: Option<(u64, Vec<u8>)>,
    /// Each cluster's time and segment position (cues).
    cues: Vec<(u64, u64)>,
}

impl<W: Write + Seek> MkvWriter<W> {
    pub(crate) fn new(out: W, track: Track, webm: bool) -> Self {
        Self {
            out,
            track,
            webm,
            segment_start: 0,
            duration_at: 0,
            info_position: 0,
            tracks_position: 0,
            started: false,
            cluster: None,
            cues: Vec::new(),
        }
    }

    fn position(&mut self) -> io::Result<u64> {
        self.out.stream_position()
    }

    /// The EBML header, segment start, reserved `SeekHead` space, `Info` and
    /// `Tracks`, once the first key frame gives the configuration.
    fn start(&mut self, config: &CodecConfig) -> io::Result<()> {
        let doc_type = if self.webm { "webm" } else { "matroska" };
        let header = [
            uint(EBML_VERSION, 1),
            uint(EBML_READ_VERSION, 1),
            uint(EBML_MAX_ID_LENGTH, 4),
            uint(EBML_MAX_SIZE_LENGTH, 8),
            string(DOC_TYPE, doc_type),
            uint(DOC_TYPE_VERSION, 4),
            uint(DOC_TYPE_READ_VERSION, 2),
        ]
        .concat();
        self.out.write_all(&element(EBML, &header))?;
        self.out.write_all(&open(SEGMENT))?;
        self.segment_start = self.position()?;
        self.out.write_all(&void(SEEK_HEAD_SPACE))?;

        self.info_position = self.position()? - self.segment_start;
        let app = format!("nanocodec {}", env!("CARGO_PKG_VERSION"));
        let mut info = uint(TIMESTAMP_SCALE, NANOS_PER_TICK);
        let duration_offset = info.len() + 3; // DURATION ID (2) and size (1)
        info.extend(float(DURATION, 0.));
        info.extend(string(MUXING_APP, &app));
        info.extend(string(WRITING_APP, &app));
        let info = element(INFO, &info);
        // INFO's ID (4) and its size (1 byte: the body is short).
        self.duration_at = self.position()? + 5 + duration_offset as u64;
        self.out.write_all(&info)?;

        self.tracks_position = self.position()? - self.segment_start;
        self.out
            .write_all(&element(TRACKS, &self.track_entry(config)))?;
        self.started = true;
        Ok(())
    }

    fn track_entry(&self, config: &CodecConfig) -> Vec<u8> {
        let track = &self.track;
        let codec_id = match track.codec {
            Codec::H264 => "V_MPEG4/ISO/AVC",
            Codec::Hevc => "V_MPEGH/ISO/HEVC",
            Codec::Vvc => "V_MPEGI/ISO/VVC",
            Codec::Av1 => "V_AV1",
            Codec::Vp9 | Codec::Av2 => "V_VP9",
        };
        let mut video = [
            uint(PIXEL_WIDTH, u64::from(track.width)),
            uint(PIXEL_HEIGHT, u64::from(track.height)),
        ]
        .concat();
        if track.colour.is_signalled() {
            let mut colour = uint(
                RANGE,
                if track.colour.full_range {
                    RANGE_FULL
                } else {
                    RANGE_LIMITED
                },
            );
            if let Some(d) = track.colour.description {
                colour.extend(uint(MATRIX_COEFFICIENTS, u64::from(d.matrix)));
                colour.extend(uint(TRANSFER_CHARACTERISTICS, u64::from(d.transfer)));
                colour.extend(uint(PRIMARIES, u64::from(d.primaries)));
            }
            video.extend(element(COLOUR, &colour));
        }
        let frame_ns = u64::from(track.frame_rate.den) * 1_000_000_000
            / u64::from(track.frame_rate.num.max(1));
        let mut entry = [
            uint(TRACK_NUMBER, 1),
            uint(TRACK_UID, 1),
            uint(TRACK_TYPE, TRACK_TYPE_VIDEO),
            uint(FLAG_LACING, 0),
            string(CODEC_ID, codec_id),
        ]
        .concat();
        if track.codec != Codec::Vp9 {
            entry.extend(element(CODEC_PRIVATE, &config.record));
        }
        entry.extend(uint(DEFAULT_DURATION, frame_ns));
        entry.extend(element(VIDEO, &video));
        element(TRACK_ENTRY, &entry)
    }

    /// Append one coded frame presented at `pts`.
    pub(crate) fn write_frame(&mut self, unit: &[u8], pts: u64, key: bool) -> io::Result<()> {
        if !self.started {
            if !key {
                return Err(invalid("the stream must start with a key frame"));
            }
            let config = CodecConfig::from_key(&self.track, unit)?;
            self.start(&config)?;
        }
        let ms = self.track.time_base.millis(pts);
        let new_cluster = match &self.cluster {
            None => true,
            Some((start, _)) => key || ms >= start + MAX_CLUSTER_SPAN_MS,
        };
        if new_cluster {
            self.flush_cluster()?;
            self.cluster = Some((ms, Vec::new()));
        }
        let sample = codec_config::sample(self.track.codec, unit)?;
        let (start, blocks) = self.cluster.as_mut().expect("a cluster is open");
        let relative = i16::try_from(ms - *start).map_err(|_| invalid("block time"))?;
        let mut block = vec![TRACK_ONE];
        block.extend_from_slice(&relative.to_be_bytes());
        block.push(if key { KEYFRAME_FLAG } else { 0 });
        block.extend_from_slice(&sample);
        blocks.extend(element(SIMPLE_BLOCK, &block));
        Ok(())
    }

    /// Write the gathered cluster, noting it for the cues.
    fn flush_cluster(&mut self) -> io::Result<()> {
        if let Some((start, blocks)) = self.cluster.take() {
            let position = self.position()? - self.segment_start;
            let mut body = uint(TIMESTAMP, start);
            body.extend(blocks);
            self.out.write_all(&element(CLUSTER, &body))?;
            self.cues.push((start, position));
        }
        Ok(())
    }

    /// Write the cues and complete the segment size, duration and seek
    /// head; the last frame shows until `end_pts`.
    pub(crate) fn finish(mut self, end_pts: u64) -> io::Result<W> {
        if !self.started {
            return Err(invalid("no frames were coded"));
        }
        self.flush_cluster()?;
        let cues_position = self.position()? - self.segment_start;
        let cues: Vec<u8> = self
            .cues
            .iter()
            .flat_map(|&(time, position)| {
                let positions = [uint(CUE_TRACK, 1), uint(CUE_CLUSTER_POSITION, position)].concat();
                element(
                    CUE_POINT,
                    &[
                        uint(CUE_TIME, time),
                        element(CUE_TRACK_POSITIONS, &positions),
                    ]
                    .concat(),
                )
            })
            .collect();
        self.out.write_all(&element(CUES, &cues))?;
        let end = self.position()?;

        let seeks: Vec<u8> = [
            (INFO, self.info_position),
            (TRACKS, self.tracks_position),
            (CUES, cues_position),
        ]
        .into_iter()
        .flat_map(|(id, position)| {
            element(
                SEEK,
                &[
                    element(SEEK_ID, &id.to_be_bytes()),
                    uint8(SEEK_POSITION, position),
                ]
                .concat(),
            )
        })
        .collect();
        let mut seek_head = element(SEEK_HEAD, &seeks);
        seek_head.extend(void(SEEK_HEAD_SPACE - seek_head.len()));
        self.out.seek(SeekFrom::Start(self.segment_start))?;
        self.out.write_all(&seek_head)?;

        let duration_ms = self.track.time_base.millis(end_pts) as f64;
        self.out.seek(SeekFrom::Start(self.duration_at))?;
        self.out.write_all(&duration_ms.to_be_bytes())?;

        self.out.seek(SeekFrom::Start(self.segment_start - 8))?;
        self.out.write_all(&ebml::vint8(end - self.segment_start))?;
        self.out.seek(SeekFrom::Start(end))?;
        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use nc_common::ColourInfo;
    use nc_encoder::FrameRate;

    use super::super::TimeBase;
    use super::*;

    fn vp9_track() -> Track {
        Track {
            codec: Codec::Vp9,
            width: 64,
            height: 48,
            frame_rate: FrameRate { num: 30, den: 1 },
            colour: ColourInfo::default(),
            time_base: TimeBase::frames(FrameRate { num: 30, den: 1 }),
        }
    }

    #[test]
    fn held_frames_keep_their_time_and_the_duration_covers_the_input() {
        let mut writer = MkvWriter::new(Cursor::new(Vec::new()), vp9_track(), true);
        writer.write_frame(&[0x82, 0x49], 0, true).unwrap();
        // Frames 1 and 2 were held: frame 3 follows at 100 ms.
        writer.write_frame(&[0x86, 0x00], 3, false).unwrap();
        let duration_at = writer.duration_at as usize;
        let segment_start = writer.segment_start as usize;
        let bytes = writer.finish(30).unwrap().into_inner();
        // The second block (track 1, +100 ms, not key, its two bytes).
        assert!(
            bytes
                .windows(6)
                .any(|w| w == [0x81, 0x00, 100, 0x00, 0x86, 0x00])
        );
        let duration = f64::from_be_bytes(bytes[duration_at..duration_at + 8].try_into().unwrap());
        assert!((duration - 1000.).abs() < 1e-9);
        let size = u64::from_be_bytes(bytes[segment_start - 8..segment_start].try_into().unwrap());
        assert_eq!(
            size & 0x00FF_FFFF_FFFF_FFFF,
            (bytes.len() - segment_start) as u64
        );
    }
}
