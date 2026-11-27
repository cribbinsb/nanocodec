//! Container output: MP4 (ISO BMFF) and Matroska / `WebM` files around the
//! encoder's frames, with each coded frame's real presentation time, so a
//! player shows the right frame rate and duration and can seek to key
//! frames (`--output` ending `.mp4`, `.m4v`, `.mkv` or `.webm`).
//!
//! Raw Annex-B carries no timestamps: an input frame the encoder holds
//! (`nc_encoder` `hold.rs`) leaves no trace in it, so a raw stream of a
//! repeat-frame source plays fast once remuxed at its nominal rate. A
//! container keeps every coded frame's time, the held time included.
//!
//! [`codec_config`] turns the encoder's units into container samples and
//! builds each codec's configuration record; [`mp4`] and [`mkv`] write the
//! files. Both need a seekable file: their headers are completed (sizes,
//! duration, index) when the stream ends.

mod codec_config;
mod ebml;
mod mkv;
mod mp4;

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use nc_common::ColourInfo;
use nc_encoder::{Codec, FrameRate};

/// A container format the CLI writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Container {
    /// ISO BMFF (`.mp4`, `.m4v`).
    Mp4,
    /// Matroska (`.mkv`).
    Matroska,
    /// `WebM` (`.webm`): Matroska restricted to VP9 and AV1.
    WebM,
}

impl Container {
    /// The container `path`'s extension names, if any; an error when it
    /// cannot carry `codec`.
    pub(crate) fn for_path(path: &Path, codec: Codec) -> Result<Option<Self>, String> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let container = match extension.as_deref() {
            Some("mp4" | "m4v") => Self::Mp4,
            Some("mkv") => Self::Matroska,
            Some("webm") => Self::WebM,
            _ => return Ok(None),
        };
        match (container, codec) {
            (_, Codec::Av2) => Err("AV2 has no MP4 or Matroska mapping yet: write IVF".into()),
            (Self::WebM, Codec::H264 | Codec::Hevc | Codec::Vvc) => {
                Err("WebM carries VP9 and AV1 only: use .mkv or .mp4".into())
            }
            // `V_MPEGI/ISO/VVC` is not yet read by FFmpeg 8 (its Matroska
            // demuxer reports an unknown codec): MP4 only.
            (Self::Matroska, Codec::Vvc) => {
                Err("VVC in Matroska is not yet readable: use .mp4".into())
            }
            _ => Ok(Some(container)),
        }
    }
}

/// The units presentation times count: `units_per_second` per second, each
/// time `unit` of them (frame `n` of a nominal 30000/1001 stream is at
/// `n` x 1001 of 30000 per second).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TimeBase {
    /// Ticks per second.
    pub units_per_second: u32,
    /// Ticks per time value.
    pub unit: u32,
}

impl TimeBase {
    /// One time value per nominal frame of `rate`.
    pub(crate) fn frames(rate: FrameRate) -> Self {
        Self {
            units_per_second: rate.num,
            unit: rate.den,
        }
    }

    /// Microsecond time values.
    pub(crate) const MICROSECONDS: Self = Self {
        units_per_second: 1_000_000,
        unit: 1,
    };

    /// `pts` in ticks.
    fn ticks(self, pts: u64) -> u64 {
        pts * u64::from(self.unit)
    }

    /// `pts` in milliseconds, rounded.
    fn millis(self, pts: u64) -> u64 {
        (u128::from(self.ticks(pts)) * 1000 + u128::from(self.units_per_second) / 2)
            .checked_div(u128::from(self.units_per_second))
            .and_then(|ms| u64::try_from(ms).ok())
            .unwrap_or(u64::MAX)
    }
}

/// What the container's track header says about the video.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Track {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    /// The nominal frame rate (Matroska `DefaultDuration`).
    pub frame_rate: FrameRate,
    pub colour: ColourInfo,
    pub time_base: TimeBase,
}

/// A container file being written.
pub(crate) enum Writer {
    Mp4(mp4::Mp4Writer<BufWriter<File>>),
    Matroska(mkv::MkvWriter<BufWriter<File>>),
}

impl Writer {
    /// Create `path` for `container` with `track`.
    pub(crate) fn create(path: &Path, container: Container, track: Track) -> Result<Self, String> {
        if crate::stdio::is_stdio(path) {
            return Err(
                "MP4 and Matroska output need a file (headers are completed at the end)".into(),
            );
        }
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let out = BufWriter::new(file);
        Ok(match container {
            Container::Mp4 => Self::Mp4(mp4::Mp4Writer::new(out, track)),
            Container::Matroska => Self::Matroska(mkv::MkvWriter::new(out, track, false)),
            Container::WebM => Self::Matroska(mkv::MkvWriter::new(out, track, true)),
        })
    }

    /// Append the encoder's unit `data`, presented at `pts`.
    pub(crate) fn write_frame(&mut self, data: &[u8], pts: u64, key: bool) -> Result<(), String> {
        match self {
            Self::Mp4(w) => w.write_frame(data, pts, key),
            Self::Matroska(w) => w.write_frame(data, pts, key),
        }
        .map_err(|e| e.to_string())
    }

    /// Complete the file; the last frame shows until `end_pts`.
    pub(crate) fn finish(self, end_pts: u64) -> Result<(), String> {
        match self {
            Self::Mp4(w) => w.finish(end_pts).map(drop),
            Self::Matroska(w) => w.finish(end_pts).map(drop),
        }
        .map_err(|e| e.to_string())
    }
}

/// An I/O error carrying a container-level reason.
fn invalid(reason: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, reason.into())
}
