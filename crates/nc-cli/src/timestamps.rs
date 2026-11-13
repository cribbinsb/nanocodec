//! Frame timestamp files in mkvmerge "timestamp format v2": a header line,
//! then one presentation time per frame in (possibly fractional)
//! milliseconds. mkvmerge and mp4fpsmod read the same files to mux
//! variable-frame-rate streams.
//!
//! `--timestamps FILE` gives each input frame its capture time;
//! `--timestamps-out FILE` writes the time of every coded frame.

use std::io::{self, BufRead, Write};

use nc_encoder::MICROS_PER_SECOND;

/// Header lines accepted (mkvmerge's current and historical spelling);
/// the first is written.
const HEADERS: [&str; 2] = ["# timestamp format v2", "# timecode format v2"];
/// Microseconds per millisecond.
const MICROS_PER_MILLI: i64 = MICROS_PER_SECOND / 1000;

/// Read a v2 timestamp file into microsecond timestamps.
pub fn read(input: impl BufRead) -> Result<Vec<i64>, String> {
    let mut lines = input.lines();
    let header = lines
        .next()
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    if !HEADERS
        .iter()
        .any(|h| header.trim().eq_ignore_ascii_case(h))
    {
        return Err(format!("timestamp file must start with '{}'", HEADERS[0]));
    }
    let mut timestamps = Vec::new();
    for line in lines {
        let line = line.map_err(|e| e.to_string())?;
        let text = line.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let millis: f64 = text
            .parse()
            .map_err(|_| format!("bad timestamp '{text}'"))?;
        if !millis.is_finite() {
            return Err(format!("bad timestamp '{text}'"));
        }
        timestamps.push((millis * MICROS_PER_MILLI as f64).round() as i64);
    }
    Ok(timestamps)
}

/// Writes a v2 timestamp file, one line per coded frame.
pub struct Writer<W: Write>(W);

impl<W: Write> Writer<W> {
    /// Start a file (writes the header).
    pub fn new(mut out: W) -> io::Result<Self> {
        writeln!(out, "{}", HEADERS[0])?;
        Ok(Self(out))
    }

    /// Append one frame's time in microseconds, as exact milliseconds.
    pub fn write(&mut self, timestamp_us: i64) -> io::Result<()> {
        let sign = if timestamp_us < 0 { "-" } else { "" };
        let us = timestamp_us.unsigned_abs();
        let milli = MICROS_PER_MILLI.unsigned_abs();
        writeln!(self.0, "{sign}{}.{:03}", us / milli, us % milli)
    }

    /// Flush the file.
    pub fn finish(mut self) -> io::Result<()> {
        self.0.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::{Writer, read};

    #[test]
    fn round_trips_through_the_v2_format() {
        let mut out = Writer::new(Vec::new()).unwrap();
        for us in [0, 33_367, 100_001, -5] {
            out.write(us).unwrap();
        }
        let bytes = out.0;
        assert_eq!(
            String::from_utf8(bytes.clone()).unwrap(),
            "# timestamp format v2\n0.000\n33.367\n100.001\n-0.005\n"
        );
        assert_eq!(read(&bytes[..]).unwrap(), [0, 33_367, 100_001, -5]);
    }

    #[test]
    fn rejects_missing_header_and_bad_lines() {
        assert!(read(&b"0\n33\n"[..]).is_err());
        assert!(read(&b"# timecode format v2\n0\nx\n"[..]).is_err());
        assert_eq!(
            read(&b"# timecode format v2\n# comment\n\n12.5\n"[..]).unwrap(),
            [12_500]
        );
    }
}
