//! The error type of every packetizer, depacketizer and capture reader.

use std::fmt;

/// Why a packet, unit or configuration was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtpError {
    /// The data ended before a required field.
    Truncated(&'static str),
    /// A field breaks the RTP or payload format syntax.
    Malformed(&'static str),
    /// Valid syntax this implementation does not carry (e.g. H.264
    /// interleaved mode, HEVC DONL).
    Unsupported(&'static str),
    /// A packetizer or depacketizer setting is out of range.
    Config(String),
    /// Reading or writing a capture file failed.
    Io(String),
}

impl fmt::Display for RtpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated(what) => write!(f, "truncated RTP data: {what}"),
            Self::Malformed(what) => write!(f, "malformed RTP data: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported RTP feature: {what}"),
            Self::Config(what) => write!(f, "invalid RTP configuration: {what}"),
            Self::Io(what) => write!(f, "capture I/O: {what}"),
        }
    }
}

impl std::error::Error for RtpError {}

impl From<std::io::Error> for RtpError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Result type of this crate.
pub type Result<T> = std::result::Result<T, RtpError>;
