//! The error type for picture construction and raw YUV I/O.

use std::fmt;
use std::io;

/// Errors from `nc-common` operations.
#[derive(Debug)]
pub enum Error {
    /// Picture or plane dimensions that cannot be represented.
    InvalidDimensions {
        /// Requested width in samples.
        width: usize,
        /// Requested height in samples.
        height: usize,
        /// Why the dimensions were rejected.
        reason: &'static str,
    },
    /// The input ended part-way through a frame.
    TruncatedFrame {
        /// Bytes one full frame needs.
        expected: usize,
        /// Bytes that were available.
        got: usize,
    },
    /// An underlying I/O error.
    Io(io::Error),
}

/// Result alias for `nc-common` operations.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions {
                width,
                height,
                reason,
            } => write!(f, "invalid dimensions {width}x{height}: {reason}"),
            Error::TruncatedFrame { expected, got } => {
                write!(f, "truncated frame: expected {expected} bytes, got {got}")
            }
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
