//! Colour signalling: what the samples mean, for displays and players.
//!
//! The code points are ISO/IEC 23091-2 (CICP), the one table shared by
//! H.264 (Annex E, Tables E-3 to E-5), HEVC (Annex E, Tables E.3 to E.5)
//! and AV1 (§6.4.2, `color_primaries` / `transfer_characteristics` /
//! `matrix_coefficients`). Signalling never changes how a stream decodes;
//! it only tells the receiver how to convert decoded YUV to RGB.
//!
//! [`ColourInfo::default`] signals nothing, which keeps every stream
//! byte-identical to an encoder without colour support.

use crate::bitwriter::BitWriter;
use crate::decode::{BitReader, Result};

/// `video_format` 5: "unspecified video format" (H.264 Table E-2, HEVC
/// Table E.2); a camera stream is none of component/PAL/NTSC/SECAM/MAC.
const VIDEO_FORMAT_UNSPECIFIED: u32 = 5;
/// Bits of `video_format`.
const VIDEO_FORMAT_BITS: u32 = 3;
/// Bits of each colour description code point.
const CODE_POINT_BITS: u32 = 8;

/// Colour primaries code points (CICP §8.1) used by the presets.
pub mod primaries {
    /// ITU-R BT.709 (also sRGB).
    pub const BT709: u8 = 1;
    /// ITU-R BT.601 625-line (PAL, BT.470 B/G).
    pub const BT601_625: u8 = 5;
    /// ITU-R BT.601 525-line (NTSC, SMPTE 170M).
    pub const BT601_525: u8 = 6;
}

/// Transfer characteristics code points (CICP §8.2) used by the presets.
pub mod transfer {
    /// ITU-R BT.709.
    pub const BT709: u8 = 1;
    /// ITU-R BT.601 (SMPTE 170M; the same curve as BT.709).
    pub const BT601: u8 = 6;
}

/// Matrix coefficients code points (CICP §8.3) used by the presets.
pub mod matrix {
    /// Identity (GBR): only valid for 4:4:4, so never for our 4:2:0 output.
    pub const IDENTITY: u8 = 0;
    /// ITU-R BT.709.
    pub const BT709: u8 = 1;
    /// ITU-R BT.601 (BT.470 B/G; identical to SMPTE 170M, code point 6).
    pub const BT601: u8 = 5;
    /// ITU-R BT.601 525-line (SMPTE 170M).
    pub const BT601_525: u8 = 6;
}

/// The three colour description code points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourDescription {
    /// `colour_primaries`.
    pub primaries: u8,
    /// `transfer_characteristics`.
    pub transfer: u8,
    /// `matrix_coefficients`.
    pub matrix: u8,
}

impl ColourDescription {
    /// HD video: BT.709 primaries, transfer and matrix.
    pub const BT709: Self = Self {
        primaries: primaries::BT709,
        transfer: transfer::BT709,
        matrix: matrix::BT709,
    };
    /// SD video, 525-line (NTSC) primaries, BT.601 matrix.
    pub const BT601_525: Self = Self {
        primaries: primaries::BT601_525,
        transfer: transfer::BT601,
        matrix: matrix::BT601_525,
    };
    /// SD video, 625-line (PAL) primaries, BT.601 matrix.
    pub const BT601_625: Self = Self {
        primaries: primaries::BT601_625,
        transfer: transfer::BT601,
        matrix: matrix::BT601,
    };
}

/// Colour signalling of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColourInfo {
    /// Primaries, transfer and matrix; `None` leaves them unsignalled
    /// (players then usually assume BT.601 for SD and BT.709 for HD).
    pub description: Option<ColourDescription>,
    /// Full-range samples (0–255, as JPEG and most webcams in MJPEG/NV12
    /// "full" mode), instead of limited (16–235 luma, 16–240 chroma).
    pub full_range: bool,
}

impl ColourInfo {
    /// Whether anything differs from "unsignalled, limited range", i.e.
    /// whether headers carry a colour description at all.
    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.description.is_some() || self.full_range
    }

    /// Whether 4:2:0 output can carry this description (the identity
    /// matrix needs 4:4:4: H.264 §E.2.1, HEVC §E.3.1, AV1 §6.4.2).
    #[must_use]
    pub fn is_valid_for_420(&self) -> bool {
        self.description
            .is_none_or(|d| d.matrix != matrix::IDENTITY)
    }
}

/// The video signal type fields shared by H.264 `vui_parameters()`
/// (§E.1.1) and HEVC `vui_parameters()` (§E.2.1), from `video_format`
/// through `matrix_coefficients`; the caller writes
/// `video_signal_type_present_flag = 1` before it.
pub fn write_video_signal_type(w: &mut BitWriter, colour: ColourInfo) {
    w.put_bits(VIDEO_FORMAT_UNSPECIFIED, VIDEO_FORMAT_BITS); // video_format
    w.put_bit(colour.full_range); // video_full_range_flag
    w.put_bit(colour.description.is_some()); // colour_description_present_flag
    if let Some(d) = colour.description {
        write_code_points(w, d);
    }
}

/// Parse the fields [`write_video_signal_type`] writes (any `video_format`).
///
/// # Errors
///
/// `Truncated` if the data ends inside them.
pub fn read_video_signal_type(r: &mut BitReader<'_>) -> Result<ColourInfo> {
    r.read_bits(VIDEO_FORMAT_BITS)?; // video_format: informative only
    let full_range = r.read_bit()?;
    let description = if r.read_bit()? {
        Some(read_code_points(r)?)
    } else {
        None
    };
    Ok(ColourInfo {
        description,
        full_range,
    })
}

/// `colour_primaries`, `transfer_characteristics`, `matrix_coefficients`:
/// 8 bits each, in this order, in all three codecs (AV1 §5.5.2 too).
pub fn write_code_points(w: &mut BitWriter, d: ColourDescription) {
    for value in [d.primaries, d.transfer, d.matrix] {
        w.put_bits(u32::from(value), CODE_POINT_BITS);
    }
}

/// Parse what [`write_code_points`] writes.
///
/// # Errors
///
/// `Truncated` if the data ends inside them.
pub fn read_code_points(r: &mut BitReader<'_>) -> Result<ColourDescription> {
    let mut next = || r.read_bits(CODE_POINT_BITS).map(|v| v as u8);
    Ok(ColourDescription {
        primaries: next()?,
        transfer: next()?,
        matrix: next()?,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ColourDescription, ColourInfo, matrix, read_video_signal_type, write_video_signal_type,
    };
    use crate::bitwriter::BitWriter;
    use crate::decode::BitReader;

    #[test]
    fn default_is_unsignalled() {
        assert!(!ColourInfo::default().is_signalled());
        let range_only = ColourInfo {
            description: None,
            full_range: true,
        };
        assert!(range_only.is_signalled());
    }

    #[test]
    fn identity_matrix_is_rejected_for_420() {
        let identity = ColourInfo {
            description: Some(ColourDescription {
                matrix: matrix::IDENTITY,
                ..ColourDescription::BT709
            }),
            full_range: false,
        };
        assert!(!identity.is_valid_for_420());
        assert!(
            ColourInfo {
                description: Some(ColourDescription::BT709),
                full_range: true
            }
            .is_valid_for_420()
        );
    }

    #[test]
    fn video_signal_type_round_trips() {
        for colour in [
            ColourInfo {
                description: Some(ColourDescription::BT601_525),
                full_range: true,
            },
            ColourInfo {
                description: None,
                full_range: true,
            },
            ColourInfo {
                description: Some(ColourDescription::BT709),
                full_range: false,
            },
        ] {
            let mut w = BitWriter::new();
            write_video_signal_type(&mut w, colour);
            w.put_bit(true); // a following field
            w.trailing_bits();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_video_signal_type(&mut r).unwrap(), colour);
            assert!(r.read_bit().unwrap());
        }
    }
}
