//! The encoder's units as container samples, and each codec's decoder
//! configuration record, built from the first key frame's headers:
//!
//! - H.264 `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §5.3.2.1),
//!   HEVC `HEVCDecoderConfigurationRecord` (§8.3.2.1), VVC
//!   `VvcDecoderConfigurationRecord` (§11.2.4.2): samples are the access
//!   unit's NAL units with 4-byte lengths in place of start codes; the
//!   parameter sets stay in-band too (sample entries `avc3`, `hev1`,
//!   `vvi1`), since every key frame repeats them and a resolution change
//!   brings new ones.
//! - AV1 `AV1CodecConfigurationRecord` (AV1-ISOBMFF §2.3): samples are
//!   temporal units without temporal delimiter OBUs (§2.4; Matroska's AV1
//!   mapping likewise).
//! - VP9 `VPCodecConfigurationRecord` (VP-codec-ISOBMFF §2.2), from the
//!   stream settings: VP9 frames carry no parameter sets.

use std::io;

use nc_common::decode::{BitReader, annex_b_units, unescape_rbsp};
use nc_encoder::Codec;

use super::{Track, invalid};

/// Bytes of each NAL unit's length field (`lengthSizeMinusOne` + 1).
const NAL_LENGTH_BYTES: usize = 4;
/// `lengthSizeMinusOne` of every record.
const LENGTH_SIZE_MINUS_ONE: u8 = (NAL_LENGTH_BYTES - 1) as u8;

/// H.264 NAL unit types (Table 7-1).
const H264_SPS: u8 = 7;
const H264_PPS: u8 = 8;
/// HEVC NAL unit types (Table 7-1).
const HEVC_VPS: u8 = 32;
const HEVC_SPS: u8 = 33;
const HEVC_PPS: u8 = 34;
/// VVC NAL unit types (Table 5).
const VVC_SPS: u8 = 15;
const VVC_PPS: u8 = 16;
/// AV1 OBU types (§6.2.2).
const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;

/// A codec's sample conversion and configuration record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodecConfig {
    /// The configuration record (MP4 box payload; Matroska `CodecPrivate`,
    /// for VVC without the MP4 box's version and flags).
    pub record: Vec<u8>,
}

impl CodecConfig {
    /// The record of `track`'s stream, whose first key frame is `unit`.
    pub(crate) fn from_key(track: &Track, unit: &[u8]) -> io::Result<Self> {
        let record = match track.codec {
            Codec::H264 => avc_record(unit)?,
            Codec::Hevc => hevc_record(unit)?,
            Codec::Vvc => vvc_record(unit, track)?,
            Codec::Av1 => av1_record(unit)?,
            Codec::Vp9 => vp9_record(track),
            Codec::Av2 => return Err(invalid("AV2 has no container mapping")),
        };
        Ok(Self { record })
    }
}

/// `unit` as a container sample of `codec`.
pub(crate) fn sample(codec: Codec, unit: &[u8]) -> io::Result<Vec<u8>> {
    match codec {
        Codec::H264 | Codec::Hevc | Codec::Vvc => {
            let mut sample = Vec::with_capacity(unit.len());
            for nal in nal_units(unit)? {
                let length = u32::try_from(nal.len()).map_err(|_| invalid("NAL unit too long"))?;
                sample.extend_from_slice(&length.to_be_bytes());
                sample.extend_from_slice(nal);
            }
            Ok(sample)
        }
        Codec::Av1 => {
            let mut sample = Vec::with_capacity(unit.len());
            for obu in obus(unit)? {
                if obu.kind != OBU_TEMPORAL_DELIMITER {
                    sample.extend_from_slice(obu.bytes);
                }
            }
            Ok(sample)
        }
        Codec::Vp9 => Ok(unit.to_vec()),
        Codec::Av2 => Err(invalid("AV2 has no container mapping")),
    }
}

/// The NAL units of an Annex-B access unit (headers included).
fn nal_units(unit: &[u8]) -> io::Result<Vec<&[u8]>> {
    annex_b_units(unit).map_err(|e| invalid(e.to_string()))
}

/// The NAL units of `unit` whose type `kind_of` reads as `kind`.
fn units_of<'a>(units: &[&'a [u8]], kind_of: fn(&[u8]) -> u8, kind: u8) -> Vec<&'a [u8]> {
    units
        .iter()
        .copied()
        .filter(|nal| kind_of(nal) == kind)
        .collect()
}

fn h264_kind(nal: &[u8]) -> u8 {
    nal[0] & 0x1F
}

fn hevc_kind(nal: &[u8]) -> u8 {
    (nal[0] >> 1) & 0x3F
}

fn vvc_kind(nal: &[u8]) -> u8 {
    nal.get(1).map_or(0, |b| b >> 3)
}

/// One `[length][NAL unit]` entry of a record's array.
fn put_nal(out: &mut Vec<u8>, nal: &[u8]) -> io::Result<()> {
    let length = u16::try_from(nal.len()).map_err(|_| invalid("parameter set too long"))?;
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(nal);
    Ok(())
}

/// The first unit of `kind`, or an error naming `what`.
fn first<'a>(units: &[&'a [u8]], what: &str) -> io::Result<&'a [u8]> {
    units
        .first()
        .copied()
        .ok_or_else(|| invalid(format!("first key frame without {what}")))
}

/// `AVCDecoderConfigurationRecord` (14496-15 §5.3.2.1.2), High profile
/// fields included (`chroma_format` 4:2:0, 8-bit).
fn avc_record(unit: &[u8]) -> io::Result<Vec<u8>> {
    let units = nal_units(unit)?;
    let sps = units_of(&units, h264_kind, H264_SPS);
    let pps = units_of(&units, h264_kind, H264_PPS);
    let first_sps = first(&sps, "an SPS")?;
    let rbsp = unescape_rbsp(&first_sps[1..]).map_err(|e| invalid(e.to_string()))?;
    let [profile, compatibility, level, ..] = rbsp[..] else {
        return Err(invalid("short SPS"));
    };
    let mut out = vec![
        1,
        profile,
        compatibility,
        level,
        0xFC | LENGTH_SIZE_MINUS_ONE,
    ];
    out.push(0xE0 | u8::try_from(sps.len()).map_err(|_| invalid("too many SPS"))?);
    for nal in &sps {
        put_nal(&mut out, nal)?;
    }
    out.push(u8::try_from(pps.len()).map_err(|_| invalid("too many PPS"))?);
    for nal in &pps {
        put_nal(&mut out, nal)?;
    }
    // High profile: chroma_format 1, bit depths 8, no SPS extensions.
    out.extend_from_slice(&[0xFC | 1, 0xF8, 0xF8, 0]);
    Ok(out)
}

/// Bytes of HEVC's `general_profile_tier_level` fields, after the SPS's
/// first byte (`sps_video_parameter_set_id`, `sps_max_sub_layers_minus1`,
/// `sps_temporal_id_nesting_flag`): profile space, tier and idc (1),
/// compatibility flags (4), constraint flags (6), level (1).
const HEVC_GENERAL_PTL_BYTES: usize = 12;

/// `HEVCDecoderConfigurationRecord` (14496-15 §8.3.2.1.2) with the VPS,
/// SPS and PPS arrays.
fn hevc_record(unit: &[u8]) -> io::Result<Vec<u8>> {
    let units = nal_units(unit)?;
    let vps = units_of(&units, hevc_kind, HEVC_VPS);
    let sps = units_of(&units, hevc_kind, HEVC_SPS);
    let pps = units_of(&units, hevc_kind, HEVC_PPS);
    let rbsp = unescape_rbsp(&first(&sps, "an SPS")?[2..]).map_err(|e| invalid(e.to_string()))?;
    let ptl = rbsp
        .get(1..=HEVC_GENERAL_PTL_BYTES)
        .ok_or_else(|| invalid("short SPS"))?;
    let sub_layers = ((rbsp[0] >> 1) & 0x7) + 1;
    let nested = rbsp[0] & 1;
    let mut out = vec![1];
    out.extend_from_slice(ptl);
    out.extend_from_slice(&[
        0xF0,
        0x00,     // min_spatial_segmentation_idc 0
        0xFC,     // parallelismType 0
        0xFC | 1, // chromaFormat 4:2:0
        0xF8,     // bitDepthLumaMinus8 0
        0xF8,     // bitDepthChromaMinus8 0
        0,
        0, // avgFrameRate unspecified
        (sub_layers << 3) | (nested << 2) | LENGTH_SIZE_MINUS_ONE,
        3, // numOfArrays
    ]);
    for (kind, nals) in [(HEVC_VPS, vps), (HEVC_SPS, sps), (HEVC_PPS, pps)] {
        // array_completeness 0: `hev1`, more may come in-band.
        out.push(kind);
        out.extend_from_slice(&u16::try_from(nals.len()).unwrap_or(u16::MAX).to_be_bytes());
        for nal in &nals {
            put_nal(&mut out, nal)?;
        }
    }
    Ok(out)
}

/// `VvcDecoderConfigurationRecord` (14496-15 §11.2.4.2.2) with its
/// `VvcPTLRecord`, read from our SPS (profile-tier-level present, no
/// general constraint info, no sub-profiles: `nc_vvc` `headers.rs`).
fn vvc_record(unit: &[u8], track: &Track) -> io::Result<Vec<u8>> {
    let units = nal_units(unit)?;
    let sps = units_of(&units, vvc_kind, VVC_SPS);
    let pps = units_of(&units, vvc_kind, VVC_PPS);
    let rbsp = unescape_rbsp(&first(&sps, "an SPS")?[2..]).map_err(|e| invalid(e.to_string()))?;
    let bad = |e: nc_common::decode::DecodeError| invalid(e.to_string());
    let mut r = BitReader::new(&rbsp);
    r.skip_bits(8).map_err(bad)?; // SPS and VPS ids
    let max_sub_layers_minus1 = r.read_bits(3).map_err(bad)?;
    let chroma_format = r.read_bits(2).map_err(bad)?;
    r.skip_bits(2).map_err(bad)?; // CTU size
    if !r.read_bit().map_err(bad)? {
        return Err(invalid("VVC SPS without profile, tier and level"));
    }
    let profile_tier = r.read_bits(8).map_err(bad)?;
    let level = r.read_bits(8).map_err(bad)?;
    let frame_only = r.read_bit().map_err(bad)?;
    let multilayer = r.read_bit().map_err(bad)?;
    if r.read_bit().map_err(bad)? {
        return Err(invalid("VVC general constraint info not supported"));
    }
    let num_sublayers = max_sub_layers_minus1 + 1;
    let mut out = vec![0xFF]; // reserved, lengthSizeMinusOne 3, ptl_present_flag
    // ols_idx 0, num_sublayers, constant_frame_rate 0, chroma_format_idc.
    let packed = (num_sublayers << 4) | chroma_format;
    out.extend_from_slice(&u16::try_from(packed).unwrap_or(0).to_be_bytes());
    out.push(0x1F); // bit_depth_minus8 0, reserved
    // VvcPTLRecord: one byte of general constraint info (the absent flag).
    out.push(1); // reserved 0, num_bytes_constraint_info 1
    out.push(u8::try_from(profile_tier).unwrap_or(0));
    out.push(u8::try_from(level).unwrap_or(0));
    out.push((u8::from(frame_only) << 7) | (u8::from(multilayer) << 6)); // gci_present_flag 0
    if num_sublayers > 1 {
        out.push(0); // no sublayer levels, reserved bits
    }
    out.push(0); // ptl_num_sub_profiles
    let dims = |v: u32| u16::try_from(v).unwrap_or(u16::MAX).to_be_bytes();
    out.extend_from_slice(&dims(track.width));
    out.extend_from_slice(&dims(track.height));
    out.extend_from_slice(&[0, 0]); // avg_frame_rate unspecified
    out.push(2); // num_of_arrays
    for (kind, nals) in [(VVC_SPS, sps), (VVC_PPS, pps)] {
        out.push(kind); // array_completeness 0: `vvi1`
        out.extend_from_slice(&u16::try_from(nals.len()).unwrap_or(u16::MAX).to_be_bytes());
        for nal in &nals {
            put_nal(&mut out, nal)?;
        }
    }
    Ok(out)
}

/// One OBU of a temporal unit.
struct Obu<'a> {
    kind: u8,
    /// The whole OBU, header included.
    bytes: &'a [u8],
    /// Its payload.
    payload: &'a [u8],
}

/// The OBUs of `unit` (each with `obu_has_size_field`, as ours are).
fn obus(unit: &[u8]) -> io::Result<Vec<Obu<'_>>> {
    let mut obus = Vec::new();
    let mut at = 0;
    while at < unit.len() {
        let header = unit[at];
        let kind = (header >> 3) & 0xF;
        let extension = header & 0x4 != 0;
        if header & 0x2 == 0 {
            return Err(invalid("AV1 OBU without a size field"));
        }
        let mut r = BitReader::new(&unit[at + 1 + usize::from(extension)..]);
        let size = r.read_leb128().map_err(|e| invalid(e.to_string()))?;
        let size_bytes = (unit.len() - at - 1 - usize::from(extension)) - r.remaining_bits() / 8;
        let start = at + 1 + usize::from(extension) + size_bytes;
        let end = usize::try_from(size)
            .ok()
            .and_then(|s| start.checked_add(s))
            .filter(|&end| end <= unit.len())
            .ok_or_else(|| invalid("AV1 OBU past the unit"))?;
        obus.push(Obu {
            kind,
            bytes: &unit[at..end],
            payload: &unit[start..end],
        });
        at = end;
    }
    Ok(obus)
}

/// `AV1CodecConfigurationRecord` (AV1-ISOBMFF §2.3.3): profile, the first
/// operating point's level and tier, 8-bit 4:2:0, and the sequence header
/// OBU as `configOBUs`.
fn av1_record(unit: &[u8]) -> io::Result<Vec<u8>> {
    let obus = obus(unit)?;
    let header = obus
        .iter()
        .find(|o| o.kind == OBU_SEQUENCE_HEADER)
        .ok_or_else(|| invalid("first key frame without a sequence header"))?;
    let bad = |e: nc_common::decode::DecodeError| invalid(e.to_string());
    let mut r = BitReader::new(header.payload);
    let profile = r.read_bits(3).map_err(bad)?;
    r.skip_bits(1).map_err(bad)?; // still_picture
    if r.read_bit().map_err(bad)? {
        return Err(invalid("reduced still-picture header"));
    }
    if r.read_bit().map_err(bad)? {
        // timing_info(): two 32-bit fields, then the picture interval.
        r.skip_bits(64).map_err(bad)?;
        if r.read_bit().map_err(bad)? {
            r.read_uvlc().map_err(bad)?;
        }
        if r.read_bit().map_err(bad)? {
            return Err(invalid("AV1 decoder model info not supported"));
        }
    }
    if r.read_bit().map_err(bad)? {
        return Err(invalid("AV1 initial display delay not supported"));
    }
    r.skip_bits(5 + 12).map_err(bad)?; // operating_points_cnt_minus_1, idc[0]
    let level = r.read_bits(5).map_err(bad)?;
    let tier = if level > 7 {
        r.read_bits(1).map_err(bad)?
    } else {
        0
    };
    let profile_level = u8::try_from((profile << 5) | level).map_err(|_| invalid("AV1 level"))?;
    // high_bitdepth, twelve_bit, monochrome 0; subsampling x and y 1;
    // chroma_sample_position 0.
    let colour = (u8::try_from(tier).unwrap_or(0) << 7) | 0b0000_1100;
    let mut out = vec![0x81, profile_level, colour, 0];
    out.extend_from_slice(header.bytes);
    Ok(out)
}

/// VP9 levels (VP9 profile and level definitions): (level, `MaxLumaSampleRate`,
/// `MaxLumaPictureSize`).
const VP9_LEVELS: [(u8, u64, u64); 14] = [
    (10, 829_440, 36_864),
    (11, 2_764_800, 73_728),
    (20, 4_608_000, 122_880),
    (21, 9_216_000, 245_760),
    (30, 20_736_000, 552_960),
    (31, 36_864_000, 983_040),
    (40, 83_558_400, 2_228_224),
    (41, 160_432_128, 2_228_224),
    (50, 311_951_360, 8_912_896),
    (51, 588_251_136, 8_912_896),
    (52, 1_176_502_272, 8_912_896),
    (60, 1_176_502_272, 35_651_584),
    (61, 2_353_004_544, 35_651_584),
    (62, 4_706_009_088, 35_651_584),
];

/// `VPCodecConfigurationRecord` (VP-codec-ISOBMFF §2.2), the payload of the
/// version-1 `vpcC` box: profile 0, the level of the size and rate, 8-bit
/// 4:2:0 (colocated with luma: `chromaSubsampling` 1), the colour the
/// stream signals (VP9 itself says BT.601 unless told otherwise).
fn vp9_record(track: &Track) -> Vec<u8> {
    let picture = u64::from(track.width) * u64::from(track.height);
    let rate = (picture * u64::from(track.frame_rate.num))
        .div_ceil(u64::from(track.frame_rate.den.max(1)));
    let level = VP9_LEVELS
        .iter()
        .find(|&&(_, max_rate, max_size)| rate <= max_rate && picture <= max_size)
        .map_or(62, |&(level, ..)| level);
    // Unspecified primaries and transfer; BT.601 matrix (VP9 CS_BT_601).
    let (primaries, transfer, matrix) = track
        .colour
        .description
        .map_or((2, 2, 6), |d| (d.primaries, d.transfer, d.matrix));
    let bit_depth_subsampling_range = (8 << 4) | (1 << 1) | u8::from(track.colour.full_range);
    vec![
        0, // profile
        level,
        bit_depth_subsampling_range,
        primaries,
        transfer,
        matrix,
        0,
        0, // codecInitializationDataSize
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_units_become_length_prefixed() {
        let unit = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x65, 9];
        assert_eq!(
            sample(Codec::H264, &unit).unwrap(),
            [0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 9]
        );
    }

    #[test]
    fn av1_samples_drop_temporal_delimiters() {
        // Temporal delimiter (size 0), then a padding OBU of two bytes.
        let unit = [0x12, 0, 0x7A, 2, 0xAA, 0xBB];
        assert_eq!(sample(Codec::Av1, &unit).unwrap(), [0x7A, 2, 0xAA, 0xBB]);
    }

    #[test]
    fn avc_record_lists_the_parameter_sets() {
        let unit = [
            0, 0, 0, 1, 0x67, 100, 0, 13, 0xAC, 0, 0, 0, 1, 0x68, 0xEE, 0, 0, 0, 1, 0x65, 1,
        ];
        let record = avc_record(&unit).unwrap();
        assert_eq!(record[..6], [1, 100, 0, 13, 0xFF, 0xE1]);
        assert_eq!(record[6..8], [0, 5]);
        assert_eq!(record[13..17], [1, 0, 2, 0x68]);
    }
}
