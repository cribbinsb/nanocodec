//! OBU packaging of one coded AV2 picture: temporal delimiter, sequence header
//! (key pictures) and the frame OBU carrying header plus entropy-coded tile.
use crate::mv_refs::ListConfig;
use crate::{frame_header, obu, sequence};
use nc_common::decode::Result;
use nc_common::{BitWriter, ColourInfo};
use nc_encoder::FrameRate;

/// OBU type of the temporal delimiter.
const TEMPORAL_DELIMITER: u8 = 2;
/// OBU type of the sequence header.
const SEQUENCE_HEADER: u8 = 1;
/// OBU type of a regular (non-key) frame.
const REGULAR_FRAME: u8 = 7;
/// `OBU_CONTENT_INTERPRETATION`: how to show the pictures (AVM v1.0.0
/// `avm_codec.h`).
const CONTENT_INTERPRETATION: u8 = 24;
/// `ci_scan_type_idc` progressive frames (AVM `AVM_SCAN_TYPE_PROGRESSIVE`).
const SCAN_TYPE_PROGRESSIVE: u32 = 1;
/// `color_description_idc` 0: the three code points follow (AVM
/// `AVM_COLOR_DESC_IDC_EXPLICIT`), as a Rice-Golomb code with k = 2.
const COLOR_DESCRIPTION_EXPLICIT_BITS: [bool; 3] = [false, false, false];
/// Unspecified code points (ISO/IEC 23091-2), for a range-only signal.
const UNSPECIFIED_CODE_POINT: u32 = 2;
/// `OBU_METADATA_SHORT`: one metadata unit (AVM v1.0.0 `avm_codec.h`).
const METADATA_SHORT: u8 = 8;
/// `metadata_type` `OBU_METADATA_TYPE_USER_DATA_UNREGISTERED`: a 16-byte
/// UUID then free bytes (AVM v1.0.0 `avm_codec.h`,
/// `read_metadata_user_data_unregistered` in `av2/decoder/obu.c`).
const METADATA_TYPE_USER_DATA_UNREGISTERED: u64 = 10;
/// The short metadata header byte (`metadata_is_suffix` 0, `muh_layer_idc`
/// 0 = unspecified, `muh_cancel_flag` 0, `muh_persistence_idc` 0 = global:
/// the message describes the whole sequence). Read as AV1 syntax, as dav2d
/// still does, the byte is metadata type 0, which it skips.
const METADATA_SHORT_HEADER: u8 = 0;
/// `trailing_bits()` after a byte-aligned payload.
const TRAILING_ONE_BYTE: u8 = 0x80;
/// Order-hint bits signalled in the sequence header.
/// `order_hint_bits_minus_1 + 1`: references within half the hint range
/// (127 frames) reach the engine's 64-frame second reference.
pub const ORDER_HINT_BITS: u8 = 8;
/// With a long-held second reference (`nc_encoder::holds_second_reference`):
/// 10 bits, so references up to `nc_encoder::MAX_REFERENCE_AGE` back keep
/// their display order (`get_relative_dist` stays within half the range).
pub const LONG_ORDER_HINT_BITS: u8 = 10;

/// Coded frame sizes are whole 8x8 blocks: the subset codes nothing smaller,
/// and dav2d's deblocking assumes it (`db_apply_tmpl.c`).
const FRAME_SIZE_MULTIPLE: usize = 8;

/// The frame size coded for a `width` x `height` picture: the next whole
/// 8x8 blocks; the conformance window crops the padding off again.
#[must_use]
pub fn coded_size(width: usize, height: usize) -> (usize, usize) {
    (
        width.next_multiple_of(FRAME_SIZE_MULTIPLE),
        height.next_multiple_of(FRAME_SIZE_MULTIPLE),
    )
}

/// The video sequence header for visible `width` x `height` pictures with
/// `layering` and motion-vector reference lists `reference_lists`. Frames
/// may code a chroma quantizer delta (one bit per frame when zero), hide
/// luma DC parities when `parity_hiding`, and signal TCQ per frame when
/// `tcq`; order hints have `order_hint_bits`.
#[must_use]
pub fn sequence_header(
    (width, height): (usize, usize),
    layering: sequence::Layering,
    reference_lists: ListConfig,
    (parity_hiding, tcq, ccso): (bool, bool, bool),
    order_hint_bits: u8,
) -> sequence::SequenceHeader {
    let (coded_width, coded_height) = coded_size(width, height);
    sequence::SequenceHeader {
        id: 0,
        max_width: coded_width as u32,
        max_height: coded_height as u32,
        // The backend sets the level of each key frame (`level.rs`).
        level: crate::level::UNCONSTRAINED,
        single_picture: false,
        order_hint_bits,
        crop: sequence::CropWindow {
            right: (coded_width - width) as u32,
            bottom: (coded_height - height) as u32,
        },
        layering,
        chroma_delta_q: true,
        dc_delta_q: true,
        parity_hiding,
        tcq: if tcq {
            sequence::TcqFrames::Signalled
        } else {
            sequence::TcqFrames::Off
        },
        reference_lists,
        ccso,
        variants: sequence::Variants::OURS,
    }
}

fn base_header(obu_type: u8) -> obu::Header {
    obu::Header {
        obu_type,
        temporal_id: 0,
        embedded_id: 0,
        extended_id: 0,
    }
}

fn delimiter(packet: &mut Vec<u8>) -> Result<()> {
    obu::write(
        obu::Header {
            extended_id: obu::GLOBAL_XLAYER_ID,
            ..base_header(TEMPORAL_DELIMITER)
        },
        &[],
        packet,
    )
}

/// The payload of a content interpretation OBU (AVM v1.0.0
/// `av2_write_content_interpretation_obu`): progressive frames, the colour
/// description and range when signalled, and the timing (one display tick
/// per nominal frame; a constant picture interval when `constant`).
#[must_use]
pub fn content_interpretation(colour: ColourInfo, rate: FrameRate, constant: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(SCAN_TYPE_PROGRESSIVE, 2); // ci_scan_type_idc
    w.put_bit(colour.is_signalled()); // ci_color_description_present_flag
    w.put_bit(false); // ci_chroma_sample_position_present_flag
    w.put_bit(false); // ci_aspect_ratio_info_present_flag
    w.put_bit(true); // ci_timing_info_present_flag
    w.put_bits(0, 2); // reserved
    if colour.is_signalled() {
        for bit in COLOR_DESCRIPTION_EXPLICIT_BITS {
            w.put_bit(bit); // color_description_idc 0
        }
        let (primaries, transfer, matrix) = colour.description.map_or(
            (
                UNSPECIFIED_CODE_POINT,
                UNSPECIFIED_CODE_POINT,
                UNSPECIFIED_CODE_POINT,
            ),
            |d| {
                (
                    u32::from(d.primaries),
                    u32::from(d.transfer),
                    u32::from(d.matrix),
                )
            },
        );
        w.put_bits(primaries, 8);
        w.put_bits(transfer, 8);
        w.put_bits(matrix, 8);
        w.put_bit(colour.full_range); // full_range_flag
    }
    w.put_bits(rate.den, 32); // num_units_in_display_tick
    w.put_bits(rate.num, 32); // time_scale
    w.put_bit(constant); // equal_elemental_interval
    if constant {
        w.put_uvlc(0); // num_ticks_per_elemental_duration_minus_1
    }
    w.put_bit(false); // ci_extension_present_flag
    w.trailing_bits();
    w.into_bytes()
}

/// A closed-loop key picture of `sequence` (base temporal layer), preceded
/// by the sequence header, the content interpretation OBU `content` (the
/// first frame-unit OBU, AVM `check_temporal_unit_structure`) and, if
/// given, the encoder information message
/// (`nc_encoder` `encoder_info.rs`: UUID then text) as a short
/// user-data-unregistered metadata OBU, which in a temporal unit follows
/// the sequence header as frame-unit data (AVM
/// `check_temporal_unit_structure`).
/// # Errors
/// Propagates header and framing errors.
pub fn key(
    sequence: &sequence::SequenceHeader,
    header: frame_header::KeyframeHeader,
    tile: &[u8],
    (content, encoder_info): (&[u8], Option<&[u8]>),
) -> Result<Vec<u8>> {
    let mut payload = frame_header::write_keyframe(sequence, header)?;
    payload.extend_from_slice(tile);
    let mut packet = Vec::new();
    delimiter(&mut packet)?;
    obu::write(
        base_header(SEQUENCE_HEADER),
        &sequence::write(sequence)?,
        &mut packet,
    )?;
    obu::write(base_header(CONTENT_INTERPRETATION), content, &mut packet)?;
    if let Some(info) = encoder_info {
        let mut metadata = vec![METADATA_SHORT_HEADER];
        metadata.extend(nc_common::leb128(METADATA_TYPE_USER_DATA_UNREGISTERED));
        metadata.extend_from_slice(info);
        metadata.push(TRAILING_ONE_BYTE);
        obu::write(base_header(METADATA_SHORT), &metadata, &mut packet)?;
    }
    obu::write(
        base_header(frame_header::CLOSED_LOOP_KEY),
        &payload,
        &mut packet,
    )?;
    Ok(packet)
}

/// A forward-predicted picture of `sequence` in temporal layer
/// `temporal_layer` (the tile-group OBU's `temporal_id`).
/// # Errors
/// Propagates header and framing errors.
pub fn inter(
    sequence: &sequence::SequenceHeader,
    temporal_layer: u8,
    header: frame_header::InterHeader,
    tile: &[u8],
) -> Result<Vec<u8>> {
    let mut payload = frame_header::write_inter(sequence, header)?;
    payload.extend_from_slice(tile);
    let mut packet = Vec::new();
    delimiter(&mut packet)?;
    let frame = obu::Header {
        temporal_id: temporal_layer,
        ..base_header(REGULAR_FRAME)
    };
    obu::write(frame, &payload, &mut packet)?;
    Ok(packet)
}
