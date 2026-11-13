//! OBU packaging of one coded AV2 picture: temporal delimiter, sequence header
//! (key pictures) and the frame OBU carrying header plus entropy-coded tile.
use crate::mv_refs::ListConfig;
use crate::{frame_header, obu, sequence};
use nc_common::decode::Result;

/// OBU type of the temporal delimiter.
const TEMPORAL_DELIMITER: u8 = 2;
/// OBU type of the sequence header.
const SEQUENCE_HEADER: u8 = 1;
/// OBU type of a regular (non-key) frame.
const REGULAR_FRAME: u8 = 7;
/// Order-hint bits signalled in the sequence header.
const ORDER_HINT_BITS: u8 = 8;
/// Level 31: no level-specific limits are claimed.
const UNCONSTRAINED_LEVEL: u8 = 31;

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
/// `tcq`.
#[must_use]
pub fn sequence_header(
    width: usize,
    height: usize,
    layering: sequence::Layering,
    reference_lists: ListConfig,
    (parity_hiding, tcq, ccso): (bool, bool, bool),
) -> sequence::SequenceHeader {
    let (coded_width, coded_height) = coded_size(width, height);
    sequence::SequenceHeader {
        id: 0,
        max_width: coded_width as u32,
        max_height: coded_height as u32,
        level: UNCONSTRAINED_LEVEL,
        single_picture: false,
        order_hint_bits: ORDER_HINT_BITS,
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

/// A closed-loop key picture of `sequence` (base temporal layer), preceded
/// by the sequence header.
/// # Errors
/// Propagates header and framing errors.
pub fn key(
    sequence: &sequence::SequenceHeader,
    header: frame_header::KeyframeHeader,
    tile: &[u8],
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
