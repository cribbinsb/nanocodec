//! Readers of the headers [`super`] writes: the round-trip oracle of its
//! tests. The decoder parses every frame header with
//! [`crate::general::frame_header`].
use super::{
    DELTA_Q_BITS, InterHeader, KeyframeHeader, LOG_SWITCHABLE_FILTERS, PRIMARY_REF_BITS,
    REFERENCE_COUNT_BITS, REFERENCE_SLOTS, REFRESH_ALL, SLOT_BITS, primary_reference, validate,
    validate_inter,
};
use crate::ccso::CcsoParams;
use crate::cdef::CdefParams;
use crate::coeff::LevelCoding;
use crate::deblock::DeblockParams;
use crate::interp_filter::FrameFilter;
use crate::prediction::InterpFilter;
use crate::sequence::{SequenceHeader, SequenceTiles, TcqFrames};
use crate::transform::QuantDeltas;
use crate::tx_type::{self, TxSetMode};
use nc_common::decode::{BitReader, DecodeError, Result};

/// The tools a lossy frame codes after its quantizer, as read.
struct LossyTools {
    delta_q: Option<u8>,
    levels: LevelCoding,
    deblock: DeblockParams,
    cdef: CdefParams,
    ccso: CcsoParams,
    transform_select: bool,
}

/// Parse the emitted keyframe prefix and return its byte length.
///
/// The caller must verify the OBU type is [`CLOSED_LOOP_KEY`] and decode the
/// remaining bytes as one tile. This function does not certify tile conformance.
/// # Errors
/// Rejects unsupported tools, truncation, invalid order hints and alignment bits.
pub fn parse_keyframe(seq: &SequenceHeader, payload: &[u8]) -> Result<(KeyframeHeader, usize)> {
    validate(
        seq,
        &KeyframeHeader {
            order_hint: 0,
            quantizer: 0,
            quant_deltas: QuantDeltas::NONE,
            screen_content: false,
            transform_select: false,
            tx_set: TxSetMode::IntraDct,
            levels: LevelCoding::Plain,
            adaptive_cdf: false,
            deblock: DeblockParams::OFF,
            cdef: CdefParams::OFF,
            ccso: CcsoParams::OFF,
            delta_q: None,
        },
    )?;
    let mut bits = BitReader::new(payload);
    fixed(&mut bits, 1, 1, "AV2 continuation tile group")?;
    if bits.read_uvlc()? != 0 {
        return Err(DecodeError::Unsupported("AV2 multi-frame header"));
    }
    if bits.read_uvlc()? != u32::from(seq.id) {
        return Err(DecodeError::Malformed("AV2 frame sequence id mismatch"));
    }
    let order_hint = if seq.single_picture {
        0
    } else {
        fixed(&mut bits, 1, 1, "AV2 delayed picture output")?;
        fixed(&mut bits, 0, 1, "AV2 frame size override")?;
        bits.read_bits(u32::from(seq.order_hint_bits))? as u16
    };
    let screen_content = read_screen_content(&mut bits, seq)?;
    fixed(&mut bits, 0, 1, "AV2 intra block copy")?;
    let adaptive_cdf = !bits.read_bit()?;
    read_single_tile(&mut bits, seq)?;
    let quantizer = bits.read_bits(8)? as u8;
    let quant_deltas = read_quant_deltas(&mut bits, seq)?;
    fixed(&mut bits, 0, 1, "AV2 frame segmentation")?;
    fixed(&mut bits, 0, 1, "AV2 quantizer matrices")?;
    let LossyTools {
        delta_q,
        levels,
        deblock,
        cdef,
        ccso,
        transform_select,
    } = read_lossy_tools(&mut bits, seq, (quantizer, seq.single_picture, true))?;
    let tx_set = TxSetMode::from_header(bits.read_bits(tx_type::HEADER_BITS)?)?;
    while !bits.is_byte_aligned() {
        if bits.read_bit()? {
            return Err(DecodeError::Malformed("AV2 tile-group alignment"));
        }
    }
    Ok((
        KeyframeHeader {
            order_hint,
            quantizer,
            quant_deltas,
            screen_content,
            transform_select,
            tx_set,
            adaptive_cdf,
            deblock,
            cdef,
            ccso,
            levels,
            delta_q,
        },
        bits.position() / 8,
    ))
}

/// Read what [`write_primary_reference`] writes, or the form that leaves
/// the primary reference to the decoder; returns whether the frame
/// inherits a reference's CDFs and whether the decoder chooses which.
fn read_primary_reference(bits: &mut BitReader<'_>) -> Result<(bool, bool)> {
    let signalled = bits.read_bit()?;
    let cross_frame_context = !bits.read_bit()?;
    if signalled && bits.read_bits(PRIMARY_REF_BITS)? != primary_reference(cross_frame_context) {
        return Err(DecodeError::Unsupported("AV2 primary reference context"));
    }
    Ok((cross_frame_context, !signalled))
}

/// `allow_screen_content_tools` where the sequence leaves it to the frame,
/// then with the tools the frame's `force_integer_mv` likewise (read and
/// dropped: no syntax of the subset depends on it).
fn read_screen_content(bits: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<bool> {
    let (tools, integer_mv) = seq.screen_content();
    let tools = match tools {
        Some(tools) => tools,
        None => bits.read_bit()?,
    };
    if tools && integer_mv.is_none() {
        bits.read_bit()?;
    }
    Ok(tools)
}

/// The tile layout of our frames: uniform spacing, one tile (no
/// `log2_tile_cols` / `log2_tile_rows` increment bits). A frame of a
/// sequence that carries its (one-tile) layout reuses that, always or by
/// a flag (AVM `read_tile_info`; reuse is always allowed for one tile).
fn read_single_tile(bits: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<()> {
    match seq.variants.tiles {
        Some(SequenceTiles::Fixed) => return Ok(()),
        Some(SequenceTiles::PerFrame) if bits.read_bit()? => return Ok(()),
        _ => {}
    }
    fixed(bits, 1, 1, "AV2 non-uniform tile spacing")?;
    if seq.max_width > 64 {
        fixed(bits, 0, 1, "AV2 multiple tile columns")?;
    }
    if seq.max_height > 64 {
        fixed(bits, 0, 1, "AV2 multiple tile rows")?;
    }
    Ok(())
}

/// The tools a lossy frame (`quantizer` nonzero) codes after its
/// quantizer: `delta_q_present` (with `delta_q_res`), the level coding,
/// deblocking, CDEF (`single`: of a single picture), CCSO (`intra`: of an
/// intra frame) and the transform mode; lossless frames have none of them.
fn read_lossy_tools(
    bits: &mut BitReader<'_>,
    seq: &SequenceHeader,
    (quantizer, single, intra): (u8, bool, bool),
) -> Result<LossyTools> {
    if quantizer == 0 {
        return Ok(LossyTools {
            delta_q: None,
            levels: LevelCoding::Plain,
            deblock: DeblockParams::OFF,
            cdef: CdefParams::OFF,
            ccso: CcsoParams::OFF,
            transform_select: false,
        });
    }
    let delta_q = if bits.read_bit()? {
        Some(bits.read_bits(crate::delta_q::DELTA_Q_RES_BITS)? as u8)
    } else {
        None
    };
    let levels = read_level_coding(bits, seq)?;
    let deblock = DeblockParams::read(bits, seq.variants.deblock_delta_bits)?;
    let cdef = CdefParams::read(bits, single, seq.cdef_on_skipped())?;
    let ccso = if seq.ccso {
        CcsoParams::read(bits, single, intra)?
    } else {
        CcsoParams::OFF
    };
    Ok(LossyTools {
        delta_q,
        levels,
        deblock,
        cdef,
        ccso,
        transform_select: bits.read_bit()?,
    })
}

fn read_level_coding(bits: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<LevelCoding> {
    let tcq = match seq.tcq {
        TcqFrames::Off => false,
        TcqFrames::Every => true,
        TcqFrames::Signalled => bits.read_bit()?,
    };
    let hiding = seq.parity_hiding && !tcq && bits.read_bit()?;
    Ok(LevelCoding::of(hiding, tcq))
}

/// Read what [`write_frame_filter`] writes.
fn read_frame_filter(bits: &mut BitReader<'_>) -> Result<FrameFilter> {
    if bits.read_bit()? {
        return Ok(FrameFilter::Switchable);
    }
    let code = bits.read_bits(LOG_SWITCHABLE_FILTERS)?;
    // Every two-bit value names a filter.
    Ok(FrameFilter::Fixed(
        InterpFilter::from_code(code).expect("two-bit filter codes"),
    ))
}

/// Parse a basic forward inter prefix, returning the entropy-tile byte offset.
/// The caller must verify OBU type and the single-picture reference invariant.
/// # Errors
/// Rejects unsupported syntax, invalid sequence parameters and truncated headers.
pub fn parse_inter(seq: &SequenceHeader, payload: &[u8]) -> Result<(InterHeader, usize)> {
    validate_inter(
        seq,
        &InterHeader {
            order_hint: 0,
            quantizer: 0,
            quant_deltas: QuantDeltas::NONE,
            transform_select: false,
            tx_set: TxSetMode::IntraDct,
            levels: LevelCoding::Plain,
            adaptive_cdf: false,
            cross_frame_context: false,
            derived_primary: false,
            refresh_flags: REFRESH_ALL,
            reference: seq.layering.explicit_references.then_some(0),
            second_reference: None,
            deblock: DeblockParams::OFF,
            cdef: CdefParams::OFF,
            ccso: CcsoParams::OFF,
            filter: FrameFilter::Fixed(InterpFilter::Regular),
            screen_content: false,
            skip_mode: false,
            delta_q: None,
        },
    )?;
    let mut bits = BitReader::new(payload);
    fixed(&mut bits, 1, 1, "AV2 continuation tile group")?;
    if bits.read_uvlc()? != 0 {
        return Err(DecodeError::Unsupported("AV2 multi-frame header"));
    }
    if bits.read_uvlc()? != u32::from(seq.id) {
        return Err(DecodeError::Malformed("AV2 frame sequence id mismatch"));
    }
    fixed(&mut bits, 1, 1, "AV2 regular intra-only frame")?;
    fixed(&mut bits, 1, 1, "AV2 delayed picture output")?;
    fixed(&mut bits, 0, 1, "AV2 frame size override")?;
    let order_hint = bits.read_bits(u32::from(seq.order_hint_bits))? as u16;
    let (cross_frame_context, derived_primary) = read_primary_reference(&mut bits)?;
    let refresh_flags = bits.read_bits(u32::from(REFERENCE_SLOTS))? as u8;
    let (reference, second_reference) = if seq.layering.explicit_references {
        read_reference_map(&mut bits)?
    } else {
        (None, None)
    };
    let screen_content = read_screen_content(&mut bits, seq)?;
    fixed(&mut bits, 0, 1, "AV2 intra block copy")?;
    fixed(&mut bits, 1, 1, "AV2 non-quarter-pel motion")?;
    let filter = read_frame_filter(&mut bits)?;
    let adaptive_cdf = !bits.read_bit()?;
    read_single_tile(&mut bits, seq)?;
    let quantizer = bits.read_bits(8)? as u8;
    let quant_deltas = read_quant_deltas(&mut bits, seq)?;
    fixed(&mut bits, 0, 1, "AV2 frame segmentation")?;
    fixed(&mut bits, 0, 1, "AV2 quantizer matrices")?;
    let LossyTools {
        delta_q,
        levels,
        deblock,
        cdef,
        ccso,
        transform_select,
    } = read_lossy_tools(&mut bits, seq, (quantizer, false, false))?;
    fixed(&mut bits, 0, 1, "AV2 compound reference selection")?;
    let skip_mode = bits.read_bit()?;
    let tx_set = TxSetMode::from_header(bits.read_bits(tx_type::HEADER_BITS)?)?;
    while !bits.is_byte_aligned() {
        if bits.read_bit()? {
            return Err(DecodeError::Malformed("AV2 tile-group alignment"));
        }
    }
    let frame = InterHeader {
        order_hint,
        quantizer,
        quant_deltas,
        transform_select,
        tx_set,
        adaptive_cdf,
        cross_frame_context,
        derived_primary,
        refresh_flags,
        reference,
        second_reference,
        deblock,
        cdef,
        ccso,
        filter,
        screen_content,
        skip_mode,
        levels,
        delta_q,
    };
    validate_inter(seq, &frame)?;
    Ok((frame, bits.position() / 8))
}

/// The explicit reference map (`explicit_ref_frame_map` 1,
/// `num_total_refs`, then each rank's slot): the rank-0 slot and any rank-1
/// slot.
fn read_reference_map(bits: &mut BitReader<'_>) -> Result<(Option<u8>, Option<u8>)> {
    fixed(bits, 1, 1, "AV2 derived references in an explicit sequence")?;
    let count = bits.read_bits(REFERENCE_COUNT_BITS)?;
    if !(1..=2).contains(&count) {
        return Err(DecodeError::Unsupported(
            "AV2 other than one or two references",
        ));
    }
    let first = bits.read_bits(SLOT_BITS)? as u8;
    let second = if count == 2 {
        Some(bits.read_bits(SLOT_BITS)? as u8)
    } else {
        None
    };
    Ok((Some(first), second))
}

fn read_quant_deltas(bits: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<QuantDeltas> {
    let mut deltas = QuantDeltas::NONE;
    if seq.dc_delta_q {
        deltas.luma_dc = read_delta_q(bits)?;
        deltas.chroma_dc = read_delta_q(bits)?;
    }
    if seq.chroma_delta_q {
        deltas.chroma_ac = read_delta_q(bits)?;
    }
    if !seq.dc_delta_q {
        // Equal AC/DC steps (AVM `equal_ac_dc_q`): U's DC takes its AC delta.
        deltas.chroma_dc = deltas.chroma_ac;
    }
    Ok(deltas)
}

fn read_delta_q(bits: &mut BitReader<'_>) -> Result<i8> {
    if !bits.read_bit()? {
        return Ok(0);
    }
    let shift = u8::BITS - DELTA_Q_BITS;
    Ok(((bits.read_bits(DELTA_Q_BITS)? as u8) << shift) as i8 >> shift)
}

fn fixed(bits: &mut BitReader<'_>, value: u32, count: u32, name: &'static str) -> Result<()> {
    if bits.read_bits(count)? != value {
        return Err(DecodeError::Unsupported(name));
    }
    Ok(())
}
