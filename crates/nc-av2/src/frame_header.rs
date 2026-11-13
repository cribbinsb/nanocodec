//! Closed-loop keyframe tile-group headers for the basic AV2 sequence subset.
//!
//! Follows AVM v1.0.0 `write_tilegroup_header` / `write_uncompressed_header`
//! and final AV2 frame/tile syntax. This writes only the byte-aligned prefix:
//! a valid entropy-coded tile must follow before the OBU is a picture.
//! AVM reference terms: BSD-3-Clause-Clear and separate AOM patent terms.

use crate::ccso::CcsoParams;
use crate::cdef::CdefParams;
use crate::coeff::LevelCoding;
use crate::deblock::DeblockParams;
use crate::interp_filter::FrameFilter;
use crate::prediction::InterpFilter;
use crate::sequence::{SequenceHeader, TcqFrames};
use crate::transform::{MAX_DELTA_Q, MIN_DELTA_Q, QuantDeltas};
use crate::tx_type::{self, TxSetMode};
use nc_common::BitWriter;
use nc_common::decode::{BitReader, DecodeError, Result};

/// `OBU_CLOSED_LOOP_KEY` in final AV2 (different from AV1 frame OBUs).
pub const CLOSED_LOOP_KEY: u8 = 4;

/// Bits of a non-switchable frame interpolation filter (AVM
/// `read_frame_interp_filter`).
const LOG_SWITCHABLE_FILTERS: u32 = 2;
/// Bits of a coded quantizer delta value (AVM `read_delta_q`:
/// `avm_rb_read_inv_signed_literal(rb, 6)` reads six bits plus the sign).
const DELTA_Q_BITS: u32 = 7;

/// Fields that vary in the initial one-tile keyframe header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyframeHeader {
    /// Display/coding order hint; zero in single-picture sequences.
    pub order_hint: u16,
    /// Native eight-bit qindex; zero enables lossless 4x4 transforms.
    pub quantizer: u8,
    /// Quantizer offsets: each nonzero only in sequences that code it
    /// (`chroma_delta_q` for chroma AC, `dc_delta_q` for both DC offsets),
    /// and all zero when lossless.
    pub quant_deltas: QuantDeltas,
    /// Signal per-block transform-size selection; false uses largest transforms.
    /// Must be false for lossless frames, where 4x4 is inferred.
    pub transform_select: bool,
    /// The frame's reduced transform set (`reduced_tx_set_used`).
    pub tx_set: TxSetMode,
    /// Update tile-local CDFs; cross-frame context inheritance remains disabled.
    pub adaptive_cdf: bool,
    /// Deblocking parameters; must be off for lossless frames, which have
    /// no deblocking syntax.
    pub deblock: DeblockParams,
    /// CDEF parameters; must be off for lossless frames, which have no CDEF
    /// syntax.
    pub cdef: CdefParams,
    /// CCSO parameters; off for lossless frames and in sequences without
    /// CCSO, which have no CCSO syntax.
    pub ccso: CcsoParams,
    /// How lossy luma levels are coded: luma DC parities hidden
    /// (`allow_parity_hiding`) or trellis-coded (`tcq_mode`), each only in
    /// sequences that enable it; lossless frames are plain.
    pub levels: LevelCoding,
    /// Superblock quantizer deltas (`delta_q_present`): `log2(delta_q_res)`
    /// when coded, `None` without; lossless frames have none.
    pub delta_q: Option<u8>,
}

fn validate(seq: &SequenceHeader, frame: &KeyframeHeader) -> Result<()> {
    seq.validate()?;
    let allowed = match frame.levels {
        _ if frame.quantizer == 0 => frame.levels == LevelCoding::Plain,
        _ if seq.tcq == TcqFrames::Every => frame.levels == LevelCoding::Tcq,
        LevelCoding::Plain => true,
        LevelCoding::ParityHiding => seq.parity_hiding,
        LevelCoding::Tcq => seq.tcq == TcqFrames::Signalled,
    };
    if !allowed {
        return Err(DecodeError::Malformed(
            "AV2 level coding the frame or its sequence cannot signal",
        ));
    }
    if frame.quantizer == 0 && frame.transform_select {
        return Err(DecodeError::Malformed(
            "AV2 lossless transform mode is inferred",
        ));
    }
    if frame.quantizer == 0 && frame.deblock.is_on() {
        return Err(DecodeError::Malformed(
            "AV2 lossless frames are not deblocked",
        ));
    }
    if frame.quantizer == 0 && frame.cdef.is_on() {
        return Err(DecodeError::Malformed("AV2 lossless frames have no CDEF"));
    }
    if (frame.quantizer == 0 || !seq.ccso) && frame.ccso.is_on() {
        return Err(DecodeError::Malformed(
            "AV2 CCSO needs a lossy frame of a CCSO sequence",
        ));
    }
    if frame.delta_q.is_some_and(|res_log2| {
        frame.quantizer == 0 || res_log2 >= 1 << crate::delta_q::DELTA_Q_RES_BITS
    }) {
        return Err(DecodeError::Malformed(
            "AV2 quantizer deltas need a lossy frame and a coded resolution",
        ));
    }
    // Without DC deltas the steps are equal: U's DC takes its AC delta.
    let deltas = frame.quant_deltas;
    let uncoded = (!seq.chroma_delta_q && deltas.chroma_ac != 0)
        || (!seq.dc_delta_q && (deltas.luma_dc != 0 || deltas.chroma_dc != deltas.chroma_ac));
    if uncoded || (frame.quantizer == 0 && deltas != QuantDeltas::NONE) {
        return Err(DecodeError::Malformed(
            "AV2 quantizer delta needs a lossy frame and the sequence flag",
        ));
    }
    if deltas
        .all()
        .iter()
        .any(|d| !(MIN_DELTA_Q..=MAX_DELTA_Q).contains(d))
    {
        return Err(DecodeError::Malformed("AV2 quantizer delta out of range"));
    }
    // Conservative single-tile limits, valid at every supported level.
    if seq.max_width > 4096 || seq.max_width.div_ceil(64) * seq.max_height.div_ceil(64) > 2304 {
        return Err(DecodeError::Unsupported("AV2 basic frame exceeds one tile"));
    }
    if (seq.single_picture && frame.order_hint != 0)
        || (!seq.single_picture && u32::from(frame.order_hint) >= 1 << seq.order_hint_bits)
    {
        return Err(DecodeError::Malformed(
            "AV2 frame order hint exceeds sequence width",
        ));
    }
    Ok(())
}

/// Write a byte-aligned closed-loop keyframe header for one entropy-coded tile.
///
/// Dimensions are the sequence maximum, all reference slots refresh, CDF updates
/// follow the requested tile-local policy; segmentation, quantizer matrices and delta-Q
/// are off; deblocking is as requested and the other in-loop filters are off.
/// Lossy frames use the requested transform-size mode and reduced luma set.
/// # Errors
/// Rejects invalid sequence parameters, order hints and oversized single tiles.
pub fn write_keyframe(seq: &SequenceHeader, frame: KeyframeHeader) -> Result<Vec<u8>> {
    validate(seq, &frame)?;
    let mut bits = BitWriter::new();
    bits.put_bit(true); // first tile group in frame
    bits.put_uvlc(0); // no multi-frame header
    bits.put_uvlc(u32::from(seq.id));
    if seq.single_picture {
        // Single-picture sequences infer SELECT_SCREEN_CONTENT_TOOLS.
        bits.put_bit(false); // allow_screen_content_tools
    } else {
        bits.put_bit(true); // immediate_output_picture
        bits.put_bit(false); // frame_size_override
        bits.put_bits(u32::from(frame.order_hint), u32::from(seq.order_hint_bits));
    }
    bits.put_bit(false); // allow_intrabc, signalled even when screen tools are off
    bits.put_bit(!frame.adaptive_cdf); // disable_cdf_update
    bits.put_bit(true); // uniform tile spacing
    if seq.max_width > 64 {
        bits.put_bit(false);
    } // log2_tile_cols = 0
    if seq.max_height > 64 {
        bits.put_bit(false);
    } // log2_tile_rows = 0
    bits.put_bits(u32::from(frame.quantizer), 8);
    write_quant_deltas(&mut bits, seq, frame.quant_deltas);
    bits.put_bit(false); // segmentation_enabled
    bits.put_bit(false); // using_qmatrix
    if frame.quantizer != 0 {
        write_delta_q_present(&mut bits, frame.delta_q);
        write_level_coding(&mut bits, seq, frame.levels);
        frame.deblock.write(&mut bits);
        frame.cdef.write(&mut bits, seq.single_picture);
        if seq.ccso {
            frame.ccso.write(&mut bits, seq.single_picture, true);
        }
        bits.put_bit(frame.transform_select); // lossless infers ONLY_4X4
    }
    bits.put_bits(frame.tx_set.header_value(), tx_type::HEADER_BITS);
    bits.byte_align_zero();
    Ok(bits.into_bytes())
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
        fixed(&mut bits, 0, 1, "AV2 screen-content tools")?;
        0
    } else {
        fixed(&mut bits, 1, 1, "AV2 delayed picture output")?;
        fixed(&mut bits, 0, 1, "AV2 frame size override")?;
        bits.read_bits(u32::from(seq.order_hint_bits))? as u16
    };
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

/// Reference slots (`REF_FRAMES`, the default DPB size).
pub const REFERENCE_SLOTS: u8 = 8;
/// Bits of a slot index (`ref_frames_log2`).
const SLOT_BITS: u32 = 3;
/// Bits of `num_total_refs` (`MAX_REFS_PER_FRAME_LOG2`).
const REFERENCE_COUNT_BITS: u32 = 3;
/// `primary_ref_frame` bits and values: the frame's first (here only)
/// reference, or none (`PRIMARY_REF_NONE`: default CDFs).
const PRIMARY_REF_BITS: u32 = 3;
const PRIMARY_REF_FIRST: u32 = 0;
const PRIMARY_REF_NONE: u32 = 7;
/// Every slot (`refresh_frame_flags` of a picture that replaces them all).
pub const REFRESH_ALL: u8 = u8::MAX;

/// Final AV2 regular tile-group OBU containing a forward inter picture.
pub const REGULAR_TILE_GROUP: u8 = 7;

/// Single-reference forward inter header for the native basic subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct InterHeader {
    /// Coding/display order hint modulo the sequence order-hint width.
    pub order_hint: u16,
    /// Native eight-bit quantizer; zero selects reversible lossless transforms.
    pub quantizer: u8,
    /// Quantizer offsets, as [`KeyframeHeader::quant_deltas`].
    pub quant_deltas: QuantDeltas,
    /// Select transform partition syntax for lossy frames.
    pub transform_select: bool,
    /// The frame's reduced transform set (`reduced_tx_set_used`).
    pub tx_set: TxSetMode,
    /// Update tile-local CDFs.
    pub adaptive_cdf: bool,
    /// Start from the reference picture's final CDFs (AVM
    /// `cross_frame_context` forward, `primary_ref_frame` 0, the one
    /// reference) instead of the defaults. Requires `adaptive_cdf`.
    pub cross_frame_context: bool,
    /// Reference slots this picture replaces (`refresh_frame_flags`). With
    /// implicit references all eight must refresh (255), so the slots hold
    /// one distinct picture.
    pub refresh_flags: u8,
    /// The slot holding the reference picture, signalled explicitly
    /// (`explicit_ref_frame_map`, one reference) exactly when the sequence
    /// enables explicit references; `None` otherwise.
    pub reference: Option<u8>,
    /// Deblocking parameters (off for lossless frames).
    pub deblock: DeblockParams,
    /// CDEF parameters (off for lossless frames).
    pub cdef: CdefParams,
    /// CCSO parameters, as [`KeyframeHeader::ccso`].
    pub ccso: CcsoParams,
    /// The frame's interpolation filter, or per-block switchable filters.
    pub filter: FrameFilter,
    /// Blocks may code skip mode (`skip_mode_flag`): the same-reference
    /// compound `NEAR_NEARMV` of the skip-mode list.
    pub skip_mode: bool,
    /// How lossy luma levels are coded, as [`KeyframeHeader::levels`].
    pub levels: LevelCoding,
    /// Superblock quantizer deltas, as [`KeyframeHeader::delta_q`].
    pub delta_q: Option<u8>,
}

fn validate_inter(seq: &SequenceHeader, frame: &InterHeader) -> Result<()> {
    validate(
        seq,
        &KeyframeHeader {
            order_hint: frame.order_hint,
            quantizer: frame.quantizer,
            quant_deltas: frame.quant_deltas,
            transform_select: frame.transform_select,
            tx_set: frame.tx_set,
            levels: frame.levels,
            adaptive_cdf: frame.adaptive_cdf,
            deblock: frame.deblock,
            cdef: frame.cdef,
            ccso: frame.ccso,
            delta_q: frame.delta_q,
        },
    )?;
    if seq.single_picture {
        return Err(DecodeError::Unsupported("AV2 inter pictures require video"));
    }
    if frame.cross_frame_context && !frame.adaptive_cdf {
        return Err(DecodeError::Unsupported(
            "AV2 cross-frame context without CDF updates",
        ));
    }
    match (seq.layering.explicit_references, frame.reference) {
        (true, Some(slot)) if slot < REFERENCE_SLOTS => Ok(()),
        (false, None) if frame.refresh_flags == REFRESH_ALL => Ok(()),
        (false, None) => Err(DecodeError::Unsupported(
            "AV2 implicit references require every slot to refresh",
        )),
        _ => Err(DecodeError::Malformed(
            "AV2 reference slot signalling does not match the sequence",
        )),
    }
}

/// The primary reference: signalled, with cross-frame context from the
/// frame's first reference, or none (default CDFs).
fn primary_reference(cross_frame_context: bool) -> u32 {
    if cross_frame_context {
        PRIMARY_REF_FIRST
    } else {
        PRIMARY_REF_NONE
    }
}

/// `signal_primary_ref_frame`, `cross_frame_context` and
/// `primary_ref_frame` (AVM `read_uncompressed_header`).
fn write_primary_reference(bits: &mut BitWriter, cross_frame_context: bool) {
    bits.put_bit(true); // signal primary reference
    bits.put_bit(!cross_frame_context); // disable cross-frame context
    bits.put_bits(primary_reference(cross_frame_context), PRIMARY_REF_BITS);
}

/// Read what [`write_primary_reference`] writes; returns whether the frame
/// inherits its reference's CDFs.
fn read_primary_reference(bits: &mut BitReader<'_>) -> Result<bool> {
    fixed(bits, 1, 1, "AV2 inferred primary reference")?;
    let cross_frame_context = !bits.read_bit()?;
    if bits.read_bits(PRIMARY_REF_BITS)? != primary_reference(cross_frame_context) {
        return Err(DecodeError::Unsupported("AV2 primary reference context"));
    }
    Ok(cross_frame_context)
}

/// Write a forward inter header referencing the sole previous reconstructed picture.
/// Requires all reference slots to contain that picture. Uses identity global
/// motion, quarter-pel precision, the frame's (or switchable) interpolation filter, optional tile-local CDF updates,
/// one tile, single-reference prediction, the requested deblocking and no other
/// in-loop filters.
/// # Errors
/// Rejects unsupported reference refresh, still sequences and invalid frame fields.
pub fn write_inter(seq: &SequenceHeader, frame: InterHeader) -> Result<Vec<u8>> {
    validate_inter(seq, &frame)?;
    let mut bits = BitWriter::new();
    bits.put_bit(true); // first tile group
    bits.put_uvlc(0); // no multi-frame header
    bits.put_uvlc(u32::from(seq.id));
    bits.put_bit(true); // INTER_FRAME
    bits.put_bit(true); // immediate output
    bits.put_bit(false); // frame size override
    bits.put_bits(u32::from(frame.order_hint), u32::from(seq.order_hint_bits));
    write_primary_reference(&mut bits, frame.cross_frame_context);
    bits.put_bits(u32::from(frame.refresh_flags), u32::from(REFERENCE_SLOTS));
    if let Some(slot) = frame.reference {
        bits.put_bit(true); // explicit_ref_frame_map
        bits.put_bits(1, REFERENCE_COUNT_BITS); // num_total_refs
        bits.put_bits(u32::from(slot), SLOT_BITS);
    }
    // Otherwise the mapping is derived: every slot holds the previous picture.
    bits.put_bit(false); // allow_intrabc
    bits.put_bit(true); // quarter-pel precision (no second precision flag)
    write_frame_filter(&mut bits, frame.filter);
    bits.put_bit(!frame.adaptive_cdf); // disable CDF update
    bits.put_bit(true); // uniform tile spacing
    if seq.max_width > 64 {
        bits.put_bit(false);
    }
    if seq.max_height > 64 {
        bits.put_bit(false);
    }
    bits.put_bits(u32::from(frame.quantizer), 8);
    write_quant_deltas(&mut bits, seq, frame.quant_deltas);
    bits.put_bit(false); // segmentation
    bits.put_bit(false); // quantization matrix
    if frame.quantizer != 0 {
        write_delta_q_present(&mut bits, frame.delta_q);
        write_level_coding(&mut bits, seq, frame.levels);
        frame.deblock.write(&mut bits);
        frame.cdef.write(&mut bits, false);
        if seq.ccso {
            frame.ccso.write(&mut bits, false, false);
        }
        bits.put_bit(frame.transform_select);
    }
    bits.put_bit(false); // SINGLE_REFERENCE
    // Final AV2 allows skip mode even with one distinct reference (AVM
    // `av2_setup_skip_mode_allowed`: both skip references are then slot 0).
    bits.put_bit(frame.skip_mode);
    bits.put_bits(frame.tx_set.header_value(), tx_type::HEADER_BITS);
    bits.byte_align_zero();
    Ok(bits.into_bytes())
}

/// The tile layout of our frames: uniform spacing, one tile (no
/// `log2_tile_cols` / `log2_tile_rows` increment bits).
fn read_single_tile(bits: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<()> {
    fixed(bits, 1, 1, "AV2 non-uniform tile spacing")?;
    if seq.max_width > 64 {
        fixed(bits, 0, 1, "AV2 multiple tile columns")?;
    }
    if seq.max_height > 64 {
        fixed(bits, 0, 1, "AV2 multiple tile rows")?;
    }
    Ok(())
}

/// The tools a lossy frame codes after its quantizer, as read.
struct LossyTools {
    delta_q: Option<u8>,
    levels: LevelCoding,
    deblock: DeblockParams,
    cdef: CdefParams,
    ccso: CcsoParams,
    transform_select: bool,
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
    let deblock = DeblockParams::read(bits)?;
    let cdef = CdefParams::read(bits, single)?;
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

/// `delta_q_present`, then `log2(delta_q_res)` when present (AVM
/// `read_uncompressed_header`, after the quantizer matrices).
fn write_delta_q_present(bits: &mut BitWriter, delta_q: Option<u8>) {
    bits.put_bit(delta_q.is_some());
    if let Some(res_log2) = delta_q {
        bits.put_bits(u32::from(res_log2), crate::delta_q::DELTA_Q_RES_BITS);
    }
}

/// A lossy frame's `tcq_mode` (coded only when the sequence lets each
/// frame choose) then `allow_parity_hiding` (coded in sequences enabling
/// it, unless the frame uses TCQ), after `delta_q_present` (AVM
/// `read_uncompressed_header`: without segments or quantizer matrices
/// nothing comes between).
fn write_level_coding(bits: &mut BitWriter, seq: &SequenceHeader, levels: LevelCoding) {
    let tcq = levels == LevelCoding::Tcq;
    if seq.tcq == TcqFrames::Signalled {
        bits.put_bit(tcq);
    }
    if seq.parity_hiding && !tcq {
        bits.put_bit(levels == LevelCoding::ParityHiding);
    }
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

/// The frame's interpolation filter mode (AVM `read_frame_interp_filter`):
/// a switchable flag, else the filter's two-bit code.
fn write_frame_filter(bits: &mut BitWriter, filter: FrameFilter) {
    match filter {
        FrameFilter::Switchable => bits.put_bit(true),
        FrameFilter::Fixed(filter) => {
            bits.put_bit(false);
            bits.put_bits(filter.code(), LOG_SWITCHABLE_FILTERS);
        }
    }
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
            refresh_flags: REFRESH_ALL,
            reference: seq.layering.explicit_references.then_some(0),
            deblock: DeblockParams::OFF,
            cdef: CdefParams::OFF,
            ccso: CcsoParams::OFF,
            filter: FrameFilter::Fixed(InterpFilter::Regular),
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
    let cross_frame_context = read_primary_reference(&mut bits)?;
    let refresh_flags = bits.read_bits(u32::from(REFERENCE_SLOTS))? as u8;
    let reference = if seq.layering.explicit_references {
        fixed(
            &mut bits,
            1,
            1,
            "AV2 derived references in an explicit sequence",
        )?;
        fixed(
            &mut bits,
            1,
            REFERENCE_COUNT_BITS,
            "AV2 more than one reference",
        )?;
        Some(bits.read_bits(SLOT_BITS)? as u8)
    } else {
        None
    };
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
        refresh_flags,
        reference,
        deblock,
        cdef,
        ccso,
        filter,
        skip_mode,
        levels,
        delta_q,
    };
    validate_inter(seq, &frame)?;
    Ok((frame, bits.position() / 8))
}

/// The deltas of AVM `setup_quantization` after `base_qindex`, in our
/// sequence subset (`separate_uv_delta_q = 0`, so V shares U's):
/// `y_dc_delta_q` and `u_dc_delta_q` when the sequence codes DC deltas
/// (`equal_ac_dc_q = 0`), then `u_ac_delta_q` when it codes chroma AC
/// deltas (`uv_ac_delta_q_enabled`).
fn write_quant_deltas(bits: &mut BitWriter, seq: &SequenceHeader, deltas: QuantDeltas) {
    if seq.dc_delta_q {
        write_delta_q(bits, deltas.luma_dc);
        write_delta_q(bits, deltas.chroma_dc);
    }
    if seq.chroma_delta_q {
        write_delta_q(bits, deltas.chroma_ac);
    }
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

/// AVM `read_delta_q`: a present flag, then the delta in two's complement.
fn write_delta_q(bits: &mut BitWriter, delta: i8) {
    bits.put_bit(delta != 0);
    if delta != 0 {
        // Two's complement in DELTA_Q_BITS (`avm_rb_read_inv_signed_literal`).
        bits.put_bits(
            u32::from(delta as u8) & ((1 << DELTA_Q_BITS) - 1),
            DELTA_Q_BITS,
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const INTER_REFERENCE: [&[u8]; 8] = [
        &[248, 7, 255, 232, 192, 2],
        &[248, 7, 255, 232, 192, 64, 144],
        &[248, 7, 255, 232, 217, 0, 144],
        &[248, 7, 255, 232, 255, 192, 144],
        &[248, 7, 255, 232, 192, 1, 0],
        &[248, 7, 255, 232, 192, 32, 72],
        &[248, 7, 255, 232, 204, 128, 72],
        &[248, 7, 255, 232, 223, 224, 72],
    ];

    const ADAPTIVE_INTER_REFERENCE: [&[u8]; 8] = [
        &[248, 7, 255, 232, 64, 2],
        &[248, 7, 255, 232, 64, 64, 144],
        &[248, 7, 255, 232, 89, 0, 144],
        &[248, 7, 255, 232, 127, 192, 144],
        &[248, 7, 255, 232, 64, 1, 0],
        &[248, 7, 255, 232, 64, 32, 72],
        &[248, 7, 255, 232, 76, 128, 72],
        &[248, 7, 255, 232, 95, 224, 72],
    ];

    #[test]
    fn adaptive_policy_roundtrips_for_key_and_inter_headers() {
        let sequence = SequenceHeader {
            id: 0,
            max_width: 72,
            max_height: 40,
            level: 31,
            single_picture: false,
            order_hint_bits: 8,
            crop: crate::sequence::CropWindow::NONE,
            layering: crate::sequence::Layering::FLAT,
            chroma_delta_q: false,
            dc_delta_q: false,
            parity_hiding: false,
            tcq: TcqFrames::Off,
            ccso: false,
            reference_lists: crate::mv_refs::ListConfig::BASIC,
        };
        for quantizer in [0, 90, 191, 255] {
            for adaptive_cdf in [false, true] {
                let key = KeyframeHeader {
                    order_hint: 0,
                    quantizer,
                    quant_deltas: QuantDeltas::NONE,
                    transform_select: quantizer != 0,
                    tx_set: TxSetMode::IntraDct,
                    levels: LevelCoding::Plain,
                    adaptive_cdf,
                    deblock: DeblockParams::from_qindex(quantizer),
                    cdef: CdefParams::OFF,
                    ccso: CcsoParams::OFF,
                    delta_q: None,
                };
                let bytes = write_keyframe(&sequence, key).unwrap();
                assert_eq!(
                    parse_keyframe(&sequence, &bytes).unwrap(),
                    (key, bytes.len())
                );
                let inter = InterHeader {
                    order_hint: 1,
                    quantizer,
                    quant_deltas: QuantDeltas::NONE,
                    transform_select: quantizer != 0,
                    tx_set: TxSetMode::IntraDct,
                    levels: LevelCoding::Plain,
                    refresh_flags: 255,
                    reference: None,
                    adaptive_cdf,
                    cross_frame_context: false,
                    deblock: DeblockParams::from_qindex(quantizer),
                    cdef: CdefParams::OFF,
                    ccso: CcsoParams::OFF,
                    filter: FrameFilter::Fixed(InterpFilter::Regular),
                    skip_mode: false,
                    delta_q: None,
                };
                let filters = [
                    InterpFilter::Regular,
                    InterpFilter::Smooth,
                    InterpFilter::Sharp,
                    InterpFilter::Bilinear,
                ]
                .map(FrameFilter::Fixed);
                for filter in filters.into_iter().chain([FrameFilter::Switchable]) {
                    let inter = InterHeader { filter, ..inter };
                    let bytes = write_inter(&sequence, inter).unwrap();
                    assert_eq!(
                        parse_inter(&sequence, &bytes).unwrap(),
                        (inter, bytes.len())
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_reference_slots_round_trip() {
        let seq = SequenceHeader {
            id: 0,
            max_width: 72,
            max_height: 40,
            level: 31,
            single_picture: false,
            order_hint_bits: 8,
            crop: crate::sequence::CropWindow::NONE,
            layering: crate::sequence::Layering {
                temporal_layers: 3,
                explicit_references: true,
            },
            chroma_delta_q: false,
            dc_delta_q: false,
            parity_hiding: false,
            tcq: TcqFrames::Off,
            ccso: false,
            reference_lists: crate::mv_refs::ListConfig::BASIC,
        };
        for (refresh_flags, slot) in [(0, 0), (1, 0), (0b0110, 7), (REFRESH_ALL, 3)] {
            let frame = InterHeader {
                order_hint: 5,
                quantizer: 120,
                quant_deltas: QuantDeltas::NONE,
                transform_select: true,
                tx_set: TxSetMode::IntraDct,
                levels: LevelCoding::Plain,
                adaptive_cdf: true,
                cross_frame_context: false,
                refresh_flags,
                reference: Some(slot),
                deblock: DeblockParams::from_qindex(120),
                cdef: CdefParams::OFF,
                ccso: CcsoParams::OFF,
                filter: FrameFilter::Fixed(InterpFilter::Regular),
                skip_mode: false,
                delta_q: None,
            };
            let bytes = write_inter(&seq, frame).unwrap();
            assert_eq!(parse_inter(&seq, &bytes).unwrap(), (frame, bytes.len()));
            let implicit = InterHeader {
                reference: None,
                ..frame
            };
            assert!(write_inter(&seq, implicit).is_err());
        }
    }

    /// A plain sequence and one with chroma AC deltas (`chroma_delta_q`).
    fn delta_sequences() -> (SequenceHeader, SequenceHeader) {
        let plain = SequenceHeader {
            id: 0,
            max_width: 72,
            max_height: 40,
            level: 31,
            single_picture: false,
            order_hint_bits: 8,
            crop: crate::sequence::CropWindow::NONE,
            layering: crate::sequence::Layering::FLAT,
            chroma_delta_q: false,
            dc_delta_q: false,
            parity_hiding: false,
            tcq: TcqFrames::Off,
            ccso: false,
            reference_lists: crate::mv_refs::ListConfig::BASIC,
        };
        let sequence = SequenceHeader {
            chroma_delta_q: true,
            ..plain
        };
        (plain, sequence)
    }

    /// A lossy key and inter header without deltas.
    fn delta_test_headers() -> (KeyframeHeader, InterHeader) {
        let key = KeyframeHeader {
            order_hint: 0,
            quantizer: 120,
            quant_deltas: QuantDeltas::NONE,
            transform_select: true,
            tx_set: TxSetMode::IntraDct,
            levels: LevelCoding::Plain,
            adaptive_cdf: true,
            deblock: DeblockParams::from_qindex(120),
            cdef: CdefParams::OFF,
            ccso: CcsoParams::OFF,
            delta_q: None,
        };
        let inter = InterHeader {
            order_hint: 1,
            quantizer: 120,
            quant_deltas: QuantDeltas::NONE,
            transform_select: true,
            tx_set: TxSetMode::IntraDct,
            levels: LevelCoding::Plain,
            refresh_flags: REFRESH_ALL,
            reference: None,
            adaptive_cdf: true,
            cross_frame_context: false,
            deblock: DeblockParams::from_qindex(120),
            cdef: CdefParams::OFF,
            ccso: CcsoParams::OFF,
            filter: FrameFilter::Fixed(InterpFilter::Regular),
            skip_mode: false,
            delta_q: None,
        };
        (key, inter)
    }

    #[test]
    fn quant_deltas_round_trip_and_are_validated() {
        let (plain, sequence) = delta_sequences();
        let with_dc = SequenceHeader {
            dc_delta_q: true,
            parity_hiding: false,
            ..sequence
        };
        let (key, inter) = delta_test_headers();
        let round_trip = |seq: &SequenceHeader, quant_deltas| {
            let key = KeyframeHeader {
                quant_deltas,
                ..key
            };
            let bytes = write_keyframe(seq, key).unwrap();
            assert_eq!(parse_keyframe(seq, &bytes).unwrap(), (key, bytes.len()));
            let inter = InterHeader {
                quant_deltas,
                ..inter
            };
            let bytes = write_inter(seq, inter).unwrap();
            assert_eq!(parse_inter(seq, &bytes).unwrap(), (inter, bytes.len()));
        };
        let rejects = |seq: &SequenceHeader, quant_deltas| {
            assert!(
                write_keyframe(
                    seq,
                    KeyframeHeader {
                        quant_deltas,
                        ..key
                    }
                )
                .is_err()
            );
            let inter = InterHeader {
                quant_deltas,
                ..inter
            };
            assert!(write_inter(seq, inter).is_err());
        };
        for delta in [MIN_DELTA_Q, -24, -1, 0, 1, 17, MAX_DELTA_Q] {
            let chroma = QuantDeltas {
                chroma_ac: delta,
                chroma_dc: delta,
                ..QuantDeltas::NONE
            };
            round_trip(&sequence, chroma);
            round_trip(&with_dc, chroma);
            let dc = QuantDeltas {
                luma_dc: delta,
                chroma_dc: -delta / 2,
                chroma_ac: 5,
            };
            round_trip(&with_dc, dc);
            if delta != 0 {
                rejects(&plain, chroma);
                rejects(&sequence, dc);
            }
        }
    }

    #[test]
    fn lossless_frames_code_no_quant_deltas() {
        let (_, sequence) = delta_sequences();
        let with_dc = SequenceHeader {
            dc_delta_q: true,
            parity_hiding: false,
            ..sequence
        };
        let (key, _) = delta_test_headers();
        let lossless = KeyframeHeader {
            quantizer: 0,
            quant_deltas: QuantDeltas {
                luma_dc: -1,
                ..QuantDeltas::NONE
            },
            transform_select: false,
            deblock: DeblockParams::OFF,
            ..key
        };
        assert!(write_keyframe(&with_dc, lossless).is_err());
        let lossless = KeyframeHeader {
            quant_deltas: QuantDeltas::NONE,
            ..lossless
        };
        let bytes = write_keyframe(&sequence, lossless).unwrap();
        assert_eq!(
            parse_keyframe(&sequence, &bytes).unwrap(),
            (lossless, bytes.len())
        );
    }

    #[test]
    fn matches_original_avm_inter_header_writer() {
        let mut index = 0;
        for (max_width, max_height) in [(8, 8), (72, 40)] {
            let seq = SequenceHeader {
                id: 0,
                max_width,
                max_height,
                level: 31,
                single_picture: false,
                order_hint_bits: 8,
                crop: crate::sequence::CropWindow::NONE,
                layering: crate::sequence::Layering::FLAT,
                chroma_delta_q: false,
                dc_delta_q: false,
                parity_hiding: false,
                tcq: TcqFrames::Off,
                ccso: false,
                reference_lists: crate::mv_refs::ListConfig::BASIC,
            };
            for quantizer in [0, 1, 100, 255] {
                let frame = InterHeader {
                    order_hint: 1,
                    quantizer,
                    quant_deltas: QuantDeltas::NONE,
                    transform_select: quantizer != 0,
                    tx_set: TxSetMode::IntraDct,
                    levels: LevelCoding::Plain,
                    adaptive_cdf: false,
                    cross_frame_context: false,
                    refresh_flags: 255,
                    reference: None,
                    deblock: DeblockParams::OFF,
                    cdef: CdefParams::OFF,
                    ccso: CcsoParams::OFF,
                    filter: FrameFilter::Fixed(InterpFilter::Regular),
                    skip_mode: false,
                    delta_q: None,
                };
                let prefix = write_inter(&seq, frame).unwrap();
                assert_eq!(prefix, INTER_REFERENCE[index]);
                let adaptive = InterHeader {
                    adaptive_cdf: true,
                    cross_frame_context: false,
                    ..frame
                };
                assert_eq!(
                    write_inter(&seq, adaptive).unwrap(),
                    ADAPTIVE_INTER_REFERENCE[index]
                );
                index += 1;
                for length in 0..prefix.len() {
                    assert!(parse_inter(&seq, &prefix[..length]).is_err());
                }
                let mut payload = prefix.clone();
                payload.extend([0xa5, 0x81, 0x00]);
                assert_eq!(parse_inter(&seq, &payload).unwrap(), (frame, prefix.len()));
                assert!(
                    write_inter(
                        &seq,
                        InterHeader {
                            refresh_flags: 1,
                            ..frame
                        }
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn keyframe_prefixes_preserve_tile_boundary() {
        for single_picture in [false, true] {
            for (max_width, max_height) in [(8, 8), (64, 64), (72, 40), (1920, 1080)] {
                let seq = SequenceHeader {
                    id: 3,
                    max_width,
                    max_height,
                    level: 31,
                    single_picture,
                    order_hint_bits: if single_picture { 0 } else { 8 },
                    crop: crate::sequence::CropWindow::NONE,
                    layering: crate::sequence::Layering::FLAT,
                    chroma_delta_q: false,
                    dc_delta_q: false,
                    parity_hiding: false,
                    tcq: TcqFrames::Off,
                    ccso: false,
                    reference_lists: crate::mv_refs::ListConfig::BASIC,
                };
                for quantizer in [0, 1, 100, 255] {
                    let frame = KeyframeHeader {
                        order_hint: if single_picture { 0 } else { 254 },
                        quantizer,
                        quant_deltas: QuantDeltas::NONE,
                        transform_select: false,
                        tx_set: TxSetMode::IntraDct,
                        levels: LevelCoding::Plain,
                        adaptive_cdf: false,
                        deblock: DeblockParams::OFF,
                        cdef: CdefParams::OFF,
                        ccso: CcsoParams::OFF,
                        delta_q: None,
                    };
                    let prefix = write_keyframe(&seq, frame).unwrap();
                    for len in 0..prefix.len() {
                        assert!(parse_keyframe(&seq, &prefix[..len]).is_err());
                    }
                    let mut packet = prefix.clone();
                    packet.extend([0xa5, 0x81, 0x00]);
                    assert_eq!(
                        parse_keyframe(&seq, &packet).unwrap(),
                        (frame, prefix.len())
                    );
                }
            }
        }
    }
}
