//! VP9 Annex B superframe packing, parsing, and layer stripping (VP9
//! Bitstream & Decoding Process Specification, §6.2 and Annex B §B.1–§B.3;
//! libvpx `vp9/vp9_cx_iface.c` `write_superframe_index` and
//! `vp9/decoder/vp9_decoder.c` `vp9_parse_superframe_index`).
//!
//! # Superframe format (VP9 Annex B)
//!
//! A temporal unit containing $N \in [2, 8]$ coded frames concatenates the
//! frames in decoding order (base layer `S0` first, enhancement layer `S1`
//! second) followed by a superframe index:
//!
//! ```text
//! +----------+----------+--------+-------------+-------------+--------+
//! | frame[0] | frame[1] | marker | size[0] LE  | size[1] LE  | marker |
//! +----------+----------+--------+-------------+-------------+--------+
//! ```
//!
//! where `marker` is `0b110_mm_fff` (`0xc0 | ((mag - 1) << 3) | (N - 1)`),
//! `mag` $\in [1, 4]$ is the number of little-endian bytes per frame size,
//! and the trailing `marker` byte duplicates the leading `marker` byte so a
//! parser can locate the index from the end of the packet (`index_size = 2 +
//! mag * N`).
//!
//! A single-frame packet has no superframe index, but if its final byte
//! matches `(byte & 0xe0) == 0xc0` a decoder scanning from the end of the
//! packet would mistake it for a superframe marker and reject the packet as
//! corrupt (`VP9 Annex B §B.3`). Appending one `0x00` padding byte escapes
//! that collision (`escape_single_frame`).
//!
//! # Display semantics (`show_frame`, VP9 §6.2 / §7.1)
//!
//! `S0` is always coded with `show_frame = 0` (`intra_only = 0` on inter
//! frames) and refreshes at least one reference slot:
//! - in a 2-layer temporal unit `[S0, S1]`, `S1` (`show_frame = 1`) is the
//!   single picture output for the unit;
//! - a unit without `S1` (the non-enhancement units of a decimated schedule
//!   such as `480p30 + 720p5`, or `S1` dropped by rate control) is `[S0,
//!   show_existing_frame]`, the second frame showing a slot `S0` refreshed;
//! - stripping `S1` for a base-only receiver ([`strip_superframe_to_base`])
//!   gives the same `[S0, show_existing_frame]` unit.
//!
//! A hidden `S0` keeps every decoder's `use_prev_frame_mvs` (VP9 §7.2: the
//! previous decoded frame must be shown and of the same size) at 0 whether
//! or not it decodes `S1`, so `S0` need not be error resilient and both
//! layers keep their own adapted probability contexts (`backend.rs`
//! `BackendSpatial::context_use`). Rewriting `S0` to `show_frame = 1`
//! instead, as before, let a base-only decoder use `S0`'s vectors in the
//! next `S0` while a decoder of both layers could not. The base-only
//! stream pays 5-6 bytes a unit for it (the one-byte frame and the index).
//! Streams whose base frames are shown (libvpx's SVC) strip to that frame.

use nc_common::decode::{BitReader, DecodeError, Result};

/// Maximum number of frames an Annex B superframe can hold (`VP9 Annex B §B.2`).
pub const MAX_SUPERFRAME_FRAMES: usize = 8;

/// Top 3-bit mask of an Annex B superframe marker byte (`0b1110_0000`).
pub const SUPERFRAME_MARKER_MASK: u8 = 0xe0;

/// Top 3-bit value of an Annex B superframe marker byte (`0b1100_0000`).
pub const SUPERFRAME_MARKER: u8 = 0xc0;
/// `frames_in_superframe_minus_1`: the marker's low 3 bits (Annex B §B.2).
const FRAMES_MINUS_1_MASK: u8 = 0x07;
/// `bytes_per_framesize_minus_1`: the marker's bits 3..4 (Annex B §B.2).
const SIZE_BYTES_SHIFT: u8 = 3;
const SIZE_BYTES_MINUS_1_MASK: u8 = 0x03;

/// VP9 2-bit `frame_marker` at the start of every uncompressed header (`VP9 §6.2`).
const FRAME_MARKER: u32 = 0b10;
/// A Profile 0 `show_existing_frame` header without its 3-bit slot
/// (`frame_marker`, `profile_low_bit`, `profile_high_bit`,
/// `show_existing_frame = 1`): one byte with the slot (`VP9 §6.2`).
const SHOW_EXISTING_FRAME: u8 = 0b1000_1000;

/// VP9 24-bit `sync_code` on key and intra-only frames (`VP9 §6.2.2`).
const SYNC_CODE: u32 = 0x49_83_42;

/// Number of active inter reference slots per frame (`LAST`, `GOLDEN`, `ALTREF`).
const REFS_PER_FRAME: usize = 3;

/// `MAX_TILE_WIDTH_B64` and `MIN_TILE_WIDTH_B64` in 64x64 superblocks (`VP9 §6.2.10`).
const MAX_TILE_WIDTH_SB: usize = 64;
const MIN_TILE_WIDTH_SB: usize = 4;

/// Segmentation feature bit widths and sign flags (`VP9 §6.2.9`).
const SEG_FEATURE_MAX: [u32; 4] = [255, 63, 3, 0];
const SEG_FEATURE_SIGNED: [bool; 4] = [true, true, false, false];

/// Parsed summary of a VP9 frame's uncompressed header (`VP9 §6.2`), sufficient
/// for packet-level inspection and `show_frame` rewriting.
#[allow(
    clippy::struct_excessive_bools,
    reason = "VP9 uncompressed header flags"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UncompressedHeaderInfo {
    /// Whether this frame is a `show_existing_frame` 1-byte pointer (`VP9 §6.2`).
    pub show_existing_frame: bool,
    /// Whether this frame is a `KEY_FRAME` (`frame_type == 0`).
    pub key_frame: bool,
    /// Whether this frame is marked for display (`show_frame == 1`).
    pub show_frame: bool,
    /// Whether `error_resilient_mode == 1`.
    pub error_resilient: bool,
    /// Whether this is an `intra_only` inter frame (`frame_type == 1 && !show_frame && intra_only`).
    pub intra_only: bool,
    /// 8-bit `refresh_frame_flags` mask (`0xff` on key frames, `0x00` on `show_existing_frame`).
    pub refresh_frame_flags: u8,
    /// Coded frame dimensions `(width, height)` if explicitly present in the header.
    pub coded_size: Option<(usize, usize)>,
    /// Bit length of the uncompressed header before `trailing_bits()` (`VP9 §6.2.1`).
    pub header_bits: usize,
    /// Byte length of the uncompressed header including `trailing_bits()`.
    pub header_bytes: usize,
    /// Byte length of the arithmetic-coded compressed header (`header_size_in_bytes`).
    pub compressed_header_bytes: usize,
}

/// Split an Annex B superframe packet into its constituent frame slices
/// (`VP9 Annex B §B.3`).
///
/// Returns `None` if `packet` does not end with a valid Annex B superframe
/// index (i.e. `packet` is a single-frame packet).
#[must_use]
pub fn split_superframe(packet: &[u8]) -> Option<Vec<&[u8]>> {
    let &marker = packet.last()?;
    if marker & SUPERFRAME_MARKER_MASK != SUPERFRAME_MARKER {
        return None;
    }
    let frames = usize::from(marker & FRAMES_MINUS_1_MASK) + 1;
    let size_bytes = usize::from((marker >> SIZE_BYTES_SHIFT) & SIZE_BYTES_MINUS_1_MASK) + 1;
    let index_bytes = 2 + size_bytes * frames;
    if packet.len() < index_bytes || packet[packet.len() - index_bytes] != marker {
        return None;
    }
    let index = &packet[packet.len() - index_bytes + 1..packet.len() - 1];
    let mut data = &packet[..packet.len() - index_bytes];
    let mut result = Vec::with_capacity(frames);
    for size in index.chunks_exact(size_bytes) {
        let size = size
            .iter()
            .rev()
            .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        if size == 0 {
            return None;
        }
        let frame = data.get(..size)?;
        data = &data[size..];
        result.push(frame);
    }
    Some(result)
}

/// Escape a single-frame VP9 packet so its final byte cannot collide with an
/// Annex B superframe marker (`VP9 Annex B §B.3`; libvpx `vp9_parse_superframe_index`).
#[must_use]
pub fn escape_single_frame(frame: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.len() + 1);
    out.extend_from_slice(frame);
    if frame
        .last()
        .is_some_and(|&b| (b & SUPERFRAME_MARKER_MASK) == SUPERFRAME_MARKER)
    {
        out.push(0x00);
    }
    out
}

/// Pack `1..=8` coded VP9 frames into a single temporal unit (`VP9 Annex B §B.2`).
///
/// - For a single frame (`frames.len() == 1`), returns the frame with
///   [`escape_single_frame`] applied (no multi-frame index overhead).
/// - For `2..=8` frames, concatenates the frames and appends the Annex B
///   superframe index using the smallest magnitude (`1..=4` bytes per size)
///   that fits the largest frame in `frames`.
///
/// # Panics
///
/// Panics if `frames.len() > MAX_SUPERFRAME_FRAMES` or any frame exceeds `u32::MAX` bytes.
#[must_use]
pub fn pack_superframe(frames: &[&[u8]]) -> Vec<u8> {
    assert!(
        frames.len() <= MAX_SUPERFRAME_FRAMES,
        "VP9 Annex B superframes support at most {MAX_SUPERFRAME_FRAMES} frames"
    );
    match frames {
        [] => Vec::new(),
        [single] => escape_single_frame(single),
        many => {
            let payload_bytes: usize = many.iter().map(|f| f.len()).sum();
            let mut out = Vec::with_capacity(payload_bytes + 2 + 4 * many.len());
            for frame in many {
                out.extend_from_slice(frame);
            }
            let lengths: Vec<usize> = many.iter().map(|f| f.len()).collect();
            write_superframe_index(&mut out, &lengths);
            out
        }
    }
}

/// Append the superframe index of frames of `lengths` bytes (Annex B §B.2):
/// the marker, each size in the fewest little-endian bytes that hold the
/// largest, and the marker again.
fn write_superframe_index(out: &mut Vec<u8>, lengths: &[usize]) {
    assert!(
        (2..=MAX_SUPERFRAME_FRAMES).contains(&lengths.len()),
        "VP9 superframes hold 2 to {MAX_SUPERFRAME_FRAMES} frames"
    );
    let largest = lengths.iter().copied().max().unwrap_or(0);
    let largest = u32::try_from(largest).expect("VP9 superframe frame exceeds 4 GiB");
    let size_bytes = (u32::BITS - largest.leading_zeros())
        .div_ceil(u8::BITS)
        .max(1) as usize;
    let marker = SUPERFRAME_MARKER
        | (((size_bytes - 1) as u8) << SIZE_BYTES_SHIFT)
        | ((lengths.len() - 1) as u8);
    out.reserve(2 + lengths.len() * size_bytes);
    out.push(marker);
    for &len in lengths {
        out.extend_from_slice(&(len as u32).to_le_bytes()[..size_bytes]);
    }
    out.push(marker);
}

/// Inspect a VP9 Profile 0 frame's uncompressed header (`VP9 §6.2`).
///
/// If the frame is an inter frame that derives its size from a reference
/// (`found_ref == 1` in `frame_size_with_refs`, `VP9 §6.2.6`) without a
/// distinct `render_size`, `fallback_width` sizes `tile_info` (without one
/// such a frame is an error: its tile syntax depends on the width). Our
/// spatial SVC frames take their size from a reference of their own layer
/// when one has it.
///
/// # Errors
///
/// Returns [`DecodeError`] if `frame` is truncated or not a valid VP9 Profile 0
/// uncompressed header.
pub fn parse_uncompressed_header_info(
    frame: &[u8],
    fallback_width: Option<usize>,
) -> Result<UncompressedHeaderInfo> {
    let mut bits = BitReader::new(frame);
    if bits.read_bits(2)? != FRAME_MARKER {
        return Err(DecodeError::Malformed("VP9 frame marker"));
    }
    let profile = u8::from(bits.read_bit()?) | (u8::from(bits.read_bit()?) << 1);
    if profile == 3 && bits.read_bit()? {
        return Err(DecodeError::Unsupported("VP9 reserved profile"));
    }
    if profile != 0 {
        return Err(DecodeError::Unsupported("VP9 profile other than 0"));
    }
    if bits.read_bit()? {
        let _slot = bits.read_bits(3)?;
        let header_bits = bits.position();
        return Ok(UncompressedHeaderInfo {
            show_existing_frame: true,
            key_frame: false,
            show_frame: true,
            error_resilient: false,
            intra_only: false,
            refresh_frame_flags: 0,
            coded_size: None,
            header_bits,
            header_bytes: header_bits.div_ceil(8),
            compressed_header_bytes: 0,
        });
    }
    let key_frame = !bits.read_bit()?;
    let show_frame = bits.read_bit()?;
    let error_resilient = bits.read_bit()?;
    let (intra_only, refresh_frame_flags, coded_size, render_size) = if key_frame {
        if bits.read_bits(24)? != SYNC_CODE {
            return Err(DecodeError::Malformed("VP9 sync code"));
        }
        read_color_config_profile0(&mut bits)?;
        let w = bits.read_bits(16)? as usize + 1;
        let h = bits.read_bits(16)? as usize + 1;
        let render = if bits.read_bit()? {
            let rw = bits.read_bits(16)? as usize + 1;
            let rh = bits.read_bits(16)? as usize + 1;
            Some((rw, rh))
        } else {
            None
        };
        (false, 0xff, Some((w, h)), render)
    } else {
        read_inter_geometry(&mut bits, show_frame, error_resilient)?
    };

    if !error_resilient {
        let _refresh_frame_context = bits.read_bit()?;
        let _frame_parallel = bits.read_bit()?;
    }
    let _frame_context_idx = bits.read_bits(2)?;

    // `tile_info` depends on the width (VP9 §6.2.14: tile-column bits
    // exist from 8 superblock columns on), so a frame sized from a
    // reference needs it from the caller.
    let width = coded_size
        .map(|(w, _)| w)
        .or_else(|| render_size.map(|(w, _)| w))
        .or(fallback_width)
        .ok_or(DecodeError::Unsupported(
            "VP9 frame sized from a reference: its width is needed",
        ))?;
    skip_loop_quant_seg_tile(&mut bits, width)?;

    let compressed_header_bytes = bits.read_bits(16)? as usize;
    if compressed_header_bytes == 0 {
        return Err(DecodeError::Malformed("VP9 zero compressed header size"));
    }
    let header_bits = bits.position();
    let header_bytes = header_bits.div_ceil(8);
    if frame.len() < header_bytes + compressed_header_bytes {
        return Err(DecodeError::Truncated);
    }
    Ok(UncompressedHeaderInfo {
        show_existing_frame: false,
        key_frame,
        show_frame,
        error_resilient,
        intra_only,
        refresh_frame_flags,
        coded_size,
        header_bits,
        header_bytes,
        compressed_header_bytes,
    })
}

type InterGeometry = (bool, u8, Option<(usize, usize)>, Option<(usize, usize)>);

fn read_inter_geometry(
    bits: &mut BitReader<'_>,
    show_frame: bool,
    error_resilient: bool,
) -> Result<InterGeometry> {
    let intra_only = !show_frame && bits.read_bit()?;
    if !error_resilient {
        let _reset_frame_context = bits.read_bits(2)?;
    }
    let mut coded_size = None;
    let mut render_size = None;
    let refresh_frame_flags = if intra_only {
        if bits.read_bits(24)? != SYNC_CODE {
            return Err(DecodeError::Malformed("VP9 intra-only sync code"));
        }
        let refresh = bits.read_bits(8)? as u8;
        let w = bits.read_bits(16)? as usize + 1;
        let h = bits.read_bits(16)? as usize + 1;
        coded_size = Some((w, h));
        if bits.read_bit()? {
            let rw = bits.read_bits(16)? as usize + 1;
            let rh = bits.read_bits(16)? as usize + 1;
            render_size = Some((rw, rh));
        }
        refresh
    } else {
        let refresh = bits.read_bits(8)? as u8;
        for _ in 0..REFS_PER_FRAME {
            let _ref_idx = bits.read_bits(3)?;
            let _sign_bias = bits.read_bit()?;
        }
        let mut found_ref = false;
        for _ in 0..REFS_PER_FRAME {
            if bits.read_bit()? {
                found_ref = true;
                break;
            }
        }
        if !found_ref {
            let w = bits.read_bits(16)? as usize + 1;
            let h = bits.read_bits(16)? as usize + 1;
            coded_size = Some((w, h));
        }
        if bits.read_bit()? {
            let rw = bits.read_bits(16)? as usize + 1;
            let rh = bits.read_bits(16)? as usize + 1;
            render_size = Some((rw, rh));
        }
        let _allow_hp = bits.read_bit()?;
        let is_switchable = bits.read_bit()?;
        if !is_switchable {
            let _filter = bits.read_bits(2)?;
        }
        refresh
    };
    Ok((intra_only, refresh_frame_flags, coded_size, render_size))
}

fn skip_loop_quant_seg_tile(bits: &mut BitReader<'_>, width: usize) -> Result<()> {
    // `loop_filter_params` (VP9 §6.2.8).
    let _filter_level = bits.read_bits(6)?;
    let _sharpness = bits.read_bits(3)?;
    if bits.read_bit()? && bits.read_bit()? {
        for _ in 0..4 {
            if bits.read_bit()? {
                let _ = bits.read_bits(7)?;
            }
        }
        for _ in 0..2 {
            if bits.read_bit()? {
                let _ = bits.read_bits(7)?;
            }
        }
    }

    // `quantization_params` (VP9 §6.2.7).
    let _base_q_idx = bits.read_bits(8)?;
    for _ in 0..3 {
        if bits.read_bit()? {
            let _ = bits.read_bits(5)?;
        }
    }

    // `segmentation_params` (VP9 §6.2.9).
    if bits.read_bit()? {
        if bits.read_bit()? {
            for _ in 0..7 {
                if bits.read_bit()? {
                    let _ = bits.read_bits(8)?;
                }
            }
            if bits.read_bit()? {
                for _ in 0..3 {
                    if bits.read_bit()? {
                        let _ = bits.read_bits(8)?;
                    }
                }
            }
        }
        if bits.read_bit()? {
            let _abs_or_delta = bits.read_bit()?;
            for _ in 0..8 {
                for (feature, &max) in SEG_FEATURE_MAX.iter().enumerate() {
                    if bits.read_bit()? {
                        let width = u32::BITS - max.leading_zeros();
                        let _ = bits.read_bits(width)?;
                        if SEG_FEATURE_SIGNED[feature] {
                            let _ = bits.read_bit()?;
                        }
                    }
                }
            }
        }
    }

    // `tile_info` (VP9 §6.2.10).
    let sb_cols = width.div_ceil(64);
    let mut min_log2 = 0u32;
    while (MAX_TILE_WIDTH_SB << min_log2) < sb_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb_cols >> max_log2) >= MIN_TILE_WIDTH_SB {
        max_log2 += 1;
    }
    max_log2 -= 1;
    let mut cols_log2 = min_log2;
    while cols_log2 < max_log2 && bits.read_bit()? {
        cols_log2 += 1;
    }
    if bits.read_bit()? {
        let _ = bits.read_bit()?;
    }
    Ok(())
}

fn read_color_config_profile0(bits: &mut BitReader<'_>) -> Result<()> {
    let color_space = bits.read_bits(3)?;
    if color_space == 7 {
        return Err(DecodeError::Unsupported("VP9 sRGB in profile 0"));
    }
    let _color_range = bits.read_bit()?;
    Ok(())
}

/// Strip enhancement layers from a VP9 packet (a 2-frame Annex B superframe or
/// a 1-frame base-only packet) so the returned packet decodes and displays the
/// base layer (`S0`) as a standalone VP9 stream (`VP9 Annex B §B.3`).
///
/// # Errors
///
/// Returns [`DecodeError`] if the base frame header is malformed.
pub fn strip_superframe_to_base(packet: &[u8]) -> Result<Vec<u8>> {
    let base = match split_superframe(packet) {
        Some(frames) => frames
            .first()
            .copied()
            .ok_or(DecodeError::Malformed("empty VP9 superframe"))?,
        None => trim_escape_padding(packet),
    };
    if shows_frame(base)? {
        Ok(escape_single_frame(base))
    } else {
        show_hidden_base(base)
    }
}

/// The unit that shows the hidden base frame `base`: it and a
/// `show_existing_frame` of the lowest slot it refreshes, as a superframe.
///
/// # Errors
///
/// Returns [`DecodeError`] if the header is malformed or refreshes no slot.
fn show_hidden_base(base: &[u8]) -> Result<Vec<u8>> {
    let refresh = refresh_frame_flags(base)?;
    if refresh == 0 {
        return Err(DecodeError::Malformed(
            "VP9 hidden base frame refreshes no slot to show",
        ));
    }
    let slot = refresh.trailing_zeros() as u8;
    Ok(pack_superframe(&[base, &[SHOW_EXISTING_FRAME | slot]]))
}

/// The `refresh_frame_flags` of a Profile 0 frame, read from the header
/// fields before the frame size, so a frame sized from a reference needs no
/// width (`VP9 §6.2`).
fn refresh_frame_flags(frame: &[u8]) -> Result<u8> {
    let mut bits = BitReader::new(frame);
    if bits.read_bits(2)? != FRAME_MARKER {
        return Err(DecodeError::Malformed("VP9 frame marker"));
    }
    if bits.read_bits(2)? != 0 {
        return Err(DecodeError::Unsupported("VP9 profile other than 0"));
    }
    if bits.read_bit()? {
        return Ok(0); // show_existing_frame
    }
    let key_frame = !bits.read_bit()?;
    let show_frame = bits.read_bit()?;
    let error_resilient = bits.read_bit()?;
    if key_frame {
        return Ok(u8::MAX);
    }
    let intra_only = !show_frame && bits.read_bit()?;
    if !error_resilient {
        let _reset_frame_context = bits.read_bits(2)?;
    }
    if intra_only && bits.read_bits(24)? != SYNC_CODE {
        return Err(DecodeError::Malformed("VP9 intra-only sync code"));
    }
    Ok(bits.read_bits(8)? as u8)
}

/// Whether `frame` is shown (`show_existing_frame` or `show_frame`, read
/// from the first bits of its uncompressed header, VP9 §6.2), without the
/// frame size the rest of the header needs.
fn shows_frame(frame: &[u8]) -> Result<bool> {
    let mut bits = BitReader::new(frame);
    if bits.read_bits(2)? != FRAME_MARKER {
        return Err(DecodeError::Malformed("VP9 frame marker"));
    }
    let profile = u8::from(bits.read_bit()?) | (u8::from(bits.read_bit()?) << 1);
    if profile != 0 {
        return Err(DecodeError::Unsupported("VP9 profile other than 0"));
    }
    if bits.read_bit()? {
        return Ok(true);
    }
    let _frame_type = bits.read_bit()?;
    bits.read_bit()
}

fn trim_escape_padding(packet: &[u8]) -> &[u8] {
    if packet.len() >= 2
        && packet[packet.len() - 1] == 0x00
        && (packet[packet.len() - 2] & SUPERFRAME_MARKER_MASK) == SUPERFRAME_MARKER
    {
        &packet[..packet.len() - 1]
    } else {
        packet
    }
}

/// Finalize a spatial temporal unit (`data` containing concatenated layer frames
/// with byte lengths `layer_lengths`):
/// - When `layer_lengths` has 2..=8 frames (`[S0, S1]`), appends the VP9 Annex B
///   superframe index (`§B.2`).
/// - When `layer_lengths` has 1 frame (`S0`-only TU on a sparse enhancement
///   schedule or when `S1` was dropped by rate control), appends a
///   `show_existing_frame` of `S0` and the index of the two (the module
///   docs).
pub fn pack_spatial_tu(data: &mut Vec<u8>, layer_lengths: &[usize]) {
    match layer_lengths.len() {
        0 => {}
        1 => {
            *data = show_hidden_base(data)
                .expect("the encoder's base frame is hidden and refreshes a slot");
        }
        _ => write_superframe_index(data, layer_lengths),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{ALL_SLOTS, FrameHeader};
    use crate::probs::CoefUpdates;
    use crate::segmentation::{SegmentFeatures, Segmentation};
    use crate::tables::COEFFICIENTS;

    fn sample_coded_frame(key: bool, show_frame: bool, width: usize, height: usize) -> Vec<u8> {
        let header = FrameHeader {
            key,
            show_frame,
            error_resilient: true,
            context_index: 0,
            refresh_context: false,
            width,
            height,
            ref_slots: [0; 3],
            size_from_ref: None,
            refresh: if key { ALL_SLOTS } else { 0x01 },
            filter_level: 12,
            sharpness: 0,
            interp_filter: Some(nc_dsp::Vp9Filter::Regular),
            allow_hp: true,
            base_q_idx: 80,
            segmentation: Segmentation::disabled(SegmentFeatures::default()),
            chroma_delta_q: 0,
            colour: crate::header::ColourSpace::default(),
        };
        let updates = CoefUpdates {
            sent: [false; 4],
            coef: COEFFICIENTS,
        };
        let compressed = header.write_compressed(&COEFFICIENTS, &updates);
        let mut frame = header.write(compressed.len());
        frame.extend_from_slice(&compressed);
        // Dummy tile payload ending with 0xc9 to exercise 0xc0 marker collision escaping.
        frame.extend_from_slice(&[0x12, 0x34, 0xc9]);
        frame
    }

    #[test]
    fn single_frame_escapes_trailing_superframe_marker_collision() {
        let raw = vec![0x82, 0x49, 0x83, 0x42, 0xc5];
        let packed = pack_superframe(&[&raw]);
        assert_eq!(packed.len(), raw.len() + 1);
        assert_eq!(*packed.last().unwrap(), 0x00);
        assert!(split_superframe(&packed).is_none());
    }

    #[test]
    fn two_layer_superframe_pack_split_and_strip_roundtrip() {
        for key in [true, false] {
            let s0 = sample_coded_frame(key, false, 320, 180);
            let s1 = sample_coded_frame(false, true, 640, 360);

            let info_s0 = parse_uncompressed_header_info(&s0, Some(320)).unwrap();
            assert_eq!(info_s0.key_frame, key);
            assert!(!info_s0.show_frame);

            let info_s1 = parse_uncompressed_header_info(&s1, Some(640)).unwrap();
            assert!(!info_s1.key_frame);
            assert!(info_s1.show_frame);

            let superframe = pack_superframe(&[&s0, &s1]);
            let parts = split_superframe(&superframe).expect("valid 2-frame superframe");
            assert_eq!(parts.len(), 2);
            assert_eq!(parts[0], s0.as_slice());
            assert_eq!(parts[1], s1.as_slice());

            // The hidden base and a `show_existing_frame` of the slot it
            // refreshes (slot 0 in both cases).
            let stripped = strip_superframe_to_base(&superframe).unwrap();
            let parts = split_superframe(&stripped).expect("base unit is a superframe");
            assert_eq!(parts, [s0.as_slice(), &[SHOW_EXISTING_FRAME]]);
            let shown = parse_uncompressed_header_info(parts[1], None).unwrap();
            assert!(shown.show_existing_frame);

            // A unit with the base alone packs the same way, and strips to itself.
            let mut alone = s0.clone();
            pack_spatial_tu(&mut alone, &[s0.len()]);
            assert_eq!(alone, stripped);
            assert_eq!(strip_superframe_to_base(&alone).unwrap(), stripped);
        }
    }

    #[test]
    fn shown_base_frames_strip_to_themselves() {
        let s0 = sample_coded_frame(false, true, 320, 180);
        let s1 = sample_coded_frame(false, true, 640, 360);
        let superframe = pack_superframe(&[&s0, &s1]);
        let stripped = strip_superframe_to_base(&superframe).unwrap();
        assert!(
            split_superframe(&stripped).is_none(),
            "a lone shown base must not look like a superframe even when it ends with 0xc9"
        );
        assert_eq!(stripped, escape_single_frame(&s0));
    }
}
