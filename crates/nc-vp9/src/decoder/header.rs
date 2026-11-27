//! Full VP9 frame headers (profiles 0 to 3) for the decoder: the uncompressed
//! header (VP9 §6.2, §7.2) and the compressed header (§6.3, §7.3). Field
//! order and the state carried between frames follow libvpx
//! `vp9_decodeframe.c` `read_uncompressed_header`, `setup_loopfilter`,
//! `setup_segmentation`, `setup_tile_info` and `read_compressed_header`
//! (BSD-3-Clause, see `THIRD_PARTY.md`).

use nc_common::decode::{BitReader, DecodeError, Result};
use nc_common::{BitDepth, ChromaFormat};

use super::block::{ALTREF_FRAME, GOLDEN_FRAME, LAST_FRAME, REFS_PER_FRAME, RefFrame};
use crate::bool_coder::BoolReader;
use crate::probs::FrameContext;
use crate::segmentation::{MAX_SEGMENTS, PREDICTION_PROBS, TREE_PROBS};
use crate::subexp::read_update;
use crate::transform::TxSize;

const FRAME_MARKER: u32 = 2;
const SYNC_CODE: u32 = 0x49_83_42;
/// `VPX_CS_SRGB`: 4:4:4 only, not allowed in profiles 0 and 2.
const COLOR_SPACE_SRGB: u32 = 7;
/// The profiles (VP9 §7.2): 0 is 8-bit 4:2:0, 1 8-bit 4:2:2, 4:4:0 or
/// 4:4:4, 2 and 3 the same at 10 or 12 bits.
const PROFILE_0: u32 = 0;
const PROFILE_2: u32 = 2;
const PROFILE_3: u32 = 3;
/// Profile 2 depths (`ten_or_twelve_bit`).
const TEN_BITS: u32 = 10;
const TWELVE_BITS: u32 = 12;
/// Reference slots (`REF_FRAMES`) and the bits naming one.
pub(super) const SLOTS: usize = 8;
const SLOT_BITS: u32 = 3;
/// `FRAME_CONTEXTS`.
pub(super) const FRAME_CONTEXTS: usize = 4;
/// `MAX_LOOP_FILTER`.
pub(super) const MAX_LOOP_FILTER: i32 = 63;
/// `MAXQ`.
const MAX_Q: i32 = 255;
/// Largest coded picture accepted, in luma samples: the highest VP9 level's
/// (6.2, libvpx `vp9_level_defs` `max_luma_picture_size`). VP9 signals no
/// level, so only the area is bounded: the libvpx vector
/// `vp90-2-13-largescaling` is 19200 wide, beyond level 6.2's longer side
/// (16832). The 16-bit size fields alone admit 65536x65536 pictures, whose
/// buffers and decoding time a stranger's stream must not command.
const MAX_PICTURE_SIZE: usize = 35_651_584;
/// Loop-filter reference deltas (intra, LAST, GOLDEN, ALTREF) and mode
/// deltas (ZEROMV, other inter modes).
const REF_DELTAS: usize = 4;
const MODE_DELTAS: usize = 2;
const DELTA_BITS: u32 = 6;
const Q_DELTA_BITS: u32 = 4;
/// `MAX_PROB`: a segmentation probability that is not sent.
const MAX_PROB: u8 = 255;
/// `MIN_TILE_WIDTH_B64` / `MAX_TILE_WIDTH_B64`, in superblocks.
const MIN_TILE_WIDTH_SB: usize = 4;
const MAX_TILE_WIDTH_SB: usize = 64;
/// `MV_UPDATE_PROB` and the 7 bits of an updated vector probability.
const MV_UPDATE_PROB: u8 = 252;
const MV_PROB_BITS: u32 = 7;
/// `literal_to_filter`: frame filter codes as libvpx filter types
/// (`EIGHTTAP_SMOOTH`, `EIGHTTAP`, `EIGHTTAP_SHARP`, `BILINEAR`).
const LITERAL_TO_FILTER: [u8; 4] = [1, 0, 2, 3];

/// Segment features (`SEG_LVL_*`), in bitstream order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SegFeature {
    AltQ = 0,
    AltLf = 1,
    RefFrame = 2,
    Skip = 3,
}
const SEG_FEATURES: usize = 4;
/// `vp9_seg_feature_data_max` and `vp9_is_segfeature_signed`.
const SEG_FEATURE_MAX: [u32; SEG_FEATURES] = [255, 63, 3, 0];
const SEG_FEATURE_SIGNED: [bool; SEG_FEATURES] = [true, true, false, false];

/// Loop-filter parameters; the deltas persist between frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LoopFilter {
    pub level: u8,
    pub sharpness: u8,
    /// `mode_ref_delta_enabled`.
    pub deltas_enabled: bool,
    pub ref_deltas: [i8; REF_DELTAS],
    pub mode_deltas: [i8; MODE_DELTAS],
}

impl Default for LoopFilter {
    /// `set_default_lf_deltas` (`vp9_setup_past_independence`).
    fn default() -> Self {
        Self {
            level: 0,
            sharpness: 0,
            deltas_enabled: true,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        }
    }
}

/// Segmentation parameters; the features persist between frames.
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub tree_probs: [u8; TREE_PROBS],
    pub pred_probs: [u8; PREDICTION_PROBS],
    /// `abs_delta`: feature data replaces the frame value.
    pub absolute: bool,
    pub features: [[Option<i16>; SEG_FEATURES]; MAX_SEGMENTS],
}

impl Default for Segmentation {
    fn default() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal_update: false,
            tree_probs: [MAX_PROB; TREE_PROBS],
            pred_probs: [MAX_PROB; PREDICTION_PROBS],
            absolute: false,
            features: [[None; SEG_FEATURES]; MAX_SEGMENTS],
        }
    }
}

impl Segmentation {
    /// `get_segdata` of an active feature (`segfeature_active`).
    pub(super) fn feature(&self, segment: u8, feature: SegFeature) -> Option<i16> {
        if !self.enabled {
            return None;
        }
        self.features[usize::from(segment)][feature as usize]
    }

    /// `vp9_get_qindex`.
    pub(super) fn qindex(&self, segment: u8, base: u8) -> u8 {
        match self.feature(segment, SegFeature::AltQ) {
            Some(data) => {
                let q = if self.absolute {
                    i32::from(data)
                } else {
                    i32::from(base) + i32::from(data)
                };
                q.clamp(0, MAX_Q) as u8
            }
            None => base,
        }
    }
}

/// Quantizer parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Quantizer {
    pub base: u8,
    pub y_dc_delta: i8,
    pub uv_dc_delta: i8,
    pub uv_ac_delta: i8,
}

impl Quantizer {
    /// `xd->lossless`: qindex 0 with no deltas (WHT, 4x4 only).
    pub(super) fn lossless(self) -> bool {
        self.base == 0 && self.y_dc_delta == 0 && self.uv_dc_delta == 0 && self.uv_ac_delta == 0
    }
}

/// What the parser needs from the decoder state.
#[derive(Debug, Clone, Copy)]
pub(super) struct HeaderState {
    /// Size of the picture in each slot.
    pub slot_sizes: [Option<(usize, usize)>; SLOTS],
    pub loop_filter: LoopFilter,
    pub segmentation: Segmentation,
}

impl HeaderState {
    /// Fails unless reference slot `slot` holds a picture: with no slot
    /// filled the key frame was lost, otherwise the frame that filled it.
    fn filled(&self, slot: usize) -> Result<()> {
        if self.slot_sizes[slot].is_some() {
            Ok(())
        } else if self.slot_sizes.iter().all(Option::is_none) {
            Err(DecodeError::MissingKey)
        } else {
            Err(DecodeError::MissingReference)
        }
    }
}

/// The fields of one uncompressed header.
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
#[derive(Debug, Clone, Copy)]
pub(super) struct FrameHeader {
    /// `show_existing_frame`: the slot to show; nothing else is coded.
    pub show_existing: Option<usize>,
    /// The depth and chroma format a key or intra-only frame signals
    /// (`None`: the stream's).
    pub color: Option<ColorConfig>,
    pub key: bool,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub reset_context: u8,
    pub refresh: u8,
    /// Slot of LAST, GOLDEN and ALTREF.
    pub ref_slots: [usize; REFS_PER_FRAME],
    /// `ref_frame_sign_bias`, indexed by reference frame.
    pub sign_bias: [bool; 4],
    pub width: usize,
    pub height: usize,
    pub allow_hp: bool,
    /// The frame's filter type, `None` when switchable per block.
    pub filter: Option<u8>,
    pub refresh_context: bool,
    pub parallel: bool,
    pub context_index: usize,
    pub loop_filter: LoopFilter,
    pub quantizer: Quantizer,
    pub segmentation: Segmentation,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    /// Bytes of the uncompressed header, then of the compressed header.
    pub header_bytes: usize,
    pub compressed_bytes: usize,
}

impl FrameHeader {
    /// `frame_is_intra_only`.
    pub(super) fn is_intra(&self) -> bool {
        self.key || self.intra_only
    }

    /// `vp9_setup_past_independence` applies.
    pub(super) fn resets_past(&self) -> bool {
        self.is_intra() || self.error_resilient
    }

    /// Parse the uncompressed header of `data`.
    pub(super) fn parse(data: &[u8], state: &HeaderState) -> Result<Self> {
        let mut bits = BitReader::new(data);
        let profile = read_profile(&mut bits)?;
        let mut header = Self {
            color: None,
            show_existing: None,
            key: false,
            show_frame: true,
            error_resilient: false,
            intra_only: false,
            reset_context: 0,
            refresh: 0,
            ref_slots: [0; REFS_PER_FRAME],
            sign_bias: [false; 4],
            width: 0,
            height: 0,
            allow_hp: false,
            filter: None,
            refresh_context: false,
            parallel: true,
            context_index: 0,
            loop_filter: state.loop_filter,
            quantizer: Quantizer::default(),
            segmentation: state.segmentation,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            header_bytes: 0,
            compressed_bytes: 0,
        };
        if bits.read_bit()? {
            let slot = bits.read_bits(SLOT_BITS)? as usize;
            state.filled(slot)?;
            header.show_existing = Some(slot);
            return Ok(header);
        }
        header.key = !bits.read_bit()?;
        header.show_frame = bits.read_bit()?;
        header.error_resilient = bits.read_bit()?;
        if header.key {
            read_sync_code(&mut bits)?;
            header.color = Some(read_color_config(&mut bits, profile)?);
            header.refresh = 0xff;
            (header.width, header.height) = read_coded_size(&mut bits)?;
            skip_render_size(&mut bits)?;
        } else {
            header.intra_only = !header.show_frame && bits.read_bit()?;
            if !header.error_resilient {
                header.reset_context = bits.read_bits(2)? as u8;
            }
            if header.intra_only {
                read_sync_code(&mut bits)?;
                // Profile 0 intra-only frames are 8-bit 4:2:0 by definition;
                // later profiles signal their colour configuration.
                header.color = Some(if profile == PROFILE_0 {
                    ColorConfig::PROFILE_0
                } else {
                    read_color_config(&mut bits, profile)?
                });
                header.refresh = bits.read_bits(8)? as u8;
                (header.width, header.height) = read_coded_size(&mut bits)?;
                skip_render_size(&mut bits)?;
            } else {
                header.refresh = bits.read_bits(8)? as u8;
                for reference in 0..REFS_PER_FRAME {
                    let slot = bits.read_bits(SLOT_BITS)? as usize;
                    state.filled(slot)?;
                    header.ref_slots[reference] = slot;
                    header.sign_bias[reference + 1] = bits.read_bit()?;
                }
                (header.width, header.height) = read_size_with_refs(&mut bits, &header, state)?;
                skip_render_size(&mut bits)?;
                header.allow_hp = bits.read_bit()?;
                header.filter = if bits.read_bit()? {
                    None
                } else {
                    Some(LITERAL_TO_FILTER[bits.read_bits(2)? as usize])
                };
            }
        }
        if !header.error_resilient {
            header.refresh_context = bits.read_bit()?;
            header.parallel = bits.read_bit()?;
        }
        header.context_index = bits.read_bits(2)? as usize;
        if header.resets_past() {
            // `vp9_setup_past_independence`: default deltas, no segment
            // features, no sign bias, context 0.
            header.loop_filter = LoopFilter::default();
            header.segmentation.features = [[None; SEG_FEATURES]; MAX_SEGMENTS];
            header.segmentation.absolute = false;
            header.sign_bias = [false; 4];
        }
        read_loop_filter(&mut bits, &mut header.loop_filter)?;
        header.quantizer = read_quantizer(&mut bits)?;
        read_segmentation(&mut bits, &mut header.segmentation)?;
        (header.tile_cols_log2, header.tile_rows_log2) = read_tile_info(&mut bits, header.width)?;
        header.compressed_bytes = bits.read_bits(16)? as usize;
        expect(header.compressed_bytes != 0, "VP9 empty compressed header")?;
        header.header_bytes = bits.position().div_ceil(8);
        Ok(header)
    }

    /// The frame context the frame decodes from and refreshes.
    pub(super) fn frame_context(&self) -> usize {
        if self.resets_past() {
            0
        } else {
            self.context_index
        }
    }
}

fn expect(condition: bool, what: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(DecodeError::Unsupported(what))
    }
}

fn read_sync_code(bits: &mut BitReader<'_>) -> Result<()> {
    if bits.read_bits(24)? == SYNC_CODE {
        Ok(())
    } else {
        Err(DecodeError::Malformed("VP9 sync code"))
    }
}

/// A frame's bit depth and chroma format (`color_config`, VP9 §6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ColorConfig {
    pub depth: BitDepth,
    pub format: ChromaFormat,
}

impl ColorConfig {
    /// Profile 0's only configuration (also of its intra-only frames).
    pub(super) const PROFILE_0: Self = Self {
        depth: BitDepth::EIGHT,
        format: ChromaFormat::Yuv420,
    };
}

/// `frame_marker` and the profile (`vp9_read_profile`, VP9 §6.2): profile
/// 3 is followed by a reserved zero bit.
fn read_profile(bits: &mut BitReader<'_>) -> Result<u32> {
    expect(bits.read_bits(2)? == FRAME_MARKER, "VP9 frame marker")?;
    let low = bits.read_bits(1)?;
    let profile = (bits.read_bits(1)? << 1) | low;
    if profile == PROFILE_3 {
        expect(!bits.read_bit()?, "VP9 reserved profile")?;
    }
    Ok(profile)
}

/// Bytes of the `show_existing_frame` header that starts with
/// `first_byte` (libvpx `vp9_decode_frame`): profile 3's reserved bit
/// makes it nine bits.
pub(super) fn show_existing_bytes(first_byte: u8) -> usize {
    // `frame_marker` (2 bits), then `profile_low_bit`, `profile_high_bit`.
    let low = u32::from((first_byte >> 5) & 1);
    let high = u32::from((first_byte >> 4) & 1);
    if (high << 1) | low == PROFILE_3 { 2 } else { 1 }
}

/// `read_bitdepth_colorspace_sampling` (VP9 §6.2.2): the bit depth (8 in
/// profiles 0 and 1, 10 or 12 in 2 and 3) and the chroma subsampling
/// (4:2:0 in profiles 0 and 2; signalled in 1 and 3, where 4:2:0 is not
/// allowed and sRGB means 4:4:4).
fn read_color_config(bits: &mut BitReader<'_>, profile: u32) -> Result<ColorConfig> {
    let depth = if profile >= PROFILE_2 {
        if bits.read_bit()? {
            BitDepth::new(TWELVE_BITS)
        } else {
            BitDepth::new(TEN_BITS)
        }
        .expect("10 and 12 are depths")
    } else {
        BitDepth::EIGHT
    };
    let subsampled_profile = profile == PROFILE_0 || profile == PROFILE_2;
    let color_space = bits.read_bits(3)?;
    let format = if color_space == COLOR_SPACE_SRGB {
        expect(!subsampled_profile, "VP9 sRGB in profile 0 or 2")?;
        expect(!bits.read_bit()?, "VP9 reserved bit")?;
        ChromaFormat::Yuv444
    } else {
        bits.read_bit()?; // color_range: output metadata only
        if subsampled_profile {
            ChromaFormat::Yuv420
        } else {
            let x = bits.read_bits(1)? as usize;
            let y = bits.read_bits(1)? as usize;
            expect((x, y) != (1, 1), "VP9 4:2:0 in profile 1 or 3")?;
            expect(!bits.read_bit()?, "VP9 reserved bit")?;
            ChromaFormat::from_subsampling(x, y).expect("one-bit subsampling")
        }
    };
    Ok(ColorConfig { depth, format })
}

/// The bit depth a frame signals: that of a key frame or an intra-only
/// frame, `None` for an inter frame (which keeps the stream's) and a
/// `show_existing_frame` header, or for data that is not a frame header.
pub(super) fn signalled_depth(data: &[u8]) -> Option<BitDepth> {
    let mut bits = BitReader::new(data);
    let profile = read_profile(&mut bits).ok()?;
    if bits.read_bit().ok()? {
        return None;
    }
    let key = !bits.read_bit().ok()?;
    let show_frame = bits.read_bit().ok()?;
    let error_resilient = bits.read_bit().ok()?;
    if key {
        read_sync_code(&mut bits).ok()?;
    } else {
        let intra_only = !show_frame && bits.read_bit().ok()?;
        if !intra_only {
            return None;
        }
        if !error_resilient {
            bits.read_bits(2).ok()?;
        }
        read_sync_code(&mut bits).ok()?;
        if profile == PROFILE_0 {
            return Some(BitDepth::EIGHT);
        }
    }
    read_color_config(&mut bits, profile).ok().map(|c| c.depth)
}

fn read_frame_size(bits: &mut BitReader<'_>) -> Result<(usize, usize)> {
    let width = bits.read_bits(16)? as usize + 1;
    let height = bits.read_bits(16)? as usize + 1;
    Ok((width, height))
}

/// `frame_size` of the picture to decode, within the level 6.2 area
/// ([`MAX_PICTURE_SIZE`]).
fn read_coded_size(bits: &mut BitReader<'_>) -> Result<(usize, usize)> {
    let (width, height) = read_frame_size(bits)?;
    if width * height > MAX_PICTURE_SIZE {
        return Err(DecodeError::Unsupported("VP9 frame size beyond level 6.2"));
    }
    Ok((width, height))
}

/// `setup_render_size`: the render size is display metadata only.
fn skip_render_size(bits: &mut BitReader<'_>) -> Result<()> {
    if bits.read_bit()? {
        read_frame_size(bits)?;
    }
    Ok(())
}

/// `valid_ref_frame_size`: a reference at most 2x larger and 16x smaller.
pub(super) fn valid_reference_size((rw, rh): (usize, usize), (w, h): (usize, usize)) -> bool {
    2 * w >= rw && 2 * h >= rh && w <= 16 * rw && h <= 16 * rh
}

/// `setup_frame_size_with_refs`.
fn read_size_with_refs(
    bits: &mut BitReader<'_>,
    header: &FrameHeader,
    state: &HeaderState,
) -> Result<(usize, usize)> {
    let mut size = None;
    for &slot in &header.ref_slots {
        if bits.read_bit()? {
            size = state.slot_sizes[slot];
            break;
        }
    }
    let size = match size {
        Some(size) => size,
        None => read_coded_size(bits)?,
    };
    let valid = header
        .ref_slots
        .iter()
        .filter_map(|&slot| state.slot_sizes[slot])
        .any(|reference| valid_reference_size(reference, size));
    if !valid {
        return Err(DecodeError::Malformed("VP9 references of invalid size"));
    }
    Ok(size)
}

fn read_signed(bits: &mut BitReader<'_>, magnitude_bits: u32) -> Result<i32> {
    let magnitude = bits.read_bits(magnitude_bits)? as i32;
    Ok(if bits.read_bit()? {
        -magnitude
    } else {
        magnitude
    })
}

/// `setup_loopfilter`: deltas update only where flagged.
fn read_loop_filter(bits: &mut BitReader<'_>, lf: &mut LoopFilter) -> Result<()> {
    lf.level = bits.read_bits(6)? as u8;
    lf.sharpness = bits.read_bits(3)? as u8;
    lf.deltas_enabled = bits.read_bit()?;
    if lf.deltas_enabled && bits.read_bit()? {
        for delta in lf.ref_deltas.iter_mut().chain(&mut lf.mode_deltas) {
            if bits.read_bit()? {
                *delta = read_signed(bits, DELTA_BITS)? as i8;
            }
        }
    }
    Ok(())
}

/// `setup_quantization`.
fn read_quantizer(bits: &mut BitReader<'_>) -> Result<Quantizer> {
    let base = bits.read_bits(8)? as u8;
    let mut delta = || -> Result<i8> {
        Ok(if bits.read_bit()? {
            read_signed(bits, Q_DELTA_BITS)? as i8
        } else {
            0
        })
    };
    Ok(Quantizer {
        base,
        y_dc_delta: delta()?,
        uv_dc_delta: delta()?,
        uv_ac_delta: delta()?,
    })
}

fn read_prob(bits: &mut BitReader<'_>) -> Result<u8> {
    Ok(if bits.read_bit()? {
        bits.read_bits(8)? as u8
    } else {
        MAX_PROB
    })
}

/// `setup_segmentation`.
fn read_segmentation(bits: &mut BitReader<'_>, seg: &mut Segmentation) -> Result<()> {
    seg.update_map = false;
    seg.enabled = bits.read_bit()?;
    if !seg.enabled {
        return Ok(());
    }
    seg.update_map = bits.read_bit()?;
    if seg.update_map {
        for prob in &mut seg.tree_probs {
            *prob = read_prob(bits)?;
        }
        seg.temporal_update = bits.read_bit()?;
        for prob in &mut seg.pred_probs {
            *prob = if seg.temporal_update {
                read_prob(bits)?
            } else {
                MAX_PROB
            };
        }
    }
    if bits.read_bit()? {
        seg.absolute = bits.read_bit()?;
        for segment in &mut seg.features {
            for (feature, data) in segment.iter_mut().enumerate() {
                *data = None;
                if bits.read_bit()? {
                    // `decode_unsigned_max`: `get_unsigned_bits(max)` bits.
                    let max = SEG_FEATURE_MAX[feature];
                    let width = u32::BITS - max.leading_zeros();
                    let magnitude = bits.read_bits(width)?.min(max) as i16;
                    let negative = SEG_FEATURE_SIGNED[feature] && bits.read_bit()?;
                    *data = Some(if negative { -magnitude } else { magnitude });
                }
            }
        }
    }
    Ok(())
}

/// `setup_tile_info` (`vp9_get_tile_n_bits`).
fn read_tile_info(bits: &mut BitReader<'_>, width: usize) -> Result<(u32, u32)> {
    let sb_cols = width.div_ceil(64);
    let mut min_log2 = 0;
    while (MAX_TILE_WIDTH_SB << min_log2) < sb_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1;
    while (sb_cols >> max_log2) >= MIN_TILE_WIDTH_SB {
        max_log2 += 1;
    }
    max_log2 -= 1;
    let mut cols_log2 = min_log2;
    while cols_log2 < max_log2 && bits.read_bit()? {
        cols_log2 += 1;
    }
    let mut rows_log2 = u32::from(bits.read_bit()?);
    if rows_log2 != 0 {
        rows_log2 += u32::from(bits.read_bit()?);
    }
    Ok((cols_log2, rows_log2))
}

/// `TX_MODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TxMode {
    Only4x4,
    Allow8x8,
    Allow16x16,
    Allow32x32,
    Select,
}

impl TxMode {
    /// `tx_mode_to_biggest_tx_size`.
    pub(super) fn largest(self) -> TxSize {
        match self {
            Self::Only4x4 => TxSize::Tx4,
            Self::Allow8x8 => TxSize::Tx8,
            Self::Allow16x16 => TxSize::Tx16,
            Self::Allow32x32 | Self::Select => TxSize::Tx32,
        }
    }
}

/// `REFERENCE_MODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReferenceMode {
    Single,
    Compound,
    Select,
}

/// What the compressed header decides beyond probabilities.
#[derive(Debug, Clone, Copy)]
pub(super) struct CompressedHeader {
    pub tx_mode: TxMode,
    pub reference_mode: ReferenceMode,
    /// `comp_fixed_ref` and `comp_var_ref` (`vp9_setup_compound_reference_mode`).
    pub comp_fixed_ref: RefFrame,
    pub comp_var_ref: [RefFrame; 2],
}

fn update_all(reader: &mut BoolReader<'_>, probs: &mut [u8]) {
    for prob in probs {
        read_update(reader, prob);
    }
}

/// `update_mv_probs`.
fn update_mv(reader: &mut BoolReader<'_>, probs: &mut [u8]) {
    for prob in probs {
        if reader.read(MV_UPDATE_PROB) {
            *prob = ((reader.read_literal(MV_PROB_BITS) << 1) | 1) as u8;
        }
    }
}

/// Parse the compressed header, applying its forward updates to `fc`
/// (`read_compressed_header`).
pub(super) fn read_compressed(
    data: &[u8],
    header: &FrameHeader,
    fc: &mut FrameContext,
) -> Result<CompressedHeader> {
    let mut reader = BoolReader::new(data)?;
    let tx_mode = if header.quantizer.lossless() {
        TxMode::Only4x4
    } else {
        let mut mode = reader.read_literal(2);
        if mode == TxMode::Allow32x32 as u32 {
            mode += reader.read_literal(1);
        }
        [
            TxMode::Only4x4,
            TxMode::Allow8x8,
            TxMode::Allow16x16,
            TxMode::Allow32x32,
            TxMode::Select,
        ][mode as usize]
    };
    if tx_mode == TxMode::Select {
        for probs in &mut fc.tx8 {
            update_all(&mut reader, probs);
        }
        for probs in &mut fc.tx16 {
            update_all(&mut reader, probs);
        }
        for probs in &mut fc.tx32 {
            update_all(&mut reader, probs);
        }
    }
    // `read_coef_probs`: up to the largest transform the mode allows.
    for table in fc.coef.iter_mut().take(tx_mode.largest() as usize + 1) {
        if reader.read_literal(1) != 0 {
            for planes in table.iter_mut() {
                for references in planes.iter_mut() {
                    for (band, contexts) in references.iter_mut().enumerate() {
                        let count = if band == 0 { 3 } else { 6 };
                        for probs in contexts.iter_mut().take(count) {
                            update_all(&mut reader, probs);
                        }
                    }
                }
            }
        }
    }
    update_all(&mut reader, &mut fc.skip);
    let mut compressed = CompressedHeader {
        tx_mode,
        reference_mode: ReferenceMode::Single,
        comp_fixed_ref: ALTREF_FRAME,
        comp_var_ref: [LAST_FRAME, GOLDEN_FRAME],
    };
    if header.is_intra() {
        reader.finish()?;
        return Ok(compressed);
    }
    for probs in &mut fc.inter_mode {
        update_all(&mut reader, probs);
    }
    if header.filter.is_none() {
        for probs in &mut fc.switchable_interp {
            update_all(&mut reader, probs);
        }
    }
    update_all(&mut reader, &mut fc.intra_inter);
    compressed.reference_mode = read_reference_mode(&mut reader, header);
    if compressed.reference_mode != ReferenceMode::Single {
        (compressed.comp_fixed_ref, compressed.comp_var_ref) = compound_references(header);
    }
    if compressed.reference_mode == ReferenceMode::Select {
        update_all(&mut reader, &mut fc.comp_inter);
    }
    if compressed.reference_mode != ReferenceMode::Compound {
        for probs in &mut fc.single_ref {
            update_all(&mut reader, probs);
        }
    }
    if compressed.reference_mode != ReferenceMode::Single {
        update_all(&mut reader, &mut fc.comp_ref);
    }
    for probs in &mut fc.y_mode {
        update_all(&mut reader, probs);
    }
    for probs in &mut fc.partition {
        update_all(&mut reader, probs);
    }
    read_mv_probs(&mut reader, fc, header.allow_hp);
    reader.finish()?;
    Ok(compressed)
}

/// `read_frame_reference_mode`: compound prediction needs references
/// of both sign biases (`vp9_compound_reference_allowed`).
fn read_reference_mode(reader: &mut BoolReader<'_>, header: &FrameHeader) -> ReferenceMode {
    let bias = &header.sign_bias;
    let allowed = bias[GOLDEN_FRAME as usize] != bias[LAST_FRAME as usize]
        || bias[ALTREF_FRAME as usize] != bias[LAST_FRAME as usize];
    if !allowed || reader.read_literal(1) == 0 {
        return ReferenceMode::Single;
    }
    if reader.read_literal(1) != 0 {
        ReferenceMode::Select
    } else {
        ReferenceMode::Compound
    }
}

/// `vp9_setup_compound_reference_mode`: the reference whose sign bias
/// differs is fixed; the other two vary.
fn compound_references(header: &FrameHeader) -> (RefFrame, [RefFrame; 2]) {
    let bias = |frame: RefFrame| header.sign_bias[frame as usize];
    if bias(LAST_FRAME) == bias(GOLDEN_FRAME) {
        (ALTREF_FRAME, [LAST_FRAME, GOLDEN_FRAME])
    } else if bias(LAST_FRAME) == bias(ALTREF_FRAME) {
        (GOLDEN_FRAME, [LAST_FRAME, ALTREF_FRAME])
    } else {
        (LAST_FRAME, [GOLDEN_FRAME, ALTREF_FRAME])
    }
}

/// `read_mv_probs`.
fn read_mv_probs(reader: &mut BoolReader<'_>, fc: &mut FrameContext, allow_hp: bool) {
    update_mv(reader, &mut fc.mv_joints);
    for component in &mut fc.mv {
        update_mv(reader, std::slice::from_mut(&mut component.sign));
        update_mv(reader, &mut component.classes);
        update_mv(reader, &mut component.class0);
        update_mv(reader, &mut component.bits);
    }
    for component in &mut fc.mv {
        for probs in &mut component.class0_fp {
            update_mv(reader, probs);
        }
        update_mv(reader, &mut component.fp);
    }
    if allow_hp {
        for component in &mut fc.mv {
            update_mv(reader, std::slice::from_mut(&mut component.class0_hp));
            update_mv(reader, std::slice::from_mut(&mut component.hp));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probs::CoefUpdates;
    use crate::segmentation::{SegmentFeatures, Segmentation as WrittenSegmentation};

    /// A key frame header the encoder writes, declaring `width` x `height`.
    fn key_header(width: usize, height: usize) -> Vec<u8> {
        let written = crate::header::FrameHeader {
            key: true,
            error_resilient: false,
            context_index: 0,
            refresh_context: true,
            width,
            height,
            ref_slots: [0; 3],
            size_from_ref: None,
            refresh: 0xff,
            filter_level: 0,
            sharpness: 0,
            interp_filter: None,
            allow_hp: false,
            base_q_idx: 99,
            segmentation: WrittenSegmentation::disabled(SegmentFeatures::default()),
            chroma_delta_q: 0,
            colour: crate::header::ColourSpace::default(),
        };
        written.write(1)
    }

    /// Pictures within level 6.2's area pass the size check, whatever their
    /// shape (they may fail later header fields this one-tile writer does not
    /// set); larger ones fail it.
    #[test]
    fn picture_size_is_capped_at_level_6_2() {
        let state = HeaderState {
            slot_sizes: [None; SLOTS],
            loop_filter: LoopFilter::default(),
            segmentation: Segmentation::default(),
        };
        let too_large = DecodeError::Unsupported("VP9 frame size beyond level 6.2");
        for (width, height) in [(8192, 4352), (19200, 108), (65536, 544), (544, 65536)] {
            let parsed = FrameHeader::parse(&key_header(width, height), &state);
            assert_ne!(parsed.err(), Some(too_large.clone()), "{width}x{height}");
        }
        for (width, height) in [(8192, 4353), (65536, 545), (4081, 53478), (2896, 65536)] {
            let parsed = FrameHeader::parse(&key_header(width, height), &state);
            assert_eq!(parsed.err(), Some(too_large.clone()), "{width}x{height}");
        }
    }

    /// Headers the encoder writes parse back to the same fields: key and
    /// error-resilient inter frames, and (`layered`) a spatial layer's
    /// inter frame that is not error resilient, reads context 1 without
    /// refreshing it and takes its size from its second reference.
    #[test]
    fn encoder_headers_round_trip() {
        for (key, layered) in [(true, false), (false, false), (false, true)] {
            for (width, height) in [(64, 48), (640, 360), (1920, 1080)] {
                let written = crate::header::FrameHeader {
                    key,
                    error_resilient: !key && !layered,
                    context_index: u8::from(layered),
                    refresh_context: !layered,
                    width,
                    height,
                    ref_slots: if key { [0; 3] } else { [3, 2, 3] },
                    size_from_ref: Some(usize::from(layered)),
                    refresh: if key { 0xff } else { 0b100 },
                    filter_level: 17,
                    sharpness: if key { 0 } else { 5 },
                    interp_filter: Some(if key {
                        nc_dsp::Vp9Filter::Regular
                    } else {
                        nc_dsp::Vp9Filter::Sharp
                    }),
                    allow_hp: !key,
                    base_q_idx: 99,
                    segmentation: WrittenSegmentation::disabled(SegmentFeatures::default()),
                    chroma_delta_q: if key { -7 } else { 3 },
                    colour: crate::header::ColourSpace::default(),
                };
                let coef = crate::tables::COEFFICIENTS;
                let compressed = written.write_compressed(
                    &coef,
                    &CoefUpdates {
                        coef,
                        sent: [false; 4],
                    },
                );
                let mut packet = written.write(compressed.len());
                let uncompressed = packet.len();
                packet.extend(&compressed);
                let mut slot_sizes = [None; SLOTS];
                slot_sizes[2] = Some((width, height));
                slot_sizes[3] = Some((if layered { width / 2 } else { width }, height));
                let state = HeaderState {
                    slot_sizes,
                    loop_filter: LoopFilter::default(),
                    segmentation: Segmentation::default(),
                };
                let parsed = FrameHeader::parse(&packet, &state).unwrap();
                assert_eq!(parsed.key, key);
                assert_eq!(parsed.error_resilient, written.error_resilient);
                if !written.error_resilient {
                    assert_eq!(parsed.context_index, usize::from(written.context_index));
                    assert_eq!(parsed.refresh_context, written.refresh_context);
                    assert_eq!(parsed.parallel, !written.refresh_context);
                }
                assert_eq!((parsed.width, parsed.height), (width, height));
                assert_eq!(parsed.refresh, written.refresh);
                if !key {
                    assert_eq!(parsed.ref_slots, [3, 2, 3]);
                    // `EIGHTTAP_SHARP`.
                    assert_eq!(parsed.filter, Some(2));
                    assert!(parsed.allow_hp);
                }
                assert_eq!(
                    (parsed.loop_filter.level, parsed.loop_filter.sharpness),
                    (17, written.sharpness)
                );
                assert!(!parsed.loop_filter.deltas_enabled);
                assert_eq!(parsed.quantizer.base, 99);
                let delta = written.chroma_delta_q;
                assert_eq!(
                    (parsed.quantizer.uv_dc_delta, parsed.quantizer.uv_ac_delta),
                    (delta, delta)
                );
                assert_eq!((parsed.tile_cols_log2, parsed.tile_rows_log2), (0, 0));
                assert_eq!(parsed.header_bytes, uncompressed);
                assert_eq!(parsed.compressed_bytes, compressed.len());
                let mut fc = FrameContext::default();
                let modes = read_compressed(&compressed, &parsed, &mut fc).unwrap();
                assert_eq!(modes.tx_mode, TxMode::Select);
                assert_eq!(modes.reference_mode, ReferenceMode::Single);
                assert_eq!(fc, FrameContext::default());
            }
        }
    }

    #[test]
    fn tile_columns_follow_the_superblock_count() {
        // 1920 wide: 30 superblocks allow up to 4 tile columns (log2 2).
        let data = [0b1100_0000];
        let mut bits = BitReader::new(&data);
        assert_eq!(read_tile_info(&mut bits, 1920).unwrap(), (2, 0));
        // 256 wide: no tile-column bit; one tile-row bit pair.
        let data = [0b1100_0000];
        let mut bits = BitReader::new(&data);
        assert_eq!(read_tile_info(&mut bits, 256).unwrap(), (0, 2));
    }

    #[test]
    fn segment_qindex_applies_absolute_and_delta_data() {
        let mut seg = Segmentation {
            enabled: true,
            ..Segmentation::default()
        };
        seg.features[1][SegFeature::AltQ as usize] = Some(-120);
        assert_eq!(seg.qindex(1, 100), 0);
        assert_eq!(seg.qindex(0, 100), 100);
        seg.absolute = true;
        seg.features[2][SegFeature::AltQ as usize] = Some(40);
        assert_eq!(seg.qindex(2, 100), 40);
        seg.enabled = false;
        assert_eq!(seg.qindex(2, 100), 100);
    }
}
