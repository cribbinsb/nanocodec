//! Final AV2 v1.0.0 sequence-header payloads for the native basic subset.
//!
//! Syntax follows `sequence_header_obu` and the grouped tool syntax in AVM
//! v1.0.0 `av2/encoder/bitstream.c` and `av2/decoder/{obu,decodeframe}.c`.
//! Reference license: BSD-3-Clause-Clear plus separate AOM patent terms (see
//! `docs/licenses/avm`). This module codes sequence configuration, not pictures.

use crate::general::sequence::SegmentFeatures;
use crate::general::tiles::{self, TileGeometry};
use crate::mv_refs::ListConfig;
use nc_common::BitWriter;
use nc_common::decode::{BitReader, DecodeError, Result};

/// Sequence configuration for 8-bit 4:2:0, Main 4:2:0 IP0, one layer.
///
/// The emitted tool contract uses 64x64 superblocks, shared luma/chroma
/// partitions, simple translational motion, reduced transform partitions,
/// chroma DCT only, and equal AC/DC quantization. Optional enhancement tools
/// are disabled. Frame/tile syntax must still implement the normative baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct SequenceHeader {
    /// Sequence header identifier, 0..15.
    pub id: u8,
    /// Maximum coded picture width, 1..65536.
    pub max_width: u32,
    /// Maximum coded picture height, 1..65536.
    pub max_height: u32,
    /// AV2 level index: 0..21 or 31 (maximum level).
    /// Choosing a level does not itself prove level conformance of pictures.
    pub level: u8,
    /// A single-picture sequence omits inter-picture configuration.
    pub single_picture: bool,
    /// Order-hint width, 1..16 for video; must be zero for single pictures.
    pub order_hint_bits: u8,
    /// Luma samples cropped from the right and bottom of each decoded picture.
    pub crop: CropWindow,
    /// Temporal layers and reference signalling.
    pub layering: Layering,
    /// Frames code their chroma quantizer delta (AVM `uv_ac_delta_q_enabled`;
    /// with equal AC/DC steps the delta moves both).
    pub chroma_delta_q: bool,
    /// Frames code luma and chroma DC quantizer deltas (AVM `equal_ac_dc_q =
    /// 0`, zero base DC deltas, `y_dc_delta_q_enabled` and
    /// `uv_dc_delta_q_enabled`).
    pub dc_delta_q: bool,
    /// Motion-vector reference lists: DRL size (`def_max_drl_bits`, no
    /// per-frame override) and the reference MV bank (`enable_refmvbank`).
    /// Single pictures have no inter frames and keep
    /// [`ListConfig::BASIC`].
    pub reference_lists: ListConfig,
    /// Frames may hide luma DC parities (`enable_parity_hiding`; each frame
    /// signals `allow_parity_hiding`).
    pub parity_hiding: bool,
    /// Which lossy frames use trellis-coded quantisation (`enable_tcq`).
    pub tcq: TcqFrames,
    /// Lossy frames may filter with CCSO (`enable_ccso`; units of
    /// [`crate::ccso::UNIT`], not matched to superblocks).
    pub ccso: bool,
    /// Fields our encoder writes one value of and other encoders vary.
    pub variants: Variants,
}

/// Sequence fields our encoder always writes [`Variants::OURS`] for; the
/// decoder accepts the values of other encoders listed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variants {
    /// Bits of a frame's deblocking offsets (`df_par_bits_minus2 + 2`),
    /// 2..=5.
    pub deblock_delta_bits: u8,
    /// Whether CDEF filters blocks whose transforms are all skipped
    /// (`cdef_on_skip_txfm`); `None` when each frame says. Single pictures
    /// do not code it (each frame says): see
    /// [`SequenceHeader::cdef_on_skipped`].
    pub cdef_on_skipped: Option<bool>,
    /// Compound pairs of one reference frames may list
    /// (`num_same_ref_compound`, 0..=3; video sequences).
    pub same_reference_compounds: u8,
    /// Whether frames allow screen-content tools
    /// (`force_screen_content_tools`) and, with them, force whole-sample
    /// vectors (`force_integer_mv`); each `None` when every frame says, as
    /// in single pictures (which do not code them).
    pub screen_content: Option<bool>,
    /// See `screen_content`.
    pub integer_mv: Option<bool>,
    /// Frames are output in coding order (`frame_output_order_flag`
    /// monotonic); other encoders reorder (hidden frames shown later).
    pub monotonic_output: bool,
    /// The sequence carries a tile layout (`seq_tile_info_present_flag`,
    /// of one tile: others are rejected); frames reuse it.
    pub tiles: Option<SequenceTiles>,
}

/// How frames use the sequence's (one-tile) layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceTiles {
    /// Every frame uses it and codes no layout.
    Fixed,
    /// Each frame says whether it does (`allow_tile_info_change`).
    PerFrame,
}

impl Variants {
    /// What our encoder writes (single pictures infer `cdef_on_skipped`
    /// per frame whatever this says).
    pub const OURS: Self = Self {
        deblock_delta_bits: 2,
        cdef_on_skipped: Some(false),
        same_reference_compounds: 0,
        screen_content: Some(false),
        integer_mv: None,
        monotonic_output: true,
        tiles: None,
    };
}

/// Which lossy frames code their luma with trellis-coded quantisation
/// (AVM `enable_tcq`: `TCQ_DISABLE`, `TCQ_8ST`, `TCQ_8ST_FR`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TcqFrames {
    /// None.
    #[default]
    Off,
    /// Every lossy frame (the only choice of single pictures).
    Every,
    /// Each lossy frame says (`tcq_mode`).
    Signalled,
}

/// The sequence's temporal layers and how frames name their reference.
///
/// A flat sequence (one layer, implicit references) refreshes every slot
/// with each picture, so AVM's implicit reference ranking
/// (`av2_get_ref_frames`) finds one distinct reference. Layered or
/// long-term structures keep several pictures in the slots; their frames
/// then signal their one reference slot explicitly
/// (`enable_explicit_ref_frame_map`, `num_total_refs = 1`), which keeps
/// the tile syntax single-reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layering {
    /// Temporal layers, 1..=4 (`max_tlayer_id + 1`); each layer may
    /// reference only itself and lower layers (the default dependency map).
    pub temporal_layers: u8,
    /// Frames signal their reference slot (`enable_explicit_ref_frame_map`).
    pub explicit_references: bool,
}

impl Layering {
    /// One layer, implicit references.
    pub const FLAT: Self = Self {
        temporal_layers: 1,
        explicit_references: false,
    };
}

/// Bits of `max_tlayer_id` (`TLAYER_BITS`).
const TEMPORAL_LAYER_BITS: u32 = 2;

/// The conformance window of AVM `av2_write_conformance_window`, in luma
/// samples, restricted to the right and bottom edges: frame sizes are whole
/// 8x8 blocks, so a picture of any even size is coded padded and cropped back.
/// Offsets are even, which keeps the window on 4:2:0 chroma samples
/// (`av2_validate_frame_level_conformance`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CropWindow {
    /// Columns removed at the right edge.
    pub right: u32,
    /// Rows removed at the bottom edge.
    pub bottom: u32,
}

impl CropWindow {
    /// No cropping: `conf_win_enabled_flag` is zero.
    pub const NONE: Self = Self {
        right: 0,
        bottom: 0,
    };

    fn is_enabled(self) -> bool {
        self != Self::NONE
    }
}

/// 4:2:0 subsampling factor a crop offset must be a multiple of.
const CHROMA_SUBSAMPLING: u32 = 2;

impl SequenceHeader {
    /// Whether CDEF filters blocks whose transforms are all skipped;
    /// `None` when each frame says (always so in single pictures).
    #[must_use]
    pub fn cdef_on_skipped(self) -> Option<bool> {
        if self.single_picture {
            None
        } else {
            self.variants.cdef_on_skipped
        }
    }

    /// Whether frames allow screen-content tools and force whole-sample
    /// vectors; each `None` when every frame says (always so in single
    /// pictures).
    #[must_use]
    pub fn screen_content(self) -> (Option<bool>, Option<bool>) {
        if self.single_picture {
            (None, None)
        } else {
            (self.variants.screen_content, self.variants.integer_mv)
        }
    }

    /// The decoded picture size after the conformance window.
    #[must_use]
    pub fn visible_size(self) -> (usize, usize) {
        (
            (self.max_width - self.crop.right) as usize,
            (self.max_height - self.crop.bottom) as usize,
        )
    }

    pub(crate) fn validate(self) -> Result<()> {
        // av2_validate_seq_conformance_window: offsets inside the frame.
        if self.crop.right >= self.max_width
            || self.crop.bottom >= self.max_height
            || !self.crop.right.is_multiple_of(CHROMA_SUBSAMPLING)
            || !self.crop.bottom.is_multiple_of(CHROMA_SUBSAMPLING)
        {
            return Err(DecodeError::Malformed(
                "AV2 conformance window out of range",
            ));
        }
        if self.id > 15
            || !(1..=65536).contains(&self.max_width)
            || !(1..=65536).contains(&self.max_height)
            || !(self.level <= 21 || self.level == 31)
            || if self.single_picture {
                self.order_hint_bits != 0 || self.reference_lists != ListConfig::BASIC
            } else {
                !(1..=16).contains(&self.order_hint_bits)
            }
        {
            return Err(DecodeError::Malformed(
                "AV2 basic sequence field out of range",
            ));
        }
        Ok(())
    }
}

/// Encode a complete sequence-header payload, including trailing bits.
/// # Errors
/// Rejects invalid dimensions, identifiers, levels and order-hint widths.
pub fn write(header: &SequenceHeader) -> Result<Vec<u8>> {
    header.validate()?;
    let mut syntax = Syntax::Write(BitWriter::new());
    sequence(&mut syntax, *header)?;
    let Syntax::Write(mut writer) = syntax else {
        unreachable!()
    };
    writer.trailing_bits();
    Ok(writer.into_bytes())
}

/// Parse the emitted basic sequence subset without allocating picture storage.
/// # Errors
/// Rejects truncation, unsupported tools, invalid fields and trailing garbage.
pub fn parse(payload: &[u8]) -> Result<SequenceHeader> {
    let mut syntax = Syntax::Read(BitReader::new(payload));
    let header = sequence(
        &mut syntax,
        SequenceHeader {
            id: 0,
            max_width: 1,
            max_height: 1,
            level: 0,
            single_picture: true,
            order_hint_bits: 0,
            crop: CropWindow::NONE,
            layering: Layering::FLAT,
            chroma_delta_q: false,
            dc_delta_q: false,
            parity_hiding: false,
            tcq: TcqFrames::Off,
            ccso: false,
            variants: Variants::OURS,
            reference_lists: ListConfig::BASIC,
        },
    )?;
    let Syntax::Read(mut reader) = syntax else {
        unreachable!()
    };
    if !reader.read_bit()? {
        return Err(DecodeError::Malformed("AV2 sequence trailing one bit"));
    }
    while !reader.is_byte_aligned() {
        if reader.read_bit()? {
            return Err(DecodeError::Malformed("AV2 sequence trailing zero bits"));
        }
    }
    if reader.remaining_bits() != 0 {
        return Err(DecodeError::Malformed("AV2 sequence trailing bytes"));
    }
    header.validate()?;
    Ok(header)
}

enum Syntax<'a> {
    Write(BitWriter),
    Read(BitReader<'a>),
}

impl<'a> Syntax<'a> {
    /// One flag whose value the encoder writes as `ours`; `Some(reader)`
    /// when a stream being read sets it otherwise, to read what follows
    /// (syntax our encoder never writes).
    fn other_encoders(&mut self, ours: bool) -> Result<Option<&mut BitReader<'a>>> {
        match self {
            Self::Write(writer) => {
                writer.put_bit(ours);
                Ok(None)
            }
            Self::Read(reader) => Ok((reader.read_bit()? != ours).then_some(reader)),
        }
    }
}

impl Syntax<'_> {
    fn field(&mut self, value: u32, bits: u32) -> Result<u32> {
        match self {
            Self::Write(writer) => {
                writer.put_bits(value, bits);
                Ok(value)
            }
            Self::Read(reader) => reader.read_bits(bits),
        }
    }

    /// AVM `avm_wb_write_primitive_quniform` / `avm_rb_read_primitive_quniform`:
    /// `value` in 0..`count`, the low values one bit shorter.
    fn quniform(&mut self, value: u32, count: u32) -> Result<u32> {
        let bits = count.ilog2() + 1;
        let short = (1 << bits) - count;
        match self {
            Self::Write(writer) => {
                if value < short {
                    writer.put_bits(value, bits - 1);
                } else {
                    writer.put_bits(short + ((value - short) >> 1), bits - 1);
                    writer.put_bits((value - short) & 1, 1);
                }
                Ok(value)
            }
            Self::Read(reader) => {
                let prefix = reader.read_bits(bits - 1)?;
                Ok(if prefix < short {
                    prefix
                } else {
                    (prefix << 1) - short + reader.read_bits(1)?
                })
            }
        }
    }

    fn uvlc(&mut self, value: u32) -> Result<u32> {
        match self {
            Self::Write(writer) => {
                writer.put_uvlc(value);
                Ok(value)
            }
            Self::Read(reader) => reader.read_uvlc(),
        }
    }

    fn fixed(&mut self, value: u32, bits: u32, name: &'static str) -> Result<()> {
        if self.field(value, bits)? != value {
            return Err(DecodeError::Unsupported(name));
        }
        Ok(())
    }

    fn off(&mut self, name: &'static str) -> Result<()> {
        self.fixed(0, 1, name)
    }
}

fn sequence(s: &mut Syntax<'_>, h: SequenceHeader) -> Result<SequenceHeader> {
    let id = s.uvlc(u32::from(h.id))?;
    if id > 15 {
        return Err(DecodeError::Malformed("AV2 sequence id exceeds 15"));
    }
    s.fixed(0, 5, "AV2 profile other than Main 420 IP0")?;
    let single = s.field(u32::from(h.single_picture), 1)? != 0;
    let level = s.field(u32::from(h.level), 5)? as u8;
    if level >= 4 && !single {
        s.off("AV2 high tier")?;
    }
    if s.uvlc(0)? != 0 {
        return Err(DecodeError::Unsupported("AV2 chroma format other than 420"));
    }
    if s.uvlc(1)? != 1 {
        return Err(DecodeError::Unsupported("AV2 bit depth other than eight"));
    }
    let mut temporal_layers = 1;
    let mut monotonic_output = true;
    if !single {
        s.fixed(0, 3, "AV2 explicit level conformance record")?;
        s.off("AV2 still picture without single-picture header")?;
        temporal_layers = s.field(
            u32::from(h.layering.temporal_layers - 1),
            TEMPORAL_LAYER_BITS,
        )? as u8
            + 1;
        s.fixed(0, 3, "AV2 embedded layers")?;
        monotonic_output = s.field(u32::from(h.variants.monotonic_output), 1)? != 0;
    }
    let width_bits = s.field(32 - (h.max_width - 1).leading_zeros().min(31) - 1, 4)? + 1;
    let height_bits = s.field(32 - (h.max_height - 1).leading_zeros().min(31) - 1, 4)? + 1;
    let max_width = s.field(h.max_width - 1, width_bits)? + 1;
    let max_height = s.field(h.max_height - 1, height_bits)? + 1;
    let crop = conformance_window(s, h.crop)?;
    if !single {
        s.off("AV2 display model")?;
        s.off("AV2 decoder model")?;
    }
    if temporal_layers > 1 {
        s.off("AV2 temporal layer dependency map")?;
    }
    partition(s)?;
    s.off("AV2 extended segmentation")?;
    segmentation_data(s)?;
    intra(s)?;
    let (order_hint_bits, explicit_references, reference_lists, same_reference_compounds) = inter(
        s,
        single,
        (h.order_hint_bits, h.variants.same_reference_compounds),
        h.layering.explicit_references,
        h.reference_lists,
    )?;
    let (screen_content, integer_mv) = if single {
        (h.variants.screen_content, h.variants.integer_mv)
    } else {
        screen_content(s, (h.variants.screen_content, h.variants.integer_mv))?
    };
    let (chroma_delta_q, dc_delta_q, parity_hiding, tcq) = transform(
        s,
        single,
        (h.chroma_delta_q, h.dc_delta_q, h.parity_hiding, h.tcq),
    )?;
    let (ccso, variants) = filters(s, single, h.ccso, h.variants)?;
    let variants = Variants {
        same_reference_compounds,
        screen_content,
        integer_mv,
        monotonic_output,
        tiles: sequence_tiles(s, h.variants.tiles, (max_width, max_height, level))?,
        ..variants
    };
    s.off("AV2 film grain")?;
    s.off("AV2 sequence extension")?;
    Ok(SequenceHeader {
        id: id as u8,
        max_width,
        max_height,
        level,
        single_picture: single,
        order_hint_bits,
        crop,
        layering: Layering {
            temporal_layers,
            explicit_references,
        },
        chroma_delta_q,
        dc_delta_q,
        reference_lists,
        parity_hiding,
        tcq,
        ccso,
        variants,
    })
}

/// `seg_info_present_flag` and the sequence's segment features: read and
/// dropped (frames of the subset never enable segmentation, so they never
/// refer to them).
fn segmentation_data(s: &mut Syntax<'_>) -> Result<()> {
    if let Some(reader) = s.other_encoders(false)? {
        reader.read_bit()?; // seg_info_adaptive_flag
        SegmentFeatures::parse(reader, BASE_SEGMENTS)?;
    }
    Ok(())
}

/// Segments without `enable_ext_seg`.
const BASE_SEGMENTS: usize = 8;

/// `force_screen_content_tools` of a video sequence and, with the tools
/// possible, `force_integer_mv` (per frame without them): each "adaptive",
/// else on or off.
fn screen_content(
    s: &mut Syntax<'_>,
    (tools, integer_mv): (Option<bool>, Option<bool>),
) -> Result<(Option<bool>, Option<bool>)> {
    let tools = optional_flag(s, tools)?;
    let integer_mv = if tools == Some(false) {
        None
    } else {
        optional_flag(s, integer_mv)?
    };
    Ok((tools, integer_mv))
}

/// A three-way sequence choice: a flag for "each frame says" (`None`),
/// else the fixed value.
fn optional_flag(s: &mut Syntax<'_>, value: Option<bool>) -> Result<Option<bool>> {
    if s.field(u32::from(value.is_none()), 1)? != 0 {
        return Ok(None);
    }
    Ok(Some(s.field(u32::from(value == Some(true)), 1)? != 0))
}

/// `seq_tile_info_present_flag` with the sequence's tile layout, accepted
/// when it is one tile.
fn sequence_tiles(
    s: &mut Syntax<'_>,
    ours: Option<SequenceTiles>,
    (max_width, max_height, level): (u32, u32, u8),
) -> Result<Option<SequenceTiles>> {
    let Syntax::Read(reader) = s else {
        if ours.is_some() {
            return Err(DecodeError::Unsupported(
                "AV2 writing a sequence tile layout",
            ));
        }
        s.off("AV2 sequence tile configuration")?;
        return Ok(None);
    };
    if !reader.read_bit()? {
        return Ok(None);
    }
    let use_of = if reader.read_bit()? {
        SequenceTiles::PerFrame
    } else {
        SequenceTiles::Fixed
    };
    let geometry = TileGeometry {
        size: (max_width, max_height),
        superblock_log2: SUPERBLOCK_LOG2,
        sequence_superblock_log2: SUPERBLOCK_LOG2,
        level,
        high_tier: false,
    };
    let layout = tiles::parse(reader, geometry)?;
    if (layout.columns(), layout.rows()) != (1, 1) {
        return Err(DecodeError::Unsupported("AV2 multiple tiles"));
    }
    Ok(Some(use_of))
}

/// log2 of the subset's 64×64 superblocks.
const SUPERBLOCK_LOG2: u8 = 6;

/// AVM `av2_write_conformance_window` / `av2_read_conformance_window`.
fn conformance_window(s: &mut Syntax<'_>, crop: CropWindow) -> Result<CropWindow> {
    if s.field(u32::from(crop.is_enabled()), 1)? == 0 {
        return Ok(CropWindow::NONE);
    }
    if s.uvlc(0)? != 0 {
        return Err(DecodeError::Unsupported(
            "AV2 conformance window left offset",
        ));
    }
    let right = s.uvlc(crop.right)?;
    if s.uvlc(0)? != 0 {
        return Err(DecodeError::Unsupported(
            "AV2 conformance window top offset",
        ));
    }
    let bottom = s.uvlc(crop.bottom)?;
    Ok(CropWindow { right, bottom })
}

fn partition(s: &mut Syntax<'_>) -> Result<()> {
    s.off("AV2 256x256 superblocks")?;
    s.off("AV2 128x128 superblocks")?;
    s.off("AV2 separate chroma partitions")?;
    s.off("AV2 extended partitions")?;
    s.fixed(1, 1, "AV2 unrestricted partition aspect ratio")?;
    s.fixed(0, 1, "AV2 partition aspect ratio above 2")
}

fn intra(s: &mut Syntax<'_>) -> Result<()> {
    for name in [
        "AV2 intra DIP",
        "AV2 intra edge filter",
        "AV2 multiple reference lines",
        "AV2 chroma from luma",
    ] {
        s.off(name)?;
    }
    s.fixed(0, 2, "AV2 alternate CfL downsample filter")?;
    s.off("AV2 multi-hypothesis chroma prediction")?;
    s.off("AV2 intra bi-prediction")
}

/// The inter tool group; returns the order-hint width,
/// `enable_explicit_ref_frame_map`, the reference-list configuration and
/// `num_same_ref_compound`.
fn inter(
    s: &mut Syntax<'_>,
    single: bool,
    (order_hint_bits, same_reference_compounds): (u8, u8),
    explicit_references: bool,
    lists: ListConfig,
) -> Result<(u8, bool, ListConfig, u8)> {
    let order_hint_bits = if single {
        0
    } else {
        for name in [
            "AV2 inter-intra",
            "AV2 causal warp",
            "AV2 warp delta",
            "AV2 warp extension",
            "AV2 masked compound",
            "AV2 reference motion fields",
        ] {
            s.off(name)?;
        }
        (s.field(u32::from(order_hint_bits.saturating_sub(1)), 4)? + 1) as u8
    };
    let ref_mv_bank = s.field(u32::from(lists.ref_mv_bank), 1)? != 0;
    s.fixed(1, 1, "AV2 DRL reordering")?;
    let mut explicit = false;
    let mut max_drl_bits = ListConfig::BASIC.max_drl_bits;
    if !single {
        explicit = s.field(u32::from(explicit_references), 1)? != 0;
        s.off("AV2 non-default reference buffer size")?; // default eight slots
        s.fixed(0, 3, "AV2 long-term frame IDs")?;
        max_drl_bits = s.quniform(
            u32::from(lists.max_drl_bits - MIN_DRL_BITS),
            DRL_BITS_CHOICES,
        )? as u8
            + MIN_DRL_BITS;
        s.off("AV2 per-frame DRL count")?;
    }
    s.fixed(0, 1, "AV2 IBC DRL count above one")?; // quniform(3, 0) = 0
    s.off("AV2 per-frame IBC DRL count")?;
    let mut same_reference = 0;
    if !single {
        same_reference = s.field(u32::from(same_reference_compounds), 2)? as u8;
        s.off("AV2 temporal interpolated prediction")?;
        s.off("AV2 motion trajectories")?;
    }
    s.off("AV2 block adaptive weighted prediction")?;
    if !single {
        for name in [
            "AV2 compound weights",
            "AV2 implicit masked blending",
            "AV2 sub-PU filter",
        ] {
            s.off(name)?;
        }
        s.fixed(0, 2, "AV2 optical flow refinement")?;
        for name in [
            "AV2 MV refinement",
            "AV2 reference upsampling",
            "AV2 adaptive MVD",
            "AV2 MVD sign derivation",
            "AV2 flexible MV resolution",
            "AV2 global motion",
            "AV2 short refresh flags",
        ] {
            s.off(name)?;
        }
    }
    let lists = ListConfig {
        max_drl_bits,
        ref_mv_bank,
    };
    Ok((order_hint_bits, explicit, lists.validate()?, same_reference))
}

/// AVM `MIN_MAX_DRL_BITS`, and the number of `def_max_drl_bits` values
/// (`MAX_MAX_DRL_BITS - MIN_MAX_DRL_BITS + 1`).
const MIN_DRL_BITS: u8 = 1;
const DRL_BITS_CHOICES: u32 = 5;

/// A zero base quantizer delta: the 5-bit literal offset by
/// `DELTA_DCQUANT_MIN` (−23).
const BASE_DELTA_ZERO: u32 = 23;

/// The transform and quantizer tools; returns whether frames code the
/// chroma AC delta (`uv_ac_delta_q_enabled`) and the DC deltas, and may
/// hide DC parities (`enable_parity_hiding`), and which use TCQ
/// (`enable_tcq`).
fn transform(
    s: &mut Syntax<'_>,
    single: bool,
    (chroma_delta_q, dc_delta_q, parity_hiding, tcq): (bool, bool, bool, TcqFrames),
) -> Result<(bool, bool, bool, TcqFrames)> {
    for name in [
        "AV2 forward skip coding",
        "AV2 intra identity transform",
        "AV2 intra secondary transform",
        "AV2 inter secondary transform",
    ] {
        s.off(name)?;
    }
    s.fixed(1, 1, "AV2 non-DCT chroma transforms")?;
    if !single {
        s.off("AV2 inter DDT")?;
    }
    s.fixed(1, 1, "AV2 full transform partition set")?;
    s.off("AV2 cross-component transform")?;
    let tcq = tcq_frames(s, single, tcq)?;
    // enable_parity_hiding follows unless every frame uses TCQ.
    let parity_hiding = tcq != TcqFrames::Every && s.field(u32::from(parity_hiding), 1)? != 0;
    if !single {
        s.off("AV2 average CDF")?;
    }
    // Single pictures infer enable_avg_cdf=avg_cdf_type=1, not zero.
    s.off("AV2 separate UV delta quantizers")?;
    let dc_delta_q = s.field(u32::from(!dc_delta_q), 1)? == 0; // equal_ac_dc_q
    if dc_delta_q {
        s.fixed(BASE_DELTA_ZERO, 5, "AV2 base luma DC quantizer delta")?;
        s.fixed(1, 1, "AV2 frames without a luma DC delta")?;
        s.fixed(BASE_DELTA_ZERO, 5, "AV2 base UV DC quantizer delta")?;
        s.fixed(1, 1, "AV2 frames without a UV DC delta")?;
    }
    s.fixed(BASE_DELTA_ZERO, 5, "AV2 base UV AC quantizer delta")?;
    let chroma_delta_q = s.field(u32::from(chroma_delta_q), 1)? != 0;
    Ok((chroma_delta_q, dc_delta_q, parity_hiding, tcq))
}

/// `enable_tcq`: a flag, then for sequences of more than one picture
/// whether each frame signals it (`TCQ_8ST_FR`) or all use it (`TCQ_8ST`).
fn tcq_frames(s: &mut Syntax<'_>, single: bool, tcq: TcqFrames) -> Result<TcqFrames> {
    if single && tcq == TcqFrames::Signalled {
        return Err(DecodeError::Malformed(
            "AV2 single pictures cannot signal TCQ per frame",
        ));
    }
    if s.field(u32::from(tcq != TcqFrames::Off), 1)? == 0 {
        return Ok(TcqFrames::Off);
    }
    if single || s.field(u32::from(tcq == TcqFrames::Signalled), 1)? == 0 {
        Ok(TcqFrames::Every)
    } else {
        Ok(TcqFrames::Signalled)
    }
}

/// The loop-filter tools; returns `enable_ccso` and the fields other
/// encoders vary.
fn filters(
    s: &mut Syntax<'_>,
    single: bool,
    ccso: bool,
    variants: Variants,
) -> Result<(bool, Variants)> {
    s.off("AV2 disabled tile-boundary filtering")?;
    s.fixed(1, 1, "AV2 sequence without CDEF")?; // frames choose CDEF themselves
    s.off("AV2 guided directional filter")?;
    s.off("AV2 restoration")?;
    let ccso = s.field(u32::from(ccso), 1)? != 0;
    if ccso {
        s.off("AV2 CCSO units matched to superblocks")?;
    }
    let cdef_on_skipped = if single {
        // Inferred per frame; the field keeps the caller's value.
        variants.cdef_on_skipped
    } else if s.field(u32::from(variants.cdef_on_skipped == Some(true)), 1)? != 0 {
        Some(true)
    } else if s.field(u32::from(variants.cdef_on_skipped == Some(false)), 1)? != 0 {
        Some(false)
    } else {
        None
    };
    let deblock_delta_bits = s.field(
        u32::from(variants.deblock_delta_bits - MIN_DEBLOCK_DELTA_BITS),
        2,
    )? as u8
        + MIN_DEBLOCK_DELTA_BITS;
    Ok((
        ccso,
        Variants {
            deblock_delta_bits,
            cdef_on_skipped,
            ..variants
        },
    ))
}

/// `df_par_bits_minus2` codes the excess over two bits.
const MIN_DEBLOCK_DELTA_BITS: u8 = 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn still_and_video_sequences_and_bounds() {
        for single_picture in [false, true] {
            for (max_width, max_height) in [(1, 1), (64, 48), (1920, 1080), (65536, 65536)] {
                let h = SequenceHeader {
                    id: 15,
                    max_width,
                    max_height,
                    level: 31,
                    single_picture,
                    order_hint_bits: if single_picture { 0 } else { 8 },
                    crop: CropWindow::NONE,
                    layering: Layering::FLAT,
                    chroma_delta_q: false,
                    dc_delta_q: false,
                    parity_hiding: false,
                    tcq: TcqFrames::Off,
                    ccso: false,
                    variants: Variants::OURS,
                    reference_lists: crate::mv_refs::ListConfig::BASIC,
                };
                let payload = write(&h).unwrap();
                assert_eq!(parse(&payload).unwrap(), h);
                for len in 0..payload.len() {
                    assert!(parse(&payload[..len]).is_err());
                }
                let mut extended = payload.clone();
                extended.push(0);
                assert!(parse(&extended).is_err());
                let mut bad_stop = payload;
                *bad_stop.last_mut().unwrap() = 0;
                assert!(parse(&bad_stop).is_err());
            }
        }
    }

    #[test]
    fn conformance_window_round_trips_and_is_bounded() {
        let h = SequenceHeader {
            id: 0,
            max_width: 1888,
            max_height: 720,
            level: 31,
            single_picture: false,
            order_hint_bits: 8,
            crop: CropWindow {
                right: 4,
                bottom: 6,
            },
            layering: Layering {
                temporal_layers: 3,
                explicit_references: true,
            },
            chroma_delta_q: true,
            dc_delta_q: true,
            parity_hiding: false,
            tcq: TcqFrames::Off,
            ccso: false,
            variants: Variants::OURS,
            reference_lists: crate::mv_refs::ListConfig::BASIC,
        };
        assert_eq!(parse(&write(&h).unwrap()).unwrap(), h);
        assert_eq!(h.visible_size(), (1884, 714));
        for crop in [
            CropWindow {
                right: 1888,
                bottom: 0,
            },
            CropWindow {
                right: 0,
                bottom: 720,
            },
            CropWindow {
                right: 3,
                bottom: 0,
            },
        ] {
            assert!(write(&SequenceHeader { crop, ..h }).is_err());
        }
    }

    #[test]
    fn layered_sequences_round_trip() {
        for temporal_layers in 1..=3 {
            for explicit_references in [false, true] {
                let h = SequenceHeader {
                    id: 0,
                    max_width: 640,
                    max_height: 360,
                    level: 31,
                    single_picture: false,
                    order_hint_bits: 8,
                    crop: CropWindow::NONE,
                    layering: Layering {
                        temporal_layers,
                        explicit_references,
                    },
                    chroma_delta_q: explicit_references,
                    dc_delta_q: explicit_references,
                    parity_hiding: false,
                    tcq: TcqFrames::Off,
                    ccso: false,
                    variants: Variants::OURS,
                    reference_lists: crate::mv_refs::ListConfig::BASIC,
                };
                assert_eq!(parse(&write(&h).unwrap()).unwrap(), h);
            }
        }
        assert_eq!(Layering::FLAT.temporal_layers, 1);
    }

    #[test]
    fn matches_original_avm_sequence_writer() {
        let mut vectors = include_str!("../tests/data/avm-sequence-headers.hex").lines();
        for single_picture in [false, true] {
            for (max_width, max_height) in [(1, 1), (64, 48), (1920, 1080), (65536, 65536)] {
                let h = SequenceHeader {
                    id: 15,
                    max_width,
                    max_height,
                    level: 31,
                    single_picture,
                    order_hint_bits: if single_picture { 0 } else { 8 },
                    crop: CropWindow::NONE,
                    layering: Layering::FLAT,
                    chroma_delta_q: false,
                    dc_delta_q: false,
                    parity_hiding: false,
                    tcq: TcqFrames::Off,
                    ccso: false,
                    variants: Variants::OURS,
                    reference_lists: crate::mv_refs::ListConfig::BASIC,
                };
                let hex = vectors.next().unwrap();
                let oracle: Vec<u8> = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect();
                assert_eq!(write(&h).unwrap(), oracle);
                assert_eq!(parse(&oracle).unwrap(), h);
            }
        }
        assert!(vectors.next().is_none());
    }

    #[test]
    fn rejects_invalid_fields_and_hostile_payloads() {
        let h = SequenceHeader {
            id: 0,
            max_width: 64,
            max_height: 64,
            level: 0,
            single_picture: false,
            order_hint_bits: 8,
            crop: CropWindow::NONE,
            layering: Layering::FLAT,
            chroma_delta_q: false,
            dc_delta_q: false,
            parity_hiding: false,
            tcq: TcqFrames::Off,
            ccso: false,
            variants: Variants::OURS,
            reference_lists: crate::mv_refs::ListConfig::BASIC,
        };
        for bad in [
            SequenceHeader { id: 16, ..h },
            SequenceHeader { max_width: 0, ..h },
            SequenceHeader {
                max_height: 65537,
                ..h
            },
            SequenceHeader { level: 22, ..h },
            SequenceHeader {
                order_hint_bits: 0,
                ..h
            },
            SequenceHeader {
                single_picture: true,
                ..h
            },
        ] {
            assert!(write(&bad).is_err());
        }
        for value in 0..=255 {
            assert!(parse(&[value; 64]).is_err());
        }
    }
}
