//! `sequence_header_obu()` (AV1 §5.5): every field of an 8-bit 4:2:0 Main
//! profile sequence header, with operating point 0 selected (§7.18 leaves
//! the choice to the application; libaom and dav1d default to 0).
use nc_common::ColourInfo;
use nc_common::colour::{matrix, read_code_points};
use nc_common::decode::{BitReader, DecodeError};

/// `seq_profile` of the Main profile (AV1 §6.4.1).
const MAIN_PROFILE: u32 = 0;
/// `seq_level_idx` above which `seq_tier` is coded (AV1 §5.5.1).
const LEVEL_WITH_TIER: u32 = 7;
/// `SELECT_SCREEN_CONTENT_TOOLS` / `SELECT_INTEGER_MV` (AV1 §6.4.1): the
/// frame header chooses.
pub(crate) const SELECT_PER_FRAME: u8 = 2;
/// Largest picture accepted: the level 6.3 limits (AV1 Annex A.3,
/// `MaxHSize`, `MaxVSize`, `MaxPicSize`) bound memory and decoding time for
/// untrusted input. The side limits alone would admit a 16384x16384 picture
/// (about 400 MB per frame buffer).
const MAX_WIDTH: usize = 16384;
const MAX_HEIGHT: usize = 8704;
const MAX_PICTURE_SIZE: usize = 35_651_584;

/// `OperatingPointIdc` bit of spatial layer 0 (AV1 §6.4.1).
const SPATIAL_LAYER_SHIFT: u32 = 8;

/// The decoder-relevant fields of a sequence header (names follow §5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(crate) struct Sequence {
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    /// `decoder_model_info_present_flag` and the lengths it codes, which
    /// frame headers need to skip their buffer-removal times.
    pub decoder_model: Option<DecoderModel>,
    /// `equal_picture_interval`: frame headers carry no presentation time.
    pub equal_picture_interval: bool,
    /// `OperatingPointIdc` of the chosen operating point 0.
    pub operating_point_idc: u32,
    /// `decoder_model_present_for_this_op[]` for every operating point:
    /// frame headers code a removal time for each one present.
    pub operating_points: [OperatingPoint; MAX_OPERATING_POINTS],
    pub operating_point_count: usize,
    pub frame_width_bits: u32,
    pub frame_height_bits: u32,
    pub max_frame_width: usize,
    pub max_frame_height: usize,
    /// `frame_id_numbers_present_flag` with the two id lengths.
    pub frame_ids: Option<FrameIdLengths>,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    /// `seq_force_screen_content_tools` (0, 1 or [`SELECT_PER_FRAME`]).
    pub force_screen_content_tools: u8,
    /// `seq_force_integer_mv` (0, 1 or [`SELECT_PER_FRAME`]).
    pub force_integer_mv: u8,
    /// `OrderHintBits` (0 without order hints).
    pub order_hint_bits: u32,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub colour: ColourInfo,
    /// `mono_chrome`: one plane coded.
    pub mono_chrome: bool,
    pub separate_uv_delta_q: bool,
    pub film_grain_params_present: bool,
}

/// Most operating points a sequence header may code (5-bit count).
pub(crate) const MAX_OPERATING_POINTS: usize = 32;

/// `decoder_model_info()` lengths (AV1 §5.5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecoderModel {
    pub buffer_delay_length: u32,
    pub buffer_removal_time_bits: u32,
    pub frame_presentation_time_bits: u32,
}

/// Per operating point: what frame headers code for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct OperatingPoint {
    pub idc: u32,
    pub decoder_model_present: bool,
}

/// `delta_frame_id_length_minus_2 + 2` and the full id length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameIdLengths {
    pub delta: u32,
    pub id: u32,
}

impl Sequence {
    /// Whether this operating point decodes the layer (§7.18: OBUs of other
    /// layers are dropped).
    pub(crate) fn in_operating_point(&self, temporal_id: u8, spatial_id: u8) -> bool {
        let idc = self.operating_point_idc;
        idc == 0
            || (idc >> temporal_id) & 1 != 0
                && (idc >> (SPATIAL_LAYER_SHIFT + u32::from(spatial_id))) & 1 != 0
    }
}

/// Parse a sequence header OBU payload.
///
/// # Errors
/// Truncated or malformed data, or a stream outside 8-bit 4:2:0 Main.
pub(crate) fn parse(payload: &[u8]) -> Result<Sequence, DecodeError> {
    let mut r = BitReader::new(payload);
    if r.read_bits(3)? != MAIN_PROFILE {
        return Err(DecodeError::Unsupported("AV1 profile other than Main"));
    }
    let still_picture = r.read_bit()?;
    let reduced_still_picture_header = r.read_bit()?;
    let layers = if reduced_still_picture_header {
        if !still_picture {
            return Err(DecodeError::Malformed(
                "AV1 reduced header on a moving picture",
            ));
        }
        r.read_bits(5)?; // seq_level_idx[0]
        Layers::default()
    } else {
        Layers::parse(&mut r)?
    };
    let frame_width_bits = r.read_bits(4)? + 1;
    let frame_height_bits = r.read_bits(4)? + 1;
    let max_frame_width = r.read_bits(frame_width_bits)? as usize + 1;
    let max_frame_height = r.read_bits(frame_height_bits)? as usize + 1;
    if max_frame_width > MAX_WIDTH
        || max_frame_height > MAX_HEIGHT
        || max_frame_width * max_frame_height > MAX_PICTURE_SIZE
    {
        return Err(DecodeError::Unsupported("AV1 frame size limit"));
    }
    let frame_ids = if !reduced_still_picture_header && r.read_bit()? {
        let delta = r.read_bits(4)? + 2;
        let id = r.read_bits(3)? + 1 + delta;
        Some(FrameIdLengths { delta, id })
    } else {
        None
    };
    let use_128x128_superblock = r.read_bit()?;
    let enable_filter_intra = r.read_bit()?;
    let enable_intra_edge_filter = r.read_bit()?;
    let mut tools = InterTools::default();
    if !reduced_still_picture_header {
        tools = InterTools::parse(&mut r)?;
    }
    let enable_superres = r.read_bit()?;
    let enable_cdef = r.read_bit()?;
    let enable_restoration = r.read_bit()?;
    let colour = color_config(&mut r)?;
    let film_grain_params_present = r.read_bit()?;
    Ok(Sequence {
        still_picture,
        reduced_still_picture_header,
        decoder_model: layers.decoder_model,
        equal_picture_interval: layers.equal_picture_interval,
        operating_point_idc: layers.operating_points[0].idc,
        operating_points: layers.operating_points,
        operating_point_count: layers.operating_point_count,
        frame_width_bits,
        frame_height_bits,
        max_frame_width,
        max_frame_height,
        frame_ids,
        use_128x128_superblock,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound: tools.interintra_compound,
        enable_masked_compound: tools.masked_compound,
        enable_warped_motion: tools.warped_motion,
        enable_dual_filter: tools.dual_filter,
        enable_order_hint: tools.order_hint_bits > 0,
        enable_jnt_comp: tools.jnt_comp,
        enable_ref_frame_mvs: tools.ref_frame_mvs,
        force_screen_content_tools: tools.force_screen_content_tools,
        force_integer_mv: tools.force_integer_mv,
        order_hint_bits: tools.order_hint_bits,
        enable_superres,
        enable_cdef,
        enable_restoration,
        colour: colour.info,
        mono_chrome: colour.mono_chrome,
        separate_uv_delta_q: colour.separate_uv_delta_q,
        film_grain_params_present,
    })
}

/// Timing, decoder model and operating points of a full sequence header
/// (§5.5.1); a reduced still-picture header implies the defaults.
struct Layers {
    decoder_model: Option<DecoderModel>,
    equal_picture_interval: bool,
    operating_points: [OperatingPoint; MAX_OPERATING_POINTS],
    operating_point_count: usize,
}
impl Default for Layers {
    /// One operating point decoding everything, no timing information.
    fn default() -> Self {
        Self {
            decoder_model: None,
            equal_picture_interval: false,
            operating_points: [OperatingPoint::default(); MAX_OPERATING_POINTS],
            operating_point_count: 1,
        }
    }
}
impl Layers {
    fn parse(r: &mut BitReader<'_>) -> Result<Self, DecodeError> {
        let mut layers = Self::default();
        if r.read_bit()? {
            layers.equal_picture_interval = timing_info(r)?;
            if r.read_bit()? {
                layers.decoder_model = Some(decoder_model_info(r)?);
            }
        }
        let initial_display_delay_present = r.read_bit()?;
        layers.operating_point_count = r.read_bits(5)? as usize + 1;
        for point in &mut layers.operating_points[..layers.operating_point_count] {
            point.idc = r.read_bits(12)?;
            if r.read_bits(5)? > LEVEL_WITH_TIER {
                r.read_bit()?; // seq_tier
            }
            if let Some(model) = layers.decoder_model {
                point.decoder_model_present = r.read_bit()?;
                if point.decoder_model_present {
                    // decoder/encoder_buffer_delay, low_delay_mode_flag.
                    r.skip_bits(2 * model.buffer_delay_length as usize + 1)?;
                }
            }
            if initial_display_delay_present && r.read_bit()? {
                r.read_bits(4)?; // initial_display_delay_minus_1
            }
        }
        Ok(layers)
    }
}

/// The inter-coding tool flags of a full sequence header (§5.5.1).
#[allow(clippy::struct_excessive_bools)] // the specification's flags
struct InterTools {
    interintra_compound: bool,
    masked_compound: bool,
    warped_motion: bool,
    dual_filter: bool,
    jnt_comp: bool,
    ref_frame_mvs: bool,
    force_screen_content_tools: u8,
    force_integer_mv: u8,
    order_hint_bits: u32,
}
impl Default for InterTools {
    /// The values a reduced still-picture header implies.
    fn default() -> Self {
        Self {
            interintra_compound: false,
            masked_compound: false,
            warped_motion: false,
            dual_filter: false,
            jnt_comp: false,
            ref_frame_mvs: false,
            force_screen_content_tools: SELECT_PER_FRAME,
            force_integer_mv: SELECT_PER_FRAME,
            order_hint_bits: 0,
        }
    }
}
impl InterTools {
    fn parse(r: &mut BitReader<'_>) -> Result<Self, DecodeError> {
        let interintra_compound = r.read_bit()?;
        let masked_compound = r.read_bit()?;
        let warped_motion = r.read_bit()?;
        let dual_filter = r.read_bit()?;
        let order_hint = r.read_bit()?;
        let (jnt_comp, ref_frame_mvs) = if order_hint {
            (r.read_bit()?, r.read_bit()?)
        } else {
            (false, false)
        };
        let force_screen_content_tools = if r.read_bit()? {
            SELECT_PER_FRAME
        } else {
            u8::from(r.read_bit()?)
        };
        let force_integer_mv = if force_screen_content_tools > 0 {
            if r.read_bit()? {
                SELECT_PER_FRAME
            } else {
                u8::from(r.read_bit()?)
            }
        } else {
            SELECT_PER_FRAME
        };
        let order_hint_bits = if order_hint { r.read_bits(3)? + 1 } else { 0 };
        Ok(Self {
            interintra_compound,
            masked_compound,
            warped_motion,
            dual_filter,
            jnt_comp,
            ref_frame_mvs,
            force_screen_content_tools,
            force_integer_mv,
            order_hint_bits,
        })
    }
}

/// `timing_info()` (§5.5.3): returns `equal_picture_interval`.
fn timing_info(r: &mut BitReader<'_>) -> Result<bool, DecodeError> {
    r.skip_bits(64)?; // num_units_in_display_tick, time_scale
    let equal_picture_interval = r.read_bit()?;
    if equal_picture_interval {
        r.read_uvlc()?; // num_ticks_per_picture_minus_1
    }
    Ok(equal_picture_interval)
}

/// `decoder_model_info()` (§5.5.4).
fn decoder_model_info(r: &mut BitReader<'_>) -> Result<DecoderModel, DecodeError> {
    let buffer_delay_length = r.read_bits(5)? + 1;
    r.skip_bits(32)?; // num_units_in_decoding_tick
    let buffer_removal_time_bits = r.read_bits(5)? + 1;
    let frame_presentation_time_bits = r.read_bits(5)? + 1;
    Ok(DecoderModel {
        buffer_delay_length,
        buffer_removal_time_bits,
        frame_presentation_time_bits,
    })
}

/// What `color_config()` (§5.5.2) yields for an 8-bit stream.
struct ColourConfig {
    info: ColourInfo,
    mono_chrome: bool,
    separate_uv_delta_q: bool,
}

fn color_config(r: &mut BitReader<'_>) -> Result<ColourConfig, DecodeError> {
    if r.read_bit()? {
        return Err(DecodeError::Unsupported("AV1 high bit depth"));
    }
    let mono_chrome = r.read_bit()?;
    let description = if r.read_bit()? {
        Some(read_code_points(r)?)
    } else {
        None
    };
    if !mono_chrome && description.is_some_and(|d| d.matrix == matrix::IDENTITY) {
        // The identity matrix (sRGB included) requires 4:4:4, which the
        // Main profile allows only without chroma (§6.4.2).
        return Err(DecodeError::Unsupported("AV1 identity matrix (4:4:4)"));
    }
    let full_range = r.read_bit()?;
    if !mono_chrome {
        r.read_bits(2)?; // chroma_sample_position: informative
    }
    let separate_uv_delta_q = !mono_chrome && r.read_bit()?;
    Ok(ColourConfig {
        info: ColourInfo {
            description,
            full_range,
        },
        mono_chrome,
        separate_uv_delta_q,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::BitWriter;

    /// A sequence header prefix (§5.5.1) of one operating point at level
    /// 31 (no limit signalled) up to `max_frame_width/height_minus_1`.
    fn header_up_to_size(width: u32, height: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_bits(MAIN_PROFILE, 3);
        w.put_bits(0, 2); // still_picture, reduced_still_picture_header
        w.put_bits(0, 2); // timing_info_present, initial_display_delay_present
        w.put_bits(0, 5); // operating_points_cnt_minus_1
        w.put_bits(0, 12); // operating_point_idc[0]
        w.put_bits(31, 5); // seq_level_idx[0]
        w.put_bit(false); // seq_tier[0]
        let bits = 16;
        w.put_bits(bits - 1, 4);
        w.put_bits(bits - 1, 4);
        w.put_bits(width - 1, bits);
        w.put_bits(height - 1, bits);
        w.put_bits(0, 32); // the rest of the header, never reached
        while !w.is_byte_aligned() {
            w.put_bit(false);
        }
        w.into_bytes()
    }

    #[test]
    fn rejects_pictures_beyond_level_6_3() {
        for (width, height) in [(16384, 16384), (16384, 2304), (4096, 8705), (16385, 16)] {
            assert_eq!(
                parse(&header_up_to_size(width, height)),
                Err(DecodeError::Unsupported("AV1 frame size limit")),
                "{width}x{height}"
            );
        }
        let largest = parse(&header_up_to_size(8192, 4352));
        assert_ne!(
            largest,
            Err(DecodeError::Unsupported("AV1 frame size limit"))
        );
    }
}
