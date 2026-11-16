//! AV2 film grain: the film grain model OBU (up to eight parameter sets,
//! by id) and the grain laid over a picture when it is output, never over
//! the reference it stays in. A shown frame names a model and a seed in its
//! header ([`super::frame_header::FilmGrain`]); the synthesis is AV1's with
//! AV2's block layout (`nc_dsp::av1_film_grain`,
//! [`nc_dsp::Dsp::av2_add_film_grain`]).
//!
//! Reimplemented from dav2d 237229e (BSD-2-Clause, see `THIRD_PARTY.md`):
//! `src/obu.c` `parse_fgm_hdr`, `src/lib.c` `has_grain`, `src/fg_apply_tmpl.c`
//! `dav2d_apply_grain`. Where dav2d and AVM v1.0.0 differ (dav2d ignores
//! `mc_identity`), AVM's `avm_dsp/grain_synthesis.c` is followed.
use std::rc::Rc;

use nc_common::decode::{BitReader, DecodeError, Result};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;
use nc_dsp::av1_film_grain::{Av1FilmGrainParams, Av1ScalingPoints};

/// Film grain model ids (`fgm_id`): one bit each in the OBU's mask.
pub const MAX_MODELS: usize = 8;

/// The models an FGM OBU may set, by id.
pub type Models = [Option<Rc<GrainModel>>; MAX_MODELS];

/// Bits of the OBU's model mask.
const MASK_BITS: u32 = 8;
/// `chroma_format_idc` of 4:2:0 (the only layout the decoder takes).
const CHROMA_FORMAT_420: u32 = 0;
/// Bits of a plane's point count, and the most points it may have.
const POINT_COUNT_BITS: u32 = 4;
const MAX_POINTS: usize = 14;
/// Bits of the coded widths of point values (`1 + n`) and scalings (`5 + n`).
const VALUE_WIDTH_BITS: u32 = 3;
const SCALING_WIDTH_BITS: u32 = 2;
const MIN_VALUE_WIDTH: u32 = 1;
const MIN_SCALING_WIDTH: u32 = 5;
/// Largest point value: values are increments summed along the points.
const MAX_POINT_VALUE: u32 = 255;
/// Bits of `scaling_shift − 8`, `ar_coeff_lag`, `ar_coeff_shift − 6` and
/// `grain_scale_shift`.
const SHIFT_BITS: u32 = 2;
/// Bits of the coded width of the AR coefficients (`5 + n`).
const COEFFICIENT_WIDTH_BITS: u32 = 2;
const MIN_COEFFICIENT_WIDTH: u32 = 5;
/// The bias of stored AR coefficients (`ar_coeffs_*_plus_128`).
const COEFFICIENT_BIAS: i32 = 128;
/// Bits of the chroma multipliers and offset (stored with their biases, as
/// AV1 codes them).
const MULTIPLIER_BITS: u32 = 8;
const OFFSET_BITS: u32 = 9;

/// One parameter set of an FGM OBU: AV1's synthesis parameters (the seed
/// comes from each frame) and AV2's block size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrainModel {
    params: Av1FilmGrainParams,
    /// `block_size`: 32x32 grain blocks rather than 16x16.
    large_blocks: bool,
}

impl GrainModel {
    /// Whether the model changes a picture (dav2d `has_grain`): scaling
    /// points, or chroma clipped to the restricted range.
    fn changes_picture(&self) -> bool {
        let p = &self.params;
        p.y_points.count > 0
            || p.cb_points.count > 0
            || p.cr_points.count > 0
            || (p.clip_to_restricted_range && p.chroma_scaling_from_luma)
    }
}

/// The grain a picture takes when output: a model and the frame's seed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputGrain {
    model: Rc<GrainModel>,
    seed: u16,
}

impl OutputGrain {
    /// The grain of model `id` with `seed`, or `None` when the model leaves
    /// the picture unchanged.
    /// # Errors
    /// Rejects a model no FGM OBU has set.
    pub fn new(models: &Models, id: u8, seed: u16) -> Result<Option<Self>> {
        let model = models[usize::from(id)]
            .clone()
            .ok_or(DecodeError::Malformed("AV2 film grain model not set"))?;
        Ok(model.changes_picture().then_some(Self { model, seed }))
    }

    /// Lay the grain over `input` into `output`, a picture of its size.
    pub fn apply(&self, dsp: &Dsp, input: &Picture, output: &mut Picture) {
        let params = Av1FilmGrainParams {
            grain_seed: self.seed,
            ..self.model.params
        };
        let size = (input.width(), input.height());
        let [y, u, v] = PlaneIndex::ALL.map(|plane| input.plane(plane).view());
        let [out_y, out_u, out_v] = output.planes_mut();
        dsp.av2_add_film_grain(
            &params,
            self.model.large_blocks,
            [&y, &u, &v],
            size,
            [
                &mut out_y.view_mut(),
                &mut out_u.view_mut(),
                &mut out_v.view_mut(),
            ],
        );
    }
}

/// Parse an FGM OBU (dav2d `parse_fgm_hdr`) into `models`: each id in its
/// mask takes a new parameter set.
/// # Errors
/// Rejects malformed syntax and layouts other than 4:2:0.
pub fn parse(payload: &[u8], models: &mut Models) -> Result<()> {
    let mut bits = BitReader::new(payload);
    let mask = bits.read_bits(MASK_BITS)?;
    if bits.read_uvlc()? != CHROMA_FORMAT_420 {
        return Err(DecodeError::Malformed(
            "AV2 film grain model of another chroma format",
        ));
    }
    for (id, model) in models.iter_mut().enumerate() {
        if mask & (1 << id) != 0 {
            *model = Some(Rc::new(parse_model(&mut bits)?));
        }
    }
    bits.trailing_bits()
}

/// One parameter set, 4:2:0.
fn parse_model(bits: &mut BitReader<'_>) -> Result<GrainModel> {
    let mut p = Av1FilmGrainParams {
        ar_coeffs_y_plus_128: [COEFFICIENT_BIAS as u8; 24],
        ar_coeffs_cb_plus_128: [COEFFICIENT_BIAS as u8; 25],
        ar_coeffs_cr_plus_128: [COEFFICIENT_BIAS as u8; 25],
        ..Av1FilmGrainParams::default()
    };
    p.chroma_scaling_from_luma = bits.read_bit()?;
    p.y_points = scaling_points(bits)?;
    if !p.chroma_scaling_from_luma {
        p.cb_points = scaling_points(bits)?;
        p.cr_points = scaling_points(bits)?;
    }
    if (p.cb_points.count > 0) != (p.cr_points.count > 0) {
        return Err(DecodeError::Malformed(
            "AV2 film grain on one chroma plane of 4:2:0",
        ));
    }
    p.grain_scaling_minus_8 = bits.read_bits(SHIFT_BITS)? as u8;
    p.ar_coeff_lag = bits.read_bits(SHIFT_BITS)? as u8;
    let lag = usize::from(p.ar_coeff_lag);
    let positions = 2 * lag * (lag + 1);
    if p.y_points.count > 0 {
        ar_coefficients(bits, &mut p.ar_coeffs_y_plus_128[..positions])?;
    }
    // Chroma's AR filter also weighs the co-located luma grain.
    let chroma_positions = positions + usize::from(p.y_points.count > 0);
    for (points, coefficients) in [
        (p.cb_points.count, &mut p.ar_coeffs_cb_plus_128),
        (p.cr_points.count, &mut p.ar_coeffs_cr_plus_128),
    ] {
        if points > 0 || p.chroma_scaling_from_luma {
            ar_coefficients(bits, &mut coefficients[..chroma_positions])?;
        }
    }
    p.ar_coeff_shift_minus_6 = bits.read_bits(SHIFT_BITS)? as u8;
    p.grain_scale_shift = bits.read_bits(SHIFT_BITS)? as u8;
    for (points, multipliers) in [
        (p.cb_points.count, &mut p.cb_mult),
        (p.cr_points.count, &mut p.cr_mult),
    ] {
        if points > 0 {
            *multipliers = (
                bits.read_bits(MULTIPLIER_BITS)? as u8,
                bits.read_bits(MULTIPLIER_BITS)? as u8,
                bits.read_bits(OFFSET_BITS)? as u16,
            );
        }
    }
    p.overlap = bits.read_bit()?;
    p.clip_to_restricted_range = bits.read_bit()?;
    if p.clip_to_restricted_range {
        p.mc_identity = bits.read_bit()?;
    }
    let large_blocks = bits.read_bit()?;
    Ok(GrainModel {
        params: p,
        large_blocks,
    })
}

/// A plane's scaling points: their value increments and scalings, each of
/// a coded width.
fn scaling_points(bits: &mut BitReader<'_>) -> Result<Av1ScalingPoints<MAX_POINTS>> {
    let mut points = Av1ScalingPoints::<MAX_POINTS> {
        count: bits.read_bits(POINT_COUNT_BITS)? as usize,
        ..Av1ScalingPoints::default()
    };
    if points.count > MAX_POINTS {
        return Err(DecodeError::Malformed("AV2 film grain point count"));
    }
    if points.count == 0 {
        return Ok(points);
    }
    let value_width = MIN_VALUE_WIDTH + bits.read_bits(VALUE_WIDTH_BITS)?;
    let scaling_width = MIN_SCALING_WIDTH + bits.read_bits(SCALING_WIDTH_BITS)?;
    let mut value = 0;
    for i in 0..points.count {
        let increment = bits.read_bits(value_width)?;
        value += increment;
        if (i > 0 && increment == 0) || value > MAX_POINT_VALUE {
            return Err(DecodeError::Malformed(
                "AV2 film grain point values not increasing",
            ));
        }
        points.value[i] = value as u8;
        // At most 5 + 3 bits.
        points.scaling[i] = bits.read_bits(scaling_width)? as u8;
    }
    Ok(points)
}

/// AR coefficients of a coded width, stored with AV1's bias of 128.
fn ar_coefficients(bits: &mut BitReader<'_>, coefficients: &mut [u8]) -> Result<()> {
    let width = MIN_COEFFICIENT_WIDTH + bits.read_bits(COEFFICIENT_WIDTH_BITS)?;
    let half = 1 << (width - 1);
    for coefficient in coefficients {
        // Within ±128 for widths up to 8 bits.
        *coefficient = (bits.read_bits(width)? as i32 - half + COEFFICIENT_BIAS) as u8;
    }
    Ok(())
}
