//! Final AV2 8-bit transforms, quantization and reconstruction: square
//! transforms of every type from 4×4 to 64×64 (`DCT_DCT` only at 64×64) and
//! the 4×4 lossless WHT.
//!
//! Audited against AVM v1.0.0 `hybrid_fwd_txfm.c`, `idct.c`,
//! `av2_inv_txfm2d.c`, `common_data.h`, `txb_common.c`, `quant_common.c` and
//! `decodetxb.c` (Alliance for Open Media, BSD-3-Clause-Clear and separate
//! AOM patent terms; see `THIRD_PARTY.md`). The DCT kernels live in `nc-dsp`
//! (`Dsp::av2_forward` / `Dsp::av2_inverse`, module `av2_txfm`); this module
//! adds the lossless path, quantization and dequantization. The WHT (at
//! qindex zero, 4×4 `DCT_DCT` only), frame-level luma DC, chroma AC and
//! chroma DC deltas ([`QuantDeltas`], [`PlaneQindex`]), no matrices, TCQ or secondary transforms
//! are supported. Quantization is
//! nearest-step, not an RD search.
//!
//! Coefficient blocks are row-major at the coded side
//! ([`TxSize::coded_width`]): a 64×64 transform codes only its top-left 32×32
//! coefficients. Residual blocks are row-major at the full side.

use nc_common::PlaneIndex;
use nc_common::decode::{DecodeError, Result};
use nc_dsp::{Av1Quantizer, Av2TxSize, Dsp};

pub use crate::tx_size::TxSize;
pub use crate::tx_type::TxType;

const QUANT_BITS: u32 = 3;
/// One quantizer step in [`quantize`]'s rounding offsets (1/128 step).
pub const ROUNDING_Q7_ONE: i32 = 128;
/// Round-to-nearest offset: no deadzone.
pub const NEAREST_ROUNDING_Q7: i32 = ROUNDING_Q7_ONE / 2;

/// [`quantize`] rounding offsets (1/128 step) of one transform's DC (first)
/// coefficient and of its AC coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rounding {
    /// Offset of the DC coefficient.
    pub dc_q7: i32,
    /// Offset of every other coefficient.
    pub ac_q7: i32,
}

impl Rounding {
    /// The same offset for every coefficient.
    #[must_use]
    pub const fn uniform(q7: i32) -> Self {
        Self {
            dc_q7: q7,
            ac_q7: q7,
        }
    }
}
const UNIT_QUANT: i32 = 1 << QUANT_BITS;
const MAX_LEVEL: i32 = (1 << 20) - 1;
const DEQUANT_MASK: u32 = 0x00ff_ffff;
const COEFF_MIN: i32 = -32768;
const COEFF_MAX: i32 = 32767;
/// Largest residual magnitude: the difference of two 8-bit samples.
const RESIDUAL_LIMIT: i32 = 255;
/// Levels per chunk of the sparse quantization and dequantization passes
/// (the 4×4 coded area: every coded area is whole chunks).
const LEVEL_CHUNK: usize = 16;
/// Scratch capacities, one per transform area (4×4 to 64×64).
const SCRATCH_CLASSES: [usize; 5] = [16, 64, 256, 1024, 4096];
const WHT_AREA: usize = 16;
const QUANT_BASE: [u16; 25] = [
    64, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76,
    78,
];

/// [`quantizer_step_q3`] of every qindex: the trellis prices several
/// dequantized levels per coefficient, and the curve's division and
/// remainder by 24 were ~0.5% of an encode's instructions.
const QUANTIZER_STEPS_Q3: [i32; 256] = {
    let mut steps = [0; 256];
    let mut q = 0;
    while q < steps.len() {
        steps[q] = if q < QUANT_BASE.len() {
            QUANT_BASE[q] as i32
        } else {
            (QUANT_BASE[(q - 1) % 24 + 1] as i32) << ((q - 1) / 24)
        };
        q += 1;
    }
    steps
};

/// Native equal-DC/AC quantizer in Q3 units for an 8-bit frame without deltas.
/// Qindex zero is the special lossless value 64, not the curve extrapolation.
#[must_use]
pub fn quantizer_step_q3(qindex: u8) -> i32 {
    QUANTIZER_STEPS_Q3[usize::from(qindex)]
}

/// Smallest frame-level quantizer delta (AVM `read_delta_q`: a present
/// flag, then a 7-bit two's-complement value).
pub const MIN_DELTA_Q: i8 = -64;
/// Largest frame-level quantizer delta.
pub const MAX_DELTA_Q: i8 = 63;
/// Largest native 8-bit qindex (`MAXQ_8_BITS`).
const MAX_QINDEX: i32 = 255;

/// A frame's quantizer offsets from its base qindex, in qindex units (AVM
/// `setup_quantization` with `separate_uv_delta_q = 0`: V shares U's).
/// Each is `y_dc_delta_q`, `u_ac_delta_q` or `u_dc_delta_q`, coded only
/// when the sequence enables it; all are zero on lossless frames.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantDeltas {
    /// Luma DC offset (`y_dc_delta_q`).
    pub luma_dc: i8,
    /// Chroma AC offset (`u_ac_delta_q`).
    pub chroma_ac: i8,
    /// Chroma DC offset (`u_dc_delta_q`).
    pub chroma_dc: i8,
}

impl QuantDeltas {
    /// No offsets: every coefficient of every plane at the base qindex.
    pub const NONE: Self = Self {
        luma_dc: 0,
        chroma_ac: 0,
        chroma_dc: 0,
    };

    /// Every offset, in header order.
    #[must_use]
    pub const fn all(self) -> [i8; 3] {
        [self.luma_dc, self.chroma_dc, self.chroma_ac]
    }
}

/// The AC and DC qindex of one plane (AVM `final_qindex_ac` /
/// `final_qindex_dc`): each sets its coefficients' quantizer step
/// (`get_q`); the AC one also sets the plane's deblocking thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qindices {
    /// Every coefficient but the first.
    pub ac: u8,
    /// The first (DC) coefficient.
    pub dc: u8,
}

impl Qindices {
    /// The same qindex for DC and AC.
    #[must_use]
    pub const fn uniform(qindex: u8) -> Self {
        Self {
            ac: qindex,
            dc: qindex,
        }
    }

    /// The qindex of the coefficient at raster position `position`.
    #[must_use]
    pub const fn at(self, position: usize) -> u8 {
        if position == 0 { self.dc } else { self.ac }
    }
}

impl From<u8> for Qindices {
    fn from(qindex: u8) -> Self {
        Self::uniform(qindex)
    }
}

/// The qindices of each plane of a frame; V shares U's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneQindex {
    /// Y: the frame's base qindex for AC, offset by the luma DC delta.
    pub luma: Qindices,
    /// U and V: the base plus the chroma deltas.
    pub chroma: Qindices,
}

impl PlaneQindex {
    /// A frame at `base` with `deltas`.
    #[must_use]
    pub fn new(base: u8, deltas: QuantDeltas) -> Self {
        Self {
            luma: Qindices {
                ac: base,
                dc: offset_qindex(base, deltas.luma_dc),
            },
            chroma: Qindices {
                ac: offset_qindex(base, deltas.chroma_ac),
                dc: offset_qindex(base, deltas.chroma_dc),
            },
        }
    }

    /// The qindices of `plane`.
    #[must_use]
    pub fn of(self, plane: PlaneIndex) -> Qindices {
        if plane == PlaneIndex::Y {
            self.luma
        } else {
            self.chroma
        }
    }
}

/// AVM `get_q` / `av2_q_clamped`: `base` offset by `delta`; a lossy
/// coefficient never reaches the lossless qindex zero.
fn offset_qindex(base: u8, delta: i8) -> u8 {
    if base == 0 && delta <= 0 {
        0
    } else {
        (i32::from(base) + i32::from(delta)).clamp(1, MAX_QINDEX) as u8
    }
}

/// AVM `av2_get_tx_scale`: the extra dequantization shift of transforms
/// above 256 samples (one bit) and above 1024 samples (two bits), which
/// compensates their smaller forward scaling.
fn dequant_shift(size: TxSize) -> u32 {
    match size {
        TxSize::Tx4 | TxSize::Tx8 | TxSize::Tx16 => 0,
        TxSize::Tx32 => 1,
        TxSize::Tx64 => 2,
    }
}

/// The `nc-dsp` kernel size of `size` (both follow AVM's `TX_SIZE` order).
fn kernel_size(size: TxSize) -> Av2TxSize {
    Av2TxSize::ALL[size.index()]
}

/// Whether `qindex` selects the lossless WHT, which exists only at 4×4
/// (as `DCT_DCT`), and whether the transform type exists at the size.
fn is_lossless(size: TxSize, tx_type: TxType, qindex: u8) -> Result<bool> {
    match (qindex, size, tx_type) {
        (0, TxSize::Tx4, TxType::DctDct) => Ok(true),
        (0, ..) => Err(DecodeError::Unsupported(
            "AV2 lossless coding uses only the 4x4 WHT",
        )),
        _ if !tx_type.kernels().supports(kernel_size(size)) => Err(DecodeError::Unsupported(
            "AV2 transform type does not exist at this size",
        )),
        _ => Ok(false),
    }
}

// The unscaled lifting stages are shared with VP9. AV2's normative forward
// factor is eight, and its inverse discards those three fractional bits.
fn wht_forward(input: &[i32; WHT_AREA]) -> [i32; WHT_AREA] {
    nc_dsp::reversible_wht4::forward(input).map(|value| value * UNIT_QUANT)
}

fn wht_inverse(input: &[i32; WHT_AREA]) -> [i32; WHT_AREA] {
    nc_dsp::reversible_wht4::inverse(&input.map(|value| value >> QUANT_BITS))
}

fn wht_block(values: &[i32]) -> &[i32; WHT_AREA] {
    values[..WHT_AREA]
        .try_into()
        .expect("slice has the 4x4 length")
}

/// Forward transform of a row-major `side × side` 8-bit residual into the
/// coded coefficients: the reversible WHT at qindex zero (4×4 only),
/// otherwise AVM's forward transform of type `tx_type`.
///
/// # Errors
/// Rejects residuals outside the difference range of two 8-bit samples,
/// lossless coding above 4×4 or with another type than `DCT_DCT`, and
/// types that do not exist at the size.
///
/// # Panics
/// If `residual` holds fewer than `side²` or `coefficients` fewer than
/// `size.coded_len()` values.
pub fn forward(
    dsp: &Dsp,
    (size, tx_type): (TxSize, TxType),
    residual: &[i32],
    qindex: u8,
    coefficients: &mut [i32],
) -> Result<()> {
    let area = size.width() * size.width();
    with_scratch(area, |samples: &mut [i16]| {
        for (sample, &value) in samples.iter_mut().zip(&residual[..area]) {
            *sample = i16::try_from(value)
                .ok()
                .filter(|v| i32::from(*v).abs() <= RESIDUAL_LIMIT)
                .ok_or(DecodeError::Unsupported(
                    "AV2 transform requires an 8-bit residual",
                ))?;
        }
        forward_samples(dsp, (size, tx_type), samples, qindex, coefficients)
    })
}

/// [`forward`] of a residual already known to be a difference of two 8-bit
/// samples (as `Dsp::residual` produces), without the range check or a
/// scratch copy.
///
/// # Errors
/// As [`forward`], except for the residual range.
///
/// # Panics
/// As [`forward`].
pub fn forward_samples(
    dsp: &Dsp,
    (size, tx_type): (TxSize, TxType),
    residual: &[i16],
    qindex: u8,
    coefficients: &mut [i32],
) -> Result<()> {
    if is_lossless(size, tx_type, qindex)? {
        let block: [i32; WHT_AREA] = std::array::from_fn(|i| i32::from(residual[i]));
        coefficients[..WHT_AREA].copy_from_slice(&wht_forward(&block));
    } else {
        dsp.av2_forward_typed(
            kernel_size(size),
            tx_type.kernels(),
            residual,
            size.width(),
            coefficients,
        );
    }
    Ok(())
}

/// Run `body` on `len` zeroed stack values, clearing only the smallest
/// [`SCRATCH_CLASSES`] capacity that holds them. Clearing a 64×64 maximum
/// for every transform was ~60% of AV2 encoder instructions (libc memset,
/// callgrind), since nearly all transforms are 4×4 or 8×8.
///
/// # Panics
/// If `len` exceeds the largest class.
fn with_scratch<T: Copy + Default, R>(len: usize, body: impl FnOnce(&mut [T]) -> R) -> R {
    let zero = T::default();
    match SCRATCH_CLASSES.iter().position(|&capacity| len <= capacity) {
        Some(0) => body(&mut [zero; SCRATCH_CLASSES[0]][..len]),
        Some(1) => body(&mut [zero; SCRATCH_CLASSES[1]][..len]),
        Some(2) => body(&mut [zero; SCRATCH_CLASSES[2]][..len]),
        Some(3) => body(&mut [zero; SCRATCH_CLASSES[3]][..len]),
        Some(4) => body(&mut [zero; SCRATCH_CLASSES[4]][..len]),
        _ => panic!("AV2 transform scratch of {len} values exceeds 64x64"),
    }
}

/// Quantization of coded coefficients into signed levels, the inverse of
/// [`dequantize`]'s scaling (`|c| · 2^(3 + shift) / step`, with the size's
/// dequantization shift), capped at the 20-bit level limit.
///
/// The rounding offsets (in 1/128 step, below one step; the DC coefficient
/// has its own) are added before truncation: [`NEAREST_ROUNDING_Q7`] rounds
/// to nearest, smaller values form an encoder-side deadzone (as libaom's /
/// nc-av1's rounding offset). The choice is not normative. Lossless
/// magnitudes are exact multiples of the step, so any offset keeps them
/// exact.
///
/// The AC levels come from nc-dsp's AV1 quantizer kernel, which computes
/// this same formula (`(|c| · 2^shift · 128 + step · rounding) / (step ·
/// 128)`, capped) with SIMD and an exact reciprocal; only the AV2 shift
/// (`3 +` the dequantization shift) differs from AV1's. AV2 coefficients
/// of 8-bit video stay below 2^16, far inside the kernel's numerator
/// contract. The scalar per-coefficient division it replaces was
/// `Quantizer::code`'s largest self cost on the Jetson.
///
/// Returns whether any level is nonzero (the kernel counts them as it
/// stores), so an all-zero result needs no end-of-block pass.
///
/// # Panics
/// If `coefficients` or `levels` holds fewer than `size.coded_len()` values.
pub fn quantize(
    dsp: &Dsp,
    size: TxSize,
    coefficients: &[i32],
    (qindex, rounding): (impl Into<Qindices>, Rounding),
    levels: &mut [i32],
) -> bool {
    let qindex = qindex.into();
    let area = size.coded_len();
    let levels = &mut levels[..area];
    let step = quantizer_step_q3(qindex.ac);
    let nonzero = dsp.av1_quantize_from(
        &coefficients[..area],
        &Av1Quantizer {
            dc_step: quantizer_step_q3(qindex.dc),
            ac_step: step,
            rounding_q7: rounding.ac_q7,
            max_level: MAX_LEVEL,
            log_scale: QUANT_BITS + dequant_shift(size),
        },
        levels,
    );
    let ac_nonzero = nonzero > u32::from(levels[0] != 0);
    // The kernel has one rounding offset; DC has its own.
    levels[0] = LevelQuantizer::new(size, qindex.dc, rounding.dc_q7).level(coefficients[0]);
    ac_nonzero || levels[0] != 0
}

/// Largest squared Euclidean norm of a `DCT_DCT` basis row (one coded
/// coefficient's weights over the residual samples) of each size, in
/// 1/256, as the integer kernels compute it, rounded up with a small margin
/// (measured by `dct_zero_test_bounds_every_coefficient`: 64, 64, 64.25,
/// 16.11 and 4.08).
const DCT_ROW_NORM_SQUARED_Q8: [u64; 5] = [16_512, 16_512, 16_576, 4_160, 1_056];
/// Coefficient units by which the kernels' rounding can lift a coefficient
/// above its exact linear value (measured at most 0.1).
const DCT_ROUNDING_SLACK: u64 = 2;

/// Whether every level of a `DCT_DCT` transform of `size` is zero for any
/// residual whose squared sum is `sse`, without transforming it. By
/// Cauchy–Schwarz each coefficient is at most its basis row's norm times
/// the residual's norm, plus the kernels' rounding, so when that bound is
/// below the quantizer's zero bound (the smaller of DC's and AC's) every
/// level is zero. Exact: it only skips work whose output is known.
#[must_use]
pub fn dct_quantizes_to_zero(
    size: TxSize,
    qindex: impl Into<Qindices>,
    rounding: Rounding,
    sse: u64,
) -> bool {
    let qindex = qindex.into();
    let zero_below = LevelQuantizer::new(size, qindex.dc, rounding.dc_q7)
        .zero_below
        .min(LevelQuantizer::new(size, qindex.ac, rounding.ac_q7).zero_below);
    let Some(room) = u64::from(zero_below).checked_sub(DCT_ROUNDING_SLACK) else {
        return false;
    };
    // ‖row‖·‖residual‖ < room, squared and in 1/256.
    sse * DCT_ROW_NORM_SQUARED_Q8[size.index()] < (room * room) << 8
}

/// The dequantized magnitude of a level of one (as [`dequantize`]), in
/// coefficient units.
#[must_use]
pub fn level_one_magnitude(size: TxSize, qindex: u8) -> i32 {
    dequantized_magnitude(size, qindex, 1)
}

/// The dequantized magnitude of a level of magnitude `level` (as
/// [`dequantize`]), in coefficient units.
#[must_use]
pub fn dequantized_magnitude(size: TxSize, qindex: u8, level: u32) -> i32 {
    let product = level.wrapping_mul(quantizer_step_q3(qindex).unsigned_abs()) & DEQUANT_MASK;
    ((((product + 4) >> QUANT_BITS) >> dequant_shift(size)) as i32).min(COEFF_MAX)
}

/// The dequantized magnitude of a TCQ level whose reconstruction index is
/// `half_steps` half steps (AVM `av2_read_coeffs_txb` with `tcq_mode`: the
/// index times the step, masked to 24 bits, rounded from Q3 and shifted one
/// bit more than [`dequantized_magnitude`]), in coefficient units.
#[must_use]
pub fn dequantized_half_steps(size: TxSize, qindex: u8, half_steps: u32) -> i32 {
    HalfStepDequantizer::new(size, qindex).magnitude(half_steps)
}

/// [`dequantized_half_steps`] of one transform size and qindex, with its
/// step and shift looked up once.
#[derive(Debug, Clone, Copy)]
pub struct HalfStepDequantizer {
    step_q3: u32,
    shift: u32,
}

impl HalfStepDequantizer {
    /// The dequantizer of `size` transforms at `qindex`.
    #[must_use]
    pub fn new(size: TxSize, qindex: u8) -> Self {
        Self {
            step_q3: quantizer_step_q3(qindex).unsigned_abs(),
            shift: dequant_shift(size) + 1,
        }
    }

    /// The dequantized magnitude of `half_steps` half steps.
    #[must_use]
    pub fn magnitude(self, half_steps: u32) -> i32 {
        let product = half_steps.wrapping_mul(self.step_q3) & DEQUANT_MASK;
        ((((product + 4) >> QUANT_BITS) >> self.shift) as i32).min(COEFF_MAX)
    }
}

/// A signed dequantized coefficient clipped to the normative 8-bit range.
#[must_use]
pub fn clamp_coefficient(value: i32) -> i32 {
    value.clamp(COEFF_MIN, COEFF_MAX)
}

/// Squared pixel error per squared coefficient error: AVM's transforms are
/// orthonormal times eight, divided by two per [`dequant_shift`] bit.
#[must_use]
pub fn coefficient_error_scale(size: TxSize) -> f64 {
    f64::from(1u32 << (2 * dequant_shift(size))) / f64::from(UNIT_QUANT * UNIT_QUANT)
}

/// How much the squared coefficient error of a transform changes when its
/// `coefficients` (forward transform, coded area) are coded as `levels`
/// instead of all zero: the sum over the nonzero levels of
/// `(c − dq)² − c²`, with `dq` the dequantized level exactly as
/// [`dequantize`] forms it. Times [`coefficient_error_scale`] this is the
/// change in squared pixel error the levels buy over the prediction alone,
/// up to the integer kernels' rounding and the reconstruction's clipping
/// (Parseval: the transforms are orthonormal up to that scale).
#[must_use]
pub fn coefficient_error_change(
    size: TxSize,
    levels: &[i32],
    qindex: impl Into<Qindices>,
    coefficients: &[i32],
) -> i64 {
    let qindex = qindex.into();
    let area = size.coded_len();
    let change = |position: usize, step_qindex: u8| {
        let level = levels[position];
        let magnitude = dequantized_magnitude(size, step_qindex, level.unsigned_abs());
        let dequantized = i64::from((magnitude * level.signum()).clamp(COEFF_MIN, COEFF_MAX));
        let coefficient = i64::from(coefficients[position]);
        dequantized * (dequantized - 2 * coefficient)
    };
    let mut total = 0;
    // Sparse levels: most chunks of a coded transform are all zero.
    for (chunk, values) in levels[..area].chunks(LEVEL_CHUNK).enumerate() {
        if values.iter().fold(0, |bits, &c| bits | c) == 0 {
            continue;
        }
        for (i, _) in values.iter().enumerate().filter(|&(_, &l)| l != 0) {
            let position = chunk * LEVEL_CHUNK + i;
            // DC has its own step, as in `dequantize`.
            let step_qindex = if position == 0 { qindex.dc } else { qindex.ac };
            total += change(position, step_qindex);
        }
    }
    total
}

/// Mean squared error per sample that the integer kernels' rounding adds
/// to (or, negative, removes from) a coded transform's reconstruction
/// beyond its coefficient error, by size: the bias of
/// [`coefficient_error_change`]'s estimate, measured on uniform residuals
/// at qindex 40..130 (about constant across quantizers). It matters only
/// at fine quantizers, where the coding error itself is a fraction of a
/// sample (`bodycam_airport`, ~50 dB). 64-point transforms are not
/// estimated (their estimate is ~10% low: not an orthogonal projection).
const KERNEL_ROUNDING_ERROR: [f64; 5] = [-0.17, -0.05, 0.04, 0.15, 0.0];

/// The squared pixel error a coded transform's reconstruction adds to the
/// prediction's, estimated from its coefficients: Parseval over the
/// levels' quantization error ([`coefficient_error_change`] times
/// [`coefficient_error_scale`]) plus the kernels' rounding
/// ([`KERNEL_ROUNDING_ERROR`]).
#[must_use]
pub fn estimated_error_change(
    size: TxSize,
    levels: &[i32],
    qindex: impl Into<Qindices>,
    coefficients: &[i32],
) -> f64 {
    estimated_error_from_change(
        size,
        coefficient_error_change(size, levels, qindex, coefficients),
    )
}

/// [`estimated_error_change`] from a coefficient error change already
/// formed (for levels another dequantizer reconstructs).
#[must_use]
pub fn estimated_error_from_change(size: TxSize, change: i64) -> f64 {
    let area = (size.width() * size.width()) as f64;
    change as f64 * coefficient_error_scale(size) + KERNEL_ROUNDING_ERROR[size.index()] * area
}

/// [`quantize`]'s scaling for one transform size and quantizer.
#[derive(Debug, Clone, Copy)]
struct LevelQuantizer {
    shift: u32,
    divisor: i64,
    offset: i64,
    /// Magnitudes below this quantize to zero: the exact condition
    /// `scaled + offset < divisor` solved for the magnitude, so that the
    /// common zero level costs one comparison (most coefficients of a
    /// low-rate residual are zero).
    zero_below: u32,
}

impl LevelQuantizer {
    fn new(size: TxSize, qindex: u8, rounding_q7: i32) -> Self {
        debug_assert!((0..ROUNDING_Q7_ONE).contains(&rounding_q7));
        let step = i64::from(quantizer_step_q3(qindex));
        let divisor = step * i64::from(ROUNDING_Q7_ONE);
        let offset = step * i64::from(rounding_q7);
        let shift = QUANT_BITS + dequant_shift(size);
        // `scaled = magnitude · unit` is below `divisor − offset` exactly when
        // the magnitude is below the ceiling of their quotient.
        let unit = i64::from(ROUNDING_Q7_ONE) << shift;
        let zero_below = (divisor - offset + unit - 1) / unit;
        Self {
            shift,
            divisor,
            offset,
            zero_below: u32::try_from(zero_below).expect("bounded by the quantizer step"),
        }
    }

    fn level(self, value: i32) -> i32 {
        if value.unsigned_abs() < self.zero_below {
            return 0;
        }
        let scaled = (i64::from(value.unsigned_abs()) << self.shift) * i64::from(ROUNDING_Q7_ONE);
        let magnitude = (scaled + self.offset) / self.divisor;
        magnitude.min(i64::from(MAX_LEVEL)) as i32 * value.signum()
    }
}

/// [`forward`] (`DCT_DCT`) then [`quantize`]: the coded levels of a residual.
///
/// # Errors
/// As [`forward`].
///
/// # Panics
/// As [`forward`], with `levels` in place of `coefficients`.
pub fn quantize_residual(
    dsp: &Dsp,
    size: TxSize,
    residual: &[i32],
    (qindex, rounding_q7): (u8, i32),
    levels: &mut [i32],
) -> Result<()> {
    with_scratch(size.coded_len(), |coefficients: &mut [i32]| {
        forward(dsp, (size, TxType::DctDct), residual, qindex, coefficients)?;
        quantize(
            dsp,
            size,
            coefficients,
            (qindex, Rounding::uniform(rounding_q7)),
            levels,
        );
        Ok(())
    })
}

/// Apply AV2's no-TCQ, no-matrix dequantization to coded levels.
///
/// Magnitudes are multiplied, masked to 24 bits, rounded from Q3, shifted by
/// the size's dequantization shift (AVM `av2_read_coeffs_txb`) and clipped
/// with their sign to the normative 8-bit coefficient range.
///
/// # Errors
/// Rejects coded levels exceeding AV2's 20-bit limit, including `i32::MIN`.
///
/// # Panics
/// If `levels` or `out` holds fewer than `size.coded_len()` values.
pub fn dequantize(
    size: TxSize,
    levels: &[i32],
    qindex: impl Into<Qindices>,
    out: &mut [i32],
) -> Result<()> {
    let qindex = qindex.into();
    let area = size.coded_len();
    let levels = &levels[..area];
    check_level_range(levels)?;
    let step = quantizer_step_q3(qindex.ac).unsigned_abs();
    let shift = dequant_shift(size);
    // Sparse levels: all-zero chunks dequantize to zero without arithmetic.
    for (coefficients, levels) in out[..area]
        .as_chunks_mut::<LEVEL_CHUNK>()
        .0
        .iter_mut()
        .zip(levels.as_chunks::<LEVEL_CHUNK>().0)
    {
        if levels.iter().fold(0, |bits, &c| bits | c) == 0 {
            coefficients.fill(0);
            continue;
        }
        for (coefficient, &value) in coefficients.iter_mut().zip(levels) {
            *coefficient = dequantize_level(value, step, shift);
        }
    }
    if qindex.dc != qindex.ac {
        out[0] = dequantize_dc(size, levels[0], qindex);
    }
    Ok(())
}

/// One AC level dequantized with the step `step` (Q3) and `shift`: only
/// the low 24 bits of the product survive the mask, so a wrapping 32-bit
/// product is exact (and vectorizes, unlike 64 bits).
#[inline]
fn dequantize_level(value: i32, step: u32, shift: u32) -> i32 {
    let product = value.unsigned_abs().wrapping_mul(step) & DEQUANT_MASK;
    let magnitude = (((product + 4) >> QUANT_BITS) >> shift) as i32;
    (magnitude * value.signum()).clamp(COEFF_MIN, COEFF_MAX)
}

/// The DC coefficient of `level`: DC has its own step (`dequant[pos != 0]`
/// in AVM `av2_read_coeffs_txb`).
fn dequantize_dc(size: TxSize, level: i32, qindex: Qindices) -> i32 {
    if qindex.dc == qindex.ac {
        dequantize_level(
            level,
            quantizer_step_q3(qindex.ac).unsigned_abs(),
            dequant_shift(size),
        )
    } else {
        let magnitude = dequantized_magnitude(size, qindex.dc, level.unsigned_abs());
        (magnitude * level.signum()).clamp(COEFF_MIN, COEFF_MAX)
    }
}

/// [`dequantize`] of the top-left `side` x `side` square of the coded
/// levels only, the rest of `out` left as it is: levels that end within
/// the square's corner (the end of block bounds them) need nothing else,
/// and a small corner of a large transform is a fraction of its area.
fn dequantize_square(
    size: TxSize,
    levels: &[i32],
    qindex: Qindices,
    out: &mut [i32],
    side: usize,
) -> Result<()> {
    let coded = size.coded_width();
    if side == coded {
        return dequantize(size, levels, qindex, out);
    }
    let step = quantizer_step_q3(qindex.ac).unsigned_abs();
    let shift = dequant_shift(size);
    for (out, levels) in out
        .chunks_exact_mut(coded)
        .zip(levels.chunks_exact(coded))
        .take(side)
    {
        let levels = &levels[..side];
        check_level_range(levels)?;
        for (coefficient, &value) in out[..side].iter_mut().zip(levels) {
            *coefficient = dequantize_level(value, step, shift);
        }
    }
    out[0] = dequantize_dc(size, levels[0], qindex);
    Ok(())
}

/// Reject levels past AV2's 20-bit limit, including `i32::MIN`.
fn check_level_range(levels: &[i32]) -> Result<()> {
    // An OR of the magnitudes exceeds the all-ones 20-bit limit exactly when
    // one magnitude does (and vectorizes, unlike `any`).
    let bits = levels.iter().fold(0, |bits, c| bits | c.unsigned_abs());
    if bits > MAX_LEVEL.unsigned_abs() {
        return Err(DecodeError::Malformed(
            "AV2 coefficient exceeds 20-bit level range",
        ));
    }
    Ok(())
}

/// Reconstruct the row-major `side × side` residual of coded levels of
/// transform type `tx_type`; the caller adds its predictor and clips to
/// 0..255.
///
/// # Errors
/// Rejects coded levels outside the normative bounded range, and the
/// size/type/quantizer combinations [`forward`] rejects.
///
/// # Panics
/// If `levels` holds fewer than `size.coded_len()` or `residual` fewer than
/// `side²` values.
pub fn reconstruct(
    dsp: &Dsp,
    transform: (TxSize, TxType),
    levels: &[i32],
    qindex: impl Into<Qindices>,
    residual: &mut [i32],
) -> Result<()> {
    let eob = crate::coeff::end_of_block(&levels[..transform.0.coded_len()], transform.0);
    reconstruct_with(
        dsp,
        transform,
        (levels, eob, Dequantizer::Plain),
        qindex,
        residual,
    )
}

/// How coded levels map to coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dequantizer {
    /// Each level times its step ([`dequantize`]).
    Plain,
    /// The trellis-coded quantizers of a TCQ frame's lossy luma
    /// ([`crate::coeff::tcq::dequantize`]).
    Tcq,
}

/// [`reconstruct`] of levels dequantized by `dequantizer`, whose end of
/// block (in the coefficient scan) is `eob`.
///
/// # Errors
/// As [`reconstruct`].
///
/// # Panics
/// As [`reconstruct`].
pub fn reconstruct_with(
    dsp: &Dsp,
    transform: (TxSize, TxType),
    (levels, eob, dequantizer): (&[i32], usize, Dequantizer),
    qindex: impl Into<Qindices>,
    residual: &mut [i32],
) -> Result<()> {
    let qindex = qindex.into();
    with_scratch(transform.0.coded_len(), |coefficients: &mut [i32]| {
        reconstruct_into(
            dsp,
            transform,
            (levels, eob, dequantizer),
            qindex,
            coefficients,
            residual,
        )
    })
}

/// [`reconstruct`] of levels whose end of block `eob` (in the coefficient
/// scan) is known, with the caller's dequantization scratch `coefficients`
/// (at least `size.coded_len()` values, overwritten), so a coder
/// reconstructing many transforms does not clear a stack scratch for each.
/// The end of block bounds the nonzero corner, whose zero rows the inverse
/// DCT skips.
///
/// # Errors
/// As [`reconstruct`].
///
/// # Panics
/// As [`reconstruct`], or if `coefficients` is too short.
pub fn reconstruct_into(
    dsp: &Dsp,
    (size, tx_type): (TxSize, TxType),
    (levels, eob, dequantizer): (&[i32], usize, Dequantizer),
    qindex: impl Into<Qindices>,
    coefficients: &mut [i32],
    residual: &mut [i32],
) -> Result<()> {
    let qindex = qindex.into();
    let lossless = is_lossless(size, tx_type, qindex.ac)?;
    let coefficients = &mut coefficients[..size.coded_len()];
    let corner = nonzero_corner(size, eob);
    match dequantizer {
        Dequantizer::Tcq if !lossless => {
            check_level_range(&levels[..size.coded_len()])?;
            crate::coeff::tcq::dequantize(size, (levels, eob), qindex, coefficients);
        }
        // The DCT kernels read only the corner's square of lane tiles.
        _ if tx_type == TxType::DctDct && !lossless => {
            let read = kernel_size(size).inverse_read_side(corner);
            dequantize_square(size, levels, qindex, coefficients, read)?;
        }
        _ => dequantize(size, levels, qindex, coefficients)?,
    }
    if lossless {
        residual[..WHT_AREA].copy_from_slice(&wht_inverse(wht_block(coefficients)));
    } else {
        dsp.av2_inverse_typed_corner(
            kernel_size(size),
            tx_type.kernels(),
            coefficients,
            corner,
            residual,
        );
    }
    Ok(())
}

/// The residual of a transform whose only nonzero level is the DC (end of
/// block 1), when it is one value for every sample: a lossy `DCT_DCT` with
/// plain dequantization ([`Av2TxSize::inverse_dc_residual`]). The DC is
/// dequantized exactly as [`dequantize`] does. `None` otherwise, and for a
/// level outside the 20-bit range, which the full path rejects. A quarter
/// of the encoder's reconstructions are DC only (bodycam 720p CBR 600), and
/// this skips their dequantization, inverse DCT and residual raster.
#[must_use]
pub fn dc_only_residual(
    (size, tx_type): (TxSize, TxType),
    (levels, eob, dequantizer): (&[i32], usize, Dequantizer),
    qindex: impl Into<Qindices>,
) -> Option<i32> {
    let qindex = qindex.into();
    let level = levels[0];
    let applies = eob == 1
        && tx_type == TxType::DctDct
        && dequantizer == Dequantizer::Plain
        && level.unsigned_abs() <= MAX_LEVEL.unsigned_abs()
        && matches!(is_lossless(size, tx_type, qindex.ac), Ok(false));
    if !applies {
        return None;
    }
    Some(kernel_size(size).inverse_dc_residual(dequantize_dc(size, level, qindex)))
}

/// The side of the top-left corner holding every nonzero coefficient of a
/// block whose end of block is `eob`: the default scans walk anti-diagonals
/// from the DC, so no coded position lies past the last one's diagonal.
fn nonzero_corner(size: TxSize, eob: usize) -> usize {
    let Some(last) = eob.checked_sub(1) else {
        return 0;
    };
    let pos = usize::from(crate::coeff::scan(size)[last]);
    let log2 = size.coded_log2();
    let diagonal = (pos >> log2) + (pos & ((1 << log2) - 1));
    (diagonal + 1).min(size.coded_width())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_dsp::Isa;
    /// Coded coefficients of the largest transform (64×64 codes 32×32).
    const MAX_CODED_AREA: usize = 32 * 32;

    /// The `DCT_DCT` basis matrix of `size` as the kernels compute it, row
    /// per coded coefficient, from scaled impulse responses.
    fn dct_rows(dsp: &Dsp, size: TxSize) -> Vec<Vec<f64>> {
        const IMPULSE: i16 = 256;
        let area = size.width() * size.width();
        let mut rows = vec![vec![0.0; area]; size.coded_len()];
        let (mut impulse, mut out) = (vec![0; area], vec![0; size.coded_len()]);
        for i in 0..area {
            impulse.fill(0);
            impulse[i] = IMPULSE;
            forward_samples(dsp, (size, TxType::DctDct), &impulse, 1, &mut out).unwrap();
            for (row, &value) in rows.iter_mut().zip(&out) {
                row[i] = f64::from(value) / f64::from(IMPULSE);
            }
        }
        rows
    }

    /// The kernel route equals the per-coefficient reference division
    /// (`LevelQuantizer::level`) for every size, a spread of quantizers and
    /// rounding offsets, and magnitudes up to the 8-bit coefficient range.
    #[test]
    fn quantize_matches_reference_division() {
        let dsp = Dsp::new();
        let mut seed = 0x9e37_79b9_u32;
        for size in TxSize::ALL {
            let area = size.coded_len();
            for qindex in (0..=255u8).step_by(7) {
                for (dc_q7, ac_q7) in [(64, 64), (48, 32), (0, 0), (127, 100)] {
                    let rounding = Rounding { dc_q7, ac_q7 };
                    let coefficients: Vec<i32> = (0..area)
                        .map(|i| {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            let limit = if i % 5 == 0 { 1 << 16 } else { 1 << 9 };
                            (seed % (2 * limit + 1)) as i32 - limit as i32
                        })
                        .collect();
                    let mut levels = vec![0; area];
                    quantize(&dsp, size, &coefficients, (qindex, rounding), &mut levels);
                    let dc = LevelQuantizer::new(size, qindex, dc_q7);
                    let ac = LevelQuantizer::new(size, qindex, ac_q7);
                    for (i, (&c, &level)) in coefficients.iter().zip(&levels).enumerate() {
                        let reference = if i == 0 { dc.level(c) } else { ac.level(c) };
                        assert_eq!(
                            level, reference,
                            "{size:?} q{qindex} {rounding:?} [{i}] {c}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn coefficient_error_change_tracks_the_reconstruction() {
        // The transform-domain estimate of a coded transform's squared
        // error stays within a few percent of the reconstruction's (12% at
        // 64x64 on white noise, whose energy sits in the uncoded half), on
        // smooth and noisy residuals at fine and coarse quantizers (no
        // clipping: the prediction sits mid-range).
        let dsp = Dsp::new();
        let mut seed = 777_u32;
        for size in TxSize::ALL {
            let side = size.width();
            let area = side * side;
            for (qindex, amplitude) in [(60_u8, 40_i32), (140, 40), (200, 90), (140, 6)] {
                let residual: Vec<i16> = (0..area)
                    .map(|i| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let noise = (seed >> 8) as i32 % (2 * amplitude + 1) - amplitude;
                        let ramp = (i % side) as i32 * amplitude / side as i32;
                        i32::midpoint(noise, ramp) as i16
                    })
                    .collect();
                let mut coefficients = vec![0; size.coded_len()];
                forward_samples(
                    &dsp,
                    (size, TxType::DctDct),
                    &residual,
                    qindex,
                    &mut coefficients,
                )
                .unwrap();
                let mut levels = vec![0; size.coded_len()];
                let rounding = Rounding::uniform(NEAREST_ROUNDING_Q7);
                quantize(&dsp, size, &coefficients, (qindex, rounding), &mut levels);
                let eob = levels.iter().rposition(|&l| l != 0).map_or(0, |p| p + 1);
                let mut dequantized = vec![0; size.coded_len()];
                let mut inverse = vec![0; area];
                reconstruct_into(
                    &dsp,
                    (size, TxType::DctDct),
                    (
                        &levels,
                        if eob == 0 { 0 } else { size.coded_len() },
                        Dequantizer::Plain,
                    ),
                    qindex,
                    &mut dequantized,
                    &mut inverse,
                )
                .unwrap();
                let sse =
                    |error: &dyn Fn(usize) -> i64| (0..area).map(|i| error(i).pow(2)).sum::<i64>();
                let predicted = sse(&|i| i64::from(residual[i]));
                let actual = if eob == 0 {
                    predicted
                } else {
                    sse(&|i| i64::from(residual[i]) - i64::from(inverse[i]))
                };
                let estimate = if eob == 0 {
                    predicted as f64
                } else {
                    predicted as f64 + estimated_error_change(size, &levels, qindex, &coefficients)
                };
                // 64-point transforms code only their low 32x32 frequencies,
                // and the kernels' rounding is about one per sample.
                let relative = if size == TxSize::Tx64 { 0.12 } else { 0.05 };
                let tolerance = relative * actual as f64 + 2.0 * area as f64;
                assert!(
                    (estimate - actual as f64).abs() <= tolerance,
                    "{size:?} q{qindex} a{amplitude}: estimate {estimate} actual {actual}"
                );
            }
        }
    }

    #[test]
    fn dct_zero_test_bounds_every_coefficient() {
        let dsp = Dsp::new();
        for size in TxSize::ALL {
            let area = size.width() * size.width();
            let rows = dct_rows(&dsp, size);
            let norm_squared = rows
                .iter()
                .map(|row| row.iter().map(|t| t * t).sum::<f64>())
                .fold(0.0, f64::max);
            let bound_squared = DCT_ROW_NORM_SQUARED_Q8[size.index()] as f64 / 256.0;
            assert!(norm_squared <= bound_squared, "{size:?}: {norm_squared}");
            // Residuals aligned with each basis row reach the norm bound;
            // random ones probe the kernels' rounding.
            let mut residuals: Vec<Vec<i16>> = Vec::new();
            for row in &rows {
                let peak = row.iter().map(|t| t.abs()).fold(0.0, f64::max);
                for amplitude in [255.0, 17.0, 3.0] {
                    residuals.push(
                        row.iter()
                            .map(|t| (amplitude * t / peak).round() as i16)
                            .collect(),
                    );
                }
            }
            let mut seed = 12_345_u32;
            for trial in 0..400 {
                let amplitude = [255, 8, 3, 1][trial % 4];
                residuals.push(
                    (0..area)
                        .map(|_| {
                            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            ((seed >> 8) % (2 * amplitude + 1)) as i16 - amplitude as i16
                        })
                        .collect(),
                );
            }
            let mut out = vec![0; size.coded_len()];
            for residual in &residuals {
                forward_samples(&dsp, (size, TxType::DctDct), residual, 1, &mut out).unwrap();
                let sse: f64 = residual.iter().map(|&r| f64::from(r) * f64::from(r)).sum();
                let limit = (sse * bound_squared).sqrt() + DCT_ROUNDING_SLACK as f64;
                assert!(
                    out.iter().all(|c| f64::from(c.unsigned_abs()) <= limit),
                    "{size:?}"
                );
            }
        }
    }

    #[test]
    fn dct_zero_test_implies_zero_levels() {
        let dsp = Dsp::new();
        let rounding = Rounding {
            dc_q7: 48,
            ac_q7: 32,
        };
        let mut seed = 7_u32;
        for size in TxSize::ALL {
            let area = size.width() * size.width();
            let (mut coefficients, mut levels) =
                (vec![0; size.coded_len()], vec![0; size.coded_len()]);
            for qindex in [40, 120, 200, 255] {
                for amplitude in [1, 2, 4, 8, 16] {
                    let residual: Vec<i16> = (0..area)
                        .map(|_| {
                            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            ((seed >> 8) % (2 * amplitude + 1)) as i16 - amplitude as i16
                        })
                        .collect();
                    let sse = residual
                        .iter()
                        .map(|&r| u64::from(r.unsigned_abs()).pow(2))
                        .sum();
                    if dct_quantizes_to_zero(size, qindex, rounding, sse) {
                        forward_samples(
                            &dsp,
                            (size, TxType::DctDct),
                            &residual,
                            qindex,
                            &mut coefficients,
                        )
                        .unwrap();
                        quantize(&dsp, size, &coefficients, (qindex, rounding), &mut levels);
                        assert!(levels.iter().all(|&l| l == 0), "{size:?} q{qindex}");
                    }
                }
            }
        }
    }

    #[test]
    fn plane_qindices_follow_av2_q_clamped() {
        let chroma = |base, delta| {
            PlaneQindex::new(
                base,
                QuantDeltas {
                    chroma_ac: delta,
                    ..QuantDeltas::NONE
                },
            )
            .chroma
            .ac
        };
        assert_eq!(chroma(120, -20), 100);
        assert_eq!(chroma(10, -20), 1);
        assert_eq!(chroma(250, 20), 255);
        assert_eq!(chroma(0, -5), 0);
        let deltas = QuantDeltas {
            luma_dc: -12,
            chroma_ac: -16,
            chroma_dc: -30,
        };
        assert_eq!(
            PlaneQindex::new(120, deltas),
            PlaneQindex {
                luma: Qindices { ac: 120, dc: 108 },
                chroma: Qindices { ac: 104, dc: 90 },
            }
        );
        assert_eq!(
            PlaneQindex::new(0, deltas),
            PlaneQindex {
                luma: Qindices::uniform(0),
                chroma: Qindices::uniform(0),
            }
        );
    }

    /// Every kernel set this machine runs: the scalar reference and the best.
    fn dsps() -> [Dsp; 2] {
        [Dsp::with_isa(Isa::Scalar).unwrap(), Dsp::new()]
    }

    fn levels_of(dsp: &Dsp, size: TxSize, residual: &[i32], qindex: u8) -> Vec<i32> {
        let mut levels = vec![0; size.coded_len()];
        quantize_residual(
            dsp,
            size,
            residual,
            (qindex, NEAREST_ROUNDING_Q7),
            &mut levels,
        )
        .unwrap();
        levels
    }

    fn reconstruct_of(dsp: &Dsp, size: TxSize, levels: &[i32], qindex: u8) -> Vec<i32> {
        let mut residual = vec![0; size.width() * size.width()];
        reconstruct(dsp, (size, TxType::DctDct), levels, qindex, &mut residual).unwrap();
        residual
    }

    fn hash(hash: &mut u64, values: &[i32]) {
        for &value in values {
            *hash = (*hash ^ u64::from(value as u32)).wrapping_mul(1_099_511_628_211);
        }
    }

    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;

    /// `tools/codegen/check_av2_dct.py` (original AVM `fwd_txfm_c`,
    /// `inv_txfm_c` and decoder dequant), by size: forward, levels, dequant,
    /// reconstruction, clipped extreme levels, all-qindex dequant. The 8×8
    /// row equals the earlier standalone TX8 oracle's hashes.
    const DCT_HASHES: [[u64; 6]; 5] = [
        [
            9_102_275_776_819_511_632,
            15_382_130_019_203_076_871,
            14_976_315_994_595_328_546,
            5_848_680_624_197_813_388,
            5_455_307_702_738_382_849,
            530_158_922_434_143_729,
        ],
        [
            7_394_461_545_706_306_953,
            13_470_341_489_985_369_725,
            17_945_927_254_867_994_320,
            834_658_859_093_866_737,
            11_172_653_938_983_297_178,
            530_158_922_434_143_729,
        ],
        [
            8_949_498_047_874_491_295,
            926_883_795_850_287_007,
            10_146_913_358_275_822_781,
            2_519_957_390_725_468_903,
            3_602_894_678_082_665_994,
            530_158_922_434_143_729,
        ],
        [
            5_327_027_826_085_476_935,
            12_288_279_349_139_452_451,
            5_432_341_847_771_408_680,
            13_584_179_198_312_964_650,
            2_348_052_779_513_892_033,
            2_331_820_005_946_854_674,
        ],
        [
            4_068_628_199_328_138_521,
            4_519_847_499_956_397_172,
            959_808_737_485_194_618,
            460_946_408_838_823_053,
            2_704_284_973_351_323_761,
            6_701_961_739_401_030_818,
        ],
    ];

    /// The oracle's residual patterns: zero, ±255 flat, a ±255 alternation,
    /// single first/last impulses and LCG noise.
    fn pattern(n: usize, qindex: u8, kind: u32) -> Vec<i32> {
        let mut state = u32::from(qindex) * 7919 + kind * 104_729;
        (0..n * n)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match kind {
                    1 => 255,
                    2 => -255,
                    3 => [-255, 255][i % 2],
                    4 => i32::from(i == 0) * 255,
                    5 => i32::from(i == n * n - 1) * -255,
                    6 | 7 => (state % 511) as i32 - 255,
                    _ => 0,
                }
            })
            .collect()
    }

    #[test]
    fn dct_matches_original_avm_all_sizes_and_quantizers() {
        for dsp in dsps() {
            for (size, expected) in TxSize::ALL.into_iter().zip(DCT_HASHES) {
                let (n, area) = (size.width(), size.coded_len());
                let mut hashes = [FNV_OFFSET; 6];
                let mut coefficients = vec![0; area];
                let mut dequantized = vec![0; area];
                for q in 1..=255u8 {
                    for kind in 0..8 {
                        let residual = pattern(n, q, kind);
                        forward(
                            &dsp,
                            (size, TxType::DctDct),
                            &residual,
                            q,
                            &mut coefficients,
                        )
                        .unwrap();
                        hash(&mut hashes[0], &coefficients);
                        let levels = levels_of(&dsp, size, &residual, q);
                        hash(&mut hashes[1], &levels);
                        dequantize(size, &levels, q, &mut dequantized).unwrap();
                        hash(&mut hashes[2], &dequantized);
                        hash(&mut hashes[3], &reconstruct_of(&dsp, size, &levels, q));
                    }
                    let extreme: Vec<i32> = (0..area)
                        .map(|i| {
                            let magnitude = [1_048_575, 32767, 1][i % 3];
                            if i % 2 == 1 { -magnitude } else { magnitude }
                        })
                        .collect();
                    hash(&mut hashes[4], &reconstruct_of(&dsp, size, &extreme, q));
                }
                let mut one = [0];
                for q in 0..=255u8 {
                    for i in 0..16 {
                        let magnitude = [0, 1, 2, 17, 1023, 32767, 65535, 1_048_575][i / 2];
                        let level = if i % 2 == 1 { -magnitude } else { magnitude };
                        let mut levels = [0; MAX_CODED_AREA];
                        levels[0] = level;
                        let mut out = [0; MAX_CODED_AREA];
                        dequantize(size, &levels, q, &mut out).unwrap();
                        one[0] = out[0];
                        hash(&mut hashes[5], &one);
                    }
                }
                assert_eq!(hashes, expected, "{size:?} on {:?}", dsp.isa());
            }
        }
    }

    /// `tools/codegen/check_av2_tx_types.py` (original AVM `fwd_txfm_c` and
    /// `inv_txfm_c` with the identity, ADST and flipped-ADST kernels): for
    /// each (size index, AVM `TX_TYPE`) pair whose kernels exist, at
    /// qindex 1, 18, .., 256 − 17 and the eight [`pattern`]s: forward,
    /// nearest-step levels, dequant, reconstruction, clipped extreme levels.
    const TX_TYPE_HASHES: [(usize, usize, [u64; 5]); 52] = [
        (
            0,
            0,
            [
                6_231_623_349_171_855_240,
                14_516_010_950_994_835_953,
                689_700_345_831_353_772,
                13_500_632_621_784_509_920,
                6_839_436_290_488_901_026,
            ],
        ),
        (
            0,
            1,
            [
                9_936_487_579_174_670_545,
                16_083_041_983_298_187_192,
                5_606_922_661_167_560_835,
                14_872_782_850_455_680_290,
                2_337_783_282_052_740_579,
            ],
        ),
        (
            0,
            2,
            [
                11_367_487_336_338_096_988,
                16_200_688_255_157_056_846,
                8_077_078_261_937_659_226,
                5_029_065_079_914_447_519,
                9_537_875_232_535_843_052,
            ],
        ),
        (
            0,
            3,
            [
                1_849_523_447_002_156_374,
                9_728_714_348_809_294_070,
                14_183_025_807_276_364_505,
                11_828_696_139_320_654_684,
                10_004_138_652_553_990_938,
            ],
        ),
        (
            0,
            4,
            [
                10_065_503_513_915_778_189,
                11_205_022_452_823_190_526,
                6_333_019_320_578_215_020,
                13_656_250_766_597_680_998,
                5_510_061_229_547_120_515,
            ],
        ),
        (
            0,
            5,
            [
                15_727_253_143_476_922_081,
                9_740_483_025_989_507_422,
                6_019_072_612_820_948_843,
                14_523_856_997_731_637_086,
                4_405_350_953_231_564_170,
            ],
        ),
        (
            0,
            6,
            [
                3_407_596_556_927_851_819,
                2_671_754_439_534_177_967,
                8_623_140_996_088_769_096,
                6_167_534_977_547_701_933,
                12_493_448_954_768_079_542,
            ],
        ),
        (
            0,
            7,
            [
                5_063_531_850_343_839_830,
                8_740_761_607_987_325_446,
                18_279_479_068_715_811_232,
                17_007_003_733_777_058_125,
                16_904_886_203_720_583_145,
            ],
        ),
        (
            0,
            8,
            [
                7_475_202_721_599_507_207,
                16_144_234_221_321_548_607,
                4_802_191_160_570_290_241,
                1_205_789_971_756_576_065,
                6_188_572_286_830_690_490,
            ],
        ),
        (
            0,
            9,
            [
                6_594_712_573_633_600_693,
                2_868_082_218_066_299_219,
                2_995_542_589_314_221_146,
                16_295_472_144_888_761_012,
                6_591_245_599_443_831_195,
            ],
        ),
        (
            0,
            10,
            [
                3_201_113_982_885_224_635,
                15_937_892_788_883_157_034,
                6_607_853_812_315_793_298,
                12_535_516_448_521_712_492,
                10_790_198_354_843_233_390,
            ],
        ),
        (
            0,
            11,
            [
                7_441_396_813_519_784_959,
                15_234_623_831_938_624_891,
                18_446_108_167_630_112_317,
                6_075_473_699_764_304_351,
                7_423_799_037_254_912_439,
            ],
        ),
        (
            0,
            12,
            [
                83_791_991_366_367_206,
                258_121_205_432_088_233,
                15_897_401_891_300_794_440,
                12_796_099_973_871_987_252,
                4_210_820_051_428_743_840,
            ],
        ),
        (
            0,
            13,
            [
                18_163_252_875_962_711_783,
                11_297_081_247_184_879_733,
                9_999_937_569_292_282_728,
                12_981_527_865_406_236_282,
                8_576_328_452_794_466_755,
            ],
        ),
        (
            0,
            14,
            [
                10_081_821_704_566_352_675,
                3_978_706_686_252_074_558,
                4_665_781_705_472_009_497,
                4_737_375_230_892_789_220,
                5_113_170_790_419_408_617,
            ],
        ),
        (
            0,
            15,
            [
                8_605_286_915_680_177_264,
                12_054_733_818_522_361_916,
                14_005_394_121_125_878_016,
                7_281_468_717_147_881_616,
                2_485_396_241_909_558_232,
            ],
        ),
        (
            1,
            0,
            [
                12_368_564_192_712_680_291,
                14_282_855_462_013_700_128,
                4_246_283_970_922_706_631,
                12_775_715_024_812_532_620,
                8_301_031_412_150_929_572,
            ],
        ),
        (
            1,
            1,
            [
                5_245_627_562_014_150_139,
                1_436_018_393_965_086_855,
                2_632_214_495_856_932_963,
                10_936_989_430_156_923_468,
                11_104_376_354_545_240_393,
            ],
        ),
        (
            1,
            2,
            [
                15_411_146_789_191_508_881,
                1_647_623_516_276_602_768,
                4_961_622_511_427_296_924,
                5_666_760_087_893_839_548,
                5_880_242_770_139_972_229,
            ],
        ),
        (
            1,
            3,
            [
                5_523_009_679_013_998_810,
                12_723_965_332_190_308_609,
                14_271_401_811_802_856_157,
                4_839_704_773_878_087_775,
                5_088_054_081_169_313_645,
            ],
        ),
        (
            1,
            4,
            [
                11_238_969_613_497_830_515,
                4_012_646_660_160_327_894,
                7_163_733_437_312_889_414,
                9_294_938_632_090_428_298,
                11_858_890_034_634_481_281,
            ],
        ),
        (
            1,
            5,
            [
                15_743_702_409_874_682_636,
                9_502_334_777_933_831_304,
                5_672_300_903_146_757_012,
                11_877_262_730_858_524_286,
                2_193_427_878_169_001_273,
            ],
        ),
        (
            1,
            6,
            [
                5_968_136_687_429_125_989,
                3_666_664_846_920_298_844,
                1_151_063_073_475_533_596,
                10_888_728_963_983_757_106,
                8_922_770_465_834_496_573,
            ],
        ),
        (
            1,
            7,
            [
                6_317_418_480_968_050_034,
                6_913_944_451_348_353_654,
                18_200_534_757_038_125_524,
                9_315_785_020_500_207_119,
                4_477_233_703_115_579_777,
            ],
        ),
        (
            1,
            8,
            [
                12_771_284_945_372_090_650,
                974_051_328_478_413_020,
                7_420_432_693_598_124_743,
                3_963_608_699_436_504_117,
                214_792_882_830_504_793,
            ],
        ),
        (
            1,
            9,
            [
                6_051_309_211_163_631_275,
                1_486_988_222_499_507_421,
                14_703_294_015_375_725_532,
                4_996_744_852_163_451_165,
                4_583_361_723_003_180_515,
            ],
        ),
        (
            1,
            10,
            [
                16_048_001_151_213_520_343,
                13_159_715_717_171_014_543,
                11_196_540_370_562_896_618,
                16_760_843_486_733_384_111,
                9_898_527_948_181_066_924,
            ],
        ),
        (
            1,
            11,
            [
                2_403_744_062_840_580_330,
                5_558_936_750_371_073_503,
                7_582_824_219_970_042_177,
                1_165_522_925_085_563_336,
                14_019_110_315_731_894_531,
            ],
        ),
        (
            1,
            12,
            [
                3_727_774_368_577_464_776,
                6_907_314_370_584_620_038,
                6_820_955_015_129_563_627,
                2_831_423_824_666_086_034,
                5_772_429_594_897_905_233,
            ],
        ),
        (
            1,
            13,
            [
                11_486_642_417_132_783_575,
                15_354_776_172_421_528_143,
                5_376_952_067_460_870_536,
                3_105_344_258_681_611_037,
                10_673_855_292_848_194_818,
            ],
        ),
        (
            1,
            14,
            [
                14_850_783_397_594_483_344,
                16_066_303_932_828_767_523,
                3_781_861_037_923_842_031,
                4_549_173_030_562_456_238,
                17_872_494_817_869_103_229,
            ],
        ),
        (
            1,
            15,
            [
                8_760_233_444_834_190_837,
                11_557_035_021_340_774_260,
                7_338_996_934_200_302_189,
                12_184_132_499_408_688_684,
                12_092_333_020_377_385_796,
            ],
        ),
        (
            2,
            0,
            [
                1_771_977_376_750_673_641,
                4_187_342_628_266_873_466,
                3_083_910_395_297_900_563,
                10_297_174_052_343_738_828,
                14_289_339_488_805_302_290,
            ],
        ),
        (
            2,
            1,
            [
                14_252_271_528_743_259_229,
                17_210_249_610_281_391_040,
                971_635_846_769_189_907,
                13_626_518_485_227_549_093,
                11_917_325_139_766_463_119,
            ],
        ),
        (
            2,
            2,
            [
                5_018_391_309_605_278_118,
                9_214_084_760_173_297_624,
                16_267_130_590_689_335_086,
                11_891_378_004_642_370_026,
                9_890_583_775_257_561_564,
            ],
        ),
        (
            2,
            3,
            [
                17_737_928_614_487_153_758,
                8_390_933_040_386_125_324,
                635_997_380_926_442_071,
                12_453_800_683_041_351_186,
                12_538_197_543_669_191_157,
            ],
        ),
        (
            2,
            4,
            [
                4_422_285_535_642_123_672,
                14_540_864_342_413_698_392,
                4_240_061_054_445_163_798,
                15_372_105_003_492_380_671,
                1_798_653_914_139_216_601,
            ],
        ),
        (
            2,
            5,
            [
                18_176_661_912_445_381_275,
                9_835_921_048_538_260_527,
                7_427_371_451_564_754_182,
                8_200_136_974_805_042_476,
                9_577_390_588_131_613_873,
            ],
        ),
        (
            2,
            6,
            [
                6_248_991_797_099_498_483,
                15_449_728_408_255_029_380,
                8_286_151_926_234_206_612,
                8_514_423_107_604_225_598,
                6_901_592_190_321_022_073,
            ],
        ),
        (
            2,
            7,
            [
                14_611_596_680_620_956_899,
                9_235_711_353_331_048_666,
                11_193_517_343_695_722_556,
                14_647_416_934_238_524_856,
                12_366_898_863_062_113_585,
            ],
        ),
        (
            2,
            8,
            [
                14_703_796_806_840_174_044,
                7_900_194_493_703_437_170,
                6_210_406_827_914_528_072,
                15_442_713_382_731_190_943,
                5_329_538_371_765_928_322,
            ],
        ),
        (
            2,
            9,
            [
                14_870_416_739_889_334_893,
                7_236_951_743_152_170_245,
                10_312_324_870_064_531_378,
                12_999_873_909_229_603_947,
                1_111_056_939_021_215_675,
            ],
        ),
        (
            2,
            10,
            [
                4_240_346_075_400_754_419,
                15_271_735_901_513_219_086,
                1_924_275_219_727_732_969,
                2_280_983_192_506_738_849,
                7_568_534_443_867_952_731,
            ],
        ),
        (
            2,
            11,
            [
                8_893_064_739_304_228_189,
                9_626_632_958_016_066_121,
                15_398_979_443_531_378_699,
                707_171_532_313_571_230,
                2_518_916_622_520_531_137,
            ],
        ),
        (
            2,
            12,
            [
                4_556_742_789_351_170_646,
                10_920_014_665_827_646_242,
                16_757_771_366_193_850_001,
                10_405_582_859_415_121_323,
                8_930_058_949_911_894_929,
            ],
        ),
        (
            2,
            13,
            [
                11_487_834_730_271_786_178,
                2_120_032_669_539_527_419,
                9_860_113_670_353_311_264,
                6_440_020_416_066_181_841,
                9_718_727_284_030_373_048,
            ],
        ),
        (
            2,
            14,
            [
                11_222_541_480_855_219_711,
                3_746_398_670_235_197_906,
                139_664_336_606_917_067,
                9_649_061_421_360_347_684,
                17_319_844_703_178_815_724,
            ],
        ),
        (
            2,
            15,
            [
                14_486_212_346_943_562_714,
                1_781_665_589_804_702_645,
                6_362_399_114_256_347_170,
                1_222_975_618_236_756_628,
                7_106_674_991_606_141_877,
            ],
        ),
        (
            3,
            0,
            [
                7_799_869_383_842_854_015,
                3_595_431_080_924_172_452,
                11_793_788_934_563_889_740,
                8_543_583_365_955_934_251,
                16_950_012_047_235_618_135,
            ],
        ),
        (
            3,
            9,
            [
                16_578_959_536_116_741_109,
                11_822_084_232_943_496_486,
                17_055_869_177_615_089_006,
                11_265_164_791_773_044_434,
                7_620_812_682_859_775_045,
            ],
        ),
        (
            3,
            10,
            [
                11_678_892_302_045_943_433,
                2_332_637_036_302_754_067,
                9_130_609_076_689_174_499,
                16_775_730_086_451_336_373,
                16_446_884_440_838_555_953,
            ],
        ),
        (
            3,
            11,
            [
                10_388_959_819_932_905_204,
                2_572_861_879_853_319_607,
                3_837_631_181_801_189_164,
                11_308_616_953_819_823_056,
                3_441_258_091_895_976_203,
            ],
        ),
    ];

    /// The qindex step of [`TX_TYPE_HASHES`].
    const TX_TYPE_QINDEX_STEP: usize = 17;

    #[test]
    fn transform_types_match_original_avm() {
        let dsp = Dsp::new();
        for &(size_index, type_index, expected) in &TX_TYPE_HASHES {
            let (size, tx_type) = (TxSize::ALL[size_index], TxType::ALL[type_index]);
            let (n, area) = (size.width(), size.coded_len());
            let mut hashes = [FNV_OFFSET; 5];
            let mut coefficients = vec![0; area];
            let mut levels = vec![0; area];
            let mut dequantized = vec![0; area];
            let mut residual = vec![0; n * n];
            for q in (1..=255u8).step_by(TX_TYPE_QINDEX_STEP) {
                for kind in 0..8 {
                    let source = pattern(n, q, kind);
                    forward(&dsp, (size, tx_type), &source, q, &mut coefficients).unwrap();
                    hash(&mut hashes[0], &coefficients);
                    quantize(
                        &dsp,
                        size,
                        &coefficients,
                        (q, Rounding::uniform(NEAREST_ROUNDING_Q7)),
                        &mut levels,
                    );
                    hash(&mut hashes[1], &levels);
                    dequantize(size, &levels, q, &mut dequantized).unwrap();
                    hash(&mut hashes[2], &dequantized);
                    reconstruct(&dsp, (size, tx_type), &levels, q, &mut residual).unwrap();
                    hash(&mut hashes[3], &residual);
                }
                let extreme: Vec<i32> = (0..area)
                    .map(|i| {
                        let magnitude = [1_048_575, 32767, 1][i % 3];
                        if i % 2 == 1 { -magnitude } else { magnitude }
                    })
                    .collect();
                reconstruct(&dsp, (size, tx_type), &extreme, q, &mut residual).unwrap();
                hash(&mut hashes[4], &residual);
            }
            assert_eq!(hashes, expected, "{size:?} {tx_type:?}");
        }
    }

    const SOURCE: [i32; 16] = [
        17, -23, 55, -101, 37, 0, -7, 85, -92, 41, 5, -64, 3, 72, -18, 29,
    ];
    const WHT_FORWARD: [i32; 16] = [
        88, 136, -424, -80, 168, -144, 472, 640, 56, 272, -136, 392, -728, 48, -856, 600,
    ];
    const DCT_FORWARD: [i32; 16] = [
        78, 102, -422, -125, -112, 200, 118, 799, 58, 402, -130, 264, -734, 212, -961, 255,
    ];
    const LEVELS: [[i32; 16]; 7] = [
        [
            11, 17, -53, -10, 21, -18, 59, 80, 7, 34, -17, 49, -91, 6, -107, 75,
        ],
        [
            16, 20, -84, -25, -22, 40, 24, 160, 12, 80, -26, 53, -147, 42, -192, 51,
        ],
        [
            15, 20, -82, -24, -22, 39, 23, 156, 11, 78, -25, 52, -143, 41, -188, 50,
        ],
        [
            8, 10, -43, -13, -11, 21, 12, 82, 6, 41, -13, 27, -75, 22, -99, 26,
        ],
        [
            8, 10, -42, -13, -11, 20, 12, 80, 6, 40, -13, 26, -73, 21, -96, 26,
        ],
        [1, 1, -5, -1, -1, 2, 1, 9, 1, 5, -1, 3, -8, 2, -11, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    const DEQUANT: [[i32; 16]; 7] = [
        [
            88, 136, -424, -80, 168, -144, 472, 640, 56, 272, -136, 392, -728, 48, -856, 600,
        ],
        [
            80, 100, -420, -125, -110, 200, 120, 800, 60, 400, -130, 265, -735, 210, -960, 255,
        ],
        [
            77, 103, -420, -123, -113, 200, 118, 800, 56, 400, -128, 267, -733, 210, -964, 256,
        ],
        [
            78, 98, -419, -127, -107, 205, 117, 800, 59, 400, -127, 263, -731, 215, -965, 254,
        ],
        [
            80, 100, -420, -130, -110, 200, 120, 800, 60, 400, -130, 260, -730, 210, -960, 260,
        ],
        [
            88, 88, -440, -88, -88, 176, 88, 792, 88, 440, -88, 264, -704, 176, -968, 264,
        ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    const RECON: [[i32; 16]; 7] = [
        [
            17, -23, 55, -101, 37, 0, -7, 85, -92, 41, 5, -64, 3, 72, -18, 29,
        ],
        [
            17, -22, 54, -99, 37, 0, -7, 84, -91, 41, 5, -63, 3, 71, -17, 28,
        ],
        [
            16, -23, 55, -100, 37, 0, -7, 84, -91, 41, 5, -63, 3, 71, -17, 28,
        ],
        [
            17, -22, 54, -100, 37, 0, -7, 84, -91, 41, 5, -63, 3, 71, -18, 29,
        ],
        [
            16, -22, 54, -99, 37, 0, -7, 84, -91, 40, 5, -63, 3, 72, -18, 29,
        ],
        [
            18, -21, 59, -97, 33, 0, -5, 81, -95, 39, 11, -62, 9, 68, -19, 26,
        ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    const EXTREME_DEQUANT: [[i32; 16]; 3] = [
        [
            0, 8, -8, 16, -16, 136, -136, 8184, -8184, 32767, -32768, 32767, -32768, 32767, -32768,
            32767,
        ],
        [
            0, 5, -5, 10, -10, 87, -87, 5243, -5243, 32767, -32768, 32767, -32768, 32767, -32768,
            32767,
        ],
        [
            0, 7680, -7680, 15360, -15360, 32767, -32768, 32767, -32768, 32767, -32768, 32767,
            -32768, 32767, -32768, 32767,
        ],
    ];
    const QUANT_HASH: [u64; 1] = [0xaa57_e874_9ce9_9364];

    #[test]
    fn matches_original_avm_transforms_and_dequantization() {
        assert_eq!(wht_forward(&SOURCE), WHT_FORWARD);
        for dsp in dsps() {
            let mut coefficients = [0; 16];
            forward(
                &dsp,
                (TxSize::Tx4, TxType::DctDct),
                &SOURCE,
                1,
                &mut coefficients,
            )
            .unwrap();
            assert_eq!(coefficients, DCT_FORWARD);
            for (index, qindex) in [0, 1, 2, 24, 25, 100, 255].into_iter().enumerate() {
                assert_eq!(levels_of(&dsp, TxSize::Tx4, &SOURCE, qindex), LEVELS[index]);
                let mut dequantized = [0; 16];
                dequantize(TxSize::Tx4, &LEVELS[index], qindex, &mut dequantized).unwrap();
                assert_eq!(dequantized, DEQUANT[index]);
                assert_eq!(
                    reconstruct_of(&dsp, TxSize::Tx4, &LEVELS[index], qindex),
                    RECON[index]
                );
            }
        }
        let extremes = [
            0, 1, -1, 2, -2, 17, -17, 1023, -1023, 32767, -32767, 65535, -65535, 1_048_575,
            -1_048_575, 123_456,
        ];
        for (index, qindex) in [0, 2, 255].into_iter().enumerate() {
            let mut dequantized = [0; 16];
            dequantize(TxSize::Tx4, &extremes, qindex, &mut dequantized).unwrap();
            assert_eq!(dequantized, EXTREME_DEQUANT[index]);
        }
        let mut hash = FNV_OFFSET;
        for qindex in 0..=255 {
            for byte in quantizer_step_q3(qindex).to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
            }
        }
        assert_eq!([hash], QUANT_HASH);
    }

    #[test]
    fn kernel_sizes_have_the_same_geometry() {
        for size in TxSize::ALL {
            let kernel = kernel_size(size);
            assert_eq!(
                (kernel.side(), kernel.coded_area()),
                (size.width(), size.coded_len())
            );
        }
    }

    #[test]
    fn lossless_is_exact_for_extremes_and_texture() {
        let dsp = Dsp::new();
        let round_trip = |source: &[i32; 16]| {
            reconstruct_of(
                &dsp,
                TxSize::Tx4,
                &levels_of(&dsp, TxSize::Tx4, source, 0),
                0,
            )
        };
        for value in [-255, 0, 255] {
            assert_eq!(round_trip(&[value; 16]), [value; 16]);
        }
        let mut state = 0x7a15_3b91u32;
        for _ in 0..2048 {
            let source = std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state % 511) as i32 - 255
            });
            assert_eq!(round_trip(&source), source);
        }
    }

    #[test]
    fn rejects_hostile_ranges_before_arithmetic() {
        let dsp = Dsp::new();
        let mut out = [0; 4096];
        assert!(forward(&dsp, (TxSize::Tx4, TxType::DctDct), &[256; 16], 1, &mut out).is_err());
        assert!(
            forward(
                &dsp,
                (TxSize::Tx4, TxType::DctDct),
                &[i32::MIN; 16],
                0,
                &mut out
            )
            .is_err()
        );
        assert!(
            forward(
                &dsp,
                (TxSize::Tx64, TxType::DctDct),
                &[-256; 4096],
                9,
                &mut out
            )
            .is_err()
        );
        for size in TxSize::ALL {
            for value in [i32::MIN, i32::MAX, MAX_LEVEL + 1, -MAX_LEVEL - 1] {
                assert!(
                    reconstruct(&dsp, (size, TxType::DctDct), &[value; 1024], 255, &mut out)
                        .is_err()
                );
            }
            for qindex in 1..=255 {
                reconstruct(&dsp, (size, TxType::DctDct), &[0; 1024], qindex, &mut out).unwrap();
                assert!(out.iter().all(|&v| v == 0));
            }
        }
        for size in [TxSize::Tx8, TxSize::Tx16, TxSize::Tx32, TxSize::Tx64] {
            assert!(forward(&dsp, (size, TxType::DctDct), &[0; 4096], 0, &mut out).is_err());
            assert!(reconstruct(&dsp, (size, TxType::DctDct), &[0; 1024], 0, &mut out).is_err());
        }
        assert_eq!(reconstruct_of(&dsp, TxSize::Tx4, &[0; 16], 0), [0; 16]);
    }
}
