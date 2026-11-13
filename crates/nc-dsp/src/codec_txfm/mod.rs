//! HEVC and H.264 integer transforms and quantisers.
//!
//! Every kernel here is bit-exact with the scalar code of the backend it
//! serves (`nc-hevc`'s `transform.rs` / `quant.rs`, `nc-h264`'s
//! `transform.rs`); the unit tests below check the scalar kernels against
//! literal ports of that code, and `nc-checkasm` checks every ISA against
//! the scalar kernels.
//!
//! # HEVC (§8.6.2–8.6.4)
//!
//! - **Forward DCT 4..32 and DST 4x4** (encoder choice; the HM encoder's
//!   scaling): rows first with rounding shift `log2 − 1`, then columns with
//!   shift `log2 + 6`, coefficients clamped to 16 bits. Coefficients are
//!   row-major: `coeffs[v * n + u]` is vertical frequency `v`, horizontal
//!   frequency `u`.
//! - **Inverse DCT / DST** (normative): columns, rounding shift 7, clamp to
//!   16 bits, rows, rounding shift 12; the residual is row-major `n x n`.
//!   Inputs are clamped to 16 bits on load (the scaling process guarantees
//!   that range, so this changes nothing for conformant input).
//! - **Quantise** (HM's dead-zone quantiser) and **dequantise** (eq. 8-309,
//!   flat scaling lists): [`HevcQuantizer`].
//!
//! # H.264 (§8.5)
//!
//! - **4x4 forward core transform** `Cf · X · Cfᵀ` and **inverse**
//!   (§8.5.12, with the final `(x + 32) >> 6`), on blocks in raster order.
//! - **Intra16x16 DC Hadamard** (eq. 8-320; its own inverse up to scale).
//! - **8x8 forward and inverse** (§8.5.13). The backend has no 8x8
//!   transform yet; the forward is the exact product `F · X · Fᵀ` with the
//!   integer matrix whose rows are 8 times the inverse basis, rounded by
//!   `(x + 32) >> 6`, i.e. the reference encoder's scale, and its quantiser
//!   multipliers are derived from that scale ([`H264Quantizer`]).
//! - **Quantise** (rounding 1/3 intra, 1/6 inter; DC blocks with one more
//!   bit) and **dequantise** (§8.5.12.1, §8.5.10, §8.5.11.2, §8.5.13.1),
//!   flat scaling matrices.
//!
//! # Ranges
//!
//! The SIMD kernels compute in 32-bit lanes; the backends compute HEVC in
//! 64 bits. Both agree within these input contracts, which the encoders
//! always meet:
//! - forward transforms: residuals of 8-bit samples, `|r| <= 255` (HEVC
//!   32x32 second pass: `32 · 90 · 45930 < 2^28`);
//! - HEVC quantiser: `|c| <= 32768` (forward DCT output is clamped there);
//! - inverse transforms: coefficients within 16 bits (dequantiser output).
//!
//! The HEVC dequantiser is exact for every `i32` level: levels beyond
//! `±2^16` and products beyond the saturation threshold are clamped before
//! the shift, where they would saturate anyway.

pub(crate) mod generic;
#[allow(unsafe_code)]
mod inverse_add;
#[allow(unsafe_code)]
pub(crate) mod scalar;
pub(crate) mod vvc_forward;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod vvc_forward_narrow;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
mod vvc_forward_simd;
pub(crate) mod vvc_inverse;
mod vvc_lfnst;
pub(crate) mod vvc_primary;
pub use vvc_lfnst::VvcLfnst;
pub use vvc_primary::VvcTransformKind;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512_h264;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2_paired;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128_inverse;

/// The HEVC transforms: the DST of 4x4 intra luma and the DCTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HevcTransform {
    /// 4x4 DST-VII (intra 4x4 luma, HEVC eq. 8-314).
    Dst4 = 0,
    /// 4x4 DCT.
    Dct4 = 1,
    /// 8x8 DCT.
    Dct8 = 2,
    /// 16x16 DCT.
    Dct16 = 3,
    /// 32x32 DCT.
    Dct32 = 4,
}

impl HevcTransform {
    /// Every transform.
    pub const ALL: [HevcTransform; 5] = [
        HevcTransform::Dst4,
        HevcTransform::Dct4,
        HevcTransform::Dct8,
        HevcTransform::Dct16,
        HevcTransform::Dct32,
    ];

    /// The DCT of side `2^log2` (2..=5), or `None`.
    #[must_use]
    pub const fn dct(log2: u8) -> Option<Self> {
        match log2 {
            2 => Some(Self::Dct4),
            3 => Some(Self::Dct8),
            4 => Some(Self::Dct16),
            5 => Some(Self::Dct32),
            _ => None,
        }
    }

    /// `log2` of the side.
    #[must_use]
    pub const fn log2(self) -> u8 {
        match self {
            Self::Dst4 | Self::Dct4 => 2,
            Self::Dct8 => 3,
            Self::Dct16 => 4,
            Self::Dct32 => 5,
        }
    }

    /// Side length in samples.
    #[must_use]
    pub const fn side(self) -> usize {
        1 << self.log2()
    }
}

/// Bit depth of every sample (Main profiles).
const BIT_DEPTH: u32 = 8;

/// HEVC quantiser and scaling parameters of one block: `qp` (0..=51), the
/// transform size and the prediction type (HM's rounding offsets: 171/512
/// of a step for intra blocks, 85/512 for inter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HevcQuantizer {
    /// Forward shift: `14 + qp / 6 + (15 − 8 − log2)`.
    pub(crate) qbits: u32,
    /// Forward multiplier, `quantScales[qp % 6]`.
    pub(crate) scale: i32,
    /// Forward rounding offset.
    pub(crate) offset: i32,
    /// Scaling multiplier `16 · levelScale[qp % 6]`.
    pub(crate) level_scale: i32,
    /// `qp / 6`.
    pub(crate) per: u32,
    /// Scaling shift `bdShift = 8 + log2 − 5`.
    pub(crate) bd_shift: u32,
}

/// `quantScales` of HM, indexed by `qp % 6`.
const HEVC_QUANT_SCALE: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
/// `levelScale` of HEVC §8.6.3, indexed by `qp % 6`.
const HEVC_LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
/// H.266 §8.7.3 `levelScale[1]`, for blocks whose area is not a power of
/// four.
const VVC_RECT_LEVEL_SCALE: [i32; 6] = [57, 64, 72, 80, 90, 102];
/// The forward scales matching [`VVC_RECT_LEVEL_SCALE`] (`HEVC_QUANT_SCALE`
/// over √2).
const VVC_RECT_QUANT_SCALE: [i32; 6] = [18396, 16384, 14564, 13107, 11651, 10280];
/// Largest HEVC level magnitude the quantiser emits.
pub(crate) const HEVC_MAX_LEVEL: i32 = 32767;

impl HevcQuantizer {
    /// Parameters for `qp` (0..=51) and a `2^log2` transform (2..=5).
    ///
    /// # Panics
    ///
    /// If `qp` or `log2` is out of range.
    #[must_use]
    pub fn new(qp: u8, log2: u8, intra: bool) -> Self {
        Self::with_rounding(qp, log2, if intra { 171 } else { 85 })
    }

    /// Like [`new`](Self::new), with the forward rounding offset given in
    /// 1/512 of a step (HM: 171 intra, 85 inter; smaller is a wider dead
    /// zone).
    ///
    /// # Panics
    ///
    /// If `qp`, `log2` or `rounding_512ths` (0..=511) is out of range.
    #[must_use]
    pub fn with_rounding(qp: u8, log2: u8, rounding_512ths: i32) -> Self {
        assert!(qp <= 51, "HEVC qp {qp} out of range");
        assert!((2..=5).contains(&log2), "HEVC transform log2 {log2}");
        assert!(
            (0..512).contains(&rounding_512ths),
            "rounding {rounding_512ths}"
        );
        let (per, rem) = (u32::from(qp / 6), usize::from(qp % 6));
        let qbits = 14 + per + (15 - BIT_DEPTH - u32::from(log2));
        let rounding = rounding_512ths;
        Self {
            qbits,
            scale: HEVC_QUANT_SCALE[rem],
            offset: rounding << (qbits - 9),
            level_scale: 16 * HEVC_LEVEL_SCALE[rem],
            per,
            bd_shift: BIT_DEPTH + u32::from(log2) - 5,
        }
    }
}

/// VVC independent quantization for 8-bit DCT-II blocks.
/// Uses the shared HEVC scaling arithmetic, with QP extended to 63 and
/// nearest rounding. Dependent quantization and scaling lists are excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VvcQuantizer(pub(crate) HevcQuantizer);

impl VvcQuantizer {
    /// Parameters for QP 0–63 and transform log2 2–6 (a 64-point side
    /// codes its 32 lowest frequencies, `crate::vvc_tx64`).
    /// # Panics
    /// If either parameter is outside its range.
    #[must_use]
    pub fn new(qp: u8, log2: u8) -> Self {
        Self::rect(qp, log2, log2)
    }

    /// Parameters for QP 0–63 and a `2^log2_width` x `2^log2_height`
    /// block, each side 4–64, or 1–64 with at least 16 samples (the narrow
    /// luma blocks of intra sub-partitions; a 64-point side codes 32
    /// frequencies, `crate::vvc_tx64`). Where the area is not a power of four the
    /// transform's gain is off by √2, which H.266 §8.7.3 compensates with
    /// its second `levelScale` row and one more shift (`rectNonTsFlag`);
    /// the forward side uses the matching scale row (`VVenC`
    /// `g_quantScales`, BSD-3-Clause-Clear, `THIRD_PARTY.md`).
    /// # Panics
    /// If a parameter is outside its range.
    #[must_use]
    pub fn rect(qp: u8, log2_width: u8, log2_height: u8) -> Self {
        assert!(qp <= 63 && log2_width <= 6 && log2_height <= 6 && log2_width + log2_height >= 4);
        let (per, rem) = (u32::from(qp / 6), usize::from(qp % 6));
        let log2_area = u32::from(log2_width + log2_height);
        let odd = log2_area & 1;
        let mean = log2_area / 2;
        let (scale, level_scale) = if odd == 0 {
            (HEVC_QUANT_SCALE[rem], HEVC_LEVEL_SCALE[rem])
        } else {
            (VVC_RECT_QUANT_SCALE[rem], VVC_RECT_LEVEL_SCALE[rem])
        };
        let qbits = 21 + per - mean - odd;
        Self(HevcQuantizer {
            qbits,
            scale,
            offset: 1 << (qbits - 1),
            level_scale: 16 * level_scale,
            per,
            bd_shift: 3 + mean + odd,
        })
    }

    /// The inverse scaling of a `2^log2_width` x `2^log2_height` block
    /// under dependent quantisation (§8.7.3 with `sh_dep_quant_used_flag`):
    /// the level scale and shift of QP + 1 and one more bit of `bdShift`,
    /// applied to levels already mapped to their reconstruction index
    /// (2|k| minus one in quantiser states 2 and 3). Inverse only: the
    /// forward fields stay QP's.
    ///
    /// # Panics
    ///
    /// If a parameter is outside its range.
    #[must_use]
    pub fn rect_dependent(qp: u8, log2_width: u8, log2_height: u8) -> Self {
        let mut quantizer = Self::rect(qp, log2_width, log2_height);
        let adjusted = usize::from(qp) + 1;
        let odd = (log2_width + log2_height) & 1 != 0;
        let level_scale = if odd {
            VVC_RECT_LEVEL_SCALE
        } else {
            HEVC_LEVEL_SCALE
        }[adjusted % 6];
        quantizer.0.level_scale = 16 * level_scale;
        quantizer.0.per = (adjusted / 6) as u32;
        quantizer.0.bd_shift += 1;
        quantizer
    }

    /// Like [`new`](Self::new), with the encoder's forward rounding offset
    /// in 1/512 of a step (256 is nearest; smaller is a wider dead zone).
    /// Inverse scaling is unchanged: rounding is an encoder choice.
    ///
    /// # Panics
    ///
    /// If `qp`, `log2` or `rounding_512ths` (0..=511) is out of range.
    #[must_use]
    pub fn with_rounding(qp: u8, log2: u8, rounding_512ths: i32) -> Self {
        Self::rect_with_rounding(qp, log2, log2, rounding_512ths)
    }

    /// [`rect`](Self::rect) with the encoder's forward rounding offset, as
    /// [`with_rounding`](Self::with_rounding).
    ///
    /// # Panics
    ///
    /// If a parameter is out of range.
    #[must_use]
    pub fn rect_with_rounding(
        qp: u8,
        log2_width: u8,
        log2_height: u8,
        rounding_512ths: i32,
    ) -> Self {
        assert!(
            (0..512).contains(&rounding_512ths),
            "rounding {rounding_512ths}"
        );
        let mut quantizer = Self::rect(qp, log2_width, log2_height);
        quantizer.0.offset = rounding_512ths << (quantizer.0.qbits - 9);
        quantizer
    }

    /// The largest coefficient magnitude this quantiser maps to level 0
    /// (`|c| · scale + offset < 2^qbits`), the `zero_limit` of
    /// [`crate::Dsp::hevc_forward_unless_zero`].
    #[must_use]
    pub fn largest_zero_coefficient(&self) -> u32 {
        let q = &self.0;
        let limit = ((1i64 << q.qbits) - i64::from(q.offset) - 1) / i64::from(q.scale);
        u32::try_from(limit.max(0)).unwrap_or(u32::MAX)
    }

    /// `coefficient`'s magnitude in quantiser steps before rounding
    /// (`|c| · scale / 2^qbits`): what an encoder's level adjustments (sign
    /// hiding parity, trellis) weigh their squared error in.
    #[must_use]
    pub fn magnitude_in_steps(&self, coefficient: i32) -> f64 {
        let q = &self.0;
        (f64::from(coefficient.unsigned_abs()) * f64::from(q.scale)) / f64::from(1u32 << q.qbits)
    }
}

/// H.264 quantiser and scaling parameters for one QP and prediction type
/// (default rounding offset 1/3 of a step intra, 1/6 inter: the reference
/// encoder's dead zones). [`Self::with_rounding_divisor`] allows tuning
/// the dead zone while keeping the zero-SAD bounds consistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H264Quantizer {
    /// `qP / 6`.
    pub(crate) per: u32,
    /// `qP % 6`.
    pub(crate) rem: usize,
    /// Forward rounding at 4x4-AC and DC/8x8 precision. Compute division
    /// once per QP, rather than at every quantisation and zero-SAD test.
    offsets: [i32; 2],
    /// Conservative 4x4-AC and chroma-DC zero-SAD bounds, fixed per QP.
    zero_sad_limits: [u32; 2],
}

impl H264Quantizer {
    /// Parameters for `qp` (0..=51).
    ///
    /// # Panics
    ///
    /// If `qp > 51`.
    #[must_use]
    pub fn new(qp: u8, intra: bool) -> Self {
        Self::with_rounding_divisor(qp, if intra { 3 } else { 6 })
    }

    /// Parameters for `qp`, rounding forward quantisation by `1 / divisor`
    /// of a step. A smaller divisor preserves more small coefficients.
    /// The reference offsets use 3 for intra and 6 for inter.
    ///
    /// # Panics
    ///
    /// If `qp > 51` or `divisor` is outside 2..=16.
    #[must_use]
    pub fn with_rounding_divisor(qp: u8, divisor: u8) -> Self {
        assert!(qp <= 51, "H.264 qp {qp} out of range");
        assert!(
            (2..=16).contains(&divisor),
            "H.264 rounding divisor {divisor}"
        );
        let qbits = 15 + u32::from(qp / 6);
        let offsets = [
            (1 << qbits) / i32::from(divisor),
            (1 << (qbits + 1)) / i32::from(divisor),
        ];
        let mf = QUANT_MF_4X4[usize::from(qp % 6)];
        let worst = mf[0].max(4 * mf[1]).max(2 * mf[2]);
        let ac_limit = (1i64 << qbits) - i64::from(offsets[0]);
        let dc_limit = (1i64 << (qbits + 1)) - i64::from(offsets[1]);
        Self {
            per: u32::from(qp / 6),
            rem: usize::from(qp % 6),
            offsets,
            zero_sad_limits: [
                ((ac_limit - 1) / i64::from(worst)) as u32,
                ((dc_limit - 1) / i64::from(mf[0])) as u32,
            ],
        }
    }

    /// The largest SAD of a 4x4 residual block whose 4x4 core transform
    /// is guaranteed to quantise to all zeros (DC included).
    ///
    /// Each coefficient of the core transform `Cf · X · Cfᵀ` (§8.5.12) is a
    /// weighted sum of the residual with weights `wu · wv`, `w = 1` for even
    /// and 2 for odd frequencies, so `|F(u, v)| <= wu · wv · SAD`. A level is
    /// zero when `|F| · MF + offset < 2^qbits`; taking the worst position
    /// class gives a bound that needs no transform at all.
    #[must_use]
    pub fn zero_sad_threshold_4x4(&self) -> u32 {
        self.zero_sad_limits[0]
    }

    /// The largest SAD of an 8x8 chroma residual block (4:2:0) whose 2x2
    /// DC transform is guaranteed to quantise to zero: each 4x4 block's DC
    /// is the sum of its residual, so every Hadamard output is bounded by
    /// the SAD of the whole 8x8 block (§8.5.11).
    #[must_use]
    pub fn zero_sad_threshold_chroma_dc(&self) -> u32 {
        self.zero_sad_limits[1]
    }

    /// The forward multipliers by position and `qbits` of 4x4 (and chroma
    /// AC) blocks: `|c| · mf[pos] / 2^qbits` is a coefficient's magnitude
    /// in quantisation steps, before rounding (encoders' rate-distortion
    /// quantisation reads it).
    #[must_use]
    pub fn forward_scale_4x4(&self) -> (&'static [i32; 16], u32) {
        (&QUANT_MF_4X4_POS[self.rem], self.forward_terms(0).0)
    }

    /// [`Self::forward_scale_4x4`] for 8x8 blocks.
    #[must_use]
    pub fn forward_scale_8x8(&self) -> (&'static [i32; 64], u32) {
        (&QUANT_MF_8X8_POS[self.rem], self.forward_terms(1).0)
    }

    /// `Qstep`: the change of an orthonormal transform coefficient per
    /// level (1 at qP 4, doubling every 6), in sample units. The
    /// position-dependent multipliers make it the same for every position
    /// and both transform sizes.
    #[must_use]
    pub fn step(&self) -> f64 {
        QSTEP_AT_REM[self.rem] * f64::from(1_u32 << self.per)
    }

    /// `(qbits, offset)` of forward quantisation with `extra` more bits of
    /// precision than 4x4 AC (1 for DC blocks, 1 for 8x8).
    pub(crate) fn forward_terms(&self, extra: u32) -> (u32, i32) {
        let qbits = 15 + self.per + extra;
        (qbits, self.offsets[extra as usize])
    }
}

/// `Qstep` at `qP` 0..5 (the H.264 step table; it doubles every 6 qP).
const QSTEP_AT_REM: [f64; 6] = [0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125];

/// `normAdjust4x4` (H.264 eq. 8-315), by `qP % 6` and position class.
const NORM_ADJUST_4X4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// The reference encoder's forward multipliers matching
/// [`NORM_ADJUST_4X4`] (`nc-h264`'s `QUANT_MF_4X4`).
const QUANT_MF_4X4: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

/// `normAdjust8x8` (H.264 eq. 8-318), by `qP % 6` and position class.
const NORM_ADJUST_8X8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// The 4x4 class of raster position `pos`: 0 for (even, even), 1 for
/// (odd, odd), 2 otherwise.
const fn class_4x4(pos: usize) -> usize {
    match ((pos / 4) % 2, pos % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// The 8x8 class of raster position `pos` (H.264 eq. 8-318).
const fn class_8x8(pos: usize) -> usize {
    let (i, j) = (pos / 8, pos % 8);
    if i % 4 == 0 && j % 4 == 0 {
        0
    } else if i % 2 == 1 && j % 2 == 1 {
        1
    } else if i % 4 == 2 && j % 4 == 2 {
        2
    } else if (i % 4 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 4 == 0) {
        3
    } else if (i % 4 == 0 && j % 4 == 2) || (i % 4 == 2 && j % 4 == 0) {
        4
    } else {
        5
    }
}

/// Per-position tables expanded from class tables: `[qP % 6][pos]`.
const fn expand_4x4(classes: &[[i32; 3]; 6], factor: i32) -> [[i32; 16]; 6] {
    let mut out = [[0; 16]; 6];
    let mut m = 0;
    while m < 6 {
        let mut pos = 0;
        while pos < 16 {
            out[m][pos] = factor * classes[m][class_4x4(pos)];
            pos += 1;
        }
        m += 1;
    }
    out
}

/// `LevelScale4x4` with flat weights (`16 · normAdjust4x4`).
pub(crate) const LEVEL_SCALE_4X4: [[i32; 16]; 6] = expand_4x4(&NORM_ADJUST_4X4, 16);
/// Forward 4x4 multipliers by position.
pub(crate) const QUANT_MF_4X4_POS: [[i32; 16]; 6] = expand_4x4(&QUANT_MF_4X4, 1);

/// `LevelScale8x8` with flat weights (`16 · normAdjust8x8`).
pub(crate) const LEVEL_SCALE_8X8: [[i32; 64]; 6] = {
    let mut out = [[0; 64]; 6];
    let mut m = 0;
    while m < 6 {
        let mut pos = 0;
        while pos < 64 {
            out[m][pos] = 16 * NORM_ADJUST_8X8[m][class_8x8(pos)];
            pos += 1;
        }
        m += 1;
    }
    out
};

/// Squared norms of the rows of the forward 8x8 matrix `F`, over 64 (the
/// forward output is scaled by `1/64`): 8 for rows 0 and 4, 5 for 2 and 6,
/// 578/64 for the odd rows; kept as numerators over 64.
const F8_ROW_NORM_64THS: [i64; 8] = [512, 578, 320, 578, 512, 578, 320, 578];

/// Forward 8x8 multipliers by position: `round(2^24 / (normAdjust8x8 · nᵢ ·
/// nⱼ))` with `n` the scaled row norms, which makes quantisation with
/// `qbits = 16 + qP / 6` the inverse of the 8x8 scaling. For `qP % 6 == 0`
/// this reproduces the reference encoder's `quant_coef8` (13107, 11428,
/// 20972, 12222, 16777, 15481).
pub(crate) const QUANT_MF_8X8_POS: [[i32; 64]; 6] = {
    let mut out = [[0; 64]; 6];
    let mut m = 0;
    while m < 6 {
        let mut pos = 0;
        while pos < 64 {
            let v = NORM_ADJUST_8X8[m][class_8x8(pos)] as i64;
            let denominator = v * F8_ROW_NORM_64THS[pos / 8] * F8_ROW_NORM_64THS[pos % 8];
            let numerator = (1i64 << 24) * 64 * 64;
            out[m][pos] = ((2 * numerator + denominator) / (2 * denominator)) as i32;
            pos += 1;
        }
        m += 1;
    }
    out
};

/// The 32-point HEVC DCT matrix, `transMatrix` of HEVC §8.6.4.2 (the same
/// specification values as `nc-hevc`'s table, which took them from
/// kvazaar: see `THIRD_PARTY.md`). The `N`-point basis vector
/// `k` is row `k * 32 / N`, truncated to `N` entries.
#[rustfmt::skip]
pub(crate) const DCT32: [[i8; 32]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
    [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
    [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
    [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
    [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

/// The 4x4 DST-VII matrix of HEVC eq. 8-314 (`transMatrix` for
/// `trType == 1`): row `k` is basis vector `k`.
pub(crate) const DST4: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

/// Forward HEVC transform: residual (row-major at `stride`) to row-major
/// coefficients (`n^2`).
pub(crate) type HevcForwardFn = unsafe fn(residual: &[i16], stride: usize, out: &mut [i32]);
/// Forward DCT, returning false if every coefficient provably quantizes to zero.
pub(crate) type HevcForwardBoundedFn =
    unsafe fn(&[i16], usize, u32, &mut [i32], HevcForwardMode) -> bool;

/// Which passes of an encoder-side HEVC 16/32-point forward DCT run AV1's
/// staged butterflies instead of HEVC's exact matrix (`generic::
/// staged_hevc_pass`). The coefficients only steer the encoder (the decoder
/// inverts what was coded, exactly), so approximation costs coding
/// efficiency, never conformance. Both approximations keep the exact DC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HevcForwardStaging {
    /// HEVC's exact matrix in both passes.
    Exact,
    /// Staged rows, exact columns.
    Rows,
    /// Staged rows and columns.
    RowsAndColumns,
}
/// An encoder-side HEVC forward DCT: its [`HevcForwardStaging`], and
/// whether the exact column pass writes zeros for every group of columns
/// (one lane vector wide) whose coefficients the proven-zero bound shows
/// all quantise to zero, instead of computing them (`zero_columns`). The
/// quantised levels are unchanged; the raw coefficients there (all within
/// the zero limit) become zero, which only steers sign hiding and the
/// trellis, so it is an encoder approximation like staging. With
/// `low_frequencies`, only the low `N/2 x N/2` quarter of the coefficients
/// is computed and the rest is zero (SVT-AV1's partial-frequency forward,
/// AV1's `av1.partial_forward`): an encoder approximation that drops the
/// high frequencies from coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HevcForwardMode {
    /// Which passes run AV1's staged butterflies.
    pub staging: HevcForwardStaging,
    /// Skip proven-zero column groups of the exact column pass.
    pub zero_columns: bool,
    /// Keep only the low-frequency quarter of the coefficients.
    pub low_frequencies: bool,
}

impl HevcForwardMode {
    /// HEVC's exact forward.
    pub const EXACT: Self = Self::staged(HevcForwardStaging::Exact);

    /// `staging` alone, every column computed.
    #[must_use]
    pub const fn staged(staging: HevcForwardStaging) -> Self {
        Self {
            staging,
            zero_columns: false,
            low_frequencies: false,
        }
    }
}

/// Inverse HEVC transform: row-major coefficients to row-major residual.
pub(crate) type HevcInverseFn = unsafe fn(coeffs: &[i32], out: &mut [i32]);
/// Forward H.264 4x4 transform of `out.len()` blocks, `blocks_wide` per row
/// of blocks, from a residual region at `stride`.
pub(crate) type H264Forward4Fn =
    unsafe fn(residual: &[i16], stride: usize, blocks_wide: usize, out: &mut [[i32; 16]]);
/// A transform of each 4x4 block of `input` into `out` (same length).
pub(crate) type H264Blocks4Fn = unsafe fn(input: &[[i32; 16]], out: &mut [[i32; 16]]);
/// Forward H.264 8x8 transform of a residual at `stride`.
pub(crate) type H264Forward8Fn = unsafe fn(residual: &[i16], stride: usize, out: &mut [i32; 64]);
/// Four 8x8 forward transforms and quantisation of one 16x16 macroblock.
pub(crate) type H264ForwardQuantizeMbFn = unsafe fn(
    residual: &[i16],
    stride: usize,
    zero_mask: u8,
    q: &H264Quantizer,
    out: &mut [[i32; 64]; 4],
) -> [u32; 4];
/// Inverse H.264 8x8 transform.
pub(crate) type H264Inverse8Fn = unsafe fn(coeffs: &[i32; 64], out: &mut [i32; 64]);
/// H.264 reconstruction of 4x4 blocks in place: scaling (`ls`, `shift`),
/// optional pre-scaled DCs, inverse transform and the add to the prediction
/// in `dst` (`generic::h264_add_inverse_4x4`).
pub(crate) type H264AddInverse4Fn = unsafe fn(
    levels: &[[i32; 16]],
    dc: &[i32],
    scale: (&[i32; 16], i32),
    blocks_wide: usize,
    dst: &mut [u8],
    stride: usize,
);
/// H.264 reconstruction of already-scaled 4x4 blocks in place
/// (`generic::h264_add_scaled_4x4`).
pub(crate) type H264AddScaled4Fn =
    unsafe fn(coeffs: &[[i32; 16]], blocks_wide: usize, dst: &mut [u8], stride: usize);
/// H.264 reconstruction of a macroblock's already-scaled luma 4x4 blocks
/// with known coded and AC masks (`generic::h264_add_scaled_mb`).
pub(crate) type H264AddScaledMbFn =
    unsafe fn(coeffs: &[[i32; 16]; 16], masks: (u16, u16), dst: &mut [u8], stride: usize);
/// H.264 reconstruction of an already-scaled 8x8 block in place
/// (`generic::h264_add_scaled_8x8`).
pub(crate) type H264AddScaled8Fn = unsafe fn(coeffs: &[i32; 64], dst: &mut [u8], stride: usize);
/// H.264 chroma transform and quantisation of one 4:2:0 component
/// (`generic::h264_forward_quantize_chroma`).
pub(crate) type H264ForwardQuantizeChromaFn = unsafe fn(
    residual: &[i16],
    stride: usize,
    q: &H264Quantizer,
    ac: &mut [[i32; 16]; 4],
) -> ([i32; 4], u32);
/// H.264 reconstruction of an 8x8 block in place
/// (`generic::h264_add_inverse_8x8`).
pub(crate) type H264AddInverse8Fn =
    unsafe fn(levels: &[i32; 64], scale: (&[i32; 64], i32), dst: &mut [u8], stride: usize);
/// Dead-zone quantisation in place: `level = (|c| · mf[i % mf.len()] +
/// offset) >> qbits`, capped at `max_level`, signed like `c`. Returns the
/// number of non-zero levels. `coeffs.len()` is a multiple of `mf.len()`.
pub(crate) type ScaleQuantizeFn =
    unsafe fn(coeffs: &mut [i32], mf: &[i32], offset: i32, qbits: u32, max_level: i32) -> u32;
/// H.264 scaling: `d = l · ls[i % ls.len()]`, then shifted left by `shift`
/// if `shift >= 0`, else rounding-shifted right by `-shift`.
pub(crate) type H264ScaleFn = unsafe fn(levels: &[i32], ls: &[i32], shift: i32, out: &mut [i32]);
/// Bit `i` set where `levels[i]` is non-zero; at most [`MAX_MASK_LEVELS`]
/// levels.
pub(crate) type NonzeroMaskFn = unsafe fn(levels: &[i32]) -> u64;

/// Most levels [`NonzeroMaskFn`] takes: one bit each in a `u64` (an 8x8
/// block).
pub(crate) const MAX_MASK_LEVELS: usize = 64;
/// HEVC scaling (eq. 8-309) of `levels` into `out`.
pub(crate) type HevcScaleFn = unsafe fn(levels: &[i32], q: &HevcQuantizer, out: &mut [i32]);

/// Fused HEVC inverse of i32 coefficients, added into an 8-bit picture.
/// The `(rows, columns)` extent covers every nonzero coefficient (see
/// `Dsp::hevc_inverse_add`).
pub(crate) type HevcInverseAddFn = unsafe fn(&[i32], (usize, usize), &mut [u8], usize);

/// One ISA's HEVC and H.264 transform kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CodecTxfmFns {
    /// Indexed by [`HevcTransform`].
    pub hevc_forward: [HevcForwardFn; 5],
    /// Exact rectangular DCT-II, with independent row/column shifts.
    pub vvc_forward_dct: unsafe fn(usize, usize, &[i16], &mut [i32]),
    pub hevc_forward_bounded: [HevcForwardBoundedFn; 4],
    /// Indexed by [`HevcTransform`].
    pub hevc_inverse: [HevcInverseFn; 5],
    pub hevc_inverse_add: [HevcInverseAddFn; 5],
    pub hevc_dequantize: HevcScaleFn,
    pub hevc_quantize_from: unsafe fn(&[i32], &HevcQuantizer, &mut [i32]) -> u32,
    pub h264_forward_4x4: H264Forward4Fn,
    pub h264_inverse_4x4: H264Blocks4Fn,
    pub h264_hadamard_4x4: H264Blocks4Fn,
    pub h264_forward_8x8: H264Forward8Fn,
    pub h264_forward_quantize_mb: H264ForwardQuantizeMbFn,
    pub h264_inverse_8x8: H264Inverse8Fn,
    pub h264_dequantize: H264ScaleFn,
    pub h264_add_inverse_4x4: H264AddInverse4Fn,
    pub h264_add_inverse_8x8: H264AddInverse8Fn,
    pub h264_add_scaled_4x4: H264AddScaled4Fn,
    pub h264_add_scaled_8x8: H264AddScaled8Fn,
    pub h264_add_scaled_mb: H264AddScaledMbFn,
    pub h264_forward_quantize_chroma: H264ForwardQuantizeChromaFn,
    /// Shared by the HEVC and H.264 quantisers.
    pub quantize: ScaleQuantizeFn,
    /// The non-zero levels of a block as a bit mask (the H.264 CABAC
    /// residual writer finds its significance map from it).
    pub nonzero_mask: NonzeroMaskFn,
}

#[cfg(test)]
mod tests;
