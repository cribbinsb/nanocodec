//! CABAC binarisations and context selection of the slice data and
//! macroblock syntax elements (§9.3.2, §9.3.3.1) for frame-coded 4:2:0 and
//! 4:0:0 I, P and B slices.

use core::marker::PhantomData;

use super::cabac::Cabac;
use super::mbstate::{MbKind, MbState, Neighbours};
use super::slice::SliceType;
use super::syntax::{Block4x4, Coeffs, MbSyntax};
use super::{DecodeError, Result};
use crate::tables::{LAST_CTX_INC_8X8, SIGNIFICANT_CTX_INC_8X8, SIGNIFICANT_CTX_INC_8X8_FIELD};

// ctxIdxOffset of each syntax element (Table 9-34, frame coded).
const MB_TYPE_I: usize = 3;
const MB_SKIP_P: usize = 11;
const MB_TYPE_P_PREFIX: usize = 14;
const MB_TYPE_P_SUFFIX: usize = 17;
const SUB_MB_TYPE_P: usize = 21;
const MB_SKIP_B: usize = 24;
const MB_TYPE_B_PREFIX: usize = 27;
const MB_TYPE_B_SUFFIX: usize = 32;
const SUB_MB_TYPE_B: usize = 36;
const MVD: [usize; 2] = [40, 47];
const REF_IDX: usize = 54;
const MB_QP_DELTA: usize = 60;
const INTRA_CHROMA_PRED_MODE: usize = 64;
const PREV_INTRA_PRED_MODE: usize = 68;
const REM_INTRA_PRED_MODE: usize = 69;
const CBP_LUMA: usize = 73;
const CBP_CHROMA: usize = 77;
const CODED_BLOCK_FLAG: usize = 85;
const SIGNIFICANT: usize = 105;
const LAST_SIGNIFICANT: usize = 166;
/// The significance contexts of field macroblocks (Table 9-34).
const SIGNIFICANT_FIELD: usize = 277;
const LAST_SIGNIFICANT_FIELD: usize = 338;
const ABS_LEVEL: usize = 227;
const TRANSFORM_8X8: usize = 399;
const SIGNIFICANT_8X8: usize = 402;
const LAST_SIGNIFICANT_8X8: usize = 417;
const SIGNIFICANT_8X8_FIELD: usize = 436;
const LAST_SIGNIFICANT_8X8_FIELD: usize = 451;
const ABS_LEVEL_8X8: usize = 426;

/// `mb_type` values (Table 7-11) the binarisation produces directly.
const MB_TYPE_I_NXN: u32 = 0;
const MB_TYPE_B_DIRECT: u32 = 0;
const MB_TYPE_I_PCM: u32 = 25;
/// First intra `mb_type` in P slices (Table 7-13) and B slices (Table 7-14).
const P_INTRA_BASE: u32 = 5;
const B_INTRA_BASE: u32 = 23;
/// B `mb_type`s the binarisation names directly (Table 9-37): the prefix
/// bins `111101` (intra), `111110` and `111111`.
const B_PREFIX_INTRA: u32 = 13;
const B_PREFIX_L1_L0_8X16: u32 = 14;
const B_PREFIX_8X8: u32 = 15;
const MB_TYPE_B_L1_L0_8X16: u32 = 11;
const MB_TYPE_B_8X8: u32 = 22;
/// First `mb_type` of B prefixes `110xxx` (`B_Bi_16x16`) and how far below
/// its 7-bin value a `1110xxx`-`11110xx` one lies.
const B_FOUR_BIN_BASE: u32 = 3;
const B_SEVEN_BIN_OFFSET: u32 = 4;
/// B `sub_mb_type`s the binarisation names directly (Table 9-38).
const SUB_B_FIRST_THREE_BIN: u8 = 3;
const SUB_B_FIRST_FOUR_BIN: u8 = 7;
const SUB_B_L1_4X4: u8 = 11;
/// `rem_intra_pred_mode` is a 3-bit fixed-length value.
const REM_INTRA_PRED_MODE_BITS: u32 = 3;
/// `intra_chroma_pred_mode` is truncated unary with `cMax` 3.
const MAX_CHROMA_PRED_MODE: u8 = 3;
/// Mapped `mb_qp_delta` limit: values −32..=31 (§7.4.5 at 10 bits).
const MAX_MAPPED_QP_DELTA: u32 = 64;
/// `mvd` prefix length (UEG3, `uCoff` 9) and suffix order.
const MVD_PREFIX_MAX: u32 = 9;
const MVD_SUFFIX_ORDER: u32 = 3;
/// `coeff_abs_level_minus1` prefix length (UEG0, `uCoff` 14).
const LEVEL_PREFIX_MAX: u32 = 14;
/// Largest `|mvd|` of a conforming stream (§7.4.5.1, level limits).
const MAX_ABS_MVD: u32 = 1 << 15;
/// Largest coefficient magnitude of 10-bit video (§7.4.5.3.3 bound,
/// 2^(7 + `bitDepth`)).
const MAX_ABS_LEVEL: u32 = 1 << 17;
/// `absMvdComp` sums below 3 select `ctxIdxInc` 0, above 32 `ctxIdxInc` 2.
const MVD_SUM_LOW: u32 = 3;
const MVD_SUM_HIGH: u32 = 32;

/// `ctxBlockCat` (Table 9-42) and its context offsets (Table 9-40).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cat {
    LumaDc = 0,
    LumaAc = 1,
    Luma4x4 = 2,
    ChromaDc = 3,
    ChromaAc = 4,
}

impl Cat {
    fn coded_block_flag_offset(self) -> usize {
        4 * self as usize
    }
    fn significance_offset(self) -> usize {
        [0, 15, 29, 44, 47][self as usize]
    }
    fn level_offset(self) -> usize {
        [0, 10, 20, 30, 39][self as usize]
    }
    fn coefficients(self) -> usize {
        match self {
            Self::LumaDc | Self::Luma4x4 => 16,
            Self::LumaAc | Self::ChromaAc => 15,
            Self::ChromaDc => 4,
        }
    }
}

/// The CABAC syntax decoder of one slice, of a decoder of `T` samples. `T`
/// is unused but gives each sample type's instance its own copy of every
/// method (`docs/handoff/high-bit-depth.md`, "Keeping the 8-bit instance
/// as fast"): shared by both, the hot ones stopped being inlined into the
/// 8-bit macroblock layer.
pub(super) struct CabacSyntax<'a, T = u8> {
    engine: Cabac<'a>,
    /// Whether the previous macroblock in decoding order had a non-zero
    /// `mb_qp_delta` (§9.3.3.1.1.5).
    prev_qp_delta_nonzero: bool,
    sample: PhantomData<T>,
}

/// `condTermFlag` of a whole neighbouring macroblock.
fn cond(n: Option<&MbState>, flag: impl Fn(&MbState) -> bool) -> usize {
    usize::from(n.is_some_and(flag))
}

impl<'a, T> CabacSyntax<'a, T> {
    pub fn new(engine: Cabac<'a>) -> Self {
        Self {
            engine,
            prev_qp_delta_nonzero: false,
            sample: PhantomData,
        }
    }

    /// `mb_skip_flag` of a P or B slice (`ctxIdxInc` §9.3.3.1.1.1).
    pub fn mb_skip_flag(&mut self, slice: SliceType, nb: &Neighbours<'_>) -> bool {
        let not_skip = |m: &MbState| !m.is_skip();
        let offset = if slice == SliceType::B {
            MB_SKIP_B
        } else {
            MB_SKIP_P
        };
        self.engine
            .decision(offset + cond(nb.a, not_skip) + cond(nb.b, not_skip))
    }

    /// `end_of_slice_flag`.
    pub fn end_of_slice(&mut self) -> bool {
        self.engine.terminate()
    }

    /// The bins of an intra `mb_type` after its first bin (Table 9-36):
    /// `I_PCM` by a terminate bin, else `Intra_16x16` coded block patterns and
    /// prediction mode. `suffix` is the P or B suffix's `ctxIdxOffset`, `None`
    /// in I slices.
    fn intra_16x16_or_pcm(&mut self, suffix: Option<usize>) -> u32 {
        if self.engine.terminate() {
            return MB_TYPE_I_PCM;
        }
        // ctxIdx of the luma, chroma, chroma-AC and the two mode bins.
        let (luma, chroma, chroma_ac, mode) = if let Some(base) = suffix {
            (base + 1, base + 2, base + 2, [base + 3, base + 3])
        } else {
            let base = MB_TYPE_I;
            (base + 3, base + 4, base + 5, [base + 6, base + 7])
        };
        let luma_coded = u32::from(self.engine.decision(luma));
        let mut chroma_cbp = 0;
        if self.engine.decision(chroma) {
            chroma_cbp = 1 + u32::from(self.engine.decision(chroma_ac));
        }
        let pred =
            2 * u32::from(self.engine.decision(mode[0])) + u32::from(self.engine.decision(mode[1]));
        1 + pred + 4 * chroma_cbp + 12 * luma_coded
    }

    /// A B-slice `mb_type` (Table 9-37, `ctxIdxInc` §9.3.3.1.1.3: bin 0 by
    /// the neighbours that are not `B_Skip` / `B_Direct_16x16`). Out of
    /// line: P slices keep `mb_type` small enough to inline.
    #[inline(never)]
    fn b_mb_type(&mut self, nb: &Neighbours<'_>) -> u32 {
        let not_direct = |m: &MbState| !matches!(m.kind, MbKind::BSkip | MbKind::BDirect16x16);
        let inc = cond(nb.a, not_direct) + cond(nb.b, not_direct);
        if !self.engine.decision(MB_TYPE_B_PREFIX + inc) {
            return MB_TYPE_B_DIRECT;
        }
        // ctxIdxInc 3 for bin 1; bin 2 takes 4 after a 1, else 5, as every
        // later bin.
        let ctx = |inc: usize| MB_TYPE_B_PREFIX + inc;
        if !self.engine.decision(ctx(3)) {
            // 10x: B_L0_16x16, B_L1_16x16.
            return 1 + u32::from(self.engine.decision(ctx(5)));
        }
        let mut bits = u32::from(self.engine.decision(ctx(4))) << 3;
        for shift in (0..3).rev() {
            bits |= u32::from(self.engine.decision(ctx(5))) << shift;
        }
        match bits {
            0..8 => B_FOUR_BIN_BASE + bits,
            B_PREFIX_INTRA => B_INTRA_BASE + self.b_intra_type(),
            B_PREFIX_L1_L0_8X16 => MB_TYPE_B_L1_L0_8X16,
            B_PREFIX_8X8 => MB_TYPE_B_8X8,
            _ => ((bits << 1) | u32::from(self.engine.decision(ctx(5)))) - B_SEVEN_BIN_OFFSET,
        }
    }

    /// The intra suffix of a B `mb_type` (Table 9-36 with the B contexts).
    fn b_intra_type(&mut self) -> u32 {
        if !self.engine.decision(MB_TYPE_B_SUFFIX) {
            return MB_TYPE_I_NXN;
        }
        self.intra_16x16_or_pcm(Some(MB_TYPE_B_SUFFIX))
    }

    /// A B `sub_mb_type` (Table 9-38; bin 2 takes `ctxIdxInc` 2 after a 1,
    /// the later bins 3).
    fn b_sub_mb_type(&mut self) -> u8 {
        let ctx = |inc: usize| SUB_MB_TYPE_B + inc;
        if !self.engine.decision(ctx(0)) {
            return 0;
        }
        if !self.engine.decision(ctx(1)) {
            return 1 + u8::from(self.engine.decision(ctx(3)));
        }
        let mut sub = SUB_B_FIRST_THREE_BIN;
        if self.engine.decision(ctx(2)) {
            if self.engine.decision(ctx(3)) {
                return SUB_B_L1_4X4 + u8::from(self.engine.decision(ctx(3)));
            }
            sub = SUB_B_FIRST_FOUR_BIN;
        }
        sub += 2 * u8::from(self.engine.decision(ctx(3)));
        sub + u8::from(self.engine.decision(ctx(3)))
    }

    /// One residual block of category `cat` of a frame or `field`
    /// macroblock: `coded_block_flag` with `ctxIdxInc` `cbf_inc`, then the
    /// significance map and levels.
    fn residual(
        &mut self,
        (cat, field): (Cat, bool),
        cbf_inc: usize,
        out: &mut Coeffs,
    ) -> Result<bool> {
        out.count = 0;
        if !self
            .engine
            .decision(CODED_BLOCK_FLAG + cat.coded_block_flag_offset() + cbf_inc)
        {
            return Ok(false);
        }
        let (sig, last) = if field {
            (SIGNIFICANT_FIELD, LAST_SIGNIFICANT_FIELD)
        } else {
            (SIGNIFICANT, LAST_SIGNIFICANT)
        };
        let sig = sig + cat.significance_offset();
        let last = last + cat.significance_offset();
        let chroma_dc = cat == Cat::ChromaDc;
        // ctxIdxInc of chroma DC significance is Min(i / NumC8x8, 2) (eq. 9-23);
        // a separate instance keeps that choice out of the scan loop.
        if chroma_dc {
            let inc = |i: usize| i.min(2);
            self.significance_map(cat.coefficients(), out, |i| sig + inc(i), |i| last + inc(i));
        } else {
            self.significance_map(cat.coefficients(), out, |i| sig + i, |i| last + i);
        }
        self.levels(ABS_LEVEL + cat.level_offset(), chroma_dc, out)?;
        Ok(true)
    }

    /// `significant_coeff_flag` / `last_significant_coeff_flag` of a block
    /// of `count` coefficients (§7.3.5.3.3).
    fn significance_map(
        &mut self,
        count: usize,
        out: &mut Coeffs,
        sig: impl Fn(usize) -> usize,
        last: impl Fn(usize) -> usize,
    ) {
        for i in 0..count - 1 {
            if self.engine.decision(sig(i)) {
                out.push(i, 0);
                if self.engine.decision(last(i)) {
                    return;
                }
            }
        }
        out.push(count - 1, 0);
    }

    /// `coeff_abs_level_minus1` and `coeff_sign_flag`, highest frequency
    /// first (`ctxIdxInc` §9.3.3.1.3).
    fn levels(&mut self, base: usize, chroma_dc: bool, out: &mut Coeffs) -> Result<()> {
        let max_gt1_inc = if chroma_dc { 3 } else { 4 };
        let (mut eq1, mut gt1) = (0usize, 0usize);
        for k in (0..out.count).rev() {
            let first = if gt1 != 0 { 0 } else { (1 + eq1).min(4) };
            let mut minus1 = 0;
            if self.engine.decision(base + first) {
                let rest = base + 5 + gt1.min(max_gt1_inc);
                minus1 = 1;
                while minus1 < LEVEL_PREFIX_MAX && self.engine.decision(rest) {
                    minus1 += 1;
                }
                if minus1 == LEVEL_PREFIX_MAX {
                    minus1 += self.engine.exp_golomb(0)?;
                }
                if minus1 >= MAX_ABS_LEVEL {
                    return Err(DecodeError::Malformed("coefficient level"));
                }
                gt1 += 1;
            } else {
                eq1 += 1;
            }
            let level = minus1 as i32 + 1;
            out.level[k] = if self.engine.bypass() { -level } else { level };
        }
        Ok(())
    }
}

/// `coded_block_flag` `condTermFlagN` of a neighbouring block (§9.3.3.1.1.9):
/// unavailable macroblocks count as coded for intra macroblocks, `I_PCM`
/// macroblocks always, skipped ones never.
fn cbf_cond(n: Option<&MbState>, intra: bool, coded: impl Fn(&MbState) -> bool) -> usize {
    usize::from(match n {
        None => intra,
        Some(m) if m.kind == MbKind::IPcm => true,
        Some(m) => coded(m),
    })
}

impl<T> MbSyntax for CabacSyntax<'_, T> {
    #[cfg(feature = "analysis")]
    fn bits_consumed(&self) -> usize {
        self.engine.bits_consumed()
    }

    #[inline]
    fn mb_type(&mut self, slice: SliceType, nb: &Neighbours<'_>) -> Result<u32> {
        match slice {
            SliceType::I => {
                let not_nxn = |m: &MbState| m.kind != MbKind::IntraNxN;
                let inc = cond(nb.a, not_nxn) + cond(nb.b, not_nxn);
                if !self.engine.decision(MB_TYPE_I + inc) {
                    return Ok(MB_TYPE_I_NXN);
                }
                Ok(self.intra_16x16_or_pcm(None))
            }
            SliceType::P => {
                if self.engine.decision(MB_TYPE_P_PREFIX) {
                    if !self.engine.decision(MB_TYPE_P_SUFFIX) {
                        return Ok(P_INTRA_BASE + MB_TYPE_I_NXN);
                    }
                    return Ok(P_INTRA_BASE + self.intra_16x16_or_pcm(Some(MB_TYPE_P_SUFFIX)));
                }
                // Table 9-37: 000 16x16, 011 16x8, 010 8x16, 001 8x8.
                let b1 = self.engine.decision(MB_TYPE_P_PREFIX + 1);
                let b2 = self.engine.decision(MB_TYPE_P_PREFIX + 2 + usize::from(b1));
                Ok(match (b1, b2) {
                    (false, false) => 0,
                    (true, true) => 1,
                    (true, false) => 2,
                    (false, true) => 3,
                })
            }
            SliceType::B => Ok(self.b_mb_type(nb)),
            // Refused at the slice header (`switching::parse`): no profile
            // has CABAC SI slices.
            SliceType::Si => Err(DecodeError::Unsupported(
                "H264 SP/SI slices outside the Extended profile",
            )),
        }
    }

    fn sub_mb_type(&mut self, slice: SliceType) -> Result<u8> {
        if slice == SliceType::B {
            return Ok(self.b_sub_mb_type());
        }
        // Table 9-38: 1 8x8, 00 8x4, 011 4x8, 010 4x4.
        if self.engine.decision(SUB_MB_TYPE_P) {
            return Ok(0);
        }
        if !self.engine.decision(SUB_MB_TYPE_P + 1) {
            return Ok(1);
        }
        Ok(if self.engine.decision(SUB_MB_TYPE_P + 2) {
            2
        } else {
            3
        })
    }

    fn transform_size_8x8_flag(&mut self, nb: &Neighbours<'_>) -> Result<bool> {
        let t8 = |m: &MbState| m.transform_8x8;
        Ok(self
            .engine
            .decision(TRANSFORM_8X8 + cond(nb.a, t8) + cond(nb.b, t8)))
    }

    fn intra_pred_mode(&mut self) -> Result<Option<u8>> {
        if self.engine.decision(PREV_INTRA_PRED_MODE) {
            return Ok(None);
        }
        let mut rem = 0;
        for bit in 0..REM_INTRA_PRED_MODE_BITS {
            rem |= u8::from(self.engine.decision(REM_INTRA_PRED_MODE)) << bit;
        }
        Ok(Some(rem))
    }

    fn intra_chroma_pred_mode(&mut self, nb: &Neighbours<'_>) -> Result<u8> {
        let directional =
            |m: &MbState| m.is_intra() && m.kind != MbKind::IPcm && m.chroma_pred_mode != 0;
        let inc = cond(nb.a, directional) + cond(nb.b, directional);
        let mut mode = 0;
        while mode < MAX_CHROMA_PRED_MODE
            && self
                .engine
                .decision(INTRA_CHROMA_PRED_MODE + if mode == 0 { inc } else { 3 })
        {
            mode += 1;
        }
        Ok(mode)
    }

    fn ref_idx(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &MbState,
        list: usize,
        x4: usize,
        y4: usize,
        num_ref_idx_active: usize,
    ) -> Result<u8> {
        // condTermFlagN (§9.3.3.1.1.6): 0 for skipped, intra and
        // direct-predicted partitions and those not using the list (-1).
        let positive = |n: Option<(&MbState, usize, usize)>| {
            usize::from(n.is_some_and(|(m, x, y)| {
                let b8 = (y / 2) * 2 + x / 2;
                !m.is_skip() && (m.direct >> b8) & 1 == 0 && m.ref_idx[list][b8] > 0
            }))
        };
        let inc = positive(nb.left(cur, x4, y4)) + 2 * positive(nb.above(cur, x4, y4));
        let mut value = 0usize;
        let mut ctx = REF_IDX + inc;
        while self.engine.decision(ctx) {
            value += 1;
            if value >= num_ref_idx_active {
                return Err(DecodeError::Malformed("ref_idx"));
            }
            ctx = REF_IDX + if value == 1 { 4 } else { 5 };
        }
        Ok(value as u8)
    }

    fn mvd(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &MbState,
        list: usize,
        x4: usize,
        y4: usize,
    ) -> Result<[i32; 2]> {
        // Intra, skipped and direct blocks and those of the other list keep
        // a zero record.
        let abs = |n: Option<(&MbState, usize, usize)>, comp: usize| {
            n.map_or(0, |(m, x, y)| {
                if m.is_skip() {
                    0
                } else {
                    u32::from(m.abs_mvd[list][y * 4 + x][comp])
                }
            })
        };
        let (left, above) = (nb.left(cur, x4, y4), nb.above(cur, x4, y4));
        let mut mvd = [0i32; 2];
        for (comp, value) in mvd.iter_mut().enumerate() {
            let sum = abs(left, comp) + abs(above, comp);
            let first_inc = if sum < MVD_SUM_LOW {
                0
            } else if sum > MVD_SUM_HIGH {
                2
            } else {
                1
            };
            let base = MVD[comp];
            let mut magnitude = 0u32;
            while magnitude < MVD_PREFIX_MAX {
                let inc = if magnitude == 0 {
                    first_inc
                } else {
                    (magnitude as usize + 2).min(6)
                };
                if !self.engine.decision(base + inc) {
                    break;
                }
                magnitude += 1;
            }
            if magnitude == MVD_PREFIX_MAX {
                magnitude += self.engine.exp_golomb(MVD_SUFFIX_ORDER)?;
            }
            if magnitude > MAX_ABS_MVD {
                return Err(DecodeError::Malformed("mvd"));
            }
            *value = magnitude as i32;
            if magnitude != 0 && self.engine.bypass() {
                *value = -*value;
            }
        }
        Ok(mvd)
    }

    fn coded_block_pattern(
        &mut self,
        nb: &Neighbours<'_>,
        _cur: &MbState,
        monochrome: bool,
    ) -> Result<u8> {
        let mut luma = 0u8;
        for b8 in 0..4 {
            let (x8, y8) = (b8 % 2, b8 / 2);
            // condTermFlagN is 1 when the neighbouring 8x8 block is
            // uncoded (a skipped macroblock's are), 0 when it is coded or
            // its macroblock is unavailable (§9.3.3.1.1.4).
            let uncoded = |m: Option<&MbState>, bit: usize| {
                usize::from(m.is_some_and(|m| m.is_skip() || (m.cbp_luma() >> bit) & 1 == 0))
            };
            let a = if x8 == 1 {
                usize::from((luma >> (b8 - 1)) & 1 == 0)
            } else {
                uncoded(nb.a, b8 + 1)
            };
            let b = if y8 == 1 {
                usize::from((luma >> (b8 - 2)) & 1 == 0)
            } else {
                uncoded(nb.b, b8 + 2)
            };
            if self.engine.decision(CBP_LUMA + a + 2 * b) {
                luma |= 1 << b8;
            }
        }
        if monochrome {
            return Ok(luma);
        }
        let chroma_cond = |m: Option<&MbState>, ac: bool| {
            cond(m, |m| {
                !m.is_skip()
                    && if ac {
                        m.cbp_chroma() == 2
                    } else {
                        m.cbp_chroma() != 0
                    }
            })
        };
        let mut chroma = 0u8;
        if self
            .engine
            .decision(CBP_CHROMA + chroma_cond(nb.a, false) + 2 * chroma_cond(nb.b, false))
        {
            chroma =
                1 + u8::from(self.engine.decision(
                    CBP_CHROMA + 4 + chroma_cond(nb.a, true) + 2 * chroma_cond(nb.b, true),
                ));
        }
        Ok(luma | chroma << 4)
    }

    fn mb_qp_delta(&mut self) -> Result<i32> {
        let mut mapped = 0u32;
        let mut ctx = MB_QP_DELTA + usize::from(self.prev_qp_delta_nonzero);
        while self.engine.decision(ctx) {
            mapped += 1;
            if mapped > MAX_MAPPED_QP_DELTA {
                return Err(DecodeError::Malformed("mb_qp_delta"));
            }
            ctx = MB_QP_DELTA + if mapped == 1 { 2 } else { 3 };
        }
        self.prev_qp_delta_nonzero = mapped != 0;
        // Table 9-3: 1, -1, 2, -2, ...
        let magnitude = mapped.div_ceil(2) as i32;
        Ok(if mapped % 2 == 1 {
            magnitude
        } else {
            -magnitude
        })
    }

    fn no_qp_delta(&mut self) {
        self.prev_qp_delta_nonzero = false;
    }

    fn luma_dc(&mut self, nb: &Neighbours<'_>, cur: &mut MbState, out: &mut Coeffs) -> Result<()> {
        let dc = |m: &MbState| m.dc_coded[0];
        let inc = cbf_cond(nb.a, true, dc) + 2 * cbf_cond(nb.b, true, dc);
        cur.dc_coded[0] = self.residual((Cat::LumaDc, cur.field), inc, out)?;
        Ok(())
    }

    fn block_4x4(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        block: Block4x4,
        out: &mut Coeffs,
    ) -> Result<()> {
        let intra = cur.is_intra();
        let (cat, left, above) = match block {
            Block4x4::LumaAc(x, y) | Block4x4::Luma(x, y) => {
                let coded = |n: Option<(&MbState, usize, usize)>| match n {
                    None => usize::from(intra),
                    Some((m, bx, by)) => {
                        cbf_cond(Some(m), intra, |m| m.total_coeff[by * 4 + bx] != 0)
                    }
                };
                let cat = if matches!(block, Block4x4::LumaAc(..)) {
                    Cat::LumaAc
                } else {
                    Cat::Luma4x4
                };
                (cat, coded(nb.left(cur, x, y)), coded(nb.above(cur, x, y)))
            }
            Block4x4::ChromaAc(comp, b) => {
                let (x, y) = (b % 2, b / 2);
                let flag =
                    |m: &MbState, x: usize, y: usize| m.chroma_total_coeff[comp][y * 2 + x] != 0;
                let left = if x > 0 {
                    usize::from(flag(cur, 0, y))
                } else {
                    cbf_cond(nb.a, intra, |m| flag(m, 1, y))
                };
                let above = if y > 0 {
                    usize::from(flag(cur, x, 0))
                } else {
                    cbf_cond(nb.b, intra, |m| flag(m, x, 1))
                };
                (Cat::ChromaAc, left, above)
            }
        };
        let coded = self.residual((cat, cur.field), left + 2 * above, out)?;
        let flag = u8::from(coded);
        match block {
            Block4x4::LumaAc(x, y) | Block4x4::Luma(x, y) => cur.total_coeff[y * 4 + x] = flag,
            Block4x4::ChromaAc(comp, b) => cur.chroma_total_coeff[comp][b] = flag,
        }
        Ok(())
    }

    fn luma_8x8(
        &mut self,
        _nb: &Neighbours<'_>,
        cur: &mut MbState,
        b8: usize,
        out: &mut Coeffs,
    ) -> Result<()> {
        // 4:2:0 8x8 blocks have no coded_block_flag: a coded 8x8 block has
        // at least one coefficient (§7.3.5.3.3).
        out.count = 0;
        let last = |i: usize| usize::from(LAST_CTX_INC_8X8[i]);
        if cur.field {
            self.significance_map(
                64,
                out,
                |i| SIGNIFICANT_8X8_FIELD + usize::from(SIGNIFICANT_CTX_INC_8X8_FIELD[i]),
                |i| LAST_SIGNIFICANT_8X8_FIELD + last(i),
            );
        } else {
            self.significance_map(
                64,
                out,
                |i| SIGNIFICANT_8X8 + usize::from(SIGNIFICANT_CTX_INC_8X8[i]),
                |i| LAST_SIGNIFICANT_8X8 + last(i),
            );
        }
        self.levels(ABS_LEVEL_8X8, false, out)?;
        let (x8, y8) = (b8 % 2, b8 / 2);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            cur.total_coeff[(2 * y8 + dy) * 4 + 2 * x8 + dx] = 1;
        }
        Ok(())
    }

    fn chroma_dc(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        comp: usize,
        out: &mut Coeffs,
    ) -> Result<()> {
        let intra = cur.is_intra();
        let dc = |m: &MbState| m.dc_coded[1 + comp];
        let inc = cbf_cond(nb.a, intra, dc) + 2 * cbf_cond(nb.b, intra, dc);
        cur.dc_coded[1 + comp] = self.residual((Cat::ChromaDc, cur.field), inc, out)?;
        Ok(())
    }

    fn pcm_samples(&mut self, out: &mut [u16], bits: [u32; 2]) -> Result<()> {
        self.engine.pcm_samples(out, bits)
    }
}
