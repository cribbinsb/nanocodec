//! CABAC binarisations and context selection of the slice data and
//! macroblock syntax elements (§9.3.2, §9.3.3.1) for frame-coded 4:2:0 and
//! 4:0:0 P and I slices.

use super::cabac::Cabac;
use super::mbstate::{MbKind, MbState, Neighbours};
use super::slice::SliceType;
use super::syntax::{Block4x4, Coeffs, MbSyntax};
use super::{DecodeError, Result};
use crate::tables::{LAST_CTX_INC_8X8, SIGNIFICANT_CTX_INC_8X8};

// ctxIdxOffset of each syntax element (Table 9-34, frame coded).
const MB_TYPE_I: usize = 3;
const MB_SKIP_P: usize = 11;
const MB_TYPE_P_PREFIX: usize = 14;
const MB_TYPE_P_SUFFIX: usize = 17;
const SUB_MB_TYPE_P: usize = 21;
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
const ABS_LEVEL: usize = 227;
const TRANSFORM_8X8: usize = 399;
const SIGNIFICANT_8X8: usize = 402;
const LAST_SIGNIFICANT_8X8: usize = 417;
const ABS_LEVEL_8X8: usize = 426;

/// `mb_type` values (Table 7-11) the binarisation produces directly.
const MB_TYPE_I_NXN: u32 = 0;
const MB_TYPE_I_PCM: u32 = 25;
/// First intra `mb_type` in P slices (Table 7-13).
const P_INTRA_BASE: u32 = 5;
/// `rem_intra_pred_mode` is a 3-bit fixed-length value.
const REM_INTRA_PRED_MODE_BITS: u32 = 3;
/// `intra_chroma_pred_mode` is truncated unary with `cMax` 3.
const MAX_CHROMA_PRED_MODE: u8 = 3;
/// Mapped `mb_qp_delta` limit: values -26..=25 (§7.4.5).
const MAX_MAPPED_QP_DELTA: u32 = 52;
/// `mvd` prefix length (UEG3, `uCoff` 9) and suffix order.
const MVD_PREFIX_MAX: u32 = 9;
const MVD_SUFFIX_ORDER: u32 = 3;
/// `coeff_abs_level_minus1` prefix length (UEG0, `uCoff` 14).
const LEVEL_PREFIX_MAX: u32 = 14;
/// Largest `|mvd|` of a conforming stream (§7.4.5.1, level limits).
const MAX_ABS_MVD: u32 = 1 << 15;
/// Largest coefficient magnitude of 8-bit video (§7.4.5.3.3 bound,
/// 2^(7 + `bitDepth`)).
const MAX_ABS_LEVEL: u32 = 1 << 15;
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

/// The CABAC syntax decoder of one slice.
pub(super) struct CabacSyntax<'a> {
    engine: Cabac<'a>,
    /// Whether the previous macroblock in decoding order had a non-zero
    /// `mb_qp_delta` (§9.3.3.1.1.5).
    prev_qp_delta_nonzero: bool,
}

/// `condTermFlag` of a whole neighbouring macroblock.
fn cond(n: Option<&MbState>, flag: impl Fn(&MbState) -> bool) -> usize {
    usize::from(n.is_some_and(flag))
}

impl<'a> CabacSyntax<'a> {
    pub fn new(engine: Cabac<'a>) -> Self {
        Self {
            engine,
            prev_qp_delta_nonzero: false,
        }
    }

    /// `mb_skip_flag` (`ctxIdxInc` §9.3.3.1.1.1).
    pub fn mb_skip_flag(&mut self, nb: &Neighbours<'_>) -> bool {
        let not_skip = |m: &MbState| !m.is_skip();
        self.engine
            .decision(MB_SKIP_P + cond(nb.a, not_skip) + cond(nb.b, not_skip))
    }

    /// `end_of_slice_flag`.
    pub fn end_of_slice(&mut self) -> bool {
        self.engine.terminate()
    }

    /// The bins of an intra `mb_type` after its first bin (Table 9-36):
    /// `I_PCM` by a terminate bin, else `Intra_16x16` coded block patterns and
    /// prediction mode. `offset` selects the I or the P-slice suffix contexts.
    fn intra_16x16_or_pcm(&mut self, suffix: bool) -> u32 {
        if self.engine.terminate() {
            return MB_TYPE_I_PCM;
        }
        // ctxIdx of the luma, chroma, chroma-AC and the two mode bins.
        let (luma, chroma, chroma_ac, mode) = if suffix {
            let base = MB_TYPE_P_SUFFIX;
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

    /// One residual block of category `cat`: `coded_block_flag` with
    /// `ctxIdxInc` `cbf_inc` (unless `cat` is `None`: 8x8 blocks, whose flag
    /// is inferred), then the significance map and levels.
    fn residual(&mut self, cat: Cat, cbf_inc: usize, out: &mut Coeffs) -> Result<bool> {
        out.count = 0;
        if !self
            .engine
            .decision(CODED_BLOCK_FLAG + cat.coded_block_flag_offset() + cbf_inc)
        {
            return Ok(false);
        }
        let sig = SIGNIFICANT + cat.significance_offset();
        let last = LAST_SIGNIFICANT + cat.significance_offset();
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

impl MbSyntax for CabacSyntax<'_> {
    fn bits_consumed(&self) -> usize {
        self.engine.bits_consumed()
    }

    fn mb_type(&mut self, slice: SliceType, nb: &Neighbours<'_>) -> Result<u32> {
        match slice {
            SliceType::I => {
                let not_nxn = |m: &MbState| m.kind != MbKind::IntraNxN;
                let inc = cond(nb.a, not_nxn) + cond(nb.b, not_nxn);
                if !self.engine.decision(MB_TYPE_I + inc) {
                    return Ok(MB_TYPE_I_NXN);
                }
                Ok(self.intra_16x16_or_pcm(false))
            }
            SliceType::P => {
                if self.engine.decision(MB_TYPE_P_PREFIX) {
                    if !self.engine.decision(MB_TYPE_P_SUFFIX) {
                        return Ok(P_INTRA_BASE + MB_TYPE_I_NXN);
                    }
                    return Ok(P_INTRA_BASE + self.intra_16x16_or_pcm(true));
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
        }
    }

    fn sub_mb_type(&mut self) -> Result<u8> {
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
        x4: usize,
        y4: usize,
        num_ref_idx_active: usize,
    ) -> Result<u8> {
        let positive = |n: Option<(&MbState, usize, usize)>| {
            usize::from(n.is_some_and(|(m, x, y)| {
                !m.is_skip() && !m.is_intra() && m.ref_idx[(y / 2) * 2 + x / 2] > 0
            }))
        };
        let inc = positive(nb.left(cur, x4, y4)) + 2 * positive(nb.above(cur, x4, y4));
        let mut value = 0usize;
        let mut ctx = REF_IDX + inc;
        while self.engine.decision(ctx) {
            value += 1;
            if value >= num_ref_idx_active {
                return Err(DecodeError::Malformed("ref_idx_l0"));
            }
            ctx = REF_IDX + if value == 1 { 4 } else { 5 };
        }
        Ok(value as u8)
    }

    fn mvd(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &MbState,
        x4: usize,
        y4: usize,
    ) -> Result<[i32; 2]> {
        let abs = |n: Option<(&MbState, usize, usize)>, comp: usize| {
            n.map_or(0, |(m, x, y)| {
                if m.is_skip() || m.is_intra() {
                    0
                } else {
                    u32::from(m.abs_mvd[y * 4 + x][comp])
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
                return Err(DecodeError::Malformed("mvd_l0"));
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
        cur.dc_coded[0] = self.residual(Cat::LumaDc, inc, out)?;
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
        let coded = self.residual(cat, left + 2 * above, out)?;
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
        self.significance_map(
            64,
            out,
            |i| SIGNIFICANT_8X8 + usize::from(SIGNIFICANT_CTX_INC_8X8[i]),
            |i| LAST_SIGNIFICANT_8X8 + usize::from(LAST_CTX_INC_8X8[i]),
        );
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
        cur.dc_coded[1 + comp] = self.residual(Cat::ChromaDc, inc, out)?;
        Ok(())
    }

    fn pcm_samples(&mut self, out: &mut [u8]) -> Result<()> {
        self.engine.pcm_samples(out)
    }
}
