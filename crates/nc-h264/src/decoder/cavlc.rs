//! CAVLC: the exp-Golomb macroblock syntax elements (§9.1) and
//! `residual_block_cavlc()` (§7.3.5.3.2, §9.2), with `nC` from the
//! neighbouring blocks' `TotalCoeff` (§9.2.1).

use super::mbstate::{MbState, Neighbours};
use super::reader::Reader;
use super::slice::SliceType;
use super::syntax::{Block4x4, Coeffs, MbSyntax};
use super::vlc::VlcTable;
use super::{DecodeError, Result};
use crate::tables::cavlc::{
    CODED_BLOCK_PATTERN_400, CODED_BLOCK_PATTERN_420, COEFF_TOKEN, RUN_BEFORE, TOTAL_ZEROS,
    TOTAL_ZEROS_CHROMA_DC,
};

/// Largest `mb_type` of I slices (Table 7-11) and P slices (Table 7-13 + 26).
const MAX_MB_TYPE_I: u32 = 25;
const MAX_MB_TYPE_P: u32 = 30;
/// Largest `sub_mb_type` of P slices (Table 7-17).
const MAX_SUB_MB_TYPE_P: u32 = 3;
const MAX_CHROMA_PRED_MODE: u32 = 3;
const REM_INTRA_PRED_MODE_BITS: u32 = 3;
/// `mb_qp_delta` range (§7.4.5).
const MIN_QP_DELTA: i32 = -26;
const MAX_QP_DELTA: i32 = 25;
/// Bound on `|mvd|` for robustness against hostile input.
const MAX_ABS_MVD: i32 = 1 << 15;
/// `nC` from which `coeff_token` is a 6-bit fixed-length code (Table 9-5).
const FLC_NC: i32 = 8;
const FLC_BITS: u8 = 6;
/// The fixed-length code of `TotalCoeff` 0 (`0000 11`).
const FLC_NO_COEFFS: u32 = 3;
/// `coeff_token` tables: the three `nC` classes of Table 9-5 (0 <= `nC` < 2,
/// 2 <= `nC` < 4, 4 <= `nC` < 8), 4:2:0 chroma DC (`nC` == -1) and the
/// fixed-length code.
const CHROMA_DC_TABLE: usize = 3;
const FLC_TABLE: usize = 4;
const COEFF_TOKEN_TABLES: usize = 5;
/// `coeff_token` table by `nC + 1`, `nC` clamped to [`FLC_NC`].
const NC_TABLE: [usize; FLC_NC as usize + 2] = [CHROMA_DC_TABLE, 0, 0, 1, 1, 2, 2, 2, 2, FLC_TABLE];
/// `coeff_token` symbols are `TotalCoeff * 4 + TrailingOnes`.
const TRAILING_ONES_BITS: u32 = 2;
/// Maximum trailing ones (`TrailingOnes` is 0..=3).
const MAX_TRAILING_ONES: usize = 3;
/// `level_prefix` from which `levelSuffixSize` grows (§9.2.2.1); escape
/// codes longer than a 16-bit level are rejected.
const LEVEL_PREFIX_ESCAPE: u32 = 15;
const MAX_LEVEL_PREFIX: u32 = 25;
/// `level_prefix` 14 with `suffixLength` 0 has a 4-bit suffix (§9.2.2.1).
const LEVEL_PREFIX_14: u32 = 14;
const LEVEL_PREFIX_14_SUFFIX: u32 = 4;
/// `suffixLength` grows while it is below this (§9.2.2.1).
const MAX_SUFFIX_LENGTH: u32 = 6;
/// Coefficients of 4:2:0 chroma DC, 4x4 and AC blocks.
const CHROMA_DC_COEFFS: usize = 4;
const BLOCK_COEFFS: usize = 16;
const AC_COEFFS: usize = 15;
/// `total_zeros` tables: 15 for 4x4 blocks, then 3 for chroma DC.
const TOTAL_ZEROS_CHROMA_DC_BASE: usize = 15;
/// `run_before` tables by `min(zerosLeft, 7)`; table 0 is the empty code.
const RUN_BEFORE_TABLES: usize = 7;

/// The CAVLC code tables, built once per decoder.
#[derive(Debug, Clone)]
pub(super) struct CavlcTables {
    /// `coeff_token` per table of [`NC_TABLE`].
    coeff_token: [VlcTable; COEFF_TOKEN_TABLES],
    /// `total_zeros` by `TotalCoeff - 1`, 4x4 blocks then chroma DC.
    total_zeros: Vec<VlcTable>,
    /// `run_before` by `min(zerosLeft, 7)`.
    run_before: Vec<VlcTable>,
}

impl CavlcTables {
    pub fn new() -> Self {
        let token = |class: usize| {
            VlcTable::new(
                COEFF_TOKEN[class]
                    .iter()
                    .enumerate()
                    .flat_map(|(total, row)| {
                        row.iter().enumerate().map(move |(ones, &(code, length))| {
                            (u32::from(code), length, (total * 4 + ones) as u16)
                        })
                    }),
            )
        };
        // §9.2.1: six bits, `TotalCoeff - 1` then `TrailingOnes`, except
        // `0000 11` for no coefficients.
        let fixed = VlcTable::new((0..1u32 << FLC_BITS).map(|code| {
            let symbol = if code == FLC_NO_COEFFS {
                0
            } else {
                (((code >> TRAILING_ONES_BITS) + 1) << TRAILING_ONES_BITS) | (code & 3)
            };
            (code, FLC_BITS, symbol as u16)
        }));
        let simple = |codes: &[(u8, u8)]| {
            VlcTable::new(
                codes
                    .iter()
                    .enumerate()
                    .map(|(symbol, &(code, length))| (u32::from(code), length, symbol as u16)),
            )
        };
        Self {
            coeff_token: [token(0), token(1), token(2), token(CHROMA_DC_TABLE), fixed],
            total_zeros: TOTAL_ZEROS
                .iter()
                .map(|codes| simple(codes))
                .chain(TOTAL_ZEROS_CHROMA_DC.iter().map(|codes| simple(codes)))
                .collect(),
            run_before: std::iter::once(VlcTable::new([]))
                .chain(RUN_BEFORE.iter().map(|codes| simple(codes)))
                .collect(),
        }
    }
}

/// The CAVLC syntax decoder of one slice.
pub(super) struct CavlcSyntax<'a, 't> {
    pub reader: Reader<'a>,
    tables: &'t CavlcTables,
}

impl<'a, 't> CavlcSyntax<'a, 't> {
    pub fn new(reader: Reader<'a>, tables: &'t CavlcTables) -> Self {
        Self { reader, tables }
    }

    /// `mb_skip_run`.
    pub fn mb_skip_run(&mut self) -> Result<u32> {
        self.reader.ue()
    }

    /// `residual_block_cavlc()` of up to `max_coeffs` coefficients with
    /// context `nc` (-1: chroma DC); returns `TotalCoeff`.
    ///
    /// The block is parsed from 32-bit peeks with unchecked consumption and
    /// one end-of-data check (as openh264 parses a block from a local bit
    /// cache): every code is one table lookup, the trailing ones are read
    /// together, and the only data-dependent loops are over the coefficients.
    fn residual(&mut self, nc: i32, max_coeffs: usize, out: &mut Coeffs) -> Result<u8> {
        out.count = 0;
        let r = &mut self.reader;
        let table = NC_TABLE[(nc.min(FLC_NC) + 1) as usize];
        let (symbol, length) = self.tables.coeff_token[table]
            .lookup(r.peek32())
            .ok_or(DecodeError::Malformed("coeff_token"))?;
        r.consume(length);
        let total = usize::from(symbol >> TRAILING_ONES_BITS);
        let trailing_ones = usize::from(symbol & 3);
        if total > max_coeffs || trailing_ones > total {
            return Err(DecodeError::Malformed("coeff_token"));
        }
        if total == 0 {
            r.check_end()?;
            return Ok(0);
        }
        let mut levels = [0i32; BLOCK_COEFFS];
        Self::levels(r, total, trailing_ones, &mut levels)?;
        let mut zeros_left = 0;
        if total < max_coeffs {
            let base = if max_coeffs == CHROMA_DC_COEFFS {
                TOTAL_ZEROS_CHROMA_DC_BASE
            } else {
                0
            };
            let (zeros, length) = self.tables.total_zeros[base + total - 1]
                .lookup(r.peek32())
                .ok_or(DecodeError::Malformed("total_zeros"))?;
            r.consume(length);
            zeros_left = usize::from(zeros);
            if zeros_left > max_coeffs - total {
                return Err(DecodeError::Malformed("total_zeros"));
            }
        }
        // Coefficients from the highest frequency down, each followed by
        // its `run_before` (§9.2.3): the zero-length code once no zeros
        // are left, so there is no branch on it.
        let mut position = total + zeros_left - 1;
        for &level in &levels[..total - 1] {
            out.push(position, level);
            let table = &self.tables.run_before[zeros_left.min(RUN_BEFORE_TABLES)];
            let (run, length) = table
                .lookup(r.peek32())
                .ok_or(DecodeError::Malformed("run_before"))?;
            r.consume(length);
            let run = usize::from(run);
            if run > zeros_left {
                return Err(DecodeError::Malformed("run_before"));
            }
            zeros_left -= run;
            position -= run + 1;
        }
        out.push(position, levels[total - 1]);
        r.check_end()?;
        Ok(total as u8)
    }

    /// The `total` levels, highest frequency first (§9.2.2).
    #[inline]
    fn levels(
        r: &mut Reader<'_>,
        total: usize,
        trailing_ones: usize,
        levels: &mut [i32; BLOCK_COEFFS],
    ) -> Result<()> {
        // `trailing_ones_sign_flag`s, all at once: 1 - 2 * flag.
        let signs = r.peek32();
        for (i, level) in levels.iter_mut().take(MAX_TRAILING_ONES).enumerate() {
            *level = 1 - ((signs >> (30 - i)) & 2) as i32;
        }
        r.consume(trailing_ones as u32);
        let mut suffix_length = u32::from(total > 10 && trailing_ones < MAX_TRAILING_ONES);
        // The first level after fewer than 3 trailing ones cannot be +-1, so
        // its code is offset by 2 (§9.2.2.1).
        let mut first_offset = if trailing_ones < MAX_TRAILING_ONES {
            2
        } else {
            0
        };
        for level in &mut levels[trailing_ones..total] {
            let word = r.peek32();
            let prefix = word.leading_zeros();
            let code = if prefix < LEVEL_PREFIX_ESCAPE {
                let suffix_size = if prefix == LEVEL_PREFIX_14 && suffix_length == 0 {
                    LEVEL_PREFIX_14_SUFFIX
                } else {
                    suffix_length
                };
                // The suffix follows the prefix's one; `>> 1 >> (31 - n)`
                // is a shift by `32 - n` that stays defined for `n == 0`.
                let suffix = (word << prefix << 1) >> 1 >> (31 - suffix_size);
                r.consume(prefix + 1 + suffix_size);
                (prefix << suffix_length) + suffix
            } else {
                Self::escape_level_code(r, prefix, suffix_length)?
            };
            let code = code + first_offset;
            first_offset = 0;
            // Even codes are positive, odd ones negative.
            let magnitude = ((code >> 1) + 1) as i32;
            let value = if code & 1 == 0 { magnitude } else { -magnitude };
            *level = value;
            suffix_length = suffix_length.max(1);
            let grows = value.unsigned_abs() > (3 << (suffix_length - 1))
                && suffix_length < MAX_SUFFIX_LENGTH;
            suffix_length += u32::from(grows);
        }
        Ok(())
    }

    /// `levelCode` of a `level_prefix` of 15 or more (§9.2.2.1).
    #[cold]
    fn escape_level_code(r: &mut Reader<'_>, prefix: u32, suffix_length: u32) -> Result<u32> {
        if prefix > MAX_LEVEL_PREFIX {
            return Err(DecodeError::Malformed("level_prefix"));
        }
        r.consume(prefix + 1);
        let suffix_size = prefix - 3;
        let mut code = (LEVEL_PREFIX_ESCAPE << suffix_length) + r.peek(suffix_size);
        r.consume(suffix_size);
        if suffix_length == 0 {
            code += 15;
        }
        if prefix >= 16 {
            code += (1 << (prefix - 3)) - 4096;
        }
        Ok(code)
    }
}

/// One `residual_block_cavlc()` of `max_coeffs` coefficients with context
/// `nc` read from `bytes`, as levels in scan order: the encoder's tests
/// check its CAVLC writer against this parser.
#[cfg(test)]
pub(crate) fn block_for_tests(bytes: &[u8], nc: i32, max_coeffs: usize) -> Vec<i32> {
    let tables = CavlcTables::new();
    let mut syntax = CavlcSyntax::new(Reader::new(bytes), &tables);
    let mut coeffs = Coeffs::new();
    syntax
        .residual(nc, max_coeffs, &mut coeffs)
        .expect("a well-formed block");
    let mut levels = vec![0; max_coeffs];
    for (scan, level) in coeffs.iter() {
        levels[scan] = level;
    }
    levels
}

/// `nC` from the neighbouring blocks' counts (§9.2.1).
fn predicted_nc(left: Option<u8>, above: Option<u8>) -> i32 {
    match (left, above) {
        (Some(a), Some(b)) => (i32::from(a) + i32::from(b) + 1) >> 1,
        (Some(n), None) | (None, Some(n)) => i32::from(n),
        (None, None) => 0,
    }
}

/// `nC` of the luma 4x4 block (`x4`, `y4`) of `cur`.
fn luma_nc(nb: &Neighbours<'_>, cur: &MbState, x4: usize, y4: usize) -> i32 {
    let count = |n: Option<(&MbState, usize, usize)>| n.map(|(m, x, y)| m.total_coeff[y * 4 + x]);
    predicted_nc(count(nb.left(cur, x4, y4)), count(nb.above(cur, x4, y4)))
}

impl MbSyntax for CavlcSyntax<'_, '_> {
    fn bits_consumed(&self) -> usize {
        self.reader.position()
    }

    fn mb_type(&mut self, slice: SliceType, _nb: &Neighbours<'_>) -> Result<u32> {
        let max = match slice {
            SliceType::I => MAX_MB_TYPE_I,
            SliceType::P => MAX_MB_TYPE_P,
        };
        self.reader.ue_max(max, "mb_type")
    }

    fn sub_mb_type(&mut self) -> Result<u8> {
        Ok(self.reader.ue_max(MAX_SUB_MB_TYPE_P, "sub_mb_type")? as u8)
    }

    fn transform_size_8x8_flag(&mut self, _nb: &Neighbours<'_>) -> Result<bool> {
        self.reader.flag()
    }

    fn intra_pred_mode(&mut self) -> Result<Option<u8>> {
        if self.reader.flag()? {
            return Ok(None);
        }
        Ok(Some(self.reader.bits(REM_INTRA_PRED_MODE_BITS)? as u8))
    }

    fn intra_chroma_pred_mode(&mut self, _nb: &Neighbours<'_>) -> Result<u8> {
        Ok(self
            .reader
            .ue_max(MAX_CHROMA_PRED_MODE, "intra_chroma_pred_mode")? as u8)
    }

    fn ref_idx(
        &mut self,
        _nb: &Neighbours<'_>,
        _cur: &MbState,
        _x4: usize,
        _y4: usize,
        num_ref_idx_active: usize,
    ) -> Result<u8> {
        Ok(self.reader.te(num_ref_idx_active as u32 - 1)? as u8)
    }

    fn mvd(
        &mut self,
        _nb: &Neighbours<'_>,
        _cur: &MbState,
        _x4: usize,
        _y4: usize,
    ) -> Result<[i32; 2]> {
        let mut mvd = [0; 2];
        for value in &mut mvd {
            *value = self.reader.se_range(-MAX_ABS_MVD, MAX_ABS_MVD, "mvd_l0")?;
        }
        Ok(mvd)
    }

    fn coded_block_pattern(
        &mut self,
        _nb: &Neighbours<'_>,
        cur: &MbState,
        monochrome: bool,
    ) -> Result<u8> {
        let column = usize::from(!cur.is_intra());
        let code = self.reader.ue()? as usize;
        let table: &[u8] = if monochrome {
            &CODED_BLOCK_PATTERN_400[column]
        } else {
            &CODED_BLOCK_PATTERN_420[column]
        };
        table
            .get(code)
            .copied()
            .ok_or(DecodeError::Malformed("coded_block_pattern"))
    }

    fn mb_qp_delta(&mut self) -> Result<i32> {
        self.reader
            .se_range(MIN_QP_DELTA, MAX_QP_DELTA, "mb_qp_delta")
    }

    fn luma_dc(&mut self, nb: &Neighbours<'_>, cur: &mut MbState, out: &mut Coeffs) -> Result<()> {
        let nc = luma_nc(nb, cur, 0, 0);
        self.residual(nc, BLOCK_COEFFS, out)?;
        Ok(())
    }

    fn block_4x4(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        block: Block4x4,
        out: &mut Coeffs,
    ) -> Result<()> {
        match block {
            Block4x4::LumaAc(x, y) | Block4x4::Luma(x, y) => {
                let max = if matches!(block, Block4x4::LumaAc(..)) {
                    AC_COEFFS
                } else {
                    BLOCK_COEFFS
                };
                let nc = luma_nc(nb, cur, x, y);
                cur.total_coeff[y * 4 + x] = self.residual(nc, max, out)?;
            }
            Block4x4::ChromaAc(comp, b) => {
                let (x, y) = (b % 2, b / 2);
                let count = |m: &MbState, x: usize, y: usize| m.chroma_total_coeff[comp][y * 2 + x];
                let left = if x > 0 {
                    Some(count(cur, 0, y))
                } else {
                    nb.a.map(|m| count(m, 1, y))
                };
                let above = if y > 0 {
                    Some(count(cur, x, 0))
                } else {
                    nb.b.map(|m| count(m, x, 1))
                };
                cur.chroma_total_coeff[comp][b] =
                    self.residual(predicted_nc(left, above), AC_COEFFS, out)?;
            }
        }
        Ok(())
    }

    fn luma_8x8(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        b8: usize,
        out: &mut Coeffs,
    ) -> Result<()> {
        // Four interleaved 4x4 parses: coefficient k of part i is 8x8 scan
        // position 4k + i (§7.3.5.3.2).
        let mut part = Coeffs::new();
        let mut combined = Coeffs::new();
        for i in 0..4 {
            let (x, y) = ((b8 % 2) * 2 + i % 2, (b8 / 2) * 2 + i / 2);
            let nc = luma_nc(nb, cur, x, y);
            cur.total_coeff[y * 4 + x] = self.residual(nc, BLOCK_COEFFS, &mut part)?;
            for (scan, level) in part.iter() {
                combined.push(4 * scan + i, level);
            }
        }
        *out = combined;
        Ok(())
    }

    fn chroma_dc(
        &mut self,
        _nb: &Neighbours<'_>,
        _cur: &mut MbState,
        _comp: usize,
        out: &mut Coeffs,
    ) -> Result<()> {
        self.residual(-1, CHROMA_DC_COEFFS, out)?;
        Ok(())
    }

    fn pcm_samples(&mut self, out: &mut [u8]) -> Result<()> {
        self.reader.align(false)?;
        let bytes = self.reader.remaining_bytes();
        let samples = bytes.get(..out.len()).ok_or(DecodeError::Truncated)?;
        out.copy_from_slice(samples);
        self.reader.skip(out.len() * 8)
    }
}
