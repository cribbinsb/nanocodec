//! Coefficient syntax of one transform block (AV1 §5.11.39 `coeffs()`,
//! contexts §8.3.2) for every transform size and type: the all-zero flag,
//! the luma transform type (§5.11.47), the end of block, levels in reverse
//! scan order with the context of their transform class, signs and Golomb
//! remainders, dequantised (§7.12.3) straight into the coefficient buffer.
//!
//! The structure is dav1d 1.5.4 `src/recon_tmpl.c` `decode_coefs`
//! (BSD-2-Clause, Copyright (c) 2018, `VideoLAN` and dav1d authors; see
//! `THIRD_PARTY.md`): the end of block as a scan index, the level pass of
//! each transform class in [`levels`], then the signs and remainders along
//! the chain of nonzero coefficients that pass leaves in the coefficient
//! buffer. Syntax tables follow libaom `av1/decoder/decodetxb.c`
//! (BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media).
//! Coefficients are column-major (`col * coded_height + row`), as the
//! inverse transforms take them.
mod levels;

use super::transform::{TX_SIZE_CONTEXTS, TxClass, TxSet, TxSize};
use crate::cdf::ResetCounters;
use crate::coeff::{PlaneType, TxbContext};
use crate::decoder_range::{Cdf, DecoderDefaults, SymbolReader};
use crate::tables::{
    COEFF_BASE_CDF, COEFF_BASE_EOB_CDF, COEFF_BR_CDF, DC_SIGN_CDF, EOB_EXTRA_CDF, EOB_PT_16_CDF,
    EOB_PT_32_CDF, EOB_PT_64_CDF, EOB_PT_128_CDF, EOB_PT_256_CDF, EOB_PT_512_CDF, EOB_PT_1024_CDF,
    INTER_EXT_TX_CDF, INTRA_EXT_TX_CDF, INTRA_MODES, TXB_SKIP_CDF,
};
use levels::{
    CLASS_2D, CLASS_H, CLASS_V, LINK_MASK, LevelBuffer, LevelCdfs, MAX_TOKEN, TOKEN_SHIFT, Tokens,
    Walk,
};
use nc_common::decode::DecodeError;
use nc_dsp::Av1TxType;

/// Levels `coeff_br` adds at most (`COEFF_BASE_RANGE`).
const COEFF_BASE_RANGE: u8 = 12;
/// Symbols per `coeff_br` (`BR_CDF_SIZE`).
const BR_CDF_SIZE: usize = 4;
/// Longest Golomb prefix (libaom rejects longer codes).
const MAX_GOLOMB_LENGTH: u32 = 20;
/// `eob_pt` symbols below which no `eob_extra` is coded: they stand for
/// the last scan index 0 or 1 themselves.
const EOB_EXTRA_FIRST: usize = 2;
/// Coefficient magnitudes keep 20 bits and dequantised values 24 before
/// the shift (§7.12.3; libaom masks the same).
const LEVEL_MASK: u32 = 0xf_ffff;
const DEQUANT_MASK: u64 = 0xff_ffff;
/// Dequantised coefficients clamp to `BitDepth + 8` bits.
const COEFF_MIN: i32 = -(1 << 15);
const COEFF_MAX: i32 = (1 << 15) - 1;
/// The largest 8-bit quantiser step (`Ac_Qlookup[0][255]`). A token below
/// [`MAX_TOKEN`] times it stays below both the 24-bit wrap and
/// [`COEFF_MAX`], so those coefficients need neither (dav1d's `ac_noqm`).
const MAX_STEP: i32 = 1828;
const _: () = assert!((MAX_TOKEN as i32 - 1) * MAX_STEP <= COEFF_MAX);
/// The entropy context a block leaves: the level sum saturates at 3 bits,
/// the DC sign category above it (libaom `COEFF_CONTEXT_MASK`, `set_dc_sign`).
const CONTEXT_LEVEL_MAX: u32 = 7;
const DC_SIGN_SHIFT: u32 = 3;
const DC_NEGATIVE: u8 = 1 << DC_SIGN_SHIFT;
const DC_POSITIVE: u8 = 2 << DC_SIGN_SHIFT;

/// CDF array shapes (libaom `entropy.h`).
const PLANE_TYPES: usize = 2;
const TXB_SKIP_CONTEXTS: usize = 13;
const EOB_MULTI_CONTEXTS: usize = 2;
const EOB_COEF_CONTEXTS: usize = 9;
const SIG_COEF_CONTEXTS_EOB: usize = 4;
const SIG_COEF_CONTEXTS: usize = 42;
const LEVEL_CONTEXTS: usize = 21;
const DC_SIGN_CONTEXTS: usize = 3;
const TX_TYPES: usize = 16;
const EXT_TX_SIZES: usize = 4;
const EXT_TX_SETS_INTRA: usize = 3;
const EXT_TX_SETS_INTER: usize = 4;
/// `coeff_br` CDFs stop at the 32x32 context (libaom `AOMMIN(txs_ctx,
/// TX_32X32)`).
const BR_SIZE_CONTEXTS: usize = 4;

/// The transform-type and coefficient CDFs of a tile.
#[derive(Debug, Clone)]
pub(crate) struct CoefficientCdfs {
    intra_tx_type: [[[Cdf<TX_TYPES>; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA],
    inter_tx_type: [[Cdf<TX_TYPES>; EXT_TX_SIZES]; EXT_TX_SETS_INTER],
    txb_skip: [[Cdf<2>; TXB_SKIP_CONTEXTS]; TX_SIZE_CONTEXTS],
    eob_pt_16: [[Cdf<5>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_32: [[Cdf<6>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_64: [[Cdf<7>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_128: [[Cdf<8>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_256: [[Cdf<9>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_512: [[Cdf<10>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_1024: [[Cdf<11>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_extra: [[[Cdf<2>; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZE_CONTEXTS],
    base_eob: [[[Cdf<3>; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES]; TX_SIZE_CONTEXTS],
    base: [[[Cdf<4>; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZE_CONTEXTS],
    br: [[[Cdf<BR_CDF_SIZE>; LEVEL_CONTEXTS]; PLANE_TYPES]; BR_SIZE_CONTEXTS],
    dc_sign: [[Cdf<2>; DC_SIGN_CONTEXTS]; PLANE_TYPES],
}

impl CoefficientCdfs {
    /// The defaults for coefficient CDF set `q_ctx` (§7.20
    /// `init_coeff_cdfs`; the transform-type CDFs do not depend on it).
    pub(crate) fn new(q_ctx: usize) -> Self {
        let br = COEFF_BR_CDF[q_ctx].cdfs();
        Self {
            intra_tx_type: INTRA_EXT_TX_CDF.cdfs(),
            inter_tx_type: INTER_EXT_TX_CDF.cdfs(),
            txb_skip: TXB_SKIP_CDF[q_ctx].cdfs(),
            eob_pt_16: EOB_PT_16_CDF[q_ctx].cdfs(),
            eob_pt_32: EOB_PT_32_CDF[q_ctx].cdfs(),
            eob_pt_64: EOB_PT_64_CDF[q_ctx].cdfs(),
            eob_pt_128: EOB_PT_128_CDF[q_ctx].cdfs(),
            eob_pt_256: EOB_PT_256_CDF[q_ctx].cdfs(),
            eob_pt_512: EOB_PT_512_CDF[q_ctx].cdfs(),
            eob_pt_1024: EOB_PT_1024_CDF[q_ctx].cdfs(),
            eob_extra: EOB_EXTRA_CDF[q_ctx].cdfs(),
            base_eob: COEFF_BASE_EOB_CDF[q_ctx].cdfs(),
            base: COEFF_BASE_CDF[q_ctx].cdfs(),
            br: std::array::from_fn(|i| br[i]),
            dc_sign: DC_SIGN_CDF[q_ctx].cdfs(),
        }
    }
}

impl ResetCounters for CoefficientCdfs {
    fn reset_counters(&mut self) {
        self.intra_tx_type.reset_counters();
        self.inter_tx_type.reset_counters();
        self.txb_skip.reset_counters();
        self.eob_pt_16.reset_counters();
        self.eob_pt_32.reset_counters();
        self.eob_pt_64.reset_counters();
        self.eob_pt_128.reset_counters();
        self.eob_pt_256.reset_counters();
        self.eob_pt_512.reset_counters();
        self.eob_pt_1024.reset_counters();
        self.eob_extra.reset_counters();
        self.base_eob.reset_counters();
        self.base.reset_counters();
        self.br.reset_counters();
        self.dc_sign.reset_counters();
    }
}

/// How a transform block's type is found.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TxTypeSource {
    /// Luma with a coded type (§5.11.47): the set, and the intra direction
    /// (`intraDir`) whose CDF codes it, or `None` for an inter block.
    Coded {
        set: TxSet,
        intra_dir: Option<usize>,
    },
    /// Known without syntax (chroma, a one-type set, `qindex` 0).
    Known(Av1TxType),
}

/// Dequantiser steps of one plane (§7.12.2 `dc_q`, `ac_q`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Steps {
    pub dc: i32,
    pub ac: i32,
}

/// What a transform block's coefficients decoded to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Coded {
    /// End of block: 0 when nothing is coded.
    pub eob: usize,
    /// The transform type (`DCT_DCT` when nothing is coded).
    pub kind: Av1TxType,
    /// The entropy context the block leaves its neighbours.
    pub context: u8,
}

impl Coded {
    pub(crate) const NONE: Self = Self {
        eob: 0,
        kind: Av1TxType::DctDct,
        context: 0,
    };
}

/// Coefficient scratch of the decoder, reused for every transform block.
pub(crate) struct CoefficientBuffer {
    levels: LevelBuffer,
    /// Storage positions of the block's nonzero coefficients, which
    /// [`CoefficientBuffer::clear`] zeroes.
    nonzero: [u16; MAX_CODED],
    nonzero_count: usize,
    /// Dequantised coefficients of the block being decoded; all zero
    /// between blocks ([`CoefficientBuffer::clear`]). Between the level
    /// and sign passes they hold the tokens and their chain.
    pub coefficients: [i32; MAX_CODED],
}

/// Coefficients of the largest coded area (32x32).
const MAX_CODED: usize = 1024;

impl CoefficientBuffer {
    pub(crate) fn new() -> Box<Self> {
        Box::new(Self {
            levels: LevelBuffer::new(),
            nonzero: [0; MAX_CODED],
            nonzero_count: 0,
            coefficients: [0; MAX_CODED],
        })
    }
    /// Zero what the last block wrote, so the buffer is clear for the
    /// next: only its nonzero positions, not the whole area.
    pub(crate) fn clear(&mut self) {
        for &pos in &self.nonzero[..self.nonzero_count] {
            self.coefficients[usize::from(pos)] = 0;
        }
        self.nonzero_count = 0;
    }
}

/// The entropy contexts the neighbouring transform blocks left along a
/// block's edges, from which its `all_zero` and DC sign contexts follow
/// (the latter only for a nonzero DC coefficient, as dav1d).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TxbNeighbours<'a> {
    pub above: &'a [u8],
    pub left: &'a [u8],
    /// Whether the transform covers its whole (plane) block.
    pub fills_block: bool,
}

/// One transform block to read.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Request<'a> {
    pub tx: TxSize,
    pub plane: PlaneType,
    pub neighbours: TxbNeighbours<'a>,
    pub source: TxTypeSource,
    pub steps: Steps,
}

/// Read one transform block's coefficients (§5.11.39) into
/// `buffer.coefficients` (zero on entry, column-major over the coded
/// area), dequantised with the request's steps and `dqDenom`.
///
/// # Errors
/// Malformed entropy-coded data.
#[inline(never)]
pub(crate) fn read(
    cdfs: &mut CoefficientCdfs,
    r: &mut SymbolReader<'_>,
    buffer: &mut CoefficientBuffer,
    request: Request<'_>,
) -> Result<Coded, DecodeError> {
    let Request {
        tx,
        plane,
        neighbours,
        source,
        steps,
    } = request;
    let size_ctx = tx.context();
    let p = plane as usize;
    let TxbNeighbours {
        above,
        left,
        fills_block,
    } = neighbours;
    let skip_ctx = TxbContext::skip_context(plane, fills_block, above, left);
    if r.boolean(&mut cdfs.txb_skip[size_ctx][skip_ctx]) {
        return Ok(Coded::NONE);
    }
    let kind = match source {
        TxTypeSource::Known(kind) => kind,
        TxTypeSource::Coded { set, intra_dir } => read_tx_type(cdfs, r, tx, set, intra_dir)?,
    };
    let class = TxClass::of(kind);
    let last = read_last(cdfs, r, tx, size_ctx, p, class);
    let size_log2 = tx.eob_alphabet();
    let level_cdfs = LevelCdfs::of(cdfs, size_ctx, size_ctx.min(BR_SIZE_CONTEXTS - 1), p);
    let CoefficientBuffer {
        levels,
        coefficients,
        ..
    } = &mut *buffer;
    let tokens = match class {
        TxClass::TwoD => levels::read::<CLASS_2D>(
            level_cdfs,
            r,
            levels,
            coefficients,
            Walk::new::<CLASS_2D>(tx),
            last,
            size_log2,
        ),
        TxClass::Horizontal => levels::read::<CLASS_H>(
            level_cdfs,
            r,
            levels,
            coefficients,
            Walk::new::<CLASS_H>(tx),
            last,
            size_log2,
        ),
        TxClass::Vertical => levels::read::<CLASS_V>(
            level_cdfs,
            r,
            levels,
            coefficients,
            Walk::new::<CLASS_V>(tx),
            last,
            size_log2,
        ),
    };
    let dc_sign = &mut cdfs.dc_sign[p];
    let context = read_signs(
        r,
        (dc_sign, neighbours),
        buffer,
        tokens,
        steps,
        tx.dequant_shift(),
    )?;
    Ok(Coded {
        eob: last + 1,
        kind,
        context,
    })
}

/// Signs and Golomb remainders of the nonzero coefficients in scan order
/// (DC first, with its own sign CDF, then the chain the level pass left),
/// dequantising each; returns the entropy context the block leaves. The
/// DC sign is read with one of `dc_sign`'s CDFs, which the neighbours pick.
fn read_signs(
    r: &mut SymbolReader<'_>,
    dc_sign: (&mut [Cdf<2>; DC_SIGN_CONTEXTS], TxbNeighbours<'_>),
    buffer: &mut CoefficientBuffer,
    tokens: Tokens,
    steps: Steps,
    shift: u32,
) -> Result<u8, DecodeError> {
    let CoefficientBuffer {
        nonzero,
        nonzero_count,
        coefficients,
        ..
    } = buffer;
    let mut count = 0;
    let mut level_sum = 0;
    let mut dc_category = 0;
    if tokens.dc != 0 {
        let (cdfs, TxbNeighbours { above, left, .. }) = dc_sign;
        let negative = r.boolean(&mut cdfs[TxbContext::dc_sign_context(above, left)]);
        dc_category = if negative { DC_NEGATIVE } else { DC_POSITIVE };
        let (value, level) = coefficient(r, tokens.dc, negative, steps.dc, shift)?;
        level_sum += level;
        coefficients[0] = value;
        nonzero[0] = 0;
        count = 1;
    }
    let mut rc = tokens.first;
    while rc != 0 {
        let negative = r.literal(1) != 0;
        let entry = coefficients[rc & (MAX_CODED - 1)];
        let tok = (entry >> TOKEN_SHIFT) as u32;
        let (value, level) = coefficient(r, tok, negative, steps.ac, shift)?;
        level_sum += level;
        coefficients[rc & (MAX_CODED - 1)] = value;
        nonzero[count & (MAX_CODED - 1)] = rc as u16;
        count += 1;
        rc = (entry & LINK_MASK) as usize;
    }
    *nonzero_count = count;
    Ok(level_sum.min(CONTEXT_LEVEL_MAX) as u8 | dc_category)
}

/// A nonzero coefficient with token `tok` and its sign: the dequantised
/// value (§7.12.3) and the magnitude, with any Golomb remainder.
#[inline]
fn coefficient(
    r: &mut SymbolReader<'_>,
    tok: u32,
    negative: bool,
    step: i32,
    shift: u32,
) -> Result<(i32, u32), DecodeError> {
    debug_assert!((0..=MAX_STEP).contains(&step));
    if tok < MAX_TOKEN {
        // Below the 24-bit wrap and the clamp: see [`MAX_STEP`].
        let magnitude = (tok as i32 * step) >> shift;
        return Ok((if negative { -magnitude } else { magnitude }, tok));
    }
    let level = (tok + read_golomb(r)?) & LEVEL_MASK;
    // The product wraps at 24 bits, is divided by `dqDenom` and clamped to
    // the coefficient range.
    let magnitude = (((u64::from(level) * step as u64) & DEQUANT_MASK) >> shift) as i32;
    let value = if negative { -magnitude } else { magnitude };
    Ok((value.clamp(COEFF_MIN, COEFF_MAX), level))
}

/// A Golomb-coded remainder (§5.11.39 `read_golomb`). Out of line: rare
/// (levels of 15 and above).
#[cold]
#[inline(never)]
fn read_golomb(r: &mut SymbolReader<'_>) -> Result<u32, DecodeError> {
    let mut zeros = 0;
    while r.literal(1) == 0 {
        zeros += 1;
        if zeros >= MAX_GOLOMB_LENGTH {
            return Err(DecodeError::Malformed("AV1 coefficient magnitude"));
        }
    }
    Ok((1u32 << zeros) + r.literal(zeros) - 1)
}

/// `eob_pt_*` and `eob_extra` (§5.11.39) as dav1d reads them: the scan
/// index of the last coefficient (`eob - 1`). `eob_pt` symbol `k` above 1
/// stands for indices from `1 << (k - 1)`: `eob_extra` gives the next bit,
/// literal bits the rest.
// Out of line (once per block): its seven inlined symbol decoders would
// otherwise sit inside the level loops' function and crowd the I-cache.
#[inline(never)]
fn read_last(
    cdfs: &mut CoefficientCdfs,
    r: &mut SymbolReader<'_>,
    tx: TxSize,
    size_ctx: usize,
    p: usize,
    class: TxClass,
) -> usize {
    let m = usize::from(class != TxClass::TwoD);
    let symbol = match tx.eob_alphabet() {
        0 => r.symbol(&mut cdfs.eob_pt_16[p][m], 5),
        1 => r.symbol(&mut cdfs.eob_pt_32[p][m], 6),
        2 => r.symbol(&mut cdfs.eob_pt_64[p][m], 7),
        3 => r.symbol(&mut cdfs.eob_pt_128[p][m], 8),
        4 => r.symbol(&mut cdfs.eob_pt_256[p][m], 9),
        5 => r.symbol(&mut cdfs.eob_pt_512[p][m], 10),
        _ => r.symbol(&mut cdfs.eob_pt_1024[p][m], 11),
    };
    if symbol < EOB_EXTRA_FIRST {
        return symbol;
    }
    let bits = (symbol - EOB_EXTRA_FIRST) as u32;
    let high = r.boolean(&mut cdfs.eob_extra[size_ctx][p][symbol - EOB_EXTRA_FIRST]);
    ((2 | usize::from(high)) << bits) | r.literal(bits) as usize
}

/// `inter_tx_type` / `intra_tx_type` (§5.11.47). Out of line, as
/// [`read_last`].
#[inline(never)]
fn read_tx_type(
    cdfs: &mut CoefficientCdfs,
    r: &mut SymbolReader<'_>,
    tx: TxSize,
    set: TxSet,
    intra_dir: Option<usize>,
) -> Result<Av1TxType, DecodeError> {
    let (index, size) = set
        .cdf_index(intra_dir.is_none(), tx)
        .ok_or(DecodeError::Malformed("AV1 transform set"))?;
    let cdf = match intra_dir {
        Some(dir) => &mut cdfs.intra_tx_type[index][size][dir],
        None => &mut cdfs.inter_tx_type[index][size],
    };
    Ok(set.kind(r.symbol(cdf, set.symbols())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coeff::{CoeffCdfs, end_of_block};
    use crate::range_coder::SymbolWriter;
    use crate::transform::TxSize as EncoderTxSize;

    /// The dequantised value of `level` at unit steps (§7.12.3).
    fn dequantized(level: i32, shift: u32) -> i32 {
        let magnitude =
            ((u64::from(level.unsigned_abs() & LEVEL_MASK) & DEQUANT_MASK) >> shift) as i32;
        (if level < 0 { -magnitude } else { magnitude }).clamp(COEFF_MIN, COEFF_MAX)
    }

    /// Whatever the encoder's square `DCT_DCT` coefficient writer codes,
    /// the decoder's general reader reads back, with the same adaptation.
    #[test]
    fn square_dct_blocks_round_trip_with_the_encoder() {
        let sizes = [
            EncoderTxSize::Tx4x4,
            EncoderTxSize::Tx8x8,
            EncoderTxSize::Tx16x16,
            EncoderTxSize::Tx32x32,
            EncoderTxSize::Tx64x64,
        ];
        for q in 0..4 {
            for encoder_tx in sizes {
                let tx = TxSize::new(encoder_tx.log2(), encoder_tx.log2()).unwrap();
                let mut encoder = CoeffCdfs::new(q);
                let mut decoder = CoefficientCdfs::new(q);
                let mut buffer = CoefficientBuffer::new();
                for inter in [false, true] {
                    for pattern in 0..4 {
                        let n = tx.coded_area();
                        let levels: Vec<i32> = (0..n)
                            .map(|i| match pattern {
                                1 => i32::from(i == 0) * -32_768,
                                2 if i % 7 == 0 => (i % 39) as i32 - 19,
                                3 if i % 3 == 0 => 1 << 19,
                                3 => -17,
                                _ => 0,
                            })
                            .collect();
                        let eob = end_of_block(encoder_tx, &levels);
                        // A full-block luma transform (skip context 0) with a
                        // negative DC neighbour (DC sign context 1).
                        let above = [DC_NEGATIVE];
                        let neighbours = TxbNeighbours {
                            above: &above,
                            left: &[0],
                            fills_block: true,
                        };
                        let ctx = TxbContext {
                            skip: 0,
                            dc_sign: 1,
                        };
                        assert_eq!(TxbContext::derive(PlaneType::Luma, true, &above, &[0]), ctx);
                        let mut writer = SymbolWriter::new();
                        encoder.write(
                            &mut writer,
                            encoder_tx,
                            PlaneType::Luma,
                            inter,
                            0,
                            ctx,
                            &levels,
                            eob,
                        );
                        let bytes = writer.finish();
                        let mut reader = SymbolReader::new(&bytes);
                        let set = TxSet::of(tx, inter, false);
                        let source = if set == TxSet::DctOnly {
                            TxTypeSource::Known(Av1TxType::DctDct)
                        } else {
                            TxTypeSource::Coded {
                                set,
                                intra_dir: (!inter).then_some(0),
                            }
                        };
                        let request = Request {
                            tx,
                            plane: PlaneType::Luma,
                            neighbours,
                            source,
                            steps: Steps { dc: 1, ac: 1 },
                        };
                        let coded = read(&mut decoder, &mut reader, &mut buffer, request).unwrap();
                        assert_eq!((coded.eob, coded.kind), (eob, Av1TxType::DctDct));
                        let expected: Vec<i32> = levels
                            .iter()
                            .map(|&l| dequantized(l, tx.dequant_shift()))
                            .collect();
                        assert_eq!(&buffer.coefficients[..n], &expected[..]);
                        buffer.clear();
                        assert!(buffer.coefficients.iter().all(|&c| c == 0));
                    }
                }
            }
        }
    }
}
