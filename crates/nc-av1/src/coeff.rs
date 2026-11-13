//! AV1 transform-coefficient coding (AV1 §5.11.39 `coeffs()`, contexts §8.3.2).
//!
//! Ported from libaom's `av1_write_coeffs_txb` and `txb_common.h`
//! (BSD-2-Clause, Alliance for Open Media), in libaom's coefficient layout:
//! index `c * n + r` for row `r`, column `c` (see `transform.rs`). Contexts
//! are computed from the complete set of levels; that is equivalent to the
//! decoder's partial view because every context only looks at coefficients
//! later in scan order, which the decoder has already decoded.
//!
//! Only `DCT_DCT` (transform class 2D) and square sizes are supported;
//! 64x64 transforms code their top-left 32x32 as a 32x32 block.

use crate::cdf::{Cdf, FromDefaults, ResetCounters};
use crate::range_coder::SymbolSink;
use crate::tables::{
    COEFF_BASE_CDF, COEFF_BASE_EOB_CDF, COEFF_BR_CDF, DC_SIGN_CDF, DEFAULT_SCAN_4X4,
    DEFAULT_SCAN_8X8, DEFAULT_SCAN_16X16, DEFAULT_SCAN_32X32, EOB_EXTRA_CDF, EOB_PT_16_CDF,
    EOB_PT_64_CDF, EOB_PT_256_CDF, EOB_PT_1024_CDF, INTER_EXT_TX_CDF, INTRA_EXT_TX_CDF,
    INTRA_MODES, NZ_MAP_CTX_OFFSET_4X4, NZ_MAP_CTX_OFFSET_8X8, NZ_MAP_CTX_OFFSET_16X16,
    NZ_MAP_CTX_OFFSET_32X32, TXB_SKIP_CDF,
};
use crate::transform::TxSize;
use nc_encoder::symbol_cost::BitCounter;

/// SVT's linear coefficient-rate estimate ([`CoeffCdfs::eob_linear_bits`]):
/// fixed cost, per-coefficient slopes for sparse and dense blocks (an end of
/// block below area / 64 is sparse), in 1/512 bit.
const EOB_LINEAR_OFFSET: f32 = 6000.0;
const EOB_LINEAR_SPARSE_SLOPE: f32 = 1000.0;
const EOB_LINEAR_DENSE_SLOPE: f32 = 400.0;
const EOB_LINEAR_DENSE_DIVISOR: usize = 64;
const EOB_LINEAR_UNITS_PER_BIT: f32 = 512.0;

/// Levels coded by `coeff_base` / `coeff_base_eob` (AV1 `NUM_BASE_LEVELS`).
const NUM_BASE_LEVELS: i32 = 2;
/// Extra levels coded by `coeff_br` (AV1 `COEFF_BASE_RANGE`).
const COEFF_BASE_RANGE: i32 = 12;
/// Symbols per `coeff_br` CDF (AV1 `BR_CDF_SIZE`).
const BR_CDF_SIZE: i32 = 4;
/// Largest level whose value is fully coded by base + range symbols.
const MAX_BASE_BR_RANGE: i32 = NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1;
/// Entropy contexts keep the level sum saturated at 3 bits (libaom
/// `COEFF_CONTEXT_MASK`); only the classes 0 / 1–3 / 4+ matter.
const CUL_LEVEL_MASK: u8 = 7;
/// Shift of the DC-sign category within an entropy context byte.
const DC_SIGN_SHIFT: u32 = 3;
/// Start of each `eob_pt` group (libaom `av1_eob_group_start`).
const EOB_GROUP_START: [usize; 12] = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513];
/// Extra bits after each `eob_pt` (libaom `av1_eob_offset_bits`).
const EOB_OFFSET_BITS: [u32; 12] = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// `DCT_DCT`'s symbol in both intra transform sets (libaom `av1_ext_tx_ind`).
const DCT_DCT_SYMBOL: usize = 1;

/// Luma or chroma: selects the coefficient CDF set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneType {
    /// Y.
    Luma = 0,
    /// U and V.
    Chroma = 1,
}

/// Contexts derived from neighbouring transform blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxbContext {
    /// Context of the `all_zero` (`txb_skip`) symbol.
    pub skip: usize,
    /// Context of the DC sign symbol.
    pub dc_sign: usize,
}

impl TxbContext {
    /// Derive the contexts from the above and left entropy-context bytes
    /// covering the transform block (libaom `get_txb_ctx`). `tx_fills_block`
    /// is true when the transform covers the whole (plane) block.
    #[must_use]
    pub fn derive(plane: PlaneType, tx_fills_block: bool, above: &[u8], left: &[u8]) -> Self {
        Self {
            skip: Self::skip_context(plane, tx_fills_block, above, left),
            dc_sign: Self::dc_sign_context(above, left),
        }
    }

    /// The `all_zero` context alone (the decoder needs the DC sign context
    /// only for blocks with a nonzero DC coefficient).
    #[must_use]
    pub fn skip_context(
        plane: PlaneType,
        tx_fills_block: bool,
        above: &[u8],
        left: &[u8],
    ) -> usize {
        match plane {
            PlaneType::Luma if tx_fills_block => 0,
            PlaneType::Luma => {
                let top = above.iter().fold(0, |a, &v| a | v) & CUL_LEVEL_MASK;
                let left = left.iter().fold(0, |a, &v| a | v) & CUL_LEVEL_MASK;
                luma_skip_context(top.min(4), left.min(4))
            }
            PlaneType::Chroma => {
                let any = |s: &[u8]| usize::from(s.iter().any(|&v| v != 0));
                let offset = if tx_fills_block { 7 } else { 10 };
                any(above) + any(left) + offset
            }
        }
    }

    /// The DC sign context alone: the sign of the sum of the neighbours'
    /// DC sign categories (1 negative, 2 positive). A context byte's
    /// category is 0, 1 or 2, so each byte adds its bit 4 and subtracts its
    /// bit 3, without a branch.
    #[must_use]
    pub fn dc_sign_context(above: &[u8], left: &[u8]) -> usize {
        let sign_sum: i32 = above
            .iter()
            .chain(left)
            .map(|&v| {
                i32::from((v >> (DC_SIGN_SHIFT + 1)) & 1) - i32::from((v >> DC_SIGN_SHIFT) & 1)
            })
            .sum();
        match sign_sum.signum() {
            -1 => 1,
            1 => 2,
            _ => 0,
        }
    }
}

/// libaom `kTxbCtxSkipContexts[top][left]` (both already capped at 4).
fn luma_skip_context(top: u8, left: u8) -> usize {
    let (max, min) = (top.max(left), top.min(left));
    if max == 0 {
        1
    } else if min == 0 {
        2 + usize::from(max > 3)
    } else if max <= 3 {
        4
    } else if min <= 3 {
        5
    } else {
        6
    }
}

/// Scan order (coefficient index for each scan position). 64x64 transforms
/// code their top-left 32x32 with the 32x32 scan and contexts.
#[must_use]
pub fn coefficient_scan(tx: TxSize) -> &'static [u16] {
    scan(tx)
}

fn scan(tx: TxSize) -> &'static [u16] {
    match tx {
        TxSize::Tx4x4 => &DEFAULT_SCAN_4X4,
        TxSize::Tx8x8 => &DEFAULT_SCAN_8X8,
        TxSize::Tx16x16 => &DEFAULT_SCAN_16X16,
        TxSize::Tx32x32 | TxSize::Tx64x64 => &DEFAULT_SCAN_32X32,
    }
}

fn nz_map_ctx_offset(tx: TxSize) -> &'static [u8] {
    match tx {
        TxSize::Tx4x4 => &NZ_MAP_CTX_OFFSET_4X4,
        TxSize::Tx8x8 => &NZ_MAP_CTX_OFFSET_8X8,
        TxSize::Tx16x16 => &NZ_MAP_CTX_OFFSET_16X16,
        TxSize::Tx32x32 | TxSize::Tx64x64 => &NZ_MAP_CTX_OFFSET_32X32,
    }
}

/// End of block: one past the last non-zero level in scan order, over the
/// whole scan (the tests' reference; the encoder knows a shorter bound).
#[cfg(test)]
#[must_use]
pub fn end_of_block(tx: TxSize, levels: &[i32]) -> usize {
    end_of_block_within(tx, levels, scan(tx).len())
}

/// [`end_of_block`] of `levels` known to be zero from scan position
/// `scan_len` on.
#[must_use]
pub fn end_of_block_within(tx: TxSize, levels: &[i32], scan_len: usize) -> usize {
    scan(tx)[..scan_len]
        .iter()
        .rposition(|&pos| levels[usize::from(pos)] != 0)
        .map_or(0, |p| p + 1)
}

/// The entropy-context byte a transform block leaves for its neighbours
/// (libaom `av1_get_txb_entropy_context`): saturated level sum and DC sign.
#[must_use]
pub fn entropy_context(tx: TxSize, levels: &[i32], eob: usize) -> u8 {
    if eob == 0 {
        return 0;
    }
    let sum: u32 = scan(tx)[..eob]
        .iter()
        .map(|&p| levels[usize::from(p)].unsigned_abs())
        .sum();
    let cul = sum.min(u32::from(CUL_LEVEL_MASK)) as u8;
    let dc_category = match levels[0].signum() {
        -1 => 1,
        1 => 2,
        _ => 0,
    };
    cul | (dc_category << DC_SIGN_SHIFT)
}

/// Transform sizes with their own coefficient CDFs (`TX_SIZES`).
const TX_SIZES: usize = 5;
/// Contexts of `all_zero` (`TXB_SKIP_CONTEXTS`).
const TXB_SKIP_CONTEXTS: usize = 13;
/// Plane types: luma, chroma (`PLANE_TYPES`).
const PLANE_TYPES: usize = 2;
/// Contexts of `coeff_base` (`SIG_COEF_CONTEXTS`).
const SIG_COEF_CONTEXTS: usize = 42;
/// Contexts of `coeff_base_eob` (`SIG_COEF_CONTEXTS_EOB`).
const SIG_COEF_CONTEXTS_EOB: usize = 4;
/// Contexts of `coeff_br` (`LEVEL_CONTEXTS`).
const LEVEL_CONTEXTS: usize = 21;
/// Contexts of `eob_extra` (`EOB_COEF_CONTEXTS`).
const EOB_COEF_CONTEXTS: usize = 9;
/// Contexts of `dc_sign` (`DC_SIGN_CONTEXTS`).
const DC_SIGN_CONTEXTS: usize = 3;
/// `eob_pt_*` contexts (`eob_multi_ctx`; 0 for transform class 2D).
const EOB_MULTI_CONTEXTS: usize = 2;
/// Symbols of `coeff_base_eob`.
const BASE_EOB_SYMBOLS: usize = 3;
/// Symbols of `coeff_base`.
const BASE_SYMBOLS: usize = 4;
/// Square transform sizes with transform-type CDFs (`EXT_TX_SIZES`).
const EXT_TX_SIZES: usize = 4;
/// Transform-type CDF slot width (`TX_TYPES`).
const TX_TYPES: usize = 16;
/// Intra and inter transform-type sets (`EXT_TX_SETS_INTRA` / `_INTER`).
const EXT_TX_SETS_INTRA: usize = 3;
const EXT_TX_SETS_INTER: usize = 4;

/// The transform-type and coefficient CDFs of a frame. The transform-type
/// CDFs are non-coefficient CDFs in the spec (`init_non_coeff_cdfs`) but are
/// only coded here; the rest start from the default set chosen by
/// `base_q_idx` (`init_coeff_cdfs`).
#[derive(Debug, Clone)]
pub struct CoeffCdfs {
    intra_tx_type: [[[Cdf<TX_TYPES>; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA],
    inter_tx_type: [[Cdf<TX_TYPES>; EXT_TX_SIZES]; EXT_TX_SETS_INTER],
    txb_skip: [[Cdf<2>; TXB_SKIP_CONTEXTS]; TX_SIZES],
    eob_pt_16: [[Cdf<5>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_64: [[Cdf<7>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_256: [[Cdf<9>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_1024: [[Cdf<11>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_extra: [[[Cdf<2>; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES],
    base_eob: [[[Cdf<BASE_EOB_SYMBOLS>; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES]; TX_SIZES],
    base: [[[Cdf<BASE_SYMBOLS>; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES],
    br: [[[Cdf<{ BR_CDF_SIZE as usize }>; LEVEL_CONTEXTS]; PLANE_TYPES]; TX_SIZES],
    dc_sign: [[Cdf<2>; DC_SIGN_CONTEXTS]; PLANE_TYPES],
}

impl CoeffCdfs {
    /// The default CDFs for quantiser context `q_ctx` (0..4).
    #[must_use]
    pub fn new(q_ctx: usize) -> Self {
        Self {
            intra_tx_type: INTRA_EXT_TX_CDF.cdfs(),
            inter_tx_type: INTER_EXT_TX_CDF.cdfs(),
            txb_skip: TXB_SKIP_CDF[q_ctx].cdfs(),
            eob_pt_16: EOB_PT_16_CDF[q_ctx].cdfs(),
            eob_pt_64: EOB_PT_64_CDF[q_ctx].cdfs(),
            eob_pt_256: EOB_PT_256_CDF[q_ctx].cdfs(),
            eob_pt_1024: EOB_PT_1024_CDF[q_ctx].cdfs(),
            eob_extra: EOB_EXTRA_CDF[q_ctx].cdfs(),
            base_eob: COEFF_BASE_EOB_CDF[q_ctx].cdfs(),
            base: COEFF_BASE_CDF[q_ctx].cdfs(),
            br: COEFF_BR_CDF[q_ctx].cdfs(),
            dc_sign: DC_SIGN_CDF[q_ctx].cdfs(),
        }
    }

    /// Code one transform block's coefficients (AV1 §5.11.39), including the
    /// luma transform type, whose alphabet differs for intra and inter
    /// blocks. `levels` are signed, in libaom layout.
    pub fn write<W: SymbolSink>(
        &mut self,
        w: &mut W,
        tx: TxSize,
        plane: PlaneType,
        is_inter: bool,
        intra_mode: usize,
        ctx: TxbContext,
        levels: &[i32],
        eob: usize,
    ) {
        let txs = tx.index();
        let pt = plane as usize;
        w.write_bool(eob == 0, &mut self.txb_skip[txs][ctx.skip]);
        if eob == 0 {
            return;
        }
        if plane == PlaneType::Luma {
            if is_inter {
                self.write_inter_tx_type_dct(w, tx);
            } else {
                self.write_intra_tx_type_dct(w, tx, intra_mode);
            }
        }
        self.write_eob(w, tx, pt, eob);

        let scan = scan(tx);
        let n = tx.coded_size();
        let bhl = u32::from(tx.coded_log2());
        let nz_offsets = nz_map_ctx_offset(tx);
        let base_cdfs = &mut self.base[txs][pt];
        let range_cdfs = &mut self.br[txs.min(3)][pt];
        let mut map = LevelMap::new();
        // Coefficients in reverse scan order. The last one (`eob - 1`) is
        // non-zero by definition and codes `coeff_base_eob` instead of
        // `coeff_base`.
        let last = eob - 1;
        let pos = usize::from(scan[last]);
        let level = levels[pos].abs();
        w.write_symbol(
            (level.min(3) - 1) as usize,
            &mut self.base_eob[txs][pt][base_eob_context(last, n)],
            BASE_EOB_SYMBOLS,
        );
        map.code_level(w, range_cdfs, pos, LevelMap::index(pos, bhl), level);
        for c in (0..last).rev() {
            let pos = usize::from(scan[c]);
            let at = LevelMap::index(pos, bhl);
            let level = levels[pos].abs();
            let ctx = LevelMap::base_context(pos, map.base_sum(at), nz_offsets[pos]);
            w.write_symbol(level.min(3) as usize, &mut base_cdfs[ctx], BASE_SYMBOLS);
            map.code_level(w, range_cdfs, pos, at, level);
        }

        for (c, &p) in scan[..eob].iter().enumerate() {
            let v = levels[usize::from(p)];
            if v == 0 {
                continue;
            }
            if c == 0 {
                w.write_bool(v < 0, &mut self.dc_sign[pt][ctx.dc_sign]);
            } else {
                w.write_literal_bit(v < 0);
            }
            if v.abs() > COEFF_BASE_RANGE + NUM_BASE_LEVELS {
                write_golomb(w, (v.abs() - MAX_BASE_BR_RANGE) as u32);
            }
        }
    }

    /// Bits [`Self::write`] would spend on this transform block with the CDFs
    /// as they stand (nothing is coded or adapted). The same symbols as
    /// `write` priced in one reverse pass: each nonzero level's sign and
    /// Exp-Golomb remainder are counted where its magnitude is, not in a
    /// second forward pass (integer 1/512-bit costs, so the order of the sum
    /// does not matter). Rate estimation prices far more blocks than are
    /// coded (transform trials, intra RD, key-frame planning). Tested equal
    /// to `write` into a [`BitCounter`].
    #[allow(clippy::too_many_arguments)]
    pub fn cost(
        &mut self,
        tx: TxSize,
        plane: PlaneType,
        is_inter: bool,
        intra_mode: usize,
        ctx: TxbContext,
        levels: &[i32],
        eob: usize,
    ) -> f32 {
        let mut counter = BitCounter::default();
        let txs = tx.index();
        let pt = plane as usize;
        counter.add_q9(self.txb_skip[txs][ctx.skip].cost_q9(usize::from(eob == 0)));
        if eob == 0 {
            return counter.bits();
        }
        if plane == PlaneType::Luma {
            if is_inter {
                self.write_inter_tx_type_dct(&mut counter, tx);
            } else {
                self.write_intra_tx_type_dct(&mut counter, tx, intra_mode);
            }
        }
        self.write_eob(&mut counter, tx, pt, eob);

        let scan = scan(tx);
        let n = tx.coded_size();
        let bhl = u32::from(tx.coded_log2());
        let nz_offsets = nz_map_ctx_offset(tx);
        let base_cdfs = &mut self.base[txs][pt];
        let range_cdfs = &mut self.br[txs.min(3)][pt];
        let mut map = LevelMap::new();
        // A nonzero level's sign: `dc_sign` for the DC, else one bypass bit;
        // its remainder past the coded range in bypass bits (`write_golomb`).
        let mut sign_and_remainder = |counter: &mut BitCounter, c: usize, v: i32| {
            if c == 0 {
                counter.add_q9(self.dc_sign[pt][ctx.dc_sign].cost_q9(usize::from(v < 0)));
            } else {
                counter.add_bypass(1);
            }
            if v.abs() > COEFF_BASE_RANGE + NUM_BASE_LEVELS {
                counter.add_bypass(golomb_bits((v.abs() - MAX_BASE_BR_RANGE) as u32));
            }
        };
        let last = eob - 1;
        let pos = usize::from(scan[last]);
        let v = levels[pos];
        let level = v.abs();
        counter.add_q9(
            self.base_eob[txs][pt][base_eob_context(last, n)].cost_q9((level.min(3) - 1) as usize),
        );
        map.code_level(
            &mut counter,
            range_cdfs,
            pos,
            LevelMap::index(pos, bhl),
            level,
        );
        sign_and_remainder(&mut counter, last, v);
        for c in (0..last).rev() {
            let pos = usize::from(scan[c]);
            let at = LevelMap::index(pos, bhl);
            let v = levels[pos];
            let level = v.abs();
            let ctx = LevelMap::base_context(pos, map.base_sum(at), nz_offsets[pos]);
            counter.add_q9(base_cdfs[ctx].cost_q9(level.min(3) as usize));
            // A zero level leaves the map as it was (zero).
            if level != 0 {
                map.code_level(&mut counter, range_cdfs, pos, at, level);
                sign_and_remainder(&mut counter, c, v);
            }
        }
        counter.bits()
    }

    /// SVT-AV1's linear coefficient-rate estimate (`coeff_rate_est_lvl` 0,
    /// `product_coding_loop.c`): bits of coding `eob` coefficients rather
    /// than none, from the end of block alone plus the live skip-flag rate.
    /// No per-coefficient CDF walk; it ignores the levels' magnitudes.
    pub fn eob_linear_bits(&self, tx: TxSize, ctx: TxbContext, eob: usize) -> f32 {
        let n = tx.size();
        let slope = if eob < n * n / EOB_LINEAR_DENSE_DIVISOR {
            EOB_LINEAR_SPARSE_SLOPE
        } else {
            EOB_LINEAR_DENSE_SLOPE
        };
        (EOB_LINEAR_OFFSET + eob as f32 * slope) / EOB_LINEAR_UNITS_PER_BIT
            + self.skip_rate_delta(tx, ctx)
    }

    /// Current nonzero-vs-zero transform skip-flag rate, in literal bits.
    pub fn skip_rate_delta(&self, tx: TxSize, ctx: TxbContext) -> f32 {
        self.txb_skip[tx.index()][ctx.skip].bits(0) - self.txb_skip[tx.index()][ctx.skip].bits(1)
    }

    /// `eob_pt_*` and `eob_extra` (AV1 §5.11.39).
    fn write_eob<W: SymbolSink>(&mut self, w: &mut W, tx: TxSize, pt: usize, eob: usize) {
        let eob_pt = EOB_GROUP_START
            .iter()
            .rposition(|&s| s <= eob)
            .expect("eob >= 1");
        let extra = eob - EOB_GROUP_START[eob_pt];
        let symbol = eob_pt - 1;
        // `eob_multi_ctx` is 0 for transform class 2D.
        match tx {
            TxSize::Tx4x4 => w.write_symbol(symbol, &mut self.eob_pt_16[pt][0], 5),
            TxSize::Tx8x8 => w.write_symbol(symbol, &mut self.eob_pt_64[pt][0], 7),
            TxSize::Tx16x16 => w.write_symbol(symbol, &mut self.eob_pt_256[pt][0], 9),
            TxSize::Tx32x32 | TxSize::Tx64x64 => {
                w.write_symbol(symbol, &mut self.eob_pt_1024[pt][0], 11);
            }
        }
        let bits = EOB_OFFSET_BITS[eob_pt];
        if bits > 0 {
            let top = (extra >> (bits - 1)) & 1 != 0;
            w.write_bool(top, &mut self.eob_extra[tx.index()][pt][eob_pt - 3]);
            for i in (0..bits - 1).rev() {
                w.write_literal_bit((extra >> i) & 1 != 0);
            }
        }
    }

    /// `intra_tx_type` for `DCT_DCT` and the luma mode (AV1 §5.11.47; libaom
    /// `av1_write_tx_type`). Square sizes below 16x16 use the 7-type set
    /// (`ext_tx_set` 1), 16x16 the 5-type set (`ext_tx_set` 2); 32x32 and up
    /// allow `DCT_DCT` only, so nothing is coded.
    fn write_intra_tx_type_dct<W: SymbolSink>(&mut self, w: &mut W, tx: TxSize, mode: usize) {
        let (set, nsyms) = match tx {
            TxSize::Tx32x32 | TxSize::Tx64x64 => return,
            TxSize::Tx16x16 => (2, 5),
            TxSize::Tx4x4 | TxSize::Tx8x8 => (1, 7),
        };
        w.write_symbol(
            DCT_DCT_SYMBOL,
            &mut self.intra_tx_type[set][tx.index()][mode],
            nsyms,
        );
    }

    /// `inter_tx_type` for `DCT_DCT` (AV1 §5.11.47; libaom `av1_write_tx_type`).
    /// With `reduced_tx_set = 0`, 16x16 transforms use the 12-type set
    /// (`EXT_TX_SET_DTT9_IDTX_1DDCT`, inter set index 2), smaller ones all 16
    /// types (`EXT_TX_SET_ALL16`, inter set index 1), 32x32 the two-type set
    /// (`EXT_TX_SET_DCT_IDTX`, inter set index 3), and 64x64 `DCT_DCT` only (not
    /// coded); `DCT_DCT`'s symbol in each comes from libaom `av1_ext_tx_ind`.
    fn write_inter_tx_type_dct<W: SymbolSink>(&mut self, w: &mut W, tx: TxSize) {
        let (set, symbol, nsyms) = match tx {
            TxSize::Tx64x64 => return,
            TxSize::Tx32x32 => (3, 1, 2),
            TxSize::Tx16x16 => (2, 3, 12),
            TxSize::Tx4x4 | TxSize::Tx8x8 => (1, 7, 16),
        };
        w.write_symbol(symbol, &mut self.inter_tx_type[set][tx.index()], nsyms);
    }
}

impl ResetCounters for CoeffCdfs {
    fn reset_counters(&mut self) {
        self.intra_tx_type.reset_counters();
        self.inter_tx_type.reset_counters();
        self.txb_skip.reset_counters();
        self.eob_pt_16.reset_counters();
        self.eob_pt_64.reset_counters();
        self.eob_pt_256.reset_counters();
        self.eob_pt_1024.reset_counters();
        self.eob_extra.reset_counters();
        self.base_eob.reset_counters();
        self.base.reset_counters();
        self.br.reset_counters();
        self.dc_sign.reset_counters();
    }
}

/// Context of `coeff_base_eob` from the scan position (libaom
/// `get_lower_levels_ctx_eob`).
fn base_eob_context(scan_idx: usize, n: usize) -> usize {
    let area = n * n;
    if scan_idx == 0 {
        0
    } else if scan_idx <= area / 8 {
        1
    } else if scan_idx <= area / 4 {
        2
    } else {
        3
    }
}

/// How far (in rows or columns) the `coeff_base` / `coeff_br` contexts look
/// past a coefficient.
const CONTEXT_REACH: usize = 2;
/// Map stride: the largest coded transform (32x32) plus the reach, used for
/// every size so that neighbour offsets are constants.
const MAP_STRIDE: usize = 32 + CONTEXT_REACH;
/// Map entries: `MAP_STRIDE` columns of `MAP_STRIDE` rows.
const MAP_SIZE: usize = MAP_STRIDE * MAP_STRIDE;
/// Span from a coefficient to its farthest context neighbour, inclusive.
const NEIGHBOURHOOD: usize = CONTEXT_REACH * MAP_STRIDE + 1;
/// Offsets of the neighbours that both `coeff_base` and `coeff_br` read:
/// (row, col + 1), (row + 1, col), (row + 1, col + 1).
const SHARED_NEIGHBOURS: [usize; 3] = [MAP_STRIDE, 1, MAP_STRIDE + 1];
/// Offsets of the two further `coeff_base` neighbours: (row, col + 2),
/// (row + 2, col).
const FAR_BASE_NEIGHBOURS: [usize; 2] = [2 * MAP_STRIDE, 2];
/// Magnitude saturation in the `coeff_base` context sum.
const BASE_CONTEXT_LEVEL_MAX: u8 = 3;

/// Coefficient magnitudes, saturated at `MAX_BASE_BR_RANGE`, in a layout
/// padded with at least `CONTEXT_REACH` zero rows and columns so that context
/// neighbours past the block edge need no bounds tests (libaom's padded
/// `levels` buffer, `TX_PAD_HOR`). It is filled in reverse scan order as
/// coefficients are coded: every context neighbour comes later in scan order,
/// so it is already set, and coefficients from the EOB on stay zero.
struct LevelMap {
    magnitudes: [u8; MAP_SIZE],
}

impl LevelMap {
    /// An all-zero map.
    fn new() -> Self {
        Self {
            magnitudes: [0; MAP_SIZE],
        }
    }

    /// Map index of coefficient `pos` (libaom layout) of a block of coded
    /// side `1 << log2`: its column times the map stride plus its row.
    fn index(pos: usize, log2: u32) -> usize {
        pos + (pos >> log2) * (MAP_STRIDE - (1 << log2))
    }

    /// Code the `coeff_br` symbols of coefficient `pos` (map index `at`)
    /// when its magnitude `level` exceeds the base levels (only those read
    /// the range sum), then record the magnitude.
    #[allow(
        clippy::inline_always,
        reason = "called per coefficient from two sites; as a call it cost +0.8..+1.0 % av1-faster instructions"
    )]
    #[inline(always)]
    fn code_level<W: SymbolSink>(
        &mut self,
        w: &mut W,
        range_cdfs: &mut [Cdf<{ BR_CDF_SIZE as usize }>; LEVEL_CONTEXTS],
        pos: usize,
        at: usize,
        level: i32,
    ) {
        if level > NUM_BASE_LEVELS {
            let (row, col) = (at % MAP_STRIDE, at / MAP_STRIDE);
            let ctx = Self::br_context(pos, row, col, self.range_sum(at));
            write_range(w, &mut range_cdfs[ctx], level);
        }
        self.magnitudes[at] = level.unsigned_abs().min(MAX_BASE_BR_RANGE as u32) as u8;
    }

    /// The context neighbourhood of the coefficient at map index `at`.
    #[inline]
    fn near(&self, at: usize) -> &[u8; NEIGHBOURHOOD] {
        self.magnitudes[at..at + NEIGHBOURHOOD]
            .try_into()
            .expect("coefficient inside the coded block")
    }

    /// `coeff_base` magnitude sum of the context neighbours of the
    /// coefficient at map index `at`.
    #[inline]
    fn base_sum(&self, at: usize) -> u32 {
        let near = self.near(at);
        SHARED_NEIGHBOURS
            .iter()
            .chain(&FAR_BASE_NEIGHBOURS)
            .map(|&i| u32::from(near[i].min(BASE_CONTEXT_LEVEL_MAX)))
            .sum()
    }

    /// `coeff_br` magnitude sum of the context neighbours of the coefficient
    /// at map index `at`.
    #[inline]
    fn range_sum(&self, at: usize) -> u32 {
        let near = self.near(at);
        SHARED_NEIGHBOURS.iter().map(|&i| u32::from(near[i])).sum()
    }

    /// Context of `coeff_base` for transform class 2D (libaom
    /// `get_lower_levels_ctx_2d`) from its neighbour sum, plus the position's
    /// offset `nz_offset` from `nz_map_ctx_offset`.
    fn base_context(pos: usize, base_sum: u32, nz_offset: u8) -> usize {
        if pos == 0 {
            return 0;
        }
        (((base_sum + 1) >> 1).min(4) as usize) + usize::from(nz_offset)
    }

    /// Context of `coeff_br` for transform class 2D (libaom `get_br_ctx`)
    /// from its neighbour sum.
    fn br_context(pos: usize, row: usize, col: usize, br_sum: u32) -> usize {
        let mag = ((br_sum + 1) >> 1).min(6) as usize;
        if pos == 0 {
            mag
        } else if row < 2 && col < 2 {
            mag + 7
        } else {
            mag + 14
        }
    }
}

/// Exp-Golomb code of the remainder above `MAX_BASE_BR_RANGE - 1` in raw
/// bits (AV1 §5.11.39 `read_golomb`; libaom `write_golomb`).
/// The `coeff_br` symbols of a coefficient whose magnitude `level` exceeds
/// the base levels (AV1 §5.11.39), up to [`COEFF_BASE_RANGE`] more.
fn write_range<W: SymbolSink>(w: &mut W, cdf: &mut Cdf<{ BR_CDF_SIZE as usize }>, level: i32) {
    let base_range = level - 1 - NUM_BASE_LEVELS;
    let mut idx = 0;
    while idx < COEFF_BASE_RANGE {
        let k = (base_range - idx).min(BR_CDF_SIZE - 1);
        w.write_symbol(k as usize, cdf, BR_CDF_SIZE as usize);
        if k < BR_CDF_SIZE - 1 {
            break;
        }
        idx += BR_CDF_SIZE - 1;
    }
}

fn write_golomb<W: SymbolSink>(w: &mut W, value: u32) {
    let x = value + 1;
    let length = 32 - x.leading_zeros();
    for _ in 0..length - 1 {
        w.write_literal_bit(false);
    }
    w.write_literal(x, length);
}

/// Bypass bits [`write_golomb`] spends on `value`.
fn golomb_bits(value: u32) -> u32 {
    let length = 32 - (value + 1).leading_zeros();
    2 * length - 1
}

#[path = "coeff_trellis.rs"]
pub(crate) mod trellis;

#[cfg(test)]
mod tests {
    use super::*;

    /// `cost` prices exactly what `write` codes: random blocks of every size
    /// and plane, dense and sparse, levels past the Exp-Golomb threshold,
    /// both signs of the DC, after the CDFs adapted to earlier blocks.
    #[test]
    fn cost_equals_the_written_symbols() {
        let mut seed = 11_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        let mut cdfs = CoeffCdfs::new(2);
        let mut writer = crate::range_coder::SymbolWriter::new();
        for round in 0..200 {
            let tx = [
                TxSize::Tx4x4,
                TxSize::Tx8x8,
                TxSize::Tx16x16,
                TxSize::Tx32x32,
                TxSize::Tx64x64,
            ][round % 5];
            let plane = if round % 3 == 0 {
                PlaneType::Chroma
            } else {
                PlaneType::Luma
            };
            let area = tx.coded_size() * tx.coded_size();
            let reach = [1, 4, 40, area][(next() % 4) as usize].min(area);
            let mut levels = vec![0_i32; area];
            for &p in &scan(tx)[..reach] {
                let v = match next() % 10 {
                    0..=4 => 0,
                    5 | 6 => 1,
                    7 => 2,
                    8 => 3 + (next() % 12) as i32,
                    _ => 15 + (next() % 300) as i32,
                };
                levels[usize::from(p)] = if next() % 2 == 0 { v } else { -v };
            }
            let eob = end_of_block(tx, &levels);
            let ctx = TxbContext::derive(plane, true, &[(next() % 64) as u8], &[0x3]);
            let (is_inter, mode) = (round % 2 == 0, (next() % 13) as usize);
            let mut counter = BitCounter::default();
            cdfs.clone()
                .write(&mut counter, tx, plane, is_inter, mode, ctx, &levels, eob);
            let cost = cdfs.cost(tx, plane, is_inter, mode, ctx, &levels, eob);
            assert_eq!(
                cost.to_bits(),
                counter.bits().to_bits(),
                "{tx:?} round {round}"
            );
            cdfs.write(&mut writer, tx, plane, is_inter, mode, ctx, &levels, eob);
        }
    }

    #[test]
    fn eob_is_one_past_last_nonzero_in_scan_order() {
        let mut levels = vec![0i32; 16];
        assert_eq!(end_of_block(TxSize::Tx4x4, &levels), 0);
        levels[0] = 3;
        assert_eq!(end_of_block(TxSize::Tx4x4, &levels), 1);
        levels[4] = -1; // scan position 1 in libaom's 4x4 default scan
        assert_eq!(end_of_block(TxSize::Tx4x4, &levels), 2);
    }

    #[test]
    fn entropy_context_saturates_and_records_dc_sign() {
        let mut levels = vec![0i32; 16];
        levels[0] = -20;
        assert_eq!(entropy_context(TxSize::Tx4x4, &levels, 1), 7 | (1 << 3));
        levels[0] = 2;
        assert_eq!(entropy_context(TxSize::Tx4x4, &levels, 1), 2 | (2 << 3));
    }

    #[test]
    fn uncoded_coefficients_do_not_read_stale_levels() {
        use crate::range_coder::SymbolWriter;
        for tx in [
            TxSize::Tx4x4,
            TxSize::Tx8x8,
            TxSize::Tx16x16,
            TxSize::Tx32x32,
            TxSize::Tx64x64,
        ] {
            // Empty levels deliberately prove the EOB-zero contract used by
            // backend scratch reuse; poisoned old coefficients must also be inert.
            let poisoned = vec![i32::MIN; tx.coded_size() * tx.coded_size()];
            for plane in [PlaneType::Luma, PlaneType::Chroma] {
                for inter in [false, true] {
                    let ctx = TxbContext::derive(plane, true, &[0], &[0]);
                    let mut bytes = Vec::new();
                    let mut costs = Vec::new();
                    for levels in [&[][..], poisoned.as_slice()] {
                        let mut cdfs = CoeffCdfs::new(2);
                        costs.push(cdfs.cost(tx, plane, inter, 0, ctx, levels, 0));
                        let mut writer = SymbolWriter::new();
                        cdfs.write(&mut writer, tx, plane, inter, 0, ctx, levels, 0);
                        bytes.push(writer.finish());
                        assert_eq!(entropy_context(tx, levels, 0), 0);
                    }
                    assert_eq!(bytes[0], bytes[1]);
                    assert_eq!(costs[0], costs[1]);
                }
            }
        }
    }

    #[test]
    fn chroma_skip_context_counts_nonzero_neighbours() {
        let ctx = TxbContext::derive(PlaneType::Chroma, true, &[0, 3], &[0, 0]);
        assert_eq!(ctx.skip, 8);
        assert_eq!(ctx.dc_sign, 0);
    }
}
