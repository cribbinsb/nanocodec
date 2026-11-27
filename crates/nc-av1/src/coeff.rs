//! AV1 transform-coefficient coding (AV1 §5.11.39 `coeffs()`, contexts §8.3.2).
//!
//! Ported from libaom's `av1_write_coeffs_txb` and `txb_common.h`
//! (BSD-2-Clause, Alliance for Open Media), in libaom's coefficient layout:
//! index `c * n + r` for row `r`, column `c` (see `transform.rs`). Contexts
//! are computed from the complete set of levels; that is equivalent to the
//! decoder's partial view because every context only looks at coefficients
//! later in scan order, which the decoder has already decoded.
//!
//! The DC-only writer and rate shortcut follows SVT-AV1 4.3.0
//! `entropy_coding.c` / `rd_cost.c` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
//! AV2 uses different base contexts, parity and Rice remainder syntax.
//!
//! Only transform class 2D (`DCT_DCT`, and on intra blocks `ADST_DCT`,
//! `DCT_ADST`, `ADST_ADST` and `IDTX`) is supported, at the square sizes
//! and the 2:1 rectangles; 64-sample sides code their lowest 32
//! frequencies (64x64 as a 32x32 block).

use crate::cdf::{Cdf, CdfState, FromDefaults, ResetCounters};
use crate::range_coder::{MAX_LITERAL_RUN, SymbolSink};
use crate::tables::{
    COEFF_BASE_CDF, COEFF_BASE_EOB_CDF, COEFF_BR_CDF, DC_SIGN_CDF, DEFAULT_SCAN_4X4,
    DEFAULT_SCAN_4X8, DEFAULT_SCAN_8X4, DEFAULT_SCAN_8X8, DEFAULT_SCAN_8X16, DEFAULT_SCAN_16X8,
    DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X32, DEFAULT_SCAN_32X16, DEFAULT_SCAN_32X32, EOB_EXTRA_CDF,
    EOB_PT_16_CDF, EOB_PT_32_CDF, EOB_PT_64_CDF, EOB_PT_128_CDF, EOB_PT_256_CDF, EOB_PT_512_CDF,
    EOB_PT_1024_CDF, INTER_EXT_TX_CDF, INTRA_EXT_TX_CDF, INTRA_MODES, NZ_MAP_CTX_OFFSET_4X4,
    NZ_MAP_CTX_OFFSET_4X8, NZ_MAP_CTX_OFFSET_8X8, NZ_MAP_CTX_OFFSET_8X16, NZ_MAP_CTX_OFFSET_16X4,
    NZ_MAP_CTX_OFFSET_16X16, NZ_MAP_CTX_OFFSET_16X32, NZ_MAP_CTX_OFFSET_32X8,
    NZ_MAP_CTX_OFFSET_32X16, NZ_MAP_CTX_OFFSET_32X32, NZ_MAP_CTX_OFFSET_32X64,
    NZ_MAP_CTX_OFFSET_64X32, TXB_SKIP_CDF,
};
use crate::transform::TxSize;
use nc_common::eob::{GROUP_START as EOB_GROUP_START, group as eob_group};
use nc_dsp::Av1TxType;
use nc_encoder::journaled_fields;
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
/// Extra bits after each `eob_pt` (libaom `av1_eob_offset_bits`).
const EOB_OFFSET_BITS: [u32; 12] = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// The intra transform sets in symbol order (AV1 §6.10.19
/// `Tx_Type_Intra_Inv_Set1` / `Set2`; libaom `av1_ext_tx_inv`).
const INTRA_SET_1: [Av1TxType; 7] = [
    Av1TxType::Idtx,
    Av1TxType::DctDct,
    Av1TxType::VDct,
    Av1TxType::HDct,
    Av1TxType::AdstAdst,
    Av1TxType::AdstDct,
    Av1TxType::DctAdst,
];
const INTRA_SET_2: [Av1TxType; 5] = [
    Av1TxType::Idtx,
    Av1TxType::DctDct,
    Av1TxType::AdstAdst,
    Av1TxType::AdstDct,
    Av1TxType::DctAdst,
];

/// How a luma transform block signals its type (AV1 §5.11.47): the
/// alphabet differs for inter and intra blocks, and an intra block's CDF
/// depends on its prediction direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSyntax {
    /// An inter block (always `DCT_DCT` in this encoder).
    Inter,
    /// An intra block of luma direction `dir` (`intra_dir`, with filter
    /// intra mapped to its direction) coding transform type `kind`, one of
    /// transform class 2D: the coefficient scan and contexts below are
    /// those of `DCT_DCT`.
    Intra {
        /// `intra_dir` of the CDF.
        dir: usize,
        /// The transform type.
        kind: Av1TxType,
    },
}

impl TypeSyntax {
    /// `DCT_DCT` in an inter block, or in an intra block of direction `dir`.
    #[must_use]
    pub fn dct(is_inter: bool, dir: usize) -> Self {
        if is_inter {
            Self::Inter
        } else {
            Self::Intra {
                dir,
                kind: Av1TxType::DctDct,
            }
        }
    }
}

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

/// End-of-block tables of the default scans (`nc_dsp::eob_table`).
const EOB_4X4: [u16; 16] = nc_dsp::eob_table(&DEFAULT_SCAN_4X4);
const EOB_8X8: [u16; 64] = nc_dsp::eob_table(&DEFAULT_SCAN_8X8);
const EOB_16X16: [u16; 256] = nc_dsp::eob_table(&DEFAULT_SCAN_16X16);
const EOB_32X32: [u16; 1024] = nc_dsp::eob_table(&DEFAULT_SCAN_32X32);
const EOB_4X8: [u16; 32] = nc_dsp::eob_table(&DEFAULT_SCAN_4X8);
const EOB_8X4: [u16; 32] = nc_dsp::eob_table(&DEFAULT_SCAN_8X4);
const EOB_8X16: [u16; 128] = nc_dsp::eob_table(&DEFAULT_SCAN_8X16);
const EOB_16X8: [u16; 128] = nc_dsp::eob_table(&DEFAULT_SCAN_16X8);
const EOB_16X32: [u16; 512] = nc_dsp::eob_table(&DEFAULT_SCAN_16X32);
const EOB_32X16: [u16; 512] = nc_dsp::eob_table(&DEFAULT_SCAN_32X16);

/// The coded shape of `tx` (width and height log2, each capped at 32), for
/// the shapes this coder supports: the squares and the 2:1 rectangles.
fn coded_shape(tx: TxSize) -> (u8, u8) {
    let shape = (tx.coded_width_log2(), tx.coded_height_log2());
    debug_assert!(
        shape.0.abs_diff(shape.1) <= 1,
        "{tx:?}: 4:1 transforms are not coded"
    );
    shape
}

/// The end-of-block table of `tx`'s scan (`Dsp::end_of_block`): for each
/// coded position, its scan index plus one.
#[must_use]
pub fn eob_table(tx: TxSize) -> &'static [u16] {
    match coded_shape(tx) {
        (2, 2) => &EOB_4X4,
        (3, 3) => &EOB_8X8,
        (4, 4) => &EOB_16X16,
        (2, 3) => &EOB_4X8,
        (3, 2) => &EOB_8X4,
        (3, 4) => &EOB_8X16,
        (4, 3) => &EOB_16X8,
        (4, 5) => &EOB_16X32,
        (5, 4) => &EOB_32X16,
        _ => &EOB_32X32,
    }
}

/// libaom's 2-D (default) scan of `tx`'s coded shape.
fn scan(tx: TxSize) -> &'static [u16] {
    match coded_shape(tx) {
        (2, 2) => &DEFAULT_SCAN_4X4,
        (3, 3) => &DEFAULT_SCAN_8X8,
        (4, 4) => &DEFAULT_SCAN_16X16,
        (2, 3) => &DEFAULT_SCAN_4X8,
        (3, 2) => &DEFAULT_SCAN_8X4,
        (3, 4) => &DEFAULT_SCAN_8X16,
        (4, 3) => &DEFAULT_SCAN_16X8,
        (4, 5) => &DEFAULT_SCAN_16X32,
        (5, 4) => &DEFAULT_SCAN_32X16,
        _ => &DEFAULT_SCAN_32X32,
    }
}

/// Scan order (coefficient index for each scan position). 64x64 transforms
/// code their top-left 32x32 with the 32x32 scan and contexts.
#[must_use]
pub fn coefficient_scan(tx: TxSize) -> &'static [u16] {
    scan(tx)
}

/// End of block: one past the last non-zero level in scan order (the
/// tests' reference for `Dsp::end_of_block` over [`eob_table`]).
#[cfg(test)]
#[must_use]
pub fn end_of_block(tx: TxSize, levels: &[i16]) -> usize {
    scan(tx)
        .iter()
        .rposition(|&pos| levels[usize::from(pos)] != 0)
        .map_or(0, |p| p + 1)
}

/// The entropy-context byte a transform block leaves for its neighbours
/// (libaom `av1_get_txb_entropy_context`): saturated level sum and DC sign.
/// [`CoeffCdfs::write`] returns it; this is the tests' reference.
#[cfg(test)]
#[must_use]
pub fn entropy_context(tx: TxSize, levels: &[i16], eob: usize) -> u8 {
    if eob == 0 {
        return 0;
    }
    let sum: u32 = scan(tx)[..eob]
        .iter()
        .map(|&p| u32::from(levels[usize::from(p)].unsigned_abs()))
        .sum();
    context_byte(sum, levels[0])
}

/// The entropy-context byte of a block whose magnitudes add up to `sum`
/// and whose DC level is `dc`.
fn context_byte(sum: u32, dc: i16) -> u8 {
    let cul = sum.min(u32::from(CUL_LEVEL_MASK)) as u8;
    let dc_category = match dc.signum() {
        -1 => 1,
        1 => 2,
        _ => 0,
    };
    cul | (dc_category << DC_SIGN_SHIFT)
}

/// Run `$body` with `$map` a fresh level map for `tx`: the small squares'
/// own (a small block clears and touches only its own few bytes), the
/// largest for every other shape.
macro_rules! with_level_map {
    ($tx:expr, $map:ident => $body:expr) => {{
        let column = 1_usize << $tx.coded_height_log2();
        match ($tx.width_log2(), $tx.height_log2()) {
            (2, 2) => {
                let $map = &mut LevelMap4::new(column);
                $body
            }
            (3, 3) => {
                let $map = &mut LevelMap8::new(column);
                $body
            }
            (4, 4) => {
                let $map = &mut LevelMap16::new(column);
                $body
            }
            _ => {
                let $map = &mut FullLevelMap::new(column);
                $body
            }
        }
    }};
}

/// The magnitudes of one transform block, and what their contexts need:
/// [`Self::write`] codes their `coeff_base_eob`, `coeff_base` and
/// `coeff_br` symbols, [`Self::cost`] prices them.
struct Magnitudes<'a> {
    /// The scan's steps up to the end of block.
    steps: ScanSteps<'a>,
    levels: &'a [i16],
    /// Coded coefficients of the transform (`coeff_base_eob`'s context).
    area: usize,
}

/// The `coeff_base_eob`, `coeff_base` and `coeff_br` CDFs of a transform
/// size and plane type.
type MagnitudeCdfs<'a> = (
    &'a mut [Cdf<BASE_EOB_SYMBOLS>; SIG_COEF_CONTEXTS_EOB],
    &'a mut [Cdf<BASE_SYMBOLS>; SIG_COEF_CONTEXTS],
    &'a mut [Cdf<{ BR_CDF_SIZE as usize }>; LEVEL_CONTEXTS],
);

impl<'a> Magnitudes<'a> {
    /// The magnitudes of `tx`'s levels up to end of block `eob` (at least 1).
    fn of(tx: TxSize, eob: usize, levels: &'a [i16]) -> Self {
        Self {
            steps: scan_steps(tx, eob),
            levels,
            area: tx.coded_area(),
        }
    }

    /// The level of the coefficient at `pos` and its magnitude.
    fn level(&self, pos: usize) -> (i32, i32) {
        let v = i32::from(self.levels[pos]);
        (v, v.abs())
    }

    /// Code the magnitudes in reverse scan order, in a level map sized to
    /// the block (`STRIDE` its side plus the context reach), so that a small
    /// block clears and touches only its own few bytes. The last one
    /// (`eob - 1`) is non-zero by definition and codes `coeff_base_eob`
    /// instead of `coeff_base`; the DC's `coeff_base` context is 0.
    fn write<W: SymbolSink, const STRIDE: usize, const SIZE: usize, const PREPARED: bool>(
        &self,
        w: &mut W,
        (base_eob_cdfs, base_cdfs, range_cdfs): MagnitudeCdfs<'_>,
        map: &mut LevelMap<STRIDE, SIZE>,
        contexts: &[u8],
    ) {
        let last = self.steps.last();
        let (pos, at, _) = self.steps.step(last);
        let (_, level) = self.level(pos);
        w.write_symbol(
            (level.min(3) - 1) as usize,
            &mut base_eob_cdfs[base_eob_context(last, self.area)],
            BASE_EOB_SYMBOLS,
        );
        map.code_level::<W, PREPARED>(w, range_cdfs, self.levels, pos, at, level);
        for c in (1..last).rev() {
            let (pos, at, offset) = self.steps.step(c);
            let (_, level) = self.level(pos);
            let ctx = if PREPARED {
                usize::from(contexts[pos] + offset)
            } else {
                ac_base_context(map.base_sum(at), offset)
            };
            w.write_symbol(level.min(3) as usize, &mut base_cdfs[ctx], BASE_SYMBOLS);
            map.code_level::<W, PREPARED>(w, range_cdfs, self.levels, pos, at, level);
        }
        if last > 0 {
            let (_, level) = self.level(0);
            w.write_symbol(level.min(3) as usize, &mut base_cdfs[0], BASE_SYMBOLS);
            map.code_level::<W, PREPARED>(w, range_cdfs, self.levels, 0, 0, level);
        }
    }

    /// The bits [`Self::write`] and the sign pass spend on these
    /// magnitudes, priced in one reverse pass: each nonzero level's sign
    /// (`dc_sign` for the DC) and Exp-Golomb remainder are counted where its
    /// magnitude is. The `coeff_base` costs add up in a local (integer
    /// costs, so the order of the sum does not matter).
    fn cost<const STRIDE: usize, const SIZE: usize, const PREPARED: bool>(
        &self,
        counter: &mut BitCounter,
        (base_eob_cdfs, base_cdfs, range_cdfs): MagnitudeCdfs<'_>,
        dc_sign: &mut Cdf<2>,
        map: &mut LevelMap<STRIDE, SIZE>,
        contexts: &[u8],
    ) {
        // A nonzero AC level's sign is one bypass bit; any level's remainder
        // past the coded range is Exp-Golomb bypass bits (`write_golomb`).
        let remainder = |counter: &mut BitCounter, level: i32| {
            if level > COEFF_BASE_RANGE + NUM_BASE_LEVELS {
                counter.add_bypass(golomb_bits((level - MAX_BASE_BR_RANGE) as u32));
            }
        };
        let last = self.steps.last();
        let (pos, at, _) = self.steps.step(last);
        let (v, level) = self.level(pos);
        counter.add_q9(
            base_eob_cdfs[base_eob_context(last, self.area)].cost_q9((level.min(3) - 1) as usize),
        );
        map.code_level::<_, PREPARED>(counter, range_cdfs, self.levels, pos, at, level);
        if last == 0 {
            counter.add_q9(dc_sign.cost_q9(usize::from(v < 0)));
        } else {
            counter.add_bypass(1);
        }
        remainder(counter, level);
        let mut base_q9 = 0;
        for c in (1..last).rev() {
            let (pos, at, offset) = self.steps.step(c);
            let (_, level) = self.level(pos);
            let ctx = if PREPARED {
                usize::from(contexts[pos] + offset)
            } else {
                ac_base_context(map.base_sum(at), offset)
            };
            base_q9 += base_cdfs[ctx].cost_q9(level.min(3) as usize);
            // A zero level leaves the map as it was (zero).
            if level != 0 {
                map.code_level::<_, PREPARED>(counter, range_cdfs, self.levels, pos, at, level);
                counter.add_bypass(1);
                remainder(counter, level);
            }
        }
        if last > 0 {
            let (v, level) = self.level(0);
            base_q9 += base_cdfs[0].cost_q9(level.min(3) as usize);
            if level != 0 {
                map.code_level::<_, PREPARED>(counter, range_cdfs, self.levels, 0, 0, level);
                counter.add_q9(dc_sign.cost_q9(usize::from(v < 0)));
                remainder(counter, level);
            }
        }
        counter.add_q9(base_q9);
    }
}

/// The steps of a transform size's scan, by scan index: the coefficient
/// (libaom layout), its level-map index and its `coeff_base` context offset
/// (libaom `nz_map_ctx_offset`). The magnitude walk runs over these three
/// parallel arrays instead of deriving a map index and looking up the
/// offset for every coefficient (SVT-AV1 likewise prepares contexts before
/// its cost loop, `svt_av1_get_nz_map_contexts`). One array of packed steps
/// measured 0.06 % more AV1 Default instructions.
struct ScanWalk<const AREA: usize> {
    pos: [u16; AREA],
    at: [u16; AREA],
    nz_offset: [u8; AREA],
    trellis: [trellis::Position; AREA],
}

impl<const AREA: usize> ScanWalk<AREA> {
    /// The walk of a square transform with `scan` and `nz_map_ctx_offset`
    /// table `offsets`, for the level map sized to it ([`map_stride`]).
    const fn new(scan: &[u16; AREA], offsets: &[u8]) -> Self {
        let side = 1 << (AREA.ilog2() / 2);
        Self::with_map(scan, offsets, side, map_stride(side))
    }

    /// The walk of a transform of coded height `height` with `scan` and
    /// `nz_map_ctx_offset` table `offsets` (at least `AREA` entries), for a
    /// level map of stride `stride`.
    const fn with_map(scan: &[u16; AREA], offsets: &[u8], height: usize, stride: usize) -> Self {
        let log2 = height.ilog2();
        let mut walk = Self {
            pos: [0; AREA],
            at: [0; AREA],
            nz_offset: [0; AREA],
            trellis: [trellis::Position::DC; AREA],
        };
        let mut c = 0;
        while c < AREA {
            let pos = scan[c] as usize;
            walk.pos[c] = scan[c];
            // The column times the map stride plus the row.
            walk.at[c] = (pos + (pos >> log2) * (stride - height)) as u16;
            walk.nz_offset[c] = offsets[pos];
            let previous = if c + 1 < AREA {
                scan[c + 1] as usize
            } else {
                pos
            };
            walk.trellis[c] = trellis::Position::new(c, AREA, pos, previous, log2, offsets[pos]);
            c += 1;
        }
        walk
    }

    /// The first `eob` steps.
    fn steps(&self, eob: usize) -> ScanSteps<'_> {
        ScanSteps {
            pos: &self.pos[..eob],
            at: &self.at[..eob],
            nz_offset: &self.nz_offset[..eob],
        }
    }
}

const WALK_4X4: ScanWalk<16> = ScanWalk::new(&DEFAULT_SCAN_4X4, &NZ_MAP_CTX_OFFSET_4X4);
const WALK_8X8: ScanWalk<64> = ScanWalk::new(&DEFAULT_SCAN_8X8, &NZ_MAP_CTX_OFFSET_8X8);
const WALK_16X16: ScanWalk<256> = ScanWalk::new(&DEFAULT_SCAN_16X16, &NZ_MAP_CTX_OFFSET_16X16);
const WALK_32X32: ScanWalk<1024> = ScanWalk::new(&DEFAULT_SCAN_32X32, &NZ_MAP_CTX_OFFSET_32X32);
/// The rectangles' walks, for the largest map ([`FullLevelMap`]): they are
/// coded far less often than squares, so one map type serves them all.
/// Shorter rectangles use the matching prefix of the wider offset table.
const fn rect_walk<const AREA: usize>(
    scan: &[u16; AREA],
    offsets: &[u8],
    height: usize,
) -> ScanWalk<AREA> {
    ScanWalk::with_map(scan, offsets, height, MAP_STRIDE)
}
const WALK_4X8: ScanWalk<32> = rect_walk(&DEFAULT_SCAN_4X8, &NZ_MAP_CTX_OFFSET_4X8, 8);
const WALK_8X4: ScanWalk<32> = rect_walk(&DEFAULT_SCAN_8X4, &NZ_MAP_CTX_OFFSET_16X4, 4);
const WALK_8X16: ScanWalk<128> = rect_walk(&DEFAULT_SCAN_8X16, &NZ_MAP_CTX_OFFSET_8X16, 16);
const WALK_16X8: ScanWalk<128> = rect_walk(&DEFAULT_SCAN_16X8, &NZ_MAP_CTX_OFFSET_32X8, 8);
const WALK_16X32: ScanWalk<512> = rect_walk(&DEFAULT_SCAN_16X32, &NZ_MAP_CTX_OFFSET_16X32, 32);
const WALK_32X16: ScanWalk<512> = rect_walk(&DEFAULT_SCAN_32X16, &NZ_MAP_CTX_OFFSET_32X16, 16);
const WALK_32X64: ScanWalk<1024> = rect_walk(&DEFAULT_SCAN_32X32, &NZ_MAP_CTX_OFFSET_32X64, 32);
const WALK_64X32: ScanWalk<1024> = rect_walk(&DEFAULT_SCAN_32X32, &NZ_MAP_CTX_OFFSET_64X32, 32);

/// The first `eob` (at least 1) steps of `tx`'s scan walk.
fn scan_steps(tx: TxSize, eob: usize) -> ScanSteps<'static> {
    match (tx.width_log2(), tx.height_log2()) {
        (2, 2) => WALK_4X4.steps(eob),
        (3, 3) => WALK_8X8.steps(eob),
        (4, 4) => WALK_16X16.steps(eob),
        (5, 5) | (6, 6) => WALK_32X32.steps(eob),
        (2, 3) => WALK_4X8.steps(eob),
        (3, 2) => WALK_8X4.steps(eob),
        (3, 4) => WALK_8X16.steps(eob),
        (4, 3) => WALK_16X8.steps(eob),
        (4, 5) => WALK_16X32.steps(eob),
        (5, 4) => WALK_32X16.steps(eob),
        (5, 6) => WALK_32X64.steps(eob),
        (6, 5) => WALK_64X32.steps(eob),
        _ => unreachable!("{tx:?}: 4:1 transforms are not coded"),
    }
}

/// The scan positions with their prepared trellis context coordinates.
fn trellis_positions(tx: TxSize) -> &'static [trellis::Position] {
    match (tx.width_log2(), tx.height_log2()) {
        (2, 2) => &WALK_4X4.trellis,
        (3, 3) => &WALK_8X8.trellis,
        (4, 4) => &WALK_16X16.trellis,
        (5, 5) | (6, 6) => &WALK_32X32.trellis,
        (2, 3) => &WALK_4X8.trellis,
        (3, 2) => &WALK_8X4.trellis,
        (3, 4) => &WALK_8X16.trellis,
        (4, 3) => &WALK_16X8.trellis,
        (4, 5) => &WALK_16X32.trellis,
        (5, 4) => &WALK_32X16.trellis,
        (5, 6) => &WALK_32X64.trellis,
        (6, 5) => &WALK_64X32.trellis,
        _ => unreachable!("{tx:?}: 4:1 transforms are not coded"),
    }
}

/// The steps of a scan up to an end of block ([`ScanWalk::steps`]).
struct ScanSteps<'a> {
    pos: &'a [u16],
    at: &'a [u16],
    nz_offset: &'a [u8],
}

impl ScanSteps<'_> {
    /// Scan index of the last coefficient (`eob - 1`).
    fn last(&self) -> usize {
        self.pos.len() - 1
    }

    /// Coefficient, map index and context offset of scan index `c`.
    #[inline]
    fn step(&self, c: usize) -> (usize, usize, u8) {
        (
            usize::from(self.pos[c]),
            usize::from(self.at[c]),
            self.nz_offset[c],
        )
    }
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
/// log2 of the sides that decide the transform sets (AV1 §5.11.48
/// `get_tx_set`): 16 (`Tx_Size_Sqr`) and 32 (`Tx_Size_Sqr_Up`).
const SIDE_16_LOG2: u8 = 4;
const SIDE_32_LOG2: u8 = 5;
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
    eob_pt_32: [[Cdf<6>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_64: [[Cdf<7>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_128: [[Cdf<8>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_256: [[Cdf<9>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
    eob_pt_512: [[Cdf<10>; EOB_MULTI_CONTEXTS]; PLANE_TYPES],
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
            eob_pt_32: EOB_PT_32_CDF[q_ctx].cdfs(),
            eob_pt_64: EOB_PT_64_CDF[q_ctx].cdfs(),
            eob_pt_128: EOB_PT_128_CDF[q_ctx].cdfs(),
            eob_pt_256: EOB_PT_256_CDF[q_ctx].cdfs(),
            eob_pt_512: EOB_PT_512_CDF[q_ctx].cdfs(),
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
    /// blocks. `levels` are signed, in libaom layout. Returns the entropy
    /// context the block leaves for its neighbours (libaom
    /// `av1_get_txb_entropy_context`).
    pub fn write<W: SymbolSink>(
        &mut self,
        w: &mut W,
        tx: TxSize,
        plane: PlaneType,
        syntax: TypeSyntax,
        ctx: TxbContext,
        levels: &[i16],
        eob: usize,
    ) -> u8 {
        let txs = tx.context();
        let pt = plane as usize;
        w.write_bool(eob == 0, &mut self.txb_skip[txs][ctx.skip]);
        if eob == 0 {
            return 0;
        }
        if plane == PlaneType::Luma {
            self.write_tx_type(w, tx, syntax);
        }
        self.write_eob(w, tx, pt, eob);

        if eob == 1 {
            return self.write_dc(w, tx, pt, ctx.dc_sign, levels[0]);
        }
        let scan = &scan(tx)[..eob];
        let magnitudes = Magnitudes::of(tx, eob, levels);
        let cdfs = (
            &mut self.base_eob[txs][pt],
            &mut self.base[txs][pt],
            &mut self.br[txs.min(3)][pt],
        );
        with_level_map!(tx, map => {
            if levels.len() >= tx.coded_area() && dense_square(tx, eob) {
                let base = map.prepare_dense(tx, levels, eob);
                magnitudes.write::<_, _, _, true>(w, cdfs, map, &base);
            } else {
                magnitudes.write::<_, _, _, false>(w, cdfs, map, &[]);
            }
        });

        // Pack the equiprobable AC signs as a run. DC sign still adapts,
        // and each Golomb remainder follows its own sign in scan order.
        let dc = i32::from(levels[0]);
        let mut sum = dc.unsigned_abs();
        if dc != 0 {
            w.write_bool(dc < 0, &mut self.dc_sign[pt][ctx.dc_sign]);
            if sum >= MAX_BASE_BR_RANGE as u32 {
                write_golomb(w, sum - MAX_BASE_BR_RANGE as u32);
            }
        }
        let mut signs = 0;
        let mut sign_count = 0;
        for &p in &scan[1..] {
            let v = i32::from(levels[usize::from(p)]);
            if v == 0 {
                continue;
            }
            let mag = v.unsigned_abs();
            sum += mag;
            signs = (signs << 1) | u32::from(v < 0);
            sign_count += 1;
            if mag >= MAX_BASE_BR_RANGE as u32 {
                w.write_literal(signs, sign_count);
                sign_count = 0;
                write_golomb(w, mag - MAX_BASE_BR_RANGE as u32);
            } else if sign_count == MAX_LITERAL_RUN {
                w.write_literal(signs, sign_count);
                sign_count = 0;
            }
        }
        w.write_literal(signs, sign_count);
        context_byte(sum, levels[0])
    }

    /// Bits [`Self::write`] would spend on this transform block with the CDFs
    /// as they stand (nothing is coded or adapted). The same symbols as
    /// `write` priced in one reverse pass: each nonzero level's sign and
    /// Exp-Golomb remainder are counted where its magnitude is, not in a
    /// second forward pass (integer 1/512-bit costs, so the order of the sum
    /// does not matter). Rate estimation prices far more blocks than are
    /// coded (transform trials, intra RD, key-frame planning). Tested equal
    /// to `write` into a [`BitCounter`].
    pub fn cost(
        &mut self,
        tx: TxSize,
        plane: PlaneType,
        syntax: TypeSyntax,
        ctx: TxbContext,
        levels: &[i16],
        eob: usize,
    ) -> f32 {
        let mut counter = BitCounter::default();
        let txs = tx.context();
        let pt = plane as usize;
        counter.add_q9(self.txb_skip[txs][ctx.skip].cost_q9(usize::from(eob == 0)));
        if eob == 0 {
            return counter.bits();
        }
        if plane == PlaneType::Luma {
            self.write_tx_type(&mut counter, tx, syntax);
        }
        self.write_eob(&mut counter, tx, pt, eob);

        if eob == 1 {
            self.write_dc(&mut counter, tx, pt, ctx.dc_sign, levels[0]);
            return counter.bits();
        }

        let magnitudes = Magnitudes::of(tx, eob, levels);
        let cdfs = (
            &mut self.base_eob[txs][pt],
            &mut self.base[txs][pt],
            &mut self.br[txs.min(3)][pt],
        );
        let dc_sign = &mut self.dc_sign[pt][ctx.dc_sign];
        with_level_map!(tx, map => {
            if levels.len() >= tx.coded_area() && dense_square(tx, eob) {
                let base = map.prepare_dense(tx, levels, eob);
                magnitudes.cost::<_, _, true>(&mut counter, cdfs, dc_sign, map, &base);
            } else {
                magnitudes.cost::<_, _, false>(&mut counter, cdfs, dc_sign, map, &[]);
            }
        });
        counter.bits()
    }

    /// SVT-AV1's DC-only writer/rate path: `av1_write_coeffs_txb_1d` and
    /// `av1_cost_coeffs_txb_loop_cost_one_eob`. With no AC coefficient both
    /// magnitude contexts are zero; neither a scan nor a level map is needed.
    #[inline]
    fn write_dc<W: SymbolSink>(
        &mut self,
        w: &mut W,
        tx: TxSize,
        pt: usize,
        dc_sign_ctx: usize,
        value: i16,
    ) -> u8 {
        let level = i32::from(value).abs();
        w.write_symbol(
            (level.min(3) - 1) as usize,
            &mut self.base_eob[tx.context()][pt][0],
            BASE_EOB_SYMBOLS,
        );
        if level > NUM_BASE_LEVELS {
            write_range(w, &mut self.br[tx.context().min(3)][pt][0], level);
        }
        w.write_bool(value < 0, &mut self.dc_sign[pt][dc_sign_ctx]);
        if level > COEFF_BASE_RANGE + NUM_BASE_LEVELS {
            write_golomb(w, (level - MAX_BASE_BR_RANGE) as u32);
        }
        context_byte(level as u32, value)
    }

    /// SVT-AV1's linear coefficient-rate estimate (`coeff_rate_est_lvl` 0,
    /// `product_coding_loop.c`): bits of coding `eob` coefficients rather
    /// than none, from the end of block alone plus the live skip-flag rate.
    /// No per-coefficient CDF walk; it ignores the levels' magnitudes.
    pub fn eob_linear_bits(&self, tx: TxSize, ctx: TxbContext, eob: usize) -> f32 {
        let slope = if eob < tx.width() * tx.height() / EOB_LINEAR_DENSE_DIVISOR {
            EOB_LINEAR_SPARSE_SLOPE
        } else {
            EOB_LINEAR_DENSE_SLOPE
        };
        (EOB_LINEAR_OFFSET + eob as f32 * slope) / EOB_LINEAR_UNITS_PER_BIT
            + self.skip_rate_delta(tx, ctx)
    }

    /// [`Self::eob_linear_bits`] as a whole block's bits: the all-zero
    /// skip flag's rate, plus the linear estimate where `eob` is not 0.
    pub fn linear_bits(&self, tx: TxSize, ctx: TxbContext, eob: usize) -> f32 {
        let none = self.txb_skip[tx.context()][ctx.skip].bits(1);
        if eob == 0 {
            none
        } else {
            none + self.eob_linear_bits(tx, ctx, eob)
        }
    }

    /// Exact Q9 cost of the skip flag for an empty transform block.
    pub(crate) fn uncoded_bits(&mut self, tx: TxSize, ctx: TxbContext) -> f32 {
        let q9 = self.txb_skip[tx.context()][ctx.skip].cost_q9(1);
        q9 as f32 / nc_encoder::symbol_cost::COST_UNITS_PER_BIT as f32
    }

    /// Current nonzero-vs-zero transform skip-flag rate, in literal bits.
    pub fn skip_rate_delta(&self, tx: TxSize, ctx: TxbContext) -> f32 {
        self.txb_skip[tx.context()][ctx.skip].bits(0)
            - self.txb_skip[tx.context()][ctx.skip].bits(1)
    }

    /// Cost in 1/512 bit of `eob_pt` symbol `symbol` of a `tx` transform in
    /// plane type `pt` (the trellis's end-of-block rate).
    pub(crate) fn eob_pt_cost_q9(&mut self, tx: TxSize, pt: usize, symbol: usize) -> u32 {
        match tx.eob_alphabet() {
            0 => self.eob_pt_16[pt][0].cost_q9(symbol),
            1 => self.eob_pt_32[pt][0].cost_q9(symbol),
            2 => self.eob_pt_64[pt][0].cost_q9(symbol),
            3 => self.eob_pt_128[pt][0].cost_q9(symbol),
            4 => self.eob_pt_256[pt][0].cost_q9(symbol),
            5 => self.eob_pt_512[pt][0].cost_q9(symbol),
            _ => self.eob_pt_1024[pt][0].cost_q9(symbol),
        }
    }

    /// `eob_pt_*` and `eob_extra` (AV1 §5.11.39).
    fn write_eob<W: SymbolSink>(&mut self, w: &mut W, tx: TxSize, pt: usize, eob: usize) {
        let eob_pt = eob_group(eob);
        let extra = eob - usize::from(EOB_GROUP_START[eob_pt]);
        let symbol = eob_pt - 1;
        // `eob_multi_ctx` is 0 for transform class 2D.
        // The alphabet grows with the coded area (`eobMultisize`).
        match tx.eob_alphabet() {
            0 => w.write_symbol(symbol, &mut self.eob_pt_16[pt][0], 5),
            1 => w.write_symbol(symbol, &mut self.eob_pt_32[pt][0], 6),
            2 => w.write_symbol(symbol, &mut self.eob_pt_64[pt][0], 7),
            3 => w.write_symbol(symbol, &mut self.eob_pt_128[pt][0], 8),
            4 => w.write_symbol(symbol, &mut self.eob_pt_256[pt][0], 9),
            5 => w.write_symbol(symbol, &mut self.eob_pt_512[pt][0], 10),
            _ => w.write_symbol(symbol, &mut self.eob_pt_1024[pt][0], 11),
        }
        let bits = EOB_OFFSET_BITS[eob_pt];
        if bits > 0 {
            let top = (extra >> (bits - 1)) & 1 != 0;
            w.write_bool(top, &mut self.eob_extra[tx.context()][pt][eob_pt - 3]);
            for i in (0..bits - 1).rev() {
                w.write_literal_bit((extra >> i) & 1 != 0);
            }
        }
    }

    /// The luma transform type (AV1 §5.11.47 `transform_type`; libaom
    /// `av1_write_tx_type`), whose set depends on the size and on whether
    /// the block is inter (`get_tx_set`, `reduced_tx_set = 0`).
    pub(crate) fn write_tx_type<W: SymbolSink>(
        &mut self,
        w: &mut W,
        tx: TxSize,
        syntax: TypeSyntax,
    ) {
        match syntax {
            TypeSyntax::Inter => self.write_inter_tx_type_dct(w, tx),
            TypeSyntax::Intra { dir, kind } => self.write_intra_tx_type(w, tx, dir, kind),
        }
    }

    /// `intra_tx_type` of `kind` for luma direction `dir`. Square sizes
    /// below 16x16 use the 7-type set (`TX_SET_INTRA_1`, CDF set 1), 16x16
    /// the 5-type set (`TX_SET_INTRA_2`, CDF set 2); 32x32 and up allow
    /// `DCT_DCT` only, so nothing is coded. The symbol is the type's index
    /// in its set (§6.10.19 `Tx_Type_Intra_Inv_Set1/2`).
    fn write_intra_tx_type<W: SymbolSink>(
        &mut self,
        w: &mut W,
        tx: TxSize,
        dir: usize,
        kind: Av1TxType,
    ) {
        let (set, inverse): (usize, &[Av1TxType]) = if tx.square_up_log2() >= SIDE_32_LOG2 {
            debug_assert_eq!(
                kind,
                Av1TxType::DctDct,
                "sides of 32 and up are DCT_DCT only"
            );
            return;
        } else if tx.square_log2() == SIDE_16_LOG2 {
            (2, &INTRA_SET_2)
        } else {
            (1, &INTRA_SET_1)
        };
        let symbol = inverse
            .iter()
            .position(|&t| t == kind)
            .expect("intra transform type outside its set");
        w.write_symbol(
            symbol,
            &mut self.intra_tx_type[set][tx.type_cdf_size()][dir],
            inverse.len(),
        );
    }

    /// `inter_tx_type` for `DCT_DCT` (AV1 §5.11.47; libaom `av1_write_tx_type`).
    /// With `reduced_tx_set = 0`, 16x16 transforms use the 12-type set
    /// (`EXT_TX_SET_DTT9_IDTX_1DDCT`, inter set index 2), smaller ones all 16
    /// types (`EXT_TX_SET_ALL16`, inter set index 1), 32x32 the two-type set
    /// (`EXT_TX_SET_DCT_IDTX`, inter set index 3), and 64x64 `DCT_DCT` only (not
    /// coded); `DCT_DCT`'s symbol in each comes from libaom `av1_ext_tx_ind`.
    fn write_inter_tx_type_dct<W: SymbolSink>(&mut self, w: &mut W, tx: TxSize) {
        let (set, symbol, nsyms) = if tx.square_up_log2() > SIDE_32_LOG2 {
            return;
        } else if tx.square_up_log2() == SIDE_32_LOG2 {
            (3, 1, 2)
        } else if tx.square_log2() == SIDE_16_LOG2 {
            (2, 3, 12)
        } else {
            (1, 7, 16)
        };
        w.write_symbol(
            symbol,
            &mut self.inter_tx_type[set][tx.type_cdf_size()],
            nsyms,
        );
    }
}

// The level tables first: most coefficient symbols are theirs.
journaled_fields!(CoeffCdfs, CdfState;
    base, br, txb_skip, base_eob, eob_extra, dc_sign, eob_pt_16, eob_pt_32, eob_pt_64,
    eob_pt_128, eob_pt_256, eob_pt_512, eob_pt_1024, inter_tx_type, intra_tx_type,
);

impl ResetCounters for CoeffCdfs {
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

/// Context of `coeff_base_eob` from the scan position (libaom
/// `get_lower_levels_ctx_eob`).
const fn base_eob_context(scan_idx: usize, area: usize) -> usize {
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
/// Stride of the map for the largest coded transform (32x32): its side
/// plus the reach.
const MAP_STRIDE: usize = map_stride(32);
/// Entries of that map: `MAP_STRIDE` columns of `MAP_STRIDE` rows.
const MAP_SIZE: usize = MAP_STRIDE * MAP_STRIDE;
/// Magnitude saturation in the `coeff_base` context sum.
const BASE_CONTEXT_LEVEL_MAX: u8 = 3;

fn dense_square(tx: TxSize, eob: usize) -> bool {
    let side = tx.width_log2();
    cfg!(target_arch = "x86_64")
        && side == tx.height_log2()
        && side <= 4
        && eob >= if side == 2 { 12 } else { tx.coded_area() / 2 }
}

/// The map for the largest coded transform.
type FullLevelMap = LevelMap<MAP_STRIDE, MAP_SIZE>;
/// Stride of the map for coded side `side`.
const fn map_stride(side: usize) -> usize {
    side + CONTEXT_REACH
}
/// The maps for 4x4, 8x8 and 16x16 transforms.
type LevelMap4 = LevelMap<{ map_stride(4) }, { map_stride(4) * map_stride(4) }>;
type LevelMap8 = LevelMap<{ map_stride(8) }, { map_stride(8) * map_stride(8) }>;
type LevelMap16 = LevelMap<{ map_stride(16) }, { map_stride(16) * map_stride(16) }>;

/// Coefficient magnitudes saturated at `BASE_CONTEXT_LEVEL_MAX` (what the
/// `coeff_base` context sums), in a layout padded with at least
/// `CONTEXT_REACH` zero rows and columns so that context neighbours past the
/// block edge need no bounds tests (libaom's padded `levels` buffer,
/// `TX_PAD_HOR`). It is filled in reverse scan order as coefficients are
/// coded: every context neighbour comes later in scan order, so it is
/// already set, and coefficients from the EOB on stay zero.
///
/// The `coeff_br` context sums magnitudes up to `MAX_BASE_BR_RANGE`; a
/// neighbour whose saturated entry is below the cap is that magnitude, and
/// one at the cap is a coded coefficient of the block, read back from the
/// levels. So the base sum takes five plain bytes (no per-neighbour
/// saturation, which libaom's `get_nz_mag` does with `clip_max3`), and only
/// the rare coefficients above the base levels look further.
///
/// `STRIDE` is at least the coded height of the blocks it serves plus the
/// reach, and `SIZE` its square, so that neighbour offsets are constants:
/// the small squares have maps of their own, everything else the largest
/// ([`with_level_map`]).
struct LevelMap<const STRIDE: usize, const SIZE: usize> {
    magnitudes: [u8; SIZE],
    /// The coded height: one column's step in the block's levels.
    column: usize,
}

impl<const STRIDE: usize, const SIZE: usize> LevelMap<STRIDE, SIZE> {
    /// Span from a coefficient to its farthest context neighbour, inclusive.
    const NEIGHBOURHOOD: usize = CONTEXT_REACH * STRIDE + 1;
    /// Offsets of the neighbours that both `coeff_base` and `coeff_br` read:
    /// (row, col + 1), (row + 1, col), (row + 1, col + 1).
    const SHARED_NEIGHBOURS: [usize; 3] = [STRIDE, 1, STRIDE + 1];
    /// Offsets of the two further `coeff_base` neighbours: (row, col + 2),
    /// (row + 2, col).
    const FAR_BASE_NEIGHBOURS: [usize; 2] = [2 * STRIDE, 2];

    /// An all-zero map for a transform of coded height `column`.
    fn new(column: usize) -> Self {
        debug_assert!(column + CONTEXT_REACH <= STRIDE);
        Self {
            magnitudes: [0; SIZE],
            column,
        }
    }

    /// Code the `coeff_br` symbols of coefficient `pos` (map index `at`)
    /// when its magnitude `level` exceeds the base levels (only those read
    /// the range sum), then record the magnitude. `levels` are the block's
    /// signed levels, in libaom layout.
    #[allow(
        clippy::inline_always,
        reason = "called per coefficient from two sites; as a call it cost +0.8..+1.0 % AV1 Faster instructions"
    )]
    #[inline(always)]
    fn code_level<W: SymbolSink, const PREPARED: bool>(
        &mut self,
        w: &mut W,
        range_cdfs: &mut [Cdf<{ BR_CDF_SIZE as usize }>; LEVEL_CONTEXTS],
        levels: &[i16],
        pos: usize,
        at: usize,
        level: i32,
    ) {
        if level > NUM_BASE_LEVELS {
            let (row, col) = (at % STRIDE, at / STRIDE);
            let ctx = br_context(pos, row, col, self.range_sum(at, levels, pos));
            write_range(w, &mut range_cdfs[ctx], level);
        }
        if !PREPARED {
            self.magnitudes[at] = level.min(i32::from(BASE_CONTEXT_LEVEL_MAX)) as u8;
        }
    }

    /// Prepare the dense padded level map and raster base contexts before
    /// scanning, as in libaom's `av1_txb_init_levels` and
    /// `av1_get_nz_map_contexts`. The EOB table masks the uncoded tail.
    fn prepare_dense(&mut self, tx: TxSize, levels: &[i16], eob: usize) -> [u8; SIZE] {
        let side = STRIDE - CONTEXT_REACH;
        debug_assert_eq!(self.column, side);
        let area = side * side;
        let mut contexts = [0; SIZE];
        let eob = eob as u16;
        let levels = levels[..area].chunks_exact(side);
        let order = eob_table(tx)[..area].chunks_exact(side);
        for ((column, scan), padded) in levels
            .zip(order)
            .zip(self.magnitudes.as_chunks_mut::<STRIDE>().0)
        {
            for ((out, &level), &index) in padded[..side].iter_mut().zip(column).zip(scan) {
                let magnitude = level.unsigned_abs().min(u16::from(BASE_CONTEXT_LEVEL_MAX)) as u8;
                *out = if index <= eob { magnitude } else { 0 };
            }
        }

        let columns = self.magnitudes.as_chunks::<STRIDE>().0;
        for (x, base) in contexts[..area].chunks_exact_mut(side).enumerate() {
            let [column, next, far] = [&columns[x], &columns[x + 1], &columns[x + 2]];
            for (y, ctx) in base.iter_mut().enumerate() {
                let count = column[y + 1] + column[y + 2] + next[y] + next[y + 1] + far[y];
                *ctx = ((count + 1) >> 1).min(4);
            }
        }
        contexts
    }

    /// The context neighbourhood of the coefficient at map index `at`.
    #[inline]
    fn near(&self, at: usize) -> &[u8] {
        &self.magnitudes[at..at + Self::NEIGHBOURHOOD]
    }

    /// `coeff_base` magnitude sum of the context neighbours of the
    /// coefficient at map index `at`.
    #[allow(
        clippy::inline_always,
        reason = "Keep the neighbour loads in the coefficient loop"
    )]
    #[inline(always)]
    fn base_sum(&self, at: usize) -> u32 {
        let near = self.near(at);
        let [shared0, shared1, shared2] = Self::SHARED_NEIGHBOURS;
        let [far0, far1] = Self::FAR_BASE_NEIGHBOURS;
        u32::from(near[shared0])
            + u32::from(near[shared1])
            + u32::from(near[shared2])
            + u32::from(near[far0])
            + u32::from(near[far1])
    }

    /// `coeff_br` magnitude sum (each saturated at `MAX_BASE_BR_RANGE`) of
    /// the context neighbours of coefficient `pos` at map index `at`. The
    /// map's neighbour offsets are the same steps in libaom layout: one
    /// row (+1) and one column (+`STRIDE` in the map, + the coded side in
    /// `levels`).
    #[allow(
        clippy::inline_always,
        reason = "Keep the neighbour loads in the coefficient loop"
    )]
    #[inline(always)]
    fn range_sum(&self, at: usize, levels: &[i16], pos: usize) -> u32 {
        let near = self.near(at);
        let [a, b, c] = Self::SHARED_NEIGHBOURS;
        let term = |i: usize, step: usize| {
            let saturated = near[i];
            if saturated < BASE_CONTEXT_LEVEL_MAX {
                u32::from(saturated)
            } else {
                u32::from(levels[pos + step].unsigned_abs()).min(MAX_BASE_BR_RANGE as u32)
            }
        };
        term(a, self.column) + term(b, 1) + term(c, self.column + 1)
    }
}

/// Context of `coeff_base` for transform class 2D (libaom
/// `get_lower_levels_ctx_2d`) from its neighbour sum, plus the position's
/// offset `nz_offset` from `nz_map_ctx_offset`.
fn base_context(pos: usize, base_sum: u32, nz_offset: u8) -> usize {
    if pos == 0 {
        return 0;
    }
    ac_base_context(base_sum, nz_offset)
}

/// [`base_context`] of a coefficient other than the DC.
fn ac_base_context(base_sum: u32, nz_offset: u8) -> usize {
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
    w.write_literal(x, 2 * length - 1);
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

    /// Every shape the coder supports: the squares and the 2:1 rectangles.
    fn coded_shapes() -> Vec<TxSize> {
        (2..=6)
            .flat_map(|w: u8| (2..=6).filter_map(move |h: u8| TxSize::new(w, h)))
            .filter(|tx| tx.width_log2().abs_diff(tx.height_log2()) <= 1)
            .collect()
    }

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
            let tx = {
                let shapes = coded_shapes();
                shapes[round % shapes.len()]
            };
            let plane = if round % 3 == 0 {
                PlaneType::Chroma
            } else {
                PlaneType::Luma
            };
            let area = tx.coded_area();
            let reach = [1, 4, 40, area][(next() % 4) as usize].min(area);
            let mut levels = vec![0_i16; area];
            for &p in &scan(tx)[..reach] {
                let v = match next() % 10 {
                    0..=4 => 0,
                    5 | 6 => 1,
                    7 => 2,
                    8 => 3 + (next() % 12) as i16,
                    _ => 15 + (next() % 300) as i16,
                };
                levels[usize::from(p)] = if next() % 2 == 0 { v } else { -v };
            }
            let eob = end_of_block(tx, &levels);
            let ctx = TxbContext::derive(plane, true, &[(next() % 64) as u8], &[0x3]);
            let (is_inter, mode) = (round % 2 == 0, (next() % 13) as usize);
            let syntax = TypeSyntax::dct(is_inter, mode);
            let mut counter = BitCounter::default();
            cdfs.clone()
                .write(&mut counter, tx, plane, syntax, ctx, &levels, eob);
            let cost = cdfs.cost(tx, plane, syntax, ctx, &levels, eob);
            assert_eq!(
                cost.to_bits(),
                counter.bits().to_bits(),
                "{tx:?} round {round}"
            );
            cdfs.write(&mut writer, tx, plane, syntax, ctx, &levels, eob);
        }
    }

    /// The DC shortcut must match the general context walk even at signed
    /// extremes and range/Golomb boundaries, with CDFs adapted beforehand.
    #[test]
    fn dc_shortcut_matches_general_coefficient_walk() {
        let mut cdfs = CoeffCdfs::new(2);
        let mut adaptation = crate::range_coder::SymbolWriter::new();
        for tx in coded_shapes() {
            for plane in [PlaneType::Luma, PlaneType::Chroma] {
                for is_inter in [false, true] {
                    for value in [
                        i16::MIN,
                        -32767,
                        -15,
                        -14,
                        -13,
                        -6,
                        -5,
                        -4,
                        -3,
                        -2,
                        -1,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        13,
                        14,
                        15,
                        i16::MAX,
                    ] {
                        for dc_sign in 0..DC_SIGN_CONTEXTS {
                            let ctx = TxbContext { skip: 0, dc_sign };
                            let mut levels = vec![0; tx.coded_area()];
                            levels[0] = value;
                            let mut reference = cdfs.clone();
                            let mut counter = BitCounter::default();
                            let (txs, pt) = (tx.context(), plane as usize);
                            counter.add_q9(reference.txb_skip[txs][ctx.skip].cost_q9(0));
                            let syntax = TypeSyntax::dct(is_inter, 0);
                            if plane == PlaneType::Luma {
                                reference.write_tx_type(&mut counter, tx, syntax);
                            }
                            reference.write_eob(&mut counter, tx, pt, 1);
                            let magnitudes = Magnitudes::of(tx, 1, &levels);
                            let contexts = (
                                &mut reference.base_eob[txs][pt],
                                &mut reference.base[txs][pt],
                                &mut reference.br[txs.min(3)][pt],
                            );
                            let sign = &mut reference.dc_sign[pt][dc_sign];
                            with_level_map!(tx, map => magnitudes.cost::<_, _, false>(&mut counter, contexts, sign, map, &[]));
                            assert_eq!(
                                cdfs.cost(tx, plane, syntax, ctx, &levels, 1).to_bits(),
                                counter.bits().to_bits(),
                                "{tx:?} {plane:?} inter={is_inter} dc={value} sign_ctx={dc_sign}",
                            );
                            cdfs.write(&mut adaptation, tx, plane, syntax, ctx, &levels, 1);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn eob_is_one_past_last_nonzero_in_scan_order() {
        let mut levels = vec![0i16; 16];
        assert_eq!(end_of_block(TxSize::TX_4X4, &levels), 0);
        levels[0] = 3;
        assert_eq!(end_of_block(TxSize::TX_4X4, &levels), 1);
        levels[4] = -1; // scan position 1 in libaom's 4x4 default scan
        assert_eq!(end_of_block(TxSize::TX_4X4, &levels), 2);
    }

    #[test]
    fn entropy_context_saturates_and_records_dc_sign() {
        let mut levels = vec![0i16; 16];
        levels[0] = -20;
        assert_eq!(entropy_context(TxSize::TX_4X4, &levels, 1), 7 | (1 << 3));
        levels[0] = 2;
        assert_eq!(entropy_context(TxSize::TX_4X4, &levels, 1), 2 | (2 << 3));
    }

    #[test]
    fn uncoded_coefficients_do_not_read_stale_levels() {
        use crate::range_coder::SymbolWriter;
        for tx in coded_shapes() {
            // Empty levels deliberately prove the EOB-zero contract used by
            // backend scratch reuse; poisoned old coefficients must also be inert.
            let poisoned = vec![i16::MIN; tx.coded_area()];
            for plane in [PlaneType::Luma, PlaneType::Chroma] {
                for inter in [false, true] {
                    let ctx = TxbContext::derive(plane, true, &[0], &[0]);
                    let mut bytes = Vec::new();
                    let mut costs = Vec::new();
                    for levels in [&[][..], poisoned.as_slice()] {
                        let mut cdfs = CoeffCdfs::new(2);
                        costs.push(cdfs.cost(tx, plane, TypeSyntax::dct(inter, 0), ctx, levels, 0));
                        let mut writer = SymbolWriter::new();
                        cdfs.write(
                            &mut writer,
                            tx,
                            plane,
                            TypeSyntax::dct(inter, 0),
                            ctx,
                            levels,
                            0,
                        );
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
