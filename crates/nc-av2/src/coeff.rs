//! Final AV2 coefficient syntax for square DCT/WHT transforms, TX4 to TX64.
//!
//! One transform block is coded as: all-zero flag, end of block, the luma
//! transform type (`tx_type`), base and low-range levels in reverse scan
//! order, then signs and adaptive high-range Rice remainders. All planes and
//! both prediction kinds are covered; TX64 codes its top-left 32×32 only.
//! Parity hiding and trellis-coded quantisation are optional per frame
//! ([`Kind::levels`], modules [`parity`] and [`tcq`]); forward skip coding
//! and secondary transforms are off in our sequence header, so their syntax
//! never appears. CDF adaptation is
//! optional (see `entropy`).
//!
//! The coefficient walk is written once over `SymbolCoder`: the writer
//! derives every symbol from the source block, the reader decodes it, and both
//! rebuild identical level maps, so contexts cannot drift apart.
//!
//! Reimplemented from AVM v1.0.0 `av2/encoder/encodetxb.c`
//! (`av2_write_coeffs_txb`, `code_eob`), `av2/decoder/decodetxb.c`
//! (`av2_read_sig_txtype`, `decode_eob`, `av2_read_coeffs_txb`),
//! `av2/common/txb_common.h` and `av2/common/hr_coding.c`; see `THIRD_PARTY.md`
//! for attribution and patent terms.
mod context;
mod levels;
mod models;
pub(crate) mod parity;
mod symbols;
pub(crate) mod tcq;
pub(crate) mod trellis;

pub use crate::tx_size::TxSize;
pub use crate::tx_type::{TxSetMode, TxType};

use crate::tx_type::{INTER_TX_TYPE_INDEX_COUNT, InterSet, Signalling};
use context::EntropyContext;
pub use context::{Context, entropy_context};

use crate::entropy::{Reader, Writer};
use levels::{LevelMap, Neighbours, ScanPosition};
use models::BaseFamily;
use nc_common::decode::{DecodeError, Result};
use symbols::{FrozenPricer, SymbolCoder};

/// Largest coefficient magnitude: AVM rejects levels needing more than 20 bits.
const MAX_LEVEL: u32 = (1 << 20) - 1;
/// Plane indices as used by [`Context::plane`].
const PLANE_Y: u8 = 0;
const PLANE_U: u8 = 1;
const PLANE_V: u8 = 2;
/// Start of each end-of-block class (AVM `av2_eob_group_start`); class `c`
/// covers `EOB_GROUP_START[c]..EOB_GROUP_START[c + 1]`.
const EOB_GROUP_START: [usize; 12] = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513];
/// Classes below this one carry no offset bits (AVM `av2_eob_offset_bits`).
const FIRST_EOB_CLASS_WITH_OFFSET: usize = 3;
/// Last symbol of the capped EOB class models (`EOB_PT_INDEX_COUNT - 1`).
const EOB_CAPPED_SYMBOL: usize = 7;
/// Lossless inter TX4 type symbol meaning WHT rather than IDTX.
const LOSSLESS_WHT_SYMBOL: usize = 0;
/// `EXT_NEW_TX_SET` index of `DCT_DCT` for every intra mode
/// (AVM `av2_md_idx2type`).
const INTRA_DCT_INDEX: usize = 0;
/// Source used by the reader, which derives no symbols from it.
const NO_SOURCE: [i32; TxSize::Tx64.coded_len()] = [0; TxSize::Tx64.coded_len()];

/// The transform type of one block and the frame's set it is coded in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kind {
    /// The frame's `reduced_tx_set_used`.
    pub set: TxSetMode,
    /// The luma transform type; chroma and lossless intra are `DCT_DCT`.
    pub tx_type: TxType,
    /// How the frame codes luma levels.
    pub levels: LevelCoding,
}

/// How a frame codes its lossy luma levels; the two tools are exclusive
/// (AVM reads `allow_parity_hiding` only when `tcq_mode` is off).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LevelCoding {
    /// Plain levels.
    #[default]
    Plain,
    /// DC parities may be hidden (`allow_parity_hiding`, [`parity`]).
    ParityHiding,
    /// Trellis-coded quantisation (`tcq_mode`, [`tcq`]).
    Tcq,
}

impl LevelCoding {
    /// The frame's coding from its header flags.
    #[must_use]
    pub const fn of(parity_hiding: bool, tcq: bool) -> Self {
        match (parity_hiding, tcq) {
            (_, true) => Self::Tcq,
            (true, false) => Self::ParityHiding,
            (false, false) => Self::Plain,
        }
    }
}

impl Kind {
    /// `DCT_DCT` in a frame whose intra blocks infer it (the standalone
    /// picture pipelines).
    pub const DCT_ONLY: Self = Self {
        set: TxSetMode::IntraDct,
        tx_type: TxType::DctDct,
        levels: LevelCoding::Plain,
    };
}

/// Write one transform block, including its all-zero flag, in a
/// [`Kind::DCT_ONLY`] frame. See [`write_typed`].
/// # Errors
/// As [`write_typed`].
pub fn write(
    w: &mut Writer,
    coefficients: &[i32],
    tx: TxSize,
    ctx: Context,
    qindex: u8,
) -> Result<()> {
    write_typed(w, coefficients, tx, ctx, qindex, Kind::DCT_ONLY).map(|_| ())
}

/// Write one transform block, including its all-zero flag.
///
/// `coefficients` holds the quantized levels of the coded region in raster
/// order with stride [`TxSize::coded_width`]: the whole block up to TX32,
/// the top-left 32×32 for TX64 (as AVM stores them). Lossless (`qindex ==
/// 0`) blocks use TX4 WHT and signal it for inter luma. Returns the block's
/// [`entropy_context`], gathered while coding.
/// # Errors
/// Rejects invalid contexts, unsupported plane/size/quantizer combinations,
/// a wrongly sized slice, magnitudes above 20 bits and a transform type the
/// block cannot signal (outside the set, or non-DCT where `DCT_DCT` is
/// inferred: chroma, TX64, DC-only intra blocks).
pub fn write_typed(
    w: &mut Writer,
    coefficients: &[i32],
    tx: TxSize,
    ctx: Context,
    qindex: u8,
    kind: Kind,
) -> Result<u8> {
    let stats = LevelStats::of(coefficients, tx)?;
    write_typed_with_stats(w, coefficients, stats, tx, ctx, qindex, kind)
}

/// What the coefficient walk needs to know of a transform's levels before
/// it starts: their end of block, from one raster pass that also checks the
/// 20-bit magnitude limit. An encoder computes it once when it quantizes
/// and keeps it with the levels, so each rate estimate and the final write
/// of the same levels skip those passes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LevelStats {
    eob: u16,
}

impl LevelStats {
    /// The statistics of an all-zero transform.
    pub(crate) const EMPTY: Self = Self { eob: 0 };

    /// The statistics of `levels` (laid out as for [`write()`]) of `tx`.
    /// # Errors
    /// Rejects magnitudes above 20 bits.
    pub(crate) fn of(levels: &[i32], tx: TxSize) -> Result<Self> {
        let (eob, bits) = levels::end_and_magnitude_bits(levels, tx);
        if bits > MAX_LEVEL {
            return Err(DecodeError::Malformed("AV2 coefficient exceeds 20 bits"));
        }
        Ok(Self { eob: eob as u16 })
    }

    /// The statistics of quantizer output `levels` whose end of block `eob`
    /// is known. The quantizer clamps every level to 20 bits (and the
    /// trellis and trimming only lower or keep them), so no pass checks the
    /// limit; debug builds check both.
    pub(crate) fn quantized(levels: &[i32], tx: TxSize, eob: usize) -> Self {
        debug_assert_eq!(Self::of(levels, tx).ok(), Some(Self { eob: eob as u16 }));
        Self { eob: eob as u16 }
    }

    /// One past the scan index of the last nonzero level (0: all zero).
    pub(crate) fn eob(self) -> usize {
        usize::from(self.eob)
    }
}

/// [`write_typed`] of levels whose [`LevelStats`] are known.
/// # Errors
/// As [`write_typed`].
pub(crate) fn write_typed_with_stats(
    w: &mut Writer,
    coefficients: &[i32],
    stats: LevelStats,
    tx: TxSize,
    ctx: Context,
    qindex: u8,
    kind: Kind,
) -> Result<u8> {
    debug_assert_eq!(LevelStats::of(coefficients, tx).ok(), Some(stats));
    if !w.estimates_frozen_now() {
        return write_with(w, coefficients, stats, tx, ctx, qindex, kind);
    }
    // A frozen estimate prices the whole block from the live models in one
    // pass and adds the total, instead of dispatching symbol by symbol
    // through the writer: the same rate, about a third fewer instructions
    // per coded position.
    let mut pricer = FrozenPricer::new(w.live_models());
    let context = write_with(&mut pricer, coefficients, stats, tx, ctx, qindex, kind)?;
    let q9 = pricer.total_q9();
    w.add_estimate_q9(q9);
    Ok(context)
}

/// [`write_typed`] through any coder that takes its symbols from the
/// source.
fn write_with<C: SymbolCoder>(
    w: &mut C,
    coefficients: &[i32],
    stats: LevelStats,
    tx: TxSize,
    ctx: Context,
    qindex: u8,
    kind: Kind,
) -> Result<u8> {
    validate(tx, ctx, qindex, coefficients.len())?;
    let source = Source {
        levels: coefficients,
        eob: stats.eob(),
        tx_type: kind.tx_type,
    };
    let mut context = EntropyContext::default();
    code(
        w,
        Block::new(tx, ctx, qindex, (kind.set, kind.levels)),
        source,
        |pos, value| {
            context.add(pos, value);
        },
    )?;
    Ok(context.value())
}

/// Read one transform block (layout as for [`write()`]) of a
/// [`Kind::DCT_ONLY`] frame.
/// # Errors
/// As [`read_typed`], and rejects non-DCT transform types.
pub fn read(
    r: &mut Reader<'_>,
    coefficients: &mut [i32],
    tx: TxSize,
    ctx: Context,
    qindex: u8,
) -> Result<()> {
    match read_typed(
        r,
        coefficients,
        tx,
        ctx,
        qindex,
        (Kind::DCT_ONLY.set, LevelCoding::Plain),
    )? {
        TxType::DctDct => Ok(()),
        _ => Err(DecodeError::Unsupported("AV2 non-DCT transform type")),
    }
}

/// Read one transform block into `coefficients` (layout as for
/// [`write_typed`]) of a frame with transform set `set` coding luma levels
/// as `levels`, returning its transform type (`DCT_DCT` when all
/// coefficients are zero). The coefficients are the coded levels; a TCQ
/// transform dequantizes them with [`tcq::dequantize`].
/// # Errors
/// Rejects invalid contexts, unsupported combinations, a wrongly sized slice,
/// level overflow and exhausted input.
pub fn read_typed(
    r: &mut Reader<'_>,
    coefficients: &mut [i32],
    tx: TxSize,
    ctx: Context,
    qindex: u8,
    (set, levels): (TxSetMode, LevelCoding),
) -> Result<TxType> {
    validate(tx, ctx, qindex, coefficients.len())?;
    coefficients.fill(0);
    let source = Source {
        levels: &NO_SOURCE[..tx.coded_len()],
        eob: 0,
        tx_type: TxType::DctDct,
    };
    let block = Block::new(tx, ctx, qindex, (set, levels));
    code(r, block, source, |pos, value| {
        coefficients[pos] = value;
    })
}

fn validate(tx: TxSize, ctx: Context, qindex: u8, len: usize) -> Result<()> {
    ctx.validate()?;
    if len != tx.coded_len() {
        return Err(DecodeError::Malformed(
            "AV2 coefficient block size mismatch",
        ));
    }
    // AVM `get_lossless_tx_size` keeps lossless chroma and intra at TX4 and
    // makes larger lossless inter transforms IDTX, which this subset never uses.
    if qindex == 0 && tx != TxSize::Tx4 {
        return Err(DecodeError::Unsupported("AV2 lossless requires TX4"));
    }
    // AVM `av2_get_tx_size` caps chroma at TX32 (`av2_get_adjusted_tx_size`).
    if ctx.plane != PLANE_Y && tx == TxSize::Tx64 {
        return Err(DecodeError::Unsupported(
            "AV2 chroma transforms stop at TX32",
        ));
    }
    Ok(())
}

/// The coding order of a transform's coded positions (raster indices at
/// the coded side), DC first.
#[must_use]
pub fn scan(tx: TxSize) -> &'static [u16] {
    levels::scan(tx)
}

/// One past the scan index of the last nonzero level of `levels` (laid out
/// as for [`write()`]), or 0 for an all-zero block: one raster pass.
#[must_use]
pub fn end_of_block(levels: &[i32], tx: TxSize) -> usize {
    levels::end_of_block(levels, tx)
}

/// Coefficient CDF quantizer category (AVM `get_q_ctx` in `entropy.c`).
fn quantizer_category(qindex: u8) -> usize {
    match qindex {
        0..=90 => 0,
        91..=140 => 1,
        141..=190 => 2,
        _ => 3,
    }
}

/// The fixed parameters of one transform block.
#[derive(Clone, Copy, Debug)]
struct Block {
    tx: TxSize,
    ctx: Context,
    /// Coefficient CDF quantizer category.
    q: usize,
    /// Lossless (qindex zero).
    lossless: bool,
    /// The frame's transform set.
    set: TxSetMode,
    /// How the frame codes luma levels.
    levels: LevelCoding,
}

impl Block {
    fn new(tx: TxSize, ctx: Context, qindex: u8, (set, levels): (TxSetMode, LevelCoding)) -> Self {
        Self {
            tx,
            ctx,
            q: quantizer_category(qindex),
            lossless: qindex == 0,
            set,
            levels,
        }
    }

    /// Whether this transform's levels are trellis-coded (AVM `tcq_enable`:
    /// lossy luma of a TCQ frame; every type we code is of the 2D class).
    fn tcq(self) -> bool {
        self.levels == LevelCoding::Tcq && self.ctx.plane == PLANE_Y && !self.lossless
    }
}

/// What the writer codes: the levels, their end of block and the transform
/// type. The reader passes zeros and `DCT_DCT`, which it ignores.
#[derive(Clone, Copy, Debug)]
struct Source<'a> {
    levels: &'a [i32],
    eob: usize,
    tx_type: TxType,
}

/// Code one block, returning its transform type. The writer takes every
/// symbol from `source`; the reader reports each nonzero coefficient to
/// `store`.
fn code<C: SymbolCoder>(
    coder: &mut C,
    block: Block,
    source: Source<'_>,
    store: impl FnMut(usize, i32),
) -> Result<TxType> {
    if models::all_zero(coder, usize::from(source.eob == 0), block)? != 0 {
        return Ok(TxType::DctDct);
    }
    let eob = code_eob(coder, source.eob, block)?;
    let tx_type = code_tx_type(coder, eob, block, source.tx_type)?;
    // One walk per level coding, so the plain walk (most blocks of every
    // frame) carries neither the parity tally nor the TCQ state.
    if parity::applies(block, tx_type, eob) {
        code_levels::<C, HIDING_WALK>(coder, block, (source.levels, eob), store)?;
    } else if block.tcq() {
        code_levels::<C, TCQ_WALK>(coder, block, (source.levels, eob), store)?;
    } else {
        code_levels::<C, PLAIN_WALK>(coder, block, (source.levels, eob), store)?;
    }
    Ok(tx_type)
}

/// The level walks of [`code_levels`]: plain levels, a block that may hide
/// its DC parity, and trellis-coded levels.
const PLAIN_WALK: u8 = 0;
const HIDING_WALK: u8 = 1;
const TCQ_WALK: u8 = 2;

/// Code the levels of a block whose end of block is `eob` (base and
/// low-range symbols in reverse scan order, then signs and high-range
/// remainders) with the level coding `WALK` names.
fn code_levels<C: SymbolCoder, const WALK: u8>(
    coder: &mut C,
    block: Block,
    (source, eob): (&[i32], usize),
    mut store: impl FnMut(usize, i32),
) -> Result<()> {
    let (tx, ctx) = (block.tx, block.ctx);
    let (hides, tcq) = (WALK == HIDING_WALK, WALK == TCQ_WALK);
    let positions = &levels::scan_positions(tx)[..eob];
    // AVM sums the base and low-range levels after DC (`num_nz`,
    // `sum_abs1` in `av2_read_coeffs_txb`) as they are coded.
    let mut after_dc = parity::Tally::default();
    let mut hidden = false;
    let mut state = tcq::INITIAL_STATE;
    let mut map = LevelMap::new();
    // The last coefficient is coded before the loop (as nc-av1's walk), so
    // the loop does not test for it.
    let (&last_at, earlier) = positions
        .split_last()
        .expect("a block with levels has an end of block");
    let level = code_last_level(coder, block, (last_at, earlier.len()), source, tcq)?;
    map.set(last_at, level);
    if tcq {
        state = tcq::next_state(state, level);
    }
    if hides && !earlier.is_empty() {
        after_dc.add(level);
    }
    for (index, &at) in earlier.iter().enumerate().rev() {
        let region = at.region(ctx.plane);
        let target = source[at.pos()].unsigned_abs();
        let neighbours = map.neighbours(at);
        if index == 0 && hides && after_dc.hides() {
            hidden = true;
            let level = parity::code_hidden_level(coder, block, neighbours, target, after_dc)?;
            map.set(at, level);
            continue;
        }
        let base_context = levels::base_context(neighbours, at, region, ctx.plane);
        // A TCQ level codes the low part that keeps its parity.
        let (target, upper) = if tcq {
            (
                tcq::low_part(target, region.saturation()),
                tcq::upper(state),
            )
        } else {
            (target, false)
        };
        let coefficient = Coefficient {
            at,
            family: BaseFamily {
                region,
                last: false,
                upper,
            },
            target,
        };
        let level = code_level(coder, neighbours, block, coefficient, base_context)?;
        map.set(at, level);
        if tcq {
            state = tcq::next_state(state, level);
        }
        if hides && index > 0 {
            after_dc.add(level);
        }
    }
    let mut average = 0;
    for &at in positions.iter().rev() {
        let pos = at.pos();
        let coded = map.get(at);
        if coded == 0 {
            continue;
        }
        let negative = source[pos] < 0;
        // AVM codes only the luma DC sign with a model; U and V DC signs and
        // every AC sign are bypass bits.
        let sign = if pos == 0 && ctx.plane == PLANE_Y {
            models::dc_sign(coder, usize::from(negative), block, hidden)? != 0
        } else {
            coder.code_literal(u32::from(negative), 1)? != 0
        };
        let target = source[pos].unsigned_abs();
        let saturation = at.region(ctx.plane).saturation();
        let level = if hidden && pos == 0 {
            parity::code_hidden_high_range(coder, coded, target, average)?
        } else if tcq && tcq::has_remainder(coded, saturation) {
            // TCQ remainders count pairs, keeping the coded parity.
            let high = coder.code_high_range(target.saturating_sub(coded) / 2, average)?;
            average = u32::midpoint(average, high);
            coded + 2 * high
        } else if coded == saturation {
            let high = coder.code_high_range(target.saturating_sub(coded), average)?;
            average = u32::midpoint(average, high);
            coded + high
        } else {
            coded
        };
        if level > MAX_LEVEL {
            return Err(DecodeError::Malformed("AV2 coefficient exceeds 20 bits"));
        }
        let magnitude = level as i32;
        store(pos, if sign { -magnitude } else { magnitude });
    }
    Ok(())
}

/// The template neighbours of the last coefficient: nothing is coded yet.
const NO_NEIGHBOURS: Neighbours = [0; levels::NEIGHBOURS.len()];

/// Code the base symbol and low-range increment of the last coefficient,
/// at scan position `at` (scan index `index`) of `source`. It is nonzero by
/// definition; its neighbours are all zero (nothing is coded yet), which
/// gives AVM's fixed contexts for it; and its TCQ models have no second
/// quantizer set.
fn code_last_level<C: SymbolCoder>(
    coder: &mut C,
    block: Block,
    (at, index): (ScanPosition, usize),
    source: &[i32],
    tcq: bool,
) -> Result<u32> {
    let region = at.region(block.ctx.plane);
    let target = source[at.pos()].unsigned_abs();
    let coefficient = Coefficient {
        at,
        family: BaseFamily {
            region,
            last: true,
            upper: false,
        },
        target: if tcq {
            tcq::low_part(target, region.saturation())
        } else {
            target
        },
    };
    let base_context = levels::last_base_context(index, block.tx);
    code_level(coder, NO_NEIGHBOURS, block, coefficient, base_context)
}

/// One coefficient of the level pass: its scan position, its base model
/// family (whose `last` coefficient is nonzero by definition), and the
/// source magnitude its base and low-range symbols code (the reader's is
/// zero).
#[derive(Debug, Clone, Copy)]
struct Coefficient {
    at: ScanPosition,
    family: BaseFamily,
    target: u32,
}

/// Code the base symbol and low-range increment of one coefficient whose
/// template neighbours are `neighbours`, returning its level saturated at
/// the region's high-range start.
// Inlined into each walk: called once per coded position, its symbol
// selection folds away with the walk's constants (out of line it cost
// about 2% of AV2 encode instructions).
#[allow(
    clippy::inline_always,
    reason = "per-position symbol selection of every walk"
)]
#[inline(always)]
fn code_level<C: SymbolCoder>(
    coder: &mut C,
    neighbours: Neighbours,
    block: Block,
    coefficient: Coefficient,
    base_context: usize,
) -> Result<u32> {
    let Coefficient { at, family, target } = coefficient;
    let region = family.region;
    let base_limit = region.base_levels() + 1;
    // The last coefficient is nonzero by definition, so its alphabet omits zero.
    let offset = u32::from(family.last);
    let symbol = target.min(base_limit).saturating_sub(offset) as usize;
    let mut level = coder.code_base(symbol, block, family, base_context)? as u32 + offset;
    if region.has_low_range() && level > region.base_levels() {
        let increment = target.saturating_sub(base_limit).min(levels::MAX_LOW_RANGE);
        let context = levels::low_range_context(neighbours, at, region);
        level += coder.code_low_range(increment as usize, block.q, region, context)? as u32;
    }
    Ok(level)
}

/// Code the end of block (AVM `code_eob` / `decode_eob`) and return it.
/// Sizes from TX16 cap the class model and code the overflow as bypass bits.
fn code_eob<C: SymbolCoder>(coder: &mut C, source_eob: usize, block: Block) -> Result<usize> {
    let source_class = EOB_GROUP_START
        .iter()
        .rposition(|&start| start <= source_eob)
        .unwrap_or_default();
    let source_symbol = source_class.saturating_sub(1);
    let class = match eob_overflow_bits(block.tx) {
        0 => models::eob_class(coder, source_symbol, block)? + 1,
        bits => {
            let capped = models::eob_class(coder, source_symbol.min(EOB_CAPPED_SYMBOL), block)?;
            let overflow = if capped == EOB_CAPPED_SYMBOL {
                coder.code_literal(source_symbol.saturating_sub(capped) as u32, bits)?
            } else {
                0
            };
            capped + overflow as usize + 1
        }
    };
    let mut eob = EOB_GROUP_START[class];
    if class >= FIRST_EOB_CLASS_WITH_OFFSET {
        let bits = (class - FIRST_EOB_CLASS_WITH_OFFSET + 1) as u32;
        let offset = source_eob.saturating_sub(eob) as u32;
        let top = models::eob_extra(coder, (offset >> (bits - 1)) as usize, block.q)? as u32;
        let rest = coder.code_literal(offset & ((1 << (bits - 1)) - 1), bits - 1)?;
        eob += ((top << (bits - 1)) | rest) as usize;
    }
    Ok(eob)
}

/// Bypass bits after a capped EOB class symbol (AVM `code_eob` cases 4..6).
fn eob_overflow_bits(tx: TxSize) -> u32 {
    match tx {
        TxSize::Tx4 | TxSize::Tx8 => 0,
        TxSize::Tx16 => 1,
        TxSize::Tx32 | TxSize::Tx64 => 2,
    }
}

/// Luma transform type (AVM `av2_write_tx_type` / `av2_read_tx_type`):
/// the lossless inter TX4 WHT/IDTX flag (we always choose the WHT), or
/// `source` coded as [`TxSetMode::signalling`] gives the block. Returns the
/// coded type; blocks that signal nothing infer `DCT_DCT`. 1-D types and
/// intra mode-dependent indices other than `DCT_DCT` are outside our subset.
fn code_tx_type<C: SymbolCoder>(
    coder: &mut C,
    eob: usize,
    block: Block,
    source: TxType,
) -> Result<TxType> {
    let (tx, ctx) = (block.tx, block.ctx);
    let unsignalled = |source: TxType| match source {
        TxType::DctDct => Ok(TxType::DctDct),
        _ => Err(DecodeError::Unsupported(
            "AV2 transform type is inferred as DCT here",
        )),
    };
    if ctx.plane != PLANE_Y {
        return unsignalled(source);
    }
    if block.lossless {
        if ctx.is_inter {
            let symbol = models::lossless_inter_tx_type(coder, LOSSLESS_WHT_SYMBOL)?;
            if symbol != LOSSLESS_WHT_SYMBOL {
                return Err(DecodeError::Unsupported("AV2 lossless inter IDTX"));
            }
        }
        return unsignalled(source);
    }
    let outside = DecodeError::Unsupported("AV2 transform type outside the frame's set");
    let eob_context = levels::tx_type_eob_context(eob, tx);
    match block.set.signalling(tx, ctx.is_inter, eob) {
        Signalling::Inferred => unsignalled(source),
        Signalling::Symbol(types) => {
            let source_symbol = types.iter().position(|&t| t == source).ok_or(outside)?;
            let symbol = if ctx.is_inter {
                models::inter_tx_type(coder, source_symbol, tx, eob_context)?
            } else {
                models::intra_tx_type(coder, source_symbol, tx)?
            };
            Ok(types[symbol])
        }
        Signalling::SetIndex(set) => {
            let types = set.types();
            let source_index = types.iter().position(|&t| t == source).ok_or(outside)?;
            let index = code_inter_set_index(coder, source_index, set, tx, eob_context)?;
            let tx_type = types[index];
            if tx_type.is_two_dimensional() {
                Ok(tx_type)
            } else {
                Err(DecodeError::Unsupported("AV2 1-D transform type"))
            }
        }
        Signalling::IntraModeDependent => {
            unsignalled(source)?;
            match models::intra_full_tx_type(coder, INTRA_DCT_INDEX, tx)? {
                INTRA_DCT_INDEX => Ok(TxType::DctDct),
                _ => Err(DecodeError::Unsupported(
                    "AV2 intra mode-dependent transform type",
                )),
            }
        }
    }
}

/// The index of an inter type in `set` (AVM `eset` 1 and 2): a flag says
/// whether it lies below [`INTER_TX_TYPE_INDEX_COUNT`], then an index or the
/// offset past that count is coded.
fn code_inter_set_index<C: SymbolCoder>(
    coder: &mut C,
    source_index: usize,
    set: InterSet,
    tx: TxSize,
    eob_context: usize,
) -> Result<usize> {
    let offset = usize::from(source_index >= INTER_TX_TYPE_INDEX_COUNT);
    if models::inter_tx_set_flag(coder, offset, set, tx, eob_context)? == 0 {
        return models::inter_tx_index(coder, source_index, set, eob_context);
    }
    let source_offset = source_index.saturating_sub(INTER_TX_TYPE_INDEX_COUNT);
    let offset = models::inter_tx_offset(coder, source_offset, set, eob_context)?;
    Ok(INTER_TX_TYPE_INDEX_COUNT + offset)
}

#[cfg(test)]
mod tests;
