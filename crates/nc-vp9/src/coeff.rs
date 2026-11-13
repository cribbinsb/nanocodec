//! VP9 coefficient tokens for every transform size: scan orders, band and
//! neighbour contexts, default probability models and the Pareto tail
//! (VP9 §9.3; libvpx `vp9_detokenize.c` / `vp9_tokenize.c`, BSD-3-Clause,
//! see `THIRD_PARTY.md`).
//!
//! Levels are passed in raster order; the caller supplies the entropy
//! context (0..=2 nonzero neighbours). The encoder dequantizes; the
//! decoder's [`read`] dequantizes as it reads, as libvpx does.

use crate::bool_coder::{BitSink, BoolReader};
use crate::probs::{BandCounts, Counts, EOB_MODEL_TOKEN, FrameContext};
use crate::recorder::{TileRecorder, coef_slot};
use crate::tables::{self, BandProbs};
use crate::transform::{TxSize, TxType};

/// Largest magnitude the 8-bit `CATEGORY6` token can carry.
pub(crate) const MAX_LEVEL: i32 = 16_450;

// Leaves represent 2, 3, 4, and categories 1..6, respectively.
const TAIL_TREE: [[i8; 2]; 8] = [
    [1, 3],
    [-1, 2],
    [-2, -3],
    [4, 5],
    [-4, -5],
    [6, 7],
    [-6, -7],
    [-8, -9],
];
const TAIL_CODES: [(u8, u32); 9] = [
    (0, 2),
    (2, 3),
    (3, 3),
    (4, 3),
    (5, 3),
    (12, 4),
    (13, 4),
    (14, 4),
    (15, 4),
];
const BASE: [i32; 9] = [2, 3, 4, 5, 7, 11, 19, 35, 67];
const EXTRA: [&[u8]; 9] = [
    &[],
    &[],
    &[],
    &[159],
    &[165, 145],
    &[173, 148, 140],
    &[176, 155, 140, 135],
    &[180, 157, 141, 134, 130],
    &tables::CATEGORY6,
];

/// Plane type of the probability model (`PLANE_TYPES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlaneType {
    Y = 0,
    Uv = 1,
}

/// One transform block's scan and neighbour lists, and the scan index of
/// every raster position.
struct ScanOrder {
    scan: &'static [u16],
    neighbors: &'static [u16],
    inverse: &'static [u16],
}

/// The scan index of every raster position of `scan`.
const fn inverted<const N: usize>(scan: &[u16; N]) -> [u16; N] {
    let mut inverse = [0; N];
    let mut c = 0;
    while c < N {
        inverse[scan[c] as usize] = c as u16;
        c += 1;
    }
    inverse
}

const DEFAULT_SCAN_4X4_INVERSE: [u16; 16] = inverted(&tables::DEFAULT_SCAN_4X4);
const ROW_SCAN_4X4_INVERSE: [u16; 16] = inverted(&tables::ROW_SCAN_4X4);
const COL_SCAN_4X4_INVERSE: [u16; 16] = inverted(&tables::COL_SCAN_4X4);
const DEFAULT_SCAN_8X8_INVERSE: [u16; 64] = inverted(&tables::DEFAULT_SCAN_8X8);
const ROW_SCAN_8X8_INVERSE: [u16; 64] = inverted(&tables::ROW_SCAN_8X8);
const COL_SCAN_8X8_INVERSE: [u16; 64] = inverted(&tables::COL_SCAN_8X8);
const DEFAULT_SCAN_16X16_INVERSE: [u16; 256] = inverted(&tables::DEFAULT_SCAN_16X16);
const ROW_SCAN_16X16_INVERSE: [u16; 256] = inverted(&tables::ROW_SCAN_16X16);
const COL_SCAN_16X16_INVERSE: [u16; 256] = inverted(&tables::COL_SCAN_16X16);
const DEFAULT_SCAN_32X32_INVERSE: [u16; 1024] = inverted(&tables::DEFAULT_SCAN_32X32);

/// Raster levels tested together for being all zero: most of a block's
/// levels are, and a run of them costs one comparison instead of one scan
/// lookup each.
const ZERO_TEST_RUN: usize = 8;

impl ScanOrder {
    /// The scan indices of the last and second-to-last nonzero levels of
    /// raster-order `levels`.
    fn last_two_nonzero(&self, levels: &[i32]) -> (Option<usize>, Option<usize>) {
        let (mut last, mut second) = (None, None);
        for (run, chunk) in levels[..self.scan.len()].chunks(ZERO_TEST_RUN).enumerate() {
            if chunk.iter().all(|&level| level == 0) {
                continue;
            }
            for (i, _) in chunk.iter().enumerate().filter(|&(_, &level)| level != 0) {
                let c = Some(usize::from(self.inverse[run * ZERO_TEST_RUN + i]));
                if c > last {
                    (last, second) = (c, last);
                } else if c > second {
                    second = c;
                }
            }
        }
        (last, second)
    }

    /// The end of block: one past the scan index of the last nonzero level.
    fn end_of_block(&self, levels: &[i32]) -> usize {
        self.last_two_nonzero(levels).0.map_or(0, |c| c + 1)
    }
}

/// `vp9_scan_orders[tx][tx_type]`: row scan for vertical ADST, column scan
/// for horizontal ADST, otherwise (and always at 32x32) the default scan.
fn scan_order(tx: TxSize, ty: TxType) -> ScanOrder {
    use tables::{
        COL_SCAN_4X4, COL_SCAN_4X4_NEIGHBORS, COL_SCAN_8X8, COL_SCAN_8X8_NEIGHBORS, COL_SCAN_16X16,
        COL_SCAN_16X16_NEIGHBORS, DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS, DEFAULT_SCAN_8X8,
        DEFAULT_SCAN_8X8_NEIGHBORS, DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X16_NEIGHBORS,
        DEFAULT_SCAN_32X32, DEFAULT_SCAN_32X32_NEIGHBORS, ROW_SCAN_4X4, ROW_SCAN_4X4_NEIGHBORS,
        ROW_SCAN_8X8, ROW_SCAN_8X8_NEIGHBORS, ROW_SCAN_16X16, ROW_SCAN_16X16_NEIGHBORS,
    };
    let (scan, neighbors, inverse): (&'static [u16], &'static [u16], &'static [u16]) =
        match (tx, ty) {
            (TxSize::Tx4, TxType::AdstDct) => (
                &ROW_SCAN_4X4,
                &ROW_SCAN_4X4_NEIGHBORS,
                &ROW_SCAN_4X4_INVERSE,
            ),
            (TxSize::Tx4, TxType::DctAdst) => (
                &COL_SCAN_4X4,
                &COL_SCAN_4X4_NEIGHBORS,
                &COL_SCAN_4X4_INVERSE,
            ),
            (TxSize::Tx4, _) => (
                &DEFAULT_SCAN_4X4,
                &DEFAULT_SCAN_4X4_NEIGHBORS,
                &DEFAULT_SCAN_4X4_INVERSE,
            ),
            (TxSize::Tx8, TxType::AdstDct) => (
                &ROW_SCAN_8X8,
                &ROW_SCAN_8X8_NEIGHBORS,
                &ROW_SCAN_8X8_INVERSE,
            ),
            (TxSize::Tx8, TxType::DctAdst) => (
                &COL_SCAN_8X8,
                &COL_SCAN_8X8_NEIGHBORS,
                &COL_SCAN_8X8_INVERSE,
            ),
            (TxSize::Tx8, _) => (
                &DEFAULT_SCAN_8X8,
                &DEFAULT_SCAN_8X8_NEIGHBORS,
                &DEFAULT_SCAN_8X8_INVERSE,
            ),
            (TxSize::Tx16, TxType::AdstDct) => (
                &ROW_SCAN_16X16,
                &ROW_SCAN_16X16_NEIGHBORS,
                &ROW_SCAN_16X16_INVERSE,
            ),
            (TxSize::Tx16, TxType::DctAdst) => (
                &COL_SCAN_16X16,
                &COL_SCAN_16X16_NEIGHBORS,
                &COL_SCAN_16X16_INVERSE,
            ),
            (TxSize::Tx16, _) => (
                &DEFAULT_SCAN_16X16,
                &DEFAULT_SCAN_16X16_NEIGHBORS,
                &DEFAULT_SCAN_16X16_INVERSE,
            ),
            (TxSize::Tx32, _) => (
                &DEFAULT_SCAN_32X32,
                &DEFAULT_SCAN_32X32_NEIGHBORS,
                &DEFAULT_SCAN_32X32_INVERSE,
            ),
        };
    ScanOrder {
        scan,
        neighbors,
        inverse,
    }
}

fn band(tx: TxSize, c: usize) -> usize {
    usize::from(if tx == TxSize::Tx4 {
        tables::BANDS_4X4[c]
    } else {
        tables::BANDS_8X8_PLUS[c]
    })
}

/// `vp9_pt_energy_class` of a token magnitude.
fn energy(magnitude: i32) -> u8 {
    match magnitude {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=10 => 4,
        _ => 5,
    }
}

/// Context of scan position `c` from its two already-coded neighbours.
fn context(cache: &[u8], c: usize, initial: usize, neighbors: &[u16]) -> usize {
    if c == 0 {
        initial
    } else {
        let a = cache[usize::from(neighbors[2 * c])];
        let b = cache[usize::from(neighbors[2 * c + 1])];
        usize::from((1 + a + b) >> 1)
    }
}

/// The probability model of one transform block.
#[derive(Clone, Copy)]
pub(crate) struct Model {
    pub tx: TxSize,
    pub ty: TxType,
    pub plane: PlaneType,
    pub inter: bool,
}

impl Model {
    /// The model of a `tx` / `ty` transform block of `plane`.
    pub(crate) fn of(plane: nc_common::PlaneIndex, tx: TxSize, ty: TxType, inter: bool) -> Self {
        let plane = if plane == nc_common::PlaneIndex::Y {
            PlaneType::Y
        } else {
            PlaneType::Uv
        };
        Self {
            tx,
            ty,
            plane,
            inter,
        }
    }
}

/// The frame's band/context probabilities for one model, and its counts.
pub(crate) struct Coder<'a> {
    pub probs: &'a BandProbs,
    pub counts: &'a mut BandCounts,
}

impl<'a> Coder<'a> {
    /// The model's probabilities and counts within the frame's.
    pub(crate) fn new(model: Model, context: &'a FrameContext, counts: &'a mut Counts) -> Self {
        let (t, p, r) = (
            model.tx as usize,
            model.plane as usize,
            usize::from(model.inter),
        );
        Self {
            probs: &context.coef[t][p][r],
            counts: &mut counts.coef[t][p][r],
        }
    }
}

fn write_magnitude(emit: &mut impl FnMut(bool, u8), value: i32, pivot: u8) {
    let token = BASE.iter().rposition(|&base| value >= base).unwrap();
    let (code, length) = TAIL_CODES[token];
    let probs = &tables::PARETO[usize::from(pivot) - 1];
    let mut node = 0;
    for shift in (0..length).rev() {
        let bit = (code >> shift) & 1 != 0;
        emit(bit, probs[node]);
        let child = TAIL_TREE[node][usize::from(bit)];
        if child < 0 {
            break;
        }
        node = child as usize;
    }
    let offset = value - BASE[token];
    for (i, &probability) in EXTRA[token].iter().enumerate() {
        emit(
            (offset >> (EXTRA[token].len() - 1 - i)) & 1 != 0,
            probability,
        );
    }
}

#[inline]
fn read_magnitude(reader: &mut BoolReader<'_>, pivot: u8) -> i32 {
    let probs = &tables::PARETO[usize::from(pivot) - 1];
    let mut node = 0;
    let token = loop {
        let bit = reader.read(probs[node]);
        let child = TAIL_TREE[node][usize::from(bit)];
        if child < 0 {
            break (-child - 1) as usize;
        }
        node = child as usize;
    };
    let mut offset = 0;
    for &probability in EXTRA[token] {
        offset = (offset << 1) | i32::from(reader.read(probability));
    }
    BASE[token] + offset
}

/// Emit the token bits of raster-order `levels` with neighbour context
/// `initial`, counting them when `counts` is given; returns the end of
/// block (number of scan positions coded).
fn emit_tokens(
    levels: &[i32],
    model: Model,
    probs: &BandProbs,
    counts: Option<&mut BandCounts>,
    initial: usize,
    emit: &mut impl FnMut(bool, u8, Option<Node>),
) -> usize {
    // The token cache is sized to the transform: rate estimation calls this
    // for every candidate, and zeroing a 32x32 cache for every 4x4 block
    // was most of its cost in WASM profiles.
    match model.tx {
        TxSize::Tx4 => emit_tokens_sized::<16>(levels, model, probs, counts, initial, emit),
        TxSize::Tx8 => emit_tokens_sized::<64>(levels, model, probs, counts, initial, emit),
        TxSize::Tx16 => emit_tokens_sized::<256>(levels, model, probs, counts, initial, emit),
        TxSize::Tx32 => emit_tokens_sized::<1024>(levels, model, probs, counts, initial, emit),
    }
}

/// [`emit_tokens`] with a token cache of `AREA` (the transform's area).
fn emit_tokens_sized<const AREA: usize>(
    levels: &[i32],
    model: Model,
    probs: &BandProbs,
    mut counts: Option<&mut BandCounts>,
    initial: usize,
    emit: &mut impl FnMut(bool, u8, Option<Node>),
) -> usize {
    debug_assert_eq!(model.tx.area(), AREA);
    let order = scan_order(model.tx, model.ty);
    let eob = order.end_of_block(levels);
    let mut cache = [0u8; AREA];
    let mut needs_eob = true;
    for (c, &position) in order.scan.iter().enumerate() {
        let (b, ctx) = (
            band(model.tx, c),
            context(&cache, c, initial, order.neighbors),
        );
        let probs = &probs[b][ctx];
        let mut count = counts.as_deref_mut().map(|counts| &mut counts[b][ctx]);
        if needs_eob {
            if let Some(count) = count.as_deref_mut() {
                count.eob_branch += 1;
                if c == eob {
                    count.tokens[EOB_MODEL_TOKEN] += 1;
                }
            }
            emit(
                c < eob,
                probs[0],
                Some(Node {
                    band: b,
                    ctx,
                    index: 0,
                }),
            );
            if c == eob {
                break;
            }
        }
        let value = levels[usize::from(position)];
        let magnitude = value.abs();
        debug_assert!(magnitude <= MAX_LEVEL);
        if let Some(count) = count {
            count.tokens[magnitude.min(2) as usize] += 1;
        }
        emit(
            magnitude != 0,
            probs[1],
            Some(Node {
                band: b,
                ctx,
                index: 1,
            }),
        );
        needs_eob = magnitude != 0;
        if magnitude != 0 {
            emit(magnitude > 1, probs[2], None);
            if magnitude > 1 {
                write_magnitude(&mut |bit, prob| emit(bit, prob, None), magnitude, probs[2]);
            }
            emit(value < 0, 128, None);
        }
        cache[usize::from(position)] = energy(magnitude);
    }
    eob
}

/// A node decision of one band/context model (nodes 0 and 1 may take
/// forward probability updates).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    pub band: usize,
    pub ctx: usize,
    pub index: usize,
}

/// Write (into the tile recorder) and count raster-order `levels` with
/// neighbour context `initial`; returns the end of block.
pub(crate) fn write(
    recorder: &mut TileRecorder,
    levels: &[i32],
    model: Model,
    coder: &mut Coder<'_>,
    initial: usize,
) -> usize {
    let (plane, reference) = (model.plane as usize, usize::from(model.inter));
    emit_tokens(
        levels,
        model,
        coder.probs,
        Some(coder.counts),
        initial,
        &mut |bit, prob, node| match node {
            Some(n) => recorder.write_coef(
                bit,
                prob,
                coef_slot(model.tx, plane, reference, n.band, n.ctx, n.index),
            ),
            None => recorder.write(bit, prob),
        },
    )
}

/// Drop trailing ±1 levels of raster-order `levels` that stand at least
/// `min_gap` scan positions after the previous nonzero level, from the end
/// backwards (SVT-AV1 `shave_coeff`, as HEVC's `shave_isolated_ones`: an
/// isolated high-frequency ±1 pays its end-of-block placement and zero
/// tokens for almost no distortion; it is mostly noise). The first nonzero
/// level is always kept. Returns whether any level remains.
pub(crate) fn shave_isolated_ones(
    levels: &mut [i32],
    tx: TxSize,
    ty: TxType,
    min_gap: usize,
) -> bool {
    let order = scan_order(tx, ty);
    loop {
        let (Some(last), Some(previous)) = order.last_two_nonzero(levels) else {
            // No level, or only the first one, which is always kept.
            return levels[..order.scan.len()].iter().any(|&level| level != 0);
        };
        let position = usize::from(order.scan[last]);
        if levels[position].abs() != 1 || last - previous < min_gap {
            return true;
        }
        levels[position] = 0;
    }
}

/// Estimated bits of coding `levels` (as [`write`], without counting).
pub(crate) fn cost(levels: &[i32], model: Model, probs: &BandProbs, initial: usize) -> f32 {
    let mut bits = 0.0;
    emit_tokens(levels, model, probs, None, initial, &mut |bit, prob, _| {
        bits += crate::tree::bit_cost(bit, prob);
    });
    bits
}

/// Scratch of [`read`]: the energy class of each coded position
/// (`token_cache`). Every position a context reads was coded earlier in
/// the scan, so it is never cleared.
pub(crate) struct TokenCache([u8; 32 * 32]);

impl TokenCache {
    pub(crate) fn new() -> Self {
        Self([0; 32 * 32])
    }
}

/// The decoder's dequantizer of one transform block (`dqv`, `dq_shift`):
/// a coefficient is `(|level| * step) >> shift` with the level's sign.
#[derive(Clone, Copy)]
pub(crate) struct Dequantizer {
    pub dc: i32,
    pub ac: i32,
    pub shift: u32,
}

/// Where a block's coded coefficients lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Extent {
    /// End of block: scan positions coded.
    pub eob: usize,
    /// Raster rows up to the last one holding a nonzero coefficient.
    pub rows: usize,
    /// Raster columns up to the last one holding a nonzero coefficient.
    pub columns: usize,
}

/// Scan and dequantization invariants shared by an inter block-plane.
/// Intra callers still prepare these per transform because their type can vary.
pub(crate) struct ReadSetup {
    order: ScanOrder,
    bands: &'static [u8],
    row_shift: u8,
    column_mask: usize,
    dequantizer: Dequantizer,
}

impl ReadSetup {
    pub(crate) fn new(model: Model, dequantizer: Dequantizer) -> Self {
        let row_shift = model.tx.log2();
        Self {
            order: scan_order(model.tx, model.ty),
            bands: if model.tx == TxSize::Tx4 {
                &tables::BANDS_4X4
            } else {
                &tables::BANDS_8X8_PLUS
            },
            row_shift,
            column_mask: (1 << row_shift) - 1,
            dequantizer,
        }
    }

    /// Clear only scan positions visited by the preceding transform block.
    pub(crate) fn clear(&self, coefficients: &mut [i32], eob: usize) {
        for &position in &self.order.scan[..eob] {
            coefficients[usize::from(position)] = 0;
        }
    }
}

/// Read a block's levels and store them dequantized, in raster order,
/// into `coefficients`, which must be zero on entry (the caller clears
/// what it read with [`clear`]); returns where they lie.
pub(crate) fn read(
    reader: &mut BoolReader<'_>,
    coefficients: &mut [i32],
    cache: &mut TokenCache,
    model: Model,
    coder: &mut Coder<'_>,
    context: (usize, Dequantizer),
) -> Extent {
    let (initial, dequantizer) = context;
    let setup = ReadSetup::new(model, dequantizer);
    read_prepared(reader, coefficients, cache, &setup, coder, initial)
}

/// Read one transform with block-plane invariants already resolved.
/// Reader state is still committed once per transform, exactly as in `read`.
#[allow(
    clippy::inline_always,
    reason = "keep the token loop with its caller so prepared plane invariants can stay in registers"
)]
#[inline(always)]
pub(crate) fn read_prepared(
    reader: &mut BoolReader<'_>,
    coefficients: &mut [i32],
    cache: &mut TokenCache,
    setup: &ReadSetup,
    coder: &mut Coder<'_>,
    initial: usize,
) -> Extent {
    // The token loop works on a local copy of the reader, written back once
    // (libvpx `decode_coefs` keeps `value`, `count` and `range` in locals):
    // through the reference, every read reloaded and stored its state.
    let mut local = reader.clone();
    let extent = read_tokens(&mut local, coefficients, cache, setup, coder, initial);
    *reader = local;
    extent
}

#[allow(
    clippy::inline_always,
    reason = "inlined into `read`, the local reader copy stays in registers"
)]
#[inline(always)]
fn read_tokens(
    reader: &mut BoolReader<'_>,
    coefficients: &mut [i32],
    cache: &mut TokenCache,
    setup: &ReadSetup,
    coder: &mut Coder<'_>,
    initial: usize,
) -> Extent {
    let ReadSetup {
        order,
        bands,
        row_shift,
        column_mask,
        dequantizer,
    } = setup;
    let (mut rows, mut columns) = (0, 0);
    let cache = &mut cache.0;
    let mut needs_eob = true;
    let mut step = dequantizer.dc;
    let mut ctx = initial;
    let positions = order
        .scan
        .iter()
        .zip(*bands)
        .zip(order.neighbors.as_chunks::<2>().0);
    for (c, ((&position, &band), neighbors)) in positions.enumerate() {
        if c > 0 {
            // `get_coef_context`: the two already-coded neighbours.
            let a = cache[usize::from(neighbors[0])];
            let b = cache[usize::from(neighbors[1])];
            ctx = usize::from((1 + a + b) >> 1);
        }
        let band = usize::from(band);
        let probs = &coder.probs[band][ctx];
        let counts = &mut coder.counts[band][ctx];
        if needs_eob {
            counts.eob_branch += 1;
            if !reader.read(probs[0]) {
                counts.tokens[EOB_MODEL_TOKEN] += 1;
                return Extent {
                    eob: c,
                    rows,
                    columns,
                };
            }
        }
        needs_eob = reader.read(probs[1]);
        let position = usize::from(position);
        if needs_eob {
            let more = reader.read(probs[2]);
            counts.tokens[1 + usize::from(more)] += 1;
            // At most `MAX_LEVEL`: the largest category's 14 extra bits.
            let magnitude = if more {
                read_magnitude(reader, probs[2])
            } else {
                1
            };
            let value = (magnitude * step) >> dequantizer.shift;
            coefficients[position] = if reader.read(128) { -value } else { value };
            cache[position] = energy(magnitude);
            rows = rows.max((position >> *row_shift) + 1);
            columns = columns.max((position & *column_mask) + 1);
        } else {
            counts.tokens[0] += 1;
            cache[position] = 0;
        }
        step = dequantizer.ac;
    }
    Extent {
        eob: order.scan.len(),
        rows,
        columns,
    }
}

/// Zero the coefficients [`read`] may have set for an end of block `eob`.
pub(crate) fn clear(coefficients: &mut [i32], model: Model, eob: usize) {
    for &position in &scan_order(model.tx, model.ty).scan[..eob] {
        coefficients[usize::from(position)] = 0;
    }
}

#[path = "coeff_trellis.rs"]
pub(crate) mod trellis;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaving_drops_only_isolated_trailing_ones() {
        let scan = scan_order(TxSize::Tx8, TxType::DctDct).scan;
        let at = |c: usize| usize::from(scan[c]);
        let mut levels = [0; 64];
        levels[at(0)] = 3;
        levels[at(2)] = 1;
        levels[at(20)] = -1;
        levels[at(40)] = 1;
        assert!(shave_isolated_ones(
            &mut levels,
            TxSize::Tx8,
            TxType::DctDct,
            8
        ));
        // Both far ±1 levels go; the one 2 positions after the DC stays.
        assert_eq!((levels[at(40)], levels[at(20)], levels[at(2)]), (0, 0, 1));
        let mut lone = [0; 64];
        lone[at(30)] = 1;
        assert!(shave_isolated_ones(
            &mut lone,
            TxSize::Tx8,
            TxType::DctDct,
            8
        ));
        assert_eq!(lone[at(30)], 1, "the first level is always kept");
    }

    #[test]
    fn every_size_round_trips_sparse_and_extreme_levels() {
        for tx in [TxSize::Tx4, TxSize::Tx8, TxSize::Tx16, TxSize::Tx32] {
            for ty in [TxType::DctDct, TxType::AdstDct, TxType::DctAdst] {
                for (plane, inter) in [(PlaneType::Y, false), (PlaneType::Uv, true)] {
                    let model = Model {
                        tx,
                        ty,
                        plane,
                        inter,
                    };
                    let area = tx.area();
                    let cases: Vec<Vec<i32>> = vec![
                        vec![0; area],
                        (0..area)
                            .map(|i| [0, 1, -2, 5, -19, 67, -MAX_LEVEL][i % 7])
                            .collect(),
                        (0..area).map(|i| i32::from(i == area - 1) * 3).collect(),
                    ];
                    for initial in 0..3 {
                        let mut writer = crate::recorder::TileRecorder::default();
                        let (context, mut counts) = (FrameContext::default(), Counts::default());
                        let eobs: Vec<usize> = cases
                            .iter()
                            .map(|l| {
                                let mut coder = Coder::new(model, &context, &mut counts);
                                write(&mut writer, l, model, &mut coder, initial)
                            })
                            .collect();
                        let packet = writer.finish(
                            &crate::recorder::slot_probs(&context.coef),
                            [0; crate::segmentation::TREE_PROBS],
                        );
                        let mut reader = BoolReader::new(&packet).unwrap();
                        let mut decoded = vec![0; area];
                        let mut cache = TokenCache::new();
                        for (levels, eob) in cases.iter().zip(eobs) {
                            let mut coder = Coder::new(model, &context, &mut counts);
                            let extent = read(
                                &mut reader,
                                &mut decoded,
                                &mut cache,
                                model,
                                &mut coder,
                                (
                                    initial,
                                    Dequantizer {
                                        dc: 1,
                                        ac: 1,
                                        shift: 0,
                                    },
                                ),
                            );
                            assert_eq!(extent.eob, eob);
                            let side = tx.side();
                            let rows = levels
                                .iter()
                                .rposition(|&level| level != 0)
                                .map_or(0, |last| last / side + 1);
                            assert_eq!(extent.rows, rows);
                            assert_eq!(&decoded, levels);
                            clear(&mut decoded, model, eob);
                            assert!(decoded.iter().all(|&l| l == 0));
                        }
                    }
                }
            }
        }
    }
}
