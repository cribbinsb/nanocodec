//! H.266 §9.3 grouped diagonal AC/DC coefficient coding (§7.3.11.11):
//! sign data hiding, or the dependent quantisation state machine when the
//! slice uses it (`CoeffContexts::dependent`).
//!
//! Context derivation and binarization follow `VVenC` (BSD-3-Clause-Clear;
//! `THIRD_PARTY.md`). Encoder and bounded decoder share the syntax walk.
use crate::{
    cabac::CabacWriter, coeff_contexts::CoeffContexts, contexts::Context, decoder_cabac::Reader,
};
use nc_common::decode::{DecodeError, Result};
use nc_dsp::Dsp;

pub(crate) trait Bins {
    /// Whether the bins are read (decoding): the walk then skips what only
    /// writing or pricing needs, finding the coded values in the source.
    const READS: bool = false;
    /// Pricing can read already-known magnitudes without reconstructing levels.
    const PRICES: bool = false;
    fn bin(&mut self, c: &mut Context, value: bool) -> Result<bool>;
    fn bypass(&mut self, value: u32, bits: u32) -> Result<u32>;
}
impl Bins for CabacWriter {
    fn bin(&mut self, c: &mut Context, value: bool) -> Result<bool> {
        self.encode_bin(c, value);
        Ok(value)
    }
    fn bypass(&mut self, value: u32, bits: u32) -> Result<u32> {
        self.encode_bypass(value, bits);
        Ok(value)
    }
}
impl Bins for Reader<'_> {
    const READS: bool = true;
    fn bin(&mut self, c: &mut Context, _: bool) -> Result<bool> {
        self.bin(c)
    }
    fn bypass(&mut self, _: u32, bits: u32) -> Result<u32> {
        self.bypass_bits(bits)
    }
}

/// Coefficients of one luma coefficient group (§7.3.11.11).
use crate::scan::GROUP_SIZE;
/// Coefficient groups of the largest (32x32) transform block: one bit each
/// of the walk's coded-group mask.
const MAX_GROUPS: usize = 64;
const _: () = assert!(MAX_GROUPS <= u64::BITS as usize);

/// Scan or raster positions of one coefficient group, in coding order,
/// without allocating (a group holds at most `GROUP_SIZE`; a coded region
/// holds at most `MAX_SIDE` squared positions, so each fits 16 bits).
#[derive(Default)]
struct GroupPositions {
    positions: [u16; GROUP_SIZE],
    len: usize,
}
impl GroupPositions {
    fn push(&mut self, position: usize) {
        debug_assert!(position < MAX_SIDE * MAX_SIDE);
        self.positions[self.len] = position as u16;
        self.len += 1;
    }
    fn is_empty(&self) -> bool {
        self.len == 0
    }
    fn as_slice(&self) -> &[u16] {
        &self.positions[..self.len]
    }
}

const GROUP_START: [usize; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];
/// The last-position prefix group ([`GROUP_START`]) of each coordinate of a
/// coded side, looked up rather than searched for every priced block.
const POSITION_GROUPS: [u8; MAX_SIDE] = {
    let mut groups = [0; MAX_SIDE];
    let mut pos = 0;
    while pos < MAX_SIDE {
        let mut group = GROUP_START.len() - 1;
        while GROUP_START[group] > pos {
            group -= 1;
        }
        groups[pos] = group as u8;
        pos += 1;
    }
    groups
};
/// The last-position prefix group of coordinate `pos` (below [`MAX_SIDE`]).
fn position_group(pos: usize) -> usize {
    usize::from(POSITION_GROUPS[pos])
}

/// Context-coded bins one transform block may spend: H.266 caps them at
/// 1.75 times its sample count (§7.3.11.11 `remBinsPass1`).
pub(crate) const fn context_bin_budget(samples: usize) -> usize {
    samples * 28 / 16
}

/// A position's flags are context coded only while at least this many bins
/// of the budget remain (its significance, greater-than-one, parity and
/// greater-than-three flags); later positions are coded in bypass.
pub(crate) const MIN_REGULAR_BUDGET: usize = 4;

/// A group hides its first level's sign when its last and first nonzero
/// scan positions are more than this apart (§7.3.11.11 `signHidden`).
pub(crate) const SIGN_HIDING_THRESHOLD: usize = 3;
/// The contexts (indices into `last_x` / `last_y`) of the bins of a
/// last-position prefix of a transform block `side` samples along that
/// coordinate (§9.3.4.2.4: luma `offsetY[log2 − 1]`, 64-point sides too),
/// derived once per coordinate rather than for every bin.
#[derive(Clone, Copy)]
struct LastPrefixContexts {
    offset: usize,
    shift: usize,
}
impl LastPrefixContexts {
    fn new(side: usize, ch: usize) -> Self {
        let log2 = side.ilog2() as usize;
        let (offset, shift) = if ch == 0 {
            ([0, 0, 0, 3, 6, 10, 15][log2], (log2 + 1) >> 2)
        } else {
            (0, (side >> 3).min(2))
        };
        Self { offset, shift }
    }
    /// The context of bin `bin`.
    fn context(self, bin: usize) -> usize {
        self.offset + (bin >> self.shift)
    }
}

/// The coded side (§7.3.11.11 `log2ZoTbWidth`: a DCT-II keeps its 32
/// lowest frequencies) of a transform block side.
fn coded_side(side: usize) -> usize {
    side.min(MAX_SIDE)
}

/// The last-position prefix of `value` along a transform block side of
/// `side` samples: as many bins as its coded side needs (`cMax`), with
/// contexts by the whole side.
fn last_prefix(
    io: &mut impl Bins,
    c: &mut [Context],
    value: usize,
    side: usize,
    ch: usize,
) -> Result<usize> {
    let target = position_group(value);
    let largest = position_group(coded_side(side) - 1);
    let contexts = LastPrefixContexts::new(side, ch);
    let mut group = 0;
    while group < largest && io.bin(&mut c[contexts.context(group)], target > group)? {
        group += 1;
    }
    Ok(group)
}

/// Bits of coding `value` as one coordinate (`x` with `c.last_x`, `y`
/// with `c.last_y`) of a block's last significant position along a
/// transform block side of `side` samples, from the context states as they
/// stand (no adaptation between the bins).
pub(crate) fn last_coordinate_bits(
    contexts: &[Context],
    value: usize,
    side: usize,
    ch: usize,
) -> f32 {
    let largest = position_group(coded_side(side) - 1);
    let target = position_group(value);
    let prefix_contexts = LastPrefixContexts::new(side, ch);
    let prefix: f32 = (0..target.min(largest))
        .map(|bin| contexts[prefix_contexts.context(bin)].bits(true))
        .sum();
    let stop = if target < largest {
        contexts[prefix_contexts.context(target)].bits(false)
    } else {
        0.
    };
    let suffix = if target > 3 { (target - 2) >> 1 } else { 0 };
    prefix + stop + suffix as f32
}
fn last_suffix(io: &mut impl Bins, group: usize, value: usize) -> Result<usize> {
    Ok(GROUP_START[group]
        + if group > 3 {
            io.bypass(
                value.saturating_sub(GROUP_START[group]) as u32,
                ((group - 2) >> 1) as u32,
            )? as usize
        } else {
            0
        })
}
/// The context template of a position (§9.3.4.2.8): the two positions to
/// its right, the two below and the one diagonally below-right, all later in
/// scan order and so coded before it.
pub(crate) const TEMPLATE: [(usize, usize); 5] = [(1, 0), (2, 0), (1, 1), (0, 1), (0, 2)];

/// Largest coded side of a transform block, and the side of the largest
/// one (a 64-point DCT-II codes 32 frequencies).
const MAX_SIDE: usize = crate::transform_tree::MAX_CODED_SIDE;
const LARGEST_SIDE: usize = 2 * MAX_SIDE;
/// Zero columns right of, and rows below, a block in [`Levels`]: the reach
/// of [`TEMPLATE`].
const PAD: usize = 2;

/// The magnitudes (in the end, the signed levels) decoded so far of a block,
/// on a grid `PAD` wider and taller than the block whose padding stays zero: every
/// position's [`TEMPLATE`] lies inside it, so it is read without bounds
/// tests. `AREA` is the padded grid's area (one instance per block size).
struct Levels<const AREA: usize> {
    values: [i32; AREA],
    /// Row stride, log2 of the block width, and its height.
    stride: usize,
    log2: u32,
    height: usize,
}

impl<const AREA: usize> Levels<AREA> {
    fn new(width: usize, height: usize) -> Self {
        debug_assert!((width + PAD) * (height + PAD) <= AREA);
        Self {
            values: [0; AREA],
            stride: width + PAD,
            log2: width.ilog2(),
            height,
        }
    }

    /// The grid index of raster position `pos` of the block.
    fn index(&self, pos: usize) -> usize {
        (pos >> self.log2) * self.stride + (pos & ((1 << self.log2) - 1))
    }
}

/// Storage needed by the shared syntax walk. Pricing has no reconstructed output.
trait LevelValues {
    fn get(&self, pos: usize) -> i32;
    fn set(&mut self, pos: usize, value: i32);
    fn template(&self, pos: usize) -> [u32; TEMPLATE.len()];
    /// The sum of `pos`'s [`TEMPLATE`] first-pass magnitudes
    /// ([`pass_one_magnitude`]) and how many of them are nonzero.
    fn pass_one_template(&self, pos: usize) -> (u32, u32) {
        self.template(pos).iter().fold((0, 0), |(sum, count), &a| {
            (sum + pass_one_magnitude(a), count + u32::from(a != 0))
        })
    }
}

impl<const AREA: usize> LevelValues for Levels<AREA> {
    fn get(&self, pos: usize) -> i32 {
        self.values[self.index(pos)]
    }

    fn set(&mut self, pos: usize, value: i32) {
        let index = self.index(pos);
        self.values[index] = value;
    }

    /// The magnitudes at `pos`'s [`TEMPLATE`] positions (zero outside).
    fn template(&self, pos: usize) -> [u32; TEMPLATE.len()] {
        let index = self.index(pos);
        TEMPLATE.map(|(dx, dy)| self.values[index + dy * self.stride + dx].unsigned_abs())
    }
}

impl<const AREA: usize> Levels<AREA> {
    /// The block's levels in raster order into `levels`, when given.
    fn collect(&self, levels: Option<&mut Vec<i32>>) {
        let Some(levels) = levels else {
            return;
        };
        let width = 1 << self.log2;
        levels.clear();
        levels.reserve_exact(width * self.height);
        for row in self.values.chunks_exact(self.stride).take(self.height) {
            levels.extend_from_slice(&row[..width]);
        }
    }
}

/// A position's first-pass magnitude (at most 5) in the low bits of a
/// [`SourceLevels::first_pass`] entry, and whether it is nonzero above
/// them: the five template entries then add up to both template sums at
/// once (at most 25 below this shift, 5 above it: a byte holds both).
const NONZERO_SHIFT: u32 = 5;
/// The low bits of a summed [`SourceLevels::first_pass`] template.
const PASS_ONE_MASK: u32 = (1 << NONZERO_SHIFT) - 1;

/// Known encoder magnitudes, for pricing. Every template neighbour is later
/// in the grouped scan, so the walk has priced it before any position reads
/// it: [`LevelValues::set`] records its first-pass magnitude and nonzero
/// flag, packed in one byte, on a grid `PAD` wider and taller than the block
/// whose padding, and every position not coded, stays zero. The grid is
/// sized to the block like [`Levels`] (36 bytes to clear for a 4x4) and
/// written once per coded position, rather than rebuilt from the source for
/// every coefficient group. Its row stride is fixed per grid size (`SIDE`,
/// the block's longer coded side, plus [`PAD`]), so a template's five
/// offsets are constants inside one bounds-checked span. Rice parameters,
/// needed only by levels above four and by bypass-coded positions, read the
/// source itself.
struct SourceLevels<'a, const SIDE: usize, const AREA: usize> {
    source: &'a [i32],
    width: usize,
    height: usize,
    log2: u32,
    /// First-pass magnitudes and nonzero flags, packed ([`NONZERO_SHIFT`]).
    first_pass: [u8; AREA],
}

impl<'a, const SIDE: usize, const AREA: usize> SourceLevels<'a, SIDE, AREA> {
    /// The grid's row stride.
    const STRIDE: usize = SIDE + PAD;
    /// The span from a position to its farthest [`TEMPLATE`] neighbour,
    /// inclusive: [`PAD`] rows down.
    const REACH: usize = PAD * Self::STRIDE + 1;

    fn new(source: &'a [i32], width: usize, height: usize) -> Self {
        const { assert!(AREA == padded_area(SIDE)) };
        debug_assert!(width <= SIDE && height <= SIDE);
        Self {
            source,
            width,
            height,
            log2: width.ilog2(),
            first_pass: [0; AREA],
        }
    }

    /// The grid index of raster position `pos` of the block.
    fn index(&self, pos: usize) -> usize {
        (pos >> self.log2) * Self::STRIDE + (pos & (self.width - 1))
    }
}

impl<const SIDE: usize, const AREA: usize> LevelValues for SourceLevels<'_, SIDE, AREA> {
    fn get(&self, pos: usize) -> i32 {
        self.source[pos].unsigned_abs() as i32
    }

    fn set(&mut self, pos: usize, value: i32) {
        let magnitude = value.unsigned_abs();
        let index = self.index(pos);
        self.first_pass[index] =
            (pass_one_magnitude(magnitude) | u32::from(magnitude != 0) << NONZERO_SHIFT) as u8;
    }

    fn template(&self, pos: usize) -> [u32; TEMPLATE.len()] {
        let (x, y) = (pos & (self.width - 1), pos >> self.log2);
        TEMPLATE.map(|(dx, dy)| {
            if x + dx < self.width && y + dy < self.height {
                self.source[pos + dy * self.width + dx].unsigned_abs()
            } else {
                0
            }
        })
    }

    fn pass_one_template(&self, pos: usize) -> (u32, u32) {
        let reach = &self.first_pass[self.index(pos)..][..Self::REACH];
        // At most 5 x 5 below the shift, 5 above it: no carry between them.
        let packed: u32 = TEMPLATE
            .iter()
            .map(|&(dx, dy)| u32::from(reach[dy * Self::STRIDE + dx]))
            .sum();
        (packed & PASS_ONE_MASK, packed >> NONZERO_SHIFT)
    }
}

/// A neighbour's magnitude as the first pass knows it: its flags imply at
/// most 4 or 5 (`sumAbsPass1`, §9.3.4.2.8).
pub(crate) fn pass_one_magnitude(magnitude: u32) -> u32 {
    magnitude.min(4 + (magnitude & 1))
}

/// [`template_contexts`] of raster position `pos` of a block `1 << log2`
/// wide.
fn contexts(values: &impl LevelValues, pos: usize, log2: u32, ch: usize) -> (usize, usize) {
    let (sum, count) = values.pass_one_template(pos);
    let diag = (pos & ((1 << log2) - 1)) + (pos >> log2);
    template_contexts(sum, count, diag, ch)
}

/// Anti-diagonals past which a position's context offsets no longer change.
pub(crate) const DIAGONAL_CLASSES: usize = 11;
/// The `sig_coeff_flag` and greater-than/parity context offsets of a
/// position by its anti-diagonal (to [`DIAGONAL_CLASSES`]), luma then
/// chroma (§9.3.4.2.8: luma `d < 2`, `d < 5`; `d == 0`, `d < 3`, `d < 10`).
#[rustfmt::skip]
pub(crate) const DIAGONAL_OFFSETS: [[(u8, u8); DIAGONAL_CLASSES]; 2] = [
    [(8, 15), (8, 10), (4, 10), (4, 5), (4, 5), (0, 5), (0, 5), (0, 5), (0, 5), (0, 5), (0, 0)],
    [(4, 5), (4, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
];

/// The `sig_coeff_flag` and greater-than/parity context offsets of a
/// position on anti-diagonal `diag` whose template holds `count` nonzero
/// levels summing to `sum` in first-pass magnitudes (§9.3.4.2.8).
pub(crate) fn template_contexts(sum: u32, count: u32, diag: usize, ch: usize) -> (usize, usize) {
    let (sig_offset, level_offset) =
        DIAGONAL_OFFSETS[usize::from(ch != 0)][diag.min(DIAGONAL_CLASSES - 1)];
    let sig = ((sum + 1) >> 1).min(3) as usize + usize::from(sig_offset);
    let level = (sum - count).min(4) as usize + 1 + usize::from(level_offset);
    (sig, level)
}
fn rice(values: &impl LevelValues, pos: usize, base: u32) -> u32 {
    rice_parameter(values.template(pos).iter().sum(), base)
}

/// The Rice parameter of a remainder whose template magnitudes sum to
/// `sum`, above a `base` the flags already implied (§9.3.3.2 `locSumAbs`).
pub(crate) fn rice_parameter(sum: u32, base: u32) -> u32 {
    match sum.saturating_sub(5 * base).min(31) {
        0..=6 => 0,
        7..=13 => 1,
        14..=27 => 2,
        _ => 3,
    }
}
/// Bits of [`remainder`] coding `value` with Rice parameter `rice` (all
/// bypass bins).
pub(crate) fn remainder_bits(value: u32, rice: u32) -> f32 {
    let mut cost = BitCost::default();
    remainder(&mut cost, value, rice).expect("pricing a remainder cannot fail");
    cost.0
}

/// Limited Rice/escape code, cutoff five and 15-bit transform dynamic range.
fn remainder(io: &mut impl Bins, value: u32, rice: u32) -> Result<u32> {
    const CUTOFF: u32 = 5;
    const MAX_PREFIX: u32 = 12; // 32 - cutoff - transform dynamic range (15).
    let quotient = value >> rice;
    let tail = quotient.saturating_sub(CUTOFF);
    let escape = (tail + 1).ilog2().min(MAX_PREFIX);
    let target = if quotient < CUTOFF {
        quotient
    } else {
        CUTOFF + escape
    };
    let mut prefix = 0;
    while prefix < CUTOFF + MAX_PREFIX && io.bypass(u32::from(prefix < target), 1)? != 0 {
        prefix += 1;
    }
    if prefix < CUTOFF {
        return Ok((prefix << rice) + io.bypass(value & ((1 << rice) - 1), rice)?);
    }
    let ext = prefix - CUTOFF;
    let base = (CUTOFF + ((1 << ext) - 1)) << rice;
    let bits = if ext == MAX_PREFIX { 15 } else { ext + rice };
    Ok(base + io.bypass(value.saturating_sub(base), bits)?)
}

/// The scan index of `source`'s last nonzero level (0 when none), a coded
/// region `width` x `height`: the shared end-of-block kernel
/// (`Dsp::end_of_block_wide`) over the grouped scan's table.
fn last_significant(dsp: &Dsp, source: &[i32], (width, height): (usize, usize)) -> usize {
    let table = crate::scan::rect_eob(width, height);
    let (end, _) = dsp.end_of_block_wide(&source[..table.len()], table);
    end.saturating_sub(1)
}

/// The last position (scan index `last` of a writer's levels; a reader's
/// from the bins) of a coded region `width` x `height` of a transform
/// block of `sides` (§9.3.4.2.4: each coordinate's prefix
/// contexts follow its own side of the whole block).
fn last_position<B: Bins>(
    io: &mut B,
    c: &mut CoeffContexts,
    (width, height): (usize, usize),
    (ch, sides): (usize, (usize, usize)),
    (scan, last): (&[usize], usize),
) -> Result<usize> {
    let (x, y) = (scan[last] % width, scan[last] / width);
    let gx = last_prefix(io, &mut c.last_x[ch], x, sides.0, ch)?;
    let gy = last_prefix(io, &mut c.last_y[ch], y, sides.1, ch)?;
    let x = last_suffix(io, gx, x)?;
    let y = last_suffix(io, gy, y)?;
    if !B::READS {
        // A writer or pricer coded its own last position.
        return Ok(last);
    }
    Ok(crate::scan::rect_indices(width, height)[y * width + x])
}

/// §7.3.11.11 `dec_abs_level` of a bypass-coded position: magnitude `a`
/// (the encoder's) coded around §9.3.3.11 `ZeroPos`, 1 (2 in quantiser
/// states 2 and 3) shifted by the Rice parameter `rice`.
fn bypass_level(io: &mut impl Bins, a: u32, rice: u32, state: u8) -> Result<u32> {
    let zero = (if state < 2 { 1 } else { 2 }) << rice;
    let value = if a == 0 {
        zero
    } else if a <= zero {
        a - 1
    } else {
        a
    };
    let value = remainder(io, value, rice)?;
    Ok(match value.cmp(&zero) {
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Less => value + 1,
        std::cmp::Ordering::Greater => value,
    })
}

/// §7.3.11.11 `QStateTransTable`: the next quantiser state after a level
/// of this parity.
const STATE_TRANSITIONS: [[u8; 2]; 4] = [[0, 2], [2, 0], [1, 3], [3, 1]];

/// The quantiser state after coding magnitude `level` in state `state`.
pub(crate) fn next_state(state: u8, level: u32) -> u8 {
    STATE_TRANSITIONS[usize::from(state)][(level & 1) as usize]
}

/// The group's sign bins, in reverse scan order (`nonzero`: scan indices,
/// from the group's last significant one). The group's first significant
/// position hides its sign in the parity of the group's level sum when the
/// group spans more than [`SIGN_HIDING_THRESHOLD`] positions (§7.3.11.11).
fn signs(
    io: &mut impl Bins,
    source: &[i32],
    values: &mut impl LevelValues,
    scan: &[usize],
    nonzero: &[u16],
    hiding: bool,
) -> Result<()> {
    let (Some(&last_sig), Some(&first_sig)) = (nonzero.first(), nonzero.last()) else {
        return Ok(());
    };
    let sign_hidden = hiding && usize::from(last_sig - first_sig) > SIGN_HIDING_THRESHOLD;
    let mut sum = 0u32;
    for (i, &sp) in nonzero.iter().enumerate() {
        let p = scan[usize::from(sp)];
        let magnitude = values.get(p);
        if magnitude > 32767 {
            return Err(DecodeError::Malformed("VVC coefficient range"));
        }
        sum += magnitude.unsigned_abs();
        let negative = if sign_hidden && i + 1 == nonzero.len() {
            sum % 2 == 1
        } else {
            io.bypass(u32::from(source[p] < 0), 1)? != 0
        };
        if negative {
            values.set(p, -magnitude);
        }
    }
    Ok(())
}

/// Read the levels of a `side`-square block of channel `ch` (§7.3.11.11)
/// into the result (`source` is unused by a reader); writers and pricers
/// code through [`emit`].
#[cfg(test)]
pub(crate) fn code(
    io: &mut impl Bins,
    c: &mut CoeffContexts,
    source: &[i32],
    side: usize,
    ch: usize,
) -> Result<Vec<i32>> {
    code_rect(io, c, source, (side, side), ch)
}

/// [`code`] of a `width` x `height` transform block (each side 4 to 64,
/// or a luma intra sub-partition 1 or 2 wide or high): `source` and the
/// result hold its coded region's levels (`transform_tree::coded_samples`).
pub(crate) fn code_rect(
    io: &mut impl Bins,
    c: &mut CoeffContexts,
    source: &[i32],
    size: (usize, usize),
    ch: usize,
) -> Result<Vec<i32>> {
    let mut levels = Vec::new();
    walk_sized(io, c, (source, None), (ch, size), Some(&mut levels))?;
    Ok(levels)
}

/// [`code`] of an encoder's own levels, without collecting them: writing
/// them, or pricing them (`BitCost`) in the RD searches.
pub(crate) fn emit(
    io: &mut impl Bins,
    c: &mut CoeffContexts,
    dsp: &Dsp,
    levels: &[i32],
    side: usize,
    ch: usize,
) {
    emit_rect(io, c, dsp, levels, (side, side), ch);
}

/// [`emit`] of a `width` x `height` block.
pub(crate) fn emit_rect(
    io: &mut impl Bins,
    c: &mut CoeffContexts,
    dsp: &Dsp,
    levels: &[i32],
    size: (usize, usize),
    ch: usize,
) {
    walk_sized(io, c, (levels, Some(dsp)), (ch, size), None)
        .expect("an encoder's levels are valid");
}

/// [`walk`] of the coded region of channel `ch`'s transform block of
/// `size` on the smallest grid holding it (zeroing a 32x32 grid for every
/// 4x4 block cost more than the walk); the levels as coded into `levels`
/// when given.
fn walk_sized<B: Bins>(
    io: &mut B,
    c: &mut CoeffContexts,
    (source, dsp): (&[i32], Option<&Dsp>),
    (ch, size): (usize, (usize, usize)),
    levels: Option<&mut Vec<i32>>,
) -> Result<()> {
    if !is_coded_size(size.0, size.1) {
        return Err(DecodeError::Unsupported("VVC transform size"));
    }
    let coded = (coded_side(size.0), coded_side(size.1));
    let block = (ch, size);
    // A reader learns the last position from the bins; a writer finds it
    // in its levels.
    let last = if let Some(dsp) = dsp {
        last_significant(dsp, source, coded)
    } else {
        debug_assert!(B::READS, "writers code through `emit`, with the Dsp");
        0
    };
    if B::PRICES && levels.is_none() {
        // Coded sides are powers of two up to `MAX_SIDE`.
        return match coded.0.max(coded.1) {
            1..=4 => walk_source::<4, { padded_area(4) }, _>(io, c, (source, last), coded, block),
            5..=8 => walk_source::<8, { padded_area(8) }, _>(io, c, (source, last), coded, block),
            9..=16 => {
                walk_source::<16, { padded_area(16) }, _>(io, c, (source, last), coded, block)
            }
            _ => walk_source::<MAX_SIDE, { padded_area(MAX_SIDE) }, _>(
                io,
                c,
                (source, last),
                coded,
                block,
            ),
        };
    }
    match coded.0.max(coded.1) {
        1..=4 => walk_grid::<{ padded_area(4) }, _>(io, c, (source, last), coded, block, levels),
        5..=8 => walk_grid::<{ padded_area(8) }, _>(io, c, (source, last), coded, block, levels),
        9..=16 => walk_grid::<{ padded_area(16) }, _>(io, c, (source, last), coded, block, levels),
        _ => walk_grid::<{ padded_area(MAX_SIDE) }, _>(io, c, (source, last), coded, block, levels),
    }
}

/// Whether a `width` x `height` transform block is one we code: sides 4
/// to 64, or the narrow luma blocks of intra sub-partitions (at least
/// sixteen samples).
fn is_coded_size(width: usize, height: usize) -> bool {
    let side = |n: usize| n.is_power_of_two() && n <= LARGEST_SIDE;
    side(width) && side(height) && width * height >= GROUP_SIZE
}

/// [`Levels`]' area for a `side`-square block.
const fn padded_area(side: usize) -> usize {
    (side + PAD) * (side + PAD)
}

/// The reconstructed grid used by writing, decoding, or a caller collecting levels.
fn walk_grid<const AREA: usize, B: Bins>(
    io: &mut B,
    c: &mut CoeffContexts,
    source: (&[i32], usize),
    coded: (usize, usize),
    block: (usize, (usize, usize)),
    levels: Option<&mut Vec<i32>>,
) -> Result<()> {
    let mut values = Levels::<AREA>::new(coded.0, coded.1);
    walk(io, c, source, coded, block, &mut values)?;
    values.collect(levels);
    Ok(())
}

/// The pricing walk over the encoder's own levels ([`SourceLevels`]).
fn walk_source<const SIDE: usize, const AREA: usize, B: Bins>(
    io: &mut B,
    c: &mut CoeffContexts,
    source: (&[i32], usize),
    coded: (usize, usize),
    block: (usize, (usize, usize)),
) -> Result<()> {
    let mut values = SourceLevels::<SIDE, AREA>::new(source.0, coded.0, coded.1);
    walk(io, c, source, coded, block, &mut values)
}

/// The syntax walk of [`code`] over reconstructed or already-known magnitudes: the
/// coded region `width` x `height` of channel `ch`'s transform block of
/// `sides`.
fn walk<B: Bins>(
    io: &mut B,
    c: &mut CoeffContexts,
    (source, last): (&[i32], usize),
    (width, height): (usize, usize),
    (ch, sides): (usize, (usize, usize)),
    values: &mut impl LevelValues,
) -> Result<()> {
    let scan = crate::scan::rect_positions(width, height);
    let last = last_position(io, c, (width, height), (ch, sides), (scan, last))?;
    let groups = crate::scan::rect_groups(width, height);
    let (columns, rows) = crate::scan::group_grid(width, height);
    // Bit `g` set: group `g` (raster order of the group grid) is coded.
    let mut significant = 0u64;
    let mut budget = context_bin_budget(width * height);
    // §7.3.11.11 `QState`, from 0 at the last position; a group skipped by
    // its `sb_coded_flag` leaves it unchanged (sixteen zeros map every
    // state to itself).
    let mut state = 0u8;
    let log2 = width.ilog2();
    let dependent = c.dependent;
    let mut c = c.channel(ch);
    let step = |state: &mut u8, level: u32| {
        if dependent {
            *state = next_state(*state, level);
        }
    };
    for g in (0..=last / GROUP_SIZE).rev() {
        let first = g * GROUP_SIZE;
        let end = (first + GROUP_SIZE - 1).min(last);
        let gp = groups[g];
        let ctx = usize::from(
            (gp % columns + 1 < columns && significant >> (gp + 1) & 1 != 0)
                || (gp / columns + 1 < rows && significant >> (gp + columns) & 1 != 0),
        );
        let present = !B::READS && scan[first..=end].iter().any(|&p| source[p] != 0);
        if g != 0 && g != last / GROUP_SIZE && !io.bin(&mut c.group[ctx], present)? {
            continue;
        }
        significant |= 1 << gp;
        let mut nonzero = GroupPositions::default();
        let mut bypass = GroupPositions::default();
        for sp in (first..=end).rev() {
            let p = scan[sp];
            if budget < MIN_REGULAR_BUDGET {
                bypass.push(sp);
                continue;
            }
            // The last level has inferred significance and level context 0;
            // pricing it need not derive either from its zero template.
            let (sig_ctx, mut level_ctx) = if B::PRICES && sp == last {
                (0, 0)
            } else {
                contexts(values, p, log2, ch)
            };
            let inferred = sp == last || (sp == first && g != 0 && nonzero.is_empty());
            let sig = if inferred {
                true
            } else {
                budget -= 1;
                io.bin(c.significance(state, sig_ctx), source[p] != 0)?
            };
            if !sig {
                step(&mut state, 0);
                continue;
            }
            if sp == last {
                level_ctx = 0;
            }
            let abs = source[p].unsigned_abs();
            budget -= 1;
            let mut level = 1;
            if io.bin(&mut c.gt1[level_ctx], abs > 1)? {
                let parity = io.bin(&mut c.parity[level_ctx], abs & 1 != 0)?;
                let gt2 = io.bin(&mut c.gt2[level_ctx], abs > 3)?;
                budget -= 2;
                level = 2 + u32::from(parity) + 2 * u32::from(gt2);
            }
            values.set(p, level as i32);
            step(&mut state, level);
            nonzero.push(sp);
        }
        // The nonzero levels so far are exactly the context-coded ones.
        for p in nonzero.as_slice().iter().map(|&sp| scan[usize::from(sp)]) {
            if values.get(p) >= 4 {
                let rice_parameter = rice(values, p, 4);
                let rem = remainder(
                    io,
                    source[p].unsigned_abs().saturating_sub(4) >> 1,
                    rice_parameter,
                )?;
                values.set(p, values.get(p) + 2 * rem as i32);
            }
        }
        for sp in bypass.as_slice().iter().map(|&sp| usize::from(sp)) {
            let p = scan[sp];
            let level = bypass_level(io, source[p].unsigned_abs(), rice(values, p, 0), state)?;
            values.set(p, level as i32);
            step(&mut state, level);
            if level != 0 {
                nonzero.push(sp);
            }
        }
        signs(io, source, values, scan, nonzero.as_slice(), !dependent)?;
    }
    Ok(())
}

/// A private context snapshot lets the shared planner price a candidate without
/// changing the entropy state of the picture being written.
#[derive(Default)]
pub(crate) struct BitCost(pub f32);
impl Bins for BitCost {
    const PRICES: bool = true;
    fn bin(&mut self, c: &mut Context, value: bool) -> Result<bool> {
        self.0 += c.bits(value);
        c.update(value);
        Ok(value)
    }
    fn bypass(&mut self, value: u32, bits: u32) -> Result<u32> {
        self.0 += bits as f32;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Give each sign-hiding group the parity its first level's sign needs
    /// by flipping that sign (any parity-consistent levels test the walk).
    fn hide_signs_by_flipping(levels: &mut [i32], side: usize) {
        for span in crate::scan::positions(side).as_chunks::<GROUP_SIZE>().0 {
            let nonzero: Vec<usize> = (0..GROUP_SIZE).filter(|&n| levels[span[n]] != 0).collect();
            let (Some(&first), Some(&last)) = (nonzero.first(), nonzero.last()) else {
                continue;
            };
            let sum: u32 = span.iter().map(|&p| levels[p].unsigned_abs()).sum();
            if last - first > SIGN_HIDING_THRESHOLD && (sum % 2 == 1) != (levels[span[first]] < 0) {
                levels[span[first]] = -levels[span[first]];
            }
        }
    }

    #[test]
    fn hidden_signs_follow_the_group_parity() {
        let positions = crate::scan::positions(4);
        for (levels_in_scan, expect_hidden) in [
            ([-1, 0, 0, 0, 2], true),  // sum 3 odd: first is negative
            ([1, 0, 0, 0, 1], true),   // sum 2 even: first is positive
            ([-1, 0, 0, 1, 0], false), // span 3: the sign is coded
        ] {
            let mut source = vec![0; 16];
            for (n, &level) in levels_in_scan.iter().enumerate() {
                source[positions[n]] = level;
            }
            let mut writer = CabacWriter::new();
            let mut contexts = CoeffContexts::new(37, true);
            emit(&mut writer, &mut contexts, &Dsp::new(), &source, 4, 0);
            writer.encode_terminate(true);
            let mut bits = writer.finish();
            bits.trailing_bits();
            let bytes = bits.into_bytes();
            let mut reader = Reader::new(&bytes).unwrap();
            let mut contexts = CoeffContexts::new(37, true);
            assert_eq!(
                code(&mut reader, &mut contexts, &[0; 16], 4, 0).unwrap(),
                source,
                "hidden {expect_hidden}"
            );
        }
    }

    #[test]
    fn sparse_dense_and_escape_levels_roundtrip() {
        for side in [4, 8, 16, 32] {
            for inter in [false, true] {
                for ch in [0, 1] {
                    for pattern in 0..4 {
                        let mut source = vec![0; side * side];
                        for (i, level) in source.iter_mut().enumerate() {
                            *level = match pattern {
                                0 if i == side * side - 1 => -32767,
                                1 if i.is_multiple_of(17) => 32767,
                                2 => (i as i32 % 11) - 5,
                                3 => {
                                    if i.is_multiple_of(2) {
                                        -32767
                                    } else {
                                        32767
                                    }
                                }
                                _ => 0,
                            };
                        }
                        hide_signs_by_flipping(&mut source, side);
                        let mut writer = CabacWriter::new();
                        let mut contexts = CoeffContexts::new(37, inter);
                        emit(&mut writer, &mut contexts, &Dsp::new(), &source, side, ch);
                        writer.encode_terminate(true);
                        let mut bits = writer.finish();
                        bits.trailing_bits();
                        let bytes = bits.into_bytes();
                        let mut reader = Reader::new(&bytes).unwrap();
                        let mut contexts = CoeffContexts::new(37, inter);
                        let decoded =
                            code(&mut reader, &mut contexts, &vec![0; side * side], side, ch)
                                .unwrap();
                        assert_eq!(
                            decoded, source,
                            "side={side} inter={inter} ch={ch} pattern={pattern}"
                        );
                        assert!(reader.terminate().unwrap());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod pricing_tests;
