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

pub(crate) trait Bins {
    /// Whether the bins are read (decoding): the walk then skips what only
    /// writing or pricing needs, finding the coded values in the source.
    const READS: bool = false;
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
/// Side of the coefficient groups of blocks at least 4 a side.
const GROUP_SIDE: usize = 4;
/// Coefficient groups of the largest (32x32) transform block.
const MAX_GROUPS: usize = 64;

/// Scan positions of one coefficient group, in coding order, without
/// allocating (a group holds at most `GROUP_SIZE`).
#[derive(Default)]
struct GroupPositions {
    positions: [usize; GROUP_SIZE],
    len: usize,
}
impl GroupPositions {
    fn push(&mut self, position: usize) {
        self.positions[self.len] = position;
        self.len += 1;
    }
    fn is_empty(&self) -> bool {
        self.len == 0
    }
    fn as_slice(&self) -> &[usize] {
        &self.positions[..self.len]
    }
}

const GROUP_START: [usize; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];
fn position_group(pos: usize) -> usize {
    GROUP_START.iter().rposition(|&p| p <= pos).unwrap()
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
/// The context (index into `last_x` / `last_y`) of bin `bin` of a
/// last-position prefix of a transform block `side` samples along that
/// coordinate (§9.3.4.2.4: luma `offsetY[log2 − 1]`, 64-point sides too).
fn last_prefix_context(bin: usize, side: usize, ch: usize) -> usize {
    let log2 = side.ilog2() as usize;
    let (offset, shift) = if ch == 0 {
        ([0, 0, 0, 3, 6, 10, 15][log2], (log2 + 1) >> 2)
    } else {
        (0, (side >> 3).min(2))
    };
    offset + (bin >> shift)
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
    let mut group = 0;
    while group < largest && io.bin(&mut c[last_prefix_context(group, side, ch)], target > group)? {
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
    let prefix: f32 = (0..target.min(largest))
        .map(|bin| contexts[last_prefix_context(bin, side, ch)].bits(true))
        .sum();
    let stop = if target < largest {
        contexts[last_prefix_context(target, side, ch)].bits(false)
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

/// A neighbour's magnitude as the first pass knows it: its flags imply at
/// most 4 or 5 (`sumAbsPass1`, §9.3.4.2.8).
pub(crate) fn pass_one_magnitude(magnitude: u32) -> u32 {
    magnitude.min(4 + (magnitude & 1))
}

fn contexts<const AREA: usize>(
    values: &Levels<AREA>,
    pos: usize,
    width: usize,
    ch: usize,
) -> (usize, usize) {
    let (mut sum, mut count) = (0, 0);
    for a in values.template(pos) {
        sum += pass_one_magnitude(a);
        count += u32::from(a != 0);
    }
    template_contexts(sum, count, pos % width + pos / width, ch)
}

/// The `sig_coeff_flag` and greater-than/parity context offsets of a
/// position on anti-diagonal `diag` whose template holds `count` nonzero
/// levels summing to `sum` in first-pass magnitudes (§9.3.4.2.8).
pub(crate) fn template_contexts(sum: u32, count: u32, diag: usize, ch: usize) -> (usize, usize) {
    let sig = ((sum + 1) >> 1).min(3) as usize
        + if diag < 2 { 4 } else { 0 }
        + if ch == 0 && diag < 5 { 4 } else { 0 };
    let level = (sum - count).min(4) as usize
        + 1
        + if diag == 0 {
            if ch == 0 { 15 } else { 5 }
        } else if ch == 0 {
            if diag < 3 {
                10
            } else if diag < 10 {
                5
            } else {
                0
            }
        } else {
            0
        };
    (sig, level)
}
fn rice<const AREA: usize>(values: &Levels<AREA>, pos: usize, base: u32) -> u32 {
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

/// The scan index of `source`'s last nonzero level (0 when none): the last
/// coefficient group holding one, found from its four raster rows (much
/// cheaper than walking a sparse block's scan backwards), then its last
/// nonzero position.
fn last_significant(source: &[i32], (width, height): (usize, usize), scan: &[usize]) -> usize {
    let groups = crate::scan::rect_groups(width, height);
    let found = if width < GROUP_SIDE || height < GROUP_SIDE {
        // Intra sub-partitions 1 or 2 wide: a few groups, from the scan.
        scan.as_chunks::<GROUP_SIZE>()
            .0
            .iter()
            .rposition(|group| group.iter().any(|&p| source[p] != 0))
    } else {
        let columns = width / GROUP_SIDE;
        // Branch-free per group: the OR of its sixteen levels.
        let coded = |group: usize| {
            let (x, y) = (group % columns * GROUP_SIDE, group / columns * GROUP_SIDE);
            (y..y + GROUP_SIDE)
                .flat_map(|row| &source[row * width + x..][..GROUP_SIDE])
                .fold(0, |any, &v| any | v)
                != 0
        };
        (0..groups.len()).rev().find(|&g| coded(groups[g]))
    };
    let Some(g) = found else {
        return 0;
    };
    let first = g * GROUP_SIZE;
    (first..first + GROUP_SIZE)
        .rev()
        .find(|&i| source[scan[i]] != 0)
        .expect("a coded group holds a nonzero level")
}

/// The last position of `source`, a coded region `width` x `height` of a
/// transform block of `sides` (§9.3.4.2.4: each coordinate's prefix
/// contexts follow its own side of the whole block).
fn last_position<B: Bins>(
    io: &mut B,
    c: &mut CoeffContexts,
    source: &[i32],
    (width, height): (usize, usize),
    (ch, sides): (usize, (usize, usize)),
    scan: &[usize],
) -> Result<usize> {
    // A reader learns the position from the bins; the value is unused.
    let last = if B::READS {
        0
    } else {
        last_significant(source, (width, height), scan)
    };
    let (x, y) = (scan[last] % width, scan[last] / width);
    let gx = last_prefix(io, &mut c.last_x[ch], x, sides.0, ch)?;
    let gy = last_prefix(io, &mut c.last_y[ch], y, sides.1, ch)?;
    let x = last_suffix(io, gx, x)?;
    let y = last_suffix(io, gy, y)?;
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
fn signs<const AREA: usize>(
    io: &mut impl Bins,
    source: &[i32],
    values: &mut Levels<AREA>,
    scan: &[usize],
    nonzero: &[usize],
    hiding: bool,
) -> Result<()> {
    let (Some(&last_sig), Some(&first_sig)) = (nonzero.first(), nonzero.last()) else {
        return Ok(());
    };
    let sign_hidden = hiding && last_sig - first_sig > SIGN_HIDING_THRESHOLD;
    let mut sum = 0u32;
    for (i, &sp) in nonzero.iter().enumerate() {
        let p = scan[sp];
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

/// Read (or write or price, by `io`) the levels `source` of a `side`-square
/// block of channel `ch` (§7.3.11.11); returns the levels as coded.
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
    walk_sized(io, c, source, (ch, size), Some(&mut levels))?;
    Ok(levels)
}

/// [`code`] of an encoder's own levels, without collecting them: writing
/// them, or pricing them (`BitCost`) in the RD searches.
pub(crate) fn emit(
    io: &mut impl Bins,
    c: &mut CoeffContexts,
    levels: &[i32],
    side: usize,
    ch: usize,
) {
    emit_rect(io, c, levels, (side, side), ch);
}

/// [`emit`] of a `width` x `height` block.
pub(crate) fn emit_rect(
    io: &mut impl Bins,
    c: &mut CoeffContexts,
    levels: &[i32],
    size: (usize, usize),
    ch: usize,
) {
    walk_sized(io, c, levels, (ch, size), None).expect("an encoder's levels are valid");
}

/// [`walk`] of the coded region of channel `ch`'s transform block of
/// `size` on the smallest grid holding it (zeroing a 32x32 grid for every
/// 4x4 block cost more than the walk); the levels as coded into `levels`
/// when given.
fn walk_sized(
    io: &mut impl Bins,
    c: &mut CoeffContexts,
    source: &[i32],
    (ch, size): (usize, (usize, usize)),
    levels: Option<&mut Vec<i32>>,
) -> Result<()> {
    if !is_coded_size(size.0, size.1) {
        return Err(DecodeError::Unsupported("VVC transform size"));
    }
    let coded = (coded_side(size.0), coded_side(size.1));
    let block = (ch, size);
    // Coded sides are powers of two up to `MAX_SIDE`.
    match coded.0.max(coded.1) {
        1..=4 => {
            walk::<{ padded_area(4) }, _>(io, c, source, coded, block).map(|v| v.collect(levels))
        }
        5..=8 => {
            walk::<{ padded_area(8) }, _>(io, c, source, coded, block).map(|v| v.collect(levels))
        }
        9..=16 => {
            walk::<{ padded_area(16) }, _>(io, c, source, coded, block).map(|v| v.collect(levels))
        }
        _ => walk::<{ padded_area(MAX_SIDE) }, _>(io, c, source, coded, block)
            .map(|v| v.collect(levels)),
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

/// The syntax walk of [`code`] on a padded grid of `AREA` samples: the
/// coded region `width` x `height` of channel `ch`'s transform block of
/// `sides`.
fn walk<const AREA: usize, B: Bins>(
    io: &mut B,
    c: &mut CoeffContexts,
    source: &[i32],
    (width, height): (usize, usize),
    (ch, sides): (usize, (usize, usize)),
) -> Result<Levels<AREA>> {
    let scan = crate::scan::rect_positions(width, height);
    let last = last_position(io, c, source, (width, height), (ch, sides), scan)?;
    let mut values = Levels::<AREA>::new(width, height);
    let groups = crate::scan::rect_groups(width, height);
    let (columns, rows) = crate::scan::group_grid(width, height);
    let mut significant = [false; MAX_GROUPS];
    let mut budget = context_bin_budget(width * height);
    // §7.3.11.11 `QState`, from 0 at the last position; a group skipped by
    // its `sb_coded_flag` leaves it unchanged (sixteen zeros map every
    // state to itself).
    let mut state = 0u8;
    let dependent = c.dependent;
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
            (gp % columns + 1 < columns && significant[gp + 1])
                || (gp / columns + 1 < rows && significant[gp + columns]),
        );
        let present = !B::READS && scan[first..=end].iter().any(|&p| source[p] != 0);
        if g != 0 && g != last / GROUP_SIZE && !io.bin(&mut c.group[ch][ctx], present)? {
            continue;
        }
        significant[gp] = true;
        let mut nonzero = GroupPositions::default();
        let mut regular = GroupPositions::default();
        let mut bypass = GroupPositions::default();
        for sp in (first..=end).rev() {
            let p = scan[sp];
            if budget < MIN_REGULAR_BUDGET {
                bypass.push(sp);
                continue;
            }
            let (sig_ctx, mut level_ctx) = contexts(&values, p, width, ch);
            let inferred = sp == last || (sp == first && g != 0 && nonzero.is_empty());
            let sig = if inferred {
                true
            } else {
                budget -= 1;
                io.bin(c.significance(ch, state, sig_ctx), source[p] != 0)?
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
            if io.bin(&mut c.gt1[ch][level_ctx], abs > 1)? {
                let parity = io.bin(&mut c.parity[ch][level_ctx], abs & 1 != 0)?;
                let gt2 = io.bin(&mut c.gt2[ch][level_ctx], abs > 3)?;
                budget -= 2;
                level = 2 + u32::from(parity) + 2 * u32::from(gt2);
            }
            values.set(p, level as i32);
            step(&mut state, level);
            regular.push(p);
            nonzero.push(sp);
        }
        for &p in regular.as_slice() {
            if values.get(p) >= 4 {
                let rice_parameter = rice(&values, p, 4);
                let rem = remainder(
                    io,
                    source[p].unsigned_abs().saturating_sub(4) >> 1,
                    rice_parameter,
                )?;
                values.set(p, values.get(p) + 2 * rem as i32);
            }
        }
        for &sp in bypass.as_slice() {
            let p = scan[sp];
            let level = bypass_level(io, source[p].unsigned_abs(), rice(&values, p, 0), state)?;
            values.set(p, level as i32);
            step(&mut state, level);
            if level != 0 {
                nonzero.push(sp);
            }
        }
        signs(
            io,
            source,
            &mut values,
            scan,
            nonzero.as_slice(),
            !dependent,
        )?;
    }
    Ok(values)
}

/// A private context snapshot lets the shared planner price a candidate without
/// changing the entropy state of the picture being written.
#[derive(Default)]
pub(crate) struct BitCost(pub f32);
impl Bins for BitCost {
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
            let mut coded = BitCost::default();
            let mut contexts = CoeffContexts::new(37, true);
            assert_eq!(
                code(&mut coded, &mut contexts, &source, 4, 0).unwrap(),
                source
            );
            let mut writer = CabacWriter::new();
            let mut contexts = CoeffContexts::new(37, true);
            code(&mut writer, &mut contexts, &source, 4, 0).unwrap();
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
                        assert_eq!(
                            code(&mut writer, &mut contexts, &source, side, ch).unwrap(),
                            source
                        );
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
