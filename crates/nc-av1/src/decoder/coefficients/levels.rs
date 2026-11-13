//! The level pass of one transform block (AV1 §5.11.39 `coeffs()`, the
//! `coeff_base_eob` / `coeff_base` / `coeff_br` loop; contexts §8.3.2):
//! levels in reverse scan order, each from a context of the levels already
//! read around it, threaded into a forward-scan chain for the sign pass.
//!
//! A port of the level loop of dav1d 1.5.4 `src/recon_tmpl.c`
//! `decode_coefs` and `get_lo_ctx` (BSD-2-Clause, Copyright (c) 2018,
//! `VideoLAN` and dav1d authors; see `THIRD_PARTY.md`):
//! - the level buffer is unpadded for the 2-D class (index = storage
//!   position, two zero columns past the block) and transposed with a
//!   16-entry stride for the 1-D classes, so every context neighbour is a
//!   fixed offset; [`LevelBuffer`] says why the missing row padding is safe;
//! - each entry packs the level and the level saturated at 3, so one sum
//!   gives both context magnitudes;
//! - the DC coefficient is read after the loop, so the loop has no DC test;
//! - the 2-D context offset is libaom's per-size table indexed by position
//!   (`av1_nz_map_ctx_offset`), one load where dav1d's 5x5 corner table
//!   needs the row and column (fewer instructions and register spills);
//! - each coefficient's buffer entry holds its token and the storage
//!   position of the next nonzero coefficient in scan order, so the sign
//!   pass visits only nonzero coefficients without a separate list.
use super::super::transform::TxSize;
use super::{BR_CDF_SIZE, COEFF_BASE_RANGE, CoefficientCdfs, MAX_CODED, SIG_COEF_CONTEXTS};
use crate::decoder_range::{Cdf, SymbolReader};
use crate::tables::{
    DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X8, DEFAULT_SCAN_4X16, DEFAULT_SCAN_8X4, DEFAULT_SCAN_8X8,
    DEFAULT_SCAN_8X16, DEFAULT_SCAN_8X32, DEFAULT_SCAN_16X4, DEFAULT_SCAN_16X8, DEFAULT_SCAN_16X16,
    DEFAULT_SCAN_16X32, DEFAULT_SCAN_32X8, DEFAULT_SCAN_32X16, DEFAULT_SCAN_32X32,
    NZ_MAP_CTX_OFFSET_4X4, NZ_MAP_CTX_OFFSET_4X8, NZ_MAP_CTX_OFFSET_4X16, NZ_MAP_CTX_OFFSET_8X8,
    NZ_MAP_CTX_OFFSET_8X16, NZ_MAP_CTX_OFFSET_8X32, NZ_MAP_CTX_OFFSET_16X4,
    NZ_MAP_CTX_OFFSET_16X16, NZ_MAP_CTX_OFFSET_16X32, NZ_MAP_CTX_OFFSET_32X8,
    NZ_MAP_CTX_OFFSET_32X16, NZ_MAP_CTX_OFFSET_32X32, NZ_MAP_CTX_OFFSET_32X64,
    NZ_MAP_CTX_OFFSET_64X32,
};

/// A coefficient's buffer entry between the passes: its token (level up
/// to [`MAX_TOKEN`]) above [`TOKEN_SHIFT`], and below it the storage
/// position of the next nonzero coefficient in scan order (0: none).
pub(super) const TOKEN_SHIFT: u32 = 11;
pub(super) const LINK_MASK: i32 = (1 << TOKEN_SHIFT) - 1;
/// The largest token: base levels plus the whole `coeff_br` range. A
/// coefficient at this token continues with a Golomb remainder.
pub(super) const MAX_TOKEN: u32 = NUM_BASE_LEVELS + COEFF_BASE_RANGE as u32 + 1;
/// Levels coded by `coeff_base` / `coeff_base_eob` (`NUM_BASE_LEVELS`).
const NUM_BASE_LEVELS: u32 = 2;
/// `coeff_base` and `coeff_base_eob` alphabets.
const BASE_SYMBOLS: usize = 4;
const BASE_EOB_SYMBOLS: usize = 3;

/// A level entry: the level in bits 0-5 and the level saturated at 3 in
/// bits 6-7 (dav1d's `tok * 0x41` / `tok + (3 << 6)`). A sum of
/// entries is the saturated sum in units of 64 plus the plain sum.
const SATURATED_SHIFT: u32 = 6;
const PLAIN_MASK: u32 = (1 << SATURATED_SHIFT) - 1;
const BASE_NEIGHBOUR_MAX: u32 = 3;
/// `coeff_base` context from the packed sum `m` of its five neighbours:
/// `min((sat + 1) >> 1, 4)` is `(m + 64) >> 7`, or 4 above 512 (a sum of
/// at most 512 has a plain part below 64, so it cannot carry).
const BASE_SUM_SATURATED: u32 = 8 << SATURATED_SHIFT;
const BASE_SUM_ROUND: u32 = 1 << SATURATED_SHIFT;
const BASE_SUM_SHIFT: u32 = SATURATED_SHIFT + 1;
const BASE_MAG_MAX: usize = 4;
/// `coeff_br` magnitude term: `min((mag + 1) >> 1, 6)` of the plain sum.
const BR_MAG_LIMIT: u32 = 12;
const BR_MAG_MAX: usize = 6;
/// `coeff_br` context groups: DC, the low-frequency corner (2-D) or first
/// line (1-D), the rest.
const BR_LOW_GROUP: usize = 7;
const BR_HIGH_GROUP: usize = 14;
/// `coeff_base` contexts of the 1-D classes follow the 26 of the 2-D
/// class, in groups of five by distance along the transform's line
/// (1 and 2 or more).
const CONTEXTS_2D: usize = 26;
const CONTEXT_GROUP: usize = 5;
const CONTEXT_FAR: usize = 2 * CONTEXT_GROUP;
/// `coeff_base_eob` contexts by how far into the block the last
/// coefficient lies: within the first eighth, the first quarter, or beyond
/// (DC alone is context 0). A coded area is `16 << size_log2`, so its
/// eighth is `2 << size_log2` and its quarter `4 << size_log2`.
const EOB_EIGHTH: usize = 2;
const EOB_QUARTER: usize = 4;

/// The transposed stride of the 1-D classes (their coded sides are at most
/// 16).
const STRIDE_1D: usize = 16;
/// Zero lines past the block that contexts read: two beyond the last
/// column (2-D) or row of the transposed map (1-D).
const PAD_LINES: usize = 2;
/// Map indices of coefficients are below 1024 (2-D: the storage position;
/// 1-D: at most `15 * 16 + 15`) and strides at most 32. Masking both with
/// these (no-ops on real values) bounds every neighbour index below
/// [`LEVELS_LEN`], so the compiler drops the bounds checks.
const AT_MASK: usize = 1023;
const STRIDE_MASK: usize = 63;
/// Entries the buffer holds: room for any masked index plus twice any
/// masked stride.
const LEVELS_LEN: usize = 2048;
const _: () = assert!(AT_MASK + 2 * STRIDE_MASK < LEVELS_LEN);
const _: () = assert!(32 * (32 + PAD_LINES) <= LEVELS_LEN);

/// The level map of the block being read (dav1d `t->scratch.levels`).
///
/// No row padding: a neighbour below the last row reads the next column's
/// first rows (2-D), and one past the transposed line the next line's
/// start (1-D). Those positions lie on an earlier scan diagonal (2-D, as
/// the default scans are ordered by `row + column` and coded sides are at
/// least 4) or earlier in the raster (1-D, sides at most 16), so in reverse
/// scan order they are still zero when read (checked for every size in
/// the tests).
pub(super) struct LevelBuffer {
    levels: [u8; LEVELS_LEN],
}

impl LevelBuffer {
    pub(super) fn new() -> Self {
        Self {
            levels: [0; LEVELS_LEN],
        }
    }
    #[inline]
    fn set(&mut self, at: usize, entry: u8) {
        self.levels[at & AT_MASK] = entry;
    }
}

/// The three transform classes as const parameters, so each gets its own
/// loop (dav1d's `DECODE_COEFS_CLASS` expansions).
pub(super) const CLASS_2D: u8 = 0;
pub(super) const CLASS_H: u8 = 1;
pub(super) const CLASS_V: u8 = 2;

/// How scan indices map to storage positions and level entries for one
/// transform size and class.
#[derive(Clone, Copy)]
pub(super) struct Walk {
    /// The 2-D diagonal scan (unused by the 1-D classes).
    scan: &'static [u16],
    /// The 2-D context offset of each storage position.
    offsets: &'static Offsets,
    /// log2 of the coded height (2-D, H) or width (V): the split of a scan
    /// index or position into line and offset.
    shift: u32,
    /// log2 of the coded height, V only: storage position of a column.
    shift2: u32,
    /// Level map stride.
    stride: usize,
    /// Map entries [`Levels::read`] clears: the block plus its padding.
    clear: usize,
}

/// One coefficient's place: storage position `rc`, level map index `at`,
/// and for the 1-D classes its distance `y` along the transform's line
/// (dav1d's name; 0 for 2-D, whose contexts take the position instead).
#[derive(Clone, Copy)]
struct Place {
    rc: usize,
    at: usize,
    y: usize,
}

impl Walk {
    pub(super) fn new<const CLASS: u8>(tx: TxSize) -> Self {
        let width_log2 = u32::from(tx.coded_width_log2());
        let height_log2 = u32::from(tx.coded_height_log2());
        let (shift, stride, lines) = match CLASS {
            CLASS_2D => (height_log2, 1 << height_log2, 1 << width_log2),
            CLASS_H => (height_log2, STRIDE_1D, 1 << height_log2),
            _ => (width_log2, STRIDE_1D, 1 << width_log2),
        };
        let two_d = CLASS == CLASS_2D;
        Self {
            scan: if two_d { default_scan(tx) } else { &[] },
            offsets: if two_d {
                nz_map_offsets(tx)
            } else {
                &NO_OFFSETS
            },
            shift,
            shift2: height_log2,
            stride,
            clear: stride * (lines + PAD_LINES),
        }
    }

    /// The place of scan index `i`.
    #[inline]
    fn place<const CLASS: u8>(self, i: usize) -> Place {
        let mask = (1 << self.shift) - 1;
        match CLASS {
            CLASS_2D => {
                let rc = usize::from(self.scan[i]);
                Place { rc, at: rc, y: 0 }
            }
            CLASS_H => {
                // Column by column: the position is the scan index; the map
                // is transposed so the line (`y`, the column) is contiguous.
                let (x, y) = (i & mask, i >> self.shift);
                Place {
                    rc: i,
                    at: x * self.stride + y,
                    y,
                }
            }
            _ => {
                // Row by row over a column-major block.
                let (x, y) = (i & mask, i >> self.shift);
                Place {
                    rc: (x << self.shift2) | y,
                    at: x * self.stride + y,
                    y,
                }
            }
        }
    }

    /// `coeff_base` context of a coefficient other than the last
    /// (dav1d `get_lo_ctx`).
    #[inline]
    fn base_context<const CLASS: u8>(self, levels: &LevelBuffer, p: Place) -> usize {
        let s = self.stride & STRIDE_MASK;
        let at = p.at & AT_MASK;
        let near = |offset: usize| u32::from(levels.levels[at + offset]);
        let (sum, offset) = if CLASS == CLASS_2D {
            // Offsets are at most 21: the mask (a no-op) bounds the context
            // below `SIG_COEF_CONTEXTS` for the compiler.
            let offset = self.offsets[p.rc & (MAX_CODED - 1)] & OFFSET_MASK;
            (
                self.br_sum::<CLASS>(levels, p) + near(2) + near(2 * s),
                usize::from(offset),
            )
        } else {
            let offset = CONTEXTS_2D
                + if p.y > 1 {
                    CONTEXT_FAR
                } else {
                    p.y * CONTEXT_GROUP
                };
            (self.br_sum::<CLASS>(levels, p) + near(3) + near(4), offset)
        };
        let magnitude = if sum > BASE_SUM_SATURATED {
            BASE_MAG_MAX
        } else {
            ((sum + BASE_SUM_ROUND) >> BASE_SUM_SHIFT) as usize
        };
        offset + magnitude
    }

    /// The packed sum of the three neighbours the `coeff_br` context reads
    /// (also the first three of the `coeff_base` context). Recomputed on
    /// the rare `coeff_br` path rather than kept live across the symbol
    /// decode, which spills it.
    #[inline]
    fn br_sum<const CLASS: u8>(self, levels: &LevelBuffer, p: Place) -> u32 {
        let s = self.stride & STRIDE_MASK;
        let at = p.at & AT_MASK;
        let near = |offset: usize| u32::from(levels.levels[at + offset]);
        if CLASS == CLASS_2D {
            near(1) + near(s) + near(s + 1)
        } else {
            near(1) + near(s) + near(2)
        }
    }

    /// The `coeff_br` context group of an AC coefficient: in the 2x2
    /// low-frequency corner (2-D) or on the first line (1-D), or not.
    #[inline]
    fn br_group<const CLASS: u8>(self, p: Place) -> usize {
        let low = if CLASS == CLASS_2D {
            let mask = (1 << self.shift) - 1;
            (p.rc >> self.shift) | (p.rc & mask) <= 1
        } else {
            p.y == 0
        };
        if low { BR_LOW_GROUP } else { BR_HIGH_GROUP }
    }
}

/// `coeff_br` context: the group plus the magnitude term of the plain sum
/// of three neighbours.
#[inline]
fn br_context(group: usize, br_sum: u32) -> usize {
    let mag = br_sum & PLAIN_MASK;
    group
        + if mag > BR_MAG_LIMIT {
            BR_MAG_MAX
        } else {
            ((mag + 1) >> 1) as usize
        }
}

/// The level entry of token `tok`.
#[inline]
fn entry(tok: u32) -> u8 {
    (tok | tok.min(BASE_NEIGHBOUR_MAX) << SATURATED_SHIFT) as u8
}

/// The `coeff_br` run (dav1d `decode_hi_tok`): up to four symbols of up
/// to 3 each, added to the base level 3.
#[inline]
fn read_br(r: &mut SymbolReader<'_>, cdf: &mut Cdf<BR_CDF_SIZE>) -> u32 {
    let mut tok = NUM_BASE_LEVELS + 1;
    for _ in 0..COEFF_BASE_RANGE / (BR_CDF_SIZE as u8 - 1) {
        let k = r.symbol(cdf, BR_CDF_SIZE) as u32;
        tok += k;
        if k < BR_CDF_SIZE as u32 - 1 {
            break;
        }
    }
    tok
}

/// What the level pass leaves the sign pass.
#[derive(Clone, Copy)]
pub(super) struct Tokens {
    /// The DC coefficient's token.
    pub dc: u32,
    /// Storage position of the first nonzero AC coefficient in scan order,
    /// 0 when there is none; its entry starts the chain.
    pub first: usize,
}

/// The CDFs one block's levels use: its size and plane type.
pub(super) struct LevelCdfs<'a> {
    pub base_eob: &'a mut [Cdf<BASE_EOB_SYMBOLS>],
    pub base: &'a mut [Cdf<BASE_SYMBOLS>; SIG_COEF_CONTEXTS],
    pub br: &'a mut [Cdf<BR_CDF_SIZE>],
}

impl<'a> LevelCdfs<'a> {
    pub(super) fn of(
        cdfs: &'a mut CoefficientCdfs,
        size_ctx: usize,
        br_ctx: usize,
        p: usize,
    ) -> Self {
        Self {
            base_eob: &mut cdfs.base_eob[size_ctx][p],
            base: &mut cdfs.base[size_ctx][p],
            br: &mut cdfs.br[br_ctx][p],
        }
    }
}

/// Read the levels of a block whose last coefficient has scan index
/// `last`, leaving each AC token chained in `cf` (zero on entry, column-major
/// over the coded area). `size_log2` is `log2(coded area) - 4`.
#[inline]
pub(super) fn read<const CLASS: u8>(
    cdfs: LevelCdfs<'_>,
    r: &mut SymbolReader<'_>,
    levels: &mut LevelBuffer,
    cf: &mut [i32; MAX_CODED],
    walk: Walk,
    last: usize,
    size_log2: usize,
) -> Tokens {
    let LevelCdfs { base_eob, base, br } = cdfs;
    if last == 0 {
        // DC alone: no map, no chain.
        let mut dc = r.symbol(&mut base_eob[0], BASE_EOB_SYMBOLS) as u32 + 1;
        if dc > NUM_BASE_LEVELS {
            dc = read_br(r, &mut br[0]);
        }
        return Tokens { dc, first: 0 };
    }
    levels.levels[..walk.clear].fill(0);

    // The last coefficient: its context depends on the scan index alone.
    let eob_ctx = 1
        + usize::from(last > EOB_EIGHTH << size_log2)
        + usize::from(last > EOB_QUARTER << size_log2);
    let p = walk.place::<CLASS>(last);
    let mut tok = r.symbol(&mut base_eob[eob_ctx], BASE_EOB_SYMBOLS) as u32 + 1;
    if tok > NUM_BASE_LEVELS {
        tok = read_br(r, &mut br[walk.br_group::<CLASS>(p)]);
    }
    cf[p.rc & (MAX_CODED - 1)] = (tok << TOKEN_SHIFT) as i32;
    levels.set(p.at, entry(tok));

    // The AC coefficients in reverse scan order. `rc` is the position of
    // the nonzero coefficient read last, which the next nonzero one links to.
    let mut rc = p.rc;
    for i in (1..last).rev() {
        let p = walk.place::<CLASS>(i);
        let ctx = walk.base_context::<CLASS>(levels, p);
        let mut tok = r.symbol(&mut base[ctx], BASE_SYMBOLS) as u32;
        if tok > NUM_BASE_LEVELS {
            let br_sum = walk.br_sum::<CLASS>(levels, p);
            tok = read_br(r, &mut br[br_context(walk.br_group::<CLASS>(p), br_sum)]);
        }
        levels.set(p.at, entry(tok));
        // Branch-free on the token: zero entries stay zero and do not move
        // the chain.
        let link = if tok != 0 {
            (tok << TOKEN_SHIFT) as i32 | rc as i32
        } else {
            0
        };
        rc = if tok != 0 { p.rc } else { rc };
        cf[p.rc & (MAX_CODED - 1)] = link;
    }

    // DC: the 2-D class has a context of its own; the 1-D ones use the
    // general one at distance 0.
    let origin = Place { rc: 0, at: 0, y: 0 };
    let ctx = if CLASS == CLASS_2D {
        0
    } else {
        walk.base_context::<CLASS>(levels, origin)
    };
    let mut dc = r.symbol(&mut base[ctx], BASE_SYMBOLS) as u32;
    if dc > NUM_BASE_LEVELS {
        let br_sum = walk.br_sum::<CLASS>(levels, origin);
        dc = read_br(r, &mut br[br_context(0, br_sum)]);
    }
    Tokens { dc, first: rc }
}

/// A 2-D context offset table by storage position, zero-extended to the
/// largest coded area so a masked position needs no bounds check.
type Offsets = [u8; MAX_CODED];
/// 2-D context offsets never exceed 21 (`av1_nz_map_ctx_offset`).
const OFFSET_MASK: u8 = 31;
const _: () = assert!(OFFSET_MASK as usize + BASE_MAG_MAX < SIG_COEF_CONTEXTS);

/// `table` zero-extended to [`MAX_CODED`] entries.
const fn extended<const N: usize>(table: &[u8; N]) -> Offsets {
    let mut out = [0; MAX_CODED];
    let mut i = 0;
    while i < N {
        out[i] = table[i];
        i += 1;
    }
    out
}

/// The 1-D classes take no offset table.
static NO_OFFSETS: Offsets = [0; MAX_CODED];
static OFFSETS_4X4: Offsets = extended(&NZ_MAP_CTX_OFFSET_4X4);
static OFFSETS_8X8: Offsets = extended(&NZ_MAP_CTX_OFFSET_8X8);
static OFFSETS_16X16: Offsets = extended(&NZ_MAP_CTX_OFFSET_16X16);
static OFFSETS_32X32: Offsets = extended(&NZ_MAP_CTX_OFFSET_32X32);
static OFFSETS_4X8: Offsets = extended(&NZ_MAP_CTX_OFFSET_4X8);
static OFFSETS_16X4: Offsets = extended(&NZ_MAP_CTX_OFFSET_16X4);
static OFFSETS_8X16: Offsets = extended(&NZ_MAP_CTX_OFFSET_8X16);
static OFFSETS_32X8: Offsets = extended(&NZ_MAP_CTX_OFFSET_32X8);
static OFFSETS_16X32: Offsets = extended(&NZ_MAP_CTX_OFFSET_16X32);
static OFFSETS_32X16: Offsets = extended(&NZ_MAP_CTX_OFFSET_32X16);
static OFFSETS_32X64: Offsets = extended(&NZ_MAP_CTX_OFFSET_32X64);
static OFFSETS_64X32: Offsets = extended(&NZ_MAP_CTX_OFFSET_64X32);
static OFFSETS_4X16: Offsets = extended(&NZ_MAP_CTX_OFFSET_4X16);
static OFFSETS_8X32: Offsets = extended(&NZ_MAP_CTX_OFFSET_8X32);

/// libaom `av1_nz_map_ctx_offset[tx_size]` (sizes of equal coded shape
/// share tables): the 2-D context offset by storage position.
fn nz_map_offsets(tx: TxSize) -> &'static Offsets {
    match (tx.width_log2(), tx.height_log2()) {
        (2, 2) => &OFFSETS_4X4,
        (3, 3) => &OFFSETS_8X8,
        (4, 4) => &OFFSETS_16X16,
        (5, 5) | (6, 6) => &OFFSETS_32X32,
        (2, 3) => &OFFSETS_4X8,
        (3 | 4, 2) => &OFFSETS_16X4,
        (3, 4) => &OFFSETS_8X16,
        (4 | 5, 3) => &OFFSETS_32X8,
        (4, 5) => &OFFSETS_16X32,
        (5 | 6, 4) => &OFFSETS_32X16,
        (4 | 5, 6) => &OFFSETS_32X64,
        (6, 5) => &OFFSETS_64X32,
        (2, 4) => &OFFSETS_4X16,
        (3, 5) => &OFFSETS_8X32,
        _ => unreachable!("not a transform size"),
    }
}

/// libaom's 2-D scan of each coded size.
fn default_scan(tx: TxSize) -> &'static [u16] {
    match (tx.coded_width_log2(), tx.coded_height_log2()) {
        (2, 2) => &DEFAULT_SCAN_4X4,
        (3, 3) => &DEFAULT_SCAN_8X8,
        (4, 4) => &DEFAULT_SCAN_16X16,
        (2, 3) => &DEFAULT_SCAN_4X8,
        (3, 2) => &DEFAULT_SCAN_8X4,
        (3, 4) => &DEFAULT_SCAN_8X16,
        (4, 3) => &DEFAULT_SCAN_16X8,
        (4, 5) => &DEFAULT_SCAN_16X32,
        (5, 4) => &DEFAULT_SCAN_32X16,
        (2, 4) => &DEFAULT_SCAN_4X16,
        (4, 2) => &DEFAULT_SCAN_16X4,
        (3, 5) => &DEFAULT_SCAN_8X32,
        (5, 3) => &DEFAULT_SCAN_32X8,
        _ => &DEFAULT_SCAN_32X32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Every transform size.
    fn sizes() -> impl Iterator<Item = TxSize> {
        (2..=6).flat_map(|w| (2..=6).filter_map(move |h| TxSize::new(w, h)))
    }

    /// The level map needs no row padding: every neighbour a context reads
    /// lies in the cleared region and is either padding or a coefficient
    /// later in reverse scan order (still zero when read). With padding,
    /// libaom reads zero there too, so the contexts agree.
    fn check_neighbours<const CLASS: u8>(tx: TxSize) {
        let walk = Walk::new::<CLASS>(tx);
        let area = tx.coded_area();
        // Scan index of each map entry a coefficient occupies.
        let mut occupant = vec![None; walk.clear];
        let mut positions = vec![false; area];
        for i in 0..area {
            let p = walk.place::<CLASS>(i);
            assert!(
                occupant[p.at].replace(i).is_none(),
                "{tx:?}: two coefficients share an entry"
            );
            assert!(
                !std::mem::replace(&mut positions[p.rc], true),
                "{tx:?}: scan repeats"
            );
        }
        // Each neighbour's map offset and its (column, row) step: the
        // transposed 1-D maps run along the line (columns for H, rows for V)
        // at offsets 1 and 2..4, the stride across.
        let s = walk.stride;
        let neighbours: [(usize, (usize, usize)); 5] = match CLASS {
            CLASS_2D => [
                (1, (0, 1)),
                (s, (1, 0)),
                (s + 1, (1, 1)),
                (2, (0, 2)),
                (2 * s, (2, 0)),
            ],
            CLASS_H => [
                (1, (1, 0)),
                (s, (0, 1)),
                (2, (2, 0)),
                (3, (3, 0)),
                (4, (4, 0)),
            ],
            _ => [
                (1, (0, 1)),
                (s, (1, 0)),
                (2, (0, 2)),
                (3, (0, 3)),
                (4, (0, 4)),
            ],
        };
        for i in 0..area {
            let p = walk.place::<CLASS>(i);
            let height_log2 = tx.coded_height_log2();
            let (col, row) = (p.rc >> height_log2, p.rc & ((1 << height_log2) - 1));
            for &(offset, (dc, dr)) in &neighbours {
                let at = p.at + offset;
                assert!(at < walk.clear, "{tx:?}: neighbour outside the cleared map");
                // Where libaom's padded map would look: inside the block
                // only if both coordinates are.
                let inside =
                    col + dc < 1 << tx.coded_width_log2() && row + dr < 1 << tx.coded_height_log2();
                if inside {
                    let j = occupant[at].expect("an inside neighbour is a coefficient");
                    let q = walk.place::<CLASS>(j);
                    let rc = ((col + dc) << height_log2) | (row + dr);
                    assert_eq!(q.rc, rc, "{tx:?}: neighbour at the wrong entry");
                } else if let Some(j) = occupant[at] {
                    assert!(j < i, "{tx:?}: wrapped neighbour read after it is set");
                }
            }
        }
    }

    /// The offset mask in the 2-D context is a no-op.
    #[test]
    fn offsets_fit_their_mask() {
        for tx in sizes() {
            assert!(nz_map_offsets(tx).iter().all(|&o| o & OFFSET_MASK == o));
        }
    }

    #[test]
    fn unpadded_maps_read_the_same_neighbours_as_padded_ones() {
        for tx in sizes() {
            check_neighbours::<CLASS_2D>(tx);
            if tx.width_log2() <= 4 && tx.height_log2() <= 4 {
                check_neighbours::<CLASS_H>(tx);
                check_neighbours::<CLASS_V>(tx);
            }
        }
    }
}
