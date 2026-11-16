//! The motion of the blocks decoded so far: per 4x4 unit of a tile's
//! last superblocks, the 8x8 row above the current superblock row, the
//! MV and warp banks, and the frame's temporal fields (dav2d `refmvs.h`
//! `refmvs_block`, `refmvs_tile`; `refmvs.c` `dav2d_refmvs_tile_sbrow_init`,
//! `dav2d_refmvs_reset_sb`, `dav2d_refmvs_save_tmvs`, `splat_mv_c`,
//! `splat_warpmv_c`, `splat_comp_warpmv_c`, `splat_comp_wedgemv_c`;
//! `decode.c` `splat_oneref_mv`, `splat_tworef_mv`, `splat_intrabc_mv`,
//! `splat_intraref`; dav2d commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`).
//!
//! The parts: [`vector`] (vector types, quantizing, projection), [`bank`]
//! (MV and warp banks), [`frame`] (the frame's distances and projection
//! list, its temporal field), [`temporal`] and [`tip`] (the projected
//! field), [`candidates`] (a block's candidate list, dav2d
//! `dav2d_refmvs_find`).
pub mod bank;
pub mod candidates;
pub mod frame;
pub mod temporal;
pub mod tip;
pub mod vector;

use super::size::BlockSize;
use crate::general::warp::WarpModel;
use bank::{MvBank, WarpBank};
use frame::{FrameMotion, TIP_FRAME_MODE, merge_pair};
use vector::{MAX_COMPONENT, QuantizedMv, RefPair, TIP_REFERENCE, TemporalMv};

pub use candidates::{Candidate, Candidates, DEFAULT_LEFT_EXTRA};
pub use vector::Mv;

/// The units kept: 64 rows (a 256x256 superblock row) by 128 columns
/// (two of its superblocks: the current one and the one to its left),
/// indexed modulo their extent (dav2d `rt->r`).
const RING_ROWS: u32 = 64;
const RING_COLUMNS: u32 = 128;
/// Of a block's units only the bottom row and the right-most three
/// columns are ever read by later blocks (dav2d writes only those).
const READ_COLUMNS: u32 = 3;
/// At a superblock's start the banks take at most this many blocks'
/// motion from the row above.
const REFILL_MAX: u32 = 4;
/// Warp matrices' translation is in Q16 / 4 for the 8x8 vectors of a
/// warp splat; their vectors round from Q11.
const WARP_SPLAT_TRANSLATION_SHIFT: u32 = 2;
const WARP_SPLAT_SHIFT: u32 = 11;
/// The unit diagonal of a warp matrix.
const UNITY: i32 = crate::general::warp::UNITY;

/// One 4x4 unit's record (dav2d `refmvs_block`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionUnit {
    /// The vectors ([`Mv::INVALID`] first for intra or not decoded; the
    /// second for one reference). A warped block's are its model's
    /// vectors at the unit's 8x8 block.
    pub mv: [Mv; 2],
    /// The references ([`RefPair::NONE`] for intra and block copies).
    pub refs: RefPair,
    /// The size of the block the unit belongs to.
    pub size: BlockSize,
    /// Global motion (dav2d `mf` bit 0).
    pub global: bool,
    /// A warped block (`mf` bit 1): `warp_mv` and `matrix` hold.
    pub warped: bool,
    /// The compound weight index (`mf >> 2`).
    pub cwp: i8,
    /// The block's top-left in 4x4 units (dav2d `by4`, `bx4`).
    pub row: u32,
    /// See `row`.
    pub column: u32,
    /// A warped block's own vectors (dav2d `lmv`).
    pub warp_mv: [Mv; 2],
    /// A warped block's matrix (dav2d `m`).
    pub matrix: [i32; 6],
}

impl MotionUnit {
    /// Intra, or not decoded yet.
    #[must_use]
    pub fn is_intra(&self) -> bool {
        self.mv[0] == Mv::INVALID
    }

    /// The vectors the bank takes (a warped block's own).
    #[must_use]
    pub fn bank_mv(&self) -> [Mv; 2] {
        if self.warped { self.warp_mv } else { self.mv }
    }
}

/// A unit nothing has written yet.
const EMPTY: MotionUnit = MotionUnit {
    mv: [Mv::INVALID; 2],
    refs: RefPair::NONE,
    size: BlockSize::B256x256,
    global: false,
    warped: false,
    cwp: 0,
    row: 0,
    column: 0,
    warp_mv: [Mv::INVALID; 2],
    matrix: [0; 6],
};

/// Where the store's frame and tile lie, in 4x4 units.
#[derive(Debug, Clone, Copy)]
pub struct StoreGeometry {
    /// The frame's width and height (8x8-rounded: dav2d `iw4`, `ih4`).
    pub columns: u32,
    /// See `columns`.
    pub rows: u32,
    /// The superblock side, as a log2 of luma samples (6, 7 or 8).
    pub superblock_log2: u8,
    /// The tile's first column.
    pub column_start: u32,
    /// The tile's column end (the frame's, at most).
    pub column_end: u32,
    /// The tile's first row.
    pub row_start: u32,
    /// The tile's row end (the frame's, at most).
    pub row_end: u32,
}

impl StoreGeometry {
    /// The superblock side in units (dav2d `sbsz`).
    #[must_use]
    pub fn superblock_units(&self) -> u32 {
        1 << (self.superblock_log2 - 2)
    }
}

/// How a compound block's wedge splits its 8x8 units between the two
/// predictions (dav2d `WEDGE_TMVP`): per unit, 0 or 1 for the side that
/// dominates, 2 for mixed; and the wedge sign.
#[derive(Debug, Clone, Copy)]
pub struct WedgeSides<'a> {
    /// Row-major, `bw4 / 2` per row.
    pub sides: &'a [u8],
    /// `wedge_sign`.
    pub sign: bool,
}

/// What an inter block leaves for later blocks and frames.
#[derive(Debug, Clone, Copy)]
pub struct InterMotion<'a> {
    /// Top-left in 4x4 units.
    pub row: u32,
    /// See `row`.
    pub column: u32,
    /// The block size.
    pub size: BlockSize,
    /// Its references.
    pub refs: RefPair,
    /// Its final vectors.
    pub mv: [Mv; 2],
    /// Global motion mode (`GLOBALMV`, `GLOBALMV_GLOBALMV`).
    pub global_mode: bool,
    /// The warp matrices of a warped block (`motion_mode` above
    /// inter-intra), one per reference.
    pub warps: Option<[[i32; 6]; 2]>,
    /// The compound weight index (`cwp_idx`).
    pub cwp: i8,
    /// A wedge compound's split.
    pub wedge: Option<WedgeSides<'a>>,
    /// A compound block keeps temporal vectors (not refined by optical
    /// flow and MV refinement together).
    pub keep_temporal: bool,
}

/// The motion of one tile (dav2d `refmvs_tile` with the frame's above row
/// `rf->ra`), and the frame's motion state.
#[derive(Debug, Clone)]
pub struct MotionStore {
    geometry: StoreGeometry,
    units: Vec<MotionUnit>,
    /// Per 8x8 column of the frame, the unit above the current superblock
    /// row (its odd row, even column), and the one that was above the
    /// current superblock's top-left before its left neighbour replaced it.
    above: Vec<MotionUnit>,
    above_left: MotionUnit,
    /// The MV bank, when the sequence has one.
    bank: Option<MvBank>,
    warps: WarpBank,
    /// The frame's part.
    pub frame: FrameMotion,
}

impl MotionStore {
    /// A store for a tile of a frame.
    #[must_use]
    pub fn new(geometry: StoreGeometry, frame: FrameMotion) -> Self {
        Self {
            geometry,
            units: vec![EMPTY; (RING_ROWS * RING_COLUMNS) as usize],
            above: vec![EMPTY; frame.stride.max(geometry.columns.div_ceil(2) as usize)],
            above_left: EMPTY,
            bank: frame.params.bank.then(MvBank::default),
            warps: WarpBank::default(),
            frame,
        }
    }

    /// The tile's geometry.
    #[must_use]
    pub fn geometry(&self) -> &StoreGeometry {
        &self.geometry
    }

    fn index(row: u32, column: u32) -> usize {
        ((row % RING_ROWS) * RING_COLUMNS + column % RING_COLUMNS) as usize
    }

    /// The unit at (`row`, `column`) of the current or previous
    /// superblocks.
    #[must_use]
    pub fn unit(&self, row: u32, column: u32) -> &MotionUnit {
        &self.units[Self::index(row, column)]
    }

    /// The unit above the current superblock row at 8x8 column `x8`.
    #[must_use]
    pub fn above(&self, x8: u32) -> &MotionUnit {
        &self.above[x8 as usize]
    }

    /// The unit that was above the current superblock's top-left.
    #[must_use]
    pub fn above_left(&self) -> &MotionUnit {
        &self.above_left
    }

    /// At each superblock row of the tile: the banks start empty, and the
    /// row's temporal field is projected when the frame uses motion fields
    /// (dav2d `dav2d_refmvs_tile_sbrow_init`, `dav2d_refmvs_load_tmvs`).
    pub fn start_row(&mut self, row: u32) {
        if let Some(bank) = &mut self.bank {
            bank.clear();
        }
        self.warps.clear();
        if self.frame.params.use_motion_fields {
            let start8 = (row >> 1) as usize;
            let end8 = start8 + (self.geometry.superblock_units() >> 1) as usize;
            self.frame.load(start8, end8);
        }
    }

    /// At each superblock: its units are not decoded yet, the banks'
    /// limits restart, and in inter frames below the tile's first row the
    /// banks take up to four blocks' motion from the row above (dav2d
    /// `dav2d_refmvs_reset_sb`).
    pub fn start_superblock(&mut self, row: u32, column: u32) {
        let side = self.geometry.superblock_units();
        for y in row..row + side {
            for x in column..column + side {
                let unit = &mut self.units[Self::index(y, x)];
                unit.mv[0] = Mv::INVALID;
                unit.refs = RefPair::NONE;
            }
        }
        if let Some(bank) = &mut self.bank {
            bank.start_superblock();
        }
        self.warps.start_superblock();
        let params = &self.frame.params;
        if row == self.geometry.row_start || params.intra_frame || params.tip_mode == TIP_FRAME_MODE
        {
            return;
        }
        let end = (column + side).min(self.geometry.column_end);
        let (mut x, mut hits) = (column, 0);
        while x < end {
            let above = self.above[(x >> 1) as usize];
            x += above.size.units().0;
            if above.is_intra() {
                continue;
            }
            if let Some(bank) = &mut self.bank {
                bank.insert(above.refs, above.bank_mv(), above.cwp);
            }
            if above.warped {
                self.warps.insert(&above.matrix, above.refs.get(0) as usize);
            }
            hits += 1;
            if hits == REFILL_MAX {
                break;
            }
        }
    }

    /// After each superblock: keep its bottom row (at 8x8 resolution) for
    /// the superblock row below (dav2d `dav2d_refmvs_save_tmvs`).
    pub fn end_superblock(&mut self, row: u32, column: u32) {
        let side = self.geometry.superblock_units();
        // Superblock positions and sides are even in 4x4 units.
        let (row8, column8, side8) = (row >> 1, column >> 1, side >> 1);
        let row_end8 = (row8 + side8).min(self.geometry.rows >> 1);
        let column_end8 = (column8 + side8).min(self.geometry.columns.div_ceil(2));
        let source_row = (row_end8 - 1) * 2 + 1;
        self.above_left = self.above[(column_end8 - 1) as usize];
        for x8 in column8..column_end8 {
            self.above[x8 as usize] = *self.unit(source_row, x8 * 2);
        }
    }

    /// Write a block's units where later blocks read them: the bottom row
    /// and the right-most three columns; `unit` gives the record at each
    /// unit offset (dav2d's splat footprint).
    fn write_units(
        &mut self,
        row: u32,
        column: u32,
        (bw4, bh4): (u32, u32),
        unit: impl Fn(u32, u32) -> MotionUnit,
    ) {
        for y in 0..bh4 {
            let first = if y + 1 == bh4 {
                0
            } else {
                bw4.saturating_sub(READ_COLUMNS)
            };
            for x in first..bw4 {
                self.units[Self::index(row + y, column + x)] = unit(x, y);
            }
        }
    }

    /// Write the block's 8x8 units of the frame's temporal field;
    /// `entry` gives each one's vectors and references.
    fn write_temporal(
        &mut self,
        row: u32,
        column: u32,
        (bw4, bh4): (u32, u32),
        entry: impl Fn(u32, u32) -> ([QuantizedMv; 2], RefPair),
    ) {
        let Some(field) = &mut self.frame.field else {
            return;
        };
        let (row8, column8) = ((row >> 1) as usize, (column >> 1) as usize);
        for y in 0..bh4.div_ceil(2) {
            for x in 0..bw4.div_ceil(2) {
                let at = (row8 + y as usize) * field.stride + column8 + x as usize;
                let (mvs, refs) = entry(x, y);
                field.mvs[at] = mvs;
                field.refs[at] = refs;
            }
        }
    }

    /// The record every unit of a block shares.
    fn base_unit(row: u32, column: u32, size: BlockSize, refs: RefPair, mv: [Mv; 2]) -> MotionUnit {
        MotionUnit {
            mv,
            refs,
            size,
            row,
            column,
            ..EMPTY
        }
    }

    /// Record an intra block of an inter frame or a frame allowing block
    /// copies (dav2d `splat_intraref`), with the bank's limits.
    pub fn record_intra(&mut self, row: u32, column: u32, size: BlockSize) {
        let unit = Self::base_unit(row, column, size, RefPair::NONE, [Mv::INVALID; 2]);
        self.write_units(row, column, size.units(), |_, _| unit);
        self.write_temporal(row, column, size.units(), |_, _| {
            ([QuantizedMv::INVALID; 2], RefPair::NONE)
        });
        self.update_bank(row, column, size);
    }

    /// Record a block copy and offer its vector to the bank (dav2d
    /// `splat_intrabc_mv`).
    pub fn record_block_copy(&mut self, row: u32, column: u32, size: BlockSize, mv: Mv) {
        let unit = Self::base_unit(row, column, size, RefPair::NONE, [mv, Mv::INVALID]);
        self.write_units(row, column, size.units(), |_, _| unit);
        self.write_temporal(row, column, size.units(), |_, _| {
            ([QuantizedMv::INVALID; 2], RefPair::NONE)
        });
        self.add_to_bank(row, column, size, RefPair::NONE, [mv, Mv::INVALID], 0);
    }

    /// The bank's limits at a block (dav2d `dav2d_refmvs_bank_update`).
    pub fn update_bank(&mut self, row: u32, column: u32, size: BlockSize) {
        let side = self.geometry.superblock_units();
        // Bank units are 2, 4 or 8 units square by superblock size.
        let unit_log2 = u32::from(self.geometry.superblock_log2) - 5;
        if let Some(bank) = &mut self.bank {
            bank.update((row, column), size.units(), side, unit_log2);
        }
    }

    /// Offer a block's vectors to the bank (dav2d `dav2d_refmvs_bank_add`).
    pub fn add_to_bank(
        &mut self,
        row: u32,
        column: u32,
        size: BlockSize,
        refs: RefPair,
        mv: [Mv; 2],
        cwp: i8,
    ) {
        self.update_bank(row, column, size);
        if let Some(bank) = &mut self.bank {
            bank.offer(refs, mv, cwp);
        }
    }

    /// Offer a warped block's model of reference `rank` to the warp bank
    /// (dav2d `dav2d_refmvs_warp_add`); false once the superblock's limit
    /// is reached.
    pub fn add_warp(&mut self, matrix: &[i32; 6], rank: usize) -> bool {
        self.warps.insert(matrix, rank)
    }

    /// Whether the temporal field keeps the pair (`r0`, `r1`) swapped
    /// (dav2d `ref_flip` bit `r0 * 8 + r1`).
    #[must_use]
    pub fn reference_flip(&self, r0: usize, r1: usize) -> bool {
        self.frame.flip >> (r0 * 8 + r1) & 1 != 0
    }

    /// The projected TIP field at 8x8 unit (`row8`, `column8`) of the
    /// frame, zero where empty (dav2d `rp_proj` reads of `tip_pred`).
    #[must_use]
    pub fn tip_vector(&self, row8: u32, column8: u32) -> TemporalMv {
        self.frame
            .projected(row8 as usize, column8 as usize)
            .0
            .or_zero()
    }

    /// A TIP field vector scaled to TIP reference `side` (dav2d
    /// `scale_tmv(tmv, rf.tip.sf[side])`).
    #[must_use]
    pub fn tip_scaled(&self, v: TemporalMv, side: usize) -> Mv {
        vector::scale_temporal(v, self.frame.tip.factors[side])
    }

    /// Record a TIP-predicted block's vectors in the temporal field
    /// (dav2d `update_temporal`): `references` and `mvs` in block order,
    /// kept swapped when `swap`; nothing when the sequence keeps no
    /// fields.
    pub fn update_temporal(
        &mut self,
        (row8, column8): (u32, u32),
        (w8, h8): (u32, u32),
        references: [i8; 2],
        mvs: [Mv; 2],
        swap: bool,
    ) {
        let Some(field) = &mut self.frame.field else {
            return;
        };
        let s = usize::from(swap);
        field.record(
            (row8 as usize, column8 as usize),
            (w8 as usize, h8 as usize),
            RefPair([references[s], references[1 - s]]),
            [mvs[s], mvs[1 - s]],
        );
    }

    /// The global model of reference `rank` when blocks may warp by it
    /// (dav2d `gmv_warp_allowed`).
    #[must_use]
    pub fn global_warp(&self, rank: usize) -> Option<WarpModel> {
        self.frame.global_warp_allowed[rank].then_some(self.frame.global[rank])
    }

    /// Record an inter block (dav2d `splat_oneref_mv`, `splat_tworef_mv`).
    pub fn record_inter(&mut self, block: &InterMotion<'_>) {
        if block.refs.is_compound() {
            self.record_compound(block);
        } else {
            self.record_single(block);
        }
    }

    /// Whether a block warps by a reference's global motion (a global
    /// mode with a non-translation model, not whole-sample, 8x8 or more).
    fn warps_globally(&self, block: &InterMotion<'_>, rank: i8) -> bool {
        let (bw4, bh4) = block.size.units();
        block.global_mode
            && !self.frame.params.force_integer_mv
            && bw4.min(bh4) > 1
            && self
                .frame
                .global_model(rank)
                .is_some_and(|model| model.kind > crate::general::warp::kind::TRANSLATION)
    }

    fn record_single(&mut self, block: &InterMotion<'_>) {
        let (row, column, size) = (block.row, block.column, block.size);
        let (bw4, bh4) = size.units();
        let reference = block.refs.get(0);
        let refs = RefPair::single(reference);
        let mut unit = Self::base_unit(row, column, size, refs, [block.mv[0], Mv::INVALID]);
        unit.global = block.global_mode && bw4.min(bh4) > 1;
        // TIP blocks' temporal vectors are their prediction's.
        let temporal = reference != TIP_REFERENCE;
        let both = RefPair::both(reference);
        if let Some(warps) = block.warps {
            unit.global = false;
            unit.warped = true;
            unit.warp_mv = [block.mv[0], Mv::INVALID];
            unit.matrix = warps[0];
            // Whole-sample frames keep the block's own vector in the field.
            let field_matrix = (!self.frame.params.force_integer_mv).then_some(warps[0]);
            self.splat_warp(&unit, &warps[0], |mv| [mv, Mv::INVALID]);
            if temporal {
                self.write_temporal(row, column, (bw4, bh4), |x, y| {
                    let mv = field_matrix
                        .map_or(block.mv[0], |m| warp_vector(&m, (row, column), (x, y)));
                    single_entry(mv, both)
                });
            }
        } else if self.warps_globally(block, reference) {
            let matrix = self.frame.global[reference as usize].matrix;
            self.write_units(row, column, (bw4, bh4), |_, _| unit);
            if temporal {
                self.write_temporal(row, column, (bw4, bh4), |x, y| {
                    single_entry(warp_vector(&matrix, (row, column), (x, y)), both)
                });
            }
        } else {
            self.write_units(row, column, (bw4, bh4), |_, _| unit);
            if temporal {
                let entry = single_entry(block.mv[0], both);
                self.write_temporal(row, column, (bw4, bh4), |_, _| entry);
            }
        }
    }

    /// Write a warped block's units, each with the vectors `vectors` makes
    /// of its 8x8 block's model vector.
    fn splat_warp(
        &mut self,
        unit: &MotionUnit,
        matrix: &[i32; 6],
        vectors: impl Fn(Mv) -> [Mv; 2],
    ) {
        let (row, column) = (unit.row, unit.column);
        self.write_units(row, column, unit.size.units(), |x, y| {
            let mv = warp_vector(matrix, (row, column), (x >> 1, y >> 1));
            MotionUnit {
                mv: vectors(mv),
                ..*unit
            }
        });
    }

    fn record_compound(&mut self, block: &InterMotion<'_>) {
        let (row, column, size) = (block.row, block.column, block.size);
        let (bw4, bh4) = size.units();
        let refs = block.refs;
        let swap = self.frame.flipped(refs);
        let s = usize::from(swap);
        let mut field_refs = RefPair::NONE;
        field_refs.0[s] = refs.get(0);
        field_refs.0[1 - s] = refs.get(1);
        let mut unit = Self::base_unit(row, column, size, refs, block.mv);
        unit.cwp = block.cwp;
        let keep = block.keep_temporal;
        let wedge = block.wedge;
        let wedge_swap = wedge.map_or(0, |w| usize::from(w.sign ^ swap));
        let side = |x: u32, y: u32| wedge.map(|w| w.sides[(y * (bw4 >> 1) + x) as usize]);
        let global_pair =
            self.warps_globally(block, refs.get(0)) || self.warps_globally(block, refs.get(1));
        let matrices = if let Some(warps) = block.warps {
            unit.warped = true;
            unit.warp_mv = block.mv;
            unit.matrix = warps[0];
            let m = warps;
            self.write_units(row, column, (bw4, bh4), |x, y| MotionUnit {
                mv: [
                    warp_vector(&m[0], (row, column), (x >> 1, y >> 1)),
                    warp_vector(&m[1], (row, column), (x >> 1, y >> 1)),
                ],
                ..unit
            });
            Some(warps)
        } else if global_pair && block.global_mode {
            unit.global = true;
            self.write_units(row, column, (bw4, bh4), |_, _| unit);
            let global = &self.frame.global;
            Some([
                global[refs.get(0) as usize].matrix,
                global[refs.get(1) as usize].matrix,
            ])
        } else {
            unit.global = block.global_mode && bw4.min(bh4) > 1;
            self.write_units(row, column, (bw4, bh4), |_, _| unit);
            None
        };
        if !keep {
            return;
        }
        if let Some([first, second]) = matrices {
            self.write_temporal(row, column, (bw4, bh4), |x, y| {
                let mut mvs = [QuantizedMv::INVALID; 2];
                mvs[s] = QuantizedMv::quantize(warp_vector(&first, (row, column), (x, y)));
                mvs[1 - s] = QuantizedMv::quantize(warp_vector(&second, (row, column), (x, y)));
                if let Some(d) = side(x, y).filter(|&d| d != 2) {
                    mvs[usize::from(d) ^ wedge_swap] = QuantizedMv::INVALID;
                }
                merge_pair(mvs, field_refs)
            });
            return;
        }
        let quantized = [
            QuantizedMv::quantize(block.mv[s]),
            QuantizedMv::quantize(block.mv[1 - s]),
        ];
        let merged = merge_pair(quantized, field_refs);
        self.write_temporal(row, column, (bw4, bh4), |x, y| match side(x, y) {
            Some(d) if d != 2 => {
                let keep = 1 - (usize::from(d) ^ wedge_swap);
                let mv = quantized[keep];
                if mv.is_valid() {
                    ([mv; 2], RefPair::both(field_refs.get(keep)))
                } else {
                    ([mv; 2], RefPair::NONE)
                }
            }
            _ => merged,
        });
    }
}

/// A one-reference field entry (a vector out of range keeps none).
fn single_entry(mv: Mv, both: RefPair) -> ([QuantizedMv; 2], RefPair) {
    let q = QuantizedMv::quantize(mv);
    ([q; 2], if q.is_valid() { both } else { RefPair::NONE })
}

/// A warp model's vector at 8x8 unit (`x8`, `y8`) of the block at
/// (`row`, `column`) (dav2d `splat_warpmv_c`'s running sums).
fn warp_vector(m: &[i32; 6], (row, column): (u32, u32), (x8, y8): (u32, u32)) -> Mv {
    let (bx, by) = (
        column as i32 + 1 + 2 * x8 as i32,
        row as i32 + 1 + 2 * y8 as i32,
    );
    let mvx = (m[2] - UNITY)
        .wrapping_mul(bx)
        .wrapping_add(m[3].wrapping_mul(by))
        .wrapping_add(m[0] >> WARP_SPLAT_TRANSLATION_SHIFT);
    let mvy = m[4]
        .wrapping_mul(bx)
        .wrapping_add(m[1] >> WARP_SPLAT_TRANSLATION_SHIFT)
        .wrapping_add((m[5] - UNITY).wrapping_mul(by));
    let round = |v: i32| {
        let magnitude =
            ((v.unsigned_abs() + (1 << (WARP_SPLAT_SHIFT - 1))) >> WARP_SPLAT_SHIFT) as i32;
        (if v < 0 { -magnitude } else { magnitude }).clamp(-MAX_COMPONENT, MAX_COMPONENT)
    };
    Mv {
        y: round(mvy),
        x: round(mvx),
    }
}
