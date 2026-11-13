//! The decoder's block state as neighbours see it (AV1 §6.10 `MiSizes`,
//! `Skips`, `SkipModes`, `RefFrames`, `YModes`, `CompGroupIdxs`,
//! `CompoundIdxs` along the above row and left column) and
//! the contexts derived from it (§8.3.2). Unlike the encoder's square-only
//! mode info, cells describe rectangular blocks and two references.
//!
//! Keeping only the above row and left column follows dav1d (BSD-2-Clause,
//! `src/env.h` `BlockContext`): a block writes `width + height` cells, not
//! `width * height`. Cells are packed into 4 aligned bytes so those writes
//! are plain 32-bit fills.
use super::block_size::BlockSize;
use super::frame_header::{ALTREF_FRAME, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, TOTAL_REFS};
use crate::tables::INTRA_MODE_CONTEXT;

const LAST2_FRAME: usize = 2;
const LAST3_FRAME: usize = 3;
const BWDREF_FRAME: usize = 5;
const ALTREF2_FRAME: usize = 6;
/// `INTRA_FRAME` plus the seven references.
const REFERENCE_NAMES: usize = ALTREF_FRAME + 1;

/// Reference names as bits, for the reference-count contexts.
const fn refs(names: &[usize]) -> u8 {
    let mut mask = 0;
    let mut i = 0;
    while i < names.len() {
        mask |= 1 << names[i];
        i += 1;
    }
    mask
}

/// The two reference groups each `single_ref_p1..p6` bit compares
/// (§8.3.2: the context is `ref_count_ctx` of their neighbour counts).
const SINGLE_REF_GROUPS: [(u8, u8); 6] = [
    (
        refs(&[LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME]),
        refs(&[BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME]),
    ),
    (refs(&[BWDREF_FRAME, ALTREF2_FRAME]), refs(&[ALTREF_FRAME])),
    (
        refs(&[LAST_FRAME, LAST2_FRAME]),
        refs(&[LAST3_FRAME, GOLDEN_FRAME]),
    ),
    (refs(&[LAST_FRAME]), refs(&[LAST2_FRAME])),
    (refs(&[LAST3_FRAME]), refs(&[GOLDEN_FRAME])),
    (refs(&[BWDREF_FRAME]), refs(&[ALTREF2_FRAME])),
];

/// Bits of a reference nibble in [`BlockInfo::refs`].
const REF_NIBBLE_BITS: u32 = 4;
const REF_NIBBLE_MASK: u8 = (1 << REF_NIBBLE_BITS) - 1;
/// The nibble of `RefFrame[1] = NONE` (single prediction, §6.10.24); a
/// stored nibble is `RefFrame + 1`.
const NO_REFERENCE_NIBBLE: u8 = 0;
/// The nibble of `RefFrame[0] = INTRA_FRAME`.
const INTRA_NIBBLE: u8 = INTRA_FRAME as u8 + 1;
/// The first nibble of an intra block copy block: above every reference,
/// so it counts as inter (libaom `is_inter_block`). Such blocks occur only
/// in intra frames, where no context reads a neighbour's references.
const INTRABC_NIBBLE: u8 = TOTAL_REFS as u8 + 1;
/// The smooth intra modes (`SMOOTH_PRED` ..= `SMOOTH_H_PRED`).
const SMOOTH_MODES: std::ops::RangeInclusive<u8> = 9..=11;
/// `skip` in [`BlockInfo::mode`]; the low bits hold `YMode` of intra
/// blocks.
const SKIP_BIT: u8 = 0x80;
const Y_MODE_MASK: u8 = 0x0f;
/// `skip_mode`, `comp_group_idx` and `compound_idx` of inter blocks in
/// [`BlockInfo::mode`].
const SKIP_MODE_BIT: u8 = 0x10;
const COMP_GROUP_BIT: u8 = 0x20;
const COMPOUND_IDX_BIT: u8 = 0x40;
/// The `comp_mode` context of a block with no neighbours, and the
/// `comp_ref_type` context where the neighbours say nothing (§8.3.2).
const COMP_MODE_NO_NEIGHBOURS: usize = 1;
const COMP_REF_TYPE_DEFAULT: usize = 2;
/// Largest `comp_group_idx` context.
const COMP_GROUP_MAX_CONTEXT: usize = 5;
/// The `compound_idx` context offset of equally distant references.
const COMPOUND_IDX_EQUAL_DISTANCE: usize = 3;

/// One 4x4 cell: the state of the block covering it, in 4 aligned bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(4))]
pub(crate) struct BlockInfo {
    size: BlockSize,
    /// `RefFrame[0] + 1` in the low nibble, `RefFrame[1] + 1` in the high.
    refs: u8,
    /// `YMode` (intra blocks) in the low bits, plus [`SKIP_BIT`].
    mode: u8,
}

impl BlockInfo {
    /// An intra block of `size` with luma mode `y_mode`.
    pub(crate) fn intra(size: BlockSize, skip: bool, y_mode: u8) -> Self {
        debug_assert!(y_mode <= Y_MODE_MASK);
        Self {
            size,
            refs: INTRA_NIBBLE | (NO_REFERENCE_NIBBLE << REF_NIBBLE_BITS),
            mode: y_mode | if skip { SKIP_BIT } else { 0 },
        }
    }
    /// An intra block copy block (`use_intrabc`): an inter block whose
    /// mode is `DC_PRED`.
    pub(crate) fn intrabc(size: BlockSize, skip: bool) -> Self {
        Self {
            size,
            refs: INTRABC_NIBBLE | (NO_REFERENCE_NIBBLE << REF_NIBBLE_BITS),
            mode: if skip { SKIP_BIT } else { 0 },
        }
    }
    /// An inter block predicted from `refs` (`RefFrame[1]` is `NONE` or
    /// `INTRA_FRAME` unless compound).
    pub(crate) fn inter(size: BlockSize, skip: bool, refs: [i8; 2], flags: InterFlags) -> Self {
        Self {
            size,
            refs: (refs[0] + 1) as u8 | (((refs[1] + 1) as u8) << REF_NIBBLE_BITS),
            mode: flags.0 | if skip { SKIP_BIT } else { 0 },
        }
    }
    pub(crate) fn skip(self) -> bool {
        self.mode & SKIP_BIT != 0
    }
    pub(crate) fn y_mode(self) -> u8 {
        self.mode & Y_MODE_MASK
    }
    /// An intra block predicted with `SMOOTH_PRED`, `SMOOTH_V_PRED` or
    /// `SMOOTH_H_PRED` (the intra edge filter's `is_smooth`).
    pub(crate) fn smooth(self) -> bool {
        !self.is_inter() && (SMOOTH_MODES).contains(&self.y_mode())
    }
    pub(crate) fn is_inter(self) -> bool {
        self.refs & REF_NIBBLE_MASK > INTRA_NIBBLE
    }
    /// `RefFrame[0]`.
    fn first_reference(self) -> usize {
        usize::from(self.refs & REF_NIBBLE_MASK) - 1
    }
    /// A compound block: `RefFrame[1]` is a reference.
    fn is_compound(self) -> bool {
        self.refs >> REF_NIBBLE_BITS > INTRA_NIBBLE
    }
    /// Whether `RefFrame[0]` and `RefFrame[1]` point the same way in time
    /// (`is_samedir_ref_pair` of a compound block).
    fn unidirectional(self) -> bool {
        let second = usize::from(self.refs >> REF_NIBBLE_BITS) - 1;
        backward(self.first_reference()) == backward(second)
    }
    fn mode_bit(self, bit: u8) -> bool {
        self.is_inter() && self.mode & bit != 0
    }
    /// This block's references as [`ReferenceCounts`] nibbles.
    fn reference_nibbles(self) -> u32 {
        REFERENCE_NIBBLES[usize::from(self.refs)]
    }
}

/// `skip_mode`, `comp_group_idx` and `compound_idx` of an inter block,
/// packed as [`BlockInfo`] stores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InterFlags(u8);

impl InterFlags {
    /// A single-reference block: all clear (its `compound_idx` is never
    /// read).
    pub(crate) const SINGLE: Self = Self(0);
    /// A compound block's flags.
    pub(crate) fn new(skip_mode: bool, comp_group_idx: bool, compound_idx: bool) -> Self {
        let flag = |set: bool, bit: u8| if set { bit } else { 0 };
        Self(
            flag(skip_mode, SKIP_MODE_BIT)
                | flag(comp_group_idx, COMP_GROUP_BIT)
                | flag(compound_idx, COMPOUND_IDX_BIT),
        )
    }
    pub(crate) fn skip_mode(self) -> bool {
        self.0 & SKIP_MODE_BIT != 0
    }
}

/// A backward reference (`BWDREF_FRAME` ..= `ALTREF_FRAME`, §8.3.2
/// `check_backward`).
fn backward(name: usize) -> bool {
    (BWDREF_FRAME..=ALTREF_FRAME).contains(&name)
}

/// The [`ReferenceCounts`] nibble one use of the reference stored as
/// `nibble` adds (none for `NONE` or `INTRA_FRAME`).
const fn reference_count(nibble: u32) -> u32 {
    if nibble > INTRA_NIBBLE as u32 && nibble <= REFERENCE_NAMES as u32 {
        1 << (COUNT_BITS * (nibble - 1))
    } else {
        0
    }
}

/// [`BlockInfo::refs`] to [`ReferenceCounts`] nibbles: one count per
/// reference the block predicts from. Intra blocks count none, and
/// `RefFrame[1] = INTRA_FRAME` (inter-intra) counts as no reference.
const REFERENCE_NIBBLES: [u32; 256] = {
    let mut table = [0; 256];
    let mut refs = 0;
    while refs < table.len() {
        let first = refs as u32 & REF_NIBBLE_MASK as u32;
        let second = refs as u32 >> REF_NIBBLE_BITS;
        if first > INTRA_NIBBLE as u32 {
            table[refs] = reference_count(first) + reference_count(second);
        }
        refs += 1;
    }
    table
};

/// Most 4x4 rows in a superblock (128x128).
const MAX_SUPERBLOCK_MI: usize = 32;

/// The cells blocks read as neighbours (§8.3.2 reads only `[row - 1][col]`
/// and `[row][col - 1]`): the bottom row of everything decoded above, per
/// column, and the right column of what was decoded to the left, per row
/// of the superblock row.
#[derive(Debug)]
pub(crate) struct BlockContexts {
    above: Vec<BlockInfo>,
    left: [BlockInfo; MAX_SUPERBLOCK_MI],
}

/// The value cells hold before any block writes them (never read: the
/// frame edges make those neighbours unavailable).
const UNWRITTEN: BlockInfo = BlockInfo {
    size: BlockSize::BLOCK_8X8,
    refs: INTRA_NIBBLE,
    mode: 0,
};

impl BlockContexts {
    pub(crate) fn new() -> Self {
        Self {
            above: Vec::new(),
            left: [UNWRITTEN; MAX_SUPERBLOCK_MI],
        }
    }
    /// Shape the contexts for a frame. Stale cells are never read: a
    /// neighbour inside the frame was always decoded (and written) before
    /// the block asking for it, so the buffers are reused without clearing.
    /// The above row has a superblock of spare cells past the frame edge,
    /// so blocks there write whole without clipping.
    pub(crate) fn start_frame(&mut self, mi_cols: usize) {
        self.above.resize(mi_cols + MAX_SUPERBLOCK_MI, UNWRITTEN);
    }
    /// Record `info` for the block at (`mi_col`, `mi_row`). Cells past the
    /// frame edge are written too (never read), and a block never crosses
    /// its superblock, so the left column needs no clipping either.
    #[allow(
        clippy::inline_always,
        reason = "once per block; out of line it cost ~17M instructions on crowd 720p"
    )]
    #[inline(always)]
    pub(crate) fn fill(&mut self, mi_col: usize, mi_row: usize, info: BlockInfo) {
        splat(&mut self.above[mi_col..], info.size.width_mi_log2(), info);
        let first = mi_row % MAX_SUPERBLOCK_MI;
        splat(&mut self.left[first..], info.size.height_mi_log2(), info);
    }
    /// The above and left neighbours of the block at (`mi_col`, `mi_row`)
    /// where `available` (`AvailU`, `AvailL`).
    pub(crate) fn neighbours(
        &self,
        mi_col: usize,
        mi_row: usize,
        (above, left): (bool, bool),
    ) -> Neighbours {
        Neighbours {
            above: above.then(|| self.above[mi_col]),
            left: left.then(|| self.left[mi_row % MAX_SUPERBLOCK_MI]),
        }
    }
}

/// Write `info` into the first `1 << count_log2` cells as fixed-size
/// stores: blocks are mostly a few cells wide, where a general fill spends
/// more on its set-up than on the stores (dav1d's `case_set` does the same).
#[allow(
    clippy::inline_always,
    reason = "twice per block; out of line it cost ~17M instructions on crowd 720p"
)]
#[inline(always)]
fn splat(cells: &mut [BlockInfo], count_log2: u8, info: BlockInfo) {
    fn fixed<const N: usize>(cells: &mut [BlockInfo], info: BlockInfo) {
        let run: &mut [BlockInfo; N] = (&mut cells[..N]).try_into().expect("a slice of N cells");
        *run = [info; N];
    }
    match count_log2 {
        0 => fixed::<1>(cells, info),
        1 => fixed::<2>(cells, info),
        2 => fixed::<4>(cells, info),
        3 => fixed::<8>(cells, info),
        4 => fixed::<16>(cells, info),
        _ => fixed::<MAX_SUPERBLOCK_MI>(cells, info),
    }
}

/// The above and left cells of a block, where available.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Neighbours {
    pub above: Option<BlockInfo>,
    pub left: Option<BlockInfo>,
}

impl Neighbours {
    /// Partition context of a square `size` block (§8.3.2): whether the
    /// above neighbour is narrower and the left one shorter.
    pub(crate) fn partition_context(&self, size: BlockSize) -> usize {
        let bsl = size.width_mi_log2();
        let above = self.above.is_some_and(|a| a.size.width_mi_log2() < bsl);
        let left = self.left.is_some_and(|l| l.size.height_mi_log2() < bsl);
        usize::from(left) * 2 + usize::from(above)
    }
    /// `skip` context: skipped neighbours.
    pub(crate) fn skip_context(&self) -> usize {
        usize::from(self.above.is_some_and(BlockInfo::skip))
            + usize::from(self.left.is_some_and(BlockInfo::skip))
    }
    /// `is_inter` context (§8.3.2 `intra_frame` ctx).
    pub(crate) fn intra_inter_context(&self) -> usize {
        match (self.above, self.left) {
            (Some(a), Some(l)) => match (a.is_inter(), l.is_inter()) {
                (false, false) => 3,
                (true, true) => 0,
                _ => 1,
            },
            (Some(n), None) | (None, Some(n)) => 2 * usize::from(!n.is_inter()),
            (None, None) => 0,
        }
    }
    /// Key-frame `intra_frame_y_mode` contexts: `Intra_Mode_Context` of the
    /// above and left luma modes (DC where unavailable).
    pub(crate) fn y_mode_contexts(&self) -> (usize, usize) {
        let ctx = |n: Option<BlockInfo>| {
            n.map_or(0, |n| {
                usize::from(INTRA_MODE_CONTEXT[usize::from(n.y_mode())])
            })
        };
        (ctx(self.above), ctx(self.left))
    }
    /// `tx_depth` context (§8.3.2): neighbours whose transform reaches the
    /// block's largest transform width (above) or height (left) (an inter neighbour
    /// counts with its block width above, height to the left). `above_tx`
    /// and `left_tx` are the stored neighbouring transform width and height.
    pub(crate) fn tx_depth_context(
        &self,
        above_tx: usize,
        left_tx: usize,
        (max_width, max_height): (usize, usize),
    ) -> usize {
        let reaches = |n: BlockInfo, inter_side: usize, tx: usize, max: usize| {
            usize::from(if n.is_inter() { inter_side } else { tx } >= max)
        };
        let above = self
            .above
            .map(|a| reaches(a, a.size.width(), above_tx, max_width));
        let left = self
            .left
            .map(|l| reaches(l, l.size.height(), left_tx, max_height));
        match (above, left) {
            (Some(a), Some(l)) => a + l,
            (Some(n), None) | (None, Some(n)) => n,
            (None, None) => 0,
        }
    }
    /// `skip_mode` context: neighbours coded in skip mode.
    pub(crate) fn skip_mode_context(&self) -> usize {
        let skip_mode =
            |n: Option<BlockInfo>| usize::from(n.is_some_and(|n| n.mode_bit(SKIP_MODE_BIT)));
        skip_mode(self.above) + skip_mode(self.left)
    }
    /// `comp_mode` context (§8.3.2): whether the neighbours are compound,
    /// intra or predicted from backward references.
    pub(crate) fn comp_mode_context(&self) -> usize {
        // A single neighbour's term: backward or intra.
        let single = |n: BlockInfo| backward(n.first_reference()) || !n.is_inter();
        match (self.above, self.left) {
            (Some(a), Some(l)) => match (a.is_compound(), l.is_compound()) {
                (false, false) => {
                    usize::from(backward(a.first_reference()) ^ backward(l.first_reference()))
                }
                (false, true) => 2 + usize::from(single(a)),
                (true, false) => 2 + usize::from(single(l)),
                (true, true) => 4,
            },
            (Some(n), None) | (None, Some(n)) => {
                if n.is_compound() {
                    3
                } else {
                    usize::from(backward(n.first_reference()))
                }
            }
            (None, None) => COMP_MODE_NO_NEIGHBOURS,
        }
    }
    /// `comp_ref_type` context (§8.3.2): whether the neighbours are
    /// compound blocks of one direction.
    pub(crate) fn comp_ref_type_context(&self) -> usize {
        let compound = |n: Option<BlockInfo>| n.is_some_and(|n| n.is_inter() && n.is_compound());
        let unidirectional =
            |n: Option<BlockInfo>| compound(n) && n.is_some_and(BlockInfo::unidirectional);
        let (above, left) = (self.above, self.left);
        let (above_comp, left_comp) = (compound(above), compound(left));
        let (above_uni, left_uni) = (unidirectional(above), unidirectional(left));
        match (above, left) {
            (Some(a), Some(l)) if a.is_inter() && l.is_inter() => {
                let same_direction = backward(a.first_reference()) == backward(l.first_reference());
                match (above_comp, left_comp) {
                    (false, false) => 1 + 2 * usize::from(same_direction),
                    (false, true) => {
                        if left_uni {
                            3 + usize::from(same_direction)
                        } else {
                            1
                        }
                    }
                    (true, false) => {
                        if above_uni {
                            3 + usize::from(same_direction)
                        } else {
                            1
                        }
                    }
                    (true, true) => match (above_uni, left_uni) {
                        (false, false) => 0,
                        (true, true) => {
                            let bwd = |n: BlockInfo| n.first_reference() == BWDREF_FRAME;
                            3 + usize::from(bwd(a) == bwd(l))
                        }
                        _ => 2,
                    },
                }
            }
            (Some(_), Some(_)) => {
                if above_comp {
                    1 + 2 * usize::from(above_uni)
                } else if left_comp {
                    1 + 2 * usize::from(left_uni)
                } else {
                    COMP_REF_TYPE_DEFAULT
                }
            }
            _ => {
                if above_comp {
                    4 * usize::from(above_uni)
                } else if left_comp {
                    4 * usize::from(left_uni)
                } else {
                    COMP_REF_TYPE_DEFAULT
                }
            }
        }
    }
    /// `comp_group_idx` context (§8.3.2): compound neighbours' own
    /// `comp_group_idx`, single ones predicted from `ALTREF_FRAME`.
    pub(crate) fn comp_group_idx_context(&self) -> usize {
        let term = |n: Option<BlockInfo>| {
            n.map_or(0, |n| {
                if n.is_compound() {
                    usize::from(n.mode_bit(COMP_GROUP_BIT))
                } else if n.first_reference() == ALTREF_FRAME {
                    3
                } else {
                    0
                }
            })
        };
        (term(self.above) + term(self.left)).min(COMP_GROUP_MAX_CONTEXT)
    }
    /// `compound_idx` context (§8.3.2) of a block whose references are
    /// `equal_distance` from the frame: compound neighbours' own
    /// `compound_idx`, single ones predicted from `ALTREF_FRAME`.
    pub(crate) fn compound_idx_context(&self, equal_distance: bool) -> usize {
        let term = |n: Option<BlockInfo>| {
            n.map_or(0, |n| {
                if n.is_compound() {
                    usize::from(n.mode_bit(COMPOUND_IDX_BIT))
                } else {
                    usize::from(n.first_reference() == ALTREF_FRAME)
                }
            })
        };
        let base = if equal_distance {
            COMPOUND_IDX_EQUAL_DISTANCE
        } else {
            0
        };
        base + term(self.above) + term(self.left)
    }
    /// How often the neighbours use each reference name, counted once per
    /// inter block for all of its `single_ref_p*` contexts.
    pub(crate) fn reference_counts(&self) -> ReferenceCounts {
        let count = |n: Option<BlockInfo>| n.map_or(0, BlockInfo::reference_nibbles);
        ReferenceCounts(count(self.above) + count(self.left))
    }
}

/// Bits per reference name in [`ReferenceCounts`]: a group of up to four
/// names counts at most four references (two neighbours, two each), so
/// group sums fit a nibble too.
const COUNT_BITS: u32 = 4;
/// Adds every nibble of a word into its top nibble.
const NIBBLE_SUM: u32 = 0x1111_1111;
const TOP_NIBBLE_SHIFT: u32 = u32::BITS - COUNT_BITS;

/// The nibbles of the names in `group` (a name bit mask).
const fn group_nibbles(group: u8) -> u32 {
    let mut mask = 0;
    let mut name = 0;
    while name < REFERENCE_NAMES {
        if group & (1 << name) != 0 {
            mask |= ((1 << COUNT_BITS) - 1) << (COUNT_BITS as usize * name);
        }
        name += 1;
    }
    mask
}

/// [`SINGLE_REF_GROUPS`] as nibble masks.
const SINGLE_REF_NIBBLES: [(u32, u32); 6] = {
    let mut masks = [(0, 0); 6];
    let mut i = 0;
    while i < masks.len() {
        masks[i] = (
            group_nibbles(SINGLE_REF_GROUPS[i].0),
            group_nibbles(SINGLE_REF_GROUPS[i].1),
        );
        i += 1;
    }
    masks
};

/// The groups `uni_comp_ref_p1` compares, as nibble masks.
const UNI_COMP_REF_P1_NIBBLES: (u32, u32) = (
    group_nibbles(refs(&[LAST2_FRAME])),
    group_nibbles(refs(&[LAST3_FRAME, GOLDEN_FRAME])),
);

/// The neighbours' use of each reference name (§8.3.2 `count_refs`), one
/// nibble per name, so a group's count is a mask and a multiply.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReferenceCounts(u32);

impl ReferenceCounts {
    fn in_group(self, nibbles: u32) -> u32 {
        (self.0 & nibbles).wrapping_mul(NIBBLE_SUM) >> TOP_NIBBLE_SHIFT
    }
    /// Context of `single_ref_p<bit + 1>` (§8.3.2 `ref_count_ctx`), which
    /// the compound reference bits share: `comp_ref` (p3), `comp_ref_p1`
    /// (p4), `comp_ref_p2` and `uni_comp_ref_p2` (p5), `comp_bwdref` (p2),
    /// `comp_bwdref_p1` (p6) and `uni_comp_ref` (p1).
    pub(crate) fn single_ref_context(self, bit: usize) -> usize {
        self.compare(SINGLE_REF_NIBBLES[bit])
    }
    /// Context of `uni_comp_ref_p1` (§8.3.2): `LAST2_FRAME` against
    /// `LAST3_FRAME` and `GOLDEN_FRAME`.
    pub(crate) fn uni_comp_ref_p1_context(self) -> usize {
        self.compare(UNI_COMP_REF_P1_NIBBLES)
    }
    fn compare(self, (first, second): (u32, u32)) -> usize {
        match self.in_group(first).cmp(&self.in_group(second)) {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Greater => 2,
        }
    }
}

/// Whether the chroma of the blocks above and left is smooth-predicted
/// (the intra edge filter's chroma `is_smooth`, §7.11.2.8). Only blocks
/// carrying chroma write it, over their chroma's luma extent, so a
/// sub-8x8 block reads the block carrying the chroma above or left of it.
#[derive(Debug)]
pub(crate) struct ChromaSmooth {
    above: Vec<bool>,
    left: [bool; MAX_SUPERBLOCK_MI],
}

impl ChromaSmooth {
    pub(crate) fn new(mi_cols: usize) -> Self {
        Self {
            above: vec![false; mi_cols + MAX_SUPERBLOCK_MI],
            left: [false; MAX_SUPERBLOCK_MI],
        }
    }
    /// Record a block's chroma over luma units `w4` x `h4` from
    /// (`mi_col`, `mi_row`).
    pub(crate) fn fill(
        &mut self,
        (mi_col, mi_row): (usize, usize),
        (w4, h4): (usize, usize),
        smooth: bool,
    ) {
        self.above[mi_col..mi_col + w4].fill(smooth);
        let row = mi_row % MAX_SUPERBLOCK_MI;
        self.left[row..(row + h4).min(MAX_SUPERBLOCK_MI)].fill(smooth);
    }
    pub(crate) fn above(&self, mi_col: usize) -> bool {
        self.above[mi_col]
    }
    pub(crate) fn left(&self, mi_row: usize) -> bool {
        self.left[mi_row % MAX_SUPERBLOCK_MI]
    }
}

/// The transform width above and height left of each 4x4 unit (libaom
/// `above_txfm_context`, `left_txfm_context`): what the `tx_depth` and
/// `txfm_split` contexts compare against (§8.3.2). A skipped inter block
/// records its own size.
#[derive(Debug)]
pub(crate) struct TxContexts {
    above: Vec<u8>,
    left: [u8; MAX_SUPERBLOCK_MI],
}

/// The reset value of the transform contexts (`TX_SIZES_LARGEST`, 64).
const LARGEST_TX: u8 = 64;
/// Square transform classes below the largest (`TX_SIZES - 1`: 4x4 ..
/// 64x64 are 0 ..= 4).
const LARGEST_TX_CLASS: usize = 4;
/// `txfm_split` contexts per category (above and left flags).
const SPLIT_CONTEXTS_PER_CATEGORY: usize = 3;
/// The smallest transform side, log2.
const TX_4X4_LOG2: u8 = 2;

impl TxContexts {
    /// Contexts of a frame `mi_cols` units wide, reset (libaom
    /// `av1_zero_above_context`), with a superblock of spare cells past the
    /// edge, so writes there need no clipping.
    pub(crate) fn new(mi_cols: usize) -> Self {
        Self {
            above: vec![LARGEST_TX; mi_cols + MAX_SUPERBLOCK_MI],
            left: [LARGEST_TX; MAX_SUPERBLOCK_MI],
        }
    }
    /// Start of a superblock row: the left column is reset.
    pub(crate) fn start_row(&mut self) {
        self.left = [LARGEST_TX; MAX_SUPERBLOCK_MI];
    }
    /// Record `tx` (width, height in samples) over `w4` x `h4` units from
    /// (`mi_col`, `mi_row`).
    pub(crate) fn fill(
        &mut self,
        (mi_col, mi_row): (usize, usize),
        (w4, h4): (usize, usize),
        (width, height): (u8, u8),
    ) {
        self.above[mi_col..mi_col + w4].fill(width);
        let row = mi_row % MAX_SUPERBLOCK_MI;
        self.left[row..(row + h4).min(MAX_SUPERBLOCK_MI)].fill(height);
    }
    /// The transform width above unit column `mi_col`.
    pub(crate) fn above(&self, mi_col: usize) -> u8 {
        self.above[mi_col]
    }
    /// The transform height left of unit row `mi_row`.
    pub(crate) fn left(&self, mi_row: usize) -> u8 {
        self.left[mi_row % MAX_SUPERBLOCK_MI]
    }
    /// `txfm_split` context of a `tx_width_log2` x `tx_height_log2`
    /// transform at (`mi_col`, `mi_row`) of a block whose larger side is
    /// `block_side_log2` (libaom `txfm_partition_context`).
    pub(crate) fn split_context(
        &self,
        (mi_col, mi_row): (usize, usize),
        block_side_log2: u8,
        (tx_width_log2, tx_height_log2): (u8, u8),
    ) -> usize {
        let above = usize::from(u16::from(self.above(mi_col)) < 1 << tx_width_log2);
        let left = usize::from(u16::from(self.left(mi_row)) < 1 << tx_height_log2);
        let max_class = usize::from(block_side_log2.min(6) - TX_4X4_LOG2);
        let class = usize::from(tx_width_log2.max(tx_height_log2) - TX_4X4_LOG2);
        let category =
            usize::from(class != max_class && max_class > 1) + (LARGEST_TX_CLASS - max_class) * 2;
        category * SPLIT_CONTEXTS_PER_CATEGORY + above + left
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_pack_into_four_bytes() {
        assert_eq!(std::mem::size_of::<BlockInfo>(), 4);
        let info = BlockInfo::intra(BlockSize::BLOCK_8X8, true, 12);
        assert!(info.skip() && !info.is_inter());
        assert_eq!(info.y_mode(), 12);
        let inter = BlockInfo::inter(
            BlockSize::BLOCK_64X64,
            false,
            [ALTREF_FRAME as i8, -1],
            InterFlags::SINGLE,
        );
        assert!(!inter.skip() && inter.is_inter());
    }

    #[test]
    fn rectangular_neighbours_set_partition_context_per_side() {
        let mut grid = BlockContexts::new();
        grid.start_frame(16);
        // A 32x8 block above (wide, short) and an 8x32 block to the left.
        grid.fill(
            4,
            2,
            BlockInfo::intra(BlockSize::new(5, 3).unwrap(), false, 0),
        );
        grid.fill(
            2,
            4,
            BlockInfo::intra(BlockSize::new(3, 5).unwrap(), false, 0),
        );
        let n = grid.neighbours(4, 4, (true, true));
        // 16x16: above is not narrower (32 wide), left is not shorter.
        assert_eq!(n.partition_context(BlockSize::square(4)), 0);
        // 64x64: both are smaller.
        assert_eq!(n.partition_context(BlockSize::square(6)), 3);
    }

    #[test]
    fn reference_counts_compare_groups() {
        let size = BlockSize::BLOCK_8X8;
        let n = Neighbours {
            above: Some(BlockInfo::inter(
                size,
                false,
                [ALTREF_FRAME as i8, -1],
                InterFlags::SINGLE,
            )),
            left: Some(BlockInfo::inter(
                size,
                false,
                [LAST_FRAME as i8, -1],
                InterFlags::SINGLE,
            )),
        };
        let counts = n.reference_counts();
        assert_eq!(counts.single_ref_context(0), 1); // forward 1 = backward 1
        assert_eq!(counts.single_ref_context(2), 2); // LAST/LAST2 1 > 0
        let intra = Neighbours {
            above: Some(BlockInfo::intra(size, false, 0)),
            left: None,
        };
        assert_eq!(intra.reference_counts().single_ref_context(0), 1);
        assert_eq!(intra.intra_inter_context(), 2);
    }
}
