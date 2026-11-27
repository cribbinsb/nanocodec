//! Per-macroblock state kept for the whole picture: what later macroblocks
//! (context selection, intra and motion vector prediction) and the
//! deblocking filter read from their neighbours, and the neighbour
//! availability rules of §6.4 (a neighbour is available when decoded in
//! the same slice, which also covers slice groups and any slice order), of
//! pictures and of the macroblock pairs of MBAFF frames (§6.4.10, Table
//! 6-4), whose left neighbours change from one block row to the next when
//! the two pairs differ in field decoding.
//!
//! Blocks are addressed by their 4x4 position (`x4`, `y4`) inside the
//! macroblock; luma arrays are raster order (`y4 * 4 + x4`), chroma arrays
//! raster order of each component's 4x4 blocks: two per row (4:2:0, 4:2:2),
//! or four, as luma, where Cb and Cr are coded like luma (4:4:4).

use nc_encoder::Mv;

/// Intra 4x4 / 8x8 prediction mode used where a neighbour has none
/// (`Intra_4x4_DC`, §8.3.1.1).
pub(super) const DC_PRED_MODE: u8 = 2;
/// `total_coeff` assumed for an `I_PCM` neighbour (§9.2.1: `nN` = 16).
pub(super) const PCM_TOTAL_COEFF: u8 = 16;
/// `CodedBlockPattern` of an `I_PCM` macroblock as context selection reads it:
/// every luma 8x8 block and chroma DC + AC coded (§9.3.3.1.1.4).
pub(super) const PCM_CBP: u8 = 0x2f;
/// Mask of `CodedBlockPatternLuma` in `cbp`; chroma is `cbp >> 4`.
pub(super) const CBP_LUMA_MASK: u8 = 0x0f;
pub(super) const CBP_CHROMA_SHIFT: u8 = 4;

/// Inter partitioning of a P or B macroblock (Tables 7-13, 7-14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Partition {
    P16x16,
    P16x8,
    P8x16,
    P8x8,
}

/// Sub-macroblock partitioning of one 8x8 block of `P_8x8` or `B_8x8`
/// (Tables 7-17, 7-18).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SubPartition {
    P8x8,
    P8x4,
    P4x8,
    P4x4,
}

impl SubPartition {
    /// The partitioning of P `sub_mb_type` `code` (Table 7-17).
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::P8x8,
            1 => Self::P8x4,
            2 => Self::P4x8,
            3 => Self::P4x4,
            _ => return None,
        })
    }

    /// The number of sub-partitions.
    pub fn count(self) -> usize {
        match self {
            Self::P8x8 => 1,
            Self::P8x4 | Self::P4x8 => 2,
            Self::P4x4 => 4,
        }
    }

    /// Size in 4x4 blocks.
    pub fn size4(self) -> (usize, usize) {
        match self {
            Self::P8x8 => (2, 2),
            Self::P8x4 => (2, 1),
            Self::P4x8 => (1, 2),
            Self::P4x4 => (1, 1),
        }
    }
}

/// The macroblock type, by prediction.
///
/// The intra kinds come first and the skipped kinds next, so `is_intra`
/// and `is_skip` (read per neighbour and edge in hot paths) are one range
/// check each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MbKind {
    /// `I_NxN` with 4x4 or (`transform_size_8x8_flag`) 8x8 prediction.
    IntraNxN,
    /// `Intra_16x16` with its prediction mode (Table 7-11).
    Intra16x16(u8),
    IPcm,
    /// `SI` (Table 7-12): `Intra_4x4` prediction, reconstructed by §8.6.2.
    Si,
    PSkip,
    /// `B_Skip`: direct prediction, no residual.
    BSkip,
    Inter(Partition),
    /// `B_Direct_16x16`: direct prediction with a residual.
    BDirect16x16,
}

/// The two reference picture lists.
pub(super) const LISTS: usize = 2;

/// The state of one decoded macroblock.
#[derive(Debug, Clone, Copy)]
pub(super) struct MbState {
    /// Serial number of its slice: decoder-wide and increasing, so a
    /// macroblock is decoded in the current picture when this is at least
    /// the picture's first serial (stale states of earlier pictures need no
    /// reset); 0 never decoded.
    pub slice: u32,
    pub kind: MbKind,
    pub transform_8x8: bool,
    /// `CodedBlockPatternLuma | CodedBlockPatternChroma << 4`.
    pub cbp: u8,
    /// `QP_Y` (from `−QpBdOffsetY`, −12 at 10 bits, to 51).
    pub qp: i8,
    /// `intra_chroma_pred_mode` (0 for non-intra macroblocks).
    pub chroma_pred_mode: u8,
    /// Intra 4x4 / 8x8 prediction modes per 4x4 block (8x8 modes repeated);
    /// [`DC_PRED_MODE`] for other macroblocks.
    pub intra_modes: [u8; 16],
    /// `TotalCoeff(coeff_token)` (CAVLC) or the coded-block flag (CABAC) of
    /// each luma 4x4 block (AC only for `Intra_16x16`).
    pub total_coeff: [u8; 16],
    /// Likewise for the chroma AC blocks of Cb and Cr (4:2:0, 4:2:2), or
    /// their luma-like 4x4 blocks (4:4:4).
    pub chroma_total_coeff: [[u8; 16]; 2],
    /// CABAC `coded_block_flag` of the luma DC, Cb DC and Cr DC blocks.
    pub dc_coded: [bool; 3],
    /// Luma 4x4 blocks with non-zero coefficients for the deblocking filter
    /// (bit `y4 * 4 + x4`; all four of an 8x8 transform block).
    pub nonzero: u16,
    /// Motion vector of each 4x4 block, per list (zero where the list is
    /// not used; list 1's are stale in P and I macroblocks, see
    /// [`MbState::reset`]).
    pub mvs: [[Mv; 16]; LISTS],
    /// `ref_idx_l0` / `ref_idx_l1` of each 8x8 block (-1: intra, or the list
    /// not used).
    pub ref_idx: [[i8; 4]; LISTS],
    /// Identity of the picture each 8x8 block predicts from, per list (0:
    /// the list is not used).
    pub ref_pic: [[u32; 4]; LISTS],
    /// The 8x8 blocks predicted by direct prediction (bit `b8`).
    pub direct: u8,
    /// CABAC: absolute motion vector differences of each 4x4 block, per
    /// list, saturated (only sums up to 33 matter, §9.3.3.1.1.7); list 1's
    /// stale outside B slices.
    pub abs_mvd: [[[u8; 2]; 16]; LISTS],
    /// An inter macroblock whose 4x4 blocks all share their vectors and
    /// references (one prediction block; set by the macroblock layer).
    pub single_motion: bool,
    /// A field macroblock (every macroblock of a field picture): field
    /// scans and CABAC significance contexts (§8.5.6, Table 9-34).
    pub field: bool,
    /// `sub_mb_type` of each 8x8 block of a `P_8x8` macroblock (analysis).
    #[cfg(feature = "analysis")]
    pub sub_partitions: [SubPartition; 4],
}

impl MbState {
    /// A not-yet-decoded macroblock.
    pub const EMPTY: Self = Self {
        slice: 0,
        kind: MbKind::PSkip,
        transform_8x8: false,
        cbp: 0,
        qp: 0,
        chroma_pred_mode: 0,
        intra_modes: [DC_PRED_MODE; 16],
        total_coeff: [0; 16],
        chroma_total_coeff: [[0; 16]; 2],
        dc_coded: [false; 3],
        nonzero: 0,
        mvs: [[Mv::ZERO; 16]; LISTS],
        ref_idx: [[-1; 4]; LISTS],
        ref_pic: [[0; 4]; LISTS],
        direct: 0,
        abs_mvd: [[[0; 2]; 16]; LISTS],
        single_motion: false,
        field: false,
        #[cfg(feature = "analysis")]
        sub_partitions: [SubPartition::P8x8; 4],
    };

    /// Reset to [`MbState::EMPTY`] for a macroblock of slice `slice`,
    /// except outside B slices (`b_slice` false) list 1's vectors and
    /// vector differences, which stay stale: the per-macroblock reset is a
    /// hot copy (resetting them as well cost 1.2 % of the decode
    /// instructions of P streams). They are read only where `ref_idx[1]`
    /// is non-negative (prediction, deblocking, the co-located block) or
    /// from neighbours in the same B slice, which are reset whole. `field`:
    /// a field macroblock.
    pub fn reset(&mut self, slice: u32, b_slice: bool, field: bool) {
        if b_slice {
            *self = Self {
                slice,
                field,
                ..Self::EMPTY
            };
            return;
        }
        let empty = &Self::EMPTY;
        // Every field named, so a new one cannot be missed.
        let Self {
            slice: slice_no,
            kind,
            transform_8x8,
            cbp,
            qp,
            chroma_pred_mode,
            intra_modes,
            total_coeff,
            chroma_total_coeff,
            dc_coded,
            nonzero,
            mvs,
            ref_idx,
            ref_pic,
            direct,
            abs_mvd,
            single_motion,
            field: field_slot,
            #[cfg(feature = "analysis")]
            sub_partitions,
        } = self;
        *slice_no = slice;
        *kind = empty.kind;
        *transform_8x8 = empty.transform_8x8;
        *cbp = empty.cbp;
        *qp = empty.qp;
        *chroma_pred_mode = empty.chroma_pred_mode;
        *intra_modes = empty.intra_modes;
        *total_coeff = empty.total_coeff;
        *chroma_total_coeff = empty.chroma_total_coeff;
        *dc_coded = empty.dc_coded;
        *nonzero = empty.nonzero;
        mvs[0] = empty.mvs[0];
        *ref_idx = empty.ref_idx;
        *ref_pic = empty.ref_pic;
        *direct = empty.direct;
        abs_mvd[0] = empty.abs_mvd[0];
        *single_motion = empty.single_motion;
        *field_slot = field;
        #[cfg(feature = "analysis")]
        {
            *sub_partitions = empty.sub_partitions;
        }
    }

    pub fn is_intra(&self) -> bool {
        matches!(
            self.kind,
            MbKind::IntraNxN | MbKind::Intra16x16(_) | MbKind::IPcm | MbKind::Si
        )
    }

    /// `I_NxN` or `SI`: macroblocks with `Intra_4x4` / `Intra_8x8`
    /// prediction modes per block (§8.3.1.1).
    pub fn has_block_modes(&self) -> bool {
        matches!(self.kind, MbKind::IntraNxN | MbKind::Si)
    }

    /// `P_Skip` or `B_Skip`.
    pub fn is_skip(&self) -> bool {
        matches!(self.kind, MbKind::PSkip | MbKind::BSkip)
    }

    /// Whether any block predicts from list 1 (a B macroblock that is not
    /// list-0-only).
    pub fn uses_l1(&self) -> bool {
        self.ref_idx[1] != [-1; 4]
    }

    pub fn cbp_luma(&self) -> u8 {
        self.cbp & CBP_LUMA_MASK
    }

    pub fn cbp_chroma(&self) -> u8 {
        self.cbp >> CBP_CHROMA_SHIFT
    }

    /// `TotalCoeff` / coded flags of the luma-like 4x4 blocks of `plane`:
    /// luma, or Cb and Cr coded like luma (4:4:4).
    pub fn plane_total_coeff(&self, plane: usize) -> &[u8; 16] {
        if plane == 0 {
            &self.total_coeff
        } else {
            &self.chroma_total_coeff[plane - 1]
        }
    }

    /// [`MbState::plane_total_coeff`], to write.
    pub fn plane_total_coeff_mut(&mut self, plane: usize) -> &mut [u8; 16] {
        if plane == 0 {
            &mut self.total_coeff
        } else {
            &mut self.chroma_total_coeff[plane - 1]
        }
    }

    /// Motion vector and reference index of list `list` of the 4x4 block
    /// (`x4`, `y4`).
    pub fn motion(&self, list: usize, x4: usize, y4: usize) -> (Mv, i8) {
        (
            self.mvs[list][y4 * 4 + x4],
            self.ref_idx[list][(y4 / 2) * 2 + x4 / 2],
        )
    }
}

/// The available neighbouring macroblocks of the current one (§6.4.9:
/// inside the picture and in the same slice): A left, B above, C above
/// right, D above left (in MBAFF frames, those of luma locations (−1, 0),
/// (0, −1), (16, −1) and (−1, −1), Table 6-4).
#[derive(Debug, Clone, Copy)]
pub(super) struct Neighbours<'a> {
    pub a: Option<&'a MbState>,
    pub b: Option<&'a MbState>,
    pub c: Option<&'a MbState>,
    pub d: Option<&'a MbState>,
    /// In MBAFF frames, what differs from a picture's neighbours.
    pub pair: Option<PairNeighbours<'a>>,
}

/// The neighbours of a macroblock of an MBAFF frame beyond A..D (Table
/// 6-4): its coding and place in its pair, and the left pair, from which
/// the left macroblock and row of each 4x4 block row, the row of D's corner
/// and the up-left neighbours of the left column follow. Neighbours coded
/// the other way (frame or field) than the current macroblock have their
/// vectors and reference indices scaled where they are read (§8.4.1.3.1,
/// §9.3.3.1.1.6/7).
#[derive(Debug, Clone, Copy)]
pub(super) struct PairNeighbours<'a> {
    /// The current macroblock is a field macroblock.
    pub field: bool,
    /// The current macroblock is the bottom one of its pair.
    pub bottom: bool,
    /// The left pair (top, bottom), if available.
    pub left_pair: Option<(&'a MbState, &'a MbState)>,
}

impl<'a> PairNeighbours<'a> {
    /// The left macroblock and 4x4 row of 4x4 row `y4` (luma location
    /// (−1, 4 · y4)). Out of line: the callers' picture paths stay small.
    #[inline(never)]
    pub fn left(&self, y4: usize) -> Option<(&'a MbState, usize)> {
        let (upper, lower) = self.left_pair?;
        let bottom = usize::from(self.bottom);
        Some(match (self.field, upper.field) {
            // Coded alike: the macroblock in the same position.
            (false, false) | (true, true) => (if self.bottom { lower } else { upper }, y4),
            // A frame macroblock beside a field pair: rows of the top field
            // macroblock (an even yN; yM = (yN + 16 · bottom) >> 1).
            (false, true) => (upper, y4 / 2 + 2 * bottom),
            // A field macroblock beside a frame pair: frame row 2 · yN +
            // bottom, the upper half in the top frame macroblock.
            (true, false) => (if y4 < 2 { upper } else { lower }, (2 * y4) % 4),
        })
    }

    /// Where the left pair is coded the other way: the macroblock and 4x4
    /// row up-left of left-column block row `y4` (1..=3; luma location
    /// (−1, 4 · y4 − 1), on the other field's rows than the left of the
    /// row above). `None` where it is the left of the row above.
    pub fn corner(&self, y4: usize) -> Option<(&'a MbState, usize)> {
        let (upper, lower) = self.left_pair?;
        let bottom = usize::from(self.bottom);
        match (self.field, upper.field) {
            (false, false) | (true, true) => None,
            // An odd yN: the bottom field, row (yN + 16 · bottom) >> 1.
            (false, true) => Some((lower, (2 * y4 - 1 + 8 * bottom) / 4)),
            (true, false) => {
                let row = 8 * y4 - 2 + bottom;
                Some((if row < 16 { upper } else { lower }, (row % 16) / 4))
            }
        }
    }

    /// Whether the left pair is coded the other way.
    pub fn split_left(&self) -> bool {
        self.left_pair
            .is_some_and(|(upper, _)| upper.field != self.field)
    }

    /// The 4x4 row of D's corner block: row 1 of the bottom field
    /// macroblock beside a bottom frame macroblock, else the last.
    pub fn d_row(&self) -> usize {
        match self.left_pair {
            Some((upper, _)) if !self.field && self.bottom && upper.field => 1,
            _ => 3,
        }
    }
}

impl<'a> Neighbours<'a> {
    /// The neighbours of macroblock `addr` of slice `slice` in `mbs`
    /// (`width_mbs` per row).
    pub fn of(mbs: &'a [MbState], width_mbs: usize, addr: usize, slice: u32) -> Self {
        let x = addr % width_mbs;
        let (left, right, up) = (x > 0, x + 1 < width_mbs, addr >= width_mbs);
        let at = |available: bool, index: usize| -> Option<&'a MbState> {
            if !available {
                return None;
            }
            let mb = &mbs[index];
            (mb.slice == slice).then_some(mb)
        };
        // The indices are only formed (and read) where the neighbour lies
        // inside the picture.
        let above = addr.wrapping_sub(width_mbs);
        Self {
            a: at(left, addr.wrapping_sub(1)),
            b: at(up, above),
            c: at(up && right, above.wrapping_add(1)),
            d: at(up && left, above.wrapping_sub(1)),
            pair: None,
        }
    }

    /// The neighbours of macroblock `addr` of an MBAFF frame (`2 · pair +
    /// bottom`; pairs `width_mbs` per row), a `field` macroblock or not, of
    /// slice `slice` (§6.4.10, Table 6-4). Every neighbour precedes `addr`
    /// in `mbs`.
    pub fn of_pair(
        mbs: &'a [MbState],
        width_mbs: usize,
        (addr, field): (usize, bool),
        slice: u32,
    ) -> Self {
        type Pair<'p> = (&'p MbState, &'p MbState);
        let pair = addr / 2;
        let bottom = addr % 2 == 1;
        let column = pair % width_mbs;
        let (left, right, up) = (column > 0, column + 1 < width_mbs, pair >= width_mbs);
        // The (top, bottom) macroblocks of a neighbouring pair, if
        // available (a pair is in one slice).
        let pair_at = |available: bool, index: usize| -> Option<Pair<'a>> {
            if !available {
                return None;
            }
            let top = &mbs[2 * index];
            (top.slice == slice).then(|| (top, &mbs[2 * index + 1]))
        };
        let above = pair.wrapping_sub(width_mbs);
        let left_pair = pair_at(left, pair.wrapping_sub(1));
        let above_pair = pair_at(up, above);
        let above_right = pair_at(up && right, above.wrapping_add(1));
        let above_left = pair_at(up && left, above.wrapping_sub(1));
        // Above: a frame macroblock sees the bottom one of the pair above;
        // a top field macroblock the top one of a field pair, else the
        // bottom one.
        let above_mb =
            |p: Option<Pair<'a>>| p.map(|(t, b)| if field && !bottom && t.field { t } else { b });
        let (b_mb, c_mb, d_mb) = if !field && bottom {
            // The top macroblock of the pair is above; C is not available;
            // D is the left pair's top frame macroblock (row 15), or its
            // bottom field macroblock (row 7, `PairNeighbours::d_row`).
            let d_mb = left_pair.map(|(upper, lower)| if lower.field { lower } else { upper });
            (Some(&mbs[addr - 1]), None, d_mb)
        } else {
            (
                above_mb(above_pair),
                above_mb(above_right),
                above_mb(above_left),
            )
        };
        let pair = PairNeighbours {
            field,
            bottom,
            left_pair,
        };
        Self {
            a: pair.left(0).map(|(m, _)| m),
            b: b_mb,
            c: c_mb,
            d: d_mb,
            pair: Some(pair),
        }
    }

    /// The macroblock and 4x4 block row left of row `y4` of the current
    /// macroblock (its column 3), if available.
    #[inline]
    pub fn left_row(&self, y4: usize) -> Option<(&'a MbState, usize)> {
        match &self.pair {
            None => self.a.map(|m| (m, y4)),
            Some(pair) => pair.left(y4),
        }
    }

    /// The macroblock and 8x8 block row left of 8x8 row `y8`.
    pub fn left_row_8x8(&self, y8: usize) -> Option<(&'a MbState, usize)> {
        self.left_row(2 * y8).map(|(m, row)| (m, row / 2))
    }

    /// The macroblock and 4:2:0 chroma 4x4 block row left of chroma row
    /// `y` (0 or 1): the luma 8x8 row's (Table 6-4 at chroma sizes).
    pub fn left_row_chroma(&self, y: usize) -> Option<(&'a MbState, usize)> {
        self.left_row_8x8(y)
    }

    /// Where `n` is coded the other way (frame or field) than the current
    /// macroblock of an MBAFF frame: `Some(current is a field macroblock)`.
    #[inline]
    pub fn coded_otherwise(&self, n: &MbState) -> Option<bool> {
        self.pair
            .as_ref()
            .and_then(|p| (p.field != n.field).then_some(p.field))
    }

    /// The macroblock and block position of the 4x4 block left of
    /// (`x4`, `y4`) of `current` (§6.4.11.4), if available.
    pub fn left<'b>(
        &self,
        current: &'b MbState,
        x4: usize,
        y4: usize,
    ) -> Option<(&'b MbState, usize, usize)>
    where
        'a: 'b,
    {
        if x4 > 0 {
            Some((current, x4 - 1, y4))
        } else {
            self.left_row(y4).map(|(mb, row)| (mb, 3, row))
        }
    }

    /// The 4x4 block above (`x4`, `y4`) of `current`, if available.
    pub fn above<'b>(
        &self,
        current: &'b MbState,
        x4: usize,
        y4: usize,
    ) -> Option<(&'b MbState, usize, usize)>
    where
        'a: 'b,
    {
        if y4 > 0 {
            Some((current, x4, y4 - 1))
        } else {
            self.b.map(|mb| (mb, x4, 3))
        }
    }
}

/// The blocks (bit `y * side + x`) of a `w`x`h` rectangle at (`x`, `y`) in
/// a `side`x`side` grid (`side` 2 or 4). Computed rather than looped over:
/// partition sizes vary from macroblock to macroblock, and loops with
/// varying trip counts mispredict their exits.
pub(super) fn partition_blocks(x: usize, y: usize, w: usize, h: usize, side: usize) -> u16 {
    let row = ((1u16 << w) - 1) << x;
    // Bit 0 of each covered row; the product then copies `row` into each
    // (no carries: `row` fits in one row's bits).
    let first_bits: u16 = if side == 4 { 0x1111 } else { 0b0101 };
    let rows = (first_bits >> (side * (side - h))) << (side * y);
    row * rows
}

/// The decoding order index of a 4x4 block (`luma4x4BlkIdx`, §6.4.3).
pub(super) fn decode_index(x4: usize, y4: usize) -> usize {
    (y4 / 2) * 8 + (x4 / 2) * 4 + (y4 % 2) * 2 + x4 % 2
}

/// The (`x4`, `y4`) position of 4x4 block `luma4x4BlkIdx` (eq. 6-17, 6-18).
pub(super) fn block_position(index: usize) -> (usize, usize) {
    let (b8, b4) = (index / 4, index % 4);
    ((b8 % 2) * 2 + b4 % 2, (b8 / 2) * 2 + b4 / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_blocks_cover_exactly_the_rectangle() {
        for side in [2, 4] {
            for (x, y) in (0..side).flat_map(|y| (0..side).map(move |x| (x, y))) {
                for (w, h) in (1..=side - x).flat_map(|w| (1..=side - y).map(move |h| (w, h))) {
                    let expected = (0..side * side)
                        .filter(|b| {
                            (x..x + w).contains(&(b % side)) && (y..y + h).contains(&(b / side))
                        })
                        .fold(0u16, |m, b| m | 1 << b);
                    assert_eq!(
                        partition_blocks(x, y, w, h, side),
                        expected,
                        "{side} {x} {y} {w} {h}"
                    );
                }
            }
        }
    }

    /// Per 4x4 row (left is the top macroblock, row), and the corners.
    type PairRows = ([(bool, usize); 4], Option<[(bool, usize); 3]>);

    /// The rows Table 6-4 gives a macroblock of the second pair of a row
    /// whose left pair is coded the other way: `(left is the top one, row)`
    /// per 4x4 row, and the corners.
    fn pair_rows(field: bool, bottom: bool) -> PairRows {
        let mut mbs = vec![MbState::EMPTY; 4];
        for (i, mb) in mbs.iter_mut().enumerate() {
            mb.slice = 1;
            mb.field = i < 2 && !field;
        }
        let addr = 2 + usize::from(bottom);
        let nb = Neighbours::of_pair(&mbs[..addr], 2, (addr, field), 1);
        let pair = nb.pair.unwrap();
        let at = |(m, row): (&MbState, usize)| (core::ptr::eq(m, &raw const mbs[0]), row);
        (
            core::array::from_fn(|y4| at(pair.left(y4).unwrap())),
            pair.split_left()
                .then(|| core::array::from_fn(|i| at(pair.corner(i + 1).unwrap()))),
        )
    }

    #[test]
    fn mbaff_left_neighbours_follow_table_6_4() {
        // A frame macroblock beside a field pair: the top field
        // macroblock's rows yN / 2 (+ 8 for the bottom one); its corners
        // (odd yN) in the bottom field macroblock.
        let (left, corners) = pair_rows(false, true);
        assert_eq!(left, [(true, 2), (true, 2), (true, 3), (true, 3)]);
        assert_eq!(corners, Some([(false, 2), (false, 2), (false, 3)]));
        // A bottom field macroblock beside a frame pair: frame rows 2 · yN
        // + 1, the upper half in the top frame macroblock.
        let (left, corners) = pair_rows(true, true);
        assert_eq!(left, [(true, 0), (true, 2), (false, 0), (false, 2)]);
        assert_eq!(corners, Some([(true, 1), (true, 3), (false, 1)]));
    }

    #[test]
    fn decoding_order_round_trips() {
        for index in 0..16 {
            let (x, y) = block_position(index);
            assert_eq!(decode_index(x, y), index);
        }
    }
}
