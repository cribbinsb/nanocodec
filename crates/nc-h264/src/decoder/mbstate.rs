//! Per-macroblock state kept for the whole picture: what later macroblocks
//! (context selection, intra and motion vector prediction) and the
//! deblocking filter read from their neighbours, and the neighbour
//! availability rules of §6.4 for frames without slice groups.
//!
//! Blocks are addressed by their 4x4 position (`x4`, `y4`) inside the
//! macroblock; luma arrays are raster order (`y4 * 4 + x4`), chroma arrays
//! raster order of the 2x2 4x4 blocks of each component.

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

/// Inter partitioning of a P macroblock (Table 7-13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Partition {
    P16x16,
    P16x8,
    P8x16,
    P8x8,
}

/// Sub-macroblock partitioning of one 8x8 block of `P_8x8` (Table 7-17).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SubPartition {
    P8x8,
    P8x4,
    P4x8,
    P4x4,
}

impl SubPartition {
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::P8x8,
            1 => Self::P8x4,
            2 => Self::P4x8,
            3 => Self::P4x4,
            _ => return None,
        })
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

    pub fn count(self) -> usize {
        let (w, h) = self.size4();
        4 / (w * h)
    }
}

/// The macroblock type, by prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MbKind {
    /// `I_NxN` with 4x4 or (`transform_size_8x8_flag`) 8x8 prediction.
    IntraNxN,
    /// `Intra_16x16` with its prediction mode (Table 7-11).
    Intra16x16(u8),
    IPcm,
    Inter(Partition),
    PSkip,
}

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
    /// `QP_Y`.
    pub qp: u8,
    /// `intra_chroma_pred_mode` (0 for non-intra macroblocks).
    pub chroma_pred_mode: u8,
    /// Intra 4x4 / 8x8 prediction modes per 4x4 block (8x8 modes repeated);
    /// [`DC_PRED_MODE`] for other macroblocks.
    pub intra_modes: [u8; 16],
    /// `TotalCoeff(coeff_token)` (CAVLC) or the coded-block flag (CABAC) of
    /// each luma 4x4 block (AC only for `Intra_16x16`).
    pub total_coeff: [u8; 16],
    /// Likewise for the chroma AC blocks of Cb and Cr.
    pub chroma_total_coeff: [[u8; 4]; 2],
    /// CABAC `coded_block_flag` of the luma DC, Cb DC and Cr DC blocks.
    pub dc_coded: [bool; 3],
    /// Luma 4x4 blocks with non-zero coefficients for the deblocking filter
    /// (bit `y4 * 4 + x4`; all four of an 8x8 transform block).
    pub nonzero: u16,
    /// Motion vector of each 4x4 block.
    pub mvs: [Mv; 16],
    /// `ref_idx_l0` of each 8x8 block (-1: intra).
    pub ref_idx: [i8; 4],
    /// Identity of the picture each 8x8 block predicts from.
    pub ref_pic: [u32; 4],
    /// CABAC: absolute motion vector differences of each 4x4 block,
    /// saturated (only sums up to 33 matter, §9.3.3.1.1.7).
    pub abs_mvd: [[u8; 2]; 16],
    /// An inter macroblock whose 4x4 blocks all share one vector and
    /// reference (one prediction block; set by the macroblock layer).
    pub single_motion: bool,
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
        chroma_total_coeff: [[0; 4]; 2],
        dc_coded: [false; 3],
        nonzero: 0,
        mvs: [Mv::ZERO; 16],
        ref_idx: [-1; 4],
        ref_pic: [0; 4],
        abs_mvd: [[0; 2]; 16],
        single_motion: false,
        #[cfg(feature = "analysis")]
        sub_partitions: [SubPartition::P8x8; 4],
    };

    pub fn is_intra(&self) -> bool {
        matches!(
            self.kind,
            MbKind::IntraNxN | MbKind::Intra16x16(_) | MbKind::IPcm
        )
    }

    pub fn is_skip(&self) -> bool {
        self.kind == MbKind::PSkip
    }

    pub fn cbp_luma(&self) -> u8 {
        self.cbp & CBP_LUMA_MASK
    }

    pub fn cbp_chroma(&self) -> u8 {
        self.cbp >> CBP_CHROMA_SHIFT
    }

    /// Motion vector and reference index of the 4x4 block (`x4`, `y4`).
    pub fn motion(&self, x4: usize, y4: usize) -> (Mv, i8) {
        (self.mvs[y4 * 4 + x4], self.ref_idx[(y4 / 2) * 2 + x4 / 2])
    }
}

/// The available neighbouring macroblocks of the current one (§6.4.9:
/// inside the picture and in the same slice): A left, B above, C above
/// right, D above left.
#[derive(Debug, Clone, Copy)]
pub(super) struct Neighbours<'a> {
    pub a: Option<&'a MbState>,
    pub b: Option<&'a MbState>,
    pub c: Option<&'a MbState>,
    pub d: Option<&'a MbState>,
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
        }
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
            self.a.map(|mb| (mb, 3, y4))
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

    #[test]
    fn decoding_order_round_trips() {
        for index in 0..16 {
            let (x, y) = block_position(index);
            assert_eq!(decode_index(x, y), index);
        }
    }
}
