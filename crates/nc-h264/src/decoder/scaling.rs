//! Scaling matrices: `scaling_list()` syntax (§7.3.2.1.1.1), the default
//! lists (Tables 7-3, 7-4), the fall-back rules of Table 7-2, and the
//! `LevelScale` factors of the dequantisation (§8.5.9).
//!
//! Lists are numbered as in Table 7-2: 4x4 lists 0..=5 are Intra Y, Intra Cb,
//! Intra Cr, Inter Y, Inter Cb, Inter Cr; 8x8 lists 0 and 1 (Table 7-2's 6
//! and 7) are Intra Y and Inter Y. 4:2:0 and 4:0:0 have no 8x8 chroma lists.

use super::Result;
use super::reader::Reader;
use crate::tables::{ZIGZAG_4X4, ZIGZAG_8X8};

/// Number of 4x4 and 8x8 scaling lists of 4:2:0 (and 4:0:0).
pub(super) const LISTS_4X4: usize = 6;
pub(super) const LISTS_8X8: usize = 2;
/// First Inter 4x4 list; the 8x8 Inter list is 1.
const FIRST_INTER_4X4: usize = 3;
/// `Flat_4x4_16` / `Flat_8x8_16` (§7.4.2.1.1).
const FLAT: u8 = 16;
/// `QP % 6` classes of the dequantisation.
const QP_PERIOD: usize = 6;

/// Table 7-3, `Default_4x4_Intra` and `Default_4x4_Inter`, in zig-zag order.
const DEFAULT_4X4: [[u8; 16]; 2] = [
    [
        6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
    ],
    [
        10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
    ],
];

/// Table 7-4, `Default_8x8_Intra` and `Default_8x8_Inter`, in zig-zag order.
const DEFAULT_8X8: [[u8; 64]; 2] = [
    [
        6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23, 23, 23, 23, 23, 23, 25, 25,
        25, 25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31,
        31, 31, 31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
    ],
    [
        9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 22, 22,
        22, 22, 22, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27,
        27, 27, 27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
    ],
];

/// `normAdjust4x4` (eq. 8-315): `v` per `QP % 6`, columns for positions with
/// both coordinates even, both odd, and mixed.
const NORM_ADJUST_4X4: [[u16; 3]; QP_PERIOD] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// `normAdjust8x8` (eq. 8-318): `v` per `QP % 6`, six position classes.
const NORM_ADJUST_8X8: [[u16; 6]; QP_PERIOD] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// One parsed `scaling_list()` (in zig-zag order) or its absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParsedList<const N: usize> {
    /// `scaling_list_present_flag` 0: a fall-back rule applies.
    Absent,
    /// `useDefaultScalingMatrixFlag`: the Table 7-3 / 7-4 default.
    UseDefault,
    /// Explicit weights.
    Explicit([u8; N]),
}

/// The scaling lists an SPS or PPS carries, before fall-back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ParsedMatrix {
    pub lists_4x4: [ParsedList<16>; LISTS_4X4],
    pub lists_8x8: [ParsedList<64>; LISTS_8X8],
}

/// Resolved weights (zig-zag order) of the 4x4 and 8x8 lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScalingMatrix {
    pub lists_4x4: [[u8; 16]; LISTS_4X4],
    pub lists_8x8: [[u8; 64]; LISTS_8X8],
}

impl ScalingMatrix {
    /// `Flat_4x4_16` and `Flat_8x8_16` everywhere: no scaling matrix.
    pub const FLAT: Self = Self {
        lists_4x4: [[FLAT; 16]; LISTS_4X4],
        lists_8x8: [[FLAT; 64]; LISTS_8X8],
    };
}

/// Parse `count_4x4` 4x4 and `count_8x8` 8x8 `scaling_list()`s, each behind
/// its `scaling_list_present_flag` (§7.3.2.1.1, §7.3.2.2).
pub(super) fn parse_matrix(r: &mut Reader<'_>, count_8x8: usize) -> Result<ParsedMatrix> {
    let mut matrix = ParsedMatrix {
        lists_4x4: [ParsedList::Absent; LISTS_4X4],
        lists_8x8: [ParsedList::Absent; LISTS_8X8],
    };
    for list in &mut matrix.lists_4x4 {
        if r.flag()? {
            *list = parse_list(r)?;
        }
    }
    for list in &mut matrix.lists_8x8[..count_8x8] {
        if r.flag()? {
            *list = parse_list(r)?;
        }
    }
    Ok(matrix)
}

/// `scaling_list(scalingList, sizeOfScalingList, useDefaultScalingMatrixFlag)`.
fn parse_list<const N: usize>(r: &mut Reader<'_>) -> Result<ParsedList<N>> {
    let mut list = [0u8; N];
    let (mut last, mut next) = (8i32, 8i32);
    for (j, weight) in list.iter_mut().enumerate() {
        if next != 0 {
            let delta = r.se_range(-128, 127, "delta_scale")?;
            next = (last + delta + 256) % 256;
            if j == 0 && next == 0 {
                return Ok(ParsedList::UseDefault);
            }
        }
        *weight = if next == 0 { last } else { next } as u8;
        last = i32::from(*weight);
    }
    Ok(ParsedList::Explicit(list))
}

/// Resolve `parsed` with Table 7-2's fall-back rule A (`fallback` = `None`:
/// the defaults) or rule B (`fallback` = the SPS matrix).
pub(super) fn resolve(parsed: &ParsedMatrix, fallback: Option<&ScalingMatrix>) -> ScalingMatrix {
    let mut out = ScalingMatrix::FLAT;
    for i in 0..LISTS_4X4 {
        let kind = usize::from(i >= FIRST_INTER_4X4);
        out.lists_4x4[i] = match parsed.lists_4x4[i] {
            ParsedList::Explicit(list) => list,
            ParsedList::UseDefault => DEFAULT_4X4[kind],
            // The first list of each kind falls back to the default (rule A)
            // or the SPS list (rule B); the chroma lists to the previous list.
            ParsedList::Absent if i % FIRST_INTER_4X4 == 0 => {
                fallback.map_or(DEFAULT_4X4[kind], |sps| sps.lists_4x4[i])
            }
            ParsedList::Absent => out.lists_4x4[i - 1],
        };
    }
    for (i, (list, default)) in out.lists_8x8.iter_mut().zip(DEFAULT_8X8).enumerate() {
        *list = match parsed.lists_8x8[i] {
            ParsedList::Explicit(list) => list,
            ParsedList::UseDefault => default,
            ParsedList::Absent => fallback.map_or(default, |sps| sps.lists_8x8[i]),
        };
    }
    out
}

/// `LevelScale4x4` / `LevelScale8x8` (eqs. 8-314, 8-317) of every list and
/// `QP % 6`, in raster order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LevelScale {
    pub scale_4x4: [[[i32; 16]; QP_PERIOD]; LISTS_4X4],
    pub scale_8x8: [[[i32; 64]; QP_PERIOD]; LISTS_8X8],
}

impl LevelScale {
    pub fn new(matrix: &ScalingMatrix) -> Self {
        let mut scale = Self {
            scale_4x4: [[[0; 16]; QP_PERIOD]; LISTS_4X4],
            scale_8x8: [[[0; 64]; QP_PERIOD]; LISTS_8X8],
        };
        for (list, weights) in scale.scale_4x4.iter_mut().zip(&matrix.lists_4x4) {
            for (m, out) in list.iter_mut().enumerate() {
                for (k, &pos) in ZIGZAG_4X4.iter().enumerate() {
                    let (x, y) = (pos % 4, pos / 4);
                    let class = match (x % 2, y % 2) {
                        (0, 0) => 0,
                        (1, 1) => 1,
                        _ => 2,
                    };
                    out[pos] = i32::from(weights[k]) * i32::from(NORM_ADJUST_4X4[m][class]);
                }
            }
        }
        for (list, weights) in scale.scale_8x8.iter_mut().zip(&matrix.lists_8x8) {
            for (m, out) in list.iter_mut().enumerate() {
                for (k, &pos) in ZIGZAG_8X8.iter().enumerate() {
                    let class = norm_adjust_8x8_class(pos % 8, pos / 8);
                    out[pos] = i32::from(weights[k]) * i32::from(NORM_ADJUST_8X8[m][class]);
                }
            }
        }
        scale
    }
}

/// The column of [`NORM_ADJUST_8X8`] for raster position (`x`, `y`) (eq. 8-318).
fn norm_adjust_8x8_class(x: usize, y: usize) -> usize {
    if x.is_multiple_of(4) && y.is_multiple_of(4) {
        0
    } else if x % 2 == 1 && y % 2 == 1 {
        1
    } else if x % 4 == 2 && y % 4 == 2 {
        2
    } else if (x.is_multiple_of(4) && y % 2 == 1) || (x % 2 == 1 && y.is_multiple_of(4)) {
        3
    } else if (x.is_multiple_of(4) && y % 4 == 2) || (x % 4 == 2 && y.is_multiple_of(4)) {
        4
    } else {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_scale_is_the_classic_dequantiser() {
        let scale = LevelScale::new(&ScalingMatrix::FLAT);
        // 16 * v: QP%6 = 0 gives 160 at (0,0), 256 at (1,1), 208 elsewhere.
        assert_eq!(scale.scale_4x4[0][0][0], 160);
        assert_eq!(scale.scale_4x4[0][0][5], 256);
        assert_eq!(scale.scale_4x4[0][0][1], 208);
        assert_eq!(scale.scale_8x8[1][5][0], 16 * 36);
    }

    #[test]
    fn fall_back_rules() {
        let mut parsed = ParsedMatrix {
            lists_4x4: [ParsedList::Absent; LISTS_4X4],
            lists_8x8: [ParsedList::Absent; LISTS_8X8],
        };
        parsed.lists_4x4[0] = ParsedList::Explicit([7; 16]);
        let a = resolve(&parsed, None);
        assert_eq!(a.lists_4x4[2], [7; 16]);
        assert_eq!(a.lists_4x4[3], DEFAULT_4X4[1]);
        assert_eq!(a.lists_8x8[1], DEFAULT_8X8[1]);
        let mut sps = ScalingMatrix::FLAT;
        sps.lists_4x4[3] = [9; 16];
        let b = resolve(&parsed, Some(&sps));
        assert_eq!(b.lists_4x4[1], [7; 16]);
        assert_eq!(b.lists_4x4[5], [9; 16]);
        assert_eq!(b.lists_8x8[0], [FLAT; 64]);
    }
}
