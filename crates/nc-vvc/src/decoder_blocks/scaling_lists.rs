//! Explicit scaling lists (H.266 §7.4.3.20, §8.7.3): the scaling factor `m`
//! of each coefficient of a transform block, from the matrices of the
//! scaling-list APS.
use super::{Sequence, SliceHeader};
use crate::decoder_headers::{Adaptation, AdaptationData, ScalingLists};
use nc_common::decode::{DecodeError, Result};

/// Matrices of Table 38 per block size: the 2x2 pair, then six each (intra
/// and inter of Y, Cb, Cr) for 4, 8, 16 and 32, then the luma pair of 64.
const FIRST_ID: [usize; 6] = [0, 2, 8, 14, 20, 26];
/// Matrix identifiers from here on carry a separate DC factor.
const FIRST_DC_ID: usize = 14;

/// The scaling lists a slice uses, if it uses any.
pub(crate) fn of_slice<'a>(
    header: &SliceHeader,
    banks: &'a [Option<Box<Adaptation>>; 8],
) -> Result<Option<&'a ScalingLists>> {
    if !header.scaling {
        return Ok(None);
    }
    let lists = header
        .picture
        .scaling
        .and_then(|id| banks.get(usize::from(id)))
        .and_then(Option::as_deref)
        .map(|aps| &aps.data);
    match lists {
        Some(AdaptationData::Scaling(lists)) => Ok(Some(lists)),
        _ => Err(DecodeError::Malformed("VVC missing scaling list APS")),
    }
}

/// Whether a transform block scaled without transform skip uses the lists:
/// not when the SPS excludes LFNST blocks and the block applies LFNST.
pub(super) fn applies(sequence: &Sequence, lfnst: bool) -> bool {
    !(lfnst && sequence.tools.scaling_lfnst_disabled)
}

/// The factors of a `width` x `height` block of `plane` (row-major): the
/// block size and prediction type select a matrix, which is sampled at the
/// block's resolution; blocks of 16 samples and more take their DC factor.
pub(super) fn factors(
    lists: &ScalingLists,
    intra: bool,
    plane: usize,
    (width, height): (usize, usize),
    out: &mut Vec<u8>,
) {
    out.clear();
    let size_class = width.max(height).ilog2() as usize - 1;
    let id = match size_class {
        // 2x2 chroma blocks are never intra.
        0 => plane.saturating_sub(1),
        // Luma has its own 64-sample pair; 4:4:4 chroma of 64 samples
        // takes its 32-sample matrices (Table 38).
        5 if plane == 0 => FIRST_ID[5] + usize::from(!intra),
        5 => FIRST_ID[4] + 3 * usize::from(!intra) + plane,
        _ => FIRST_ID[size_class] + 3 * usize::from(!intra) + plane,
    };
    let matrix = &lists.matrices[id];
    let side_log2 = (matrix.len().ilog2() / 2) as usize;
    let side = 1 << side_log2;
    let (width_log2, height_log2) = (width.ilog2() as usize, height.ilog2() as usize);
    for y in 0..height {
        let row = ((y << side_log2) >> height_log2) * side;
        out.extend((0..width).map(|x| matrix[row + ((x << side_log2) >> width_log2)]));
    }
    if id >= FIRST_DC_ID {
        out[0] = lists.dc[id];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lists() -> ScalingLists {
        ScalingLists {
            // Entry (i, j) of matrix `id` is id + 10 * j + i, wrapped to stay nonzero.
            matrices: std::array::from_fn(|id| {
                let side = if id < 2 {
                    2
                } else if id < 8 {
                    4
                } else {
                    8
                };
                (0..side * side)
                    .map(|pos| (1 + (id + 10 * (pos / side) + pos % side) % 250) as u8)
                    .collect()
            }),
            dc: std::array::from_fn(|id| 200 + id as u8),
        }
    }

    #[test]
    fn block_size_and_prediction_select_the_matrix() {
        let lists = lists();
        let mut out = Vec::new();
        // 4x4 inter Cb is matrix 2 + 3 + 1; no separate DC below 16 samples.
        factors(&lists, false, 1, (4, 4), &mut out);
        assert_eq!(out, lists.matrices[6]);
        // 2x2 inter Cr is matrix 1; 64x64 intra luma matrix 26 with its DC.
        factors(&lists, false, 2, (2, 2), &mut out);
        assert_eq!(out, lists.matrices[1]);
        factors(&lists, true, 0, (64, 64), &mut out);
        assert_eq!(
            (out[0], out[8], out[64 * 8]),
            (226, lists.matrices[26][1], lists.matrices[26][8])
        );
    }

    #[test]
    fn rectangular_blocks_sample_the_matrix_of_their_longer_side() {
        let lists = lists();
        let mut out = Vec::new();
        // 16x4 intra luma: matrix 14 (8x8), every second column, every second row twice.
        factors(&lists, true, 0, (16, 4), &mut out);
        let matrix = &lists.matrices[14];
        assert_eq!(out[0], lists.dc[14]);
        assert_eq!(out[1..4], [matrix[0], matrix[1], matrix[1]]);
        assert_eq!(out[16], matrix[2 * 8]);
        assert_eq!(out[3 * 16 + 15], matrix[6 * 8 + 7]);
    }
}
