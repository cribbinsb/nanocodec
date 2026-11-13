//! Coefficient scan orders (HEVC §6.5.3–§6.5.5, §7.3.8.11).
//!
//! A transform block is split into 4x4 sub-blocks; the sub-blocks, and the
//! coefficients inside each, are visited in the same order: up-right
//! diagonal (`scanIdx` 0), horizontal (1) or vertical (2). The horizontal
//! and vertical scans are used only by 4x4 and 8x8 intra blocks whose mode
//! is near vertical or near horizontal (§7.4.9.11).

use crate::transform::{MAX_LOG2, MIN_LOG2};

/// A quantised level as the residual coder reads it: `i32` while a block
/// is coded, `i16` in the picture's level store (HEVC levels lie in
/// -32768..=32767, §7.4.9.11).
pub trait Level: Copy + Default + PartialEq + Into<i32> + core::ops::BitOrAssign {}

impl Level for i16 {}
impl Level for i32 {}

/// Coefficients per 4x4 sub-block.
pub const SUB_BLOCK_COEFFS: usize = 16;
/// log2 of the sub-block side.
pub const SUB_BLOCK_LOG2: u8 = 2;

/// Whether the 4x4 sub-block `(xs, ys)` (in sub-blocks) of the row-major
/// `size`-square `levels` has a non-zero level. Its four rows are read as
/// fixed-size arrays and OR-ed, so the test is a few vector operations;
/// the residual coder, sign hiding and shaving all ask it of every
/// sub-block they visit. Exact; replacing the per-element row scans cut
/// Fast instructions 2.2% per P frame and 2.8% per key frame (callgrind,
/// crowd09 QP 47).
#[must_use]
#[inline]
pub fn sub_block_coded<L: Level>(levels: &[L], size: usize, (xs, ys): (usize, usize)) -> bool {
    let origin = 4 * (ys * size + xs);
    let block = &levels[origin..origin + 3 * size + 4];
    let mut any = [L::default(); 4];
    for row in 0..4 {
        let values = block[row * size..row * size + 4]
            .first_chunk::<4>()
            .expect("four levels per sub-block row");
        for (a, &v) in any.iter_mut().zip(values) {
            *a |= v;
        }
    }
    any != [L::default(); 4]
}

/// The up-right diagonal scan of a `size`-square array (HEVC §6.5.3):
/// `(x, y)` positions in scan order.
#[must_use]
pub fn up_right_diagonal(size: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(size * size);
    let (mut x, mut y) = (0isize, 0isize);
    while order.len() < size * size {
        while y >= 0 {
            if (x as usize) < size && (y as usize) < size {
                order.push((x as usize, y as usize));
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
    order
}

/// `scanIdx` (§7.4.9.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScanKind {
    /// Up-right diagonal (`scanIdx` 0).
    #[default]
    Diagonal,
    /// Row by row (`scanIdx` 1).
    Horizontal,
    /// Column by column (`scanIdx` 2).
    Vertical,
}

impl ScanKind {
    /// The scan of an intra transform block of side `2^log2` predicted with
    /// mode `mode` (§7.4.9.11; 4:2:0, so for chroma only 4x4 blocks).
    #[must_use]
    pub fn for_intra(mode: u8, log2: u8, luma: bool) -> Self {
        let mode_dependent = if luma {
            log2 == 2 || log2 == 3
        } else {
            log2 == 2
        };
        match mode {
            6..=14 if mode_dependent => Self::Vertical,
            22..=30 if mode_dependent => Self::Horizontal,
            _ => Self::Diagonal,
        }
    }
}

/// Positions of a `size`-square array in the order of `kind`.
fn traverse(kind: ScanKind, size: usize) -> Vec<(usize, usize)> {
    match kind {
        ScanKind::Diagonal => up_right_diagonal(size),
        ScanKind::Horizontal => (0..size)
            .flat_map(|y| (0..size).map(move |x| (x, y)))
            .collect(),
        ScanKind::Vertical => (0..size)
            .flat_map(|x| (0..size).map(move |y| (x, y)))
            .collect(),
    }
}

/// Scan order of one transform size.
#[derive(Debug, Clone)]
pub struct Scan {
    /// log2 of the transform side.
    pub log2: u8,
    /// The scan order.
    pub kind: ScanKind,
    /// Sub-block positions `(xS, yS)` in scan order.
    pub sub_blocks: Vec<(usize, usize)>,
    /// Raster index (`y * size + x`) of each coefficient, in scan order:
    /// sub-block `i` covers entries `16 * i .. 16 * i + 16`.
    pub positions: Vec<usize>,
    /// The inverse of `positions`: the scan index of each raster index.
    pub index_of: Vec<u16>,
}

impl Scan {
    /// The `kind` scan of a `2^log2`-square transform block.
    #[must_use]
    pub fn new(log2: u8, kind: ScanKind) -> Self {
        let size = 1usize << log2;
        let sub_blocks = traverse(kind, size >> SUB_BLOCK_LOG2);
        let in_block = traverse(kind, 1 << SUB_BLOCK_LOG2);
        let positions = sub_blocks
            .iter()
            .flat_map(|&(xs, ys)| {
                in_block.iter().map(move |&(x, y)| {
                    ((ys << SUB_BLOCK_LOG2) + y) * size + (xs << SUB_BLOCK_LOG2) + x
                })
            })
            .collect::<Vec<usize>>();
        let mut index_of = vec![0; positions.len()];
        for (index, &raster) in positions.iter().enumerate() {
            index_of[raster] = index as u16;
        }
        Self {
            log2,
            kind,
            sub_blocks,
            positions,
            index_of,
        }
    }
}

/// Every scan of every transform size, indexed by `log2 - MIN_LOG2` and
/// scan kind.
#[derive(Debug, Clone)]
pub struct Scans(Vec<[Scan; 3]>);

impl Scans {
    /// Build all scans once per encoder.
    #[must_use]
    pub fn new() -> Self {
        Self(
            (MIN_LOG2..=MAX_LOG2)
                .map(|log2| {
                    [ScanKind::Diagonal, ScanKind::Horizontal, ScanKind::Vertical]
                        .map(|kind| Scan::new(log2, kind))
                })
                .collect(),
        )
    }

    /// The `kind` scan of a `2^log2` transform.
    #[must_use]
    pub fn get(&self, log2: u8, kind: ScanKind) -> &Scan {
        &self.0[usize::from(log2 - MIN_LOG2)][kind as usize]
    }
}

impl Default for Scans {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_by_four_diagonal_matches_the_specification() {
        // HEVC §6.5.3: (0,0) (0,1) (1,0) (0,2) (1,1) (2,0) ...
        let order = up_right_diagonal(4);
        assert_eq!(
            &order[..6],
            &[(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
        );
        assert_eq!(order[15], (3, 3));
    }

    #[test]
    fn block_scan_visits_every_coefficient_once() {
        for log2 in MIN_LOG2..=MAX_LOG2 {
            for kind in [ScanKind::Diagonal, ScanKind::Horizontal, ScanKind::Vertical] {
                let scan = Scan::new(log2, kind);
                let mut seen = scan.positions.clone();
                seen.sort_unstable();
                assert_eq!(seen, (0..1usize << (2 * log2)).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn eight_by_eight_horizontal_visits_sub_blocks_row_by_row() {
        // §6.5.4: sub-blocks (0,0) (1,0) (0,1) (1,1), each row by row.
        let scan = Scan::new(3, ScanKind::Horizontal);
        assert_eq!(scan.sub_blocks, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        assert_eq!(&scan.positions[..5], &[0, 1, 2, 3, 8]);
        let scan = Scan::new(3, ScanKind::Vertical);
        assert_eq!(&scan.positions[..5], &[0, 8, 16, 24, 1]);
    }
}
