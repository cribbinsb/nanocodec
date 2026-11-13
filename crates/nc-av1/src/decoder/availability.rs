//! Which neighbours a block may read: the tile that bounds them (AV1
//! §5.11.1 `MiRowStart`..`MiColEnd`, `is_inside`), and the neighbouring
//! samples an intra transform block may read (§7.11.2 `haveAboveRt`,
//! `haveBelowLft`, from §5.11.5 `BlockDecoded`).
//!
//! The spec keeps a decoded flag per 4x4 unit of each plane, set by every
//! transform block. Equivalently, and with one store per block row instead
//! of one per unit, this keeps a bit per 4x4 luma unit of the current
//! superblock, set when a whole block is decoded; inside the block being
//! decoded, decoding order (64x64 chunks in raster order, transform blocks
//! in raster order within each) decides. A chroma unit is decoded with the
//! block covering the bottom-right luma unit of its 8x8 area, the block that
//! carries the chroma of sub-8x8 blocks. Outside the superblock, the row
//! above is decoded inside the tile, the superblock to the left down to its
//! bottom edge, and nothing to the right or below (`clear_block_decoded_flags`).
//! libaom's `has_top_right` / `has_bottom_left` tables encode the same
//! order (`av1/common/reconintra.c`).
use super::block_size::BlockSize;

/// Most 4x4 rows (and columns) of a superblock (128x128).
const MAX_SUPERBLOCK_MI: usize = 32;
/// 4x4 units per 64x64 chunk of a large block, log2.
const CHUNK_MI_LOG2: usize = 4;

/// The tile being decoded, in 4x4 units: neighbours outside it are not
/// available (§5.11.1, `is_inside`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileBounds {
    /// `MiColStart`, `MiColEnd`.
    pub col_start: usize,
    pub col_end: usize,
    /// `MiRowStart`, `MiRowEnd`.
    pub row_start: usize,
    pub row_end: usize,
}

impl TileBounds {
    /// The whole frame of `mi_cols` x `mi_rows` units as one tile.
    pub(crate) fn frame(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            col_start: 0,
            col_end: mi_cols,
            row_start: 0,
            row_end: mi_rows,
        }
    }
    /// `is_inside(row, col)`.
    pub(crate) fn is_inside(&self, row: isize, col: isize) -> bool {
        let inside =
            |at: isize, start: usize, end: usize| (start as isize..end as isize).contains(&at);
        inside(row, self.row_start, self.row_end) && inside(col, self.col_start, self.col_end)
    }
    /// (`AvailU`, `AvailL`) of the unit (`mi_col`, `mi_row`) inside the
    /// tile.
    pub(crate) fn available(&self, mi_col: usize, mi_row: usize) -> (bool, bool) {
        (mi_row > self.row_start, mi_col > self.col_start)
    }
}

/// Decoded blocks of the current superblock.
#[derive(Debug)]
pub(super) struct DecodedUnits {
    /// One bit per 4x4 luma unit (column), per row of the superblock.
    rows: [u32; MAX_SUPERBLOCK_MI],
    /// The superblock's top-left unit and side in units.
    origin: (usize, usize),
    side: usize,
    /// The tile's right and bottom edges (`MiColEnd`, `MiRowEnd`).
    col_end: usize,
    row_end: usize,
}

/// A transform block of one plane of a block, in that plane's 4x4 units:
/// the block's top-left unit and size there, the transform's offset in the
/// block and size, and the plane's subsampling.
#[derive(Debug, Clone, Copy)]
pub(super) struct PlaneTransform {
    pub block_origin: (usize, usize),
    pub block_size: (usize, usize),
    pub offset: (usize, usize),
    pub size: (usize, usize),
    pub subsampled: bool,
}

impl DecodedUnits {
    pub(super) fn new(mi_cols: usize, mi_rows: usize, superblock: BlockSize) -> Self {
        Self {
            rows: [0; MAX_SUPERBLOCK_MI],
            origin: (0, 0),
            side: superblock.width_mi(),
            col_end: mi_cols,
            row_end: mi_rows,
        }
    }
    /// Start the tile `tile`.
    pub(super) fn start_tile(&mut self, tile: &TileBounds) {
        self.col_end = tile.col_end;
        self.row_end = tile.row_end;
    }
    /// Start the superblock at unit (`mi_col`, `mi_row`): nothing in it is
    /// decoded.
    pub(super) fn start_superblock(&mut self, mi_col: usize, mi_row: usize) {
        self.origin = (mi_col, mi_row);
        self.rows = [0; MAX_SUPERBLOCK_MI];
    }
    /// The block of `w4` x `h4` units at (`mi_col`, `mi_row`) is decoded.
    pub(super) fn mark(&mut self, (mi_col, mi_row): (usize, usize), (w4, h4): (usize, usize)) {
        let (col, row) = (mi_col - self.origin.0, mi_row - self.origin.1);
        let width = w4.min(self.side - col);
        let bits = (((1u64 << width) - 1) << col) as u32;
        for r in &mut self.rows[row..(row + h4).min(self.side)] {
            *r |= bits;
        }
    }
    /// Whether the luma unit (`row`, `col`), outside the block being
    /// decoded, is decoded.
    pub(super) fn luma_decoded(&self, row: isize, col: isize) -> bool {
        let (sb_col, sb_row) = (self.origin.0 as isize, self.origin.1 as isize);
        let side = self.side as isize;
        if row < 0 || col < 0 {
            return false;
        }
        if row < sb_row {
            return col < self.col_end as isize;
        }
        if row >= sb_row + side || row >= self.row_end as isize {
            return false;
        }
        if col < sb_col {
            return true;
        }
        if col >= sb_col + side {
            return false;
        }
        self.rows[(row - sb_row) as usize] & (1 << (col - sb_col)) != 0
    }
    /// Whether the plane unit at (`row`, `col`) relative to `tx`'s block
    /// is decoded when `tx` is predicted.
    fn decoded(&self, tx: &PlaneTransform, row: isize, col: isize) -> bool {
        let (bw, bh) = (tx.block_size.0 as isize, tx.block_size.1 as isize);
        if (0..bh).contains(&row) && (0..bw).contains(&col) {
            // Inside the block: 64x64 chunks in raster order, transform
            // rows in order within one.
            let chunk_log2 = CHUNK_MI_LOG2 - usize::from(tx.subsampled);
            let chunk = |r: isize, c: isize| (r >> chunk_log2, c >> chunk_log2);
            let current = chunk(tx.offset.1 as isize, tx.offset.0 as isize);
            let other = chunk(row, col);
            return other < current || (other == current && row < tx.offset.1 as isize);
        }
        let ss = usize::from(tx.subsampled);
        let plane_row = tx.block_origin.1 as isize + row;
        let plane_col = tx.block_origin.0 as isize + col;
        // A chroma unit belongs to the block holding its bottom-right luma
        // unit.
        self.luma_decoded(
            (plane_row << ss) + ss as isize,
            (plane_col << ss) + ss as isize,
        )
    }
    /// `haveAboveRt` and `haveBelowLft` of `tx`, given `haveAbove` and
    /// `haveLeft`.
    pub(super) fn corners(
        &self,
        tx: &PlaneTransform,
        have_above: bool,
        have_left: bool,
    ) -> (bool, bool) {
        let (x, y) = (tx.offset.0 as isize, tx.offset.1 as isize);
        let (w, h) = (tx.size.0 as isize, tx.size.1 as isize);
        let above_right = have_above && self.decoded(tx, y - 1, x + w);
        let below_left = have_left && self.decoded(tx, y + h, x - 1);
        (above_right, below_left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma(
        block_origin: (usize, usize),
        block_size: (usize, usize),
        offset: (usize, usize),
        size: (usize, usize),
    ) -> PlaneTransform {
        PlaneTransform {
            block_origin,
            block_size,
            offset,
            size,
            subsampled: false,
        }
    }

    #[test]
    fn quadtree_order_inside_a_superblock() {
        let mut units = DecodedUnits::new(64, 64, BlockSize::BLOCK_64X64);
        units.start_superblock(16, 16);
        // Top-left 32x32 (8x8 units): above-right is the row above the
        // superblock, below-left the left superblock.
        let tx = luma((16, 16), (8, 8), (0, 0), (8, 8));
        assert_eq!(units.corners(&tx, true, true), (true, true));
        units.mark((16, 16), (8, 8));
        // Top-right 32x32: below-left (the bottom-left quadrant) not yet.
        let tx = luma((24, 16), (8, 8), (0, 0), (8, 8));
        assert_eq!(units.corners(&tx, true, true), (true, false));
        units.mark((24, 16), (8, 8));
        // Bottom-left 32x32: above-right is the decoded top-right quadrant;
        // below-left is the next superblock row.
        let tx = luma((16, 24), (8, 8), (0, 0), (8, 8));
        assert_eq!(units.corners(&tx, true, true), (true, false));
        // A transform in the block's second row: above-right inside the
        // block is decoded, below-left inside it is not.
        let tx = luma((16, 24), (8, 8), (0, 4), (4, 4));
        assert_eq!(units.corners(&tx, true, true), (true, false));
        // Bottom-right 32x32: neither the right superblock nor the next
        // superblock row is decoded.
        let tx = luma((24, 24), (8, 8), (0, 0), (8, 8));
        assert_eq!(units.corners(&tx, true, true), (false, false));
    }

    #[test]
    fn chunks_of_a_128_block_decode_in_raster_order() {
        let mut units = DecodedUnits::new(64, 64, BlockSize::BLOCK_128X128);
        units.start_superblock(0, 32);
        // The first transform of the bottom-left 64x64 chunk: its above-right
        // is the top-right chunk, decoded before it.
        let tx = luma((0, 32), (32, 32), (0, 16), (16, 16));
        assert!(units.corners(&tx, true, false).0);
        // The top-right chunk's first transform: its above-right is the row
        // above; its below-left is in the bottom-left chunk, not decoded,
        // unless the transform is short enough to stay beside the top-left
        // chunk.
        let tx = luma((0, 32), (32, 32), (16, 0), (16, 16));
        assert_eq!(units.corners(&tx, true, true), (true, false));
        let tx = luma((0, 32), (32, 32), (16, 0), (2, 2));
        assert_eq!(units.corners(&tx, true, true), (true, true));
    }

    #[test]
    fn sub8x8_chroma_follows_the_block_carrying_it() {
        let mut units = DecodedUnits::new(64, 64, BlockSize::BLOCK_64X64);
        units.start_superblock(0, 0);
        // Two 4x8 blocks at units (4, 4) and (5, 4); the second carries the
        // 4x4 chroma at chroma unit (2, 2). The chroma unit right of the
        // block above it (chroma (3, 1)) belongs to luma unit (7, 3): not
        // decoded until marked. Below-left (chroma (1, 3)) is not decoded.
        units.mark((4, 4), (1, 2));
        units.mark((5, 4), (1, 2));
        let chroma = PlaneTransform {
            block_origin: (2, 2),
            block_size: (1, 1),
            offset: (0, 0),
            size: (1, 1),
            subsampled: true,
        };
        assert_eq!(units.corners(&chroma, true, true), (false, false));
        units.mark((6, 2), (2, 2));
        assert_eq!(units.corners(&chroma, true, true), (true, false));
    }
}
