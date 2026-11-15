//! CTU decoding order of a picture's tiles (H.266 §6.5.1): tiles in raster
//! order, the CTUs of each tile in raster order within it.

/// One CTU in decoding order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Ctu {
    /// Luma position of the CTU's top-left sample.
    pub x: usize,
    pub y: usize,
    /// Raster-scan address in the picture.
    pub address: usize,
    /// Index of the CTU's tile: CUs see neighbours of their own tile only
    /// (§6.4.4).
    pub tile: u16,
    /// Luma position of the tile's top-left sample.
    pub tile_origin: (usize, usize),
}
impl Ctu {
    /// Whether this CTU is the first of its tile (a new CABAC substream with
    /// initial context variables).
    pub fn starts_tile(self) -> bool {
        (self.x, self.y) == self.tile_origin
    }
    /// Whether this CTU is the first of a CTU row of its tile.
    pub fn starts_row(self) -> bool {
        self.x == self.tile_origin.0
    }
}

/// The CTUs of a picture of `dimensions` luma samples in decoding order;
/// `tiles` holds the column widths and row heights in CTUs.
pub(super) fn scan(tiles: &[Vec<u32>; 2], dimensions: [usize; 2], ctu_log2: u8) -> Vec<Ctu> {
    let side = 1usize << ctu_log2;
    let columns = dimensions[0].div_ceil(side);
    let starts = |sizes: &[u32]| {
        sizes
            .iter()
            .scan(0, |start, &size| {
                let first = *start;
                *start += size as usize;
                Some((first, size as usize))
            })
            .collect::<Vec<_>>()
    };
    let (tile_columns, tile_rows) = (starts(&tiles[0]), starts(&tiles[1]));
    let mut order = Vec::with_capacity(columns * dimensions[1].div_ceil(side));
    for (row_index, &(top, height)) in tile_rows.iter().enumerate() {
        for (column_index, &(left, width)) in tile_columns.iter().enumerate() {
            for y in top..top + height {
                for x in left..left + width {
                    order.push(Ctu {
                        x: x * side,
                        y: y * side,
                        address: y * columns + x,
                        tile: (row_index * tile_columns.len() + column_index) as u16,
                        tile_origin: (left * side, top * side),
                    });
                }
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_are_scanned_one_after_another() {
        // 5x3 CTUs of 128 in tiles of 3+2 columns and 2+1 rows.
        let order = scan(&[vec![3, 2], vec![2, 1]], [640, 360], 7);
        let addresses: Vec<_> = order.iter().map(|ctu| ctu.address).collect();
        assert_eq!(
            addresses,
            [0, 1, 2, 5, 6, 7, 3, 4, 8, 9, 10, 11, 12, 13, 14]
        );
        assert_eq!(order[6].tile, 1);
        assert!(order[6].starts_tile() && order[8].starts_row() && !order[8].starts_tile());
        assert_eq!(order[13].tile_origin, (384, 256));
    }
}
