//! The CTUs of one slice in decoding order (H.266 §6.5.1), each with its
//! tile: the slice's CTU addresses come from its header.
use crate::decoder_headers::TileGrid;

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

/// The CTUs at addresses `ctus` (a slice's, in decoding order) of a
/// picture with tile grid `tiles` and CTUs of `ctu_log2`.
pub(super) fn scan(ctus: &[u32], tiles: &TileGrid, ctu_log2: u8) -> Vec<Ctu> {
    let side = 1usize << ctu_log2;
    let width = tiles.ctus(0);
    ctus.iter()
        .map(|&address| {
            let (x, y) = (address % width, address / width);
            let column = tiles.index(0, x);
            let row = tiles.index(1, y);
            Ctu {
                x: x as usize * side,
                y: y as usize * side,
                address: address as usize,
                tile: (row * tiles.tiles(0) + column) as u16,
                tile_origin: (
                    tiles.span(0, column).start as usize * side,
                    tiles.span(1, row).start as usize * side,
                ),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_are_scanned_one_after_another() {
        // 5x3 CTUs of 128 in tiles of 3+2 columns and 2+1 rows.
        let tiles = TileGrid::new(&[vec![3, 2], vec![2, 1]]);
        let crate::decoder_headers::SliceLayout::Rectangular { slices, .. } =
            crate::decoder_headers::SliceLayout::whole(&tiles)
        else {
            unreachable!()
        };
        let order = scan(&slices[0], &tiles, 7);
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
