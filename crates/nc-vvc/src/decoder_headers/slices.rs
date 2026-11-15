//! H.266 §6.5.1 tile grid and slice layout: which CTUs each slice of a
//! picture holds, in decoding order, from the PPS's tile and rectangular
//! slice syntax (§7.3.2.5) or a raster-scan slice's address (§7.3.7.1).
//! Derivation checked against BSD-licensed VTM `PPS::initRectSliceMap`; see
//! `THIRD_PARTY.md`.
use super::{DecodeError, Reader, Result, Subpictures};
use std::ops::Range;

/// Tile column and row boundaries of a picture, in CTUs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TileGrid {
    /// Tile column `i` spans CTU columns `bounds[0][i]..bounds[0][i + 1]`;
    /// likewise rows in `bounds[1]`.
    bounds: [Vec<u32>; 2],
}
impl TileGrid {
    /// The grid of tile column widths and row heights `sizes` (in CTUs).
    pub fn new(sizes: &[Vec<u32>; 2]) -> Self {
        Self {
            bounds: sizes.each_ref().map(|sizes| {
                std::iter::once(0)
                    .chain(sizes.iter().scan(0, |end, &size| {
                        *end += size;
                        Some(*end)
                    }))
                    .collect()
            }),
        }
    }
    /// Tile columns (axis 0) or rows (axis 1).
    pub fn tiles(&self, axis: usize) -> usize {
        self.bounds[axis].len() - 1
    }
    /// `NumTilesInPic`.
    pub fn count(&self) -> usize {
        self.tiles(0) * self.tiles(1)
    }
    /// The picture's width (axis 0) or height (axis 1) in CTUs.
    pub fn ctus(&self, axis: usize) -> u32 {
        *self.bounds[axis].last().unwrap()
    }
    /// The CTU columns (axis 0) or rows (axis 1) of tile column or row `index`.
    pub fn span(&self, axis: usize, index: usize) -> Range<u32> {
        self.bounds[axis][index]..self.bounds[axis][index + 1]
    }
    /// The tile column (axis 0) or row (axis 1) holding CTU column or row `ctu`.
    pub fn index(&self, axis: usize, ctu: u32) -> usize {
        self.bounds[axis].partition_point(|&start| start <= ctu) - 1
    }
    /// The tile holding CTU (`x`, `y`), in raster order of tiles.
    pub fn tile_at(&self, x: u32, y: u32) -> usize {
        self.index(1, y) * self.tiles(0) + self.index(0, x)
    }
    /// Append the CTU addresses of rectangle `columns` x `rows` in raster order.
    fn push_rectangle(&self, columns: Range<u32>, rows: Range<u32>, out: &mut Vec<u32>) {
        let width = self.ctus(0);
        for y in rows {
            out.extend(columns.clone().map(|x| y * width + x));
        }
    }
    /// The CTU addresses of the tiles in tile columns `columns` and rows
    /// `rows`, tile by tile in raster order (§6.5.1).
    fn tiles_ctus(&self, columns: Range<usize>, rows: Range<usize>) -> Vec<u32> {
        let mut out = Vec::new();
        for row in rows {
            for column in columns.clone() {
                self.push_rectangle(self.span(0, column), self.span(1, row), &mut out);
            }
        }
        out
    }
    /// The CTU addresses of raster-scan slice tiles `first..first + count`.
    pub fn raster_slice(&self, first: usize, count: usize) -> Vec<u32> {
        let columns = self.tiles(0);
        let mut out = Vec::new();
        for tile in first..first + count {
            self.push_rectangle(
                self.span(0, tile % columns),
                self.span(1, tile / columns),
                &mut out,
            );
        }
        out
    }
}

/// How the slices of a picture are laid out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SliceLayout {
    /// `pps_rect_slice_flag` 0: each slice header names its first tile and
    /// tile count.
    Raster,
    /// Rectangular slices, given by the PPS (or one per subpicture).
    Rectangular {
        /// The CTU addresses of each slice, in decoding order.
        slices: Vec<Vec<u32>>,
        /// The picture-level indices of the slices of each subpicture, in
        /// slice address order (`SubpicIdxForSlice`, §7.4.3.5).
        subpictures: Vec<Vec<u16>>,
    },
}
impl SliceLayout {
    /// The one slice of an unpartitioned picture.
    pub fn whole(grid: &TileGrid) -> Self {
        Self::Rectangular {
            slices: vec![grid.tiles_ctus(0..grid.tiles(0), 0..grid.tiles(1))],
            subpictures: vec![vec![0]],
        }
    }

    /// One slice per subpicture (`pps_single_slice_per_subpic_flag`): the
    /// part of each tile the subpicture covers, tile by tile.
    pub fn per_subpicture(grid: &TileGrid, subpictures: &Subpictures) -> Result<Self> {
        let mut slices = Vec::with_capacity(subpictures.layout.len());
        for subpicture in &subpictures.layout {
            let [x0, y0, width, height] = subpicture.ctus;
            let (columns, rows) = (x0..x0 + width, y0..y0 + height);
            let mut ctus = Vec::new();
            for row in grid.index(1, y0)..=grid.index(1, rows.end - 1) {
                for column in grid.index(0, x0)..=grid.index(0, columns.end - 1) {
                    let (tile_columns, tile_rows) = (grid.span(0, column), grid.span(1, row));
                    grid.push_rectangle(
                        columns.start.max(tile_columns.start)..columns.end.min(tile_columns.end),
                        rows.start.max(tile_rows.start)..rows.end.min(tile_rows.end),
                        &mut ctus,
                    );
                }
            }
            slices.push(ctus);
        }
        Self::rectangular(grid, slices, subpictures)
    }

    /// Parse the rectangular slice syntax after `pps_single_slice_per_subpic_flag`
    /// (`pps_num_slices_in_pic_minus1` to the last `pps_tile_idx_delta_val`).
    pub fn parse(r: &mut Reader<'_>, grid: &TileGrid, subpictures: &Subpictures) -> Result<Self> {
        let total_ctus = grid.ctus(0) * grid.ctus(1);
        let count = r.ue(total_ctus - 1, "VVC slice count")? as usize + 1;
        let delta_present = count > 2 && r.flag()?;
        let (columns, rows) = (grid.tiles(0), grid.tiles(1));
        let mut slices = Vec::with_capacity(count);
        let mut tile = 0usize;
        let mut previous_height = 1;
        while slices.len() < count {
            let (x, y) = (tile % columns, tile / columns);
            if slices.len() + 1 == count {
                // The last slice holds the tiles from its first to the
                // bottom-right of the picture.
                slices.push(grid.tiles_ctus(x..columns, y..rows));
                break;
            }
            let width = if x + 1 == columns {
                1
            } else {
                r.ue((columns - 1 - x) as u32, "VVC slice width in tiles")? as usize + 1
            };
            let height = if y + 1 == rows {
                1
            } else if delta_present || x == 0 {
                r.ue((rows - 1 - y) as u32, "VVC slice height in tiles")? as usize + 1
            } else {
                previous_height
            };
            if y + height > rows {
                return Err(DecodeError::Malformed("VVC slice below picture"));
            }
            previous_height = height;
            let tile_rows = grid.span(1, y);
            if width == 1 && height == 1 && tile_rows.len() > 1 {
                let mut top = tile_rows.start;
                for slice_height in in_tile_heights(r, tile_rows.len() as u32)? {
                    let mut ctus = Vec::new();
                    grid.push_rectangle(grid.span(0, x), top..top + slice_height, &mut ctus);
                    slices.push(ctus);
                    top += slice_height;
                }
                if slices.len() > count {
                    return Err(DecodeError::Malformed("VVC slices in tile exceed count"));
                }
            } else {
                slices.push(grid.tiles_ctus(x..x + width, y..y + height));
            }
            if slices.len() < count {
                tile = if delta_present {
                    let delta = r.se(
                        -(grid.count() as i32),
                        grid.count() as i32,
                        "VVC tile index delta",
                    )?;
                    usize::try_from(tile as i64 + i64::from(delta))
                        .ok()
                        .filter(|&tile| tile < grid.count())
                        .ok_or(DecodeError::Malformed("VVC tile index delta"))?
                } else {
                    let mut next = tile + width;
                    if next.is_multiple_of(columns) {
                        next += (height - 1) * columns;
                    }
                    next
                };
                if tile >= grid.count() {
                    return Err(DecodeError::Malformed("VVC slice beyond last tile"));
                }
            }
        }
        Self::rectangular(grid, slices, subpictures)
    }

    /// Check that `slices` cover each CTU once, and assign them to
    /// subpictures by their first CTU.
    fn rectangular(
        grid: &TileGrid,
        slices: Vec<Vec<u32>>,
        subpictures: &Subpictures,
    ) -> Result<Self> {
        let width = grid.ctus(0);
        let mut covered = vec![false; (width * grid.ctus(1)) as usize];
        let mut by_subpicture = vec![Vec::new(); subpictures.layout.len()];
        for (index, slice) in slices.iter().enumerate() {
            let first = *slice
                .first()
                .ok_or(DecodeError::Malformed("VVC empty slice"))?;
            for &address in slice {
                let cell = covered
                    .get_mut(address as usize)
                    .ok_or(DecodeError::Malformed("VVC slice outside picture"))?;
                if std::mem::replace(cell, true) {
                    return Err(DecodeError::Malformed("VVC overlapping slices"));
                }
            }
            by_subpicture[subpictures.at(first % width, first / width)].push(index as u16);
        }
        if covered.contains(&false) {
            return Err(DecodeError::Malformed("VVC slices leave a gap"));
        }
        if by_subpicture.iter().any(Vec::is_empty) {
            return Err(DecodeError::Malformed("VVC subpicture without slice"));
        }
        Ok(Self::Rectangular {
            slices,
            subpictures: by_subpicture,
        })
    }
}

/// The CTU-row heights of the slices within one tile of `rows` CTU rows:
/// `pps_num_exp_slices_in_tile` explicit heights, then the last explicit
/// height repeated, then the remainder (§6.5.1).
fn in_tile_heights(r: &mut Reader<'_>, rows: u32) -> Result<Vec<u32>> {
    let explicit = r.ue(rows - 1, "VVC slices in tile")?;
    if explicit == 0 {
        return Ok(vec![rows]);
    }
    let mut heights = Vec::with_capacity(explicit as usize);
    let mut remaining = rows;
    for _ in 0..explicit {
        let height = r.ue(rows - 1, "VVC slice height in CTUs")? + 1;
        remaining = remaining
            .checked_sub(height)
            .ok_or(DecodeError::Malformed("VVC slice heights exceed tile"))?;
        heights.push(height);
    }
    let uniform = *heights.last().unwrap();
    while remaining >= uniform {
        heights.push(uniform);
        remaining -= uniform;
    }
    if remaining > 0 {
        heights.push(remaining);
    }
    Ok(heights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_and_raster_slices_follow_tile_scan() {
        // 5x3 CTUs in tiles of 3+2 columns and 2+1 rows.
        let grid = TileGrid::new(&[vec![3, 2], vec![2, 1]]);
        assert_eq!(grid.count(), 4);
        assert_eq!(grid.tile_at(4, 2), 3);
        assert_eq!(grid.raster_slice(1, 2), [3, 4, 8, 9, 10, 11, 12]);
        let SliceLayout::Rectangular { slices, .. } = SliceLayout::whole(&grid) else {
            unreachable!()
        };
        assert_eq!(
            slices[0],
            [0, 1, 2, 5, 6, 7, 3, 4, 8, 9, 10, 11, 12, 13, 14]
        );
    }

    #[test]
    fn subpictures_split_tiles_they_do_not_fill() {
        // One tile of 4x4 CTUs, subpictures of its top three and bottom row.
        let grid = TileGrid::new(&[vec![4], vec![4]]);
        let mut subpictures = Subpictures::whole([4, 4]);
        subpictures.layout[0].ctus = [0, 0, 4, 3];
        let mut bottom = subpictures.layout[0];
        bottom.ctus = [0, 3, 4, 1];
        subpictures.layout.push(bottom);
        let SliceLayout::Rectangular {
            slices,
            subpictures,
        } = SliceLayout::per_subpicture(&grid, &subpictures).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(slices[1], [12, 13, 14, 15]);
        assert_eq!(subpictures, [vec![0], vec![1]]);
    }
}
