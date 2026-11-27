//! Block fills of raster cell grids: the codecs' mode-information grids
//! (AV1 4x4 `ModeInfo`, VP9 8x8 `ModeInfo`, HEVC minimum coding blocks)
//! record each coded block in every cell it covers; a coding trial notes a
//! square's cells and puts them back.

/// Set the `side` x `side` cells from (`row`, `col`) of the raster grid
/// `cells` (`cols` cells per row, `rows` rows) to `value`, clipped to the
/// grid ([`fill_rect`] of a square).
///
/// # Panics
///
/// If the clipped block does not lie inside `cells`.
pub fn fill_square<T: Copy>(
    cells: &mut [T],
    grid: (usize, usize),
    at: (usize, usize),
    side: usize,
    value: T,
) {
    fill_rect(cells, grid, at, (side, side), value);
}

/// Set the `width` x `height` cells from (`row`, `col`) of the raster grid
/// `cells` (`cols` cells per row, `rows` rows) to `value`, clipped to the
/// grid.
///
/// The first row is filled cell by cell and copied whole into the others: a
/// cell of several fields is written field by field, and the row copies took
/// AV1 Faster's encode instructions down 1.0 % and Veryfast's 1.3 % (x86
/// callgrind, 2026-11-20).
///
/// # Panics
///
/// If the clipped block does not lie inside `cells`.
pub fn fill_rect<T: Copy>(
    cells: &mut [T],
    (cols, rows): (usize, usize),
    (row, col): (usize, usize),
    (width, height): (usize, usize),
    value: T,
) {
    let last_row = (row + height).min(rows);
    if row >= last_row {
        return;
    }
    let first = row * cols + col;
    let width = (col + width).min(cols) - col;
    cells[first..first + width].fill(value);
    for r in row + 1..last_row {
        cells.copy_within(first..first + width, r * cols + col);
    }
}

/// Note the `width` x `height` cells from (`row`, `col`) of `cells`, clipped
/// to the grid, in `saved` (row by row), for [`restore_rect`].
///
/// # Panics
///
/// If the clipped block does not lie inside `cells`.
pub fn save_rect<T: Copy>(
    cells: &[T],
    (cols, rows): (usize, usize),
    (row, col): (usize, usize),
    (width, height): (usize, usize),
    saved: &mut Vec<T>,
) {
    saved.clear();
    let width = (col + width).min(cols).saturating_sub(col);
    for r in row..(row + height).min(rows) {
        let first = r * cols + col;
        saved.extend_from_slice(&cells[first..first + width]);
    }
}

/// Put back the cells [`save_rect`] noted of the same rectangle.
///
/// # Panics
///
/// If `saved` does not match the clipped block.
pub fn restore_rect<T: Copy>(
    cells: &mut [T],
    (cols, rows): (usize, usize),
    (row, col): (usize, usize),
    (width, height): (usize, usize),
    saved: &[T],
) {
    let width = (col + width).min(cols).saturating_sub(col);
    if width == 0 {
        return;
    }
    for (r, saved_row) in (row..(row + height).min(rows)).zip(saved.chunks_exact(width)) {
        let first = r * cols + col;
        cells[first..first + width].copy_from_slice(saved_row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_the_clipped_square_only() {
        let (cols, rows) = (5, 4);
        let mut cells = vec![0u8; cols * rows];
        fill_square(&mut cells, (cols, rows), (2, 3), 4, 7);
        let expected: Vec<u8> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| u8::from(r >= 2 && c >= 3) * 7))
            .collect();
        assert_eq!(cells, expected);
    }

    #[test]
    fn a_saved_rectangle_is_put_back() {
        let (cols, rows) = (5, 4);
        let original: Vec<u8> = (0..20).collect();
        let mut cells = original.clone();
        let mut saved = Vec::new();
        save_rect(&cells, (cols, rows), (2, 3), (4, 4), &mut saved);
        assert_eq!(saved, [13, 14, 18, 19]);
        fill_square(&mut cells, (cols, rows), (2, 3), 4, 99);
        restore_rect(&mut cells, (cols, rows), (2, 3), (4, 4), &saved);
        assert_eq!(cells, original);
    }

    #[test]
    fn a_square_below_the_grid_writes_nothing() {
        let mut cells = vec![1u8; 4];
        fill_square(&mut cells, (2, 2), (2, 0), 2, 9);
        assert_eq!(cells, [1; 4]);
    }
}
