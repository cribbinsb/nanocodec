//! Block fills of raster cell grids: the codecs' mode-information grids
//! (AV1 4x4 `ModeInfo`, VP9 8x8 `ModeInfo`, HEVC minimum coding blocks)
//! record each coded block in every cell it covers.

/// Set the `side` x `side` cells from (`row`, `col`) of the raster grid
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
pub fn fill_square<T: Copy>(
    cells: &mut [T],
    (cols, rows): (usize, usize),
    (row, col): (usize, usize),
    side: usize,
    value: T,
) {
    let last_row = (row + side).min(rows);
    if row >= last_row {
        return;
    }
    let first = row * cols + col;
    let width = (col + side).min(cols) - col;
    cells[first..first + width].fill(value);
    for r in row + 1..last_row {
        cells.copy_within(first..first + width, r * cols + col);
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
    fn a_square_below_the_grid_writes_nothing() {
        let mut cells = vec![1u8; 4];
        fill_square(&mut cells, (2, 2), (2, 0), 2, 9);
        assert_eq!(cells, [1; 4]);
    }
}
