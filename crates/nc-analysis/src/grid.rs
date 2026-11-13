//! A 2-D grid of per-block values in raster order.

/// Per-block values for a `cols` x `rows` grid of blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid<T> {
    cols: usize,
    rows: usize,
    cells: Vec<T>,
}

impl<T: Clone> Grid<T> {
    /// A grid with every cell set to `value`.
    #[must_use]
    pub fn filled(cols: usize, rows: usize, value: T) -> Self {
        Self {
            cols,
            rows,
            cells: vec![value; cols * rows],
        }
    }
}

impl<T> Grid<T> {
    /// A grid from cells in raster order.
    ///
    /// # Panics
    ///
    /// If `cells.len() != cols * rows`.
    #[must_use]
    pub fn from_cells(cols: usize, rows: usize, cells: Vec<T>) -> Self {
        assert_eq!(cells.len(), cols * rows, "grid cell count mismatch");
        Self { cols, rows, cells }
    }

    /// Blocks per row.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Rows of blocks.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The cell at (`col`, `row`).
    ///
    /// # Panics
    ///
    /// If the position is outside the grid.
    #[must_use]
    pub fn get(&self, col: usize, row: usize) -> &T {
        assert!(
            col < self.cols && row < self.rows,
            "grid index out of range"
        );
        &self.cells[row * self.cols + col]
    }

    /// The cell at (`col`, `row`), or `None` outside the grid (signed, so
    /// neighbour lookups like `col - 1` need no special casing).
    #[must_use]
    pub fn try_get(&self, col: isize, row: isize) -> Option<&T> {
        let (c, r) = (usize::try_from(col).ok()?, usize::try_from(row).ok()?);
        (c < self.cols && r < self.rows).then(|| &self.cells[r * self.cols + c])
    }

    /// Up to `len` cells of `row` from `col` on, clipped to the grid (empty
    /// for a row or column outside it): one bounds check for a run of
    /// cells instead of one per cell.
    #[must_use]
    pub fn row_run(&self, col: usize, row: usize, len: usize) -> &[T] {
        if row >= self.rows || col >= self.cols {
            return &[];
        }
        let start = row * self.cols + col;
        &self.cells[start..start + len.min(self.cols - col)]
    }

    /// The cells of `row`, writable.
    ///
    /// # Panics
    ///
    /// If `row` is outside the grid.
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        assert!(row < self.rows, "grid row out of range");
        &mut self.cells[row * self.cols..(row + 1) * self.cols]
    }

    /// The cell at (`col`, `row`), writable.
    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut T {
        assert!(
            col < self.cols && row < self.rows,
            "grid index out of range"
        );
        &mut self.cells[row * self.cols + col]
    }

    /// All cells in raster order.
    #[must_use]
    pub fn cells(&self) -> &[T] {
        &self.cells
    }

    /// Positions (`col`, `row`) in raster order.
    pub fn positions(&self) -> impl Iterator<Item = (usize, usize)> + use<T> {
        let cols = self.cols;
        (0..self.rows).flat_map(move |r| (0..cols).map(move |c| (c, r)))
    }
}
