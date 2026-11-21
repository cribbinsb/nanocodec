//! Diagnostics of the key-frame inheritance oracle
//! (`docs/analysis/key-frame-study-2026-11-21.md`): what a hindsight
//! measurement of each key cell's reuse needs, and the map such a
//! measurement hands back.
//!
//! - [`CellAnalysis`]: the analysis of the frame just coded per 8x8 luma
//!   cell (its sum of squared deviations, the 8x8 motion search's vector
//!   and SAD), read out after every frame (`nanocodec encode
//!   --analysis-log`). An offline pass propagates the frames' reuse back to
//!   the key along the vectors, as MB-tree and TPL do with a lookahead.
//! - A key offset map ([`crate::Encoder::set_next_key_offsets`], `nanocodec
//!   encode --key-offsets`): QP offsets per 8x8 cell imposed on the next key
//!   frame on top of the block AQ terms, budgeted by rate control like any
//!   block offset.
//!
//! Both are oracles for measuring an allocation, not controls: nothing in
//! the encoder writes or reads them unless a caller does, and streams are
//! unchanged without them.

use nc_analysis::FrameAnalysis;

/// One 8x8 luma cell of [`CellAnalysis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisCell {
    /// Sum of squared deviations from the cell's mean (its intra activity).
    pub sum_sq_dev: u32,
    /// SAD of the 8x8 motion search's best match (`None` on a frame without
    /// motion analysis: a key, the first frame).
    pub motion_sad: Option<u32>,
    /// The best match's vector, in luma samples (zero without motion).
    pub mv: (i16, i16),
}

/// The analysis of one frame per 8x8 luma cell, in raster order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellAnalysis {
    /// Cells per row.
    pub cols: usize,
    /// Rows of cells.
    pub rows: usize,
    /// `cols * rows` cells.
    pub cells: Vec<AnalysisCell>,
}

impl CellAnalysis {
    /// Read `analysis` out per cell. The motion field is the 8x8 one, on
    /// the same cell grid as the pre-analysis.
    #[must_use]
    pub fn of(analysis: &FrameAnalysis) -> Self {
        let blocks = &analysis.pre.blocks;
        let fine = analysis
            .motion
            .as_ref()
            .map(|m| &m.fine)
            .filter(|f| f.cols() == blocks.cols() && f.rows() == blocks.rows());
        let cells = blocks
            .cells()
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let motion = fine.map(|f| f.cells()[i]);
                AnalysisCell {
                    sum_sq_dev: b.sum_sq_dev,
                    motion_sad: motion.map(|m| m.sad),
                    mv: motion.map_or((0, 0), |m| (m.mv.x, m.mv.y)),
                }
            })
            .collect();
        Self {
            cols: blocks.cols(),
            rows: blocks.rows(),
            cells,
        }
    }
}

/// `offsets` (the block AQ terms' unclamped offsets of every cell, if any)
/// plus a key offset map `map` of the same cells, QP units. Offsets of
/// another length are none (the reuse term before it has measured cells
/// hands an empty slice, which block AQ ignores too).
#[must_use]
pub(crate) fn with_key_map(offsets: Option<&[f64]>, map: &[f64]) -> Vec<f64> {
    match offsets {
        Some(offsets) if offsets.len() == map.len() => {
            offsets.iter().zip(map).map(|(a, b)| a + b).collect()
        }
        _ => map.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_map_adds_to_the_terms_offsets() {
        assert_eq!(with_key_map(None, &[1., -2.]), vec![1., -2.]);
        assert_eq!(with_key_map(Some(&[0.5, 0.5]), &[1., -2.]), vec![1.5, -1.5]);
        assert_eq!(with_key_map(Some(&[]), &[1., -2.]), vec![1., -2.]);
    }
}
