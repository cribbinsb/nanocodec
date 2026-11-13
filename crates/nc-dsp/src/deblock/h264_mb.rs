//! Whole-macroblock H.264 deblocking (§8.7): the reference walks a
//! [`H264MbEdges`] mask and filters each edge through the per-edge kernels,
//! vertical edges before horizontal ones (the horizontal edges read the
//! vertical results). The ISA versions keep that order.

use super::H264MbEdges;

/// Luma samples between consecutive H.264 edges.
const LUMA_EDGE_SPACING: usize = 4;
/// Chroma samples between consecutive filtered chroma edges (4:2:0: edges
/// 0 and 2 of the luma grid, 4 chroma samples apart).
const CHROMA_EDGE_SPACING: usize = 2;

/// Call `filter(vertical, offset, edge)` for every edge in `edges.mask`,
/// in filtering order; `offset` is the edge's distance into the
/// macroblock in samples of the plane (`chroma`: 4:2:0 chroma).
pub(crate) fn for_each_edge(
    edges: &H264MbEdges,
    chroma: bool,
    mut filter: impl FnMut(bool, usize, &super::H264EdgeParams),
) {
    let spacing = if chroma {
        CHROMA_EDGE_SPACING
    } else {
        LUMA_EDGE_SPACING
    };
    for (d, row) in edges.edges.iter().enumerate() {
        for (e, edge) in row.iter().enumerate() {
            if edges.mask & (1 << (4 * d + e)) != 0 {
                filter(d == 0, e * spacing, edge);
            }
        }
    }
}
