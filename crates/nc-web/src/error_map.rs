//! Per-block squared error of a decoded picture against its source, for the
//! demo's PSNR map (analysis builds only: `BrowserDecoder::error_map`;
//! `web/error-map.js` turns the sums into dB and the page draws them).
//!
//! The grid is fixed, not the codec's own partition, so two views of different
//! codecs are measured on the same blocks: `MAP_BLOCK` x `MAP_BLOCK` luma samples,
//! and the co-located half-size blocks of each 4:2:0 chroma plane, so all three
//! planes have the same number of columns and rows (`ceil(ceil(w / 2) / (B /
//! 2)) == ceil(w / B)`). Blocks at the right and bottom edges are clipped to
//! the picture. The page merges 2x2 of them for its coarser view.

use nc_common::{Picture, PlaneIndex, PlaneView, ShownPicture};
use nc_dsp::Dsp;

/// Luma samples per side of a map block (chroma blocks are half that).
pub(crate) const MAP_BLOCK: usize = 8;

/// Replace `sums` with the sum of squared errors of every map block: the
/// luma blocks row by row, then the Cb blocks, then the Cr blocks. Each sum
/// comes from nc-dsp's SSE kernel and fits a `u32` (at most 64 * 255^2).
pub(crate) fn block_sse(
    dsp: &Dsp,
    reference: &Picture,
    decoded: ShownPicture<'_>,
    sums: &mut Vec<u32>,
) {
    sums.clear();
    for (index, block) in [
        (PlaneIndex::Y, MAP_BLOCK),
        (PlaneIndex::U, MAP_BLOCK / 2),
        (PlaneIndex::V, MAP_BLOCK / 2),
    ] {
        plane_sse(
            dsp,
            &reference.plane(index).view(),
            &decoded.plane(index),
            block,
            sums,
        );
    }
}

/// Append the block sums of one plane pair, `block` samples per side.
fn plane_sse(dsp: &Dsp, a: &PlaneView<'_>, b: &PlaneView<'_>, block: usize, sums: &mut Vec<u32>) {
    let (width, height) = (a.width(), a.height());
    for y in (0..height).step_by(block) {
        let h = block.min(height - y);
        for x in (0..width).step_by(block) {
            let w = block.min(width - x);
            let block_a = a.sub_view(x, y, w, h).expect("a block inside the plane");
            let block_b = b.sub_view(x, y, w, h).expect("a block inside the plane");
            let sum = dsp.sse(&block_a, &block_b);
            sums.push(u32::try_from(sum).expect("a block's squared error fits u32"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_each_block_of_each_plane_in_order() {
        // 12x10: luma blocks 8 and 4 wide, 8 and 2 high; chroma 6x5 in 4x4 blocks.
        let reference = Picture::new(12, 10, 0).unwrap();
        let mut decoded = reference.clone();
        // One luma sample off by 3 in the bottom-right block, one Cr sample off by 2 in the first.
        decoded.plane_mut(PlaneIndex::Y).view_mut().row_mut(9)[11] = 3;
        decoded.plane_mut(PlaneIndex::V).view_mut().row_mut(0)[0] = 2;
        let mut sums = vec![7];
        block_sse(
            &Dsp::new(),
            &reference,
            ShownPicture::whole(&decoded),
            &mut sums,
        );
        assert_eq!(sums, [0, 0, 0, 9, 0, 0, 0, 0, 4, 0, 0, 0]);
    }
}
