//! H.266 CCLM template selection and luma gathering; arithmetic uses shared DSP.
use super::{Sequence, Unit, references::Availability};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::{vvc_cclm_downsample, vvc_cclm_downsample_collocated, vvc_cclm_predict};

pub(super) fn predict(
    picture: &Picture,
    available: &Availability,
    sequence: &Sequence,
    unit: &Unit,
    block: nc_encoder::Block,
    plane: PlaneIndex,
    out: &mut [u8],
) {
    let n = block;
    let (x, y, width, height) = (n.x / 2, n.y / 2, n.width() / 2, n.height() / 2);
    let left = available.has(1, x as isize - 1, y as isize);
    let above = available.has(1, x as isize, y as isize - 1);
    let luma = picture.plane(PlaneIndex::Y).view();
    let chroma = picture.plane(plane).view();
    let collocated = sequence.tools.chroma_collocated[1];
    // Coordinates are relative to the luma block. Only unavailable left/top
    // taps are replicated; templates themselves are selected by availability.
    let sample = |px: isize, py: isize| {
        let px = (n.x as isize + px).clamp(0, luma.width() as isize - 1) as usize;
        let py = (n.y as isize + py).clamp(0, luma.height() as isize - 1) as usize;
        luma.row(py)[px]
    };
    let downsample = |cx: isize, cy: isize| {
        let (px, mut py) = (2 * cx, 2 * cy);
        let single_row = cy == -1 && n.y.is_multiple_of(1 << sequence.ctu_log2);
        if single_row {
            py = -1;
        }
        let lx = if cx == 0 && !left { px } else { px - 1 };
        if collocated && !single_row {
            let uy = if cy == 0 && !above { py } else { py - 1 };
            vvc_cclm_downsample_collocated([
                sample(px, uy),
                sample(lx, py),
                sample(px, py),
                sample(px + 1, py),
                sample(px, py + 1),
            ])
        } else {
            let dy = if single_row { py } else { py + 1 };
            vvc_cclm_downsample([
                sample(lx, py),
                sample(px, py),
                sample(px + 1, py),
                sample(lx, dy),
                sample(px, dy),
                sample(px + 1, dy),
            ])
        }
    };
    let mut top_count = if above && unit.chroma_mode != 68 {
        width
    } else {
        0
    };
    let mut left_count = if left && unit.chroma_mode != 69 {
        height
    } else {
        0
    };
    if top_count > 0 && unit.chroma_mode == 69 {
        top_count += (width..width + width.min(height))
            .take_while(|&dx| available.has(1, (x + dx) as isize, y as isize - 1))
            .count();
    }
    if left_count > 0 && unit.chroma_mode == 68 {
        left_count += (height..height + width.min(height))
            .take_while(|&dy| available.has(1, x as isize - 1, (y + dy) as isize))
            .count();
    }
    let mut pairs = Vec::with_capacity(4);
    for (top, count, other) in [
        (true, top_count, left_count),
        (false, left_count, top_count),
    ] {
        if count == 0 {
            continue;
        }
        let single = usize::from(other == 0);
        let start = count >> (2 + single);
        let step = (count >> (1 + single)).max(1);
        for i in 0..count.min(2 << single) {
            let position = start + i * step;
            let (cx, cy) = if top {
                (position as isize, -1)
            } else {
                (-1, position as isize)
            };
            let value = chroma.row((y as isize + cy) as usize)[(x as isize + cx) as usize];
            pairs.push([downsample(cx, cy), value]);
        }
    }
    let values: Vec<_> = (0..height)
        .flat_map(|dy| (0..width).map(move |dx| downsample(dx as isize, dy as isize)))
        .collect();
    vvc_cclm_predict(&values, &pairs, out);
}
