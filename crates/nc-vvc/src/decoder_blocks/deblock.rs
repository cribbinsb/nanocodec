//! H.266 general dual-tree intra edge control; all filtering uses shared DSP.
use super::{PictureParameters, Sequence, SliceHeader, Transform, Unit, transform};
use nc_common::{Picture, PlaneIndex, Sample};
use nc_dsp::{DepthKernels, Pixel, vvc_deblock_beta, vvc_deblock_tc};

#[allow(
    clippy::struct_excessive_bools,
    reason = "cached per-TU flags avoid repeated Unit/Transform pointer chases in edge scan"
)]
struct EdgeBlock<'a> {
    unit: &'a Unit,
    transform: &'a Transform,
    luma_bounds: [u16; 4],
    chroma_bounds: [u16; 4],
    slice: u16,
    tile: u16,
    qp_y: i8,
    coded_y: bool,
    has_subblocks: bool,
    uniform_motion: bool,
}

impl EdgeBlock<'_> {
    #[inline]
    fn bounds(&self, channel: usize) -> (usize, usize, usize, usize) {
        let [x, y, w, h] = if channel == 0 {
            self.luma_bounds
        } else {
            self.chroma_bounds
        };
        (
            usize::from(x),
            usize::from(y),
            usize::from(w),
            usize::from(h),
        )
    }

    #[inline]
    fn size_perp(&self, channel: usize, horizontal: bool) -> usize {
        let [_, _, w, h] = if channel == 0 {
            self.luma_bounds
        } else {
            self.chroma_bounds
        };
        usize::from(if horizontal { h } else { w })
    }
}

fn edge_blocks<'a>(
    units: &'a [Unit],
    sequence: &Sequence,
    headers: &[SliceHeader],
) -> Vec<EdgeBlock<'a>> {
    let total: usize = units.iter().map(|u| u.transforms.len()).sum();
    let mut out = Vec::with_capacity(total);
    for unit in units {
        let has_subblocks =
            unit.affine_models.iter().any(Option::is_some) || !unit.subblocks.is_empty();
        let uniform_motion = unit.motion.is_some()
            && !has_subblocks
            && unit.geometric.is_none()
            && unit.affine_fields[0].is_none()
            && unit.affine_fields[1].is_none();
        let header = &headers[usize::from(unit.slice)];
        let qp_y = transform::qp(unit.quant(), 0, 0, sequence, header) as i8;
        for tf in &unit.transforms {
            let (lb, cb) = if let Some(selection) = unit.sbt
                && std::ptr::eq(tf, unit.transforms.as_ptr())
            {
                let (x, y, w, h) = selection.uncoded(unit.node);
                let rect = [x as u16, y as u16, w as u16, h as u16];
                (rect, rect)
            } else {
                let lb = tf.block;
                let cb = tf.chroma_block.map_or([0; 4], |c| {
                    [c.x as u16, c.y as u16, c.width() as u16, c.height() as u16]
                });
                (
                    [
                        lb.x as u16,
                        lb.y as u16,
                        lb.width() as u16,
                        lb.height() as u16,
                    ],
                    cb,
                )
            };
            out.push(EdgeBlock {
                unit,
                transform: tf,
                luma_bounds: lb,
                chroma_bounds: cb,
                slice: unit.slice,
                tile: unit.tile,
                qp_y,
                coded_y: tf.coded[0],
                has_subblocks,
                uniform_motion,
            });
        }
    }
    out
}

/// Vector difference (1/16 sample units) from which an edge is filtered.
const HALF_SAMPLE: i32 = 8;

/// The grid of the subblock edges deblocking filters inside an affine or
/// `SbTMVP` CU (H.266 §8.8.3.4: 8 luma samples).
const SUBBLOCK_EDGE_GRID: usize = 8;

/// The chroma edge grid in chroma samples (H.266 §8.8.3.3), and the
/// smallest chroma block side on both sides of a long-filtered edge.
const CHROMA_EDGE_GRID: usize = 8;

/// The most chroma lines one call of the chroma edge kernel filters.
const MAX_CHROMA_LINES: usize = 8;

/// Chroma lines of one 4-luma-sample cell along an edge: 2 where chroma
/// is subsampled along it, 4 where it is not.
const CHROMA_PAIR: usize = 2;
const CHROMA_QUAD: usize = 4;

/// Samples along an edge segment whose ends set the LADF luma level.
const LADF_SEGMENT: usize = 4;

#[cfg(test)]
pub(crate) fn apply<T: Pixel>(
    picture: &mut Picture<T>,
    units: &[Unit],
    sequence: &Sequence,
    parameters: &PictureParameters,
    headers: &[SliceHeader],
    dsp: DepthKernels<'_>,
) {
    let mut grid = [Vec::new(), Vec::new()];
    apply_with_scratch(
        picture, units, sequence, parameters, headers, dsp, &mut grid,
    );
}

pub(crate) fn apply_with_scratch<T: Pixel>(
    picture: &mut Picture<T>,
    units: &[Unit],
    sequence: &Sequence,
    parameters: &PictureParameters,
    headers: &[SliceHeader],
    dsp: DepthKernels<'_>,
    grid: &mut [Vec<u32>; 2],
) {
    if headers.iter().all(|h| h.deblock.disabled) {
        return;
    }
    let stride = picture.width().div_ceil(4);
    // Every slice of a picture has its picture header.
    let virtual_boundaries = &headers[0].picture.boundaries;
    let blocks = edge_blocks(units, sequence, headers);
    fill_edge_grid(&blocks, stride, picture.height(), sequence.chroma, grid);
    for horizontal in [false, true] {
        apply_luma(
            picture,
            &blocks,
            &grid[0],
            stride,
            horizontal,
            &virtual_boundaries[usize::from(horizontal)],
            (sequence, parameters, headers),
            dsp,
        );
        if sequence.chroma {
            apply_chroma(
                picture,
                &blocks,
                &grid[1],
                stride,
                horizontal,
                &virtual_boundaries[usize::from(horizontal)],
                (sequence, parameters, headers),
                dsp,
            );
        }
    }
}

#[inline]
fn same_pair_run(
    grid: &[u32],
    (at, along, along_limit): (usize, usize, usize),
    (p_off, along_step): (usize, usize),
    (p_raw, q_raw): (u32, u32),
) -> usize {
    let mut skip = 1usize;
    while along + 4 * skip < along_limit
        && grid[at + skip * along_step] == q_raw
        && grid[at + skip * along_step - p_off] == p_raw
    {
        skip += 1;
    }
    skip
}

#[inline]
fn luma_thresholds_fast(
    (qp_p, qp_q): (i8, i8),
    strength: i32,
    ladf: i32,
    bd_offset: i32,
    depth: nc_common::BitDepth,
    q_header: &SliceHeader,
) -> (i32, i32) {
    let qp = ((i32::from(qp_p) + i32::from(qp_q) + 1) >> 1) - bd_offset + ladf;
    let [beta_offset, tc_offset] = q_header.deblock.offsets[0].map(i32::from);
    let beta = vvc_deblock_beta(
        crate::filter::BETA[(qp + 2 * beta_offset).clamp(0, 63) as usize],
        depth,
    );
    let tc = vvc_deblock_tc(
        crate::filter::TC[(qp + 2 * (strength - 1) + 2 * tc_offset).clamp(0, 65) as usize],
        depth,
    );
    (beta, tc)
}

#[inline]
fn filter_luma_single_fast<T: Pixel>(
    y_view: &mut nc_common::PlaneViewMut<'_, T>,
    (vw, vh): (usize, usize),
    at: (usize, usize),
    horizontal: bool,
    (beta, tc): (i32, i32),
    lengths: [usize; 2],
    dsp: DepthKernels<'_>,
) {
    let inside = if horizontal {
        at.1 >= 8 && at.1 + 8 <= vh && at.0 + 4 <= vw
    } else {
        at.0 >= 8 && at.0 + 8 <= vw && at.1 + 4 <= vh
    };
    if inside {
        dsp.vvc_deblock_luma_segment(y_view, at, !horizontal, beta, tc, lengths);
    } else {
        filter_luma_border_view(y_view, dsp, at, horizontal, beta, tc, lengths);
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_luma_run_fast<T: Pixel>(
    y_view: &mut nc_common::PlaneViewMut<'_, T>,
    (vw, vh): (usize, usize),
    (along, perp, along_limit): (usize, usize, usize),
    run: usize,
    horizontal: bool,
    interior_perp: bool,
    (beta, tc): (i32, i32),
    lengths: [usize; 2],
    dsp: DepthKernels<'_>,
) {
    let mut k = 0;
    while interior_perp && k + 2 <= run && along + 4 * k + 8 <= along_limit {
        let pos = along + 4 * k;
        let at = if horizontal { (pos, perp) } else { (perp, pos) };
        dsp.vvc_deblock_luma_edge(y_view, at, !horizontal, beta, tc, lengths);
        k += 2;
    }
    while k < run {
        let pos = along + 4 * k;
        let at = if horizontal { (pos, perp) } else { (perp, pos) };
        filter_luma_single_fast(y_view, (vw, vh), at, horizontal, (beta, tc), lengths, dsp);
        k += 1;
    }
}

#[inline]
fn unit_luma_motion(block: &EdgeBlock<'_>, x: usize, y: usize) -> Option<super::motion::Motion> {
    if block.uniform_motion {
        block.unit.motion
    } else {
        super::motion::at(block.unit, x, y)
    }
}

#[inline]
fn subblock_motion_bs(
    p: &EdgeBlock<'_>,
    q: &EdgeBlock<'_>,
    (x, y): (usize, usize),
    horizontal: bool,
    sides: (&SliceHeader, &SliceHeader),
) -> i32 {
    let px = x - usize::from(!horizontal);
    let py = y - usize::from(horizontal);
    match (unit_luma_motion(p, px, py), unit_luma_motion(q, x, y)) {
        (Some(pm), Some(qm)) => motion_strength(pm, qm, sides),
        _ => 0,
    }
}

#[inline]
fn bs_invariant_along_run(
    p: &EdgeBlock<'_>,
    q: &EdgeBlock<'_>,
    bs: i32,
    perp: usize,
    horizontal: bool,
) -> bool {
    if std::ptr::eq(p, q) {
        return false;
    }
    if bs == 2
        || p.unit.inter.is_none()
        || q.unit.inter.is_none()
        || p.coded_y
        || q.coded_y
        || p.unit.block_vector.is_some()
        || q.unit.block_vector.is_some()
    {
        return true;
    }
    if std::ptr::eq(p.unit, q.unit) {
        let origin = if horizontal {
            q.unit.node.y
        } else {
            q.unit.node.x
        };
        return !q.has_subblocks || !(perp - origin).is_multiple_of(SUBBLOCK_EDGE_GRID);
    }
    p.uniform_motion && q.uniform_motion
}

#[allow(clippy::too_many_arguments)]
fn filter_luma_subblock_run<T: Pixel>(
    y_view: &mut nc_common::PlaneViewMut<'_, T>,
    (vw, vh): (usize, usize),
    (p, q): (&EdgeBlock<'_>, &EdgeBlock<'_>),
    (along, perp, along_limit): (usize, usize, usize),
    (run, first_bs): (usize, i32),
    (horizontal, interior_perp, boundary): (bool, bool, bool),
    (bd_offset, depth): (i32, nc_common::BitDepth),
    sides: (&SliceHeader, &SliceHeader),
    dsp: DepthKernels<'_>,
) {
    let mut cached: Option<(i32, (i32, i32), [usize; 2])> = None;
    let mut k = 0usize;
    while k < run {
        let pos = along + 4 * k;
        let at = if horizontal { (pos, perp) } else { (perp, pos) };
        let bs = if k == 0 {
            first_bs
        } else {
            subblock_motion_bs(p, q, at, horizontal, sides)
        };
        if bs == 0 {
            k += 1;
            continue;
        }
        let (thresholds, lengths) = match cached {
            Some((prev_bs, th, len)) if prev_bs == bs => (th, len),
            _ => {
                let th = luma_thresholds_fast((p.qp_y, q.qp_y), bs, 0, bd_offset, depth, sides.1);
                let len = luma_lengths(p, q, perp, horizontal, boundary);
                cached = Some((bs, th, len));
                (th, len)
            }
        };
        if interior_perp && k + 2 <= run && pos + 8 <= along_limit {
            let next_at = if horizontal {
                (pos + 4, perp)
            } else {
                (perp, pos + 4)
            };
            let next_bs = subblock_motion_bs(p, q, next_at, horizontal, sides);
            if next_bs == bs {
                dsp.vvc_deblock_luma_edge(
                    y_view,
                    at,
                    !horizontal,
                    thresholds.0,
                    thresholds.1,
                    lengths,
                );
                k += 2;
                continue;
            }
            if next_bs == 0 {
                filter_luma_single_fast(y_view, (vw, vh), at, horizontal, thresholds, lengths, dsp);
                k += 2;
                continue;
            }
        }
        filter_luma_single_fast(y_view, (vw, vh), at, horizontal, thresholds, lengths, dsp);
        k += 1;
    }
}

fn filter_luma_border_view<T: Pixel>(
    view: &mut nc_common::PlaneViewMut<'_, T>,
    dsp: DepthKernels<'_>,
    origin: (usize, usize),
    horizontal: bool,
    beta: i32,
    tc: i32,
    lengths: [usize; 2],
) {
    let (vw, vh) = (view.width(), view.height());
    let mut edge = [[T::default(); 16]; 4];
    for (line, values) in edge.iter_mut().enumerate() {
        for (tap, slot) in values.iter_mut().enumerate() {
            let (x, y) = position(origin, horizontal, line, tap, 16);
            let row = y.clamp(0, vh as isize - 1) as usize;
            let col = x.clamp(0, vw as isize - 1) as usize;
            *slot = view.row_mut(row)[col];
        }
    }
    dsp.vvc_deblock_luma(&mut edge, beta, tc, lengths);
    for (line, values) in edge.iter().enumerate() {
        for (tap, &value) in values.iter().enumerate() {
            let (x, y) = position(origin, horizontal, line, tap, 16);
            if x >= 0 && y >= 0 && (x as usize) < vw && (y as usize) < vh {
                view.row_mut(y as usize)[x as usize] = value;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_luma_no_ladf<T: Pixel>(
    picture: &mut Picture<T>,
    blocks: &[EdgeBlock<'_>],
    grid: &[u32],
    stride: usize,
    horizontal: bool,
    closed_positions: &[u32],
    (sequence, parameters, headers): (&Sequence, &PictureParameters, &[SliceHeader]),
    dsp: DepthKernels<'_>,
) {
    let (width, height) = (picture.width(), picture.height());
    let (perp_limit, along_limit) = if horizontal {
        (height, width)
    } else {
        (width, height)
    };
    let step = if horizontal { (stride, 1) } else { (1, stride) };
    let depth = nc_common::BitDepth::of::<T>(dsp.depth());
    let bd_offset = crate::decoder_headers::qp_bd_offset(depth);
    let ctu_size = 1usize << sequence.ctu_log2;
    let mut y_view = picture.plane_mut(PlaneIndex::Y).view_mut();
    for perp in (4..perp_limit).step_by(4) {
        if closed_positions.contains(&(perp as u32)) {
            continue;
        }
        let interior_perp = perp >= 8 && perp + 8 <= perp_limit;
        let boundary = horizontal && perp.is_multiple_of(ctu_size);
        let mut along = 0;
        while along < along_limit {
            let (x, y) = if horizontal {
                (along, perp)
            } else {
                (perp, along)
            };
            let at = (y >> 2) * stride + (x >> 2);
            let pair = (grid[at - step.0], grid[at]);
            let q = &blocks[pair.1 as usize];
            if pair.0 == pair.1 {
                let origin = if horizontal {
                    q.unit.node.y
                } else {
                    q.unit.node.x
                };
                if !q.has_subblocks || !(perp - origin).is_multiple_of(SUBBLOCK_EDGE_GRID) {
                    along += 4 * same_pair_run(grid, (at, along, along_limit), step, pair);
                    continue;
                }
            }
            let p = &blocks[pair.0 as usize];
            let header = &headers[usize::from(q.slice)];
            if header.deblock.disabled
                || (p.slice != q.slice || p.tile != q.tile)
                    && !open_edge(p.unit, q.unit, (sequence, parameters), headers)
            {
                along += 4 * same_pair_run(grid, (at, along, along_limit), step, pair);
                continue;
            }
            let sides = (&headers[usize::from(p.slice)], header);
            let bs = strength(p, q, 0, (x, y), horizontal, sides);
            let inv = bs_invariant_along_run(p, q, bs, perp, horizontal);
            let run = same_pair_run(grid, (at, along, along_limit), step, pair);
            // Only an SPS that enables palette has palette CUs.
            let kept = sequence
                .tools
                .palette
                .then(|| {
                    palette_sides(
                        &y_view.as_view(),
                        (p, q),
                        (x, y),
                        horizontal,
                        (4 * run, LUMA_REACH),
                    )
                })
                .flatten();
            if inv {
                if bs > 0 {
                    let thresholds =
                        luma_thresholds_fast((p.qp_y, q.qp_y), bs, 0, bd_offset, depth, header);
                    let lengths = luma_lengths(p, q, perp, horizontal, boundary);
                    filter_luma_run_fast(
                        &mut y_view,
                        (width, height),
                        (along, perp, along_limit),
                        run,
                        horizontal,
                        interior_perp,
                        thresholds,
                        lengths,
                        dsp,
                    );
                }
            } else {
                filter_luma_subblock_run(
                    &mut y_view,
                    (width, height),
                    (p, q),
                    (along, perp, along_limit),
                    (run, bs),
                    (horizontal, interior_perp, boundary),
                    (bd_offset, depth),
                    sides,
                    dsp,
                );
            }
            if let Some(kept) = kept {
                restore(&mut y_view, kept);
            }
            along += 4 * run;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_luma<T: Pixel>(
    picture: &mut Picture<T>,
    blocks: &[EdgeBlock<'_>],
    grid: &[u32],
    stride: usize,
    horizontal: bool,
    closed_positions: &[u32],
    (sequence, parameters, headers): (&Sequence, &PictureParameters, &[SliceHeader]),
    dsp: DepthKernels<'_>,
) {
    if sequence.tools.ladf.is_empty() {
        apply_luma_no_ladf(
            picture,
            blocks,
            grid,
            stride,
            horizontal,
            closed_positions,
            (sequence, parameters, headers),
            dsp,
        );
        return;
    }
    let (width, height) = (picture.width(), picture.height());
    let (perp_limit, along_limit) = if horizontal {
        (height, width)
    } else {
        (width, height)
    };
    let step = if horizontal { (stride, 1) } else { (1, stride) };
    let depth = nc_common::BitDepth::of::<T>(dsp.depth());
    let bd_offset = crate::decoder_headers::qp_bd_offset(depth);
    for perp in (4..perp_limit).step_by(4) {
        if closed_positions.contains(&(perp as u32)) {
            continue;
        }
        let mut along = 0;
        while along < along_limit {
            let (x, y) = if horizontal {
                (along, perp)
            } else {
                (perp, along)
            };
            let at = (y >> 2) * stride + (x >> 2);
            let pair = (grid[at - step.0], grid[at]);
            let q = &blocks[pair.1 as usize];
            if pair.0 == pair.1 {
                let origin = if horizontal {
                    q.unit.node.y
                } else {
                    q.unit.node.x
                };
                if !q.has_subblocks || !(perp - origin).is_multiple_of(SUBBLOCK_EDGE_GRID) {
                    along += 4 * same_pair_run(grid, (at, along, along_limit), step, pair);
                    continue;
                }
            }
            let p = &blocks[pair.0 as usize];
            let header = &headers[usize::from(q.slice)];
            if header.deblock.disabled
                || (p.slice != q.slice || p.tile != q.tile)
                    && !open_edge(p.unit, q.unit, (sequence, parameters), headers)
            {
                along += 4 * same_pair_run(grid, (at, along, along_limit), step, pair);
                continue;
            }
            let sides = (&headers[usize::from(p.slice)], header);
            let bs = strength(p, q, 0, (x, y), horizontal, sides);
            let inv = bs_invariant_along_run(p, q, bs, perp, horizontal);
            if bs == 0 {
                along += 4 * if inv {
                    same_pair_run(grid, (at, along, along_limit), step, pair)
                } else {
                    1
                };
                continue;
            }
            // Only an SPS that enables palette has palette CUs.
            let kept = sequence
                .tools
                .palette
                .then(|| {
                    palette_sides(
                        &picture.plane(PlaneIndex::Y).view(),
                        (p, q),
                        (x, y),
                        horizontal,
                        (4, LUMA_REACH),
                    )
                })
                .flatten();
            filter_luma_edge(
                picture,
                (p, q),
                (x, y),
                horizontal,
                bs,
                false,
                (depth, bd_offset),
                sequence,
                sides,
                dsp,
            );
            if let Some(kept) = kept {
                restore(&mut picture.plane_mut(PlaneIndex::Y).view_mut(), kept);
            }
            along += 4;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_chroma<T: Pixel>(
    picture: &mut Picture<T>,
    blocks: &[EdgeBlock<'_>],
    grid: &[u32],
    stride: usize,
    horizontal: bool,
    closed_positions: &[u32],
    (sequence, parameters, headers): (&Sequence, &PictureParameters, &[SliceHeader]),
    dsp: DepthKernels<'_>,
) {
    let (width, height) = (picture.width(), picture.height());
    let (perp_limit, along_limit) = if horizontal {
        (height, width)
    } else {
        (width, height)
    };
    let (p_off, along_step) = if horizontal { (stride, 1) } else { (1, stride) };
    let depth = nc_common::BitDepth::of::<T>(dsp.depth());
    let bd_offset = crate::decoder_headers::qp_bd_offset(depth);
    // Chroma edges lie on an 8-sample chroma grid (§8.8.3.3); a 4-luma
    // grid cell holds `cell_lines` chroma lines along the edge.
    let (sx, sy) = sequence.subsampling();
    let (perp_shift, along_shift) = if horizontal { (sy, sx) } else { (sx, sy) };
    let edge_step = CHROMA_EDGE_GRID << perp_shift;
    let cell_lines = 4 >> along_shift;
    let max_run = MAX_CHROMA_LINES / cell_lines;
    for perp in (edge_step..perp_limit).step_by(edge_step) {
        if closed_positions.contains(&(perp as u32)) {
            continue;
        }
        let boundary = horizontal && perp.is_multiple_of(1 << sequence.ctu_log2);
        let mut along = 0;
        while along < along_limit {
            let (x, y) = if horizontal {
                (along, perp)
            } else {
                (perp, along)
            };
            let at = (y >> 2) * stride + (x >> 2);
            let (p_raw, q_raw) = (grid[at - p_off], grid[at]);
            let mut run = 1usize;
            while run < max_run
                && along + 4 * run < along_limit
                && grid[at + run * along_step - p_off] == p_raw
                && grid[at + run * along_step] == q_raw
            {
                run += 1;
            }
            if p_raw == q_raw {
                along += 4 * run;
                continue;
            }
            let (p, q) = (&blocks[p_raw as usize], &blocks[q_raw as usize]);
            let header = &headers[usize::from(q.unit.slice)];
            if header.deblock.disabled
                || !open_edge(p.unit, q.unit, (sequence, parameters), headers)
            {
                along += 4 * run;
                continue;
            }
            let sides = (&headers[usize::from(p.unit.slice)], header);
            // The long filter needs 8 chroma samples on each side.
            let large = p.size_perp(1, horizontal) >> perp_shift >= CHROMA_EDGE_GRID
                && q.size_perp(1, horizontal) >> perp_shift >= CHROMA_EDGE_GRID;
            for plane in [PlaneIndex::U, PlaneIndex::V] {
                let bs = chroma_strength(p, q, plane as usize);
                if bs == 0 || (bs == 1 && !large) {
                    continue;
                }
                let (beta, tc) = edge_thresholds(
                    (p, q),
                    plane as usize,
                    bs,
                    0,
                    bd_offset,
                    depth,
                    sequence,
                    sides,
                );
                // Only an SPS that enables palette has palette CUs.
                let kept = sequence
                    .tools
                    .palette
                    .then(|| {
                        palette_sides(
                            &picture.plane(plane).view(),
                            (p, q),
                            (x >> sx, y >> sy),
                            horizontal,
                            (run * cell_lines, CHROMA_REACH),
                        )
                    })
                    .flatten();
                filter_chroma_run(
                    picture,
                    dsp,
                    plane,
                    (x >> sx, y >> sy),
                    (run * cell_lines, cell_lines),
                    horizontal,
                    (beta, tc),
                    (large, boundary),
                );
                if let Some(kept) = kept {
                    restore(&mut picture.plane_mut(plane).view_mut(), kept);
                }
            }
            along += 4 * run;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn edge_thresholds(
    (p, q): (&EdgeBlock<'_>, &EdgeBlock<'_>),
    plane: usize,
    strength: i32,
    ladf: i32,
    bd_offset: i32,
    depth: nc_common::BitDepth,
    sequence: &Sequence,
    (p_header, header): (&SliceHeader, &SliceHeader),
) -> (i32, i32) {
    let qp = (i32::from(transform::qp(
        p.unit.quant(),
        p.transform.joint,
        plane,
        sequence,
        p_header,
    )) + i32::from(transform::qp(
        q.unit.quant(),
        q.transform.joint,
        plane,
        sequence,
        header,
    )) + 1)
        >> 1;
    let qp = qp - bd_offset + ladf;
    let [beta_offset, tc_offset] = header.deblock.offsets[plane].map(i32::from);
    let beta = vvc_deblock_beta(
        crate::filter::BETA[(qp + 2 * beta_offset).clamp(0, 63) as usize],
        depth,
    );
    let tc = vvc_deblock_tc(
        crate::filter::TC[(qp + 2 * (strength - 1) + 2 * tc_offset).clamp(0, 65) as usize],
        depth,
    );
    (beta, tc)
}

#[allow(clippy::too_many_arguments)]
fn filter_luma_edge<T: Pixel>(
    picture: &mut Picture<T>,
    (p, q): (&EdgeBlock<'_>, &EdgeBlock<'_>),
    (x, y): (usize, usize),
    horizontal: bool,
    strength: i32,
    eight_lines: bool,
    (depth, bd_offset): (nc_common::BitDepth, i32),
    sequence: &Sequence,
    sides: (&SliceHeader, &SliceHeader),
    dsp: DepthKernels<'_>,
) {
    let perpendicular = if horizontal { y } else { x };
    let boundary = horizontal && y.is_multiple_of(1 << sequence.ctu_log2);
    let ladf = ladf_offset(picture, &sequence.tools.ladf, (x, y), horizontal);
    let (beta, tc) = edge_thresholds((p, q), 0, strength, ladf, bd_offset, depth, sequence, sides);
    let lengths = luma_lengths(p, q, perpendicular, horizontal, boundary);
    if eight_lines {
        dsp.vvc_deblock_luma_edge(
            &mut picture.plane_mut(PlaneIndex::Y).view_mut(),
            (x, y),
            !horizontal,
            beta,
            tc,
            lengths,
        );
    } else {
        filter_luma(picture, dsp, (x, y), horizontal, beta, tc, lengths);
    }
}

fn open_edge(
    p: &Unit,
    q: &Unit,
    (sequence, parameters): (&Sequence, &PictureParameters),
    headers: &[SliceHeader],
) -> bool {
    if p.slice == q.slice && p.tile == q.tile {
        return true;
    }
    let partitioning = &parameters.partitioning;
    let subpictures = [p, q].map(|unit| headers[usize::from(unit.slice)].subpicture);
    let layout = &sequence.subpictures.layout;
    (p.tile == q.tile || partitioning.filter_across_tiles)
        && (p.slice == q.slice || partitioning.filter_across_slices)
        && (subpictures[0] == subpictures[1]
            || subpictures.iter().all(|&index| layout[index].filter_across))
}

fn ladf_offset<T: Sample>(
    picture: &Picture<T>,
    intervals: &[(u16, i8)],
    (x, y): (usize, usize),
    horizontal: bool,
) -> i32 {
    let Some((&(_, lowest), upper)) = intervals.split_first() else {
        return 0;
    };
    let luma = picture.plane(PlaneIndex::Y).view();
    let level = [0, LADF_SEGMENT - 1]
        .into_iter()
        .map(|along| {
            let (p, q) = if horizontal {
                ((x + along, y - 1), (x + along, y))
            } else {
                ((x - 1, y + along), (x, y + along))
            };
            luma.row(p.1)[p.0].to_i32() + luma.row(q.1)[q.0].to_i32()
        })
        .sum::<i32>()
        >> 2;
    upper
        .iter()
        .take_while(|&&(lower, _)| level > i32::from(lower))
        .last()
        .map_or(i32::from(lowest), |&(_, offset)| i32::from(offset))
}

fn filter_luma<T: Pixel>(
    picture: &mut Picture<T>,
    dsp: DepthKernels<'_>,
    at: (usize, usize),
    horizontal: bool,
    beta: i32,
    tc: i32,
    lengths: [usize; 2],
) {
    const REACH: usize = 8;
    const LINES: usize = 4;
    let (x, y) = at;
    let (width, height) = (picture.width(), picture.height());
    let inside = if horizontal {
        y >= REACH && y + REACH <= height && x + LINES <= width
    } else {
        x >= REACH && x + REACH <= width && y + LINES <= height
    };
    if inside {
        dsp.vvc_deblock_luma_segment(
            &mut picture.plane_mut(PlaneIndex::Y).view_mut(),
            at,
            !horizontal,
            beta,
            tc,
            lengths,
        );
    } else {
        let mut edge = gather::<T, 4, 16>(picture, PlaneIndex::Y, at, horizontal);
        dsp.vvc_deblock_luma(&mut edge, beta, tc, lengths);
        store(picture, PlaneIndex::Y, at, horizontal, &edge);
    }
}

#[allow(clippy::too_many_arguments)]
/// Filter `lines` chroma lines of an edge from `(cx, cy)`, decided in
/// segments of `segment` lines: the chroma lines of one 4-luma-sample grid
/// cell (VTM `xEdgeFilterChroma`).
fn filter_chroma_run<T: Pixel>(
    picture: &mut Picture<T>,
    dsp: DepthKernels<'_>,
    plane: PlaneIndex,
    (cx, cy): (usize, usize),
    (lines, segment): (usize, usize),
    horizontal: bool,
    (beta, tc): (i32, i32),
    (large, boundary): (bool, bool),
) {
    const REACH: usize = 4;
    let (vw, vh) = {
        let view = picture.plane(plane).view();
        (view.width(), view.height())
    };
    let inside = if horizontal {
        cy >= REACH && cy + REACH <= vh && cx + lines <= vw
    } else {
        cx >= REACH && cx + REACH <= vw && cy + lines <= vh
    };
    if inside {
        let mut view = picture.plane_mut(plane).view_mut();
        dsp.vvc_deblock_chroma_edge_segments(
            &mut view,
            ((cx, cy), !horizontal),
            (lines, segment),
            (beta, tc),
            (large, boundary),
        );
        return;
    }
    for line in (0..lines).step_by(segment) {
        let at = if horizontal {
            (cx + line, cy)
        } else {
            (cx, cy + line)
        };
        if segment == CHROMA_PAIR {
            let mut edge = gather::<T, CHROMA_PAIR, { 2 * REACH }>(picture, plane, at, horizontal);
            dsp.vvc_deblock_chroma(&mut edge, beta, tc, large, boundary);
            store(picture, plane, at, horizontal, &edge);
        } else {
            let mut edge = gather::<T, CHROMA_QUAD, { 2 * REACH }>(picture, plane, at, horizontal);
            dsp.vvc_deblock_chroma_quad(&mut edge, (beta, tc), (large, boundary));
            store(picture, plane, at, horizontal, &edge);
        }
    }
}

fn position(
    (x, y): (usize, usize),
    horizontal: bool,
    line: usize,
    tap: usize,
    width: usize,
) -> (isize, isize) {
    let offset = tap as isize - (width / 2) as isize;
    if horizontal {
        ((x + line) as isize, y as isize + offset)
    } else {
        (x as isize + offset, (y + line) as isize)
    }
}

fn gather<T: Sample, const L: usize, const W: usize>(
    picture: &Picture<T>,
    plane: PlaneIndex,
    origin: (usize, usize),
    horizontal: bool,
) -> [[T; W]; L] {
    let view = picture.plane(plane).view();
    let (x, y) = origin;
    let inside = if horizontal {
        y >= W / 2 && y + W / 2 <= view.height() && x + L <= view.width()
    } else {
        x >= W / 2 && x + W / 2 <= view.width() && y + L <= view.height()
    };
    if inside {
        return crate::filter::gather(picture, plane, x, y, horizontal);
    }
    std::array::from_fn(|line| {
        std::array::from_fn(|tap| {
            let (x, y) = position(origin, horizontal, line, tap, W);
            view.row(y.clamp(0, view.height() as isize - 1) as usize)
                [x.clamp(0, view.width() as isize - 1) as usize]
        })
    })
}

fn store<T: Sample, const L: usize, const W: usize>(
    picture: &mut Picture<T>,
    plane: PlaneIndex,
    origin: (usize, usize),
    horizontal: bool,
    edge: &[[T; W]; L],
) {
    let mut view = picture.plane_mut(plane).view_mut();
    let (x, y) = origin;
    let inside = if horizontal {
        y >= W / 2 && y + W / 2 <= view.height() && x + L <= view.width()
    } else {
        x >= W / 2 && x + W / 2 <= view.width() && y + L <= view.height()
    };
    if inside {
        crate::filter::store(picture, plane, x, y, horizontal, edge);
        return;
    }
    for (line, values) in edge.iter().enumerate() {
        for (tap, &value) in values.iter().enumerate() {
            let (x, y) = position(origin, horizontal, line, tap, W);
            if x >= 0 && y >= 0 && (x as usize) < view.width() && (y as usize) < view.height() {
                view.row_mut(y as usize)[x as usize] = value;
            }
        }
    }
}

#[inline]
fn chroma_strength(p: &EdgeBlock<'_>, q: &EdgeBlock<'_>, plane: usize) -> i32 {
    if p.unit.intra() || q.unit.intra() {
        return if p.unit.bdpcm[1] != 0 && q.unit.bdpcm[1] != 0 {
            0
        } else {
            2
        };
    }
    if matches!(p.unit.inter, Some(super::Inter::Ciip(_)))
        || matches!(q.unit.inter, Some(super::Inter::Ciip(_)))
    {
        return 2;
    }
    i32::from(
        p.transform.coded[plane]
            || q.transform.coded[plane]
            || p.transform.joint != 0
            || q.transform.joint != 0,
    )
}

// H.266 §8.8.3: intra edges, coded residuals, then reference/motion changes.
fn strength(
    p: &EdgeBlock<'_>,
    q: &EdgeBlock<'_>,
    plane: usize,
    (x, y): (usize, usize),
    horizontal: bool,
    headers: (&SliceHeader, &SliceHeader),
) -> i32 {
    if p.unit.intra() || q.unit.intra() {
        let channel = usize::from(plane != 0);
        return if p.unit.bdpcm[channel] != 0 && q.unit.bdpcm[channel] != 0 {
            0
        } else {
            2
        };
    }
    if matches!(p.unit.inter, Some(super::Inter::Ciip(_)))
        || matches!(q.unit.inter, Some(super::Inter::Ciip(_)))
    {
        return 2;
    }
    if !std::ptr::eq(p, q)
        && (p.transform.coded[plane]
            || q.transform.coded[plane]
            || (plane != 0 && (p.transform.joint != 0 || q.transform.joint != 0)))
    {
        return 1;
    }
    if std::ptr::eq(p.unit, q.unit) {
        let (position, origin) = if horizontal {
            (y, q.unit.node.y)
        } else {
            (x, q.unit.node.x)
        };
        if !q.has_subblocks || !(position - origin).is_multiple_of(SUBBLOCK_EDGE_GRID) {
            return 0;
        }
    }
    if plane == 0 {
        // A palette CU beside a CU of another prediction mode (§8.8.3.5).
        if p.unit.palette.is_some() != q.unit.palette.is_some() {
            return 1;
        }
        match (p.unit.block_vector, q.unit.block_vector) {
            (Some(p), Some(q)) => {
                return i32::from(
                    (p.x - q.x).abs() >= HALF_SAMPLE || (p.y - q.y).abs() >= HALF_SAMPLE,
                );
            }
            (None, None) => {}
            _ => return 1,
        }
        return subblock_motion_bs(p, q, (x, y), horizontal, headers);
    }
    0
}

/// The deepest samples the luma and chroma filters modify on each side of
/// an edge (§8.8.3.6: `maxFilterLength` 7 and 3).
const LUMA_REACH: usize = 7;
const CHROMA_REACH: usize = 3;

/// The samples of an edge segment's palette-coded sides, which deblocking
/// leaves as they are (§8.8.3.6.6, §8.8.3.6.8: `nDp` or `nDq` is 0 when the
/// CU has `pred_mode_plt_flag`); `None` without palette CUs. `(x, y)` is
/// the first q sample in plane samples; `length` samples along the edge.
#[inline]
fn palette_sides<T: Pixel>(
    view: &nc_common::PlaneView<'_, T>,
    (p, q): (&EdgeBlock<'_>, &EdgeBlock<'_>),
    (x, y): (usize, usize),
    horizontal: bool,
    (length, reach): (usize, usize),
) -> Option<Vec<(usize, usize, T)>> {
    let sides = [p.unit.palette.is_some(), q.unit.palette.is_some()];
    if sides == [false; 2] {
        return None;
    }
    Some(palette_samples(
        view,
        sides,
        (x, y),
        horizontal,
        (length, reach),
    ))
}

/// The samples of [`palette_sides`] when either side is a palette CU (out
/// of line: other edges pay only the check).
#[cold]
#[inline(never)]
fn palette_samples<T: Pixel>(
    view: &nc_common::PlaneView<'_, T>,
    sides: [bool; 2],
    (x, y): (usize, usize),
    horizontal: bool,
    (length, reach): (usize, usize),
) -> Vec<(usize, usize, T)> {
    let (across, along) = if horizontal { (y, x) } else { (x, y) };
    let (across_limit, along_limit) = if horizontal {
        (view.height(), view.width())
    } else {
        (view.width(), view.height())
    };
    let mut kept = Vec::new();
    for (side, palette) in sides.into_iter().enumerate() {
        if !palette {
            continue;
        }
        let span = if side == 0 {
            across.saturating_sub(reach)..across
        } else {
            across..(across + reach).min(across_limit)
        };
        for a in span {
            for b in along..(along + length).min(along_limit) {
                let (sx, sy) = if horizontal { (b, a) } else { (a, b) };
                kept.push((sx, sy, view.row(sy)[sx]));
            }
        }
    }
    kept
}

/// Put back the samples [`palette_sides`] kept.
#[cold]
#[inline(never)]
fn restore<T: Pixel>(view: &mut nc_common::PlaneViewMut<'_, T>, kept: Vec<(usize, usize, T)>) {
    for (x, y, value) in kept {
        view.row_mut(y)[x] = value;
    }
}

fn fill_edge_grid(
    blocks: &[EdgeBlock<'_>],
    stride: usize,
    height: usize,
    chroma: bool,
    grid: &mut [Vec<u32>; 2],
) {
    let cells = stride * height.div_ceil(4);
    if grid[0].len() != cells {
        grid[0].resize(cells, 0);
    }
    if chroma && grid[1].len() != cells {
        grid[1].resize(cells, 0);
    }
    let ch_end = 1 + usize::from(chroma);
    for (index, block) in blocks.iter().enumerate() {
        let idx = index as u32;
        let unit = block.unit;
        let range = if unit.combined {
            0..ch_end
        } else {
            unit.channel..(unit.channel + 1).min(ch_end)
        };
        for channel in range {
            let n = if channel == 0 {
                block.transform.block
            } else {
                let Some(n) = block.transform.chroma_block else {
                    continue;
                };
                n
            };
            let cols = n.width().div_ceil(4);
            let x4 = n.x >> 2;
            for ry in (n.y..n.y + n.height()).step_by(4) {
                let base = (ry >> 2) * stride + x4;
                grid[channel][base..base + cols].fill(idx);
            }
        }
    }
}

fn motion_strength(
    p: super::motion::Motion,
    q: super::motion::Motion,
    headers: (&SliceHeader, &SliceHeader),
) -> i32 {
    fn entry(
        motion: super::motion::Motion,
        header: &SliceHeader,
        list: usize,
    ) -> Option<(&crate::decoder_headers::ReferenceEntry, crate::motion::Mv16)> {
        motion.lists[list].map(|v| {
            (
                &header.references[list].entries[usize::from(v.reference)],
                v.vector,
            )
        })
    }
    let (p0, p1) = (entry(p, headers.0, 0), entry(p, headers.0, 1));
    let (q0, q1) = (entry(q, headers.1, 0), entry(q, headers.1, 1));
    let same =
        |a: Option<(&crate::decoder_headers::ReferenceEntry, crate::motion::Mv16)>,
         b: Option<(&crate::decoder_headers::ReferenceEntry, crate::motion::Mv16)>| match (
            a, b,
        ) {
            (Some((ar, av)), Some((br, bv))) => {
                ar == br && (av.x - bv.x).abs() < 8 && (av.y - bv.y).abs() < 8
            }
            (None, None) => true,
            _ => false,
        };
    i32::from(!((same(p0, q0) && same(p1, q1)) || (same(p0, q1) && same(p1, q0))))
}

fn luma_lengths(
    p: &EdgeBlock<'_>,
    q: &EdgeBlock<'_>,
    position: usize,
    horizontal: bool,
    boundary: bool,
) -> [usize; 2] {
    let size_p = p.size_perp(0, horizontal);
    let size_q = q.size_perp(0, horizontal);
    if std::ptr::eq(p, q) {
        let (x, y, _, _) = q.bounds(0);
        let origin = if horizontal { y } else { x };
        let distance = (position - origin).min(origin + size_q - position);
        return [if distance <= 4 {
            1
        } else if distance <= 8 {
            2
        } else {
            3
        }; 2];
    }
    if size_p <= 4 || size_q <= 4 {
        return [1, 1];
    }
    let reach_p = if size_p < 32 || boundary {
        3
    } else if p.has_subblocks {
        5
    } else {
        7
    };
    let reach_q = if size_q < 32 {
        3
    } else if q.has_subblocks {
        5
    } else {
        7
    };
    [reach_p, reach_q]
}

#[cfg(test)]
mod sample_tests {
    use super::*;

    #[test]
    fn edge_buffers_preserve_border_and_orientation_semantics() {
        check_samples::<4, 16>(PlaneIndex::Y);
        check_samples::<2, 8>(PlaneIndex::U);
    }

    fn check_samples<const L: usize, const W: usize>(plane: PlaneIndex) {
        let mut picture = Picture::new(40, 36, 0).unwrap();
        let mut view = picture.plane_mut(plane).view_mut();
        let (width, height) = (view.width(), view.height());
        for y in 0..height {
            for (x, value) in view.row_mut(y).iter_mut().enumerate() {
                *value = (x * 17 + y * 31) as u8;
            }
        }
        for horizontal in [false, true] {
            for y in (0..height).step_by(2) {
                for x in (0..width).step_by(2) {
                    let actual = gather::<u8, L, W>(&picture, plane, (x, y), horizontal);
                    let view = picture.plane(plane).view();
                    let expected: [[u8; W]; L] = std::array::from_fn(|line| {
                        std::array::from_fn(|tap| {
                            let (px, py) = position((x, y), horizontal, line, tap, W);
                            view.row(py.clamp(0, height as isize - 1) as usize)
                                [px.clamp(0, width as isize - 1) as usize]
                        })
                    });
                    assert_eq!(actual, expected);
                    let mut copied = picture.clone();
                    let changed = actual.map(|line| line.map(|value| value.wrapping_add(53)));
                    store(&mut copied, plane, (x, y), horizontal, &changed);
                    let view = copied.plane(plane).view();
                    for py in 0..height {
                        for px in 0..width {
                            let across = if horizontal {
                                py as isize - y as isize
                            } else {
                                px as isize - x as isize
                            };
                            let along = if horizontal {
                                px as isize - x as isize
                            } else {
                                py as isize - y as isize
                            };
                            let inside = (-(W as isize / 2)..W as isize / 2).contains(&across)
                                && (0..L as isize).contains(&along);
                            let original = picture.plane(plane).view().row(py)[px];
                            assert_eq!(
                                view.row(py)[px],
                                if inside {
                                    original.wrapping_add(53)
                                } else {
                                    original
                                }
                            );
                        }
                    }
                }
            }
        }
    }
}
