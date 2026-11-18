//! H.266 general dual-tree intra edge control; all filtering uses shared DSP.
use super::{PictureParameters, Sequence, SliceHeader, Transform, Unit, transform};
use nc_common::{Picture, PlaneIndex, Sample};
use nc_dsp::{DepthKernels, Pixel, vvc_deblock_beta, vvc_deblock_tc};

struct EdgeBlock<'a> {
    unit: &'a Unit,
    transform: &'a Transform,
    luma_bounds: (usize, usize, usize, usize),
    chroma_bounds: (usize, usize, usize, usize),
    has_subblocks: bool,
    uniform_motion: bool,
}

impl EdgeBlock<'_> {
    #[inline]
    fn bounds(&self, channel: usize) -> (usize, usize, usize, usize) {
        if channel == 0 {
            self.luma_bounds
        } else {
            self.chroma_bounds
        }
    }

    #[inline]
    fn size_perp(&self, channel: usize, horizontal: bool) -> usize {
        let (_, _, w, h) = self.bounds(channel);
        if horizontal { h } else { w }
    }
}

fn edge_blocks(units: &[Unit]) -> Vec<EdgeBlock<'_>> {
    let total: usize = units.iter().map(|u| u.transforms.len()).sum();
    let mut out = Vec::with_capacity(total);
    for unit in units {
        let has_subblocks =
            unit.affine_models.iter().any(Option::is_some) || !unit.subblocks.is_empty();
        let uniform_motion = !has_subblocks && unit.geometric.is_none();
        for transform in &unit.transforms {
            let (luma_bounds, chroma_bounds) = if let Some(selection) = unit.sbt
                && std::ptr::eq(transform, unit.transforms.as_ptr())
            {
                let b = selection.uncoded(unit.node);
                (b, b)
            } else {
                let lb = transform.block;
                let cb = transform
                    .chroma_block
                    .map_or((0, 0, 0, 0), |c| (c.x, c.y, c.width(), c.height()));
                ((lb.x, lb.y, lb.width(), lb.height()), cb)
            };
            out.push(EdgeBlock {
                unit,
                transform,
                luma_bounds,
                chroma_bounds,
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

/// Samples along an edge segment whose ends set the LADF luma level.
const LADF_SEGMENT: usize = 4;

pub(crate) fn apply<T: Pixel>(
    picture: &mut Picture<T>,
    units: &[Unit],
    sequence: &Sequence,
    parameters: &PictureParameters,
    headers: &[SliceHeader],
    dsp: DepthKernels<'_>,
) {
    if headers.iter().all(|h| h.deblock.disabled) {
        return;
    }
    let stride = picture.width().div_ceil(4);
    // Every slice of a picture has its picture header.
    let virtual_boundaries = &headers[0].picture.boundaries;
    let blocks = edge_blocks(units);
    let grid = edge_grid(&blocks, stride, picture.height());
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

#[allow(clippy::too_many_arguments)]
fn apply_luma<T: Pixel>(
    picture: &mut Picture<T>,
    blocks: &[EdgeBlock<'_>],
    grid: &[usize],
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
    let no_ladf = sequence.tools.ladf.is_empty();
    for perp in (4..perp_limit).step_by(4) {
        if closed_positions.contains(&(perp as u32)) {
            continue;
        }
        let interior_perp = perp >= 8 && perp + 8 <= perp_limit;
        let mut along = 0;
        while along < along_limit {
            let (x, y) = if horizontal {
                (along, perp)
            } else {
                (perp, along)
            };
            let at = (y >> 2) * stride + (x >> 2);
            let (p_idx, q_idx) = (grid[at - p_off], grid[at]);
            let q = &blocks[q_idx];
            if p_idx == q_idx {
                let cu_origin = if horizontal {
                    q.unit.node.y
                } else {
                    q.unit.node.x
                };
                if !q.has_subblocks || !(perp - cu_origin).is_multiple_of(SUBBLOCK_EDGE_GRID) {
                    let mut skip = 1usize;
                    while along + 4 * skip < along_limit
                        && grid[at + skip * along_step] == q_idx
                        && grid[at + skip * along_step - p_off] == p_idx
                    {
                        skip += 1;
                    }
                    along += 4 * skip;
                    continue;
                }
            }
            let p = &blocks[p_idx];
            let header = &headers[usize::from(q.unit.slice)];
            if header.deblock.disabled
                || !open_edge(p.unit, q.unit, (sequence, parameters), headers)
            {
                along += 4;
                continue;
            }
            let sides = (&headers[usize::from(p.unit.slice)], header);
            let bs = strength(p, q, 0, (x, y), horizontal, sides);
            if bs == 0 {
                along += 4;
                continue;
            }
            let can_batch8 = interior_perp
                && no_ladf
                && p_idx != q_idx
                && along + 8 <= along_limit
                && grid[at + along_step - p_off] == p_idx
                && grid[at + along_step] == q_idx
                && (bs == 2
                    || !p.transform.levels[0].is_empty()
                    || !q.transform.levels[0].is_empty()
                    || (p.uniform_motion && q.uniform_motion));
            filter_luma_edge(
                picture,
                (p, q),
                (x, y),
                horizontal,
                bs,
                can_batch8,
                sequence,
                sides,
                dsp,
            );
            along += if can_batch8 { 8 } else { 4 };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_chroma<T: Pixel>(
    picture: &mut Picture<T>,
    blocks: &[EdgeBlock<'_>],
    grid: &[usize],
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
    for perp in (16..perp_limit).step_by(16) {
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
            let (p_idx, q_idx) = (grid[at - p_off], grid[at]);
            let mut run = 1usize;
            while run < 4
                && along + 4 * run < along_limit
                && grid[at + run * along_step - p_off] == p_idx
                && grid[at + run * along_step] == q_idx
            {
                run += 1;
            }
            if p_idx == q_idx {
                along += 4 * run;
                continue;
            }
            let (p, q) = (&blocks[p_idx], &blocks[q_idx]);
            let header = &headers[usize::from(q.unit.slice)];
            if header.deblock.disabled
                || !open_edge(p.unit, q.unit, (sequence, parameters), headers)
            {
                along += 4 * run;
                continue;
            }
            let sides = (&headers[usize::from(p.unit.slice)], header);
            let large = p.size_perp(1, horizontal) >= 16 && q.size_perp(1, horizontal) >= 16;
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
                filter_chroma_run(
                    picture,
                    dsp,
                    plane,
                    (x / 2, y / 2),
                    run,
                    horizontal,
                    (beta, tc),
                    (large, boundary),
                );
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
    sequence: &Sequence,
    sides: (&SliceHeader, &SliceHeader),
    dsp: DepthKernels<'_>,
) {
    let perpendicular = if horizontal { y } else { x };
    let boundary = horizontal && y.is_multiple_of(1 << sequence.ctu_log2);
    let depth = nc_common::BitDepth::of::<T>(dsp.depth());
    let bd_offset = crate::decoder_headers::qp_bd_offset(depth);
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
fn filter_chroma_run<T: Pixel>(
    picture: &mut Picture<T>,
    dsp: DepthKernels<'_>,
    plane: PlaneIndex,
    (cx, cy): (usize, usize),
    run: usize,
    horizontal: bool,
    (beta, tc): (i32, i32),
    (large, boundary): (bool, bool),
) {
    const REACH: usize = 4;
    let lines = 2 * run;
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
        dsp.vvc_deblock_chroma_edge(
            &mut view,
            (cx, cy),
            !horizontal,
            lines,
            beta,
            tc,
            large,
            boundary,
        );
        return;
    }
    for k in 0..run {
        let at = if horizontal {
            (cx + 2 * k, cy)
        } else {
            (cx, cy + 2 * k)
        };
        let mut edge = gather::<T, 2, { 2 * REACH }>(picture, plane, at, horizontal);
        dsp.vvc_deblock_chroma(&mut edge, beta, tc, large, boundary);
        store(picture, plane, at, horizontal, &edge);
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
    if p.unit.inter.is_none() || q.unit.inter.is_none() {
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
        !p.transform.levels[plane].is_empty()
            || !q.transform.levels[plane].is_empty()
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
    if p.unit.inter.is_none() || q.unit.inter.is_none() {
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
        && (!p.transform.levels[plane].is_empty()
            || !q.transform.levels[plane].is_empty()
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
        match (p.unit.block_vector, q.unit.block_vector) {
            (Some(p), Some(q)) => {
                return i32::from(
                    (p.x - q.x).abs() >= HALF_SAMPLE || (p.y - q.y).abs() >= HALF_SAMPLE,
                );
            }
            (None, None) => {}
            _ => return 1,
        }
        if let (Some(p), Some(q)) = (
            super::motion::at(
                p.unit,
                x - usize::from(!horizontal),
                y - usize::from(horizontal),
            ),
            super::motion::at(q.unit, x, y),
        ) {
            return motion_strength(p, q, headers);
        }
    }
    0
}

fn edge_grid(blocks: &[EdgeBlock<'_>], stride: usize, height: usize) -> [Vec<usize>; 2] {
    let mut grid = [
        vec![0; stride * height.div_ceil(4)],
        vec![0; stride * height.div_ceil(4)],
    ];
    for (index, block) in blocks.iter().enumerate() {
        let unit = block.unit;
        for channel in if unit.combined {
            0..2
        } else {
            unit.channel..unit.channel + 1
        } {
            let n = if channel == 0 {
                block.transform.block
            } else {
                let Some(n) = block.transform.chroma_block else {
                    continue;
                };
                n
            };
            let cols = n.width().div_ceil(4);
            for y in (n.y..n.y + n.height()).step_by(4) {
                let base = (y >> 2) * stride + (n.x >> 2);
                grid[channel][base..base + cols].fill(index);
            }
        }
    }
    grid
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
