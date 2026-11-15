//! H.266 general dual-tree intra edge control; all filtering uses shared DSP.
use super::{PictureParameters, Sequence, SliceHeader, Transform, Unit, transform};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

struct EdgeBlock<'a> {
    unit: &'a Unit,
    transform: &'a Transform,
}

impl EdgeBlock<'_> {
    fn bounds(&self, channel: usize) -> (usize, usize, usize, usize) {
        if let Some(selection) = self.unit.sbt
            && std::ptr::eq(self.transform, self.unit.transforms.as_ptr())
        {
            return selection.uncoded(self.unit.node);
        }
        let block = if channel == 0 {
            self.transform.block
        } else {
            self.transform.chroma_block.unwrap()
        };
        (block.x, block.y, block.width(), block.height())
    }
}

fn edge_blocks(units: &[Unit]) -> Vec<EdgeBlock<'_>> {
    units
        .iter()
        .flat_map(|unit| {
            unit.transforms
                .iter()
                .map(move |transform| EdgeBlock { unit, transform })
        })
        .collect()
}

/// Vector difference (1/16 sample units) from which an edge is filtered.
const HALF_SAMPLE: i32 = 8;

/// Samples along an edge segment whose ends set the LADF luma level.
const LADF_SEGMENT: usize = 4;

pub(crate) fn apply(
    picture: &mut Picture,
    units: &[Unit],
    sequence: &Sequence,
    parameters: &PictureParameters,
    headers: &[SliceHeader],
    dsp: &Dsp,
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
        for plane in PlaneIndex::ALL {
            if !sequence.chroma && plane != PlaneIndex::Y {
                continue;
            }
            let channel = usize::from(plane != PlaneIndex::Y);
            for y in (0..picture.height()).step_by(4) {
                for x in (0..picture.width()).step_by(4) {
                    let perpendicular = if horizontal { y } else { x };
                    // No edge on a virtual boundary is filtered (§8.8.3.2).
                    if perpendicular == 0
                        || (channel == 1 && !perpendicular.is_multiple_of(16))
                        || virtual_boundaries[usize::from(horizontal)]
                            .contains(&(perpendicular as u32))
                    {
                        continue;
                    }
                    let at = (y >> 2) * stride + (x >> 2);
                    let p = &blocks[grid[channel][at - if horizontal { stride } else { 1 }]];
                    let q = &blocks[grid[channel][at]];
                    let transform_edge = !std::ptr::eq(p, q);
                    let cu_origin = if horizontal {
                        q.unit.node.y
                    } else {
                        q.unit.node.x
                    };
                    let affine = q.unit.affine_models.iter().any(Option::is_some)
                        || !q.unit.subblocks.is_empty();
                    if !(transform_edge
                        || (channel == 0
                            && affine
                            && (perpendicular - cu_origin).is_multiple_of(8)))
                    {
                        continue;
                    }
                    // The edges of a CU in a slice with deblocking disabled
                    // are not filtered, nor are tile, slice and subpicture
                    // edges closed to the loop filters (§8.8.3.2).
                    let header = &headers[usize::from(q.unit.slice)];
                    if header.deblock.disabled
                        || !open_edge(p.unit, q.unit, (sequence, parameters), headers)
                    {
                        continue;
                    }
                    let sides = (&headers[usize::from(p.unit.slice)], header);
                    let sizes = [p, q].map(|u| {
                        let (_, _, width, height) = u.bounds(channel);
                        if horizontal { height } else { width }
                    });
                    let strength = strength(p, q, plane as usize, (x, y), horizontal, sides);
                    if strength == 0
                        || (channel == 1 && strength == 1 && sizes.iter().any(|&size| size < 16))
                    {
                        continue;
                    }
                    filter_edge(
                        picture,
                        (p, q),
                        plane,
                        (x, y),
                        horizontal,
                        strength,
                        sequence,
                        sides,
                        dsp,
                    );
                }
            }
        }
    }
}

/// Filter one four-sample edge segment at (`x`, `y`) (luma samples) between
/// blocks `p` and `q` of boundary strength `strength` (§8.8.3.6).
#[allow(
    clippy::too_many_arguments,
    reason = "the edge's position, sides and slice"
)]
fn filter_edge(
    picture: &mut Picture,
    (p, q): (&EdgeBlock<'_>, &EdgeBlock<'_>),
    plane: PlaneIndex,
    (x, y): (usize, usize),
    horizontal: bool,
    strength: i32,
    sequence: &Sequence,
    (p_header, header): (&SliceHeader, &SliceHeader),
    dsp: &Dsp,
) {
    let channel = usize::from(plane != PlaneIndex::Y);
    let perpendicular = if horizontal { y } else { x };
    let sizes = [p, q].map(|u| {
        let (_, _, width, height) = u.bounds(channel);
        if horizontal { height } else { width }
    });
    let boundary = horizontal && y.is_multiple_of(1 << sequence.ctu_log2);
    let qp = (i32::from(transform::qp(
        p.unit.quant(),
        p.transform.joint,
        plane as usize,
        sequence,
        p_header,
    )) + i32::from(transform::qp(
        q.unit.quant(),
        q.transform.joint,
        plane as usize,
        sequence,
        header,
    )) + 1)
        >> 1;
    let qp = if channel == 0 {
        qp + ladf_offset(picture, &sequence.tools.ladf, (x, y), horizontal)
    } else {
        qp
    };
    let [beta_offset, tc_offset] = header.deblock.offsets[plane as usize].map(i32::from);
    let beta = crate::filter::BETA[(qp + 2 * beta_offset).clamp(0, 63) as usize];
    let tc = (crate::filter::TC[(qp + 2 * (strength - 1) + 2 * tc_offset).clamp(0, 65) as usize]
        + 2)
        >> 2;
    if channel == 0 {
        let lengths = luma_lengths(p, q, perpendicular, horizontal, boundary);
        filter_luma(picture, dsp, (x, y), horizontal, beta, tc, lengths);
    } else {
        filter_chroma(
            picture,
            dsp,
            plane,
            (x / 2, y / 2),
            horizontal,
            beta,
            tc,
            sizes.iter().all(|&size| size >= 16),
            boundary,
        );
    }
}

/// Whether the loop filters may cross the edge between CUs `p` and `q`:
/// `pps_loop_filter_across_tiles_enabled_flag`,
/// `pps_loop_filter_across_slices_enabled_flag` and, between subpictures,
/// `sps_loop_filter_across_subpic_enabled_flag` of both (§8.8.3.2).
fn open_edge(
    p: &Unit,
    q: &Unit,
    (sequence, parameters): (&Sequence, &PictureParameters),
    headers: &[SliceHeader],
) -> bool {
    // Inside one slice and tile (most edges) nothing is closed.
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

/// Luma-adaptive deblocking QP offset of a four-sample edge segment (H.266
/// §8.8.3.6.2): the interval holding the mean of the edge's corner samples.
fn ladf_offset(
    picture: &Picture,
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
            u16::from(luma.row(p.1)[p.0]) + u16::from(luma.row(q.1)[q.0])
        })
        .sum::<u16>()
        >> 2;
    upper
        .iter()
        .take_while(|&&(lower, _)| level > lower)
        .last()
        .map_or(i32::from(lowest), |&(_, offset)| i32::from(offset))
}

/// Interior segments use the shared in-place SIMD kernel. Narrow ISP
/// boundaries retain clamped access for taps outside the visible picture.
fn filter_luma(
    picture: &mut Picture,
    dsp: &Dsp,
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
        let mut edge = gather::<4, 16>(picture, PlaneIndex::Y, at, horizontal);
        dsp.vvc_deblock_luma(&mut edge, beta, tc, lengths);
        store(picture, PlaneIndex::Y, at, horizontal, &edge);
    }
}

/// Chroma uses the same partial-line plane access as HEVC; two lines
/// correspond to one four-line luma segment in 4:2:0.
fn filter_chroma(
    picture: &mut Picture,
    dsp: &Dsp,
    plane: PlaneIndex,
    at: (usize, usize),
    horizontal: bool,
    beta: i32,
    tc: i32,
    large: bool,
    boundary: bool,
) {
    const REACH: usize = 4;
    const LINES: usize = 2;
    let (x, y) = at;
    let mut view = picture.plane_mut(plane).view_mut();
    let inside = if horizontal {
        y >= REACH && y + REACH <= view.height() && x + LINES <= view.width()
    } else {
        x >= REACH && x + REACH <= view.width() && y + LINES <= view.height()
    };
    if inside {
        dsp.vvc_deblock_chroma_edge(&mut view, at, !horizontal, LINES, beta, tc, large, boundary);
    } else {
        let mut edge = gather::<LINES, { 2 * REACH }>(picture, plane, at, horizontal);
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
fn gather<const L: usize, const W: usize>(
    picture: &Picture,
    plane: PlaneIndex,
    origin: (usize, usize),
    horizontal: bool,
) -> [[u8; W]; L] {
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
    // Narrow ISP edges can approach the picture border more closely than
    // the common DSP buffer's far taps, which their kernel never reads.
    std::array::from_fn(|line| {
        std::array::from_fn(|tap| {
            let (x, y) = position(origin, horizontal, line, tap, W);
            view.row(y.clamp(0, view.height() as isize - 1) as usize)
                [x.clamp(0, view.width() as isize - 1) as usize]
        })
    })
}
fn store<const L: usize, const W: usize>(
    picture: &mut Picture,
    plane: PlaneIndex,
    origin: (usize, usize),
    horizontal: bool,
    edge: &[[u8; W]; L],
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
    if [p, q]
        .iter()
        .any(|u| matches!(u.unit.inter, Some(super::Inter::Ciip(_))))
    {
        return 2;
    }
    if !std::ptr::eq(p, q)
        && [p, q].iter().any(|unit| {
            !unit.transform.levels[plane].is_empty() || (plane != 0 && unit.transform.joint != 0)
        })
    {
        return 1;
    }
    // H.266 §8.8.3.5: a transform-only boundary can gain strength from
    // residuals, but motion is compared only at CU or coding-subblock edges.
    // In particular, geometric motion changes inside a CU do not create one.
    if std::ptr::eq(p.unit, q.unit)
        && q.unit.affine_models.iter().all(Option::is_none)
        && q.unit.subblocks.is_empty()
    {
        return 0;
    }
    // IBC CUs compare block vectors; an IBC CU beside a motion-compensated
    // one is a prediction-mode change (§8.8.3.5).
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
    }
    if plane == 0
        && let (Some(p), Some(q)) = (
            super::motion::at(
                p.unit,
                x - usize::from(!horizontal),
                y - usize::from(horizontal),
            ),
            super::motion::at(q.unit, x, y),
        )
    {
        return motion_strength(p, q, headers);
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
            for y in (n.y..n.y + n.height()).step_by(4) {
                for x in (n.x..n.x + n.width()).step_by(4) {
                    grid[channel][(y >> 2) * stride + (x >> 2)] = index;
                }
            }
        }
    }
    grid
}

/// Whether motion `p` and `q` (of CUs in slices with headers `headers`)
/// differ enough to filter their edge. References compare by POC delta,
/// which names the same picture in any slice of the picture.
fn motion_strength(
    p: super::motion::Motion,
    q: super::motion::Motion,
    headers: (&SliceHeader, &SliceHeader),
) -> i32 {
    fn describe(
        motion: super::motion::Motion,
        header: &SliceHeader,
    ) -> [Option<(&crate::decoder_headers::ReferenceEntry, crate::motion::Mv16)>; 2] {
        std::array::from_fn(|list| {
            motion.lists[list].map(|v| {
                let reference = &header.references[list].entries[usize::from(v.reference)];
                (reference, v.vector)
            })
        })
    }
    let (p, q) = (describe(p, headers.0), describe(q, headers.1));
    let same =
        |p: Option<(&crate::decoder_headers::ReferenceEntry, crate::motion::Mv16)>,
         q: Option<(&crate::decoder_headers::ReferenceEntry, crate::motion::Mv16)>| match (
            p, q,
        ) {
            (Some((pr, p)), Some((qr, q))) => {
                pr == qr && (p.x - q.x).abs() < 8 && (p.y - q.y).abs() < 8
            }
            (None, None) => true,
            _ => false,
        };
    i32::from(!((same(p[0], q[0]) && same(p[1], q[1])) || (same(p[0], q[1]) && same(p[1], q[0]))))
}

fn luma_lengths(
    p: &EdgeBlock<'_>,
    q: &EdgeBlock<'_>,
    position: usize,
    horizontal: bool,
    boundary: bool,
) -> [usize; 2] {
    let size = |u: &EdgeBlock<'_>| {
        let (_, _, width, height) = u.bounds(0);
        if horizontal { height } else { width }
    };
    if std::ptr::eq(p, q) {
        let (x, y, _, _) = q.bounds(0);
        let origin = if horizontal { y } else { x };
        let distance = (position - origin).min(origin + size(q) - position);
        return [if distance <= 4 {
            1
        } else if distance <= 8 {
            2
        } else {
            3
        }; 2];
    }
    if size(p) <= 4 || size(q) <= 4 {
        return [1, 1];
    }
    [p, q].map(|unit| {
        if size(unit) < 32 || (boundary && std::ptr::eq(unit, p)) {
            3
        } else if unit.unit.affine_models.iter().any(Option::is_some)
            || !unit.unit.subblocks.is_empty()
        {
            5
        } else {
            7
        }
    })
}

#[cfg(test)]
mod sample_tests {
    use super::*;

    // Check both the row-copy path and the clamped narrow-ISP fallback
    // against direct coordinate access, including partial final segments.
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
                    let actual = gather::<L, W>(&picture, plane, (x, y), horizontal);
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
