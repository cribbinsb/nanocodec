//! Bit-exactness checks of the loop-filter kernels (AV1 CDEF, HEVC SAO edge
//! offset, H.264, HEVC and AV2 deblocking) against scalar.
//!
//! Deblocking cases use smooth content with optional steps across the edge
//! (HEVC also nearly flat content, for the strong filter).
//! Direction-search cases use random, extreme, flat and striped blocks.
//! CDEF cases filter every block of a small plane (so frame edges, where the
//! padding is unavailable, are exercised on every side) with random
//! strengths, direction and damping, and compare the whole output plane. SAO
//! cases use random rectangles (any width, so the scalar row tails run too)
//! and random classes and offsets. Block lists (`cdef_filter_blocks`) mix
//! interior and edge blocks with per-block directions and a few primary
//! strengths, so ISAs that filter two 4x4 blocks at once pair blocks of
//! different directions, and leave some unpaired.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{
    AV2_EDGE_LINES, Av2EdgeFilter, CDEF_MAX_SETS, CDEF_PADDING, CdefBlock, CdefLines, CdefSets,
    CdefSource, CdefStrengths, CdefUnit, Dsp, H264EdgeParams, H264MbEdges, SAO_EO_NEIGHBOURS,
    cdef_adjust_strength,
};

use crate::check::CheckReport;
use crate::check_codec::record;
use crate::rng::Rng;

/// Planes of 24x16 samples: three 8x8 luma blocks across, two down.
const PLANE_W: usize = 24;
const PLANE_H: usize = 16;

pub(crate) fn check_filters(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    check_sao_apply_boundaries(candidate, reference, report);
    check_sao_stats_boundaries(candidate, reference, report);
    check_hevc_chroma_footprint(candidate, reference, report);
    check_hevc_strong_boundaries(candidate, reference, report);
    for _ in 0..iterations * 8 {
        check_cdef(candidate, reference, rng, report);
        check_cdef_blocks(candidate, reference, rng, report);
        check_cdef_direction(candidate, reference, rng, report);
        check_cdef_band(candidate, rng, report);
        check_sao(candidate, reference, rng, report);
        check_h264_deblock(candidate, reference, rng, report);
        check_h264_deblock_mb(candidate, reference, rng, report);
        check_hevc_deblock(candidate, reference, rng, report);
        check_av2_deblock(candidate, reference, rng, report);
        check_av2_deblock_line(candidate, reference, rng, report);
    }
}

fn check_cdef(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut samples = vec![0u8; PLANE_W * PLANE_H];
    match rng.below(4) {
        0 => rng.fill(&mut samples),
        // Small noise around a level: exercises the constraint's clipping.
        1 => {
            let level = rng.byte();
            for s in &mut samples {
                *s = level.saturating_add(rng.byte() % 8).saturating_sub(4);
            }
        }
        2 => samples.fill(if rng.below(2) == 0 { 0 } else { 255 }),
        _ => {
            for (i, s) in samples.iter_mut().enumerate() {
                *s = if (i / PLANE_W + i % PLANE_W).is_multiple_of(2) {
                    0
                } else {
                    255
                };
            }
        }
    }
    let view = PlaneView::new(&samples, PLANE_W, PLANE_W, PLANE_H).expect("plane");
    let mut source = CdefSource::new();
    source.fill_from(&view);
    let size = if rng.below(2) == 0 { 8 } else { 4 };
    let pri = rng.below(16) as u8;
    let sec = [0u8, 1, 2, 4][rng.below(4)];
    let dir = rng.below(8) as u8;
    let damping = 2 + rng.below(5) as u8;
    let mut outputs = [samples.clone(), samples.clone()];
    for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
        let mut dst = PlaneViewMut::new(out, PLANE_W, PLANE_W, PLANE_H).expect("plane");
        for y in (0..PLANE_H).step_by(size) {
            for x in (0..PLANE_W).step_by(size) {
                dsp.cdef_filter(size, &source, (x, y), (pri, sec), dir, damping, &mut dst);
            }
        }
    }
    record(report, candidate.isa(), outputs[0] == outputs[1], || {
        format!("cdef {size}x{size} pri {pri} sec {sec} dir {dir} damping {damping}")
    });
    // The 8-bit interior kernels, the reference's included, must match the
    // padded 16-bit reference on blocks away from the plane's edges. Every
    // block uses the same filter, so a sample's output does not depend on
    // the block grid: the interior blocks start CDEF_PADDING from the edge.
    let starts = |extent: usize| (CDEF_PADDING..=extent - size - CDEF_PADDING).step_by(size);
    let covered = |extent: usize| CDEF_PADDING..CDEF_PADDING + starts(extent).count() * size;
    for dsp in [candidate, reference] {
        let mut interior = samples.clone();
        let mut dst = PlaneViewMut::new(&mut interior, PLANE_W, PLANE_W, PLANE_H).expect("plane");
        for y in starts(PLANE_H) {
            for x in starts(PLANE_W) {
                dsp.cdef_filter_interior(size, &view, (x, y), (pri, sec), dir, damping, &mut dst);
            }
        }
        let inside = |i: usize| {
            covered(PLANE_W).contains(&(i % PLANE_W)) && covered(PLANE_H).contains(&(i / PLANE_W))
        };
        let matches = interior
            .iter()
            .enumerate()
            .all(|(i, &v)| v == if inside(i) { outputs[1][i] } else { samples[i] });
        record(report, dsp.isa(), matches, || {
            format!("cdef interior {size}x{size} pri {pri} sec {sec} dir {dir} damping {damping}")
        });
    }
}

fn check_cdef_blocks(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut samples = vec![0u8; PLANE_W * PLANE_H];
    rng.fill(&mut samples);
    let view = PlaneView::new(&samples, PLANE_W, PLANE_W, PLANE_H).expect("plane");
    let mut source = CdefSource::new();
    source.fill_from(&view);
    let size = if rng.below(2) == 0 { 8 } else { 4 };
    // Two primaries (one may be 0) so that neighbouring blocks sometimes
    // share a strength and sometimes not.
    let primaries = [rng.below(16) as u8, rng.below(16) as u8];
    let sec = [0u8, 1, 2, 4][rng.below(4)];
    let damping = 2 + rng.below(5) as u8;
    let mut blocks = Vec::new();
    for y in (0..PLANE_H).step_by(size) {
        for x in (0..PLANE_W).step_by(size) {
            if rng.below(4) != 0 {
                blocks.push(CdefBlock {
                    x,
                    y,
                    pri: primaries[rng.below(2)],
                    dir: rng.below(8) as u8,
                });
            }
        }
    }
    let mut outputs = [samples.clone(), samples.clone()];
    for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
        let mut dst = PlaneViewMut::new(out, PLANE_W, PLANE_W, PLANE_H).expect("plane");
        dsp.cdef_filter_blocks(size, &source, &blocks, sec, damping, &mut dst);
    }
    record(report, candidate.isa(), outputs[0] == outputs[1], || {
        format!(
            "cdef blocks {size}x{size} primaries {primaries:?} sec {sec} damping {damping} blocks {blocks:?}"
        )
    });
}

/// Band-driver pictures: 17 x 10 luma units, two bands of 64 rows (the
/// second partial), so units touch every edge and both bands' line buffers
/// slide.
const BAND_W: usize = 136;
const BAND_H: usize = 80;
const BAND_ROWS: usize = 64;

/// `Dsp::cdef_filter_band` over a picture, band by band, against an
/// independent per-block filter: directions from the unfiltered luma, each
/// block filtered with `cdef_filter` from a copy of its whole plane.
fn check_cdef_band(candidate: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut picture = Picture::new(BAND_W, BAND_H, 0).expect("picture");
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            // Smooth rows with noise, so directions and contrasts vary.
            let base = rng.byte();
            for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                *s = base.wrapping_add((x * 3) as u8) ^ (rng.byte() & 0x1f);
            }
        }
    }
    let count = 1 + rng.below(CDEF_MAX_SETS);
    let secondary = |rng: &mut Rng| [0u8, 1, 2, 4][rng.below(4)];
    let strengths: Vec<CdefStrengths> = (0..count)
        .map(|_| CdefStrengths {
            luma_primary: rng.below(16) as u8,
            luma_secondary: secondary(rng),
            chroma_primary: rng.below(16) as u8,
            chroma_secondary: secondary(rng),
        })
        .collect();
    let damping = 3 + rng.below(4) as u8;
    let units: Vec<CdefUnit> = (0..BAND_H / 8)
        .flat_map(|row| (0..BAND_W / 8).map(move |column| (column * 8, row * 8)))
        .filter_map(|(x, y)| {
            let set = rng.below(count) as u8;
            (rng.below(4) != 0).then_some(CdefUnit { x, y, set })
        })
        .collect();

    let mut expected = picture.clone();
    let sources = PlaneIndex::ALL.map(|plane| {
        let mut source = CdefSource::new();
        source.fill_from(&picture.plane(plane).view());
        source
    });
    for unit in &units {
        let set = strengths[usize::from(unit.set)];
        let (dir, var) = candidate.cdef_find_dir(&sources[0].view(), (unit.x, unit.y));
        let luma_primary = cdef_adjust_strength(set.luma_primary, var);
        let planes = [
            (
                PlaneIndex::Y,
                8,
                luma_primary,
                set.luma_primary,
                set.luma_secondary,
            ),
            (
                PlaneIndex::U,
                4,
                set.chroma_primary,
                set.chroma_primary,
                set.chroma_secondary,
            ),
            (
                PlaneIndex::V,
                4,
                set.chroma_primary,
                set.chroma_primary,
                set.chroma_secondary,
            ),
        ];
        for (plane, size, primary, coded_primary, secondary) in planes {
            if primary | secondary == 0 {
                continue;
            }
            let shift = usize::from(size == 4);
            let chroma = u8::from(size == 4);
            candidate.cdef_filter(
                size,
                &sources[plane as usize],
                (unit.x >> shift, unit.y >> shift),
                (primary, secondary),
                if coded_primary == 0 { 0 } else { dir },
                damping - chroma,
                &mut expected.plane_mut(plane).view_mut(),
            );
        }
    }

    let sets = CdefSets::new(&strengths, damping);
    let mut lines = CdefLines::new();
    lines.start(&picture);
    for start in (0..BAND_H).step_by(BAND_ROWS) {
        let rows = start..(start + BAND_ROWS).min(BAND_H);
        let band: Vec<CdefUnit> = units
            .iter()
            .filter(|u| rows.contains(&u.y))
            .copied()
            .collect();
        candidate.cdef_filter_band(&mut lines, &mut picture, rows, &band, &sets);
    }
    let same = PlaneIndex::ALL.iter().all(|&plane| {
        let (a, b) = (picture.plane(plane).view(), expected.plane(plane).view());
        (0..a.height()).all(|y| a.row(y) == b.row(y))
    });
    record(report, candidate.isa(), same, || {
        format!("cdef band: sets {strengths:?} damping {damping} units {units:?}")
    });
}

/// Direction-search planes: room for an 8x8 block at any offset 0..8.
const DIR_SIDE: usize = 16;

fn check_cdef_direction(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut samples = vec![0u8; DIR_SIDE * DIR_SIDE];
    match rng.below(4) {
        0 => rng.fill(&mut samples),
        // Extremes: the largest partial sums and costs.
        1 => {
            for s in &mut samples {
                *s = if rng.below(2) == 0 { 0 } else { 255 };
            }
        }
        2 => samples.fill(rng.byte()),
        // Lines at a random slope: a clear dominant direction.
        _ => {
            let (a, b) = (rng.below(5), rng.below(5));
            let (low, high) = (rng.byte(), rng.byte());
            for (i, s) in samples.iter_mut().enumerate() {
                let (r, c) = (i / DIR_SIDE, i % DIR_SIDE);
                *s = if (a * r + b * c) % 4 < 2 { low } else { high };
            }
        }
    }
    let view = PlaneView::new(&samples, DIR_SIDE, DIR_SIDE, DIR_SIDE).expect("plane");
    let at = (rng.below(DIR_SIDE - 7), rng.below(DIR_SIDE - 7));
    let found = [candidate, reference].map(|d| d.cdef_find_dir(&view, at));
    record(report, candidate.isa(), found[0] == found[1], || {
        format!(
            "cdef direction at {at:?}: {:?}, reference {:?}",
            found[0], found[1]
        )
    });
}

/// SAO planes: wide enough for several vector steps per row, and tall
/// enough that the 16-bit statistics flush on every ISA.
const SAO_W: usize = 70;
const SAO_H: usize = 72;

fn check_sao(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut rec = vec![0u8; SAO_W * SAO_H];
    let mut src = vec![0u8; SAO_W * SAO_H];
    if rng.below(2) == 0 {
        rng.fill(&mut rec);
    } else {
        // Few levels: many equal neighbours, so every category occurs.
        for r in &mut rec {
            *r = 120 + (rng.byte() % 3) * 4;
        }
    }
    if rng.below(4) == 0 {
        // The largest differences, all of one sign, on a checkerboard of
        // local extremes: the 16-bit sums' limit.
        src.fill(255);
        for (i, r) in rec.iter_mut().enumerate() {
            *r = ((i / SAO_W + i % SAO_W) % 2) as u8;
        }
    } else {
        rng.fill(&mut src);
    }
    let class = rng.below(4) as u8;
    let [(ax, ay), (bx, by)] = SAO_EO_NEIGHBOURS[usize::from(class)];
    let (left, right) = (usize::from(ax < 0 || bx < 0), usize::from(ax > 0 || bx > 0));
    let (top, bottom) = (usize::from(ay < 0 || by < 0), usize::from(ay > 0 || by > 0));
    let x0 = left + rng.below(8);
    let x1 = (x0 + rng.below(SAO_W)).min(SAO_W - right);
    let y0 = top + rng.below(4);
    let y1 = (y0 + 1 + rng.below(SAO_H)).min(SAO_H - bottom);
    let rect = (x0, y0, x1, y1);
    let offsets = [
        (rng.below(8)) as i8,
        (rng.below(8)) as i8,
        -(rng.below(8) as i8),
        -(rng.below(8) as i8),
    ];
    let rec_view = PlaneView::new(&rec, SAO_W, SAO_W, SAO_H).expect("plane");
    let src_view = PlaneView::new(&src, SAO_W, SAO_W, SAO_H).expect("plane");
    let stats = [candidate, reference].map(|d| d.sao_edge_stats(class, &src_view, &rec_view, rect));
    record(report, candidate.isa(), stats[0] == stats[1], || {
        format!("sao stats class {class} rect {rect:?}")
    });
    let mut outputs = [rec.clone(), rec.clone()];
    for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
        let mut dst = PlaneViewMut::new(out, SAO_W, SAO_W, SAO_H).expect("plane");
        dsp.sao_edge_apply(class, offsets, &rec_view, &mut dst, rect);
    }
    record(report, candidate.isa(), outputs[0] == outputs[1], || {
        format!("sao apply class {class} offsets {offsets:?} rect {rect:?}")
    });
}

/// H.264 deblocking planes: room for an edge of 16 lines with 4 samples on
/// each side, in either direction.
const DB_W: usize = 24;
const DB_H: usize = 24;

fn check_h264_deblock(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut plane = vec![0u8; DB_W * DB_H];
    // Smooth content with small steps, so the filter conditions often hold.
    let level = rng.byte();
    let spread = 1 + rng.below(40) as u8;
    for s in &mut plane {
        *s = level.saturating_add(rng.byte() % spread);
    }
    if rng.below(2) == 0 {
        // A step across the middle: exercises the strong filter.
        let step = rng.byte() % 24;
        for (i, s) in plane.iter_mut().enumerate() {
            if i % DB_W >= DB_W / 2 {
                *s = s.saturating_add(step);
            }
        }
    }
    let luma = rng.below(2) == 0;
    let vertical = rng.below(2) == 0;
    let bs: [u8; 4] = core::array::from_fn(|_| rng.below(5) as u8);
    let tc0: [u8; 4] = core::array::from_fn(|_| rng.below(26) as u8);
    let alpha = rng.below(256) as u8;
    let beta = rng.below(19) as u8;
    let pos = if vertical {
        (DB_W / 2, 4)
    } else {
        (4, DB_H / 2)
    };
    let mut outputs = [plane.clone(), plane.clone()];
    for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
        let mut dst = PlaneViewMut::new(out, DB_W, DB_W, DB_H).expect("plane");
        if luma {
            dsp.h264_deblock_luma_edge(&mut dst, pos, vertical, bs, tc0, (alpha, beta));
        } else {
            dsp.h264_deblock_chroma_edge(&mut dst, pos, vertical, bs, tc0, (alpha, beta));
        }
    }
    record(report, candidate.isa(), outputs[0] == outputs[1], || {
        format!(
            "h264 deblock luma {luma} vertical {vertical} bs {bs:?} tc0 {tc0:?} alpha {alpha} beta {beta}"
        )
    });
}

/// H.264 whole-macroblock deblocking planes: a 16x16 luma macroblock (8x8
/// chroma) at (8, 8) with room for its edge-0 reach on the left and top.
const MB_DB_SIDE: usize = 32;
const MB_DB_POS: usize = 8;

/// The candidate's whole-macroblock entries against the reference's
/// per-edge kernels in the standard's order (vertical edges, then
/// horizontal), on random edge masks and parameters.
fn check_h264_deblock_mb(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let level = rng.byte();
    let spread = 1 + rng.below(40) as u8;
    let mut planes: [Vec<u8>; 2] = core::array::from_fn(|_| {
        (0..MB_DB_SIDE * MB_DB_SIDE)
            .map(|_| level.saturating_add(rng.byte() % spread))
            .collect()
    });
    let chroma = rng.below(2) == 0;
    let mut edges = H264MbEdges {
        mask: rng.byte(),
        ..H264MbEdges::default()
    };
    if chroma {
        // 4:2:0 chroma has edges 0 and 2 only.
        edges.mask &= 0b0101_0101;
    }
    for row in &mut edges.edges {
        for edge in row {
            *edge = H264EdgeParams {
                bs: core::array::from_fn(|_| rng.below(5) as u8),
                tc0: core::array::from_fn(|_| rng.below(26) as u8),
                alpha: rng.below(256) as u8,
                beta: rng.below(19) as u8,
            };
        }
    }
    let pos = (MB_DB_POS, MB_DB_POS);
    let mut want = planes.clone();
    {
        let [u, v] = &mut want;
        let mut views = [u, v]
            .map(|p| PlaneViewMut::new(p, MB_DB_SIDE, MB_DB_SIDE, MB_DB_SIDE).expect("plane"));
        let spacing = if chroma { 2 } else { 4 };
        let planes_used = if chroma { 2 } else { 1 };
        for view in views.iter_mut().take(planes_used) {
            for (d, row) in edges.edges.iter().enumerate() {
                for (e, edge) in row.iter().enumerate() {
                    if edges.mask & (1 << (4 * d + e)) == 0 {
                        continue;
                    }
                    let vertical = d == 0;
                    let at = if vertical {
                        (pos.0 + e * spacing, pos.1)
                    } else {
                        (pos.0, pos.1 + e * spacing)
                    };
                    let params = (edge.alpha, edge.beta);
                    if chroma {
                        reference.h264_deblock_chroma_edge(
                            view, at, vertical, edge.bs, edge.tc0, params,
                        );
                    } else {
                        reference
                            .h264_deblock_luma_edge(view, at, vertical, edge.bs, edge.tc0, params);
                    }
                }
            }
        }
    }
    {
        let [u, v] = &mut planes;
        let mut u = PlaneViewMut::new(u, MB_DB_SIDE, MB_DB_SIDE, MB_DB_SIDE).expect("plane");
        let mut v = PlaneViewMut::new(v, MB_DB_SIDE, MB_DB_SIDE, MB_DB_SIDE).expect("plane");
        if chroma {
            candidate.h264_deblock_mb_chroma(&mut u, &mut v, pos, &edges);
        } else {
            candidate.h264_deblock_mb_luma(&mut u, pos, &edges);
        }
    }
    record(report, candidate.isa(), planes == want, || {
        format!(
            "h264 deblock macroblock chroma {chroma} mask {:#010b}",
            edges.mask
        )
    });
}

// Minimal final-row allocations expose short-edge overreads/writes. Compare
// the whole allocation, including untouched taps and row padding.
fn check_hevc_chroma_footprint(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for vertical in [false, true] {
        for lines in [4, 8] {
            for padding in [0, 3] {
                let (width, height) = if vertical { (8, lines) } else { (lines, 8) };
                let stride = width + padding;
                let pos = if vertical { (4, 0) } else { (0, 4) };
                for tc in [[0; 4], [255; 4], [0, 1, 127, 255], [255, 127, 1, 0]] {
                    for pattern in 0..5 {
                        let plane: Vec<u8> = (0..stride * (height - 1) + width)
                            .map(|i| match pattern {
                                0 => (i * 97 + 13) as u8,
                                1 => {
                                    if i % 3 == 0 {
                                        255
                                    } else {
                                        0
                                    }
                                }
                                2 => (120 + i % 7) as u8,
                                3 => 0,
                                _ => 255,
                            })
                            .collect();
                        let mut outputs = [plane.clone(), plane];
                        for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
                            let mut dst =
                                PlaneViewMut::new(out, stride, width, height).expect("plane");
                            dsp.hevc_deblock_chroma_edge(&mut dst, pos, vertical, lines, tc);
                        }
                        record(report, candidate.isa(), outputs[0] == outputs[1], || {
                            format!(
                                "hevc chroma footprint vertical {vertical} lines {lines} padding {padding} tc {tc:?} pattern {pattern}"
                            )
                        });
                    }
                }
            }
        }
    }
}

fn check_hevc_deblock(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut plane = vec![0u8; DB_W * DB_H];
    // Smooth content, often flat (strong filter) or with a step across the
    // edge, so every decision outcome occurs.
    let level = rng.byte();
    let max_spread = if rng.below(2) == 0 { 4 } else { 40 };
    let spread = 1 + rng.below(max_spread) as u8;
    for s in &mut plane {
        *s = level.saturating_add(rng.byte() % spread);
    }
    let vertical = rng.below(2) == 0;
    if rng.below(2) == 0 {
        let step = rng.byte() % 32;
        for (i, s) in plane.iter_mut().enumerate() {
            let (r, c) = (i / DB_W, i % DB_W);
            if (if vertical { c } else { r }) >= DB_W / 2 {
                *s = s.saturating_add(step);
            }
        }
    }
    let luma = rng.below(2) == 0;
    let lines = if luma || rng.below(2) == 0 { 8 } else { 4 };
    let tc: [u8; 4] = core::array::from_fn(|_| {
        if rng.below(4) == 0 {
            0
        } else {
            rng.below(25) as u8
        }
    });
    let beta = rng.below(65) as u8;
    let pos = if vertical {
        (DB_W / 2, 4 + rng.below(8))
    } else {
        (4 + rng.below(8), DB_H / 2)
    };
    let mut outputs = [plane.clone(), plane.clone()];
    for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
        let mut dst = PlaneViewMut::new(out, DB_W, DB_W, DB_H).expect("plane");
        if luma {
            dsp.hevc_deblock_luma_edge(&mut dst, pos, vertical, [tc[0], tc[1]], beta);
        } else {
            dsp.hevc_deblock_chroma_edge(&mut dst, pos, vertical, lines, tc);
        }
    }
    record(report, candidate.isa(), outputs[0] == outputs[1], || {
        format!("hevc deblock luma {luma} vertical {vertical} lines {lines} tc {tc:?} beta {beta}")
    });
}

/// AV2 filter lengths (`filter_length_neg` / `_pos`).
const AV2_LENGTHS: [u8; 6] = [4, 6, 8, 10, 14, 18];
/// Side of the AV2 deblocking test plane: edges at every position from the
/// border reach to the middle, so both the vector windows and the scalar
/// lines near the plane edges run.
const AV2_PLANE: usize = 32;

fn check_av2_deblock(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut plane = vec![0u8; AV2_PLANE * AV2_PLANE];
    // Smooth ramps with a step across the edge (wide filters engage), or
    // extreme values (the clipping of the moved samples).
    let extreme = rng.below(4) == 0;
    let level = rng.byte();
    for s in &mut plane {
        *s = if extreme {
            [0, u8::MAX][rng.below(2)]
        } else {
            level.saturating_add(rng.byte() % 6)
        };
    }
    let vertical = rng.below(2) == 0;
    let filter = Av2EdgeFilter {
        length_neg: AV2_LENGTHS[rng.below(AV2_LENGTHS.len())],
        length_pos: AV2_Q_LENGTHS[rng.below(AV2_Q_LENGTHS.len())],
        q_threshold: rng.below(120) as u16,
        side_threshold: rng.below(120) as u16,
    };
    let (reach_p, reach_q) = filter.reach();
    let edge = reach_p + rng.below(AV2_PLANE - reach_p - reach_q + 1);
    let along = rng.below(AV2_PLANE - AV2_EDGE_LINES + 1);
    let step = rng.byte() % 48;
    if !extreme {
        for (i, s) in plane.iter_mut().enumerate() {
            let (r, c) = (i / AV2_PLANE, i % AV2_PLANE);
            if (if vertical { c } else { r }) >= edge {
                *s = s.saturating_add(step);
            }
        }
    }
    let pos = if vertical {
        (edge, along)
    } else {
        (along, edge)
    };
    let mut outputs = [plane.clone(), plane];
    for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
        let mut dst = PlaneViewMut::new(out, AV2_PLANE, AV2_PLANE, AV2_PLANE).expect("plane");
        dsp.av2_deblock_edge(&mut dst, pos, vertical, filter);
    }
    record(report, candidate.isa(), outputs[0] == outputs[1], || {
        format!("av2 deblock vertical {vertical} at {pos:?} {filter:?} extreme {extreme}")
    });
}

/// Side of the AV2 edge-line test plane: twelve segments along each edge
/// line (one batch of eight and a remainder), and edge lines from the
/// border to the middle (batched and one by one).
const AV2_LINE_PLANE: usize = 48;
/// Spacing of AV2 edge lines and segments.
const AV2_EDGE_STEP: usize = 4;
/// Noise amplitudes of the edge-line content: flat sides reach the widest
/// filters, rough ones each early exit of the decision.
const AV2_NOISE: [u8; 5] = [1, 2, 3, 6, 40];
/// AV2 q-side filter lengths: only p sides are ever shortened to 6 (a q
/// side of 6 would let the decision read past its reach).
const AV2_Q_LENGTHS: [u8; 5] = [4, 8, 10, 14, 18];

fn check_av2_deblock_line(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let mut plane = vec![0u8; AV2_LINE_PLANE * AV2_LINE_PLANE];
    let extreme = rng.below(8) == 0;
    let noise = AV2_NOISE[rng.below(AV2_NOISE.len())];
    let level = rng.byte();
    for s in &mut plane {
        *s = if extreme {
            [0, u8::MAX][rng.below(2)]
        } else {
            level.saturating_add(rng.byte() % noise)
        };
    }
    let vertical = rng.below(2) == 0;
    let coordinate = AV2_EDGE_STEP * (1 + rng.below(AV2_LINE_PLANE / AV2_EDGE_STEP - 1));
    if !extreme {
        // A step across the edge, different per segment.
        for (i, s) in plane.iter_mut().enumerate() {
            let (r, c) = (i / AV2_LINE_PLANE, i % AV2_LINE_PLANE);
            let (across, along) = if vertical { (c, r) } else { (r, c) };
            if across >= coordinate {
                *s = s.saturating_add((along / AV2_EDGE_STEP * 7 % 48) as u8);
            }
        }
    }
    // Any segment count, so lines end in a short last chunk too.
    let segments = 1 + rng.below(AV2_LINE_PLANE / AV2_EDGE_STEP);
    let filters: Vec<Option<Av2EdgeFilter>> = (0..segments)
        .map(|_| {
            let filter = Av2EdgeFilter {
                length_neg: AV2_LENGTHS[rng.below(AV2_LENGTHS.len())],
                length_pos: AV2_Q_LENGTHS[rng.below(AV2_Q_LENGTHS.len())],
                q_threshold: rng.below(120) as u16,
                side_threshold: rng.below(400) as u16,
            };
            let (reach_p, reach_q) = filter.reach();
            let inside = coordinate >= reach_p && coordinate + reach_q <= AV2_LINE_PLANE;
            (inside && rng.below(4) != 0).then_some(filter)
        })
        .collect();
    let mut outputs = [plane.clone(), plane.clone(), plane];
    for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
        let mut dst =
            PlaneViewMut::new(out, AV2_LINE_PLANE, AV2_LINE_PLANE, AV2_LINE_PLANE).expect("plane");
        dsp.av2_deblock_line(&mut dst, coordinate, vertical, &filters);
    }
    // The line equals its segments filtered one by one.
    let mut dst = PlaneViewMut::new(
        &mut outputs[2],
        AV2_LINE_PLANE,
        AV2_LINE_PLANE,
        AV2_LINE_PLANE,
    )
    .expect("plane");
    for (segment, filter) in filters.iter().enumerate() {
        if let Some(filter) = *filter {
            let along = segment * AV2_EDGE_STEP;
            let pos = if vertical {
                (coordinate, along)
            } else {
                (along, coordinate)
            };
            reference.av2_deblock_edge(&mut dst, pos, vertical, filter);
        }
    }
    let agree = outputs[0] == outputs[1] && outputs[1] == outputs[2];
    record(report, candidate.isa(), agree, || {
        format!(
            "av2 deblock line vertical {vertical} at {coordinate} noise {noise} extreme {extreme} {filters:?}"
        )
    });
}

/// Every band position, vector tail, mask mode and signed offset extreme.
/// Tight final rows plus untouched padding exercise raw SIMD access contracts.
fn check_sao_apply_boundaries(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for width in [0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
        let stride = width + 9;
        let height = 5;
        let samples: Vec<u8> = (0..stride * (height - 1) + width + 2)
            .map(|i| (i * 19 + i / stride * 73) as u8)
            .collect();
        let bounds = (1, 1, width + 1, 4);
        let mask: Vec<bool> = (0..width * 3).map(|i| i % 3 == 0).collect();
        for position in 0..32 {
            for (offsets, masked) in [
                ([7, -7, 3, -3], false),
                ([-128, 127, -1, 1], false),
                ([127, -128, 1, -1], true),
            ] {
                let mut outputs = [samples.clone(), samples.clone()];
                for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
                    let mut plane = PlaneViewMut::new(out, stride, width + 2, height).unwrap();
                    dsp.sao_band_apply_masked(
                        position,
                        offsets,
                        &mut plane,
                        bounds,
                        if masked { &mask } else { &[] },
                    );
                }
                record(report, candidate.isa(), outputs[0] == outputs[1], || {
                    format!(
                        "sao band position={position} width={width} masked={masked} offsets={offsets:?}"
                    )
                });
            }
        }
        for class in 0..4 {
            for offsets in [[7, 3, -3, -7], [-128, 127, -1, 1]] {
                let source = PlaneView::new(&samples, stride, width + 2, height).unwrap();
                let mut outputs = [samples.clone(), samples.clone()];
                for (dsp, out) in [candidate, reference].into_iter().zip(&mut outputs) {
                    let mut plane = PlaneViewMut::new(out, stride, width + 2, height).unwrap();
                    dsp.sao_edge_apply(class, offsets, &source, &mut plane, bounds);
                }
                record(report, candidate.isa(), outputs[0] == outputs[1], || {
                    format!("sao edge class={class} width={width} offsets={offsets:?}")
                });
            }
        }
    }
}

/// Each endpoint alone can reject strong filtering, including equality at
/// the strict threshold. Keep the other segment strong and exercise tC0.
fn check_hevc_strong_boundaries(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const SIDE: usize = 16;
    for vertical in [false, true] {
        for endpoint in [0, 3, 4, 7] {
            for kind in 0..3 {
                for delta in [-1i16, 0, 1] {
                    for tc in [[4, 4], [0, 4], [4, 0]] {
                        let mut input = vec![100u8; SIDE * SIDE];
                        for line in 0..8 {
                            let mut taps = [100u8, 100, 100, 100, 103, 103, 103, 103];
                            if line == endpoint {
                                match kind {
                                    // beta64: 2*dpq <16, ends<8, step<10.
                                    0 => taps[1] = (108 + delta) as u8,
                                    1 => taps[0] = (108 + delta) as u8,
                                    _ => taps[4..].fill((110 + delta) as u8),
                                }
                            }
                            for (tap, value) in taps.into_iter().enumerate() {
                                let (x, y) = if vertical {
                                    (4 + tap, 4 + line)
                                } else {
                                    (4 + line, 4 + tap)
                                };
                                input[y * SIDE + x] = value;
                            }
                        }
                        let mut output = [input.clone(), input];
                        for (dsp, pixels) in [candidate, reference].into_iter().zip(&mut output) {
                            let mut view =
                                PlaneViewMut::new(pixels, SIDE, SIDE, SIDE).expect("plane");
                            let at = if vertical { (8, 4) } else { (4, 8) };
                            dsp.hevc_deblock_luma_edge(&mut view, at, vertical, tc, 64);
                        }
                        record(report, candidate.isa(), output[0] == output[1], || {
                            format!(
                                "hevc strong boundary vertical {vertical} endpoint {endpoint} kind {kind} delta {delta} tc {tc:?}"
                            )
                        });
                    }
                }
            }
        }
    }
}

/// Sampled statistics with independent strides and unaligned exact-extent
/// planes. Width32 crosses the 128-vector flush boundary at rows127/128/129;
/// width64 does so at rows63/64/65. Short and cropped rectangles also cover
/// dispatch setup, sampled final-row weighting and the AVX2 tail fallback.
fn check_sao_stats_boundaries(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for width in [16, 31, 32, 33, 63, 64, 65] {
        let heights: &[usize] = match width {
            32 => &[0, 1, 2, 3, 127, 128, 129],
            64 => &[0, 1, 2, 3, 63, 64, 65],
            _ => &[1, 3],
        };
        for &height in heights {
            for step in 1..=4 {
                // Keep selected-row counts at the flush boundary even when
                // sampling. The extra row exercises the partial final group.
                let rows = if height <= 3 {
                    height
                } else {
                    height * step + usize::from(step > 1)
                };
                let (w, h) = (width + 2, rows + 2);
                let (src_stride, rec_stride) = (w + 7, w + 13);
                let (src_offset, rec_offset) = (1, 7);
                let mut src = vec![0; src_offset + (h - 1) * src_stride + w];
                let mut rec = vec![0; rec_offset + (h - 1) * rec_stride + w];
                for pattern in 0..3 {
                    for y in 0..h {
                        for x in 0..w {
                            let (a, b) = match pattern {
                                0 => (255, if (x + y).is_multiple_of(2) { 0 } else { 255 }),
                                1 => (0, if (x + y).is_multiple_of(2) { 255 } else { 0 }),
                                _ => (
                                    ((x * 73 + y * 19) % 256) as u8,
                                    ((x * 41 + y * 37 + x / 3) % 256) as u8,
                                ),
                            };
                            src[src_offset + y * src_stride + x] = a;
                            rec[rec_offset + y * rec_stride + x] = b;
                        }
                    }
                    let src = PlaneView::new(&src[src_offset..], src_stride, w, h).unwrap();
                    let rec = PlaneView::new(&rec[rec_offset..], rec_stride, w, h).unwrap();
                    for class in 0..4 {
                        let bounds = (1, 1, width + 1, rows + 1);
                        let got = candidate.sao_edge_stats_sampled(class, &src, &rec, bounds, step);
                        let want =
                            reference.sao_edge_stats_sampled(class, &src, &rec, bounds, step);
                        record(report, candidate.isa(), got == want, || {
                            format!(
                                "sao sampled width{width} rows{rows} step{step} class{class} pattern{pattern}"
                            )
                        });
                    }
                }
            }
        }
    }
}
