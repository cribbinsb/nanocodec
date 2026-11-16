//! The deblocking filter of the general decoder (dav2d `db_apply_tmpl.c`
//! and `deblock_tmpl.c`): every vertical edge of the frame, then every
//! horizontal one, each 4-line segment by nc-dsp's AV2 edge filter (AVM
//! `filt_choice` and the asymmetric ramp, the kernel the encoder's
//! reconstruction uses too).
//!
//! An edge's thresholds come from the quantizer of the 64x64 filter block
//! on each side with the side's segment delta (the mean of the two when
//! both filter, dav2d `setup_thr_sb_*`), offset per channel by the frame
//! header; sub-PU edges divide them by eight. Its maximum width comes from
//! its level ([`super::edges`]); edges on the top row of each 64-row band
//! leave the rows above shorter (luma at most 6 samples, chroma 2). A side
//! in a lossless segment is left unchanged.
//!
//! Not yet: the tile-boundary level fix-ups (several tiles, rejected).

use nc_common::PlaneViewMut;
use nc_dsp::{AV2_EDGE_LINES, Av2EdgeFilter, Dsp};

use super::FilterMaps;
use super::edges::{Direction, EdgeMap, LEVELS, MAX_LUMA_LEVEL, SUB_PU};
use crate::deblock::thresholds::{q_threshold, side_threshold};
use crate::general::block::pixels::Planes;
use crate::general::block::segmentation::SegmentMap;
use crate::general::block::{delta_qindex, segment_qindex};
use crate::general::frame_header::FrameHeader;
use crate::general::sequence::{MAX_SEGMENTS, SequenceHeader};

/// Filter length (dav2d `max_width` × 2 + 2) by luma and chroma level.
const LUMA_LENGTHS: [u8; 4] = [4, 8, 14, 18];
const CHROMA_LENGTHS: [u8; 3] = [4, 8, 10];
/// The p side of a horizontal edge on a band's top row reaches at most 6
/// luma or 2 chroma samples up (dav2d `deblock_v_sb64*`'s `edge`).
const LUMA_BAND_TOP_LENGTH: u8 = 14;
const CHROMA_BAND_TOP_LENGTH: u8 = 6;
/// Rows of luma 4x4 units in a band (64 rows).
const BAND_UNITS: usize = 16;
/// Qindex units per deblocking delta step (dav2d `8 * delta_q`).
const DELTA_SCALE: i32 = 8;
/// Sub-PU edges scale both thresholds down by this shift.
const SUB_PU_SHIFT: u32 = 3;
/// The bottom crop (dav2d "Crop deblock size on the bottom of the frame"):
/// a band within this many units of the frame's bottom has its 32-long
/// edges two units above the bottom shortened to level 2.
const CROP_REACH: usize = 4;
const CROP_ROWS_ABOVE: usize = 2;

/// Quantizer and side thresholds of one filter block for one channel.
type Thresholds = (u16, u16);

/// A plane's filter channel: which edges it filters and their thresholds.
struct Channel<'a> {
    plane: usize,
    edges: &'a EdgeMap,
    chroma: bool,
    /// Per filter block and segment ([`Segments::count`] per block), by
    /// direction.
    thresholds: [Vec<Thresholds>; 2],
    /// Which directions the frame filters.
    directions: [bool; 2],
    /// The segment of each of the plane's 4x4 units.
    segments: Segments<'a>,
}

/// A plane's segments as the filter reads them.
#[derive(Clone, Copy)]
struct Segments<'a> {
    /// The segment per unit, in frames with segmentation.
    map: Option<&'a SegmentMap>,
    /// Segments with thresholds per filter block.
    count: usize,
    /// The lossless segments, as a bit mask.
    lossless: u16,
}

impl Segments<'_> {
    /// The segment of unit (`column`, `row`).
    fn at(self, column: usize, row: usize) -> usize {
        self.map.map_or(0, |map| usize::from(map.get(row, column)))
    }

    /// Whether `segment` is lossless.
    fn lossless(self, segment: usize) -> bool {
        self.lossless >> segment & 1 != 0
    }
}

/// Deblock the reconstruction `planes` (padded to whole superblocks) of a
/// frame whose block layer filled `maps`.
pub fn filter(
    planes: &mut Planes,
    maps: &mut FilterMaps,
    header: &FrameHeader,
    sequence: &SequenceHeader,
    dsp: &Dsp,
) {
    let deblock = &header.deblock;
    // Chroma is filtered only beside luma (dav2d builds no masks without
    // a luma direction).
    if !deblock.level_y.iter().any(|&on| on) {
        return;
    }
    crop_bottom(&mut maps.luma_edges);
    let count = if header.segmentation.is_some() {
        MAX_SEGMENTS
    } else {
        1
    };
    // Each filter block's luma AC quantizer per segment (dav2d
    // `init_deblock_thr_lut_*`).
    let segment_qindices = || {
        maps.qindex.iter().flat_map(move |&qindex| {
            (0..count).map(move |segment| segment_qindex(header, qindex, segment))
        })
    };
    let luma = |direction: usize| {
        let offset = DELTA_SCALE * i32::from(deblock.delta_q_y[direction]);
        segment_qindices()
            .map(|qindex| thresholds(qindex + offset))
            .collect()
    };
    let chroma = |ac: i8, delta: i8| -> Vec<Thresholds> {
        let base = i32::from(ac) + i32::from(sequence.quant.base_uv_ac_delta);
        segment_qindices()
            .map(|qindex| {
                let qindex = delta_qindex(qindex, base);
                thresholds(qindex + DELTA_SCALE * i32::from(delta))
            })
            .collect()
    };
    let segments = |chroma: bool| Segments {
        map: maps
            .segments
            .as_ref()
            .map(|maps| if chroma { &maps.chroma } else { &maps.luma }),
        count,
        lossless: header.lossless.segments,
    };
    let quantizer = &header.quantizer;
    let u = chroma(quantizer.u_ac, deblock.delta_q_u);
    let channels = [
        Channel {
            plane: 0,
            edges: &maps.luma_edges,
            chroma: false,
            thresholds: [luma(0), luma(1)],
            directions: deblock.level_y,
            segments: segments(false),
        },
        Channel {
            plane: 1,
            edges: &maps.chroma_edges,
            chroma: true,
            thresholds: [u.clone(), u],
            directions: [deblock.level_u; 2],
            segments: segments(true),
        },
        Channel {
            plane: 2,
            edges: &maps.chroma_edges,
            chroma: true,
            thresholds: [
                chroma(quantizer.v_ac, deblock.delta_q_v),
                chroma(quantizer.v_ac, deblock.delta_q_v),
            ],
            directions: [deblock.level_v; 2],
            segments: segments(true),
        },
    ];
    for channel in &channels {
        let plane = channel.plane;
        let stride = planes.stride[plane];
        let height = planes.data[plane].len() / stride;
        let Some(mut view) = PlaneViewMut::new(&mut planes.data[plane], stride, stride, height)
        else {
            continue;
        };
        for direction in [Direction::Vertical, Direction::Horizontal] {
            if channel.directions[direction as usize] {
                filter_direction(&mut view, channel, maps.blocks_across, direction, dsp);
            }
        }
    }
}

/// Both thresholds of a channel's qindex (block qindex plus offsets).
fn thresholds(qindex: i32) -> Thresholds {
    (q_threshold(qindex), side_threshold(qindex))
}

/// Shorten the 32-long horizontal luma edges two units above the bottom
/// of the frame's last bands (dav2d `dav2d_deblock_slice_cols`).
fn crop_bottom(edges: &mut EdgeMap) {
    let rows = edges.rows;
    for band in 0..rows.div_ceil(BAND_UNITS) {
        let top = band * BAND_UNITS;
        if top + BAND_UNITS + CROP_REACH > rows {
            let height = (rows - top).min(BAND_UNITS);
            let row = top + height - CROP_ROWS_ABOVE;
            edges.lower_row(row, MAX_LUMA_LEVEL, MAX_LUMA_LEVEL - 1);
        }
    }
}

/// Every edge of one direction of a channel, edge line by edge line.
fn filter_direction(
    view: &mut PlaneViewMut<'_>,
    channel: &Channel<'_>,
    blocks_across: usize,
    direction: Direction,
    dsp: &Dsp,
) {
    let edges = channel.edges;
    let vertical = direction == Direction::Vertical;
    let (lines, along) = if vertical {
        (edges.columns, edges.rows)
    } else {
        (edges.rows, edges.columns)
    };
    let shift = u32::from(channel.chroma);
    let block = |column: usize, row: usize| {
        let to_block = |units: usize| (units << shift) >> super::UNITS64_LOG2;
        to_block(row) * blocks_across + to_block(column)
    };
    let thresholds = &channel.thresholds[direction as usize];
    let segments = channel.segments;
    let side = |column: usize, row: usize| {
        let segment = segments.at(column, row);
        (
            thresholds[block(column, row) * segments.count + segment],
            segments.lossless(segment),
        )
    };
    let band = BAND_UNITS >> shift;
    let mut filters = vec![None; along];
    // Segments with a lossless side, filtered after the rest of the line.
    let mut keeping = Vec::new();
    // The frame's left and top borders are not edges.
    for line in 1..lines {
        let mut any = false;
        for (position, filter) in filters.iter_mut().enumerate() {
            let (column, row) = if vertical {
                (line, position)
            } else {
                (position, line)
            };
            let set = edges.get(direction, column, row);
            *filter = None;
            if set & LEVELS == 0 {
                continue;
            }
            let (p_column, p_row) = if vertical {
                (column - 1, row)
            } else {
                (column, row - 1)
            };
            let (p, keep_p) = side(p_column, p_row);
            let (q, keep_q) = side(column, row);
            let (mut q, mut side) = combine(p, q);
            if set & SUB_PU != 0 {
                q >>= SUB_PU_SHIFT;
                side >>= SUB_PU_SHIFT;
                if side == 0 {
                    q = 0;
                }
            }
            // The highest level set (dav2d's kernels test the masks from
            // the top level down).
            let level = (set & LEVELS).ilog2() as usize;
            let (length_pos, band_top_length) = if channel.chroma {
                (CHROMA_LENGTHS[level], CHROMA_BAND_TOP_LENGTH)
            } else {
                (LUMA_LENGTHS[level], LUMA_BAND_TOP_LENGTH)
            };
            let length_neg = if !vertical && row % band == 0 {
                length_pos.min(band_top_length)
            } else {
                length_pos
            };
            let edge = Av2EdgeFilter {
                length_neg,
                length_pos,
                q_threshold: q,
                side_threshold: side,
            };
            if keep_p || keep_q {
                keeping.push((position, edge, (keep_p, keep_q)));
            } else {
                *filter = Some(edge);
                any = true;
            }
        }
        let coordinate = line * AV2_EDGE_LINES;
        if any {
            dsp.av2_deblock_line(view, coordinate, vertical, &filters);
        }
        // Segments of one edge line share no sample: their order is free.
        for (position, edge, keep) in keeping.drain(..) {
            let along = position * AV2_EDGE_LINES;
            let at = if vertical {
                (coordinate, along)
            } else {
                (along, coordinate)
            };
            dsp.av2_deblock_edge_keeping(view, at, vertical, edge, keep);
        }
    }
}

/// The thresholds of an edge from those of its two sides: their rounded
/// mean when both filter, else the one that does.
fn combine(p: Thresholds, q: Thresholds) -> Thresholds {
    let mean = |a: u16, b: u16| {
        if a != 0 && b != 0 {
            (a + b).div_ceil(2)
        } else {
            a | b
        }
    };
    (mean(p.0, q.0), mean(p.1, q.1))
}
