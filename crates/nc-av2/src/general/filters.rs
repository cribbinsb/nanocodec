//! The general decoder's in-loop filters, in dav2d's order
//! (`dav2d_filter_slice`): deblocking, then CDEF with CCSO, then loop
//! restoration with GDF. dav2d runs them in 64-row bands behind the block
//! layer; each reads only what an earlier stage left for the whole frame
//! (the deblocking edges of one band reach no unfiltered rows of the next,
//! CDEF and CCSO read the deblocked frame, restoration the CDEF output with
//! deblocked rows at its stripe boundaries), so here each stage runs over
//! the whole frame in turn.
//!
//! The block layer fills [`FilterMaps`] as it decodes: the deblocking edges
//! ([`edges`]), each 64x64 filter block's quantizer, CDEF set and CCSO and
//! GDF flags, which 4x4 units code a luma residual, and each unit's
//! segment.
//!
//! Every filter leaves the 4x4 units of lossless segments as decoded
//! (dav2d's `lossless_mask_y`/`_uv`): deblocking moves no lossless side,
//! CCSO skips their samples, and CDEF (by 8x8 luma unit), restoration and
//! GDF read only copies of the picture made before them, so their lossless
//! units are put back from that copy afterwards ([`keep_lossless`]).

pub mod deblock;
pub mod edges;
pub mod gdf;
pub mod restoration;
pub mod stripes;

use edges::EdgeMap;
use nc_common::decode::{DecodeError, Result};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

use super::block::segmentation::SegmentMaps;
use super::frame_header::FrameHeader;
use super::frame_header::filters::{CCSO_BAND_STRIDE, CCSO_CLASS_STRIDE, CcsoPlane};
use super::frame_header::restoration::{MAX_CLASSES, Taps};
use crate::ccso::{self, CcsoParams, CcsoSource, PlaneFilter};
use crate::cdef::{self, CdefParams, PictureFilterScratch};

/// Planes of 4:2:0.
const PLANES: usize = 3;
/// Luma samples per 4x4 unit, as a shift.
const UNIT_LOG2: u32 = 2;

/// Side of the luma units CDEF filters or leaves alone.
const CDEF_LUMA_UNIT: usize = 8;
/// Side of the 4x4 units restoration and GDF leave alone.
pub(crate) const LOSSLESS_UNIT: usize = 1 << UNIT_LOG2;

/// Luma 4x4 units per 64x64 filter block, as a shift.
pub const UNITS64_LOG2: u32 = 4;

/// What the block layer records for the loop filters of one frame.
#[derive(Debug, Clone, Default)]
pub struct FilterMaps {
    /// 64x64 filter blocks across the frame (dav2d `sb64w`).
    pub blocks_across: usize,
    /// Per filter block: the quantizer index of its superblock (dav2d
    /// `lf_mask->qidx`).
    pub qindex: Vec<u16>,
    /// Per filter block: the CDEF strength set read, `None` while no
    /// block has read one (dav2d `cdef_idx` -1).
    pub cdef_index: Vec<Option<u8>>,
    /// Per filter block and plane: the CCSO flag.
    pub ccso_flags: Vec<[bool; 3]>,
    /// Per filter block: the GDF flag.
    pub gdf_flags: Vec<bool>,
    /// The deblocking edges of luma and of chroma.
    pub luma_edges: EdgeMap,
    /// See `luma_edges`.
    pub chroma_edges: EdgeMap,
    /// Per luma 4x4 unit (row-major, `luma_edges.columns` across): the
    /// unit's block codes a luma residual (dav2d `noskip_mask`).
    pub coded: Vec<bool>,
    /// Per luma 4x4 unit: the unit's transform has coefficients (dav2d
    /// `lr_noskip_mask`).
    pub residual: Vec<bool>,
    /// Per plane and filter block: the restoration unit whose top-left
    /// luma position the block holds (dav2d `lr_mask`).
    pub restoration: [Vec<Option<RestorationUnit>>; PLANES],
    /// The segment of each luma and chroma 4x4 unit, in frames with
    /// segmentation.
    pub segments: Option<SegmentMaps>,
}

/// One loop restoration unit: its type (`restoration::kind`) and, for a
/// non-separable Wiener unit of a plane without frame filters, its filter
/// per class (chroma: one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestorationUnit {
    /// The unit's restoration type.
    pub kind: u8,
    /// Its filters.
    pub filters: [Taps; MAX_CLASSES],
}

impl FilterMaps {
    /// Empty maps for a frame `columns` x `rows` luma 4x4 units (rounded
    /// to 8x8), every filter block at quantizer `qindex`.
    #[must_use]
    pub fn new(columns: u32, rows: u32, qindex: u16) -> Self {
        let blocks = |units: u32| (units as usize).div_ceil(1 << UNITS64_LOG2);
        let (across, down) = (blocks(columns), blocks(rows));
        let filter_blocks = across * down;
        let (columns, rows) = (columns as usize, rows as usize);
        Self {
            blocks_across: across,
            qindex: vec![qindex; filter_blocks],
            cdef_index: vec![None; filter_blocks],
            ccso_flags: vec![[false; 3]; filter_blocks],
            gdf_flags: vec![false; filter_blocks],
            luma_edges: EdgeMap::new(columns, rows),
            chroma_edges: EdgeMap::new(columns / 2, rows / 2),
            coded: vec![false; columns * rows],
            residual: vec![false; columns * rows],
            restoration: std::array::from_fn(|_| vec![None; filter_blocks]),
            segments: None,
        }
    }

    /// The index of the filter block holding luma unit (`row`, `column`).
    #[must_use]
    pub fn filter_block(&self, row: u32, column: u32) -> usize {
        (row >> UNITS64_LOG2) as usize * self.blocks_across + (column >> UNITS64_LOG2) as usize
    }

    /// Mark the luma units of a block with a residual (`columns`, `rows`:
    /// its extent in units, clipped to the frame).
    pub fn mark_coded(&mut self, at: (u32, u32), extent: (u32, u32)) {
        let stride = self.luma_edges.columns;
        fill(&mut self.coded, stride, at, extent);
    }

    /// Whether the 4x4 unit (`row`, `column`) of a luma or chroma plane is
    /// in a lossless segment of `header`.
    #[must_use]
    pub fn lossless(
        &self,
        header: &FrameHeader,
        chroma: bool,
        (row, column): (usize, usize),
    ) -> bool {
        self.segments.as_ref().is_some_and(|maps| {
            let map = if chroma { &maps.chroma } else { &maps.luma };
            header.lossless.segments >> map.get(row, column) & 1 != 0
        })
    }

    /// Mark the luma units of a transform with coefficients (extent as
    /// [`Self::mark_coded`]).
    pub fn mark_residual(&mut self, at: (u32, u32), extent: (u32, u32)) {
        let stride = self.luma_edges.columns;
        fill(&mut self.residual, stride, at, extent);
    }
}

/// Set a rectangle of a per-unit flag map.
fn fill(map: &mut [bool], stride: usize, (row, column): (u32, u32), (columns, rows): (u32, u32)) {
    for y in row..row + rows {
        let start = y as usize * stride + column as usize;
        map[start..start + columns as usize].fill(true);
    }
}

/// The frame's CDEF parameters (`single_picture`: a single-picture
/// sequence, whose one zero set means off).
/// # Errors
/// Rejects strengths out of range.
pub fn cdef_params(header: &FrameHeader, single_picture: bool) -> Result<CdefParams> {
    let Some(cdef) = header.cdef else {
        return Ok(CdefParams::OFF);
    };
    let sets: Vec<(u8, u8)> = (0..usize::from(cdef.strengths))
        .map(|set| (cdef.y_strength[set], cdef.uv_strength[set]))
        .collect();
    CdefParams::decoded(cdef.damping, &sets, cdef.on_skipped, single_picture)
}

/// The frame's CCSO filters (a reused plane's parameters are already the
/// reference's in the header).
/// # Errors
/// Rejects parameters out of range.
pub fn ccso_params(header: &FrameHeader) -> Result<CcsoParams> {
    if !header.ccso.enabled {
        return Ok(CcsoParams::OFF);
    }
    let mut planes = [None; PLANES];
    for (plane, coded) in planes.iter_mut().zip(&header.ccso.planes) {
        if coded.enabled {
            *plane = Some(ccso_filter(coded)?);
        }
    }
    CcsoParams::new(planes)
}

/// One plane's CCSO filter from its header fields, its shape checked
/// before its offsets fill the filter's table.
fn ccso_filter(coded: &CcsoPlane) -> Result<PlaneFilter> {
    let mut filter = PlaneFilter::zero(
        coded.band_only,
        coded.scale_index,
        coded.quant_index,
        coded.filter_support,
        coded.edge_classifier,
        coded.max_band_log2,
    );
    if !filter.is_valid() {
        return Err(DecodeError::Malformed("AV2 CCSO parameter out of range"));
    }
    for class0 in 0..filter.intervals() {
        for class1 in 0..filter.intervals() {
            for band in 0..filter.bands() {
                let code =
                    coded.offsets[(class0 * CCSO_CLASS_STRIDE + class1) * CCSO_BAND_STRIDE + band];
                filter.set_offset(filter.entry(band, class0, class1), code);
            }
        }
    }
    Ok(filter)
}

/// CDEF, then CCSO classifying on the deblocked luma (dav2d
/// `dav2d_cdef_slice`): the picture filters of our own block layer, fed
/// from the general block layer's maps.
/// # Errors
/// Rejects filter parameters out of range.
pub(crate) fn cdef_and_ccso(
    picture: &mut Picture,
    maps: &FilterMaps,
    (header, single_picture): (&FrameHeader, bool),
    dsp: &Dsp,
    (cdef_scratch, ccso_source): (&mut PictureFilterScratch, &mut CcsoSource),
) -> Result<()> {
    let cdef = cdef_params(header, single_picture)?;
    let ccso = ccso_params(header)?;
    if ccso.is_on() {
        ccso_source.fill_from(&picture.plane(PlaneIndex::Y).view());
    }
    let lossless = header.lossless.segments != 0;
    let deblocked = lossless.then(|| picture.clone());
    let unit = |sample: usize| (sample >> UNIT_LOG2) as u32;
    let stride = maps.luma_edges.columns;
    let unit_set = |x: usize, y: usize| {
        let (column, row) = (unit(x), unit(y));
        let set = maps.cdef_index[maps.filter_block(row, column)]?;
        // dav2d `noskip_mask`: one of the 8x8 unit's four 4x4 units codes
        // a luma residual.
        let first = row as usize * stride + column as usize;
        let coded = maps.coded[first..first + 2]
            .iter()
            .chain(&maps.coded[first + stride..first + stride + 2])
            .any(|&coded| coded);
        (coded || !cdef.passes_over(true)).then_some(set)
    };
    cdef::filter_picture(picture, unit_set, cdef, dsp, cdef_scratch);
    if let Some(deblocked) = &deblocked {
        keep_lossless(picture, deblocked, maps, header, CDEF_LUMA_UNIT);
    }
    if ccso.is_on() {
        let flag_width = if lossless {
            LOSSLESS_UNIT
        } else {
            1 << (UNITS64_LOG2 + UNIT_LOG2)
        };
        ccso::filter_picture_runs(picture, ccso_source, &ccso, flag_width, |plane, x, y| {
            let chroma = plane != PlaneIndex::Y;
            // (x, y) is in luma samples: the plane's unit is a shift less.
            let at = |luma: usize| luma >> (UNIT_LOG2 + u32::from(chroma));
            maps.ccso_flags[maps.filter_block(unit(y), unit(x))][plane as usize]
                && !(lossless && maps.lossless(header, chroma, (at(y), at(x))))
        });
    }
    Ok(())
}

/// Put back into `picture`, from `before` (the picture before a filter
/// that reads only such a copy), every lossless 4x4 unit of a frame with
/// lossless segments; in luma, every `luma_unit`-sided unit holding one
/// (CDEF leaves whole 8x8 luma units alone, dav2d `y_lossless`).
pub(crate) fn keep_lossless(
    picture: &mut Picture,
    before: &Picture,
    maps: &FilterMaps,
    header: &FrameHeader,
    luma_unit: usize,
) {
    const UNIT: usize = 1 << UNIT_LOG2;
    for plane in PlaneIndex::ALL {
        let chroma = plane != PlaneIndex::Y;
        let side = if chroma { UNIT } else { luma_unit };
        let source = before.plane(plane).view();
        let mut output = picture.plane_mut(plane).view_mut();
        let (width, height) = (output.width(), output.height());
        for top in (0..height).step_by(side) {
            for left in (0..width).step_by(side) {
                let units = |start: usize, end: usize| (start / UNIT)..end.div_ceil(UNIT);
                let (bottom, right) = ((top + side).min(height), (left + side).min(width));
                let lossless = units(top, bottom).any(|row| {
                    units(left, right).any(|column| maps.lossless(header, chroma, (row, column)))
                });
                if lossless {
                    for y in top..bottom {
                        output.row_mut(y)[left..right].copy_from_slice(&source.row(y)[left..right]);
                    }
                }
            }
        }
    }
}
