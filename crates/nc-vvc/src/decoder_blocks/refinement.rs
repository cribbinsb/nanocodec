//! General B-picture BDOF/DMVR control through shared interpolation/refinement DSP.
//! Spatial motion remains unrefined; only temporal storage receives DMVR (§8.5.3).
use super::{Inter, SliceHeader, Unit, motion::Motion};
use crate::motion::Mv16;
use nc_common::{BitDepth, Plane, PlaneIndex, PlaneView, PlaneViewMut, Sample};
use nc_dsp::{DepthKernels, Pixel};

const REGION_SIDE: usize = 16;
const SEARCH_BORDER: usize = 2;
const SEARCH_SIDE: usize = REGION_SIDE + 2 * SEARCH_BORDER;
const CU_BILINEAR_SIDE: usize = 128 + 2 * SEARCH_BORDER;
const CU_PIXELS_SIDE: usize = CU_BILINEAR_SIDE + 1;
const OPTICAL_SIDE: usize = REGION_SIDE + 2;
/// Bits of the high-precision samples BDOF and PROF pad with whole
/// reference samples (H.266 §8.5.6.3.1: `shift3 = Max(2, 14 − BitDepth)`).
const PRECISION: u32 = 14;

#[derive(Debug, Clone, Copy)]
pub(super) struct Region {
    pub delta: Mv16,
    bdof: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Bounds {
    pub(super) minimum: (i32, i32),
    pub(super) maximum: (i32, i32),
}
/// Where the reference samples of one plane may be read from (H.266
/// §8.5.6.3.2): inside the current subpicture when it is treated as a
/// picture, else across the left and right picture edges when the picture
/// wraps around, else inside the picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Edges {
    /// The wrap-around offset in samples of the plane.
    pub wrap: Option<i32>,
    /// The subpicture's first and last columns and rows in samples of the
    /// plane (`SubpicLeftBoundaryPos`... divided by the subsampling).
    pub area: Option<[i32; 4]>,
}
impl Edges {
    /// Whether a window of `size` samples at `origin` reads only what the
    /// plane holds there, its replicated border included: only those may
    /// be borrowed directly instead of fetched sample by sample.
    pub(super) fn direct<T: Sample>(
        self,
        reference: &Plane<T>,
        origin: (i32, i32),
        size: (usize, usize),
    ) -> bool {
        let (right, bottom) = (origin.0 + size.0 as i32, origin.1 + size.1 as i32);
        match (self.area, self.wrap) {
            (Some([left, top, last_x, last_y]), _) => {
                origin.0 >= left && origin.1 >= top && right <= last_x + 1 && bottom <= last_y + 1
            }
            (None, Some(_)) => origin.0 >= 0 && right <= reference.width() as i32,
            (None, None) => true,
        }
    }
}

/// The reference sample positions a prediction window reads (H.266
/// §8.5.6.3.2), its clamps worked out once per window: inside a refinement
/// window's padding bounds, then inside the subpicture treated as a picture,
/// or wrapped horizontally when the picture wraps around (`ClipH`), then
/// inside the picture.
#[derive(Clone, Copy)]
pub(super) struct Fetch {
    bounds: Option<Bounds>,
    /// The wrap-around offset and the plane width, when the picture wraps.
    wrap: Option<(i32, i32)>,
    /// The first and last readable column and row.
    columns: (i32, i32),
    rows: (i32, i32),
}
impl Fetch {
    pub fn new<T: Sample>(reference: &Plane<T>, bounds: Option<Bounds>, edges: Edges) -> Self {
        let (width, height) = (reference.width() as i32, reference.height() as i32);
        let (columns, rows) = match edges.area {
            Some([left, top, right, bottom]) => ((left, right), (top, bottom)),
            None => ((0, width - 1), (0, height - 1)),
        };
        Self {
            bounds,
            wrap: edges
                .wrap
                .filter(|_| edges.area.is_none())
                .map(|offset| (offset, width)),
            columns,
            rows,
        }
    }
    /// The column read for column `x`.
    pub fn column(self, mut x: i32) -> usize {
        if let Some(bounds) = self.bounds {
            x = x.clamp(bounds.minimum.0, bounds.maximum.0);
        }
        if let Some((offset, width)) = self.wrap {
            if x < 0 {
                x += offset;
            } else if x >= width {
                x -= offset;
            }
        }
        x.clamp(self.columns.0, self.columns.1) as usize
    }
    /// The row read for row `y`.
    pub fn row(self, mut y: i32) -> usize {
        if let Some(bounds) = self.bounds {
            y = y.clamp(bounds.minimum.1, bounds.maximum.1);
        }
        y.clamp(self.rows.0, self.rows.1) as usize
    }
    /// Fill `dst` from `source` starting at column `x0`, using contiguous
    /// slice copies and edge fills when wrap-around is inactive.
    pub fn fill_row<T: Copy>(self, source: &[T], x0: i32, dst: &mut [T]) {
        if self.wrap.is_some() {
            for (i, pixel) in dst.iter_mut().enumerate() {
                *pixel = source[self.column(x0 + i as i32)];
            }
            return;
        }
        let len = dst.len() as i32;
        if len == 0 {
            return;
        }
        let (min_x, max_x) = match self.bounds {
            Some(b) => (
                b.minimum.0.clamp(self.columns.0, self.columns.1),
                b.maximum.0.clamp(self.columns.0, self.columns.1),
            ),
            None => self.columns,
        };
        let left_pad = (min_x - x0).clamp(0, len) as usize;
        let right_start = (max_x + 1 - x0).clamp(left_pad as i32, len) as usize;
        let mid_len = right_start - left_pad;
        if left_pad > 0 {
            dst[..left_pad].fill(source[min_x as usize]);
        }
        if mid_len > 0 {
            let src_start = (x0 + left_pad as i32) as usize;
            dst[left_pad..right_start].copy_from_slice(&source[src_start..src_start + mid_len]);
        }
        if right_start < dst.len() {
            dst[right_start..].fill(source[max_x as usize]);
        }
    }
    /// Fill an entire `(w, h)` window starting at `origin`, hoisting horizontal
    /// and vertical clamp bounds out of the row loop.
    pub fn fill_window<T: Sample>(
        self,
        view: &PlaneView<'_, T>,
        origin: (i32, i32),
        (w, _h): (usize, usize),
        samples: &mut [T],
    ) {
        if self.wrap.is_some() || w == 0 {
            for (y, row) in samples.chunks_exact_mut(w).enumerate() {
                self.fill_row(view.row(self.row(origin.1 + y as i32)), origin.0, row);
            }
            return;
        }
        let ((min_x, max_x), (min_y, max_y)) = match self.bounds {
            Some(b) => (
                (
                    b.minimum.0.clamp(self.columns.0, self.columns.1),
                    b.maximum.0.clamp(self.columns.0, self.columns.1),
                ),
                (
                    b.minimum.1.clamp(self.rows.0, self.rows.1),
                    b.maximum.1.clamp(self.rows.0, self.rows.1),
                ),
            ),
            None => (self.columns, self.rows),
        };
        let len = w as i32;
        let left_pad = (min_x - origin.0).clamp(0, len) as usize;
        let right_start = (max_x + 1 - origin.0).clamp(left_pad as i32, len) as usize;
        let mid_len = right_start - left_pad;
        let src_start = (origin.0 + left_pad as i32) as usize;
        let stride = view.stride();
        let data = view.data();
        if left_pad == 0 && right_start == w {
            for (y, row) in samples.chunks_exact_mut(w).enumerate() {
                let ry = (origin.1 + y as i32).clamp(min_y, max_y) as usize;
                nc_common::copy_row(row, &data[ry * stride + src_start..][..w]);
            }
            return;
        }
        let (min_xu, max_xu) = (min_x as usize, max_x as usize);
        let right_pad = w - right_start;
        for (y, row) in samples.chunks_exact_mut(w).enumerate() {
            let ry = (origin.1 + y as i32).clamp(min_y, max_y) as usize;
            let src_row = &data[ry * stride..];
            if left_pad > 0 {
                let lv = src_row[min_xu];
                if left_pad <= 4 {
                    for d in &mut row[..left_pad] {
                        *d = lv;
                    }
                } else {
                    row[..left_pad].fill(lv);
                }
            }
            if mid_len > 0 {
                nc_common::copy_row(
                    &mut row[left_pad..right_start],
                    &src_row[src_start..src_start + mid_len],
                );
            }
            if right_pad > 0 {
                let rv = src_row[max_xu];
                if right_pad <= 4 {
                    for d in &mut row[right_start..] {
                        *d = rv;
                    }
                } else {
                    row[right_start..].fill(rv);
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct RefinementScratch<T: Pixel> {
    inputs: Box<[[i16; SEARCH_SIDE * SEARCH_SIDE]; 2]>,
    cu_inputs: [Vec<i16>; 2],
    pixels: Vec<T>,
}

impl<T: Pixel> Default for RefinementScratch<T> {
    fn default() -> Self {
        Self {
            inputs: Box::new([[0; SEARCH_SIDE * SEARCH_SIDE]; 2]),
            cu_inputs: [
                vec![0; CU_BILINEAR_SIDE * CU_BILINEAR_SIDE],
                vec![0; CU_BILINEAR_SIDE * CU_BILINEAR_SIDE],
            ],
            pixels: vec![T::default(); CU_PIXELS_SIDE * CU_PIXELS_SIDE],
        }
    }
}

pub(crate) fn derive<T: Pixel>(
    units: &mut [Unit],
    header: &SliceHeader,
    poc: i32,
    pocs: [&[i32]; 2],
    references: super::inter_prediction::Sources<'_, T>,
    dsp: DepthKernels<'_>,
    scratch: &mut RefinementScratch<T>,
) {
    let (pictures, edges) = (references.pictures, references.edges);
    if header.picture.motion.bdof_disabled && header.picture.motion.dmvr_disabled {
        return;
    }
    for unit in units {
        let n = unit.node;
        if n.width < 8
            || n.height < 8
            || n.width * n.height < 128
            || unit.affine_models.iter().any(Option::is_some)
            || !unit.subblocks.is_empty()
            || matches!(unit.inter, Some(Inter::Ciip(_) | Inter::Geometric { .. }))
        {
            continue;
        }
        let Some(motion) = unit.motion else {
            continue;
        };
        let [Some(first), Some(second)] = motion.lists else {
            continue;
        };
        let distance = poc - pocs[0][usize::from(first.reference)];
        // Signalled weights, a long-term or a scaled reference in either
        // list switch both refinements off (H.266 §8.5.1 dmvrFlag, §8.5.6.1
        // bdofFlag).
        if motion.weight != 0
            || references.scaled(motion)
            || header
                .weights
                .signalled([first.reference, second.reference])
            || header.long_term(0, first.reference)
            || header.long_term(1, second.reference)
            || distance == 0
            || distance != pocs[1][usize::from(second.reference)] - poc
        {
            continue;
        }
        unit.bdof = !header.picture.motion.bdof_disabled && !unit.smvd;
        if header.picture.motion.dmvr_disabled || !matches!(unit.inter, Some(Inter::Merge(_))) {
            continue;
        }
        derive_unit_dmvr(
            dsp,
            pictures,
            edges,
            unit,
            [first, second],
            (
                &mut scratch.inputs,
                &mut scratch.cu_inputs,
                &mut scratch.pixels,
            ),
        );
    }
}

fn derive_unit_dmvr<T: Pixel>(
    dsp: DepthKernels<'_>,
    pictures: [&[&nc_common::Picture<T>]; 2],
    edges: Edges,
    unit: &mut Unit,
    vectors: [super::motion::Vector; 2],
    (inputs, cu_inputs, pixels): (
        &mut [[i16; SEARCH_SIDE * SEARCH_SIDE]; 2],
        &mut [Vec<i16>; 2],
        &mut [T],
    ),
) {
    let n = unit.node;
    let (width, height) = (n.width.min(REGION_SIDE), n.height.min(REGION_SIDE));
    let (sub_bw, sub_bh) = (width + 4, height + 4);
    unit.dmvr.reserve((n.width / width) * (n.height / height));
    if n.width == width && n.height == height {
        for (list, vector) in vectors.into_iter().enumerate() {
            bilinear(
                dsp,
                pictures[list][usize::from(vector.reference)].plane(PlaneIndex::Y),
                (n.x, n.y),
                (sub_bw, sub_bh),
                (vector.vector, edges),
                pixels,
                &mut inputs[list][..sub_bw * sub_bh],
            );
        }
        let (delta, bdof) = dsp.vvc_dmvr((width, height), [&inputs[0], &inputs[1]]);
        unit.dmvr.push(Region {
            delta: Mv16 {
                x: delta[0],
                y: delta[1],
            },
            bdof: unit.bdof && bdof,
        });
        return;
    }
    let (cu_bw, cu_bh) = (n.width + 4, n.height + 4);
    for (list, vector) in vectors.into_iter().enumerate() {
        bilinear(
            dsp,
            pictures[list][usize::from(vector.reference)].plane(PlaneIndex::Y),
            (n.x, n.y),
            (cu_bw, cu_bh),
            (vector.vector, edges),
            pixels,
            &mut cu_inputs[list][..cu_bw * cu_bh],
        );
    }
    for dy in (0..n.height).step_by(height) {
        for dx in (0..n.width).step_by(width) {
            for list in 0..2 {
                let src_cu = &cu_inputs[list];
                let dst_sub = &mut inputs[list][..sub_bw * sub_bh];
                for (r, dst_row) in dst_sub.chunks_exact_mut(sub_bw).enumerate() {
                    let off = (dy + r) * cu_bw + dx;
                    nc_common::copy_row(dst_row, &src_cu[off..off + sub_bw]);
                }
            }
            let (delta, bdof) = dsp.vvc_dmvr((width, height), [&inputs[0], &inputs[1]]);
            unit.dmvr.push(Region {
                delta: Mv16 {
                    x: delta[0],
                    y: delta[1],
                },
                bdof: unit.bdof && bdof,
            });
        }
    }
}

fn bilinear<T: Pixel>(
    dsp: DepthKernels<'_>,
    reference: &Plane<T>,
    position: (usize, usize),
    size: (usize, usize),
    (vector, edges): (Mv16, Edges),
    scratch: &mut [T],
    out: &mut [i16],
) {
    let origin = (
        position.0 as i32 + (vector.x >> 4) - 2,
        position.1 as i32 + (vector.y >> 4) - 2,
    );
    let (w, h) = (size.0 + 1, size.1 + 1);
    let phases = ((vector.x & 15) as usize, (vector.y & 15) as usize);
    if let Some(window) = edges
        .direct(reference, origin, (w, h))
        .then(|| reference.region(origin.0 as isize, origin.1 as isize, w, h))
        .flatten()
    {
        dsp.vvc_dmvr_bilinear(size, &window, phases, out);
        return;
    }
    let pixels = &mut scratch[..w * h];
    let view = reference.view();
    let fetch = Fetch::new(reference, None, edges);
    fetch.fill_window(&view, origin, (w, h), pixels);
    dsp.vvc_dmvr_bilinear(size, &PlaneView::new(pixels, w, w, h).unwrap(), phases, out);
}

fn refined(mut motion: Motion, delta: Mv16) -> Motion {
    for (list, vector) in motion.lists.iter_mut().enumerate() {
        if let Some(vector) = vector {
            let sign = if list == 0 { 1 } else { -1 };
            vector.vector = Mv16 {
                x: vector.vector.x + sign * delta.x,
                y: vector.vector.y + sign * delta.y,
            }
            .clip();
        }
    }
    motion
}
pub(super) fn temporal_with_fields(
    unit: &Unit,
    x: usize,
    y: usize,
    fields: &[Option<crate::affine::Field>; 2],
) -> Option<Motion> {
    let motion = super::motion::at_with_fields(unit, x, y, fields)?;
    if unit.dmvr.is_empty() {
        return Some(motion);
    }
    let stride = unit.node.width.div_ceil(16);
    let region = unit.dmvr[(y - unit.node.y) / 16 * stride + (x - unit.node.x) / 16];
    Some(refined(motion, region.delta))
}

pub(super) fn predict<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: super::inter_prediction::Sources<'_, T>,
    unit: &Unit,
    plane: PlaneIndex,
    block: nc_encoder::Block,
    scratch: &mut super::inter_prediction::InterScratch<T>,
    out: &mut [T],
) {
    let (_, size) = references.plane_rect(plane, block);
    let mut dst = PlaneViewMut::new(&mut out[..size.0 * size.1], size.0, size.0, size.1).unwrap();
    predict_direct(dsp, references, unit, plane, block, scratch, &mut dst);
}

struct SubPuScratch<'a, T: Sample> {
    samples: [&'a mut [i16]; 2],
    padded: [&'a mut [i16]; 2],
    pixels: &'a mut [T; 16 * 16],
    window_buf: &'a mut [T],
}

pub(super) fn predict_direct<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: super::inter_prediction::Sources<'_, T>,
    unit: &Unit,
    plane: PlaneIndex,
    block: nc_encoder::Block,
    scratch: &mut super::inter_prediction::InterScratch<T>,
    dst: &mut PlaneViewMut<'_, T>,
) {
    let chroma = usize::from(plane != PlaneIndex::Y);
    let (sx, sy) = references.subsampling(plane);
    let (position, size) = references.plane_rect(plane, block);
    let step = (REGION_SIDE >> sx, REGION_SIDE >> sy);
    let base = unit.motion.unwrap();
    let (s0, p0_rest) = scratch.s0.split_at_mut(REGION_SIDE * REGION_SIDE);
    let (s1, p1_rest) = scratch.s1.split_at_mut(REGION_SIDE * REGION_SIDE);
    let mut sub_scratch = SubPuScratch {
        samples: [s0, s1],
        padded: [
            &mut p0_rest[..OPTICAL_SIDE * OPTICAL_SIDE],
            &mut p1_rest[..OPTICAL_SIDE * OPTICAL_SIDE],
        ],
        pixels: &mut scratch.pixels,
        window_buf: &mut scratch.window[..23 * 23],
    };
    for y in (0..size.1).step_by(step.1) {
        for x in (0..size.0).step_by(step.0) {
            let (width, height) = (step.0.min(size.0 - x), step.1.min(size.1 - y));
            let index = (block.y - unit.node.y + (y << sy)) / REGION_SIDE
                * unit.node.width.div_ceil(REGION_SIDE)
                + (block.x - unit.node.x + (x << sx)) / REGION_SIDE;
            let region = unit.dmvr.get(index).copied();
            let mut sub_dst = dst.sub_view_mut(x, y, width, height).unwrap();
            predict_sub_pu(
                dsp,
                references,
                (plane, chroma),
                (base, region, unit.bdof),
                ((position.0 + x, position.1 + y), (width, height)),
                &mut sub_scratch,
                &mut sub_dst,
            );
        }
    }
}

fn predict_sub_pu<T: Pixel>(
    dsp: DepthKernels<'_>,
    references: super::inter_prediction::Sources<'_, T>,
    (plane, chroma): (PlaneIndex, usize),
    (base, region, unit_bdof): (Motion, Option<Region>, bool),
    (at, (width, height)): ((usize, usize), (usize, usize)),
    bufs: &mut SubPuScratch<'_, T>,
    sub_dst: &mut PlaneViewMut<'_, T>,
) {
    // Chroma is refined by the luma delta, then scaled to its own units.
    let motion = references.plane_motion(region.map_or(base, |r| refined(base, r.delta)), plane);
    let base = references.plane_motion(base, plane);
    let bdof = chroma == 0 && region.map_or(unit_bdof, |r| r.bdof);
    let edges = references.edges(plane);
    let bits = 4 + chroma;
    for list in 0..2 {
        let vector = motion.lists[list].unwrap();
        let original = base.lists[list].unwrap().vector;
        let reference = references.pictures[list][usize::from(vector.reference)].plane(plane);
        let bounds = region
            .filter(|_| {
                (original.x >> bits, original.y >> bits)
                    != (vector.vector.x >> bits, vector.vector.y >> bits)
            })
            .map(|_| {
                let before = if chroma == 0 { 3 } else { 1 };
                let origin = (
                    at.0 as i32 + (original.x >> bits),
                    at.1 as i32 + (original.y >> bits),
                );
                Bounds {
                    minimum: (origin.0 - before, origin.1 - before),
                    maximum: (
                        origin.0 + width as i32 + before,
                        origin.1 + height as i32 + before,
                    ),
                }
            });
        super::inter_prediction::precision_window(
            dsp,
            reference,
            (at, (width, height)),
            vector.vector,
            (chroma == 0, motion.half, false),
            (bounds, edges),
            bufs.window_buf,
            bufs.samples[list],
        );
        if bdof {
            padding(
                (reference, dsp),
                at,
                (width, height),
                vector.vector,
                (bounds, edges),
                bufs.samples[list],
                &mut bufs.padded[list][..(width + 2) * (height + 2)],
            );
        }
    }
    if bdof {
        dsp.vvc_bdof(
            (width, height),
            [bufs.padded[0], bufs.padded[1]],
            bufs.pixels,
        );
        for row in 0..height {
            nc_common::copy_row(
                sub_dst.row_mut(row),
                &bufs.pixels[row * width..(row + 1) * width],
            );
        }
    } else {
        super::inter_prediction::combine_into(
            dsp,
            (motion, None, plane),
            bufs.samples[0],
            bufs.samples[1],
            sub_dst,
        );
    }
}

/// The `(width + 2) x (height + 2)` high-precision samples BDOF or PROF
/// read: `interior`, bordered by the whole reference samples at the nearest
/// integer positions, scaled to the precision of samples of `depth`.
pub(super) fn padding<T: Pixel>(
    (reference, dsp): (&Plane<T>, DepthKernels<'_>),
    position: (usize, usize),
    size: (usize, usize),
    vector: Mv16,
    (bounds, edges): (Option<Bounds>, Edges),
    interior: &[i16],
    samples: &mut [i16],
) {
    let (width, height) = size;
    let stride = width + 2;
    let pad_h = height + 2;
    let origin = (
        position.0 as i32 + (vector.x >> 4) + ((vector.x & 15) >> 3),
        position.1 as i32 + (vector.y >> 4) + ((vector.y & 15) >> 3),
    );
    assert_eq!(samples.len(), stride * pad_h);
    if bounds.is_none()
        && edges.direct(reference, (origin.0 - 1, origin.1 - 1), (stride, pad_h))
        && let Some(win) =
            reference.region(origin.0 as isize - 1, origin.1 as isize - 1, stride, pad_h)
    {
        dsp.vvc_bdof_pad(size, &win, interior, samples);
        return;
    }
    let view = reference.view();
    let fetch = Fetch::new(reference, bounds, edges);
    if fetch.wrap.is_none() {
        let ((min_x, max_x), (min_y, max_y)) = match fetch.bounds {
            Some(b) => (
                (
                    b.minimum.0.clamp(fetch.columns.0, fetch.columns.1),
                    b.maximum.0.clamp(fetch.columns.0, fetch.columns.1),
                ),
                (
                    b.minimum.1.clamp(fetch.rows.0, fetch.rows.1),
                    b.maximum.1.clamp(fetch.rows.0, fetch.rows.1),
                ),
            ),
            None => (fetch.columns, fetch.rows),
        };
        if origin.0 > min_x
            && origin.0 + width as i32 <= max_x
            && origin.1 > min_y
            && origin.1 + height as i32 <= max_y
        {
            let x_l = (origin.0 - 1) as usize;
            let y_t = (origin.1 - 1) as usize;
            let ref_stride = view.stride();
            let base = y_t * ref_stride + x_l;
            let span = (height + 1) * ref_stride + stride;
            let win = PlaneView::new(&view.data()[base..base + span], ref_stride, stride, pad_h)
                .expect("in-bounds BDOF window");
            dsp.vvc_bdof_pad(size, &win, interior, samples);
            return;
        }
    }
    let shift = PRECISION - BitDepth::of::<T>(dsp.depth()).bits();
    let top_src = view.row(fetch.row(origin.1 - 1));
    fill_scaled_row(fetch, top_src, origin.0 - 1, shift, &mut samples[..stride]);
    for y in 1..=height {
        let src = view.row(fetch.row(origin.1 + y as i32 - 1));
        let row = &mut samples[y * stride..(y + 1) * stride];
        row[0] = (src[fetch.column(origin.0 - 1)].to_i32() << shift) as i16;
        row[1..=width].copy_from_slice(&interior[(y - 1) * width..y * width]);
        row[width + 1] = (src[fetch.column(origin.0 + width as i32)].to_i32() << shift) as i16;
    }
    let bot_src = view.row(fetch.row(origin.1 + height as i32));
    fill_scaled_row(
        fetch,
        bot_src,
        origin.0 - 1,
        shift,
        &mut samples[(height + 1) * stride..],
    );
}

fn fill_scaled_row<T: Sample>(fetch: Fetch, source: &[T], x0: i32, shift: u32, dst: &mut [i16]) {
    if fetch.wrap.is_none() {
        let (min_x, max_x) = match fetch.bounds {
            Some(b) => (
                b.minimum.0.clamp(fetch.columns.0, fetch.columns.1),
                b.maximum.0.clamp(fetch.columns.0, fetch.columns.1),
            ),
            None => fetch.columns,
        };
        let len = dst.len() as i32;
        if x0 >= min_x && x0 + len - 1 <= max_x {
            for (out, &s) in dst.iter_mut().zip(&source[x0 as usize..]) {
                *out = (s.to_i32() << shift) as i16;
            }
            return;
        }
    }
    for (x, out) in dst.iter_mut().enumerate() {
        *out = (source[fetch.column(x0 + x as i32)].to_i32() << shift) as i16;
    }
}
