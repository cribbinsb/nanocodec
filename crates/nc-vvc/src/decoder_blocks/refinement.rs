//! General B-picture BDOF/DMVR control through shared interpolation/refinement DSP.
//! Spatial motion remains unrefined; only temporal storage receives DMVR (§8.5.3).
use super::{Inter, SliceHeader, Unit, motion::Motion};
use crate::motion::Mv16;
use nc_common::{Picture, Plane, PlaneIndex, PlaneView};
use nc_dsp::Dsp;

const REGION_SIDE: usize = 16;
const SEARCH_BORDER: usize = 2;
const SEARCH_SIDE: usize = REGION_SIDE + 2 * SEARCH_BORDER;
const BILINEAR_SIDE: usize = SEARCH_SIDE + 1;
const OPTICAL_SIDE: usize = REGION_SIDE + 2;

#[derive(Debug, Clone, Copy)]
pub(super) struct Region {
    pub delta: Mv16,
    bdof: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Bounds {
    minimum: (i32, i32),
    maximum: (i32, i32),
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
    pub(super) fn direct(
        self,
        reference: &Plane,
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
    pub fn new(reference: &Plane, bounds: Option<Bounds>, edges: Edges) -> Self {
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
}

pub(crate) fn derive(
    units: &mut [Unit],
    header: &SliceHeader,
    poc: i32,
    pocs: [&[i32]; 2],
    (pictures, edges): ([&[&Picture]; 2], Edges),
    dsp: &Dsp,
) {
    if header.picture.motion.bdof_disabled && header.picture.motion.dmvr_disabled {
        return;
    }
    // All search windows are at most 20×20. Reuse bounded workspaces across
    // regions; each interpolation overwrites its complete active rectangle.
    let mut inputs = [[0i16; SEARCH_SIDE * SEARCH_SIDE]; 2];
    let mut pixels = [0u8; BILINEAR_SIDE * BILINEAR_SIDE];
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
        // Signalled weights of either reference switch both refinements off
        // (H.266 §8.5.1 dmvrFlag, §8.5.6.1 bdofFlag).
        if motion.weight != 0
            || header
                .weights
                .signalled([first.reference, second.reference])
            || distance == 0
            || distance != pocs[1][usize::from(second.reference)] - poc
        {
            continue;
        }
        unit.bdof = !header.picture.motion.bdof_disabled && !unit.smvd;
        if header.picture.motion.dmvr_disabled || !matches!(unit.inter, Some(Inter::Merge(_))) {
            continue;
        }
        let (width, height) = (n.width.min(REGION_SIDE), n.height.min(REGION_SIDE));
        unit.dmvr.reserve((n.width / width) * (n.height / height));
        for y in (n.y..n.y + n.height).step_by(height) {
            for x in (n.x..n.x + n.width).step_by(width) {
                for (list, vector) in [first, second].into_iter().enumerate() {
                    bilinear(
                        dsp,
                        pictures[list][usize::from(vector.reference)].plane(PlaneIndex::Y),
                        (x, y),
                        (width + 4, height + 4),
                        (vector.vector, edges),
                        &mut pixels,
                        &mut inputs[list],
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
            }
        }
    }
}

fn bilinear(
    dsp: &Dsp,
    reference: &Plane,
    position: (usize, usize),
    size: (usize, usize),
    (vector, edges): (Mv16, Edges),
    scratch: &mut [u8; BILINEAR_SIDE * BILINEAR_SIDE],
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
    for (y, row) in pixels.chunks_exact_mut(w).enumerate() {
        let source = view.row(fetch.row(origin.1 + y as i32));
        for (x, pixel) in row.iter_mut().enumerate() {
            *pixel = source[fetch.column(origin.0 + x as i32)];
        }
    }
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
pub(super) fn temporal(unit: &Unit, x: usize, y: usize) -> Option<Motion> {
    let motion = super::motion::at(unit, x, y)?;
    if unit.dmvr.is_empty() {
        return Some(motion);
    }
    let stride = unit.node.width.div_ceil(16);
    let region = unit.dmvr[(y - unit.node.y) / 16 * stride + (x - unit.node.x) / 16];
    Some(refined(motion, region.delta))
}

pub(super) fn predict(
    dsp: &Dsp,
    references: super::inter_prediction::Sources<'_>,
    unit: &Unit,
    plane: PlaneIndex,
    block: nc_encoder::Block,
    out: &mut [u8],
) {
    let chroma = usize::from(plane != PlaneIndex::Y);
    let position = (block.x >> chroma, block.y >> chroma);
    let size = (block.width() >> chroma, block.height() >> chroma);
    let step = 16 >> chroma;
    let base = unit.motion.unwrap();
    // The active rectangles vary at region boundaries, but never exceed 16².
    // Keep both predictions, their optical-flow borders and output reusable.
    let mut samples = [[0i16; REGION_SIDE * REGION_SIDE]; 2];
    let mut padded = [[0i16; OPTICAL_SIDE * OPTICAL_SIDE]; 2];
    let mut pixels = [0u8; REGION_SIDE * REGION_SIDE];
    for y in (0..size.1).step_by(step) {
        for x in (0..size.0).step_by(step) {
            let (width, height) = (step.min(size.0 - x), step.min(size.1 - y));
            let index = (block.y - unit.node.y + (y << chroma)) / 16 * unit.node.width.div_ceil(16)
                + (block.x - unit.node.x + (x << chroma)) / 16;
            let region = unit.dmvr.get(index).copied();
            let motion = region.map_or(base, |r| refined(base, r.delta));
            let bdof = chroma == 0 && region.map_or(unit.bdof, |r| r.bdof);
            for list in 0..2 {
                let vector = motion.lists[list].unwrap();
                let original = base.lists[list].unwrap().vector;
                let reference =
                    references.pictures[list][usize::from(vector.reference)].plane(plane);
                let at = (position.0 + x, position.1 + y);
                let bits = 4 + chroma;
                // Reference padding is needed only if refinement changes the
                // integer origin; fractional-only refinement uses the normal window.
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
                    (bounds, references.edges(plane)),
                    &mut samples[list],
                );
                if bdof {
                    padding(
                        reference,
                        at,
                        (width, height),
                        vector.vector,
                        (bounds, references.edges(plane)),
                        &samples[list],
                        &mut padded[list][..(width + 2) * (height + 2)],
                    );
                }
            }
            if bdof {
                dsp.vvc_bdof((width, height), [&padded[0], &padded[1]], &mut pixels);
            } else {
                // A refined block has no signalled weights (`derive`).
                super::inter_prediction::combine(
                    dsp,
                    (motion, None, plane),
                    &samples[0],
                    &samples[1],
                    (width, height),
                    &mut pixels,
                );
            }
            for row in 0..height {
                out[(y + row) * size.0 + x..(y + row) * size.0 + x + width]
                    .copy_from_slice(&pixels[row * width..(row + 1) * width]);
            }
        }
    }
}

pub(super) fn padding(
    reference: &Plane,
    position: (usize, usize),
    size: (usize, usize),
    vector: Mv16,
    (bounds, edges): (Option<Bounds>, Edges),
    interior: &[i16],
    samples: &mut [i16],
) {
    let (width, height) = size;
    let stride = width + 2;
    let origin = (
        position.0 as i32 + (vector.x >> 4) + ((vector.x & 15) >> 3),
        position.1 as i32 + (vector.y >> 4) + ((vector.y & 15) >> 3),
    );
    let view = reference.view();
    let fetch = Fetch::new(reference, bounds, edges);
    assert_eq!(samples.len(), stride * (height + 2));
    for (y, row) in samples.chunks_exact_mut(stride).enumerate() {
        let source = view.row(fetch.row(origin.1 + y as i32 - 1));
        for (x, value) in row.iter_mut().enumerate() {
            if y > 0 && y <= height && x > 0 && x <= width {
                *value = interior[(y - 1) * width + x - 1];
            } else {
                *value = i16::from(source[fetch.column(origin.0 + x as i32 - 1)]) << 6;
            }
        }
    }
}
