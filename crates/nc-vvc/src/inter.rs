//! H.266 §8.5.6.3 sixteenth-luma / 1/32-chroma interpolation of one-vector
//! blocks, and the window placement shared with affine sub-blocks.
//!
//! Vectors are in sixteenth luma samples (`motion::Mv16`); a 4:2:0 chroma
//! block reads the same vector as 1/32 chroma samples. On the quarter
//! (luma) or eighth (chroma) grid VVC's taps and 8-bit rounding equal
//! HEVC's, so those vectors use the HEVC kernels; finer ones (merged from
//! an affine neighbour) use the VVC tables. The alternate half-sample
//! filter is not enabled.
use crate::motion::{Motion, Mv16};
use nc_common::{Picture, Plane, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter};
use nc_encoder::Block;

/// Fractional bits of a luma vector.
const LUMA_FRAC_BITS: u32 = 4;
/// Fractional bits of the same vector read at 4:2:0 chroma resolution.
const CHROMA_FRAC_BITS: u32 = 5;
/// Sixteenths per HEVC luma quarter, 1/32 per HEVC chroma eighth.
const HEVC_PHASE_STEP: i32 = 4;

/// A `w`x`h` block's prediction buffer, row-major, as a view to predict into.
pub(crate) fn packed(out: &mut [u8], (w, h): (usize, usize)) -> PlaneViewMut<'_> {
    PlaneViewMut::new(out, w, w, h).expect("the prediction buffer covers the block")
}

/// The inter prediction of `plane` of CU `cu` at `motion` from `reference`
/// into `prediction`, resized to the CU's samples (row-major): the encoder
/// predicts several times per CU and keeps the buffer, so no allocation or
/// zeroing is repeated.
pub(crate) fn predict_cu_into(
    dsp: &Dsp,
    reference: &Picture,
    plane: PlaneIndex,
    cu: &Block,
    motion: &Motion,
    prediction: &mut Vec<u8>,
) {
    let shift = usize::from(plane != PlaneIndex::Y);
    let size = (cu.width() >> shift, cu.height() >> shift);
    // Every sample is written below; only a longer buffer's tail is zeroed.
    prediction.resize(size.0 * size.1, 0);
    predict_cu_view(
        dsp,
        reference,
        plane,
        cu,
        motion,
        &mut packed(prediction, size),
    );
}

/// [`predict_cu_into`] into `dst`, at least the CU's samples of `plane`:
/// the decoder predicts straight into the picture it reconstructs.
pub(crate) fn predict_cu_view(
    dsp: &Dsp,
    reference: &Picture,
    plane: PlaneIndex,
    cu: &Block,
    motion: &Motion,
    prediction: &mut PlaneViewMut<'_>,
) {
    let shift = usize::from(plane != PlaneIndex::Y);
    let (x, y) = (cu.x >> shift, cu.y >> shift);
    let (w, h) = (cu.width() >> shift, cu.height() >> shift);
    let reference = reference.plane(plane);
    match motion {
        Motion::Vector(mv) if plane == PlaneIndex::Y => {
            predict_luma(dsp, reference, (x, y), (w, h), *mv, prediction);
        }
        Motion::Vector(mv) => {
            predict_chroma(dsp, reference, (x, y), (w, h), *mv, prediction);
        }
        Motion::Affine(model) => {
            let field = model.field(cu.width(), cu.height());
            crate::affine::predict(dsp, reference, plane, cu, &field, prediction);
        }
    }
}

/// Predict the `w`x`h` luma block at `(x, y)` from `reference` displaced by
/// `mv` into `out`.
pub(crate) fn predict_luma(
    dsp: &Dsp,
    reference: &Plane,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Mv16,
    out: &mut PlaneViewMut<'_>,
) {
    let filter = if on_hevc_grid(mv) {
        Filter::HevcLuma
    } else {
        Filter::VvcLuma
    };
    predict_with(dsp, filter, reference, (x, y), (w, h), mv, out);
}

/// Predict the `w`x`h` chroma block at chroma position `(x, y)` displaced
/// by the luma vector `mv` (§8.5.6.3.4: 1/32 chroma samples for 4:2:0).
pub(crate) fn predict_chroma(
    dsp: &Dsp,
    reference: &Plane,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Mv16,
    out: &mut PlaneViewMut<'_>,
) {
    let filter = if on_hevc_grid(mv) {
        Filter::HevcChroma
    } else {
        Filter::VvcChroma
    };
    predict_with(dsp, filter, reference, (x, y), (w, h), mv, out);
}

/// Whether `mv` lies on the HEVC filters' phase grid (quarter luma,
/// eighth chroma samples): both are four phases of the VVC tables.
fn on_hevc_grid(mv: Mv16) -> bool {
    mv.x % HEVC_PHASE_STEP == 0 && mv.y % HEVC_PHASE_STEP == 0
}

/// Predict the `w`x`h` block at `(x, y)` of `reference` displaced by `mv`
/// with `filter`, whose positions set the vector's fractional bits (an
/// HEVC filter reads `mv` at its coarser phases).
pub(crate) fn predict_with(
    dsp: &Dsp,
    filter: Filter,
    reference: &Plane,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Mv16,
    out: &mut PlaneViewMut<'_>,
) {
    let ((wx, wy), (frac_x, frac_y)) = window_of(filter, (x, y), mv);
    let (ww, wh) = (w + filter.taps() - 1, h + filter.taps() - 1);
    let size = BlockSize::new(w, h).expect("prediction blocks are 4..64 a side");
    if let Some(window) = reference.region(wx as isize, wy as isize, ww, wh) {
        dsp.predict(filter, size, &window, frac_x, frac_y, out);
    } else {
        let samples = gather(reference, wx, wy, ww, wh);
        let window = PlaneView::new(&samples, ww, ww, wh).expect("gathered window");
        dsp.predict(filter, size, &window, frac_x, frac_y, out);
    }
}

/// Where the `filter` window of the block at `(x, y)` displaced by `mv`
/// starts in the reference (it may lie outside the picture), and the
/// vector's sub-sample parts in the filter's units (an HEVC filter reads
/// `mv` at its coarser phases).
fn window_of(filter: Filter, (x, y): (usize, usize), mv: Mv16) -> ((i32, i32), (usize, usize)) {
    let (frac_bits, phase_shift) = match filter {
        Filter::HevcLuma => (LUMA_FRAC_BITS, 2),
        Filter::HevcChroma => (CHROMA_FRAC_BITS, 2),
        Filter::VvcChroma => (CHROMA_FRAC_BITS, 0),
        _ => (LUMA_FRAC_BITS, 0),
    };
    let (int_x, frac_x) = split(mv.x, frac_bits);
    let (int_y, frac_y) = split(mv.y, frac_bits);
    let before = filter.before() as i32;
    (
        (x as i32 + int_x - before, y as i32 + int_y - before),
        (frac_x >> phase_shift, frac_y >> phase_shift),
    )
}

/// Most sub-blocks of one batch: a 64x64 CU's 4x4 luma sub-blocks.
pub(crate) const MAX_SUB_BLOCKS: usize = 256;

/// Predict the block `w` samples wide at `(x, y)` of `reference` in
/// `side`-sample sub-blocks, sub-block `i` (raster order) displaced by
/// `vectors[i]` with `filter`, into `out`: one
/// `Dsp::predict_sub_blocks` batch where every sub-block's window lies
/// inside the padded reference, else sub-block by sub-block with windows
/// clamped to the picture (§8.5.6.3.2).
pub(crate) fn predict_sub_blocks(
    dsp: &Dsp,
    filter: Filter,
    reference: &Plane,
    (x, y): (usize, usize),
    w: usize,
    side: usize,
    vectors: &[Mv16],
    out: &mut PlaneViewMut<'_>,
) {
    let per_row = w / side;
    let extra = filter.taps() - 1;
    let mut motion = [nc_dsp::SubBlockMotion::default(); MAX_SUB_BLOCKS];
    let mut origins = [(0i32, 0i32); MAX_SUB_BLOCKS];
    let (mut low, mut high) = ((i32::MAX, i32::MAX), (i32::MIN, i32::MIN));
    for (i, &mv) in vectors.iter().enumerate() {
        let at = (x + side * (i % per_row), y + side * (i / per_row));
        let (origin, (frac_x, frac_y)) = window_of(filter, at, mv);
        origins[i] = origin;
        motion[i].frac_x = frac_x as u8;
        motion[i].frac_y = frac_y as u8;
        low = (low.0.min(origin.0), low.1.min(origin.1));
        high = (high.0.max(origin.0), high.1.max(origin.1));
    }
    let span = |from: i32, to: i32| (to - from) as usize + side + extra;
    let window = reference.region(
        low.0 as isize,
        low.1 as isize,
        span(low.0, high.0),
        span(low.1, high.1),
    );
    let Some(window) = window else {
        let mut block = [0u8; 8 * 8];
        let block = &mut block[..side * side];
        for (i, &mv) in vectors.iter().enumerate() {
            let (col, row) = (side * (i % per_row), side * (i / per_row));
            predict_with(
                dsp,
                filter,
                reference,
                (x + col, y + row),
                (side, side),
                mv,
                &mut packed(block, (side, side)),
            );
            for (r, line) in block.chunks_exact(side).enumerate() {
                out.row_mut(row + r)[col..col + side].copy_from_slice(line);
            }
        }
        return;
    };
    for (m, origin) in motion.iter_mut().zip(&origins).take(vectors.len()) {
        m.x = (origin.0 - low.0) as u16;
        m.y = (origin.1 - low.1) as u16;
    }
    let size = BlockSize::new(side, side).expect("sub-blocks are 4 or 8 a side");
    dsp.predict_sub_blocks(
        filter,
        size,
        &window,
        &motion[..vectors.len()],
        per_row,
        out,
    );
}

/// Integer and fractional parts of a vector component with `frac_bits`
/// fractional bits.
fn split(v: i32, frac_bits: u32) -> (i32, usize) {
    (v >> frac_bits, (v & ((1 << frac_bits) - 1)) as usize)
}

/// The `w`x`h` reference window at `(x, y)` with coordinates clamped to the
/// coded picture edges (H.266 §8.5.6.3.2), through the shared
/// [`nc_dsp::emulate_edges`].
fn gather(reference: &Plane, x: i32, y: i32, w: usize, h: usize) -> Vec<u8> {
    let view = reference.view();
    let mut window = vec![0u8; w * h];
    nc_dsp::emulate_edges(
        (view.data(), view.stride()),
        (reference.width(), reference.height()),
        (x as isize, y as isize),
        (w, h),
        (&mut window, w),
    );
    window
}
