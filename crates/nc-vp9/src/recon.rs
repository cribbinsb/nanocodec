//! Block reconstruction shared by the encoder backend and the decoder:
//! transform-block traversal, prediction into the reconstruction, and
//! residual addition (VP9 §8.6.2, libvpx `vp9_decodeframe.c`
//! `predict_and_reconstruct_intra_block` / `reconstruct_inter_block`).
//!
//! Both sides call these with identical arguments, so the encoder's
//! reconstruction is the decoder's by construction.

use nc_common::{Picture, PlaneIndex, PlaneView};

use crate::coeff::Extent;
use crate::inter_pred::{self, BlockPosition};
use crate::intra_pred::{self, Edges};
use crate::transform::{self, Steps, TxSize, TxType};
use crate::types::{IntraMode, MI_SIZE, Mv};

/// One transform block of a plane.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TxBlock {
    pub plane: PlaneIndex,
    /// Top-left in plane samples.
    pub x: usize,
    pub y: usize,
    pub tx: TxSize,
    /// Row/column offsets inside the coded block, in plane samples (the
    /// intra edge availability uses them).
    pub offset_x: usize,
    pub offset_y: usize,
    /// Side of the coded block in plane samples.
    pub block_side: usize,
}

impl TxBlock {
    /// Edge availability: inner transform blocks always have the edges
    /// above and to the left; above-right lies inside the block unless the
    /// transform block is in its rightmost column (`have_right`).
    fn edges(&self, top: bool, left: bool) -> Edges {
        Edges {
            top: top || self.offset_y > 0,
            left: left || self.offset_x > 0,
            right: self.offset_x + self.tx.side() < self.block_side,
        }
    }
}

/// Transform blocks of `plane` in a square block at luma `(x, y)` of side
/// `size`, raster order.
pub(crate) fn tx_blocks(
    plane: PlaneIndex,
    x: usize,
    y: usize,
    size: usize,
    tx: TxSize,
) -> impl Iterator<Item = TxBlock> {
    let ss = usize::from(plane != PlaneIndex::Y);
    let (px, py, side) = (x >> ss, y >> ss, size >> ss);
    let step = tx.side();
    (0..side).step_by(step).flat_map(move |dy| {
        (0..side).step_by(step).map(move |dx| TxBlock {
            plane,
            x: px + dx,
            y: py + dy,
            tx,
            offset_x: dx,
            offset_y: dy,
            block_side: side,
        })
    })
}

/// Intra prediction of one transform block into `recon`. `top`/`left`
/// say whether the coded block has a neighbour above/left in the frame;
/// inner transform blocks always do.
pub(crate) fn predict_intra(
    dsp: &nc_dsp::Dsp,
    recon: &mut Picture,
    block: &TxBlock,
    mode: IntraMode,
    top: bool,
    left: bool,
) {
    let side = block.tx.side();
    let mut prediction = [0u8; 32 * 32];
    let edges = block.edges(top, left);
    intra_pred::predict(
        dsp,
        &recon.plane(block.plane).view(),
        (block.x, block.y),
        side,
        mode,
        edges,
        &mut prediction,
    );
    write_block(
        recon,
        block.plane,
        block.x,
        block.y,
        side,
        &prediction[..side * side],
    );
}

/// How the intra mode search measures a prediction against the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntraMetric {
    /// Squared error (paired with lambda per bit).
    Sse,
    /// Hadamard SATD (paired with sqrt(lambda) per bit): tracks the coded
    /// residual's cost better than its energy does.
    Satd,
}

/// `metric` between `source` and the intra prediction of one transform
/// block from `recon`'s current edges (mode decision; writes nothing).
pub(crate) fn intra_prediction_cost(
    dsp: &nc_dsp::Dsp,
    recon: &Picture,
    source: &Picture,
    block: &TxBlock,
    (mode, metric): (IntraMode, IntraMetric),
    top: bool,
    left: bool,
) -> u64 {
    let side = block.tx.side();
    let mut prediction = [0u8; 32 * 32];
    let edges = block.edges(top, left);
    intra_pred::predict(
        dsp,
        &recon.plane(block.plane).view(),
        (block.x, block.y),
        side,
        mode,
        edges,
        &mut prediction,
    );
    let (x, y) = (block.x as isize, block.y as isize);
    let src = source
        .plane(block.plane)
        .region(x, y, side, side)
        .expect("transform block inside the source");
    let pred = PlaneView::new(&prediction, side, side, side).expect("prediction covers the block");
    let size = nc_dsp::BlockSize::new(side, side).expect("square VP9 transform block");
    match metric {
        IntraMetric::Sse => u64::from(dsp.variance(size, &src, &pred).sse),
        IntraMetric::Satd => u64::from(dsp.satd(size, &src, &pred)),
    }
}

/// Scratch of inter prediction, reused across blocks (zeroing a fresh
/// buffer per block cost about 4% of an encode).
#[derive(Debug)]
pub(crate) struct InterScratch {
    pub window: Vec<u8>,
}

impl InterScratch {
    pub(crate) fn new() -> Self {
        Self {
            window: vec![0; inter_pred::WINDOW_SAMPLES],
        }
    }
}

/// Inter prediction of every plane of a block into `packed`, plane after
/// plane at the block's size (the layout of the encoder's saved
/// prediction, which it then copies into the reconstruction once).
/// `luma`, when given, is the block's luma prediction already made with
/// `filter` (the encoder's filter choice keeps it).
pub(crate) fn predict_inter(
    dsp: &nc_dsp::Dsp,
    scratch: &mut InterScratch,
    packed: &mut [u8],
    (reference, ref_size, position): (&Picture, (usize, usize), &BlockPosition),
    (mv, filter): (Mv, nc_dsp::Vp9Filter),
    luma: Option<&[u8]>,
) {
    let mut offset = 0;
    for plane in PlaneIndex::ALL {
        let ss = usize::from(plane != PlaneIndex::Y);
        let side = (position.size * MI_SIZE) >> ss;
        let out = &mut packed[offset..offset + side * side];
        offset += side * side;
        if let (PlaneIndex::Y, Some(luma)) = (plane, luma) {
            out.copy_from_slice(&luma[..side * side]);
            continue;
        }
        inter_pred::predict_with_ref_size(
            dsp,
            reference,
            ref_size,
            plane,
            position,
            mv,
            filter,
            &mut scratch.window,
            out,
        );
    }
}

fn write_block(
    recon: &mut Picture,
    plane: PlaneIndex,
    x: usize,
    y: usize,
    side: usize,
    samples: &[u8],
) {
    let mut view = recon.plane_mut(plane).view_mut();
    for (dy, row) in samples.chunks_exact(side).enumerate() {
        view.row_mut(y + dy)[x..x + side].copy_from_slice(row);
    }
}

/// The rounded mean of the source samples above and left of the `n`x`n`
/// block at plane `(x, y)`, 128 with neither (DC prediction from source
/// neighbours, for planning before reconstruction exists).
pub(crate) fn source_dc(plane: &nc_common::Plane, x: usize, y: usize, n: usize) -> u8 {
    let view = plane.view();
    let mut sum = 0u32;
    let mut count = 0u32;
    if y > 0 {
        sum += view.row(y - 1)[x..x + n]
            .iter()
            .map(|&s| u32::from(s))
            .sum::<u32>();
        count += n as u32;
    }
    if x > 0 {
        sum += (0..n)
            .map(|i| u32::from(view.row(y + i)[x - 1]))
            .sum::<u32>();
        count += n as u32;
    }
    (sum + count / 2)
        .checked_div(count)
        .map_or(128, |dc| dc as u8)
}

/// Reusable dequantization and inverse-transform buffers (reused rather
/// than zeroed per transform block: only the block's area is read).
#[derive(Debug, Clone)]
pub(crate) struct ResidualBuffers {
    dequantized: [i32; 32 * 32],
    residual: [i32; 32 * 32],
    /// The instance's kernels (the dispatched inverse transform).
    dsp: nc_dsp::Dsp,
}

impl ResidualBuffers {
    pub(crate) fn new(dsp: &nc_dsp::Dsp) -> Box<Self> {
        Box::new(Self {
            dequantized: [0; 32 * 32],
            residual: [0; 32 * 32],
            dsp: *dsp,
        })
    }

    /// The residual `levels` reconstruct to (dequantization and inverse),
    /// `tx.area()` samples.
    pub(crate) fn residual_of(
        &mut self,
        tx: TxSize,
        ty: TxType,
        lossless: bool,
        levels: &[i32],
        steps: Steps,
    ) -> &[i32] {
        let area = tx.area();
        for (index, (&level, value)) in levels[..area].iter().zip(&mut self.dequantized).enumerate()
        {
            *value = transform::dequantize_level(tx, index, level, steps);
        }
        transform::inverse(
            &self.dsp,
            tx,
            ty,
            lossless,
            &self.dequantized,
            &mut self.residual,
        );
        &self.residual[..area]
    }

    /// Dequantize `levels` (raster) and add the inverse transform to the
    /// prediction already in `recon`.
    pub(crate) fn add_residual(
        &mut self,
        recon: &mut Picture,
        block: &TxBlock,
        ty: TxType,
        lossless: bool,
        levels: &[i32],
        steps: Steps,
    ) {
        let area = block.tx.area();
        // Exact shortcut: a DC-only DCT block adds one constant (most coded
        // blocks at low rates), without dequantizing and inverting the
        // whole block.
        if dc_only_applies(block.tx, ty, lossless)
            && levels[1..area].iter().all(|&level| level == 0)
        {
            let dc = transform::dequantize_level(block.tx, 0, levels[0], steps);
            add_constant(recon, block, transform::dc_only_residual(block.tx, dc));
            return;
        }
        self.residual_of(block.tx, ty, lossless, levels, steps);
        add_block(recon, block, &self.residual[..area]);
    }

    /// The decoder's [`Self::add_residual`]: `coefficients` are already
    /// dequantized and `extent` bounds them (its end of block is at least
    /// 1). 8x8 and larger blocks invert and add in one fused kernel, which
    /// picks its DC-only or zero-corner variant from the extent.
    pub(crate) fn add_coefficients(
        &mut self,
        recon: &mut Picture,
        block: &TxBlock,
        (ty, lossless): (TxType, bool),
        coefficients: &[i32],
        extent: Extent,
    ) {
        let area = block.tx.area();
        if block.tx == TxSize::Tx4 || lossless {
            transform::inverse(
                &self.dsp,
                block.tx,
                ty,
                lossless,
                &coefficients[..area],
                &mut self.residual,
            );
            add_block(recon, block, &self.residual[..area]);
            return;
        }
        let mut view = recon.plane_mut(block.plane).view_mut();
        let stride = view.stride();
        let dst = &mut view.data_mut()[block.y * stride + block.x..];
        self.dsp.vp9_inverse_add(
            block.tx.log2(),
            transform::passes(ty),
            &coefficients[..area],
            (extent.rows, extent.columns),
            (dst, stride),
        );
    }
}

/// Whether a DC-only block reconstructs as one constant
/// ([`transform::dc_only_residual`]): 8x8 and larger DCT blocks.
fn dc_only_applies(tx: TxSize, ty: TxType, lossless: bool) -> bool {
    ty == TxType::DctDct && !lossless && tx != TxSize::Tx4
}

/// Add `value` to the prediction under `block`, clipping to 8 bits.
fn add_constant(recon: &mut Picture, block: &TxBlock, value: i32) {
    let side = block.tx.side();
    let mut view = recon.plane_mut(block.plane).view_mut();
    for dy in 0..side {
        for sample in &mut view.row_mut(block.y + dy)[block.x..block.x + side] {
            *sample = (i32::from(*sample) + value).clamp(0, 255) as u8;
        }
    }
}

/// Add a raster `residual` to the prediction under `block`, clipping.
fn add_block(recon: &mut Picture, block: &TxBlock, residual: &[i32]) {
    let side = block.tx.side();
    let mut view = recon.plane_mut(block.plane).view_mut();
    for (dy, row) in residual.chunks_exact(side).enumerate() {
        let samples = &mut view.row_mut(block.y + dy)[block.x..block.x + side];
        for (sample, &r) in samples.iter_mut().zip(row) {
            *sample = (i32::from(*sample) + r).clamp(0, 255) as u8;
        }
    }
}

/// `get_tx_type` / `intra_mode_to_tx_type_lookup`: luma of intra blocks
/// follows the mode (32x32 has only the DCT);
/// chroma, inter blocks, 32x32 and lossless use the DCT (WHT).
pub(crate) fn tx_type(
    plane: PlaneIndex,
    intra_mode: Option<IntraMode>,
    tx: TxSize,
    lossless: bool,
) -> TxType {
    match intra_mode {
        Some(mode) if plane == PlaneIndex::Y && tx != TxSize::Tx32 && !lossless => match mode {
            IntraMode::V | IntraMode::D117 | IntraMode::D63 => TxType::AdstDct,
            IntraMode::H | IntraMode::D153 | IntraMode::D207 => TxType::DctAdst,
            IntraMode::Tm | IntraMode::D135 => TxType::AdstAdst,
            IntraMode::Dc | IntraMode::D45 => TxType::DctDct,
        },
        _ => TxType::DctDct,
    }
}
