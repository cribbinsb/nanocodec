//! Block-level QP: the engine's per-block quantizer offsets
//! (`BlockDecision::qp_offset`) as `cu_qp_delta` (HEVC §7.3.8.14), one per
//! CTB quantization group (`crate::quant_group`).
//!
//! A CU is quantised, its λ taken and its residual tested at its block QP
//! by swapping that QP's [`FrameQuantization`] in for the frame's while the
//! CU is coded; each is built once per frame with the frame's settings
//! (rounding, λ multiple, planned-step headroom), so a block QP means
//! exactly what the same frame QP would. The CU's resulting `QpY` goes to
//! the grid, where deblocking and the syntax writer read it.

use std::mem;

use super::{FrameQuantization, HevcBackend, QP_MAX, QP_MIN};
use crate::quant_group::QuantGroup;

/// Quantizers a block may take: every HEVC QP.
const QP_COUNT: usize = QP_MAX as usize + 1;

/// What a [`FrameQuantization`] depends on besides its QP, for one frame.
#[derive(Debug, Clone, Copy)]
pub(super) struct QuantizationSettings {
    pub chroma_offset: i32,
    pub rounding: [(i32, i32); 2],
    pub lambda_scale: f32,
    pub column_zero_scale: f32,
    pub pre_transform_zero_scale: f32,
    pub headroom_step: f32,
}

impl QuantizationSettings {
    /// The quantisation at `qp`.
    pub fn at(&self, qp: i32) -> FrameQuantization {
        FrameQuantization::new(
            qp,
            self.chroma_offset,
            self.rounding,
            self.lambda_scale,
            self.column_zero_scale,
            self.pre_transform_zero_scale,
            self.headroom_step,
        )
    }
}

/// Block-QP state of one stream (present only once the engine enables
/// block QP).
#[derive(Debug)]
pub(super) struct BlockQp {
    group: QuantGroup,
    settings: Option<QuantizationSettings>,
    /// The current frame's quantisations by QP, built on first use.
    by_qp: Vec<Option<FrameQuantization>>,
    /// QP of the quantisation swapped in for the CU being coded.
    active: Option<i32>,
}

impl BlockQp {
    pub fn new() -> Self {
        Self {
            group: QuantGroup::new(0),
            settings: None,
            by_qp: std::iter::repeat_with(|| None).take(QP_COUNT).collect(),
            active: None,
        }
    }

    /// Start a slice at `slice_qp` with these quantisation settings.
    pub fn begin_frame(&mut self, slice_qp: i32, settings: QuantizationSettings) {
        self.group = QuantGroup::new(slice_qp);
        self.settings = Some(settings);
        self.by_qp.iter_mut().for_each(|q| *q = None);
    }

    /// Start the quantization group of the next CTB.
    pub fn begin_group(&mut self) {
        self.group.begin_group();
    }
}

impl HevcBackend {
    /// Quantise the next CU at the frame QP moved by `offset` native steps
    /// (or at its group's QP once that is coded); [`Self::end_block_qp`]
    /// restores the frame's quantisation.
    pub(super) fn begin_block_qp(&mut self, offset: i16) {
        let Some(block_qp) = self.block_qp.as_mut() else {
            return;
        };
        let wanted = (self.qp + i32::from(offset)).clamp(QP_MIN as i32, QP_MAX as i32);
        let qp = block_qp.group.quantizer(wanted);
        if qp == self.qp {
            return;
        }
        let settings = block_qp
            .settings
            .expect("begin_frame sets the quantisation settings");
        let quantization = block_qp.by_qp[qp as usize]
            .take()
            .unwrap_or_else(|| settings.at(qp));
        block_qp.by_qp[self.qp as usize] =
            Some(mem::replace(&mut self.frame_quantization, quantization));
        block_qp.active = Some(qp);
    }

    /// Whether the current CU is quantised finer than the frame (a block-QP
    /// refinement, `nc_encoder` `reuse.rs`).
    pub(super) fn refining_block(&self) -> bool {
        self.block_qp
            .as_ref()
            .and_then(|block_qp| block_qp.active)
            .is_some_and(|qp| qp < self.qp)
    }

    /// Put the frame's quantisation back after a CU.
    pub(super) fn end_block_qp(&mut self) {
        let Some(block_qp) = self.block_qp.as_mut() else {
            return;
        };
        if let Some(qp) = block_qp.active.take() {
            let frame = block_qp.by_qp[self.qp as usize]
                .take()
                .expect("the frame's quantisation is parked while a block QP is active");
            block_qp.by_qp[qp as usize] = Some(mem::replace(&mut self.frame_quantization, frame));
        }
    }

    /// `QpY` of a CU just coded at the active quantisation, with or without
    /// a coded residual.
    pub(super) fn cu_qp(&mut self, residual: bool) -> i8 {
        let quantizer = self.frame_quantization.planes[0].qp;
        let qp = match self.block_qp.as_mut() {
            Some(block_qp) => block_qp.group.finish_cu(quantizer, residual),
            None => quantizer,
        };
        qp as i8
    }
}
