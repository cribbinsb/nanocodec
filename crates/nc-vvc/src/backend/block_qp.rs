//! Block-level QP: the engine's per-block quantizer offsets
//! (`BlockDecision::qp_offset`) as `cu_qp_delta` (H.266 §7.3.11.10), one per
//! CTB quantization group (`crate::quant_group`).
//!
//! While a CU is coded the backend's QP and λ are those of its block QP
//! (λ follows the squared step, doubling every 3 QP, so the frame's planned
//! `lambda_scale` carries over); every quantiser, level search and intra or
//! motion choice of the CU reads them. The CU's resulting `QpY` goes to the
//! [`QpMap`], which deblocking reads, and the delta it carries, if any, to
//! the syntax writer.

use super::VvcBackend;
use crate::headers;
use crate::quant_group::{QpMap, QuantGroup, row_start_prediction};

/// λ doubles every this many QP (the squared step, §8.7.3).
const QP_PER_LAMBDA_OCTAVE: f32 = 3.;

/// Block-QP state of one stream (present only once the engine enables
/// block QP).
#[derive(Debug)]
pub(super) struct BlockQp {
    group: QuantGroup,
    /// `QpY` of every CU of the current picture.
    pub(super) map: QpMap,
    /// The picture's QP and λ, restored after each CU.
    frame: (u32, f32),
}

impl BlockQp {
    pub(super) fn new() -> Self {
        Self {
            group: QuantGroup::new(0),
            map: QpMap::default(),
            frame: (0, 0.),
        }
    }
}

impl VvcBackend {
    /// Start the picture's quantization groups at the slice QP and λ.
    pub(super) fn begin_block_qp_frame(&mut self) {
        let (qp, lambda) = (self.qp, self.lambda);
        let (width, height) = (self.geometry.coded_width, self.geometry.coded_height);
        if let Some(block_qp) = self.block_qp.as_mut() {
            block_qp.group = QuantGroup::new(qp);
            block_qp.map.reset(width, height, qp);
            block_qp.frame = (qp, lambda);
        }
    }

    /// Start the quantization group of the CTB at luma (`x`, `y`).
    pub(super) fn begin_quant_group(&mut self, x: usize, y: usize) {
        if let Some(block_qp) = self.block_qp.as_mut() {
            let above = row_start_prediction(&block_qp.map, x, y);
            block_qp.group.begin_group(above);
        }
    }

    /// Code the next CU at the frame QP moved by `offset` (or at its
    /// group's QP once that is coded); [`Self::end_block_qp`] restores the
    /// frame's QP and λ.
    pub(super) fn begin_block_qp(&mut self, offset: i16) {
        let Some(block_qp) = self.block_qp.as_ref() else {
            return;
        };
        let (frame_qp, frame_lambda) = block_qp.frame;
        let wanted = (i64::from(frame_qp) + i64::from(offset)).clamp(0, i64::from(headers::MAX_QP));
        let qp = block_qp.group.quantizer(wanted as u32);
        self.qp = qp;
        self.lambda =
            frame_lambda * 2f32.powf((qp as f32 - frame_qp as f32) / QP_PER_LAMBDA_OCTAVE);
    }

    /// Finish CU `cu`, with a coded residual or not: record its `QpY`,
    /// restore the frame's QP and λ, and return the `CuQpDeltaVal` it
    /// carries, if any.
    pub(super) fn end_block_qp(&mut self, cu: &nc_encoder::Block, residual: bool) -> Option<i32> {
        let quantizer = self.qp;
        let block_qp = self.block_qp.as_mut()?;
        let (qp_y, delta) = block_qp.group.finish_cu(quantizer, residual);
        block_qp.map.record(cu, qp_y);
        (self.qp, self.lambda) = block_qp.frame;
        delta
    }
}
