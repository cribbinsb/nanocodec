//! General inter CABAC banks, including B/P initialization switching.
//! Normative tables from BSD-Clear `VVdeC`; see `THIRD_PARTY.md`.
use super::{Context, initialization::bank};
pub(super) struct Models {
    pub direction: [Context; 6],
    pub affine: [Context; 3],
    pub affine_type: Context,
    pub subblock: [Context; 3],
    pub subblock_index: Context,
    pub regular: [Context; 2],
    pub mmvd: Context,
    pub mmvd_base: Context,
    pub mmvd_step: Context,
    pub ciip: Context,
    pub symmetric: Context,
    pub precision: [Context; 5],
    pub weight: Context,
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            direction: bank(
                [[14, 13, 5, 4, 3, 40], [7, 6, 5, 12, 4, 40], [35; 6]],
                [0, 0, 1, 4, 4, 0],
                qp,
                kind,
            ),
            affine: bank([[19, 13, 6], [12, 13, 14], [35; 3]], [4, 0, 0], qp, kind),
            affine_type: Context::new(35, 4, qp),
            subblock: bank([[25, 58, 45], [48, 57, 44], [35; 3]], [4, 4, 4], qp, kind),
            subblock_index: bank::<1, 1>([[4], [5], [35]], [0], qp, kind)[0],
            regular: bank([[46, 15], [38, 7], [35; 2]], [5, 5], qp, kind),
            mmvd: bank::<1, 1>([[25], [26], [35]], [4], qp, kind)[0],
            mmvd_base: Context::new(43, 10, qp),
            mmvd_step: bank::<1, 1>([[59], [60], [35]], [0], qp, kind)[0],
            ciip: Context::new(57, 1, qp),
            symmetric: Context::new(28, 5, qp),
            precision: bank(
                [
                    [59, 26, 50, 60, 38],
                    [59, 48, 58, 60, 60],
                    [35, 34, 35, 35, 35],
                ],
                [0, 5, 0, 0, 4],
                qp,
                kind,
            ),
            weight: bank::<1, 1>([[5], [4], [35]], [1], qp, kind)[0],
        }
    }
}
