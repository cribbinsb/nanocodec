//! The border-and-rows VP9 intra predictors compiled for SIMD128 (the
//! whole build enables it).

use super::{Vp9IntraEdges, Vp9IntraMode};

pub(super) unsafe fn predict(
    mode: Vp9IntraMode,
    side: usize,
    edges: &Vp9IntraEdges,
    out: &mut [u8],
) {
    super::rows::predict(mode, side, edges, out);
}
