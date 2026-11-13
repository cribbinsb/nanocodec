//! The border-and-rows VP9 intra predictors compiled for AVX2.

use super::{Vp9IntraEdges, Vp9IntraMode};

#[target_feature(enable = "avx2")]
pub(super) unsafe fn predict(
    mode: Vp9IntraMode,
    side: usize,
    edges: &Vp9IntraEdges,
    out: &mut [u8],
) {
    super::rows::predict(mode, side, edges, out);
}
