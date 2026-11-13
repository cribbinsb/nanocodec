//! NEON kernels: the generic code on 4 lanes.

use crate::simd::neon::I32x4;

super::generic::av2_txfm_kernels!(#[target_feature(enable = "neon")]; I32x4, I32x4);
super::typed_lanes::typed_fns!(av2_typed_fns, #[target_feature(enable = "neon")]; I32x4, I32x4);
