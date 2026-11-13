//! WebAssembly SIMD128 kernels: the generic code on 4 lanes. This module
//! only exists in builds with SIMD128 enabled, so its kernels need no
//! run-time check.

use crate::simd::simd128::I32x4;

super::generic::av2_txfm_kernels!(; I32x4, I32x4);
super::typed_lanes::typed_fns!(av2_typed_fns, ; I32x4, I32x4);
