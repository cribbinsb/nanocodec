//! WebAssembly SIMD128 AV2 intra kernels: the generic code on 8 lanes (4
//! for 4×4 blocks and run remainders). This module only exists in builds
//! with SIMD128 enabled, so its kernels need no run-time check.

use crate::simd::simd128::{I16x4, I16x8};

super::runs::av2_intra_kernels!(; I16x4, I16x4, I16x8);
