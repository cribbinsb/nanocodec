//! AVX2 AV2 intra kernels: the generic code on 16 lanes (8 and 4 for small
//! blocks and run remainders).

use crate::simd::avx2::{I16x4, I16x8, I16x16};

super::runs::av2_intra_kernels!(#[target_feature(enable = "avx2")]; I16x4, I16x8, I16x16);
