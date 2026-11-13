//! Scalar reference kernels: the generic code on one `i32` lane.

#![allow(clippy::inline_always)]

super::generic::codec_txfm_kernels!(; i32, i32, i32);
