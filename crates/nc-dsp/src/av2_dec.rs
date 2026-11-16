//! Scalar kernels of the general AV2 decoder (`nc_av2::general`): the
//! inverse transforms of every size and type with the secondary and
//! cross-chroma transforms ([`itx`], [`stx`]), every intra predictor
//! with its edge preparation ([`ipred`]), motion compensation (bilinear
//! [`mc`], the switchable filters and extended warp [`mc_8tap`], the
//! affine [`warp`], compound blends [`compound`], boundary-aware compound
//! [`bacp`], decoder-side refinement [`opfl`], wedge and inter-intra
//! masks [`wedge`]), and the guided detail filter ([`gdf`]), for 8-bit
//! 4:2:0; the inter kernels are generic over the sample type.
//!
//! Ported from dav2d (commit 237229e; Copyright © 2018-2026 `VideoLAN` and
//! dav2d authors, BSD-2-Clause, see `docs/licenses/dav2d`), which is
//! bit-exact with AVM v1.0.0, and reimplemented in safe Rust with dav2d's
//! function contracts (pointer + stride become slice + offset + stride).
//! Our own streams' subset keeps its own SIMD kernels ([`crate::Dsp`]); these
//! serve only the paths other encoders' streams take, so they are scalar.

pub mod bacp;
pub mod compound;
pub mod gdf;
pub mod gdf_tables;
pub mod ipred;
pub mod itx;
mod itx_1d;
mod itx_kernels;
pub mod mc;
pub mod mc_8tap;
mod mc_tables;
pub mod opfl;
pub mod stx;
pub mod stx_tables;
pub mod warp;
pub mod wedge;
pub mod wiener;
pub mod wiener_tables;
