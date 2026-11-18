//! VVC reconstruction kernels at any bit depth: intra prediction, the
//! residual add, the inverse transforms and LMCS, as [`DepthKernels`]
//! methods (`docs/handoff/high-bit-depth.md`).
//!
//! Each method has the name and argument order of the 8-bit [`crate::Dsp`]
//! method, with samples of the decoder's type `T`. Three routes:
//!
//! - kernels with 8-bit SIMD tiers (the square intra predictor, the
//!   residual add) route through [`VvcReconPixel`]: `u8` calls the 8-bit
//!   [`crate::Dsp`] method, `u16` validates as it does and calls the
//!   [`VvcReconFns`] table, whose entries are the 8-bit scalar references
//!   instantiated at `u16`;
//! - scalar-only kernels (rectangular, ISP, MRL, MIP and BDPCM prediction,
//!   LMCS) call their one generic reference directly, which the 8-bit
//!   method also calls (at `u8`, 8 bits);
//! - the inverse transforms, whose data are `i16` coefficients and
//!   residuals at every depth, take the depth: at 8 bits they are the
//!   8-bit methods' code.
//!
//! Reference samples of the predictors stay `i32` (their range,
//! `0..=(1 << BitDepth) − 1`, is checked against the depth); QPs reach
//! `63 + QpBdOffset` (`Qp′`, H.266 §7.4.3.4) in the transforms.
//! `vvc_intra_filter_references` needs no depth form: the filter decision
//! depends on the block shape and direction only (H.266 §8.4.5.2.8).

use nc_common::{BitDepth, PlaneViewMut, Sample};

use super::{DepthKernels, Pixel};
use crate::simd::HIGH_LOOKUP_TABLE;
use crate::vvc_intra::{self, VvcIntraMode, rectangular};
use crate::vvc_recon::{add_clipped, check_add_residual};
use crate::{Isa, VvcLfnst, VvcLmcs, VvcTransformKind};

/// The square intra predictor at a depth (`crate::vvc_intra::PredictFn`
/// with the depth).
pub(crate) type VvcIntra16Fn =
    unsafe fn(usize, VvcIntraMode, &[i32], &[i32], i32, bool, &mut [u16], BitDepth) -> bool;
/// The rectangular intra predictor at a depth.
pub(crate) type VvcIntraRect16Fn =
    unsafe fn(usize, usize, VvcIntraMode, &[i32], &[i32], i32, &mut [u16], BitDepth);
/// Add a packed residual to a packed prediction, clipping to the depth.
pub(crate) type VvcAddResidual16Fn = unsafe fn(&mut [u16], &[i16], BitDepth);
/// Map `u16` luma samples in place through an LMCS table.
pub(crate) type VvcLmcsMap16Fn = unsafe fn(&[i16; HIGH_LOOKUP_TABLE], &mut [u16]);

/// The `u16` VVC reconstruction kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VvcReconFns {
    pub intra: VvcIntra16Fn,
    pub intra_rect: VvcIntraRect16Fn,
    pub add_residual: VvcAddResidual16Fn,
    pub lmcs_map: VvcLmcsMap16Fn,
}

impl VvcReconFns {
    /// The scalar references at `u16`.
    pub(crate) const SCALAR: Self = Self {
        intra: vvc_intra::scalar_at::<u16>,
        intra_rect: rectangular::predict::<u16>,
        add_residual: add_clipped::<u16>,
        lmcs_map: crate::vvc_lmcs::map_u16_scalar,
    };

    pub(crate) fn for_isa(isa: Isa) -> Self {
        let mut fns = Self::SCALAR;
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => {
                fns.intra = vvc_intra::high_simd::avx2::predict;
                fns.intra_rect = vvc_intra::high_simd::avx2::predict_rect;
                fns.add_residual = crate::vvc_recon::high_simd::avx2::add_clipped;
                fns.lmcs_map = crate::vvc_lmcs::high_simd::map_avx2;
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                fns.intra = vvc_intra::high_simd::avx512::predict;
                fns.intra_rect = vvc_intra::high_simd::avx512::predict_rect;
                fns.add_residual = crate::vvc_recon::high_simd::avx512::add_clipped;
                fns.lmcs_map = crate::vvc_lmcs::high_simd::map_avx512;
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => {
                fns.intra = vvc_intra::high_simd::neon::predict;
                fns.intra_rect = vvc_intra::high_simd::neon::predict_rect;
                fns.add_residual = crate::vvc_recon::high_simd::neon::add_clipped;
                fns.lmcs_map = crate::vvc_lmcs::high_simd::map_neon;
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => {
                fns.intra = vvc_intra::high_simd::simd128::predict;
                fns.intra_rect = vvc_intra::high_simd::simd128::predict_rect;
                fns.add_residual = crate::vvc_recon::high_simd::simd128::add_clipped;
                fns.lmcs_map = crate::vvc_lmcs::high_simd::map_simd128;
            }
            _ => {}
        }
        fns
    }
}

/// The routing of the VVC kernels that have 8-bit SIMD tiers (a part of
/// [`Pixel`]): `u8` calls the [`crate::Dsp`] method, `u16` the table.
#[allow(clippy::too_many_arguments)]
pub trait VvcReconPixel: Sample {
    #[doc(hidden)]
    fn vvc_intra_square(
        kernels: DepthKernels<'_>,
        side: usize,
        mode: VvcIntraMode,
        edges: (&[i32], &[i32], i32),
        transposed_ok: bool,
        out: &mut [Self],
    ) -> bool;
    #[doc(hidden)]
    fn vvc_intra_predict_rect(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        mode: VvcIntraMode,
        edges: (&[i32], &[i32], i32),
        out: &mut [Self],
    );
    #[doc(hidden)]
    fn vvc_add_residual(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        prediction: &mut [Self],
        residual: &[i16],
    );
    #[doc(hidden)]
    fn vvc_forward_lmcs(kernels: DepthKernels<'_>, model: &VvcLmcs<Self>, samples: &mut [Self]);
    #[doc(hidden)]
    fn vvc_inverse_lmcs(
        kernels: DepthKernels<'_>,
        model: &VvcLmcs<Self>,
        plane: &mut PlaneViewMut<'_, Self>,
    );
}

impl VvcReconPixel for u8 {
    fn vvc_intra_square(
        kernels: DepthKernels<'_>,
        side: usize,
        mode: VvcIntraMode,
        (top, left, corner): (&[i32], &[i32], i32),
        transposed_ok: bool,
        out: &mut [u8],
    ) -> bool {
        if transposed_ok {
            kernels
                .dsp
                .vvc_intra_predict_oriented(side, mode, top, left, corner, out)
        } else {
            kernels
                .dsp
                .vvc_intra_predict(side, mode, top, left, corner, out);
            false
        }
    }

    fn vvc_intra_predict_rect(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        mode: VvcIntraMode,
        (top, left, corner): (&[i32], &[i32], i32),
        out: &mut [u8],
    ) {
        kernels
            .dsp
            .vvc_intra_predict_rect(size, mode, top, left, corner, out);
    }

    fn vvc_add_residual(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        prediction: &mut [u8],
        residual: &[i16],
    ) {
        kernels.dsp.vvc_add_residual(size, prediction, residual);
    }

    fn vvc_forward_lmcs(_kernels: DepthKernels<'_>, model: &VvcLmcs<u8>, samples: &mut [u8]) {
        model.forward_samples(samples);
    }

    fn vvc_inverse_lmcs(
        _kernels: DepthKernels<'_>,
        model: &VvcLmcs<u8>,
        plane: &mut PlaneViewMut<'_, u8>,
    ) {
        model.inverse_plane(plane);
    }
}

impl VvcReconPixel for u16 {
    fn vvc_intra_square(
        kernels: DepthKernels<'_>,
        side: usize,
        mode: VvcIntraMode,
        edges: (&[i32], &[i32], i32),
        transposed_ok: bool,
        out: &mut [u16],
    ) -> bool {
        vvc_intra::check_square(side, mode, edges, out.len(), kernels.depth);
        let (top, left, corner) = edges;
        // SAFETY: `check_square` checked the side, direction, every slice
        // length and the reference range; the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_recon.intra)(
                side,
                mode,
                top,
                left,
                corner,
                transposed_ok,
                out,
                kernels.depth,
            )
        }
    }

    fn vvc_intra_predict_rect(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        mode: VvcIntraMode,
        edges: (&[i32], &[i32], i32),
        out: &mut [u16],
    ) {
        let (top, left) = vvc_intra::check_rect(size, mode, edges, out.len(), kernels.depth);
        let corner = edges.2;
        if vvc_intra::square_kernel_fits(size) {
            // SAFETY: as for `vvc_intra_square` (`check_rect` covers the
            // square kernel's contract for these sizes).
            unsafe {
                (kernels.dsp.high.vvc_recon.intra)(
                    size.0,
                    mode,
                    top,
                    left,
                    corner,
                    false,
                    out,
                    kernels.depth,
                );
            }
        } else {
            // SAFETY: `check_rect` validated dimensions, mode, and slice bounds;
            // the table matches a supported ISA.
            unsafe {
                (kernels.dsp.high.vvc_recon.intra_rect)(
                    size.0,
                    size.1,
                    mode,
                    top,
                    left,
                    corner,
                    out,
                    kernels.depth,
                );
            }
        }
    }

    fn vvc_add_residual(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        prediction: &mut [u16],
        residual: &[i16],
    ) {
        let length = check_add_residual(size, prediction.len(), residual.len());
        // SAFETY: both slices are cut to the block (checked above); the
        // table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_recon.add_residual)(
                &mut prediction[..length],
                &residual[..length],
                kernels.depth,
            );
        }
    }

    fn vvc_forward_lmcs(kernels: DepthKernels<'_>, model: &VvcLmcs<u16>, samples: &mut [u16]) {
        if let Some(table) = model.forward_wide_table() {
            // SAFETY: `table` has `HIGH_LOOKUP_TABLE` entries; table matches a supported ISA.
            unsafe {
                (kernels.dsp.high.vvc_recon.lmcs_map)(table, samples);
            }
        } else {
            model.forward_samples(samples);
        }
    }

    fn vvc_inverse_lmcs(
        kernels: DepthKernels<'_>,
        model: &VvcLmcs<u16>,
        plane: &mut PlaneViewMut<'_, u16>,
    ) {
        if let Some(table) = model.inverse_wide_table() {
            for y in 0..plane.height() {
                // SAFETY: `table` has `HIGH_LOOKUP_TABLE` entries; table matches a supported ISA.
                unsafe {
                    (kernels.dsp.high.vvc_recon.lmcs_map)(table, plane.row_mut(y));
                }
            }
        } else {
            model.inverse_plane(plane);
        }
    }
}

impl DepthKernels<'_> {
    /// [`crate::Dsp::vvc_intra_predict`] at this depth: reference samples in
    /// `0..=(1 << BitDepth) − 1`, the prediction clipped to the depth.
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_intra_predict`], references outside the depth.
    pub fn vvc_intra_predict<T: Pixel>(
        self,
        side: usize,
        mode: VvcIntraMode,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [T],
    ) {
        T::vvc_intra_square(self, side, mode, (top, left, corner), false, out);
    }

    /// [`crate::Dsp::vvc_intra_predict_oriented`] at this depth (references as
    /// [`Self::vvc_intra_predict`]).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_intra_predict_oriented`].
    #[must_use]
    pub fn vvc_intra_predict_oriented<T: Pixel>(
        self,
        side: usize,
        mode: VvcIntraMode,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [T],
    ) -> bool {
        T::vvc_intra_square(self, side, mode, (top, left, corner), true, out)
    }

    /// [`crate::Dsp::vvc_intra_predict_rect`] at this depth (references as
    /// [`Self::vvc_intra_predict`]).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_intra_predict_rect`].
    pub fn vvc_intra_predict_rect<T: Pixel>(
        self,
        size: (usize, usize),
        mode: VvcIntraMode,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [T],
    ) {
        T::vvc_intra_predict_rect(self, size, mode, (top, left, corner), out);
    }

    /// [`crate::Dsp::vvc_intra_predict_isp`] at this depth.
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_intra_predict_isp`].
    #[allow(clippy::too_many_arguments)]
    pub fn vvc_intra_predict_isp<T: Pixel>(
        self,
        prediction_size: (usize, usize),
        size: (usize, usize),
        direction: u8,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [T],
    ) {
        rectangular::isp_predict(
            prediction_size,
            size,
            direction,
            (top, left, corner),
            out,
            self.depth,
        );
    }

    /// [`crate::Dsp::vvc_mrl_predict`] at this depth.
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_mrl_predict`].
    #[allow(clippy::too_many_arguments)]
    pub fn vvc_mrl_predict<T: Pixel>(
        self,
        size: (usize, usize),
        direction: u8,
        reference_line: u8,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [T],
    ) {
        rectangular::mrl_predict(
            size,
            direction,
            reference_line,
            (top, left, corner),
            out,
            self.depth,
        );
    }

    /// [`crate::Dsp::vvc_mip_predict`] at this depth: edge samples in `0..=(1 <<
    /// BitDepth) − 1`, the mid value and the clip of the depth (H.266
    /// §8.4.5.2.2).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_mip_predict`], edges outside the depth.
    pub fn vvc_mip_predict<T: Pixel>(
        self,
        size: (usize, usize),
        mode: u8,
        transpose: bool,
        top: &[i32],
        left: &[i32],
        out: &mut [T],
    ) {
        vvc_intra::mip_predict(size, mode, transpose, (top, left), out, self.depth);
    }

    /// [`crate::Dsp::vvc_bdpcm_predict`] at this depth.
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_bdpcm_predict`].
    pub fn vvc_bdpcm_predict<T: Pixel>(
        self,
        size: (usize, usize),
        vertical: bool,
        top: &[i32],
        left: &[i32],
        out: &mut [T],
    ) {
        vvc_intra::bdpcm_predict(size, vertical, (top, left), out, self.depth);
    }

    /// [`crate::Dsp::vvc_add_residual`] at this depth (`Clip1` to it).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_add_residual`].
    pub fn vvc_add_residual<T: Pixel>(
        self,
        size: (usize, usize),
        prediction: &mut [T],
        residual: &[i16],
    ) {
        T::vvc_add_residual(self, size, prediction, residual);
    }

    /// [`crate::Dsp::vvc_inverse_primary`] at this depth: `qp` is `Qp′` (up to
    /// `63 + QpBdOffset`); scaling `bdShift = BitDepth + … − 5` (H.266
    /// §8.7.3) and the second stage `20 − BitDepth` (§8.7.2).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_inverse_primary`], with the QP bound of the depth; if
    /// the depth exceeds 10 bits (Main 10).
    pub fn vvc_inverse_primary(
        self,
        size: (usize, usize),
        kinds: [VvcTransformKind; 2],
        levels: &[i32],
        qp: u8,
        dependent: bool,
        out: &mut [i16],
    ) {
        self.dsp
            .vvc_inverse_primary_at(size, kinds, levels, (qp, dependent), out, self.depth);
    }

    /// [`crate::Dsp::vvc_inverse_skip`] at this depth: only the QP range (`Qp′`,
    /// up to `63 + QpBdOffset`) depends on it (H.266 §8.7.3: `bdShift =
    /// 10`; §8.7.2: the residual is the scaled coefficients).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_inverse_skip`], with the QP bound of the depth; if the
    /// depth exceeds 10 bits.
    pub fn vvc_inverse_skip(
        self,
        size: (usize, usize),
        levels: &[i32],
        qp: u8,
        bdpcm: u8,
        out: &mut [i16],
    ) {
        self.dsp
            .vvc_inverse_skip_at(size, levels, qp, bdpcm, out, self.depth);
    }

    /// [`crate::Dsp::vvc_inverse_lfnst`] at this depth (QP and shifts as
    /// [`Self::vvc_inverse_primary`]).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_inverse_lfnst`], with the QP bound of the depth; if the
    /// depth exceeds 10 bits.
    pub fn vvc_inverse_lfnst(
        self,
        size: (usize, usize),
        transform: VvcLfnst,
        levels: &[i32],
        qp: u8,
        dependent: bool,
        out: &mut [i16],
    ) {
        self.dsp
            .vvc_inverse_lfnst_at(size, transform, levels, (qp, dependent), out, self.depth);
    }

    /// [`crate::Dsp::vvc_inverse_with_matrix`] at this depth (QP and shifts as
    /// [`Self::vvc_inverse_primary`]).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_inverse_with_matrix`], with the QP bound of the depth;
    /// if the depth exceeds 10 bits.
    #[allow(clippy::too_many_arguments)]
    pub fn vvc_inverse_with_matrix(
        self,
        size: (usize, usize),
        kinds: [VvcTransformKind; 2],
        lfnst: Option<VvcLfnst>,
        levels: &[i32],
        quantizer: (u8, bool),
        matrix: &[u8],
        out: &mut [i16],
    ) {
        crate::codec_txfm::vvc_inverse::inverse_with_matrix(
            self.dsp,
            size,
            (kinds, lfnst),
            levels,
            quantizer,
            matrix,
            out,
            self.depth,
        );
    }

    /// [`crate::Dsp::vvc_forward_lmcs`] with a model of this depth
    /// ([`VvcLmcs::new_at`]).
    ///
    /// # Panics
    /// If the model's depth is not this one.
    pub fn vvc_forward_lmcs<T: Pixel>(self, model: &VvcLmcs<T>, samples: &mut [T]) {
        assert_eq!(model.depth(), self.depth, "LMCS model of another depth");
        T::vvc_forward_lmcs(self, model, samples);
    }

    /// [`crate::Dsp::vvc_inverse_lmcs`] with a model of this depth.
    ///
    /// # Panics
    /// If the model's depth is not this one.
    pub fn vvc_inverse_lmcs<T: Pixel>(self, model: &VvcLmcs<T>, plane: &mut PlaneViewMut<'_, T>) {
        assert_eq!(model.depth(), self.depth, "LMCS model of another depth");
        T::vvc_inverse_lmcs(self, model, plane);
    }

    /// [`crate::Dsp::vvc_scale_chroma_residual`] at this (chroma) depth: the
    /// residual clips to `−(1 << BitDepth) ..= (1 << BitDepth) − 1` before
    /// scaling (H.266 §8.7.5.3).
    ///
    /// # Panics
    /// As [`crate::Dsp::vvc_scale_chroma_residual`].
    pub fn vvc_scale_chroma_residual(self, scale: i32, residual: &mut [i16]) {
        crate::vvc_lmcs::scale_chroma_residual(scale, residual, self.depth);
    }
}

#[cfg(test)]
#[path = "vvc_recon_tests.rs"]
mod tests;
