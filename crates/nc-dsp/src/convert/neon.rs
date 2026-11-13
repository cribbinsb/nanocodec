//! NEON conversion kernels (`AArch64`).
//!
//! Deinterleave: `vld2q_u8` splits 16 (Cb, Cr) pairs into two registers in
//! one load. The scalar row function handles the tail. (On the Orin, LLVM
//! auto-vectorizes the scalar loop into the same instructions: both take
//! about 205 ns per 64x32 block; this kernel makes that independent of the
//! optimizer.)
//!
//! Kernel contract: see the `convert` module docs.

use core::arch::aarch64::{vld2q_u8, vst1q_u8};

use super::{ConvertFns, scalar};

/// Pairs split per iteration of the deinterleave loop.
const DEINTERLEAVE_STEP: usize = 16;

/// The NEON kernel table. NEON is always available on `AArch64`.
pub(crate) fn convert_fns() -> ConvertFns {
    ConvertFns {
        deinterleave_uv,
        i420_to_rgba_rows: super::to_rgba_neon::i420_to_rgba_rows,
        rgba_to_i420: if std::arch::is_aarch64_feature_detected!("dotprod") {
            super::rgba_neon::rgba_to_i420_dotprod
        } else {
            super::rgba_neon::rgba_to_i420
        },
        sse_row: scalar::sse_row,
    }
}

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[allow(clippy::too_many_arguments, reason = "matches DeinterleaveUvFn")]
#[target_feature(enable = "neon")]
unsafe fn deinterleave_uv(
    src: &[u8],
    src_stride: usize,
    dst_u: &mut [u8],
    u_stride: usize,
    dst_v: &mut [u8],
    v_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DEINTERLEAVE_STEP;
    for y in 0..h {
        let pairs = &src[y * src_stride..y * src_stride + 2 * w];
        let cb_row = &mut dst_u[y * u_stride..y * u_stride + w];
        let cr_row = &mut dst_v[y * v_stride..y * v_stride + w];
        for x in (0..simd_width).step_by(DEINTERLEAVE_STEP) {
            // SAFETY: source bytes 2x .. 2x + 32 lie inside the 2w-byte row
            // and outputs x .. x + 16 inside the w-sample rows, since
            // x + 16 <= w (kernel contract).
            unsafe {
                let split = vld2q_u8(pairs.as_ptr().add(2 * x));
                vst1q_u8(cb_row.as_mut_ptr().add(x), split.0);
                vst1q_u8(cr_row.as_mut_ptr().add(x), split.1);
            }
        }
        scalar::deinterleave_uv_row(
            &pairs[2 * simd_width..],
            &mut cb_row[simd_width..],
            &mut cr_row[simd_width..],
        );
    }
}
