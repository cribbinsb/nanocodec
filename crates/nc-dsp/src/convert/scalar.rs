//! Scalar reference implementations of the conversion kernels.

use super::{ALPHA_OPAQUE, ConvertFns, bt601, bt601_forward};

/// The scalar kernel table.
pub(crate) fn convert_fns() -> ConvertFns {
    ConvertFns {
        deinterleave_uv,
        i420_to_rgba_rows,
        rgba_to_i420,
        sse_row,
    }
}

/// Sum of squared differences of two equally long rows (kernel contract).
pub(crate) fn sse_row(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u32::from(x.abs_diff(y)).pow(2))
        .sum()
}

#[allow(clippy::too_many_arguments, reason = "matches DeinterleaveUvFn")]
fn deinterleave_uv(
    src: &[u8],
    src_stride: usize,
    dst_u: &mut [u8],
    u_stride: usize,
    dst_v: &mut [u8],
    v_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        deinterleave_uv_row(
            &src[y * src_stride..y * src_stride + 2 * w],
            &mut dst_u[y * u_stride..y * u_stride + w],
            &mut dst_v[y * v_stride..y * v_stride + w],
        );
    }
}

/// One row: `src` holds `u.len()` (Cb, Cr) pairs. SIMD versions use it for
/// the tail columns that do not fill a register.
pub(crate) fn deinterleave_uv_row(src: &[u8], cb_row: &mut [u8], cr_row: &mut [u8]) {
    let pairs = src.as_chunks::<2>().0;
    for ((&[cb, cr], cb_out), cr_out) in pairs.iter().zip(cb_row).zip(cr_row) {
        *cb_out = cb;
        *cr_out = cr;
    }
}

/// One or two display rows (kernel contract in the `convert` module docs).
fn i420_to_rgba_rows(luma: [&[u8]; 2], cb: &[u8], cr: &[u8], rgba: [&mut [u8]; 2]) {
    for (luma, rgba) in luma.into_iter().zip(rgba) {
        i420_to_rgba_row(luma, cb, cr, rgba);
    }
}

/// One display row (`convert` module docs). SIMD versions use it for the
/// tail columns that do not fill a register.
pub(crate) fn i420_to_rgba_row(luma: &[u8], cb: &[u8], cr: &[u8], rgba: &mut [u8]) {
    let pixels = rgba.as_chunks_mut::<4>().0;
    for (x, (out, &l)) in pixels.iter_mut().zip(luma).enumerate() {
        let y = bt601::Y * (i32::from(l) - bt601::LUMA_OFFSET) + bt601::ROUND;
        let u = i32::from(cb[x / 2]) - bt601::CHROMA_OFFSET;
        let v = i32::from(cr[x / 2]) - bt601::CHROMA_OFFSET;
        let channel = |value: i32| (value >> bt601::SHIFT).clamp(0, 255) as u8;
        *out = [
            channel(y + bt601::R_CR * v),
            channel(y - bt601::G_CB * u - bt601::G_CR * v),
            channel(y + bt601::B_CB * u),
            ALPHA_OPAQUE,
        ];
    }
}

/// The encoder input conversion (`convert` module docs).
#[allow(clippy::too_many_arguments, reason = "matches RgbaToI420Fn")]
pub(crate) fn rgba_to_i420(
    rgba: &[u8],
    luma: &mut [u8],
    luma_stride: usize,
    cb: &mut [u8],
    cr: &mut [u8],
    chroma_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgba_to_luma_row(
            &rgba[4 * w * y..][..4 * w],
            &mut luma[luma_stride * y..][..w],
        );
    }
    for y in 0..h / 2 {
        let rows = &rgba[4 * w * 2 * y..][..8 * w];
        rgba_to_chroma_row(
            rows,
            0,
            &mut cb[chroma_stride * y..][..w / 2],
            &mut cr[chroma_stride * y..][..w / 2],
        );
    }
}

/// Luma of one packed RGBA row. SIMD versions use it for tail columns.
pub(crate) fn rgba_to_luma_row(rgba: &[u8], luma: &mut [u8]) {
    for (pixel, out) in rgba.as_chunks::<4>().0.iter().zip(luma) {
        *out = forward(
            bt601_forward::Y,
            bt601_forward::LUMA_OFFSET,
            pixel_rgb(*pixel),
        );
    }
}

/// Chroma of `cb.len()` 2x2 groups from two packed RGBA rows (`rows`: the
/// top row then the bottom one, each `rows.len() / 2` bytes), starting at
/// group `first`. SIMD versions use it for tail columns.
pub(crate) fn rgba_to_chroma_row(rows: &[u8], first: usize, cb: &mut [u8], cr: &mut [u8]) {
    let (top, bottom) = rows.split_at(rows.len() / 2);
    for (x, (cb, cr)) in cb.iter_mut().zip(cr).enumerate() {
        let group = 8 * (first + x);
        let mut sum = [0i32; 3];
        for row in [top, bottom] {
            for pixel in row[group..group + 8].as_chunks::<4>().0 {
                for (total, value) in sum.iter_mut().zip(pixel_rgb(*pixel)) {
                    *total += value;
                }
            }
        }
        let mean = sum.map(|v| (v + 2) / 4);
        *cb = forward(bt601_forward::CB, bt601_forward::CHROMA_OFFSET, mean);
        *cr = forward(bt601_forward::CR, bt601_forward::CHROMA_OFFSET, mean);
    }
}

fn pixel_rgb(pixel: [u8; 4]) -> [i32; 3] {
    [0, 1, 2].map(|c| i32::from(pixel[c]))
}

fn forward(weights: [i32; 3], offset: i32, rgb: [i32; 3]) -> u8 {
    let sum: i32 = weights.iter().zip(rgb).map(|(w, v)| w * v).sum();
    (offset + ((sum + bt601_forward::ROUND) >> bt601_forward::SHIFT)).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::deinterleave_uv;

    #[test]
    fn splits_pairs_and_respects_strides() {
        // Two rows of two pairs, source stride 5 (one padding byte).
        let src = [1u8, 2, 3, 4, 99, 5, 6, 7, 8];
        let (mut u, mut v) = ([0u8; 6], [0u8; 4]);
        deinterleave_uv(&src, 5, &mut u, 3, &mut v, 2, 2, 2);
        assert_eq!(u, [1, 3, 0, 5, 7, 0]);
        assert_eq!(v, [2, 4, 6, 8]);
    }
}
