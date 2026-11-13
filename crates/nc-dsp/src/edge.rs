//! Edge emulation for motion compensation: a reference window that leaves
//! the picture, rebuilt in a small buffer with every coordinate clamped to
//! the nearest visible sample (libvpx `extend_and_predict`, dav1d
//! `emu_edge`; reimplemented, BSD-3-Clause / BSD-2-Clause, see
//! `THIRD_PARTY.md`).
//!
//! Codecs whose references have a border read most windows in place and
//! call this only for windows that leave the border. Each output row is
//! one fill, one copy and one fill, which the standard library lowers to
//! vector stores; no codec-specific kernel is needed.

/// Copy the `width` x `height` window whose top-left sample is at
/// (`left`, `top`) of a plane whose visible size is `visible` into `dst`
/// (rows `dst_stride` apart), reading every sample at its coordinates
/// clamped into the visible area. The plane's sample (x, y) is
/// `src[y * stride + x]`.
///
/// # Panics
///
/// If the visible area is empty or not inside `src`, or `dst` is too small.
pub fn emulate_edges(
    (src, stride): (&[u8], usize),
    visible: (usize, usize),
    (left, top): (isize, isize),
    (width, height): (usize, usize),
    (dst, dst_stride): (&mut [u8], usize),
) {
    let (visible_w, visible_h) = (visible.0 as isize, visible.1 as isize);
    assert!(visible_w > 0 && visible_h > 0 && dst_stride >= width);
    // Columns of the window left of, inside and right of the picture.
    let before = (-left).clamp(0, width as isize) as usize;
    let first = left.clamp(0, visible_w) as usize;
    let end = (left + width as isize).clamp(0, visible_w) as usize;
    let inside = end - first;
    for (y, line) in dst.chunks_mut(dst_stride).take(height).enumerate() {
        let row = (top + y as isize).clamp(0, visible_h - 1) as usize * stride;
        let source = &src[row..row + visible.0];
        let (head, rest) = line[..width].split_at_mut(before);
        head.fill(source[0]);
        let (middle, tail) = rest.split_at_mut(inside);
        middle.copy_from_slice(&source[first..end]);
        tail.fill(source[visible.0 - 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_sample_reads_its_clamped_coordinate() {
        let (w, h, stride) = (5usize, 4usize, 7usize);
        let src: Vec<u8> = (0..stride * h).map(|i| i as u8).collect();
        for (left, top) in [(-3, -2), (2, 1), (3, 3), (-9, 6), (0, 0)] {
            let mut dst = [0u8; 6 * 4];
            emulate_edges((&src, stride), (w, h), (left, top), (6, 4), (&mut dst, 6));
            for y in 0..4 {
                for x in 0..6 {
                    let cx = (left + x as isize).clamp(0, w as isize - 1) as usize;
                    let cy = (top + y as isize).clamp(0, h as isize - 1) as usize;
                    assert_eq!(
                        dst[y * 6 + x],
                        src[cy * stride + cx],
                        "{left},{top} {x},{y}"
                    );
                }
            }
        }
    }
}
