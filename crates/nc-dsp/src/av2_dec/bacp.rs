//! The mask of boundary-aware compound prediction (BACP) in the general
//! AV2 decoder: where one of a compound block's two predictions reads
//! outside its reference frame, the other takes its weight
//! ([`bacp_mask`]).
//!
//! Ported from dav2d `src/bacp.c` (`bacp_mask_c`; commit 237229e,
//! Copyright © 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see
//! `docs/licenses/dav2d`).

/// Half of a mask's full weight (6-bit weights): the weight when both or
/// neither prediction is inside the frame.
const HALF_WEIGHT: u8 = 32;

/// Write the `bw` x `bh` blend mask (weights of the first prediction,
/// rows `stride` apart) of a block whose predictions start at (`x0`,
/// `y0`) and (`x1`, `y1`) in a `fw` x `fh` reference: 64 where only the
/// first is inside the frame, 0 where only the second is, 32 otherwise
/// (dav2d `bacp_mask_c`).
///
/// # Panics
///
/// If `mask` is too short.
pub fn bacp_mask(
    mask: &mut [u8],
    stride: usize,
    (bw, bh): (usize, usize),
    (x0, y0): (i32, i32),
    (x1, y1): (i32, i32),
    (fw, fh): (u32, u32),
) {
    let inside = |x: i32, y: i32| u8::from((x as u32) < fw && (y as u32) < fh);
    for (y, row) in mask.chunks_mut(stride).take(bh).enumerate() {
        let y = y as i32;
        for (x, weight) in row[..bw].iter_mut().enumerate() {
            let x = x as i32;
            let first = inside(x0 + x, y0 + y);
            let second = inside(x1 + x, y1 + y);
            *weight = HALF_WEIGHT * (1 + first - second);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block straddling the left frame edge with the second prediction
    /// inside: the first's outside columns weigh 0.
    #[test]
    fn edge_columns() {
        let mut mask = [0_u8; 4 * 2];
        bacp_mask(&mut mask, 4, (4, 2), (-2, 0), (0, 0), (16, 16));
        assert_eq!(mask, [0, 0, 32, 32, 0, 0, 32, 32]);
        bacp_mask(&mut mask, 4, (4, 2), (0, 0), (14, 0), (16, 16));
        assert_eq!(mask, [32, 32, 64, 64, 32, 32, 64, 64]);
    }
}
