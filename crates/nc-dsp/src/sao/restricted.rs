//! SAO edge offset of one CTB whose neighbours are restricted (H.265
//! §8.7.3.2, H.266 §8.8.4.2): a sample is left unchanged when a neighbour it
//! compares with lies in a CTB the filter may not read (across a slice,
//! tile or subpicture edge closed to the loop filters), beyond the picture,
//! or across a virtual boundary (H.266), and when it is exempt (PCM or
//! lossless, H.265). Scalar: only CTBs at such edges take this path; the
//! others use the vector kernels of [`crate::Dsp::sao_edge_apply`].
//! Shared by the HEVC and VVC decoders.

use nc_common::{BitDepth, PlaneView, PlaneViewMut, Sample};

use super::SAO_EO_NEIGHBOURS;

/// `edgeIdx` remapping of H.265 §8.7.3.2 / H.266 §8.8.4.2: 0, 1, 2 become
/// 1, 2, 0.
const EDGE_REMAP: [usize; 5] = [1, 2, 0, 3, 4];

/// One CTB of a plane, in the coordinates of the views it is filtered in.
#[derive(Debug, Clone, Copy)]
pub struct SaoCtb {
    /// The CTB's first column.
    pub x0: usize,
    /// The CTB's first row.
    pub y0: usize,
    /// The column after the CTB's last.
    pub x1: usize,
    /// The row after the CTB's last.
    pub y1: usize,
    /// The CTB size in samples of the plane.
    pub size: usize,
    /// The columns of the plane that may be read.
    pub width: usize,
    /// The rows of the plane (or of the views' window) that may be read.
    pub height: usize,
}

/// What restricts the neighbours a CTB's edge offsets compare with.
#[derive(Debug, Clone, Copy)]
pub struct SaoLimits<'a> {
    /// Which of the 3x3 CTBs around the CTB (row-major, `[1][1]` itself)
    /// may be read. Positions outside the plane are checked separately.
    pub ctbs: [[bool; 3]; 3],
    /// The columns, then the rows, at which a virtual boundary starts: a
    /// sample and a neighbour on opposite sides of one are not compared.
    pub cuts: [&'a [usize]; 2],
    /// Samples left unchanged, packed in CTB rows (`(y1 - y0) x (x1 - x0)`);
    /// empty when there are none.
    pub exempt: &'a [bool],
}

/// Apply edge offset class `class` with `offsets` (`SaoOffsetVal[1..=4]`,
/// already scaled to the bit depth) to CTB `ctb` of `dst`, classifying
/// from `src` (the deblocked samples, same coordinates), within `limits`;
/// results clip to `depth`.
///
/// # Panics
///
/// If the CTB lies outside either view or `exempt` has the wrong length.
pub fn sao_edge_restricted<T: Sample>(
    src: &PlaneView<'_, T>,
    dst: &mut PlaneViewMut<'_, T>,
    ctb: SaoCtb,
    (class, offsets, depth): (u8, [i16; 4], BitDepth),
    limits: &SaoLimits<'_>,
) {
    let SaoCtb {
        x0,
        y0,
        x1,
        y1,
        size,
        width,
        height,
    } = ctb;
    assert!(x1 <= width && y1 <= height && width <= src.width() && height <= src.height());
    assert!(x1 <= dst.width() && y1 <= dst.height());
    let exempt = limits.exempt;
    assert!(exempt.is_empty() || exempt.len() == (x1 - x0) * (y1 - y0));
    let [(ax, ay), (bx, by)] = SAO_EO_NEIGHBOURS[usize::from(class)];
    let region = |v: isize, start: usize| -> usize {
        if v < start as isize {
            0
        } else if v >= (start + size) as isize {
            2
        } else {
            1
        }
    };
    // Whether `a` and `b` lie on opposite sides of a cut.
    let split = |cuts: &[usize], a: isize, b: isize| {
        cuts.iter()
            .any(|&cut| (a < cut as isize) != (b < cut as isize))
    };
    let row_at = |v: isize| src.row(v.clamp(0, height as isize - 1) as usize);
    for y in y0..y1 {
        // Only samples on the CTB's border, or next to a cut, can have a
        // neighbour that may not be read.
        let border_row = y == y0 || y + 1 == y1;
        let (row_a, row, row_b) = (row_at(y as isize + ay), src.row(y), row_at(y as isize + by));
        let out = dst.row_mut(y);
        for x in x0..x1 {
            if !exempt.is_empty() && exempt[(y - y0) * (x1 - x0) + x - x0] {
                continue;
            }
            let (xa, xb) = (x as isize + ax, x as isize + bx);
            let readable = |xn: isize, yn: isize| {
                xn >= 0
                    && yn >= 0
                    && (xn as usize) < width
                    && (yn as usize) < height
                    && limits.ctbs[region(yn, y0)][region(xn, x0)]
                    && !split(limits.cuts[0], x as isize, xn)
                    && !split(limits.cuts[1], y as isize, yn)
            };
            let checked = border_row
                || x == x0
                || x + 1 == x1
                || !limits.cuts[0].is_empty()
                || !limits.cuts[1].is_empty();
            if checked && !(readable(xa, y as isize + ay) && readable(xb, y as isize + by)) {
                continue;
            }
            let s = row[x].to_i32();
            let a = row_a[xa as usize].to_i32();
            let b = row_b[xb as usize].to_i32();
            let edge = EDGE_REMAP[(2 + (s - a).signum() + (s - b).signum()) as usize];
            if edge != 0 {
                out[x] = depth.clip_to(s + i32::from(offsets[edge - 1]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dsp;

    fn plane() -> Vec<u8> {
        (0..32 * 32)
            .map(|i| ((i * 37 + i / 32 * 11) % 251) as u8)
            .collect()
    }

    #[test]
    fn unrestricted_interior_matches_the_vector_kernel() {
        let source = plane();
        let input = PlaneView::new(&source, 32, 32, 32).unwrap();
        for class in 0..4 {
            let (mut expected, mut actual) = (source.clone(), source.clone());
            let mut out = PlaneViewMut::new(&mut expected, 32, 32, 32).unwrap();
            Dsp::new().sao_edge_apply(class, [3, 1, -1, -3], &input, &mut out, (8, 8, 24, 24));
            let mut out = PlaneViewMut::new(&mut actual, 32, 32, 32).unwrap();
            let ctb = SaoCtb {
                x0: 8,
                y0: 8,
                x1: 24,
                y1: 24,
                size: 16,
                width: 32,
                height: 32,
            };
            let limits = SaoLimits {
                ctbs: [[true; 3]; 3],
                cuts: [&[], &[]],
                exempt: &[],
            };
            sao_edge_restricted(
                &input,
                &mut out,
                ctb,
                (class, [3, 1, -1, -3], BitDepth::EIGHT),
                &limits,
            );
            assert_eq!(expected, actual, "class {class}");
        }
    }

    #[test]
    fn samples_beside_a_cut_or_closed_ctb_keep_their_value() {
        let source = plane();
        let input = PlaneView::new(&source, 32, 32, 32).unwrap();
        let mut actual = source.clone();
        let mut out = PlaneViewMut::new(&mut actual, 32, 32, 32).unwrap();
        let ctb = SaoCtb {
            x0: 8,
            y0: 8,
            x1: 24,
            y1: 24,
            size: 16,
            width: 32,
            height: 32,
        };
        let mut ctbs = [[true; 3]; 3];
        ctbs[1][0] = false;
        let limits = SaoLimits {
            ctbs,
            cuts: [&[16], &[]],
            exempt: &[],
        };
        sao_edge_restricted(
            &input,
            &mut out,
            ctb,
            (0, [9, 9, -9, -9], BitDepth::EIGHT),
            &limits,
        );
        for y in 8..24 {
            for x in [8, 15, 16] {
                assert_eq!(actual[y * 32 + x], source[y * 32 + x], "({x}, {y})");
            }
        }
        assert!((8..24).any(|y| actual[y * 32 + 12] != source[y * 32 + 12]));
    }
}
