//! Exhaustive strength/direction checks on minimum-size interior CDEF windows.
//! Odd strides, byte offsets and exact final-row lengths exercise short row
//! loads; full-buffer comparisons also protect padding and untouched pixels.

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{CDEF_PADDING, CdefSource, Dsp};

use crate::check::CheckReport;
use crate::check_codec::record;

const GUARD: u8 = 0xA5;
const PATTERNS: usize = 6;
const DAMPING: [u8; 4] = [0, 2, 4, 8];

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for side in [4, 8] {
        let extent = side + 2 * CDEF_PADDING;
        for extra_stride in [0, 3] {
            let stride = extent + extra_stride;
            let len = (extent - 1) * stride + extent;
            for pattern in 0..PATTERNS {
                let mut input = vec![GUARD; len + 2];
                for row in 0..extent {
                    for col in 0..extent {
                        input[1 + row * stride + col] = pixel(row, col, pattern);
                    }
                }
                let src = PlaneView::new(&input[1..=len], stride, extent, extent).expect("source");
                let mut padded = CdefSource::new();
                padded.fill_from(&src);
                for primary in 0..=15 {
                    for secondary in [0, 1, 2, 4] {
                        for direction in 0..8 {
                            for damping in DAMPING {
                                let mut actual = vec![GUARD; len + 2];
                                let mut expected = actual.clone();
                                let at = (CDEF_PADDING, CDEF_PADDING);
                                let strengths = (primary, secondary);
                                let mut dst =
                                    PlaneViewMut::new(&mut actual[1..=len], stride, extent, extent)
                                        .expect("output");
                                candidate.cdef_filter_interior(
                                    side, &src, at, strengths, direction, damping, &mut dst,
                                );
                                let mut dst = PlaneViewMut::new(
                                    &mut expected[1..=len],
                                    stride,
                                    extent,
                                    extent,
                                )
                                .expect("oracle");
                                reference.cdef_filter(
                                    side, &padded, at, strengths, direction, damping, &mut dst,
                                );
                                record(report, candidate.isa(), actual == expected, || {
                                    format!(
                                        "cdef interior exact window {side} stride {stride} pattern {pattern} pri {primary} sec {secondary} dir {direction} damping {damping}"
                                    )
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    check_strided(candidate, reference, report);
}

fn pixel(row: usize, col: usize, pattern: usize) -> u8 {
    match pattern {
        0 => (row * 137 + col * 71 + row * col * 17) as u8,
        1 => 120 + ((row * 7 + col * 11) % 16) as u8,
        2 => {
            if (row + col).is_multiple_of(2) {
                0
            } else {
                255
            }
        }
        3 => 255,
        4 => 0,
        _ => {
            let near_edge = ((row + 3 * col) % 16) as u8;
            if col.is_multiple_of(2) {
                near_edge
            } else {
                255 - near_edge
            }
        }
    }
}

/// Frame-like strides and independent alignments supplement the exhaustive
/// minimum-window cases without multiplying the full strength/damping sweep.
fn check_strided(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const SIDE: usize = 8;
    let extent = SIDE + 2 * CDEF_PADDING;
    for (src_stride, dst_stride) in [(extent, extent + 3), (1280, 1293)] {
        for (src_offset, dst_offset) in [(0, 7), (1, 0), (7, 1)] {
            let src_len = (extent - 1) * src_stride + extent;
            let dst_len = (extent - 1) * dst_stride + extent;
            let mut input = vec![GUARD; src_offset + src_len + 1];
            for row in 0..extent {
                for col in 0..extent {
                    // Independent row identities expose assembly and packing
                    // across the four 128-bit chunks of a whole-block vector.
                    input[src_offset + row * src_stride + col] = (row * 13 + col) as u8;
                }
            }
            let src = PlaneView::new(
                &input[src_offset..src_offset + src_len],
                src_stride,
                extent,
                extent,
            )
            .expect("strided source");
            let mut padded = CdefSource::new();
            padded.fill_from(&src);
            for strengths in [(0, 0), (15, 0), (0, 4), (15, 4)] {
                for direction in 0..8 {
                    let mut actual = vec![GUARD; dst_offset + dst_len + 1];
                    let mut expected = actual.clone();
                    let at = (CDEF_PADDING, CDEF_PADDING);
                    let mut dst = PlaneViewMut::new(
                        &mut actual[dst_offset..dst_offset + dst_len],
                        dst_stride,
                        extent,
                        extent,
                    )
                    .expect("strided output");
                    candidate
                        .cdef_filter_interior(SIDE, &src, at, strengths, direction, 4, &mut dst);
                    let mut dst = PlaneViewMut::new(
                        &mut expected[dst_offset..dst_offset + dst_len],
                        dst_stride,
                        extent,
                        extent,
                    )
                    .expect("strided oracle");
                    reference.cdef_filter(SIDE, &padded, at, strengths, direction, 4, &mut dst);
                    record(report, candidate.isa(), actual == expected, || {
                        format!(
                            "cdef strided src {src_stride}+{src_offset} dst {dst_stride}+{dst_offset} strengths {strengths:?} dir {direction}"
                        )
                    });
                }
            }
        }
    }
}
