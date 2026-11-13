//! Exact width-32 HEVC put footprints, every phase and all public heights.
//! Different source/output strides and offsets expose packing-order mistakes;
//! whole-buffer comparisons retain guards around and between output rows.

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter};

use crate::{check::CheckReport, check_codec::record};

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const WIDTH: usize = 32;
    const HALO: usize = 7;
    const GUARD: u8 = 0xA5;
    for height in [4, 8, 16, 32, 64] {
        for (src_stride, dst_stride, src_offset, dst_offset) in [(39, 32, 1, 3), (1280, 1293, 7, 1)]
        {
            let rows = height + HALO;
            let src_len = (rows - 1) * src_stride + WIDTH + HALO;
            let dst_len = (height - 1) * dst_stride + WIDTH;
            for pattern in 0..6 {
                let mut input = vec![GUARD; src_offset + src_len];
                for y in 0..rows {
                    for x in 0..WIDTH + HALO {
                        input[src_offset + y * src_stride + x] = match pattern {
                            0 => (x * 73 + y * 137 + x * y * 17) as u8,
                            1 => 120 + ((x * 3 + y * 7) % 16) as u8,
                            2 => {
                                if (x + y).is_multiple_of(2) {
                                    0
                                } else {
                                    255
                                }
                            }
                            3 => 0,
                            4 => 255,
                            _ => {
                                if x % 7 < 3 && y % 7 < 3 {
                                    255
                                } else {
                                    0
                                }
                            }
                        };
                    }
                }
                let src =
                    PlaneView::new(&input[src_offset..], src_stride, WIDTH + HALO, rows).unwrap();
                let size = BlockSize::new(WIDTH, height).unwrap();
                for fx in 0..4 {
                    for fy in 0..4 {
                        let run = |dsp: &Dsp| {
                            let mut out = vec![GUARD; dst_offset + dst_len + 9];
                            let mut dst = PlaneViewMut::new(
                                &mut out[dst_offset..dst_offset + dst_len],
                                dst_stride,
                                WIDTH,
                                height,
                            )
                            .unwrap();
                            dsp.predict(Filter::HevcLuma, size, &src, fx, fy, &mut dst);
                            out
                        };
                        record(
                            report,
                            candidate.isa(),
                            run(candidate) == run(reference),
                            || {
                                format!(
                                    "HEVC put32 height {height} strides {src_stride}/{dst_stride} offsets {src_offset}/{dst_offset} pattern {pattern} phase {fx}/{fy}"
                                )
                            },
                        );
                    }
                }
            }
        }
    }
}
