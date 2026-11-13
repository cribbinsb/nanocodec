//! Picture order counts of frames (§8.2.1): types 0, 1 and 2, including the
//! reset after memory management operation 5. Only output order depends on
//! them in streams without B slices.

use super::slice::SliceHeader;
use super::sps::{PocType, Sps};

/// What the next picture's derivation needs from earlier pictures.
#[derive(Debug, Clone, Copy, Default)]
#[allow(
    clippy::struct_field_names,
    reason = "the fields are the standard's prev* variables"
)]
pub(super) struct PocState {
    /// `prevPicOrderCntMsb` / `prevPicOrderCntLsb` (type 0): from the
    /// previous reference picture.
    prev_msb: i64,
    prev_lsb: i64,
    /// `prevFrameNumOffset` and `prevFrameNum` (types 1 and 2): from the
    /// previous picture.
    prev_frame_num_offset: i64,
    prev_frame_num: u32,
}

/// The order counts of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FramePoc {
    /// `TopFieldOrderCnt`.
    pub top: i64,
    /// `BottomFieldOrderCnt`.
    pub bottom: i64,
    /// `PicOrderCntMsb` (type 0) or `FrameNumOffset` (types 1, 2).
    msb_or_offset: i64,
    lsb: i64,
}

impl FramePoc {
    /// `PicOrderCnt()` of the frame (eq. 8-1).
    pub fn value(self) -> i64 {
        self.top.min(self.bottom)
    }
}

impl PocState {
    /// Derive the order counts of the picture `header` starts (§8.2.1.1..3).
    pub fn derive(&self, sps: &Sps, header: &SliceHeader) -> FramePoc {
        let frame_num = i64::from(header.frame_num);
        let frame_num_offset = || {
            if header.idr {
                0
            } else if self.prev_frame_num > header.frame_num {
                self.prev_frame_num_offset + i64::from(sps.max_frame_num())
            } else {
                self.prev_frame_num_offset
            }
        };
        match &sps.poc {
            PocType::Lsb { log2_max_lsb } => {
                let max_lsb = 1i64 << log2_max_lsb;
                let lsb = i64::from(header.pic_order_cnt_lsb);
                let (prev_msb, prev_lsb) = if header.idr {
                    (0, 0)
                } else {
                    (self.prev_msb, self.prev_lsb)
                };
                // eq. 8-3: wrap the MSB when the LSB jumps by half its range.
                let msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                    prev_msb + max_lsb
                } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                    prev_msb - max_lsb
                } else {
                    prev_msb
                };
                let top = msb + lsb;
                FramePoc {
                    top,
                    bottom: top + i64::from(header.delta_pic_order_cnt_bottom),
                    msb_or_offset: msb,
                    lsb,
                }
            }
            PocType::Cycle {
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offsets_for_ref_frame,
                ..
            } => {
                let offset = frame_num_offset();
                let cycle = offsets_for_ref_frame.len() as i64;
                let mut abs_frame_num = if cycle != 0 { offset + frame_num } else { 0 };
                if !header.reference && abs_frame_num > 0 {
                    abs_frame_num -= 1;
                }
                // eqs. 8-7..8-10: whole cycles, then the partial one.
                let mut expected = 0;
                if abs_frame_num > 0 {
                    let per_cycle: i64 = offsets_for_ref_frame.iter().map(|&o| i64::from(o)).sum();
                    let (count, in_cycle) =
                        ((abs_frame_num - 1) / cycle, (abs_frame_num - 1) % cycle);
                    expected = count * per_cycle
                        + offsets_for_ref_frame[..=in_cycle as usize]
                            .iter()
                            .map(|&o| i64::from(o))
                            .sum::<i64>();
                }
                if !header.reference {
                    expected += i64::from(*offset_for_non_ref_pic);
                }
                let top = expected + i64::from(header.delta_pic_order_cnt[0]);
                FramePoc {
                    top,
                    bottom: top
                        + i64::from(*offset_for_top_to_bottom_field)
                        + i64::from(header.delta_pic_order_cnt[1]),
                    msb_or_offset: offset,
                    lsb: 0,
                }
            }
            PocType::FrameNum => {
                let offset = frame_num_offset();
                // eq. 8-12: twice the frame count, one less for non-reference.
                let value = if header.idr {
                    0
                } else if header.reference {
                    2 * (offset + frame_num)
                } else {
                    2 * (offset + frame_num) - 1
                };
                FramePoc {
                    top: value,
                    bottom: value,
                    msb_or_offset: offset,
                    lsb: 0,
                }
            }
        }
    }

    /// Record a decoded picture. After memory management operation 5 the
    /// picture counts as `frame_num` 0 with its order counts rebased to 0
    /// (§8.2.1: `tempPicOrderCnt`).
    pub fn picture_done(&mut self, sps: &Sps, header: &SliceHeader, poc: FramePoc, reset: bool) {
        match sps.poc {
            PocType::Lsb { .. } if header.reference => {
                if reset {
                    self.prev_msb = 0;
                    self.prev_lsb = poc.top - poc.value();
                } else {
                    self.prev_msb = poc.msb_or_offset;
                    self.prev_lsb = poc.lsb;
                }
            }
            PocType::Lsb { .. } => {}
            PocType::Cycle { .. } | PocType::FrameNum => {
                self.prev_frame_num_offset = if reset { 0 } else { poc.msb_or_offset };
                self.prev_frame_num = if reset { 0 } else { header.frame_num };
            }
        }
    }

    /// Record a frame inferred for a `frame_num` gap (§8.2.5.2), which
    /// advances `prevFrameNum` (and the offset when `frame_num` wraps).
    pub fn gap_frame(&mut self, sps: &Sps, frame_num: u32) {
        if self.prev_frame_num > frame_num {
            self.prev_frame_num_offset += i64::from(sps.max_frame_num());
        }
        self.prev_frame_num = frame_num;
    }
}
