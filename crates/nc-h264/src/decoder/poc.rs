//! Picture order counts of frames and fields (§8.2.1): types 0, 1 and 2,
//! including the reset after memory management operation 5. Only output
//! order depends on them in streams without B slices.

use super::slice::SliceHeader;
use super::sps::{PocType, Sps};
use super::structure::{Parity, Structure};

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

/// The order counts of one picture: both of a frame, one of a field (the
/// other field's is then the same value, unused).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FramePoc {
    /// `TopFieldOrderCnt`.
    pub top: i64,
    /// `BottomFieldOrderCnt`.
    pub bottom: i64,
    structure: Structure,
    /// `PicOrderCntMsb` (type 0) or `FrameNumOffset` (types 1, 2).
    msb_or_offset: i64,
    lsb: i64,
}

impl FramePoc {
    /// `PicOrderCnt()` of the picture (eq. 8-1): the smaller of a frame's
    /// two counts, a field's own.
    pub fn value(self) -> i64 {
        match self.structure {
            Structure::Frame => self.top.min(self.bottom),
            Structure::Field(Parity::Top) => self.top,
            Structure::Field(Parity::Bottom) => self.bottom,
        }
    }

    /// The counts after memory management operation 5 (§8.2.1:
    /// `tempPicOrderCnt` subtracted, so the picture's own count is 0).
    pub fn rebased(self) -> Self {
        let temp = self.value();
        Self {
            top: self.top - temp,
            bottom: self.bottom - temp,
            ..self
        }
    }

    /// The counts of a picture of `structure` whose frame would have
    /// `top` and `bottom`: a field keeps its own (and sets the other to it).
    fn of(structure: Structure, (top, bottom): (i64, i64), msb_or_offset: i64, lsb: i64) -> Self {
        let (top, bottom) = match structure {
            Structure::Frame => (top, bottom),
            Structure::Field(Parity::Top) => (top, top),
            Structure::Field(Parity::Bottom) => (bottom, bottom),
        };
        Self {
            top,
            bottom,
            structure,
            msb_or_offset,
            lsb,
        }
    }
}

impl PocState {
    /// Derive the order counts of the picture `header` starts (§8.2.1.1..3).
    /// The bottom field's own syntax elements are parsed only in frames
    /// (0 in fields), so each type's frame equations give a field's count
    /// too (eqs. 8-4..8-6, 8-8..8-10, 8-12).
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
                let bottom = top + i64::from(header.delta_pic_order_cnt_bottom);
                FramePoc::of(header.structure, (top, bottom), msb, lsb)
            }
            PocType::Cycle {
                offset_for_top_to_bottom_field,
                ..
            } => {
                let offset = frame_num_offset();
                let top = expected_poc(&sps.poc, offset + frame_num, header.reference)
                    + i64::from(header.delta_pic_order_cnt[0]);
                let bottom = top
                    + i64::from(*offset_for_top_to_bottom_field)
                    + i64::from(header.delta_pic_order_cnt[1]);
                FramePoc::of(header.structure, (top, bottom), offset, 0)
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
                FramePoc::of(header.structure, (value, value), offset, 0)
            }
        }
    }

    /// Record a decoded picture. After memory management operation 5 the
    /// picture counts as `frame_num` 0 with its order counts rebased to 0
    /// (§8.2.1: `tempPicOrderCnt`); a bottom field then leaves
    /// `prevPicOrderCntLsb` 0, anything else its rebased top count.
    pub fn picture_done(&mut self, sps: &Sps, header: &SliceHeader, poc: FramePoc, reset: bool) {
        match sps.poc {
            PocType::Lsb { .. } if header.reference => {
                if reset {
                    self.prev_msb = 0;
                    self.prev_lsb = match header.structure {
                        Structure::Field(Parity::Bottom) => 0,
                        _ => poc.rebased().top,
                    };
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

/// `expectedPicOrderCnt` of type 1 (eqs. 8-6..8-11) for the frame
/// `abs_frame_num` (`FrameNumOffset + frame_num`) frames in: whole cycles of
/// `offset_for_ref_frame`, then the partial one, plus
/// `offset_for_non_ref_pic` for a non-reference picture.
fn expected_poc(poc: &PocType, frame: i64, reference: bool) -> i64 {
    let PocType::Cycle {
        offset_for_non_ref_pic,
        offsets_for_ref_frame,
        ..
    } = poc
    else {
        return 0;
    };
    let cycle = offsets_for_ref_frame.len() as i64;
    let mut abs_frame_num = if cycle != 0 { frame } else { 0 };
    if !reference && abs_frame_num > 0 {
        abs_frame_num -= 1;
    }
    let mut expected = 0;
    if abs_frame_num > 0 {
        let per_cycle: i64 = offsets_for_ref_frame.iter().map(|&o| i64::from(o)).sum();
        let (count, in_cycle) = ((abs_frame_num - 1) / cycle, (abs_frame_num - 1) % cycle);
        expected = count * per_cycle
            + offsets_for_ref_frame[..=in_cycle as usize]
                .iter()
                .map(|&o| i64::from(o))
                .sum::<i64>();
    }
    if !reference {
        expected += i64::from(*offset_for_non_ref_pic);
    }
    expected
}
