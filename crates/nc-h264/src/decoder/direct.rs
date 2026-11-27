//! Direct prediction of B macroblocks and sub-macroblocks (§8.4.1.2):
//! spatial (§8.4.1.2.2: reference indices and vectors from the
//! neighbours, zeroed where the co-located block is still) and temporal
//! (§8.4.1.2.3: the co-located vector scaled by order count distances), and
//! the distance scale factor implicit weighted prediction shares
//! (§8.4.2.3.1).
//!
//! The co-located picture is `RefPicList1[0]` or the frame or field of its
//! frame that Table 8-6 selects; its macroblock states are kept with it in
//! the DPB ([`RefPicture::motion`]). Where the two are coded differently
//! (a field predicting from a frame, or the reverse) Table 8-8 maps the
//! macroblock and block rows, and temporal prediction rescales the
//! vertical vector (`vertMvScale`).

use nc_common::Sample;
use nc_encoder::Mv;

use super::dpb::{FrameMotion, RefPicture};
use super::mbstate::{LISTS, MbState};
use super::mvpred::MotionCache;
use super::structure::{Parity, Structure, frame_of};
use super::{DecodeError, Result};

/// Clipping range of the order count distances `tb` and `td` (eqs. 8-201,
/// 8-202).
const MAX_DISTANCE: i64 = 127;
/// Range of `DistScaleFactor` (eq. 8-198).
const MAX_DIST_SCALE: i32 = 1023;
/// Largest co-located vector component of a "still" block (colZeroFlag).
const STILL_MV: i16 = 1;

/// `DistScaleFactor` of a picture with order count `current` between
/// `poc0` and `poc1` (eqs. 8-197..8-202), or `None` when the two are at the
/// same distance (`td` 0), where the callers use their fallbacks.
pub(super) fn dist_scale_factor(current: i64, poc0: i64, poc1: i64) -> Option<i32> {
    let tb = (current - poc0).clamp(-MAX_DISTANCE - 1, MAX_DISTANCE) as i32;
    let td = (poc1 - poc0).clamp(-MAX_DISTANCE - 1, MAX_DISTANCE) as i32;
    if td == 0 {
        return None;
    }
    let tx = (16384 + (td / 2).abs()) / td;
    Some(((tb * tx + 32) >> 6).clamp(-MAX_DIST_SCALE - 1, MAX_DIST_SCALE))
}

/// How the co-located picture is coded against the current one (Table
/// 8-8: `vertMvScale`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColLayout {
    /// Both frames or both fields: the same macroblock (`One_To_One`).
    Same,
    /// A field predicting from a frame (`Frm_To_Fld`): two frame
    /// macroblock rows per field row, vertical vectors halved.
    FrameToField(Parity),
    /// A frame predicting from a field (`Fld_To_Frm`): a field macroblock
    /// row per two frame rows, vertical vectors doubled.
    FieldToFrame,
}

/// What direct prediction reads in one B slice.
#[derive(Debug)]
pub(super) struct DirectContext<'a> {
    /// `direct_spatial_mv_pred_flag`.
    pub spatial: bool,
    /// `direct_8x8_inference_flag`: each 8x8 block takes the co-located
    /// corner 4x4 block's motion.
    pub inference_8x8: bool,
    /// The co-located picture's macroblock states.
    col: &'a [MbState],
    layout: ColLayout,
    /// `PicWidthInMbs`.
    width_mbs: usize,
    /// Whether `RefPicList1[0]` is a short-term reference (spatial
    /// colZeroFlag).
    col_short_term: bool,
    /// Identities of the list 0 entries (temporal: the co-located block's
    /// reference becomes the lowest index naming it).
    l0_ids: &'a [Option<u32>],
    /// Temporal: `DistScaleFactor` of each list 0 entry against
    /// `RefPicList1[0]`, `None` where the vector is copied unscaled (a
    /// long-term entry or `td` 0, eqs. 8-191, 8-192).
    scale: Vec<Option<i32>>,
}

/// The reference indices and vectors of one 4x4 block (per list; -1 and a
/// zero vector for a list it does not use).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DirectMotion {
    pub ref_idx: [i8; LISTS],
    pub mv: [Mv; LISTS],
}

/// Spatial direct prediction's macroblock-wide part (eqs. 8-184..8-189).
#[derive(Debug, Clone, Copy)]
pub(super) struct Spatial {
    motion: DirectMotion,
    /// `directZeroPredictionFlag`: both lists from index 0 with zero
    /// vectors, whatever the co-located block.
    zero: bool,
}

impl Spatial {
    /// From the macroblock's two fresh motion caches (no partition of it
    /// recorded yet).
    pub fn new(caches: &[MotionCache; LISTS]) -> Self {
        let ref_idx = caches.each_ref().map(MotionCache::direct_ref_idx);
        if ref_idx.iter().all(|&r| r < 0) {
            return Self {
                motion: DirectMotion {
                    ref_idx: [0; LISTS],
                    mv: [Mv::ZERO; LISTS],
                },
                zero: true,
            };
        }
        let mut mv = [Mv::ZERO; LISTS];
        for (list, cache) in caches.iter().enumerate() {
            if ref_idx[list] >= 0 {
                mv[list] = cache.predict_16x16(ref_idx[list]);
            }
        }
        Self {
            motion: DirectMotion { ref_idx, mv },
            zero: false,
        }
    }
}

impl<'a> DirectContext<'a> {
    /// The context of a B slice of picture `current` (with order count
    /// `poc`, `width_mbs` by `mbs / width_mbs` macroblocks) with lists `l0`
    /// and `l1` (whose first entry gives the co-located picture) and
    /// identities `l0_ids`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::MissingReference`] when `RefPicList1[0]` has no
    /// picture or lacks the field Table 8-6 selects; malformed when its
    /// size differs (checked by the caller for prediction, here for its
    /// macroblock states).
    pub fn new<T: Sample>(
        (spatial, inference_8x8): (bool, bool),
        lists: [&'a [Option<RefPicture<T>>]; LISTS],
        l0_ids: &'a [Option<u32>],
        (current, poc): (Structure, i64),
        (width_mbs, mbs): (usize, usize),
    ) -> Result<Self> {
        let col = lists[1]
            .first()
            .and_then(Option::as_ref)
            .ok_or(DecodeError::MissingReference)?;
        let (states, layout, col_mbs) = co_located_picture(col, current, poc, mbs)?;
        if states.len() != col_mbs {
            return Err(DecodeError::Malformed("co-located picture of another size"));
        }
        let scale = if spatial {
            Vec::new()
        } else {
            lists[0]
                .iter()
                .map(|entry| {
                    entry.as_ref().and_then(|pic0| {
                        if pic0.long_term {
                            None
                        } else {
                            dist_scale_factor(poc, pic0.poc, col.poc)
                        }
                    })
                })
                .collect()
        };
        Ok(Self {
            spatial,
            inference_8x8,
            col: states,
            layout,
            width_mbs,
            col_short_term: !col.long_term,
            l0_ids,
            scale,
        })
    }

    /// The co-located block of 4x4 block (`x4`, `y4`) of macroblock
    /// `addr` (§8.4.1.2.1: the corner block of its 8x8 block under
    /// `direct_8x8_inference_flag`, its macroblock and row mapped by Table
    /// 8-8): its vector and reference index (list 0's when it predicts
    /// from list 0, else list 1's; -1 and zero for an intra block), and the
    /// identity of the picture it references.
    fn co_located(&self, addr: usize, x4: usize, y4: usize) -> (Mv, i8, u32) {
        let (x4, y4) = if self.inference_8x8 {
            ((x4 / 2) * 3, (y4 / 2) * 3)
        } else {
            (x4, y4)
        };
        let width = self.width_mbs;
        let (col_addr, y4) = match self.layout {
            ColLayout::Same => (addr, y4),
            // mbAddrCol1 = 2 · PicWidthInMbs · (CurrMbAddr / PicWidthInMbs)
            // + CurrMbAddr % PicWidthInMbs + PicWidthInMbs · (yCol / 8),
            // yM = (2 · yCol) % 16.
            ColLayout::FrameToField(_) => (
                2 * width * (addr / width) + addr % width + width * (y4 / 2),
                (2 * y4) % 4,
            ),
            // mbAddrCol4 = PicWidthInMbs · (CurrMbAddr / (2 · PicWidthInMbs))
            // + CurrMbAddr % PicWidthInMbs, yM = 8 · ((CurrMbAddr /
            // PicWidthInMbs) % 2) + 4 · (yCol / 8).
            ColLayout::FieldToFrame => (
                width * (addr / (2 * width)) + addr % width,
                2 * ((addr / width) % 2) + y4 / 2,
            ),
        };
        let col = &self.col[col_addr];
        if col.is_intra() {
            return (Mv::ZERO, -1, 0);
        }
        let b8 = (y4 / 2) * 2 + x4 / 2;
        let list = usize::from(col.ref_idx[0][b8] < 0);
        let (mv, ref_idx) = col.motion(list, x4, y4);
        (mv, ref_idx, col.ref_pic[list][b8])
    }

    /// The motion of 4x4 block (`x4`, `y4`) of macroblock `addr`; `spatial`
    /// is the macroblock's spatial part (spatial mode).
    pub fn block(
        &self,
        spatial: Option<&Spatial>,
        addr: usize,
        x4: usize,
        y4: usize,
    ) -> Result<DirectMotion> {
        let (mv_col, ref_col, pic_col) = self.co_located(addr, x4, y4);
        if let Some(spatial) = spatial {
            let mut motion = spatial.motion;
            // colZeroFlag (§8.4.1.2.2): a still co-located block on a
            // short-term picture zeroes the vectors of index 0.
            let still = self.col_short_term
                && ref_col == 0
                && mv_col.x.abs() <= STILL_MV
                && mv_col.y.abs() <= STILL_MV;
            if still && !spatial.zero {
                for (mv, &ref_idx) in motion.mv.iter_mut().zip(&motion.ref_idx) {
                    if ref_idx == 0 {
                        *mv = Mv::ZERO;
                    }
                }
            }
            return Ok(motion);
        }
        // Temporal (§8.4.1.2.3): list 0 names the co-located block's
        // reference (index 0 for an intra block) as the current picture
        // sees it: the field of a frame with the current field's parity,
        // the frame of a field; list 1 index 0.
        let (target, mv_col) = match self.layout {
            ColLayout::Same => (pic_col, mv_col),
            ColLayout::FrameToField(parity) => (
                Structure::Field(parity).reference_id(frame_of(pic_col)),
                Mv {
                    x: mv_col.x,
                    y: mv_col.y / 2,
                },
            ),
            ColLayout::FieldToFrame => (
                Structure::Frame.reference_id(frame_of(pic_col)),
                Mv {
                    x: mv_col.x,
                    y: mv_col.y.saturating_mul(2),
                },
            ),
        };
        let ref_l0 = if ref_col < 0 {
            0
        } else {
            self.l0_ids
                .iter()
                .position(|&id| id == Some(target))
                .ok_or(DecodeError::MissingReference)?
        };
        let (mv0, mv1) = match self.scale.get(ref_l0).copied().flatten() {
            Some(scale) => {
                // Conforming streams keep the result in range (§8.4.1.2.3).
                let scaled = |c: i16| {
                    ((scale * i32::from(c) + 128) >> 8).clamp(i16::MIN.into(), i16::MAX.into())
                        as i16
                };
                let mv0 = Mv {
                    x: scaled(mv_col.x),
                    y: scaled(mv_col.y),
                };
                let mv1 = Mv {
                    x: mv0.x.wrapping_sub(mv_col.x),
                    y: mv0.y.wrapping_sub(mv_col.y),
                };
                (mv0, mv1)
            }
            None => (mv_col, Mv::ZERO),
        };
        Ok(DirectMotion {
            ref_idx: [ref_l0 as i8, 0],
            mv: [mv0, mv1],
        })
    }
}

/// The co-located picture of a B picture `current` with order count
/// `poc` and `mbs` macroblocks whose `RefPicList1[0]` is `first` (Table
/// 8-6): its macroblock states, how they map, and how many there must be.
/// A frame predicting from a field pair takes the field nearer in order
/// count (the bottom one at equal distance).
fn co_located_picture<'a, T: Sample>(
    first: &'a RefPicture<T>,
    current: Structure,
    poc: i64,
    mbs: usize,
) -> Result<(&'a [MbState], ColLayout, usize)> {
    let field = |fields: &'a [Option<std::sync::Arc<Vec<MbState>>>; 2], parity: Parity| {
        fields[parity.index()]
            .as_deref()
            .map(Vec::as_slice)
            .ok_or(DecodeError::MissingReference)
    };
    Ok(match (&first.motion, current) {
        (FrameMotion::Frame(states), Structure::Frame) => (states, ColLayout::Same, mbs),
        (FrameMotion::Frame(states), Structure::Field(parity)) => {
            (states, ColLayout::FrameToField(parity), 2 * mbs)
        }
        (FrameMotion::Fields(fields), Structure::Field(_)) => {
            let parity = first
                .structure
                .parity()
                .ok_or(DecodeError::Malformed("frame reference of a field"))?;
            (field(fields, parity)?, ColLayout::Same, mbs)
        }
        (FrameMotion::Fields(fields), Structure::Frame) => {
            let [top, bottom] = first.field_poc.map(|p| (p - poc).abs());
            let parity = if top < bottom {
                Parity::Top
            } else {
                Parity::Bottom
            };
            (field(fields, parity)?, ColLayout::FieldToFrame, mbs / 2)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_scale_factor_matches_the_standard_examples() {
        // Halfway between: tb 1, td 2 -> tx 8192, DSF 128 (a half).
        assert_eq!(dist_scale_factor(1, 0, 2), Some(128));
        // Same distance: no scale.
        assert_eq!(dist_scale_factor(1, 2, 2), None);
        // Clipped to the range.
        assert_eq!(dist_scale_factor(500, 0, 1), Some(MAX_DIST_SCALE));
    }
}
