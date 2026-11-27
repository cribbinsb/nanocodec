//! Direct prediction of B macroblocks and sub-macroblocks (§8.4.1.2):
//! spatial (§8.4.1.2.2: reference indices and vectors from the
//! neighbours, zeroed where the co-located block is still) and temporal
//! (§8.4.1.2.3: the co-located vector scaled by order count distances), and
//! the distance scale factor implicit weighted prediction shares
//! (§8.4.2.3.1).
//!
//! The co-located picture is `RefPicList1[0]` or the frame or field of its
//! frame that Table 8-6 selects; its macroblock states are kept with it in
//! the DPB ([`RefPicture::motion`]; one set per colour plane when the planes
//! are coded separately, the co-located block being in the same plane).
//! Where the two are coded differently (a field predicting from a frame, or
//! the reverse) Table 8-8 maps the macroblock and block rows, and temporal
//! prediction rescales the vertical vector (`vertMvScale`).

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

/// The picture or macroblock direct prediction is for, as Table 8-8 tells
/// them apart (`PicCodingStruct(CurrPic)` and, in MBAFF frames,
/// `mb_field_decoding_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Current {
    /// A frame (FRM), macroblocks in raster order.
    Frame,
    /// A field picture (FLD).
    Field(Parity),
    /// A frame macroblock of an MBAFF frame (AFRM), pair addresses.
    PairFrame,
    /// A field macroblock of that parity of an MBAFF frame.
    PairField(Parity),
}

impl Current {
    /// A picture of `structure` (an MBAFF frame's frame macroblocks when
    /// `mbaff`).
    pub fn of(structure: Structure, mbaff: bool) -> Self {
        match structure {
            Structure::Frame if mbaff => Self::PairFrame,
            Structure::Frame => Self::Frame,
            Structure::Field(parity) => Self::Field(parity),
        }
    }

    /// The parity of a field picture or field macroblock.
    fn parity(self) -> Option<Parity> {
        match self {
            Self::Field(parity) | Self::PairField(parity) => Some(parity),
            Self::Frame | Self::PairFrame => None,
        }
    }
}

/// How the co-located picture is coded (`PicCodingStruct(colPic)`), with
/// its macroblock states.
#[derive(Debug, Clone, Copy)]
enum ColSource<'a> {
    /// A frame (FRM): raster order.
    Frame(&'a [MbState]),
    /// A field (FLD): raster order within the field.
    Field(&'a [MbState]),
    /// An MBAFF frame (AFRM): pair addresses, each macroblock with its
    /// field decoding flag.
    Pair(&'a [MbState]),
}

impl<'a> ColSource<'a> {
    /// The states of colour plane `colour_plane` of `planes` (one set of
    /// `mbs` per plane coded separately).
    fn plane(self, mbs: usize, (colour_plane, planes): (usize, usize)) -> Result<Self> {
        let pick = |states: &'a [MbState]| {
            if states.len() != mbs * planes {
                return Err(DecodeError::Malformed("co-located picture of another size"));
            }
            Ok(&states[mbs * colour_plane..mbs * (colour_plane + 1)])
        };
        Ok(match self {
            Self::Frame(states) => Self::Frame(pick(states)?),
            Self::Field(states) => Self::Field(pick(states)?),
            Self::Pair(states) => Self::Pair(pick(states)?),
        })
    }
}

/// `vertMvScale` (§8.4.1.2.1): how a co-located vertical vector and its
/// reference translate to the current picture or macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertMvScale {
    OneToOne,
    /// A field predicting from a frame macroblock: halved, the field of
    /// the frame with the current parity.
    FrmToFld,
    /// A frame predicting from a field macroblock: doubled, the frame of
    /// the field.
    FldToFrm,
}

/// What direct prediction reads in one B slice.
#[derive(Debug)]
pub(super) struct DirectContext<'a> {
    /// `direct_spatial_mv_pred_flag`.
    pub spatial: bool,
    /// `direct_8x8_inference_flag`: each 8x8 block takes the co-located
    /// corner 4x4 block's motion.
    pub inference_8x8: bool,
    /// The co-located picture.
    col: ColSource<'a>,
    current: Current,
    /// Of an MBAFF co-located field pair seen from a frame macroblock: the
    /// field nearer in order count (Table 8-8, `mbAddrCol6`).
    col_parity: Parity,
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
    /// `poc`; `mbs` macroblocks of the picture, a frame's for MBAFF frames,
    /// `width_mbs` wide) with lists `l0` and `l1` (whose first entry gives
    /// the co-located picture) and identities `l0_ids`, in colour plane
    /// `colour_plane` of `planes` coded separately (0 of 1 otherwise).
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
        (current, poc): (Current, i64),
        (width_mbs, mbs): (usize, usize),
        (colour_plane, planes): (usize, usize),
    ) -> Result<Self> {
        let col = lists[1]
            .first()
            .and_then(Option::as_ref)
            .ok_or(DecodeError::MissingReference)?;
        let (source, col_mbs) = co_located_picture(col, current, poc, mbs)?;
        // Separately coded colour planes keep one set of states per plane:
        // the co-located block is in the same plane (JM's `JVmv_info`).
        let source = source.plane(col_mbs, (colour_plane, planes))?;
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
            col: source,
            current,
            col_parity: nearer_field(col, poc),
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
    fn co_located(&self, addr: usize, x4: usize, y4: usize) -> (Mv, i8, u32, VertMvScale) {
        let (x4, y4) = if self.inference_8x8 {
            ((x4 / 2) * 3, (y4 / 2) * 3)
        } else {
            (x4, y4)
        };
        let (col, col_addr, y4, scale) = self.co_located_block(addr, y4);
        let col = &col[col_addr];
        if col.is_intra() {
            return (Mv::ZERO, -1, 0, scale);
        }
        let b8 = (y4 / 2) * 2 + x4 / 2;
        let list = usize::from(col.ref_idx[0][b8] < 0);
        let (mv, ref_idx) = col.motion(list, x4, y4);
        (mv, ref_idx, col.ref_pic[list][b8], scale)
    }

    /// Table 8-8: the co-located macroblock states, macroblock address and
    /// 4x4 row (`yM / 4`) of row `y4` (`yCol / 4`) of macroblock `addr`,
    /// and `vertMvScale`.
    fn co_located_block(
        &self,
        addr: usize,
        y4: usize,
    ) -> (&'a [MbState], usize, usize, VertMvScale) {
        use VertMvScale::{FldToFrm, FrmToFld, OneToOne};
        let width = self.width_mbs;
        // A frame macroblock's rows of a field macroblock: yM = 8 ·
        // (bottom) + 4 · (yCol / 8).
        let frame_rows_of_field = |bottom: usize| 2 * bottom + y4 / 2;
        // A field macroblock's rows of a frame pair: the frame macroblock
        // yCol / 8, yM = (2 · yCol) % 16.
        let field_rows_of_frame = (2 * y4) % 4;
        match (self.current, self.col) {
            (Current::Frame, ColSource::Frame(col))
            | (Current::Field(_) | Current::PairField(_), ColSource::Field(col)) => {
                // An MBAFF field macroblock's field is the pair's row.
                let col_addr = if matches!(self.current, Current::PairField(_)) {
                    addr / 2
                } else {
                    addr
                };
                (col, col_addr, y4, OneToOne)
            }
            // mbAddrCol1.
            (Current::Field(_), ColSource::Frame(col)) => (
                col,
                2 * width * (addr / width) + addr % width + width * (y4 / 2),
                field_rows_of_frame,
                FrmToFld,
            ),
            // mbAddrCol4.
            (Current::Frame, ColSource::Field(col)) => (
                col,
                width * (addr / (2 * width)) + addr % width,
                frame_rows_of_field((addr / width) % 2),
                FldToFrm,
            ),
            // mbAddrCol5.
            (Current::PairFrame, ColSource::Field(col)) => {
                (col, addr / 2, frame_rows_of_field(addr % 2), FldToFrm)
            }
            // mbAddrCol2 / mbAddrCol3: the pair at the field macroblock's
            // place.
            (Current::Field(parity), ColSource::Pair(col)) => {
                if col[2 * addr].field {
                    (col, 2 * addr + parity.index(), y4, OneToOne)
                } else {
                    (col, 2 * addr + y4 / 2, field_rows_of_frame, FrmToFld)
                }
            }
            // mbAddrCol6, mbAddrCol7: the pair at the same place.
            (Current::PairFrame, ColSource::Pair(col)) => {
                if col[addr].field {
                    let field_mb = 2 * (addr / 2) + self.col_parity.index();
                    (col, field_mb, frame_rows_of_field(addr % 2), FldToFrm)
                } else {
                    (col, addr, y4, OneToOne)
                }
            }
            (Current::PairField(_), ColSource::Pair(col)) => {
                if col[addr].field {
                    (col, addr, y4, OneToOne)
                } else {
                    (col, 2 * (addr / 2) + y4 / 2, field_rows_of_frame, FrmToFld)
                }
            }
            // A frame sequence has no MBAFF frames and the reverse
            // (`co_located_picture` refuses these).
            (Current::Frame, ColSource::Pair(col))
            | (Current::PairFrame | Current::PairField(_), ColSource::Frame(col)) => {
                (col, addr, y4, OneToOne)
            }
        }
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
        let (mv_col, ref_col, pic_col, scale) = self.co_located(addr, x4, y4);
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
        let (target, mv_col) = match scale {
            VertMvScale::OneToOne => (pic_col, mv_col),
            VertMvScale::FrmToFld => (
                Structure::Field(
                    self.current
                        .parity()
                        .expect("a field predicts from a frame"),
                )
                .reference_id(frame_of(pic_col)),
                Mv {
                    x: mv_col.x,
                    y: mv_col.y / 2,
                },
            ),
            VertMvScale::FldToFrm => (
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

/// The co-located picture of a B picture or macroblock `current` with
/// order count `poc` and `mbs` macroblocks (a frame's for MBAFF frames)
/// whose `RefPicList1[0]` is `first` (Table 8-6), and how many macroblock
/// states it must have. A frame predicting from a field pair takes the
/// nearer field ([`nearer_field`]); a field takes the field of its own
/// list entry.
fn co_located_picture<'a, T: Sample>(
    first: &'a RefPicture<T>,
    current: Current,
    poc: i64,
    mbs: usize,
) -> Result<(ColSource<'a>, usize)> {
    let field = |fields: &'a [Option<std::sync::Arc<Vec<MbState>>>; 2], parity: Parity| {
        fields[parity.index()]
            .as_deref()
            .map(|states| ColSource::Field(states))
            .ok_or(DecodeError::MissingReference)
    };
    let mixed = DecodeError::Malformed("MBAFF and non-MBAFF frames in one sequence");
    Ok(match (&first.motion, current) {
        (FrameMotion::Frame(states), Current::Frame) => (ColSource::Frame(states), mbs),
        (FrameMotion::Frame(states), Current::Field(_)) => (ColSource::Frame(states), 2 * mbs),
        (FrameMotion::Frame(_), Current::PairFrame | Current::PairField(_))
        | (FrameMotion::Mbaff(_), Current::Frame) => return Err(mixed),
        (FrameMotion::Mbaff(states), Current::Field(_)) => (ColSource::Pair(states), 2 * mbs),
        (FrameMotion::Mbaff(states), Current::PairFrame | Current::PairField(_)) => {
            (ColSource::Pair(states), mbs)
        }
        (FrameMotion::Fields(fields), Current::Field(_)) => {
            let parity = first
                .structure
                .parity()
                .ok_or(DecodeError::Malformed("frame reference of a field"))?;
            (field(fields, parity)?, mbs)
        }
        (FrameMotion::Fields(fields), Current::PairField(parity)) => {
            (field(fields, parity)?, mbs / 2)
        }
        (FrameMotion::Fields(fields), Current::Frame | Current::PairFrame) => {
            (field(fields, nearer_field(first, poc))?, mbs / 2)
        }
    })
}

/// The field of `first`'s frame nearer in order count to a frame with
/// order count `poc`, the bottom one at equal distance (Table 8-6:
/// `topAbsDiffPOC < bottomAbsDiffPOC`).
fn nearer_field<T: Sample>(first: &RefPicture<T>, poc: i64) -> Parity {
    let [top, bottom] = first.field_poc.map(|p| (p - poc).abs());
    if top < bottom {
        Parity::Top
    } else {
        Parity::Bottom
    }
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
