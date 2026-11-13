//! Reference frame marking and reference picture lists of frames:
//! sliding-window and adaptive marking with all memory management
//! operations (§8.2.5), `frame_num` gap filling with "non-existing" frames
//! (§8.2.5.2), P list initialisation (§8.2.4.2.1) and modification
//! (§8.2.4.3).
//!
//! Output order is not handled here: decoded pictures are shared with the
//! output queue (`output.rs`) through `Arc`s.

use std::sync::Arc;

use nc_common::Picture;

use super::{DecodeError, Result};

/// `LongTermFrameIdx` of an IDR picture with `long_term_reference_flag`.
const IDR_LONG_TERM_FRAME_IDX: u32 = 0;

/// A decoded picture with the identity the deblocking filter compares
/// ("different reference pictures", §8.7.2.1) and its frame size.
#[derive(Debug, Clone)]
pub(super) struct RefPicture {
    pub id: u32,
    pub picture: Arc<Picture>,
}

/// One reference frame. Frames inferred for a `frame_num` gap
/// ("non-existing" frames) have no picture and must not be referenced.
#[derive(Debug, Clone)]
struct RefFrame {
    picture: Option<RefPicture>,
    frame_num: u32,
    /// `LongTermFrameIdx` of a long-term frame; `None`: short-term.
    long_term_idx: Option<u32>,
}

/// A `ref_pic_list_modification()` command (§7.3.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Modification {
    /// `modification_of_pic_nums_idc` 0: subtract `abs_diff_pic_num_minus1 + 1`.
    Subtract(u32),
    /// `modification_of_pic_nums_idc` 1: add `abs_diff_pic_num_minus1 + 1`.
    Add(u32),
    /// `modification_of_pic_nums_idc` 2: this `long_term_pic_num`.
    LongTerm(u32),
}

/// A memory management control operation (§7.3.3.3, Table 7-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mmco {
    /// 1: release the short-term frame `difference_of_pic_nums_minus1 + 1`
    /// below the current picture number.
    ReleaseShortTerm(u32),
    /// 2: release the long-term frame with this `long_term_pic_num`.
    ReleaseLongTerm(u32),
    /// 3: make a short-term frame long-term frame `long_term_frame_idx`.
    ShortTermToLongTerm {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// 4: `max_long_term_frame_idx_plus1`.
    MaxLongTermFrameIdx(u32),
    /// 5: release every reference frame (and reset `frame_num` and POC).
    ReleaseAll,
    /// 6: make the current frame long-term frame `long_term_frame_idx`.
    CurrentToLongTerm(u32),
}

/// The frame a list modification moves: by `PicNum` or `LongTermPicNum`.
#[derive(Debug, Clone, Copy)]
enum Target {
    ShortTerm(i64),
    LongTerm(u32),
}

/// How a decoded picture is marked (`dec_ref_pic_marking()`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Marking {
    /// `nal_ref_idc` 0.
    NonReference,
    /// An IDR picture, long-term when `long_term_reference_flag` is set.
    Idr { long_term: bool },
    /// `adaptive_ref_pic_marking_mode_flag` 0.
    SlidingWindow,
    /// Memory management operations, in order.
    Adaptive(Vec<Mmco>),
}

impl Marking {
    /// Whether the marking contains memory management operation 5.
    pub fn resets(&self) -> bool {
        matches!(self, Self::Adaptive(ops) if ops.contains(&Mmco::ReleaseAll))
    }
}

/// The reference frames of the coded video sequence.
#[derive(Debug, Default)]
pub(super) struct Dpb {
    frames: Vec<RefFrame>,
    /// `MaxLongTermFrameIdx` (`None`: "no long-term frame indices").
    max_long_term_frame_idx: Option<u32>,
    /// `PrevRefFrameNum` (§7.4.3).
    prev_ref_frame_num: u32,
    /// `max_num_ref_frames` of the active SPS (at least 1).
    max_frames: usize,
    /// `MaxFrameNum` of the active SPS.
    max_frame_num: u32,
}

impl Dpb {
    /// Test hook: overwrite every reference picture with `value`, as a
    /// decoder that lost a frame would hold wrong references.
    #[cfg(test)]
    pub(super) fn corrupt_pictures(&mut self, value: u8) {
        for frame in &mut self.frames {
            if let Some(reference) = &mut frame.picture {
                let picture = Arc::make_mut(&mut reference.picture);
                for plane in nc_common::PlaneIndex::ALL {
                    picture.plane_mut(plane).view_mut().fill(value);
                }
                picture.extend_borders();
            }
        }
    }

    /// Start a coded video sequence at an IDR picture (§8.2.5.1: every
    /// reference frame becomes unused).
    pub fn start_idr(&mut self, max_frames: usize, max_frame_num: u32) {
        self.release_all();
        self.max_frames = max_frames.max(1);
        self.max_frame_num = max_frame_num;
    }

    fn release_all(&mut self) {
        self.frames.clear();
        self.max_long_term_frame_idx = None;
        self.prev_ref_frame_num = 0;
    }

    /// The `frame_num` values a picture with `frame_num` implies are missing
    /// (§8.2.5.2), in order; empty without a gap. A gap the SPS does not
    /// allow is an unintentional loss (§8.2.5.2): the frames are inferred
    /// all the same, so a picture predicting from one is refused
    /// ([`DecodeError::MissingReference`]) and the pictures that do not
    /// decode.
    ///
    /// A frame (this decoder has no fields) never repeats `PrevRefFrameNum`
    /// (§7.4.3): one that does follows lost frames that wrapped
    /// `frame_num`, typically a lost IDR picture after a reference with
    /// `frame_num` 1, so every other `frame_num` is inferred missing.
    pub fn gap(&self, frame_num: u32) -> Vec<u32> {
        let expected = (self.prev_ref_frame_num + 1) % self.max_frame_num;
        if frame_num == expected {
            return Vec::new();
        }
        let mut missing = Vec::new();
        let mut unused = expected;
        while unused != frame_num {
            missing.push(unused);
            unused = (unused + 1) % self.max_frame_num;
        }
        missing
    }

    /// A `frame_num` gap ran through 0: an IDR picture may have been lost.
    /// Our encoder keeps `LongTermFrameIdx` 0 for the IDR picture alone and
    /// releases it at the first mark, long before `frame_num` wraps
    /// (`references.rs`), so a long-term picture still under index 0 is
    /// the previous IDR picture's; it goes, and a picture that names index
    /// 0 is refused instead of predicting from it.
    pub fn lost_idr(&mut self) {
        self.frames
            .retain(|f| f.long_term_idx != Some(IDR_LONG_TERM_FRAME_IDX));
    }

    /// Store a "non-existing" short-term frame for a gap `frame_num`
    /// (sliding window marking, §8.2.5.2).
    pub fn insert_gap_frame(&mut self, frame_num: u32) -> Result<()> {
        self.sliding_window(frame_num)?;
        self.frames.push(RefFrame {
            picture: None,
            frame_num,
            long_term_idx: None,
        });
        self.prev_ref_frame_num = frame_num;
        Ok(())
    }

    /// `FrameNumWrap` / `PicNum` of a short-term frame seen from a picture
    /// with `current` `frame_num` (eqs. 8-27, 8-28).
    fn pic_num(&self, frame_num: u32, current: u32) -> i64 {
        if frame_num > current {
            i64::from(frame_num) - i64::from(self.max_frame_num)
        } else {
            i64::from(frame_num)
        }
    }

    /// The P reference list (`RefPicList0`) of a slice of the picture with
    /// `frame_num`: `count` entries after `modifications`. Entries are
    /// `None` where no reference picture exists (they must not be used).
    ///
    /// # Errors
    ///
    /// [`DecodeError::MissingReference`] for a modification that names no
    /// reference frame: the frame (or the long-term marking) was lost.
    pub fn ref_list(
        &self,
        frame_num: u32,
        count: usize,
        modifications: &[Modification],
    ) -> Result<Vec<Option<RefPicture>>> {
        // §8.2.4.2.1: short-term frames by descending PicNum, then
        // long-term frames by ascending LongTermPicNum.
        let mut short: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.long_term_idx.is_none())
            .collect();
        short.sort_by_key(|f| std::cmp::Reverse(self.pic_num(f.frame_num, frame_num)));
        let mut long: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.long_term_idx.is_some())
            .collect();
        long.sort_by_key(|f| f.long_term_idx);
        // One spare entry: the modification process works on count + 1.
        let mut list: Vec<Option<&RefFrame>> = short.into_iter().chain(long).map(Some).collect();
        list.resize(count + 1, None);
        self.modify(&mut list, frame_num, count, modifications)?;
        Ok(list[..count]
            .iter()
            .map(|entry| entry.and_then(|f| f.picture.clone()))
            .collect())
    }

    /// §8.2.4.3.1 and §8.2.4.3.2 on a list of `count + 1` entries.
    fn modify<'a>(
        &'a self,
        list: &mut [Option<&'a RefFrame>],
        frame_num: u32,
        count: usize,
        modifications: &[Modification],
    ) -> Result<()> {
        let max_pic_num = i64::from(self.max_frame_num);
        let current = i64::from(frame_num);
        let mut pred = current;
        for (ref_idx, &command) in modifications.iter().enumerate() {
            if ref_idx >= count {
                return Err(DecodeError::Malformed("too many list modifications"));
            }
            let target = match command {
                Modification::Subtract(diff) | Modification::Add(diff) => {
                    let diff = i64::from(diff) + 1;
                    if diff > max_pic_num {
                        return Err(DecodeError::Malformed("abs_diff_pic_num_minus1"));
                    }
                    // eqs. 8-34, 8-35: picNumLXNoWrap, then wrapped picNumLX.
                    let mut no_wrap = if matches!(command, Modification::Subtract(_)) {
                        pred - diff
                    } else {
                        pred + diff
                    };
                    if no_wrap < 0 {
                        no_wrap += max_pic_num;
                    } else if no_wrap >= max_pic_num {
                        no_wrap -= max_pic_num;
                    }
                    pred = no_wrap;
                    Target::ShortTerm(if no_wrap > current {
                        no_wrap - max_pic_num
                    } else {
                        no_wrap
                    })
                }
                Modification::LongTerm(long_term_pic_num) => Target::LongTerm(long_term_pic_num),
            };
            let is_target = |f: &RefFrame| match target {
                Target::ShortTerm(pic_num) => {
                    f.long_term_idx.is_none() && self.pic_num(f.frame_num, frame_num) == pic_num
                }
                Target::LongTerm(long_term_pic_num) => f.long_term_idx == Some(long_term_pic_num),
            };
            let frame = self
                .frames
                .iter()
                .find(|f| is_target(f))
                .ok_or(DecodeError::MissingReference)?;
            // Insert at ref_idx, then drop the later duplicate (eqs. 8-37, 8-38).
            list.copy_within(ref_idx..count, ref_idx + 1);
            list[ref_idx] = Some(frame);
            let mut next = ref_idx + 1;
            for c in ref_idx + 1..=count {
                if list[c].is_none_or(|f| !is_target(f)) {
                    list[next] = list[c];
                    next += 1;
                }
            }
        }
        Ok(())
    }

    /// Mark the decoded `picture` with `frame_num` (§8.2.5.1). Non-reference
    /// pictures are not kept.
    ///
    /// # Errors
    ///
    /// Operations naming no frame, or long-term indices above the maximum.
    pub fn mark(&mut self, frame_num: u32, picture: RefPicture, marking: &Marking) -> Result<()> {
        let mut long_term_idx = None;
        match marking {
            Marking::NonReference => return Ok(()),
            Marking::Idr { long_term } => {
                // §8.2.5.1: a long-term IDR sets MaxLongTermFrameIdx to 0.
                self.max_long_term_frame_idx = long_term.then_some(IDR_LONG_TERM_FRAME_IDX);
                long_term_idx = long_term.then_some(IDR_LONG_TERM_FRAME_IDX);
            }
            Marking::SlidingWindow => self.sliding_window(frame_num)?,
            Marking::Adaptive(operations) => {
                for &operation in operations {
                    if let Some(idx) = self.apply(frame_num, operation)? {
                        long_term_idx = Some(idx);
                    }
                }
            }
        }
        let frame_num = if marking.resets() { 0 } else { frame_num };
        if long_term_idx.is_none() {
            // A non-conforming stream can exceed max_num_ref_frames after
            // adaptive marking; the oldest short-term frame goes first.
            while self.frames.len() >= self.max_frames {
                self.remove_oldest_short_term(frame_num)?;
            }
        }
        self.frames.push(RefFrame {
            picture: Some(picture),
            frame_num,
            long_term_idx,
        });
        self.prev_ref_frame_num = frame_num;
        Ok(())
    }

    /// One memory management operation (§8.2.5.4); returns the current
    /// frame's `LongTermFrameIdx` for operation 6.
    fn apply(&mut self, frame_num: u32, operation: Mmco) -> Result<Option<u32>> {
        let short_term = |dpb: &Self, difference: u32| -> Result<usize> {
            // eq. 8-39: picNumX = CurrPicNum - (difference_of_pic_nums_minus1 + 1).
            let pic_num = i64::from(frame_num) - i64::from(difference) - 1;
            dpb.frames
                .iter()
                .position(|f| {
                    f.long_term_idx.is_none() && dpb.pic_num(f.frame_num, frame_num) == pic_num
                })
                .ok_or(DecodeError::Malformed("MMCO names no short-term frame"))
        };
        match operation {
            Mmco::ReleaseShortTerm(difference) => {
                let index = short_term(self, difference)?;
                self.frames.remove(index);
            }
            Mmco::ReleaseLongTerm(long_term_pic_num) => {
                self.frames
                    .retain(|f| f.long_term_idx != Some(long_term_pic_num));
            }
            Mmco::ShortTermToLongTerm {
                difference_of_pic_nums_minus1,
                long_term_frame_idx,
            } => {
                self.check_long_term_idx(long_term_frame_idx)?;
                short_term(self, difference_of_pic_nums_minus1)?;
                self.frames
                    .retain(|f| f.long_term_idx != Some(long_term_frame_idx));
                let index = short_term(self, difference_of_pic_nums_minus1)?;
                self.frames[index].long_term_idx = Some(long_term_frame_idx);
            }
            Mmco::MaxLongTermFrameIdx(plus1) => {
                self.max_long_term_frame_idx = plus1.checked_sub(1);
                let max = self.max_long_term_frame_idx;
                self.frames.retain(|f| {
                    f.long_term_idx
                        .is_none_or(|idx| max.is_some_and(|m| idx <= m))
                });
            }
            Mmco::ReleaseAll => self.release_all(),
            Mmco::CurrentToLongTerm(long_term_frame_idx) => {
                self.check_long_term_idx(long_term_frame_idx)?;
                self.frames
                    .retain(|f| f.long_term_idx != Some(long_term_frame_idx));
                return Ok(Some(long_term_frame_idx));
            }
        }
        Ok(None)
    }

    fn check_long_term_idx(&self, idx: u32) -> Result<()> {
        if self.max_long_term_frame_idx.is_none_or(|max| idx > max) {
            return Err(DecodeError::Malformed("LongTermFrameIdx above the maximum"));
        }
        Ok(())
    }

    /// §8.2.5.3: when the frames fill `max_num_ref_frames`, the short-term
    /// frame with the smallest `FrameNumWrap` becomes unused.
    fn sliding_window(&mut self, current: u32) -> Result<()> {
        if self.frames.len() >= self.max_frames {
            self.remove_oldest_short_term(current)?;
        }
        Ok(())
    }

    /// Release the short-term frame with the smallest `FrameNumWrap` seen
    /// from `frame_num` `current`.
    fn remove_oldest_short_term(&mut self, current: u32) -> Result<()> {
        let oldest = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.long_term_idx.is_none())
            .min_by_key(|(_, f)| self.pic_num(f.frame_num, current))
            .map(|(index, _)| index)
            .ok_or(DecodeError::Malformed("reference frames are all long-term"))?;
        self.frames.remove(oldest);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(id: u32) -> RefPicture {
        RefPicture {
            id,
            picture: Arc::new(Picture::new(16, 16, 0).unwrap()),
        }
    }

    fn ids(list: &[Option<RefPicture>]) -> Vec<Option<u32>> {
        list.iter().map(|r| r.as_ref().map(|r| r.id)).collect()
    }

    fn dpb(max_frames: usize) -> Dpb {
        let mut dpb = Dpb::default();
        dpb.start_idr(max_frames, 16);
        dpb
    }

    #[test]
    fn sliding_window_keeps_the_latest_frames_in_pic_num_order() {
        let mut dpb = dpb(3);
        dpb.mark(0, picture(0), &Marking::Idr { long_term: false })
            .unwrap();
        for n in 1..20u32 {
            dpb.mark(n % 16, picture(n), &Marking::SlidingWindow)
                .unwrap();
        }
        // frame_num 3 is current: 19 (3), 18 (2), 17 (1).
        let list = dpb.ref_list(4, 3, &[]).unwrap();
        assert_eq!(ids(&list), [Some(19), Some(18), Some(17)]);
    }

    #[test]
    fn modification_moves_a_frame_to_the_front_without_duplicates() {
        let mut dpb = dpb(4);
        dpb.mark(0, picture(0), &Marking::Idr { long_term: false })
            .unwrap();
        for n in 1..4u32 {
            dpb.mark(n, picture(n), &Marking::SlidingWindow).unwrap();
        }
        // Current frame_num 4: default order 3, 2, 1, 0. Move PicNum 1 first.
        let list = dpb.ref_list(4, 4, &[Modification::Subtract(2)]).unwrap();
        assert_eq!(ids(&list), [Some(1), Some(3), Some(2), Some(0)]);
    }

    #[test]
    fn long_term_frames_follow_short_term_and_are_selected_by_number() {
        let mut dpb = dpb(3);
        dpb.mark(0, picture(0), &Marking::Idr { long_term: true })
            .unwrap();
        for n in 1..6u32 {
            dpb.mark(n, picture(n), &Marking::SlidingWindow).unwrap();
        }
        let list = dpb.ref_list(6, 3, &[]).unwrap();
        assert_eq!(ids(&list), [Some(5), Some(4), Some(0)]);
        let list = dpb.ref_list(6, 2, &[Modification::LongTerm(0)]).unwrap();
        assert_eq!(ids(&list), [Some(0), Some(5)]);
    }

    #[test]
    fn mmco_operations() {
        let mut dpb = dpb(4);
        dpb.mark(0, picture(0), &Marking::Idr { long_term: false })
            .unwrap();
        dpb.mark(1, picture(1), &Marking::SlidingWindow).unwrap();
        dpb.mark(2, picture(2), &Marking::SlidingWindow).unwrap();
        let ops = vec![
            Mmco::MaxLongTermFrameIdx(2),
            Mmco::ShortTermToLongTerm {
                difference_of_pic_nums_minus1: 2,
                long_term_frame_idx: 1,
            },
            Mmco::ReleaseShortTerm(0),
            Mmco::CurrentToLongTerm(0),
        ];
        dpb.mark(3, picture(3), &Marking::Adaptive(ops)).unwrap();
        // Short-term 1; long-term 0 (picture 3) and 1 (picture 0).
        let list = dpb.ref_list(4, 3, &[]).unwrap();
        assert_eq!(ids(&list), [Some(1), Some(3), Some(0)]);
        dpb.mark(4, picture(4), &Marking::Adaptive(vec![Mmco::ReleaseAll]))
            .unwrap();
        assert_eq!(ids(&dpb.ref_list(1, 1, &[]).unwrap()), [Some(4)]);
    }

    #[test]
    fn gaps_insert_non_existing_frames() {
        let mut dpb = dpb(2);
        dpb.mark(0, picture(0), &Marking::Idr { long_term: false })
            .unwrap();
        assert_eq!(dpb.gap(3), [1, 2]);
        // A repeated frame_num: a whole cycle was lost.
        assert_eq!(dpb.gap(0).len(), 15);
        for n in [1, 2] {
            dpb.insert_gap_frame(n).unwrap();
        }
        assert_eq!(ids(&dpb.ref_list(3, 2, &[]).unwrap()), [None, None]);
        // A lost long-term picture: the modification names nothing.
        assert_eq!(
            dpb.ref_list(3, 1, &[Modification::LongTerm(1)])
                .unwrap_err(),
            DecodeError::MissingReference
        );
    }
}
