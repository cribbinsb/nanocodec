//! Reference marking and reference picture lists of frames and fields:
//! sliding-window and adaptive marking with all memory management
//! operations (§8.2.5), `frame_num` gap filling with "non-existing" frames
//! (§8.2.5.2), picture numbers (§8.2.4.1), P and B list initialisation for
//! frames (§8.2.4.2.1, §8.2.4.2.3) and fields (§8.2.4.2.2, §8.2.4.2.4,
//! §8.2.4.2.5) and list modification (§8.2.4.3).
//!
//! A reference frame is a frame store: one frame buffer whose two fields
//! are marked separately (a frame marks both; a field picture its own, the
//! second field of a pair joins its first field's store). Frame pictures
//! predict from stores whose two fields are both short-term or both
//! long-term, field pictures from single fields.
//!
//! Output order is not handled here: decoded pictures are shared with the
//! output queue (`output.rs`) through `Arc`s.

use std::sync::Arc;

use nc_common::{Picture, Sample};

use super::mbstate::MbState;
use super::structure::{Parity, Structure};
use super::{DecodeError, Result};

/// `LongTermFrameIdx` of an IDR picture with `long_term_reference_flag`.
const IDR_LONG_TERM_FRAME_IDX: u32 = 0;

/// The macroblock states of a decoded frame as its pictures were coded,
/// which the co-located macroblock of direct prediction is found in
/// (§8.4.1.2.1).
#[derive(Debug, Clone)]
pub(super) enum FrameMotion {
    /// Coded as one frame: its macroblocks in raster order.
    Frame(Arc<Vec<MbState>>),
    /// Coded as two field pictures: each field's macroblocks in raster
    /// order (`None`: that field was not decoded).
    Fields([Option<Arc<Vec<MbState>>>; 2]),
}

/// A frame or field picture a reference list entry names, with the
/// identity the deblocking filter compares ("different reference
/// pictures", §8.7.2.1), its order count and its frame's macroblock states
/// (§8.4.1.2.1).
#[derive(Debug, Clone)]
pub(super) struct RefPicture<T: Sample = u8> {
    /// Reference identity of the frame or field (`structure.rs`).
    pub id: u32,
    /// The frame buffer (a field is every other row of it).
    pub picture: Arc<Picture<T>>,
    /// The frame, or which of its fields.
    pub structure: Structure,
    /// `PicOrderCnt()` of the frame or field.
    pub poc: i64,
    /// Order counts of the frame's top and bottom fields.
    pub field_poc: [i64; 2],
    /// Whether it was a long-term reference when the list holding this
    /// copy was built (set by the list builders only).
    pub long_term: bool,
    /// Its frame's macroblock states.
    pub motion: FrameMotion,
}

/// A decoded reference picture to mark.
#[derive(Debug, Clone)]
pub(super) struct NewReference<T: Sample = u8> {
    /// Identity of its frame (the same for both fields of a pair).
    pub frame_id: u32,
    pub structure: Structure,
    /// Its frame buffer (holding the first field too for a second field).
    pub picture: Arc<Picture<T>>,
    /// Order counts (top, bottom; a field sets its own).
    pub poc: [i64; 2],
    /// The picture's macroblock states.
    pub motion: Arc<Vec<MbState>>,
}

/// The reference marking of one field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Unused,
    ShortTerm,
    /// Long-term with this `LongTermFrameIdx`.
    LongTerm(u32),
}

/// One reference frame store. Frames inferred for a `frame_num` gap
/// ("non-existing" frames) have no picture and must not be referenced.
#[derive(Debug, Clone)]
struct FrameStore<T: Sample> {
    id: u32,
    picture: Option<Arc<Picture<T>>>,
    motion: FrameMotion,
    frame_num: u32,
    /// Order counts of the top and bottom fields (0 for "non-existing").
    poc: [i64; 2],
    /// Which fields are decoded (a field pair's first field alone, or both).
    decoded: [bool; 2],
    marks: [Mark; 2],
}

impl<T: Sample> FrameStore<T> {
    fn mark(&self, parity: Parity) -> Mark {
        self.marks[parity.index()]
    }

    fn is_reference(&self) -> bool {
        self.marks != [Mark::Unused; 2]
    }

    /// A short-term frame (or complementary field pair): both fields.
    fn is_short_frame(&self) -> bool {
        self.marks == [Mark::ShortTerm; 2]
    }

    /// The `LongTermFrameIdx` of a long-term frame or pair (both fields).
    fn long_frame_idx(&self) -> Option<u32> {
        match self.marks {
            [Mark::LongTerm(a), Mark::LongTerm(b)] if a == b => Some(a),
            _ => None,
        }
    }

    fn has_short_field(&self) -> bool {
        self.marks.contains(&Mark::ShortTerm)
    }

    /// The smallest `LongTermFrameIdx` of its long-term fields.
    fn long_field_idx(&self) -> Option<u32> {
        self.marks
            .iter()
            .filter_map(|m| match m {
                Mark::LongTerm(idx) => Some(*idx),
                _ => None,
            })
            .min()
    }

    /// `PicOrderCnt()` of the frame or pair: the smaller field count.
    fn frame_poc(&self) -> i64 {
        self.poc[0].min(self.poc[1])
    }

    /// The order count a B field list sorts this store by: of its decoded
    /// fields, the smaller (§8.2.4.2.4: a lone first field's own).
    fn entry_poc(&self) -> i64 {
        Parity::BOTH
            .iter()
            .filter(|p| self.decoded[p.index()])
            .map(|p| self.poc[p.index()])
            .min()
            .unwrap_or(0)
    }

    /// Mark every field holding `LongTermFrameIdx` `idx` unused.
    fn release_long_term_idx(&mut self, idx: u32) {
        for mark in &mut self.marks {
            if *mark == Mark::LongTerm(idx) {
                *mark = Mark::Unused;
            }
        }
    }
}

/// The picture whose lists are built or whose marking is applied: what
/// picture numbers are relative to (§8.2.4.1).
#[derive(Debug, Clone, Copy)]
pub(super) struct ListPicture {
    pub frame_num: u32,
    pub structure: Structure,
    /// `PicOrderCnt()` of the picture (B lists).
    pub poc: i64,
}

/// A list entry: a frame store's frame or one of its fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    store: usize,
    structure: Structure,
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
    /// 1: release the short-term picture `difference_of_pic_nums_minus1 +
    /// 1` below the current picture number.
    ReleaseShortTerm(u32),
    /// 2: release the long-term picture with this `long_term_pic_num`.
    ReleaseLongTerm(u32),
    /// 3: make a short-term picture long-term frame `long_term_frame_idx`.
    ShortTermToLongTerm {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// 4: `max_long_term_frame_idx_plus1`.
    MaxLongTermFrameIdx(u32),
    /// 5: release every reference picture (and reset `frame_num` and POC).
    ReleaseAll,
    /// 6: make the current picture long-term frame `long_term_frame_idx`.
    CurrentToLongTerm(u32),
}

/// The picture a list modification or memory management operation names:
/// by `PicNum` or `LongTermPicNum`.
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
#[derive(Debug)]
pub(super) struct Dpb<T: Sample = u8> {
    frames: Vec<FrameStore<T>>,
    /// `MaxLongTermFrameIdx` (`None`: "no long-term frame indices").
    max_long_term_frame_idx: Option<u32>,
    /// `PrevRefFrameNum` (§7.4.3).
    prev_ref_frame_num: u32,
    /// `max_num_ref_frames` of the active SPS (at least 1).
    max_frames: usize,
    /// `MaxFrameNum` of the active SPS.
    max_frame_num: u32,
}

impl<T: Sample> Default for Dpb<T> {
    fn default() -> Self {
        Self {
            frames: Vec::new(),
            max_long_term_frame_idx: None,
            prev_ref_frame_num: 0,
            max_frames: 0,
            max_frame_num: 0,
        }
    }
}

impl<T: Sample> Dpb<T> {
    /// Test hook: overwrite every reference picture with `value`, as a
    /// decoder that lost a frame would hold wrong references.
    #[cfg(test)]
    pub(super) fn corrupt_pictures(&mut self, value: T) {
        for frame in &mut self.frames {
            if let Some(picture) = &mut frame.picture {
                let picture = Arc::make_mut(picture);
                for plane in nc_common::PlaneIndex::ALL {
                    picture.plane_mut(plane).view_mut().fill(value);
                }
                picture.extend_borders();
            }
        }
    }

    /// Start a coded video sequence at an IDR picture (§8.2.5.1: every
    /// reference picture becomes unused).
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

    /// Drop the stores none of whose fields is a reference any more.
    fn drop_unused(&mut self) {
        self.frames.retain(FrameStore::is_reference);
    }

    /// The `frame_num` values a picture with `frame_num` implies are missing
    /// (§8.2.5.2), in order; empty without a gap. Not asked for the second
    /// field of a pair, which repeats its first field's `frame_num`. A gap
    /// the SPS does not allow is an unintentional loss (§8.2.5.2): the
    /// frames are inferred all the same, so a picture predicting from one
    /// is refused ([`DecodeError::MissingReference`]) and the pictures
    /// that do not decode.
    ///
    /// A new frame or first field never repeats `PrevRefFrameNum`
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
        for frame in &mut self.frames {
            frame.release_long_term_idx(IDR_LONG_TERM_FRAME_IDX);
        }
        self.drop_unused();
    }

    /// Store a "non-existing" short-term frame for a gap `frame_num`
    /// (sliding window marking, §8.2.5.2).
    pub fn insert_gap_frame(&mut self, frame_num: u32) -> Result<()> {
        self.sliding_window(frame_num)?;
        self.frames.push(FrameStore {
            id: 0,
            picture: None,
            motion: FrameMotion::Fields([None, None]),
            frame_num,
            poc: [0; 2],
            decoded: [true; 2],
            marks: [Mark::ShortTerm; 2],
        });
        self.prev_ref_frame_num = frame_num;
        Ok(())
    }

    /// `FrameNumWrap` of a short-term store seen from a picture with
    /// `current` `frame_num` (eq. 8-27).
    fn frame_num_wrap(&self, frame_num: u32, current: u32) -> i64 {
        if frame_num > current {
            i64::from(frame_num) - i64::from(self.max_frame_num)
        } else {
            i64::from(frame_num)
        }
    }

    /// `PicNum` of a short-term entry (eqs. 8-28..8-30): a frame's
    /// `FrameNumWrap`; a field's doubled, plus one for the current field's
    /// parity.
    fn pic_num(&self, entry: Entry, cur: &ListPicture) -> i64 {
        let wrap = self.frame_num_wrap(self.frames[entry.store].frame_num, cur.frame_num);
        match entry.structure {
            Structure::Frame => wrap,
            Structure::Field(parity) => {
                2 * wrap + i64::from(Some(parity) == cur.structure.parity())
            }
        }
    }

    /// `LongTermPicNum` of a long-term entry (eqs. 8-29..8-32), as
    /// [`Self::pic_num`].
    fn long_term_pic_num(&self, entry: Entry, cur: &ListPicture) -> Option<u32> {
        let store = &self.frames[entry.store];
        match entry.structure {
            Structure::Frame => store.long_frame_idx(),
            Structure::Field(parity) => match store.mark(parity) {
                Mark::LongTerm(idx) => {
                    Some(2 * idx + u32::from(Some(parity) == cur.structure.parity()))
                }
                _ => None,
            },
        }
    }

    /// `CurrPicNum` and `MaxPicNum` (§7.4.3).
    fn current_pic_num(&self, cur: &ListPicture) -> (i64, i64) {
        let frame_num = i64::from(cur.frame_num);
        let max = i64::from(self.max_frame_num);
        if cur.structure.is_field() {
            (2 * frame_num + 1, 2 * max)
        } else {
            (frame_num, max)
        }
    }

    /// The P reference list (`RefPicList0`) of a slice of `cur`: `count`
    /// entries after `modifications`. Entries are `None` where no
    /// reference picture exists (they must not be used).
    ///
    /// # Errors
    ///
    /// [`DecodeError::MissingReference`] for a modification that names no
    /// reference picture: it (or its long-term marking) was lost.
    pub fn ref_list(
        &self,
        cur: &ListPicture,
        count: usize,
        modifications: &[Modification],
    ) -> Result<Vec<Option<RefPicture<T>>>> {
        let wrap = |i: &usize| self.frame_num_wrap(self.frames[*i].frame_num, cur.frame_num);
        let list = match cur.structure {
            Structure::Frame => {
                // §8.2.4.2.1: short-term frames by descending PicNum, then
                // long-term frames by ascending LongTermPicNum.
                let mut short = self.stores(FrameStore::is_short_frame);
                short.sort_by_key(|i| std::cmp::Reverse(wrap(i)));
                frames(short.into_iter().chain(self.long_frames()))
            }
            Structure::Field(parity) => {
                // §8.2.4.2.2: stores with a short-term field by descending
                // FrameNumWrap (the current frame's first field included),
                // stores with a long-term field by ascending
                // LongTermFrameIdx, each alternated into fields.
                let mut short = self.stores(FrameStore::has_short_field);
                short.sort_by_key(|i| std::cmp::Reverse(wrap(i)));
                let mut list = self.alternate(&short, parity, false);
                list.extend(self.alternate(&self.long_stores(), parity, true));
                list
            }
        };
        self.finish_list(list, cur, count, modifications)
    }

    /// The B reference lists (`RefPicList0`, `RefPicList1`) of a slice of
    /// `cur`: `counts` entries after `modifications`, as
    /// [`Dpb::ref_list`].
    ///
    /// # Errors
    ///
    /// As [`Dpb::ref_list`].
    pub fn b_ref_lists(
        &self,
        cur: &ListPicture,
        counts: [usize; 2],
        modifications: &[Vec<Modification>; 2],
    ) -> Result<[Vec<Option<RefPicture<T>>>; 2]> {
        // §8.2.4.2.3, §8.2.4.2.4: list 0 has the short-term pictures up to
        // the current one by descending order count, then those after it
        // ascending; list 1 the two halves the other way round; long-term
        // ones follow by ascending LongTermPicNum / LongTermFrameIdx.
        // Each short-term store with the order count it sorts by.
        let short: Vec<Ranked> = match cur.structure {
            Structure::Frame => self
                .stores(FrameStore::is_short_frame)
                .into_iter()
                .map(|i| (i, self.frames[i].frame_poc()))
                .collect(),
            Structure::Field(_) => self
                .stores(FrameStore::has_short_field)
                .into_iter()
                .map(|i| (i, self.frames[i].entry_poc()))
                .collect(),
        };
        let (mut before, mut after): (Vec<Ranked>, Vec<Ranked>) =
            short.into_iter().partition(|&(_, poc)| poc <= cur.poc);
        before.sort_by_key(|&(_, poc)| std::cmp::Reverse(poc));
        after.sort_by_key(|&(_, poc)| poc);
        let order0: Vec<usize> = before.iter().chain(&after).map(|&(i, _)| i).collect();
        let order1: Vec<usize> = after.iter().chain(&before).map(|&(i, _)| i).collect();
        let (list0, mut list1) = match cur.structure {
            Structure::Frame => {
                let long = self.long_frames();
                (
                    frames(order0.into_iter().chain(long.iter().copied())),
                    frames(order1.into_iter().chain(long)),
                )
            }
            Structure::Field(parity) => {
                let long = self.alternate(&self.long_stores(), parity, true);
                let mut list0 = self.alternate(&order0, parity, false);
                list0.extend(&long);
                let mut list1 = self.alternate(&order1, parity, false);
                list1.extend(long);
                (list0, list1)
            }
        };
        // A list 1 equal to list 0 (of more than one entry) swaps its first
        // two entries.
        if list1.len() > 1 && list1 == list0 {
            list1.swap(0, 1);
        }
        Ok([
            self.finish_list(list0, cur, counts[0], &modifications[0])?,
            self.finish_list(list1, cur, counts[1], &modifications[1])?,
        ])
    }

    /// The indices of the stores `keep` selects.
    fn stores(&self, keep: impl Fn(&FrameStore<T>) -> bool) -> Vec<usize> {
        (0..self.frames.len())
            .filter(|&i| keep(&self.frames[i]))
            .collect()
    }

    /// The long-term frames by ascending `LongTermPicNum`.
    fn long_frames(&self) -> Vec<usize> {
        let mut long = self.stores(|f| f.long_frame_idx().is_some());
        long.sort_by_key(|&i| self.frames[i].long_frame_idx());
        long
    }

    /// The stores with a long-term field by ascending `LongTermFrameIdx`.
    fn long_stores(&self) -> Vec<usize> {
        let mut long = self.stores(|f| f.long_field_idx().is_some());
        long.sort_by_key(|&i| self.frames[i].long_field_idx());
        long
    }

    /// §8.2.4.2.5: the reference fields of `stores` (short-term, or
    /// `long_term`) in their order, alternating parity from `parity`; once
    /// one parity runs out, the rest of the other follow in order.
    fn alternate(&self, stores: &[usize], parity: Parity, long_term: bool) -> Vec<Entry> {
        let usable = |store: usize, parity: Parity| {
            let mark = self.frames[store].mark(parity);
            if long_term {
                matches!(mark, Mark::LongTerm(_))
            } else {
                mark == Mark::ShortTerm
            }
        };
        let mut next = [0usize; 2];
        let mut out = Vec::new();
        let mut want = parity;
        loop {
            let cursor = &mut next[want.index()];
            while *cursor < stores.len() && !usable(stores[*cursor], want) {
                *cursor += 1;
            }
            if *cursor == stores.len() {
                // No more fields of this parity: the other's remaining ones.
                let other = want.opposite();
                for &store in &stores[next[other.index()]..] {
                    if usable(store, other) {
                        out.push(Entry {
                            store,
                            structure: Structure::Field(other),
                        });
                    }
                }
                return out;
            }
            out.push(Entry {
                store: stores[*cursor],
                structure: Structure::Field(want),
            });
            *cursor += 1;
            want = want.opposite();
        }
    }

    /// An initial list cut or padded to `count` entries and modified
    /// (§8.2.4.2, §8.2.4.3), as pictures.
    fn finish_list(
        &self,
        initial: Vec<Entry>,
        cur: &ListPicture,
        count: usize,
        modifications: &[Modification],
    ) -> Result<Vec<Option<RefPicture<T>>>> {
        // One spare entry: the modification process works on count + 1.
        let mut list: Vec<Option<Entry>> = initial.into_iter().map(Some).collect();
        list.resize(count + 1, None);
        self.modify(&mut list, cur, count, modifications)?;
        Ok(list[..count]
            .iter()
            .map(|entry| entry.and_then(|e| self.picture_of(e)))
            .collect())
    }

    /// The picture of a list entry (`None` for a "non-existing" frame).
    fn picture_of(&self, entry: Entry) -> Option<RefPicture<T>> {
        let store = &self.frames[entry.store];
        let picture = store.picture.clone()?;
        let (poc, long_term) = match entry.structure {
            Structure::Frame => (store.frame_poc(), store.long_frame_idx().is_some()),
            Structure::Field(parity) => (
                store.poc[parity.index()],
                matches!(store.mark(parity), Mark::LongTerm(_)),
            ),
        };
        Some(RefPicture {
            id: entry.structure.reference_id(store.id),
            picture,
            structure: entry.structure,
            poc,
            field_poc: store.poc,
            long_term,
            motion: store.motion.clone(),
        })
    }

    /// The entry of the current picture's structure that `target` names.
    fn find(&self, target: Target, cur: &ListPicture) -> Option<Entry> {
        let structures: &[Structure] = match cur.structure {
            Structure::Frame => &[Structure::Frame],
            Structure::Field(_) => &[
                Structure::Field(Parity::Top),
                Structure::Field(Parity::Bottom),
            ],
        };
        (0..self.frames.len())
            .flat_map(|store| {
                structures
                    .iter()
                    .map(move |&structure| Entry { store, structure })
            })
            .find(|&entry| self.is_target(entry, target, cur))
    }

    /// Whether `entry` is the picture `target` names.
    fn is_target(&self, entry: Entry, target: Target, cur: &ListPicture) -> bool {
        let store = &self.frames[entry.store];
        match target {
            Target::ShortTerm(pic_num) => {
                let short = match entry.structure {
                    Structure::Frame => store.is_short_frame(),
                    Structure::Field(parity) => store.mark(parity) == Mark::ShortTerm,
                };
                short && self.pic_num(entry, cur) == pic_num
            }
            Target::LongTerm(long_term_pic_num) => {
                self.long_term_pic_num(entry, cur) == Some(long_term_pic_num)
            }
        }
    }

    /// §8.2.4.3.1 and §8.2.4.3.2 on a list of `count + 1` entries.
    fn modify(
        &self,
        list: &mut [Option<Entry>],
        cur: &ListPicture,
        count: usize,
        modifications: &[Modification],
    ) -> Result<()> {
        let (current, max_pic_num) = self.current_pic_num(cur);
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
            let entry = self
                .find(target, cur)
                .ok_or(DecodeError::MissingReference)?;
            // Insert at ref_idx, then drop the later duplicate (eqs. 8-37, 8-38).
            list.copy_within(ref_idx..count, ref_idx + 1);
            list[ref_idx] = Some(entry);
            let mut next = ref_idx + 1;
            for c in ref_idx + 1..=count {
                if list[c].is_none_or(|e| !self.is_target(e, target, cur)) {
                    list[next] = list[c];
                    next += 1;
                }
            }
        }
        Ok(())
    }

    /// Mark the decoded reference picture `current` with `frame_num`
    /// (§8.2.5.1). Non-reference pictures are not kept. The second field of
    /// a pair joins its first field's store when that is still a reference.
    ///
    /// # Errors
    ///
    /// Operations naming no picture, or long-term indices above the maximum.
    pub fn mark(
        &mut self,
        frame_num: u32,
        current: NewReference<T>,
        marking: &Marking,
    ) -> Result<()> {
        let cur = ListPicture {
            frame_num,
            structure: current.structure,
            poc: 0,
        };
        let second_field =
            current.structure.is_field() && self.frames.iter().any(|f| f.id == current.frame_id);
        let mut long_term_idx = None;
        match marking {
            Marking::NonReference => return Ok(()),
            Marking::Idr { long_term } => {
                // §8.2.5.1: a long-term IDR sets MaxLongTermFrameIdx to 0.
                self.max_long_term_frame_idx = long_term.then_some(IDR_LONG_TERM_FRAME_IDX);
                long_term_idx = long_term.then_some(IDR_LONG_TERM_FRAME_IDX);
            }
            // §8.2.5.3 is not applied to the second field of a pair.
            Marking::SlidingWindow if second_field => {}
            Marking::SlidingWindow => self.sliding_window(frame_num)?,
            Marking::Adaptive(operations) => {
                for &operation in operations {
                    if let Some(idx) = self.apply(&cur, current.frame_id, operation)? {
                        long_term_idx = Some(idx);
                    }
                    self.drop_unused();
                }
            }
        }
        let frame_num = if marking.resets() { 0 } else { frame_num };
        let mark = long_term_idx.map_or(Mark::ShortTerm, Mark::LongTerm);
        self.prev_ref_frame_num = frame_num;
        if let Structure::Field(parity) = current.structure
            && let Some(store) = self.frames.iter_mut().find(|f| f.id == current.frame_id)
        {
            let p = parity.index();
            store.marks[p] = mark;
            store.picture = Some(current.picture);
            store.poc[p] = current.poc[p];
            store.decoded[p] = true;
            match &mut store.motion {
                FrameMotion::Fields(fields) => fields[p] = Some(current.motion),
                FrameMotion::Frame(_) => {
                    return Err(DecodeError::Malformed("field of a frame picture"));
                }
            }
            return Ok(());
        }
        if long_term_idx.is_none() {
            // A non-conforming stream can exceed max_num_ref_frames after
            // adaptive marking; the oldest short-term frame goes first.
            while self.frames.len() >= self.max_frames {
                self.remove_oldest_short_term(frame_num)?;
            }
        }
        let (marks, decoded, motion) = match current.structure {
            Structure::Frame => ([mark; 2], [true; 2], FrameMotion::Frame(current.motion)),
            Structure::Field(parity) => {
                let mut marks = [Mark::Unused; 2];
                let mut decoded = [false; 2];
                let mut fields = [None, None];
                marks[parity.index()] = mark;
                decoded[parity.index()] = true;
                fields[parity.index()] = Some(current.motion);
                (marks, decoded, FrameMotion::Fields(fields))
            }
        };
        self.frames.push(FrameStore {
            id: current.frame_id,
            picture: Some(current.picture),
            motion,
            frame_num,
            poc: current.poc,
            decoded,
            marks,
        });
        Ok(())
    }

    /// The entry `picNumX` = `CurrPicNum − (difference_of_pic_nums_minus1 +
    /// 1)` names (eq. 8-39).
    fn short_term_target(&self, cur: &ListPicture, difference: u32) -> Result<Entry> {
        let (current, _) = self.current_pic_num(cur);
        let pic_num = current - i64::from(difference) - 1;
        self.find(Target::ShortTerm(pic_num), cur)
            .ok_or(DecodeError::Malformed("MMCO names no short-term picture"))
    }

    /// Set the marking of `entry` (both fields of a frame).
    fn set_mark(&mut self, entry: Entry, mark: Mark) {
        let store = &mut self.frames[entry.store];
        match entry.structure {
            Structure::Frame => store.marks = [mark; 2],
            Structure::Field(parity) => store.marks[parity.index()] = mark,
        }
    }

    /// One memory management operation (§8.2.5.4) of the picture `cur` of
    /// frame `current_frame`; returns the current picture's
    /// `LongTermFrameIdx` for operation 6.
    fn apply(
        &mut self,
        cur: &ListPicture,
        current_frame: u32,
        operation: Mmco,
    ) -> Result<Option<u32>> {
        match operation {
            Mmco::ReleaseShortTerm(difference) => {
                let entry = self.short_term_target(cur, difference)?;
                self.set_mark(entry, Mark::Unused);
            }
            Mmco::ReleaseLongTerm(long_term_pic_num) => {
                if let Some(entry) = self.find(Target::LongTerm(long_term_pic_num), cur) {
                    self.set_mark(entry, Mark::Unused);
                }
            }
            Mmco::ShortTermToLongTerm {
                difference_of_pic_nums_minus1,
                long_term_frame_idx,
            } => {
                self.check_long_term_idx(long_term_frame_idx)?;
                let entry = self.short_term_target(cur, difference_of_pic_nums_minus1)?;
                // The index leaves every other frame (a field keeps it only
                // when it pairs with the picture being marked).
                for (i, frame) in self.frames.iter_mut().enumerate() {
                    if i != entry.store || entry.structure == Structure::Frame {
                        frame.release_long_term_idx(long_term_frame_idx);
                    }
                }
                self.set_mark(entry, Mark::LongTerm(long_term_frame_idx));
            }
            Mmco::MaxLongTermFrameIdx(plus1) => {
                self.max_long_term_frame_idx = plus1.checked_sub(1);
                let max = self.max_long_term_frame_idx;
                for frame in &mut self.frames {
                    for mark in &mut frame.marks {
                        if let Mark::LongTerm(idx) = *mark
                            && max.is_none_or(|m| idx > m)
                        {
                            *mark = Mark::Unused;
                        }
                    }
                }
            }
            Mmco::ReleaseAll => self.release_all(),
            Mmco::CurrentToLongTerm(long_term_frame_idx) => {
                self.check_long_term_idx(long_term_frame_idx)?;
                // The first field of the current frame keeps its index.
                for frame in &mut self.frames {
                    if frame.id != current_frame || cur.structure == Structure::Frame {
                        frame.release_long_term_idx(long_term_frame_idx);
                    }
                }
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

    /// §8.2.5.3: when the reference frames, pairs and lone fields fill
    /// `max_num_ref_frames`, the short-term one with the smallest
    /// `FrameNumWrap` becomes unused.
    fn sliding_window(&mut self, current: u32) -> Result<()> {
        if self.frames.len() >= self.max_frames {
            self.remove_oldest_short_term(current)?;
        }
        Ok(())
    }

    /// Release the short-term fields of the store with the smallest
    /// `FrameNumWrap` seen from `frame_num` `current`.
    fn remove_oldest_short_term(&mut self, current: u32) -> Result<()> {
        let oldest = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.has_short_field())
            .min_by_key(|(_, f)| self.frame_num_wrap(f.frame_num, current))
            .map(|(index, _)| index)
            .ok_or(DecodeError::Malformed("reference frames are all long-term"))?;
        for mark in &mut self.frames[oldest].marks {
            if *mark == Mark::ShortTerm {
                *mark = Mark::Unused;
            }
        }
        self.drop_unused();
        Ok(())
    }
}

/// A store index with the order count a B list sorts it by.
type Ranked = (usize, i64);

/// Frame entries of `stores`.
fn frames(stores: impl IntoIterator<Item = usize>) -> Vec<Entry> {
    stores
        .into_iter()
        .map(|store| Entry {
            store,
            structure: Structure::Frame,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(id: u32, structure: Structure, poc: i64) -> NewReference {
        NewReference {
            frame_id: id,
            structure,
            picture: Arc::new(Picture::new(16, 16, 0).unwrap()),
            poc: [poc; 2],
            motion: Arc::new(Vec::new()),
        }
    }

    fn picture(id: u32) -> NewReference {
        reference(id, Structure::Frame, i64::from(id))
    }

    fn frame(frame_num: u32) -> ListPicture {
        ListPicture {
            frame_num,
            structure: Structure::Frame,
            poc: 0,
        }
    }

    /// Frame identities of a frame list.
    fn ids(list: &[Option<RefPicture>]) -> Vec<Option<u32>> {
        list.iter()
            .map(|r| r.as_ref().map(|r| super::super::structure::frame_of(r.id)))
            .collect()
    }

    /// (frame identity, top field) of a field list.
    fn fields(list: &[Option<RefPicture>]) -> Vec<(u32, bool)> {
        list.iter()
            .map(|r| {
                let r = r.as_ref().unwrap();
                (
                    super::super::structure::frame_of(r.id),
                    r.structure == Structure::Field(Parity::Top),
                )
            })
            .collect()
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
        let list = dpb.ref_list(&frame(4), 3, &[]).unwrap();
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
        let list = dpb
            .ref_list(&frame(4), 4, &[Modification::Subtract(2)])
            .unwrap();
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
        let list = dpb.ref_list(&frame(6), 3, &[]).unwrap();
        assert_eq!(ids(&list), [Some(5), Some(4), Some(0)]);
        let list = dpb
            .ref_list(&frame(6), 2, &[Modification::LongTerm(0)])
            .unwrap();
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
        let list = dpb.ref_list(&frame(4), 3, &[]).unwrap();
        assert_eq!(ids(&list), [Some(1), Some(3), Some(0)]);
        dpb.mark(4, picture(4), &Marking::Adaptive(vec![Mmco::ReleaseAll]))
            .unwrap();
        assert_eq!(ids(&dpb.ref_list(&frame(1), 1, &[]).unwrap()), [Some(4)]);
    }

    #[test]
    fn b_lists_order_by_order_count_around_the_current_frame() {
        let mut dpb = dpb(4);
        dpb.mark(0, picture(0), &Marking::Idr { long_term: false })
            .unwrap();
        for (n, id) in [(1, 8), (2, 4), (3, 2)] {
            dpb.mark(n, picture(id), &Marking::SlidingWindow).unwrap();
        }
        let none = [Vec::new(), Vec::new()];
        let cur = ListPicture { poc: 3, ..frame(4) };
        let [l0, l1] = dpb.b_ref_lists(&cur, [4, 4], &none).unwrap();
        assert_eq!(ids(&l0), [Some(2), Some(0), Some(4), Some(8)]);
        assert_eq!(ids(&l1), [Some(4), Some(8), Some(2), Some(0)]);
        // Every reference after the current frame: the lists are equal and
        // list 1 swaps its first two entries.
        let cur = ListPicture {
            poc: -1,
            ..frame(4)
        };
        let [l0, l1] = dpb.b_ref_lists(&cur, [2, 2], &none).unwrap();
        assert_eq!(ids(&l0), [Some(0), Some(2)]);
        assert_eq!(ids(&l1), [Some(2), Some(0)]);
    }

    #[test]
    fn field_lists_alternate_parity_from_the_current_field() {
        let mut dpb = dpb(4);
        let top = Structure::Field(Parity::Top);
        let bottom = Structure::Field(Parity::Bottom);
        // Frame 1 (frame_num 0) as two fields, frame 2 (frame_num 1) as a
        // frame, then the top field of frame 3 (frame_num 2).
        dpb.mark(0, reference(1, top, 0), &Marking::Idr { long_term: false })
            .unwrap();
        dpb.mark(0, reference(1, bottom, 1), &Marking::SlidingWindow)
            .unwrap();
        dpb.mark(
            1,
            reference(2, Structure::Frame, 4),
            &Marking::SlidingWindow,
        )
        .unwrap();
        dpb.mark(2, reference(3, top, 8), &Marking::SlidingWindow)
            .unwrap();
        // The bottom field of frame 3: same parity first, the first field
        // of its own frame (highest FrameNumWrap) leads the top fields.
        let cur = ListPicture {
            frame_num: 2,
            structure: bottom,
            poc: 9,
        };
        let list = dpb.ref_list(&cur, 5, &[]).unwrap();
        assert_eq!(
            fields(&list),
            [(2, false), (3, true), (1, false), (2, true), (1, true)]
        );
        // PicNum of the same-parity field of frame 2 is 2 * 1 + 1 = 3:
        // CurrPicNum 5 minus 2.
        let list = dpb.ref_list(&cur, 2, &[Modification::Subtract(1)]).unwrap();
        assert_eq!(fields(&list), [(2, false), (3, true)]);
        let list = dpb.ref_list(&cur, 2, &[Modification::Subtract(0)]).unwrap();
        assert_eq!(fields(&list), [(3, true), (2, false)]);
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
        assert_eq!(ids(&dpb.ref_list(&frame(3), 2, &[]).unwrap()), [None, None]);
        // A lost long-term picture: the modification names nothing.
        assert_eq!(
            dpb.ref_list(&frame(3), 1, &[Modification::LongTerm(1)])
                .unwrap_err(),
            DecodeError::MissingReference
        );
    }
}
