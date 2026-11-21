//! An undo journal of adaptive-CDF writes, for the RD partition search's
//! node trials (`rd_split.rs`) in the codecs that adapt their CDFs symbol
//! by symbol (AV1 `backend/node_trial.rs`).
//!
//! Why: a node trial must put the CDFs back as they were when it began, and
//! a kept coding's CDFs back when it wins. Copying the whole CDF context
//! (tens of KiB with its symbol-cost caches) at every trial and undo was
//! 16.5 % of AV1 Default's instructions with the search on, about half of
//! what the search added, while a trial adapts only the CDFs its symbols
//! use (measured gain: docs/handoff/av1.md "RD partition search").
//!
//! How: before a CDF adapts, the coder records its whole state (the codec's
//! [`Journaled::State`], wide enough for every alphabet) and its place: the
//! byte offset of the CDF within the context, from addresses, which are
//! stable while the context is journaled (it is boxed and never moved
//! during a trial). The codec's context finds the CDF at an offset through
//! [`Journaled::swap_at`]: arrays by index arithmetic, structs by field
//! offset ([`journaled_fields!`]).
//!
//! Every operation is an exchange of an entry with the CDF it names, so
//! the journal is its own redo log: undoing a run (exchanging its entries
//! newest first) leaves each entry holding the state its write produced,
//! and exchanging them again oldest first redoes the run and turns them
//! back into undo entries. A kept coding is a run undone and set aside
//! ([`UndoJournal::undo_keeping`]), put back by [`UndoJournal::redo`].
//!
//! Exact: entries hold the whole state, cost caches included, so an undone
//! CDF is bit for bit what it was. A CDF outside the context (a scratch
//! copy) is not recorded, as a whole-context copy never restored it either.

/// A context whose adaptive CDFs a journal restores (the codec's CDF set,
/// its nested tables, arrays of them, and each CDF).
pub trait Journaled {
    /// The state of any one CDF of the set (probabilities, counter and cost
    /// cache), in one type for every alphabet size.
    type State;

    /// Exchange the CDF that starts `offset` bytes into `self` with
    /// `state`.
    ///
    /// # Panics
    ///
    /// If no CDF starts at `offset`: the journal recorded a place that is
    /// not one of this context's CDFs.
    fn swap_at(&mut self, offset: usize, state: &mut Self::State);
}

impl<T: Journaled, const M: usize> Journaled for [T; M] {
    type State = T::State;

    #[inline]
    fn swap_at(&mut self, offset: usize, state: &mut T::State) {
        let size = size_of::<T>();
        self[offset / size].swap_at(offset % size, state);
    }
}

/// Implements [`Journaled`] for a struct of journaled fields by field
/// offset: `journaled_fields!(Type, State; field, ...)`. The fields are
/// searched in the order given, so list the most often coded first. Used
/// inside the struct's own module, where its fields are visible.
#[macro_export]
macro_rules! journaled_fields {
    ($ty:ty, $state:ty; $($field:ident),+ $(,)?) => {
        impl $crate::undo_journal::Journaled for $ty {
            type State = $state;

            #[inline]
            fn swap_at(&mut self, offset: usize, state: &mut $state) {
                $(
                    let start = ::core::mem::offset_of!($ty, $field);
                    let size = ::core::mem::size_of_val(&self.$field);
                    if let Some(inner) = offset.checked_sub(start).filter(|&inner| inner < size) {
                        return $crate::undo_journal::Journaled::swap_at(
                            &mut self.$field,
                            inner,
                            state,
                        );
                    }
                )+
                panic!("no CDF of {} at offset {offset}", stringify!($ty));
            }
        }
    };
}

/// One recorded write: where, and the state before it (or, once undone
/// for a kept coding, after it).
#[derive(Debug, Clone, Copy)]
struct Entry<S> {
    offset: u32,
    state: S,
}

/// The writes of a coding set aside for a later redo
/// ([`UndoJournal::undo_keeping`]); buffers reused coding to coding.
#[derive(Debug, Clone)]
pub struct KeptRun<S> {
    entries: Vec<Entry<S>>,
}

impl<S> Default for KeptRun<S> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

/// The journal of one CDF context (module docs). Inactive until
/// [`Self::start`], so coding outside trials records nothing.
#[derive(Debug, Clone)]
pub struct UndoJournal<S> {
    entries: Vec<Entry<S>>,
    /// Address of the journaled context.
    base: usize,
    /// Size of the journaled context in bytes (0: inactive).
    len: usize,
}

impl<S> Default for UndoJournal<S> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            base: 0,
            len: 0,
        }
    }
}

/// The address of `value`, to place it within a journaled context.
#[inline]
fn address<T: ?Sized>(value: &T) -> usize {
    std::ptr::from_ref(value).cast::<u8>().addr()
}

impl<S> UndoJournal<S> {
    /// Record the writes to `context`'s CDFs from now on.
    pub fn start<T: Journaled<State = S>>(&mut self, context: &T) {
        debug_assert!(self.entries.is_empty());
        self.base = address(context);
        self.len = size_of::<T>();
    }

    /// Stop recording and forget the entries (the outermost trial ended).
    pub fn stop(&mut self) {
        self.entries.clear();
        self.len = 0;
    }

    /// Whether writes are being recorded.
    #[inline]
    #[must_use]
    pub fn active(&self) -> bool {
        self.len != 0
    }

    /// Note that the CDF `cdf` is about to change, with its current state
    /// `state()`. A CDF outside the journaled context is not recorded.
    #[inline]
    pub fn record<C>(&mut self, cdf: &C, state: impl FnOnce() -> S) {
        let offset = address(cdf).wrapping_sub(self.base);
        if offset < self.len {
            self.entries.push(Entry {
                offset: offset as u32,
                state: state(),
            });
        }
    }

    /// A mark: the writes after it can be undone.
    #[must_use]
    pub fn mark(&self) -> usize {
        self.entries.len()
    }

    /// Put `context`'s CDFs back as they were at `mark`, forgetting the
    /// writes after it.
    pub fn undo<T: Journaled<State = S>>(&mut self, mark: usize, context: &mut T) {
        self.exchange_newest_first(mark, context);
        self.entries.truncate(mark);
    }

    /// As [`Self::undo`], but set the undone writes aside in `kept`, to be
    /// redone by [`Self::redo`].
    pub fn undo_keeping<T: Journaled<State = S>>(
        &mut self,
        mark: usize,
        context: &mut T,
        kept: &mut KeptRun<S>,
    ) {
        self.exchange_newest_first(mark, context);
        kept.entries.clear();
        kept.entries.extend(self.entries.drain(mark..));
    }

    /// Redo the writes `kept` holds onto `context`, recording them again
    /// (so they can be undone in turn), and empty `kept`.
    pub fn redo<T: Journaled<State = S>>(&mut self, kept: &mut KeptRun<S>, context: &mut T) {
        self.check_context(context);
        for mut entry in kept.entries.drain(..) {
            context.swap_at(entry.offset as usize, &mut entry.state);
            self.entries.push(entry);
        }
    }

    /// Exchange the entries after `mark` with their CDFs, newest first.
    fn exchange_newest_first<T: Journaled<State = S>>(&mut self, mark: usize, context: &mut T) {
        self.check_context(context);
        for entry in self.entries[mark..].iter_mut().rev() {
            context.swap_at(entry.offset as usize, &mut entry.state);
        }
    }

    /// The offsets are `context`'s: it is the context journaled, unmoved.
    fn check_context<T: Journaled<State = S>>(&self, context: &T) {
        assert!(
            self.active() && address(context) == self.base,
            "the journaled CDF context moved or is not journaled"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy CDF: its state is its value.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Toy(u32);

    impl Journaled for Toy {
        type State = u32;
        fn swap_at(&mut self, offset: usize, state: &mut u32) {
            assert_eq!(offset, 0);
            std::mem::swap(&mut self.0, state);
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Set {
        one: Toy,
        rows: [[Toy; 3]; 2],
    }

    journaled_fields!(Set, u32; rows, one);

    /// Write `value` to `cdf`, recording it.
    fn write(journal: &mut UndoJournal<u32>, cdf: &mut Toy, value: u32) {
        journal.record(cdf, || cdf.0);
        cdf.0 = value;
    }

    fn set() -> Set {
        Set {
            one: Toy(1),
            rows: [[Toy(10), Toy(11), Toy(12)], [Toy(20), Toy(21), Toy(22)]],
        }
    }

    #[test]
    fn undo_restores_nested_marks_and_ignores_outside_writes() {
        let mut context = Box::new(set());
        let start = (*context).clone();
        let mut journal = UndoJournal::default();
        journal.start(&*context);
        write(&mut journal, &mut context.one, 2);
        let outer = journal.mark();
        write(&mut journal, &mut context.rows[1][2], 5);
        let after_outer = (*context).clone();
        let inner = journal.mark();
        write(&mut journal, &mut context.rows[1][2], 6);
        write(&mut journal, &mut context.rows[0][0], 7);
        let mut outside = Toy(0);
        write(&mut journal, &mut outside, 9);
        journal.undo(inner, &mut *context);
        assert_eq!(*context, after_outer);
        journal.undo(outer, &mut *context);
        journal.undo(0, &mut *context);
        assert_eq!(*context, start);
        journal.stop();
        assert!(!journal.active());
    }

    #[test]
    fn a_kept_run_is_redone_and_undoable_again() {
        let mut context = Box::new(set());
        let start = (*context).clone();
        let mut journal = UndoJournal::default();
        let mut kept = KeptRun::default();
        journal.start(&*context);
        write(&mut journal, &mut context.rows[0][1], 3);
        write(&mut journal, &mut context.rows[0][1], 4);
        write(&mut journal, &mut context.one, 8);
        let coded = (*context).clone();
        journal.undo_keeping(0, &mut *context, &mut kept);
        assert_eq!(*context, start);
        write(&mut journal, &mut context.rows[1][0], 30);
        journal.undo(0, &mut *context);
        journal.redo(&mut kept, &mut *context);
        assert_eq!(*context, coded);
        journal.undo(0, &mut *context);
        assert_eq!(*context, start);
    }
}
