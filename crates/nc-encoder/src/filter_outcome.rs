//! What each frame-level filter search ended up deciding, per unit (CTB or
//! superblock): the cheap counters behind a usage-driven search density.
//!
//! A codec's filter search (HEVC/VVC SAO, VVC ALF, AV1/AV2 CDEF) gathers
//! statistics on every unit it searches. When the unit then ends with the
//! filter off, or with the same parameters as the co-located unit of the
//! previous picture, those statistics bought nothing a cheaper rule (reuse,
//! or skip) would not have. The backend counts its units by outcome
//! ([`CodecBackend::filter_outcomes`](crate::CodecBackend::filter_outcomes));
//! the engine sums the committed frames by frame role
//! ([`Encoder::filter_outcome_stats`](crate::Encoder::filter_outcome_stats)).

use std::collections::BTreeMap;

/// A frame-level filter whose parameters the encoder searches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FilterTool {
    /// Sample adaptive offset (HEVC, VVC).
    Sao,
    /// Adaptive loop filter (VVC).
    Alf,
    /// Constrained directional enhancement filter (AV1, AV2).
    Cdef,
}

impl FilterTool {
    /// Lower-case name for reports.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sao => "sao",
            Self::Alf => "alf",
            Self::Cdef => "cdef",
        }
    }
}

/// A unit whose residual-free share of its area is at least this is
/// "quiet": most of it copies its reference with no residual, so the
/// filter sees little new texture there.
pub const QUIET_SHARE: f32 = 0.75;

/// The condition on a searched unit's own history (and content) that its
/// outcome is also counted under: how well the history predicts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Given {
    /// Every searched unit with a history.
    Any,
    /// The unit is quiet ([`QUIET_SHARE`]).
    Quiet,
    /// The unit ended off at its last search.
    OffLast,
    /// ... at its last two searches.
    OffLast2,
    /// ... at its last four searches.
    OffLast4,
    /// Off at its last search, and quiet now.
    OffLastQuiet,
    /// On at its last search.
    OnLast,
    /// On and unchanged at its last search.
    UnchangedLast,
    /// ... at its last two searches.
    UnchangedLast2,
}

impl Given {
    /// Every condition, in report order.
    pub const ALL: [Self; 9] = [
        Self::Any,
        Self::Quiet,
        Self::OffLast,
        Self::OffLast2,
        Self::OffLast4,
        Self::OffLastQuiet,
        Self::OnLast,
        Self::UnchangedLast,
        Self::UnchangedLast2,
    ];

    /// Lower-case name for reports.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Quiet => "quiet",
            Self::OffLast => "off1",
            Self::OffLast2 => "off2",
            Self::OffLast4 => "off4",
            Self::OffLastQuiet => "off1_quiet",
            Self::OnLast => "on1",
            Self::UnchangedLast => "unchanged1",
            Self::UnchangedLast2 => "unchanged2",
        }
    }

    fn holds(self, history: UnitHistory, quiet: bool) -> bool {
        match self {
            Self::Any => true,
            Self::Quiet => quiet,
            Self::OffLast => history.off_run >= 1,
            Self::OffLast2 => history.off_run >= 2,
            Self::OffLast4 => history.off_run >= 4,
            Self::OffLastQuiet => history.off_run >= 1 && quiet,
            Self::OnLast => history.off_run == 0,
            Self::UnchangedLast => history.unchanged_run >= 1,
            Self::UnchangedLast2 => history.unchanged_run >= 2,
        }
    }
}

/// Searched units meeting one [`Given`] condition, by outcome.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Units meeting the condition.
    pub units: u64,
    /// Of them, units that ended off.
    pub off: u64,
    /// Of them, units that ended on with unchanged parameters.
    pub unchanged: u64,
}

/// One picture's (or a sum of pictures') units by search outcome.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterOutcome {
    /// Every unit of the picture.
    pub units: u64,
    /// Units whose statistics were gathered.
    pub searched: u64,
    /// Searched units that ended with the filter off.
    pub searched_off: u64,
    /// Searched units that ended on, with the co-located unit's parameters
    /// of the previous picture.
    pub searched_unchanged: u64,
    /// Searched units with a history, by [`Given`] condition (indexed as
    /// [`Given::ALL`]).
    pub given: [Tally; Given::ALL.len()],
}

impl FilterOutcome {
    /// Count one unit: whether it was searched, whether the filter is on,
    /// and whether its parameters equal the previous picture's co-located
    /// unit's.
    pub fn count(&mut self, searched: bool, on: bool, unchanged: bool) {
        self.units += 1;
        if searched {
            self.searched += 1;
            if !on {
                self.searched_off += 1;
            } else if unchanged {
                self.searched_unchanged += 1;
            }
        }
    }

    /// Count a searched unit's outcome under each condition its `history`
    /// and `quiet` meet.
    fn count_given(&mut self, history: UnitHistory, quiet: bool, on: bool, unchanged: bool) {
        for (tally, given) in self.given.iter_mut().zip(Given::ALL) {
            if given.holds(history, quiet) {
                tally.units += 1;
                tally.off += u64::from(!on);
                tally.unchanged += u64::from(on && unchanged);
            }
        }
    }

    fn add(&mut self, other: &Self) {
        self.units += other.units;
        self.searched += other.searched;
        self.searched_off += other.searched_off;
        self.searched_unchanged += other.searched_unchanged;
        for (tally, other) in self.given.iter_mut().zip(&other.given) {
            tally.units += other.units;
            tally.off += other.off;
            tally.unchanged += other.unchanged;
        }
    }
}

/// A unit's outcomes at its last searches: runs of consecutive searches
/// that ended off, or on and unchanged (saturating). Pictures that did not
/// search the unit leave its history alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UnitHistory {
    /// Whether the unit has been searched at all.
    known: bool,
    off_run: u8,
    unchanged_run: u8,
}

impl UnitHistory {
    fn after(self, on: bool, unchanged: bool) -> Self {
        Self {
            known: true,
            off_run: if on {
                0
            } else {
                self.off_run.saturating_add(1)
            },
            unchanged_run: if on && unchanged {
                self.unchanged_run.saturating_add(1)
            } else {
                0
            },
        }
    }
}

/// A backend's outcome counter for one filter: each picture's units against
/// the last committed picture's, under the recode contract (a re-coded or
/// discarded picture does not become the previous one). `T` is what makes
/// two co-located units' parameters equal.
#[derive(Debug, Clone)]
pub struct OutcomeTracker<T> {
    previous: Vec<T>,
    history: Vec<UnitHistory>,
    pending: Option<(Vec<T>, Vec<UnitHistory>)>,
    frame: Option<FilterOutcome>,
}

impl<T> Default for OutcomeTracker<T> {
    fn default() -> Self {
        Self {
            previous: Vec::new(),
            history: Vec::new(),
            pending: None,
            frame: None,
        }
    }
}

impl<T: PartialEq> OutcomeTracker<T> {
    /// Forget the last coding of the picture (a new picture, or a recode).
    pub fn begin_picture(&mut self) {
        self.pending = None;
        self.frame = None;
    }

    /// The picture's decided `units` (raster order): `searched(i)` whether
    /// unit `i`'s statistics were gathered, `on(unit)` whether its filter is
    /// on, `residual_free(i)` the share of unit `i`'s area coded without
    /// residual (its content signal, [`QUIET_SHARE`]).
    pub fn observe(
        &mut self,
        units: Vec<T>,
        searched: impl Fn(usize) -> bool,
        on: impl Fn(&T) -> bool,
        residual_free: impl Fn(usize) -> f32,
    ) {
        let mut outcome = FilterOutcome::default();
        let known_history = self.history.len() == units.len();
        let mut history = if known_history {
            self.history.clone()
        } else {
            vec![UnitHistory::default(); units.len()]
        };
        for (index, unit) in units.iter().enumerate() {
            let unchanged = self.previous.get(index) == Some(unit);
            let (searched, on) = (searched(index), on(unit));
            outcome.count(searched, on, unchanged);
            if searched {
                let last = history[index];
                if last.known {
                    let quiet = residual_free(index) >= QUIET_SHARE;
                    outcome.count_given(last, quiet, on, unchanged);
                }
                history[index] = last.after(on, unchanged);
            }
        }
        self.frame = Some(outcome);
        self.pending = Some((units, history));
    }

    /// The committed picture's units become the previous ones.
    pub fn commit(&mut self) {
        if let Some((units, history)) = self.pending.take() {
            self.previous = units;
            self.history = history;
        }
    }

    /// The units `gate` skips this picture (raster order, `units` of them):
    /// `residual_free(i)` is unit `i`'s share of its area coded without
    /// residual; the unit's run of searches that ended off comes from the
    /// committed pictures (none for a unit never searched).
    #[must_use]
    pub fn usage_skips(
        &self,
        gate: &crate::UsageGate,
        units: usize,
        residual_free: impl Fn(usize) -> f32,
    ) -> Vec<bool> {
        let known = self.history.len() == units;
        (0..units)
            .map(|unit| {
                let off_run = known
                    .then(|| self.history[unit])
                    .filter(|history| history.known)
                    .map(|history| history.off_run);
                gate.skips(unit, residual_free(unit), off_run)
            })
            .collect()
    }

    /// This picture's outcome, if its filter was searched.
    #[must_use]
    pub fn frame(&self) -> Option<FilterOutcome> {
        self.frame
    }
}

/// The role a frame's outcomes are summed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutcomeRole {
    /// Key frames.
    Key,
    /// Base inter frames of the effort cadence.
    Base,
    /// The other inter frames.
    NonBase,
}

impl OutcomeRole {
    /// Lower-case name for reports.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Base => "base",
            Self::NonBase => "nonbase",
        }
    }
}

/// The committed frames' outcomes, by tool and role.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterOutcomeStats(BTreeMap<(FilterTool, OutcomeRole), FilterOutcome>);

impl FilterOutcomeStats {
    pub(crate) fn add(&mut self, role: OutcomeRole, frame: &[(FilterTool, FilterOutcome)]) {
        for (tool, outcome) in frame {
            self.0.entry((*tool, role)).or_default().add(outcome);
        }
    }

    /// Every (tool, role) total, in order.
    pub fn iter(&self) -> impl Iterator<Item = (FilterTool, OutcomeRole, &FilterOutcome)> {
        self.0
            .iter()
            .map(|(&(tool, role), outcome)| (tool, role, outcome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_by_outcome_and_sums_by_role() {
        let mut frame = FilterOutcome::default();
        frame.count(false, false, false);
        frame.count(true, false, true);
        frame.count(true, true, true);
        frame.count(true, true, false);
        assert_eq!(
            frame,
            FilterOutcome {
                units: 4,
                searched: 3,
                searched_off: 1,
                searched_unchanged: 1,
                ..FilterOutcome::default()
            }
        );
        let mut stats = FilterOutcomeStats::default();
        stats.add(OutcomeRole::Base, &[(FilterTool::Sao, frame)]);
        stats.add(OutcomeRole::Base, &[(FilterTool::Sao, frame)]);
        let (tool, role, total) = stats.iter().next().unwrap();
        assert_eq!(
            (tool, role, total.units, total.searched_off),
            (FilterTool::Sao, OutcomeRole::Base, 8, 2)
        );
    }

    #[test]
    fn history_conditions_count_the_next_search() {
        let mut tracker = OutcomeTracker::<bool>::default();
        // Unit 0 off twice then on; unit 1 on and unchanged throughout;
        // unit 2 is quiet and off, then not searched, then off.
        let pictures = [
            ([false, true, false], [true, true, true]),
            ([false, true, false], [true, true, false]),
            ([true, true, false], [true, true, true]),
        ];
        let mut totals = FilterOutcome::default();
        for (on, searched) in pictures {
            tracker.begin_picture();
            tracker.observe(
                on.to_vec(),
                |i| searched[i],
                |&o| o,
                |i| if i == 2 { 1.0 } else { 0.0 },
            );
            tracker.commit();
            totals.add(&tracker.frame().unwrap());
        }
        let tally =
            |given: Given| totals.given[Given::ALL.iter().position(|&g| g == given).unwrap()];
        // Pictures 2 and 3: unit 0 twice, unit 1 twice, unit 2 once.
        assert_eq!(tally(Given::Any).units, 5);
        // Unit 0 at picture 2 (off) and 3 (on); unit 2 at picture 3 (off).
        assert_eq!(
            tally(Given::OffLast),
            Tally {
                units: 3,
                off: 2,
                unchanged: 0
            }
        );
        assert_eq!(
            tally(Given::OffLast2),
            Tally {
                units: 1,
                off: 0,
                unchanged: 0
            }
        );
        assert_eq!(
            tally(Given::OffLastQuiet),
            Tally {
                units: 1,
                off: 1,
                unchanged: 0
            }
        );
        assert_eq!(
            tally(Given::UnchangedLast),
            Tally {
                units: 1,
                off: 0,
                unchanged: 1
            }
        );
    }
}
