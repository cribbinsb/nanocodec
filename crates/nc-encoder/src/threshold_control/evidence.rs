//! What the adapted tools report: paired evidence from the blocks that ran
//! them, gathered per coding of a frame.
//!
//! Every observation is a block on which the outcome is known both with
//! the tool's extra work and without it (the cheap alternative: the
//! unrefined seed, the pruned candidate, the reused choice, the predicted
//! step). It carries a
//! weight and the part of that weight the cheap alternative would have
//! lost. A tool that only knows whether its work changed the decision
//! reports weight 1 and loss 0 or 1 (a *run* and a *win*); a tool that
//! knows what each choice is worth reports that value and the part its
//! work changed, so one large change counts for more than many negligible
//! ones.

use super::tool::{Size, TOOL_COUNT, Tool};

/// A tool's evidence: the weight of its observations (`runs`), the part
/// of it the cheap alternative would have lost (`wins`), and how many
/// observations there were.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Evidence {
    pub(super) runs: u64,
    pub(super) wins: u64,
    pub(super) samples: u32,
}

impl Evidence {
    pub(super) fn add(&mut self, other: Self) {
        self.runs = self.runs.saturating_add(other.runs);
        self.wins = self.wins.saturating_add(other.wins);
        self.samples = self.samples.saturating_add(other.samples);
    }

    /// The share of the weight the cheap alternative would have lost.
    pub(super) fn rate(self) -> f32 {
        self.wins as f32 / self.runs as f32
    }

    /// How much evidence this is, in the unit `adapt.min_runs` counts for
    /// its tool.
    pub(super) fn size(self, size: Size) -> f32 {
        match size {
            Size::Weight => self.runs as f32,
            Size::Observations => self.samples as f32,
        }
    }
}

/// The margin band (`adapt.band`): how far inside its gate a block still
/// informs the gate's threshold, as a multiple of the threshold. A
/// threshold of 0 (the gate off) has no band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Band(pub(super) f32);

impl Band {
    /// Whether gate value `value` of a tool that runs where it is at least
    /// `threshold` lies in the band above the threshold.
    pub fn above(self, value: f32, threshold: f32) -> bool {
        threshold > 0.0 && value >= threshold && value < self.0 * threshold
    }

    /// Whether gate value `value` of a tool that runs where it is at most
    /// `threshold` lies in the band below the threshold.
    pub fn below(self, value: f32, threshold: f32) -> bool {
        threshold > 0.0 && value <= threshold && value > threshold / self.0
    }
}

/// How a switched tool runs on a frame ([`Actuator::Switch`]).
///
/// [`Actuator::Switch`]: super::tool::Actuator::Switch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Switch {
    /// No controller switches it: the preset's setting stands and nothing
    /// is recorded.
    #[default]
    Preset,
    /// The tool runs; each run is evidence.
    Full,
    /// The cheap alternative replaces the tool, bar one opportunity in
    /// `period`, which runs the tool as a probe: the evidence that can
    /// switch it back.
    Cheap { period: u32 },
}

/// What one opportunity of a switched tool does
/// ([`FrameEvidence::quarter_turn`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Turn {
    /// As the preset says, unrecorded.
    Preset,
    /// Run the tool and record what it saved.
    Measure,
    /// Take the cheap alternative.
    Cheap,
}

/// The evidence of one coding of a frame, gathered block by block while a
/// controller wants it (`band` set); otherwise every record is a single
/// test, so default encodes pay nothing measurable.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct FrameEvidence {
    band: Option<Band>,
    /// The tools a controller adapts, one bit each: only their evidence is
    /// worth the work of noting it.
    wanted: u8,
    pub(super) tools: [Evidence; TOOL_COUNT],
    quarter: Switch,
    /// The switched quarter stage's opportunities so far (its probes' phase).
    quarter_turns: u32,
}

impl FrameEvidence {
    /// A tool that runs where its gate value `value` is at least
    /// `threshold`: a run when `value` is in the band, won or not.
    pub fn record_above(&mut self, tool: Tool, value: f32, threshold: f32, won: bool) {
        if self.band.is_some_and(|band| band.above(value, threshold)) {
            self.add(tool, 1, u64::from(won));
        }
    }

    /// A tool that runs where its gate value `value` is at least
    /// `threshold` and knows the block's cost without its work (`before`)
    /// and with it (`after`): when `value` is in the band, an observation
    /// weighing `before`, of which the tool saved the difference.
    pub fn record_saving_above(
        &mut self,
        tool: Tool,
        value: f32,
        threshold: f32,
        (before, after): (f32, f32),
    ) {
        if self.band.is_some_and(|band| band.above(value, threshold)) {
            self.add(tool, before as u64, (before - after).max(0.0) as u64);
        }
    }

    /// Adds one observation of `tool` of weight `runs`, `wins` of it lost
    /// without the tool.
    pub fn add(&mut self, tool: Tool, runs: u64, wins: u64) {
        self.tools[tool as usize].add(Evidence {
            runs,
            wins,
            samples: 1,
        });
    }

    /// The band this evidence is gathered in, `None` while none is.
    pub fn band(&self) -> Option<Band> {
        self.band
    }

    /// Whether a controller adapts `tool`, so that its evidence is wanted.
    pub fn gathers(&self, tool: Tool) -> bool {
        self.wanted & tool_bit(tool) != 0
    }

    /// What the next opportunity of the switched quarter stage does
    /// ([`Tool::Quarter`]).
    pub fn quarter_turn(&mut self) -> Turn {
        match self.quarter {
            Switch::Preset => Turn::Preset,
            Switch::Full => Turn::Measure,
            Switch::Cheap { period } => {
                self.quarter_turns = self.quarter_turns.wrapping_add(1);
                if self.quarter_turns.is_multiple_of(period.max(1)) {
                    Turn::Measure
                } else {
                    Turn::Cheap
                }
            }
        }
    }

    /// A measured quarter stage: an observation weighing the cost `before`
    /// it, of which the cheap alternative would have lost `before - kept`.
    pub fn record_quarter(&mut self, before: f32, kept: f32) {
        self.add(
            Tool::Quarter,
            before as u64,
            (before - kept).max(0.0) as u64,
        );
    }

    /// Starts a frame's (or a recoding's) evidence afresh, gathered in
    /// `band` (`None`: not gathered) for the `wanted` tools, with the
    /// quarter stage switched as `quarter`.
    pub fn clear(
        &mut self,
        band: Option<Band>,
        wanted: impl IntoIterator<Item = Tool>,
        quarter: Switch,
    ) {
        *self = Self {
            band,
            wanted: wanted
                .into_iter()
                .fold(0, |bits, tool| bits | tool_bit(tool)),
            quarter,
            ..Self::default()
        };
    }
}

fn tool_bit(tool: Tool) -> u8 {
    1 << tool as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_band_above_the_threshold_counts() {
        let mut evidence = FrameEvidence::default();
        evidence.record_above(Tool::Subpel, 1.2, 1.0, true);
        assert_eq!(evidence.tools[Tool::Subpel as usize], Evidence::default());
        evidence.clear(Some(Band(2.0)), Tool::ALL, Switch::Preset);
        evidence.record_above(Tool::Subpel, 0.9, 1.0, true);
        evidence.record_above(Tool::Subpel, 2.5, 1.0, true);
        evidence.record_above(Tool::Subpel, 1.0, 0.0, true);
        evidence.record_above(Tool::Subpel, 1.2, 1.0, false);
        assert_eq!(
            evidence.tools[Tool::Subpel as usize],
            Evidence {
                runs: 1,
                wins: 0,
                samples: 1
            }
        );
        let band = Band(2.0);
        assert!(band.below(0.08, 0.1));
        assert!(!band.below(0.04, 0.1));
        assert!(!band.below(0.12, 0.1));
        assert!(!band.below(0.0, 0.0));
    }

    #[test]
    fn a_cheap_switch_probes_one_opportunity_in_its_period() {
        let mut evidence = FrameEvidence::default();
        assert_eq!(evidence.quarter_turn(), Turn::Preset);
        assert!(!evidence.gathers(Tool::Quarter));
        evidence.clear(Some(Band(2.0)), [Tool::Quarter], Switch::Full);
        assert_eq!(evidence.quarter_turn(), Turn::Measure);
        assert!(evidence.gathers(Tool::Quarter) && !evidence.gathers(Tool::Rank));
        evidence.clear(
            Some(Band(2.0)),
            [Tool::Quarter],
            Switch::Cheap { period: 3 },
        );
        let turns: Vec<Turn> = (0..6).map(|_| evidence.quarter_turn()).collect();
        assert_eq!(
            turns,
            [
                Turn::Cheap,
                Turn::Cheap,
                Turn::Measure,
                Turn::Cheap,
                Turn::Cheap,
                Turn::Measure
            ]
        );
        evidence.record_quarter(200.0, 190.0);
        evidence.record_quarter(300.0, 300.0);
        assert_eq!(
            evidence.tools[Tool::Quarter as usize],
            Evidence {
                runs: 500,
                wins: 10,
                samples: 2
            }
        );
    }

    #[test]
    fn a_saving_weighs_its_block_by_the_cost_at_stake() {
        let mut evidence = FrameEvidence::default();
        evidence.clear(Some(Band(2.0)), Tool::ALL, Switch::Preset);
        evidence.record_saving_above(Tool::Refine, 150.0, 100.0, (400.0, 380.0));
        evidence.record_saving_above(Tool::Refine, 150.0, 100.0, (600.0, 600.0));
        evidence.record_saving_above(Tool::Refine, 250.0, 100.0, (900.0, 100.0));
        let refine = evidence.tools[Tool::Refine as usize];
        assert_eq!(
            refine,
            Evidence {
                runs: 1000,
                wins: 20,
                samples: 2
            }
        );
        assert_eq!(refine.rate(), 0.02);
        assert_eq!(refine.size(Size::Observations), 2.0);
        assert_eq!(refine.size(Size::Weight), 1000.0);
    }
}
