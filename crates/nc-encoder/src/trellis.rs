//! Trellis quantisation (rate-distortion optimised quantisation, RDOQ),
//! shared by the codec backends.
//!
//! Plain quantisation rounds every coefficient on its own. Trellis
//! quantisation chooses the levels of one transform block together, so that
//! distortion plus λ times the bits of the codec's actual entropy coding is
//! minimal: a coefficient may round down, or to zero, when the bits it saves
//! (its own, its neighbours' contexts, a shorter end of block) are worth more
//! than the error it adds. The bitstream is unchanged in kind: only the
//! levels differ.
//!
//! The search is a dynamic program over scan positions in *reverse* scan
//! order, the order in which H.264 CABAC, HEVC and AV1 code their levels.
//! Each survivor ("node") carries the codec's entropy-context state after the
//! levels it has decided; at each position every node is extended with each
//! candidate level ([`Candidates`]), nodes with equal states are merged (the
//! cheaper survives, which makes the search exact for codecs whose state is
//! small, as in x264's CABAC trellis), and at most `max_states` of the
//! cheapest are kept (a beam, for codecs whose state cannot merge). A
//! separate "not started" path carries the all-zero tail: from it, any
//! position may become the last coded coefficient, paying the codec's
//! end-of-block or last-position signalling ([`CoeffRateModel::eob_rate`]).
//! The all-zero block is the "not started" path at the end.
//!
//! Each codec supplies two things: the candidates with their distortion
//! (squared error in its transform domain, scaled to sample SSE, the units
//! of λ), and a [`CoeffRateModel`] that prices levels with its adapted
//! entropy tables. Backends apply it only to the coefficients of the block
//! mode decision chose (as x264's `trellis 1`), so it costs one search per
//! coded transform block.

/// Largest nearest level for which zero is still a candidate. Larger
/// coefficients are only ever rounded to the nearest level or one below it:
/// zeroing them costs far more distortion than any rate it could save (x264
/// and HM's RDOQ likewise only drop small levels to zero).
pub const ZERO_CANDIDATE_MAX_LEVEL: u32 = 2;
// `Candidates::around`'s branch-free slots assume zero is reached from
// levels 1 (as the level below) and 2 (as a third slot) only.
const _: () = assert!(ZERO_CANDIDATE_MAX_LEVEL == 2);

/// At most this many candidate levels per coefficient: the nearest level,
/// the one below and zero.
pub const MAX_CANDIDATES: usize = 3;

/// The levels one coefficient may take, with the distortion of each.
#[derive(Debug, Clone, Copy, Default)]
pub struct Candidates {
    /// Candidate magnitudes, best-distortion first.
    pub levels: [u32; MAX_CANDIDATES],
    /// Distortion of each candidate, in sample-SSE units.
    pub distortion: [f32; MAX_CANDIDATES],
    /// Number of candidates in use.
    pub count: usize,
}

impl Candidates {
    /// The candidates around `nearest`, the magnitude rounded to the nearest
    /// level: `nearest`, `nearest − 1`, and zero while `nearest` is at most
    /// [`ZERO_CANDIDATE_MAX_LEVEL`]. `distortion(level)` prices each one.
    pub fn around(nearest: u32, mut distortion: impl FnMut(u32) -> f32) -> Self {
        // Branch-free: all three slots are always filled and `count` says
        // which are candidates. With `nearest` 1 the level below is zero
        // itself; with 2 zero follows it; above 2 zero is not one. Whether a
        // coefficient rounds to 0, 1, 2 or more is data-dependent, so the
        // branching form mispredicted on most positions.
        let below = nearest.saturating_sub(1);
        let levels = [nearest, below, 0];
        // Zero as a third slot is rare; pricing it only then is cheaper
        // than always (AV1's distortion is the costlier to evaluate).
        let zero = if nearest == ZERO_CANDIDATE_MAX_LEVEL {
            distortion(0)
        } else {
            0.0
        };
        Self {
            levels,
            distortion: [distortion(nearest), distortion(below), zero],
            count: 1 + usize::from(nearest >= 1) + usize::from(nearest == ZERO_CANDIDATE_MAX_LEVEL),
        }
    }

    /// Whether zero is a candidate, and its distortion.
    fn zero(&self) -> Option<f32> {
        self.iter().find(|&(level, _)| level == 0).map(|(_, d)| d)
    }

    fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.levels[..self.count]
            .iter()
            .copied()
            .zip(self.distortion[..self.count].iter().copied())
    }
}

/// A codec's coefficient rate in a given entropy-context state. Positions
/// are scan indices; levels are magnitudes. Methods take `&mut self` so a
/// model can fill lazy cost caches of its adaptive tables.
pub trait CoeffRateModel {
    /// What the codec's contexts need to know about the levels decided so
    /// far (those later in scan order). Equal states price every remaining
    /// level identically, which is what makes merging them exact.
    type State: Clone + PartialEq;

    /// The state before any level is coded.
    fn initial_state(&self) -> Self::State;

    /// Bits of signalling that the block is coded and that `scan_pos` holds
    /// its last nonzero level (coded-block flag, end of block or last
    /// position, anything else paid once per coded block).
    fn eob_rate(&mut self, scan_pos: usize) -> f32;

    /// Bits of coding each of `levels` at `scan_pos` in `state`, sign
    /// included, into `bits` (one call per position and state, so a model
    /// derives its contexts once for all candidates). From
    /// [`Self::initial_state`] the position is the last coded one and the
    /// levels are nonzero.
    fn level_rates(
        &mut self,
        state: &Self::State,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32],
    );

    /// Move `state` past `level` coded at `scan_pos`.
    fn advance(&self, state: &mut Self::State, scan_pos: usize, level: u32);

    /// Bits of coding a zero at `scan_pos` in `state`, moving `state` past
    /// it: [`Self::level_rates`] of the zero then [`Self::advance`], which a
    /// model may fuse (most positions of a large block can only be zero).
    fn zero_step(&mut self, state: &mut Self::State, scan_pos: usize) -> f32 {
        let mut rate = [0.0];
        self.level_rates(state, scan_pos, &[0], &mut rate);
        self.advance(state, scan_pos, 0);
        rate[0]
    }

    /// Bits paid once the first scan position is coded (0 by default).
    fn final_rate(&mut self, _state: &Self::State) -> f32 {
        0.0
    }

    /// Bits of signalling an all-zero block.
    fn empty_rate(&mut self) -> f32;
}

/// Marks a node that started at its own position (no earlier decision).
const NO_PARENT: u32 = u32::MAX;

/// One decided level and the decision it extends.
#[derive(Debug, Clone, Copy)]
struct Step {
    parent: u32,
    level: u32,
}

/// A survivor extended by one level, before its state is built: `from` is
/// the survivor's index, or [`NO_PARENT`] for the all-zero tail.
#[derive(Debug, Clone, Copy)]
struct Extension {
    from: u32,
    level: u32,
    cost: f64,
}

/// A survivor of the search: its context state, its cost so far and its
/// latest decision.
#[derive(Debug, Clone)]
struct Node<S> {
    state: S,
    cost: f64,
    step: u32,
}

/// Reusable search buffers (one per backend, so no allocation per block).
#[derive(Debug, Clone)]
pub struct Trellis<S> {
    nodes: Vec<Node<S>>,
    next: Vec<Node<S>>,
    extensions: Vec<Extension>,
    steps: Vec<Step>,
}

impl<S> Default for Trellis<S> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            next: Vec::new(),
            extensions: Vec::new(),
            steps: Vec::new(),
        }
    }
}

impl<S: Clone + PartialEq> Trellis<S> {
    /// Choose the levels of one transform block. `candidates[i]` belongs to
    /// scan position `i`; positions past the slice are zero in every
    /// solution. Writes the chosen magnitude of every scan position of the
    /// slice to `levels` and returns the end of block (one past the last
    /// nonzero scan position; 0 for an all-zero block).
    ///
    /// `lambda` converts bits to distortion units; `max_states` (at least
    /// one) bounds the survivors per position.
    pub fn quantize<M: CoeffRateModel<State = S>>(
        &mut self,
        model: &mut M,
        candidates: &[Candidates],
        lambda: f32,
        max_states: usize,
        levels: &mut [u32],
    ) -> usize {
        if max_states <= 1 {
            return Self::quantize_single(model, candidates, f64::from(lambda), levels);
        }
        self.quantize_beam(model, candidates, lambda, max_states, levels)
    }

    /// [`Self::quantize`] with a beam of `max_states` survivors.
    fn quantize_beam<M: CoeffRateModel<State = S>>(
        &mut self,
        model: &mut M,
        candidates: &[Candidates],
        lambda: f32,
        max_states: usize,
        levels: &mut [u32],
    ) -> usize {
        let lambda = f64::from(lambda);
        self.nodes.clear();
        self.steps.clear();
        // Cost of the all-zero tail so far (None once a position must code).
        let mut tail = Some(0.0_f64);
        for (pos, cands) in candidates.iter().enumerate().rev() {
            if cands.count == 1 && cands.levels[0] == 0 {
                self.advance_with_zero(model, pos, cands.distortion[0], lambda);
                tail = tail.map(|t| t + f64::from(cands.distortion[0]));
                continue;
            }
            self.extensions.clear();
            let mut rates = [0.0; MAX_CANDIDATES];
            for (from, node) in self.nodes.iter().enumerate() {
                model.level_rates(&node.state, pos, &cands.levels[..cands.count], &mut rates);
                for ((level, distortion), &rate) in cands.iter().zip(&rates) {
                    let cost = node.cost + f64::from(distortion) + lambda * f64::from(rate);
                    self.extensions.push(Extension {
                        from: from as u32,
                        level,
                        cost,
                    });
                }
            }
            if let Some(tail_cost) = tail {
                let initial = model.initial_state();
                // Candidates are ordered nearest first, so zero is last.
                let nonzero = cands.levels[..cands.count].partition_point(|&l| l > 0);
                if nonzero > 0 {
                    let eob = model.eob_rate(pos);
                    model.level_rates(&initial, pos, &cands.levels[..nonzero], &mut rates);
                    rates[..nonzero].iter_mut().for_each(|r| *r += eob);
                }
                for ((level, distortion), &rate) in cands.iter().zip(&rates).take(nonzero) {
                    let cost = tail_cost + f64::from(distortion) + lambda * f64::from(rate);
                    self.extensions.push(Extension {
                        from: NO_PARENT,
                        level,
                        cost,
                    });
                }
            }
            tail = tail.zip(cands.zero()).map(|(t, d)| t + f64::from(d));
            self.select_survivors(model, pos, max_states);
        }

        let n = candidates.len();
        levels[..n].fill(0);
        let empty = tail.map(|t| t + lambda * f64::from(model.empty_rate()));
        let mut best: Option<(f64, u32)> = None;
        for node in &self.nodes {
            let cost = node.cost + lambda * f64::from(model.final_rate(&node.state));
            if best.is_none_or(|(c, _)| cost < c) {
                best = Some((cost, node.step));
            }
        }
        let Some((cost, mut step)) = best else {
            return 0;
        };
        if empty.is_some_and(|e| e <= cost) {
            return 0;
        }
        // The best path decided position 0 last; walk back towards its start.
        let mut pos = 0;
        loop {
            let s = self.steps[step as usize];
            levels[pos] = s.level;
            if s.parent == NO_PARENT {
                return pos + 1;
            }
            step = s.parent;
            pos += 1;
        }
    }
}

impl<S: Clone + PartialEq> Trellis<S> {
    /// [`Self::quantize`] with one survivor: the same decisions (the
    /// cheapest extension, the first of equal ones in the order the beam
    /// ranks them) without the beam's ranking, state copies or decision
    /// arena. The survivor's path is kept as one level per position, from
    /// its start (the last coded position) down.
    fn quantize_single<M: CoeffRateModel<State = S>>(
        model: &mut M,
        candidates: &[Candidates],
        lambda: f64,
        levels: &mut [u32],
    ) -> usize {
        let n = candidates.len();
        levels[..n].fill(0);
        // The survivor: its state, cost and first (highest) coded position.
        let mut survivor: Option<(S, f64, usize)> = None;
        let mut tail = Some(0.0_f64);
        let mut rates = [0.0; MAX_CANDIDATES];
        for (pos, cands) in candidates.iter().enumerate().rev() {
            if cands.count == 1 && cands.levels[0] == 0 {
                let distortion = f64::from(cands.distortion[0]);
                if let Some((state, cost, _)) = survivor.as_mut() {
                    *cost += distortion + lambda * f64::from(model.zero_step(state, pos));
                }
                tail = tail.map(|t| t + distortion);
                levels[pos] = 0;
                continue;
            }
            // The tail's extensions are priced first and the survivor's
            // last, so a model's per-position cache holds the survivor's
            // (`advance` mostly follows it); they are compared in the
            // beam's order (survivor first), which breaks ties alike.
            let nonzero = cands.levels[..cands.count].partition_point(|&l| l > 0);
            let mut tail_rates = [0.0; MAX_CANDIDATES];
            let initial = (tail.is_some() && nonzero > 0).then(|| {
                let state = model.initial_state();
                let eob = model.eob_rate(pos);
                model.level_rates(&state, pos, &cands.levels[..nonzero], &mut tail_rates);
                tail_rates[..nonzero].iter_mut().for_each(|r| *r += eob);
                state
            });
            // (cost, level, from the tail)
            let mut best: Option<(f64, u32, bool)> = None;
            let mut consider = |cost: f64, level: u32, from_tail: bool| {
                if best.is_none_or(|(c, _, _)| cost < c) {
                    best = Some((cost, level, from_tail));
                }
            };
            if let Some((state, cost, _)) = survivor.as_ref() {
                model.level_rates(state, pos, &cands.levels[..cands.count], &mut rates);
                for ((level, distortion), &rate) in cands.iter().zip(&rates) {
                    consider(
                        cost + f64::from(distortion) + lambda * f64::from(rate),
                        level,
                        false,
                    );
                }
            }
            if let (Some(tail_cost), Some(_)) = (tail, initial.as_ref()) {
                for ((level, distortion), &rate) in cands.iter().zip(&tail_rates).take(nonzero) {
                    consider(
                        tail_cost + f64::from(distortion) + lambda * f64::from(rate),
                        level,
                        true,
                    );
                }
            }
            tail = tail.zip(cands.zero()).map(|(t, d)| t + f64::from(d));
            let (cost, level, from_tail) = best.expect("a position with a choice extends a path");
            if from_tail {
                let mut state = initial.expect("tail extensions start from the initial state");
                model.advance(&mut state, pos, level);
                survivor = Some((state, cost, pos));
            } else if let Some((state, survivor_cost, _)) = survivor.as_mut() {
                model.advance(state, pos, level);
                *survivor_cost = cost;
            }
            levels[pos] = level;
        }

        let empty = tail.map(|t| t + lambda * f64::from(model.empty_rate()));
        let Some((state, cost, start)) = survivor else {
            levels[..n].fill(0);
            return 0;
        };
        let cost = cost + lambda * f64::from(model.final_rate(&state));
        if empty.is_some_and(|e| e <= cost) {
            levels[..n].fill(0);
            return 0;
        }
        // Positions above the path's start were decided by paths it
        // replaced: they are zero.
        levels[start + 1..n].fill(0);
        start + 1
    }

    /// A position whose only candidate is zero (most of a large block's high
    /// frequencies): every survivor codes a zero, in place. Survivors are
    /// not merged here even if their states become equal; the next position
    /// with a choice merges them.
    fn advance_with_zero<M: CoeffRateModel<State = S>>(
        &mut self,
        model: &mut M,
        pos: usize,
        distortion: f32,
        lambda: f64,
    ) {
        for node in &mut self.nodes {
            let rate = model.zero_step(&mut node.state, pos);
            node.cost += f64::from(distortion) + lambda * f64::from(rate);
            self.steps.push(Step {
                parent: node.step,
                level: 0,
            });
            node.step = self.steps.len() as u32 - 1;
        }
    }
}

impl<S: Clone + PartialEq> Trellis<S> {
    /// Keep the `max_states` cheapest of the extended nodes whose states
    /// differ: of two equal states only the cheaper survives, since every
    /// later price is the same for both. Extensions are ranked by cost
    /// first, so only the ones that might survive build their state, and
    /// each is compared with the survivors only.
    fn select_survivors<M: CoeffRateModel<State = S>>(
        &mut self,
        model: &M,
        pos: usize,
        max_states: usize,
    ) {
        self.extensions
            .sort_unstable_by(|a, b| a.cost.total_cmp(&b.cost));
        self.next.clear();
        for ext in &self.extensions {
            if self.next.len() == max_states {
                break;
            }
            let (mut state, parent) = match self.nodes.get(ext.from as usize) {
                Some(node) => (node.state.clone(), node.step),
                None => (model.initial_state(), NO_PARENT),
            };
            model.advance(&mut state, pos, ext.level);
            if self.next.iter().any(|kept| kept.state == state) {
                continue;
            }
            self.steps.push(Step {
                parent,
                level: ext.level,
            });
            let step = self.steps.len() as u32 - 1;
            self.next.push(Node {
                state,
                cost: ext.cost,
                step,
            });
        }
        std::mem::swap(&mut self.nodes, &mut self.next);
    }
}

/// A codec whose level contexts read *earlier* scan positions and which
/// codes its end of block in the scan, after a nonzero level (VP9 tokens:
/// `more_coefs` / the EOB token, VP9 §9.3). The levels are decided in scan
/// order by [`ForwardTrellis`].
pub trait ForwardCoeffRateModel {
    /// What the contexts of later levels need to know about the levels
    /// decided so far.
    type State: Clone + PartialEq;

    /// The state before the first scan position.
    fn initial_state(&self) -> Self::State;

    /// Bits of ending the block at `scan_pos` in `state` (no level from
    /// here on), or `None` where the syntax cannot end there. `scan_pos`
    /// may be one past the candidates (the search's last chance to end).
    fn end_rate(&mut self, state: &Self::State, scan_pos: usize) -> Option<f32>;

    /// Bits of coding each of `levels` at `scan_pos` in `state` into
    /// `bits`, sign included.
    fn level_rates(
        &mut self,
        state: &Self::State,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32],
    );

    /// Move `state` past `level` coded at `scan_pos`.
    fn advance(&self, state: &mut Self::State, scan_pos: usize, level: u32);
}

/// The trellis of [`ForwardCoeffRateModel`] codecs: the same dynamic
/// program as [`Trellis`] (merged equal states, a beam of the cheapest),
/// run in scan order. Ending the block is a choice at every position where
/// the syntax allows it; the positions after the end take their zero
/// distortion.
#[derive(Debug, Clone)]
pub struct ForwardTrellis<S> {
    search: Trellis<S>,
    /// Zero-level distortion of the positions from each one to the end.
    zero_suffix: Vec<Option<f64>>,
}

impl<S> Default for ForwardTrellis<S> {
    fn default() -> Self {
        Self {
            search: Trellis::default(),
            zero_suffix: Vec::new(),
        }
    }
}

impl<S: Clone + PartialEq> ForwardTrellis<S> {
    /// Choose the levels of one transform block: `candidates[i]` belongs
    /// to scan position `i`, and positions past the slice are zero.
    /// Writes the chosen magnitude of every position of the slice to
    /// `levels` and returns the end of block (one past the last nonzero).
    pub fn quantize<M: ForwardCoeffRateModel<State = S>>(
        &mut self,
        model: &mut M,
        candidates: &[Candidates],
        lambda: f32,
        max_states: usize,
        levels: &mut [u32],
    ) -> usize {
        let lambda = f64::from(lambda);
        self.prepare_zero_suffix(candidates);
        if max_states <= 1 {
            return self.quantize_single(model, candidates, lambda, levels);
        }
        self.quantize_beam(model, candidates, lambda, max_states, levels)
    }

    /// The zero-level distortion of the positions from each one to the end:
    /// ending at position i leaves i.. at zero, which every one of them must
    /// allow (`None` where one does not).
    fn prepare_zero_suffix(&mut self, candidates: &[Candidates]) {
        let n = candidates.len();
        self.zero_suffix.clear();
        self.zero_suffix.resize(n + 1, Some(0.0));
        for (i, cands) in candidates.iter().enumerate().rev() {
            self.zero_suffix[i] = self.zero_suffix[i + 1]
                .zip(cands.zero())
                .map(|(rest, d)| rest + f64::from(d));
        }
    }

    /// [`Self::quantize`] with a beam of `max_states` survivors, after
    /// `zero_suffix` is set.
    fn quantize_beam<M: ForwardCoeffRateModel<State = S>>(
        &mut self,
        model: &mut M,
        candidates: &[Candidates],
        lambda: f64,
        max_states: usize,
        levels: &mut [u32],
    ) -> usize {
        let n = candidates.len();
        let search = &mut self.search;
        search.nodes.clear();
        search.steps.clear();
        search.nodes.push(Node {
            state: model.initial_state(),
            cost: 0.0,
            step: NO_PARENT,
        });
        // The cheapest ended block: (cost, its last step, end position).
        let mut best: Option<(f64, u32, usize)> = None;
        let mut rates = [0.0; MAX_CANDIDATES];
        for (pos, cands) in candidates.iter().enumerate() {
            search.end_paths(model, pos, self.zero_suffix[pos], lambda, &mut best);
            search.extensions.clear();
            for (from, node) in search.nodes.iter().enumerate() {
                model.level_rates(&node.state, pos, &cands.levels[..cands.count], &mut rates);
                for ((level, distortion), &rate) in cands.iter().zip(&rates) {
                    let cost = node.cost + f64::from(distortion) + lambda * f64::from(rate);
                    search.extensions.push(Extension {
                        from: from as u32,
                        level,
                        cost,
                    });
                }
            }
            search.select_forward(model, pos, max_states);
        }
        search.end_paths(model, n, Some(0.0), lambda, &mut best);

        levels[..n].fill(0);
        let Some((_, mut step, end)) = best else {
            // No legal ending: keep the nearest levels.
            for (level, cands) in levels.iter_mut().zip(candidates) {
                *level = cands.levels[0];
            }
            return levels[..n]
                .iter()
                .rposition(|&l| l != 0)
                .map_or(0, |p| p + 1);
        };
        // The ended path decided positions 0..end; walk back from end − 1.
        let mut pos = end;
        while step != NO_PARENT {
            pos -= 1;
            let s = self.search.steps[step as usize];
            levels[pos] = s.level;
            step = s.parent;
        }
        levels[..n]
            .iter()
            .rposition(|&l| l != 0)
            .map_or(0, |p| p + 1)
    }

    /// [`Self::quantize`] with one survivor, after `zero_suffix` is set:
    /// the same decisions without the beam's ranking, state copies or
    /// decision arena. A single forward path never branches back, so its
    /// levels are written as they are decided and an ending remembers only
    /// its position.
    fn quantize_single<M: ForwardCoeffRateModel<State = S>>(
        &mut self,
        model: &mut M,
        candidates: &[Candidates],
        lambda: f64,
        levels: &mut [u32],
    ) -> usize {
        let n = candidates.len();
        let mut state = model.initial_state();
        let mut cost = 0.0;
        // The cheapest ended block: (cost, end position).
        let mut best: Option<(f64, usize)> = None;
        let mut end_here = |model: &mut M, state: &S, cost: f64, pos: usize, rest: Option<f64>| {
            if let Some(rest) = rest
                && let Some(rate) = model.end_rate(state, pos)
            {
                let total = cost + rest + lambda * f64::from(rate);
                if best.is_none_or(|(c, _)| total < c) {
                    best = Some((total, pos));
                }
            }
        };
        let mut rates = [0.0; MAX_CANDIDATES];
        for (pos, cands) in candidates.iter().enumerate() {
            end_here(model, &state, cost, pos, self.zero_suffix[pos]);
            model.level_rates(&state, pos, &cands.levels[..cands.count], &mut rates);
            // The first of the cheapest, as the beam's stable ranking.
            let mut chosen: Option<(f64, u32)> = None;
            for ((level, distortion), &rate) in cands.iter().zip(&rates) {
                let extended = cost + f64::from(distortion) + lambda * f64::from(rate);
                if chosen.is_none_or(|(c, _)| extended < c) {
                    chosen = Some((extended, level));
                }
            }
            let (extended, level) = chosen.expect("every position has a candidate");
            model.advance(&mut state, pos, level);
            cost = extended;
            levels[pos] = level;
        }
        end_here(model, &state, cost, n, Some(0.0));
        let Some((_, end)) = best else {
            // No legal ending: keep the nearest levels.
            for (level, cands) in levels.iter_mut().zip(candidates) {
                *level = cands.levels[0];
            }
            return levels[..n]
                .iter()
                .rposition(|&l| l != 0)
                .map_or(0, |p| p + 1);
        };
        levels[end..n].fill(0);
        levels[..end]
            .iter()
            .rposition(|&l| l != 0)
            .map_or(0, |p| p + 1)
    }
}

impl<S: Clone + PartialEq> Trellis<S> {
    /// Consider ending every survivor's block at `pos`.
    fn end_paths<M: ForwardCoeffRateModel<State = S>>(
        &self,
        model: &mut M,
        pos: usize,
        zero_suffix: Option<f64>,
        lambda: f64,
        best: &mut Option<(f64, u32, usize)>,
    ) {
        let Some(rest) = zero_suffix else {
            return;
        };
        for node in &self.nodes {
            if let Some(rate) = model.end_rate(&node.state, pos) {
                let cost = node.cost + rest + lambda * f64::from(rate);
                if best.is_none_or(|(c, _, _)| cost < c) {
                    *best = Some((cost, node.step, pos));
                }
            }
        }
    }

    /// [`Self::select_survivors`] for a forward model.
    fn select_forward<M: ForwardCoeffRateModel<State = S>>(
        &mut self,
        model: &M,
        pos: usize,
        max_states: usize,
    ) {
        self.extensions
            .sort_unstable_by(|a, b| a.cost.total_cmp(&b.cost));
        self.next.clear();
        for ext in &self.extensions {
            if self.next.len() == max_states {
                break;
            }
            let node = &self.nodes[ext.from as usize];
            let mut state = node.state.clone();
            model.advance(&mut state, pos, ext.level);
            if self.next.iter().any(|kept| kept.state == state) {
                continue;
            }
            self.steps.push(Step {
                parent: node.step,
                level: ext.level,
            });
            let step = self.steps.len() as u32 - 1;
            self.next.push(Node {
                state,
                cost: ext.cost,
                step,
            });
        }
        std::mem::swap(&mut self.nodes, &mut self.next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy codec: each level costs `1 + level` bits after a nonzero level
    /// and `2 + level` after a zero (the state is "previous level was
    /// nonzero"), zeros cost half a bit, the end of block `eob_bits`.
    struct Toy {
        eob_bits: f32,
    }

    impl CoeffRateModel for Toy {
        type State = bool;
        fn initial_state(&self) -> bool {
            true
        }
        fn eob_rate(&mut self, _scan_pos: usize) -> f32 {
            self.eob_bits
        }
        fn level_rates(
            &mut self,
            state: &bool,
            _scan_pos: usize,
            levels: &[u32],
            bits: &mut [f32],
        ) {
            for (&level, bits) in levels.iter().zip(bits) {
                *bits = match (level, state) {
                    (0, _) => 0.5,
                    (l, true) => 1.0 + l as f32,
                    (l, false) => 2.0 + l as f32,
                };
            }
        }
        fn advance(&self, state: &mut bool, _scan_pos: usize, level: u32) {
            *state = level != 0;
        }
        fn empty_rate(&mut self) -> f32 {
            1.0
        }
    }

    fn squared(original: f32) -> impl Fn(u32) -> f32 {
        move |level| (original - level as f32).powi(2)
    }

    /// Cost of one level choice, priced as the bitstream would code it.
    fn path_cost(model: &mut Toy, cands: &[Candidates], levels: &[u32], lambda: f32) -> f64 {
        let dist: f64 = cands
            .iter()
            .zip(levels)
            .map(|(c, &l)| f64::from(c.iter().find(|&(cl, _)| cl == l).expect("candidate").1))
            .sum();
        let Some(last) = levels.iter().rposition(|&l| l != 0) else {
            return dist + f64::from(lambda * model.empty_rate());
        };
        let mut rate = model.eob_rate(last);
        let mut state = model.initial_state();
        for pos in (0..=last).rev() {
            let mut bits = [0.0];
            model.level_rates(&state, pos, &levels[pos..=pos], &mut bits);
            rate += bits[0];
            model.advance(&mut state, pos, levels[pos]);
        }
        dist + f64::from(lambda) * f64::from(rate)
    }

    /// Exhaustive minimum over all candidate combinations.
    fn brute_force(model: &mut Toy, cands: &[Candidates], lambda: f32) -> f64 {
        let total: usize = cands.iter().map(|c| c.count).product();
        (0..total)
            .map(|mut code| {
                let levels: Vec<u32> = cands
                    .iter()
                    .map(|c| {
                        let level = c.levels[code % c.count];
                        code /= c.count;
                        level
                    })
                    .collect();
                path_cost(model, cands, &levels, lambda)
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn candidates_follow_the_nearest_level() {
        let c = Candidates::around(0, squared(0.2));
        assert_eq!(&c.levels[..c.count], &[0]);
        let c = Candidates::around(2, squared(2.2));
        assert_eq!(&c.levels[..c.count], &[2, 1, 0]);
        let c = Candidates::around(5, squared(4.6));
        assert_eq!(&c.levels[..c.count], &[5, 4]);
    }

    #[test]
    fn zero_lambda_keeps_the_nearest_levels() {
        let originals: [f32; 4] = [3.2, 0.9, 0.1, 1.4];
        let cands: Vec<_> = originals
            .iter()
            .map(|&o| Candidates::around(o.round() as u32, squared(o)))
            .collect();
        let mut levels = [9; 4];
        let eob =
            Trellis::default().quantize(&mut Toy { eob_bits: 3.0 }, &cands, 0.0, 4, &mut levels);
        assert_eq!(levels, [3, 1, 0, 1]);
        assert_eq!(eob, 4);
    }

    #[test]
    fn exact_search_matches_brute_force() {
        // Deterministic pseudo-random coefficients and λ values.
        let mut seed = 12345_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) as f32 / 65536.0
        };
        for _ in 0..200 {
            let lambda = next() * 4.0;
            let originals: Vec<f32> = (0..7).map(|_| next() * 3.5).collect();
            let cands: Vec<_> = originals
                .iter()
                .map(|&o| Candidates::around(o.round() as u32, squared(o)))
                .collect();
            let mut model = Toy {
                eob_bits: 1.0 + next() * 4.0,
            };
            let best = brute_force(&mut model, &cands, lambda);
            let mut levels = [0; 7];
            // Two states (bool) merge exactly; a beam of 2 is the full DP.
            let eob = Trellis::default().quantize(&mut model, &cands, lambda, 2, &mut levels);
            let cost = path_cost(&mut model, &cands, &levels, lambda);
            assert!((cost - best).abs() < 1e-3, "{levels:?}: {cost} vs {best}");
            assert_eq!(
                eob,
                levels.iter().rposition(|&l| l != 0).map_or(0, |p| p + 1)
            );
        }
    }

    #[test]
    fn single_survivor_matches_a_beam_of_one() {
        let mut seed = 4242_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) as f32 / 65536.0
        };
        for _ in 0..500 {
            let lambda = next() * 4.0;
            let originals: Vec<f32> = (0..9).map(|_| next() * 3.5).collect();
            let cands: Vec<_> = originals
                .iter()
                .map(|&o| Candidates::around(o.round() as u32, squared(o)))
                .collect();
            let mut model = Toy {
                eob_bits: 1.0 + next() * 4.0,
            };
            let (mut single, mut beam) = ([0; 9], [0; 9]);
            let mut trellis = Trellis::default();
            let eob_single = trellis.quantize(&mut model, &cands, lambda, 1, &mut single);
            let eob_beam = trellis.quantize_beam(&mut model, &cands, lambda, 1, &mut beam);
            assert_eq!((eob_single, single), (eob_beam, beam));
        }
    }

    #[test]
    fn expensive_end_of_block_empties_a_weak_block() {
        let cands = [Candidates::around(1, squared(0.6))];
        let mut levels = [7];
        let eob =
            Trellis::default().quantize(&mut Toy { eob_bits: 20.0 }, &cands, 1.0, 4, &mut levels);
        assert_eq!((eob, levels), (0, [0]));
    }

    /// A forward toy: the end of block may follow only a nonzero level
    /// (as VP9's) and costs `eob_bits`, except after the last of `length`
    /// positions; levels cost as in [`Toy`] by whether the previous level
    /// was nonzero.
    struct ForwardToy {
        eob_bits: f32,
        length: usize,
    }

    impl ForwardCoeffRateModel for ForwardToy {
        type State = (bool, bool); // (started, previous nonzero)
        fn initial_state(&self) -> (bool, bool) {
            (false, true)
        }
        fn end_rate(&mut self, state: &(bool, bool), scan_pos: usize) -> Option<f32> {
            let bits = if scan_pos == self.length {
                0.0
            } else {
                self.eob_bits
            };
            state.1.then_some(bits)
        }
        fn level_rates(
            &mut self,
            state: &(bool, bool),
            _: usize,
            levels: &[u32],
            bits: &mut [f32],
        ) {
            for (&level, bits) in levels.iter().zip(bits) {
                *bits = match (level, state.1) {
                    (0, _) => 0.5,
                    (l, true) => 1.0 + l as f32,
                    (l, false) => 2.0 + l as f32,
                };
            }
        }
        fn advance(&self, state: &mut (bool, bool), _: usize, level: u32) {
            *state = (true, level != 0);
        }
    }

    /// The cost of one forward level choice, ending right after its last
    /// nonzero level (or at once).
    fn forward_cost(
        model: &mut ForwardToy,
        cands: &[Candidates],
        levels: &[u32],
        lambda: f32,
    ) -> f64 {
        let dist: f64 = cands
            .iter()
            .zip(levels)
            .map(|(c, &l)| f64::from(c.iter().find(|&(cl, _)| cl == l).expect("candidate").1))
            .sum();
        let end = levels.iter().rposition(|&l| l != 0).map_or(0, |p| p + 1);
        let mut state = model.initial_state();
        let mut rate = 0.0;
        for pos in 0..end {
            let mut bits = [0.0];
            model.level_rates(&state, pos, &levels[pos..=pos], &mut bits);
            rate += bits[0];
            model.advance(&mut state, pos, levels[pos]);
        }
        if end < levels.len() {
            rate += model.end_rate(&state, end).expect("ends after a nonzero");
        }
        dist + f64::from(lambda) * f64::from(rate)
    }

    #[test]
    fn forward_single_survivor_matches_a_beam_of_one() {
        let mut seed = 99_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) as f32 / 65536.0
        };
        for _ in 0..500 {
            let lambda = next() * 4.0;
            let originals: Vec<f32> = (0..9).map(|_| next() * 3.5).collect();
            let cands: Vec<_> = originals
                .iter()
                .map(|&o| Candidates::around(o.round() as u32, squared(o)))
                .collect();
            let mut model = ForwardToy {
                eob_bits: 1.0 + next() * 4.0,
                length: 9,
            };
            let (mut single, mut beam) = ([0; 9], [0; 9]);
            let mut trellis = ForwardTrellis::default();
            let eob_single = trellis.quantize(&mut model, &cands, lambda, 1, &mut single);
            // The beam path, forced (a beam of one through the general code).
            trellis.prepare_zero_suffix(&cands);
            let eob_beam =
                trellis.quantize_beam(&mut model, &cands, f64::from(lambda), 1, &mut beam);
            assert_eq!((eob_single, single), (eob_beam, beam));
        }
    }

    #[test]
    fn forward_search_matches_brute_force() {
        let mut seed = 777_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) as f32 / 65536.0
        };
        for _ in 0..200 {
            let lambda = next() * 4.0;
            let originals: Vec<f32> = (0..7).map(|_| next() * 3.5).collect();
            let cands: Vec<_> = originals
                .iter()
                .map(|&o| Candidates::around(o.round() as u32, squared(o)))
                .collect();
            let mut model = ForwardToy {
                eob_bits: 1.0 + next() * 4.0,
                length: 7,
            };
            let total: usize = cands.iter().map(|c| c.count).product();
            let best = (0..total)
                .map(|mut code| {
                    let levels: Vec<u32> = cands
                        .iter()
                        .map(|c| {
                            let level = c.levels[code % c.count];
                            code /= c.count;
                            level
                        })
                        .collect();
                    forward_cost(&mut model, &cands, &levels, lambda)
                })
                .fold(f64::INFINITY, f64::min);
            let mut levels = [0; 7];
            // Two states merge exactly.
            let eob =
                ForwardTrellis::default().quantize(&mut model, &cands, lambda, 4, &mut levels);
            let cost = forward_cost(&mut model, &cands, &levels, lambda);
            assert!((cost - best).abs() < 1e-3, "{levels:?}: {cost} vs {best}");
            assert_eq!(
                eob,
                levels.iter().rposition(|&l| l != 0).map_or(0, |p| p + 1)
            );
        }
    }
}
