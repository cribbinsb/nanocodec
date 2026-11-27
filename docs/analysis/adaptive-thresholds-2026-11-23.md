# Adaptive tool thresholds: review, redesign and measurements (2026-11-23)

Owner request: "completely review if it can be improved; can it be extended to more tools? Can it
be made more aggressive and used at the fast end e.g. trying if parabola estimate subpel works and
switching when it does? Can it be used more aggressively at the slow end? Can we use a learned
model on the history data to adjust/predict?"

Module: `crates/nc-encoder/src/threshold_control.rs` and `threshold_control/` (`adapt.*`).
Earlier work: `docs/handoff/light-pipeline.md` "Adaptive tool thresholds" (2026-11-17),
`docs/analysis/adaptive-effort-2026-11-16.md`, `docs/handoff/fast-decision.md`.
Branch `adaptive-thresholds-overhaul`; scripts and logs in
`~/nanocodec-data/scratch/adaptive-thresholds/` (scratch, not kept).

## 0. Summary

- **Critique** (section 1): the bilinear ranking path recorded no evidence, so every rung that
  ranks on bilinear predictions was outside the controller; a gate at 0 or a tool without a
  threshold could not join; wins were counted, not weighed; the update stepped on noise at its
  sample sizes; adding a tool meant four parallel `match` statements.
- **Design** (section 2, landed default-off: edd8195bf, 9b9b18708, ab9a70f08, a0c3b7381): one
  update over a tool table; evidence that may be weighed (cost saved over cost at stake); a
  sequential rule (`adapt.confidence`) and one effort knob (`adapt.aggression`); three
  actuators (threshold, interval, and a new switch that probes the tool one time in 8 while its
  cheap alternative runs); two new tools, `adapt.refine` (bilinear winner refinement) and
  `adapt.quarter` (the walk's quarter stage against a parabola step). Off is free in streams;
  gathering costs +0.21 % instructions with one tool adapted.
- **Measured** (sections 3 to 5b): a tool's value differs up to 100-fold between clips and is
  stable within one. Stage 3 with CPU on each decision's own encodes passed `adapt.quarter`
  on AV1 veryfast (a speed lever at equal quality: quality 0.00 %, CPU -0.6 to -1.5 %) and
  `adapt.refine` on H.264 veryfast (a quality lever, L3 quality -0.34 %, CPU -0.35 % on L3).
  On AV1 veryfast the gain is the parabola replacing the stage nearly everywhere, not
  per-clip switching. The slow end does not want the decision gates adapted; AV2, VP9 and the
  medium rungs gave nothing the pool resolves.
- **Landed presets**: AV1 veryfast `adapt.quarter` 0.02 (71e88d783) and H.264 veryfast
  `adapt.refine` 0.02 (522ed5e23), both judged at the CBR front slopes (front_k e227e9c71);
  carried over (section 5c): AV1 faster quarter target 0.02 (c5442d58a), VP9 faster
  `adapt.refine` 0.02 (ee61d9e89).
  AV1 fast failed on L3 (AMBIGUOUS).

## 1. Critique of the controller as it was (bf6de9198)

Line numbers are of `crates/nc-encoder/src/threshold_control.rs`, `decision.rs` and `encoder.rs`
at bf6de9198, before the refactor.

**What it did.** Six tools (sub-pel seed gate, exact-rank margin, integer probe, CDEF interval,
depth removal, motion exits). A block whose gate value lies between the threshold T and
`adapt.band` x T on the running side is a *run* (`Band::above`, threshold_control.rs:120), and a
*win* when the vector only that tool found won the ranking (decision.rs:1606). Per tool and frame
role (base / non-base) a multiplier on T moves by `adapt.step` (1.25) at the end of each cycle of
`adapt.period` (4) inter frames once `adapt.min_runs` (64) runs are in: towards less work when the
win rate is under target / 1.5, towards more when over target x 1.5 (`end_cycle`,
threshold_control.rs:305-333). Measuring costs nothing extra: both outcomes are known on the same
block.

1. **The bilinear ranking path recorded no evidence.** `GateOutcomes` is built only in the
   exact-filter branch (decision.rs:654); the bilinear branch (decision.rs:839 onwards) refines its
   winner without recording anything. `adapt.subpel` did scale `codec_subpel_seed_lambdas`, which
   that branch reads, but with no evidence the multiplier never left 1. Every rung that ranks on
   bilinear predictions (H.264 faster and veryfast, VP9 fast, faster and veryfast) was therefore
   outside the controller whatever its settings, and H.264's own gate there
   (`subpel_skip_variance`, an absolute variance) was not scaled at all. This is why the
   2026-11-17 trial found "identical streams" on H.264 Faster and VP9, and why the 2026-11-23
   H.264 veryfast trial of `adapt.subpel` measured 0.
2. **A gate at 0 has no band** (threshold_control.rs:121), so a tool the preset runs everywhere
   (H.264 veryfast's seed gate is 0) or not at all cannot be adapted: there is nothing "just
   inside the gate". Only tools the preset already gates at a useful place could join.
3. **Tools without a threshold could not join.** Anything that is a choice between a tool and a
   cheaper substitute (a search stage against a predicted step, a search against a reuse) had no
   form here; CDEF got in by treating the multiplier, rounded, as an interval, with its own role
   mapping and bound written into the update (threshold_control.rs:273, :288-293, :327-329).
4. **A win is counted, not weighed.** A refinement that saves 0.1 % of a block's cost and one that
   halves it are both one win. `docs/handoff/fast-decision.md` measured that the decision-time
   gain per win differs several-fold between stages, and the six `*_target` values (0.15, 0.1,
   0.1, 0.05, 0.15, 0.2) are win rates of six different events, so they cannot be compared or
   set from one price.
5. **The update is a raw comparison on small samples.** At 64 runs and a target of 0.15 one
   standard error of the rate is 0.045; the hold band is 0.10 to 0.225. A gate sitting exactly at
   its target therefore steps in about one cycle in four, and each decision discards its evidence
   (threshold_control.rs:333). The 2026-11-17 traces show it: the sub-pel gate's multiplier
   "wanders x0.5-x4 with band win rates 2-32 % from cycle to cycle".
6. **Slow to arrive, never reset.** From 1 to the bound 4 is seven steps: 28 inter frames at best,
   more for base frames (one frame in four in a cadence gets a cycle's evidence). On the chart's
   150-frame clips that is a fifth of the clip. Nothing resets at a scene cut or a key frame: the
   multiplier walks back at the same pace. Not changed here (see section 6).
7. **Two roles only.** Key frames are not controlled (the controller observes inter frames only,
   encoder.rs:2193), and every non-base temporal layer shares one multiplier.
8. **Two proxies for a win.** Motion exits count a searched superblock as a win when its best
   64x64 SAD beats the zero vector's by 1/8 (encoder.rs:4624, `MOTION_WIN_SHARE`), not whether
   the exit would have changed anything coded; depth removal counts the planner's own decision to
   split (encoder.rs:4648), which no coding confirms.
9. **Structure.** Adding a tool meant editing the enum, two arrays, `adapts`, `target`, `apply`,
   the update's special cases and the parameter group: four parallel `match` statements
   (threshold_control.rs:199-222) and state in two parallel arrays.
10. **It is symmetric, so on a fast rung it is not a speed lever.** With `adapt.min_scale` 0.5 a
   tool that wins often gets *more* work. On AV2 Faster and AV1 Veryfast the three decision gates
   cost +0.70 % / +1.36 % instructions (fast-decision.md): the controller loosened gates the
   preset had set tight. That is the mechanism working as designed; whether it is wanted is the
   target's and the bound's choice per rung.

## 2. Design

One update, shared by every tool; what differs per tool is a table entry.

- **Evidence is paired and may be weighed** (`threshold_control/evidence.rs`). An observation is a
  block on which the outcome is known with the tool's work and with its cheap alternative. It has a
  weight and the part of the weight the alternative would have lost. Tools that only know "the
  decision changed" report 1 and 0/1 as before. Tools that know the costs report the cost at stake
  and the cost saved, so the rate is "share of the cost of these blocks the tool saves": one unit
  for every such tool, and the thing a price can be put on.
- **One rule** (`threshold_control/rule.rs`): `Rule::judge` gives less work / hold / more work /
  wait; `Travel::moved` applies one step inside the tool's bounds.
  - `adapt.confidence` z (default 0, the old rule): move only when the rate is outside the
    hysteresis band by z standard errors, hold as soon as it is inside by as many, otherwise keep
    the evidence and wait (up to 8 x `min_runs` observations). A sequential test: clear cases
    still move after one cycle, unclear ones stop moving on noise.
  - `adapt.aggression` (default 1) multiplies every tool's target: the rung's one effort knob.
- **Three actuators on the same rule** (`threshold_control/tool.rs`):
  - *Threshold*: the multiplier scales a gate (the five old gates, and `adapt.refine`).
  - *Interval*: the multiplier is frames between runs (CDEF).
  - *Switch* (new): two positions, the tool or its cheap alternative. While the tool runs, every
    run is evidence. While the alternative runs, one opportunity in `adapt.probe_period` (8) runs
    the tool as a probe, whose result is used, so the evidence that can switch it back keeps
    coming at an eighth of the tool's cost. This is the owner's "probe and switch", and it is also
    how a tool with no threshold, or one the preset leaves off, can be controlled.
- **Per clip and frame role** as before: a `Gate` (multiplier and evidence) per tool and role.
- **Deterministic and free when off.** Counts of coding decisions, updated between frames. The
  decision's non-gathering instance takes a no-op switch type, so it compiles as before.

New tools:

| tool | actuator | evidence | acts on |
|---|---|---|---|
| `adapt.refine` | threshold | cost at stake and cost saved by refining winners just above the gate | `decision.winner_refine_gate_scale` (new, default 1): the bilinear ranking's winner-refinement gate, the larger of the codec's variance gate and the seed gate |
| `adapt.quarter` | switch | cost before the quarter stage and what the alternative would have kept of the stage's gain | the walk's quarter stage after a half stage that did not move, against the step a parabola through the half stage's costs predicts (`decision.subpel_quarter_predict`: 0 no step, 1 probed once, 2 taken unmeasured) |

`adapt.quarter` is the owner's example. The parabola (Hill et al., TCSVT 2006; `quarter_predict.rs`)
already existed for the fixed-stage search as a static setting; the walk (AV1, AV2) now has it as
the switch's alternative. The recorded loss is an upper bound: zero where the parabola predicts
the vector the quarter stage found, the stage's whole gain otherwise.

Which static levers are candidates for a switch: the ones the ledger rejected for one content
class while the rest gained. The quarter stage after an unmoved half stage is exactly that
(fast-decision.md: AV1 Veryfast x86 -1.84 %, stopped by the bodycam class at +4.44 % quality).

### The learned predictor

Not built. What exists to learn from is the evidence stream per clip, role and tool, and the
controller's update is already an online estimator of it. The measurements in section 3 show
where a predictor could add anything: the value of a tool differs 100-fold between clips and is
stable within a clip (the 10th and 90th percentile cycles of one clip are within a factor of
2 to 11 of each other on the quarter stage, the clips' medians up to 100 apart), so the first cycle's evidence already
places a clip, and `adapt.confidence` covers the noise between cycles. A model would have to
predict the first four frames before any evidence, or the response to a scene cut. Its form, if
tried: the target multiplied by e^(w . features) with w = 0 the present rule
(`linear_threshold.rs`), features causal and per frame (noise sigma, previous skip share, role),
tuned closed-loop with `nceval.tune spsa`. Queued in the ideas ledger, not started: six offline
models failed closed-loop on 2026-11-22, and nothing measured here says the remaining error is in
what a model would predict.

## 3. What the evidence says (controller traces)

A temporary print of each judged cycle (role, tool, loss rate, size, multiplier; not landed).
150 frames, CBR 600 kbit/s, the five quick clips, gate build, no switching (targets 0, bounds 1),
so the rates are the tool's own. Rate = share of the observed blocks' decision cost (distortion +
lambda x bits) that the tool saved; median over cycles, base / non-base frames.

**The walk's quarter stage after an unmoved half stage, AV1 veryfast** (`adapt.quarter`):

| clip | no alternative (predict 0) | against the probed parabola step (predict 1), upper bound |
|---|---|---|
| static_chirla | 0.02 / 0.01 % | 0.01 / 0.00 % |
| crowd_mot17_09 | 0.07 / 0.03 % | 0.03 / 0.03 % |
| bodycam_airport | 0.23 / 0.59 % | 0.08 / 0.22 % |
| pan_jaad_0117 | 0.23 / 0.46 % | 0.15 / 0.33 % |
| lowlight_pp22_004 | 1.95 / 2.94 % | 0.97 / 1.09 % |

**The bilinear ranking's winner refinement** (`adapt.refine`, band widened to 16 x the gate so that
most refined winners are observed):

| clip | H.264 veryfast | VP9 veryfast |
|---|---|---|
| static_chirla | 0.02 % / too few | 0.01 / 0.01 % |
| crowd_mot17_09 | 3.6 / 2.9 % | 2.0 / 2.9 % |
| bodycam_airport | fewer than 64 observations in 150 frames | 1.9 / 2.6 % |
| pan_jaad_0117 | 2.9 / 3.3 % | 1.3 / 1.6 % |
| lowlight_pp22_004 | 13.5 / 10.1 % | 2.3 / 2.5 % |

With the default band (2 x the gate) H.264 veryfast has few observations (65-85 a cycle on
crowd, five judged cycles in 150 frames on pan, none on bodycam): its gate (a variance of 24 per
sample) sits far below most of the winners it refines, so a multiplier bounded at 4 reaches few
blocks.

What the traces say:

- **A tool's value differs up to 100-fold between clips and is stable within one**: this is what
  the 2026-11-16 study found for tool groups (a 7-fold range), here per tool and for free.
- **Where a tool is worth least it also runs least.** On the static clip the refinement saves
  nothing, and few blocks reach it (a median of 122 observed winners a cycle against 1 468 on
  crowd, H.264). The CPU a switch can recover on such content is small for that reason
  (2026-11-16 study, section 3.1: gain and cost rise together).
- **Near the H.264 gate, refining pays**: 1.4-3.9 % of the block's cost in the default band. The
  gate is not over-spending there, so adapting it is a quality lever on H.264 veryfast (it spends
  on lowlight), not a speed lever.
- **The parabola step recovers part of the quarter stage** (the bound halves on lowlight and
  bodycam), which is what makes it usable as the switch's alternative.

## 4. CPU

x86 callgrind instructions, release build, whole encode, 61 frames (slow rungs 31), CBR 600, the
five quick clips; against the same binary without the parameters. "geo" is the geometric mean.
Rented-core cycles follow where measured.

**Off means free.** Gate builds, shipped presets, 31 frames, the base bf6de9198 against the three
landed commits before the last: AV1 veryfast +0.05 %, H.264 veryfast +0.03 %, HEVC faster
+0.10 %, AV1 medium +0.01 %, H.264 medium +0.07 % instructions. Not zero in the gate
profile. On rented cores (release builds of 51c564220 against a0c3b7381, which also spans two
unrelated speed commits, so it is not a clean pair) cycles were inside their intervals on
Graviton 3/4 and mixed on Zen 5 (H.264 veryfast +0.91 % [+0.33, +1.49] and +0.66 % at the two key
shares with instructions -0.11 %; HEVC faster -0.53 % / -0.17 %): no off-path cost is shown, and
none is excluded on Zen 5. A clean release pair of the four landed commits alone was not run.

**What measuring costs.** With only `adapt.quarter` on and the switch held on the tool, AV1
veryfast paid +0.49 %: the decision's gathering instance noted every tool's evidence. After
a0c3b7381 (evidence only for the adapted tools; streams identical) it is +0.21 %.

**`adapt.quarter`, AV1 veryfast** (after a0c3b7381):

| target, alternative | static | crowd | pan | bodycam | lowlight | geo |
|---|---:|---:|---:|---:|---:|---:|
| 0.002, probe | -2.83 | -0.59 | -0.30 | -0.35 | +0.28 | -0.76 % |
| 0.005, probe | -2.83 | -0.65 | -0.69 | -0.51 | +0.24 | -0.89 % |
| 0.005, probe, probe period 32 | -3.16 | -0.68 | -0.72 | -0.81 | +0.05 | -1.07 % |
| 0.02, probe | -2.83 | -0.65 | -0.70 | -0.68 | -0.74 | -1.12 % |
| 0.005, model | -6.05 | -0.65 | -0.64 | -0.93 | +0.07 | -1.66 % |
| 0.02, model | -6.04 | -0.64 | -0.66 | -0.94 | -0.84 | -1.85 % |
| `decision.subpel_quarter_predict` 2 alone | | | | | | -0.01 % (inert on the walk) |

At 0.005 the switch leaves lowlight on the tool, as the traces say it should, and pays the
measuring cost there.

**`adapt.quarter` 0.005 with the model, other walk rungs:** AV2 fast -1.10 % (static -5.12),
AV2 faster -0.38 %, AV1 fast -0.14 %, AV1 faster +0.02 % (bodycam +1.86), H.264 medium -0.18 %,
AV2 slower -0.11 %, VP9 slower -0.01 %, VVC slower +0.05 %. On the slow rungs the refinement is
too small a share of the encode for the switch to matter either way.

**`adapt.refine`:**

| rung, setting | static | crowd | pan | bodycam | lowlight | geo |
|---|---:|---:|---:|---:|---:|---:|
| H.264 veryfast, target 0.01 | -0.61 | +0.31 | +0.32 | +0.07 | +4.07 | +0.82 % |
| H.264 veryfast, target 0.08, bounds 16 | -0.59 | -0.36 | +0.15 | +0.07 | +1.16 | +0.08 % |
| VP9 veryfast, target 0.01 | -1.78 | +1.43 | -0.19 | -0.20 | +0.21 | -0.11 % |
| VP9 veryfast, target 0.02 | -1.83 | +0.62 | -1.31 | -1.84 | -0.59 | -0.99 % |
| VP9 veryfast, target 0.02, no extra work (`adapt.min_scale` 1) | -1.81 | -0.07 | -1.41 | -1.16 | -1.32 | -1.16 % |
| VP9 veryfast, target 0.04, bounds 16 | -1.84 | -1.69 | -2.22 | -2.03 | -2.31 | -2.02 % |

**`adapt.confidence` 2 on the rungs that run the controller:** AV1 faster +0.82 % (static +2.36,
bodycam -1.34), HEVC faster -0.05 %, HEVC fast -0.01 %, VVC fast -0.17 %.

**Rented cores** (`tools/dev/aws_cpu_ab.py --key-shares`, cycles, base and candidate the same
build a0c3b7381, three 900-frame clips, 5 rounds, 95 % intervals; 1 key in 150 / 1 key in 300):

| candidate | Graviton 3 | Graviton 4 | Zen 5 | x86 instructions |
|---|---|---|---|---|
| AV1 veryfast, `adapt.quarter` 0.02, model | -1.26 / -0.52 % | -1.33 / -0.93 % | -0.69 / -0.10 % | -1.13 / -1.08 % |
| AV1 veryfast, 0.005, model | -0.22 / -0.10 % | +0.23 / -0.60 % | +0.38 / -0.05 % | +0.18 / -0.05 % |
| AV1 veryfast, 0.005, probe | +0.16 / +0.04 % | +0.32 / +0.21 % | +1.21 / +0.83 % | +0.61 / +0.39 % |
| AV2 fast, 0.005, model | -0.31 / -0.62 % | -0.50 / -0.72 % | 0.00 / -0.65 % | -0.26 / -0.66 % |
| H.264 veryfast, `adapt.refine` 0.01 | +0.58 / +0.78 % | +0.53 / +0.41 % | -0.20 / +0.86 % | +1.26 / +1.37 % |
| H.264 veryfast, `adapt.refine` 0.02 | (box failed) | +0.46 / +0.04 % | +0.55 / +1.24 % | +1.27 / +0.78 % |
| VP9 veryfast, `decision.subpel_quarter_predict` 2 | (box failed) | +3.46 / +3.32 % | +1.91 / +2.21 % | +2.91 / +2.81 % |
| H.264 faster, `decision.subpel_quarter_predict` 2 | (box failed) | -0.58 / -0.04 % | +0.35 / +1.09 % | -0.09 / +0.45 % |

- **The callgrind figures overstate the saving two to four times.** The five quick clips at 61
  frames include a static clip that saves 3-6 % of instructions; on the long clips at the
  acceptance key shares the 0.02 model saves 1.1 % of instructions and 0.1-1.3 % of cycles, and
  the 0.005 settings, which keep the stage (and its measuring) on part of the content, save
  nothing on real cores. Only the setting that takes the alternative nearly everywhere is a
  speed-up.
- **The unmeasured parabola step is not free on VP9 veryfast**: +2 to +3.5 % cycles (more
  fractional vectors to predict from). At that rung's slope (0.94) this cancels its L1 gain
  (-2.49 % was charged no CPU): dropped.

## 5. Quality

Stage 1 only (`nceval.quick` on rented boxes, draw 2026-11-23: L1 = 24 clips, 150 f; probe = 6
long clips at 1 key in 150 and 1 in 300, mean of the shares; 90 % intervals; k = each rung's front
slope; the CPU charged is the callgrind figure of section 4, so J moves by k x the difference where
rented cores disagree). Anchor: the rung at head. A quick screen ranks and kills; it adopts nothing.

**`adapt.quarter`, AV1 veryfast (k 0.2459):**

| setting | dCPU charged | L1 J | probe J | verdict |
|---|---:|---|---|---|
| target 0.005, probe | -0.59 % | -0.24 % [-0.37, -0.14] | -0.29 % [-0.68, -0.07] | continue |
| target 0.005, model | -1.36 % | -0.47 % [-0.63, -0.34] | -0.65 % [-0.90, -0.52] | continue |
| target 0.02, model | -1.85 % | -0.66 % [-0.84, -0.52] | -0.83 % [-1.03, -0.75] | continue |
| target 0.05, model | -1.90 % | -0.64 % [-0.80, -0.50] | -0.83 % [-1.06, -0.72] | continue |
| always the alternative (target 1), model | -1.95 % | -0.61 % [-0.76, -0.48] | -0.84 % [-1.08, -0.73] | continue |

- **The gain is the parabola model replacing the quarter stage, not the switching.** Always taking
  the alternative scores the same as the switch at 0.02 or 0.05, and the switch at 0.005 (which
  keeps the stage on lowlight-like clips) scores slightly lower. On this rung the decision-time
  loss the switch measures on lowlight (1-3 % of those blocks' cost) did not show up as a quality
  loss: the probes' worst class was lowlight at +0.42 % with the stage replaced everywhere.
- No class guard, no pulse: the bodycam class that stopped the static "quarter stage only after a
  half-sample move" lever at +4.44 % (fast-decision.md) is at +0.08 % here; the difference is the
  predicted step.

**`adapt.quarter` 0.005 with the model, AV2:** fast L1 -0.44 % [-0.82, -0.04], probe 0.00 %
[-0.25, +0.32] (1/300 +0.47 %, static +1.41 %); faster L1 -0.02 %, probe -0.23 % [-0.50, +0.05].
Continue, undecided.

**`adapt.refine`:**

| rung (k) | setting | dCPU charged | L1 J | probe J | verdict |
|---|---|---:|---|---|---|
| H.264 veryfast (0.1362) | target 0.01 | +0.82 % | -0.52 % [-0.98, -0.12] | -0.05 % [-0.21, +0.17] (1/300 +0.26 %) | continue |
| H.264 veryfast | target 0.02 | +0.40 % (estimate, not counted) | -0.47 % [-0.98, -0.09] | -0.24 % [-0.50, +0.01] | continue |
| H.264 veryfast | target 0.08, bounds 16 | +0.08 % | +0.38 % [-0.23, +0.99] | +1.52 % [+0.72, +2.17] | **kill** |
| VP9 veryfast (0.9392) | target 0.04, bounds 16 | -2.0 % | +1.08 % [+0.06, +2.45] | +1.55 % [+0.05, +2.55], crowd +13.8 % at 1/150 | **kill** |
| VP9 veryfast | target 0.02 | -0.99 % | +0.30 % [-0.38, +1.10] | -0.41 % [-1.20, +0.50], videocall +5.97 % (L1) | continue, undecided |
| VP9 veryfast | target 0.02, no extra work | -1.16 % | +0.28 % [-0.42, +1.11] | -0.18 % [-0.96, +0.63] | continue, undecided |

- On H.264 veryfast the controller is a quality lever (it refines more on lowlight-like clips):
  quality about -0.6 % on L1 for +0.4 to +0.8 % instructions. As a speed lever (high target) it
  is killed. On VP9 veryfast, at a front slope of 0.94, cutting refinement harder than the
  preset loses more quality than the CPU is worth.

**`adapt.confidence` 2 on rungs that already run the controller:** AV1 faster L1 -0.31 %
[-0.51, -0.06], probe +0.16 % [-0.04, +0.40]; HEVC faster L1 -0.05 %, probe -0.21 % [-0.39, -0.04];
VVC fast L1 -0.02 %, probe -0.21 % [-0.33, -0.04]. All continue; gains of 0.2 % are below what the
pool resolves (about 0.7 % for a half-pool SPRT, PROCEDURES "Fishtest-style tuning"): undecided.

**The three decision gates (`adapt.subpel`, `rank`, `probe`) on medium and slow rungs**, with
`adapt.aggression` 0.5 (spend more), 1, 2 (cut more). x86 callgrind first: HEVC, VP9 medium and
HEVC slow have none of the three gates set, so the controller is inert there and costs +0.15 to
+0.28 % for gathering.

| rung (k) | aggression | dCPU charged | L1 J | probe J | verdict |
|---|---:|---:|---|---|---|
| VVC medium (0.1634) | 1 | -0.95 % | -0.07 % [-0.32, +0.18] | +0.47 % [+0.14, +0.83] | continue (probe worse) |
| VVC medium | 2 | -1.13 % | +0.23 % | +0.82 % [+0.58, +1.09] | **kill** |
| H.264 medium (0.2044) | 1 | -0.70 % | +0.28 % [+0.04, +0.58] | +0.46 % [+0.20, +0.73] | **kill** |
| AV1 medium (0.1277) | 0.5 | +0.64 % | -0.25 % [-0.38, -0.06] | -0.13 % [-0.21, -0.03] | continue |
| AV1 medium | 2 | -1.12 % | +0.70 % [+0.32, +1.17] | +0.31 % [-0.09, +0.71] | continue (worse) |
| AV2 medium (0.1313) | 2 | -0.69 % | +0.34 % [+0.13, +0.60] | +0.87 % [+0.65, +0.99] | **kill** |
| AV2 medium | 0.5 | +0.28 % | 0.00 % | +0.21 % [+0.06, +0.40] | continue (worse) |
| AV1 slow (0.0677) | 0.5 | +0.42 % | +0.12 % [-0.12, +0.34] | -0.44 % [-1.01, +0.03] | continue, undecided |

- **The slow end does not want these gates adapted.** Cutting (aggression 1 or 2) loses more than
  the CPU is worth at every medium rung tried; spending (0.5) is a small gain on AV1 medium only
  (0.1 to 0.25 %, below what the pool resolves).

**A static setting found on the way** (`decision.subpel_quarter_predict` 2, the unmeasured parabola
step, on the fixed-stage rungs that did not have it; no CPU expected, not yet measured):
VP9 veryfast L1 -2.49 % [-4.57, -0.90], probe -0.39 % [-1.11, +0.33], texture class +3.71 %:
continue. H.264 faster L1 -0.08 %, probe -0.33 % [-0.64, -0.13]: continue. H.264 veryfast L1
+1.31 %, probe +1.51 % [+0.48, +2.19], crowd +7.9 %: **kill**.

## 5b. Stage 3 (nceval.train, CPU on each decision's own encodes)

`nceval.train run` on rented boxes, base 00f777521 (landed cache), draw 2026-11-23, after
1f425f5ef: every arm's dCPU is timed (aws_cpu_ab, Zen 5 + Graviton 3/4 combined) on the declared
subset of that arm's own screen encodes, per protocol group (L3: 150 f one key; final check:
600 f keys every 150, 900 f keys every 300), and each screen is charged its own group. Earlier
CPU figures in this document (callgrind at 61 f, aws_cpu_ab on three fixed clips) are other
encodes and are not combined with these. The AV1 veryfast arms ran in one round against one base
together with av1-retune's key moves (a950812b53); boxes $0.33 (AV1) and $0.17 (H.264).

| arm | dCPU L3 | dCPU final 1/150, 1/300 | L3 J | final check J (mean of shares) | verdict |
|---|---:|---|---|---|---|
| AV1 veryfast `adapt.quarter` 0.02, model (962f542925) | -0.64 % | -1.54, -1.20 % | -0.16 % [-0.30, -0.01] | -0.35 % [-0.49, -0.19] | PASS |
| AV1 veryfast key moves (a950812b53, av1-retune) | +0.55 % | +1.59, +0.27 % | -0.60 % [-0.95, -0.25] | -0.59 % [-0.91, -0.25] | PASS |
| H.264 veryfast `adapt.refine` 0.02 (58a50afc48) | -0.35 % | -1.44, +0.55 % | -0.39 % [-0.55, -0.23] | -0.43 % [-0.64, -0.23] | PASS |

The verdicts above are at the fixed-QP-timed slopes (AV1 veryfast 0.2459, H.264 veryfast
0.1362), given as an explicit `--k` because `front_k.json` was refused (it was timed at fixed
QP while the screens judge CBR). Re-deciding from the same saved screens and charges
(J + k dCPU per clip, the same bootstrap and share combination; it reproduces the train's
numbers at those k):

| arm | k 0 (quality only) | k 0.10 | k 0.20 | k 0.35 |
|---|---|---|---|---|
| AV1 quarter, L3 / final | -0.01 % AMBIGUOUS / -0.02 % AMBIGUOUS | -0.07 % BETTER / -0.15 % BETTER | -0.13 / -0.29 % BETTER | -0.23 / -0.50 % BETTER |
| AV1 key moves | -0.73 / -0.82 % BETTER | -0.68 / -0.72 % | -0.62 / -0.63 % | -0.54 / -0.49 % |
| H.264 refine | -0.34 / -0.37 % BETTER | -0.38 / -0.41 % | -0.41 / -0.46 % | -0.46 / -0.52 % |

- **The quarter switch is a speed lever at equal quality**: quality alone is 0.00 % on 108 L3
  clips and 64 long windows; it passes whenever the rung's slope is about 0.1 or more.
- **`adapt.refine` on H.264 veryfast is a quality lever that also costs no CPU on these encodes**
  (L3 -0.35 %, the 1/300 share +0.55 %), unlike the callgrind (+0.8 %) and three-clip aws_cpu_ab
  (+0.04 to +1.24 %) figures of section 4, which measured other encodes.
- **AV2 fast at target 0.02** (stage 1, draw 2026-11-23, no CPU charged): L1 -0.26 % [-0.60,
  +0.09], probe +0.32 % [+0.03, +0.72] (1/300 +0.73 %, static +1.28 %); with section 4's
  -0.3 to -0.7 % cycles at k 0.2052 the probe stays above 0. Not trained; neither was 0.005
  (probe 0.00 %, 1/300 +0.47 %).

**At the CBR-timed front slopes** (front_k e227e9c71, from the main chart at 7c1c5334f with
speed timed on its own CBR encodes), re-decided from the same screens and charges:

| arm | k | L3 J | final check J (1/150, 1/300) | verdict |
|---|---:|---|---|---|
| AV1 veryfast `adapt.quarter` | 0.4074 | -0.27 % [-0.40, -0.11] | -0.58 % [-0.71, -0.41] (-0.76, -0.39 %) | PASS, landed 71e88d783 |
| AV1 veryfast key moves (av1-retune) | 0.4074 | -0.51 % [-0.86, -0.16] | -0.44 % [-0.76, -0.10] (-0.59, -0.29 %) | PASS, landed by av1-retune |
| H.264 veryfast `adapt.refine` | 0.3217 | -0.45 % [-0.62, -0.30] | -0.51 % [-0.72, -0.31] (-1.03, +0.01 %) | PASS, landed 522ed5e23 |

The two AV1 changes were judged each alone against one base; their combination was not
measured.

## 5c. Carry-over to the other codecs and rungs

**Where each tool acts** (stream identity, crowd 720p, CBR 1500, 20 frames, 90 for refine on
VP9 / H.264 faster; head fa55faa23): `adapt.quarter` changes nothing where the walk does not run
(VP9 fast/faster/veryfast, HEVC fast/faster, VVC faster, H.264 fast/faster): there only the
static parabola step (`decision.subpel_quarter_predict`) changes streams. `adapt.refine` changes
streams only on the bilinear ranking: VP9 fast, faster, veryfast and H.264 veryfast; it is
identical on H.264 fast/faster, HEVC, VVC and AV1 (exact ranking).

Stage 1 (`nceval.quick`, draw 2026-11-23, no CPU charged) and stage 3 (`nceval.train`, CPU on
each arm's own encodes, k = the CBR front slope):

| rung, change | stage 1 L1 / probe (quality) | train dCPU key1; 1/150, 1/300 | L3 J | final check J | outcome |
|---|---|---|---|---|---|
| AV1 faster, quarter target 0.005 -> 0.02, predict 1 -> 2 (k 0.4499) | +0.12 % / -0.16 % | -1.24 %; -0.60, -0.58 % | -0.49 % [-0.71, -0.28] | -0.15 % [-0.32, +0.02] | PASS, landed c5442d58a |
| VP9 faster, `adapt.refine` 0.02 (k 0.3258) | +0.01 % / +0.21 % | -1.34 %; -1.13, +0.06 % | -0.35 % [-0.42, -0.29] | -0.20 % [-0.31, -0.08] | PASS, landed ee61d9e89 |
| H.264 faster, predict 2 (k 0.2435) | -0.08 % / -0.33 % (earlier draw) | +0.92 %; +0.63, +0.45 % | +0.04 % [-0.15, +0.21] AMBIGUOUS | -0.14 % [-0.36, +0.08] | FAIL |
| AV1 fast, quarter 0.02 + predict 2 (k 0.2372) | +0.01 % / +0.09 % | -0.45 %; -3.02, -1.39 % | -0.04 % [-0.19, +0.13] AMBIGUOUS | -0.46 % [-0.63, -0.25] | FAIL |
| VP9 fast, `adapt.refine` 0.02 (k 0.1709) | +0.33 % [+0.15, +0.57] / +0.57 % [+0.21, +1.05] | not trained: would need about -3 % CPU | | | dropped |
| VP9 veryfast, `adapt.refine` 0.02 | L1 +1.23 % [+0.55, +2.03] KILL (videocall +5.97 %) | | | | killed |

## 6. Not done, and what did not work

- **The learned predictor** (section 2): not built. Section 3's traces show per-clip values that
  are stable after the first cycle, which the rule already reads; nothing measured locates an
  error a model would remove.
- **Resetting at scene cuts and keys** (critique 6): not changed. The quarter switch's gain does not
  come from switching (section 5), so its reset behaviour does not affect it.
- **Key-frame and per-layer gates** (critique 7): not built.
- **Weighed evidence for depth removal and motion exits** (critique 8): they still report win
  counts through the shared update.
- **Killed**: `adapt.refine` as a speed lever (H.264 veryfast target 0.08, VP9 veryfast 0.04);
  the three decision gates cut at the medium rungs (VVC, H.264, AV2 medium); the unmeasured
  parabola step on H.264 veryfast (quality) and on VP9 veryfast (+2 to +3.5 % cycles).
- **Undecided, not trained** (below what the pool resolves or with a cost not yet shown to pay):
  `adapt.quarter` on AV2 fast and faster (section 5b), `adapt.refine` 0.02 on VP9 veryfast, `adapt.confidence`
  2 on AV1 faster, HEVC faster and VVC fast, aggression 0.5 on AV1 medium and AV1 slow.
