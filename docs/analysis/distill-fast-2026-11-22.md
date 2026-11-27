# Distilling the slow rungs' decisions into the fast rungs (2026-11-22)

**Question (owner, via the learned-search line, docs/analysis/learned-search-2026-11-22.md).**
At medium and faster there is little or no search to steer: the planner decides depths,
shapes and modes from cheap rules, and a guide must cost about 1 % of a superblock. Can a tiny
model teach the fast rungs the slowest rung's outcomes (Stockfish trains on deeper search than
it can afford at runtime), and how much is there to win?

**Status.** Work in progress; every number names its data. Nothing here changes a stream.

Scripts: `eval/distill/`. Hook: the partition transplant oracle
(`crates/nc-encoder/src/transplant.rs`, `nanocodec encode --partition-record FILE` /
`--partition-replay FILE`, default off).

## 1. The oracle

### 1.1 Method

For each fast rung (AV1 medium, fast, faster, veryfast; HEVC and VVC medium, fast, faster)
and each decision type, one arm: the rung with every parameter of that decision group set to
the slowest rung's effective value (AV1 slower, HEVC and VVC veryslow; `distill.arms`). The
groups, by parameter name:

- **partition**: the inter planner (`planner.*` but the key plan): with `planner.rd_split` the
  slowest rung's RD partition search, and the slowest rung's planner thresholds;
- **skip**: skip / merge exits (`av1.pre_tx_skip*`, `av1.skip_bias*`, `hevc.early_merge*`,
  `*.merge_candidates`, `decision.skip_*`, zero-out gates);
- **transform**: transform size / type search (`av1.inter_tx_split`, `av1.transform_rd*`,
  `hevc.inter_tu_split`, `hevc.max_transform_log2`, partial forward transforms, `vvc.tx64`);
- **mode**: the mode and motion search (`decision.*`, `adapt.*`, inter-intra, intra candidates);
- **all**: the four together. Filters, quantisation (trellis), rate control, references,
  layers and key frames stay the rung's.

Each arm is the slowest rung's *mechanism* for that decision inside the fast rung: the
decisions a perfect distilled guide would predict, in the fast rung's own context (its
references and other tools). Its J is the ceiling of such a guide; its CPU is the price of
getting those decisions by search instead.

For partitions the ceiling's CPU is measured too: the arm's encode records the partition it
coded in every inter superblock, and the plain rung codes those partitions without the search
(`--partition-replay`). The replayed stream is byte-identical to the arm's (section 1.3), so
the replay is exactly the arm's coding at the cost of a perfect partition predictor.

J: `nceval.screen --clips adopt --levels quick --k 0`, the day's L1 draw (2026-11-22, 24
clips, 150 frames, single-point, CBR), quality only, on rented boxes (`aws_measure --batch`),
build 7adbd3646. CPU: `tools/dev/aws_cpu_ab.py`, Graviton4 and Zen 5, the four quick clips,
150 frames, CBR 600, 3 interleaved rounds, cycles.

### 1.2 Ceiling per decision type

Quality J (k 0) against the rung, with its 90 % interval, and cycles on Graviton4 and Zen 5.

| codec | rung | partition | skip | transform | mode | all four |
|---|---|---|---|---|---|---|
| AV1 | medium | -3.0 [-4.3,-2.2] / +57, +62 | -1.3 / +9, +6 | none | -2.0 / +17, +11 | -6.8 / +99, +91 |
| AV1 | fast | -4.3 [-5.1,-3.4] / +55, +59 | -2.3 / +6, +3 | +8.1 (bug, below) | -2.9 / +36, +25 | -8.7 / +130, +114 |
| AV1 | faster | -6.8 [-7.4,-6.1] / +61, +69 | -2.5 / +7, +4 | +7.3 (bug) | -3.5 / +48, +31 | -12.7 / +167, +149 |
| AV1 | veryfast | -11.9 [-13.2,-10.6] / +81, +108 | -2.6 / +11, +9 | -1.1 / +13, +13 | -5.9 / +69, +42 | -19.1 / +254, +250 |
| HEVC | medium | -7.8 [-8.5,-7.1] / +78, +101 | 0.0 (inert) | none | -2.1 / +44, +37 | -9.2 / +173, +190 |
| HEVC | fast | -6.5 [-7.5,-5.8] / +90, +120 | -1.6 / +9, +8 | -0.5 / 0, 0 | -8.2 / +76, +70 | -15.7 / +296, +318 |
| HEVC | faster | -11.7 [-13.2,-10.3] / +105, +147 | -0.6 / +9, +9 | -1.7 / +12, +7 | -11.2 / +110, +92 | -23.2 / +402, +447 |
| VVC | medium | -8.0 [-9.1,-6.9] / +72, +91 | -0.1 / +1, 0 | +0.3 / +6, +2 | -1.3 / +28, +27 | -9.2 / +138, +154 |
| VVC | fast | -9.6 [-10.4,-8.6] / +73, +95 | -0.8 / +12, +9 | -1.0 / +4, +3 | -7.7 / +75, +73 | -18.1 / +274, +288 |
| VVC | faster | -9.6 [-10.7,-8.9] / +91, +113 | -0.5 / +16, +11 | -1.2 / +5, +4 | -8.7 / +99, +92 | -18.4 / +352, +357 |

Partitions hold the largest prize at every rung but HEVC fast (mode -8.2 against -6.5), with
mode second at the HEVC and VVC fast rungs; skip and transform hold little.

The partition arm split into the RD search alone and the planner's thresholds alone
(`partition_search`, `partition_rules`; only where the arm moves both):

| codec | rung | search only | planner thresholds only |
|---|---|---|---|
| AV1 | faster | -5.9 [-7.4,-5.0] / +63, +67 | -2.1 [-2.7,-1.0] / +2, +5 |
| AV1 | veryfast | -9.0 [-10.2,-7.9] / +84, +97 | -7.9 [-9.4,-5.8] / +10, +20 |
| HEVC | medium | -5.6 [-6.7,-4.8] / +52, +55 | +0.6 [-0.7,+1.7] / +13, +20 |
| HEVC | fast | -6.1 [-7.2,-5.3] / +54, +56 | +3.6 [+2.7,+4.5] / +12, +20 |
| HEVC | faster | -10.1 [-11.3,-9.1] / +71, +86 | -0.6 [-2.0,+0.6] / +22, +36 |
| VVC | medium | -5.6 [-6.4,-4.9] / +44, +45 | +1.1 [-0.1,+2.1] / +12, +19 |
| VVC | fast | -6.9 [-8.0,-6.2] / +44, +51 | -0.0 [-0.6,+1.2] / +7, +11 |
| VVC | faster | -6.9 [-8.0,-6.3] / +54, +56 | +0.6 [-0.1,+1.3] / +11, +15 |

The search is the prize, not the thresholds: the slowest rung's planner thresholds alone
gain little or lose, except at AV1 veryfast (-7.9 % for +10 / +20 %: its light superblocks'
32x32 floor). At the presets' k 0.6 none of the thresholds-only arms is a clear win.

### 1.3 The perfect partition guide's own cost (the transplant)

The arm's encode records its partitions; the plain rung codes them (`eval/distill/replay_check.py`,
build box, the four quick clips, 150 frames, CBR 600). The replayed stream equals the arm's on
31 of 40 (rung, clip): every AV1 rung but faster bodycam, every VVC rung, HEVC medium. Where it
differs (HEVC fast and faster, all clips; AV1 faster bodycam), the arm also moves the
planner's thresholds, which the replaying rung keeps, so its leaf statistics and evidence
differ; the stream sizes stay within 0.5 %.

Bug found on the way, since fixed in 07a44b9f6 (ideas ledger
`av1-inter-tx-split-starved-cbr-level-overrun-open-bug`, repro
`eval/distill/repro_inter_tx_split.sh`): `av1.inter_tx_split=1` on AV1 fast and veryfast at a
starved 1080p CBR drives the planned level past the top quantizer (362 against 283) and
undershoots the target by 18 %, the transform arm's +119 % J on one clip.

Cycles of the replay against the plain rung (`tools/dev/aws_cpu_ab.py`, base and candidate the
same build ab2bae287, the four quick clips, 150 f, CBR 600, 3 rounds), beside the arm's J
(section 1.2) and the search's own cycles:

| codec | rung | J of the partition arm | replay: Graviton4 / Zen 5 | the search: Graviton4 / Zen 5 |
|---|---|---|---|---|
| AV1 | medium | -3.0 | +1.3 / +0.3 | +57 / +62 |
| AV1 | fast | -4.3 | +0.1 / -1.8 | +55 / +59 |
| AV1 | faster | -6.8 | +2.1 / +2.3 | +61 / +69 |
| AV1 | veryfast | -11.9 | +9.3 / +17.8 | +81 / +108 |
| HEVC | medium | -7.8 | +6.7 / +10.4 | +78 / +101 |
| HEVC | fast | -6.5 (replay not identical) | +10.8 / +15.2 | +90 / +120 |
| HEVC | faster | -11.7 (replay not identical) | +19.8 / +33.0 | +105 / +147 |
| VVC | medium | -8.0 | +7.5 / +12.6 | +72 / +91 |
| VVC | fast | -9.6 | +5.5 / +10.1 | +73 / +95 |
| VVC | faster | -9.6 | +9.2 / +14.3 | +91 / +113 |

A perfect partition guide would give AV1 medium, fast and faster their partition arm's J for
-2 to +2 % cycles: the search's partitions cost no more to code than the plan's there. On HEVC
and VVC, and AV1 veryfast, the searched partitions are finer and cost 6-33 % more to code, still
a fifth to a tenth of the search. The J and the cycles come from different clips (L1 draw and
the four quick clips): the J column is the arm's, and equals the replay's only where the
streams are identical.

## 2. The model: a distilled zero-trial flip

Data: nn dataset v0, the fast rungs' ungated search twins (`planner.rd_split=1`, gates off; AV1
medium / fast / faster / veryfast, HEVC medium / fast / faster, CBR and qp-high), every node the
planner weighed both ways on inter frames (AV1 14.7 M, HEVC 12.4 M). Label: the J the other
choice of leaf or quad split saves, per sample over λ. Split: whole source groups
(`nceval.nndata.split`, seed `nn-fast-1`; HEVC 957 / 164 clips). Score: the net share of the flip
oracle's gain kept on the held-out side (flips made, losses included, over every right flip),
at the threshold the training side chose. Scripts `eval/distill/features.py`, `models.py`,
`deployable.py` (results in `eval/distill/trees/`, the box runs' JSON under
`~/nanocodec-data/work/nn-fast/box*`).

### 2.1 Best-model front (no cost limit)

| model | HEVC (twin / plain context) | AV1 (twin / plain) |
|---|---|---|
| the planner's margin (rule) | 0.0 % | -0.1 % |
| 11 trial bytes, MLP 32-8 | 8.4 / 8.6 % | 5.8 / 5.9 % |
| boosted trees, every input | 15.5 / 14.3 % (fast 21.1, faster 8.0, medium 10.3) | 11.5 / 10.3 % (veryfast 21.7 / 23.0) |
| wide MLP 512-256-128, every input | 14.0 / 12.2 % | 5.2 / 9.3 % |
| CNN on the patches + every input | not known (a scoring bug, fixed after) | not known |

Fit-side shares are 25-31 %: the gap is generalisation across source families. The worst class
is texture everywhere (about 0 %). Twin against plain context (the neighbours' costs from the
search twin's coding or the plain rung's) moves 1-2 points. The 11 trial bytes alone keep a
third to a half of the best model's share.

### 2.2 Closed loop: the deployable tree

`planner.flip_tree` (default off; `crates/nc-encoder/src/flip_guide.rs`, generated
`flip_trees.rs`): one depth-8 tree per rung on the planner's own inputs (the trial bytes, the
node's position in its superblock, λ), at most eight compares per weighed node. Offline
(plain context, the rung's rows only): HEVC fast 11.3 % (boosted trees on these inputs 14.5 %,
on every input 20.2 %), AV1 veryfast 12.0 % (17.9 %, 26.4 %).

In the encoder (`nceval.fixedqp ab --level quick`, draw 2026-11-22, against the rung as
shipped; J positive is worse):

| rung | flips | J | PSNR / VMAF / MS-SSIM |
|---|---|---|---|
| HEVC fast | both | +1.7 % [+1.3, +2.1] | +0.9 / +1.0 / +1.9 |
| HEVC fast | merges only (planned split coded as its leaf) | +2.1 % [+1.7, +2.4] | +1.3 / +1.6 / +2.2 |
| HEVC fast | splits only (planned leaf split) | -0.2 % [-0.3, -0.0] | -0.2 / -0.3 / -0.1 |
| AV1 veryfast | both | +0.2 % [-0.2, +0.6] | -0.2 / +0.1 / +0.4 |

Cycles (`aws_cpu_ab`, four quick clips, CBR 600): HEVC fast +1.5 % Graviton4, -2.0 % Zen 5;
AV1 veryfast +1.9 % / +2.8 %. The guide's own cost (evidence computed, no flip): HEVC +0.45 /
+0.57 % instructions, AV1 0.0 %.

**The offline share does not carry into the encoder.** The merges, whose label is the planner's
own leaf, lose; the splits gain a sliver. Why the closed loop disagrees with the labels is not
known yet. This is the fifth offline lead of the day that did not carry (with nn-lead's value
gate at the slow rungs); the learned fast-end line stops here (coordinator, 2026-11-22).

## 3. The hand-tuned answer: the context-gated search at the fast rungs

Coordinator's closing experiment: the shipped context-gated RD split search (`planner.rd_split=1`,
`planner.rd_split_context=1`) at its default thresholds and two tighter settings, against the
rung as shipped, and the step to the next slower rung (that rung's parameters set on top of the
faster one: an approximation of the rung, since a preset's `light.` copies are mapped by name).
J: `nceval.fixedqp ab --level quick` (draw 2026-11-22). Cycles: `aws_cpu_ab`, the four quick
clips, CBR 600, 3 rounds. "Slope" is J per 100 × ln(1 + mean cycle change): the rung step sets the
line a cheaper point must beat.

| rung | arm | J | cycles G4 / Z5 | slope |
|---|---|---|---|---|
| HEVC fast | context gate | -4.0 % [-4.7, -3.6] | +30.0 / +30.3 % | -0.15 |
| | + deepen -2, merge +1 | -0.3 % [-0.5, -0.2] | +4.3 / +4.5 % | -0.07 |
| | + residual 1, merge-skipped | +0.0 % | +2.4 / +3.2 % | 0.00 |
| | the step to medium | -14.3 % [-15.9, -12.6] | +56.4 / +57.6 % | -0.32 |
| AV1 faster | context gate | -3.6 % [-3.8, -3.3] | +35.7 / +36.7 % | -0.12 |
| | + deepen -2, merge +1 | -0.5 % [-0.7, -0.4] | +4.9 / +4.8 % | -0.11 |
| | + residual 1, merge-skipped | -0.2 % [-0.3, -0.0] | +2.4 / +3.0 % | -0.07 |
| | the step to fast | -7.7 % [-9.0, -6.1] | +18.6 / +19.4 % | -0.44 |
| AV1 veryfast | context gate | -2.4 % [-2.7, -2.1] | +31.2 / +31.9 % | -0.09 |
| | + deepen -2, merge +1 | -0.1 % [-0.2, -0.0] | +2.6 / +3.3 % | -0.03 |
| | + residual 1, merge-skipped | -0.0 % | +2.3 / +2.8 % | 0.00 |
| | the step to faster | -9.5 % [-11.1, -7.8] | +31.0 / +41.8 % | -0.31 |

No point lands below the rung-to-rung line: the search buys a third or less of the J per cycle
that the next slower rung's settings buy. The fast rungs gain more from their other settings than
from the partition search; the search is no ladder candidate there.

## 4. Conclusion

- The prize is real (section 1): the slowest rung's partitions in the fast rung are worth
  AV1 -3..-12 %, HEVC -7..-12 %, VVC -8..-10 % L1 J, and a perfect predictor would cost almost
  nothing at AV1 medium / fast / faster (replay cycles within ±2 %).
- No model reaches it without trials: the best offline model keeps 10-21 % of the zero-trial
  flip oracle on held-out sources, and the deployable tree's 11-12 % turns into J ≈ 0 to +2 %
  in the encoder (section 2). The hand-tuned context gate at the fast rungs is below the ladder's
  own slope (section 3).
- What stays: the transplant oracle (`--partition-record / --partition-replay`) and
  `aws_cpu_ab` per-job inputs (landed 11867aa0b); `planner.flip_tree` / `flip_direction`
  (default off, branch nn-fast) as the closed-loop harness for any later flip model.
- Retires nothing: no hand-tuned parameter can be dropped on this evidence.
