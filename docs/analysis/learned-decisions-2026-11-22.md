# Learned encoder decisions: a fitted key-trial gate, and a content-adaptive controller oracle (2026-11-22)

Owner asks: (1) a learned gate for which nodes the RD partition search trials; (2) "something that
runs on analysis frame statistics, history, etc and outputs what set of tools / QP cadence etc?".
Both as probes: oracle first, build only what the numbers justify. Adoption bar (owner, mid-probe):
a model that matches the hand-tuned mechanism at equal CPU is a win if it retires hand-tuned
parameters. Inference must be bit-identical on every target.

Scripts: branch `learned-probe`, `probe/` (disposable; not landed). Heavy work ran on the build box
and rented boxes; nothing loaded the node dump on the PC.

## Part 1: a fitted gate for the key RD search

### Data

The key RD label dump of docs/analysis/intra-selection-2026-11-21.md: AV1, HEVC and VP9 medium,
`planner.key_rd_split=1`, window 3/3 (every square node 64 to 8 coded as leaf and as quad split),
frame 0 and the mid frame of 56 clips each coded as one key at three quantizers (AV1 / VP9 qindex
80 / 130 / 190, HEVC QP 24 / 30 / 36), 918 runs, 12.5 M nodes. Scoring as there: recursive regret,
a split's cost holding its children at their best; the offline key planner reproduces the dumped
plan exactly.

**Split by source dataset, not clip.** Four folds, each source wholly in one: xiph | chirla, otw,
vc1k | bdd100k, cevo, personpath22, roundabouthd, video | antare, chad, jaad, meva, svt. Every
number below is out-of-fold (fit on three folds, scored on the fourth).

**Coverage.** Content classes bodycam, crowd, lowlight, other, pan, static, videocall (8 clips
each); 360p-720p plus 720x1280 portrait (vc1k), no 1080p; three quantizers spanning medium's
range per codec; key frames only (the P-frame search is a different label set); source grades A
(clean) and B, and 13 ungraded clips (vc1k, uvg, texture). Product content: still cameras (static:
chirla, meva, otw, vidyo), low light, sensor noise (noise sigma 0.2-6.8), crowds and talking heads
(vc1k videocall, vidyo) are all present. Gaps: no 1080p, no screen content, no H.264/VVC/AV2
labels (the tree is codec-free, see below). Rate rule: the dump codes one key at fixed QP, so
the <= 50 %-of-source rule does not bind it; source bitrate per clip is in `probe/clips.json`.

### The gate's target

Per node with both candidates, the gain of a trial = cost of the plan's choice minus the cheaper of
the two, children at their best; divided by λ and the node's area: what a trial is worth per sample
of coded area. A regression model predicts it; a node is trialled where the prediction is above a
threshold; sweeping the threshold traces gain recovered against extra coded area (reached
trialled nodes' area over the frame's, one extra candidate each, as the margin study measured).

### Gain against trials, held out by source

Share of the full key search's gain recovered at the same extra area as margin 1.1 (AV1 0.78,
HEVC 1.10, VP9 1.06 frames):

| gate | AV1 | HEVC | VP9 |
|---|---|---|---|
| margin 1.1 (landed) | 75 % | 72 % | 72 % |
| boosted trees, content + margin + size + codec | 90 % | 94 % | 90 % |
| tree depth 6, same features | 87 % | 94 % | 90 % |
| tree depth 6, planner features only (margin, direction, size, codec, estimate / λ, λ) | 86 % | 95 % | 87 % |
| tree depth 6, planner features without the codec (the built table) | 84 % | 95 % | 87 % |
| tree depth 6, margin, direction, size, codec | 85 % | 89 % | 85 % |
| tree depth 4, margin, direction, size, codec | 76 % | 88 % | 82 % |
| linear score | 69 % | 87 % | 79 % |

Curves (extra area : recovered): margin AV1 0.43:0.53 0.80:0.76 1.19:0.90; depth-6 tree
0.42:0.64 0.84:0.91. HEVC margin 0.69:0.52 1.12:0.73; tree 0.49:0.68 0.78:0.84.

Per class and source the depth-6 tree beats the margin in every cell (AV1 +0.02 on svt to +0.20
on cevo; HEVC +0.13 to +0.32; VP9 +0.11 to +0.28); worst class for the tree AV1 videocall 0.82
(margin 0.71). Source quality: grade A AV1/HEVC/VP9 +0.10/+0.18/+0.16, grade B +0.14/+0.27/+0.23,
so the gain does not come from compressed sources. Out of range (a held-out node outside its
training folds' 0.1-99.9 % range on any content feature): 1.0 % of trial candidates; on them the
margin gate recovers 44 % of the local gain, the tree 91 %.

**The caveat: cost per node.** At the same number of trialled NODES as margin 1.1 the tree loses
(planner features: AV1 75 % vs 75 %, HEVC 61 % vs 72 %, VP9 53 % vs 72 %; codec-free 61 / 61 / 53 %): it buys its area
efficiency by trialling many small nodes. Which one prices a trial depends on how a trial's CPU
splits between per-sample and per-node work, so only real key CPU decides; see "Real encodes".

**Label stability.** The labels come from our encoder, so one landed lever was changed and the
HEVC part of the dump re-coded (`hevc.intra_rd_candidates` 2 instead of 1: the key leaves' mode
search, which moves every coded cost; 145 runs survived, the rest lost to a scheduling fault).
On the new labels, at margin 1.1's area: margin 1.1 70 %, the tree fitted on the OLD labels
96 %, a tree refitted on the new labels (out of fold, less data) 91 %; the old and refitted
trees agree on 88 % of the top-20 % nodes. A leaf-mode lever does not move the gate enough to
need a refit; a refit is one dump (about 7 min on the 96-vCPU build box for three codecs) plus
a 10-minute fit.

**Five codecs.** VVC and AV2 key labels were added (same clips and frames, VVC QP 26 / 32 / 38,
AV2 qindex 80 / 130 / 190; 1530 runs, 20.8 M nodes; the offline planner agrees with every
codec's dumped plan exactly). Codec-free depth-6 tree, held out by source, at margin 1.1's area:

| codec | margin 1.1 | tree |
|---|---|---|
| AV1 | 75 % | 85 % |
| HEVC | 72 % | 95 % |
| VP9 | 72 % | 86 % |
| VVC | 68 % | 86 % |
| AV2 | 60 % | 66 % |

At margin 1.1's node count it loses on every codec (AV1 51, AV2 39, HEVC 60, VP9 50, VVC 58 %).

### Built (default off)

`planner.key_rd_gate` (`crates/nc-encoder/src/trial_gate.rs`): a depth-6 regression tree on the
key plan's margin, its direction, the node size, the plan's leaf estimate per sample over λ and λ,
fitted on every source, codec-free (one table for all six codecs; 123 nodes, 62 leaves; the
score at which it trials margin 1.1's area offline: AV1 33, HEVC 23, VP9 16). Each test is stored in the
form the encoder evaluates with one IEEE multiply and compare (thresholds exponentiated offline),
so no logarithm runs and the decision is the same on every target. Where set it replaces
`planner.key_rd_margin` and `planner.key_rd_merge_margin`; its one value is the predicted gain
per sample (1e-4 λ) a trial must promise.

### Real encodes

One-frame fixed-QP ab (`nceval.fixedqp ab --frames 1 --key-share 0.15`), draw
fixedqp-2026-11-22, n 23-24, BD-J against medium (margin 1.1; AV2 1.2) with the whole one-frame
encode's CPU against the same box's margin 1.1 run (AV2: 1.2). The gate's value is in 1e-4 λ per
sample; "vs margin" is the gate's J minus the margin curve (1.05 / 1.1 / 1.2) interpolated at the
gate's CPU. CIs are about +-0.3-0.5.

Three-codec tree (AV1, HEVC, VP9 labels), commit 22c741ca9:

| codec | g16 | g24 | g40 | vs margin at equal CPU (g16 / g24 / g40) |
|---|---|---|---|---|
| AV1 | -0.3 at 1.08x | -0.3 at 1.07x | +1.0 at 0.93x | +0.07 / +0.03 / +0.54 |
| HEVC | -1.3 at 1.08x | -1.1 at 1.05x | +0.4 at 0.98x | -0.33 / -0.39 / +0.03 |
| VP9 | -0.5 at 1.06x | -0.3 at 1.03x | -0.3 at 0.91x | -0.21 / -0.11 / -1.46 |
| VVC | -1.6 at 1.09x | -1.5 at 1.07x | +1.3 at 0.96x | -0.28 / -0.45 / +0.53 |
| AV2 | +0.8 at 0.94x | +1.2 at 0.90x | +2.3 at 0.79x | +0.35 / +0.39 / +0.58 |

Five-codec tree (VVC and AV2 labels added), commit 7ef23e253:

| codec | g16 | g24 | g32 | g40 | vs margin at equal CPU (g16 / g24 / g32 / g40) |
|---|---|---|---|---|---|
| AV1 | -0.6 at 1.08x | -0.4 at 1.05x | -0.2 at 1.07x | +1.0 at 0.93x | -0.24 / -0.11 / +0.12 / +0.52 |
| HEVC | -1.3 at 1.05x | -1.1 at 1.04x | -1.0 at 1.02x | +0.4 at 0.94x | -0.68 / -0.49 / -0.56 / -0.33 |
| VP9 | -1.3 at 1.11x | -0.3 at 1.05x | -0.3 at 1.07x | 0.0 at 0.97x | -0.81 / -0.11 / +0.01 / -0.54 |
| VVC | -2.4 at 1.01x | -1.9 at 1.03x | -1.9 at 1.02x | +1.4 at 0.92x | -1.94 / -0.97 / -1.46 / +0.28 |
| AV2 | -0.1 at 0.98x | +0.7 at 0.94x | +0.6 at 0.94x | +2.2 at 0.84x | -0.35 / 0.00 / -0.09 / +0.58 |

With AV2's labels in, one value (g16) is at parity or better than every codec's margin curve
at equal CPU (AV1 -0.24, HEVC -0.68, VP9 -0.81, VVC -1.94, AV2 -0.35), so it can replace the five
per-codec `planner.key_rd_margin` settings (1.1, AV2 1.2) with one shared value; against medium
it is AV1 -0.6 % J at +8 % one-frame CPU, HEVC -1.3 at +5 %, VP9 -1.3 at +11 %, VVC -2.4 at +1 %,
AV2 -0.1 at -2 %. The real gain is far below the offline area figure: per node a trial costs more
than its area says. Not screened on the pool levels; the medium preset change (planner.key_rd_gate
16 in each codec's DEFAULT_PARAMS, key_rd_margin unset) needs L3 and final-check.

Retraining cost: a refit is one key dump (1530 one-frame runs: about 7 min for three codecs and
5 min for two on the 96-vCPU build box) and a 10-minute fit; the held-out scoring adds 35 min.

### Pool screens (planner.key_rd_gate=16 against each rung as shipped)

Landed off at ff9548c0f. Stage 1 (`nceval.quick`, draw 2026-11-22: L1 150 f CBR n24, J % with
90 % CI; probe 600 f keys every 10 s, n6) and whole-encode CPU (`aws_cpu_ab`, 150 f CBR with one
key, cycles, zen5 / graviton4, 95 % CI under 0.5 points except where noted):

| rung | margin it replaces | L1 J | worst L1 class | probe J | CPU zen5 / graviton4 |
|---|---|---|---|---|---|
| AV1 medium | 1.1 | -0.83 [-1.11, +0.27] | static +0.7 | +0.56 | +1.63 / +1.53 % |
| AV1 slow | 1.2 | +0.13 [-0.04, +0.35] | texture +0.7 | +0.03 | -0.64 / -0.30 % |
| AV1 slower | 1.2 | -0.40 [-0.58, +0.09] | crowd +0.4 | +0.15 | -0.17 / -0.25 % |
| HEVC medium | 1.1 | +0.39 [-0.02, +0.66] | drone +1.8 | -0.42 | +1.07 / +0.91 % |
| HEVC slow | 1.2 | +0.21 [-0.20, +0.64] | texture +1.7 | -0.95 | -0.06 / -0.07 % |
| HEVC slower | 1.5 | +0.67 [+0.04, +1.18] | texture +3.7 | -0.69 | -1.06 / -0.82 % |
| HEVC veryslow | 1.5 | -0.13 [-0.40, +0.20] | static +0.8 | +1.24 [+0.18, +1.85] | -1.02 / -0.72 % |
| VP9 medium | 1.1 | -0.55 [-0.72, -0.30] | pan +0.2 | -0.13 | +1.87 / +1.40 % |
| VP9 slow | 1.1 | +0.34 [0.00, +0.57] | drone +1.6 | 0.00 | +1.99 / +1.36 % |
| VP9 slower | 1.5 | -0.19 [-0.43, +0.22] | bodycam +1.5 | +0.22 | -2.28 / -1.81 % |
| VVC medium | 1.1 | +0.79 [-0.01, +1.30] | drone +4.1 | -0.15 | +2.49 / +1.89 % |
| VVC slower | 1.5 | +0.59 [-0.06, +1.05] | drone +2.6 | +0.45 | -1.07 / -1.18 % |
| VVC veryslow | 1.5 | +0.25 [-0.16, +0.53] | drone +1.4 | +0.10 | -0.34 / -0.61 % |
| AV2 medium | 1.2 | +0.64 [-0.39, +0.97] | drone +6.5 | +0.52 | +0.43 / +0.24 % |
| AV2 slower | 1.5 | +0.15 [-0.11, +0.31] | drone +1.3 | +0.41 [+0.13, +0.60] | +0.76 (CI +-0.7) / +0.02 % |

At medium the gate costs 0.2-2.5 % whole-encode CPU and is mixed on J; at the slower rungs
(margins 1.2-1.5) it is CPU-neutral or cheaper and within +-0.7 % J. The drone cells are one
clip (roundabouthd_c004) at one rate point (89 kbps, 720p 15 fps); that clip is in the training
dump. Re-coded with per-frame logs, its key is unchanged by the gate (same QP, PSNR-Y 31.98 vs
31.98 VVC, 31.20 vs 31.20 AV2, key bytes -1.5 / -0.4 %); the difference is the later CBR
trajectory at a starved rate (mean PSNR-Y VVC +0.08 dB, AV2 -0.17, HEVC -0.03): single-point
noise, not a coverage gap.

Stages 2-3 (`nceval.train`, base ff9548c0f, draw 2026-11-22; L3 150 f on 108 clips at the
preset's k with the candidate's CPU; final-check 600 f from frame 0, keys every 10 s, n39):

| rung | L3 J | final-check J | train verdict |
|---|---|---|---|
| AV1 medium | +0.29 [+0.10, +0.51] WORSE | +0.13 [-0.08, +0.36] | ejected |
| AV1 slow | +0.07 [-0.16, +0.31] | -0.35 [-0.65, -0.05] BETTER | ejected (L3 ambiguous) |
| AV1 slower | -0.18 [-0.69, +0.35] | -0.33 [-0.66, +0.03] BETTER | ejected (L3 ambiguous) |
| HEVC medium | +0.17 [-0.42, +0.76] WORSE (key pulse +0.32 > 0.25) | -0.13 [-0.40, +0.15] BETTER | ejected |
| HEVC slow | +0.47 [+0.05, +0.89] WORSE | -0.01 [-0.23, +0.20] | ejected |
| HEVC slower | -0.02 [-0.28, +0.24] | -0.05 [-0.36, +0.25] BETTER | ejected (L3 ambiguous) |
| HEVC veryslow | +0.57 [-0.27, +1.41] (drone +4.6) | -0.12 [-0.42, +0.18] BETTER | ejected |
| VP9 medium | +0.56 [+0.18, +0.90] WORSE | +0.01 [-0.22, +0.22] | ejected |
| VP9 slow | +0.25 [+0.05, +0.46] WORSE | -0.33 [-0.53, -0.14] BETTER | ejected |
| VP9 slower | -0.10 [-0.49, +0.31] | +0.15 [-0.18, +0.46] | ejected |
| VVC medium | -0.02 [-0.20, +0.14] | +0.05 [-0.17, +0.27] | ejected (ambiguous) |
| VVC slower | +0.30 [+0.08, +0.50] WORSE | -0.11 [-0.31, +0.09] BETTER | ejected |
| VVC veryslow | +0.12 [-0.09, +0.33] | -0.09 [-0.30, +0.12] BETTER | ejected (L3 ambiguous) |
| AV2 medium | +0.06 [-0.21, +0.34] | -0.63 [-0.83, -0.44] BETTER | ejected (L3 ambiguous) |
| AV2 slower | -0.39 [-0.70, -0.06] BETTER | +0.25 [-0.22, +0.73] | PASS |

**Verdict.** On the pool, the shared gate value is at parity with the hand-tuned margins
(L3 within -0.4..+0.6 %, final check -0.6..+0.25 %), worse on L3 at five rungs (AV1 / HEVC /
VP9 medium, HEVC and VP9 slow), better on the 600-frame final check at nine. It passes one rung
(AV2 slower). It does not retire the per-codec margins, and no preset change follows (coordinator,
2026-11-22).

**Why the one-frame wins did not carry over (read this before re-running it).** The one-frame
harness scores the key alone, where the gate gained 0.1-2.4 % key J. In a real encode the key is
one frame in 150 (L3) or one per 10 s (final check), so its share of the stream's J is small:
even a 2 % key gain becomes tenths of a percent overall, below the per-clip noise. The gate only
changes which key nodes are trialled, so P frames gain nothing directly. Meanwhile at medium
the gate trials more than margin 1.1 (that is where its key gain came from) and pays 0.4-2.5 %
whole-encode CPU, which L3 charges at the preset's k. Net: parity at best, worse at medium on
AV1, HEVC and VP9. Only a key-heavy use (very short key intervals, or the one-frame key chart
itself) would show the harness's gain. The mechanism stays landed off as the bar for a learned key evaluator
(`nn-key`).

## Part 2: a content-adaptive controller, oracle first

### Method

`nceval.arm_oracle` (new): each arm, the preset plus parameters, measured like a ladder family in
CBR (the four ladder rates) and fixed QP (four quantizers spanning the CBR curve), 600 frames,
keys every 10 s, per-frame bits, PSNR-YUV and VMAF v1 kept. AV1 and HEVC medium, draw
`learned-probe-2026-11-22` (not the day's adoption draw), adopt + final levels, 4 per class:
21 clips, 19 with two full key periods (bodycam 4, crowd 4, drone 2, lowlight 4, pan 3, static 4;
no videocall clip is 600 frames long). Arms (each changes another mechanism):

| arm | parameters |
|---|---|
| cad8b4 | `rate.cadence_period=8 rate.cadence_base_boost=4 rate.cadence_startup_period=0` |
| gate0 | `rate.cadence_motion_gate=0` |
| keyrd0 | `planner.key_rd_split=0` |
| prd | `planner.rd_split=1 planner.rd_split_margin=1.2` (the slow presets' P search; much more CPU) |
| aq1 | `rate.aq_strength=1` |
| noise2 | `decision.noise_skip=2` (still-camera skip) |
| dblk | `filter.deblock_skip_share=0.75` |

Per key period (10 s) J2 = the mean of the PSNR-YUV and VMAF v1 BD-rates, full and lower half,
from the per-frame series pooled over the period (no per-frame MS-SSIM, so it is two thirds of
J); whole-clip J (with MS-SSIM) agrees in sign and size. **Split-half:** each clip's arm is chosen
on one key period and scored on the other, both ways, averaged: no selection noise. It is also
the zero-latency causal choice with one key period of lag (choose period 2 from period 1).

### Result (BD-J2 % against medium, class-balanced; negative is better)

| codec, mode, arms | best single (hindsight) | best single (chosen on one half) | per-clip hindsight | per-clip split-half | split-half minus best single, by class |
|---|---|---|---|---|---|
| AV1 CBR, all 7 | prd -1.46 | -1.46 | -2.26 | -1.82 | bodycam +0.4 crowd -0.3 drone -2.2 lowlight +0.2 pan +0.3 static -0.6 |
| AV1 fixed QP, all 7 | prd -2.04 | -2.04 | -2.30 | -1.74 | bodycam 0.0 crowd -0.5 drone +0.7 lowlight +0.3 pan 0.0 static +1.3 |
| AV1 CBR, without prd | cad8b4 -0.14 | +0.13 | -1.11 | -0.58 | bodycam +0.4 crowd -0.3 drone -1.5 lowlight -1.2 pan -0.7 static -1.0 |
| AV1 fixed QP, without prd | cad8b4 -0.51 | -0.51 | -0.97 | -0.28 | bodycam +0.3 crowd -0.2 drone +0.1 lowlight +0.2 pan -0.6 static +1.6 |
| HEVC CBR, all 7 | prd -2.95 | -2.95 | -3.17 | -2.71 | bodycam 0.0 crowd -0.2 drone +0.5 lowlight +0.4 pan 0.0 static +0.7 |
| HEVC fixed QP, all 7 | prd -3.05 | -3.05 | -3.44 | -2.87 | bodycam 0.0 crowd +0.4 drone +0.9 lowlight 0.0 pan 0.0 static -0.3 |
| HEVC CBR, without prd | cad8b4 -0.05 | +0.16 | -0.91 | -0.11 | bodycam +1.6 crowd -1.4 drone -1.1 lowlight -0.2 pan -0.7 static +0.1 |
| HEVC fixed QP, without prd | cad8b4 -1.42 | -1.42 | -1.57 | -0.91 | bodycam -0.1 crowd +0.5 drone +1.0 lowlight -0.4 pan +0.6 static +1.4 |

Share of the per-clip hindsight gain (over the best single arm) that survives split-half: AV1 CBR
45 % (all arms) / 73 % (without prd), 0.36 / 0.71 % J2; every other row none (split-half is worse
than the single best arm by 0.2-0.5). Per-clip SD of an arm's J2 between the two halves:
1.1-2.9 % (aq1 3.6-4.2), the same size as the arm differences, so most of the hindsight spread is
selection noise, as the 2026-11-21 cadence oracle found (docs/analysis/frame-allocation-2026-11-21.md).

**Conclusion: a per-clip (per key period) controller does not pay at this granularity.** The
largest causal gain is 0.7 % J2 (AV1 CBR, equal-CPU arms), under the ~2 % bar; fixed QP and HEVC
lose. Step 2 (a classifier on encoder features) and its mechanism were not built, and no
registered parameter is retired. The period-8 / base-4 cadence's class split (docs/handoff/vvc.md)
did not reproduce as a choosable per-clip signal here: cad8b4 is the single best equal-CPU arm in
both modes and both codecs, so it is a preset question, not a controller one. Rate note: the top
fixed-QP point of 7 clips reached 0.53-0.69x the source bitrate (the harness's span of the CBR
curve), above the 50 % rule; the CBR ladder stayed under it.

## Part 3 (a), handed to nn-lead: how accurate a trial-replacing estimate must be

Each node's coded leaf and split J replaced by c (1 + e N(0,1)), decisions by the noisy costs,
scored on the true ones (key dump; share of the full key search's gain kept, AV1 / HEVC / VP9):

| e | independent errors | errors correlated 0.5 between leaf and split |
|---|---|---|
| 2 % | 87 / 89 / 89 % | 93 / 94 / 94 % |
| 5 % | 40 / 54 / 53 % | 65 / 73 / 72 % |
| 10 % | -68 / -13 / -3 % | -5 / 27 / 28 % |
| 20 % | -279 / -115 / -70 % | -161 / -65 / -39 % |

An evaluator that replaces trials must reach about 2 % relative J error per candidate (5 % if its
errors correlate across the two candidates); guiding which trials to run is the realistic role.
