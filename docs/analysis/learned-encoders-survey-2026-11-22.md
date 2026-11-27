# Learned encoder decisions: what the reference encoders and the literature do (2026-11-22)

Agent nn-survey, for the coordinator. Question: our learned search gates (the P-frame RD partition
gate in five forms, the fast-rung flip tree, the key-frame tree) all led offline and lost or tied
in closed-loop encodes against the hand-tuned context gate
(`docs/analysis/learned-search-2026-11-22.md` on branch nn-lead, sections 7-11;
`docs/analysis/distill-fast-2026-11-22.md`). What do the established encoders and the literature
do, what do they really gain, and did we miss something?

Sources read: libaom at d24ecae38 (2026-11-22, after tag v3.15.3), libvpx at 29b28a3
(2026-11-20), SVT-AV1 at tag v4.3.0 (e3d63405f), VVenC at 450905fa3 (2026-11-20), all cloned to
`~/nanocodec-data/scratch/nn-survey/`. Every libaom number quoted from a commit message is the
authors' own claim from their internal test sets (lowres / midres / hdres, "borg tests"),
not something we re-measured, except in section 2. x264 / x265 / FFmpeg source was not read.
**UNVERIFIED** marks a claim taken from an abstract, a blog or a second-hand source.

## Summary

- **libaom is the only reference encoder with many learned models**: about 16 small decision
  models (section 1.1). Every one of them is a *pruner inside an RD search that still makes the
  final choice*. Nearly all are one-hidden-layer MLPs (8 to 64 hidden units) or linear models,
  one per block size, plus one 8.5k-parameter CNN for key-frame partitions. Their inputs are
  mostly **the RD results of candidates already coded at this node** (NONE's rate, distortion and
  cost; SPLIT's four sub-block costs; the rectangular costs), plus a cheap simple-motion search.
  The output is compared with a **threshold set per speed level, per block size and often per
  resolution**. Each model was accepted by a full closed-loop encode of the authors' test sets
  against the encoder *without* it (exhaustive search there; speed in timings or, in at least
  one commit, instruction counts). Typical claims are 2-25 % less encode time for 0.02-0.1 % BD-rate at speeds 0-2, with up to
  +0.8 % for 27 % less time at speed 5.
- **The real-time paths use almost none of it.** libaom `--rt` uses variance-based partitioning.
  Its RT ML partitioning has been compiled out since 2020 (`#if 0`, "TODO(kyslov) Re-enable when
  AV1 models are trained"). At speed >= 7 inter frames use the non-RD pick mode and call no
  network. The key frame's RD intra search still uses some models (section 2). libvpx VP9 RT uses its ML
  partition only at speed >= 8 for frames of CIF or smaller. **SVT-AV1 has no learned model at
  all.** Its "classifiers" (light PD0 / PD1 detectors, depth refinement, depth removal) are
  hand-set thresholds on ME distortions, costs of earlier passes and reference-frame statistics,
  and the PD0 detector is switched off for `--rtc`. **VVenC has none either.** Its fast
  partitioning is hand-set cost-ratio and gradient rules. So in the codebases that set our bar
  (SVT-AV1 M12 `--rtc`), real-time search control is hand-tuned.
- **Measured (section 2)**: libaom's learned pruning, on against a build with every model off. 720p
  pool clips, 150 frames, low delay, fixed q, overall-PSNR BD.
  - Good quality, cpu-used 2 and 4: **2.6-2.7x faster for +3.0-3.4 % BD-rate**. That is about half
    the BD cost per unit of time of libaom's own cpu-used 2 to 4 step.
  - Real time, rt6 and rt8: **+0.04-0.08 % BD for 7-18 % less time** (indicative timing).
  - Learned pruning pays against an exhaustive search; in the real-time paths it moves little.
- **Literature (section 3)**: almost all of the CU-partition / mode-pruning work reports time
  saved against *slow reference software* (HM, VTM, libaom at cpu-used 0-1), with labels taken
  open-loop from that software's exhaustive RDO. Few papers compare at equal time against a strong fast
  heuristic, and those few report small or no margins. The clear, verified wins of learned control
  in production are elsewhere: **sequence-level rate control trained by RL on the real encoder**
  (MuZero-RC on libvpx VP9 two-pass: -4.7 to -6.3 % PSNR BD-rate, about 4 % stated for YouTube
  production), and **networking / bitrate controllers trained on deployment data** (Teams
  bandwidth estimation, OnRL). The randomized Puffer trial found simulator-trained ABR policies did
  not beat a simple buffer rule; what won there used ML only for predictions checkable soon after.
- **What we did differently from libaom** (section 4): our value gates read *planner evidence
  before any trial*, while libaom's partition models read the *coded result of the first candidate*
  (and its siblings) and prune what remains. libaom tunes one threshold per speed, block size and
  resolution by full encodes, and accepts a model only for a measured speed-up at about 0.05 %
  loss **against no pruning**. It never had to beat a strong hand-tuned gate. Where a strong
  hand-tuned pruner exists (RT variance partitioning), libaom did not ship a learned one.
- **Next (section 4.3)**: (1) the *post-trial* pruner: the context gate decides as today, and a
  learned score of the planned candidate's own coded outcome (its J, rate, distortion, skip flag,
  nonzero count, beside the context bytes) only *removes* trials the gate admitted. Threshold
  infinity is today's gate, and the threshold is set per rung and block size by encodes. (2) Tune
  the context gate's own thresholds jointly per rung and block size by closed-loop black-box
  search (the literature's most reliable gain; nothing learned), which also gives the bar for (1).

## 1. Reference encoders, from source

### 1.1 libaom (BSD-2-Clause)

The inference code is `av1/encoder/ml.c` (`av1_nn_predict`: dense layers with ReLU, float; SIMD
in `x86/ml_avx2.c`, `x86/ml_sse3.c`, `arm/ml_neon.c`; `av1_nn_softmax`, `av1_nn_fast_softmax_16`)
and `cnn.c` (the key-frame CNN). Weights are C arrays in `partition_model_weights.h` (8119
lines), `partition_cnn_weights.h`, `tx_prune_model_weights.h`, `mode_prune_model_weights.h`,
`misc_model_weights.h` and `intra_mode_search_utils.h`. **No training code or data is in the
repository** (searched `tools/`, the encoder sources and commit messages). The hooks that dump
features are `write_features_to_file` (`--partition-info-path`) and the experimental
external-partition API (`aom/aom_external_partition.h`, the `ext_ml_model_decision_*` calls in
`partition_strategy.c`). Commit messages describe the labels as the encoder's own RD outcome. One
retrain was done on the target configuration: 748f93bfba (2019-03-21), "the model is retrained
using simple_motion_search results from PARTITION_NONE and PARTITION_SPLIT without
two_pass_partition_search". That is, the models were refit when the encoder around them
changed: on-policy in the weak sense of training on the configuration that ships. How the
training scripts set labels or loss is **not in the repository (UNVERIFIED beyond commit messages)**.

Models in use (all in the good-quality / all-intra RD path; the speed levels are from
`speed_features.c` at d24ecae38; origin commit, date and claims from its message):

| decision | where | model | inputs | output use | active (good) | origin and claim |
|---|---|---|---|---|---|---|
| prune HORZ / VERT after NONE and SPLIT | `av1_ml_prune_rect_partition` | 9-32-3 MLP per bsize 8..128 | NONE cost / best, the 4 SPLIT sub-costs / best, 4 quadrant variances / block variance | softmax; prune a direction if its prob <= a per-bsize threshold | `ml_prune_partition` = 1 at every good speed; this model runs where `ml_early_term_after_part_split_level` = 0 (speed >= 3, and speed 0 at 720p+) | 9b5fb2ce6f 2018-08-27 (Bokov): 5-6 % faster at speed 0 for +0.02-0.03 % |
| terminate after SPLIT | `av1_ml_early_term_after_split` | 31-32-1 MLP per bsize | QP, NONE / SPLIT / sub-block RD results, simple-motion-search variances | score > threshold ends the node's search | speed 0 (< 720p), speeds 1-2 | 7918e478aa 2019-04-08 (Su): 3.7-5.4 % faster for +0.02-0.05 % |
| prune AB partitions | `ml_prune_ab_partition` | 10-64-16 MLP per bsize 16..128 | rect / split costs, source variance (comment: "may not be the current block's variance ... Need to re-train") | 16-way bitmask of HORZ_A/B, VERT_A/B | with `ml_prune_partition` | e29094a3b4 2018-06-06 (Su): 12.5-24.7 % faster for +0.02-0.03 % |
| prune HORZ4 / VERT4 | `av1_ml_prune_4_partition` | 18-?-4 MLP per bsize, 3 aggressiveness models (`ml_4_partition_search_level_index` 1-3) | costs of the basic partitions' sub-blocks, variance | per-label thresholds | with `ml_prune_partition` | 74c260ba96 2018-06-29 (Su): 4.2-9.1 % faster for +0.07-0.10 % at speed 0, about 0 at speed 1 |
| break out after NONE | `av1_ml_predict_breakout` | 4-input MLP per bsize ("hd" models) | NONE's rate and distortion, source variance, quantizer | score >= threshold: no SPLIT, no rect; threshold scale per level, per bsize, per resolution | levels 1-3 at every good speed, if NONE is skippable | a2a035f638 2018-08-09 (Su): 7-21 % faster for +0.05-0.09 % (< 720p models first) |
| split-only / no-split before search | `simple_motion_search_based_split` | 17-20/24-1 MLP per bsize, 2 model sets | a simple motion search of the block and its quadrants (SSE, variance, ...) | two thresholds per aggressiveness x resolution x bsize: force SPLIT only, or forbid SPLIT | speed >= 1 (speed 0 at low q) | 748f93bfba 2019-03-21 (Tsai, retrain): 8.2-10.6 % faster, -0.14 to +0.08 % |
| prune rect before search | `simple_motion_search_prune_rect` | 25-8..32-4/10 MLP per bsize | simple motion search features | per-direction thresholds by aggressiveness | every good speed | 79cd069485 2018-11-29; 7de1d8dcdd 2019-08-14 made speed 5 aggressive: +0.73-0.83 % for 25-30 % faster |
| terminate after NONE | `av1_simple_motion_search_early_term_none` | linear (logistic) per bsize | simple-motion features + NONE's log rate / dist / cost | score >= 0 | speed >= 1 | 2b059127b1 2019-02-11 (Tsai): +0.004 % for 1.7 % ("459:1") |
| key-frame split / no-split | `intra_mode_cnn_partition` | CNN on 65x65 luma + log qindex, about 8.5k parameters (34 kB), plus 4 small dense branches | pixels | prune split or all non-split at levels 1-2 | speed >= 1, not screen content | 36035d1cc8 2019-06-19 (Tsai): key frames only -30 to -36 % time for +0.12 %; 33 frames 2.6-2.9 % for +0.05 % |
| max partition size per superblock | `av1_get_max_min_partition_features` + `av1_max_part_pred_nn_config` | 13-48-4 MLP | motion-field stats from a 16x16 motion search | start the search at the predicted size (DIRECT / RELAXED / ADAPT) | by resolution and speed (`auto_max_partition_based_on_simple_motion`) | fd8ece66e6 2018-11-15 (Chen): lowres 4.2 % faster for +0.03 % |
| inter transform type | `prune_tx_2D` (tx_search.c) | separable horizontal and vertical 4/8-8/16-4 MLPs per transform size | horizontal / vertical energy distribution of the residual | softmax over 16 2-D types; keep the top ones by adaptive thresholds per prune level | `prune_2d_txfm_mode` >= 1, inter blocks (good speed >= 1; rt sets level 3) | dd36e78d82 2018-09-05 (Bokov, retrain): 1.7-4.6 % faster, -0.07 to +0.02 % |
| transform split | `ml_predict_tx_split` | 8/12-12..64-1 MLP per transform size | residual mean and std of the block and its sub-blocks | score < -threshold (4000 good, 8500 init): no split | `ml_tx_split_thresh` >= 0 | 79a37242c0 2017-09-29 (Bokov): 3-4 % for +0.02-0.06 %; reworked d9b9ebabac 2018-05-03 (Su): < 0.05 %, 4-22 % on samples |
| skip intra modes in inter frames | `rdopt.c` (after hand rules) | 6-24-2 MLP, separate <= 480p and HD models | best inter mode's skip flag, block w/h, intra cost and inter cost estimates, quantizer ratio | score margin 1.4 | `skip_intra_in_interframe` >= 1 | 8317cac121 2019-09-25, 79abf51abb 2019-10-01 (Wang): 0.7-1.9 % faster for +0.00-0.05 % |
| frame MV precision | `mv_prec.c` | 18-32-1 MLP, frame level | MV statistics of the last frame | high precision if score >= 0 | `high_precision_mv_usage = LAST_MV_DATA` (speeds 0-2) | 51bd648aa9 2019-12-20 (Tsai): **a gain**, -0.06 to -0.15 %, speed neutral |
| directional intra pruning | HOG | linear (0 hidden layers) on HOG bins | gradient histogram | prune directional modes by threshold level | `intra_pruning_with_hog` | bd55259635 2019-11-12 (Su): all-intra about 5 % faster, -0.08 to +0.05 % |
| intra transform depth (all-intra) | `ml_predict_intra_tx_depth_prune` | 14-16-1 MLP | source variance, dc quant, residual stats | prune 8x8 or 4x4 depth | all-intra speed >= 6 | a00e59608f 2022-06-07: 4.7-10.4 % faster for +0.12-0.19 % PSNR |

Patterns:

- **A model is a speed feature like any other.** It is accepted on a closed-loop encode of a test
  set, against the encoder without it, and its aggressiveness is a threshold picked per speed
  level (often per resolution class and block size). The quality cost of each model is tiny
  (hundredths of a percent) because the thresholds are conservative. The speed-ups compound
  because they act on an exhaustive search with many shapes (10 AV1 partition types, 16 transform
  types).
- **Inputs are mostly post-trial.** Six of the partition models read RD results of partitions
  already searched at this node. The others read a dedicated cheap pre-search (simple motion
  search, SMS) run for this purpose. Only the max-partition model, the CNN and the MV-precision
  model decide before any coding.
- **Per-model value at speed 0-2 is a few percent of time each.** The libaom team stacked
  around a dozen. The single large one is the key-frame CNN (a third of key-frame time) and the
  aggressive SMS rect pruning at speed 5 (+0.8 % for 27 %).
- **Real time, `set_rt_speed_features_framesize_independent`**: it sets `ml_prune_partition = 1`
  and similar flags under a comment "Following set of speed features are not impacting encoder's
  decisions as the relevant tools are disabled by default". `partition_search_type =
  VAR_BASED_PARTITION`: a hand-tuned variance-threshold partition with `adjust_var_based_rd_partitioning`
  merges, so no RD partition tree exists for the partition models to prune. `ML_BASED_PARTITION`
  (`ml_predict_var_partitioning`, 6-8-1 MLPs in `av1_ml_partition_models.h`) is behind
  `CONFIG_RT_ML_PARTITIONING` (default 0) *and* `#if 0` in `speed_features.c:1687`
  ("TODO(kyslov) Re-enable when AV1 models are trained"). The weights header says
  "TODO(kyslov): Replace with proper weights after training AV1 models"; the code came in
  28af7875cc (2020-09-25) and data collection in 2c9f625a2b (2020-10-06). From speed 7,
  `use_nonrd_pick_mode = 1`. RT speeds 5-6 still use RD mode search, where the transform-type and
  transform-split networks and the intra-skip network can run (section 2 measures them).

### 1.2 libvpx VP9 (BSD-3-Clause)

- Good quality: `rd_ml_partition` (search breakout, early termination, variance-based pruning
  of NONE / SPLIT, rect pruning), `vp9_speed_features.c` lines 84-342. Example: 6b7848d4c
  (2018-09-29, Su) `ml_var_partition_pruning`: 17.7-18.3 % faster at speed 0 for +0.07-0.16 %.
- **Real time: the one learned search decision in an RT path among the encoders read.**
  `nonrd_use_ml_partition` turns `partition_search_type` into `ML_BASED_PARTITION`
  (`vp9_speed_features.c:764-771`) at speed >= 8, inter frames, no SVC, 8-bit, **frame area <=
  352x288 only**. Commit eed8d4776 (2019-02-19, Jiang): "~10% speed up with no quality change for
  speed 8. 7% quality gain for speed 9 with no speed change". Reverted and relanded
  (152358da7, 78c44e2dc, 2019-05). An earlier experiment, 0bedb351f (2018-10-15, Su), replaced
  REFERENCE_PARTITION at speed 5: -1.63 % (ytlivehr), -0.07 % (ytlivelr), 1.5-14 % faster. It
  was not kept for larger frames. Against libvpx's own variance partitioning, a hand-tuned fast
  heuristic, the learned partition won only at low resolution.

### 1.3 SVT-AV1 (BSD-3-Clause-Clear)

No neural network, regression or other learned model in `Source/Lib` (searched for nn / mlp /
weights / trained / classifier / neural). The decisions named "classifier" in
`enc_dec_process.c` are hand-set threshold rules:

- `pd0_detector` (light-PD0 classifier): the reference frames' intra share of the superblock,
  ME distortion, variance thresholds. `use_pd0_classifier = !scs->static_config.rtc`
  (`enc_dec_process.c:2956`): **off for RTC**.
- `lpd1_detector_post_pd0` / `lpd1_detector_skip_pd0` (light-PD1 classifiers): the PD0 cost, the
  coefficient count and the reference frames' information.
- Depth refinement and depth removal (`Docs/Appendix-Mode-Decision.md`): costs from the previous
  PD stage. Depth removal uses ME distortions normalised to a 64x64 area; a depth is dropped when
  (cost16 - cost8) / cost8 < TH, and so on.

So the reference real-time encoder in our design guide steers its search with hand rules on the
costs of earlier passes, neighbour and reference-frame statistics. That is the same family as our
context gate.

### 1.4 VVenC and VTM (BSD-3-Clause)

VVenC: no learned model. Fast partitioning is hand-set:

- `m_qtbttSpeedUp` (split-cost reuse and skip rules, `EncCu.cpp`, `EncModeCtrl.cpp`);
- `m_fastTTSplit` (skip TT when the BT cost exceeds the parent's or the other direction's by
  `TT_THRESHOLDS` = 1.1 ... 0.95, `vvencCfg.cpp:2110`);
- `m_contentBasedFastQtbt` (`EncModeCtrl.cpp:538-680`): skip horizontal or vertical splits when
  absolute gradient sums in four directions differ by fixed ratios (1.0 / 1.2 and that /
  sqrt 2).

VTM ships none either. JVET has studied CNN partitioning in contributions and its exploration
work, and several published VVC speed-ups target VTM (section 3). None was adopted into VTM or
VVenC as far as their sources show.

### 1.5 x264 / x265 (GPL: documentation only)

Their documentation (x264 man page; x265 CLI docs, read in part) describes hand-designed AQ
(variance and auto-variance, a dark-scene bias, edge information in x265), psy-rd / psy-rdoq,
cutree and early-skip rules. It names no learned decision model. `--analysis-load` /
`--analysis-save` reuse an earlier encode's decisions; they are not learned. Not exhaustive:
UNVERIFIED for undocumented code.

## 2. Measured: libaom's learned pruning on our content

**Setup.** There is no runtime switch for libaom's models. The only exception is the
external-partition API, which replaces them rather than switching them off. So the A/B is a
build-time switch: a ~60-line patch, `patch_noml.py`, kept with the driver `run.sh`, the BD script `bd.py` and the raw
`results.jsonl` in `~/nanocodec-data/work/nn-survey/aom-ml-ab/` (marked keep). The NC_NO_ML build forces these speed features off after each of the three speed-feature
setters:

- `ml_prune_partition` (the rect, AB and 4-way networks);
- `ml_early_term_after_part_split_level`, `ml_predict_breakout_level`;
- the simple-motion-search split / rect / term-none models;
- the key-frame CNN and the max-partition model;
- `ml_tx_split_thresh` = -1, the intra transform-depth network;
- HOG pruning (luma and chroma);
- frame MV precision falls back from LAST_MV_DATA to CURRENT_Q.

It also bypasses the transform-type network (`prune_tx_2D`) and the intra-skip network in
`rdopt.c`, keeping the hand rules beside them. Everything else is unchanged.

**"ML off" means no pruning at those points, not a hand-tuned substitute.** That is the same
comparison libaom's commits make.

libaom d24ecae38, aomenc, one pass, `--lag-in-frames=0` (low delay, as we code),
`--end-usage=q`, cq-level 32 / 38 / 44 / 50 / 56, 150 frames, `--threads=1`. Five 720p pool
clips:

- crowd_cevo_ukof_hd_indoor_light_ohcam_0;
- drone_rounda_roundabouthd_c001;
- pan_jaad_video_0194;
- static_bdd100_bdd_00dc5030_237e7f71;
- vidyo1.

Points above half the source's bitrate were dropped. That left pan_jaad with three points and no
BD figure, so the BD columns are four clips. Modes: `--good --cpu-used=2`, `--good
--cpu-used=4`, `--rt --cpu-used=6` (RD mode search), `--rt --cpu-used=8` (non-RD).

BD-rate is on overall PSNR (aomenc `--psnr`; PCHIP over log rate), not our J. CPU is the user
time of each encode on the shared build box (c7i.24xlarge, 16 encodes in parallel, shuffled
order). It is *indicative*: perf counters are blocked on that box. Box time was about 26 min of
one slot on the shared build box, so no extra rental.

In the table, "ML on" is
libaom as shipped, measured against the NC_NO_ML build:

| mode | BD-rate of ML on vs off | CPU of ML on vs off | per clip BD (crowd / drone / static / vidyo) |
|---|---|---|---|
| good, cpu-used 2 | +3.38 % | -61 % (2.6x faster) | +2.8 / +1.8 / +3.4 / +5.5 % |
| good, cpu-used 4 | +3.04 % | -63 % (2.7x faster) | +3.3 / +1.5 / +3.4 / +4.0 % |
| rt, cpu-used 6 | +0.08 % | -7 % | +0.2 / -0.2 / +0.2 / +0.1 % |
| rt, cpu-used 8 | +0.04 % | -18 % (noisy) | +0.1 / +0.1 / +0.2 / -0.2 % |

The speed ladder for scale (both builds as shipped / both NC_NO_ML):

| step | BD-rate | CPU |
|---|---|---|
| good cpu-used 2 -> 4, ML on | +9.8 % | -77 % |
| good cpu-used 2 -> 4, ML off | +10.3 % | -76 % |
| good cpu-used 4 -> rt cpu-used 6 | +31.6 % | -78 % |
| rt cpu-used 6 -> 8 | +18.0 % | -61 % |

Reading:

- **Inside libaom's exhaustive good-quality search, the learned pruners are worth a lot.** Together
  they make the encoder 2.6-2.7x faster for +3.0-3.4 % BD-rate on our content. That is about
  3.4 % BD per unit of ln(time), against about 6.6 for the ladder's own step from cpu-used 2 to 4.
  They are cheaper per unit of time than the rest of libaom's speed features, so they stay on at
  every good speed. The loss is far above the hundredths each commit claimed alone. The models
  stack (about 15 of them). Our content is camera video at low rate with `--lag-in-frames=0`,
  which the models were not trained on; which of the two causes the gap is not known.
- **In the real-time modes they are worth almost nothing.** Per clip the BD is within +-0.25 %, and
  time is -7 % at rt6. rt8 streams differ from the first frame on: a 1-frame encode already
  differs, so the key frame's RD intra search is where the models act. At speed >= 7 the inter
  frames use non-RD mode decision with variance partitioning and no network. Which model acts on
  the key frame was not isolated. The rt8 time figure is within the box's noise for 5-60 s
  encodes.
- So libaom confirms the pattern of our day. Learned pruning pays where it prunes a large
  exhaustive search against no pruning. Where the search is already narrowed by hand rules (RT),
  the same team shipped no learned partitioning, and the remaining models move little.

## 3. Literature

Two literature sweeps (web, 2026-11-22). "Full text" = the paper itself was read; otherwise
UNVERIFIED (abstract, catalogue or second-hand). TS = time saved, BD = BD-rate loss, both against
the paper's own anchor.

### 3.1 Learned partition and mode decisions (HEVC, VVC, AV1 / VP9)

| paper | decision, model | labels | claim | baseline | closed loop? | vs a fast heuristic? |
|---|---|---|---|---|---|---|
| Xu et al., "Reducing Complexity of HEVC: A Deep Learning Approach", IEEE TIP 2018, arXiv:1710.01218 (full text) | CU split map; ETH-CNN 1.29 M parameters (intra), ETH-LSTM (inter, on a forced-64x64 precode residual) | HM-16.5 RDO, open loop | AI TS 57-67 % for +2.25 %; LDP TS 44-63 % for +1.50 % | HM-16.5 | yes, inside HM | no (other learned methods only) |
| Li et al., "DeepQTMT", IEEE TIP 2021, arXiv:2006.13125 (abstract) | VVC intra QTMT; multi-stage early-exit CNN, loss includes RD cost | VTM RDO | TS 45-67 % for +1.3-3.2 % | VTM | yes | no |
| Tissier et al., IEEE TCSVT 2023, arXiv:2103.05319 (full text) | CNN edge probabilities + LightGBM picks top-N splits for RDO | VTM AI with VTM's speed-ups *off* (open loop, slower than the anchor) | top-3 TS 47 % for +0.79 %; top-2 70 % for +2.49 % | VTM-10.2 with its speed-ups on | yes | no |
| Liu et al., arXiv:2310.13838, 2023 (full text) | VVC inter partition path; light U-Net on motion fields | VTM-10 RA | TS 17-60 % for +0.44-4.6 % | VTM-10 RA GOP32 | yes | re-implements a decision-tree method's fast setting in VTM: 0.84 % less loss at equal time. **States that the Feng et al. CNN costs 2.34 % of VTM time but about 67 % of VVenC-fast time; VTM-10 AI is 27x VVenC fast** |
| Feng et al., IEEE TIP 2023 (abstract) and arXiv:2504.18398, 2025 (full text) | partition-map CNN, intra and inter | VTM-10 with fast tools off "to ensure an accurate block partitioning structure" | RA TS 44-51 % for +1.7-2.1 % | VTM-10 | yes; the authors state "the fast algorithm cannot account for the encoding context"; VTM-10 to VTM-23 without retraining costs +0.43-0.85 % | no |
| Amestoy et al., IEEE TIP 2020 (abstract) | cascade of random-forest skip decisions with a "risk interval" falling back to full search | VTM | TS 25-61 % for +0.4-2.2 % | VTM-5 | yes | no |
| Saldanha et al., IEEE TCSVT 2022 (abstract) | five LightGBM classifiers, texture + coding + context features | VTM | TS 35-61 % for +0.5-2.4 % | VTM-10 | yes | one of the two methods a 2026 survey finds above VTM's own configuration curve |
| **Tech et al., "Rate-Distortion-Time Cost Aware CNN Training for Fast VVC Intra-Picture Partitioning Decisions", PCS 2021** (slides full text; paper UNVERIFIED) | CNN outputs two numbers per 32x32 block: minimum MTT width and height | **each setting is coded per block, its D + lambda R + mu T recorded; the encoder continues from the chosen setting's state (labels on the restricted encoder's own path); loss = the interpolated recorded cost, not cross-entropy**; mu per QP fixes the time saving | TS 50 % for +0.7-0.9 %; 73 % for +3.7 % | VTM-7 / VTM-10.2, defaults on | yes | the other method above VTM's configuration curve |
| Tech et al., SPIE 11842, 2021 (abstract, UNVERIFIED) | the same, ported into VVenC | | "is efficient" across the five VVenC presets; numbers not seen | VVenC presets | yes | the nearest learned-vs-fast-preset Pareto test for VVC; numbers not obtained |
| Taabane et al., Electronics 2023 (abstract) and IEEE Access 2024 (catalogue) | LightGBM split classifiers in VVenC, features and labels from VVenC itself | VVenC | intra TS 30-83 % for +0.7-3.0 % by preset | VVenC presets | yes | equal-time comparison with the next faster preset not seen |
| Galpin et al., DCC 2019, arXiv:2011.06691 (full text) | CNN split probabilities vs depth thresholds, about 225 k parameters | JEM-type codec | 2x at no loss, > 4x for < 1 %, 10x for 3.8 % (AI) | **the codec with its standard speed-up heuristics on; the CNN added on top** | yes | partly: on top of heuristics, not against a faster preset |
| **Paul, Norkin, Bovik, IEEE TIP 2020, arXiv:1906.06476** (full text) | H-FCN (from 26 k parameters) predicts the whole 64x64 partition tree, replacing the RD partition search (intra) | libvpx `--good --cpu-used 1` | TS 69.7 % for +1.71 % (inference counted) | libvpx cpu-used 1 | yes | **yes: against `--cpu-used 4` it is 4.3 % faster with 1.44 % less loss. The only verified learned-beats-a-fast-setting result found; intra only, good-quality mode, not real time** |
| Kim et al., arXiv:1908.11166 (abstract) | AV1 decision trees: test all inter modes or not | libaom | -43 % time, BD not in the abstract | libaom, setting not stated | ? | no |
| Kherchouche et al., "Partition Tree Search Acceleration for VVC: Survey and Evaluation with VTM Evolution", arXiv:2605.21526, 2026 (full text) | survey + their own DQN trained offline on about 12 M VTM-18 trajectories | | typical AI claims TS 25-61 % for +0.7-2.8 %; their DQN "comparable" | VTM | survey plots reported numbers (no re-implementation) | **"Only the methods proposed by Tech et al. and Saldanha et al. consistently outperform the corresponding VTM baseline"** (VTM at other MaxMTT depths) |
| Wieckowski et al., "Review and evaluation of VVC fast partitioning search methods using a common baseline", MHV 2024 (abstract only, UNVERIFIED) | Fraunhofer re-places published heuristic and neural methods on one baseline | | | | | likely the most relevant critique; PDF not obtained |
| HEVC classics (Zhang, Kwong et al. TIP 2015, cascaded SVM; Li et al. JVCIR 2019, actor-critic; online-updated models of Kuang et al., Mallikarachchi et al.) (UNVERIFIED) | | | about 50 % TS for about +2 % vs HM | HM | | no |

Patterns:

1. The typical claim is 40-70 % TS for +0.7-2.5 % (intra) or 20-50 % for +0.4-2 % (inter),
   **against HM or VTM**, which is 27x (AI) to about 140x slower than VVenC's fast presets. Almost
   any reasonable gate wins against an exhaustive search. Most papers do run full closed-loop
   encodes; the weakness is the anchor, not open-loop evaluation.
2. **Learned vs a strong fast heuristic at equal time: one verified win** (Paul et al., VP9 intra,
   good quality). There is one survey finding (two VVC methods above the VTM configuration
   curve) and one unquantified VVenC port (Tech et al. SPIE). No paper was found comparing
   against an x265 fast preset or SVT-AV1 / libaom real time.
3. **Overhead does not transfer**: a gate that costs 2 % of a slow encoder costs a large share of a
   fast one (2.34 % of VTM became about 67 % of VVenC fast).
4. **Labels are usually open loop** (the reference encoder's exhaustive RDO, sometimes with its
   speed-ups switched *off*). Methods that do address the closed loop: Tech et al.
   (rate-distortion-time cost recorded per setting along the gated encoder's own path, cost as
   the loss, one mu per speed target); RD-cost-aware losses (DeepQTMT, Kherchouche); labels and
   features from the target fast encoder (Taabane / VVenC); thresholds swept by full encodes
   (nearly all). RL agents (Li 2019, Kherchouche 2025/2026) are trained offline on recorded
   trajectories and report no advantage. No DAgger-style iterated retraining was found.
5. Production practice matches our result: small learned gates at slow good-quality speeds,
   hand rules (variance, context) at fast and real-time speeds (section 1).

### 3.2 Learned rate control, QP and lambda

- **MuZero-RC** (Mandhane et al., "MuZero with Self-competition for Rate Control in VP9 Video
  Compression", arXiv:2202.06626, 2022; full text). Frame QP in libvpx VP9 two-pass VBR,
  `--cpu-used=1`. MuZero trained **by RL on the real encoder** (libvpx SimpleEncode with the QP
  overridden; 20 k 480p clips). Reward is "self-competition": +-1 against the agent's own running
  average for that (video, bitrate), constraint first. **Lagrangian constrained RL was tried first
  and was "difficult to tune"** (their appendix A.4). Against libvpx's own two-pass VBR, a strong
  production heuristic, on 3062 YouTube UGC clips: PSNR BD -4.72 % (augmented -6.28 %), SSIM -3.68
  / -5.11 %, **VMAF only -0.53 / -1.88 %**; overshooting clips 64 % -> 20 / 16 %. DeepMind's blog
  (2022-02-11) states it was launched on part of YouTube's live traffic with an average 4 % bitrate
  reduction (the production figure is the blog's). A 2023 blog says it was extended to GOP grouping
  and referencing (UNVERIFIED: no paper or numbers).
- **Imitation-learned RC** (Mao, Gu et al., arXiv:2012.05339, 2020): the experts are per-video
  optimal QP sequences found by evolution strategies over real encodes, imitated by a policy.
  -8.5 % median bitrate against libvpx two-pass VBR; they measure a 13 % gap between that VBR
  policy and the per-video optimum.
- RL frame-level bit allocation for HEVC (Chen, Hu, Peng 2018; Ho et al. 2021, 2022; UNVERIFIED /
  abstracts). Their anchors are HM rate control (weak) or x265 two-pass ABR. Against x265
  two-pass they are "comparable": a tie against a strong heuristic.
- CNN-predicted R-lambda parameters per CTU (Li et al., VCIP 2017, UNVERIFIED): -0.7 % against HM
  rate control (weak).
- libvpx / libaom **external rate-control and partition interfaces** (`vpx/vpx_ext_ratectrl.h`,
  `aom/aom_ext_ratectrl.h`, `aom/aom_external_partition.h`). They expose QP, GOP, rdmult and
  partition decisions to an outside model, fed by first-pass and TPL statistics. Their link to the
  DeepMind work is plausible from timing; UNVERIFIED.
- **Not learned, but closed-loop and deployed**: Netflix's Dynamic Optimizer (per-shot convex hull
  of real encodes; about 30 % against a fixed ladder), and the "fast DO" (search with a fast
  encoder, apply to the slow one; Wu et al., SPIE 2021).

### 3.3 RL for encoder and transport control

- Partition / mode decision by RL (Chung 2017, Li 2019, Zhao 2022, Kherchouche 2025/2026): HM / VTM
  anchors, offline trajectories, no advantage over supervised methods reported.
- **Deployed, against production heuristics**, all trained on real deployment data or online in the
  live system and judged by large A/B tests:
  - Microsoft Teams bandwidth estimation: offline distributional RL on call logs against the
    production Kalman estimator. Khairy et al., arXiv:2510.12265; AAAI 2026 deployed track.
  - OnRL: online RL in Taobao live, with a fallback; stalls -14 % at similar quality. MobiCom 2020.
  - Facebook ABR: 30 M sessions; +1.6 % bitrate, -0.4 % stalls. Its reward was tuned by Bayesian
    optimisation. arXiv:2008.12858.
- **Counter-example**: Puffer (Yan et al., NSDI 2020). In a randomized trial on real users,
  simulator-trained ABR policies (Pensieve) did not beat a simple buffer-based rule. The winner
  (Fugu) was a classical model-predictive controller around a learned predictor, trained in
  place, with ML limited to "predictions that can be checked soon after".

### 3.4 Learned perceptual weighting and AQ

- x264 and x265 documentation: AQ modes (variance, auto-variance, dark bias, edge), psy-rd,
  psy-rdoq and cutree are all hand-designed; neither documents a learned model.
- Learned pre-processing, not encoder control: iSize deep perceptual preprocessing (Chadha and
  Andreopoulos, CVPR 2021) claims 11-17 % against the same encoder without it, with a risk of
  gaming VMAF. The Netflix deep downscaler gives about 5.4 % VMAF BD on VP9 and is deployed in
  Cosmos.
- Learned saliency / JND QP maps: typically 7-9 % against fixed QP or reference software. No case
  was found against x264 / x265's tuned AQ, and no confirmed production deployment.
- A differentiable learned-codec proxy for macroblock QP maps in H.264 intra (Xu and Bajic,
  arXiv:2607.10478, 2026): -15 % MS-SSIM against **fixed-QP** H.264 (a weak baseline).

### 3.5 Black-box tuning of hand-written heuristics by full encodes

Consistently positive and cheap:

- Ringis et al., Frontiers in Signal Processing 2023 (full text). Per-clip lambda search: x265
  -1.87 % PSNR over 10 k UGC clips; libaom -2.5 % with per-frame-type lambda. One global x265
  lambda x0.782: -0.63 %. Searching at libaom speed 6 instead of speed 2 found all of the gain at
  230x less cost.
- Disney Research's Bayesian optimisation of x264 parameters per title: -14.5 % PSNR BD against a
  default preset (a weak baseline). A learned predictor recovers about 80 % of it from one fast
  encode. Venue UNVERIFIED.

## 4. Synthesis for nanocodec

### 4.1 Where learned control demonstrably beats strong hand-tuned heuristics, and where it doesn't

Demonstrated:

- **Sequence-level rate control trained by RL on the real encoder** (MuZero-RC against libvpx
  two-pass VBR: -4.7 to -6.3 % PSNR BD, mostly on the trained metric; a 4 % production claim).
  Also imitation of per-video optima found by search (-8.5 %).
- **Transport and bitrate controllers trained on deployment data or online**, judged by large A/B
  tests (Teams, OnRL, Facebook ABR).
- **Pruning an exhaustive RD search, against no pruning**: libaom (measured here, 2.6x for +3 %),
  libvpx good quality, the VTM / HM literature.
- **One equal-time win against a fast setting**: Paul et al., VP9 intra, good quality.
- **One real-time partition win**: libvpx VP9 speed >= 8, frames of CIF and below.

Not demonstrated:

- **Replacing a strong fast hand-tuned search gate in a real-time encoder.** libaom compiled its RT
  ML partitioning out for six years. SVT-AV1 and VVenC use none. The VVC literature reports
  against VTM, and its gates' overhead becomes a large share of a fast encoder's time. A 2026
  survey finds only 2 methods consistently above VTM's own configuration curve.
- Learned AQ / perceptual weighting against tuned AQ.
- RL partition or mode agents.
- Simulator-trained policies in the field (Puffer).

Our result (offline lead, closed-loop tie or loss against the context gate) is the expected
outcome in the second group, not an anomaly.

### 4.2 What they did that we didn't

1. **Post-trial inputs.** libaom's partition models fire *after* NONE (and SPLIT, rect) are
   coded, and read those candidates' coded rate, distortion and cost. Our gates read the planner's
   evidence before the alternative is coded (11 bytes; 79 in the wide gate).
   - The planned candidate's coded outcome was in group D of the inventory (section 8.1 of the
     nn-lead document). Only its J was used; its bits, distortion, skip flag and nonzero count
     were "missing", and the v0.3 test with them was dropped.
   - libaom's breakout model uses exactly those four numbers (NONE's rate and distortion,
     variance, quantizer). It is the cheapest model in its set, on at every good speed.
2. **Pruning, not replacing.** Each libaom model only removes candidates. The RD search keeps the
   decision, and a pruned-out choice costs only what it would have gained. Our replacement arms
   (value gate instead of the context gate) also *admit* trials the context gate rejects, and
   those bought little (the OR arm, section 9.6). The AND arm was the right shape but used the
   pre-trial inputs.
3. **The threshold is the product.** Every libaom model's output is cut at a threshold per speed
   level, per block size and often per resolution class, picked by full encodes. Tech et al. do
   the same through mu per QP. We swept one price per rung.
4. **The bar.** libaom accepted a model for a measured speed-up at about 0.05 % against *no
   pruning*. It never had to beat a strong pruner of the same decision, and where one existed (RT
   variance partitioning) it shipped none. We required the learned gate to beat the context gate,
   which is a much harder bar.
5. **Labels on the shipping configuration.** libaom retrained when the encoder around a model
   changed (748f93bfba). Tech et al. collect labels along the restricted encoder's own path with a
   rate-distortion-time cost as the loss. Our first fits used a diagnostic dump; the later ones
   used the engine's own bytes (on-policy shards). That part we did.
6. **Not tried by anyone, as far as found**: DAgger-style iterated retraining on the deployed
   gate's own decisions. The closed-loop methods that exist are cost-aware losses, on-path labels,
   threshold sweeps by full encodes and RL on the real encoder (rate control only).

### 4.3 Worth trying next

Rule: the learned version contains today's heuristic as a degenerate case, judged by real
encodes.

1. **A post-trial pruner inside the context gate.** At a node the context gate admits, the planned
   candidate is coded first (as now). A small score then reads that coding outcome:

   - its J, rate, distortion, skip flag, nonzero count and largest transform;
   - the planner's margin and the two context bytes.

   The score decides whether to *skip* the alternatives, like libaom's breakout and
   early-termination models.

   - The gate stays exactly as it is. The pruner can only remove trials it admitted, so a
     threshold of infinity is today's encoder bit for bit.
   - The threshold is set per rung and per block size by the fixed-QP A/B and cycles.
   - Accept it if it moves the rung below the straight line from the shipped rung to "search off".
     The AND hybrid with pre-trial inputs sat on that line (section 9.4).
   - Cheapest first: a logistic score on 6-8 numbers (libaom's term-none model is linear) before
     any MLP.
   - Data: the dump already codes the planned candidate first. The missing columns are the v0.3
     ones. The same idea applies at the slower rungs (they ship the ungated search) and on key
     frames (`key_rd_split`).

2. **Closed-loop black-box tuning of the context gate's own thresholds**, jointly per rung and
   per block size. Nothing is learned; this is the joint-parameter tuning path (`nceval.tune`).
   The literature's most consistent gains are of this kind (lambda search, Bayesian optimisation
   of encoder parameters, libaom's per-speed thresholds). Section 9.4 notes that looser context
   thresholds at the slower rungs were not tried. This both raises the bar honestly and is the
   comparison idea 1 must beat at each rung.

Not recommended now:

- **RL rate control.** The only verified learned win against a strong production heuristic, but
  for two-pass VOD frame QP, at thousands of encodes per clip, and gaining mostly PSNR. For live
  CBR it is a separate project with its own budget.
- **Pixel CNNs for partitioning.** In fast encoders the overhead does not transfer.
- **RL partition agents.** No reported advantage.
