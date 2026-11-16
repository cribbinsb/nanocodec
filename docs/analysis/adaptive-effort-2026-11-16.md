# Adaptive effort: what a per-superblock and per-frame effort choice can gain (2026-11-16)

Branch `feature/adaptive-effort`. Question (owner): an encoder that chooses per superblock and per
frame how much effort to spend should beat fixed presets at every CPU budget. Today's budget
experiments (`docs/handoff/light-pipeline.md`) did not show that. This note measures how much
there is to gain, what limits it, and what to build.

Everything below is measured unless marked **[inferred]** or **[proposal]**.

## 0. Findings

1. **A perfect per-superblock choice gains a lot.** With the true gain and cost of every
   superblock known (each frame coded both ways first), giving the Default's per-block tools to
   the best 20-28 % of superblocks gets 74-87 % of the Default's whole BD-rate gain for 23-31 %
   of its extra coding time: 4.6 BD points better than the straight line between the two uniform
   settings on six AOM CTC sequences, 6.7 on four of our clips (§2).
2. **Today's proxy recovers little of it, and nothing else cheap does better.** The proxy
   (motion-search SAD and spread in quantizer steps) is 0.3-0.6 BD points better than the line on
   the CTC sequences and 1.4-2.6 on our clips, where almost all of it comes from the one
   static-camera clip. A random choice is on the line (§2).
3. **The oracle's advantage is not predictable before coding.** Coding only the superblocks in
   one quintile of the proxy, of the zero-vector SAD, of the source texture or of the motion
   vector length gives, on the CTC sequences, the same BD-rate per unit of time in every
   quintile (within 0.5 points of the line): gain and cost rise together. A gradient-boosted
   model on ten pre-coding features, the previous frame's coded statistics, and gain maps
   measured on earlier frames all rank superblocks no better than the proxy (§3).
4. **What is predictable is the value per clip and per frame.** The same tools are worth 0.06 BD %
   per % of coding time on a moving-camera clip and 0.43 on a static one (seven times more); a
   frame coded both ways gives a local gain that correlates 0.92 with the clip's realised BD gain
   over ten clips (§3.3).
5. **Realisable today: effort by frame role.** Regular settings on base frames only is 1.6 BD
   points better than the line on the CTC sequences at fixed QP, 0.9 in CBR, 4.0 on our clips; a
   frame controller from probe frames 0.2-1.4; per-superblock maps learned from probes lose (§5).
6. **The tool set per scene is the larger prize, and it reaches above the Default.** Tool groups
   differ eight-fold in value per time and their order changes by clip; with perfect knowledge of
   each clip, real encodes are 1.1-1.6 BD points better than one fixed tool order at every budget
   below the Default, and the second reference is worth 9.7 % on one clip and costs 1.8 % on
   another (§3.5, §5.3, §5.5).
7. **The controller prototype does not beat the presets yet**: 1.5-5.9 BD points worse than the
   line between Faster, Fast, Default and Slow on whole-encoder time. Its measurements are the
   limit: sampling 5 % of superblocks inside each frame is unbiased but too noisy per tool at
   100-300 frames; a frame coded twice is accurate (0.86-0.99 against realised) but costs time on
   short clips (§5.4, §5.6).
8. **Why today's budget did not show a gain** (§4): its heavy end point was itself below the
   Fast-to-Default line by about 2 Weighted points **[computed from the recorded numbers]**, the
   proxy adds about half a point at fixed QP on moving content, and in CBR the proxy picks worse
   than a random choice (+0.2 to +1.0 against the line where random is -0.1 to -0.7).

## 1. Method: the effort oracle

`crates/nc-encoder/src/effort_oracle.rs` (`oracle.*`, default off, branch only). With
`oracle.mode` set, every inter frame is coded several times from the same committed state (the
recode contract, `CodecBackend::RECODE`):

1. every superblock **light** (the `light.*` overlay);
2. every superblock **regular** (the frame's own settings);
3. for real, with the light/regular map the mode chooses.

Each coding records per superblock its entropy-coded bits (`CodecBackend::coded_bits`, AV1 only so
far), the squared error of its reconstruction after the loop filters (all planes) and its wall time
(planner, decisions and coding; the fastest of two identical codings). So for every superblock, with
the references and everything else held fixed:

- gain = [(SSE_light - SSE_regular) + λ (bits_light - bits_regular)] / (λ · frame bits), the
  rate the regular settings save at the encoder's own λ, as a share of the frame;
- cost = (time_regular - time_light) / (frame's all-light time).

Checks: mode 6 (all regular) writes the Default's stream byte for byte, mode 5 (all light) the
stream of `effort.regular_fraction 0`.

Policies (the map of step 3):

| name | map |
|---|---|
| `L`, `R` | all light, all regular (the two uniform settings) |
| `tau T` | oracle: regular where gain > T · cost and gain > 0 |
| `rank F` / `gain F` | oracle: the share F with the highest gain per cost / gain |
| `proxy F` | today's classifier (`effort.regular_fraction` F, `spread_weight` 1) |
| `rand F` | a random share F |
| `rot K` | each superblock regular every K-th frame, phases staggered by position |
| `bin-X-k` | the k-th quintile of feature X within the frame |
| `probe N f F` | realisable: every N-th frame coded both ways (both paid for), the others give the share F with the highest mean measured gain so far the regular settings |
| `frame N t T` | realisable frame controller: every N-th frame coded both ways (both paid for); all regular while the probes' mean gain per cost is above T, all light otherwise |

What is reported for a policy:

- **BD-rate** of its real streams against `L`, per clip (stream bytes, PSNR-YUV of the summed
  squared error of every frame, piecewise-cubic over four quantizers), averaged over clips;
- **time**: the real coding's `code_picture` time over the all-light coding's, both measured inside
  the same run (so machine load cancels), averaged over clips and quantizers. Probe frames count
  both codings. Motion search, pre-analysis, the key frame and I/O are outside it: on lowlight and
  crowd at qindex 170 the whole encoder goes x1.29 and x1.37 from `L` to `R` where the coding time
  goes x1.59 and x1.45;
- **line**: the BD-rate a uniform mix of `L` and `R` would have at that time
  (BD_R · (time - 1) / (time_R - 1)); "vs line" negative is better than the line.

Setting: AV1, the Default as the regular setting, light = Fast's 25 per-block levers
(`tools/research/effort_oracle/bundles.json`, the union of the light-pipeline ablation groups:
transform split off, trellis off, partial forward 2, depth removal 25/50, the pre-transform and
transform-RD skips, P-frame intra off, the motion levers, linear coefficient rate). Fixed QP (the
CLI's fixed-QP profile), one key frame.

- CTC: Johnny, ControlledBurn, WestWindEasy (720p), RedKayak, SpeedBag, TouchdownPass (360p);
  qindex 110, 160, 210, 235; 100 frames; release build on the build box (c6i, shared).
- Ours: lowlight_city_night, crowd_mot17_09, bodycam_bc1, static_meva_hospital (720p); qindex 135,
  170, 205, 235; 120 frames; gate-profile build on the PC under `nice -n 19` (shared).

The oracle is **greedy**: it ranks by the gain inside the frame and does not see that a better
reconstruction also helps later frames. A choice that looked ahead could do better, so it is a
lower bound on a true oracle; its streams are real, so its BD-rate is an achieved result, not a
prediction.

## 2. The oracle, the proxy and a random choice

Six CTC sequences (`remote-out/ctc6`):

| policy | regular share | time | BD-rate % | line | vs line |
|---|---:|---:|---:|---:|---:|
| L | 0 | 0.99 | 0.00 | | |
| tau 0.8 | 0.10 | 1.037 | -3.62 | -0.61 | -3.02 |
| tau 0.4 | 0.14 | 1.071 | -4.90 | -1.16 | -3.74 |
| **tau 0.2** | 0.20 | 1.123 | **-6.58** | -2.02 | **-4.56** |
| tau 0.1 | 0.29 | 1.188 | -7.71 | -3.09 | -4.62 |
| tau 0.03 | 0.44 | 1.318 | -8.98 | -5.24 | -3.74 |
| rank 0.25 | 0.24 | 1.182 | -6.75 | -2.99 | -3.76 |
| proxy 0.1 | 0.10 | 1.126 | -2.34 | -2.08 | -0.26 |
| proxy 0.25 | 0.25 | 1.265 | -4.99 | -4.36 | -0.63 |
| proxy 0.5 | 0.50 | 1.414 | -7.34 | -6.82 | -0.51 |
| proxy 0.75 | 0.75 | 1.489 | -8.46 | -8.06 | -0.40 |
| rand 0.1 / 0.25 / 0.5 / 0.75 | | 1.059 / 1.146 / 1.286 / 1.424 | -1.20 / -2.69 / -4.61 / -6.98 | | -0.23 / -0.30 / +0.11 / 0.00 |
| R | 1 | 1.542 | -8.92 | -8.92 | 0 |

Four of our clips (`out-priv`):

| policy | regular share | time | BD-rate % | line | vs line |
|---|---:|---:|---:|---:|---:|
| tau 0.4 | 0.20 | 1.085 | -8.48 | -1.98 | -6.50 |
| **tau 0.2** | 0.28 | 1.159 | **-10.42** | -3.71 | **-6.71** |
| tau 0.1 | 0.37 | 1.244 | -11.57 | -5.70 | -5.87 |
| proxy 0.1 | 0.10 | 1.155 | -6.19 | -3.62 | -2.57 |
| proxy 0.25 | 0.25 | 1.294 | -8.99 | -6.89 | -2.10 |
| proxy 0.5 | 0.50 | 1.414 | -11.10 | -9.68 | -1.41 |
| rand 0.1 / 0.25 / 0.5 / 0.75 | | 1.074 / 1.163 / 1.284 / 1.412 | -2.78 / -5.12 / -7.97 / -10.32 | | -1.05 / -1.30 / -1.34 / -0.68 |
| R | 1 | 1.509 | -11.91 | -11.91 | 0 |

Per clip, BD-rate %: bodycam / crowd / hospital / lowlight: R -1.8 / -6.0 / -29.5 / -10.4;
tau 0.2 -2.5 / -3.5 / -27.7 / -8.0; proxy 0.25 -3.1 / -4.1 / -22.6 / -6.1; rand 0.25 -0.5 / -2.0 /
-14.4 / -3.7.

- The oracle at about a fifth of the superblocks takes three quarters of the gain for a quarter of
  the cost. Its superblocks are also the cheap ones: `rank 0.25` costs 32 % of R's extra time on
  the CTC sequences where `proxy 0.25` costs 48 % for the same share of superblocks.
- Choosing every superblock with any positive gain (`tau 0.03`) ends level with R (-8.98 against
  -8.92; ours -12.45 against -11.91): picking the better of the two codings per superblock is not
  worth more than coding everything regular. The oracle's value is in what it leaves out.
- A random choice is on the line on every clip but the static-camera one, where a quarter of the
  superblocks at random already gets half of the gain (hospital: -14.4 of -29.5): a good
  reconstruction of a static area keeps being reused.
- The proxy is level with the oracle at a tenth of the superblocks on hospital (-16.8 against
  -15.4) and close to random on the moving-camera clips (bodycam, WestWindEasy, ControlledBurn).

## 3. What predicts the benefit

### 3.1 Per superblock, realised: the gain and the cost rise together

Coding regular only the superblocks of one quintile of a feature (rank within the frame; quintile
4 is the highest), six CTC sequences (`remote-out/ctc6-bins`); BD-rate % at time, and against the
line:

| feature | quintile 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| benefit proxy | -0.37 at 1.031 (+0.13) | -0.73 at 1.052 (+0.10) | -1.16 at 1.088 (+0.24) | -2.65 at 1.150 (-0.27) | -4.21 at 1.232 (-0.52) |
| zero-vector SAD | +0.70 at 1.037 (+1.28) | -0.87 at 1.057 (+0.04) | -1.06 at 1.092 (+0.40) | -2.32 at 1.150 (+0.06) | -3.87 at 1.221 (-0.35) |
| source standard deviation | +0.18 at 1.040 (+0.81) | -0.48 at 1.069 (+0.62) | -1.61 at 1.118 (+0.28) | -2.66 at 1.151 (-0.26) | -3.20 at 1.168 (-0.54) |
| zero-vector SAD / best 16x16 SAD | -1.20 at 1.077 (+0.02) | -1.91 at 1.114 (-0.10) | -1.96 at 1.134 (+0.17) | -2.48 at 1.130 (-0.42) | -0.96 at 1.107 (+0.74) |
| vector length | -1.51 at 1.085 (-0.15) | -1.80 at 1.102 (-0.17) | -1.28 at 1.119 (+0.62) | -2.17 at 1.127 (-0.15) | -1.65 at 1.122 (+0.29) |

The top proxy quintile gains eleven times what the bottom one gains and costs eight times as much.
No quintile of any feature is more than 0.55 points better than the line. On our four clips
(`out-priv-bins`, timed on a loaded PC) the top quintiles are 1.4-1.9 points better than the line
(proxy -8.31 at 1.224) and the bottom ones 1.2-2.2 worse; hospital is 21 of the top quintile's
points.

### 3.2 Per superblock, inside the frame: nothing ranks better than the proxy

From the all-light runs' rows (356,400 superblocks on the CTC sequences, 257,040 on three of our
clips). "Capture at 0.25" is the share of the frame's all-regular gain obtained when a ranking has
spent a quarter of its cost; a random ranking gives 0.25.

| ranking | CTC | ours |
|---|---:|---:|
| oracle (gain per cost) | 1.98 | 1.22 |
| benefit proxy | 0.41 | 0.44 |
| best single pre-coding feature | 0.44 (8x8 SAD spread) | 0.44 (spread, proxy) |
| gradient-boosted model, ten pre-coding features, leave one clip out | 0.49 | 0.44 |
| the same plus the previous frame's coded bits and time | 0.48 | 0.43 |
| the same plus this frame's light coding (bits, error, time) | 0.58 | 0.46 |
| the gain measured for the same superblock one frame earlier | 0.77 | 0.53 |
| the gain map of a probe frame every 8 frames | 0.52 | 0.43 |
| ridge fit of gain on the features, refitted at each probe (every 8) | 0.44 | 0.47 |

- 41-48 % of superblocks have a *negative* gain inside their frame (the regular settings cost more
  at the encoder's λ than they save) and the negative gains cancel 20-29 % of the positive ones;
  the oracle's capture above 1 is from leaving those out.
- The frame's own measured gain is the only strong ranking, and it is what a real encoder does not
  have. The previous frame's measured map keeps part of it (0.53-0.77) and is gone by a lag of
  eight frames.
- Realised, the probe maps lose: `probe 16 f 0.25` is -5.13 % at time 1.383 on the CTC sequences
  (1.07 points *worse* than the line), `probe 4` 4-5 points worse: the second coding of the probe
  frames costs more than the map recovers.

### 3.3 Per clip and per frame: a wide, measurable range

Realised value of the regular settings per clip (R against L), and the mean local gain of a frame
coded both ways (share of the frame's bits, all-light trajectory):

| clip | BD-rate % | coding time % | BD % per time % | local gain % |
|---|---:|---:|---:|---:|
| hospital (static camera) | -29.47 | +68.1 | -0.43 | +47.8 |
| Johnny | -12.71 | +40.7 | -0.31 | +27.5 |
| SpeedBag | -10.20 | +43.7 | -0.23 | +20.5 |
| lowlight | -10.41 | +58.0 | -0.18 | +20.4 |
| TouchdownPass | -12.33 | +68.2 | -0.18 | +15.4 |
| RedKayak | -9.76 | +73.0 | -0.13 | +6.0 |
| crowd | -5.99 | +46.9 | -0.13 | +6.2 |
| WestWindEasy | -4.92 | +51.2 | -0.10 | +8.3 |
| ControlledBurn | -3.62 | +48.4 | -0.07 | +7.0 |
| bodycam (moving camera) | -1.78 | +30.7 | -0.06 | +8.9 |

- The local gain against the realised BD-rate over the ten clips: Pearson 0.92, Spearman 0.72
  (bodycam is the outlier: the regular settings save 15 % of its bits per frame at 2 % more error,
  and most of that is given back because every later frame predicts from the worse picture).
  Measured on the all-regular trajectory instead: 0.92 and 0.88.
- Inside a clip the value also moves with the quantizer (ControlledBurn 0.02 at qindex 110, 0.32
  at 235) and with the frame's role (on the static clips a non-base frame returns 1.2-2.3 times
  what a base frame returns per cost).
- No single cheap statistic orders the ten clips (static share, median zero-vector SAD in steps,
  mean proxy, share of superblocks coded as skip were tried): Johnny and lowlight have low
  proxies and high value, RedKayak and WestWindEasy high proxies and low value. An absolute proxy
  threshold (`effort.regular_proxy`) would therefore move effort *towards* the clips where it pays
  least **[inferred from the table; not run]**.
- With perfect knowledge of the per-cell value, choosing all-light or all-regular per clip and
  quantizer is 1.3-2.1 points better than the line on the CTC sequences at times 1.03-1.37
  (`oracle_frontier.py`) and 2.2-4.2 on our four clips.

### 3.4 Can the encoder measure the gain without coding twice?

A crossover probe (two consecutive frames coded once each with a checkerboard of light and regular
superblocks and its complement; the change of the two halves' J estimates the gain, of their times
the cost; `oracle_cross.py`, `out-priv-cross`):

- **cost: yes.** Crossover against coded twice, % of light time: bodycam 41.8 / 46.7, crowd 55.4 /
  59.4, hospital 64.7 / 71.1, lowlight 54.4 / 57.6 (one pair's standard deviation 5-19).
- **gain: no.** Share of J: bodycam 3.17 / 2.82, crowd 1.76 / 2.28, hospital 0.83 / 3.03, lowlight
  0.10 / 1.73. On static content a superblock that is regular in one frame pays for a better
  picture that its light coding in the next frame then gets for nothing, so the alternation books
  the benefit to the wrong side.

So the gain needs the frame coded twice from the same reference; the cost does not.

### 3.5 Per tool group: an eight-fold range, and a different order on every clip

Each of the nine groups of the bundle alone (the group at its Fast value on every superblock against
the Default; six CTC sequences, box, `remote-out/tools`, `oracle_tools.py`). BD-rate % / coding
time %:

| group | ControlledBurn | Johnny | RedKayak | SpeedBag | TouchdownPass | WestWindEasy | mean | BD per time |
|---|---|---|---|---|---|---|---|---:|
| partial forward (`pf`) | -0.07 / +0.9 | -0.60 / +0.4 | -3.39 / +2.5 | -0.37 / +0.4 | -0.55 / +0.6 | -0.11 / -0.2 | -0.85 / +0.8 | -1.11 |
| P-frame intra (`intra`) | -0.25 / +0.9 | -0.15 / -1.7 | -0.76 / +2.3 | -1.51 / +0.4 | -0.16 / +0.2 | -1.42 / +3.0 | -0.71 / +0.9 | -0.83 |
| coefficient rate (`rate`) | -1.05 / +1.4 | -2.51 / +0.9 | +0.18 / +0.2 | -3.20 / +4.1 | -0.07 / +2.2 | -0.20 / +0.5 | -1.14 / +1.5 | -0.74 |
| depth removal off (`depth`) | -0.40 / +1.3 | -1.34 / -0.2 | +0.03 / +0.1 | +0.10 / +0.4 | -1.24 / +1.8 | -0.06 / +1.1 | -0.48 / +0.7 | -0.65 |
| motion levers (`motion`) | -0.23 / +5.2 | -5.04 / +10.7 | +0.06 / +5.5 | -2.65 / +7.1 | -5.25 / +10.0 | -1.36 / +7.4 | -2.41 / +7.7 | -0.32 |
| transform split (`txsplit`) | -0.82 / +8.7 | -1.18 / +4.2 | -2.05 / +8.8 | -1.15 / +4.1 | -6.18 / +9.1 | -0.98 / +9.8 | -2.06 / +7.5 | -0.28 |
| transform-RD skip off (`txrd`) | -0.39 / +3.2 | -0.93 / +0.7 | -0.10 / +0.4 | -1.39 / +4.5 | +0.80 / +1.8 | -0.06 / +0.1 | -0.34 / +1.8 | -0.19 |
| trellis | -0.31 / +11.0 | -0.83 / +5.7 | -4.72 / +24.1 | -1.46 / +8.9 | -1.05 / +16.1 | -2.14 / +15.4 | -1.75 / +13.5 | -0.13 |
| pre-transform skip model (`pretx`) | -0.08 / -2.4 | +0.05 / -2.8 | +1.57 / -1.6 | -1.79 / +1.8 | -1.31 / -2.8 | +1.51 / -3.8 | -0.01 / -1.9 | saves time |

- The nine single costs sum to +32.5 % of coding time; the whole bundle costs +54 % (and gains
  -8.92 % where the singles sum to -9.75 %): the groups' costs interact. Which pairs is not
  measured.
- The order of the groups differs by clip: the motion levers are worth 5 % on Johnny and
  TouchdownPass and nothing on RedKayak and ControlledBurn for the same 5 % of time; trellis 4.7 %
  on RedKayak and 0.3 % on ControlledBurn; transform split 6.2 % on TouchdownPass and about 1 %
  elsewhere.
- So "uniform mix of two bundles" (the line of §2) is not the right bar for a preset: a preset that
  takes the groups in order of value is far above it (the first four groups are -3.2 % for +3.9 %
  of time), and the oracle of §2 is about 1-2 points better than that order at the same time
  **[inferred: assumes the groups' effects add]**.

## 4. Why today's budget experiments did not show a gain

1. **The end point.** From `docs/handoff/light-pipeline.md`: Fast with the four tools on every
   superblock (`regular_fraction` 1) is 186 fps at -11.8 Weighted. The line from Fast (226 fps,
   -11.0) to the Default (152 fps, -17.9) is at -14.0 there, so the budget's own heavy end is 2.2
   points below the line it was judged against. Between its own end points (taking the 250 fps,
   -6.5 point as the light end, which is an assumption) the budget is about 1.3 points *above* the
   line at fractions 0.25 and 0.5: it did interpolate with a small bow.
2. **The tools.** Trellis and transform split are the two lowest-value groups per time here as in
   the group ablation (-0.13 and -0.28 BD % per % of coding time against -0.65 to -1.1 for the
   cheap groups, which today's Fast already has).
3. **The proxy at fixed QP is not the limit**: nothing computable before coding does better (§3).
4. **In CBR the proxy picks worse than random.** Six CTC sequences, 150-1200 kbit/s
   (`remote-out/ctc6-cbr`), against the line: proxy 0.1 / 0.25 / 0.5 / 0.75: +0.22 / +0.50 / +1.02 /
   -0.09 (worse than a uniform mix); random -0.14 / -0.60 / -0.65 / -0.39; the oracle (tau 0.2)
   -5.39. WestWindEasy at half the superblocks: proxy -2.8 %, random -7.6 %. Under rate control the
   superblocks with the largest residual in quantizer steps are not where the tools pay; why is not
   measured.
5. **One measurement of why** (the all-light runs' rows by proxy quintile, `remote-out/ctc6` and
   `ctc6-cbr`): on the top proxy quintile the regular settings spend +12.7 % of the frame's bits in
   CBR against +7.4 % at fixed QP, for the same -2.4 % of the frame's squared error; its local gain
   per cost falls from 0.24 to 0.19 and the second quintile becomes the best (0.26). Under rate
   control those bits are taken back from later frames by a coarser quantizer **[inferred]**.

## 5. Realisable policies, measured

All as real streams, with every extra coding paid for in the time. "vs line" as in §2.

### 5.1 Effort by frame role: the best simple policy found

Regular settings on base frames (the cadence's high-quality frames, a quarter of the inter frames
here), light on the others; `base+proxy F` adds the proxy's best share F of the non-base frames'
superblocks.

| policy | CTC, fixed QP: BD % at time (vs line) | ours, fixed QP |
|---|---|---|
| base frames regular | -4.80 at 1.204 (**-1.56**) | -9.09 at 1.187 (**-4.02**) |
| base + proxy 0.1 | -5.93 at 1.294 (-1.27) | |
| base + proxy 0.25 | -7.22 at 1.390 (-1.03) | |
| non-base frames regular | -4.61 at 1.367 (+1.22) | -6.33 at 1.273 (+1.07) |
| proxy 0.25 (every frame) | -4.99 at 1.280 (-0.54) | -8.99 at 1.259 (-1.96) |
| each superblock every 4th frame, staggered (`rot 4`) | -2.70 at 1.150 (-0.27) | -5.18 at 1.144 (-1.27) |

In CBR (six CTC sequences, 150-1200 kbit/s, `remote-out/ctc6-roles-cbr`): base frames regular -4.58
at 1.207 (-0.91); base + proxy 0.1 -6.19 at 1.306 (-0.77); non-base frames regular -7.00 at 1.371
(-0.44); proxy 0.25 -4.75 at 1.293 (+0.44); random 0.25 -3.38 at 1.158 (-0.58). The base-frame rule
keeps about half of its fixed-QP advantage; the proxy on top of it adds nothing in CBR.

A base frame is predicted from by every frame until the next one, so what the regular settings add
to it is used several times; the same settings on the frames in between are worth less than a
uniform mix. This is SVT's rule (regular PD1 on base frames) and needs no measurement at run
time: `effort.regular_fraction 0` with `effort.light_nonbase_only 1` codes it today. It was not
among the budget points screened on 2026-11-16 (their `light_nonbase_only` was 0).

### 5.2 Frame controller from probe frames (the whole bundle on or off)

`frame N t T` (§1), six CTC sequences, 100 frames: N = 64: -3.47 at 1.134 (-1.34), -5.43 at 1.254
(-1.41), -6.20 at 1.336 (-0.87); N = 32: -5.17 at 1.267 (-0.85), -6.45 at 1.388 (-0.15); N = 16:
0.5 to 2.3 points *worse* than the line (a probe every 16 frames costs about 6 % of time). With the
proxy's share as a floor on the frames that are not regular (`ladder`): no better (-0.94 to +1.5).
Ours, 120 frames, N = 32: -1.30, -1.38, -0.49; N = 16 worse than the line.

Per-superblock gain maps learned from the probes (`probe N f F`) are worse than the line at every
setting (+0.4 to +7.4): §3.2.

### 5.3 Tool sets per clip against one fixed tool order (the bound for a per-scene tool controller)

Real encodes, six CTC sequences, fixed QP (`remote-out/toolsel`, `oracle_toolsel.py`). `fixed-k`:
the k best groups by pooled value on, the rest at their Fast values, for every clip. `clip-T`: per
clip, the groups whose own BD per time on that clip beats T (perfect knowledge of the clip).

| configuration | coding time / Default | BD % vs Default |
|---|---:|---:|
| fixed-5 / fixed-6 / fixed-7 / fixed-8 | 0.686 / 0.762 / 0.841 / 0.876 | +7.59 / +4.79 / +2.37 / +1.79 |
| clip-0.6 | 0.684 | +6.03 (1.57 better than the fixed order at that time) |
| clip-0.3 | 0.766 | +3.21 (1.45 better) |
| clip-0.2 | 0.806 | +2.15 (1.31 better) |
| clip-0.12 | 0.905 | +0.28 (1.09 better) |
| clip-0.06 | 0.957 | -0.48 (1.08 better; better than the Default at 4 % less time) |

Choosing the tool set per clip is worth 1.1-1.6 BD points at every budget between x0.68 and x0.96
of the Default's coding time, with perfect knowledge. At `clip-0.06` the pre-transform skip model
is off on four clips and the motion levers off on two: the Default runs tools that cost rate on
some clips.

### 5.4 How the controller can measure a tool's value

Three estimators of one group's gain on the current content, each compared with the realised
BD-rate of that group on that clip (54 tool x clip cells, six CTC sequences, 100 frames):

| estimator | extra coding | against realised BD |
|---|---|---|
| **probe frame**: one frame coded with and without the group, same reference | one frame per probe | Pearson 0.71-0.74 at the encoder's λ; per group over the six clips 0.90-0.99 with the squared error weighted double (`toolctl.lambda_scale` 0.5). The probe's squared-error change alone: 0.68; its bit change alone: -0.17 |
| **sampling inside every frame** (owner's question): a fresh random 5 % of superblocks coded with the group toggled, compared with the others, each against its own value one frame earlier (`oracle_sparse.py`) | none | whole bundle, ten clips: agrees with the probe (r 0.65-1.00), standard error 0.6-1.0 points of J after 100-120 frames at 720p. Per group (signals of 0.2-2 points), over the six clips: -0.78 to +0.89 across eight of the nine groups (the pre-transform skip model gave no estimate; transform split -0.78, depth -0.54, motion levers -0.14, partial forward +0.31, trellis +0.55, P-frame intra +0.59, coefficient rate +0.66, transform-RD skip +0.89), where the same frames coded twice give 0.86-0.99 for every group: at 5 % over 100 frames the per-group estimate is mostly noise |
| **crossover** (a checkerboard and its complement on consecutive frames) | none | cost yes; gain no (§3.4) |

- **The pairing matters more than the sample.** A probe frame compares 240 superblocks with
  themselves; 5 % sampling compares 12 superblocks a frame with different ones. For the whole
  bundle (gains of 1-6 points of J) that is enough after about 100 frames at 720p; for one tool
  group it is not, and at 360p (one sampled superblock a frame) not for the bundle either.
- So the owner's first point is the one to build: **paired** samples. Trial tools (transform
  split, intra check, refinement) compute both outcomes on the block already; for the others a
  sampled superblock has to be coded both ways, which needs a superblock-level restart in the
  backend (the AV1 writer already logs its symbols before coding them). Neither is built here.
- **Effect on later frames** (owner's second question). Delayed attribution (superblocks whose
  vector points at least half into sampled superblocks, followed back four frames, against those
  that point into none) finds a carried gain where there should be one: lowlight local +3.7 % of
  J, then +3.2 / +3.2 / +2.0 / +1.4 over the next four frames; hospital and Johnny alike; crowd
  about 0. Its standard error at 5 % over 100 frames is 2-10 points of J, as large as the signal,
  and adding it lowered the correlation with realised BD on the CTC set (local 0.49-0.57, with
  the carried term 0.29-0.58; coded twice 0.75-0.83). It needs about ten times the samples.
- The prototype therefore uses the fallback: a per-scene persistence s / (1 - s), s the mean
  share of superblocks coded as skip, multiplying the squared-error part of the gain
  (`toolctl.persistence`). The carried-to-local ratios above (about 2.5 on lowlight and Johnny, 0
  on crowd) are what it stands in for; it has not been fitted to them.

### 5.5 Above the Default: Slow's groups, per clip

Each group on top of the Default (regular = the Default with the group, light = the Default; six CTC
sequences, fixed QP, `remote-out/up`). BD-rate % / coding time %:

| group | ControlledBurn | Johnny | RedKayak | SpeedBag | TouchdownPass | WestWindEasy | mean | BD per time |
|---|---|---|---|---|---|---|---|---:|
| second reference tried per block (`refs.second 2`, `second_full 1`, `structure.unified 0` for the stream; light superblocks do not try it) | -0.64 / +20.0 | -4.09 / +28.2 | **+1.82** / +11.9 | -3.22 / +24.2 | -9.71 / +27.5 | -9.48 / +16.8 | -4.22 / +21.5 | -0.20 |
| refinement gates off (seed refine, seed λ, skip-no-refine, cheap-refine floor) | -0.10 / +8.2 | -2.36 / +22.7 | -0.26 / +5.6 | -0.99 / +15.8 | -0.38 / +13.5 | -0.15 / +8.0 | -0.71 / +12.3 | -0.06 |
| P-frame intra search 2 | -0.12 / +0.9 | -0.21 / -1.8 | +0.13 / +14.7 | -0.60 / +0.7 | -0.22 / -0.2 | -0.02 / +8.0 | -0.17 / +3.7 | -0.05 |
| transform-split energy gate off | -0.29 / +13.3 | +0.11 / +7.6 | -0.37 / +5.5 | -1.12 / +19.6 | +0.69 / +7.3 | -0.10 / +7.2 | -0.18 / +10.1 | -0.02 |
| residual RD at every vector (`residual_rd_max_mv` 64) | +0.04 / -0.5 | +0.25 / -2.4 | +0.46 / -0.8 | -0.29 / -0.8 | +1.42 / -1.4 | +0.72 / -0.5 | +0.43 / -1.1 | costs rate |

- The second reference is worth 9.5-9.7 % on two clips, 3-4 % on two, 0.6 % on one and *costs*
  1.8 % on RedKayak, for 12-28 % of coding time: the widest spread of any group measured.
  Chosen per clip with perfect knowledge, the five groups are 1.2-1.6 BD points better than
  their one fixed order at +6 to +18 % of coding time **[assumes the groups' effects add]**.
- Not measured this way (they do not switch per superblock): trellis states 4, the key-frame RD
  candidates, the second reference's frame structure itself. OBMC and warped motion are not in
  the encoder.

### 5.6 The tool controller prototype

`crates/nc-encoder/src/tool_control.rs` (`tool<g>.<name> = cheap value`, `toolctl.*`; default
off). Tool groups are parameter overlays; a group runs while its measured gain per cost is above
`toolctl.threshold`. Two ways to measure: probe frames (`toolctl.probe_period`), or sampling inside
every frame with the persistence factor (`toolctl.sample_fraction`), for which the AV1 backend takes
a second per-superblock parameter set (`CodecBackend::set_sample_params`). It starts from the tool
set `toolctl.initial_mask` names.

**It does not beat the presets yet.** Whole range, six CTC sequences, 130 frames, four quantizers,
nine groups below the Default and the five of §5.5 above it, every stream starting as the Default
(`oracle_range.py`, `remote-out/range`; time is the whole encoder's user CPU on the shared build
box, fastest of two runs, so it includes the controller's bookkeeping):

| point | time / Default | BD % vs Default | presets' line at that time | difference |
|---|---:|---:|---:|---:|
| Faster / Fast / Default / Slow | 0.439 / 0.647 / 1 / 1.586 | +59.64 / +9.79 / 0 / -3.31 | | |
| probe frames, threshold 1.0 / 0.32 / 0.16 | 0.898 / 0.900 / 0.951 | +8.69 / +7.77 / +7.22 | +2.83 / +2.78 / +1.36 | +5.9 / +5.0 / +5.9 |
| probe frames, threshold 0.08 / 0.04 / 0.02 / 0.01 / 0 | 1.008 / 1.066 / 1.161 / 1.162 / 1.268 | +4.28 / +2.56 / +1.27 / +0.80 / -0.05 | -0.04 to -1.52 | +4.3 / +2.9 / +2.2 / +1.7 / +1.5 |
| sampling 25 %, thresholds 0 to 1.0 | 1.03 to 1.12 | +2.22 to +3.54 | -0.16 to -0.69 | +2.9 to +3.7 |
| sampling 5 %, thresholds 0 to 1.0 | 1.08 to 1.16 | +2.15 (the same stream at every threshold) | -0.44 to -0.88 | +2.6 to +3.0 |

Every controller point is 1.5 to 5.9 BD points *worse* than the line between the presets. On four of
our clips at 300 frames (`out-range-priv`) the BD-rates are +5.4 to +12.1 % against the Default for
the sampling controller and -0.1 to +10.9 % for the probe controller (hospital alone +18 to +38 %
when its tools are switched off); the times of that run are not usable (a loaded PC: Fast timed
slower than the Default).

What is measured about why:

- At 5 % sampling no threshold changes the stream in 130 frames: a group needs 150 sampled
  superblocks before it is decided and gets about one frame in fourteen.
- With fourteen groups the probe controller codes 16 of 130 frames twice.
- The per-group estimates from sampling are noise at this length (§5.4); on hospital, where the
  lower bundle is worth 29.5 %, the sampling controller ends 18 to 38 % worse than the Default.

Not separated: what the second-reference frame structure costs while its group is off (every
controller stream carries it), and the tracing overhead (per-superblock bit pricing, clock reads and
a squared-error pass on every frame).

## 6. Design conclusions **[proposal]**

What the measurements support, in order of confidence:

1. **Effort by frame role first.** Regular settings on base frames and light on the rest is the one
   policy here that beats a uniform mix everywhere it was run (1.6 points on CTC at fixed QP, 0.9
   in CBR, 4.0 on our clips) and it needs nothing measured at run time. The next screen should be
   `effort.regular_fraction 0` + `effort.light_nonbase_only 1` on the pool levels and final-check.
2. **Per-superblock choice is a small extra and only at fixed QP on static-camera content.** In CBR
   the proxy is worse than chance; it should not carry a budget there until that is understood.
3. **The larger prize is the tool set per scene**: 1.1-1.6 BD points at every budget below the
   Default with perfect knowledge (real encodes), about the same above it (additive estimate), and
   single groups that cost rate on some clips (pre-transform skip model, second reference on
   RedKayak, residual RD at every vector).
4. **The controller needs paired measurements.** A probe frame ranks a group's value across clips
   at 0.86-0.99 against the realised BD-rate; unpaired sampling of 5 % does not at 100 frames. The
   two ways to get pairs without coding frames twice are the owner's: read the trial tools' own
   two outcomes (the transform-split trial has both costs at `choose_tx_split`,
   `nc-av1/src/backend.rs`; not hooked up here), and code a few sampled superblocks both ways,
   which needs a superblock restart in the backend.
5. **Weight quality over bits in the value** (the probe's squared-error change tracks realised
   BD-rate, its bit change does not), and carry it forward by a persistence factor until a
   measured carried term is affordable.
6. **One threshold as the CPU knob** is kept as the design, but the prototype shows it only pays
   once the estimates are good: with noisy estimates it moves away from the Default in the wrong
   places.
7. **Make the controller deterministic** before it leaves the branch: its cost is wall time today.

## 7. Not done, and limits

- Everything is AV1, PSNR-YUV BD-rate (no VMAF, MS-SSIM or J), six CTC sequences and four of our
  clips, 100-130 frames (ours 120-300). No pool screen, no `final-check`, no c8a timing (no c8a
  money was spent), no Jetson run: nothing here beat the presets on the whole-encoder axis, so
  there was no candidate to time.
- Coding time (§2-§5.5) is `code_picture` time inside the run on shared machines; the whole encoder
  moves about 0.6 times as much. §5.6 is whole-encoder user time on the shared box.
- The oracle is greedy inside the frame; a choice that looked ahead could gain more.
- The trial tools' own gain readings, the superblock-level paired sample, the 300-frame per-tool
  runs (the lens copies of the CTC sequences are 130 frames) and trellis states 4 / key-frame RD
  candidates per clip were not done.
- The costs of the nine lower groups interact (+32.5 % as singles, +54 % together); the pairs are
  not identified.
- Why the proxy fails in CBR is one measurement and an inference (§4.5).

## 8. Reproduce

Scripts: `tools/research/effort_oracle/` (`oracle_run.py SET OUTDIR`, `oracle_report.py`,
`oracle_predict.py`, `oracle_sim.py`, `oracle_lag.py`, `oracle_frontier.py`, `oracle_cross.py`,
`oracle_sparse.py`, `oracle_tools.py`, `oracle_toolsel.py`, `oracle_range.py`, `oracle_ledger.py`).
Data: the worktree's `remote-out/` (box runs: `ctc6`, `ctc6-bins`, `ctc6-real`, `ctc6-cbr`,
`ctc6-roles`, `tools`, `toolsel`, `up`, `range`) and `~/nanocodec-data/work/adaptive-effort/`
(our clips: `out-priv*`, `out-sparse-*`, `out-range-priv`; box logs `box-*.log`). The lever ledger
has the end points of the nine lower groups and of the bundle at fixed QP and in CBR
(`doc:effort-oracle end points`).
