# Frame-level bit allocation under CBR: the prize, measured (2026-11-21)

**Question (owner).** CBR plans each frame's quantizer to hit a budget, but the budget between P frames
is near-uniform. Would planning each frame's bits from content statistics (spend where later frames
reuse it, less where they do not) raise CBR quality, and improve how well frames and windows meet their
targets?

**Answer, from oracles, before any mechanism was built.**

- **Moving bits between P frames by reuse gains nothing.** A hindsight reuse weighting of every inter
  level moved J by -0.3..+1.1 %, and its opposite-sign control by -0.4..+0.2 %.
- **Choosing the cadence per clip with hindsight** gains at most 0.6-2.8 %, which is the size of the
  per-clip noise of any live parameter change (SD 1.3-2.2 %).
- **The cadence that exists is worth 5-7 %** on the product window (off: AV1 +5.1 %, HEVC +6.9 %), and
  its depth already follows a measured reuse signal (`structure.inherited_depth`, `inheritance.rs`).
- **The CBR-to-fixed-QP gap (+15..+32 % BD-J) is the key frame.** A fixed-QP key is 3.4 (static) to 8
  (drone) buffers large and 3-6 dB above the P frames before it. CBR's key is bounded near the 1.4-buffer
  re-code ceiling and, on static and drone clips, lands 1.1-2.2 dB below them. On the product window
  93-95 % of the fixed-QP-key replays break the buffer CBR runs in (71-92 % on 150 frames).
- **What is left at frame level is the still-scene burst cycle** (static, drone): the largest size-model
  errors, the re-codes, the overflows and the worst one-second windows are all there.
- **The key's size is the lever, and it is priced in delay** (section 2). Live, on periodic keys of still
  scenes (`rate.key_hold_still`): 2 buffers gives J -1.7..-2.9 % for 130-180 ms more delay at a
  constant-rate channel, 3 buffers -3.1..-4.3 % for 200-270 ms; the stream stays under target.
- **A better size model was built, does not pay, and was deleted** (section 4): windows improve a little, J loses
  0.9-2.0 % on the product window.

Nothing here changes a stream.

## Method

`python3 -m nceval.rc_oracle` (2026-11-14) replays a CBR point's logged quantizers with one part changed
(`--qp-replay`; the controller's budgets and buffers run as usual, a forced frame is never re-coded).
Added for this study:

| Addition | What it measures |
|---|---|
| `reuse1`, `reuse2`, `reuse_anti` | each inter level 1 (2, -1) QP unit finer per doubling of what the next second of frames inherits from the frame, at the sequence's mean offset. The inherited share per frame is `1 - inter cost / intra cost` from the coded sizes (a macroblock tree's propagation fraction, per frame), accumulated over one second, cut at keys |
| `key_finer`, `key_coarser` | every key 4 QP units finer (coarser), inter frames as logged |
| `--arm NAME:p=v` | a live CBR arm with parameters; `best arm per clip` picks each clip's best with hindsight (it includes selection noise, so it is an upper bound) |
| `--preset`, `--rotation` | a shipped preset; a draw key of its own |
| conformity table | per point from the rate log: stream and one-second windows over target, cadence-cycle bits over budget, buffer fullness, overflows, re-codes, size-model error per role |
| per-frame logs and PSNR-Y in the report | offline fits |

The rate log gained four columns at its end (`spatial_hint`, `temporal_hint`, `camera_motion`,
`compensable`). `tools/dev/aws_measure.py` runs `nceval.rc_oracle` on a rented box.

**Runs.** Build f0845c1eb (variance-integration d83c432a8 plus the log columns; streams unchanged).
Draw key `frame-alloc-diag`, not the day's, so the adoption draw was not looked at. CPU scoring on
rented c7a/c7i.24xlarge boxes; the whole study, sections 2 and 4 included, about $5.

- **150 f**: the draw's quick level, 24 clips (crowd 4, lowlight 4, pan 4, static 3, bodycam 3,
  videocall 3, texture 2, drone 1), one key, the 4-rung ladder. AV1 medium, HEVC medium, AV1 faster
  (layered).
- **600 f**: 12 clips, two per class (bodycam, crowd, drone, lowlight, pan, static; no videocall clip is
  600 frames long), keys every 10 s. AV1 medium, HEVC medium.

Class columns are means of 1-4 clips: indicative. BD-J is the project's J (PSNR-YUV, VMAF v1, MS-SSIM in
thirds; mean of the full and lower-half BD-rates), class-balanced. Positive costs bits.

## 1a. What CBR loses against fixed QP

| BD-J % | all | bodycam | crowd | drone | lowlight | pan | static | texture | videocall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV1 medium, 150 f | +20.7 | +5.1 | +22.9 | +72.3 | +12.8 | +7.1 | +42.5 | +5.0 | -2.3 |
| HEVC medium, 150 f | +30.0 | +8.0 | +44.2 | +89.2 | +16.7 | +18.7 | +57.8 | +5.3 | -0.1 |
| AV1 faster, 150 f | +15.0 | -3.4 | +16.7 | +63.9 | +10.2 | -0.6 | +43.2 | -2.0 | -7.7 |
| AV1 medium, 600 f | +23.3 | +3.2 | +14.1 | +62.0 | 0.0 | +4.9 | +55.6 | | |
| HEVC medium, 600 f | +31.8 | +8.3 | +20.4 | +73.9 | 0.0 | +8.8 | +79.4 | | |

### By part (the replay chain)

| Step, BD-J % | AV1 150 | HEVC 150 | AV1 faster 150 | AV1 600 | HEVC 600 |
|---|---:|---:|---:|---:|---:|
| key allocation (fixed-QP key ratio against CBR's key, levels flat) | +13.3 | +16.2 | +9.4 | +22.2 | +23.1 |
| slow drift (flat against centred 1 s mean) | +2.5 | +8.4 | +2.2 | | |
| fast jitter (centred 1 s mean against as logged) | +2.5 | +1.1 | +1.9 | | |
| drift and jitter together (600 f) | | | | +1.2 | +5.6 |
| re-codes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| the rest | +1.1 | +2.1 | +0.9 | -1.1 | +0.8 |

Key allocation by class, 600 f (AV1 / HEVC): static +61.8 / +61.3, drone +49.4 / +43.5, crowd
+16.8 / +24.9, pan +6.9 / +11.5, bodycam +0.4 / +0.8, lowlight -2.0 / -3.1.

### The fixed-QP key does not fit the buffer

Points whose replay goes past CBR's own worst (largest one-second window by more than 5 %, or the buffer
past the larger of 1.0 and CBR's own peak):

| Replay | AV1 150 | HEVC 150 | AV1 faster 150 | AV1 600 | HEVC 600 |
|---|---|---|---|---|---|
| fixed-QP key (`flat_key`) | 66 / 90 | 64 / 90 | 83 / 90 | 41 / 43 | 40 / 43 |
| constant level, CBR's key (`flat`) | 22 / 90 | 14 / 90 | 52 / 90 | 28 / 43 | 20 / 43 |

AV1 medium, 600 f (class means over the rungs):

| | bodycam | crowd | drone | lowlight | pan | static |
|---|---:|---:|---:|---:|---:|---:|
| CBR: peak buffer fullness (buffers) | 0.97 | 0.89 | 1.38 | 0.42 | 0.57 | 1.32 |
| fixed-QP key: peak buffer fullness | 2.58 | 1.97 | 8.02 | 3.95 | 1.22 | 3.44 |
| CBR: largest 1 s window (x target) | 1.74 | 1.70 | 2.12 | 1.29 | 1.43 | 2.09 |
| fixed-QP key: largest 1 s window | 2.29 | 2.32 | 5.40 | 1.96 | 1.73 | 3.96 |
| CBR: key at frame 300 against the 12 frames before (PSNR-Y dB) | +2.41 | +0.85 | -2.16 | +3.09 | no key | -1.14 |
| fixed QP: the same | +4.87 | +3.12 | +5.16 | +2.34 | no key | +5.85 |
| CBR: first key / second key (seconds of the stream's rate) | 0.73 / 0.43 | 0.83 / 0.58 | 1.28 / 1.14 | 0.27 / 0.28 | 0.46 / none | 1.25 / 1.10 |

The pan clips' key period is longer than their 600 frames (one key each); the drone clips have four
keys.

So raw fixed QP is not the ceiling of a stream that keeps a one-second buffer and the periodic-key pulse
guard (+0.25 dB). What a constraint-keeping stream could reach was not measured as a single number; the
two rows below bound it.

### The local slope of the key (AV1 medium, 600 f, 12 clips)

| Replay against CBR | J % | bodycam | crowd | drone | lowlight | pan | static | peak fullness, mean (max) | largest 1 s window, mean (max) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| as logged | | | | | | | | 0.93 (1.40) | 1.73 (2.32) |
| keys 4 QP finer | -6.2 | -0.3 | -9.0 | -12.5 | +1.1 | -2.9 | -13.7 | 1.39 (2.45) | 2.08 (2.94) |
| keys 4 QP coarser | +1.2 | +1.2 | +5.4 | +1.7 | +0.1 | -1.7 | +0.4 | 1.06 (2.16) | 1.84 (2.83) |

Key pulse at frame 300, static / drone: -1.14 / -2.16 dB as logged, +1.71 / +0.39 dB with finer keys.

Mean PSNR-Y on the static clips by frame range (7 points):

| Stream (rate / target) | key 0 | 1-14 | 15-59 | 60-149 | 150-299 | key 300 | 301-314 | 315-359 | 360-449 | 450-599 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CBR (0.97) | 32.18 | 31.27 | 32.10 | 32.36 | 33.11 | 31.35 | 30.92 | 31.30 | 31.85 | 32.19 |
| keys 4 QP finer (0.96) | 35.18 | 33.29 | 33.02 | 32.77 | 33.26 | 34.19 | 32.97 | 32.28 | 32.30 | 32.33 |
| fixed-QP key, flat level (0.94) | 38.77 | 35.74 | 34.50 | 33.72 | 33.32 | 38.77 | 35.90 | 34.52 | 33.73 | 33.22 |

A larger key, then cheap frames that inherit it, beats a buffer-sized key followed by refinement through
P frames, at a lower rate. The finer key is the one lever in this study with a large measured gain, and
it is a rate-rule question, not an allocation one: it needs keys of about 2 buffers on still and
near-still content (repaid inside the 5 s horizon as now). Not built: key sizing is other agents' area
(`docs/handoff/rate-control.md`, 2026-11-19 to 21), and whether a key second may reach 2.5x the target
is the owner's decision.

## 1b. Hindsight allocation between inter frames

BD-J % against the replayed CBR stream (replays) or CBR (live arms).

| Oracle | AV1 150 | HEVC 150 | AV1 faster 150 | AV1 600 | HEVC 600 |
|---|---:|---:|---:|---:|---:|
| reuse-weighted levels, strength 1 | +1.14 | +0.31 | +0.51 | -0.10 | -0.27 |
| reuse-weighted, opposite sign (control) | -0.40 | +0.19 | -0.19 | | |
| constant level (`flat`, the clip's mean, known from frame 0) | -3.86 | -6.40 | -2.76 | -0.97 | -4.13 |
| the same, only clips where every rung stays legal (n) | -4.0 (15) | -5.5 (19) | -3.2 (9) | +2.5 (3) | +7.5 (3) |
| centred 1 s mean (`smooth`) | -2.47 | -1.12 | -1.87 | | |
| trailing 1 s mean (causal, open loop) | -0.09 | -0.61 | +1.93 | | |
| first second after each key at the period's mean (`startup`) | | | | +0.46 | +1.16 |
| live: `rate.level_hint_weight=0.25` | -0.51 | -0.75 | +0.35 | +0.34 | |
| live: cadence off (`rate.cadence_period=1`) | +0.32 | +1.33 | +0.40 | +5.08 | +6.88 |
| live: `rate.cadence_period=8` | -0.07 | -0.73 | +0.11 | -0.68 | +0.32 |
| live: `rate.cadence_base_boost=6` | +1.70 | +0.79 | +0.41 | | |
| live: `rate.cadence_scale=0.5` | +3.66 | +2.74 | +3.40 | +9.49 | |
| best live arm per clip (hindsight) | -2.61 | -2.75 | -0.60 | -1.00 | -0.77 |

- **Reuse weighting.** Both signs are within 1.2 % of zero in every run, and the intended sign is the
  worse one on the 150-frame windows. Per class the largest effect is bodycam +2.8 / +1.6 / +1.1.
  Strength 2 was not run: strength 1 has no gain to scale.
- **Constant level.** On 150 frames it gains 2.8-6.4 %. On 600 frames it splits by class (AV1 / HEVC):
  drone -7.8 / -17.9, static +0.6 / -11.2, bodycam -2.6 / -6.1, against crowd +1.6 / +4.2 and
  pan +2.8 / +8.0: on moving content the level CBR follows is better than a constant one. The causal
  trailing mean recovers none of it (as on 2026-11-14), and flattening the start-up second loses.
- **Cadence.** Off costs 5-7 % on the product window (pan +14.7 / +20.7, static +8.8 / +7.3). Period 8
  is within 0.75 % everywhere, with pan -2.5 and crowd -1.7 (AV1, 600 f) against static +0.9 / +3.4.
  The per-clip best is a selection among arms whose per-clip SD is 1.3-2.2 % (`cadence_scale=0.5`:
  4.8-8.0 %), so it is not evidence of a gain.
- **`rate.cadence_scale=0.5` is not "half a cadence".** It loses 2.7-9.5 %, more than turning the cadence
  off does on 150 frames, with single clips from -19.6 to +20.8 %. Why is not known. Test: the rate log's
  `cadence_offset` column under 0.5 against the default on one clip.

## 1c. Rate conformity today

Class means over each class's points. Windows are one-second sliding windows in units of the target; a
keyless window holds no key frame. A cycle is four consecutive inter frames from a key.

**AV1 medium / HEVC medium, 600 f, keys every 10 s:**

| | bodycam | crowd | drone | lowlight | pan | static |
|---|---|---|---|---|---|---|
| stream / target | 0.991 / 0.990 | 0.993 / 0.990 | 0.975 / 0.969 | 0.988 / 0.987 | 0.988 / 0.986 | 0.969 / 0.972 |
| keyless window p95 | 1.21 / 1.26 | 1.06 / 1.08 | 1.57 / 1.53 | 1.07 / 1.07 | 1.06 / 1.15 | 1.36 / 1.47 |
| keyless window max | 1.40 / 1.44 | 1.11 / 1.16 | 1.86 / 1.92 | 1.14 / 1.16 | 1.12 / 1.35 | 1.66 / 1.75 |
| any window max | 1.74 / 1.74 | 1.70 / 1.63 | 2.12 / 2.13 | 1.29 / 1.30 | 1.43 / 1.45 | 2.09 / 2.00 |
| cycle bits / budget, p50 | 0.94 / 0.85 | 1.01 / 0.95 | 0.74 / 0.57 | 0.96 / 0.92 | 0.97 / 0.92 | 0.86 / 0.75 |
| cycle bits / budget, p95 | 1.94 / 1.99 | 1.61 / 1.57 | 2.69 / 2.68 | 1.33 / 1.33 | 1.82 / 1.62 | 2.67 / 3.26 |
| largest inter frame (frame budgets) | 8.8 / 9.6 | 6.2 / 12.4 | 17.1 / 17.8 | 4.3 / 5.1 | 12.7 / 29.1 | 24.5 / 30.4 |
| buffer overflow events per point | 0.3 / 0.3 | 0 / 0 | 3.6 / 2.4 | 0 / 0 | 0 / 0.4 | 2.0 / 0.7 |
| re-codes per 1000 frames | 0.8 / 3.9 | 3.6 / 0.5 | 34.6 / 44.8 | 0 / 0 | 0 / 0.8 | 23.1 / 9.8 |
| level move per frame (qindex / QP) | 5.9 / 1.2 | 4.1 / 1.0 | 5.1 / 1.6 | 2.4 / 0.4 | 2.1 / 0.4 | 4.6 / 1.2 |
| size model, log2(actual / predicted) SD: base frames | 0.88 / 1.06 | 0.48 / 0.54 | 1.42 / 1.96 | 0.27 / 0.30 | 0.53 / 0.81 | 1.40 / 1.81 |
| the same: other inter frames | 0.50 / 0.62 | 0.44 / 0.56 | 0.80 / 1.04 | 0.27 / 0.29 | 0.37 / 0.43 | 0.65 / 0.81 |
| base frames missed by 2x or more | 19 % / 23 % | 4 % / 8 % | 49 % / 68 % | 0 % / 1 % | 10 % / 13 % | 41 % / 54 % |

- **The ceiling holds on the whole stream** in every class (0.97-0.99).
- **Lowlight, crowd and pan conform**: keyless windows p95 at most 1.15x, almost no re-codes.
- **Drone and static do not**: keyless windows p95 1.36-1.57x and up to 1.92x, 10-45 re-codes per 1000
  frames, overflow events on most points, and half the cadence base frames mispredicted by 2x or more.
  The stream still runs under target, so these are bursts paid back, the still-scene cycle of
  `docs/analysis/rc-cost-decomposition-2026-11-14.md` ("What slow drift is on a still scene").
- On 150 frames (AV1 / HEVC) the pattern is the same: base-frame SD drone 1.92 / 2.48, static
  1.04 / 1.72, crowd 1.08 / 1.28, against videocall 0.49 / 0.50 and texture 0.44 / 0.47.

### Is sqrt(SAD) the right statistic?

An offline replica of the size model on the logged frames: `bits x step^1.1 ~ m x predictor`, `m` a
running geometric mean (0.25) per role (base, other inter), restarted at keys, first codings only. It is
not the controller (no restarts on scene changes, no refinement price, two roles instead of three), so
its absolute values are lower than the logged ones above; compare rows only. SD of
log2(actual / predicted), mean over classes:

| Predictor | AV1 150 base / inter | HEVC 150 | AV1 600 | HEVC 600 |
|---|---|---|---|---|
| temporal hint (today's) | 0.65 / 0.50 | 0.74 / 0.55 | 0.65 / 0.50 | 0.79 / 0.58 |
| no hint (the role's last sizes alone) | 0.56 / 0.48 | 0.67 / 0.52 | 0.55 / 0.46 | 0.68 / 0.50 |
| hint^0.5 | 0.58 / 0.47 | 0.69 / 0.51 | 0.56 / 0.45 | 0.70 / 0.51 |
| hint x (1 - compensable / 2) | 0.65 / 0.52 | | 0.65 / 0.52 | |
| (reference step / step)^0.5, no hint | 0.52 / 0.40 | 0.62 / 0.43 | 0.53 / 0.37 | 0.65 / 0.39 |
| hint^0.5 x (reference step / step)^0.5 | 0.53 / 0.38 | 0.62 / 0.41 | 0.53 / 0.35 | 0.65 / 0.39 |
| hint^0.5 x (reference step / step)^1 | 0.50 / 0.48 | 0.61 / 0.50 | 0.51 / 0.45 | 0.63 / 0.45 |

- **The hint at exponent 1 predicts worse than no hint at all** in every run, most on drone (base
  1.21 against 0.90, AV1 150) and static inter frames (0.70 against 0.51).
- **The reference frame's step is the better predictor**: with the hint at exponent 0.5 the error falls
  by 24-33 % on ordinary inter frames and 16-19 % on base frames.
- **Static and drone base frames stay near 0.8-1.1** under every predictor tried (HEVC 600: static
  1.23 -> 1.07, drone 1.26 -> 1.00). A frame-level statistic does not see those bursts coming.

## What this says about the candidates

| Candidate | Evidence | Verdict |
|---|---|---|
| frame budget scaled by causal reuse | the hindsight version is 0 to +1.1 %, the control no worse | rejected: no prize |
| adaptive cadence (period, depth from content) | cadence off +5..7 %, so the cadence matters; depth already follows inheritance; period 8 within 0.75 %; per-clip best inside noise | not pursued; period by motion class (pan and crowd -1.7..-2.5 on AV1 600 f) is the only lead, and earlier period-8 attempts failed on worst-1 %, flicker and long static |
| borrowing across the window on content change | start-up flattening +0.5 / +1.2; `rate.burst_loan` rejected 2026-11-15 | not pursued |
| the hint statistic | exponent 0.5 and a reference-step term cut the replica's error 16-33 %; built (section 4): windows a little better, J +0.9..+2.0 % at 600 f | built, measured, deleted |
| `rate.level_hint_weight` < 1 | live at 0.25: -0.51, -0.75, +0.35 (150 f), +0.34 (600 f); not on in any preset | not adoptable as it is (2026-11-14: L2 +0.30) |
| model per role by content class | static and drone base frames miss under every predictor | not pursued as a model; it is the burst cycle |
| key size on still content (not in the brief) | section 2: `rate.key_hold_still` 2 buffers, J -1.7..-2.9 % for 130-180 ms more delay | the large lever; the owner picks the point |

## 2. The key-size trade-off (for the owner's rate rule)

**Setup.** AV1 medium, HEVC medium, AV1 faster; the 12 long clips, 600 f, keys every 10 s; and a
single-key run (900 f, 10 clips: static, crowd, bodycam, pan, lowlight, two each), AV1 and HEVC medium.
Means over all points unless a class is named. **Delay** is the peak of a bucket drained at the target
rate, in ms: what a constant-rate channel at the target adds to the worst frame (excess bits / channel
rate). **Key second** is the bits of the second that starts at the key, over the target.

**What bounds a key today** (`RateController::key_allowance`). A periodic key may take the buffer plus one
frame at the peak rate: 1.07 buffers at 30 fps. Measured periodic keys on static clips: 1.07 (AV1),
0.99 (HEVC); on drone 1.11, 1.01. The 1.4-buffer re-code ceiling is not the binding limit.

**The parameter exists**: `rate.key_hold_still` = N buffers (2026-11-14, off in every preset, never
pool-screened). A periodic key of a still scene (running unchanged share at least 0.5) is planned at the
quality its inter frames hold, with room of N buffers; the frames after it repay it. It is the knob for
periodic keys, 0 = today. The start-up key has no such parameter.

### Periodic keys, live (closed loop: the frames after the key repay it)

`rate.key_hold_still` = N. Cells: AV1 medium / HEVC medium / AV1 faster.

| N | J % | key second (x target) | delay after the key (ms) | stream / target |
|---|---|---|---|---|
| off | | 1.58 / 1.50 / 1.61 | 853 / 902 / 881 | 0.98 / 0.98 / 0.98 |
| 1.4 | -0.8 / -1.3 / -0.45 | 1.64 / 1.56 / 1.65 | 896 / 928 / 918 | 0.98 / 0.98 / 0.98 |
| 2 | -2.2 / -2.9 / -1.7 | 1.75 / 1.67 / 1.73 | 1030 / 1035 / 1034 | 0.97 / 0.97 / 0.98 |
| 3 | -3.6 / -4.3 / -3.1 | 1.85 / 1.76 / 1.84 | 1117 / 1099 / 1147 | 0.96 / 0.96 / 0.97 |
| 4 | -4.5 / -4.8 / -3.8 | 1.90 / 1.77 / 1.88 | 1161 / 1090 / 1175 | 0.94 / 0.93 / 0.95 |

So N = 2 buys 1.7-2.9 % J for 130-180 ms more delay and a key second near 1.7x; N = 3 buys 3.1-4.3 % for
200-270 ms. The stream stays under the target at every N. The worst frame (minimum PSNR-Y) is unchanged
(AV1 30.86 -> 30.85 dB at N = 2 and 3; HEVC 30.43 -> 30.52 -> 30.50).

Static clips, AV1 medium / HEVC medium:

| N | J % | key (buffers) | key second | delay (ms) | key against the 12 frames before (dB) | 30 frames after against 30 before (dB) | bucket back to its pre-key level (s) | stream / target |
|---|---|---|---|---|---|---|---|---|
| off | | 1.07 / 0.99 | 1.89 / 1.75 | 1197 / 1241 | -1.14 / -2.12 | -1.80 / -2.52 | 5.3 / 4.0 | 0.97 / 0.97 |
| 2 | -3.9 / -4.9 | 1.59 / 1.53 | 2.18 / 2.05 | 1618 / 1576 | +0.11 / -0.22 | -1.01 / -1.37 | 7.7 / 6.2 | 0.96 / 0.96 |
| 3 | -7.7 / -9.1 | 1.74 / 1.69 | 2.37 / 2.24 | 1783 / 1718 | +0.57 / +0.69 | -0.46 / -0.36 | 8.6 / 8.0 | 0.93 / 0.92 |

Other classes (AV1 / HEVC, J at N = 2 and N = 3):

- **Drone:** -8.1 / -9.0 and -10.1 / -12.2; delay 1257 -> 1629 -> 1675 ms (AV1).
- **Crowd:** -0.3 / -2.9 and -2.9 / -4.0, but the keys get smaller (AV1 0.55 -> 0.47 -> 0.45 buffers,
  31.15 -> 30.58 -> 30.35 dB) and worst-1 % PSNR-YUV falls 32.24 -> 32.00 -> 31.85 dB: the held key is
  planned without today's still boost. Not a clean gain there.
- **Bodycam, lowlight:** within 0.4 %.
- **Pan:** -0.8 / -0.3 at N = 2, although these clips code no periodic key in 600 f; stream 0.99 ->
  0.97, worst frame -0.2 dB. Not established why. `empty_key_allowance` (what `key_reserve_room` keeps
  free before the next scheduled key) grows with N, which would squeeze the frames before a key that is
  scheduled just past the clip's end; test: one pan clip's rate log with and without the parameter.

### Both kinds of key, by replay (open loop)

The key moved finer, the inter frames as logged: nothing repays, so the stream's rate rises (by at most
1 %; the BD is at equal rate). The slope of the key's size alone.

**Periodic keys** 2 / 4 / 6 QP units finer, 600 f:

| | J % (2 / 4 / 6) | key (buffers) | key second | delay (ms) | key against the frames before (dB) |
|---|---|---|---|---|---|
| AV1 medium | -1.3 / -3.7 / -6.2 | 0.71 -> 0.92 / 1.18 / 1.50 | 1.58 -> 1.67 / 1.90 / 2.20 | 853 -> 986 / 1249 / 1584 | +1.1 -> +2.3 / +3.6 / +5.0 |
| HEVC medium | -1.0 / -3.1 / -5.6 | 0.64 -> 0.84 / 1.08 / 1.38 | 1.50 -> 1.56 / 1.74 / 2.02 | 902 -> 1006 / 1214 / 1513 | +0.65 -> +1.9 / +3.3 / +4.7 |
| AV1 faster | -1.3 / -3.5 / -5.4 | 0.72 -> 0.93 / 1.19 / 1.51 | 1.61 -> 1.69 / 1.91 / 2.21 | 881 -> 1007 / 1279 / 1643 | +1.5 -> +2.8 / +4.1 / +5.4 |
| AV1 medium, static | -1.7 / -7.3 / -12.9 | 1.07 -> 1.37 / 1.79 / 2.26 | 1.89 -> 1.95 / 2.29 / 2.74 | 1197 -> 1358 / 1764 / 2237 | -1.14 -> +0.18 / +1.71 / +3.23 |

**The start-up key** (the first frame) 2 / 4 / 6 QP units finer:

| | J %, 600 f with 10 s keys | J %, 900 f, one key | first key (buffers) | first second | start-up delay (ms) | bucket under a quarter second again (s) |
|---|---|---|---|---|---|---|
| AV1 medium | -1.1 / -2.5 / -3.6 | -1.0 / -1.6 / -2.3 | 0.79 -> 1.02 / 1.30 / 1.63 | 1.59 -> 1.74 / 1.99 / 2.31 | 834 -> 1021 / 1300 / 1632 | 2.2 -> 2.7 / 4.7 / 6.3 |
| HEVC medium | -0.65 / -1.8 / -3.1 | -0.7 / -1.4 / -2.4 | 0.65 -> 0.84 / 1.07 / 1.36 | 1.45 -> 1.54 / 1.73 / 1.98 | 829 -> 929 / 1119 / 1376 | 1.5 -> 1.9 / 2.8 / 4.6 |
| AV1 faster | -1.2 / -2.6 / -3.4 | | 0.82 -> 1.05 / 1.33 / 1.67 | 1.65 -> 1.80 / 2.04 / 2.36 | 870 -> 1063 / 1337 / 1674 | 2.3 -> 2.8 / 4.7 / 6.2 |
| AV1 medium, static | -2.3 / -6.4 / -9.5 | -2.9 / -5.3 / -8.6 | 1.21 -> 1.57 / 2.01 / 2.50 | 1.92 -> 2.03 / 2.38 / 2.84 | 1264 -> 1535 / 1972 / 2464 | 3.6 -> 3.7 / 4.7 / 6.1 |

(The first-key columns of the first three rows are the 600-frame run's; the 900-frame clips' first keys
are smaller, AV1 0.65 -> 0.82 / 1.03 / 1.27 buffers, delay 751 -> 875 / 1067 / 1305 ms.)

A finer start-up key is spent within about ten seconds. AV1 medium, static, one key, mean PSNR-Y of the
stream with the key 4 QP finer less CBR's: +3.00 dB at the key, +1.55 dB over frames 1-29, +0.67 over
30-89, +0.35 over 90-149, +0.16 over 150-299, 0 after. Its delay is paid once, at the start of the
stream, where a viewer waits or a sender bursts; a periodic key's is paid every key period.

### The alternative with no rule change: the same quality spread over the frames after the key

- **It is what the stream does today.** On static clips the key second is 1.89x (AV1) / 1.75x (HEVC)
  with a key of about one buffer: the frames after the key spend about 0.8 buffer more in that second.
- **How far short it falls.** The 30 frames after a static key are 1.80 dB (AV1) / 2.52 dB (HEVC) below
  the 30 before it; the bucket takes 5.3 / 4.0 s to drain; AV1's static clips return to 32.2 dB by the end
  of the 10 s period against 33.1 dB before the key. With N = 3 the same 30 frames are 0.46 / 0.36 dB
  below.
- **Why it does not avoid the delay.** The delay is the bucket's peak, and the peak counts the bits sent
  early whether they are in one frame or in thirty. A stream that refines faster after a small key raises
  the same peak. An arm at equal bucket peak with a different split between key and refinement was not
  measured.

### What is proposed

- **Periodic keys:** `rate.key_hold_still`, default 0 as today; the owner picks N from the live table.
  Before any preset takes it: the crowd case (the held key loses the still boost), the pan case, and
  `nceval.quick` / `nceval.train` on the day's draw. Its 2026-11-14 measurements (six still points) are in
  `docs/analysis/persistent-region-coding-2026-11-14.md` section 5.
- **Start-up key:** not built. It needs the first key's room lifted in `budget_at` and `strict_room`
  (`rate.first_key_buffers`, 0 = today) and a boost that asks for it; the replay table is the evidence.

## 3. The constant-level oracle on drone and static is not the post-key dip

The `flat` replay's gain there (AV1 / HEVC, 600 f: drone -7.8 / -17.9 %, static +0.6 / -11.2 %) comes
from a stream that ran at 0.70-0.82 of the target (HEVC static 0.695, AV1 0.784) and lost 1-2 dB late in
each period, not from the seconds after a key: over frames 301-314 the flat and CBR streams are within
0.03 dB (HEVC static 30.78 against 30.75) at those rates. Its BD gain says that on still scenes the last
20-30 % of CBR's bits buy less than the first. The level's SD inside a key period is 3.8 QP on HEVC static
and drone against 1.2 on crowd.

What a causal controller would need is the level the scene sustains, before it has found it by descending
onto a refinement burst and climbing back: the still-scene cycle of 2026-11-15 (three causal levers
rejected) and `rate.still_level_damping` (2026-11-18/19; pinned off on HEVC medium for the pulse guard).
It does not belong to the key rule above.

## 4. The size model's content terms (built, measured, deleted)

Two parameters, `rate.hint_exponent` (1 = today) and `rate.reference_step_exponent` (0 = off), in a module
`size_terms.rs`; the code was deleted after this measurement (coordinator, 2026-11-21: a failed lever) and
is kept on branch `frame-rc-size-terms`.
The inter size model became `bits = m x hint^a x (previous coded frame's step / step)^g / step^1.1`; the level is solved for
a cycle that runs as planned. `preset_identity` was 387/387 against the same tree without it.

Live arms against CBR (the oracle tool's `--arm`, each arm's own rate logs). 150 f: 24 clips, one key.
600 f: 12 clips, keys every 10 s. Cells: AV1 medium / HEVC medium.

| Arm (a, g) | J %, 150 f | J %, 600 f | keyless 1 s window p95, 600 f | max, 600 f | re-codes per 1000, 600 f | overflow events per point, 600 f | base-frame model SD, 600 f | level move per frame, 600 f |
|---|---|---|---|---|---|---|---|---|
| today (1, 0) | | | 1.22 / 1.26 | 1.38 / 1.46 | 10.3 / 10.0 | 0.99 / 0.63 | 0.83 / 1.08 | 4.0 / 0.97 |
| (0.5, 0) | -0.60 / -0.51 | +0.93 / +1.59 | 1.15 / 1.20 | 1.30 / 1.38 | 10.2 / 8.0 | 1.09 / 0.55 | 0.64 / 0.85 | 3.2 / 0.72 |
| (0.5, 0.5) | -0.41 / +0.28 | +0.88 / +1.99 | 1.16 / 1.18 | 1.31 / 1.37 | 10.0 / 7.2 | 0.97 / 0.49 | 0.60 / 0.77 | 3.2 / 0.71 |
| (0, 0.5) | +0.46 / +0.88 | +1.42 / +2.69 | 1.13 / 1.17 | 1.28 / 1.33 | 9.7 / 5.8 | 0.76 / 0.36 | 0.56 / 0.70 | 1.9 / 0.39 |
| (1, 0.5), 150 f only | -0.12 / | | | | | | | |
| (0.5, 1), 150 f only | -0.10 / | | | | | | | |

By class at (0.5, 0.5), 600 f, AV1 / HEVC:

| | bodycam | crowd | drone | lowlight | pan | static |
|---|---|---|---|---|---|---|
| J % | -1.4 / -0.6 | +1.2 / +2.9 | +0.8 / +1.2 | +0.1 / -0.5 | +0.6 / +2.8 | +4.0 / +6.1 |
| keyless window p95, today -> arm | 1.21 -> 1.12 / 1.26 -> 1.13 | 1.06 -> 1.04 / 1.08 -> 1.05 | 1.57 -> 1.48 / 1.53 -> 1.54 | 1.07 -> 1.07 / 1.07 -> 1.07 | 1.06 -> 1.02 / 1.15 -> 1.06 | 1.36 -> 1.19 / 1.47 -> 1.24 |
| base frames missed by 2x, today -> arm | 19 -> 8 % / 23 -> 11 % | 4 -> 1 % / 8 -> 2 % | 49 -> 38 % / 68 -> 56 % | 0 / 1 -> 0 % | 10 -> 6 % / 13 -> 8 % | 41 -> 23 % / 54 -> 27 % |

- **The model predicts better, as the offline fit said**: base-frame error SD down 21-28 %, base frames
  missed by 2x from 21-28 % to 13-17 %, the level moves 20-27 % less per frame.
- **Windows improve a little**: keyless p95 1.22 -> 1.16 (AV1), 1.26 -> 1.18 (HEVC); static 1.36 -> 1.19 and
  1.47 -> 1.24. Drone does not move, and AV1's re-codes do not.
- **It costs quality on the product window**: J +0.9 % (AV1) and +1.6..+2.0 % (HEVC) at 600 f, +4..+6 % on
  static, against -0.4..-0.6 % on 150 f. The per-clip SD of these arms is 1.9-2.8 %, so the 12-clip means
  are not tight, but the sign is the same in both codecs and all three arms.
- **Verdict: not queued for `nceval.train`.** It is a conformity gain bought with J, not one at equal J:
  the bursts it trims on static clips are refinement the picture keeps. The stiller level is what loses
  there; whether the reference term alone, on a level solved per frame instead of per cycle, keeps the
  window gain without it was not tried.

## 5. The buffer our CBR charts run in, against the references' (2026-11-21)

Owner, on section 2: a key larger than the buffer is not better CBR, it is a looser buffer.
`rate.key_hold_still` is therefore held as a user's buffer/latency option, never a preset. The same
question then applies to what ships.

**What each encoder is given** (`eval/nceval/cbr.py`, `BUFFER_MS` 1000 for all):

| Encoder | Rate-control options |
|---|---|
| nanocodec | `--bitrate T --buffer-ms 1000`. Inside: peak 2 x target by default; `rate.recode_overflow` 1.4 (a frame is re-coded only past 1.4 buffers); `rate.window_seconds` 0 (no short-window rule); a key borrows and is repaid over 5 s (first key budget 8 x 2.5 inter budgets, AV2 8 x 3); key seconds exempt (2026-11-13) |
| x264, x265 | `--bitrate T --vbv-maxrate T --vbv-bufsize T x 1 s`, `--tune zerolatency`, no B frames, no lookahead |
| SVT-AV1 | `--rtc 1 --rc 2 --tbr T --buf-sz 1000 --buf-initial-sz 600 --buf-optimal-sz 600` |
| libaom, libvpx | `--rt --end-usage=cbr --buf-sz=1000 --buf-initial-sz=600 --buf-optimal-sz=600` |
| openh264 | `--bitrate T` (RC_BITRATE_MODE), frame skip off; no buffer option |
| NVENC | `-b:v T -maxrate T -bufsize T x 1 s` |

**What the streams do.** The chart point cache (`~/nanocodec-data/work/realuse`, each point's frame sizes
and the harness's own leaky-bucket replay; no new encodes), the newest stored point per family, medium
presets of ours. Only (clip, target) points that every listed encoder coded within 5 % over target: 670 of
1378 common points at 150 f (one key), 119 of 198 at 600 f (keys every 10 s). Whether these are exactly
the points a published chart read was not checked.

150 f:

| Encoder | stream / target (median) | peak buffer (x 1 s): median / p95 / max | points over 1 buffer | first key (buffers, median) | key second (x target): median / p95 |
|---|---|---|---|---|---|
| nanocodec H.264 | 0.986 | 0.43 / 1.27 / 1.40 | 13 % | 0.33 | 1.28 / 1.88 |
| nanocodec HEVC | 0.980 | 0.58 / 1.29 / 1.40 | 16 % | 0.38 | 1.35 / 1.83 |
| nanocodec AV1 | 0.986 | 0.58 / 1.30 / 1.40 | 15 % | 0.37 | 1.34 / 2.02 |
| nanocodec VP9 | 0.986 | 0.55 / 1.32 / 1.40 | 16 % | 0.37 | 1.33 / 2.01 |
| nanocodec VVC | 0.977 | 0.53 / 1.23 / 1.39 | 10 % | 0.37 | 1.32 / 1.88 |
| nanocodec AV2 | 0.986 | 0.61 / 1.22 / 1.40 | 14 % | 0.45 | 1.40 / 1.86 |
| x264 medium | 0.923 | 0.16 / 0.54 / 0.74 | 0 % | 0.12 | 0.91 / 1.07 |
| x264 veryfast | 0.921 | 0.17 / 0.54 / 0.74 | 0 % | 0.12 | 0.89 / 1.06 |
| x265 medium | 0.979 | 0.24 / 0.61 / 0.85 | 0 % | 0.15 | 1.04 / 1.38 |
| SVT-AV1 M12 | 1.002 | 0.26 / 0.62 / 1.17 | 0 % | 0.10 | 1.00 / 1.23 |
| SVT-AV1 M8 | 0.997 | 0.19 / 0.52 / 1.03 | 0 % | 0.09 | 0.96 / 1.12 |
| libaom cpu 8 | 1.013 | 0.33 / 0.65 / 1.34 | 0 % | 0.24 | 1.06 / 1.32 |
| libvpx VP9 cpu 9 | 1.009 | 0.30 / 0.57 / 1.83 | 1 % | 0.24 | 1.05 / 1.32 |
| openh264 | 1.001 | 0.40 / 1.47 / 3.14 | 13 % | 0.38 | 1.02 / 2.06 |

600 f: nanocodec (six codecs) peak buffer p95 1.08-1.29, max 1.20-1.40, 12-18 % of points over one
buffer, first key 0.41-0.53 buffers, key second median 1.36-1.47; x264 medium 0.79 / 0.80, 0 %, 0.13,
1.09; x265 medium 0.94 / 1.16, 3 %, 0.17, 1.14; SVT-AV1 M12 0.81 / 0.99, 0 %, 0.12, 1.03; libaom
0.81 / 1.44, 3 %, 0.30, 1.14; libvpx 1.03 / 1.70, 5 %, 0.29, 1.12.

- **Ours is looser than x264, x265, SVT-AV1, libaom and libvpx**: a first key 3-4 times x264's and
  SVT's, a key second of 1.3-1.5x target against 0.9-1.1x, and the one-second buffer passed on 10-18 %
  of points (to 1.40 by design) where theirs is passed on 0-5 %.
- **openh264 is comparable or looser.** Nothing of ours is tighter than any reference.
- **On the excluded points** at least one encoder missed the target by over 5 %; over all 1378 points
  the references' peaks include points of 16-130 buffers (libvpx, libaom, openh264, and single points of
  x264 and SVT): on low rungs several references do not hold the rate at all.

## 6. What the looser buffer is worth in the comparison

One run, the release build 4fc9a1de4, parameters only. A class-spread seeded sample of the held-out
chart's sets: 24 clips of `independent` at 150 f (one key), 12 clips of `independent_long` at 600 f with
10 s keys (9 with cached reference points); the 4-rung ladder. Our arms were coded and CPU-scored on rented
boxes; x264 medium and SVT-AV1 M12 are the cached chart points of the same clips (GPU-scored: a mix the
tools normally refuse, of about 1e-6 in MS-SSIM and VMAF).

- **shipped**;
- **mid**: `rate.key_cap_buffer` 0.3, `rate.recode_overflow` 1.0;
- **tight**: `rate.key_cap_buffer` 0.15, `rate.recode_overflow` 1.0, `rate.window_seconds` 1.

BD-J % (negative: ours cheaper); shipped / mid / tight:

| | AV1 medium | HEVC medium | H.264 medium |
|---|---|---|---|
| 150 f, against SVT-AV1 M12 (n 24) | -31.4 / -28.4 / -24.8 | -32.5 / -29.2 / -25.4 | +0.7 / +5.1 / +9.9 |
| 150 f, against x264 medium (n 15-24, differs by arm) | -40.5 / -38.4 / -37.8 | -41.5 / -40.9 / -39.6 | -25.5 / -23.3 / -22.0 |
| 150 f, arm against our shipped (n 24) | 0 / +4.8 / +10.4 | 0 / +5.1 / +10.6 | 0 / +5.5 / +9.9 |
| 600 f, against SVT-AV1 M12 (n 9) | -20.2 / -19.2 / -16.4 | -23.1 / -22.0 / -18.2 | +23.4 / +25.2 / +31.6 |
| 600 f, against x264 medium (n 7-9) | -42.3 / -41.7 / -40.0 | -43.1 / -42.5 / -40.9 | -20.6 / -19.3 / -15.5 |
| 600 f, arm against our shipped (n 12) | 0 / +2.8 / +11.3 | 0 / +3.4 / +12.6 | 0 / +3.7 / +12.7 |

Per metric at 150 f against SVT-AV1 M12 (PSNR-YUV / VMAF v1 / MS-SSIM), shipped then tight: AV1
-35.3 / -32.4 / -24.9 then -29.7 / -27.3 / -16.5; HEVC -31.7 / -34.3 / -29.3 then -25.4 / -29.1 / -21.2;
H.264 -6.8 / -5.9 / +5.1 then +1.6 / +1.8 / +14.4.

The arms' streams, all rungs (150 f; AV1 / HEVC / H.264):

| | peak buffer p95 | max | points over 1 buffer | first key (buffers) | key second, median | stream / target |
|---|---|---|---|---|---|---|
| shipped | 1.28 / 1.36 / 1.23 | 1.38 / 1.40 / 1.37 | 22 / 27 / 12 % | 0.43 / 0.48 / 0.39 | 1.42 / 1.41 / 1.30 | 0.986 / 0.983 / 0.992 |
| mid | 0.95 / 0.93 / 0.92 | 0.98 / 0.99 / 0.98 | 0 % | 0.34 / 0.26 / 0.30 | 1.29 / 1.22 / 1.24 | 0.986 / 0.983 / 0.992 |
| tight | 0.78 / 0.71 / 0.79 | 0.94 / 0.82 / 0.97 | 0 % | 0.17 / 0.13 / 0.16 | 1.10 / 1.05 / 1.09 | 0.969 / 0.950 / 0.965 |

The tight arm's first key and key second match SVT's and x264's (section 5); its buffer peak is still a
little above theirs at p95 (0.71-0.79 against 0.54-0.62).

What the tight arm does to the picture (mean PSNR-Y, 150 f, AV1 / HEVC / H.264, shipped -> tight): first
key 34.3 -> 29.3, 34.5 -> 28.2, 33.1 -> 28.3 dB; frames 1-30 31.7 -> 30.3, 31.6 -> 30.0, 30.5 -> 29.0;
frames 30-150 31.59 -> 31.46, 31.63 -> 31.42, 30.45 -> 30.19; worst frame 29.3 -> 27.5, 29.3 -> 27.4,
27.6 -> 26.3 (it becomes the key). No collapse in these means; individual clips were not inspected. For
scale, SVT-AV1 M12's first key on these clips is 27.9 dB and x264 medium's 25.2.

- **Held inside one buffer like the references, our CBR costs about 10 % more bits at 150 f and 11-13 %
  at 600 f**, the same in the three codecs. The lead over SVT-AV1 M12 falls by 4-7 points and over x264
  medium by 2-5.
- **Never passing one buffer, with a first key of about 0.3 buffer (mid), costs 3-5 %.**
- A sample of 24 and 12 clips, one run, no interval. The full held-out figure with a reference-like
  buffer was not measured.
- Against x264 the number of clips with a reliable BD changes with the arm, so those rows are not over
  identical clips; the rows against SVT-AV1 M12 and against our own shipped arm are.

**Mid rows in full** (BD % against SVT-AV1 M12, J then PSNR-YUV / VMAF v1 / MS-SSIM). 150 f: AV1 -28.4
(-32.6 / -30.3 / -21.1), HEVC -29.2 (-28.3 / -32.1 / -25.7), H.264 +5.1 (-2.9 / -2.5 / +10.1). Against x264
medium, 150 f: AV1 -38.4 (-43.0 / -44.5 / -26.7), HEVC -40.9 (-40.8 / -45.8 / -31.4), H.264 -23.3
(-26.2 / -30.8 / -9.1). Shipped and tight against x264 medium, 150 f: AV1 -40.5 (-44.6 / -45.2 / -30.0) and
-37.8 (-41.8 / -43.7 / -25.0); HEVC -41.5 (-41.6 / -46.7 / -32.9) and -39.6 (-39.5 / -45.3 / -28.7); H.264
-25.5 (-28.6 / -32.3 / -11.6) and -22.0 (-25.0 / -28.6 / -6.8). Streams at 600 f (AV1): peak buffer
p95 / max shipped 1.37 / 1.39, mid 0.99 / 1.00, tight 0.73 / 0.80; first key 0.51 / 0.30 / 0.16 buffers.

## 7. Can key-frame CPU buy back the tight buffer's cost?

Owner's idea for the buffer fix: with the key held to about 0.15 buffer, spend more CPU on the key. Same
build, clips and windows as section 6; every arm sits on the **tight** arm, and J is against tight. Times
are wall ms per frame from the rate logs on a shared box: ratios between arms of one run, not timings.

**Key tools on tight** (150 f / 600 f):

| Codec | What the arm adds on key frames | J % | key frame time | whole encode |
|---|---|---|---|---|
| AV1 | `av1.keyframe_intra_rd_candidates` 8, `av1.keyframe_angle_deltas` 2, `quant.trellis_key` 1 | +0.2 / 0.0 | 32 -> 40 ms | x1.021 / x1.012 |
| H.264 | the slowest rung's intra tools through `key.`: 9 RD candidates, all 8x8 modes, no 4x4 / 8x8 gates, transform-size RD, chroma final stage, residual-RD λ | +0.5 / +0.2 | 16 -> 39 ms | x1.024 / x1.004 |
| HEVC | the SAO settings (the only ones `key.` accepts) | +0.4 / +0.2 | unchanged | x0.97 / x1.003 |

`key.` refuses `hevc.intra_rd_candidates`, `hevc.intra_nxn_min_cost` and `hevc.intra_tu_split_min_cost`
("not re-read per frame"); `quant.*` and `planner.*` take no per-role value in any codec.

**Everything the slowest rung has, on a key frame alone** (1 frame, 4 fixed quantizers, 24 clips; the
slowest rung against medium, BD-rate): H.264 PSNR-YUV -6.9 %, VMAF -5.2 %, MS-SSIM -8.7 %; HEVC
-2.5 / -2.6 / -3.9 %; AV1 -2.9 / -7.6 / -2.4 %. A key 2.5-9 % cheaper is the ceiling of the present intra
tools. By the sweep below (47 % more key bits is J -2.0 to -3.2 %), 5 % of the key is about 0.2-0.35 % J,
which agrees with the direct arms. The 2026-11 finding that a key-frame RD search is worth at most 0.5 %
holds under the tight key too. A key λ tuned for inheritance, palette, intra block copy and a CfL search
are not parameters: not tested.

**Effort on the frames after the key.** No role targets them (`frame_role.rs` has `key.` and `nonbase.`).
The bracket is the slowest rung's whole parameter set on every frame, on tight (150 f / 600 f): AV1 J
-9.8 / -10.3 % at x2.5 / x2.3 the encode time, HEVC -12.7 / -14.2 % at x3.6 / x3.2, H.264 -2.2 / -3.1 %
at x1.5. That is the rung's gain on all frames; the rung was not coded under the shipped buffer in this
run, so whether it is larger under tight is not known. Inside it the first key is no better (AV1 29.33 ->
29.12 dB, HEVC 28.16 -> 28.38) and frames 1-30 gain 0.18 (AV1) to 0.42 dB (HEVC), against a dip of about
1.4-1.6 dB below the shipped stream.

**The split between the key and the frames after it** (re-code ceiling 1.0 and the 1 s window rule in
every arm; `rate.key_cap_buffer` swept):

| key cap (buffers) | J % vs tight, 150 f: AV1 / HEVC / H.264 | 600 f | first key (buffers) | first key PSNR-Y, AV1 150 f | frames 1-30 | peak buffer p95, the six runs |
|---|---|---|---|---|---|---|
| 0.10 | +3.0 / +2.0 / +2.4 | +1.6 / +1.5 / +3.1 | 0.08-0.11 | 27.7 | 30.03 | 0.57-0.66 |
| 0.15 (tight) | 0 | 0 | 0.12-0.17 | 29.3 | 30.30 | 0.66-0.79 |
| 0.22 | -2.3 / -3.1 / -3.2 | -2.0 / -2.1 / -2.3 | 0.17-0.25 | 31.0 | 30.63 | 0.71-0.91 |
| 0.30 | -3.9 / -4.4 / -3.4 | -3.4 / -3.8 / -4.3 | 0.23-0.34 | 32.4 | 30.93 | 0.83-0.93 |

The larger key wins in all six runs, and the frames after it are better too. At 0.30 the stream still
never passes one buffer.

**What "tight" is made of.** It imposes three things: the buffer is never passed (re-code ceiling 1.0); the
key is as small as the references' (cap 0.15); every one-second window is within 3 % of the target. Only
the first is what x264's VBV and SVT's buffer do: their own worst one-second windows are 1.33-1.41x target
at p95 (section 5's points), and their small first keys are their choice.

| Arm, against our shipped stream (J %) | 150 f: AV1 / HEVC / H.264 | 600 f |
|---|---|---|
| mid: the buffer rule alone (ceiling 1.0, key cap 0.3) | +4.8 / +5.1 / +5.5 | +2.8 / +3.4 / +3.7 |
| key cap 0.30 with the window rule | +5.5 / +5.1 / +5.8 | +7.0 / +8.1 / +7.7 |
| tight: all three | +10.4 / +10.6 / +9.9 | +11.3 / +12.6 / +12.7 |

So holding one buffer costs about 3-5.5 %; copying the references' key size and adding the window rule
takes it to 10-13 %. Key-frame CPU recovers none of it; the key's share of the buffer recovers 3.4-4.4
points of it.

## 8. The still-scene burst cycle: no loss at equal rate

**Question.** On static and drone clips the constant-level replay gains 10-18 % BD-J (HEVC) while running
at 0.70-0.82 of the target (section 3). Is that an inefficiency of CBR inside its buffer, which a causal
controller that settles at a sustainable level could recover at the same or a lower rate and delay?

**Setup.** 11 clips of the diagnostic draw (4 static, 4 crowd as the control, 3 drone clips of one source),
600 f, keys every 10 s, the 4-rung ladder, HEVC medium and AV1 medium at the landed head.

**What the cycle is** (the 2026-11-21 logs of section 1, 600 f; HEVC / AV1):

| | static | drone | crowd | bodycam |
|---|---|---|---|---|
| inter frames over 4 budgets, per second | 0.83 / 0.88 | 0.64 / 0.60 | 0.57 / 0.49 | 0.49 / 0.43 |
| their share of the inter bits | 38 % / 30 % | 45 % / 36 % | 13 % / 9 % | 20 % / 17 % |
| that are cadence base frames | 98 % / 99 % | 94 % / 96 % | 100 % / 86 % | 100 % / 83 % |
| mean size (frame budgets) | 13.8 / 10.4 | 8.8 / 8.6 | 6.0 / 5.0 | 5.6 / 5.9 |
| re-coded | 12 % / 5 % | 41 % / 23 % | 0 | 2 % / 1 % |
| inter frames under a quarter budget | 27 % / 13 % | 37 % / 14 % | 5 % / 1 % | 5 % / 3 % |
| PSNR-Y the burst adds / left after 8 frames / after 30 | +1.37, +0.75, +0.49 / +1.00, +0.71, +0.47 | +1.49, +0.82, +0.32 / +1.28, +0.96, +0.52 | +2.05, +1.62, +0.45 / +1.57, +1.37, +0.39 | |
| level over the 8 frames before / after (QP; qindex) | -3.2, +5.7 / -14.7, +19.1 | -4.1, +4.9 / -16.4, +16.4 | -1.4, +1.2 / -8.8, +4.3 | |

Frames that cost almost nothing leave credit, the level descends, a cadence base frame crosses the quality
the picture holds and refines all of it at once, the debt lifts the level, and two thirds of what the
burst added is gone a second later.

**Oracles** (replay, BD-J % against CBR; all / crowd / drone / static):

| Replay | HEVC | AV1 | stream / target, static (HEVC / AV1) |
|---|---|---|---|
| constant level (`flat`) | -9.5 / +2.2 / -18.5 / -12.1 | -3.7 / +0.6 / -9.0 / -2.8 | 0.70 / 0.83 |
| held after 2 s (`plateau2`, causal in form) | -13.5 / -0.9 / -19.7 / -19.8 | -4.8 / +1.3 / -9.3 / -6.4 | 0.58 / 0.78 |
| held after 4 s (`plateau4`) | -10.8 / -0.2 / -18.2 / -14.0 | -6.5 / -0.3 / -10.4 / -8.7 | 0.73 / 0.80 |
| no descent below the period's median level (`floor`) | -9.1 / +0.9 / -17.0 / -11.1 | -4.3 / -1.1 / -7.9 / -3.8 | 0.66 / 0.79 |
| no climb above it (`ceil`) | -0.9 / +0.7 / -2.5 / -1.0 | +0.5 / +1.9 / -1.2 / +0.7 | 1.06 / 1.04 |
| `plateau2` held 2 QP units finer | -7.6 / +2.5 / -13.0 / -12.3 | +1.3 / +6.2 / -3.8 / +1.5 | |
| held 4 finer | -1.4 / +6.5 / -6.2 / -4.4 | +6.2 / +11.4 / +1.4 / +5.9 | |
| held 6 finer | +5.2 / +10.0 / +2.7 / +2.8 | +14.4 / +18.5 / +10.6 / +14.0 | 1.27 / 1.84 |

- **The whole gain is in the descents** (`floor` takes it, `ceil` takes none): the bits spent below the
  level a period sustains.
- **Every gaining replay spends less** (0.58-0.83 of the target on static) and has a lower PSNR at that
  target (HEVC static: 37.05 dB at 0.97 of target as coded, 35.92 at 0.58 held after 2 s, 36.45 at 0.73
  held after 4 s).
- **The gain goes as the held level is made finer**, and is a loss by 6 QP units.

**At CBR's own rate.** Per (clip, target), the held-level family (held after 2 s; 0, 2, 4, 6 QP units
finer) interpolated at the rate CBR coded, less CBR:

| | points bracketed | PSNR-YUV (dB) | VMAF v1 | MS-SSIM | worst frame (dB) | peak buffer (CBR's) | worst 1 s window (CBR's) |
|---|---|---|---|---|---|---|---|
| HEVC static | 13 of 15 | +0.07 | +0.13 | -0.0002 | -0.02 | 3.94 (1.02) | 2.44 (1.76) |
| HEVC drone | 12 of 12 | +0.22 | +0.55 | +0.0007 | +0.03 | 9.01 (1.38) | 3.75 (2.14) |
| HEVC crowd | 13 of 15 | -0.01 | -0.09 | -0.0009 | +0.11 | 2.87 (0.81) | 1.68 (1.64) |
| AV1 static | 14 of 15 | -0.04 | -0.10 | -0.0012 | +0.10 | 3.04 (1.06) | 2.09 (1.87) |
| AV1 drone | 12 of 12 | +0.08 | +0.42 | +0.0003 | -0.10 | 5.49 (1.38) | 2.61 (2.13) |
| AV1 crowd | 8 of 15 | -0.01 | -0.14 | -0.0016 | +0.37 | 2.66 (0.88) | 1.81 (1.74) |

**Answer: no.** At the same target and rate a held level is as good as the burst cycle, within 0.1 dB
(drone on HEVC +0.22 dB), and the open-loop hold breaks the buffer on the way. The constant-level BD gain
is a statement about where on its own curve the stream sits: on a still scene the last 20-40 % of the bits
buy little, so a stream that stops spending moves along a steeper part of the curve. The cycle is how CBR
spends a target the picture does not need, not a waste a better controller could turn into quality.

So the only thing a "settling" controller can deliver is the undershoot: 20-40 % fewer bits for 0.5-1.1 dB
less on fixed cameras. That lever was built, screened and deleted on 2026-11-15
(`rate.settle_seconds`, `docs/analysis/cbr-ideas-2026-11-15.md`: one clip of 27 on L3, a second that gave up
0.5-1 dB for half its rate; "whether a comfortable stream should give up 0.5-1 dB for half its rate is the
owner's call"). These runs reproduce its oracle and add nothing that changes that decision.

**What bounding the bursts costs** (live arms, J % against CBR; all / crowd / drone / static):

| Arm | HEVC | AV1 | keyless 1 s window p95, static | re-codes per 1000 frames, static |
|---|---|---|---|---|
| shipped | | | 1.38 / 1.26 | 12 / 24 |
| `rate.window_seconds` 1 | +3.9 / +3.1 / +1.7 / +6.8 | +4.3 / -0.3 / +7.7 / +5.5 | 1.01 / 1.01 | 63 / 69 |
| `rate.recode_overflow_inter` 1.1 | +2.8 / +0.1 / +3.9 / +4.2 | +2.0 / 0.0 / +2.9 / +3.0 | 1.31 / 1.24 | 19 / 31 |
| `rate.still_level_damping` 0.5 on HEVC (pinned 0 there) | -1.1 / -0.1 / -3.0 / -0.2 | | 1.31 | 14 |
| `rate.still_level_damping` 0 on AV1 (0.5 there) | | -0.15 / -0.2 / 0.0 / -0.3 | 1.30 | 20 |

Conformity on still scenes is bought with quality (the bursts are refinement the stream otherwise cannot
place). HEVC's damping pin stands for the reason it was set (2026-11-19: the pulse guard on one clip, a
real key loss there); it is worth about 1 % here, on drone.

## Open

- A live parameter for the start-up key, and `nceval.quick` / `nceval.train` for `rate.key_hold_still` once
  the owner has picked N.
- Why `rate.key_hold_still` moves the pan clips, and the still boost a held key should keep on crowd.

- The constraint-keeping ceiling as one number: the largest key that keeps a stated key-second bound
  (say 2.5x), by replay over a grid of key shifts, per class.
- The key slope on HEVC and on the layered presets (run for AV1 medium only).
- Whether a size model with the reference-step term changes J or the conformity table: build it
  default-off in `rate_control.rs` (`RateModel`), then `nceval.quick`.
- Why `rate.cadence_scale=0.5` loses more than no cadence.
- The 150-frame window has one drone clip and two texture clips; the 600-frame window has no videocall
  clip.
