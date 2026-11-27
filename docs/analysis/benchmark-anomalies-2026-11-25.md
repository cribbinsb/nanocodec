# What the 2026-11-24 re-measure showed, and what was done (2026-11-25)

The benchmark re-measure of build 889309c99 (2026-11-24 21:58) against 74aaa1fcd
(2026-11-23 20:54) is in `~/nanocodec-data/bench/bp3r-vs-74aaa1fcd.txt` (BD-J against x264
medium, speed in Mpx/CPU-s on c8i.2xlarge). This note records, for each oddity in that table,
the cause found in the two builds' per-point records (`bench/bp3r-*/run`, `bench/bp3-*/run`:
bytes, CPU seconds, per-frame sizes and scores of every point) and what was changed. The
benchmark sets are the evidence here, never the tuning set: every value below was chosen on
pool draws of 2026-11-25.

"Paired" below means the new build's curve against the old build's curve of the same preset
and clip (nceval.bdj, the benchmark's BD-J), over every clip both builds coded, with a
bootstrap interval over clips. It is a direct measure of a change; the table's differences go
through the x264 anchor and its common clip set, which is why they differ.

## 1. H.264 slower: 1.4-1.7x its old time

- **Cause:** 8cd85aec5 (2026-11-24 03:02) replaced the rung: slow plus `h264.rd_refine 1`,
  `h264.subpel_skip_variance 13.76` and `decision.final_margin_lambdas 0.217`. Intended and
  screened with CPU (cycles against slow +54 to +61 % on three long clips).
- **The old rung was off the front:** in all three CBR views and in AOM-CTC Fixed QP it was
  slower than slow and no better (ours CBR -23.10 at 90.8 against slow's -23.16 at 104.4).
  The new one is not dominated in any view.
- **Its cost in the benchmark is higher than the commit states:** against slow it takes
  1.86x / 1.87x / 1.76x the time (ours / VC-1K / AOM-CTC CBR) and 1.75x / 1.72x at Fixed QP,
  for -1.5 / -0.9 / -0.7 and -3.2 / -1.9 points: 1.2-2.5 points per e-fold of time under rate
  control, 3.4-5.7 at Fixed QP. The step before it (medium to slow) is 10-15.
- **Where the time goes** (aws_cpu_ab, CBR 600, four quick clips, 150 f, cycles against slow,
  c8i / c8g): `rd_refine` alone +62.0 / +65.3 %; the other two alone +5.2 / +5.6 %; all three
  +66.8 / +70.7 %.
- **Where the quality comes from** (nceval.quick against slow, draw 2026-11-25, quality only):
  `rd_refine` alone L1 -1.23 % [-2.22, -0.80], probe -0.91 % [-1.41, -0.59]; the other two
  alone L1 -0.27 % [-0.49, -0.04], probe +0.05 % [-0.22, +0.13]; all three L1 -1.01 % [-1.88,
  -0.67], probe -0.71 % [-1.45, -0.26].
- **Conclusion:** the rung is RD vector refinement, at about 2 points per e-fold: the slow end
  of H.264's front, as the ledger has held since 2026-11-15 (`h-264-rd-refinement-of-inter-
  vectors`: "k* <= ~0.03"). Nothing cheaper reaches its quality. The other two parameters add
  about 5 % time and no measured quality on top of the refinement; see "Left open".

## 2. H.264 medium and slow got worse at Fixed QP

- **Cause:** 6b6450187 (Brief 20): H.264's fixed-QP profile key offset 8 -> 10 and
  `rate.fixed_qp_moving_key` 6 -> 8. Nothing else reaches these rungs' fixed-QP streams: at
  equal QP the key frames grew 8-31 % (+1.0 dB), the P frames' bytes moved under 0.3 %, total
  bytes +0.8 % (QP 22) to +7.6 % (QP 47) for +0.02 to +0.23 dB PSNR-Y.
- **Paired, held-out "ours" Fixed QP (n=97):** medium +0.78 % [+0.35, +1.21], slow +0.67 %
  [+0.22, +1.10].
- **Why its screen passed:** Brief 20 ran before `nceval.fixedqp ab` was moved to the
  benchmark's six base quantizers (8d752939d, 19:44; Brief 20's tree was based on c9aade2cf,
  18:07). It coded four quantizers walked to the ladder's rates and decided on the old J.
- **Re-screen on the benchmark's quantizers** (nceval.fixedqp ab, BD-J, draw fixedqp-2026-11-25,
  n=24), candidates against 10 / 8:

  | offset / moving key | medium 150 f | slow 150 f | slower 150 f | medium 300 f |
  |---|---|---|---|---|
  | 9 / 7 | -0.35 [-0.73, -0.07] | -0.44 [-0.81, -0.14] | -0.39 [-0.79, -0.09] | -0.23 [-0.58, +0.05] |
  | 8 / 6 (before Brief 20) | -0.47 [-1.31, +0.14] | -0.47 [-1.29, +0.12] | -0.43 [-1.33, +0.22] | -0.15 [-0.93, +0.45] |
  | 7 / 5 | -0.34 [-1.57, +0.56] | -0.34 [-1.54, +0.56] | -0.16 [-1.41, +0.75] | +0.17 [-1.05, +1.17] |
  | 6 / 4 | +0.77 [-1.03, +1.96] | +0.73 [-1.05, +1.95] | +1.04 [-0.82, +2.23] | +1.09 [-0.71, +2.38] |

- **Fix:** 9 / 7 in `FIXED_QP_PARAMS` (nc-h264/src/backend.rs): the only value with an
  interval below zero on all three rungs at 150 f, and the best at 300 f. Key frames only, so
  no CPU run (Brief 20 measured 8 -> 10 at -0.08 / -0.05 %). veryfast, faster and fast keep
  their pinned 8 / 6 (not screened with 9).

## 3. The fast rungs paid 4-16 % speed for their Fixed QP gains

- **Cause:** Brief 19 (78835b834, d0bf64eb4): the fixed-QP profile unpinned on the fast rungs,
  coarser moving keys, and the low-quantizer gates relaxed (`planner.depth_removal_min_quantizer`,
  `h264.inter_zero_luma_scale`, `hevc.pre_transform_zero_scale`), each under the `fixed_qp.`
  prefix. Screened with CPU; the benchmark's time ratios match the screens' (HEVC faster
  +17.9 / +15.4 % screened, x1.16-1.19 measured; H.264 veryfast +8.5 / +10.2 %, x1.06).
- **No leak into CBR:** in "ours" CBR the streams of H.264 veryfast / faster / fast, VP9
  veryfast / faster and AV1 veryfast are byte-identical between the builds (373 of 373 points
  each), HEVC faster's are but for 5 points (first difference after frame 30; not traced),
  and these rungs' CBR time ratios are 0.976-1.000.
- **Against the front:** each rung's step (its old self to its new self) against the slope to
  its next slower rung, points per e-fold of time, ours / AOM-CTC Fixed QP:

  | rung | step | to the next rung | speed ratio to the next rung (was) |
  |---|---|---|---|
  | HEVC faster | 65 / 53 | 42 / 40 | 1.31 / 1.26 (1.48 / 1.45) |
  | AV1 veryfast | 76 / 96 | 5 / 18 | 1.28 / 1.27 (1.32 / 1.28) |
  | AV1 faster | 71 / 75 | -2 / 4 | 1.29 / 1.33 (1.31 / 1.37) |
  | H.264 veryfast | 95 / 129 | 41 / 21 | 1.07 / 1.06 (1.11 / 1.09) |
  | VP9 faster | 46 / 80 | 19 / 13 | 1.30 / 1.34 (1.29 / 1.32) |
  | **VP9 veryfast** | **40 / 49** | **56 / 63** | **1.13 / 1.12 (1.17 / 1.18)** |

  All are on their fronts but VP9 veryfast, whose step is flatter than its slope to VP9
  faster, at 1.13x faster's speed.
- **VP9 veryfast, the depth floor** (`fixed_qp.planner.depth_removal_min_quantizer = 0.86`).
  Brief 19's own table already had its increment under the slope (the free part -2.1 % at
  -0.8 % CPU; with the floor -6.0 % at +11.6 %: 0.31 % per 1 % against 0.464); the bundle's
  average passed. On the benchmark's quantizers (nceval.fixedqp ab, BD-J, draw
  fixedqp-2026-11-25, n=24, 150 f), against the floor at 0.86:

  | floor | VP9 veryfast J | VP9 faster J |
  |---|---|---|
  | 0 (off) | +3.73 [+2.89, +4.53] | +2.44 [+1.59, +3.24] |
  | 0.5 | +2.94 [+2.25, +3.57] | +1.94 [+1.20, +2.61] |
  | 0.6 | +1.60 [+1.16, +1.99] | +1.13 [+0.66, +1.55] |
  | 0.7 | +0.71 [+0.52, +0.89] | +0.56 [+0.28, +0.79] |

  CPU of floor 0 against 0.86 (aws_cpu_ab, cycles, four quick clips, 150 f, c8i / c8g), at
  qindex 110 / 160 / 210: veryfast -15.2 / -11.6 / -4.6 % and -12.7 / -10.2 / -2.8 %; faster
  -9.2 / -7.9 / -4.2 % and -6.9 / -6.3 / -4.3 %. Over the six quantizers (the other three
  interpolated, weighted by the benchmark's time shares) the floor costs veryfast about
  +11.5 / +9 % and faster about +7.5 / +6 %.
  - veryfast: 0.3-0.4 % J per 1 % CPU against 0.46 (Brief 19's slope) to 0.6 (the
    benchmark's): under the front. The intermediate floors lie on the same line (no knee).
  - faster: 0.33-0.41 against 0.13-0.19: over the front; it keeps its floor.
- **Fix:** the floor removed from VP9 veryfast (fixed QP only). Expected in the benchmark's
  Fixed QP views: about 3.7 points worse and 9-11 % faster, so about 1.26x faster's speed.

## 4. The AV1 period-8 cadence shows less than its screens

Screens that adopted it (ledger `av1-cbr-cadence-period-8-at-medium`): medium L3 -0.51,
final check -0.50; fast L3 -0.29, final check -0.49; faster L3 -0.62, final check -0.18.

Paired, new against old build, every clip both coded (new also carries 6efa4646a, the
inheritance measure, on slow and slower, and 60610ba58, their depth above 1):

| rung | ours CBR (n=84-85) | VC-1K CBR (n=99) | AOM-CTC CBR (n=33-34) |
|---|---|---|---|
| medium | -0.56 [-0.89, -0.26] | -0.40 [-0.52, -0.29] | +0.23 [-0.01, +0.47] |
| slow | -0.55 [-1.06, -0.11] | -0.78 [-0.96, -0.60] | +0.34 [-0.00, +0.70] |
| slower | -0.50 [-1.05, -0.04] | -0.78 [-0.96, -0.60] | +0.22 [-0.09, +0.55] |
| fast | -0.30 [-0.59, +0.01] | -0.25 [-0.37, -0.13] | +0.15 [-0.07, +0.38] |
| faster | -0.16 [-0.50, +0.11] | +0.03 [-0.09, +0.17] | -0.12 [-0.32, +0.05] |

- **"ours" and VC-1K reproduce the screens.** The table's smaller "ours" figure (-0.35) is its
  common set: 41 of the 84 clips, on which the paired figure is -0.32 [-0.78, +0.16].
- **AOM-CTC is the frame rate.** Its 13 clips at 50 / 60 fps: medium +0.72 [+0.37, +1.10],
  slow +0.78 [+0.29, +1.29], slower +0.78 [+0.33, +1.26], fast +0.34 [+0.10, +0.61]; its 21
  clips at 25 / 30 fps: medium -0.08 [-0.32, +0.15]. At equal target the new medium streams
  are +1.4 to +1.6 % larger on the 50 / 60 fps clips for +0.025 dB PSNR-Y (+0.6 to +0.9 % and
  +0.02 dB at 25 / 30 fps; -0.4 % and +0.01 dB in "ours").
- **Why the screens did not see it:** the pool has 8 clips above 30 fps of 206; "ours" and
  VC-1K have none.
- **Not established:** that the period itself (and not another change between the builds) is
  what loses at 50 / 60 fps, and why. No fix was made: a frame-rate rule chosen on these 13
  clips would be tuned on a sealed set.

## 5. The rest of the table

- **Speed noise on unchanged streams.** Rungs whose every stream is byte-identical between
  the builds: time ratio 0.976-1.009 in "ours" CBR (median 0.988) and 0.993-1.014 at Fixed
  QP (median 1.005). So a speed difference under about 1.5 % around a view's median is not
  a change.
- **VVC about 1-2 % slower on identical streams** (Fixed QP 1.013-1.026 against the other
  codecs' 0.993-1.011; table ratios 0.966-1.000 in the four views other than "ours" CBR) is not the code: plain builds
  of the two commits, interleaved (aws_cpu_ab, CBR 600, c8i / c8g cycles): VVC fast -0.63 /
  -0.25 %, faster -0.44 / -1.78 %, instructions -0.3 to -0.7 %; H.264 medium -1.5 / -1.7 %;
  AV2 fast -0.8 / -1.1 %. The benchmark's two PGO builds differ beyond that; which part is
  profile, layout or box was not separated.
- **CBR streams that changed on rungs with no preset change** (AV2 medium 160 of 373 points,
  HEVC slow 154, HEVC veryslow 152, VVC medium 162, VVC slow 155; first difference never
  before frame 9): the rungs 6efa4646a lists (the inheritance measure reads the coded block
  quantizer). Paired: AV2 medium +0.05 / -0.37 / -0.10 (ours / VC-1K / AOM-CTC), none worse
  with an interval above zero.
- **AV2 medium, AOM-CTC CBR -0.68:** its common set is 9 clips; paired over its 34 clips the
  change is -0.10 [-0.27, +0.05].
- **AV1 fast behind AV1 faster in "ours" Fixed QP** (-23.05 at 132 against -23.62 at 170; in
  the old build too, -19.41 against -19.85): through the x264 anchor only. AV1 fast against
  AV1 faster directly, same clips: fast -5.6 % [-7.5, -3.6] (n=97); AOM-CTC -7.4 % [-10.1,
  -4.8]. At equal qindex fast spends 14-58 % more bytes for +0.8 to +1.9 dB, so the two
  curves overlap x264's range differently. Which reading the charts should carry is a BD-rule
  question, not an encoder one.
- **Ladders:** no other rung is dominated in any view. Sign differences between views: the AV1
  CBR rungs of section 4 only.

## Left open

- H.264 slower without `subpel_skip_variance 13.76` and `final_margin_lambdas 0.217` (slow
  plus `rd_refine 1` alone), against slower (nceval.quick, draw 2026-11-25, CPU on its own
  encodes): L1 -0.13 % [-0.26, +0.08], probe 1/150 +0.04 % [-0.39, +0.31], 1/300 -0.11 %
  [-0.15, -0.03], cycles -2.5 / -2.7 %: CONTINUE. Queued for the train's L3 and final check
  (`nceval.train` 723c68b247, idea `h-264-slower-is-slow-plus-rd-refine-alone`); not landed.
- The fixed-QP key offset 9 / 7 on H.264 veryfast, faster and fast (pinned at 8 / 6).
- AV1 period 8 on 50 / 60 fps content: the pool needs clips above 30 fps before it can be
  screened.
