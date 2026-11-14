# What our steady-state frames do differently (2026-11-13)

**Question.** Over 600 frames, with a key every 10 s, our gaps to x264 `medium` shrink by 6–10 points.
long-window-loss-2026-11-13.md places the loss in the steady-state frames (after the first 150), mostly
on MS-SSIM. Halving the start-up spend does not recover it. What do our steady-state frames do
differently from the references' at a matched rate? This is a diagnosis; nothing was changed.

**Data.**

- **Clips.** Four pool clips of ≥ 600 frames, outside the sealed sets, one per worst class:

  | Class | Clip | Size |
  |---|---|---|
  | Crowd | `pool_chad_1_093` | 1280x720 |
  | Static | `long_static_chirla_s024c2` | 1080x720 |
  | Pan | `pool_pan_bdd100_bdd_00ce8219_d0b5582e` | 1280x720 |
  | Body-cam | `pool_bodycam_bwc_video118` | 640x360 |

- **Rates.** Two per clip: the chart ladder's 2nd and 4th rungs (350 / 999 kbps at 720p; 308 / 880 for
  the static clip). The body-cam clip's ladder has only three rungs, so it takes its outer two (71 / 212).
  All are ≤ 50% of the source rate.
- **Encoders.** 600 frames from frame 0, a key every 300, CBR with the chart's 1 s buffer and command
  lines (`nceval.cbr.Family`):
  - our AV1 and H.264 Default, built at 624a1f40;
  - SVT-AV1 M8 (`--rtc`), x264 `medium` and x265 `medium` (zerolatency, flat P).
- **Metrics.** Per frame: PSNR-Y, PSNR-YUV (6:1:1) and MS-SSIM (`nceval.msssim`, libvmaf's
  `float_ms_ssim` on luma, shown in dB as −10·log10(1 − mean)). Each is averaged per 150-frame window.
  - Ours is scored on its displayed reconstruction (held frames repeat), anchors on an ffmpeg decode.
  - No YUV is written. No VMAF.
- **Tools.** `nanocodec stat` (analysis build) on every stream, including the anchors'. Shares are of
  luma area over the window's P frames.
  - The per-block QP spread is the area-weighted SD of the block QP (or qindex) within a frame,
    averaged over frames.
  - "Steady" means windows 1 and 3. Window 2 holds the key at frame 300 and its start-up.
- **Files.** Streams, scripts and tables are in `~/nanocodec-data/work/steady-state/`: `scripts/run.py`,
  `windows.py`, `report.py`, and `report.md` for the full tables.
- **Not available.** The records do not split mode/MV bits from coefficient bits for H.264 or AV1.
  Our AV1 block bits were also wrong at tile ends (fixed in the commit before this note's), so no
  bit split is shown.

**Limits.** 4 clips × 2 rates, one clip per class. The numbers show direction, not population size. The
per-window rates are not matched: CBR matches the total, and each encoder shapes it differently. That
shape is finding 1.

## Headline

Five differences, ranked by how clearly they separate steady-state from window 0:

| # | Difference (windows 1–3 vs 0) | Ours | References | Points to |
|---|---|---|---|---|
| 1 | **Rate shape.** The references spend more after window 0; ours stays flat | window kbit, steady / w0: AV1 +1.7%, H.264 +0.3% | SVT +10.0%, x264 +3.0%, x265 +1.0%; static 880: SVT +15%, x264 +7% | Start-up allocation, as in long-window-loss cause 1 (`rate.first_key_boost_scale`, `rate.key_repay_seconds`). Most of the steady AV1–SVT gap on static, little of it on crowd |
| 2 | **Motion field on moving cameras.** The references' vectors get finer after start-up; ours stay zero or integer | H.264 pan 350: zero-MV 57 → 67%, sub-pel 24 → 15%. Body-cam 212: zero 89 → 85%. AV1 body-cam sub-pel 11 → 18% | x264 pan 350: zero 17 → 23%, sub-pel 75 → 69%. Body-cam 212: zero 60 → 41%, sub-pel 26 → 41%. SVT body-cam sub-pel 37 → 53% | H.264 skip / zero-vector lock-in (P_Skip inherits a zero vector from zero neighbours, H.264 §8.4.1.1); the sub-pel gates (`decision.codec_subpel_seed_lambdas`, `decision.subpel_*`) |
| 3 | **AV1 blocks and transforms shrink in steady state on static/crowd** | cb ≥ 32: −2 to −8 points (crowd 999: 84 → 75%); inter tx ≥ 16: −2 to −26 points (static 880: 74 → 48%); more coded area than SVT on crowd (999: 13% vs 9%) | SVT cb ≥ 32 changes −2.4 to +1.3 points (86–94%); tx ≥ 16 −2.5 to −10 | nc-av1 partition and transform-size decisions at the lower qindex of steady state (the λ–qindex relation and the split costs) |
| 4 | **Frame-QP cadence amplitude** | steady AV1 frame qindex SD 21, mean step 18 per frame; H.264 SD 4.1, step 3.3 | SVT SD 14, step 9; x264 SD 1.4, step 0.9 | `qp_cadence.rs`, but **not a cause**: coding steady state flat (`rate.cadence_period=1`) loses 0.46 / 0.42 dB PSNR-Y and 0.48 / 0.41 dB MS-SSIM in windows 1 / 3 (AV1; H.264 −0.29 / −0.21 dB MS-SSIM). Keep the cadence |
| 5 | **No block-level AQ** | per-block QP SD 0.2 | SVT 1–13 qindex (mean 6), x264 2–6 QP (mean 3.8), x265 1.1 QP | `rate.aq_strength` (0 in Default). **Not steady-state specific**: variance AQ at strength 1 trades −0.5 (AV1) / −0.8 (H.264) dB PSNR-Y for +0.1 to +0.3 dB MS-SSIM in every window alike, window 0 included |

Two smaller items:

- **The periodic key.** Our AV1 key at frame 300 costs ~248 kbit against SVT's ~46 kbit; H.264 ~249
  against x264's ~216 (x265 ~178). Window 2 is not our weak window against SVT: SVT's cheap key hurts its own
  window 2. Against x264 it is our AV1's weakest (+0.66 dB MS-SSIM vs +1.15 / +0.81 in windows 1 / 3).
- **Skip share** differs by class rather than by window. In steady state our AV1 skips 1–5 points
  less than SVT on crowd and static, and 3–14 points more on pan and body-cam. Our H.264 skips 2–9
  points more than x264 everywhere, most on body-cam.

## Window means (8 streams per encoder: 4 clips × 2 rates)

Steady means windows 1 and 3. Tool shares are % of P-frame luma area.

| Encoder | Window | kbit / window | Frame QP SD | Block QP SD | Skip | Intra | cb ≥ 32 | Sub-pel | Zero MV | PSNR-Y | MS-SSIM dB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours AV1 | 0 | 2539 | 24.3 | 0.2 | 72.5 | 0.3 | 80.6 | 22.0 | 68.7 | 34.6 | 16.4 |
| | steady | 2564 | 21.1 | 0.2 | 69.9 | 0.3 | 78.6 | 21.5 | 65.7 | 35.0 | 17.3 |
| SVT-AV1 M8 | 0 | 2485 | 19.5 | 7.7 | 66.9 | 3.7 | 89.6 | 27.5 | 64.1 | 34.4 | 15.8 |
| | steady | 2662 | 14.2 | 5.6 | 66.5 | 3.7 | 88.3 | 30.1 | 60.9 | 36.0 | 18.3 |
| ours H.264 | 0 | 2571 | 4.7 | 0.2 | 82.0 | 4.8 | — | 12.4 | 81.4 | 33.4 | 15.4 |
| | steady | 2526 | 4.1 | 0.2 | 81.9 | 5.2 | — | 10.5 | 83.1 | 33.8 | 16.3 |
| x264 medium | 0 | 2397 | 2.5 | 3.7 | 77.8 | 6.9 | — | 26.7 | 68.9 | 31.1 | 14.3 |
| | steady | 2455 | 1.4 | 3.8 | 77.7 | 6.5 | — | 27.4 | 67.4 | 32.3 | 16.3 |
| x265 medium | 0 | 2444 | 1.9 | 1.1 | 69.9 | 3.2 | 86.9 | 31.6 | 64.2 | 33.4 | 15.9 |
| | steady | 2458 | 1.2 | 1.1 | 72.2 | 2.8 | 87.6 | 32.7 | 62.4 | 34.4 | 17.4 |

Zero MV is the share of vector-carrying area (skip included) with a (0, 0) vector. H.264 coding blocks are
16x16 macroblocks, so cb ≥ 32 does not apply.

**Ours minus reference, same clip and rate** (mean of 8 pairs):

| Pair | Window | kbit | PSNR-Y | PSNR-YUV | MS-SSIM dB |
|---|---|---:|---:|---:|---:|
| AV1 − SVT M8 | 0 | +54 | +0.15 | +0.65 | +0.59 |
| | 1 | −89 | −1.04 | −0.45 | −1.08 |
| | 2 | −20 | −0.28 | +0.35 | +0.12 |
| | 3 | −108 | −1.01 | −0.47 | −0.97 |
| H.264 − x264 medium | 0 | +174 | +2.33 | +2.28 | +1.13 |
| | 1 | +69 | +1.73 | +1.73 | +0.22 |
| | 2 | +47 | +0.88 | +0.93 | −0.53 |
| | 3 | +73 | +1.27 | +1.29 | −0.24 |

From window 0 to steady state our AV1 loses about 1.2 dB PSNR-Y and 1.6 dB MS-SSIM against SVT. Our H.264
loses about 0.8 dB PSNR-Y and 1.1 dB MS-SSIM against x264. MS-SSIM moves more than PSNR, as in the
independent test.

## 1. Rate shape: the references' spend rises after start-up, ours does not

Window kbit, steady state against window 0 (mean of the per-stream ratios): ours AV1 +1.7%, H.264
+0.3%; SVT +10.0%, x264 +3.0%, x265 +1.0%. Per clip the references' rise is larger where we lose most:

- **Static 880.** SVT 4034 → 4636 kbit (+15%), x264 3918 → 4186 (+7%). Ours AV1 4386 → 4404, H.264
  4444 → 4245.
- **Crowd 999.** SVT +2%, x264 +6%. Ours AV1 +3%, H.264 −1%.

**At matched steady-state rate the AV1 gap is smaller than it looks.**

| Clip | Ours AV1 (steady) | SVT (steady) | Gap |
|---|---|---|---|
| Static 880 | 41.8 dB PSNR-Y, 24.7 dB MS-SSIM at 4404 kbit | 42.0 / 25.0 at 4636 (+5%) | About the rate difference |
| Crowd 999 | 33.7 / 17.4 at 4995 | 34.9 / 18.7 at 5125 (+3%) | Still ~1.2 dB |

So the shape explains part of the static loss, and the crowd loss needs findings 2–3. The control knobs
are the ones long-window-loss already names (first-key budget and repay).

## 2. Motion field on moving cameras

On the pan and body-cam clips (H.264's worst classes), the references' vectors become finer after
start-up. Ours stay at zero or at whole samples.

| Clip, rate | Encoder | Zero MV w0 → steady | Sub-pel w0 → steady | Sub-pel among non-zero |
|---|---|---|---|---|
| Pan 350 | ours H.264 | 57 → 67% | 24 → 15% | 55 → 46% |
| | x264 medium | 17 → 23% | 75 → 69% | 90 → 89% |
| Body-cam 212 | ours H.264 | 89 → 85% | 5 → 7% | 46 → 48% |
| | x264 medium | 60 → 41% | 26 → 41% | 66 → 69% |
| Body-cam 212 | ours AV1 | 69 → 51% | 11 → 18% | 36 → 36% |
| | SVT M8 | 54 → 35% | 37 → 53% | 80 → 82% |

**Quality follows.** On pan, our H.264 loses more in steady state than x264 at about the same rate:
- 350 kbps: PSNR-Y −2.2 dB (x264 −1.3), MS-SSIM −1.6 dB (x264 −0.9);
- 999 kbps: PSNR-Y −1.8 dB (x264 −1.0).

Our intra share also rises there (2.9 → 4.9%, x264 1.5 → 2.7%), as it does where motion fails.

**Why it points where it does.**
- In H.264 a P_Skip block's vector is the median predictor. It is forced to zero when the left or upper
  neighbour has a zero vector on reference 0 (H.264 §8.4.1.1). So zero-vector skips spread across a
  moving scene unless some blocks are coded as inter with real vectors.
- x264 skips as much as we do (79 vs 81% on pan 350) but with predicted, non-zero vectors.
- What to look at:
  - our skip / zero-vector decision on camera motion: the H.264 backend's P_Skip candidate and the
    engine's still-block bias;
  - the sub-pel gates (`decision.codec_subpel_seed_lambdas`, `decision.subpel_*`). Even among non-zero
    vectors our sub-pel share is 46–55% against x264's 66–90%.

AV1 on the pan clip is close to SVT (zero 11–16% vs 10–13%, sub-pel 60–75% vs 62–75%). AV1's gap is on
body-cam.

## 3. AV1 block and transform sizes shrink in steady state

On static and crowd (AV1's worst classes), our AV1 moves to smaller blocks and transforms as the qindex
falls after start-up. SVT keeps its sizes.

| Clip, rate | Encoder | cb ≥ 32 w0 → steady | inter tx ≥ 16 w0 → steady | Coded luma area steady | qindex w0 → steady |
|---|---|---|---|---|---|
| Crowd 999 | ours AV1 | 84 → 75% | 93 → 89% | 13.0% | 169 → 154 |
| | SVT M8 | 89 → 91% | 93 → 90% | 8.8% | 173 → 146 |
| Static 880 | ours AV1 | 85 → 81% | 74 → 48% | 5.5% | 124 → 95 |
| | SVT M8 | 89 → 86% | 79 → 69% | 8.2% | 129 → 103 |
| Static 308 | ours AV1 | 91 → 86% | 92 → 88% | 2.0% | 179 → 143 |
| | SVT M8 | 94 → 93% | 92 → 85% | 4.0% | 174 → 132 |

Both encoders lower qindex by a similar amount. Only ours trades large blocks and transforms for small
ones. On crowd 999 SVT is 1.2 dB ahead in steady state with 90% skip (ours 85%) and less coded area. This
points at nc-av1's partition and transform-size decisions as qindex falls: how λ scales with qindex, and
the split / transform-size rate terms. It is a hypothesis to test; nothing here shows the smaller blocks
are wrong.

## 4. Frame-QP cadence: larger than the references', but it helps

**Ours swings more than the references.**
- AV1: frame qindex SD 21 in steady state, mean step 18 per frame (SVT 14 / 9).
- H.264: SD 4.1, step 3.3 (x264 1.4 / 0.9).
- A steady-state stretch of crowd 999 (frames 200–223):
  - ours AV1: 190 215 216 224 180 213 194 193 156 182 … 137 163 144 143;
  - SVT: 137 140 154 163 140 157 145 140 147 140 194 … 140 131;
  - x264: 30 ± 1 throughout.

**Counterfactual.** `rate.cadence_period=1`, the steady-state cadence off with the start-up cadence kept;
all 8 points.

| | Window | kbit | PSNR-Y | PSNR-YUV | MS-SSIM dB |
|---|---|---:|---:|---:|---:|
| AV1 flat − Default | 0 | +0.7% | +0.01 | +0.01 | +0.01 |
| | 1 | −0.9% | −0.46 | −0.35 | −0.48 |
| | 3 | +2.4% | −0.42 | −0.32 | −0.41 |
| H.264 flat − Default | 1 | +1.3% | −0.24 | −0.21 | −0.29 |
| | 3 | +4.4% | −0.15 | −0.13 | −0.21 |

On crowd and static the flat steady state cuts our AV1 frame-QP SD from 24–34 to 10–18, near SVT's, and
it is worse in every steady window. The
cadence is not what loses the steady state; `qp_cadence.rs` should stay as it is.

## 5. No block-level AQ: a level difference, not a steady-state one

All three references vary the quantizer within a frame:
- SVT: per-block qindex SD 4–12;
- x264: 2–5 QP;
- x265: ~1 QP.

Ours does not (0.2; `rate.aq_strength` is 0 in Default).

**Counterfactual.** `rate.aq_strength=1`, all 8 points.

| | Window | kbit | PSNR-Y | MS-SSIM dB |
|---|---|---:|---:|---:|
| AV1 AQ − Default | 0 | +1.4% | −0.38 | +0.23 |
| | 1 / 3 | −2.1% / −0.6% | −0.55 / −0.59 | +0.08 / +0.29 |
| H.264 AQ − Default | 0 | −0.8% | −0.79 | +0.07 |
| | 1 / 3 | +0.9% / +2.1% | −0.89 / −0.92 | +0.05 / +0.14 |

The trade is the same in window 0 as in steady state. Our variance AQ gives some MS-SSIM, at a larger PSNR
cost, wherever it runs. It is not what changes after start-up. Whether a better AQ (SVT-style, or
x264's variance AQ, which keeps PSNR) would win the MS-SSIM third is a separate question.

## Suggested next steps (not done)

1. **H.264 zero-vector lock-in on moving cameras** (finding 2). Count P_Skip blocks whose vector was
   forced to zero by §8.4.1.1 on the pan and body-cam clips. Screen a cost that lets inter blocks with the
   predicted vector break the zero chain. Then check AV1's body-cam sub-pel gap through the sub-pel
   gates.
2. **AV1 partition / transform size against qindex** (finding 3). Compare the split and transform-size
   decisions at qindex 95–155 with SVT's on the crowd and static clips. The λ–qindex mapping and the
   split rate are the first suspects.
3. **Steady-state rate shape** (finding 1) is already on the long-window list. On static it is most of
   the AV1 gap.
4. Finding 4 (cadence) and finding 5 (AQ) need no action from this study.
