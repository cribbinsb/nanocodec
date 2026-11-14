# A noisy still webcam fills any CBR target (2026-11-14)

**Report (owner).** "When encoding from a live webcam pointing into a stationary empty room the encoder will
happily consume 1 Mbit even with AV1, encoding noise, with non-zero fictional motion vectors."

**Policy.** The CBR target is a ceiling. Undershoot is wanted when there is nothing to code, and the encoder
should not spend bits coding sensor noise (docs/ARCHITECTURE.md §1, §6.6).

**Result.**
- Reproduced. Every codec fills every target, from 150 kbps to 2 Mbps, on a still room.
- Quality against the scene stops improving after about 3 s while the spend stays flat.
- The reference encoders fill the target too.
- Main causes:
  - no effective quality cap;
  - static detection that reads the webcam's noise as change;
  - cadence base frames that re-code the noise (50-62% of AV1's bits).
  Noise-matching vectors carry at most 6% of the bits.
- Two registered levers landed, both off.
- **The noise skip** (`decision.noise_skip`) is recommended for every Default:
  - at the same rate, quality against clean is equal or better (up to +2 dB);
  - it is robust to slow light changes and moving objects;
  - it costs 18-23% fewer encode instructions on a still camera.
- **The noise-relative floor** (`rate.noise_floor_scale`) makes a still scene undershoot (5-30% of target),
  but it is not safe as a default: under it, slow changes accumulate.
- Undershoot without that risk needs rate control to stop re-spending the skipped area (§6, proposed).
- The captures are private and stay outside the repo; this note holds statistics only.

## 1. Captures and the camera's noise

The camera is a Microsoft LifeCam Cinema:
- 720p30 is offered only as MJPEG; raw YUYV goes up to 640x480 at 30 fps.
- In the room's light the camera delivers 15 distinct frames per second. A 30 fps capture repeats every other
  frame, so the clips were reduced to the distinct frames: 300 frames, 20 s.

Three clips were kept:
- 1280x720 MJPEG;
- 640x480 YUYV, which has a real light change near 10 s;
- the 720p clip scaled to 854x480 bilinear. This is what the demo codes: 854x480, 15 fps, CBR 300 kbps,
  peak 600, 1 s buffer, Default preset, key every 30 s (`web/view-settings.js`, `web/panel.js`).

Noise, 720p MJPEG clip, frames 30-300:

| | luma | U | V |
|---|---:|---:|---:|
| frame-to-frame sigma (mean abs diff / 1.128) | 1.2 | 1.9 | 1.4 |
| per-pixel temporal sigma, mean | 1.7 | 2.0 | 1.5 |

- Sigma depends strongly on brightness: 0.45 for luma 224-255 (42% of the picture, clipped highlights) and
  2.0-3.6 for luma 64-191.
- The noise is spatially correlated. The rms of 8x8-block-mean frame differences is 0.50, about twice what
  white noise would give.
- A slow drift sits on top: the rms difference grows from 1.3 at lag 1 to 2.2 at lag 64.

"Clean" reference for the captures: a 17-frame centred moving average of the clip. It removes the white
component (to about 0.3 sigma) and keeps the drift. It is approximate. The honest measure is the synthetic
still (§4), where the clean original is known.

## 2. Reproduction (854x480, 15 fps, demo settings)

Achieved kbps; PSNR-Y against the clean reference.

| codec / preset | 150 | 300 | 1000 | 2000 |
|---|---|---|---|---|
| AV1 Default | 148 / 41.9 | 296 / 43.4 | 988 / 44.8 | 1975 / 45.3 |
| AV1 Fast | | 296 / 40.4 | 986 / 40.5 | |
| H.264 Default | | 298 / 44.7 | 994 / 45.0 | |
| HEVC Default | | 296 / 44.6 | 987 / 45.3 | |

- Steady state, AV1 Default: PSNR against clean is flat from frame ~50 on, while every frame keeps spending
  10 / 20 / 65 / 130 kbit at 150 / 300 / 1000 / 2000 kbps.
- AV1 Fast gains 0.1 dB for 3.3 times the bits.

References in their CBR real-time modes (same clip, 1 s buffer):

| encoder | 300 | 1000 |
|---|---|---|
| SVT-AV1 M10 `--rtc` | 295 / 44.9 | 993 / 45.7 |
| libaom rt cpu-used 10 | 300 / 45.6 | 987 / 45.8 |
| libvpx VP9 rt cpu-used 8 | 302 / 44.5 | 1000 / 45.1 |
| x264 veryfast zerolatency | 284 / 43.0 | 993 / 45.4 |

- All of them fill the target.
- libvpx `--noise-sensitivity=1` gave identical bytes in this build.

## 3. Why the bits are spent

Analysis build (`nanocodec stat`), AV1 Default at 1000 kbps, P frames:
- 95.6% of the inter area uses the zero vector.
- Non-zero vectors cover 4.4%, almost all sub-sample and under 2 samples.
- Zero-vector blocks with a residual cover 70% of the area and carry 94% of the bits. NEWMV blocks carry 5.7%.

Causes, with their shares:
- **(a) No effective quality cap. Main cause.**
  - `rate.max_quality_qp` and `rate.min_refinement_gain` are 0 in every preset. The floor is therefore the
    codec minimum: AV1 inter qindex reaches 43 at 1 Mbps and 20 at 2 Mbps.
  - The static skips prove a zero residual only at the frame quantizer, so at fine steps the noise is coded.
  - Skipping more does not lower the rate. With `av1.still_skip` at limit scales 1, 3 and 8 the rate stays
    at 988 kbps, because rate control re-spends the bits through a finer quantizer.
- **(b) Static detection reads the webcam's noise as change.**
  - Pre-analysis' static threshold follows the analyzer's noise estimate from the flattest blocks. On a
    webcam those are clipped highlights with no noise.
  - Only 7% of the 854x480 picture is classed static and offered to a static skip. 95% of those cells are
    exactly unchanged highlights.
  - The rate controller's zero-vector sigma (`residual_sigma`) reads 0.74; the synthetic stills read their
    true sigma (1.15 for 1.19, 2.84 for 2.91).
- **(c) Noise-matching vectors. Small.** 4.4% of the area and at most 6% of the bits. On flat noisy areas the
  motion search picks whichever vector matches the noise best.
- **(d) QP cadence base frames. Large.**
  - Every 8th frame is about 55 qindex finer.
  - It takes 50% (300 kbps), 57% (1 Mbps) and 62% (2 Mbps) of all AV1 bits.

## 4. What was built

| parameter | default | what it does |
|---|---|---|
| `decision.noise_skip` (c) | 0 | per-block noise skip, below |
| `decision.noise_skip_chroma_ratio` | 1 | chroma noise sigma as a multiple of luma |
| `rate.noise_floor_scale` (k) | 0 | finest step = k × the running sigma of zero-vector blocks |
| `rate.quality_floor_cadence` | 0 | 1 lets cadence base frames go below the floor by their offset |

**The noise skip** (`noise_skip.rs`):
- The engine keeps a per-8x8-cell noise variance. It learns only on frames where the zero vector predicts the
  cell about as well as the motion search's best match (zero SAD at most 5/4 of the best plus 1 per sample).
  The estimate is the zero-motion SAD against the previous source, smoothed over about 4 frames.
- A cell that moves is reset.
- A cell's sigma is capped at 6 times the frame median, and floored at 0.5 code value (reconstruction
  rounding).
- A block is tested only once all its cells have been still for 4 frames in a row.
- It passes when, in every 8x8 luma cell, the zero-vector residual against the *current* source has MSE
  ≤ c·σ²_cell, and the same holds in chroma with σ times the chroma ratio. It is then coded as a zero-vector
  skip, whatever the frame quantizer.
- The expected MSE is σ² plus the reconstruction's own error against the scene. So a block whose
  reconstruction is still poor (a starved key) keeps being refined, and drift is coded once it exceeds the
  bound.
- AV1, HEVC and H.264 take it through `CodecBackend::try_static_skip`. In H.264 this happens only where
  P_Skip's predicted vector is zero.

**The floor** (`quality_cap.rs`):
- It sets the finest step to k × the running zero-vector sigma.
- It lifts at once on an inter frame where under 25% of blocks are still; keys keep the estimate.

## 5. Measurements (all at a 1000 kbps target, 854x480, 15 fps)

### Synthetic noisy still: clean original known

The still is frame 0 of `static_meva_hospital`, scaled to 854x480 and repeated 300 times. Noise comes from
`eval/clipset/add_noise.py` at three gains. Each cell: achieved kbps / PSNR-Y against the clean original.
c = 4 and chroma ratio 2 throughout.

| sigma (luma) | codec | off | noise skip | skip + floor k16, cadence kept |
|---|---|---|---|---|
| 1.15 (camera-like) | AV1 | 987 / 43.54 | 987 / 43.61 | 54 / 42.78 |
| | H.264 | 986 / 43.82 | 987 / 45.79 | 50 / 41.77 |
| 2.9 (`low`) | AV1 | 986 / 36.93 | 987 / 38.30 | 59 / 38.15 |
| | H.264 | 986 / 37.70 | 986 / 38.56 | 54 / 38.09 |
| 7 (`high`) | AV1 | 987 / 30.83 | 988 / 31.07 | 70 / 32.48 |
| | H.264 | 985 / 31.89 | 986 / 32.22 | 84 / 32.25 |

- Off at sigma 2.9, AV1 gets *worse* against the clean picture as it spends: 38.2 dB at the key, 36.7 dB at
  the end. That is coding noise, measured.
- The skip is better or equal at every sigma, at the same rate.
- The floor saves 92-95% of the rate. It gains at sigma 2.9 and 7 but loses 0.8-2 dB at sigma 1.15, where
  the reconstruction freezes near the key.
- Floor k at sigma 1.15, with the skip, kbps / dB:

  | k | AV1 | H.264 |
  |---|---|---|
  | 6 | 986 / 43.51 | 505 / 43.12 |
  | 8 | 500 / 42.55 | 207 / 42.22 |
  | 11 | 174 / 41.78 | 91 / 41.96 |
  | 16 | 54 / 42.78 | 50 / 41.77 |

  Medium floors code noise at a medium step and do worst.

### Still webcam (854x480 clip; clean = 17-frame mean)

| codec | off | noise skip c4 | skip + `max_quality_qp` 20 | skip + floor k16, cadence kept |
|---|---|---|---|---|
| AV1 | 988 / 44.77 | 988 / 45.34 | 858 / 45.19 | 668 / 45.04 |
| H.264 | 994 / 44.95 | 990 / 44.81 | 250 / 44.75 | 95 / 43.71 |
| HEVC | 987 / 45.26 | 987 / 45.70 | 521 / 45.32 | 277 / 44.95 |

For comparison, each codec uncapped at a 300 kbps target: AV1 296 / 43.35, H.264 298 / 44.66,
HEVC 296 / 44.60.

Floors without the skip:
- `max_quality_qp` 30 alone: AV1 316 kbps at 41.27 dB. That is below the uncapped curve (CBR 300 gives
  43.35), because every frame is clamped and no cadence base frame refines.
- With `rate.quality_floor_cadence` 1, k 32: 345 kbps at 43.73, on the curve.

### 60 s robustness, AV1, synthetic camera-sigma noise, key at 30 s

- **Ramp:** luma gain rises 25% over the minute.
- **Object:** a textured 96x96 patch enters at 20 s, rests, and leaves at 40 s.

Worst-frame PSNR-Y against clean in the last 4 s window, and mean kbps:

| | off | noise skip | skip + floor k16c |
|---|---|---|---|
| ramp | 990 kbps, 40.87 | 990 kbps, 40.92 | 62 kbps, 37.21 (falls steadily 42.2 → 37.2) |
| object | 990 kbps, 43.25 | 990 kbps, 43.24 | 45 kbps, 39.38 (42.8 before the object, 39.4 after it left) |

- The skip follows the slow ramp and the object exactly like the encoder without it. Its test is against the
  current source, so drift is coded once it exceeds the bound.
- The floor does not. Under a coarse floor, small corrections quantise to nothing: the ramp's error
  accumulates, and the background the object uncovered stays at the floor's quality until the next key.
- **The noise-relative floor is therefore not safe as a default.** Undershoot on still scenes needs a
  different mechanism (§6).

### CPU (x86 instructions, callgrind, webcam clip, 90 frames, 1000 kbps target)

| codec | off | noise skip c4 | skip + `max_quality_qp` 20 |
|---|---:|---:|---:|
| AV1 Default | 7.80 G | 6.43 G (−17.6%) | 6.44 G (−17.4%) |
| HEVC Default | 5.80 G | 4.45 G (−23.3%) | 4.02 G (−30.7%) |
| H.264 Default | 3.53 G | 2.83 G (−19.9%) | 2.12 G (−39.9%) |

- Skipped blocks cost no mode decision, transform, quantisation or entropy coding.
- With the levers off, the code adds one parameter test per block offered to the static skip.
- The identity streams (52/52) are byte-identical. Jetson cycles were not measured (the timing lock was not
  taken).

## 6. Recommendations and open points

**Adopt the noise skip** (`decision.noise_skip` 4, `decision.noise_skip_chroma_ratio` 2) into every
Default, subject to the tuner's L1/L2 screens. Reasons:
- It is a quality lever at the same rate, and better or equal against clean at every noise level measured.
- It follows a slow light ramp and an object passing exactly like the encoder without it.
- It saves 18-23% of encode instructions on a still camera.
- Handed over in docs/handoff/tuning.md.

**The noise-relative floor stays a setting, off.**
- It produces the undershoot the owner asked for: 5-7% of a 1 Mbps target on synthetic stills; 95-668 kbps
  instead of about 1000 on the webcam.
- It is not safe as a default. Under it, slow light changes accumulate (−5 dB over a minute) and an
  uncovered background stays coarse until the next key (−3.4 dB).
- It also loses 0.8-2 dB against clean on camera-level noise.
- A static-camera product that wants a hard spend reduction and has short key intervals can use k 16 with
  `rate.quality_floor_cadence` 1 and the noise skip, accepting those costs. Without the cadence flag, a floor
  is below the uncapped rate-quality curve.
- **The demo's webcam** should get the skip, not the floor.

**What undershoot needs instead (proposal, not built).** Rate control should stop re-spending the area the
noise skip takes. One way: scale a frame's budget by the share of the picture not noise-skipped in the
previous frame, and forgive the underspend, as held frames already are. The skip's per-block test would
still decide what is coded, so drift and objects are followed, while the frame quantizer stops falling.

**Fixed backstop** `rate.max_quality_qp` 10 (owner decision, every preset): the census shows it is never
reached on the benchmark ladder; it is handed to the tuner.

**VP9, VVC, AV2** have no static-skip path (`CodecBackend::try_static_skip` default). Each needs:
- a zero-vector skip it can code without a residual (VP9 `skip` with ZEROMV; VVC merge/skip with a zero
  candidate; AV2 as AV1);
- a `static_skip_vector` of zero;
- `try_static_skip` accepting `test.within_noise`.
The floors already apply to them. AV2's step mapping in `quality_cap.rs` is suspected to be about 8 times
too coarse (docs/handoff/rate-control.md).

**Separate findings for other agents** (not pursued here):
- *Under-refinement of a clean still.* A noise-free still at CBR 300 is coded once at the key (37.2 dB) and
  then held: every later frame is 0 bytes, with 280 kbps unused. At fixed QP the same still reaches 45.5 dB
  for 48 kbps.
- *AV1 Default against SVT-AV1 / libaom at 300 kbps on this content:* 43.4 dB against clean, where SVT M10
  `--rtc` gets 44.9 and libaom rt gets 45.6 (−1.5 to −2.3 dB). Our H.264 and HEVC get 44.6-44.7.
- *Static detection on webcams.* The analyzer's noise estimate reads clipped highlights, so pre-analysis
  classes 7% of a still webcam picture static. This also starves `still_skip` and `static_zero_vector`.

## 7. Skip-aware budget (`rate.skip_budget`, landed off)

**What it does** (`skip_budget.rs`):
- An inter frame's budget is scaled by max(floor, 1 − s × the share of the previous frame's area the noise
  skip took).
- Rate control drains only the scaled allowance from its debt, so the unspent part is forgiven, never banked.
- s follows the stream's bits per pixel (`bpp_curve.rs`, log-interpolated): `rate.skip_budget` (1) from
  0.15 bpp up, `rate.skip_budget_starved` (0.3) at 0.02 bpp and below. At a starved rate the skipped area's
  bits still buy refinement: at 300 kbps, H.264 loses 0.9 dB at s = 1 and 0.2 dB at s = 0.5.
- Floor `rate.skip_budget_floor` is 0.1.
- Keys reset the share.
- The skip already is the per-block "settled / unsettled" split: settled blocks get no residual, changing
  blocks are coded at the frame quantizer. So the budget follows the unsettled share, and recovering areas
  keep their bits.

**Webcam** (noise skip c 4, chroma ratio 2, skip budget on). Achieved kbps / PSNR-Y against the 17-frame
mean, uncapped in brackets:

| target | AV1 | HEVC | H.264 |
|---|---|---|---|
| 300 | 247 / 43.19 (296 / 43.35) | 186 / 44.61 (296 / 44.60) | 159 / 44.39 (298 / 44.66) |
| 1000 | 476 / 44.86 (988 / 44.77) | 274 / 45.05 (987 / 45.26) | 199 / 44.80 (994 / 44.95) |

**Synthetic stills at 1000 kbps** (s = 1). kbps / PSNR-Y against the clean original, off in brackets:

| sigma | AV1 | H.264 |
|---|---|---|
| 1.15 | 865 / 43.63 (987 / 43.54) | 395 / 43.75 (986 / 43.82) |
| 2.9 | 447 / 37.16 (986 / 36.93) | 109 / 38.47 (986 / 37.70) |
| 7 | 820 / 30.90 (987 / 30.83) | 509 / 32.46 (985 / 31.89) |

**60 s robustness at 1000 kbps** (AV1 / H.264, against clean, off in brackets):
- Ramp: last-window worst frame 40.87 / 40.84 (40.87 / 41.00).
- Object: the band it crosses is at 43.36 / 43.14 dB in the 2 s after it leaves (43.34 / 43.44 off), and
  back to 43.44 / 43.55 by 42-45 s. Nothing lingers to the next key.
- The moving region while the object moves: 43.39 / 43.33 (43.34 / 43.21).
- Mean rate: AV1 873-898, H.264 394-541 kbps.
- Per-second traces were reported with the work. They were measured with budget strength 1; the
  bits-per-pixel curve does not change it at 1000 kbps, which is 0.16 bpp.

**Rate-control audit still points** (long_static_chirla_s024c2 at 47/93/187 kbps, 600 f, key every 10 s,
against source):
- rate 0.956-1.049;
- PSNR-YUV −0.25 to +0.16 dB;
- worst-1% PSNR-Y unchanged or better.
Chirla is a clean, compressed source with little noise to skip.

**Why AV1 undershoots least** (webcam, 1000 kbps, instrumented):

| | AV1 | HEVC | H.264 |
|---|---|---|---|
| area noise-skipped | 52% | 73% | 80% |
| area tested and failed | 45% | 25% | 17% |
| failed area in 64x64 leaves | 77% | 21% | — |
| failed area whose 16x16 quarters pass | 25% | 12% | 0 |

AV1's partition planner gives still content 64x64 leaves, so one failing 8x8 cell fails the block. A "noise
split" (test 16x16 quarters, split where they disagree) is the next step.

**The clamped noise floor** (`rate.noise_floor_max_qp` 16 / `_starved` 30, bits-per-pixel curve):
- It passes the ramp and object tests: AV1 is identical to the skip alone, and H.264's ramp is better than
  off.
- It barely undershoots at 1 Mbps (AV1 988, HEVC 774, H.264 527 kbps).
- The skip-aware budget is the better route.
