# First-Key Size Prediction Model (`rate.first_key_model`, Brief 15)

**Date:** 2026-11-24  
**Branch:** `feature/gemini-first-key` (commits `70f720c42`, `20b7c76f2`; off `5a617ea59`, unmerged, default off)

## 1. Root Cause & Step 1 Census (184 Unsealed Pool Families, 8,256 Encodes)

`RateController::bootstrap_step` (`crates/nc-encoder/src/rate_control.rs`) predicts frame-0 key bits from a single scalar (`hint.spatial` = sum of `1 + sqrt(sum_sq_dev)` over 8×8 blocks, clamped to `[0.25, 4.0]`) and a single `bootstrap_bits_per_pixel` per quantizer scale. It has two structural flaws:

1. **Block variance ignores spatial correlation and deadzone collapse:** smooth ramps have high variance yet predict cheaply via intra prediction, while fine texture/foliage/noise has the same variance and costs 2–3.6× more bits.
2. **Uncapped `rate.key_still_boost` on frame 0:** `key_step()` applies `budget_step * finer` (up to 4 QP finer on still cameras) without clamping against `strict_room(true)` because `self.model[KEY_MODEL]` is `None` before frame 0 is coded.

Measured across all **184 unique unsealed pool families** (never sealed sets) at the benchmark ladder (`300/500/800/1200` kbps at 720p30 scaled by `(w*h*fps / (1280*720*30))^0.75`, capped at 50% source bitrate, 688 encodes per `(codec, preset)`):

| Codec | Preset | Baseline Recodes | Baseline LogRMSE | Baseline p90 / Max (1st/Pred) | `model=1` Recodes | `model=1` LogRMSE | `model=2` Recodes | `model=2` LogRMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **AV1** | `fast` | 94 / 688 (13.7%) | 0.454 | 1.96× / 3.62× | **3 / 688 (0.4%)** | 0.143 | **13 / 688 (1.9%)** | **0.135** |
| **AV1** | `medium` | 69 / 688 (10.0%) | 0.463 | 1.73× / 3.28× | **8 / 688 (1.2%)** | 0.175 | **20 / 688 (2.9%)** | **0.161** |
| **AV2** | `fast` | 37 / 688 (5.4%) | 0.440 | 1.38× / 2.40× | **0 / 688 (0.0%)** | 0.127 | **10 / 688 (1.5%)** | **0.126** |
| **AV2** | `medium` | 52 / 688 (7.6%) | 0.523 | 1.47× / 2.79× | **1 / 688 (0.1%)** | 0.149 | **13 / 688 (1.9%)** | **0.146** |
| **H.264** | `fast` | 73 / 688 (10.6%) | 0.403 | 1.72× / 2.77× | **3 / 688 (0.4%)** | 0.142 | **10 / 688 (1.5%)** | **0.139** |
| **H.264** | `medium` | 62 / 688 (9.0%) | 0.403 | 1.72× / 2.77× | **0 / 688 (0.0%)** | 0.141 | **10 / 688 (1.5%)** | **0.139** |
| **HEVC** | `fast` | 3 / 688 (0.4%) | 0.486 | 1.23× / 2.06× | **0 / 688 (0.0%)** | 0.146 | **2 / 688 (0.3%)** | **0.131** |
| **HEVC** | `medium` | 14 / 688 (2.0%) | 0.488 | 1.21× / 2.11× | **3 / 688 (0.4%)** | 0.153 | **2 / 688 (0.3%)** | **0.139** |
| **VP9** | `fast` | 8 / 688 (1.2%) | 0.352 | 1.30× / 1.69× | **0 / 688 (0.0%)** | 0.145 | **0 / 688 (0.0%)** | **0.130** |
| **VP9** | `medium` | 76 / 688 (11.0%) | 0.447 | 1.89× / 3.49× | **4 / 688 (0.6%)** | 0.167 | **16 / 688 (2.3%)** | **0.144** |
| **VVC** | `fast` | 54 / 688 (7.8%) | 0.515 | 1.56× / 2.73× | **3 / 688 (0.4%)** | 0.150 | **11 / 688 (1.6%)** | **0.136** |
| **VVC** | `medium` | 51 / 688 (7.4%) | 0.530 | 1.47× / 2.73× | **2 / 688 (0.3%)** | 0.152 | **8 / 688 (1.2%)** | **0.140** |
| **Total** | | **593 / 8,256 (7.2%)** | | | **27 / 8,256 (0.3%)** | | **115 / 8,256 (1.4%)** | |

## 2. Shared-Engine Model (`crates/nc-encoder/src/first_key_model.rs`)

Once per stream (frame 0 only, when `rate.first_key_model > 0`), `FirstKeyFeatures::extract` computes 32-bin log-spaced histograms (`0.25..256.0`, 512 bytes total, <0.2 ms) of:
- 8×8 block pixel gradient `|dx| + |dy|`,
- 8×8 block 2D Laplacian `0.25 * |4p - N - S - E - W|` (separates fine texture/noise from linear ramps),
- 2× downsampled 4×4 block gradient (`y2`) and 4× downsampled 2×2 block gradient (`y4`).

For any orthonormal sample quantizer step `Q`, it evaluates the deadzone survival curve `mean_blocks log(1 + 2 * energy / Q)` and `mean_blocks log(1 + (2 * energy / Q)^2)` in ~200 ns, with per-codec calibrated 9-weight linear combination in log-bits space plus a `fast_bias` when `quant.trellis == 0 || quant.trellis_key == 0` (`+0.136` AV1, `+0.116` AV2, `~0` H.264/HEVC). 5-fold family-CV log-RMSE is **0.141–0.157** across all 6 codecs and strictly monotonic in `Q` on **184/184** pool clips.

Two modes are exposed (default off: `rate.first_key_model = 0`, `rate.first_key_room_aim = 0.85`):
- `rate.first_key_model = 1`: replaces `bootstrap_step` for both `budget` and `strict_room` clamping.
- `rate.first_key_model = 2` (**overflow-guard**): keeps `bootstrap_step` for non-overflowing keys and clamps **only** first keys predicted to exceed `first_key_room_aim * strict_room(true)`.

## 3. First-Key Quality vs Today and vs Freed-Band In-Frame Steering

Measured on 28 stratified pool clips × 4 ladder rungs (104 points per `(codec, preset)`):

| Codec / Preset | Mode | All-Points Recodes | All-Points dPSNR-Y | Baseline-Recoded Points Recodes | Baseline-Recoded dPSNR-Y |
|---|---|---:|---:|---:|---:|
| **AV1 `fast`** | `freed_band` (`correct_qp=12`) | 0 / 104 | -0.133 dB | 0 / 16 | -0.723 dB |
| | `first_key_model=2` | 5 / 104 | **-0.056 dB** | 5 / 16 | **-0.118 dB** (**+0.605 dB vs freed-band**) |
| | `combined` (`model=2 + proactive`) | 2 / 104 | -0.074 dB | 2 / 16 | **-0.218 dB** (+0.505 dB vs freed-band) |
| **AV1 `medium`** | `freed_band` | 0 / 104 | -0.098 dB | 0 / 14 | -0.668 dB |
| | `first_key_model=2` | 5 / 104 | **-0.030 dB** | 5 / 14 | **+0.040 dB** (**+0.708 dB vs freed-band**) |
| | `combined` | 2 / 104 | -0.042 dB | 2 / 14 | **-0.014 dB** (+0.654 dB vs freed-band) |
| **H.264 `fast`** | `freed_band` | 0 / 104 | -0.060 dB | 0 / 14 | -0.372 dB |
| | `first_key_model=2` | 4 / 104 | **-0.056 dB** | 4 / 14 | **-0.211 dB** (+0.161 dB vs freed-band) |
| | `combined` | **0 / 104** | -0.069 dB | **0 / 14** | -0.261 dB (+0.111 dB vs freed-band) |
| **VP9 `medium`** | `freed_band` | 15 / 104 | -0.001 dB | 15 / 15 | -0.005 dB (steering inactive) |
| | `first_key_model=2` | 4 / 104 | **-0.017 dB** | 4 / 15 | **-0.069 dB** |

## 4. `nceval.quick` + `aws_cpu_ab.py` (`zen5`) on AV1 `fast` (`k = 0.2372`, draw `2026-11-24`)

- **`rate.first_key_model=1`** (full replacement, clamped to `0.85 * key_allowance`): **KILL** — L1 `J +1.71% [+0.56, +2.51]` (`drone +6.02%`, `texture +3.93%`, `lowlight -1.06%`); probe `1/150` `J +1.11%`, `1/300` `J +0.98%`; `zen5` cycles `-1.34%` (L1), `-0.76%` (`1/150`), `+0.99%` (`1/300`). Diagnosed: replacing `bootstrap_step` on non-overflowing `drone`/`texture` clips shrank keys that previously sat below `1.15 * capacity`, hurting all 149 subsequent inter frames.
- **`rate.first_key_model=2`** (overflow-guard, clamped to `0.85 * strict_room`): **CONTINUE** —
  - **L1 (24 clips, 150f):** `J +0.23% [-0.07, +0.54]` (zero guards tripped; worst class `texture +2.35%`); `zen5` CPU **`-1.51% [-1.69, -1.33]`** (`-1.22%` instructions) -> **net `J = -0.13%` (BETTER)**.
  - **Probe `1/150` (600f):** `J +1.25% [+0.24, +1.67]`, pulse **`-0.02 dB`** (worst `+0.17`); `zen5` CPU **`-0.21% [-0.48, +0.05]`**.
  - **Probe `1/300` (900f):** `J +0.67% [+0.48, +1.15]`, pulse **`+0.01 dB`** (worst `+0.30`); `zen5` CPU `+1.79% [+1.50, +2.09]`.
  - Cloud job CPU: `349 s → 319 s` (-8.6%).

## 5. Correctness, Identity & Carry-Over

- **Byte-identical while off:** `tools/dev/preset_identity.py` -> **`387/387 streams identical`** against base `5a617ea59`.
- **Unit & integration tests:** `first_key_model` (1/1), `rate_control` (42/42), `crates/nc-cli/tests/static_recovery.rs` (**13/13 passed**).
- **Long single-key run (600f, `--intra-period 600`, AV1 `fast` 500 kbps):** 0 first-key re-codes, 0 overflows, 600f rate error `-1.0%` to `-2.6%`, `max_buf = 0.215–0.674`, frame-0 prediction error cut from up to `2.51×` to `2–13%`.
- **Per-codec carry-over:** Applies cleanly to **all 6 codecs** (`AV1`, `AV2`, `H.264`, `HEVC`, `VP9`, `VVC`) via `RateModel::first_key`; largest re-code reductions on **AV1** (`94 → 13` fast, `69 → 20` medium), **VP9** (`76 → 16` medium — where in-frame steering cannot act), **H.264** (`73 → 10` fast, `62 → 10` medium), **VVC** (`54 → 11` fast, `51 → 8` medium), **AV2** (`37 → 10` fast, `52 → 13` medium), and **HEVC** (`14 → 2` medium).

## 6. Brief 16 Addendum: Fixing the Long-Window Probe Loss (`rate.first_key_trigger = 1.10`, `room_aim = 0.92`)

**Commit:** `7a6ec8e8c` (rebased onto `variance-integration` `d51e02c3f`, unmerged, default off; `387/387 streams identical` on `preset_identity.py`).

### Diagnosis
100% of `mode=2`'s Brief 15 probe loss (`1/150: +1.25%`, `1/300: +0.67%`) came from 3 clips (`bodycam`, `lowlight`, `pan` were all `0.00%`):
1. **False positives on non-overflowing keys:** `mode=2` used `first_key_room_aim = 0.85` as both trigger and target, clamping **29/594 (4.9%)** non-overflowing AV1 `fast` keys and **22/615 (3.6%)** H.264 `fast` keys. On `pool_static_chirla` (156 kbps, `strict_room = 179.4k`), baseline never overflowed (`162.7k`, `codings=1`), yet `mode=2` raised QP `82 → 102` (`-1.68 dB` on frame 0, `-1.07 dB` on frames 1–29, `+4.30%` BD-rate loss).
2. **Aiming too far under `strict_room`:** On genuine 3-coding overflows (`pool_drone_rounda`, `pool_chad_4_109`), `room_aim = 0.85` landed 10–13% below `overflow_recode`'s `0.90 * strict_room` target.

### Fix
Added `rate.first_key_trigger = 1.10` (in `mode=2`, clamp only when predicted first-key bits exceed `1.10 * strict_room`) and raised `rate.first_key_room_aim` from `0.85` to `0.92`.
- **184-family census:** False positives dropped **10×** on AV1 `fast` (`29 → 3/594`, **0.5%**) and **7×** on H.264 `fast` (`22 → 3/615`, **0.5%**), while eliminating **-61.7%** of extra first-key codings on AV1 `fast` (`180 → 69`) and **-59.8%** on H.264 `fast` (`117 → 47`).
- **`pool_static_chirla` (156k):** false positive eliminated (`0.00 dB` loss). `pool_chad_4_109` & `pool_drone_rounda`: `codings 3 → 1`, `0.00` to `+0.01 dB` over 300f (`+0.33/+0.49 dB` on frame 0).

### Dev-Procedure Results (`nceval.quick` + `aws_cpu_ab` on `c8i.2xlarge` / `c8g.2xlarge` `eu-north-1` spot, draw `2026-11-24`)
| Rung (`k`) | Window | Quality `J` | `c8i.2xlarge` cycles | `c8g.2xlarge` cycles | Mean dCPU | **Net `J` (`qual + k·dCPU`)** | Pulse |
|---|---|---:|---:|---:|---:|---:|---:|
| **AV1 `fast`** (`0.2372`) | **L1** (24 clips, 150f) | **-0.02%** `[-0.07, +0.09]` | **-0.77%** | **-1.03%** | **-0.90%** | **-0.23% (WIN)** | — |
| | **Probe `1/150`** (600f) | **-0.04%** `[-0.06, +0.00]` | **-0.40%** | **-0.53%** | **-0.47%** | **-0.15% (WIN)** | `-0.01 dB` |
| | **Probe `1/300`** (900f) | `+0.55% [+0.00, +0.91]` | **-0.16%** | **-0.07%** | **-0.12%** | `+0.52%` | `+0.03 dB` |
| **H.264 `fast`** (`0.2253`) | **L1** (24 clips, 150f) | `+0.21% [-0.44, +0.50]` | `+0.77%` | `+0.87%` | `+0.82%` | `+0.39%` | — |
| | **Probe `1/150`** (600f) | **`+0.00%`** `[-0.00, +0.00]` | **-0.90%** | **-0.81%** | **-0.86%** | **-0.19% (WIN)** | `+0.00 dB` |
| | **Probe `1/300`** (900f) | **`-0.00%`** `[-0.00, +0.00]` | `+0.13%` | **-0.13%** | `0.00%` | **`0.00%` (NEUTRAL)** | `+0.00 dB` |

Both rungs returned **`CONTINUE`** on `nceval.quick`. On AV1 `fast`, L1 (`-0.23%`) and `1/150` (`-0.15%`) are clear wins; `1/300` (`+0.52%`) is held back solely by one marginal-overflow clip (`pool_static_chirla` 93k). On H.264 `fast`, `1/150` (`-0.19%`) wins and `1/300` (`0.00%`) is neutral, while L1 has one `drone` clip (`+4.02%`) pushing L1 net to `+0.39%`.

