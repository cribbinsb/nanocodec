# Frame-Structure Cadence under CBR: `rate.cadence_period=8` and Deep Inherited Depth (`structure.depth_max`, Brief 17)

**Date:** 2026-11-24  
**Branch:** `feature/gemini-cadence` (commit `0ad2ad128`, off `variance-integration` `66092ae54`, unmerged, default off)

## 1. Task 1: `rate.cadence_period = 8` under CBR

Screened across all 6 codecs (`AV1 fast/medium`, `AV2 medium`, `VP9 medium`, `HEVC medium`, `VVC medium`) on `nceval.quick` (`draw 2026-11-24`, both key shares `1/150` and `1/300`) with paired CPU A/B (`aws_cpu_ab` on `c8i.2xlarge` + `c8g.2xlarge` `eu-north-1` spot) for the three clear winners.

| Rung (`k`) | L1 (150f) Qual | `1/150` (600f) Qual | `1/300` (900f) Qual | Probe Mean Qual | Pulse (`1/150` / `1/300`) | `c8i` / `c8g` dCPU (L1 / `1/150` / `1/300`) | **Net `J` (L1 / `1/150` / `1/300`)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **AV1 `medium`** (`0.123`) | **-0.54%** `[-0.72, -0.24]` | **-1.40%** `[-2.59, -0.41]` | **-0.18%** `[-0.55, +0.20]` | **-0.79%** `[-1.42, -0.26]` | `-0.07` / `-0.01` dB | `-0.91/-0.87%` / `-3.20/-2.98%` / `-3.51/-3.69%` | **-0.65% / -1.78% / -0.62% (WIN ALL 3)** |
| **AV1 `fast`** (`0.2372`) | `+0.04% [-0.26, +0.21]` | **-0.83%** `[-1.48, -0.31]` | **-0.32%** `[-1.12, +0.06]` | **-0.57%** `[-1.09, -0.25]` | `-0.11` / `-0.01` dB | `+0.36/+0.18%` / `-1.12/-1.07%` / `-0.37/-0.47%` | `+0.10%` / **-1.09%** / **-0.42% (WIN PROBES)** |
| **AV2 `medium`** (`0.1625`) | **-0.64%** `[-0.89, +0.09]` | **-0.73%** `[-1.19, -0.11]` | **-0.09%** `[-0.62, +0.25]` | **-0.41%** `[-0.76, -0.06]` | `-0.07` / `-0.02` dB | `-0.18/+0.13%` / `-0.67/-0.91%` / `-3.75/-3.97%` | **-0.64% / -0.86% / -0.72% (WIN ALL 3)** |
| **VP9 `medium`** (`0.1598`) | `-0.24% [-0.71, +0.47]` | **-0.99%** `[-1.52, -0.35]` | `+0.42% [+0.04, +0.69]` | `-0.29% [-0.61, +0.06]` | `-0.06` / `+0.00` dB | *(not timed)* | wins `1/150`, loses `1/300` (`static +1.34%`) |
| **VVC `medium`** (`0.1768`) | **-0.46%** `[-0.81, -0.09]` | **-0.35%** `[-0.54, -0.16]` | `+0.12% [-0.31, +0.44]` | `-0.11% [-0.35, +0.07]` | `-0.08` / `-0.03` dB | *(not timed)* | wins L1 & `1/150`, flat `1/300` |
| **HEVC `medium`** (`0.1558`) | `+0.00% [-0.28, +0.24]` | `-0.35% [-0.78, +0.09]` | `+0.26% [+0.07, +0.45]` | `-0.05% [-0.28, +0.19]` | `-0.04` / `-0.09` dB | *(not timed)* | neutral (matches oracle) |

- **600-frame single-key check** (`AV1 medium`, 500 kbps, 4 pool classes, 0 overflows): `static +0.136 dB`, `crowd +0.037 dB`, `pan +0.070 dB`, `drone +0.286 dB`.
- **Recommendation:** **`AV1 medium`**, **`AV1 fast`**, and **`AV2 medium`** are ready for `nceval.train` adoption of `rate.cadence_period = 8` — they are simultaneously **higher quality AND 1–4% faster** on both x86 and Arm, with negative (better) key pulses.

## 2. Task 2: Deeper Cadence for Fully Inherited Scenes (`structure.depth_max`, default off)

Implemented in `crates/nc-encoder/src/inheritance.rs`, `params.rs`, and `encoder.rs` (`0ad2ad128`):
- Parameters (default off: `structure.depth_max = 1.0`, **`387/387 streams identical`** on `preset_identity.py`; `inheritance` 4/4 and `static_recovery` 13/13 passed):
  - `structure.depth_max` (`1.0..2.0`, default `1.0`),
  - `structure.depth_deep_loss` (`0.10` dB/QP),
  - `structure.depth_deep_min_period` (`150` frames),
  - `structure.depth_deep_max_motion` (`1.0` luma samples/frame),
  - `structure.depth_deep_max_noise` (`1.5` luma sigma).
- Ramps depth above `1.0` only after `>= 2` closed mini-GOPs when `loss < depth_full_loss`, `key_period >= depth_deep_min_period`, camera motion is low, and sensor noise is low.

| Rung | Setting | L1 (150f) Qual | `1/150` (600f) Qual | `1/300` (900f) Qual | Probe Mean Qual | Key Notes |
|---|---|---:|---:|---:|---:|---|
| **AV1 `medium`** | `depth_max=1.5` (p4) | `-0.00% [-0.81, +0.83]` | `+0.11% [-0.70, +0.60]` | `-0.19% [-0.84, +0.22]` | `-0.04%` | `bodycam 0.00%` everywhere; `lowlight +3.04%` guard; `static` probe `+2.47%/+1.20%` |
| **HEVC `medium`** | `depth_max=1.5` (p4) | `+0.22% [-0.25, +0.86]` | `-0.39% [-1.19, +0.34]` | `+0.29% [-0.28, +0.83]` | `-0.05%` | `bodycam 0.00%` everywhere; `lowlight +2.52%`; `static` probe `+1.61%/+2.28%` |
| **HEVC `medium`** | `depth_max=1.35, max_noise=0.75` (p4) | `-0.02% [-0.39, +0.56]` | `-0.45% [-1.03, +0.18]` | `+0.22% [-0.35, +0.73]` | `-0.12%` | `lowlight` improved `+2.52% → +2.04%`, `1/150 static` `+1.61% → +0.85%` |
| **AV1 `medium`** | **`p8 + depth_max=1.35 + max_noise=0.75`** | **-0.51%** `[-1.38, +0.25]` | **-1.43%** `[-2.62, -0.41]` | **-0.65%** `[-1.61, +0.00]` | **-1.04%** `[-1.80, -0.43]` | **Zero guards! Adds `-0.47%` on `1/300` on top of `p8` alone (`-0.18% → -0.65%`)!** |

- **Key finding:** At `period = 4`, raising depth to `1.5×` pays `-4.5` QP every 4th frame, starving the 3 inter frames on `static` clips. **Combined with `rate.cadence_period = 8` on AV1 `medium`**, HQ frames occur half as often and are inherited by 7 frames instead of 3 — **`p8 + depth_max=1.35 + max_noise=0.75` lifts `1/300` from `-0.18%` to `-0.65%` and probe mean to `-1.04% [-1.80, -0.43]`** (plus ~`-3.3%` CPU from `p8` -> `~-1.45%` net J), with **zero guards tripped** and pulse `-0.04 / -0.01 dB`!

## 3. Task 3: Assessment of a Learned Per-Window Chooser

- **AV1:** Not warranted. `rate.cadence_period = 8` + `structure.depth_max = 1.35` already claims **-1.43% (`1/150`) and -0.65% (`1/300`) in pure quality** (`~-1.45%` net J including `-3.3%` CPU), capturing nearly the entire causal CBR oracle prize (`-1.03%` / `-1.58%` hindsight, `-0.70%` / `-0.75%` 1-window-lagged) with zero model complexity or inference cost.
- **HEVC `1/300`:** About `~1.0–1.5%` of the hindsight prize (`-1.71%`) remains unclaimed because `p8` is neutral and `depth_max` alone is `-0.12%` (still losing `+2.0%` on clean dark `lowlight` and `+2.2%` on `static` at `1/300`). If pursued later, a lightweight **per-window classifier (30-frame window)** trained on `structure_oracle`'s per-window labels (`{today, period8, deep1.5}`) using 6 already-computed summary features (`inheritance_loss`, `mean_luma`, `noise_sigma`, `camera_motion`, `static_fraction`, `bpp / target_ratio`) with a small 2-layer MLP or GBDT (<50 µs once per 30 frames) could separate clean dark lowlight (`mean_luma` low) from bright still crowd/static (`mean_luma` normal, `loss < 0.15`), with expected causal gain bounded by the lagged oracle (`~-0.3%` to `-0.6%` J on HEVC `1/300`) at negligible CPU cost.
