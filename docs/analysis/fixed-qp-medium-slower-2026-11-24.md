# Fixed-QP Medium & Slower Rungs: H.264, HEVC, VVC (Brief 20, 2026-11-24)

Branch `feature/gemini-fixed-qp` (rebased onto `variance-integration` `c9aade2cf`).

## 1. BP-3.11 Fixed-QP Front Slopes (`ours`, `c8i`)

| Codec | `medium` slope (to `slow`) | `slow` slope (to `slower`) | `slower` slope |
|---|---:|---:|---:|
| **H.264** | `-0.133` | `-0.010` | `-0.010` |
| **HEVC** | `-0.083` | `-0.048` | `-0.058` |
| **VVC** | `-0.088` | `-0.072` | `-0.052` |

---

## 2. Pool Screen Results (`nceval.fixedqp ab`, draw `fixedqp-2026-11-24`, `n=24`)

### Key-Frame QP Offset, Cadence Pattern, Structure, and Low-QP Gates (`medium`, 150 f)

| Codec `medium` | Candidate | 150 f J [95% CI] | Static / VC | Moving | Finding |
|---|---|---:|---:|---:|---|
| **H.264** | `h264.keyframe_qp_offset=4` | `+6.52% [+4.5, +7.4]` | `+7.99%` | `+4.30%` | Much worse (coarser key hurts) |
| | `h264.keyframe_qp_offset=6` | `+2.48%` | `+2.79%` | `+1.74%` | Worse |
| | **`h264.keyframe_qp_offset=10`** | **`-1.09% [-1.4, -0.6]`** | **`-1.84%`** | **`-0.56%`** | **Clear winner (optimum)** |
| | **`h264.keyframe_qp_offset=10, moving_key=8`** | **`-1.10% [-1.4, -0.7]`** | **`-1.65%`** | **`-0.58%`** | **Best (tighter CI, protects moving)** |
| | `h264.keyframe_qp_offset=12` | `-0.49% [-1.0, +0.4]` | `-1.40%` | `+0.08%` | Overshoots on moving |
| | `rate.fixed_qp_moving_key=10` | `+0.12% [-0.0, +0.3]` | `+0.59%` | `+0.08%` | Neutral/worse alone |
| | `cadence_startup_period=0, base_boost=4` | `+1.07% [+0.6, +1.3]` | `+0.71%` | `+0.99%` | Startup cadence helps |
| | `cadence_static_scaling=0` / `p8_b5` / `ref_sec` / `skip_no_refine=0` | `+0.12%` / `-0.09%` / `0.00%` / `0.00%` | — | — | Neutral |
| **HEVC** | `hevc.key_qp_offset_delta=-6` / `-3` | `+6.47%` / `+1.63%` | `+8.17%` / `+2.43%` | `+4.27%` / `+0.81%` | Coarser key hurts |
| | `hevc.key_qp_offset_delta=+3` (+ `mk=9`) | `+1.88%` / `+1.77%` | `+0.50%` / `+0.25%` | `+1.94%` / `+1.83%` | Finer key also hurts (`0` is exact optimum) |
| | `rate.fixed_qp_moving_key=10` | `0.00% [-0.1, +0.1]` | `-0.15%` | `+0.04%` | Already optimal at `6` |
| | `cadence_leaf_penalty=2` / `base_boost=4` / `p8_b6` / `nostart_b4_l2` | `+0.22%` / `+0.38%` / `+0.60%` / `+2.09%` | — | — | All worse |
| | `base_lambda_scale=0.8` / `pre_transform_zero_scale=1` / `secondary_min_depth=0` / `prop_flat=1` | `+0.11%` / `-0.05%` / `0.00%` / `0.00%` | — | — | Neutral (`propagation_flat` off at fixed QP) |
| **VVC** | `vvc.key_qp_offset_delta=-6` / `-3` | `+7.59%` / `+2.54%` | `+7.36%` / `+2.58%` | `+5.58%` / `+1.65%` | Coarser key hurts |
| | `vvc.key_qp_offset_delta=+3` (+ `mk=9`) | `+0.03%` / `+0.10%` | `-0.90%` / `-0.37%` | `+0.54%` / `+0.42%` | Already optimal at `0` (`9.0` at p60 / `12.0` at p300) |
| | `rate.fixed_qp_moving_key=10` | `+0.35% [-0.0, +0.8]` | `+0.88%` | `+0.19%` | Worse |
| | `cadence_leaf_penalty=2` / `base_boost=4` / `p8_b6` / `nostart_b4_l2` | `+0.11%` / `+0.30%` / `+0.49%` / `+2.21%` | — | — | All worse |
| | `vvc.pre_transform_zero_scale=1` | `+0.13% [-0.0, +0.2]` | `+0.10%` | `+0.08%` | Neutral |
| | `decision.base_lambda_scale=0.8` | `-0.18% [-0.4, -0.0]` | `-0.01%` | `-0.24%` | Small free win |
| | **`refs.second_full=1, secondary_min_depth=0`** | **`-0.42% [-0.8, -0.2]`** | **`-0.40%`** | **`-0.57%`** | **Clear winner (all metrics & classes)** |

### 600-Frame Confirmation & Fixed-QP CPU A/B (`c8i` / `c8g`)

| Rung (Front Slope) | Candidate | 600 f J [95% CI] | Static / VC | Moving | CPU (`c8i` / `c8g`) | Mean CPU | Slope (`%J / 1% CPU`) | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **H.264 `medium`** (`-0.133`) | `h264.keyframe_qp_offset=10` | `-0.44% [-0.9, +0.2]` | `+0.07%` | `-0.18%` | `-0.13% / +0.02%` | `-0.06%` | ∞ (free) | Superseded by `mk=8` |
| | **`h264.keyframe_qp_offset=10, moving_key=8`** | **`-0.53% [-0.9, +0.0]`** | **`-0.09%`** | **`-0.29%`** | **`-0.08% / -0.05%`** | **`-0.07%`** | **∞ (free)** | **Committed (`H264::FIXED_QP_PARAMS`)** |
| **H.264 `slow`** (`-0.010`) | `h264.keyframe_qp_offset=10` | `-0.42%` | `-0.29%` | `-0.16%` | `+0.15% / -0.38%` | `-0.12%` | **∞ (free)** | **Inherited (`H264::FIXED_QP_PARAMS`)** |
| **VVC `medium`** (`-0.088`) | **`refs.second_full=1, secondary_min_depth=0`** | **`-0.37% [-0.6, -0.2]`** | **`-0.38%`** | **`-0.43%`** | **`+1.10% / +1.32%`** | **`+1.21%`** | **`-0.31%`** | **Committed (`VVC::FIXED_QP_PARAMS`, 3.5×)** |
| | `above + decision.base_lambda_scale=0.8` | `-0.36% [-0.6, -0.2]` | `+0.00%` | `-0.57%` | `+1.75% / +1.60%` | `+1.68%` | `-0.21%` | Not taken on `medium` (VMAF `+0.3`, static `+0.6`) |
| **VVC `slow`** (`-0.072`) | **`above + decision.base_lambda_scale=0.8`** | **`-0.51% [-0.8, -0.3]`** | **`-0.94%`** | **`-0.54%`** | **`+1.64% / +1.36%`** | **`+1.50%`** | **`-0.34%`** | **Committed (`vvc/slow.toml`, 4.7×)** |

**HEVC `medium`/`slow`/`slower`:** No change — already at sharp optimum (`KEY_QP_OFFSET_AT_REFERENCE = 12.0`, `fixed_qp_moving_key = 6`, `cadence_static_scaling = 0`, `refs.second_full = 1`, startup cadence `p8 / b6`).

---

## 3. AWS Spend (All `eu-north-1` Spot, Cap `$0.60`)

- **Round 1 (150 f, 3 codecs, `i-0e07ab6bfef8b625b`, `c6i.16xlarge` `--cold`, 20.7 min @ `$0.3765/h`):** **`$0.13`** (terminated on exit).
- **Round 2 (150 f, 3 codecs, `i-0c784338bff550430`, `c6i.24xlarge` `--cold`, 13.1 min @ `$0.4531/h`):** **`$0.10`** (terminated on exit).
- **Round 3 (600 f, 4 rungs, `c6i.16xlarge` `--cold`):** attempt 1 reclaimed (`i-087b0d8d1f21360a3`, ~10 min @ `$0.3765/h` `~$0.06`) + attempt 2 (`i-070f74e2cd8af4fd0`, 22.2 min @ `$0.3540/h` `$0.14`) = **`$0.20`** (terminated on exit).
- **Fixed-QP CPU A/B (6 pairs, `c8i.2xlarge` 0.16 h @ `$0.0579/h` + `c8g.2xlarge` 0.18 h @ `$0.0648/h`):** **`$0.02`** (terminated on exit).
- **Total Brief 20 AWS spend:** **`$0.45`** (under `$0.60` cap; 0 on-demand; 0 region spills; 0 boxes left running).
