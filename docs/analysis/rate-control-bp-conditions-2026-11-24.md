# Rate Control Re-Tuned Under Benchmark Conditions (Brief 21, 2026-11-24)

Branch `feature/gemini-rate-bp` (rebased onto `variance-integration` `14ffbfffd`).

All measurements were run under the benchmark's CBR conditions (`rate.first_key_lookahead_off=1`, `--strict-buffer` 1.0, peak = target, hold off, `rate.key_cap_buffer=1.0`, decided on benchmark BD-J), using Graviton4 encodes + PC GPU scoring (`--score auto`) and `c8i.2xlarge`/`c8g.2xlarge` CBR CPU A/B.

---

## 1. One-at-a-Time Quick Matrix (`nceval.quick`, draw `2026-11-24`, L1 `n=24` + Probe `1/150` & `1/300` `n=12`)

| Arm (`--param`) | AV1 `med` (L1 / Probe) | H.264 `med` (L1 / Probe) | HEVC `med` (L1 / Probe) | Guards / Notes |
|---|---|---|---|---|
| `rate.in_frame=1, in_frame_proactive=1` | **`-0.31%` / `-0.39%`** | **`-0.22%` / `-0.18%`** | **`-0.50%` / `-0.86%`** | **0 guards on all 3 codecs; faster CPU** |
| `rate.in_frame=1` (reactive only) | `-0.04%` / `-0.30%` | — | — | Beaten by proactive |
| `rate.key_repay_seconds=10` | `-0.75%` / `-0.48%` | `+0.33%` / `-0.31%` | `-0.91%` / `-0.49%` | 150 f ceiling guard (`+0.95–1.53%` over target) |
| `rate.key_repay_seconds=20` | `-0.38%` / `-0.63%` | `+0.77%` / `-0.30%` | `-0.26%` / `-0.22%` | Worse L1 & larger ceiling (`+1.32–2.09%`) |
| `rate.key_repay_still_seconds=20` | `-0.29%` / `-0.43%` | — | — | Ceiling `+0.68%` |
| `rate.debit_margin=0` | `-0.59%` / `-0.37%` | `-0.05%` / `+0.19%` | `n/a` / `-0.35%` | 0 guards; loses on H.264 |
| `rate.debit_credit=1.5` | `0.00%` / `-0.11%` | — | — | Neutral |
| `rate.recode_fill=0.95, exp=4, max=3` | `-0.14%` / `-0.34%` | `+0.07%` / `+0.07%` | `+0.31%` / `-0.16%` | Mixed |
| `rate.recode_early=1` | `-0.07%` / `+0.36%` | — | — | Worse on 1/150 (`+0.93%`) |
| `rate.level_hint_weight=0.5` | `-0.26%` / `-0.13%` | `-0.04%` / `+0.05%` | `+0.08%` / `-0.18%` | Small / mixed |
| `rate.level_step_scale=0.35` | `-0.56%` / `-0.42%` | `+0.12%` / `-0.10%` | `-0.27%` / `-0.02%` | Mixed |
| `rate.window_seconds=1.0, tol=0.10` | `-0.94%` / `-1.42%` | `-0.71%` / `-0.09%` | `-0.68%` / `-0.59%` | Bodycam class guard (`+3.5–4.1%`) |
| `rate.window_seconds=1.0, tol=0.20` | **`-0.81%` / `-1.01%`** | `-0.22%` / `+0.40%` | `-0.92%` / `-0.12%` | 0 guards at quick; failed AV1 600 f pulse |

### Carry-Over Quicks (Batch 4: `VVC`/`VP9`/`AV2 medium` + `AV1`/`H.264`/`HEVC fast`)
- `in_frame=1, in_frame_proactive=1`: `vp9:med` `-0.09% / -0.12%` (0 guards), `av2:med` `0.00%`, `av1:fast` `+0.46% / -0.37%` (0 guards), `hevc:fast` `+0.05% / -0.38%` (0 guards), `h264:fast` `+0.32% / -0.10%` (0 guards), `vvc:med` `+0.36% / -0.22%` (drone `+3.33%`).
- `win1_t20`: `av1:fast` `+0.08% / -0.33%` (0 guards), `hevc:fast` `0.00% / +1.06%` (worse).

---

## 2. Full L3 (`n=98`), 600-Frame Final Check (`1/150` & `1/300`, `n=55`), and `c8i`/`c8g` CBR CPU A/B

| Codec `medium` | Candidate | L3 (`n=98`) Quality J [90% CI] | 600 f FC (`n=55`) Quality J [90% CI] | CPU (`c8i` / `c8g`) | Net J (L3 / FC) | Guards / Verdict |
|---|---|---:|---:|---:|---:|---|
| **AV1** | **`in_frame=1, in_frame_proactive=1`** | **`-0.34% [-0.55, +0.06]`** | **`-0.20% [-0.33, +0.15]`** | **`-0.78% / -0.88%`** (`-0.83%`) | **`-0.46% / -0.32%`** | **0 guards — ADOPTED (`AV1::DEFAULT_PARAMS`)** |
| **AV1** | `key_repay=10, in_frame=1, proactive=1` | `-0.93% [-1.24, -0.07]` | `-0.38% [-0.58, -0.01]` | `+1.97% / +1.63%` (`+1.80%`) | `-0.68% / -0.13%` | 150 f ceiling guard (`+1.60%`); worse net FC than `in_frame` alone |
| **AV1** | `window_seconds=1.0, tol=0.20` | `-0.54% [-0.80, +0.02]` | `-0.33% [-0.51, +0.09]` (`n=31`) | — | — | **REJECTED** (worst-clip pulse `+1.40 dB > +0.50`) |
| **HEVC** | `in_frame=1, in_frame_proactive=1` | `+0.16% [-0.24, +0.36]` | `-1.05% [-1.21, +0.28]` (`n=31`) | `-0.03% / -0.46%` | `+0.13% / —` | **REJECTED** (L3 `+0.16%`, worst-clip pulse `+0.73 dB > +0.50`) |
| **HEVC** | `key_repay=10, in_frame=1, proactive=1` | `+0.30% [-0.19, +0.61]` | `-0.71% [-0.86, -0.06]` | `+2.96% / +2.34%` | `+0.58% / -0.43%` | **REJECTED** (L3 `+0.58%` net + ceiling `+1.34%`) |
| **H.264** | `in_frame=1, in_frame_proactive=1` | `+0.04% [-0.08, +0.12]` | `-0.14% [-0.24, +0.04]` | `+0.01% / -0.04%` | `+0.04% / -0.14%` | **REJECTED** (L3 `+0.04%` + pulse `+0.25 dB`) |

**Codec reason (AV1-only):** Under `--strict-buffer`, proactive in-frame steering eliminates whole-frame re-codes on AV1 medium (`-0.83%` CPU, `-0.46%` L3 net J, `-0.32%` 600 f FC net J, 0 guards), whereas HEVC and H.264 medium are slightly worse at L3 and trip periodic-key pulse guards.

---

## 3. AWS Spend (All `eu-north-1` Spot, Cap `$1.00`)

- **Batch 1** (10 AV1 `medium` quicks, `c8g.8xlarge` + PC GPU): **`$0.030`**
- **Batch 2** (18 H.264/HEVC/AV1 `medium` quicks, `c8g.8xlarge` + PC GPU): **`$0.071`**
- **Batch 3** (12 combo + `win1_t20` quicks, `c8g.12xlarge` + PC GPU): **`$0.037`**
- **Batch 4** (14 carry-over quicks, `c8g.12xlarge` CPU scoring): **`$0.120`**
- **CBR CPU A/B** (5 pairs, `c8i.2xlarge` + `c8g.2xlarge` ~0.10 h each): **`$0.020`**
- **Batch 5** (12 full L3 `n=98` + 600 f FC `n=55` screens, `c8g.12xlarge`/`c8g.8xlarge` + PC GPU): **`$0.148`**
- **Total Brief 21 AWS spend:** **`$0.43`** (under `$1.00` cap; 0 on-demand; 0 region spills; 0 boxes left running).
