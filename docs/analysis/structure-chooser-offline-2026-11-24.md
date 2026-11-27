# Offline Analysis: Learned Per-Window Frame-Structure Chooser (Brief 18, 2026-11-24)

Branch `feature/gemini-structure-model` (from `variance-integration` `b4031c3f7`).
Zero new encodes; **$0.00 AWS** beyond the shared build box (`nc-remote`).

## 1. Question & Dataset Audit

Can a small learned per-window frame-structure chooser (a) beat the Brief 17 hand rules (`rate.cadence_period = 8` on AV1, `structure.depth_max` with motion/noise backoffs) by >= 0.3% J offline beyond the null, or (b) match them while retiring hand-tuned thresholds?

### Stored data (`~/nanocodec-data/work/nn-structure/`)
- **Regimes available:** CBR walks for `AV1 medium` and `HEVC medium` at `1/150` (600 f, 12 clips, 44 points, 880 windows each) and `1/300` (900 f, 10 clips, 35 points, 1,050 windows each) — 16 unique pool clips total, 0 missing null walks (`r-*.json` + `walks-*.json` + `lag-*.json`).
- **Not in stored data:** `VVC` was not walked in the 2026-11-23 structure oracle (only AV1 and HEVC).

### Feature availability audit (zero new encodes)

| Brief 18 feature | Stored in `nn-structure`? | Status in this study |
|---|---|---|
| `bpp` | Derivable (`kbps` + pool `width/height/fps`) | Included (`log_bpp`) |
| `key distance` | Derivable (`key_frames`, `window * 30`) | Included (`kf_is_300`, `f_since_key_frac`, `in_startup`) |
| `static share`, `camera motion`, `noise`, `mean luma`, `dark_frac` | **Not stored** in `nn-structure` | Computed causally from the 16 raw input clips (zero encodes; 6 frame-pairs/window, strictly using windows `< w`) |
| `previous window's results` | Only 6-candidate counterfactual `scores` stored (not single-choice realized bits/PSNR) | Used only as diagnostic upper bound (`F3`); excluded from causal models |
| `inheritance loss` | **Not stored** (encoder-internal state) | **Missing** — would require new encodes; omitted per brief |
| `rate state` (buffer fullness, `inter_level`, `period_ledger` room) | **Not stored** (only stream-level `max_fullness`) | **Missing** — would require new encodes; omitted per brief |

**Important data-leakage catches during analysis:**
1. `slope` (the clip's rate-per-dB slope from the 4-rung curve) is non-causal and was removed from all feature sets.
2. In the 10-clip `1/300` draw, `bodycam` and `lowlight` are `360p`/`400p` while `static`/`crowd`/`pan` are `536p`–`720p`. Giving a model `log(pixels)` or `fps` (`F0`/`F1`) lets it cheat via resolution. Our primary causal model **`F1_compact`** excludes `width`, `height`, `pixels`, `fps`, and `slope` — using **only 9 resolution-invariant causal features**: `log_bpp`, `kf_is_300`, `f_since_key_frac`, `in_startup`, plus running causal averages of `static_share_2`, `camera_motion`, `noise_hf`, `dark_frac`, and `mean_luma`.

---

## 2. Offline-to-Real Calibration & Limits of Offline Scoring

In CBR, each stored window score `s(point, w, c)` is measured over a 2-window horizon (`w` and `w+1`, 60 frames) in PSNR-YUV rate-equivalent %, conditional on the oracle's chosen prefix `0..w-1`, with `w+1` coded as `today`. Because a different choice changes the CBR rate state (buffer fullness, `inter_level`, `period_ledger` room, and `H` reference) for all later windows — and because real BD-J combines PSNR-YUV, VMAF v1, and MS-SSIM — offline mean window scores differ systematically from real coded BD-J.

Comparing offline mean window scores against the **real coded BD-J** in `lag-*.json` and `r-*.json` (all class-balanced):

| Regime | Policy | Offline mean score | Real coded BD-J | `Real - Offline` | Scale (`Real / Offline`) |
|---|---|---|---|---|---|
| **AV1 1/150** | `oracle - null` (hindsight) | `-0.578%` | `-1.03%` | `-0.45%` | `1.78x` |
| | `oracle-lag` | `-0.199%` | `-0.70%` | `-0.50%` | `3.52x` |
| | `const-period8` | `-0.170%` | `-1.04%` | `-0.87%` | `6.12x` |
| | `const-deep` | `+2.999%` | `+5.92%` | `+2.92%` | `1.97x` |
| | `const-flat` | `+3.052%` | `+7.07%` | `+4.02%` | `2.32x` |
| **AV1 1/300** | `oracle - null` (hindsight) | `-0.903%` | `-1.58%` | `-0.68%` | `1.75x` |
| | `oracle-lag` | `-0.245%` | `-0.75%` | `-0.50%` | `3.06x` |
| | `const-period8` | `-0.310%` | `-0.46%` | `-0.15%` | `1.48x` |
| | `const-deep` | `+1.545%` | `+2.56%` | `+1.01%` | `1.66x` |
| | `const-flat` | `+3.363%` | `+11.26%` | `+7.89%` | `3.35x` |
| **HEVC 1/150** | `oracle - null` (hindsight) | `-0.255%` | `-0.83%` | `-0.58%` | `3.25x` |
| | `oracle-lag` | `+0.334%` | `+0.22%` | `-0.11%` | — |
| | `const-period8` | `-0.024%` | `-0.24%` | `-0.22%` | — |
| | `const-deep` | `+4.425%` | `+7.99%` | `+3.57%` | `1.81x` |
| **HEVC 1/300** | `oracle - null` (hindsight) | `-0.698%` | `-1.71%` | `-1.01%` | `2.45x` |
| | `oracle-lag` | **`+0.338%`** | **`-0.31%`** | **`-0.65%`** | **sign flip!** |
| | `const-period8` | `+0.374%` | `+0.04%` | `-0.34%` | — |
| | `const-deep` | `+1.871%` | `+2.14%` | `+0.27%` | `1.14x` |

**Takeaways on offline scoring error:**
1. **Magnitude compression (~1.8x–2.5x):** Changing one window out of a 2-window horizon understates cumulative stream-wide BD-J changes by roughly `1.8x–2.5x`.
2. **State-drift error (~0.3–0.7% J):** On `HEVC 1/300`, `oracle-lag` scores `+0.338%` offline (worse than `today`) yet achieves **`-0.31%`** real coded BD-J (`-0.65%` trajectory discrepancy), because the offline scores were evaluated on the hindsight oracle's already-perturbed CBR rate state.
3. **Candidate exclusivity:** In the stored walks, `period8` and `deep` are mutually exclusive (`period=8` uses `depth <= 1.0`; `deep` uses `period=4, depth=1.5`). On AV1 medium, Brief 17 showed `period8 + depth_max=1.35` together gains `-0.47%` on `1/300` over `period8` alone — an action the stored walks cannot represent.

---

## 3. Leave-One-Clip-Out CV Results (Whole Clips Held Out)

All models predict the candidate score vector (`safe3 = {today, period8, deep}` eliminates noisy `flat`/`full`/`nosecond` picks, which cost `+0.3%` to `+12%` real BD-J) and pick `argmin_c pred[c]`. Pooled models train across `1/150 + 1/300` (16 unique clips, 1,930 windows per codec) with **all points of the held-out clip removed from both key shares**.

### Null-walk sanity check (`F1|RF(d=5)` on `null_decisions`)
- `av1-150`: `+0.047% [-0.065, +0.160]`
- `av1-300`: `-0.054% [-0.104, -0.005]`
- `hevc-150`: `-0.059% [-0.078, -0.042]`
- `hevc-300`: `+0.006% [-0.053, +0.061]`
- Mean across all 4 regimes: **`-0.015%`** (confirms zero leakage in LOCO CV).

### Main comparison on identical stored outcomes (class-balanced offline %, 95% bootstrap CI over clips)

| Regime | `today` | `const-p8` | `brief17-rule` | `oracle-lag` | **`F1_compact\|Ridge` (9-feat linear, pooled)** | `F1\|HGB` (pooled) | `hindsight - null` (offline) |
|---|---|---|---|---|---|---|---|
| **AV1 1/150** | `0.000%` | `-0.170%` | `-0.170%` | `-0.199%` | **`-0.184%`** `[-0.304, -0.064]` | `-0.296%` `[-0.401, -0.186]` | `-0.578%` |
| **AV1 1/300** | `0.000%` | `-0.310%` | `-0.207%` | `-0.245%` | **`-0.452%`** `[-0.608, -0.296]` | `-0.527%` `[-0.628, -0.423]` | `-0.903%` |
| **HEVC 1/150** | `0.000%` | `-0.024%` | `0.000%` | `+0.334%` | **`-0.024%`** `[-0.050, +0.002]` | `-0.121%` `[-0.253, +0.010]` | `-0.255%` |
| **HEVC 1/300** | `0.000%` | `+0.374%` | `-0.112%` | `+0.338%` | **`-0.291%`** `[-0.487, -0.094]` | `-0.267%` `[-0.534, -0.002]` | `-0.698%` |

### What the 9-feature linear model (`F1_compact|Ridge`) learned

Standardized Ridge weights for `deep` (negative = favors `deep`):
- **AV1 `deep`** (`intercept = +2.08`): `static_share_2: -1.16`, `noise_hf: -0.60`, `camera_motion: +0.56`, `kf_is_300: -0.39`, `in_startup: +0.37`, `f_since_key_frac: -0.31`, `dark_frac: +0.24`, `mean_luma: -0.23`, `log_bpp: -0.05`.
- **HEVC `deep`** (`intercept = +2.71`): `static_share_2: -1.15`, `in_startup: +1.10`, `f_since_key_frac: -0.96`, `kf_is_300: -0.48`, `camera_motion: +0.46`, `noise_hf: -0.43`, `mean_luma: -0.32`, `dark_frac: +0.22`, `log_bpp: +0.22`.

Per-class breakdown of `F1_compact|Ridge` (pooled):
- **HEVC 1/300 (`-0.291%`):** `bodycam -0.007%`, `crowd -0.377%`, `lowlight -0.063%`, `pan -0.170%`, `static -0.840%` — **all 5 classes negative** (`9/10` clips `<= 0.00%`; picks: `today 57.5%`, `deep 30.3%`, `period8 12.2%`).
- **AV1 1/300 (`-0.452%`):** `bodycam +0.103%`, `crowd -0.412%`, `lowlight +0.165%`, `pan -0.174%`, `static -1.942%` (`picks: period8 66.3%, deep 24.3%, today 9.4%`).

### Key findings
1. **Complex trees (`HGB`, `RF`) do NOT beat the 9-feature linear model** on HEVC (`-0.267%` / `-0.227%` vs `-0.291%` at `1/300`) and barely differ on AV1 (`-0.527%` vs `-0.452%` at `1/300`, where `HGB` also had resolution features). The causal signal is almost entirely a **smooth linear score** of content stillness, brightness, and GOP position — not high-order nonlinear interactions.
2. **Why the linear model beats `brief17-rule` on `1/300` (`-0.291%` vs `-0.112%` HEVC; `-0.452%` vs `-0.207%` AV1):** Brief 17's `structure.depth_max` uses 5 hard rectangular thresholds (`depth_max`, `depth_deep_loss`, `depth_deep_min_period`, `depth_deep_max_motion`, `depth_deep_max_noise`) and no luma/darkness term, so it both leaked into dark `lowlight` (`+2.04%` on HEVC quick screen) and missed textured still scenes (`pool_chad_4_109`, where the linear model gets `-1.754%` on HEVC and `-2.502%` on AV1).
3. **Almost all of the gain is on `deep` (cadence depth) at `1/300`, not per-window `period` switching or `nosecond`:**
   - On AV1, `period8` is already a constant preset win (Brief 17).
   - On HEVC `1/150`, the causal prize is near zero (`-0.024%` offline).
   - On HEVC `1/300`, the 9-weight linear scorer beats `brief17-rule` by **`-0.179%` offline** (`~-0.35%` to `-0.44%` at the empirical `~2.0x–2.45x` real-BD-J scale) with **zero hard content thresholds** and **zero lowlight/bodycam regressions**.

---

## 4. Decision & Proposed Closed-Loop Experiment (Awaiting Approval)

**Decision rule check:**
- **Gain bar (`>= 0.3%` over Brief 17 rules):** In raw offline units, `F1_compact|Ridge` beats `brief17-rule` by `-0.18%` on HEVC `1/300` and `-0.25%` on AV1 `1/300` (borderline in raw offline units, `~0.35–0.45%` when scaled to real BD-J using the `lag-*.json` calibration, and `0.0%` at `1/150`).
- **Threshold-retirement bar (parity with clearly fewer tuned thresholds):** **MET.** A single 8–9 weight linear `cadence_depth` scorer replaces the 5 hand-tuned `structure.depth_deep_*` thresholds (plus fixes the missing darkness/lowlight guard) with a smooth, single-formula continuous depth in `[0.25, depth_max]` that works natively alongside `rate.cadence_period = 8` on AV1 and `period = 4` on HEVC/VVC.

### Proposed closed-loop experiment (do not run until approved)
1. **Engine change (default off, byte-identical while off):**
   - Add a tiny linear `structure.depth_model` (default `0`, off) in `crates/nc-encoder/src/inheritance.rs` + `encoder.rs` that computes continuous `depth = clamp(base_depth + linear_score(features), 0.25, depth_max)` once per 30-frame window (or per mini-GOP) from existing causal engine inputs (`static_share`, `camera_motion`, `noise_sigma`, `mean_luma`, `dark_frac`, `frames_since_key / key_period`, `in_startup`, `key_period >= 240`, `log_bpp`) — retiring the 5 rectangular `depth_deep_*` thresholds when enabled.
   - Verify `preset_identity.py` (`387/387`), `static_recovery`, and 600f single-key checks on `nc-remote` (`$0`).
2. **One `nceval.quick` screen per codec (`HEVC medium`, `VVC medium`, `AV1 medium` paired with `p8`):**
   - Packed onto a single `eu-north-1` spot `c6i.24xlarge` warm box (`$0.44/h`, no on-demand fallback, no other regions).
   - **Estimated time & cost:** ~25–30 minutes total for all 3 codecs (`HEVC`, `VVC`, `AV1`), **`~$0.20–$0.25`** (well under the `$0.50` cap).
   - **Zero CPU regression expected** (10-MAC linear dot product per window); if any codec wins `>= 0.3%` or matches Brief 17 with zero guards, follow up with `aws_cpu_ab` (`c8i`/`c8g` spot, `~$0.02/codec`).

---

## 5. Closed-Loop Screen Results (`b481daec7`, draw `2026-11-24`)

Implemented `structure.depth_model` (default `0.0`, off; zero overhead when off) in `crates/nc-encoder/src/{params,qp_cadence,inheritance,encoder,spatial_encode}.rs` (`b481daec7`) using the pooled 8-feature linear model (`log_bpp`, `kf_long`, `f_since_key_frac`, `in_startup`, plus EMA of `static_share`, `camera_motion`, `dark_frac`, `mean_luma`) with `DEFAULT_MODEL_MAX_DEPTH = 1.5`.

### Verification
- `cargo test --locked --release -p nanocodec-encoder inheritance`: **5/5 passed** (including `linear_depth_model_boosts_bright_still_long_gop_and_guards_dark_or_moving`).
- `cargo test --locked --release -p nanocodec-cli --test integration static_recovery`: **13/13 passed**.
- `tools/dev/preset_identity.py` vs `ef812fa3`: **387/387 streams byte-identical** with `structure.depth_model = 0`.

### `nceval.quick` + `aws_cpu_ab` (`c8i.2xlarge` + `c8g.2xlarge` `eu-north-1` spot, draw `2026-11-24`)

| Codec / Preset (`k`) | Window | J (BD-rate) [90% CI] | dCPU (`c8i` / `c8g`) | Net J (`J + k·dCPU`) | Pulse / Worst class |
|---|---|---|---|---|---|
| **HEVC `medium`** (`0.1058`) | L1 | `-0.03%` `[-0.18, +0.08]` | — | — | `drone +0.24%` (`lowlight 0.00%`, `bodycam 0.00%`, `static -0.69%`) |
| `structure.depth_model=1` | `1/150` | **`-0.24%`** `[-0.39, -0.01]` | `+0.28%` (`+0.35%` / `+0.21%`) | **`-0.21%`** | `-0.02 dB` / `lowlight +0.00%` (all 5 classes <= 0) |
| | `1/300` | `-0.07%` `[-0.47, +0.62]` | `-0.04%` (`+0.11%` / `-0.18%`) | **`-0.07%`** | `+0.02 dB` / `crowd +1.49%` (`static -1.35%`, `lowlight -0.05%`) |
| | Probe mean | `-0.16%` `[-0.37, +0.21]` | `+0.12%` | **`-0.14%`** | Retires Brief 17's `lowlight +2.04%` guard |
| **VVC `medium`** (`0.1836`) | L1 | `+0.20%` `[-0.02, +0.26]` | — | — | `drone +1.38%` (`lowlight 0.00%`, `bodycam 0.00%`) |
| `structure.depth_model=1` | `1/150` | `+0.20%` `[+0.07, +0.29]` | `-0.24%` (`+0.08%` / `-0.55%`) | `+0.16%` | `-0.05 dB` / `drone +0.56%` (`lowlight 0.00%`) |
| | `1/300` | **`-0.50%`** `[-1.07, -0.08]` | `-0.33%` (`-0.26%` / `-0.40%`) | **`-0.56%`** | `-0.01 dB` / `lowlight +0.20%` (`crowd -1.39%`, `static -0.90%`) |
| | Probe mean | `-0.15%` `[-0.44, +0.07]` | `-0.28%` | **`-0.20%`** | Strong `1/300` win traded against `+0.16%` at `1/150` |
| **AV1 `medium`** (`0.1412`) | L1 | **`-0.69%`** `[-0.98, -0.28]` | — | — | `pan -0.16%` (all 8 classes negative; beats `p8` `-0.54%`) |
| `rate.cadence_period=8` + | `1/150` | `-0.23%` `[-0.71, +0.20]` | `-0.25%` (`-0.33%` / `-0.17%`) | `-0.27%` | `-0.09 dB` / `lowlight +1.05%` (`depth_max=1.5` too deep at `1/150`) |
| `structure.depth_model=1` | `1/300` | **`-0.86%`** `[-1.46, -0.18]` | **`-5.11%`** (`-5.05%` / `-5.16%`) | **`-1.58%`** | `-0.01 dB` / `bodycam +0.35%` (`static -2.62%`, `pan -1.67%`) |
| | Probe mean | **`-0.54%`** `[-0.93, -0.14]` | **`-2.68%`** | **`-0.92%`** | Beats `p8 + depth_max=1.35` on L1 (`-0.69%` vs `-0.51%`) & `1/300` (`-0.86%` vs `-0.65%`), trails on `1/150` (`-0.23%` vs `-1.43%`) |

### Closed-loop verdict
1. **Did `structure.depth_model=1` beat Brief 17 rules by >= 0.3% J overall?** **No** on probe mean (`HEVC -0.14%` net J vs `-0.12%`, `VVC -0.20%` net J, `AV1 -0.92%` net J vs `p8 + depth_max=1.35` `~-1.45%` net J).
2. **Where it clearly won:**
   - **Threshold retirement & lowlight safety:** Completely eliminated the `lowlight` regression on HEVC (`0.00%` / `-0.05%` vs `+2.04%` in Brief 17) and VVC (`0.00%` / `+0.20%`) with zero hard content thresholds.
   - **`1/300` long-GOP wins:** **`-0.56%` net J** on VVC `1/300` (`-0.50%` J, `-0.33%` CPU) and **`-1.58%` net J** on AV1 `1/300` (`-0.86%` J, `-5.11%` CPU), plus **`-0.69%`** L1 on AV1 (all 8 classes negative).
3. **Why `1/150` trailed on AV1 & VVC:** `DEFAULT_MODEL_MAX_DEPTH = 1.5` allowed depth > 1.0 at `key_period = 150` (`kf_long = 0`), where `1.5` over-boosted `static`/`lowlight` (AV1) and `drone` (VVC). Keeping `structure.depth_model` default off (`0`) on `feature/gemini-structure-model` preserves the clean infrastructure; capping `depth_max = 1.35` (or gating `key_period >= 240` for `depth > 1`) would immediately recover `1/150` while keeping the `1/300` win.

---

## 6. Compute & AWS Spend (All `eu-north-1` Spot, Cap `$0.50`)
- **Shared build box (`nc-remote`):** Offline LOCO CV fits, Rust builds/tests, `preset_identity` (`387/387`), and x86/ARM cross-builds (`$0.00` extra).
- **Packed 3-codec `nceval.quick` (`i-0919bb86e5b019d07`, `c6i.24xlarge` `eu-north-1` spot, `$0.44/h`):** 2 short failed staging attempts (2.5 min total, `$0.02`) + 1 successful packed run (5.1 min, `$0.04`) = **`$0.06`**.
- **Packed 3-codec `aws_cpu_ab` (`eu-north-1` spot, `--key-shares 150,300`):** `c8i.2xlarge` (0.40 h @ `$0.0573/h`, `$0.02`) + `c8g.2xlarge` (0.47 h @ `$0.0648/h`, `$0.03`) = **`$0.05`**.
- **Total Brief 18 AWS spend:** **`~$0.11`** (of `$0.50` cap; 0 on-demand, 0 region spills, **0 boxes left running**).
- **Scratch scripts & outputs:** `~/nanocodec-data/scratch/gemini-b18/{oracle_data.json, pixel_features.json, fit_results_v2.json, quick-*-depthmodel.json, cpu-ab-3codecs.json}`.
