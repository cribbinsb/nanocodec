# Fixed-QP Fast-Rung Swing Diagnosis & Fixes (Brief 19, 2026-11-24)

Branch `feature/gemini-fixed-qp` (rebased onto `variance-integration` `85b788218`).

## 1. Diagnosis of the CBR -> Fixed-QP Swing (BP-3.11 "ours", `$0` offline)

Decomposed on the **exact same 80 common clips** (eliminating the 41-vs-97 set-composition bias of `4–8` pts) using the stored `scores.jsonl` (`frame_bytes` + `pooled` metrics; sealed "ours" set used for diagnosis only):

1. **Anchor effect (`+11–14 pts`, legitimate anchor property):** In CBR, `x264-medium` starves frame 0 (`key% = 2.6%`, `Ydrift = +1.81 dB`) and loses `-13.6%` J to `x264-medium~psnr`. In Fixed QP, it codes a proper key (`9.6%`) and narrows to `-2.0%` (`+11.7%` swing of the anchor itself). Against `x264-medium~psnr`, `av2@med` (`-5.7%`), `vp9@med` (`-0.6%`), and `av1@med` (`+1.3%`) have zero swing.
2. **Pinned `rate.fixed_qp_profile = 0` & `rate.fixed_qp_moving_key = 0` (`+9–18 pts` on `HEVC faster/fast`; `+4–8 pts` on `H.264/VVC/VP9/AV1` fast rungs — DEFECT):** Under CBR, `structure.inherited_depth = 1` keeps cadence active. At Fixed QP, `inherited_depth` turns off (`encoder.rs:3260`); `medium` compensates via `FIXED_QP_PARAMS` (`rate.cadence_static_scaling = 0`, `refs.second_full = 1`) and `fixed_qp_moving_key = 6`. Pinning both to `0` on fast rungs collapsed cadence (`HEVC faster` `r4: 6.82 -> 4.19`) and inflated key share (`22.5%` overall, `37.9%` at QP 47).
3. **CBR-tuned low-QP skip/zero/depth-removal gates (`+8–12 pts` on `H.264 veryfast` & `HEVC faster` — DEFECT):** At QP 22–27, `hevc/faster.toml`'s `planner.depth_removal_min_quantizer = 0` stripped partition depth at low QP, and `h264/veryfast.toml`'s `inter_zero_luma_scale = 2` (nonbase `3`) zeroed real detail (`+39.5%` high-rate VMAF gap vs `x264-medium`).
4. **AV1 `fast` vs `faster` inversion (`+9–16 pts` on moving/drone/texture clips — DEFECT):** `av1/fast.toml` had `structure.unified = 0` and `rate.fixed_qp_moving_key = 0`. Giving `av1/fast` `av1/faster`'s 3-layer hierarchy hurt (`+2.0%` J at 150 f), whereas enabling the second reference (`structure.unified = 1`, `secondary_min_depth = 0`) + `fixed_qp_moving_key = 6` won cleanly (`-2.9%` J at 600 f).

---

## 2. Pool Screen Results (`nceval.fixedqp ab`, draw `fixedqp-2026-11-24`, `n=24`) & Fixed-QP CPU A/B

| Rung (BP-3 QP ladder slope) | Candidate (`fixed_qp.` params) | 150 f J [95% CI] | 600 f J [95% CI] | CPU (`c8i` / `c8g`) | Slope (`%J / 1% CPU`) | Decision |
|---|---|---:|---:|---:|---:|---|
| **H.264 `veryfast`** (`-0.528`) | `f1_profile_only`: `fixed_qp_profile=1` | — | `-2.7% [-3.2, -1.6]` | `-0.30% / -0.52%` | ∞ (faster!) | Superseded by `f2_zeroscale1` |
| | `f1_profile_mk`: `fixed_qp_profile=1`, `moving_key=6` | `-3.4% [-4.1, -2.1]` | `-3.0% [-3.4, -2.0]` | `+0.00% / -0.17%` | ∞ (free!) | Strong zero-CPU option |
| | **`f2_zeroscale1`**: above + `inter_zero_luma_scale=1`, `nonbase=2` | — | **`-6.8% [-7.3, -6.0]`** | `+8.52% / +10.19%` | **`-0.73%`** | **Adopted (`fixed_qp.`)** |
| **HEVC `faster`** (`-0.384`) | `f1_nostatic`: `cadence_static_scaling=0`, `moving_key=6` | `-1.4% [-2.0, -0.8]` | `-1.3% [-1.8, -0.9]` | `+1.41% / +1.00%` | **`-1.08%`** | Strong low-CPU option |
| | **`f2_depth_only`**: above + `depth_removal_min_quantizer=0.86` | — | **`-7.3% [-8.5, -5.5]`** | `+17.91% / +15.42%` | **`-0.44%`** | **Adopted (`fixed_qp.`)** |
| | `f2_gates_nostatic`: above + `nonbase.pre_transform_zero_scale=2`, `key_intra_coarse=0` | `-7.1% [-8.5, -5.0]` | `-7.9% [-9.1, -5.9]` | `+21.83% / +20.25%` | `-0.37%` | Diminishing return |
| **AV1 `fast`** (`-0.140`) | `f1_movingkey`: `fixed_qp_moving_key=6` | `-0.7% [-1.3, -0.3]` | `-0.7% [-1.0, -0.4]` | `+0.10% / -0.00%` | ∞ (free!) | Subset of winner |
| | `f1_layers3`: `layers.count=3`, `adaptive=1`, `propagation=1` | `+2.0% [+0.1, +3.8]` | — | — | — | **Rejected** |
| | `f1_unified`: `unified=1`, `secondary_min_depth=0`, `static_scaling=0`, `moving_key=6` | `-2.2% [-3.1, -1.3]` | `-2.4% [-3.4, -1.4]` | `+4.46% / +4.89%` | `-0.51%` | Beaten by `keepstatic` |
| | **`f1_unified_keepstatic`**: `unified=1`, `secondary_min_depth=0`, `moving_key=6` | — | **`-2.9% [-3.3, -2.3]`** | `+5.00% / +5.66%` | **`-0.54%`** | **Adopted (`fixed_qp.`)** |

---

## 3. Step 4 Carry-Over Results (10 Rungs, `nceval.fixedqp ab`, draw `fixedqp-2026-11-24`, `n=24`) & Fixed-QP CPU A/B

Every one of the 10 carry-over rungs passed both 150 f and 600 f screens and beat its BP-3.11 fixed-QP front slope (`%J / 1% CPU`) with **0 CBR stream changes**:

| Rung (BP-3 QP front slope) | Candidate (`fixed_qp.` params) | 150 f J [95% CI] | 600 f J [95% CI] | CPU (`c8i` / `c8g`) | Mean CPU | Slope (`%J / 1% CPU`) | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| **H.264 `faster`** (`-0.209`) | `f1_prof_mk`: unpin `fixed_qp_profile`, `moving_key=6` | `-3.8% [-4.5, -2.5]` | `-3.2% [-3.6, -2.2]` | `-0.26% / -0.84%` | `-0.55%` | ∞ (faster) | Superseded by `f2_zero` |
| | **`f2_zero`**: above + `inter_zero_luma_scale=1`, `nonbase=2` | `-4.3% [-5.0, -3.0]` | **`-3.7% [-4.1, -2.9]`** | `+3.04% / +3.09%` | `+3.07%` | **`-1.21%`** | **Adopted (`fixed_qp.`, 5.8×)** |
| **H.264 `fast`** (`-0.502`) | `f1_prof_mk`: unpin `fixed_qp_profile`, `moving_key=6` | `-3.2% [-3.9, -2.0]` | `-2.7% [-3.1, -1.8]` | `-0.11% / -0.06%` | `-0.09%` | ∞ (faster) | Superseded by `f2_zero` |
| | **`f2_zero`**: above + `inter_zero_luma_scale=1`, `nonbase=2` | `-3.7% [-4.4, -2.5]` | **`-3.1% [-3.4, -2.2]`** | `+2.98% / +2.68%` | `+2.83%` | **`-1.10%`** | **Adopted (`fixed_qp.`, 2.2×)** |
| **HEVC `fast`** (`-0.091`) | `f1_nostatic`: unpin `fixed_qp_profile`, `refs.second_full=0`, `cadence_static_scaling=0`, `moving_key=6` | `-1.9% [-3.1, -0.9]` | `-1.5% [-2.5, -0.8]` | `+1.75% / +2.02%` | `+1.89%` | `-0.79%` | Superseded by `f2_all` |
| | **`f2_all`**: above + `nonbase.pre_transform_zero_scale=2` | `-2.3% [-3.5, -1.3]` | **`-1.9% [-2.9, -1.3]`** | `+4.91% / +5.17%` | `+5.04%` | **`-0.38%`** | **Adopted (`fixed_qp.`, 4.2×)** |
| **VVC `faster`** (`-0.033`) | **`f1_nostatic`**: unpin `fixed_qp_profile`, `cadence_static_scaling=0`, `moving_key=6` | `-1.4% [-2.8, -0.2]` | **`-1.6% [-2.6, -0.9]`** | `+1.16% / +0.30%` | `+0.73%` | **`-2.19%`** | **Adopted (`fixed_qp.`, 66×)** |
| **VVC `fast`** (`-0.290`) | **`f1_nostatic`**: unpin `fixed_qp_profile`, `cadence_static_scaling=0`, `moving_key=6` | `-1.6% [-3.0, -0.3]` | **`-1.7% [-2.7, -1.0]`** | `+1.07% / +0.94%` | `+1.01%` | **`-1.69%`** | **Adopted (`fixed_qp.`, 5.8×)** |
| **VP9 `veryfast`** (`-0.464`) | `f1_p8_mk`: `cadence_period=8`, `base_boost=4`, `startup=0`, `moving_key=6` | `-1.0% [-1.6, -0.6]` | `-2.1% [-2.3, -1.6]` | `-0.66% / -1.00%` | `-0.83%` | ∞ (faster) | Superseded by `f2_depth` |
| | **`f2_depth`**: above + `depth_removal_min_quantizer=0.86` | `-4.3% [-5.2, -2.8]` | **`-6.0% [-6.7, -4.0]`** | `+13.20% / +10.05%` | `+11.63%` | **`-0.52%`** | **Adopted (`fixed_qp.`, 1.11×)** |
| **VP9 `faster`** (`-0.177`) | `f1_p8_mk`: `cadence_period=8`, `base_boost=4`, `startup=0`, `moving_key=6` | `-1.0% [-1.5, -0.5]` | `-2.3% [-2.6, -1.7]` | `-0.83% / -0.24%` | `-0.54%` | ∞ (faster) | Superseded by `f2_depth` |
| | **`f2_depth`**: above + `depth_removal_min_quantizer=0.86` | `-3.6% [-4.4, -2.3]` | **`-5.2% [-5.8, -3.5]`** | `+7.95% / +6.18%` | `+7.07%` | **`-0.74%`** | **Adopted (`fixed_qp.`, 4.2×)** |
| **VP9 `fast`** (`-0.160`) | **`f2_light_depth`**: `moving_key=6`, `light.depth_removal_min_quantizer=0.86` | `-1.7% [-2.3, -0.7]` | **`-2.1% [-2.6, -1.2]`** | `+5.61% / +4.77%` | `+5.19%` | **`-0.40%`** | **Adopted (`fixed_qp.`, 2.5×)** |
| **AV1 `veryfast`** (`-0.117`) | `f1_mk`: `moving_key=6` | `-1.3% [-1.9, -0.8]` | `-1.1% [-1.6, -0.6]` | `+0.04% / -0.13%` | `-0.05%` | ∞ (faster) | Superseded by `f2_depth` |
| | **`f2_depth`**: above + `depth_removal_min_quantizer=0.8` | `-8.9% [-10.4, -6.9]` | **`-9.7% [-11.2, -7.5]`** | `+11.74% / +8.68%` | `+10.21%` | **`-0.95%`** | **Adopted (`fixed_qp.`, 8.1×)** |
| **AV1 `faster`** (`-0.074`) | `f1_mk`: `moving_key=6` | `-1.5% [-2.4, -0.8]` | `-1.2% [-1.8, -0.7]` | `+0.21% / +0.26%` | `+0.24%` | `-5.00%` | Superseded by `f2_light_depth` |
| | **`f2_light_depth`**: above + `light.depth_removal_min_quantizer=0.8` | `-5.0% [-5.7, -3.6]` | **`-3.7% [-4.3, -2.8]`** | `+4.13% / +3.15%` | `+3.64%` | **`-1.02%`** | **Adopted (`fixed_qp.`, 13.8×)** |

---

## 4. AWS Spend (All `eu-north-1` Spot)

### Step 3 (Cap `$1.00`)
- **150-frame 3-codec screen (`i-0f3fe2c9f27cd88ed`, `c6i.24xlarge` spot, 11.1 min @ `$0.4368/h`):** **`$0.08`**.
- **600-frame 3-codec confirmation (same warm box `i-0f3fe2c9f27cd88ed`, 14.1 min @ `$0.4368/h`):** **`$0.10`** (+ idle tail `~$0.07`).
- **Fixed-QP CPU A/B #1 (`c8i.2xlarge` + `c8g.2xlarge` spot, 0.09 h each):** **`$0.01`**.
- **Fixed-QP CPU A/B #2 (`c8i.2xlarge` + `c8g.2xlarge` spot, 0.06 h each):** **`$0.01`**.
- **Step 3 total:** **`~$0.27`** (cap `$1.00`).

### Step 4 (Cap `$1.00`)
- **Warm box `i-092760fbc45400517` (`c7i.24xlarge` spot @ `$0.5621/h`, 48.0 min total incl. idle auto-stop):** **`$0.46`** (Batch A 150 f + Batch B 600 f).
- **Warm box `i-059a8698480923320` (`c7i.24xlarge` spot @ `$0.5621/h`, 49.8 min total):** **`$0.48`** (Batch B 150 f + Batch A 600 f).
- **Packed 17-pair Fixed-QP CPU A/B (`c8i.2xlarge` 0.20 h @ `$0.0579/h` `$0.01` + `c8g.2xlarge` 0.22 h @ `$0.0648/h` `$0.015`):** **`$0.03`**.
- **Step 4 total:** **`$0.97`** (cap `$1.00`, 0 on-demand, 0 region spills, 0 boxes left running).
